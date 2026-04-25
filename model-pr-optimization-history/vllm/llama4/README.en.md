# vllm Llama 4 Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/tool_chat_template_llama4_json.jinja` | [#16428](https://github.com/vllm-project/vllm/pull/16428) |
| `examples/tool_chat_template_llama4_pythonic.jinja` | [#16463](https://github.com/vllm-project/vllm/pull/16463), [#17917](https://github.com/vllm-project/vllm/pull/17917) |
| `tests/models/multimodal/processing/test_llama4.py` | [#16113](https://github.com/vllm-project/vllm/pull/16113) |
| `tests/models/multimodal/processing/test_mllama4.py` | no direct PR-number commit |
| `tests/tool_parsers/test_llama4_pythonic_tool_parser.py` | no direct PR-number commit |
| `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` | [#25145](https://github.com/vllm-project/vllm/pull/25145), [#25889](https://github.com/vllm-project/vllm/pull/25889), [#26790](https://github.com/vllm-project/vllm/pull/26790), [#30709](https://github.com/vllm-project/vllm/pull/30709) |
| `vllm/model_executor/models/llama4.py` | [#16113](https://github.com/vllm-project/vllm/pull/16113), [#16311](https://github.com/vllm-project/vllm/pull/16311), [#16439](https://github.com/vllm-project/vllm/pull/16439), [#16512](https://github.com/vllm-project/vllm/pull/16512), [#16801](https://github.com/vllm-project/vllm/pull/16801), [#17315](https://github.com/vllm-project/vllm/pull/17315), [#19997](https://github.com/vllm-project/vllm/pull/19997), [#20419](https://github.com/vllm-project/vllm/pull/20419), [#20788](https://github.com/vllm-project/vllm/pull/20788), [#21499](https://github.com/vllm-project/vllm/pull/21499), [#22691](https://github.com/vllm-project/vllm/pull/22691), [#22701](https://github.com/vllm-project/vllm/pull/22701), ... (18 total) |
| `vllm/model_executor/models/llama4_eagle.py` | [#20591](https://github.com/vllm-project/vllm/pull/20591), [#20788](https://github.com/vllm-project/vllm/pull/20788), [#27136](https://github.com/vllm-project/vllm/pull/27136), [#29926](https://github.com/vllm-project/vllm/pull/29926) |
| `vllm/model_executor/models/mllama4.py` | [#16113](https://github.com/vllm-project/vllm/pull/16113), [#16201](https://github.com/vllm-project/vllm/pull/16201), [#16365](https://github.com/vllm-project/vllm/pull/16365), [#16746](https://github.com/vllm-project/vllm/pull/16746), [#18368](https://github.com/vllm-project/vllm/pull/18368), [#20419](https://github.com/vllm-project/vllm/pull/20419), [#22021](https://github.com/vllm-project/vllm/pull/22021), [#22107](https://github.com/vllm-project/vllm/pull/22107), [#25961](https://github.com/vllm-project/vllm/pull/25961), [#28602](https://github.com/vllm-project/vllm/pull/28602), [#30709](https://github.com/vllm-project/vllm/pull/30709), [#35147](https://github.com/vllm-project/vllm/pull/35147) |
| `vllm/tool_parsers/llama4_pythonic_tool_parser.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 37
- Extra PRs preserved from existing docs: 3
- Total PRs in this document: 40
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
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
| 2025-09-11 | [#24444](https://github.com/vllm-project/vllm/pull/24444) | merged | [Bugfix] Fix platform-specific routing in CustomOp implementations | `vllm/model_executor/layers/rotary_embedding/mrope.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` |
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

## Per-PR Diff Audit Cards

### PR #16104 - [Model] Support Llama4 in vLLM

- Link: https://github.com/vllm-project/vllm/pull/16104
- Status/date: merged / 2025-04-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 35 files, +2369/-142, 3141 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support Llama4 in vLLM"; model line: Llama 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json`; PR body summary: Add the support for Llama4 Scout (17B x 16 Experts) and Maverick (17B x 128 Experts) in vLLM. Using 8xH100, vLLM can serve Scout with 1M context and Maverick with about 430K. Us....
- Key implementation: `vllm/model_executor/models/mllama4.py` added +886/-0 (886 lines); hunks: -0,0 +1,886; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward, touching `Llama4ImagePatchInputs, Llama4VisionMLP, __init__`; `vllm/model_executor/models/llama4.py` added +530/-0 (530 lines); hunks: -0,0 +1,530; symbols: Llama4MoE, custom_routing_function, __init__, forward, touching `Llama4MoE, custom_routing_function, __init__`; `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json` added +200/-0 (200 lines); hunks: -0,0 +1,200; `tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override, touching `test_processor_override`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` added +886/-0 (886 lines); hunks: -0,0 +1,886; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward
  - `vllm/model_executor/models/llama4.py` added +530/-0 (530 lines); hunks: -0,0 +1,530; symbols: Llama4MoE, custom_routing_function, __init__, forward
  - `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json` added +200/-0 (200 lines); hunks: -0,0 +1,200
  - `tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override
  - `vllm/model_executor/layers/rotary_embedding.py` modified +68/-0 (68 lines); hunks: -851,6 +851,70 @@ def _compute_inv_freq(self, base: Union[int, float]) -> tor...; -1130,6 +1194,10 @@ def get_rope(; symbols: _compute_inv_freq, Llama4VisionRotaryEmbedding, __init__, _compute_cos_sin_cache
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` added +886/-0; `vllm/model_executor/models/llama4.py` added +530/-0; `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json` added +200/-0; `vllm/model_executor/layers/rotary_embedding.py` modified +68/-0; `vllm/model_executor/layers/fused_moe/layer.py` modified +41/-24; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +24/-14
  - tests: `tests/models/multimodal/processing/test_llama4.py` added +99/-0
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_llama4.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16113 - Upstream Llama4 Support to Main

- Link: https://github.com/vllm-project/vllm/pull/16113
- Status/date: merged / 2025-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_llama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/mllama4.py`; associated commits `55dcce91df15`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 43 files, +2436/-155, 3350 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Upstream Llama4 Support to Main"; model line: Llama 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `tests/models/multimodal/processing/test_llama4.py`; PR body summary: As a follow up of https://github.com/vllm-project/vllm/pull/16104, we upstream the Llama4 support to the main branch. The goal of this PR: - Support the llama4 - Clean up some h....
- Key implementation: `vllm/model_executor/models/mllama4.py` added +895/-0 (895 lines); hunks: -0,0 +1,895; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward, touching `Llama4ImagePatchInputs, Llama4VisionMLP, __init__`; `vllm/model_executor/models/llama4.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: Llama4MoE, custom_routing_function, __init__, forward, touching `Llama4MoE, custom_routing_function, __init__`; `tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override, touching `test_processor_override`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` added +895/-0 (895 lines); hunks: -0,0 +1,895; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward
  - `vllm/model_executor/models/llama4.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: Llama4MoE, custom_routing_function, __init__, forward
  - `tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` added +895/-0; `vllm/model_executor/models/llama4.py` added +531/-0
  - tests: `tests/models/multimodal/processing/test_llama4.py` added +99/-0
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/audio_language/test_ultravox.py`, `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/test_phi3v.py`, `tests/models/decoder_only/vision_language/test_pixtral.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16201 - [Misc] Move Llama 4 projector call into encoder execution

- Link: https://github.com/vllm-project/vllm/pull/16201
- Status/date: merged / 2025-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `ed636d99caa0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-3, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Move Llama 4 projector call into encoder execution"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/mllama4.py`; PR body summary: On vLLM we consider encoder execution to generate multimodal embeddings ready to be merged into text embeddings, thus it often includes the projector call even though it's not p....
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +4/-3 (7 lines); hunks: -760,6 +760,8 @@ def _process_image_input(; -791,10 +793,9 @@ def get_input_embeddings(; symbols: _process_image_input, get_multimodal_embeddings, get_input_embeddings, touching `_process_image_input, get_multimodal_embeddings, get_input_embeddings`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +4/-3 (7 lines); hunks: -760,6 +760,8 @@ def _process_image_input(; -791,10 +793,9 @@ def get_input_embeddings(; symbols: _process_image_input, get_multimodal_embeddings, get_input_embeddings
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +4/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16311 - [BugFix] llama4 qknorm should be not shared across head

- Link: https://github.com/vllm-project/vllm/pull/16311
- Status/date: merged / 2025-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `ec7da6fcf32f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-12, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] llama4 qknorm should be not shared across head"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: QKNorm should not be across head, we need to do qknorm per head, score goes up from 0.6858 to 0.7153. This also resovles TP=4 Accuracy Issue https://github.com/vllm-project/vllm....
- Key implementation: `vllm/model_executor/models/llama4.py` modified +7/-12 (19 lines); hunks: -155,14 +155,8 @@ def __init__(self,; -226,10 +220,11 @@ def forward(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +7/-12 (19 lines); hunks: -155,14 +155,8 @@ def __init__(self,; -226,10 +220,11 @@ def forward(; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +7/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16365 - [Model] Remove image mm limit for LLaMa4

- Link: https://github.com/vllm-project/vllm/pull/16365
- Status/date: merged / 2025-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `61de3ef74b9c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-7, 84 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Remove image mm limit for LLaMa4"; model line: Llama 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/mllama4.py`; PR body summary: Follow up in https://github.com/vllm-project/vllm/pull/16104 to remove the hard limit. Added some tests to allow testing more images in vision_language_multi_image.py. Default d....
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +3/-1 (4 lines); hunks: -477,7 +477,9 @@ def get_hf_processor(self, **kwargs: object) -> Llama4Proces...; symbols: get_hf_processor, get_supported_mm_limits, get_patch_per_chunk, touching `get_hf_processor, get_supported_mm_limits, get_patch_per_chunk`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +3/-1 (4 lines); hunks: -477,7 +477,9 @@ def get_hf_processor(self, **kwargs: object) -> Llama4Proces...; symbols: get_hf_processor, get_supported_mm_limits, get_patch_per_chunk
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -477,7 +477,9 @@ def get_hf_processor(self, **kwargs: object) -> Llama4Processor:
-        return {"image": 10}
+        # Although vLLM can support more images from an infra capability
+        # perspective, we do not recommend using >10 images in practice.
+        return {"image": None}
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16439 - [Llama4] Enable attention temperature tuning by default for long context (>32k)

- Link: https://github.com/vllm-project/vllm/pull/16439
- Status/date: merged / 2025-04-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `99ef59cf7f93`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Llama4] Enable attention temperature tuning by default for long context (>32k)"; model line: Llama 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: Attention temperature tuning (on nope layers) improves accuracy on long context (>32k) tasks. Enabling it by default for long context unless explicitly disabled by the user. Eva....
- Key implementation: `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -467,11 +467,15 @@ class Llama4ForCausalLM(LlamaForCausalLM):; symbols: Llama4ForCausalLM, __init__, touching `Llama4ForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -467,11 +467,15 @@ class Llama4ForCausalLM(LlamaForCausalLM):; symbols: Llama4ForCausalLM, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +6/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16463 - [Frontend] Added chat templates for LLaMa4 pythonic tool calling

- Link: https://github.com/vllm-project/vllm/pull/16463
- Status/date: merged / 2025-04-11
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_llama4_pythonic.jinja`; associated commits `16eda8c43a9d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +182/-2, 223 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] Added chat templates for LLaMa4 pythonic tool calling"; model line: Llama 4; category: bug fix; main diff: `examples/tool_chat_template_llama4_pythonic.jinja`; PR body summary: Fixing tool calling related chat templates for llama4. LLaMa4 followed similar pythonic format. Co-authored with @wukaixingxp. Tool call messages and tool response messages shou....
- Key implementation: `examples/tool_chat_template_llama4_pythonic.jinja` added +139/-0 (139 lines); hunks: -0,0 +1,139.
- Code diff details:
  - `examples/tool_chat_template_llama4_pythonic.jinja` added +139/-0 (139 lines); hunks: -0,0 +1,139
- Key code excerpts:

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

- Reviewed files:
  - docs: `examples/tool_chat_template_llama4_pythonic.jinja` added +139/-0
- Risk and verification: The diff ships test coverage in `tests/tool_use/conftest.py`, `tests/tool_use/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16512 - Optimized topk for topk=1 (Llama-4)

- Link: https://github.com/vllm-project/vllm/pull/16512
- Status/date: merged / 2025-04-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `bd6028d6b0bb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-2, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimized topk for topk=1 (Llama-4)"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: Clear speedup for latency case, adapted from https://github.com/sgl-project/sglang/commit/86a876d883a7c7a0e2b0fca5ef86e20ab92c0694 (thank you!) Llama Scout FP8 on 2xH100, input/....
- Key implementation: `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -37,7 +37,7; -50,7 +50,7 @@ def custom_routing_function(; symbols: custom_routing_function, touching `custom_routing_function`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -37,7 +37,7; -50,7 +50,7 @@ def custom_routing_function(; symbols: custom_routing_function
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -37,7 +37,7 @@
-from .utils import (AutoWeightsLoader, extract_layer_index,
+from .utils import (AutoWeightsLoader, extract_layer_index, fast_topk,
@@ -50,7 +50,7 @@ def custom_routing_function(
-        router_scores, router_indices = torch.topk(gating_output, topk, dim=-1)
+        router_scores, router_indices = fast_topk(gating_output, topk, dim=-1)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16801 - [BugFix] Accuracy fix for llama4 int4 - improperly casted scales

- Link: https://github.com/vllm-project/vllm/pull/16801
- Status/date: merged / 2025-04-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `7eb42556281d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +6/-9, 58 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Accuracy fix for llama4 int4 - improperly casted scales"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: A few MoE fixes: 1) Scales where fp16/bf16 but were being casted to float unsafely. Add a check by using `.data_ptr ()` 2) Fix `AttributeError: 'FusedMoE' object has no attribut....
- Key implementation: `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -51,8 +51,8 @@ def custom_routing_function(; symbols: custom_routing_function, __init__, touching `custom_routing_function, __init__`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -51,8 +51,8 @@ def custom_routing_function(; symbols: custom_routing_function, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -51,8 +51,8 @@ def custom_routing_function(
-        router_scores = torch.sigmoid(router_scores.float()).to(
-            hidden_states.dtype)
+        # psuedo-standard is that the router scores are floats
+        router_scores = torch.sigmoid(router_scores.float())
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16746 - [Bugfix] fix pp for llama4

- Link: https://github.com/vllm-project/vllm/pull/16746
- Status/date: merged / 2025-04-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `e31045f95ca0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix pp for llama4"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/mllama4.py`; PR body summary: This PR fixes PP for llama4 (https://github.com/vllm-project/vllm/issues/16385) Loading language model with architecture to support PP verification and correct the prefix to sep....
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +3/-3 (6 lines); hunks: -672,9 +672,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -824,7 +824,7 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +3/-3 (6 lines); hunks: -672,9 +672,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -824,7 +824,7 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: __init__, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16428 - Add chat template for Llama 4 models

- Link: https://github.com/vllm-project/vllm/pull/16428
- Status/date: merged / 2025-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_llama4_json.jinja`; associated commits `05e1fbfc52ca`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +139/-1, 172 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add chat template for Llama 4 models"; model line: Llama 4; category: model support/runtime entry; main diff: `examples/tool_chat_template_llama4_json.jinja`; PR body summary: The main differences are that some token have changed, e.g. start_header_id was changed to header_start. Now the models also support multiple tool calls although one of our para....
- Key implementation: `examples/tool_chat_template_llama4_json.jinja` added +116/-0 (116 lines); hunks: -0,0 +1,116.
- Code diff details:
  - `examples/tool_chat_template_llama4_json.jinja` added +116/-0 (116 lines); hunks: -0,0 +1,116
- Key code excerpts:

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

- Reviewed files:
  - docs: `examples/tool_chat_template_llama4_json.jinja` added +116/-0
- Risk and verification: The diff ships test coverage in `tests/tool_use/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17315 - [model] make llama4 compatible with pure dense layers

- Link: https://github.com/vllm-project/vllm/pull/17315
- Status/date: merged / 2025-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `b4ac4fa04da1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[model] make llama4 compatible with pure dense layers"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: make llama4 compatible with pure dense layers when interleave_moe_layer_step == 0 (no moe layers).
- Key implementation: `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -273,8 +273,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -273,8 +273,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -273,8 +273,8 @@ def __init__(
-        is_moe_layer = (self.layer_idx +
-                        1) % config.interleave_moe_layer_step == 0
+        is_moe_layer = config.interleave_moe_layer_step > 0 and (
+            self.layer_idx + 1) % config.interleave_moe_layer_step == 0
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17917 - [Frontend][Bug Fix] Update llama4 pythonic jinja template and llama4_pythonic parser

- Link: https://github.com/vllm-project/vllm/pull/17917
- Status/date: merged / 2025-05-22
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_llama4_pythonic.jinja`; associated commits `c91fe7b1b9c4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +541/-72, 720 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend][Bug Fix] Update llama4 pythonic jinja template and llama4_pythonic parser"; model line: Llama 4; category: bug fix; main diff: `examples/tool_chat_template_llama4_pythonic.jinja`; PR body summary: Change the llama4 pythonic template and small fix on the edge case where llama4 model may output ` ` unexpectedly. BFCL test result: | Name | reported | Base_vllm | Pythonic_vll....
- Key implementation: `examples/tool_chat_template_llama4_pythonic.jinja` modified +36/-64 (100 lines); hunks: -1,85 +1,51; -91,10 +57,12.
- Code diff details:
  - `examples/tool_chat_template_llama4_pythonic.jinja` modified +36/-64 (100 lines); hunks: -1,85 +1,51; -91,10 +57,12
- Key code excerpts:

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

- Reviewed files:
  - docs: `examples/tool_chat_template_llama4_pythonic.jinja` modified +36/-64
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/tool_parsers/test_llama4_pythonic_tool_parser.py`, `tests/tool_use/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18368 - [Model] enable data parallel for Llama4 vision encoder

- Link: https://github.com/vllm-project/vllm/pull/18368
- Status/date: merged / 2025-06-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `ebb1ec931871`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +214/-68, 496 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] enable data parallel for Llama4 vision encoder"; model line: Llama 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/mllama4.py`; PR body summary: **Summary:** Llama4 vision encoder in dp8 is ~3x as fast as in tp8, especially when handling a large number of input images (eg. 9 images per request). Add an enable_vision_enco....
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +167/-68 (235 lines); hunks: -34,6 +34,7; -49,6 +50,7; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, pixel_shuffle, touching `Llama4ImagePatchInputs, Llama4VisionMLP, __init__`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +167/-68 (235 lines); hunks: -34,6 +34,7; -49,6 +50,7; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, pixel_shuffle
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +167/-68
- Risk and verification: Runtime changes concentrate in `vllm/config.py`, `vllm/engine/arg_utils.py`, `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19997 - [Llama4] Update `attn_temperature_tuning`

- Link: https://github.com/vllm-project/vllm/pull/19997
- Status/date: merged / 2025-06-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `1afa9948f593`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Llama4] Update `attn_temperature_tuning`"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: Since https://github.com/huggingface/transformers/pull/37501 landed It's the below on HF now: So we no longer need this comment Start the model on TP=8 on H100, loaded fine Comm....
- Key implementation: `vllm/model_executor/models/llama4.py` modified +1/-2 (3 lines); hunks: -148,9 +148,8 @@ def __init__(self,; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +1/-2 (3 lines); hunks: -148,9 +148,8 @@ def __init__(self,; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -148,9 +148,8 @@ def __init__(self,
-        # TODO: attn_temperature_tuning should be a bool in huggingface
-            config.attn_temperature_tuning > 0
+            config.attn_temperature_tuning
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20419 - Enable ModelOpt Llama4 fp8 checkpoint deployment

- Link: https://github.com/vllm-project/vllm/pull/20419
- Status/date: merged / 2025-07-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/mllama4.py`; associated commits `4afe687a8291`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +501/-35, 693 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable ModelOpt Llama4 fp8 checkpoint deployment"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +144/-20 (164 lines); hunks: -717,6 +717,7 @@ class Llama4ForConditionalGeneration(nn.Module, SupportsMult...; -902,32 +903,109 @@ def _consolidate_qkv_weights(; symbols: Llama4ForConditionalGeneration, _consolidate_qkv_weights, load_weights, _rename_weight_for_modelopt_checkpoint, touching `Llama4ForConditionalGeneration, _consolidate_qkv_weights, load_weights`; `vllm/model_executor/models/llama4.py` modified +55/-4 (59 lines); hunks: -35,7 +35,8; -432,12 +433,24 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +144/-20 (164 lines); hunks: -717,6 +717,7 @@ class Llama4ForConditionalGeneration(nn.Module, SupportsMult...; -902,32 +903,109 @@ def _consolidate_qkv_weights(; symbols: Llama4ForConditionalGeneration, _consolidate_qkv_weights, load_weights, _rename_weight_for_modelopt_checkpoint
  - `vllm/model_executor/models/llama4.py` modified +55/-4 (59 lines); hunks: -35,7 +35,8; -432,12 +433,24 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +144/-20; `vllm/model_executor/models/llama4.py` modified +55/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/model_loader/weight_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20591 - [Meta] Llama4 EAGLE Support

- Link: https://github.com/vllm-project/vllm/pull/20591
- Status/date: merged / 2025-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4_eagle.py`; associated commits `c11013db8b76`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +258/-18, 363 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Meta] Llama4 EAGLE Support"; model line: Llama 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/llama4_eagle.py`; PR body summary: Support EAGLE speculative decoding with dense-only draft model for Llama4, using official Meta based support Original Author: @zixi-qi Ran with a uploaded scout based eagle to t....
- Key implementation: `vllm/model_executor/models/llama4_eagle.py` added +214/-0 (214 lines); hunks: -0,0 +1,214; symbols: LlamaModel, __init__, forward, load_weights, touching `LlamaModel, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/llama4_eagle.py` added +214/-0 (214 lines); hunks: -0,0 +1,214; symbols: LlamaModel, __init__, forward, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4_eagle.py` added +214/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/models/test_initialization.py`, `tests/v1/e2e/test_spec_decode.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21499 - [NVIDIA] Fix Llama4 Scout FP4 functionality issues

- Link: https://github.com/vllm-project/vllm/pull/21499
- Status/date: merged / 2025-07-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `ff08e51940a7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +219/-70, 432 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NVIDIA] Fix Llama4 Scout FP4 functionality issues"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: Fix the weight loading issues and accuray issues when using the NVIDIA ModelOpt Llama4 Scout FP4 model..
- Key implementation: `vllm/model_executor/models/llama4.py` modified +205/-67 (272 lines); hunks: -342,34 +342,94 @@ def load_moe_expert_weights(; -382,6 +442,9 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights, touching `load_moe_expert_weights, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +205/-67 (272 lines); hunks: -342,34 +342,94 @@ def load_moe_expert_weights(; -382,6 +442,9 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +205/-67
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20788 - [Meta] Official Eagle mm support, first enablement on llama4

- Link: https://github.com/vllm-project/vllm/pull/20788
- Status/date: merged / 2025-07-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/llama4_eagle.py`; associated commits `9e0726e5bfd2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +206/-37, 487 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Meta] Official Eagle mm support, first enablement on llama4"; model line: Llama 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/llama4.py`; PR body summary: Enable MM inference for EAGLE, targeting mllama4 in this PR but generally easy to extend to other models Issue with this PR: MM chunked prefill needs to be disabled, or set mbnt....
- Key implementation: `vllm/model_executor/models/llama4_eagle.py` modified +31/-4 (35 lines); hunks: -37,8 +37,9; -78,15 +79,23 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, load_weights, touching `__init__, get_input_embeddings, forward`; `vllm/model_executor/models/llama4.py` modified +1/-0 (1 lines); hunks: -256,6 +256,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/llama4_eagle.py` modified +31/-4 (35 lines); hunks: -37,8 +37,9; -78,15 +79,23 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, load_weights
  - `vllm/model_executor/models/llama4.py` modified +1/-0 (1 lines); hunks: -256,6 +256,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4_eagle.py` modified +31/-4; `vllm/model_executor/models/llama4.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/v1/e2e/test_spec_decode.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22107 - [Fix] Fix llama4 modelopt weight loading error

- Link: https://github.com/vllm-project/vllm/pull/22107
- Status/date: merged / 2025-08-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `337eb23bcca6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-4, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Fix llama4 modelopt weight loading error"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/mllama4.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +5/-4 (9 lines); hunks: -906,11 +906,13 @@ def _consolidate_qkv_weights(; -929,15 +931,14 @@ def _rename_weight_for_modelopt_checkpoint(self, name: str...; symbols: _consolidate_qkv_weights, _rename_weight_for_modelopt_checkpoint, touching `_consolidate_qkv_weights, _rename_weight_for_modelopt_checkpoint`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +5/-4 (9 lines); hunks: -906,11 +906,13 @@ def _consolidate_qkv_weights(; -929,15 +931,14 @@ def _rename_weight_for_modelopt_checkpoint(self, name: str...; symbols: _consolidate_qkv_weights, _rename_weight_for_modelopt_checkpoint
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +5/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22511 - Fix Llama4 FlashInfer FP4 MoE issues

- Link: https://github.com/vllm-project/vllm/pull/22511
- Status/date: merged / 2025-08-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +9/-5, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Llama4 FlashInfer FP4 MoE issues"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`; PR body summary: FlashInfer cutlass FP4 MoE already support Llama4, so remove unncessary asserts and set group to 0 when not used..
- Key implementation: `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py` modified +6/-1 (7 lines); hunks: -60,7 +60,12 @@ def prepare(; symbols: prepare, touching `prepare`; `vllm/model_executor/layers/quantization/modelopt.py` modified +3/-2 (5 lines); hunks: -1299,8 +1299,9 @@ def apply(; symbols: apply, touching `apply`; `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` modified +0/-2 (2 lines); hunks: -170,8 +170,6 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py` modified +6/-1 (7 lines); hunks: -60,7 +60,12 @@ def prepare(; symbols: prepare
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +3/-2 (5 lines); hunks: -1299,8 +1299,9 @@ def apply(; symbols: apply
  - `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` modified +0/-2 (2 lines); hunks: -170,8 +170,6 @@ def apply(; symbols: apply
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py` modified +6/-1; `vllm/model_executor/layers/quantization/modelopt.py` modified +3/-2; `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` modified +0/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22701 - Fix cuda illegal mem access with Llama4 TP8 + rms_norm custom op

- Link: https://github.com/vllm-project/vllm/pull/22701
- Status/date: merged / 2025-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `4f0f844b1675`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-2, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix cuda illegal mem access with Llama4 TP8 + rms_norm custom op"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: The rms_norm custom op cuda kernel supports non-contiguous striding on the "num_tokens" dim, but it assumes that all the num_tokens dims (if there are more than one num_tokens d....
- Key implementation: `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -224,10 +224,14 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -224,10 +224,14 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +6/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22691 - [bug fix] Fix llama4 spec decoding

- Link: https://github.com/vllm-project/vllm/pull/22691
- Status/date: merged / 2025-08-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `5bfe0dea7a34`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-2, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bug fix] Fix llama4 spec decoding"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +4/-2 (6 lines); hunks: -195,7 +195,9 @@ def __init__(self,; -206,7 +208,7 @@ def __init__(self,; symbols: __init__, _get_attn_scale, touching `__init__, _get_attn_scale`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +4/-2 (6 lines); hunks: -195,7 +195,9 @@ def __init__(self,; -206,7 +208,7 @@ def __init__(self,; symbols: __init__, _get_attn_scale
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +4/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22021 - Migrate Llama4ImagePatchInputs to TensorSchema

- Link: https://github.com/vllm-project/vllm/pull/22021
- Status/date: merged / 2025-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `f32a5bc5058a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-18, 87 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Migrate Llama4ImagePatchInputs to TensorSchema"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/mllama4.py`; PR body summary: This PR migrates Llama4ImagePatchInputs from a TypedDict-based definition to a structured TensorSchema model with runtime shape validation. This brings it in line with recent ch....
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +23/-18 (41 lines); hunks: -19,7 +19,7; -53,35 +53,42; symbols: Llama4ImagePatchInputs, _call_hf_processor, _parse_and_validate_image_input, touching `Llama4ImagePatchInputs, _call_hf_processor, _parse_and_validate_image_input`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +23/-18 (41 lines); hunks: -19,7 +19,7; -53,35 +53,42; symbols: Llama4ImagePatchInputs, _call_hf_processor, _parse_and_validate_image_input
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +23/-18
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24444 - [Bugfix] Fix platform-specific routing in CustomOp implementations

- Link: https://github.com/vllm-project/vllm/pull/24444
- Status/date: merged / 2025-09-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +53/-30, 187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix platform-specific routing in CustomOp implementations"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/layers/rotary_embedding/mrope.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py`; PR body summary: Some layers inheriting from `CustomOp` are overwriting the `forward` method, effectively disabling the platform routing logic defined in `CustomOp.dispatch_forward()`, resulting....
- Key implementation: `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +0/-23 (23 lines); hunks: -8,7 +8,6; -202,28 +201,6 @@ def __init__(; symbols: __init__, forward, forward_native, touching `__init__, forward, forward_native`; `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +10/-1 (11 lines); hunks: -88,7 +88,7 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; -129,3 +129,12 @@ def forward(; symbols: _compute_cos_sin_cache, forward, forward_native, forward_cuda, touching `_compute_cos_sin_cache, forward, forward_native`; `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` modified +10/-1 (11 lines); hunks: -111,7 +111,7 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; -161,6 +161,15 @@ def forward(; symbols: _compute_cos_sin_cache, forward, forward_native, forward_cuda, touching `_compute_cos_sin_cache, forward, forward_native`; `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` modified +9/-1 (10 lines); hunks: -12,7 +12,7; -70,3 +70,11 @@ def forward(; symbols: Ernie4_5_VLRotaryEmbedding, forward, forward_native, forward_cuda, touching `Ernie4_5_VLRotaryEmbedding, forward, forward_native`.
- Code diff details:
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +0/-23 (23 lines); hunks: -8,7 +8,6; -202,28 +201,6 @@ def __init__(; symbols: __init__, forward, forward_native
  - `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +10/-1 (11 lines); hunks: -88,7 +88,7 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; -129,3 +129,12 @@ def forward(; symbols: _compute_cos_sin_cache, forward, forward_native, forward_cuda
  - `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` modified +10/-1 (11 lines); hunks: -111,7 +111,7 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; -161,6 +161,15 @@ def forward(; symbols: _compute_cos_sin_cache, forward, forward_native, forward_cuda
  - `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` modified +9/-1 (10 lines); hunks: -12,7 +12,7; -70,3 +70,11 @@ def forward(; symbols: Ernie4_5_VLRotaryEmbedding, forward, forward_native, forward_cuda
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +8/-1 (9 lines); hunks: -1593,7 +1593,7 @@ def maybe_all_reduce_tensor_model_parallel(; -1627,6 +1627,13 @@ def forward(; symbols: maybe_all_reduce_tensor_model_parallel, forward, forward_native, forward_cuda
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/mrope.py
@@ -8,7 +8,6 @@
-from vllm.platforms import current_platform
@@ -202,28 +201,6 @@ def __init__(
-        self.use_triton = current_platform.is_cuda_alike()
-    def forward(
-        self,
-        positions: torch.Tensor,
diff -- vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py
@@ -88,7 +88,7 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:
-    def forward(
+    def forward_native(
@@ -129,3 +129,12 @@ def forward(
+    def forward_cuda(
+        self,
+        positions: torch.Tensor,
diff -- vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py
@@ -111,7 +111,7 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +0/-23; `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +10/-1; `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` modified +10/-1; `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` modified +9/-1; `vllm/model_executor/layers/fused_moe/layer.py` modified +8/-1; `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +8/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/activation.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25889 - [Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding`

- Link: https://github.com/vllm-project/vllm/pull/25889
- Status/date: merged / 2025-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; associated commits `43b752c325d5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-1, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding`"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; PR body summary: Fix #25888. Especially, this PR roll back the misplaced dtype cast introduced by #21126 which deteriorates the vision understanding performance of llama 4 family. Note that the....
- Key implementation: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +3/-1 (4 lines); hunks: -59,7 +59,9 @@ def forward_native( # type: ignore[override]; symbols: forward_native, touching `forward_native`.
- Code diff details:
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +3/-1 (4 lines); hunks: -59,7 +59,9 @@ def forward_native( # type: ignore[override]; symbols: forward_native
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py
@@ -59,7 +59,9 @@ def forward_native(  # type: ignore[override]
-        self._match_cos_sin_cache_dtype(query)
+        # self.cos_sin_cache here is complex tensor so we cannot cast into
+        # query's dtype directly with self._match_cos_sin_cache_dtype
+        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(query.device)
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25961 - Support llama3 eagle3 head with llama4 verifier

- Link: https://github.com/vllm-project/vllm/pull/25961
- Status/date: merged / 2025-10-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `05f6846ede18`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +83/-8, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support llama3 eagle3 head with llama4 verifier"; model line: Llama 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/mllama4.py`; PR body summary: This PR enables Eagle3 speculative decoding with Llama3 drafter and Llama4 multimodal verifier support, with configurable auxiliary hidden state layers. Key Features - **Compati....
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +25/-2 (27 lines); hunks: -64,7 +64,12; -717,7 +722,9 @@ def get_dummy_mm_data(; symbols: get_dummy_mm_data, Llama4ForConditionalGeneration, __init__, set_aux_hidden_state_layers, touching `get_dummy_mm_data, Llama4ForConditionalGeneration, __init__`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +25/-2 (27 lines); hunks: -64,7 +64,12; -717,7 +722,9 @@ def get_dummy_mm_data(; symbols: get_dummy_mm_data, Llama4ForConditionalGeneration, __init__, set_aux_hidden_state_layers
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +25/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama.py`, `vllm/model_executor/models/llama_eagle3.py`, `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26790 - llama4_vision_rope: add HIP override to accept (q, k) and avoid (positions, q, k) mismatch

- Link: https://github.com/vllm-project/vllm/pull/26790
- Status/date: merged / 2025-10-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; associated commits `87efc681dbd5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-0, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "llama4_vision_rope: add HIP override to accept (q, k) and avoid (positions, q, k) mismatch"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; PR body summary: On ROCm/HIP, the base RoPE path expects `(positions, query, key)` and, in the fallback branch, calls `forward_cuda(positions, query, key)`. The Llama4‑Vision RoPE implements `fo....
- Key implementation: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +7/-0 (7 lines); hunks: -78,3 +78,10 @@ def forward_cuda( # type: ignore[override]; symbols: forward_cuda, forward_hip, touching `forward_cuda, forward_hip`.
- Code diff details:
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +7/-0 (7 lines); hunks: -78,3 +78,10 @@ def forward_cuda( # type: ignore[override]; symbols: forward_cuda, forward_hip
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27136 - [Fix][Spec Decode] Fix llama4 draft loading with different quantization

- Link: https://github.com/vllm-project/vllm/pull/27136
- Status/date: merged / 2025-10-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4_eagle.py`; associated commits `be4445072c4e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +17/-10, 34 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix][Spec Decode] Fix llama4 draft loading with different quantization"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4_eagle.py`; PR body summary: This PR takes care of the case where draft model and target model may use different quantization configs. `replace` doesn't work with pydantic dataclass `deepcopy` fails with ne....
- Key implementation: `vllm/model_executor/models/llama4_eagle.py` modified +17/-10 (27 lines); hunks: -60,16 +60,23 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/llama4_eagle.py` modified +17/-10 (27 lines); hunks: -60,16 +60,23 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4_eagle.py` modified +17/-10
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4_eagle.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25145 - [XPU][bugfix] fix rope for llama4 and deepseek

- Link: https://github.com/vllm-project/vllm/pull/25145
- Status/date: merged / 2025-10-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; associated commits `b798e39f931a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +22/-32, 116 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU][bugfix] fix rope for llama4 and deepseek"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; PR body summary: Fix more dispatch issue on xpu introduced in https://github.com/vllm-project/vllm/pull/24444.
- Key implementation: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +2/-9 (11 lines); hunks: -5,10 +5,10; -78,10 +78,3 @@ def forward_cuda( # type: ignore[override]; symbols: Llama4VisionRotaryEmbedding, __init__, forward_cuda, forward_hip, touching `Llama4VisionRotaryEmbedding, __init__, forward_cuda`.
- Code diff details:
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +2/-9 (11 lines); hunks: -5,10 +5,10; -78,10 +78,3 @@ def forward_cuda( # type: ignore[override]; symbols: Llama4VisionRotaryEmbedding, __init__, forward_cuda, forward_hip
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +2/-9
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/rotary_embedding/base.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28602 - LLaMA4 LoRA Adapter Enablement

- Link: https://github.com/vllm-project/vllm/pull/28602
- Status/date: merged / 2025-11-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `964d65deedb9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +34/-2, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "LLaMA4 LoRA Adapter Enablement"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/models/mllama4.py`; PR body summary: Adds LoRA Adapter Support for LLaMA 4 Architecture. Tested with custom LoRA and with vLLM Bench. Tests showed some performance degradation (TTFT and ITL) with concurrent use of....
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +34/-2 (36 lines); hunks: -35,6 +35,7; -45,6 +46,7; symbols: get_dummy_mm_data, Llama4ForConditionalGeneration, _load_other_weights, get_expert_mapping, touching `get_dummy_mm_data, Llama4ForConditionalGeneration, _load_other_weights`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +34/-2 (36 lines); hunks: -35,6 +35,7; -45,6 +46,7; symbols: get_dummy_mm_data, Llama4ForConditionalGeneration, _load_other_weights, get_expert_mapping
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +34/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28577 - [BugFix] Fix Llama4 Pipeline Parallelism Assert Error

- Link: https://github.com/vllm-project/vllm/pull/28577
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `dc45efc8ef7f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-0, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Fix Llama4 Pipeline Parallelism Assert Error"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: Summary: When enabling PP for Llama4 model, each rank has a few `PPMissingLayer()`. The line `assert isinstance(layer, Llama4DecoderLayer)` raises assert error when encountering....
- Key implementation: `vllm/model_executor/models/llama4.py` modified +7/-0 (7 lines); hunks: -54,6 +54,7; -738,6 +739,9 @@ def set_moe_parameters(self):; symbols: set_moe_parameters, update_physical_experts_metadata, touching `set_moe_parameters, update_physical_experts_metadata`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +7/-0 (7 lines); hunks: -54,6 +54,7; -738,6 +739,9 @@ def set_moe_parameters(self):; symbols: set_moe_parameters, update_physical_experts_metadata
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29926 - [Bugfix][llama4_eagle] Fix missing 'lm_head' attribute

- Link: https://github.com/vllm-project/vllm/pull/29926
- Status/date: merged / 2025-12-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4_eagle.py`; associated commits `962d703818c0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +16/-3, 46 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][llama4_eagle] Fix missing 'lm_head' attribute"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4_eagle.py`; PR body summary: This PR: 1. Fixes llama4_eagle model by adding the missing `lm_head` attribute to the model class. 2. Fixes `test_spec_decode.py::test_eagle_correctness[FLASH_ATTN-llama4_eagle]....
- Key implementation: `vllm/model_executor/models/llama4_eagle.py` modified +11/-2 (13 lines); hunks: -28,7 +28,10; -182,6 +185,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, transform, touching `__init__, transform`.
- Code diff details:
  - `vllm/model_executor/models/llama4_eagle.py` modified +11/-2 (13 lines); hunks: -28,7 +28,10; -182,6 +185,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, transform
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4_eagle.py` modified +11/-2
- Risk and verification: The diff ships test coverage in `tests/v1/e2e/test_spec_decode.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30709 - [Misc][LLaMa4] Compile LLaMa Vision Encoder

- Link: https://github.com/vllm-project/vllm/pull/30709
- Status/date: merged / 2026-01-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`, `vllm/model_executor/models/mllama4.py`; associated commits `ea6d067a2aeb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +85/-20, 202 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc][LLaMa4] Compile LLaMa Vision Encoder"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; PR body summary: We want to speedup up inference for mllama4 by applying `torch.compile` to the intensive workload, similar to what is done in #23207. We start by enabling the VisionEncoder + Pi....
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +29/-9 (38 lines); hunks: -31,9 +31,11; -47,6 +49,7; symbols: forward, Llama4VisionModel, __init__, touching `forward, Llama4VisionModel, __init__`; `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +5/-2 (7 lines); hunks: -60,14 +60,17 @@ def forward_native( # type: ignore[override]; symbols: forward_native, touching `forward_native`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +29/-9 (38 lines); hunks: -31,9 +31,11; -47,6 +49,7; symbols: forward, Llama4VisionModel, __init__
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +5/-2 (7 lines); hunks: -60,14 +60,17 @@ def forward_native( # type: ignore[override]; symbols: forward_native
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +29/-9; `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +5/-2
- Risk and verification: The diff ships test coverage in `tests/compile/fullgraph/test_multimodal_compile.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #32886 - [Bugfix] Fix FP8 MoE EP Weight Loading for ModelOpt Llama4

- Link: https://github.com/vllm-project/vllm/pull/32886
- Status/date: merged / 2026-01-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `1fb648bf107e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +21/-1, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix FP8 MoE EP Weight Loading for ModelOpt Llama4"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: 32862 Add a version-guarded fallback in Llama4 MoE weight loading to avoid CPU FP8 indexing on older PyTorch releases. For torch.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +21/-1 (22 lines); hunks: -51,6 +51,8; -504,7 +506,25 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, touching `load_moe_expert_weights`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +21/-1 (22 lines); hunks: -51,6 +51,8; -504,7 +506,25 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +21/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34243 - [Bugfix] Enable attn quantization of Llama-4 by correctly permuting scales for rope (int8, fp8)

- Link: https://github.com/vllm-project/vllm/pull/34243
- Status/date: merged / 2026-02-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `11c7ace34061`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +29/-5, 76 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Enable attn quantization of Llama-4 by correctly permuting scales for rope (int8, fp8)"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: Llama-4 weights of `q/k_proj` are permuted during model loading to prepare the model for interleaved/gpt-neox rope. The same permutation needs to be applied on quantization weig....
- Key implementation: `vllm/model_executor/models/llama4.py` modified +29/-5 (34 lines); hunks: -44,6 +44,9; -829,11 +832,20 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute, touching `permute_qk_weight_for_rotary, permute`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +29/-5 (34 lines); hunks: -44,6 +44,9; -829,11 +832,20 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +29/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34471 - [Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers

- Link: https://github.com/vllm-project/vllm/pull/34471
- Status/date: merged / 2026-02-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `ee1d25f199ee`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +29/-68, 114 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: This PR resolves a couple of issues in loading of Llama-4 models with quantized self-attention layers. Current logic hardcodes support only for NVFP4 modelopt ckpts, and for INT....
- Key implementation: `vllm/model_executor/models/llama4.py` modified +29/-68 (97 lines); hunks: -44,9 +44,6; -831,74 +828,38 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute, touching `permute_qk_weight_for_rotary, permute`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +29/-68 (97 lines); hunks: -44,9 +44,6; -831,74 +828,38 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +29/-68
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34997 - Revert "[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers "

- Link: https://github.com/vllm-project/vllm/pull/34997
- Status/date: merged / 2026-02-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `0e22cd618b5d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +68/-29, 114 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers ""; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: Reverts vllm-project/vllm#34471 to fix CI failures: https://github.com/vllm-project/vllm/issues/34995 FIX: https://github.com/vllm-project/vllm/issues/34995 cc @eldarkurtic.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +68/-29 (97 lines); hunks: -44,6 +44,9; -828,38 +831,74 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute, touching `permute_qk_weight_for_rotary, permute`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +68/-29 (97 lines); hunks: -44,6 +44,9; -828,38 +831,74 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +68/-29
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35033 - [Llama4,CI] Bring back Llama-4 bug fixes, and also fix Maverick tests

- Link: https://github.com/vllm-project/vllm/pull/35033
- Status/date: merged / 2026-02-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `1e8438a89a64`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +31/-70, 127 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Llama4,CI] Bring back Llama-4 bug fixes, and also fix Maverick tests"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; PR body summary: Bring back reverted Llama-4 bug fix changes from https://github.com/vllm-project/vllm/pull/34471 (reverted with https://github.com/vllm-project/vllm/pull/34997) and also fix the....
- Key implementation: `vllm/model_executor/models/llama4.py` modified +29/-68 (97 lines); hunks: -44,9 +44,6; -831,74 +828,38 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute, touching `permute_qk_weight_for_rotary, permute`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +29/-68 (97 lines); hunks: -44,9 +44,6; -831,74 +828,38 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +29/-68
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_maverick.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35147 - [Feature] Add LoRA tower/connector support for Llama 4 Vision (mllama4)

- Link: https://github.com/vllm-project/vllm/pull/35147
- Status/date: merged / 2026-02-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `012dee92331c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-1, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Add LoRA tower/connector support for Llama 4 Vision (mllama4)"; model line: Llama 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/mllama4.py`; PR body summary: Enable LoRA adapters for the vision tower and connector of Llama 4 Vision (`Llama4ForConditionalGeneration` / `mllama4.py`), as part of #31479. Previously, LoRA could only be ap....
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +23/-1 (24 lines); hunks: -1151,6 +1151,28 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens, touching `get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +23/-1 (24 lines); hunks: -1151,6 +1151,28 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +23/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
