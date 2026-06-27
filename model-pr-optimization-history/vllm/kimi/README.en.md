# vllm Kimi K2/K2.5/Linear/VL Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/reasoning/test_kimi_k2_reasoning_parser.py` | [#37438](https://github.com/vllm-project/vllm/pull/37438), [#41068](https://github.com/vllm-project/vllm/pull/41068) |
| `tests/tool_parsers/test_kimi_k2_tool_parser.py` | [#31207](https://github.com/vllm-project/vllm/pull/31207), [#38579](https://github.com/vllm-project/vllm/pull/38579) |
| `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` | no direct PR-number commit |
| `vllm/model_executor/models/kimi_audio.py` | [#36127](https://github.com/vllm-project/vllm/pull/36127), [#36903](https://github.com/vllm-project/vllm/pull/36903) |
| `vllm/model_executor/models/kimi_k25.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131), [#33320](https://github.com/vllm-project/vllm/pull/33320), [#33346](https://github.com/vllm-project/vllm/pull/33346), [#33562](https://github.com/vllm-project/vllm/pull/33562), [#33876](https://github.com/vllm-project/vllm/pull/33876), [#34427](https://github.com/vllm-project/vllm/pull/34427), [#34501](https://github.com/vllm-project/vllm/pull/34501), [#36192](https://github.com/vllm-project/vllm/pull/36192), [#36361](https://github.com/vllm-project/vllm/pull/36361), [#37693](https://github.com/vllm-project/vllm/pull/37693), [#39344](https://github.com/vllm-project/vllm/pull/39344), [#42869](https://github.com/vllm-project/vllm/pull/42869), ... (13 total) |
| `vllm/model_executor/models/kimi_k25_vit.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131), [#33346](https://github.com/vllm-project/vllm/pull/33346), [#34501](https://github.com/vllm-project/vllm/pull/34501), [#42081](https://github.com/vllm-project/vllm/pull/42081), [#44493](https://github.com/vllm-project/vllm/pull/44493) |
| `vllm/model_executor/models/kimi_linear.py` | [#27809](https://github.com/vllm-project/vllm/pull/27809), [#27834](https://github.com/vllm-project/vllm/pull/27834), [#27885](https://github.com/vllm-project/vllm/pull/27885), [#37371](https://github.com/vllm-project/vllm/pull/37371) |
| `vllm/model_executor/models/kimi_vl.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387), [#16833](https://github.com/vllm-project/vllm/pull/16833), [#17156](https://github.com/vllm-project/vllm/pull/17156), [#21769](https://github.com/vllm-project/vllm/pull/21769), [#23114](https://github.com/vllm-project/vllm/pull/23114), [#23817](https://github.com/vllm-project/vllm/pull/23817), [#31738](https://github.com/vllm-project/vllm/pull/31738), [#41992](https://github.com/vllm-project/vllm/pull/41992) |
| `vllm/model_executor/models/moonvit.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387), [#23817](https://github.com/vllm-project/vllm/pull/23817), [#29309](https://github.com/vllm-project/vllm/pull/29309), [#31738](https://github.com/vllm-project/vllm/pull/31738), [#41992](https://github.com/vllm-project/vllm/pull/41992) |
| `vllm/reasoning/kimi_k2_reasoning_parser.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131), [#33646](https://github.com/vllm-project/vllm/pull/33646), [#41068](https://github.com/vllm-project/vllm/pull/41068) |
| `vllm/tokenizers/kimi_audio.py` | [#36127](https://github.com/vllm-project/vllm/pull/36127) |
| `vllm/tool_parsers/kimi_k2_tool_parser.py` | [#31207](https://github.com/vllm-project/vllm/pull/31207), [#38579](https://github.com/vllm-project/vllm/pull/38579) |
| `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja` | [#36127](https://github.com/vllm-project/vllm/pull/36127) |
| `vllm/transformers_utils/configs/kimi_k25.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131) |
| `vllm/transformers_utils/configs/kimi_linear.py` | [#27809](https://github.com/vllm-project/vllm/pull/27809) |
| `vllm/transformers_utils/configs/kimi_vl.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387) |
| `vllm/transformers_utils/configs/moonvit.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387) |
| `vllm/transformers_utils/processors/kimi_audio.py` | [#36127](https://github.com/vllm-project/vllm/pull/36127) |
| `vllm/transformers_utils/processors/kimi_k25.py` | [#37693](https://github.com/vllm-project/vllm/pull/37693) |

## PR Coverage Summary

- Git-traced PRs: 34
- Extra PRs preserved from existing docs: 7
- Total PRs in this document: 41
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
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
| 2025-12-15 | [#30125](https://github.com/vllm-project/vllm/pull/30125) | merged | [CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it. | `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `vllm/attention/layers/mm_encoder_attention.py`, `vllm/model_executor/models/qwen2_vl.py` |
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
| 2026-05-04 | [#41068](https://github.com/vllm-project/vllm/pull/41068) | merged | [Bugfix] KimiK2ReasoningParser: guard against buffered end-token in streaming | `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py` |
| 2026-05-11 | [#42081](https://github.com/vllm-project/vllm/pull/42081) | merged | [Bug] Fix kimi dtype issue with `mm_projector_forward` | `vllm/model_executor/models/kimi_k25_vit.py` |
| 2026-05-14 | [#41778](https://github.com/vllm-project/vllm/pull/41778) | merged | [MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell | `benchmarks/attention_benchmarks/configs/mla_prefill.yaml`, `benchmarks/attention_benchmarks/configs/mla_decode.yaml`, `vllm/model_executor/layers/attention/mla_attention.py` |
| 2026-05-18 | [#42869](https://github.com/vllm-project/vllm/pull/42869) | merged | [BugFix] Kimi-K2.5: skip vision tower dtype conversion when using quantization | `vllm/model_executor/models/kimi_k25.py` |
| 2026-05-22 | [#41126](https://github.com/vllm-project/vllm/pull/41126) | merged | [Attention] Mamba attention module refactor | `vllm/model_executor/models/olmo_hybrid.py`, `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` |
| 2026-05-29 | [#43857](https://github.com/vllm-project/vllm/pull/43857) | merged | Add vLLM library info to Hugging Face Hub requests | `vllm/model_executor/model_loader/weight_utils.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/tokenizers/grok2.py` |
| 2026-06-04 | [#44493](https://github.com/vllm-project/vllm/pull/44493) | merged | [Bugfix]Fix Kimi-K2.5 FlashInfer ViT metadata | `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py` |
| 2026-06-04 | [#44539](https://github.com/vllm-project/vllm/pull/44539) | merged | [mamba] unify KDA conv states into one cache to match 2-state SSM layout | `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `vllm/model_executor/models/kimi_linear.py` |
| 2026-06-12 | [#45003](https://github.com/vllm-project/vllm/pull/45003) | merged | [Frontend] Support strict mode for tool calling | `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py` |
| 2026-06-17 | [#41992](https://github.com/vllm-project/vllm/pull/41992) | merged | [MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py` |
| 2026-06-21 | [#45424](https://github.com/vllm-project/vllm/pull/45424) | merged | [Core] Ensure memory is pinned prior to async h2d copy | `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/pooler/seqwise/methods.py`, `vllm/multimodal/inputs.py` |

## Per-PR Diff Audit Cards

### PR #16387 - [Model][VLM] Add Kimi-VL model support

- Link: https://github.com/vllm-project/vllm/pull/16387
- Status/date: merged / 2025-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`, `vllm/transformers_utils/configs/kimi_vl.py`, `vllm/transformers_utils/configs/moonvit.py`; associated commits `b1308b84a3a6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +1436/-14, 1618 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][VLM] Add Kimi-VL model support"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `vllm/transformers_utils/configs/kimi_vl.py`; technical summary: Covers "[Model][VLM] Add Kimi-VL model support"; the main implementation surface is `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `vllm/transformers_utils/configs/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` added +628/-0 (628 lines); hunks: -0,0 +1,628; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope, touching `multihead_attention, sdpa_attention, _apply_rope_input_validation`; `vllm/model_executor/models/kimi_vl.py` added +608/-0 (608 lines); hunks: -0,0 +1,608; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward, touching `MaxImageTokenMeta, KimiVLMultiModalProjector, __init__`; `vllm/transformers_utils/configs/kimi_vl.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: KimiVLConfig, __init__, touching `KimiVLConfig, __init__`; `vllm/transformers_utils/configs/moonvit.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: MoonViTConfig, __init__, touching `MoonViTConfig, __init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` added +628/-0 (628 lines); hunks: -0,0 +1,628; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope
  - `vllm/model_executor/models/kimi_vl.py` added +608/-0 (608 lines); hunks: -0,0 +1,608; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward
  - `vllm/transformers_utils/configs/kimi_vl.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: KimiVLConfig, __init__
  - `vllm/transformers_utils/configs/moonvit.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: MoonViTConfig, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` added +628/-0; `vllm/model_executor/models/kimi_vl.py` added +608/-0; `vllm/transformers_utils/configs/kimi_vl.py` added +36/-0; `vllm/transformers_utils/configs/moonvit.py` added +32/-0
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16833 - [Misc] Clean up Kimi-VL

- Link: https://github.com/vllm-project/vllm/pull/16833
- Status/date: merged / 2025-04-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`; associated commits `aadb6565628c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +20/-44, 139 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Clean up Kimi-VL"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[Misc] Clean up Kimi-VL"; the main implementation surface is `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_vl.py` modified +17/-40 (57 lines); hunks: -56,7 +56,6; -70,22 +69,20; symbols: KimiVLProcessingInfo, get_hf_config, get_supported_mm_limits, get_num_image_tokens, touching `KimiVLProcessingInfo, get_hf_config, get_supported_mm_limits`.
- Code diff details:
  - `vllm/model_executor/models/kimi_vl.py` modified +17/-40 (57 lines); hunks: -56,7 +56,6; -70,22 +69,20; symbols: KimiVLProcessingInfo, get_hf_config, get_supported_mm_limits, get_num_image_tokens
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +17/-40
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17156 - fix float16 support for kimi-vl

- Link: https://github.com/vllm-project/vllm/pull/17156
- Status/date: merged / 2025-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`; associated commits `69bff9bc8934`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix float16 support for kimi-vl"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "fix float16 support for kimi-vl"; the main implementation surface is `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_vl.py` modified +1/-2 (3 lines); hunks: -340,8 +340,7 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, touching `_parse_and_validate_image_input`.
- Code diff details:
  - `vllm/model_executor/models/kimi_vl.py` modified +1/-2 (3 lines); hunks: -340,8 +340,7 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_vl.py
@@ -340,8 +340,7 @@ def _parse_and_validate_image_input(
-        # fp32 -> bf16
-        pixel_values = pixel_values.to(torch.bfloat16)
+        pixel_values = pixel_values.to(self.vision_tower.dtype)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21769 - Migrate KimiVLImagePixelInputs to TensorSchema

- Link: https://github.com/vllm-project/vllm/pull/21769
- Status/date: merged / 2025-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`; associated commits `05fae021750b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +15/-9, 55 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Migrate KimiVLImagePixelInputs to TensorSchema"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "Migrate KimiVLImagePixelInputs to TensorSchema"; the main implementation surface is `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_vl.py` modified +15/-9 (24 lines); hunks: -46,7 +46,7; -79,6 +79,7; symbols: forward, KimiVLImagePixelInputs, _parse_and_validate_image_input, touching `forward, KimiVLImagePixelInputs, _parse_and_validate_image_input`.
- Code diff details:
  - `vllm/model_executor/models/kimi_vl.py` modified +15/-9 (24 lines); hunks: -46,7 +46,7; -79,6 +79,7; symbols: forward, KimiVLImagePixelInputs, _parse_and_validate_image_input
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +15/-9
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23114 - [Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506

- Link: https://github.com/vllm-project/vllm/pull/23114
- Status/date: merged / 2025-08-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`; associated commits `fda9537c5e61`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-13, 77 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506"; the main implementation surface is `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_vl.py` modified +17/-12 (29 lines); hunks: -54,16 +54,16; -81,7 +81,7; symbols: get_replacement, KimiVLForConditionalGeneration, get_placeholder_str, __init__, touching `get_replacement, KimiVLForConditionalGeneration, get_placeholder_str`.
- Code diff details:
  - `vllm/model_executor/models/kimi_vl.py` modified +17/-12 (29 lines); hunks: -54,16 +54,16; -81,7 +81,7; symbols: get_replacement, KimiVLForConditionalGeneration, get_placeholder_str, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +17/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23817 - [Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506

- Link: https://github.com/vllm-project/vllm/pull/23817
- Status/date: merged / 2025-09-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`; associated commits `a0e0efd6bdcf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +157/-62, 478 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506"; the main implementation surface is `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` modified +55/-22 (77 lines); hunks: -42,7 +42,6; -55,6 +54,8; symbols: MLP2, __init__, forward, MoonVitEncoderLayer, touching `MLP2, __init__, forward`; `vllm/model_executor/models/kimi_vl.py` modified +39/-15 (54 lines); hunks: -56,6 +56,7; -76,6 +77,7; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward, touching `MaxImageTokenMeta, KimiVLMultiModalProjector, __init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` modified +55/-22 (77 lines); hunks: -42,7 +42,6; -55,6 +54,8; symbols: MLP2, __init__, forward, MoonVitEncoderLayer
  - `vllm/model_executor/models/kimi_vl.py` modified +39/-15 (54 lines); hunks: -56,6 +56,7; -76,6 +77,7; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +55/-22; `vllm/model_executor/models/kimi_vl.py` modified +39/-15
- Risk and verification: The diff ships test coverage in `tests/multimodal/test_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27809 - [Model] Introduce Kimi Linear to vLLM

- Link: https://github.com/vllm-project/vllm/pull/27809
- Status/date: merged / 2025-10-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py`; associated commits `4e68cc9b6aa2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +1326/-49, 1510 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Introduce Kimi Linear to vLLM"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py`; technical summary: Covers "[Model] Introduce Kimi Linear to vLLM"; the main implementation surface is `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_linear.py` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: KimiMLP, __init__, forward, KimiMoE, touching `KimiMLP, __init__, forward`; `vllm/transformers_utils/configs/kimi_linear.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: KimiLinearConfig, __init__, is_mla, is_moe, touching `KimiLinearConfig, __init__, is_mla`.
- Code diff details:
  - `vllm/model_executor/models/kimi_linear.py` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: KimiMLP, __init__, forward, KimiMoE
  - `vllm/transformers_utils/configs/kimi_linear.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: KimiLinearConfig, __init__, is_mla, is_moe
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_linear.py` added +663/-0; `vllm/transformers_utils/configs/kimi_linear.py` added +144/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27834 - [Kimi-Linear] Correct prefixes and add compatibility to AWQ quants

- Link: https://github.com/vllm-project/vllm/pull/27834
- Status/date: merged / 2025-10-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_linear.py`; associated commits `e5ef4dfc11ab`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kimi-Linear] Correct prefixes and add compatibility to AWQ quants"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_linear.py`; technical summary: Covers "[Kimi-Linear] Correct prefixes and add compatibility to AWQ quants"; the main implementation surface is `vllm/model_executor/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_linear.py` modified +2/-1 (3 lines); hunks: -155,6 +155,7 @@ def __init__(; -340,7 +341,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/kimi_linear.py` modified +2/-1 (3 lines); hunks: -155,6 +155,7 @@ def __init__(; -340,7 +341,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -155,6 +155,7 @@ def __init__(
+                prefix=f"{prefix}.shared_experts",
@@ -340,7 +341,7 @@ def __init__(
-                prefix=f"{prefix}.mlp",
+                prefix=f"{prefix}.block_sparse_moe",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_linear.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27885 - fix incorrect type annotation in KimiMLP

- Link: https://github.com/vllm-project/vllm/pull/27885
- Status/date: merged / 2025-10-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_linear.py`; associated commits `bc306fe5e978`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix incorrect type annotation in KimiMLP"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_linear.py`; technical summary: Covers "fix incorrect type annotation in KimiMLP"; the main implementation surface is `vllm/model_executor/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_linear.py` modified +1/-2 (3 lines); hunks: -22,7 +22,6; -61,7 +60,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_linear.py` modified +1/-2 (3 lines); hunks: -22,7 +22,6; -61,7 +60,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -22,7 +22,6 @@
-    QKVParallelLinear,
@@ -61,7 +60,7 @@ def __init__(
-        quant_config: QKVParallelLinear | None = None,
+        quant_config: QuantizationConfig | None = None,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_linear.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29309 - [XPU]fix Kimi-VL-A3B-thinking on xpu

- Link: https://github.com/vllm-project/vllm/pull/29309
- Status/date: merged / 2025-11-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/moonvit.py`; associated commits `3cfa63ad9916`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +14/-6, 52 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU]fix Kimi-VL-A3B-thinking on xpu"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/moonvit.py`; technical summary: Covers "[XPU]fix Kimi-VL-A3B-thinking on xpu"; the main implementation surface is `vllm/model_executor/models/moonvit.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` modified +14/-6 (20 lines); hunks: -56,10 +56,13; -106,10 +109,10 @@ def multihead_attention(; symbols: multihead_attention, Rope2DPosEmb, __init__, touching `multihead_attention, Rope2DPosEmb, __init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` modified +14/-6 (20 lines); hunks: -56,10 +56,13; -106,10 +109,10 @@ def multihead_attention(; symbols: multihead_attention, Rope2DPosEmb, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +14/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/moonvit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30125 - [CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it.

- Link: https://github.com/vllm-project/vllm/pull/30125
- Status/date: merged / 2025-12-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 24 files, +1264/-853, 3625 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it."; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `vllm/attention/layers/mm_encoder_attention.py`, `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it."; the main implementation surface is `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `vllm/attention/layers/mm_encoder_attention.py`, `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0 (434 lines); hunks: -0,0 +1,434; symbols: build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt, build_qwen2_5_video_prompt, touching `build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt`; `vllm/attention/layers/mm_encoder_attention.py` added +284/-0 (284 lines); hunks: -0,0 +1,284; symbols: maybe_get_vit_flash_attn_backend, MMEncoderAttention, __init__, enabled, touching `maybe_get_vit_flash_attn_backend, MMEncoderAttention, __init__`; `vllm/model_executor/models/qwen2_vl.py` modified +47/-96 (143 lines); hunks: -33,7 +33,6; -45,10 +44,8; symbols: __init__, split_qkv, forward, touching `__init__, split_qkv, forward`; `vllm/model_executor/models/glm4_1v.py` modified +46/-91 (137 lines); hunks: -47,8 +47,10; -191,10 +193,15 @@ def __init__(; symbols: __init__, split_qkv, forward, touching `__init__, split_qkv, forward`.
- Code diff details:
  - `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0 (434 lines); hunks: -0,0 +1,434; symbols: build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt, build_qwen2_5_video_prompt
  - `vllm/attention/layers/mm_encoder_attention.py` added +284/-0 (284 lines); hunks: -0,0 +1,284; symbols: maybe_get_vit_flash_attn_backend, MMEncoderAttention, __init__, enabled
  - `vllm/model_executor/models/qwen2_vl.py` modified +47/-96 (143 lines); hunks: -33,7 +33,6; -45,10 +44,8; symbols: __init__, split_qkv, forward
  - `vllm/model_executor/models/glm4_1v.py` modified +46/-91 (137 lines); hunks: -47,8 +47,10; -191,10 +193,15 @@ def __init__(; symbols: __init__, split_qkv, forward
  - `vllm/model_executor/models/dots_ocr.py` modified +46/-83 (129 lines); hunks: -5,15 +5,14; -254,11 +253,15 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- tests/models/multimodal/generation/test_vit_backend_functionality.py
@@ -0,0 +1,434 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Consolidated test for ViT attention backend functionality across multiple models.
+This test validates that each multimodal model can successfully generate outputs
+using different ViT attention backends. Tests are parametrized by model and backend.
diff -- vllm/attention/layers/mm_encoder_attention.py
@@ -0,0 +1,284 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Callable
+import torch
+from vllm.attention.backends.registry import AttentionBackendEnum
+from vllm.attention.ops.vit_attn_wrappers import (
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -33,7 +33,6 @@
```

- Reviewed files:
  - tests: `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0
  - runtime: `vllm/attention/layers/mm_encoder_attention.py` added +284/-0; `vllm/model_executor/models/qwen2_vl.py` modified +47/-96; `vllm/model_executor/models/glm4_1v.py` modified +46/-91; `vllm/model_executor/models/dots_ocr.py` modified +46/-83; `vllm/model_executor/models/siglip2navit.py` modified +45/-84; `vllm/model_executor/models/qwen2_5_vl.py` modified +48/-76
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_backend_functionality.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31207 - fix: update kimi k2 tool parser logic

- Link: https://github.com/vllm-project/vllm/pull/31207
- Status/date: merged / 2025-12-30
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`; associated commits `358bfd315cad`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +211/-202, 511 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: update kimi k2 tool parser logic"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`; technical summary: Covers "fix: update kimi k2 tool parser logic"; the main implementation surface is `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +192/-191 (383 lines); hunks: -44,6 +44,33 @@ def assert_tool_calls(; -346,61 +373,32 @@ def test_token_leak_between_section_and_tool_begin(kimi_k2...; symbols: assert_tool_calls, run_streaming_sequence, test_extract_tool_calls_no_tools, test_token_leak_between_section_and_tool_begin, touching `assert_tool_calls, run_streaming_sequence, test_extract_tool_calls_no_tools`; `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +19/-11 (30 lines); hunks: -122,7 +122,6 @@ def _check_and_strip_markers(self, text: str) -> tuple[str,...; -238,6 +237,7 @@ def extract_tool_calls_streaming(; symbols: _check_and_strip_markers, _reset_section_state, extract_tool_calls_streaming, touching `_check_and_strip_markers, _reset_section_state, extract_tool_calls_streaming`.
- Code diff details:
  - `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +192/-191 (383 lines); hunks: -44,6 +44,33 @@ def assert_tool_calls(; -346,61 +373,32 @@ def test_token_leak_between_section_and_tool_begin(kimi_k2...; symbols: assert_tool_calls, run_streaming_sequence, test_extract_tool_calls_no_tools, test_token_leak_between_section_and_tool_begin
  - `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +19/-11 (30 lines); hunks: -122,7 +122,6 @@ def _check_and_strip_markers(self, text: str) -> tuple[str,...; -238,6 +237,7 @@ def extract_tool_calls_streaming(; symbols: _check_and_strip_markers, _reset_section_state, extract_tool_calls_streaming
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +192/-191
  - runtime: `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +19/-11
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_kimi_k2_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31738 - [Models]: Use `MMEncoderAttention` for MoonViT

- Link: https://github.com/vllm-project/vllm/pull/31738
- Status/date: merged / 2026-01-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`; associated commits `7101e0851f73`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +72/-158, 345 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models]: Use `MMEncoderAttention` for MoonViT"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[Models]: Use `MMEncoderAttention` for MoonViT"; the main implementation surface is `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` modified +71/-157 (228 lines); hunks: -51,118 +51,20; -411,11 +313,19 @@ def __init__(; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, __init__, touching `multihead_attention, sdpa_attention, _apply_rope_input_validation`; `vllm/model_executor/models/kimi_vl.py` modified +1/-1 (2 lines); hunks: -325,7 +325,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` modified +71/-157 (228 lines); hunks: -51,118 +51,20; -411,11 +313,19 @@ def __init__(; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, __init__
  - `vllm/model_executor/models/kimi_vl.py` modified +1/-1 (2 lines); hunks: -325,7 +325,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +71/-157; `vllm/model_executor/models/kimi_vl.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33131 - [Models] Kimi-K2.5

- Link: https://github.com/vllm-project/vllm/pull/33131
- Status/date: merged / 2026-01-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`, `vllm/transformers_utils/configs/kimi_k25.py`; associated commits `b539f988e1ee`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +1799/-8, 2011 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Kimi-K2.5"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/configs/kimi_k25.py`; technical summary: Covers "[Models] Kimi-K2.5"; the main implementation surface is `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/configs/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: _apply_rope_input_validation, get_rope_shape_decorate, wrapper, get_rope_shape, touching `_apply_rope_input_validation, get_rope_shape_decorate, wrapper`; `vllm/model_executor/models/kimi_k25.py` added +581/-0 (581 lines); hunks: -0,0 +1,581; symbols: MaxImageTokenMeta, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor, __init__, touching `MaxImageTokenMeta, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor`; `vllm/transformers_utils/configs/kimi_k25.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: KimiK25VisionConfig, __init__, KimiK25Config, hidden_size, touching `KimiK25VisionConfig, __init__, KimiK25Config`; `vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: KimiK2ReasoningParser, __init__, is_reasoning_end, is_reasoning_end_streaming, touching `KimiK2ReasoningParser, __init__, is_reasoning_end`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: _apply_rope_input_validation, get_rope_shape_decorate, wrapper, get_rope_shape
  - `vllm/model_executor/models/kimi_k25.py` added +581/-0 (581 lines); hunks: -0,0 +1,581; symbols: MaxImageTokenMeta, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor, __init__
  - `vllm/transformers_utils/configs/kimi_k25.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: KimiK25VisionConfig, __init__, KimiK25Config, hidden_size
  - `vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: KimiK2ReasoningParser, __init__, is_reasoning_end, is_reasoning_end_streaming
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0; `vllm/model_executor/models/kimi_k25.py` added +581/-0; `vllm/transformers_utils/configs/kimi_k25.py` added +129/-0; `vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33320 - [Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d…

- Link: https://github.com/vllm-project/vllm/pull/33320
- Status/date: merged / 2026-01-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `17b17c068453`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d…"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d…"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -58,6 +58,7; -320,7 +321,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -58,6 +58,7; -320,7 +321,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -58,6 +58,7 @@
+from vllm.platforms import current_platform
@@ -320,7 +321,7 @@ def __init__(
-        self.device = torch.cuda.current_device()
+        self.device = current_platform.current_device()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33346 - [Models] Refactor Kimi-K2.5 weight loading

- Link: https://github.com/vllm-project/vllm/pull/33346
- Status/date: merged / 2026-01-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `8bfc8d5600ed`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +40/-176, 282 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Refactor Kimi-K2.5 weight loading"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; technical summary: Covers "[Models] Refactor Kimi-K2.5 weight loading"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +38/-174 (212 lines); hunks: -23,16 +23,7; -64,7 +55,12; symbols: KimiK25ForConditionalGeneration, get_placeholder_str, __init__, _parse_and_validate_media_input, touching `KimiK25ForConditionalGeneration, get_placeholder_str, __init__`; `vllm/model_executor/models/kimi_k25_vit.py` modified +2/-2 (4 lines); hunks: -660,13 +660,13 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +38/-174 (212 lines); hunks: -23,16 +23,7; -64,7 +55,12; symbols: KimiK25ForConditionalGeneration, get_placeholder_str, __init__, _parse_and_validate_media_input
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +2/-2 (4 lines); hunks: -660,13 +660,13 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +38/-174; `vllm/model_executor/models/kimi_k25_vit.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33562 - [Bugfix] Enable Kimi k25 processor test

- Link: https://github.com/vllm-project/vllm/pull/33562
- Status/date: merged / 2026-02-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `4061dcf4c51a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +96/-12, 221 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Enable Kimi k25 processor test"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix] Enable Kimi k25 processor test"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +27/-5 (32 lines); hunks: -96,16 +96,20 @@ class MoonshotKimiVAutoProcessor(ProcessorMixin):; -122,13 +126,30 @@ def __call__(; symbols: MoonshotKimiVAutoProcessor, __init__, __call__, touching `MoonshotKimiVAutoProcessor, __init__, __call__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +27/-5 (32 lines); hunks: -96,16 +96,20 @@ class MoonshotKimiVAutoProcessor(ProcessorMixin):; -122,13 +126,30 @@ def __call__(; symbols: MoonshotKimiVAutoProcessor, __init__, __call__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +27/-5
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33876 - [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading

- Link: https://github.com/vllm-project/vllm/pull/33876
- Status/date: merged / 2026-02-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `a2522839d87d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +15/-5, 53 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunks: -24,7 +24,11; -302,7 +306,9 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, touching `split_video_chunks, KimiK25ForConditionalGeneration, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunks: -24,7 +24,11; -302,7 +306,9 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +14/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34427 - [Bugfix] Delete unused redundant code in Kimi-K2.5

- Link: https://github.com/vllm-project/vllm/pull/34427
- Status/date: merged / 2026-02-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `62788f99a4d0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-5, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Delete unused redundant code in Kimi-K2.5"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix] Delete unused redundant code in Kimi-K2.5"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +0/-5 (5 lines); hunks: -11,7 +11,6; -378,10 +377,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +0/-5 (5 lines); hunks: -11,7 +11,6; -378,10 +377,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +0/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34501 - [Bugfix] Add quant_config in ViT of Kimi-K2.5

- Link: https://github.com/vllm-project/vllm/pull/34501
- Status/date: merged / 2026-02-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `4a9952ec1b15`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-0, 158 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Add quant_config in ViT of Kimi-K2.5"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix] Add quant_config in ViT of Kimi-K2.5"; the main implementation surface is `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25_vit.py` modified +15/-0 (15 lines); hunks: -28,6 +28,7; -304,6 +305,7 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/model_executor/models/kimi_k25.py` modified +11/-0 (11 lines); hunks: -23,6 +23,10; -361,6 +365,7 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, _parse_and_validate_media_input, touching `__init__, _maybe_ignore_quant_config, _parse_and_validate_media_input`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +15/-0 (15 lines); hunks: -28,6 +28,7; -304,6 +305,7 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/kimi_k25.py` modified +11/-0 (11 lines); hunks: -23,6 +23,10; -361,6 +365,7 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, _parse_and_validate_media_input
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` modified +15/-0; `vllm/model_executor/models/kimi_k25.py` modified +11/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33646 - [Bugfix] Handle case when kimi ends reasoning with a tool call

- Link: https://github.com/vllm-project/vllm/pull/33646
- Status/date: merged / 2026-02-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/reasoning/kimi_k2_reasoning_parser.py`; associated commits `9251ed5c4fc6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +230/-2, 240 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Handle case when kimi ends reasoning with a tool call"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/reasoning/kimi_k2_reasoning_parser.py`; technical summary: Covers "[Bugfix] Handle case when kimi ends reasoning with a tool call"; the main implementation surface is `vllm/reasoning/kimi_k2_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/reasoning/kimi_k2_reasoning_parser.py` added +228/-0 (228 lines); hunks: -0,0 +1,228; symbols: KimiK2ReasoningParser, __init__, _is_identity_mode, is_reasoning_end, touching `KimiK2ReasoningParser, __init__, _is_identity_mode`.
- Code diff details:
  - `vllm/reasoning/kimi_k2_reasoning_parser.py` added +228/-0 (228 lines); hunks: -0,0 +1,228; symbols: KimiK2ReasoningParser, __init__, _is_identity_mode, is_reasoning_end
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/reasoning/kimi_k2_reasoning_parser.py` added +228/-0
- Risk and verification: Runtime changes concentrate in `vllm/reasoning/__init__.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36192 - [Security] Respect user trust_remote_code setting in NemotronVL and KimiK25

- Link: https://github.com/vllm-project/vllm/pull/36192
- Status/date: merged / 2026-03-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `00bd08edeee5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-2, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Security] Respect user trust_remote_code setting in NemotronVL and KimiK25"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Security] Respect user trust_remote_code setting in NemotronVL and KimiK25"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -174,7 +174,8 @@ def __init__(self, ctx: InputProcessingContext) -> None:; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -174,7 +174,8 @@ def __init__(self, ctx: InputProcessingContext) -> None:; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -174,7 +174,8 @@ def __init__(self, ctx: InputProcessingContext) -> None:
-            self.ctx.model_config.model, trust_remote_code=True
+            self.ctx.model_config.model,
+            trust_remote_code=self.ctx.model_config.trust_remote_code,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36127 - [Model] Add support for moonshotai/Kimi-Audio-7B-Instruct

- Link: https://github.com/vllm-project/vllm/pull/36127
- Status/date: merged / 2026-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja`, `vllm/transformers_utils/processors/kimi_audio.py`; associated commits `42fadebecb79`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +1446/-29, 1583 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add support for moonshotai/Kimi-Audio-7B-Instruct"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py`; technical summary: Covers "[Model] Add support for moonshotai/Kimi-Audio-7B-Instruct"; the main implementation surface is `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_audio.py` added +725/-0 (725 lines); hunks: -0,0 +1,725; symbols: _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__, KimiAudioProcessingInfo, touching `_get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__`; `vllm/tokenizers/kimi_audio.py` added +410/-0 (410 lines); hunks: -0,0 +1,410; symbols: _load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained, __init__, touching `_load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained`; `vllm/transformers_utils/processors/kimi_audio.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: _get_feat_extract_output_lengths, KimiAudioProcessor, __init__, check_argument_for_proper_class, touching `_get_feat_extract_output_lengths, KimiAudioProcessor, __init__`; `vllm/renderers/kimi_audio.py` added +49/-0 (49 lines); hunks: -0,0 +1,49; symbols: KimiAudioRenderer, from_config, touching `KimiAudioRenderer, from_config`.
- Code diff details:
  - `vllm/model_executor/models/kimi_audio.py` added +725/-0 (725 lines); hunks: -0,0 +1,725; symbols: _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__, KimiAudioProcessingInfo
  - `vllm/tokenizers/kimi_audio.py` added +410/-0 (410 lines); hunks: -0,0 +1,410; symbols: _load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained, __init__
  - `vllm/transformers_utils/processors/kimi_audio.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: _get_feat_extract_output_lengths, KimiAudioProcessor, __init__, check_argument_for_proper_class
  - `vllm/renderers/kimi_audio.py` added +49/-0 (49 lines); hunks: -0,0 +1,49; symbols: KimiAudioRenderer, from_config
  - `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja` added +13/-0 (13 lines); hunks: -0,0 +1,13
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_audio.py` added +725/-0; `vllm/tokenizers/kimi_audio.py` added +410/-0; `vllm/transformers_utils/processors/kimi_audio.py` added +163/-0; `vllm/renderers/kimi_audio.py` added +49/-0; `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja` added +13/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`, `tests/models/test_initialization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36361 - Kimi k2.5 MLA based eagle3

- Link: https://github.com/vllm-project/vllm/pull/36361
- Status/date: merged / 2026-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `557389473755`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +499/-8, 649 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Kimi k2.5 MLA based eagle3"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "Kimi k2.5 MLA based eagle3"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +14/-1 (15 lines); hunks: -28,6 +28,8; -311,7 +313,12 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, set_aux_hidden_state_layers, touching `split_video_chunks, KimiK25ForConditionalGeneration, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-1 (15 lines); hunks: -28,6 +28,8; -311,7 +313,12 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, set_aux_hidden_state_layers
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +14/-1
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36903 - [Misc] Clean up Kimi-audio whisper encoder loading

- Link: https://github.com/vllm-project/vllm/pull/36903
- Status/date: merged / 2026-03-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_audio.py`; associated commits `a8e8d62dd80f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +89/-116, 382 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Clean up Kimi-audio whisper encoder loading"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_audio.py`; technical summary: Covers "[Misc] Clean up Kimi-audio whisper encoder loading"; the main implementation surface is `vllm/model_executor/models/kimi_audio.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_audio.py` modified +61/-111 (172 lines); hunks: -3,25 +3,21; -64,15 +60,6; symbols: _get_whisper_local_path, _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__, touching `_get_whisper_local_path, _get_feat_extract_output_lengths, KimiAudioWhisperEncoder`.
- Code diff details:
  - `vllm/model_executor/models/kimi_audio.py` modified +61/-111 (172 lines); hunks: -3,25 +3,21; -64,15 +60,6; symbols: _get_whisper_local_path, _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_audio.py` modified +61/-111
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/model_loader/default_loader.py`, `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/kimi_audio.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37371 - standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01

- Link: https://github.com/vllm-project/vllm/pull/37371
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_linear.py`; associated commits `17808394bc48`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +235/-219, 527 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_linear.py`; technical summary: Covers "standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01"; the main implementation surface is `vllm/model_executor/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_linear.py` modified +97/-88 (185 lines); hunks: -46,6 +46,7; -472,94 +473,7 @@ def forward(; symbols: forward, KimiLinearForCausalLM, __init__, embed_input_ids, touching `forward, KimiLinearForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_linear.py` modified +97/-88 (185 lines); hunks: -46,6 +46,7; -472,94 +473,7 @@ def forward(; symbols: forward, KimiLinearForCausalLM, __init__, embed_input_ids
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_linear.py` modified +97/-88
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/models/minimax_text_01.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37438 - [Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support

- Link: https://github.com/vllm-project/vllm/pull/37438
- Status/date: merged / 2026-03-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_kimi_k2_reasoning_parser.py`; associated commits `c63ca2b2e696`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +173/-18, 227 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `tests/reasoning/test_kimi_k2_reasoning_parser.py`; technical summary: Covers "[Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support"; the main implementation surface is `tests/reasoning/test_kimi_k2_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled, test_extract_reasoning_with_think_tags, touching `kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled`.
- Code diff details:
  - `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled, test_extract_reasoning_with_think_tags
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_kimi_k2_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37693 - [Model] Update Kimi-K25 and Isaac processors to fit HF-style

- Link: https://github.com/vllm-project/vllm/pull/37693
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/processors/kimi_k25.py`; associated commits `37aadf623786`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +128/-95, 366 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Update Kimi-K25 and Isaac processors to fit HF-style"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/transformers_utils/processors/kimi_k25.py`, `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Model] Update Kimi-K25 and Isaac processors to fit HF-style"; the main implementation surface is `vllm/transformers_utils/processors/kimi_k25.py`, `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/processors/kimi_k25.py` modified +54/-38 (92 lines); hunks: -1,38 +1,41; -42,31 +45,44 @@ def __call__(; symbols: KimiK25Processor, __init__, __call__, touching `KimiK25Processor, __init__, __call__`; `vllm/model_executor/models/kimi_k25.py` modified +16/-18 (34 lines); hunks: -104,19 +104,25 @@ class KimiK25ProcessingInfo(BaseProcessingInfo):; -132,20 +138,15 @@ def get_supported_mm_limits(self) -> Mapping[str, int | No...; symbols: KimiK25ProcessingInfo, __init__, get_hf_processor, get_supported_mm_limits, touching `KimiK25ProcessingInfo, __init__, get_hf_processor`.
- Code diff details:
  - `vllm/transformers_utils/processors/kimi_k25.py` modified +54/-38 (92 lines); hunks: -1,38 +1,41; -42,31 +45,44 @@ def __call__(; symbols: KimiK25Processor, __init__, __call__
  - `vllm/model_executor/models/kimi_k25.py` modified +16/-18 (34 lines); hunks: -104,19 +104,25 @@ class KimiK25ProcessingInfo(BaseProcessingInfo):; -132,20 +138,15 @@ def get_supported_mm_limits(self) -> Mapping[str, int | No...; symbols: KimiK25ProcessingInfo, __init__, get_hf_processor, get_supported_mm_limits
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/kimi_k25.py` modified +54/-38; `vllm/model_executor/models/kimi_k25.py` modified +16/-18
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/isaac.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/processors/isaac.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39344 - fix(kimi_k25): resolve media_placeholder_token_id from tokenizer

- Link: https://github.com/vllm-project/vllm/pull/39344
- Status/date: merged / 2026-04-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `17e787a7792b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +24/-3, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(kimi_k25): resolve media_placeholder_token_id from tokenizer"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "fix(kimi_k25): resolve media_placeholder_token_id from tokenizer"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +24/-3 (27 lines); hunks: -113,7 +113,29 @@ def __init__(self, ctx: InputProcessingContext) -> None:; -232,8 +254,7 @@ def _get_prompt_updates(; symbols: __init__, _get_prompt_updates, get_replacement, touching `__init__, _get_prompt_updates, get_replacement`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +24/-3 (27 lines); hunks: -113,7 +113,29 @@ def __init__(self, ctx: InputProcessingContext) -> None:; -232,8 +254,7 @@ def _get_prompt_updates(; symbols: __init__, _get_prompt_updates, get_replacement
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +24/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38579 - [Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping

- Link: https://github.com/vllm-project/vllm/pull/38579
- Status/date: merged / 2026-04-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`; associated commits `03ce1c6ed908`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +684/-1405, 2206 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`; technical summary: Covers "[Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping"; the main implementation surface is `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +525/-921 (1446 lines); hunks: -3,14 +3,20; -20,959 +26,557 @@ def kimi_k2_tokenizer():; symbols: kimi_k2_tokenizer, kimi_k2_tool_parser, parser, assert_tool_calls, touching `kimi_k2_tokenizer, kimi_k2_tool_parser, parser`; `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +159/-484 (643 lines); hunks: -1,6 +1,5; -17,137 +16,59; symbols: KimiK2ToolParser, __init__, _check_and_strip_markers, _reset_section_state, touching `KimiK2ToolParser, __init__, _check_and_strip_markers`.
- Code diff details:
  - `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +525/-921 (1446 lines); hunks: -3,14 +3,20; -20,959 +26,557 @@ def kimi_k2_tokenizer():; symbols: kimi_k2_tokenizer, kimi_k2_tool_parser, parser, assert_tool_calls
  - `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +159/-484 (643 lines); hunks: -1,6 +1,5; -17,137 +16,59; symbols: KimiK2ToolParser, __init__, _check_and_strip_markers, _reset_section_state
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +525/-921
  - runtime: `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +159/-484
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_kimi_k2_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41068 - [Bugfix] KimiK2ReasoningParser: guard against buffered end-token in streaming

- Link: https://github.com/vllm-project/vllm/pull/41068
- Status/date: merged / 2026-05-04
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`; associated commits `712ad0286c9a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +70/-0, 102 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] KimiK2ReasoningParser: guard against buffered end-token in streaming"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`; technical summary: Covers "[Bugfix] KimiK2ReasoningParser: guard against buffered end-token in streaming"; the main implementation surface is `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_kimi_k2_reasoning_parser.py` modified +63/-0 (63 lines); hunks: -1,6 +1,8; -12,6 +14,20; symbols: mock_kimi_k2_tokenizer, kimi_k2_tokenizer, test_streaming_tool_section_ends_reasoning, test_streaming_end_token_id_buffered, touching `mock_kimi_k2_tokenizer, kimi_k2_tokenizer, test_streaming_tool_section_ends_reasoning`; `vllm/reasoning/kimi_k2_reasoning_parser.py` modified +7/-0 (7 lines); hunks: -221,6 +221,10 @@ def extract_reasoning_streaming(; -229,6 +233,9 @@ def extract_reasoning_streaming(; symbols: extract_reasoning_streaming, touching `extract_reasoning_streaming`.
- Code diff details:
  - `tests/reasoning/test_kimi_k2_reasoning_parser.py` modified +63/-0 (63 lines); hunks: -1,6 +1,8; -12,6 +14,20; symbols: mock_kimi_k2_tokenizer, kimi_k2_tokenizer, test_streaming_tool_section_ends_reasoning, test_streaming_end_token_id_buffered
  - `vllm/reasoning/kimi_k2_reasoning_parser.py` modified +7/-0 (7 lines); hunks: -221,6 +221,10 @@ def extract_reasoning_streaming(; -229,6 +233,9 @@ def extract_reasoning_streaming(; symbols: extract_reasoning_streaming
- Key code excerpts:

```diff
diff -- tests/reasoning/test_kimi_k2_reasoning_parser.py
@@ -1,6 +1,8 @@
+from unittest.mock import MagicMock
@@ -12,6 +14,20 @@
+@pytest.fixture
+def mock_kimi_k2_tokenizer():
+    tokenizer = MagicMock()
+    tokenizer.get_vocab.return_value = {
diff -- vllm/reasoning/kimi_k2_reasoning_parser.py
@@ -221,6 +221,10 @@ def extract_reasoning_streaming(
+            if self._end_token not in delta_text:
+                # Token ID arrived before text was flushed (stop-sequence buffering).
+                # Wait for the next delta when the text becomes visible.
+                return None
@@ -229,6 +233,9 @@ def extract_reasoning_streaming(
+            if self._tool_section_start_token not in delta_text:
```

- Reviewed files:
  - tests: `tests/reasoning/test_kimi_k2_reasoning_parser.py` modified +63/-0
  - runtime: `vllm/reasoning/kimi_k2_reasoning_parser.py` modified +7/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_kimi_k2_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42081 - [Bug] Fix kimi dtype issue with `mm_projector_forward`

- Link: https://github.com/vllm-project/vllm/pull/42081
- Status/date: merged / 2026-05-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `3f9c0c25b331`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix kimi dtype issue with `mm_projector_forward`"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25_vit.py`; technical summary: Covers "[Bug] Fix kimi dtype issue with `mm_projector_forward`"; the main implementation surface is `vllm/model_executor/models/kimi_k25_vit.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25_vit.py` modified +3/-0 (3 lines); hunks: -618,6 +618,9 @@ def mm_projector_forward(mm_projector: torch.nn.Module, vt_o...; symbols: mm_projector_forward, touching `mm_projector_forward`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +3/-0 (3 lines); hunks: -618,6 +618,9 @@ def mm_projector_forward(mm_projector: torch.nn.Module, vt_o...; symbols: mm_projector_forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -618,6 +618,9 @@ def mm_projector_forward(mm_projector: torch.nn.Module, vt_output: list[torch.Te
+    projector_dtype = mm_projector.pre_norm.weight.dtype
+    if batched.dtype != projector_dtype:
+        batched = batched.to(projector_dtype)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41778 - [MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell

- Link: https://github.com/vllm-project/vllm/pull/41778
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +640/-89, 975 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell"; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `benchmarks/attention_benchmarks/configs/mla_prefill.yaml`, `benchmarks/attention_benchmarks/configs/mla_decode.yaml`, `vllm/model_executor/layers/attention/mla_attention.py`; technical summary: Covers "[MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell"; the main implementation surface is `benchmarks/attention_benchmarks/configs/mla_prefill.yaml`, `benchmarks/attention_benchmarks/configs/mla_decode.yaml`, `vllm/model_executor/layers/attention/mla_attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `benchmarks/attention_benchmarks/configs/mla_prefill.yaml` modified +2/-0 (2 lines); hunks: -3,6 +3,7; -120,6 +121,7 @@ prefill_backends:; `benchmarks/attention_benchmarks/configs/mla_decode.yaml` modified +1/-0 (1 lines); hunks: -53,6 +53,7 @@ backends:; `vllm/model_executor/layers/attention/mla_attention.py` modified +1/-0 (1 lines); hunks: -1362,6 +1362,7 @@ def backend_supports_prefill_query_quantization() -> bool:; symbols: backend_supports_prefill_query_quantization, touching `backend_supports_prefill_query_quantization`; `vllm/v1/attention/backends/mla/tokenspeed_mla.py` added +277/-0 (277 lines); hunks: -0,0 +1,277; symbols: _get_workspace, TokenspeedMLAMetadataBuilder, TokenspeedMLABackend, get_supported_kernel_block_sizes, touching `_get_workspace, TokenspeedMLAMetadataBuilder, TokenspeedMLABackend`.
- Code diff details:
  - `benchmarks/attention_benchmarks/configs/mla_prefill.yaml` modified +2/-0 (2 lines); hunks: -3,6 +3,7; -120,6 +121,7 @@ prefill_backends:
  - `benchmarks/attention_benchmarks/configs/mla_decode.yaml` modified +1/-0 (1 lines); hunks: -53,6 +53,7 @@ backends:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +1/-0 (1 lines); hunks: -1362,6 +1362,7 @@ def backend_supports_prefill_query_quantization() -> bool:; symbols: backend_supports_prefill_query_quantization
  - `vllm/v1/attention/backends/mla/tokenspeed_mla.py` added +277/-0 (277 lines); hunks: -0,0 +1,277; symbols: _get_workspace, TokenspeedMLAMetadataBuilder, TokenspeedMLABackend, get_supported_kernel_block_sizes
  - `vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py` added +180/-0 (180 lines); hunks: -0,0 +1,180; symbols: TokenspeedMLAPrefillBackend, get_name, supports_compute_capability, is_available
- Key code excerpts:

```diff
diff -- benchmarks/attention_benchmarks/configs/mla_prefill.yaml
@@ -3,6 +3,7 @@
+#   CuTe DSL:     tokenspeed (Blackwell + R1 dims, requires tokenspeed_mla)
@@ -120,6 +121,7 @@ prefill_backends:
+  - tokenspeed
diff -- benchmarks/attention_benchmarks/configs/mla_decode.yaml
@@ -53,6 +53,7 @@ backends:
+  - TOKENSPEED_MLA  # Blackwell + R1 dims + FP8 KV (use --kv-cache-dtype fp8)
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -1362,6 +1362,7 @@ def backend_supports_prefill_query_quantization() -> bool:
+        "TOKENSPEED_MLA",
diff -- vllm/v1/attention/backends/mla/tokenspeed_mla.py
@@ -0,0 +1,277 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""TokenSpeed CuTe DSL MLA decode backend (Blackwell, FP8 KV cache only)."""
+from typing import ClassVar
+import torch
```

- Reviewed files:
  - runtime: `benchmarks/attention_benchmarks/configs/mla_prefill.yaml` modified +2/-0; `benchmarks/attention_benchmarks/configs/mla_decode.yaml` modified +1/-0; `vllm/model_executor/layers/attention/mla_attention.py` modified +1/-0; `vllm/v1/attention/backends/mla/tokenspeed_mla.py` added +277/-0; `vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py` added +180/-0
  - other: `benchmarks/attention_benchmarks/mla_runner.py` modified +67/-63
  - tests: `tests/v1/attention/test_mla_backends.py` modified +66/-7; `tests/conftest.py` modified +22/-13
- Risk and verification: The diff ships test coverage in `tests/conftest.py`, `tests/v1/attention/test_mla_backends.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42869 - [BugFix] Kimi-K2.5: skip vision tower dtype conversion when using quantization

- Link: https://github.com/vllm-project/vllm/pull/42869
- Status/date: merged / 2026-05-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `23c15acd770c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-3, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Kimi-K2.5: skip vision tower dtype conversion when using quantization"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[BugFix] Kimi-K2.5: skip vision tower dtype conversion when using quantization"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +6/-3 (9 lines); hunks: -339,9 +339,12 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +6/-3 (9 lines); hunks: -339,9 +339,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -339,9 +339,12 @@ def __init__(
-            self.vision_tower = self.vision_tower.to(
-                device=self.device, dtype=model_config.dtype
-            )
+            if self._maybe_ignore_quant_config(quant_config) is not None:
+                self.vision_tower = self.vision_tower.to(device=self.device)
+            else:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +6/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41126 - [Attention] Mamba attention module refactor

- Link: https://github.com/vllm-project/vllm/pull/41126
- Status/date: merged / 2026-05-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +765/-774, 1913 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Attention] Mamba attention module refactor"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/olmo_hybrid.py`, `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`; technical summary: Covers "[Attention] Mamba attention module refactor"; the main implementation surface is `vllm/model_executor/models/olmo_hybrid.py`, `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/olmo_hybrid.py` modified +6/-645 (651 lines); hunks: -26,73 +26,47; -107,502 +81,6; symbols: _make_fused_conv1d_weight_loader, weight_loader, OlmoHybridGatedDeltaNet, mamba_type, touching `_make_fused_conv1d_weight_loader, weight_loader, OlmoHybridGatedDeltaNet`; `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` added +634/-0 (634 lines); hunks: -0,0 +1,634; symbols: OlmoHybridGatedDeltaNetAttention, get_state_shape, __init__, rearrange_mixed_qkv, touching `OlmoHybridGatedDeltaNetAttention, get_state_shape, __init__`; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` renamed +26/-45 (71 lines); hunks: -5,39 +5,37; -83,11 +81,8 @@ def kda_attention_fake(; symbols: kda_attention_fake, KimiDeltaAttention, mamba_type, KimiGatedDeltaNetAttention, touching `kda_attention_fake, KimiDeltaAttention, mamba_type`; `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` renamed +19/-52 (71 lines); hunks: -5,7 +5,6; -15,8 +14,6; symbols: forward_native, GatedDeltaNetAttention, mamba_type, get_state_dtype, touching `forward_native, GatedDeltaNetAttention, mamba_type`.
- Code diff details:
  - `vllm/model_executor/models/olmo_hybrid.py` modified +6/-645 (651 lines); hunks: -26,73 +26,47; -107,502 +81,6; symbols: _make_fused_conv1d_weight_loader, weight_loader, OlmoHybridGatedDeltaNet, mamba_type
  - `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` added +634/-0 (634 lines); hunks: -0,0 +1,634; symbols: OlmoHybridGatedDeltaNetAttention, get_state_shape, __init__, rearrange_mixed_qkv
  - `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` renamed +26/-45 (71 lines); hunks: -5,39 +5,37; -83,11 +81,8 @@ def kda_attention_fake(; symbols: kda_attention_fake, KimiDeltaAttention, mamba_type, KimiGatedDeltaNetAttention
  - `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` renamed +19/-52 (71 lines); hunks: -5,7 +5,6; -15,8 +14,6; symbols: forward_native, GatedDeltaNetAttention, mamba_type, get_state_dtype
  - `vllm/model_executor/layers/mamba/gdn/base.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: GatedDeltaNetAttention, for, __init__, mamba_type
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/olmo_hybrid.py
@@ -26,73 +26,47 @@
-from einops import rearrange
-from transformers.activations import ACT2FN
-    CacheConfig,
-    ModelConfig,
-    SpeculativeConfig,
-    get_current_vllm_config,
diff -- vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py
@@ -0,0 +1,634 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+from einops import rearrange
+from torch import nn
+from vllm.config import (
diff -- vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py
@@ -5,39 +5,37 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/olmo_hybrid.py` modified +6/-645; `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` added +634/-0; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` renamed +26/-45; `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` renamed +19/-52; `vllm/model_executor/layers/mamba/gdn/base.py` added +58/-0; `vllm/model_executor/models/kimi_linear.py` modified +13/-27
- Risk and verification: Runtime changes concentrate in `vllm/config/compilation.py`, `vllm/model_executor/layers/mamba/gdn/__init__.py`, `vllm/model_executor/layers/mamba/gdn/base.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43857 - Add vLLM library info to Hugging Face Hub requests

- Link: https://github.com/vllm-project/vllm/pull/43857
- Status/date: merged / 2026-05-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +78/-43, 467 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add vLLM library info to Hugging Face Hub requests"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/model_loader/weight_utils.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/tokenizers/grok2.py`; technical summary: Covers "Add vLLM library info to Hugging Face Hub requests"; the main implementation surface is `vllm/model_executor/model_loader/weight_utils.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/tokenizers/grok2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/model_loader/weight_utils.py` modified +7/-7 (14 lines); hunks: -23,7 +23,6; -46,6 +45,7; symbols: get_quant_config, get_sparse_attention_config, download_weights_from_hf, touching `get_quant_config, get_sparse_attention_config, download_weights_from_hf`; `vllm/tokenizers/kimi_audio.py` modified +4/-4 (8 lines); hunks: -10,13 +10,13; -78,7 +78,7 @@ def from_pretrained(; symbols: from_pretrained, touching `from_pretrained`; `vllm/tokenizers/grok2.py` modified +3/-3 (6 lines); hunks: -8,7 +8,6; -20,6 +19,7; symbols: _maybe_load_tokenizer_config, from_pretrained, touching `_maybe_load_tokenizer_config, from_pretrained`; `vllm/model_executor/model_loader/bitsandbytes_loader.py` modified +2/-3 (5 lines); hunks: -10,7 +10,6; -48,6 +47,7; symbols: _get_weight_files, touching `_get_weight_files`.
- Code diff details:
  - `vllm/model_executor/model_loader/weight_utils.py` modified +7/-7 (14 lines); hunks: -23,7 +23,6; -46,6 +45,7; symbols: get_quant_config, get_sparse_attention_config, download_weights_from_hf
  - `vllm/tokenizers/kimi_audio.py` modified +4/-4 (8 lines); hunks: -10,13 +10,13; -78,7 +78,7 @@ def from_pretrained(; symbols: from_pretrained
  - `vllm/tokenizers/grok2.py` modified +3/-3 (6 lines); hunks: -8,7 +8,6; -20,6 +19,7; symbols: _maybe_load_tokenizer_config, from_pretrained
  - `vllm/model_executor/model_loader/bitsandbytes_loader.py` modified +2/-3 (5 lines); hunks: -10,7 +10,6; -48,6 +47,7; symbols: _get_weight_files
  - `vllm/model_executor/model_loader/gguf_loader.py` modified +2/-2 (4 lines); hunks: -8,7 +8,6; -27,6 +26,7; symbols: _prepare_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/model_loader/weight_utils.py
@@ -23,7 +23,6 @@
-from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
@@ -46,6 +45,7 @@
+from vllm.transformers_utils.repo_utils import hf_api, hf_fs
@@ -373,7 +373,7 @@ def get_quant_config(
-            hf_folder = snapshot_download(
+            hf_folder = hf_api().snapshot_download(
diff -- vllm/tokenizers/kimi_audio.py
@@ -10,13 +10,13 @@
-from huggingface_hub import hf_hub_download
+from vllm.transformers_utils.repo_utils import hf_api
@@ -78,7 +78,7 @@ def from_pretrained(
-                vocab_path = hf_hub_download(
+                vocab_path = hf_api().hf_hub_download(
@@ -87,7 +87,7 @@ def from_pretrained(
diff -- vllm/tokenizers/grok2.py
@@ -8,7 +8,6 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/model_loader/weight_utils.py` modified +7/-7; `vllm/tokenizers/kimi_audio.py` modified +4/-4; `vllm/tokenizers/grok2.py` modified +3/-3; `vllm/model_executor/model_loader/bitsandbytes_loader.py` modified +2/-3; `vllm/model_executor/model_loader/gguf_loader.py` modified +2/-2; `vllm/model_executor/model_loader/tensorizer.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/lora/test_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #44493 - [Bugfix]Fix Kimi-K2.5 FlashInfer ViT metadata

- Link: https://github.com/vllm-project/vllm/pull/44493
- Status/date: merged / 2026-06-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `1bdc60ed53ad`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +109/-28, 260 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix]Fix Kimi-K2.5 FlashInfer ViT metadata"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix]Fix Kimi-K2.5 FlashInfer ViT metadata"; the main implementation surface is `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25_vit.py` modified +108/-27 (135 lines); hunks: -154,9 +154,12 @@ def __init__(; -218,7 +221,9 @@ def __init__(; symbols: __init__, reset_parameters, forward, touching `__init__, reset_parameters, forward`; `vllm/model_executor/models/kimi_k25.py` modified +1/-1 (2 lines); hunks: -235,7 +235,7 @@ def _get_mm_fields_config(; symbols: _get_mm_fields_config, _call_hf_processor, touching `_get_mm_fields_config, _call_hf_processor`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +108/-27 (135 lines); hunks: -154,9 +154,12 @@ def __init__(; -218,7 +221,9 @@ def __init__(; symbols: __init__, reset_parameters, forward
  - `vllm/model_executor/models/kimi_k25.py` modified +1/-1 (2 lines); hunks: -235,7 +235,7 @@ def _get_mm_fields_config(; symbols: _get_mm_fields_config, _call_hf_processor
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -154,9 +154,12 @@ def __init__(
-    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
+    def forward(
+        self, x: torch.Tensor, grid_thws: torch.Tensor | list[list[int]]
+    ) -> torch.Tensor:
-        for t, h, w in grid_thws.tolist():
+        grid_thw_list = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
diff -- vllm/model_executor/models/kimi_k25.py
@@ -235,7 +235,7 @@ def _get_mm_fields_config(
-            grid_thws=MultiModalFieldConfig.batched("vision_chunk"),
+            grid_thws=MultiModalFieldConfig.batched("vision_chunk", keep_on_cpu=True),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` modified +108/-27; `vllm/model_executor/models/kimi_k25.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44539 - [mamba] unify KDA conv states into one cache to match 2-state SSM layout

- Link: https://github.com/vllm-project/vllm/pull/44539
- Status/date: merged / 2026-06-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +16/-30, 120 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[mamba] unify KDA conv states into one cache to match 2-state SSM layout"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `vllm/model_executor/models/kimi_linear.py`; technical summary: Covers "[mamba] unify KDA conv states into one cache to match 2-state SSM layout"; the main implementation surface is `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `vllm/model_executor/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/mamba/mamba_utils.py` modified +7/-19 (26 lines); hunks: -120,9 +120,9 @@ def kda_state_dtype(; -243,7 +243,7 @@ def kda_state_shape(; symbols: kda_state_dtype, MambaStateShapeCalculator, kda_state_shape, gated_delta_net_state_copy_func, touching `kda_state_dtype, MambaStateShapeCalculator, kda_state_shape`; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` modified +6/-6 (12 lines); hunks: -85,7 +85,7 @@ def kda_attention_fake(; -94,7 +94,7 @@ def get_state_dtype(; symbols: kda_attention_fake, KimiGatedDeltaNetAttention, get_state_dtype, get_state_shape, touching `kda_attention_fake, KimiGatedDeltaNetAttention, get_state_dtype`; `vllm/model_executor/models/kimi_linear.py` modified +3/-5 (8 lines); hunks: -600,15 +600,15 @@ def forward(; -628,9 +628,7 @@ def get_mamba_state_shape_from_config(; symbols: forward, get_mamba_state_dtype_from_config, get_mamba_state_shape_from_config, get_mamba_state_copy_func, touching `forward, get_mamba_state_dtype_from_config, get_mamba_state_shape_from_config`.
- Code diff details:
  - `vllm/model_executor/layers/mamba/mamba_utils.py` modified +7/-19 (26 lines); hunks: -120,9 +120,9 @@ def kda_state_dtype(; -243,7 +243,7 @@ def kda_state_shape(; symbols: kda_state_dtype, MambaStateShapeCalculator, kda_state_shape, gated_delta_net_state_copy_func
  - `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` modified +6/-6 (12 lines); hunks: -85,7 +85,7 @@ def kda_attention_fake(; -94,7 +94,7 @@ def get_state_dtype(; symbols: kda_attention_fake, KimiGatedDeltaNetAttention, get_state_dtype, get_state_shape
  - `vllm/model_executor/models/kimi_linear.py` modified +3/-5 (8 lines); hunks: -600,15 +600,15 @@ def forward(; -628,9 +628,7 @@ def get_mamba_state_shape_from_config(; symbols: forward, get_mamba_state_dtype_from_config, get_mamba_state_shape_from_config, get_mamba_state_copy_func
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/mamba/mamba_utils.py
@@ -120,9 +120,9 @@ def kda_state_dtype(
-    ):
+    ) -> tuple[torch.dtype, torch.dtype]:
-        return (state_dtype, state_dtype, state_dtype, torch.float32)
+        return (state_dtype, torch.float32)
@@ -243,7 +243,7 @@ def kda_state_shape(
-    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int, int]]:
diff -- vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py
@@ -85,7 +85,7 @@ def kda_attention_fake(
-    ) -> tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype]:
+    ) -> tuple[torch.dtype, torch.dtype]:
@@ -94,7 +94,7 @@ def get_state_dtype(
-    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
+    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
@@ -300,13 +300,13 @@ def _forward(
diff -- vllm/model_executor/models/kimi_linear.py
@@ -600,15 +600,15 @@ def forward(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/mamba/mamba_utils.py` modified +7/-19; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` modified +6/-6; `vllm/model_executor/models/kimi_linear.py` modified +3/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45003 - [Frontend] Support strict mode for tool calling

- Link: https://github.com/vllm-project/vllm/pull/45003
- Status/date: merged / 2026-06-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 29 files, +672/-1936, 3162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] Support strict mode for tool calling"; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`; technical summary: Covers "[Frontend] Support strict mode for tool calling"; the main implementation surface is `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300 (1300 lines); hunks: -1,1300 +0,0; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks, touching `StreamingXMLToolCallParser, __init__, reset_streaming_state`; `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240 (414 lines); hunks: -1,14 +1,15; -24,23 +25,51; symbols: register_model_structural_tag, register_vllm_structural_tag, decorator, get_model_structural_tag, touching `register_model_structural_tag, register_vllm_structural_tag, decorator`; `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins, test_get_model_structural_tag_supports_vllm_hermes, touching `sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins`; `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190 (203 lines); hunks: -3,6 +3,7; -19,15 +20,12; symbols: qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized, assert_tool_calls, touching `qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized`.
- Code diff details:
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300 (1300 lines); hunks: -1,1300 +0,0; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks
  - `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240 (414 lines); hunks: -1,14 +1,15; -24,23 +25,51; symbols: register_model_structural_tag, register_vllm_structural_tag, decorator, get_model_structural_tag
  - `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins, test_get_model_structural_tag_supports_vllm_hermes
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190 (203 lines); hunks: -3,6 +3,7; -19,15 +20,12; symbols: qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized, assert_tool_calls
  - `tests/tool_parsers/test_qwen3xml_tool_parser.py` removed +0/-72 (72 lines); hunks: -1,72 +0,0; symbols: TestQwen3xmlToolParser, test_config
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/qwen3xml_tool_parser.py
@@ -1,1300 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-import json
-from collections.abc import Sequence
-from typing import Any
-from xml.parsers.expat import ParserCreate
diff -- vllm/tool_parsers/structural_tag_registry.py
@@ -1,14 +1,15 @@
-# Model-specific structural tag builders adapted from XGrammar's
-# builtin structural tag implementations:
-# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/builtin_structural_tag.py
-from xgrammar import StructuralTag
+from xgrammar import StructuralTag, normalize_tool_choice
+from xgrammar import get_model_structural_tag as get_xgrammar_model_structural_tag
diff -- tests/tool_parsers/test_structural_tag_registry.py
@@ -0,0 +1,314 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300; `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240; `vllm/tool_parsers/abstract_tool_parser.py` modified +36/-28; `vllm/entrypoints/serve/render/serving.py` modified +24/-28; `vllm/tool_parsers/deepseekv4_tool_parser.py` modified +1/-15
  - tests: `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0; `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190; `tests/tool_parsers/test_qwen3xml_tool_parser.py` removed +0/-72
- Risk and verification: The diff ships test coverage in `requirements/test/rocm.txt`, `tests/entrypoints/openai/chat_completion/test_completion_with_function_calling.py`, `tests/entrypoints/openai/responses/conftest.py`, `tests/parser/test_parse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41992 - [MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL

- Link: https://github.com/vllm-project/vllm/pull/41992
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`; associated commits `fa85ead2f378`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +498/-39, 726 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL"; the main implementation surface is `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` modified +266/-37 (303 lines); hunks: -45,7 +45,9; -110,23 +112,42 @@ def __init__(; symbols: __init__, reset_parameters, forward, get_pos_embeds, touching `__init__, reset_parameters, forward`; `vllm/model_executor/models/kimi_vl.py` modified +195/-2 (197 lines); hunks: -56,7 +56,11; -79,6 +83,7; symbols: get_replacement, KimiVLForConditionalGeneration, __init__, get_encoder_cudagraph_config, touching `get_replacement, KimiVLForConditionalGeneration, __init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` modified +266/-37 (303 lines); hunks: -45,7 +45,9; -110,23 +112,42 @@ def __init__(; symbols: __init__, reset_parameters, forward, get_pos_embeds
  - `vllm/model_executor/models/kimi_vl.py` modified +195/-2 (197 lines); hunks: -56,7 +56,11; -79,6 +83,7; symbols: get_replacement, KimiVLForConditionalGeneration, __init__, get_encoder_cudagraph_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -45,7 +45,9 @@
+from typing import Any
+import numpy as np
@@ -110,23 +112,42 @@ def __init__(
-    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
-        pos_embs = []
-        for shape in grid_hws.tolist():
diff -- vllm/model_executor/models/kimi_vl.py
@@ -56,7 +56,11 @@
-from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
+from vllm.model_executor.models.interfaces import (
+    SupportsEncoderCudaGraph,
+    SupportsMultiModal,
+    SupportsPP,
+)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +266/-37; `vllm/model_executor/models/kimi_vl.py` modified +195/-2
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45424 - [Core] Ensure memory is pinned prior to async h2d copy

- Link: https://github.com/vllm-project/vllm/pull/45424
- Status/date: merged / 2026-06-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 49 files, +254/-264, 1718 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Core] Ensure memory is pinned prior to async h2d copy"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/pooler/seqwise/methods.py`, `vllm/multimodal/inputs.py`; technical summary: Covers "[Core] Ensure memory is pinned prior to async h2d copy"; the main implementation surface is `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/pooler/seqwise/methods.py`, `vllm/multimodal/inputs.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/attention/mla_attention.py` modified +10/-8 (18 lines); hunks: -1684,12 +1684,13 @@ def build(; -1746,12 +1747,13 @@ def build(; symbols: build, touching `build`; `vllm/model_executor/layers/pooler/seqwise/methods.py` modified +8/-8 (16 lines); hunks: -10,6 +10,7; -74,15 +75,14 @@ def forward(; symbols: forward, touching `forward`; `vllm/multimodal/inputs.py` modified +14/-2 (16 lines); hunks: -488,7 +488,13 @@ def _reduce_data(; -538,7 +544,13 @@ def _reduce_data(; symbols: _reduce_data, touching `_reduce_data`; `vllm/model_executor/models/moonvit.py` modified +3/-2 (5 lines); hunks: -66,6 +66,7; -758,7 +759,7 @@ def prepare_encoder_metadata(; symbols: _apply_rope_input_validation, prepare_encoder_metadata, touching `_apply_rope_input_validation, prepare_encoder_metadata`.
- Code diff details:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +10/-8 (18 lines); hunks: -1684,12 +1684,13 @@ def build(; -1746,12 +1747,13 @@ def build(; symbols: build
  - `vllm/model_executor/layers/pooler/seqwise/methods.py` modified +8/-8 (16 lines); hunks: -10,6 +10,7; -74,15 +75,14 @@ def forward(; symbols: forward
  - `vllm/multimodal/inputs.py` modified +14/-2 (16 lines); hunks: -488,7 +488,13 @@ def _reduce_data(; -538,7 +544,13 @@ def _reduce_data(; symbols: _reduce_data
  - `vllm/model_executor/models/moonvit.py` modified +3/-2 (5 lines); hunks: -66,6 +66,7; -758,7 +759,7 @@ def prepare_encoder_metadata(; symbols: _apply_rope_input_validation, prepare_encoder_metadata
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-3 (5 lines); hunks: -83,9 +83,8; -825,7 +824,7 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen, invert_permutation
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -1684,12 +1684,13 @@ def build(
-                chunk_starts = (
+                chunk_starts = torch.empty(
+                    num_chunks, num_prefills, dtype=torch.int32, pin_memory=True
+                ).copy_(
+                    .multiply_(max_context_chunk)
-                    .expand(-1, num_prefills)
diff -- vllm/model_executor/layers/pooler/seqwise/methods.py
@@ -10,6 +10,7 @@
+from vllm.utils.torch_utils import async_tensor_h2d
@@ -74,15 +75,14 @@ def forward(
-        # Build segment_ids on CPU so repeat_interleave doesn't need to sync
-        # GPU->CPU to learn its data-dependent output length, then upload
-        # non-blocking. eg. [2, 1, 3] -> [0, 0, 1, 2, 2, 2]
+        prompt_lens = async_tensor_h2d(
diff -- vllm/multimodal/inputs.py
@@ -488,7 +488,13 @@ def _reduce_data(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/attention/mla_attention.py` modified +10/-8; `vllm/model_executor/layers/pooler/seqwise/methods.py` modified +8/-8; `vllm/multimodal/inputs.py` modified +14/-2; `vllm/model_executor/models/moonvit.py` modified +3/-2; `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-3; `vllm/model_executor/layers/attention/mm_encoder_attention.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/v1/logits_processors/test_correctness.py`, `tests/v1/streaming_input/test_gpu_model_runner_streaming.py`, `tests/v1/worker/test_gpu_input_batch.py`, `tests/v1/worker/test_gpu_model_runner.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
