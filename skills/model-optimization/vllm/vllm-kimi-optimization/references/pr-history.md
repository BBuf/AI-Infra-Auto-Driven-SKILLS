# vLLM Kimi K2 / K2.5 / Linear / Audio / VL PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: Kimi-VL, Kimi-Linear, Kimi-K2.5, Kimi-Audio, parser aliases, and quantized MLA behavior in vLLM.

## Landed PRs

### PR #16387 - Add Kimi-VL model support

- Link: https://github.com/vllm-project/vllm/pull/16387
- Why it mattered: Landed the original Kimi-VL multimodal runtime.
- Runtime path: vllm/vllm/model_executor/models/kimi_vl.py, vllm/vllm/model_executor/models/kimi_linear.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #27809 - Introduce Kimi Linear to vLLM

- Link: https://github.com/vllm-project/vllm/pull/27809
- Why it mattered: Added the linear-attention Kimi family instead of only the VL path.
- Runtime path: vllm/vllm/model_executor/models/kimi_vl.py, vllm/vllm/model_executor/models/kimi_linear.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33131 - Kimi-K2.5

- Link: https://github.com/vllm-project/vllm/pull/33131
- Why it mattered: Brought the K2.5 generation into mainline.
- Runtime path: vllm/vllm/model_executor/models/kimi_vl.py, vllm/vllm/model_executor/models/kimi_linear.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33876 - Fix Kimi-K2.5 NVFP4 checkpoints weight loading

- Link: https://github.com/vllm-project/vllm/pull/33876
- Why it mattered: Closed a concrete launch blocker for quantized K2.5 checkpoints.
- Runtime path: vllm/vllm/model_executor/models/kimi_vl.py, vllm/vllm/model_executor/models/kimi_linear.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #36127 - Add support for moonshotai/Kimi-Audio-7B-Instruct

- Link: https://github.com/vllm-project/vllm/pull/36127
- Why it mattered: Extended the family to audio-conditioned serving.
- Runtime path: vllm/vllm/model_executor/models/kimi_vl.py, vllm/vllm/model_executor/models/kimi_linear.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37438 - Add Kimi-K2.5 reasoning/tool parser aliases

- Link: https://github.com/vllm-project/vllm/pull/37438
- Why it mattered: Aligned parser aliases and tool-call IDs with the newer model outputs.
- Runtime path: vllm/vllm/model_executor/models/kimi_vl.py, vllm/vllm/model_executor/models/kimi_linear.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM Kimi K2 / K2.5 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-04-10 | [#16387](https://github.com/vllm-project/vllm/pull/16387) | merged | [Model][VLM] Add Kimi-VL model support | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `examples/offline_inference/vision_language_multi_image.py` |
| 2025-10-30 | [#27809](https://github.com/vllm-project/vllm/pull/27809) | merged | [Model] Introduce Kimi Linear to vLLM | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/layers/kda.py`, `vllm/transformers_utils/configs/kimi_linear.py` |
| 2026-01-27 | [#33131](https://github.com/vllm-project/vllm/pull/33131) | merged | [Models] Kimi-K2.5 | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/entrypoints/chat_utils.py` |
| 2026-02-05 | [#33876](https://github.com/vllm-project/vllm/pull/33876) | merged | [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading | model wrapper, scheduler/runtime | `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/deepseek_v2.py` |
| 2026-03-05 | [#36127](https://github.com/vllm-project/vllm/pull/36127) | merged | [Model] Add support for moonshotai/Kimi-Audio-7B-Instruct | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py` |
| 2026-03-18 | [#37438](https://github.com/vllm-project/vllm/pull/37438) | merged | [Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support | tests/benchmarks | `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/entrypoints/chat_utils.py`, `vllm/entrypoints/openai/chat_completion/serving.py` |

## Diff Cards

### PR #16387 - [Model][VLM] Add Kimi-VL model support

- Link: https://github.com/vllm-project/vllm/pull/16387
- Status/date: `merged`, created 2025-04-10, merged 2025-04-14; author `courage17340`.
- Diff scope read: `18` files, `+1436/-14`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: vision, config, kv, spec, cache, processor, attention, doc, cuda, eagle.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` added +628/-0 (628 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope
  - `vllm/model_executor/models/kimi_vl.py` added +608/-0 (608 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: MaxImageTokenMeta:, KimiVLMultiModalProjector, __init__, forward
  - `examples/offline_inference/vision_language_multi_image.py` modified +40/-0 (40 lines); hunks: def load_llama4(question: str, image_urls: list[str]) -> ModelRequestData:; def load_qwen2_5_vl(question: str, image_urls: list[str]) -> ModelRequestData:; symbols: load_llama4, load_kimi_vl, load_mistral3, load_qwen2_5_vl
  - `vllm/transformers_utils/configs/kimi_vl.py` added +36/-0 (36 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: KimiVLConfig, __init__
  - `vllm/transformers_utils/configs/moonvit.py` added +32/-0 (32 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: MoonViTConfig, __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `examples/offline_inference/vision_language_multi_image.py`; keywords observed in patches: vision, config, kv, spec, cache, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `examples/offline_inference/vision_language_multi_image.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #27809 - [Model] Introduce Kimi Linear to vLLM

- Link: https://github.com/vllm-project/vllm/pull/27809
- Status/date: `merged`, created 2025-10-30, merged 2025-10-30; author `zhiyuan1i`.
- Diff scope read: `15` files, `+1326/-49`; areas: model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config; keywords: attention, cache, kv, config, mla, spec, moe, topk, expert, lora.
- Code diff details:
  - `vllm/model_executor/models/kimi_linear.py` added +663/-0 (663 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: KimiMLP, __init__, forward, KimiMoE
  - `vllm/model_executor/layers/kda.py` added +426/-0 (426 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: kda_attention, kda_attention_fake, KimiDeltaAttention, mamba_type
  - `vllm/transformers_utils/configs/kimi_linear.py` added +144/-0 (144 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: KimiLinearConfig, __init__, is_mla, is_moe
  - `vllm/model_executor/models/config.py` modified +25/-26 (51 lines); hunks: # SPDX-License-Identifier: Apache-2.0; def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:; symbols: verify_and_update_config, verify_and_update_config, verify_and_update_config, lcm
  - `vllm/model_executor/layers/mamba/mamba_utils.py` modified +41/-0 (41 lines); hunks: def gated_delta_net_state_dtype(; def gated_delta_net_state_shape(; symbols: gated_delta_net_state_dtype, kda_state_dtype, MambaStateShapeCalculator:, gated_delta_net_state_shape
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/layers/kda.py`, `vllm/transformers_utils/configs/kimi_linear.py`; keywords observed in patches: attention, cache, kv, config, mla, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/layers/kda.py`, `vllm/transformers_utils/configs/kimi_linear.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #33131 - [Models] Kimi-K2.5

- Link: https://github.com/vllm-project/vllm/pull/33131
- Status/date: `merged`, created 2026-01-27, merged 2026-01-27; author `ywang96`.
- Diff scope read: `16` files, `+1799/-8`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: kv, vision, config, processor, attention, cache, quant, spec, cuda, expert.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0 (678 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: _apply_rope_input_validation, get_rope_shape_decorate, wrapper, get_rope_shape
  - `vllm/model_executor/models/kimi_k25.py` added +581/-0 (581 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: MaxImageTokenMeta:, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor, __init__
  - `vllm/entrypoints/chat_utils.py` modified +182/-5 (187 lines); hunks: MultiModalBatchedField,; class ConversationMessage(TypedDict, total=False):; symbols: ConversationMessage, _get_embeds_data, rebuild_mm_uuids_from_mm_data, build_video_prompts_from_mm_data
  - `vllm/transformers_utils/configs/kimi_k25.py` added +129/-0 (129 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: KimiK25VisionConfig, __init__, KimiK25Config, __init__
  - `vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0 (80 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: KimiK2ReasoningParser, __init__, is_reasoning_end, is_reasoning_end_streaming
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/entrypoints/chat_utils.py`; keywords observed in patches: kv, vision, config, processor, attention, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/entrypoints/chat_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #33876 - [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading

- Link: https://github.com/vllm-project/vllm/pull/33876
- Status/date: `merged`, created 2026-02-05, merged 2026-02-05; author `Isotr0py`.
- Diff scope read: `2` files, `+15/-5`; areas: model wrapper, scheduler/runtime; keywords: config, expert, fp4, moe, quant.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunks: from vllm.config import VllmConfig; def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, KimiK25ForConditionalGeneration, KimiK25ForConditionalGeneration
  - `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/deepseek_v2.py`; keywords observed in patches: config, expert, fp4, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #36127 - [Model] Add support for moonshotai/Kimi-Audio-7B-Instruct

- Link: https://github.com/vllm-project/vllm/pull/36127
- Status/date: `merged`, created 2026-03-05, merged 2026-03-11; author `tunglinwood`.
- Diff scope read: `14` files, `+1446/-29`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: spec, attention, config, processor, cache, kv, test, vision, quant.
- Code diff details:
  - `vllm/model_executor/models/kimi_audio.py` added +725/-0 (725 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__, KimiAudioProcessingInfo
  - `vllm/tokenizers/kimi_audio.py` added +410/-0 (410 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: _load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained, __init__
  - `vllm/transformers_utils/processors/kimi_audio.py` added +163/-0 (163 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: _get_feat_extract_output_lengths, KimiAudioProcessor, __init__, check_argument_for_proper_class
  - `vllm/renderers/kimi_audio.py` added +49/-0 (49 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: KimiAudioRenderer, from_config
  - `vllm/transformers_utils/processors/__init__.py` modified +18/-17 (35 lines); hunks: import importlib; "GLM4VProcessor",; symbols: __getattr__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py`; keywords observed in patches: spec, attention, config, processor, cache, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #37438 - [Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support

- Link: https://github.com/vllm-project/vllm/pull/37438
- Status/date: `merged`, created 2026-03-18, merged 2026-03-19; author `DorBernsohn`.
- Diff scope read: `4` files, `+173/-18`; areas: tests/benchmarks; keywords: config, test, spec.
- Code diff details:
  - `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0 (155 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled, test_extract_reasoning_with_think_tags
  - `vllm/entrypoints/chat_utils.py` modified +14/-0 (14 lines); hunks: def get_history_tool_calls_cnt(conversation: list[ConversationMessage]):; symbols: get_history_tool_calls_cnt, get_tool_call_id_type, make_tool_call_id
  - `vllm/entrypoints/openai/chat_completion/serving.py` modified +2/-9 (11 lines); hunks: ChatTemplateContentFormatOption,; def __init__(; symbols: __init__
  - `vllm/entrypoints/openai/responses/serving.py` modified +2/-9 (11 lines); hunks: from vllm.entrypoints.chat_utils import (; def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/entrypoints/chat_utils.py`, `vllm/entrypoints/openai/chat_completion/serving.py`; keywords observed in patches: config, test, spec. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/entrypoints/chat_utils.py`, `vllm/entrypoints/openai/chat_completion/serving.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
