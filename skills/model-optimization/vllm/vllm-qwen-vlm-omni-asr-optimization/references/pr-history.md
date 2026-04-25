# vLLM Qwen2.5-VL / Qwen3-VL / Qwen3-Omni / Qwen3-ASR PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: Qwen2.5-VL, Qwen3-VL, Qwen3-Omni thinker, and Qwen3-ASR / realtime

## Landed PRs

### PR #13155 - Qwen2.5-VL Optimization

- Link: https://github.com/vllm-project/vllm/pull/13155
- State: merged
- Diff coverage: full diff reviewed, `2` files, `47` additions, `51` deletions
- Motivation:
  - The Qwen2.5-VL vision stack had an expensive SDPA fallback path and a
    model-local RMSNorm implementation that diverged from common vLLM layers.
- Key implementation:
  - Uses per-entry SDPA execution instead of building a large dense attention
    mask.
  - Switches the vision norm path to the shared `RMSNorm`.
  - Passes `use_flash_attn` into the vision rotary application path.
- Key code excerpts:

```diff
-            attention_mask = torch.zeros([1, seq_length, seq_length], ...)
-            output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
+            outputs = []
+            for i in range(1, len(cu_seqlens)):
+                output_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0)
+                outputs.append(output_i)
+            context_layer = torch.cat(outputs, dim=1)
```

```diff
-        norm_layer = partial(Qwen2RMSNorm, eps=norm_eps)
+        norm_layer = partial(RMSNorm, eps=norm_eps)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py`,
    `vllm/model_executor/models/qwen2_vl.py`
- Validation implications:
  - When Qwen2.5-VL is slow or OOMs on the fallback backend, inspect this PR
    before changing higher-level multimodal code.

### PR #24727 - Support Qwen3-VL Model Series

- Link: https://github.com/vllm-project/vllm/pull/24727
- State: merged
- Diff coverage: full diff reviewed, `13` files, `2084` additions, `17`
  deletions
- Motivation:
  - Qwen3-VL introduced native image and video placeholders, video processors,
    Qwen3 text integration, and Qwen3-VL-MoE.
- Key implementation:
  - Adds `qwen3_vl.py` and `qwen3_vl_moe.py`.
  - Registers both `Qwen3VLForConditionalGeneration` and
    `Qwen3VLMoeForConditionalGeneration`.
  - Adds image and video prompt replacement, timestamp handling, video grid
    parsing, and multimodal embedding merge for both modalities.
- Key code excerpts:

```diff
+    "Qwen3VLForConditionalGeneration": ("qwen3_vl", "Qwen3VLForConditionalGeneration"),
+    "Qwen3VLMoeForConditionalGeneration": ("qwen3_vl_moe", "Qwen3VLMoeForConditionalGeneration"),
```

```diff
+            PromptReplacement(
+                modality="video",
+                target="<|vision_start|><|video_pad|><|vision_end|>",
+                replacement=get_video_replacement_qwen3vl,
+            ),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py`,
    `vllm/model_executor/models/qwen3_vl_moe.py`,
    `vllm/model_executor/models/registry.py`
  - rotary/video support:
    `vllm/model_executor/layers/rotary_embedding/mrope.py`,
    `vllm/multimodal/video.py`
  - docs/tests/examples: `docs/models/supported_models.md`,
    `tests/models/registry.py`,
    `tests/models/multimodal/processing/test_common.py`,
    `examples/offline_inference/vision_language.py`
- Validation implications:
  - Regressions often show up as wrong video placeholder counts, wrong
    timestamps, or wrong `(3, seq_len)` MRoPE positions rather than simple load
    failures.

### PR #25055 - Add Triton kernel for Qwen3-VL interleaved MRoPE

- Link: https://github.com/vllm-project/vllm/pull/25055
- State: merged
- Diff coverage: full diff reviewed, `2` files, `88` additions, `46` deletions
- Motivation:
  - Qwen3-VL uses interleaved 3D rotary layout, which could not be handled
    cleanly by the earlier chunked MRoPE logic.
- Key implementation:
  - Adds `apply_interleaved_rope(...)`.
  - Introduces `mrope_interleaved` and updates the Triton path to distinguish
    interleaved vs. non-interleaved sections.
  - Extends MRoPE tests to Qwen3-VL models.
- Key code excerpts:

```diff
+def apply_interleaved_rope(x: torch.Tensor,
+                           mrope_section: list[int]) -> torch.Tensor:
+    """Apply interleaved MRoPE to 3D rotary embeddings."""
```

```diff
+        if self.mrope_interleaved:
+            cos = apply_interleaved_rope(cos, self.mrope_section)
+            sin = apply_interleaved_rope(sin, self.mrope_section)
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/rotary_embedding/mrope.py`
  - tests: `tests/kernels/core/test_mrope.py`
- Validation implications:
  - If Qwen3-VL image/video outputs drift only on specific backends or TP sizes,
    check interleaved MRoPE first.

### PR #25550 - Add Qwen3-Omni moe thinker

- Link: https://github.com/vllm-project/vllm/pull/25550
- State: merged
- Diff coverage: full diff reviewed, `6` files, `1795` additions, `36`
  deletions
- Motivation:
  - Qwen3-Omni thinker combines Qwen3-MoE text, a visual tower, an audio tower,
    and special `use_audio_in_video` prompt semantics that cannot be handled by
    the older Qwen2.5-Omni thinker unchanged.
- Key implementation:
  - Adds `qwen3_omni_moe_thinker.py`.
  - Registers `Qwen3OmniMoeThinkerForConditionalGeneration`.
  - Reimplements multimodal prompt updates so `use_audio_in_video` adjusts
    placeholder accounting instead of double-counting audio items.
- Key code excerpts:

```diff
+    "Qwen3OmniMoeForConditionalGeneration": (
+        "qwen3_omni_moe_thinker",
+        "Qwen3OmniMoeThinkerForConditionalGeneration",
+    ),
```

```diff
+        if use_audio_in_video and "video" in mm_item_counts:
+            assert "audio" in mm_item_counts
+            mm_item_counts["audio"] -= mm_item_counts["video"]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`,
    `vllm/model_executor/models/registry.py`
  - rotary support: `vllm/model_executor/layers/rotary_embedding/mrope.py`
  - tests/docs: `tests/models/registry.py`,
    `tests/models/multimodal/processing/test_common.py`,
    `docs/models/supported_models.md`
- Validation implications:
  - Always validate both `use_audio_in_video=True` and `False`.
  - Placeholder count mismatches can be processor bugs even when the base model
    loads successfully.

### PR #33312 - Qwen3-ASR

- Link: https://github.com/vllm-project/vllm/pull/33312
- State: merged
- Diff coverage: full diff reviewed, `9` files, `1269` additions
- Motivation:
  - Qwen3-ASR needs speech-specific configs, audio processor wiring, audio
    feature-length accounting, and transcription support on top of Qwen3 text.
- Key implementation:
  - Adds `qwen3_asr.py`, `transformers_utils/configs/qwen3_asr.py`, and
    `transformers_utils/processors/qwen3_asr.py`.
  - Reuses Omni audio encoder pieces and computes audio output lengths from
    feature lengths for prompt replacement.
- Key code excerpts:

```diff
+class Qwen3ASRMultiModalProcessor(
+    Qwen3OmniMoeThinkerMultiModalProcessor,
+):
```

```diff
+def _get_feat_extract_output_lengths(input_lengths: torch.Tensor):
+    ...
+    return output_lengths
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_asr.py`,
    `vllm/model_executor/models/registry.py`
  - configs/processors:
    `vllm/transformers_utils/configs/qwen3_asr.py`,
    `vllm/transformers_utils/processors/qwen3_asr.py`,
    `vllm/transformers_utils/config.py`
  - docs/examples/tests: `docs/models/supported_models.md`,
    `examples/offline_inference/audio_language.py`,
    `tests/models/registry.py`
- Validation implications:
  - Audio length bugs can surface as wrong prompt expansion or missing output
    segments even when transcription itself seems wired up.

### PR #34613 - Add Qwen3-ASR realtime streaming support

- Link: https://github.com/vllm-project/vllm/pull/34613
- State: merged
- Diff coverage: full diff reviewed, `5` files, `256` additions, `1` deletion
- Motivation:
  - The batch ASR path was not sufficient for realtime streaming; the endpoint
    needed buffering, per-segment prompt expansion, and a separate model class.
- Key implementation:
  - Adds `qwen3_asr_realtime.py`.
  - Extends `SupportsRealtime` with `realtime_max_tokens`.
  - Expands `<|audio_pad|>` to the true per-segment audio token count before
    MRoPE positions are computed.
- Key code excerpts:

```diff
+    realtime_max_tokens: ClassVar[int] = 1
+    """Maximum tokens to generate per streaming audio segment."""
```

```diff
+class Qwen3ASRRealtimeGeneration(Qwen3ASRForConditionalGeneration, SupportsRealtime):
+    realtime_max_tokens = 64
```

```diff
+        if tid == audio_pad_id and pad_start_idx == -1:
+            pad_start_idx = i
+            expanded_ids.extend([audio_pad_id] * audio_len)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_asr_realtime.py`,
    `vllm/model_executor/models/interfaces.py`,
    `vllm/model_executor/models/registry.py`
  - endpoint: `vllm/entrypoints/openai/realtime/connection.py`
  - tests: `tests/models/registry.py`
- Validation implications:
  - Realtime regressions can come from buffer segmentation and placeholder
    expansion, not only from endpoint glue.

## Open PR Radar

- Re-run PR search before claiming new Qwen3-Omni / ASR follow-ups beyond the
  checked mainline commit.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM Qwen VLM / Omni / ASR PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-02-12 | [#13155](https://github.com/vllm-project/vllm/pull/13155) | merged | [Misc] Qwen2.5-VL Optimization | model wrapper, scheduler/runtime | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-09-12 | [#24727](https://github.com/vllm-project/vllm/pull/24727) | merged | [Model] Support Qwen3-VL Model Series | model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py` |
| 2025-09-17 | [#25055](https://github.com/vllm-project/vllm/pull/25055) | merged | [Kernel][Performance] Add Triton kernel for Qwen3-VL interleaved MRoPE | kernel, scheduler/runtime, tests/benchmarks | `tests/kernels/core/test_mrope.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py` |
| 2025-09-24 | [#25550](https://github.com/vllm-project/vllm/pull/25550) | merged | Add Qwen3-Omni moe thinker | model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`, `tests/models/registry.py` |
| 2026-01-29 | [#33312](https://github.com/vllm-project/vllm/pull/33312) | merged | [Models] Qwen3-ASR | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py` |
| 2026-02-16 | [#34613](https://github.com/vllm-project/vllm/pull/34613) | merged | [Realtime] Add Qwen3-ASR realtime streaming support | model wrapper, scheduler/runtime, tests/benchmarks | `vllm/model_executor/models/qwen3_asr_realtime.py`, `tests/models/registry.py`, `vllm/model_executor/models/interfaces.py` |

## Diff Cards

### PR #13155 - [Misc] Qwen2.5-VL Optimization

- Link: https://github.com/vllm-project/vllm/pull/13155
- Status/date: `merged`, created 2025-02-12, merged 2025-02-13; author `wulipc`.
- Diff scope read: `2` files, `+47/-51`; areas: model wrapper, scheduler/runtime; keywords: attention, flash, vision, config, quant.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +25/-36 (61 lines); hunks: from vllm.logger import init_logger; def forward(; symbols: forward, forward, forward, Qwen2RMSNorm
  - `vllm/model_executor/models/qwen2_vl.py` modified +22/-15 (37 lines); hunks: def apply_rotary_emb_torch(x: torch.Tensor,; def forward(; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision, forward
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; keywords observed in patches: attention, flash, vision, config, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #24727 - [Model] Support Qwen3-VL Model Series

- Link: https://github.com/vllm-project/vllm/pull/24727
- Status/date: `merged`, created 2025-09-12, merged 2025-09-17; author `ywang96`.
- Diff scope read: `13` files, `+2084/-17`; areas: model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: moe, vision, config, processor, test, cuda, fp8, kv, quant, scheduler.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` added +1478/-0 (1478 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Qwen3_VisionPatchEmbed, __init__, forward, Qwen3_VisionMLP
  - `vllm/model_executor/models/qwen3_vl_moe.py` added +344/-0 (344 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Qwen3VLMoeProcessingInfo, get_hf_config, Qwen3MoeLLMModel, __init__
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +133/-11 (144 lines); hunks: def triton_mrope(; def __init__(; symbols: triton_mrope, apply_interleaved_rope, MRotaryEmbedding, __init__
  - `examples/offline_inference/vision_language.py` modified +78/-0 (78 lines); hunks: def run_qwen2_5_omni(questions: list[str], modality: str):; def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:; symbols: run_qwen2_5_omni, run_qwen3_vl, run_qwen3_vl_moe, run_r_vl
  - `tests/models/multimodal/processing/test_common.py` modified +34/-1 (35 lines); hunks: def glm4_1v_patch_mm_data(mm_data: MultiModalDataDict) -> MultiModalDataDict:; def glm4_1v_patch_mm_data(mm_data: MultiModalDataDict) -> MultiModalDataDict:; symbols: glm4_1v_patch_mm_data, glm4_1v_patch_mm_data, qwen3_vl_patch_mm_data, create_metadata
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`; keywords observed in patches: moe, vision, config, processor, test, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #25055 - [Kernel][Performance] Add Triton kernel for Qwen3-VL interleaved MRoPE

- Link: https://github.com/vllm-project/vllm/pull/25055
- Status/date: `merged`, created 2025-09-17, merged 2025-09-19; author `Isotr0py`.
- Diff scope read: `2` files, `+88/-46`; areas: kernel, scheduler/runtime, tests/benchmarks; keywords: cuda, attention, cache, config, kv, test, triton.
- Code diff details:
  - `tests/kernels/core/test_mrope.py` modified +66/-32 (98 lines); hunks: # SPDX-License-Identifier: Apache-2.0; def generate_test_data(num_tokens: int, num_q_heads: int, num_kv_heads: int,; symbols: generate_test_data, generate_test_data, unroll_model_tp_dict, MRoPETestInfo
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +22/-14 (36 lines); hunks: @triton.jit; def _triton_qwen2vl_mrope_forward(; symbols: _triton_qwen2vl_mrope_forward, _triton_mrope_forward, _triton_qwen2vl_mrope_forward, _triton_qwen2vl_mrope_forward
- Optimization/support interpretation: The concrete diff surface is `tests/kernels/core/test_mrope.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`; keywords observed in patches: cuda, attention, cache, config, kv, test. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `tests/kernels/core/test_mrope.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #25550 - Add Qwen3-Omni moe thinker

- Link: https://github.com/vllm-project/vllm/pull/25550
- Status/date: `merged`, created 2025-09-24, merged 2025-10-10; author `wangxiongts`.
- Diff scope read: `6` files, `+1795/-36`; areas: model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: moe, config, processor, test, vision, attention, cache, doc, flash, kv.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` added +1409/-0 (1409 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Qwen3_VisionPatchEmbed, __init__, forward, Qwen3_VisionMLP
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +374/-34 (408 lines); hunks: def split_thw(grid_thw: Union[torch.Tensor, list[int]]) -> list[list[int]]:; def _vl_get_input_positions_tensor(; symbols: split_thw, _vl_get_input_positions_tensor, _vl_get_input_positions_tensor, _vl_get_input_positions_tensor
  - `tests/models/registry.py` modified +5/-0 (5 lines); hunks: def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +2/-2 (4 lines); hunks: These models primarily accept the `LLM.generate` (./generative_models.md#llmgen; Some models are supported only via the [Transformers backend](#transformers).
  - `vllm/model_executor/models/registry.py` modified +4/-0 (4 lines); hunks: "qwen2_5_omni_thinker",
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`, `tests/models/registry.py`; keywords observed in patches: moe, config, processor, test, vision, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`, `tests/models/registry.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #33312 - [Models] Qwen3-ASR

- Link: https://github.com/vllm-project/vllm/pull/33312
- Status/date: `merged`, created 2026-01-29, merged 2026-01-29; author `ywang96`.
- Diff scope read: `9` files, `+1269/-0`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: spec, attention, config, doc, moe, processor, cache, flash, kv, quant.
- Code diff details:
  - `vllm/model_executor/models/qwen3_asr.py` added +567/-0 (567 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: _get_feat_extract_output_lengths, Qwen3ASRProcessingInfo, get_hf_config, get_hf_processor
  - `vllm/transformers_utils/configs/qwen3_asr.py` added +436/-0 (436 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Qwen3ASRAudioEncoderConfig, to, __init__, Qwen3ASRTextConfig
  - `vllm/transformers_utils/processors/qwen3_asr.py` added +231/-0 (231 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Qwen3ASRProcessorKwargs, _get_feat_extract_output_lengths, Qwen3ASRProcessor, to
  - `examples/offline_inference/audio_language.py` modified +20/-0 (20 lines); hunks: def run_qwen2_5_omni(question: str, audio_count: int):; def run_whisper(question: str, audio_count: int) -> ModelRequestData:; symbols: run_qwen2_5_omni, run_qwen3_asr, run_ultravox, run_whisper
  - `tests/models/registry.py` modified +6/-0 (6 lines); hunks: def check_available_online(; symbols: check_available_online
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py`; keywords observed in patches: spec, attention, config, doc, moe, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #34613 - [Realtime] Add Qwen3-ASR realtime streaming support

- Link: https://github.com/vllm-project/vllm/pull/34613
- Status/date: `merged`, created 2026-02-16, merged 2026-02-21; author `pougetat`.
- Diff scope read: `5` files, `+256/-1`; areas: model wrapper, scheduler/runtime, tests/benchmarks; keywords: cache, config, moe, processor, spec, test.
- Code diff details:
  - `vllm/model_executor/models/qwen3_asr_realtime.py` added +239/-0 (239 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Qwen3ASRRealtimeBuffer:, __init__, write_audio, read_audio
  - `tests/models/registry.py` modified +8/-0 (8 lines); hunks: def check_available_online(; symbols: check_available_online
  - `vllm/model_executor/models/interfaces.py` modified +4/-0 (4 lines); hunks: class SupportsRealtime(Protocol):; symbols: SupportsRealtime, buffer_realtime_audio
  - `vllm/model_executor/models/registry.py` modified +4/-0 (4 lines); hunks: "qwen3_asr",
  - `vllm/entrypoints/openai/realtime/connection.py` modified +1/-1 (2 lines); hunks: async def _run_generation(; symbols: _run_generation
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/qwen3_asr_realtime.py`, `tests/models/registry.py`, `vllm/model_executor/models/interfaces.py`; keywords observed in patches: cache, config, moe, processor, spec, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/qwen3_asr_realtime.py`, `tests/models/registry.py`, `vllm/model_executor/models/interfaces.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
