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
