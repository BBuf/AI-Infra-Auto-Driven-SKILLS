# Moss-VL PR History

Evidence sweep:

- SGLang `origin/main`: `bca3dd958` (`2026-04-24`)
- Manual diff review date: `2026-04-24`
- Searched paths: Moss-VL model, processor, multimodal scheduler fields, conversation template, server args.
- Searched PR terms: `Moss-VL`, `MossVL`, `moss_vl`, `moss-vl`.

## Runtime Surfaces

- `python/sglang/srt/models/moss_vl.py`
- `python/sglang/srt/multimodal/processors/moss_vl.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/parser/conversation.py`
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/configs/model_config.py`

## Diff-Reviewed PR Cards

### PR #23454 - Add Moss-VL Python runtime support

- Link: https://github.com/sgl-project/sglang/pull/23454
- State: merged at `2026-04-24T03:14:29Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `3397` lines, `10` files; current-main source rechecked at `bca3dd958`.
- Motivation: Moss-VL needs SGLang-native multimodal serving rather than being routed through a generic VLM path. The model combines a Qwen3VL-like vision encoder, Moss-specific text/cross-attention layers, per-frame visibility masks, and image/video placeholder semantics that require new scheduler and processor metadata.
- Key implementation:
  - registers `MossVLForConditionalGeneration` as multimodal, encoder-decoder, and chunked-prefill-unsupported in `model_config.py`;
  - adds `MossVLForConditionalGeneration`, vision encoder, text model, cross-attention layers, separator-token insertion, and HF-to-SGLang weight mapping in `moss_vl.py`;
  - adds `MossVLImageProcessor` to normalize image/video inputs, build `vision_position_ids`, compute visible frame counts, and clean temporary video files;
  - extends `MultimodalProcessorOutput` and `MultimodalInputs` with `vision_position_ids`, `media_nums_per_sample`, and `visible_frame_counts`;
  - registers the `moss-vl` conversation template and forces flashinfer prefill for custom cross-attention masks.
- Key code excerpts:

```diff
+    "MossVLForConditionalGeneration",
```

```python
class MossVLForConditionalGeneration(nn.Module):
    self.visual = MossVLVisionModel(...)
    self.language_model = MossVLForCausalLM(...)
    self.separator_token = nn.Parameter(torch.zeros(vision_config.out_hidden_size))
```

```python
def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
    encoder_len = self._get_encoder_len(mm_inputs)
    mm_inputs.num_image_tokens = encoder_len
    return self._build_encoder_prefix_pad_ids(mm_inputs) + input_ids
```

```python
custom_mask = self._build_cross_attention_custom_mask(forward_batch)
if custom_mask is not None:
    forward_batch.cross_attention_custom_mask = custom_mask
```

```python
elif model_arch == "MossVLForConditionalGeneration":
    self.prefill_attention_backend = "flashinfer"
    assert prefill_backend == "flashinfer"
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`
  - processor/template: `python/sglang/srt/multimodal/processors/moss_vl.py`, `python/sglang/srt/parser/conversation.py`
  - config: `python/sglang/srt/configs/model_config.py`
- Validation implications: Moss-VL tests must cover image and video processor output, encoder-prefix stripping, frame-level cross-attention masks, cached encoder KV, and the flashinfer prefill assertion. A plain model-load smoke is not enough.

## Validation Notes

- Moss-VL source touches shared multimodal scheduler structures; regressions may show up in logprob start lengths or encoder prefix stripping.
- If a bug appears only after first-token prefill, inspect `visible_frame_counts` shrinkage and `mm_input.release_features()` before looking at logits.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG MOSS-VL PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-22 | [#23454](https://github.com/sgl-project/sglang/pull/23454) | merged | [srt] Add Moss-VL Python runtime support | model wrapper, attention/backend, multimodal/processor, scheduler/runtime, docs/config | `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`, `python/sglang/srt/managers/schedule_batch.py` |

## Diff Cards

### PR #23454 - [srt] Add Moss-VL Python runtime support

- Link: https://github.com/sgl-project/sglang/pull/23454
- Status/date: `merged`, created 2026-04-22, merged 2026-04-24; author `zsj555`.
- Diff scope read: `10` files, `+2401/-6`; areas: model wrapper, attention/backend, multimodal/processor, scheduler/runtime, docs/config; keywords: attention, config, processor, spec, cuda, flash, vision, cache, kv, mla.
- Code diff details:
  - `python/sglang/srt/models/moss_vl.py` added +1643/-0 (1643 lines); hunks: +"""PyTorch Moss-VL model for SGLang - Qwen3VL Vision + Text with Cross Attention."""; symbols: MossVLVisionMLP, __init__, forward, MossVLVisionPatchEmbed
  - `python/sglang/srt/multimodal/processors/moss_vl.py` added +612/-0 (612 lines); hunks: +import asyncio; symbols: MossVLImageProcessor, __init__, _build_mm_items, _build_vision_token_info
  - `python/sglang/srt/managers/schedule_batch.py` modified +70/-0 (70 lines); hunks: class MultimodalProcessorOutput:; def from_dict(d: dict) -> "MultimodalProcessorOutput":; symbols: MultimodalProcessorOutput:, from_dict, MultimodalInputs:, release_features
  - `python/sglang/srt/parser/conversation.py` modified +29/-2 (31 lines); hunks: def get_prompt(self) -> str:; def generate_chat_conv(; symbols: get_prompt, generate_chat_conv, generate_chat_conv, generate_chat_conv
  - `python/sglang/srt/managers/tokenizer_manager.py` modified +12/-2 (14 lines); hunks: async def _tokenize_one_request(; symbols: _tokenize_one_request
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`, `python/sglang/srt/managers/schedule_batch.py`; keywords observed in patches: attention, config, processor, spec, cuda, flash. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`, `python/sglang/srt/managers/schedule_batch.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
