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
