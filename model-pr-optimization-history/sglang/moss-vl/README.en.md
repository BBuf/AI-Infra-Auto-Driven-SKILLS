# SGLang Moss-VL Support and Optimization Timeline

Scope: Moss-VL native SGLang runtime, image/video processor, conversation template, multimodal scheduler metadata, cross-attention custom masks, and flashinfer prefill requirement.

Evidence snapshot: SGLang `origin/main` `bca3dd958` (`2026-04-24`). Full dossier: `skills/model-optimization/sglang/sglang-moss-vl-optimization/references/pr-history.md`.

## Diff-Reviewed PR

#23454 added Moss-VL runtime support. The full diff was reviewed (`3397` lines, `10` files). The PR adds `moss_vl.py`, `multimodal/processors/moss_vl.py`, Moss-VL fields in `schedule_batch.py`, a `moss-vl` conversation template, model-config registration, and a `flashinfer` prefill requirement.

Key contract: Moss-VL vision tokens include separator tokens, frame visibility is converted into a packed cross-attention custom mask, and encoder-prefix placeholder tokens are stripped before text extend.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `MOSS-VL` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-22 | [#23454](https://github.com/sgl-project/sglang/pull/23454) | merged | [srt] Add Moss-VL Python runtime support | model wrapper, attention/backend, multimodal/processor, scheduler/runtime, docs/config | `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`, `python/sglang/srt/managers/schedule_batch.py` |

### File-level PR diff reading notes

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


### Gap and optimization follow-up

- Covered PRs: 1; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
