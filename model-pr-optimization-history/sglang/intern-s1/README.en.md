# SGLang Intern-S1 Support and PR History

This note tracks the SGLang runtime, key PRs, and cookbook-facing touchpoints for Intern-S1.

- Status: 当前 mainline 已支持

## Key Conclusions

- Intern-S1 leans heavily on shared InternVL processor code in SGLang.
- Most regressions come from processor compatibility, parser behavior, and video-aware serving rather than the text stack alone.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/interns1.py`
- `sglang/python/sglang/srt/models/internvl.py`

## Landed PRs

- [#9381](https://github.com/sgl-project/sglang/pull/9381) `InternS1 image token updates in InternVL processor`: Aligned the shared processor with Intern-S1 image semantics.
- [#12367](https://github.com/sgl-project/sglang/pull/12367) `Fix Intern-S1 accuracy and `/generate` input_ids support`: Closed early correctness gaps.
- [#14866](https://github.com/sgl-project/sglang/pull/14866) `Add tool calling and reasoning parser support for Intern-S1`: Added parser support that cookbook usage depends on.
- [#17040](https://github.com/sgl-project/sglang/pull/17040) `Support InternS1 text_config in InternVL processor`: Improved sub-config compatibility in shared processors.

## Matching Skill

- `skills/model-optimization/sglang/sglang-intern-s1-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-intern-s1-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Intern-S1` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-20 | [#9381](https://github.com/sgl-project/sglang/pull/9381) | merged | fix: InternS1 don't recognize image, updates image token for InternVL processor | multimodal/processor | `python/sglang/srt/conversation.py`, `python/sglang/srt/multimodal/processors/internvl.py` |
| 2025-10-30 | [#12367](https://github.com/sgl-project/sglang/pull/12367) | merged | [Bug] Fix Intern-S1 model accuracy and support /generate interface with input_ids | model wrapper, multimodal/processor | `python/sglang/srt/models/interns1.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py` |
| 2025-12-11 | [#14866](https://github.com/sgl-project/sglang/pull/14866) | merged | Adding tool calling and reasoning parser support for Intern-S1 | misc | `python/sglang/srt/function_call/internlm_detector.py`, `python/sglang/srt/constrained/base_grammar_backend.py`, `python/sglang/srt/constrained/xgrammar_backend.py` |
| 2026-01-13 | [#17040](https://github.com/sgl-project/sglang/pull/17040) | merged | fix(processor): support InternS1 text_config in InternVL processor | multimodal/processor | `python/sglang/srt/multimodal/processors/internvl.py` |

### File-level PR diff reading notes

### PR #9381 - fix: InternS1 don't recognize image, updates image token for InternVL processor

- Link: https://github.com/sgl-project/sglang/pull/9381
- Status/date: `merged`, created 2025-08-20, merged 2025-08-20; author `JustinTong0323`.
- Diff scope read: `2` files, `+9/-17`; areas: multimodal/processor; keywords: config, expert, processor, spec, vision.
- Code diff details:
  - `python/sglang/srt/conversation.py` modified +2/-15 (17 lines); hunks: def generate_chat_conv(; def generate_chat_conv(; symbols: generate_chat_conv, generate_chat_conv
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2 (9 lines); hunks: def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):; def process_image_internvl(image, input_size=448, max_num=12):; symbols: __init__, process_image_internvl
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/conversation.py`, `python/sglang/srt/multimodal/processors/internvl.py`; keywords observed in patches: config, expert, processor, spec, vision. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/conversation.py`, `python/sglang/srt/multimodal/processors/internvl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12367 - [Bug] Fix Intern-S1 model accuracy and support /generate interface with input_ids

- Link: https://github.com/sgl-project/sglang/pull/12367
- Status/date: `merged`, created 2025-10-30, merged 2025-11-03; author `hhaAndroid`.
- Diff scope read: `3` files, `+8/-41`; areas: model wrapper, multimodal/processor; keywords: attention, config, flash, fp8, processor, quant, vision.
- Code diff details:
  - `python/sglang/srt/models/interns1.py` modified +3/-21 (24 lines); hunks: -from typing import Iterable, List, Optional, Set, Tuple; def __init__(; symbols: __init__, pixel_shuffle, extract_feature, load_weights
  - `python/sglang/srt/models/internvl.py` modified +1/-19 (20 lines); hunks: -from typing import Iterable, List, Optional, Set, Tuple, Union; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights, load_weights
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +4/-1 (5 lines); hunks: async def process_mm_data_async(; symbols: process_mm_data_async
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/interns1.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py`; keywords observed in patches: attention, config, flash, fp8, processor, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/interns1.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14866 - Adding tool calling and reasoning parser support for Intern-S1

- Link: https://github.com/sgl-project/sglang/pull/14866
- Status/date: `merged`, created 2025-12-11, merged 2025-12-16; author `KennyYao2001`.
- Diff scope read: `6` files, `+290/-14`; areas: misc; keywords: kv, moe, spec.
- Code diff details:
  - `python/sglang/srt/function_call/internlm_detector.py` added +248/-0 (248 lines); hunks: +# modified from https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/tool_parser/internlm2_parser.py; symbols: InternlmDetector, __init__, has_tool_call, get_arguments
  - `python/sglang/srt/constrained/base_grammar_backend.py` modified +19/-7 (26 lines); hunks: def create_grammar_backend(; symbols: create_grammar_backend
  - `python/sglang/srt/constrained/xgrammar_backend.py` modified +18/-5 (23 lines); hunks: def __repr__(self):; def __init__(; symbols: __repr__, TokenizerNotSupportedError, XGrammarGrammarBackend, __init__
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +2/-2 (4 lines); hunks: def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:; symbols: _get_reasoning_from_request
  - `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunks: from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector; class FunctionCallParser:; symbols: FunctionCallParser:, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/internlm_detector.py`, `python/sglang/srt/constrained/base_grammar_backend.py`, `python/sglang/srt/constrained/xgrammar_backend.py`; keywords observed in patches: kv, moe, spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/internlm_detector.py`, `python/sglang/srt/constrained/base_grammar_backend.py`, `python/sglang/srt/constrained/xgrammar_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17040 - fix(processor): support InternS1 text_config in InternVL processor

- Link: https://github.com/sgl-project/sglang/pull/17040
- Status/date: `merged`, created 2026-01-13, merged 2026-01-26; author `Mahdi-CV`.
- Diff scope read: `1` files, `+12/-4`; areas: multimodal/processor; keywords: config, processor.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +12/-4 (16 lines); hunks: def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):; def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):; symbols: __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/multimodal/processors/internvl.py`; keywords observed in patches: config, processor. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/multimodal/processors/internvl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 4; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
