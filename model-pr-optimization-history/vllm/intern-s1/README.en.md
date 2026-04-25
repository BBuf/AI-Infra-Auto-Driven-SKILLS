# vLLM Intern-S1 Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for Intern-S1.

- Status: supported on current mainline

## Key Conclusions

- Intern-S1 leans heavily on shared InternVL processor code in vLLM.
- Most regressions come from processor compatibility and video-aware serving rather than the text stack alone.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/interns1.py`
- `vllm/vllm/model_executor/models/interns1_pro.py`

## Landed PRs

- [#21628](https://github.com/vllm-project/vllm/pull/21628) `Support Intern-S1`: Initial Intern-S1 support in vLLM.
- [#21671](https://github.com/vllm-project/vllm/pull/21671) `Add video support for Intern-S1`: Extended the family beyond static images.
- [#22417](https://github.com/vllm-project/vllm/pull/22417) `Fix wrong method name in Intern-S1 image processor`: Patched a processor bug after bring-up.
- [#33636](https://github.com/vllm-project/vllm/pull/33636) `Intern-S1-Pro`: Added the Pro generation / alias path.

## Matching Skill

- `skills/model-optimization/vllm/vllm-intern-s1-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-intern-s1-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Intern-S1` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-25 | [#21628](https://github.com/vllm-project/vllm/pull/21628) | merged | Support Intern-S1 | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/interns1_vit.py`, `examples/offline_inference/vision_language.py` |
| 2025-07-27 | [#21671](https://github.com/vllm-project/vllm/pull/21671) | merged | [VLM] Add video support for Intern-S1 | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/interns1.py`, `examples/offline_inference/vision_language.py`, `docs/models/supported_models.md` |
| 2025-08-07 | [#22417](https://github.com/vllm-project/vllm/pull/22417) | merged | [Bugfix] Fix wrong method name in Intern-S1 image processor | model wrapper, scheduler/runtime | `vllm/model_executor/models/interns1.py` |
| 2026-02-03 | [#33636](https://github.com/vllm-project/vllm/pull/33636) | merged | [Models] Intern-S1-Pro | model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/interns1_pro.py`, `vllm/model_executor/layers/rotary_embedding/fope.py`, `examples/offline_inference/vision_language.py` |

### File-level PR diff reading notes

### PR #21628 - Support Intern-S1

- Link: https://github.com/vllm-project/vllm/pull/21628
- Status/date: `merged`, created 2025-07-25, merged 2025-07-26; author `lvhan028`.
- Diff scope read: `7` files, `+1196/-0`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: vision, config, quant, spec, attention, doc, lora, processor, test.
- Code diff details:
  - `vllm/model_executor/models/interns1.py` added +711/-0 (711 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: InternS1MultiModalProjector, __init__, forward, InternS1ImagePixelInputs
  - `vllm/model_executor/models/interns1_vit.py` added +421/-0 (421 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: InternS1VisionPatchEmbeddings, __init__, forward, InternS1VisionEmbeddings
  - `examples/offline_inference/vision_language.py` modified +32/-0 (32 lines); hunks: def run_tarsier(questions: list[str], modality: str) -> ModelRequestData:; def run_skyworkr1v(questions: list[str], modality: str) -> ModelRequestData:; symbols: run_tarsier, run_interns1, run_internvl, run_skyworkr1v
  - `examples/offline_inference/vision_language_multi_image.py` modified +28/-0 (28 lines); hunks: def load_smolvlm(question: str, image_urls: list[str]) -> ModelRequestData:; def load_tarsier2(question: str, image_urls: list[str]) -> ModelRequestData:; symbols: load_smolvlm, load_interns1, load_internvl, load_tarsier2
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: def check_available_online(; symbols: check_available_online
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/interns1_vit.py`, `examples/offline_inference/vision_language.py`; keywords observed in patches: vision, config, quant, spec, attention, doc. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/interns1_vit.py`, `examples/offline_inference/vision_language.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21671 - [VLM] Add video support for Intern-S1

- Link: https://github.com/vllm-project/vllm/pull/21671
- Status/date: `merged`, created 2025-07-27, merged 2025-07-27; author `Isotr0py`.
- Diff scope read: `5` files, `+173/-50`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: spec, vision, config, doc, processor, test.
- Code diff details:
  - `vllm/model_executor/models/interns1.py` modified +166/-45 (211 lines); hunks: from collections.abc import Iterable, Mapping, Sequence; def get_interns1_target_ratios(; symbols: get_interns1_target_ratios, InternS1ProcessingInfo, get_hf_processor, get_supported_mm_limits
  - `examples/offline_inference/vision_language.py` modified +5/-3 (8 lines); hunks: def run_tarsier(questions: list[str], modality: str) -> ModelRequestData:; def run_interns1(questions: list[str], modality: str) -> ModelRequestData:; symbols: run_tarsier, run_interns1, run_interns1
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: Specified using `--task generate`.
  - `tests/models/multimodal/processing/test_common.py` modified +1/-0 (1 lines); hunks: def _test_processing_correctness_one(; symbols: _test_processing_correctness_one
  - `vllm/model_executor/models/internvl.py` modified +0/-1 (1 lines); hunks: def get_multimodal_embeddings(self,; symbols: get_multimodal_embeddings
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/interns1.py`, `examples/offline_inference/vision_language.py`, `docs/models/supported_models.md`; keywords observed in patches: spec, vision, config, doc, processor, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/interns1.py`, `examples/offline_inference/vision_language.py`, `docs/models/supported_models.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22417 - [Bugfix] Fix wrong method name in Intern-S1 image processor

- Link: https://github.com/vllm-project/vllm/pull/22417
- Status/date: `merged`, created 2025-08-07, merged 2025-08-07; author `DarkLight1337`.
- Diff scope read: `1` files, `+1/-1`; areas: model wrapper, scheduler/runtime; keywords: processor.
- Code diff details:
  - `vllm/model_executor/models/interns1.py` modified +1/-1 (2 lines); hunks: def get_num_image_tokens(; symbols: get_num_image_tokens
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/interns1.py`; keywords observed in patches: processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/interns1.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #33636 - [Models] Intern-S1-Pro

- Link: https://github.com/vllm-project/vllm/pull/33636
- Status/date: `merged`, created 2026-02-03, merged 2026-02-03; author `CUHKSZzxy`.
- Diff scope read: `11` files, `+942/-11`; areas: model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: cache, vision, config, moe, expert, kv, attention, flash, processor, quant.
- Code diff details:
  - `vllm/model_executor/models/interns1_pro.py` added +633/-0 (633 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: InternS1ProProcessingInfo, get_hf_config, get_hf_processor, InternS1ProMoeMLP
  - `vllm/model_executor/layers/rotary_embedding/fope.py` added +199/-0 (199 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: FourierRotaryEmbedding, __init__, _compute_inv_freq, _compute_cos_sin_cache
  - `examples/offline_inference/vision_language.py` modified +35/-0 (35 lines); hunks: def run_interns1(questions: list[str], modality: str) -> ModelRequestData:; def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:; symbols: run_interns1, run_interns1_pro, run_internvl, run_tarsier2
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +23/-0 (23 lines); hunks: from .dual_chunk_rope import DualChunkRotaryEmbedding; def get_rope(; symbols: get_rope
  - `vllm/model_executor/layers/rotary_embedding/base.py` modified +15/-6 (21 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/interns1_pro.py`, `vllm/model_executor/layers/rotary_embedding/fope.py`, `examples/offline_inference/vision_language.py`; keywords observed in patches: cache, vision, config, moe, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/interns1_pro.py`, `vllm/model_executor/layers/rotary_embedding/fope.py`, `examples/offline_inference/vision_language.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 4; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
