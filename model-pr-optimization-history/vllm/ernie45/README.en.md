# vLLM Ernie4.5 / Ernie4.5-VL Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for Ernie4.5 / Ernie4.5-VL.

- Status: supported on current mainline

## Key Conclusions

- Ernie4.5 spans dense, MoE, and VL paths in vLLM.
- The highest-risk work items are shared-expert behavior, VL rotary/timestamp logic, and long-input stability.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/ernie45.py`
- `vllm/vllm/model_executor/models/ernie45_moe.py`
- `vllm/vllm/model_executor/models/ernie45_vl.py`

## Landed PRs

- [#20220](https://github.com/vllm-project/vllm/pull/20220) `Add Ernie4.5 and Ernie4.5MoE Model Support`: Landed text and MoE support.
- [#21717](https://github.com/vllm-project/vllm/pull/21717) `Fix Ernie4_5_MoeForCausalLM shared experts`: Fixed shared-expert correctness.
- [#22514](https://github.com/vllm-project/vllm/pull/22514) `Add Ernie4.5 VL Model Support`: Added the multimodal Ernie4.5-VL lane.
- [#24074](https://github.com/vllm-project/vllm/pull/24074) `Fix Ernie4.5-VL hanging on long inputs`: Closed a production long-input stall.
- [#31274](https://github.com/vllm-project/vllm/pull/31274) `Support video metadata for timestamp rendering`: Improved VL video output fidelity.

## Matching Skill

- `skills/model-optimization/vllm/vllm-ernie45-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-ernie45-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `ERNIE 4.5 / ERNIE 4.5 VL` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-06-29 | [#20220](https://github.com/vllm-project/vllm/pull/20220) | merged | [Model] Add Ernie4.5 and Ernie4.5MoE Model Support | model wrapper, MoE/router, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45.py`, `tests/models/registry.py` |
| 2025-07-28 | [#21717](https://github.com/vllm-project/vllm/pull/21717) | merged | [Bugfix] Fix Ernie4_5_MoeForCausalLM shared experts | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/ernie45_moe.py` |
| 2025-08-08 | [#22514](https://github.com/vllm-project/vllm/pull/22514) | merged | [Model] Add Ernie4.5 VL Model Support | model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py` |
| 2025-09-02 | [#24074](https://github.com/vllm-project/vllm/pull/24074) | merged | [BugFix][Model] Fix Ernie4.5-VL hanging on long inputs | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py` |
| 2025-12-24 | [#31274](https://github.com/vllm-project/vllm/pull/31274) | merged | [Model][Ernie4.5-VL] Support video metadata for timestamp rendering | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks | `vllm/model_executor/models/ernie45_vl.py`, `tests/models/multimodal/processing/test_common.py` |

### File-level PR diff reading notes

### PR #20220 - [Model] Add Ernie4.5 and Ernie4.5MoE Model Support

- Link: https://github.com/vllm-project/vllm/pull/20220
- Status/date: `merged`, created 2025-06-29, merged 2025-07-02; author `CSWYF3634076`.
- Diff scope read: `5` files, `+634/-0`; areas: model wrapper, MoE/router, scheduler/runtime, tests/benchmarks, docs/config; keywords: kv, moe, spec, attention, config, cache, doc, expert, fp8, processor.
- Code diff details:
  - `vllm/model_executor/models/ernie45_moe.py` added +583/-0 (583 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Ernie4_5_MoeMLP, __init__, forward, Ernie4_5_MoeMoE
  - `vllm/model_executor/models/ernie45.py` added +43/-0 (43 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Ernie4_5_ForCausalLM, __init__
  - `tests/models/registry.py` modified +4/-0 (4 lines); hunks: def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +2/-0 (2 lines); hunks: Specified using `--task generate`.
  - `vllm/model_executor/models/registry.py` modified +2/-0 (2 lines); hunks: "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45.py`, `tests/models/registry.py`; keywords observed in patches: kv, moe, spec, attention, config, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45.py`, `tests/models/registry.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21717 - [Bugfix] Fix Ernie4_5_MoeForCausalLM shared experts

- Link: https://github.com/vllm-project/vllm/pull/21717
- Status/date: `merged`, created 2025-07-28, merged 2025-07-28; author `jeejeelee`.
- Diff scope read: `1` files, `+6/-5`; areas: model wrapper, MoE/router, scheduler/runtime; keywords: config, expert, moe, router.
- Code diff details:
  - `vllm/model_executor/models/ernie45_moe.py` modified +6/-5 (11 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, forward
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/ernie45_moe.py`; keywords observed in patches: config, expert, moe, router. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/ernie45_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22514 - [Model] Add Ernie4.5 VL Model Support

- Link: https://github.com/vllm-project/vllm/pull/22514
- Status/date: `merged`, created 2025-08-08, merged 2025-08-27; author `CSWYF3634076`.
- Diff scope read: `11` files, `+2463/-0`; areas: model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: moe, kv, vision, config, attention, cache, cuda, processor, quant, spec.
- Code diff details:
  - `vllm/model_executor/models/ernie45_vl.py` added +1504/-0 (1504 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: rotate_half, apply_rotary_emb_torch, apply_rotary_pos_emb_vision, all_gather_interleave
  - `vllm/model_executor/models/ernie45_vl_moe.py` added +723/-0 (723 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Ernie4_5_VLMoeMLP, Ernie4_5_VLMoeAttention, __init__, forward
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +123/-0 (123 lines); hunks: def get_input_positions_tensor(; def _glm4v_get_input_positions_tensor(; symbols: get_input_positions_tensor, _glm4v_get_input_positions_tensor, _ernie_get_input_positions_tensor, _vl_get_input_positions_tensor
  - `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` added +72/-0 (72 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Ernie4_5_VLRotaryEmbedding, forward
  - `examples/offline_inference/vision_language.py` modified +32/-0 (32 lines); hunks: def run_deepseek_vl2(questions: list[str], modality: str) -> ModelRequestData:; def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:; symbols: run_deepseek_vl2, run_ernie45_vl, run_florence2, run_tarsier2
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`; keywords observed in patches: moe, kv, vision, config, attention, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #24074 - [BugFix][Model] Fix Ernie4.5-VL hanging on long inputs

- Link: https://github.com/vllm-project/vllm/pull/24074
- Status/date: `merged`, created 2025-09-02, merged 2025-09-09; author `CSWYF3634076`.
- Diff scope read: `2` files, `+18/-7`; areas: model wrapper, MoE/router, scheduler/runtime; keywords: vision, expert, moe, processor, router.
- Code diff details:
  - `vllm/model_executor/models/ernie45_vl.py` modified +10/-4 (14 lines); hunks: logger = init_logger(__name__); def get_image_processor(self, **kwargs: object):; symbols: get_image_processor, get_supported_mm_limits, get_mm_max_tokens_per_item, _get_vision_info
  - `vllm/model_executor/models/ernie45_vl_moe.py` modified +8/-3 (11 lines); hunks: def forward(; def forward(; symbols: forward, forward
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`; keywords observed in patches: vision, expert, moe, processor, router. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #31274 - [Model][Ernie4.5-VL] Support video metadata for timestamp rendering

- Link: https://github.com/vllm-project/vllm/pull/31274
- Status/date: `merged`, created 2025-12-24, merged 2025-12-25; author `Tiiiktak`.
- Diff scope read: `2` files, `+82/-5`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks; keywords: attention, config, moe, processor, spec, test.
- Code diff details:
  - `vllm/model_executor/models/ernie45_vl.py` modified +80/-4 (84 lines); hunks: # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.; MMEncoderAttention,; symbols: get_max_video_tokens, Ernie4_5VLMultiModalProcessor, _get_data_parser, _pixel_values_norm
  - `tests/models/multimodal/processing/test_common.py` modified +2/-1 (3 lines); hunks: def create_metadata(frames: np.ndarray):; symbols: create_metadata
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/ernie45_vl.py`, `tests/models/multimodal/processing/test_common.py`; keywords observed in patches: attention, config, moe, processor, spec, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/ernie45_vl.py`, `tests/models/multimodal/processing/test_common.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 5; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
