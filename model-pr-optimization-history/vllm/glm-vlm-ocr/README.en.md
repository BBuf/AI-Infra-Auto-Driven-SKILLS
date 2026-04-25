# vLLM GLM VLM / OCR Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for GLM VLM / OCR.

- Status: supported on current mainline

## Key Conclusions

- GLM visual/OCR support in vLLM spans classic GLM4V, newer GLM4.1V, and GLM-OCR-specific processor paths.
- The main failures are processor-schema drift, MRoPE/video position handling, and OCR-specific weight or patch-merger mismatches.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/glm4v.py`
- `vllm/vllm/model_executor/models/glm4_1v.py`
- `vllm/vllm/model_executor/models/glm_ocr.py`

## Landed PRs

- [#9242](https://github.com/vllm-project/vllm/pull/9242) `Add GLM-4v support`: Landed the original GLM4V multimodal model path.
- [#19331](https://github.com/vllm-project/vllm/pull/19331) `Add GLM4.1V model`: Extended the family to the newer GLM4.1V checkpoint layout and vision stack.
- [#27860](https://github.com/vllm-project/vllm/pull/27860) `Fix broken MRoPE for GLM-4.1V/GLM-4.5V`: Closed a positional-embedding bug with large practical accuracy impact on vision inputs.
- [#33005](https://github.com/vllm-project/vllm/pull/33005) `GLM-OCR with MTP Support`: Added OCR-specific draft / MTP support rather than text-only OCR loading.
- [#33350](https://github.com/vllm-project/vllm/pull/33350) `Fix broken GLM-OCR initialization`: Fixed startup failures in the GLM-OCR path after the first bring-up.
- [#37962](https://github.com/vllm-project/vllm/pull/37962) `GLM OCR Patch Merger context_dim`: Updated the patch-merger contract for newer OCR checkpoints.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-glm-vlm-ocr-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-glm-vlm-ocr-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `GLM VLM / OCR` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2024-10-10 | [#9242](https://github.com/vllm-project/vllm/pull/9242) | merged | [Model] Add GLM-4v support and meet vllm==0.6.2 | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/chatglm.py`, `vllm/model_executor/models/glm4_vision_encoder.py`, `tests/models/decoder_only/vision_language/test_glm4.py` |
| 2025-06-08 | [#19331](https://github.com/vllm-project/vllm/pull/19331) | merged | Add GLM-4.1V model | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/layers/rotary_embedding.py`, `vllm/multimodal/parse.py` |
| 2025-10-31 | [#27860](https://github.com/vllm-project/vllm/pull/27860) | merged | [Bugfix] Fix broken MRoPE for GLM-4.1V/GLM-4.5V | model wrapper, scheduler/runtime | `vllm/model_executor/models/glm4_1v.py` |
| 2026-01-24 | [#33005](https://github.com/vllm-project/vllm/pull/33005) | merged | [GLM-OCR] GLM-OCR with MTP Support | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm_ocr_mtp.py`, `vllm/model_executor/models/glm4.py` |
| 2026-01-29 | [#33350](https://github.com/vllm-project/vllm/pull/33350) | merged | [Bugfix] Fix broken GLM-OCR initialization | model wrapper, scheduler/runtime | `vllm/model_executor/models/glm_ocr.py` |
| 2026-03-24 | [#37962](https://github.com/vllm-project/vllm/pull/37962) | merged | [bug-fix] GLM OCR Patch Merger context_dim | model wrapper, scheduler/runtime | `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm4_1v.py` |

### File-level PR diff reading notes

### PR #9242 - [Model] Add GLM-4v support and meet vllm==0.6.2

- Link: https://github.com/vllm-project/vllm/pull/9242
- Status/date: `merged`, created 2024-10-10, merged 2024-10-11; author `sixsixcoder`.
- Diff scope read: `7` files, `+776/-72`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: vision, attention, config, kv, processor, quant, cache, doc, lora, moe.
- Code diff details:
  - `vllm/model_executor/models/chatglm.py` modified +298/-52 (350 lines); hunks: # coding=utf-8; class GLMMLP(nn.Module):; symbols: calculate_image_placeholder, mm_input_mapper_for_glmv, merge_glm_vision_embeddings, GLMImagePixelInputs
  - `vllm/model_executor/models/glm4_vision_encoder.py` added +298/-0 (298 lines); hunks: +# coding=utf-8; symbols: PatchEmbedding, __init__, forward, Attention
  - `tests/models/decoder_only/vision_language/test_glm4.py` added +133/-0 (133 lines); hunks: +from typing import List, Optional, Tuple, Type; symbols: run_test, processor, test_models
  - `vllm/transformers_utils/tokenizer.py` modified +21/-18 (39 lines); hunks: def __len__(self):; def get_tokenizer(; symbols: __len__, patch_padding_side, _pad, get_tokenizer
  - `examples/offline_inference_vision_language.py` modified +16/-0 (16 lines); hunks: def run_mllama(question: str, modality: str):; def run_mllama(question: str, modality: str):; symbols: run_mllama, run_glm4v, run_mllama
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/chatglm.py`, `vllm/model_executor/models/glm4_vision_encoder.py`, `tests/models/decoder_only/vision_language/test_glm4.py`; keywords observed in patches: vision, attention, config, kv, processor, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/chatglm.py`, `vllm/model_executor/models/glm4_vision_encoder.py`, `tests/models/decoder_only/vision_language/test_glm4.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19331 - Add GLM-4.1V model

- Link: https://github.com/vllm-project/vllm/pull/19331
- Status/date: `merged`, created 2025-06-08, merged 2025-07-01; author `zRzRzRzRzRzRzR`.
- Diff scope read: `17` files, `+1946/-16`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: processor, config, test, vision, cache, spec, attention, flash, kv, lora.
- Code diff details:
  - `vllm/model_executor/models/glm4_1v.py` added +1589/-0 (1589 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs, Glm4vVideoEmbeddingInputs
  - `vllm/model_executor/layers/rotary_embedding.py` modified +119/-0 (119 lines); hunks: # See the License for the specific language governing permissions and; def get_input_positions_tensor(; symbols: get_input_positions_tensor, get_input_positions_tensor, _glm4v_get_input_positions_tensor, _vl_get_input_positions_tensor
  - `vllm/multimodal/parse.py` modified +40/-2 (42 lines); hunks: def __init__(self, data: Union[torch.Tensor, list[torch.Tensor]]) -> None:; def __init__(; symbols: __init__, VideoProcessorItems, __init__, __init__
  - `examples/offline_inference/vision_language.py` modified +39/-1 (40 lines); hunks: def run_glm4v(questions: list[str], modality: str) -> ModelRequestData:; def run_skyworkr1v(questions: list[str], modality: str) -> ModelRequestData:; symbols: run_glm4v, run_glm4_1v, run_h2ovl, run_skyworkr1v
  - `tests/models/multimodal/generation/test_common.py` modified +28/-0 (28 lines); hunks: num_logprobs=10,
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/layers/rotary_embedding.py`, `vllm/multimodal/parse.py`; keywords observed in patches: processor, config, test, vision, cache, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/layers/rotary_embedding.py`, `vllm/multimodal/parse.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #27860 - [Bugfix] Fix broken MRoPE for GLM-4.1V/GLM-4.5V

- Link: https://github.com/vllm-project/vllm/pull/27860
- Status/date: `merged`, created 2025-10-31, merged 2025-10-31; author `Isotr0py`.
- Diff scope read: `1` files, `+147/-2`; areas: model wrapper, scheduler/runtime; keywords: config, lora, processor, vision.
- Code diff details:
  - `vllm/model_executor/models/glm4_1v.py` modified +147/-2 (149 lines); hunks: # limitations under the License.; import torch.nn as nn; symbols: get_video_replacement_glm4v, Glm4vForConditionalGeneration, get_multimodal_embeddings, get_mrope_input_positions
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/glm4_1v.py`; keywords observed in patches: config, lora, processor, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/glm4_1v.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #33005 - [GLM-OCR] GLM-OCR with MTP Support

- Link: https://github.com/vllm-project/vllm/pull/33005
- Status/date: `merged`, created 2026-01-24, merged 2026-01-26; author `zRzRzRzRzRzRzR`.
- Diff scope read: `14` files, `+873/-8`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: spec, config, fp8, processor, test, kv, moe, quant, attention, cache.
- Code diff details:
  - `vllm/model_executor/models/glm_ocr.py` added +389/-0 (389 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: GlmOcrVisionMLP, GlmOcrVisionAttention, __init__, split_qkv
  - `vllm/model_executor/models/glm_ocr_mtp.py` added +285/-0 (285 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: GlmOcrMultiTokenPredictorLayer, __init__, forward, GlmOcrMultiTokenPredictor
  - `vllm/model_executor/models/glm4.py` modified +99/-2 (101 lines); hunks: from vllm.model_executor.layers.quantization import QuantizationConfig; def __init__(; symbols: Glm4Attention, __init__, __init__, load_weights
  - `examples/offline_inference/vision_language.py` modified +38/-0 (38 lines); hunks: def run_glm4_5v_fp8(questions: list[str], modality: str) -> ModelRequestData:; def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:; symbols: run_glm4_5v_fp8, run_glm_ocr, run_h2ovl, run_tarsier2
  - `tests/models/multimodal/generation/test_common.py` modified +14/-0 (14 lines); hunks: ],
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm_ocr_mtp.py`, `vllm/model_executor/models/glm4.py`; keywords observed in patches: spec, config, fp8, processor, test, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm_ocr_mtp.py`, `vllm/model_executor/models/glm4.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #33350 - [Bugfix] Fix broken GLM-OCR initialization

- Link: https://github.com/vllm-project/vllm/pull/33350
- Status/date: `merged`, created 2026-01-29, merged 2026-01-29; author `Isotr0py`.
- Diff scope read: `1` files, `+1/-1`; areas: model wrapper, scheduler/runtime; keywords: config, quant, vision.
- Code diff details:
  - `vllm/model_executor/models/glm_ocr.py` modified +1/-1 (2 lines); hunks: class GlmOcrPatchMerger(Glm4vPatchMerger):; symbols: GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/glm_ocr.py`; keywords observed in patches: config, quant, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/glm_ocr.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #37962 - [bug-fix] GLM OCR Patch Merger context_dim

- Link: https://github.com/vllm-project/vllm/pull/37962
- Status/date: `merged`, created 2026-03-24, merged 2026-03-26; author `JaredforReal`.
- Diff scope read: `2` files, `+14/-4`; areas: model wrapper, scheduler/runtime; keywords: config, quant, vision, processor.
- Code diff details:
  - `vllm/model_executor/models/glm_ocr.py` modified +8/-3 (11 lines); hunks: from einops import rearrange; class GlmOcrPatchMerger(Glm4vPatchMerger):; symbols: GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__, __init__
  - `vllm/model_executor/models/glm4_1v.py` modified +6/-1 (7 lines); hunks: import torch.nn.functional as F; def forward(; symbols: forward, Glm4vVisionTransformer, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm4_1v.py`; keywords observed in patches: config, quant, vision, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm4_1v.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 6; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
