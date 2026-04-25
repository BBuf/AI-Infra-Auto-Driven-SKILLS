# vLLM InternVL3.5 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: InternVL3.5 multimodal processor, video support, ViT DP / torch.compile, and backend compatibility.

## Landed PRs

### PR #6514 - Initialize support for InternVL2 series models

- Link: https://github.com/vllm-project/vllm/pull/6514
- Why it mattered: Historical base for current InternVL runtime code.
- Runtime path: vllm/vllm/model_executor/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #18499 - Initialize video input support for InternVL models

- Link: https://github.com/vllm-project/vllm/pull/18499
- Why it mattered: Added video processing to the family.
- Runtime path: vllm/vllm/model_executor/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #23658 - Enable video support for InternVL3.5 models

- Link: https://github.com/vllm-project/vllm/pull/23658
- Why it mattered: Carried video support into the 3.5 checkpoints.
- Runtime path: vllm/vllm/model_executor/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #23742 - Enable native HF format InternVL support

- Link: https://github.com/vllm-project/vllm/pull/23742
- Why it mattered: Removed reliance on ad hoc checkpoint rewrites.
- Runtime path: vllm/vllm/model_executor/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #38049 - Add torch.compile support for InternVL vision encoder

- Link: https://github.com/vllm-project/vllm/pull/38049
- Why it mattered: Modernized the encoder execution path.
- Runtime path: vllm/vllm/model_executor/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM InternVL3.5 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2024-07-17 | [#6514](https://github.com/vllm-project/vllm/pull/6514) | merged | [Model] Initialize support for InternVL2 series models | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/test_internvl.py` |
| 2025-05-21 | [#18499](https://github.com/vllm-project/vllm/pull/18499) | merged | [VLM] Initialize video input support for InternVL models | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `examples/offline_inference/vision_language.py` |
| 2025-08-26 | [#23658](https://github.com/vllm-project/vllm/pull/23658) | merged | [Model] Enable video support for InternVL3.5 models | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/registry.py` |
| 2025-08-27 | [#23742](https://github.com/vllm-project/vllm/pull/23742) | merged | [Model] Enable native HF format InternVL support | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`, `docs/models/supported_models.md` |
| 2026-03-25 | [#38049](https://github.com/vllm-project/vllm/pull/38049) | merged | [Model] Add torch.compile support for InternVL vision encoder | model wrapper, scheduler/runtime, docs/config | `vllm/model_executor/models/intern_vit.py`, `vllm/config/utils.py` |

## Diff Cards

### PR #6514 - [Model] Initialize support for InternVL2 series models

- Link: https://github.com/vllm-project/vllm/pull/6514
- Status/date: `merged`, created 2024-07-17, merged 2024-07-29; author `Isotr0py`.
- Diff scope read: `14` files, `+1042/-6`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: config, attention, vision, cache, kv, spec, processor, quant, cuda, lora.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` added +471/-0 (471 lines); hunks: +# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py; symbols: InternVLImagePixelInputs, build_transform, find_closest_aspect_ratio, calculate_num_blocks
  - `vllm/model_executor/models/intern_vit.py` added +270/-0 (270 lines); hunks: +# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py; symbols: InternVisionEmbeddings, __init__, _get_pos_embed, forward
  - `tests/models/test_internvl.py` added +201/-0 (201 lines); hunks: +import types; symbols: InternVLProcessor:, __init__, __call__, generate
  - `vllm/transformers_utils/configs/internvl.py` added +51/-0 (51 lines); hunks: +# Adapted from; symbols: InternVLChatConfig, __init__
  - `examples/offline_inference_vision_language.py` modified +15/-0 (15 lines); hunks: def run_minicpmv(question):; def run_blip2(question):; symbols: run_minicpmv, run_internvl, for, run_blip2
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/test_internvl.py`; keywords observed in patches: config, attention, vision, cache, kv, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/test_internvl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18499 - [VLM] Initialize video input support for InternVL models

- Link: https://github.com/vllm-project/vllm/pull/18499
- Status/date: `merged`, created 2025-05-21, merged 2025-05-25; author `Isotr0py`.
- Diff scope read: `10` files, `+596/-62`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: vision, processor, test, config, spec, attention, cache, doc, fp8.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +485/-26 (511 lines); hunks: # --------------------------------------------------------; class InternVLImageEmbeddingInputs(TypedDict):; symbols: InternVLImageEmbeddingInputs, InternVLVideoPixelInputs, InternVLVideoEmbeddingInputs, build_transform
  - `tests/models/multimodal/generation/vlm_utils/model_utils.py` modified +66/-20 (86 lines); hunks: from pathlib import PosixPath; def __init__(self, hf_runner: HfRunner):; symbols: __init__, __call__, __call__
  - `examples/offline_inference/vision_language.py` modified +11/-4 (15 lines); hunks: def run_smolvlm(questions: list[str], modality: str) -> ModelRequestData:; def run_internvl(questions: list[str], modality: str) -> ModelRequestData:; symbols: run_smolvlm, run_internvl, run_internvl
  - `vllm/model_executor/models/nvlm_d.py` modified +8/-5 (13 lines); hunks: PromptUpdateDetails); def get_hf_processor(; symbols: get_hf_processor, NVLMDummyInputsBuilder, NVLMDummyInputsBuilder, get_dummy_text
  - `tests/models/multimodal/generation/test_common.py` modified +11/-0 (11 lines); hunks: use_tokenizer_eos=True,
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `examples/offline_inference/vision_language.py`; keywords observed in patches: vision, processor, test, config, spec, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `examples/offline_inference/vision_language.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23658 - [Model] Enable video support for InternVL3.5 models

- Link: https://github.com/vllm-project/vllm/pull/23658
- Status/date: `merged`, created 2025-08-26, merged 2025-08-26; author `Isotr0py`.
- Diff scope read: `5` files, `+22/-7`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: test, fp8, moe, config, doc, spec.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +7/-3 (10 lines); hunks: def get_supported_mm_limits(self):; symbols: get_supported_mm_limits, get_video_token, get_num_frames_with_most_features
  - `tests/models/multimodal/processing/test_tensor_schema.py` modified +6/-1 (7 lines); hunks: "MiniCPMV",
  - `tests/models/registry.py` modified +4/-1 (5 lines); hunks: def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +2/-2 (4 lines); hunks: These models primarily accept the `LLM.generate` (./generative_models.md#llmgen; Some models are supported only via the [Transformers backend](#transformers).
  - `tests/models/multimodal/processing/test_common.py` modified +3/-0 (3 lines); hunks: def _test_processing_correctness_one(; symbols: _test_processing_correctness_one
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/registry.py`; keywords observed in patches: test, fp8, moe, config, doc, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/registry.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23742 - [Model] Enable native HF format InternVL support

- Link: https://github.com/vllm-project/vllm/pull/23742
- Status/date: `merged`, created 2025-08-27, merged 2025-08-27; author `Isotr0py`.
- Diff scope read: `4` files, `+18/-16`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: test, doc, fp8, moe.
- Code diff details:
  - `tests/models/multimodal/generation/test_common.py` modified +14/-15 (29 lines); hunks: },; use_tokenizer_eos=True,
  - `tests/models/registry.py` modified +2/-1 (3 lines); hunks: def check_available_online(; def check_available_online(; symbols: check_available_online, check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: These models primarily accept the [`LLM.generate`](./generative_models.md#llmgen
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: "H2OVLChatModel": ("h2ovl", "H2OVLChatModel"),
- Optimization/support interpretation: The concrete diff surface is `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`, `docs/models/supported_models.md`; keywords observed in patches: test, doc, fp8, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`, `docs/models/supported_models.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #38049 - [Model] Add torch.compile support for InternVL vision encoder

- Link: https://github.com/vllm-project/vllm/pull/38049
- Status/date: `merged`, created 2026-03-25, merged 2026-03-26; author `tianrengao`.
- Diff scope read: `2` files, `+20/-3`; areas: model wrapper, scheduler/runtime, docs/config; keywords: config, quant, vision.
- Code diff details:
  - `vllm/model_executor/models/intern_vit.py` modified +11/-2 (13 lines); hunks: import torch.nn.functional as F; def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward, InternVisionEncoderLayer, __init__, __init__
  - `vllm/config/utils.py` modified +9/-1 (10 lines); hunks: def normalize_value(x):; symbols: normalize_value
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/intern_vit.py`, `vllm/config/utils.py`; keywords observed in patches: config, quant, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/intern_vit.py`, `vllm/config/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
