# SGLang InternVL3.5 Support and PR History

This note tracks the SGLang runtime, key PRs, and cookbook-facing touchpoints for InternVL3.5.

- Status: 当前 mainline 已支持

## Key Conclusions

- InternVL3.5 is mostly a processor / encoder / video problem in SGLang.
- Video handling, data-parallel vision execution, and backend compatibility dominate the risk surface.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/internvl.py`

## Landed PRs

- [#5350](https://github.com/sgl-project/sglang/pull/5350) `Support InternVL3`: Initial InternVL family support that later carried 3.5.
- [#13640](https://github.com/sgl-project/sglang/pull/13640) `Support Piecewise CUDA Graph for InternVL`: Added graph capture support on the encoder path.
- [#13925](https://github.com/sgl-project/sglang/pull/13925) `Support InternVL Vision Encoder Data Parallelism`: Opened the multi-GPU ViT path.
- [#15942](https://github.com/sgl-project/sglang/pull/15942) `Support Video for InternVL3_5`: Extended support to 3.5 video use cases.
- [#19127](https://github.com/sgl-project/sglang/pull/19127) `Support processor and embedding inputs for InternVL`: Hardened processor / embed input interoperability.

## Matching Skill

- `skills/model-optimization/sglang/sglang-internvl35-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-internvl35-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `InternVL3.5` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-04-13 | [#5350](https://github.com/sgl-project/sglang/pull/5350) | merged | Support InternVL3 | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py` |
| 2025-11-20 | [#13640](https://github.com/sgl-project/sglang/pull/13640) | merged | [VLM] Support Piecewise CUDA Graph for InternVL | model wrapper, kernel, scheduler/runtime, tests/benchmarks | `test/srt/test_piecewise_cuda_graph.py`, `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/internvl.py` |
| 2025-11-25 | [#13925](https://github.com/sgl-project/sglang/pull/13925) | merged | [VLM] Support InternVL Vision Encoder Data Parallelism | model wrapper, multimodal/processor, tests/benchmarks | `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `test/nightly/test_encoder_dp.py` |
| 2025-12-27 | [#15942](https://github.com/sgl-project/sglang/pull/15942) | merged | [VLM] Support Video for InternVL3_5 | model wrapper, multimodal/processor | `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py` |
| 2026-02-21 | [#19127](https://github.com/sgl-project/sglang/pull/19127) | merged | [vlm][internVL] Support processor and embedding inputs for InternVL | model wrapper, multimodal/processor, tests/benchmarks | `test/registered/vlm/test_vlm_input_format.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/base_processor.py` |

### File-level PR diff reading notes

### PR #5350 - Support InternVL3

- Link: https://github.com/sgl-project/sglang/pull/5350
- Status/date: `merged`, created 2025-04-13, merged 2025-05-02; author `xiaomin-D`.
- Diff scope read: `12` files, `+1728/-9`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: vision, config, kv, processor, attention, spec, cuda, doc, flash, cache.
- Code diff details:
  - `python/sglang/srt/configs/internvl.py` added +696/-0 (696 lines); hunks: +import copy; symbols: InternLM2Config, to, __init__, _rope_scaling_validation
  - `python/sglang/srt/models/internvl.py` added +670/-0 (670 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: FlashAttention, __init__, forward, InternAttention
  - `python/sglang/srt/managers/multimodal_processors/internvl.py` added +232/-0 (232 lines); hunks: +# Adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py; symbols: InternVLImageProcessor, __init__, build_transform, resize_image
  - `python/sglang/lang/chat_template.py` modified +44/-0 (44 lines); hunks: def get_chat_template_by_model_path(model_path):; def get_chat_template_by_model_path(model_path):; symbols: get_chat_template_by_model_path, get_chat_template_by_model_path, match_gemma3_instruct, match_internvl_chat
  - `python/sglang/srt/conversation.py` modified +29/-2 (31 lines); hunks: class SeparatorStyle(IntEnum):; def get_prompt(self) -> str:; symbols: SeparatorStyle, get_prompt, generate_chat_conv, generate_chat_conv
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py`; keywords observed in patches: vision, config, kv, processor, attention, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13640 - [VLM] Support Piecewise CUDA Graph for InternVL

- Link: https://github.com/sgl-project/sglang/pull/13640
- Status/date: `merged`, created 2025-11-20, merged 2025-11-21; author `yuan-luo`.
- Diff scope read: `5` files, `+103/-13`; areas: model wrapper, kernel, scheduler/runtime, tests/benchmarks; keywords: cache, cuda, attention, spec, test, config, lora, moe, processor, quant.
- Code diff details:
  - `test/srt/test_piecewise_cuda_graph.py` modified +41/-0 (41 lines); hunks: def test_gsm8k_accuracy(self):; symbols: test_gsm8k_accuracy, TestPiecewiseCudaGraphInternVL25, setUpClass, tearDownClass
  - `python/sglang/srt/managers/mm_utils.py` modified +37/-2 (39 lines); hunks: def should_use_external_mm_preprocess(multimodal_model: nn.Module) -> bool:; symbols: should_use_external_mm_preprocess, resolve_external_mm_data_embedding_funcs, external_mm_preprocess_routine
  - `python/sglang/srt/models/internvl.py` modified +21/-10 (31 lines); hunks: from sglang.srt.layers.attention.vision import SingletonCache, VisionAttention; def __init__(; symbols: __init__, pixel_shuffle, forward, pad_input_ids
  - `python/sglang/srt/model_executor/model_runner.py` modified +3/-0 (3 lines); hunks: from sglang.srt.lora.lora_registry import LoRARef; def forward_extend(; symbols: forward_extend
  - `test/srt/run_suite.py` modified +1/-1 (2 lines); hunks: TestFile("test_original_logprobs.py", 41),
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_piecewise_cuda_graph.py`, `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/internvl.py`; keywords observed in patches: cache, cuda, attention, spec, test, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_piecewise_cuda_graph.py`, `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/internvl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13925 - [VLM] Support InternVL Vision Encoder Data Parallelism

- Link: https://github.com/sgl-project/sglang/pull/13925
- Status/date: `merged`, created 2025-11-25, merged 2025-11-26; author `yuan-luo`.
- Diff scope read: `3` files, `+118/-25`; areas: model wrapper, multimodal/processor, tests/benchmarks; keywords: vision, attention, cache, config, moe, quant, test, triton.
- Code diff details:
  - `python/sglang/srt/models/internvl.py` modified +83/-25 (108 lines); hunks: import torch.nn.functional as F; from sglang.srt.models.qwen2 import Qwen2ForCausalLM; symbols: __init__, __init__, forward, InternMLP
  - `python/sglang/srt/multimodal/mm_utils.py` modified +34/-0 (34 lines); hunks: def get_dp_encoder_lb_assignment(; symbols: get_dp_encoder_lb_assignment, run_dp_sharded_vision_model, run_dp_sharded_mrope_vision_model
  - `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: MODELS = [
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `test/nightly/test_encoder_dp.py`; keywords observed in patches: vision, attention, cache, config, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `test/nightly/test_encoder_dp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15942 - [VLM] Support Video for InternVL3_5

- Link: https://github.com/sgl-project/sglang/pull/15942
- Status/date: `merged`, created 2025-12-27, merged 2025-12-30; author `yuan-luo`.
- Diff scope read: `2` files, `+426/-118`; areas: model wrapper, multimodal/processor; keywords: cache, config, cuda, moe, processor, spec, vision.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +418/-118 (536 lines); hunks: # Adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py; MultimodalSpecialTokens,; symbols: InternVLImageProcessor, InternVLProcessor, _get_normalize_tensors, __init__
  - `python/sglang/srt/models/internvl.py` modified +8/-0 (8 lines); hunks: def __init__(; def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: __init__, get_image_feature, get_video_feature, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`; keywords observed in patches: cache, config, cuda, moe, processor, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19127 - [vlm][internVL] Support processor and embedding inputs for InternVL

- Link: https://github.com/sgl-project/sglang/pull/19127
- Status/date: `merged`, created 2026-02-21, merged 2026-02-27; author `jiangyukunok`.
- Diff scope read: `4` files, `+282/-7`; areas: model wrapper, multimodal/processor, tests/benchmarks; keywords: processor, spec, vision, config, cuda, test.
- Code diff details:
  - `test/registered/vlm/test_vlm_input_format.py` modified +153/-0 (153 lines); hunks: def _processor_output_image_data(self, processor_output):; symbols: _processor_output_image_data, TestInternVLUnderstandsImage, setUpClass, _init_visual
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +109/-1 (110 lines); hunks: from decord import VideoReader, cpu, gpu; def _resolve_video_num_frames(; symbols: _resolve_video_num_frames, _has_special_format, _process_special_format, process_and_combine_mm_data
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +13/-6 (19 lines); hunks: def __init__(; def build_input_ids(self, prompt, img_grid_thw):; symbols: __init__, build_input_ids, load_mm_data, fast_load_mm_data
  - `python/sglang/srt/models/internvl.py` modified +7/-0 (7 lines); hunks: def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: get_image_feature, get_video_feature
- Optimization/support interpretation: The concrete diff surface is `test/registered/vlm/test_vlm_input_format.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/base_processor.py`; keywords observed in patches: processor, spec, vision, config, cuda, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/vlm/test_vlm_input_format.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/base_processor.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 5; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
