# vllm DeepSeek OCR 2 Model PR Optimization History

## 2026-06-05 PR Backfill Audit

Rechecked vllm upstream `origin/main@c66b19800` on 2026-06-05; 1 additional PR-numbered merge(s) touched the tracked implementation files after the previous freshness cutoff (2026-05-19). These are not yet reflected in the timeline / diff-audit cards below and should be folded in on the next full regeneration.

| Merged | PR | Title | Tracked files touched |
| --- | --- | --- | --- |
| 2026-06-04 | [#41759](https://github.com/vllm-project/vllm/pull/41759) | [MM][Perf][CG] Support ViT full CUDA graph for InternVL | `vision_language_offline.py` |


## 2026-05-19 Coverage Addition

Generated from vllm upstream `origin/main@ef54a4d604`, `git log --name-only -- <model-files>` over model-related paths, and the GitHub Pull Request files API. This page fills the missing `DeepSeek OCR 2` history entry found from sgl-cookbook coverage.

## Implementation File Coverage

| File | PRs traced by git |
| --- | --- |
| `vllm/model_executor/models/deepseek_ocr.py` | [#35182](https://github.com/vllm-project/vllm/pull/35182), [#37289](https://github.com/vllm-project/vllm/pull/37289), [#36670](https://github.com/vllm-project/vllm/pull/36670), [#35025](https://github.com/vllm-project/vllm/pull/35025), [#34085](https://github.com/vllm-project/vllm/pull/34085), [#34330](https://github.com/vllm-project/vllm/pull/34330), [#33909](https://github.com/vllm-project/vllm/pull/33909), [#33063](https://github.com/vllm-project/vllm/pull/33063), [#31972](https://github.com/vllm-project/vllm/pull/31972), [#32632](https://github.com/vllm-project/vllm/pull/32632), [#32327](https://github.com/vllm-project/vllm/pull/32327), [#32016](https://github.com/vllm-project/vllm/pull/32016), ... (25 total) |
| `vllm/model_executor/models/deepseek_ocr2.py` | [#35182](https://github.com/vllm-project/vllm/pull/35182), [#37289](https://github.com/vllm-project/vllm/pull/37289), [#35025](https://github.com/vllm-project/vllm/pull/35025), [#34330](https://github.com/vllm-project/vllm/pull/34330), [#33909](https://github.com/vllm-project/vllm/pull/33909), [#33165](https://github.com/vllm-project/vllm/pull/33165) |
| `vllm/transformers_utils/processors/deepseek_ocr.py` | [#36024](https://github.com/vllm-project/vllm/pull/36024), [#33909](https://github.com/vllm-project/vllm/pull/33909), [#27361](https://github.com/vllm-project/vllm/pull/27361), [#27247](https://github.com/vllm-project/vllm/pull/27247) |
| `vllm/transformers_utils/chat_templates/template_deepseek_ocr.jinja` | [#27247](https://github.com/vllm-project/vllm/pull/27247) |
| `tests/models/multimodal/processing/test_deepseek_ocr.py` | [#36670](https://github.com/vllm-project/vllm/pull/36670) |
| `examples/generate/multimodal/vision_language_offline.py` | [#42224](https://github.com/vllm-project/vllm/pull/42224), [#41736](https://github.com/vllm-project/vllm/pull/41736), [#42151](https://github.com/vllm-project/vllm/pull/42151), [#40830](https://github.com/vllm-project/vllm/pull/40830), [#36464](https://github.com/vllm-project/vllm/pull/36464) |

## PR Coverage Summary

- git-traced PR count: 32
- keyword/supplemental PR count: 0
- total PR count in this document: 32
- file trace command: `git log --name-only -- <model-files>`
- diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | Status | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-10-22 | [#27247](https://github.com/vllm-project/vllm/pull/27247) | merged | [Model] Upstream Deepseek-OCR model | `docs/models/supported_models.md`, `examples/offline_inference/vision_language.py`, `tests/models/registry.py` |
| 2025-10-23 | [#27361](https://github.com/vllm-project/vllm/pull/27361) | merged | [Bugfix] Fix deepseek-ocr multi-image inference and add `merge_by_field_config=True` with tensor schema support | `examples/offline_inference/vision_language_multi_image.py`, `tests/models/multimodal/processing/test_common.py`, `vllm/model_executor/models/deepseek_ocr.py` |
| 2025-11-05 | [#27560](https://github.com/vllm-project/vllm/pull/27560) | merged | [Bugfix] Validate custom logits processor xargs for online serving | `docs/design/logits_processors.md`, `docs/features/custom_arguments.md`, `docs/features/custom_logitsprocs.md` |
| 2025-11-08 | [#28101](https://github.com/vllm-project/vllm/pull/28101) | merged | [Model] Consolidate Deepseek-MoE implementation with DeepSeek-v2 | `tests/models/registry.py`, `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/models/deepseek_ocr.py` |
| 2025-11-13 | [#27583](https://github.com/vllm-project/vllm/pull/27583) | merged | Rename clashing method names for vLLM model protocol | `docs/contributing/model/basic.md`, `docs/contributing/model/multimodal.md`, `vllm/model_executor/models/apertus.py` |
| 2025-11-13 | [#28617](https://github.com/vllm-project/vllm/pull/28617) | merged | [BugFix] DeepSeek-OCR: apply NoRepeatNGramLogitsProcessor to greedy path | `vllm/model_executor/models/deepseek_ocr.py` |
| 2025-12-02 | [#29793](https://github.com/vllm-project/vllm/pull/29793) | merged | [Chore] Move tokenizer initialization methods | `benchmarks/benchmark_prefix_caching.py`, `benchmarks/benchmark_serving_structured_output.py`, `tests/compile/test_dynamic_shapes_compilation.py` |
| 2025-12-04 | [#30035](https://github.com/vllm-project/vllm/pull/30035) | merged | [Chore] Deprecate `merge_by_field_config` arg | `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_glm4_1v.py`, `tests/models/multimodal/processing/test_tensor_schema.py` |
| 2025-12-06 | [#30170](https://github.com/vllm-project/vllm/pull/30170) | merged | [Chore] Deprecate `SupportsMultiModal.merge_by_field_config` | `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/aya_vision.py`, `vllm/model_executor/models/blip2.py` |
| 2025-12-07 | [#30145](https://github.com/vllm-project/vllm/pull/30145) | merged | [Renderer] Separate out `RendererConfig` from `ModelConfig` | `docs/contributing/model/transcription.md`, `tests/compile/distributed/test_sequence_parallelism.py`, `tests/compile/test_functionalization.py` |
| 2025-12-07 | [#30199](https://github.com/vllm-project/vllm/pull/30199) | merged | Revert "[Renderer] Separate out `RendererConfig` from `ModelConfig` (#30145)" | `docs/contributing/model/transcription.md`, `tests/compile/distributed/test_sequence_parallelism.py`, `tests/compile/test_functionalization.py` |
| 2026-01-02 | [#31569](https://github.com/vllm-project/vllm/pull/31569) | merged | feat: support LoRA for DeepSeek-OCR(Language Model part) | `docs/models/supported_models.md`, `vllm/model_executor/models/deepseek_ocr.py` |
| 2026-01-08 | [#31947](https://github.com/vllm-project/vllm/pull/31947) | merged | [Model] Standardize common vision encoders | `vllm/model_executor/models/clip.py`, `vllm/model_executor/models/deepencoder.py`, `vllm/model_executor/models/deepseek_ocr.py` |
| 2026-01-09 | [#32016](https://github.com/vllm-project/vllm/pull/32016) | merged | [Model] Remove redundant None check in DeepSeekOCR image input processing | `vllm/model_executor/models/deepseek_ocr.py` |
| 2026-01-14 | [#32327](https://github.com/vllm-project/vllm/pull/32327) | merged | [1/N] Reorganize multimodal processing code | `docs/api/README.md`, `docs/contributing/model/multimodal.md`, `docs/design/mm_processing.md` |
| 2026-01-20 | [#32632](https://github.com/vllm-project/vllm/pull/32632) | merged | [1/N] Initialize MM components in context managers (A-D) | `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/audioflamingo3.py`, `vllm/model_executor/models/aya_vision.py` |
| 2026-01-24 | [#31972](https://github.com/vllm-project/vllm/pull/31972) | merged | [Models]: Make Multimodal config implicit in ViT implementation | `vllm/model_executor/layers/attention/mm_encoder_attention.py`, `vllm/model_executor/models/clip.py`, `vllm/model_executor/models/deepencoder.py` |
| 2026-01-26 | [#33063](https://github.com/vllm-project/vllm/pull/33063) | merged | [Chore] Update type annotation of `input_ids` in model forward | `docs/contributing/model/basic.md`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`, `vllm/model_executor/models/afmoe.py` |
| 2026-02-02 | [#33165](https://github.com/vllm-project/vllm/pull/33165) | merged | [Model] Support DeepSeek-OCR-2 | `docs/models/supported_models.md`, `examples/offline_inference/vision_language.py`, `tests/models/registry.py` |
| 2026-02-05 | [#33909](https://github.com/vllm-project/vllm/pull/33909) | merged | [Models] Consolidate Deepseek-OCR2 processor | `vllm/model_executor/models/deepencoder2.py`, `vllm/model_executor/models/deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr2.py` |
| 2026-02-11 | [#34330](https://github.com/vllm-project/vllm/pull/34330) | merged | [Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder` | `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/audioflamingo3.py`, `vllm/model_executor/models/aya_vision.py` |
| 2026-02-12 | [#34085](https://github.com/vllm-project/vllm/pull/34085) | merged | Fix DeepSeek-OCR tensor validation for all size variants | `vllm/model_executor/models/deepseek_ocr.py` |
| 2026-02-23 | [#35025](https://github.com/vllm-project/vllm/pull/35025) | merged | [Refactor] Simplify dummy data generation | `docs/contributing/model/multimodal.md`, `tests/models/multimodal/processing/test_audioflamingo3.py`, `tests/models/multimodal/processing/test_common.py` |
| 2026-03-06 | [#36024](https://github.com/vllm-project/vllm/pull/36024) | merged | [Misc] Lazy import registered processors | `tests/models/registry.py`, `vllm/model_executor/models/deepseek_vl2.py`, `vllm/model_executor/models/fireredasr2.py` |
| 2026-03-12 | [#36670](https://github.com/vllm-project/vllm/pull/36670) | merged | [Bugfix][Model] Fix DeepSeek-OCR TensorSchema crash on empty images_crop | `tests/models/multimodal/processing/test_deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr.py` |
| 2026-03-17 | [#37289](https://github.com/vllm-project/vllm/pull/37289) | merged | [Bugfix] Standardize custom HF Processor init | `vllm/model_executor/models/deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr2.py`, `vllm/model_executor/models/glm4v.py` |
| 2026-03-25 | [#35182](https://github.com/vllm-project/vllm/pull/35182) | merged | [Misc] Reorganize inputs | `docs/api/README.md`, `docs/contributing/model/transcription.md`, `docs/features/multimodal_inputs.md` |
| 2026-04-27 | [#36464](https://github.com/vllm-project/vllm/pull/36464) | merged | [Examples] Resettle generate examples. | `.buildkite/test-amd.yaml`, `.buildkite/test_areas/misc.yaml`, `.buildkite/test_areas/model_runner_v2.yaml` |
| 2026-05-02 | [#40830](https://github.com/vllm-project/vllm/pull/40830) | merged | [MM][CG] Support ViT CG for Qwen2.5-VL | `docs/design/cuda_graphs_multimodal.md`, `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py` |
| 2026-05-13 | [#41736](https://github.com/vllm-project/vllm/pull/41736) | merged | [MM][CG] Support ViT CG for Qwen2-VL | `docs/design/cuda_graphs_multimodal.md`, `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py` |
| 2026-05-13 | [#42151](https://github.com/vllm-project/vllm/pull/42151) | merged | [MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5 | `docs/design/cuda_graphs_multimodal.md`, `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py` |
| 2026-05-18 | [#42224](https://github.com/vllm-project/vllm/pull/42224) | merged | [MM][CG] Enable encoder Cudagraph for Step3VL | `docs/design/cuda_graphs_multimodal.md`, `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py` |

## Per-PR Diff Audit Cards

### PR #27247 - [Model] Upstream Deepseek-OCR model

- Link: https://github.com/vllm-project/vllm/pull/27247
- Status/date: merged / 2025-10-22
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 10 files, +1821/-40, with 1953 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Upstream Deepseek-OCR model"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `docs/models/supported_models.md`, `examples/offline_inference/vision_language.py`, `tests/models/registry.py`.
- Key implementation:
  - `docs/models/supported_models.md` modified +1/-0
  - `examples/offline_inference/vision_language.py` modified +69/-20; symbols: run_deepseek_ocr, run_dots_ocr
  - `tests/models/registry.py` modified +3/-0
  - `vllm/model_executor/models/deepencoder.py` added +673/-0; symbols: MLPBlock, __init__, forward, LayerNorm2d
- Code diff details:
  - `docs/models/supported_models.md` modified +1/-0
  - `examples/offline_inference/vision_language.py` modified +69/-20
  - `tests/models/registry.py` modified +3/-0
  - `vllm/model_executor/models/deepencoder.py` added +673/-0
- Key code excerpts:

```diff
diff -- docs/models/supported_models.md
@@ -639,6 +639,7 @@ These models primarily accept the [`LLM.generate`](./generative_models.md#llmgen
 | `ChameleonForConditionalGeneration` | Chameleon | T + I | `facebook/chameleon-7b`, etc. | | ✅︎ |
 | `Cohere2VisionForConditionalGeneration` | Command A Vision | T + I<sup>+</sup> | `CohereLabs/command-a-vision-07-2025`, etc. | | ✅︎ |
 | `DeepseekVLV2ForCausalLM`<sup>^</sup> | DeepSeek-VL2 | T + I<sup>+</sup> | `deepseek-ai/deepseek-vl2-tiny`, `deepseek-ai/deepseek-vl2-small`, `deepseek-ai/deepseek-vl2`, etc. | | ✅︎ |
+| `DeepseekOCRForCausalLM` | DeepSeek-OCR | T + I<sup>+</sup> | `deepseek-ai/DeepSeek-OCR`, etc. | | ✅︎ |
 | `Ernie4_5_VLMoeForConditionalGeneration` | Ernie4.5-VL | T + I<sup>+</sup>/ V<sup>+</sup> | `baidu/ERNIE-4.5-VL-28B-A3B-PT`, `baidu/ERNIE-4.5-VL-424B-A47B-PT` | | ✅︎ |
 | `FuyuForCausalLM` | Fuyu | T + I | `adept/fuyu-8b`, etc. | | ✅︎ |
 | `Gemma3nForConditionalGeneration` | Gemma 3n | T + I + A | `google/gemma-3n-E2B-it`, `google/gemma-3n-E4B-it`, etc. | | |
diff -- examples/offline_inference/vision_language.py
@@ -30,6 +30,7 @@ class ModelRequestData(NamedTuple):
     prompts: list[str]
     stop_token_ids: list[int] | None = None
     lora_requests: list[LoRARequest] | None = None
+    sampling_params: list[SamplingParams] | None = None


 # NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
@@ -153,23 +154,6 @@ def run_chameleon(questions: list[str], modality: str) -> ModelRequestData:
     )


-# Dots-OCR
-def run_dots_ocr(questions: list[str], modality: str) -> ModelRequestData:
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/deepencoder.py` added +673/-0; `vllm/model_executor/models/deepseek_ocr.py` added +594/-0; `vllm/model_executor/models/deepseek_vl2.py` modified +23/-20; `vllm/model_executor/models/registry.py` modified +1/-0; `vllm/transformers_utils/chat_templates/registry.py` modified +1/-0; `vllm/transformers_utils/chat_templates/template_deepseek_ocr.jinja` added +14/-0; `vllm/transformers_utils/processors/deepseek_ocr.py` added +442/-0
  - tests: `tests/models/registry.py` modified +3/-0
  - docs/bench: `docs/models/supported_models.md` modified +1/-0; `examples/offline_inference/vision_language.py` modified +69/-20
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #27361 - [Bugfix] Fix deepseek-ocr multi-image inference and add `merge_by_field_config=True` with tensor schema support

- Link: https://github.com/vllm-project/vllm/pull/27361
- Status/date: merged / 2025-10-23
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 4 files, +112/-66, with 306 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix deepseek-ocr multi-image inference and add `merge_by_field_config=True` with tensor schema support"; model line: DeepSeek OCR 2; category: bug fix; main diff: `examples/offline_inference/vision_language_multi_image.py`, `tests/models/multimodal/processing/test_common.py`, `vllm/model_executor/models/deepseek_ocr.py`.
- Key implementation:
  - `examples/offline_inference/vision_language_multi_image.py` modified +48/-2; symbols: load_deepseek_ocr
  - `tests/models/multimodal/processing/test_common.py` modified +1/-0
  - `vllm/model_executor/models/deepseek_ocr.py` modified +58/-55; symbols: DeepseekOCRImagePixelInputs, _parse_and_validate_image_input, _process_image_input
  - `vllm/transformers_utils/processors/deepseek_ocr.py` modified +5/-9
- Code diff details:
  - `examples/offline_inference/vision_language_multi_image.py` modified +48/-2
  - `tests/models/multimodal/processing/test_common.py` modified +1/-0
  - `vllm/model_executor/models/deepseek_ocr.py` modified +58/-55
  - `vllm/transformers_utils/processors/deepseek_ocr.py` modified +5/-9
- Key code excerpts:

```diff
diff -- examples/offline_inference/vision_language_multi_image.py
@@ -44,6 +44,7 @@ class ModelRequestData(NamedTuple):
     stop_token_ids: list[int] | None = None
     chat_template: str | None = None
     lora_requests: list[LoRARequest] | None = None
+    sampling_params: SamplingParams | None = None


 # NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
@@ -201,6 +202,46 @@ def load_deepseek_vl2(question: str, image_urls: list[str]) -> ModelRequestData:
     )


+def load_deepseek_ocr(question: str, image_urls: list[str]) -> ModelRequestData:
+    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
diff -- tests/models/multimodal/processing/test_common.py
@@ -332,6 +332,7 @@ def _test_processing_correctness_one(
         "facebook/chameleon-7b",
         "CohereLabs/command-a-vision-07-2025",
         "deepseek-ai/deepseek-vl2-tiny",
+        "deepseek-ai/DeepSeek-OCR",
         "baidu/ERNIE-4.5-VL-28B-A3B-PT",
         "adept/fuyu-8b",
         "google/gemma-3-4b-it",
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +58/-55; `vllm/transformers_utils/processors/deepseek_ocr.py` modified +5/-9
  - tests: `tests/models/multimodal/processing/test_common.py` modified +1/-0
  - docs/bench: `examples/offline_inference/vision_language_multi_image.py` modified +48/-2
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #27560 - [Bugfix] Validate custom logits processor xargs for online serving

- Link: https://github.com/vllm-project/vllm/pull/27560
- Status/date: merged / 2025-11-05
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 18 files, +232/-49, with 574 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Validate custom logits processor xargs for online serving"; model line: DeepSeek OCR 2; category: bug fix; main diff: `docs/design/logits_processors.md`, `docs/features/custom_arguments.md`, `docs/features/custom_logitsprocs.md`.
- Key implementation:
  - `docs/design/logits_processors.md` modified +13/-1; symbols: validate_params
  - `docs/features/custom_arguments.md` modified +3/-0
  - `docs/features/custom_logitsprocs.md` modified +33/-9; symbols: validate_params
  - `examples/offline_inference/logits_processor/custom.py` modified +17/-2; symbols: validate_params, extract_extra_arg
- Code diff details:
  - `docs/design/logits_processors.md` modified +13/-1
  - `docs/features/custom_arguments.md` modified +3/-0
  - `docs/features/custom_logitsprocs.md` modified +33/-9
  - `examples/offline_inference/logits_processor/custom.py` modified +17/-2
- Key code excerpts:

```diff
diff -- docs/design/logits_processors.md
@@ -254,7 +254,15 @@ The previous sections alluded to the interfaces which vLLM logits processors mus
                 changes to the batch makeup.
             """
             raise NotImplementedError
-
+
+        @classmethod
+        def validate_params(cls, sampling_params: SamplingParams):
+            """Validate sampling params for this logits processor.
+
+            Raise ValueError for invalid ones.
+            """
+            return None
+
diff -- docs/features/custom_arguments.md
@@ -4,6 +4,9 @@ You can use vLLM *custom arguments* to pass in arguments which are not part of t

 Custom arguments can be useful if, for example, you want to use a [custom logits processor](./custom_logitsprocs.md) without modifying the vLLM source code.

+!!! note
+    Make sure your custom logits processor have implemented `validate_params` for custom arguments. Otherwise invalid custom arguments can cause unexpected behaviour.
+
 ## Offline Custom Arguments

 Custom arguments passed to `SamplingParams.extra_args` as a `dict` will be visible to any code which has access to `SamplingParams`:
```
- Reviewed files:
  - runtime: `vllm/entrypoints/openai/protocol.py` modified +2/-2; `vllm/entrypoints/openai/serving_chat.py` modified +8/-0; `vllm/entrypoints/openai/serving_completion.py` modified +9/-0; `vllm/model_executor/models/deepseek_ocr.py` modified +23/-17; `vllm/transformers_utils/configs/deepseek_vl2.py` modified +6/-0; `vllm/utils/torch_utils.py` modified +28/-0; `vllm/v1/sample/logits_processor/__init__.py` modified +20/-2; `vllm/v1/sample/logits_processor/interface.py` modified +8/-0
  - tests: `tests/entrypoints/openai/test_lora_resolvers.py` modified +1/-0; `tests/entrypoints/openai/test_serving_chat.py` modified +1/-0; `tests/v1/logits_processors/test_custom_online.py` modified +29/-0; `tests/v1/logits_processors/utils.py` modified +15/-2
  - docs/bench: `docs/design/logits_processors.md` modified +13/-1; `docs/features/custom_arguments.md` modified +3/-0; `docs/features/custom_logitsprocs.md` modified +33/-9; `examples/offline_inference/logits_processor/custom.py` modified +17/-2; `examples/offline_inference/logits_processor/custom_req.py` modified +8/-7; `examples/offline_inference/logits_processor/custom_req_init.py` modified +8/-7
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #28101 - [Model] Consolidate Deepseek-MoE implementation with DeepSeek-v2

- Link: https://github.com/vllm-project/vllm/pull/28101
- Status/date: merged / 2025-11-08
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 6 files, +144/-548, with 825 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Consolidate Deepseek-MoE implementation with DeepSeek-v2"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `tests/models/registry.py`, `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/models/deepseek_ocr.py`.
- Key implementation:
  - `tests/models/registry.py` modified +4/-1
  - `vllm/model_executor/models/deepseek.py` removed +0/-517
  - `vllm/model_executor/models/deepseek_ocr.py` modified +0/-8
  - `vllm/model_executor/models/deepseek_v2.py` modified +139/-13; symbols: DeepseekAttention, __init__, forward, DeepseekForCausalLM
- Code diff details:
  - `tests/models/registry.py` modified +4/-1
  - `vllm/model_executor/models/deepseek.py` removed +0/-517
  - `vllm/model_executor/models/deepseek_ocr.py` modified +0/-8
  - `vllm/model_executor/models/deepseek_v2.py` modified +139/-13
- Key code excerpts:

```diff
diff -- tests/models/registry.py
@@ -219,7 +219,10 @@ def check_available_online(
         "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
         trust_remote_code=True,
     ),
-    "DeepseekForCausalLM": _HfExamplesInfo("deepseek-ai/deepseek-llm-7b-chat"),
+    "DeepseekForCausalLM": _HfExamplesInfo(
+        "deepseek-ai/deepseek-moe-16b-base",
+        trust_remote_code=True,
+    ),
     "DeepseekV2ForCausalLM": _HfExamplesInfo(
         "deepseek-ai/DeepSeek-V2-Lite-Chat",
         trust_remote_code=True,
diff -- vllm/model_executor/models/deepseek.py
@@ -1,517 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
-# Copyright 2023 The vLLM team.
-# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
-#
-# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
-# and OPT implementations in this library. It has been modified from its
-# original forms to accommodate minor architectural differences compared
-# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
-#
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek.py` removed +0/-517; `vllm/model_executor/models/deepseek_ocr.py` modified +0/-8; `vllm/model_executor/models/deepseek_v2.py` modified +139/-13; `vllm/model_executor/models/deepseek_vl2.py` modified +0/-8; `vllm/model_executor/models/registry.py` modified +1/-1
  - tests: `tests/models/registry.py` modified +4/-1
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #27583 - Rename clashing method names for vLLM model protocol

- Link: https://github.com/vllm-project/vllm/pull/27583
- Status/date: merged / 2025-11-13
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +368/-367, with 2559 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Rename clashing method names for vLLM model protocol"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `docs/contributing/model/basic.md`, `docs/contributing/model/multimodal.md`, `vllm/model_executor/models/apertus.py`.
- Key implementation:
  - `docs/contributing/model/basic.md` modified +2/-2; symbols: embed_input_ids
  - `docs/contributing/model/multimodal.md` modified +3/-3; symbols: embed_multimodal
  - `vllm/model_executor/models/apertus.py` modified +4/-4; symbols: embed_input_ids
  - `vllm/model_executor/models/arcee.py` modified +4/-4; symbols: embed_input_ids
- Code diff details:
  - `docs/contributing/model/basic.md` modified +2/-2
  - `docs/contributing/model/multimodal.md` modified +3/-3
  - `vllm/model_executor/models/apertus.py` modified +4/-4
  - `vllm/model_executor/models/arcee.py` modified +4/-4
- Key code excerpts:

```diff
diff -- docs/contributing/model/basic.md
@@ -56,13 +56,13 @@ The initialization code should look like this:

 ### Computation Code

-- Add a `get_input_embeddings` method inside `MyModel` module that returns the text embeddings given `input_ids`. This is equivalent to directly calling the text embedding layer, but provides a unified interface in case `MyModel` is used within a composite multimodal model.
+- Add a `embed_input_ids` method inside `MyModel` module that returns the text embeddings given `input_ids`. This is equivalent to directly calling the text embedding layer, but provides a unified interface in case `MyModel` is used within a composite multimodal model.

 ```python
 class MyModel(nn.Module):
         ...

-    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
+    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
         ...
diff -- docs/contributing/model/multimodal.md
@@ -36,7 +36,7 @@ Further update the model as follows:

   More conveniently, you can simply pass `**kwargs` to the [forward][torch.nn.Module.forward] method and retrieve the keyword parameters for multimodal inputs from it.

-- Implement [get_multimodal_embeddings][vllm.model_executor.models.interfaces.SupportsMultiModal.get_multimodal_embeddings] that returns the embeddings from running the multimodal inputs through the multimodal tokenizer of the model. Below we provide a boilerplate of a typical implementation pattern, but feel free to adjust it to your own needs.
+- Implement [embed_multimodal][vllm.model_executor.models.interfaces.SupportsMultiModal.embed_multimodal] that returns the embeddings from running the multimodal inputs through the multimodal tokenizer of the model. Below we provide a boilerplate of a typical implementation pattern, but feel free to adjust it to your own needs.

     ??? code

@@ -49,7 +49,7 @@ Further update the model as follows:
                 image_features = self.vision_encoder(image_input)
                 return self.multi_modal_projector(image_features)

-            def get_multimodal_embeddings(
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/apertus.py` modified +4/-4; `vllm/model_executor/models/arcee.py` modified +4/-4; `vllm/model_executor/models/arctic.py` modified +4/-4; `vllm/model_executor/models/aria.py` modified +3/-3; `vllm/model_executor/models/aya_vision.py` modified +1/-1; `vllm/model_executor/models/baichuan.py` modified +4/-4; `vllm/model_executor/models/bailing_moe.py` modified +4/-4; `vllm/model_executor/models/bamba.py` modified +4/-4
  - docs/bench: `docs/contributing/model/basic.md` modified +2/-2; `docs/contributing/model/multimodal.md` modified +3/-3
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #28617 - [BugFix] DeepSeek-OCR: apply NoRepeatNGramLogitsProcessor to greedy path

- Link: https://github.com/vllm-project/vllm/pull/28617
- Status/date: merged / 2025-11-13
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, with 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] DeepSeek-OCR: apply NoRepeatNGramLogitsProcessor to greedy path"; model line: DeepSeek OCR 2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_ocr.py`.
- Key implementation:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-1
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-1
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -161,7 +161,7 @@ def validate_params(cls, params: SamplingParams):
             )

     def is_argmax_invariant(self) -> bool:
-        return True
+        return False

     def new_req_logits_processor(
         self,
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +1/-1
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #29793 - [Chore] Move tokenizer initialization methods

- Link: https://github.com/vllm-project/vllm/pull/29793
- Status/date: merged / 2025-12-02
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 51 files, +150/-129, with 761 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Chore] Move tokenizer initialization methods"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `benchmarks/benchmark_prefix_caching.py`, `benchmarks/benchmark_serving_structured_output.py`, `tests/compile/test_dynamic_shapes_compilation.py`.
- Key implementation:
  - `benchmarks/benchmark_prefix_caching.py` modified +1/-1
  - `benchmarks/benchmark_serving_structured_output.py` modified +1/-1
  - `tests/compile/test_dynamic_shapes_compilation.py` modified +1/-1
  - `tests/entrypoints/openai/test_chat_template.py` modified +1/-1
- Code diff details:
  - `benchmarks/benchmark_prefix_caching.py` modified +1/-1
  - `benchmarks/benchmark_serving_structured_output.py` modified +1/-1
  - `tests/compile/test_dynamic_shapes_compilation.py` modified +1/-1
  - `tests/entrypoints/openai/test_chat_template.py` modified +1/-1
- Key code excerpts:

```diff
diff -- benchmarks/benchmark_prefix_caching.py
@@ -40,7 +40,7 @@
 from vllm.utils.argparse_utils import FlexibleArgumentParser

 try:
-    from vllm.transformers_utils.tokenizer import get_tokenizer
+    from vllm.tokenizers import get_tokenizer
 except ImportError:
     from backend_request_func import get_tokenizer

diff -- benchmarks/benchmark_serving_structured_output.py
@@ -46,7 +46,7 @@
 from transformers import PreTrainedTokenizerBase

 try:
-    from vllm.transformers_utils.tokenizer import get_tokenizer
+    from vllm.tokenizers import get_tokenizer
 except ImportError:
     from backend_request_func import get_tokenizer

```
- Reviewed files:
  - tests: `tests/compile/test_dynamic_shapes_compilation.py` modified +1/-1; `tests/entrypoints/openai/test_chat_template.py` modified +1/-1; `tests/entrypoints/openai/test_lora_resolvers.py` modified +1/-1; `tests/entrypoints/openai/test_return_token_ids.py` modified +1/-1; `tests/entrypoints/openai/test_return_tokens_as_ids.py` modified +1/-1; `tests/entrypoints/openai/test_serving_chat.py` modified +1/-1; `tests/entrypoints/openai/test_token_in_token_out.py` modified +1/-1; `tests/entrypoints/openai/test_tokenization.py` modified +1/-1
  - docs/bench: `benchmarks/benchmark_prefix_caching.py` modified +1/-1; `benchmarks/benchmark_serving_structured_output.py` modified +1/-1
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #30035 - [Chore] Deprecate `merge_by_field_config` arg

- Link: https://github.com/vllm-project/vllm/pull/30035
- Status/date: merged / 2025-12-04
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 19 files, +90/-302, with 728 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Chore] Deprecate `merge_by_field_config` arg"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_glm4_1v.py`, `tests/models/multimodal/processing/test_tensor_schema.py`.
- Key implementation:
  - `tests/models/multimodal/processing/test_common.py` modified +2/-2
  - `tests/models/multimodal/processing/test_glm4_1v.py` modified +4/-3
  - `tests/models/multimodal/processing/test_tensor_schema.py` modified +2/-3
  - `tests/multimodal/test_cache.py` modified +3/-6
- Code diff details:
  - `tests/models/multimodal/processing/test_common.py` modified +2/-2
  - `tests/models/multimodal/processing/test_glm4_1v.py` modified +4/-3
  - `tests/models/multimodal/processing/test_tensor_schema.py` modified +2/-3
  - `tests/multimodal/test_cache.py` modified +3/-6
- Key code excerpts:

```diff
diff -- tests/models/multimodal/processing/test_common.py
@@ -20,7 +20,7 @@
 )
 from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalDataDict
 from vllm.multimodal.cache import MultiModalProcessorOnlyCache
-from vllm.multimodal.inputs import MultiModalInputs
+from vllm.multimodal.inputs import MultiModalInputs, batched_tensors_equal
 from vllm.multimodal.processing import BaseMultiModalProcessor, InputProcessingContext
 from vllm.tokenizers import (
     MistralTokenizer,
@@ -440,4 +440,4 @@ def _assert_inputs_equal(
         a_data.pop(key, None)
         b_data.pop(key, None)

-    assert a_data == b_data, msg
diff -- tests/models/multimodal/processing/test_glm4_1v.py
@@ -5,6 +5,7 @@

 from vllm.assets.video import VideoAsset
 from vllm.multimodal import MULTIMODAL_REGISTRY
+from vllm.multimodal.inputs import batched_tensors_equal
 from vllm.multimodal.video import OpenCVDynamicVideoBackend, OpenCVVideoBackend

 from ...utils import build_model_context
@@ -103,7 +104,7 @@ def test_video_loader_consistency(
     dynamic_outputs = processor.apply(prompt, dynamic_mm_data, hf_processor_mm_kwargs)

     assert static_outputs["prompt_token_ids"] == dynamic_outputs["prompt_token_ids"]
-    assert (
-        static_outputs["mm_kwargs"].get_data()
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +2/-2; `vllm/model_executor/models/interfaces.py` modified +1/-1; `vllm/model_executor/models/lightonocr.py` modified +2/-2; `vllm/model_executor/models/nano_nemotron_vl.py` modified +6/-6; `vllm/model_executor/models/opencua.py` modified +2/-2; `vllm/model_executor/models/paddleocr_vl.py` modified +2/-2; `vllm/model_executor/models/paligemma.py` modified +3/-5; `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-2
  - tests: `tests/models/multimodal/processing/test_common.py` modified +2/-2; `tests/models/multimodal/processing/test_glm4_1v.py` modified +4/-3; `tests/models/multimodal/processing/test_tensor_schema.py` modified +2/-3; `tests/multimodal/test_cache.py` modified +3/-6; `tests/multimodal/test_inputs.py` removed +0/-91
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #30170 - [Chore] Deprecate `SupportsMultiModal.merge_by_field_config`

- Link: https://github.com/vllm-project/vllm/pull/30170
- Status/date: merged / 2025-12-06
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 61 files, +23/-110, with 568 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Chore] Deprecate `SupportsMultiModal.merge_by_field_config`"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/aya_vision.py`, `vllm/model_executor/models/blip2.py`.
- Key implementation:
  - `vllm/model_executor/models/aria.py` modified +0/-2
  - `vllm/model_executor/models/aya_vision.py` modified +0/-2
  - `vllm/model_executor/models/blip2.py` modified +0/-2
  - `vllm/model_executor/models/chameleon.py` modified +0/-2
- Code diff details:
  - `vllm/model_executor/models/aria.py` modified +0/-2
  - `vllm/model_executor/models/aya_vision.py` modified +0/-2
  - `vllm/model_executor/models/blip2.py` modified +0/-2
  - `vllm/model_executor/models/chameleon.py` modified +0/-2
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/aria.py
@@ -499,8 +499,6 @@ class AriaForConditionalGeneration(nn.Module, SupportsMultiModal):
     model to perform tasks that involve both image and text inputs.
     """

-    merge_by_field_config = True
-
     hf_to_vllm_mapper = WeightsMapper(
         orig_to_new_prefix={
             # mapping for new names in checkpoint saved after transformers v4.52
diff -- vllm/model_executor/models/aya_vision.py
@@ -318,8 +318,6 @@ def _get_layer_index(feature_layer_index: int, num_hidden_layers: int) -> int:
     dummy_inputs=AyaVisionDummyInputsBuilder,
 )
 class AyaVisionForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
-    merge_by_field_config = True
-
     hf_to_vllm_mapper = WeightsMapper(
         orig_to_new_prefix={
             # mapping for new names in checkpoint saved after transformers v4.52
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/aria.py` modified +0/-2; `vllm/model_executor/models/aya_vision.py` modified +0/-2; `vllm/model_executor/models/blip2.py` modified +0/-2; `vllm/model_executor/models/chameleon.py` modified +0/-2; `vllm/model_executor/models/clip.py` modified +0/-1; `vllm/model_executor/models/cohere2_vision.py` modified +0/-2; `vllm/model_executor/models/deepseek_ocr.py` modified +0/-2; `vllm/model_executor/models/deepseek_vl2.py` modified +0/-2
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #30145 - [Renderer] Separate out `RendererConfig` from `ModelConfig`

- Link: https://github.com/vllm-project/vllm/pull/30145
- Status/date: merged / 2025-12-07
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +944/-773, with 4691 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Renderer] Separate out `RendererConfig` from `ModelConfig`"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `docs/contributing/model/transcription.md`, `tests/compile/distributed/test_sequence_parallelism.py`, `tests/compile/test_functionalization.py`.
- Key implementation:
  - `docs/contributing/model/transcription.md` modified +6/-6
  - `tests/compile/distributed/test_sequence_parallelism.py` modified +2/-0
  - `tests/compile/test_functionalization.py` modified +5/-1
  - `tests/compile/test_fusion.py` modified +5/-1
- Code diff details:
  - `docs/contributing/model/transcription.md` modified +6/-6
  - `tests/compile/distributed/test_sequence_parallelism.py` modified +2/-0
  - `tests/compile/test_functionalization.py` modified +5/-1
  - `tests/compile/test_fusion.py` modified +5/-1
- Key code excerpts:

```diff
diff -- docs/contributing/model/transcription.md
@@ -22,7 +22,7 @@ Declare supported languages and capabilities:
     import torch
     from torch import nn

-    from vllm.config import ModelConfig, SpeechToTextConfig
+    from vllm.config import RendererConfig, SpeechToTextConfig
     from vllm.inputs.data import PromptType
     from vllm.model_executor.models.interfaces import SupportsTranscription

@@ -52,7 +52,7 @@ This is for controlling general behavior of the API when serving your model:
         @classmethod
         def get_speech_to_text_config(
             cls,
-            model_config: ModelConfig,
diff -- tests/compile/distributed/test_sequence_parallelism.py
@@ -17,6 +17,7 @@
     DeviceConfig,
     ModelConfig,
     PassConfig,
+    RendererConfig,
     VllmConfig,
     get_current_vllm_config,
     set_current_vllm_config,
@@ -276,6 +277,7 @@ def sequence_parallelism_pass_on_test_model(

     vllm_config = VllmConfig(
         model_config=model_config,
+        renderer_config=RendererConfig(model_config=model_config),
         device_config=device_config,
```
- Reviewed files:
  - tests: `tests/compile/distributed/test_sequence_parallelism.py` modified +2/-0; `tests/compile/test_functionalization.py` modified +5/-1; `tests/compile/test_fusion.py` modified +5/-1; `tests/compile/test_fusion_attn.py` modified +2/-0; `tests/compile/test_pass_manager.py` modified +6/-2; `tests/compile/test_qk_norm_rope_fusion.py` modified +4/-1; `tests/distributed/test_kvlayout.py` modified +3/-0; `tests/entrypoints/openai/test_chat_template.py` modified +4/-18
  - docs/bench: `docs/contributing/model/transcription.md` modified +6/-6
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #30199 - Revert "[Renderer] Separate out `RendererConfig` from `ModelConfig` (#30145)"

- Link: https://github.com/vllm-project/vllm/pull/30199
- Status/date: merged / 2025-12-07
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +773/-944, with 4691 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[Renderer] Separate out `RendererConfig` from `ModelConfig` (#30145)""; model line: DeepSeek OCR 2; category: bug fix; main diff: `docs/contributing/model/transcription.md`, `tests/compile/distributed/test_sequence_parallelism.py`, `tests/compile/test_functionalization.py`.
- Key implementation:
  - `docs/contributing/model/transcription.md` modified +6/-6
  - `tests/compile/distributed/test_sequence_parallelism.py` modified +0/-2
  - `tests/compile/test_functionalization.py` modified +1/-5
  - `tests/compile/test_fusion.py` modified +1/-5
- Code diff details:
  - `docs/contributing/model/transcription.md` modified +6/-6
  - `tests/compile/distributed/test_sequence_parallelism.py` modified +0/-2
  - `tests/compile/test_functionalization.py` modified +1/-5
  - `tests/compile/test_fusion.py` modified +1/-5
- Key code excerpts:

```diff
diff -- docs/contributing/model/transcription.md
@@ -22,7 +22,7 @@ Declare supported languages and capabilities:
     import torch
     from torch import nn

-    from vllm.config import RendererConfig, SpeechToTextConfig
+    from vllm.config import ModelConfig, SpeechToTextConfig
     from vllm.inputs.data import PromptType
     from vllm.model_executor.models.interfaces import SupportsTranscription

@@ -52,7 +52,7 @@ This is for controlling general behavior of the API when serving your model:
         @classmethod
         def get_speech_to_text_config(
             cls,
-            renderer_config: RendererConfig,
diff -- tests/compile/distributed/test_sequence_parallelism.py
@@ -17,7 +17,6 @@
     DeviceConfig,
     ModelConfig,
     PassConfig,
-    RendererConfig,
     VllmConfig,
     get_current_vllm_config,
     set_current_vllm_config,
@@ -277,7 +276,6 @@ def sequence_parallelism_pass_on_test_model(

     vllm_config = VllmConfig(
         model_config=model_config,
-        renderer_config=RendererConfig(model_config=model_config),
         device_config=device_config,
```
- Reviewed files:
  - tests: `tests/compile/distributed/test_sequence_parallelism.py` modified +0/-2; `tests/compile/test_functionalization.py` modified +1/-5; `tests/compile/test_fusion.py` modified +1/-5; `tests/compile/test_fusion_attn.py` modified +0/-2; `tests/compile/test_pass_manager.py` modified +2/-6; `tests/compile/test_qk_norm_rope_fusion.py` modified +1/-4; `tests/distributed/test_kvlayout.py` modified +0/-3; `tests/entrypoints/openai/test_chat_template.py` modified +18/-4
  - docs/bench: `docs/contributing/model/transcription.md` modified +6/-6
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #31569 - feat: support LoRA for DeepSeek-OCR(Language Model part)

- Link: https://github.com/vllm-project/vllm/pull/31569
- Status/date: merged / 2026-01-02
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-2, with 44 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: support LoRA for DeepSeek-OCR(Language Model part)"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `docs/models/supported_models.md`, `vllm/model_executor/models/deepseek_ocr.py`.
- Key implementation:
  - `docs/models/supported_models.md` modified +1/-1
  - `vllm/model_executor/models/deepseek_ocr.py` modified +13/-1; symbols: DeepseekOCRForCausalLM, get_mm_mapping
- Code diff details:
  - `docs/models/supported_models.md` modified +1/-1
  - `vllm/model_executor/models/deepseek_ocr.py` modified +13/-1
- Key code excerpts:

```diff
diff -- docs/models/supported_models.md
@@ -677,7 +677,7 @@ These models primarily accept the [`LLM.generate`](./generative_models.md#llmgen
 | `ChameleonForConditionalGeneration` | Chameleon | T + I | `facebook/chameleon-7b`, etc. | | ✅︎ |
 | `Cohere2VisionForConditionalGeneration` | Command A Vision | T + I<sup>+</sup> | `CohereLabs/command-a-vision-07-2025`, etc. | | ✅︎ |
 | `DeepseekVLV2ForCausalLM`<sup>^</sup> | DeepSeek-VL2 | T + I<sup>+</sup> | `deepseek-ai/deepseek-vl2-tiny`, `deepseek-ai/deepseek-vl2-small`, `deepseek-ai/deepseek-vl2`, etc. | | ✅︎ |
-| `DeepseekOCRForCausalLM` | DeepSeek-OCR | T + I<sup>+</sup> | `deepseek-ai/DeepSeek-OCR`, etc. | | ✅︎ |
+| `DeepseekOCRForCausalLM` | DeepSeek-OCR | T + I<sup>+</sup> | `deepseek-ai/DeepSeek-OCR`, etc. | ✅︎ | ✅︎ |
 | `Ernie4_5_VLMoeForConditionalGeneration` | Ernie4.5-VL | T + I<sup>+</sup>/ V<sup>+</sup> | `baidu/ERNIE-4.5-VL-28B-A3B-PT`, `baidu/ERNIE-4.5-VL-424B-A47B-PT` | | ✅︎ |
 | `FuyuForCausalLM` | Fuyu | T + I | `adept/fuyu-8b`, etc. | | ✅︎ |
 | `Gemma3ForConditionalGeneration` | Gemma 3 | T + I<sup>E+</sup> | `google/gemma-3-4b-it`, `google/gemma-3-27b-it`, etc. | ✅︎ | ✅︎ |
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -14,9 +14,11 @@
 from vllm.config.multimodal import BaseDummyOptions
 from vllm.model_executor.models.interfaces import (
     MultiModalEmbeddings,
+    SupportsLoRA,
     SupportsMultiModal,
     SupportsPP,
 )
+from vllm.model_executor.models.module_mapping import MultiModelKeys
 from vllm.model_executor.models.utils import (
     AutoWeightsLoader,
     WeightsMapper,
@@ -343,7 +345,7 @@ def get_replacement_deepseek_vl2(item_idx: int):
     info=DeepseekOCRProcessingInfo,
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +13/-1
  - docs/bench: `docs/models/supported_models.md` modified +1/-1
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #31947 - [Model] Standardize common vision encoders

- Link: https://github.com/vllm-project/vllm/pull/31947
- Status/date: merged / 2026-01-08
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 19 files, +254/-174, with 1287 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Standardize common vision encoders"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/model_executor/models/clip.py`, `vllm/model_executor/models/deepencoder.py`, `vllm/model_executor/models/deepseek_ocr.py`.
- Key implementation:
  - `vllm/model_executor/models/clip.py` modified +57/-12
  - `vllm/model_executor/models/deepencoder.py` modified +3/-0
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-0
  - `vllm/model_executor/models/hyperclovax_vision.py` modified +9/-10; symbols: __init__
- Code diff details:
  - `vllm/model_executor/models/clip.py` modified +57/-12
  - `vllm/model_executor/models/deepencoder.py` modified +3/-0
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-0
  - `vllm/model_executor/models/hyperclovax_vision.py` modified +9/-10
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/clip.py
@@ -17,7 +17,7 @@
 from vllm.attention.layer import Attention
 from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
 from vllm.config import VllmConfig
-from vllm.config.multimodal import BaseDummyOptions
+from vllm.config.multimodal import BaseDummyOptions, MultiModalConfig
 from vllm.distributed import divide, get_tensor_model_parallel_world_size
 from vllm.model_executor.layers.activation import get_act_fn
 from vllm.model_executor.layers.conv import Conv2dLayer
@@ -353,6 +353,7 @@ def __init__(
         self,
         config: CLIPTextConfig | CLIPVisionConfig,
         quant_config: QuantizationConfig | None = None,
+        multimodal_config: MultiModalConfig | None = None,
diff -- vllm/model_executor/models/deepencoder.py
@@ -19,6 +19,7 @@
 from transformers import CLIPVisionConfig

 from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
+from vllm.config import MultiModalConfig
 from vllm.model_executor.layers.conv import Conv2dLayer
 from vllm.model_executor.layers.quantization import QuantizationConfig
 from vllm.model_executor.model_loader.weight_utils import default_weight_loader
@@ -608,6 +609,7 @@ def __init__(
         self,
         config: CLIPVisionConfig,
         quant_config: QuantizationConfig | None = None,
+        multimodal_config: MultiModalConfig | None = None,
         *,
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/clip.py` modified +57/-12; `vllm/model_executor/models/deepencoder.py` modified +3/-0; `vllm/model_executor/models/deepseek_ocr.py` modified +1/-0; `vllm/model_executor/models/hyperclovax_vision.py` modified +9/-10; `vllm/model_executor/models/isaac.py` modified +1/-1; `vllm/model_executor/models/keye.py` modified +2/-0; `vllm/model_executor/models/lightonocr.py` modified +3/-1; `vllm/model_executor/models/llava.py` modified +7/-2
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #32016 - [Model] Remove redundant None check in DeepSeekOCR image input processing

- Link: https://github.com/vllm-project/vllm/pull/32016
- Status/date: merged / 2026-01-09
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-13, with 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Remove redundant None check in DeepSeekOCR image input processing"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_ocr.py`.
- Key implementation:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +10/-13
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +10/-13
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -437,19 +437,16 @@ def _parse_and_validate_image_input(
         if pixel_values is None or torch.sum(pixel_values).item() == 0:
             return None

-        if pixel_values is not None:
-            base_size = self.vision_config.image_size
-            return DeepseekOCRImagePixelInputs(
-                type="pixel_values",
-                data=pixel_values,
-                images_crop=images_crop,
-                images_spatial_crop=images_spatial_crop,
-                resolve_bindings={
-                    "base_size": base_size,
-                },
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +10/-13
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #32327 - [1/N] Reorganize multimodal processing code

- Link: https://github.com/vllm-project/vllm/pull/32327
- Status/date: merged / 2026-01-14
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 76 files, +717/-670, with 2419 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[1/N] Reorganize multimodal processing code"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `docs/api/README.md`, `docs/contributing/model/multimodal.md`, `docs/design/mm_processing.md`.
- Key implementation:
  - `docs/api/README.md` modified +0/-4
  - `docs/contributing/model/multimodal.md` modified +4/-6
  - `docs/design/mm_processing.md` modified +1/-1
  - `tests/multimodal/test_processing.py` modified +2/-2
- Code diff details:
  - `docs/api/README.md` modified +0/-4
  - `docs/contributing/model/multimodal.md` modified +4/-6
  - `docs/design/mm_processing.md` modified +1/-1
  - `tests/multimodal/test_processing.py` modified +2/-2
- Key code excerpts:

```diff
diff -- docs/api/README.md
@@ -82,10 +82,6 @@ Internal data structures.

 - [vllm.multimodal.processing][]

-### Memory Profiling
-
-- [vllm.multimodal.profiling][]
-
 ### Registry

 - [vllm.multimodal.registry][]
diff -- docs/contributing/model/multimodal.md
@@ -116,12 +116,10 @@ def get_supported_mm_limits(self) -> Mapping[str, int | None]:

 ## 3. Specify dummy inputs

-Then, inherit [BaseDummyInputsBuilder][vllm.multimodal.profiling.BaseDummyInputsBuilder] to construct dummy inputs for
-HF processing as well as memory profiling.
+Then, inherit [BaseDummyInputsBuilder][vllm.multimodal.processing.BaseDummyInputsBuilder] to construct dummy inputs for
+HF processing. The processed outputs are also used for memory profiling.

-### For memory profiling
-
-Override the abstract methods [get_dummy_text][vllm.multimodal.profiling.BaseDummyInputsBuilder.get_dummy_text] and [get_dummy_mm_data][vllm.multimodal.profiling.BaseDummyInputsBuilder.get_dummy_mm_data] to construct dummy inputs for memory profiling. These dummy inputs should result in the worst-case memory usage of the model so that vLLM can reserve the correct amount of memory for it.
+Override the abstract methods [get_dummy_text][vllm.multimodal.processing.BaseDummyInputsBuilder.get_dummy_text] and [get_dummy_mm_data][vllm.multimodal.processing.BaseDummyInputsBuilder.get_dummy_mm_data] to construct dummy inputs. These dummy inputs should result in the worst-case memory usage of the model so that vLLM can reserve the correct amount of memory for it.

```
- Reviewed files:
  - runtime: `vllm/benchmarks/mm_processor.py` modified +1/-1; `vllm/model_executor/models/aria.py` modified +1/-1; `vllm/model_executor/models/audioflamingo3.py` modified +1/-1; `vllm/model_executor/models/aya_vision.py` modified +6/-3; `vllm/model_executor/models/bagel.py` modified +1/-1; `vllm/model_executor/models/blip2.py` modified +1/-1; `vllm/model_executor/models/chameleon.py` modified +1/-1; `vllm/model_executor/models/clip.py` modified +1/-1
  - tests: `tests/multimodal/test_processing.py` modified +2/-2
  - docs/bench: `docs/api/README.md` modified +0/-4; `docs/contributing/model/multimodal.md` modified +4/-6; `docs/design/mm_processing.md` modified +1/-1
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #32632 - [1/N] Initialize MM components in context managers (A-D)

- Link: https://github.com/vllm-project/vllm/pull/32632
- Status/date: merged / 2026-01-20
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 11 files, +239/-267, with 721 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[1/N] Initialize MM components in context managers (A-D)"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/audioflamingo3.py`, `vllm/model_executor/models/aya_vision.py`.
- Key implementation:
  - `vllm/model_executor/models/aria.py` modified +21/-32; symbols: compute_logits
  - `vllm/model_executor/models/audioflamingo3.py` modified +13/-15
  - `vllm/model_executor/models/aya_vision.py` modified +17/-18
  - `vllm/model_executor/models/bagel.py` modified +36/-43
- Code diff details:
  - `vllm/model_executor/models/aria.py` modified +21/-32
  - `vllm/model_executor/models/audioflamingo3.py` modified +13/-15
  - `vllm/model_executor/models/aya_vision.py` modified +17/-18
  - `vllm/model_executor/models/bagel.py` modified +36/-43
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/aria.py
@@ -15,9 +15,7 @@
 from vllm.model_executor.layers.activation import get_act_fn
 from vllm.model_executor.layers.fused_moe import SharedFusedMoE
 from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
-from vllm.model_executor.layers.logits_processor import LogitsProcessor
 from vllm.model_executor.layers.quantization import QuantizationConfig
-from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
 from vllm.model_executor.model_loader.weight_utils import (
     default_weight_loader,
     maybe_remap_kv_scale_name,
@@ -539,30 +537,22 @@ def __init__(
         quant_config = vllm_config.quant_config

         self.config = config
diff -- vllm/model_executor/models/audioflamingo3.py
@@ -460,20 +460,21 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
         multimodal_config = vllm_config.model_config.multimodal_config
         self.config = config
         self.multimodal_config = multimodal_config
-
-        self.audio_tower = AudioFlamingo3Encoder(
-            config.audio_config,
-        )
-        self.multi_modal_projector = AudioFlamingo3MultiModalProjector(config)
-
         self.quant_config = quant_config

-        self.language_model = init_vllm_registered_model(
-            vllm_config=vllm_config,
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/aria.py` modified +21/-32; `vllm/model_executor/models/audioflamingo3.py` modified +13/-15; `vllm/model_executor/models/aya_vision.py` modified +17/-18; `vllm/model_executor/models/bagel.py` modified +36/-43; `vllm/model_executor/models/blip2.py` modified +24/-30; `vllm/model_executor/models/clip.py` modified +23/-24; `vllm/model_executor/models/cohere2_vision.py` modified +17/-18; `vllm/model_executor/models/deepseek_ocr.py` modified +40/-41
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #31972 - [Models]: Make Multimodal config implicit in ViT implementation

- Link: https://github.com/vllm-project/vllm/pull/31972
- Status/date: merged / 2026-01-24
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 38 files, +118/-470, with 2781 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models]: Make Multimodal config implicit in ViT implementation"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/model_executor/layers/attention/mm_encoder_attention.py`, `vllm/model_executor/models/clip.py`, `vllm/model_executor/models/deepencoder.py`.
- Key implementation:
  - `vllm/model_executor/layers/attention/mm_encoder_attention.py` modified +0/-9
  - `vllm/model_executor/models/clip.py` modified +4/-24
  - `vllm/model_executor/models/deepencoder.py` modified +0/-3
  - `vllm/model_executor/models/deepseek_ocr.py` modified +0/-1
- Code diff details:
  - `vllm/model_executor/layers/attention/mm_encoder_attention.py` modified +0/-9
  - `vllm/model_executor/models/clip.py` modified +4/-24
  - `vllm/model_executor/models/deepencoder.py` modified +0/-3
  - `vllm/model_executor/models/deepseek_ocr.py` modified +0/-1
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/attention/mm_encoder_attention.py
@@ -4,7 +4,6 @@

 import torch

-from vllm.config import MultiModalConfig
 from vllm.logger import init_logger
 from vllm.model_executor.custom_op import CustomOp
 from vllm.model_executor.models.vision import get_vit_attn_backend
@@ -32,7 +31,6 @@ def __init__(
         scale: float | None = None,
         num_kv_heads: int | None = None,
         prefix: str = "",
-        multimodal_config: MultiModalConfig | None = None,
     ) -> None:
diff -- vllm/model_executor/models/clip.py
@@ -16,7 +16,7 @@

 from vllm.attention.layer import Attention
 from vllm.config import VllmConfig
-from vllm.config.multimodal import BaseDummyOptions, MultiModalConfig
+from vllm.config.multimodal import BaseDummyOptions
 from vllm.distributed import divide, get_tensor_model_parallel_world_size
 from vllm.model_executor.layers.activation import get_act_fn
 from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
@@ -59,6 +59,7 @@
     VisionFeatureSelectStrategy,
     VisionFeatureSelectStrategyStr,
     get_num_selected_vision_tokens,
+    is_vit_use_data_parallel,
```
- Reviewed files:
  - runtime: `vllm/model_executor/layers/attention/mm_encoder_attention.py` modified +0/-9; `vllm/model_executor/models/clip.py` modified +4/-24; `vllm/model_executor/models/deepencoder.py` modified +0/-3; `vllm/model_executor/models/deepseek_ocr.py` modified +0/-1; `vllm/model_executor/models/dots_ocr.py` modified +5/-34; `vllm/model_executor/models/eagle2_5_vl.py` modified +0/-1; `vllm/model_executor/models/ernie45_vl.py` modified +1/-12; `vllm/model_executor/models/glm4_1v.py` modified +5/-30
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #33063 - [Chore] Update type annotation of `input_ids` in model forward

- Link: https://github.com/vllm-project/vllm/pull/33063
- Status/date: merged / 2026-01-26
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +146/-143, with 1304 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Chore] Update type annotation of `input_ids` in model forward"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `docs/contributing/model/basic.md`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`, `vllm/model_executor/models/afmoe.py`.
- Key implementation:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +1/-1
  - `vllm/model_executor/models/afmoe.py` modified +2/-2
  - `vllm/model_executor/models/apertus.py` modified +1/-1
- Code diff details:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +1/-1
  - `vllm/model_executor/models/afmoe.py` modified +2/-2
  - `vllm/model_executor/models/apertus.py` modified +1/-1
- Key code excerpts:

```diff
diff -- docs/contributing/model/basic.md
@@ -71,7 +71,7 @@ class MyModel(nn.Module):
 ```python
 def forward(
     self,
-    input_ids: torch.Tensor,
+    input_ids: torch.Tensor | None,
     positions: torch.Tensor,
     intermediate_tensors: IntermediateTensors | None = None,
     inputs_embeds: torch.Tensor | None = None,
diff -- tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py
@@ -36,7 +36,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):

     def forward(
         self,
-        input_ids: torch.Tensor,
+        input_ids: torch.Tensor | None,
         positions: torch.Tensor,
         intermediate_tensors: IntermediateTensors | None = None,
         inputs_embeds: torch.Tensor | None = None,
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/afmoe.py` modified +2/-2; `vllm/model_executor/models/apertus.py` modified +1/-1; `vllm/model_executor/models/arcee.py` modified +1/-1; `vllm/model_executor/models/arctic.py` modified +2/-2; `vllm/model_executor/models/aria.py` modified +1/-1; `vllm/model_executor/models/audioflamingo3.py` modified +1/-1; `vllm/model_executor/models/aya_vision.py` modified +1/-1; `vllm/model_executor/models/bagel.py` modified +1/-1
  - tests: `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +1/-1
  - docs/bench: `docs/contributing/model/basic.md` modified +1/-1
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #33165 - [Model] Support DeepSeek-OCR-2

- Link: https://github.com/vllm-project/vllm/pull/33165
- Status/date: merged / 2026-02-02
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 9 files, +1099/-1, with 1159 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support DeepSeek-OCR-2"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `docs/models/supported_models.md`, `examples/offline_inference/vision_language.py`, `tests/models/registry.py`.
- Key implementation:
  - `docs/models/supported_models.md` modified +1/-0
  - `examples/offline_inference/vision_language.py` modified +44/-0; symbols: run_deepseek_ocr2
  - `tests/models/registry.py` modified +3/-0
  - `vllm/model_executor/models/deepencoder.py` modified +2/-1
- Code diff details:
  - `docs/models/supported_models.md` modified +1/-0
  - `examples/offline_inference/vision_language.py` modified +44/-0
  - `tests/models/registry.py` modified +3/-0
  - `vllm/model_executor/models/deepencoder.py` modified +2/-1
- Key code excerpts:

```diff
diff -- docs/models/supported_models.md
@@ -672,6 +672,7 @@ These models primarily accept the [`LLM.generate`](./generative_models.md#llmgen
 | `Cohere2VisionForConditionalGeneration` | Command A Vision | T + I<sup>+</sup> | `CohereLabs/command-a-vision-07-2025`, etc. | | ✅︎ |
 | `DeepseekVLV2ForCausalLM`<sup>^</sup> | DeepSeek-VL2 | T + I<sup>+</sup> | `deepseek-ai/deepseek-vl2-tiny`, `deepseek-ai/deepseek-vl2-small`, `deepseek-ai/deepseek-vl2`, etc. | | ✅︎ |
 | `DeepseekOCRForCausalLM` | DeepSeek-OCR | T + I<sup>+</sup> | `deepseek-ai/DeepSeek-OCR`, etc. | ✅︎ | ✅︎ |
+| `DeepseekOCR2ForCausalLM` | DeepSeek-OCR-2 | T + I<sup>+</sup> | `deepseek-ai/DeepSeek-OCR-2`, etc. | ✅︎ | ✅︎ |
 | `Eagle2_5_VLForConditionalGeneration` | Eagle2.5-VL | T + I<sup>E+</sup> | `nvidia/Eagle2.5-8B`, etc. | ✅︎ | ✅︎ |
 | `Ernie4_5_VLMoeForConditionalGeneration` | Ernie4.5-VL | T + I<sup>+</sup>/ V<sup>+</sup> | `baidu/ERNIE-4.5-VL-28B-A3B-PT`, `baidu/ERNIE-4.5-VL-424B-A47B-PT` | | ✅︎ |
 | `FuyuForCausalLM` | Fuyu | T + I | `adept/fuyu-8b`, etc. | | ✅︎ |
diff -- examples/offline_inference/vision_language.py
@@ -270,6 +270,49 @@ def run_deepseek_ocr(questions: list[str], modality: str) -> ModelRequestData:
     )


+def run_deepseek_ocr2(questions: list[str], modality: str) -> ModelRequestData:
+    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
+
+    assert modality == "image"
+
+    model_name = "deepseek-ai/DeepSeek-OCR-2"
+
+    engine_args = EngineArgs(
+        model=model_name,
+        limit_mm_per_prompt={modality: 1},
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/deepencoder.py` modified +2/-1; `vllm/model_executor/models/deepencoder2.py` added +283/-0; `vllm/model_executor/models/deepseek_ocr2.py` added +444/-0; `vllm/model_executor/models/registry.py` modified +1/-0; `vllm/transformers_utils/chat_templates/registry.py` modified +1/-0; `vllm/transformers_utils/processors/deepseek_ocr2.py` added +320/-0
  - tests: `tests/models/registry.py` modified +3/-0
  - docs/bench: `docs/models/supported_models.md` modified +1/-0; `examples/offline_inference/vision_language.py` modified +44/-0
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #33909 - [Models] Consolidate Deepseek-OCR2 processor

- Link: https://github.com/vllm-project/vllm/pull/33909
- Status/date: merged / 2026-02-05
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 5 files, +52/-336, with 480 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Consolidate Deepseek-OCR2 processor"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/model_executor/models/deepencoder2.py`, `vllm/model_executor/models/deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr2.py`.
- Key implementation:
  - `vllm/model_executor/models/deepencoder2.py` modified +1/-1
  - `vllm/model_executor/models/deepseek_ocr.py` modified +10/-2
  - `vllm/model_executor/models/deepseek_ocr2.py` modified +12/-4
  - `vllm/transformers_utils/processors/deepseek_ocr.py` modified +29/-9
- Code diff details:
  - `vllm/model_executor/models/deepencoder2.py` modified +1/-1
  - `vllm/model_executor/models/deepseek_ocr.py` modified +10/-2
  - `vllm/model_executor/models/deepseek_ocr2.py` modified +12/-4
  - `vllm/transformers_utils/processors/deepseek_ocr.py` modified +29/-9
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepencoder2.py
@@ -31,7 +31,7 @@ def __init__(
         num_key_value_heads: int = 2,
         intermediate_size: int = 4864,
         vocab_size: int = 151936,
-        attn_implementation: str = "sdpa",  # ⭐
+        attn_implementation: str = "sdpa",
         rms_norm_eps: float = 1e-06,
         rope_theta: float = 1000000.0,
         attention_dropout: float = 0.0,
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -52,7 +52,6 @@
 from vllm.transformers_utils.processors.deepseek_ocr import (
     BASE_SIZE,
     CROP_MODE,
-    IMAGE_SIZE,
     DeepseekOCRProcessor,
     count_tiles,
 )
@@ -66,6 +65,7 @@
 from .deepseek_vl2 import MlpProjector

 # The image token id may be various
+IMAGE_SIZE = 640
 _IMAGE_TOKEN = "<image>"
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/deepencoder2.py` modified +1/-1; `vllm/model_executor/models/deepseek_ocr.py` modified +10/-2; `vllm/model_executor/models/deepseek_ocr2.py` modified +12/-4; `vllm/transformers_utils/processors/deepseek_ocr.py` modified +29/-9; `vllm/transformers_utils/processors/deepseek_ocr2.py` removed +0/-320
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #34330 - [Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder`

- Link: https://github.com/vllm-project/vllm/pull/34330
- Status/date: merged / 2026-02-11
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 72 files, +131/-27, with 784 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder`"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/audioflamingo3.py`, `vllm/model_executor/models/aya_vision.py`.
- Key implementation:
  - `vllm/model_executor/models/aria.py` modified +1/-0
  - `vllm/model_executor/models/audioflamingo3.py` modified +4/-1
  - `vllm/model_executor/models/aya_vision.py` modified +1/-0
  - `vllm/model_executor/models/bagel.py` modified +1/-0
- Code diff details:
  - `vllm/model_executor/models/aria.py` modified +1/-0
  - `vllm/model_executor/models/audioflamingo3.py` modified +4/-1
  - `vllm/model_executor/models/aya_vision.py` modified +1/-0
  - `vllm/model_executor/models/bagel.py` modified +1/-0
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/aria.py
@@ -445,6 +445,7 @@ def get_dummy_mm_data(
         seq_len: int,
         mm_counts: Mapping[str, int],
         mm_options: Mapping[str, BaseDummyOptions] | None = None,
+        mm_processor_kwargs: Mapping[str, object] | None = None,
     ) -> MultiModalDataDict:
         vision_config = self.info.get_vision_config()

diff -- vllm/model_executor/models/audioflamingo3.py
@@ -253,8 +253,11 @@ def get_dummy_mm_data(
         seq_len: int,
         mm_counts: Mapping[str, int],
         mm_options: Mapping[str, BaseDummyOptions] | None = None,
+        mm_processor_kwargs: Mapping[str, object] | None = None,
     ) -> MultiModalDataDict:
-        feature_extractor = self.info.get_feature_extractor()
+        feature_extractor = self.info.get_feature_extractor(
+            **(mm_processor_kwargs or {})
+        )
         sampling_rate = feature_extractor.sampling_rate
         audio_len = MAX_AUDIO_LEN * sampling_rate
         num_audios = mm_counts.get("audio", 0)
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/aria.py` modified +1/-0; `vllm/model_executor/models/audioflamingo3.py` modified +4/-1; `vllm/model_executor/models/aya_vision.py` modified +1/-0; `vllm/model_executor/models/bagel.py` modified +1/-0; `vllm/model_executor/models/bee.py` modified +1/-0; `vllm/model_executor/models/blip2.py` modified +1/-0; `vllm/model_executor/models/chameleon.py` modified +1/-0; `vllm/model_executor/models/clip.py` modified +1/-0
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #34085 - Fix DeepSeek-OCR tensor validation for all size variants

- Link: https://github.com/vllm-project/vllm/pull/34085
- Status/date: merged / 2026-02-12
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-1, with 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix DeepSeek-OCR tensor validation for all size variants"; model line: DeepSeek OCR 2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_ocr.py`.
- Key implementation:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +11/-1
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +11/-1
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -448,14 +448,24 @@ def _parse_and_validate_image_input(
         if pixel_values is None or torch.sum(pixel_values).item() == 0:
             return None

-        base_size = self.vision_config.image_size
+        # Use actual tensor spatial dim instead of hardcoded
+        # vision_config.image_size (1024). The vision encoders (SAM & CLIP)
+        # support arbitrary resolutions via pos-encoding interpolation,
+        # so Tiny/Small/Base/Large variants all work with the same weights.
+        base_size = pixel_values.shape[-1]
+        if images_crop is not None and images_crop.numel() > 0:
+            image_size = images_crop.shape[-1]
+        else:
+            image_size = base_size
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +11/-1
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #35025 - [Refactor] Simplify dummy data generation

- Link: https://github.com/vllm-project/vllm/pull/35025
- Status/date: merged / 2026-02-23
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 78 files, +282/-367, with 1791 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Simplify dummy data generation"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `docs/contributing/model/multimodal.md`, `tests/models/multimodal/processing/test_audioflamingo3.py`, `tests/models/multimodal/processing/test_common.py`.
- Key implementation:
  - `docs/contributing/model/multimodal.md` modified +11/-11
  - `tests/models/multimodal/processing/test_audioflamingo3.py` modified +1/-1
  - `tests/models/multimodal/processing/test_common.py` modified +2/-0
  - `tests/models/multimodal/processing/test_tensor_schema.py` modified +1/-0
- Code diff details:
  - `docs/contributing/model/multimodal.md` modified +11/-11
  - `tests/models/multimodal/processing/test_audioflamingo3.py` modified +1/-1
  - `tests/models/multimodal/processing/test_common.py` modified +2/-0
  - `tests/models/multimodal/processing/test_tensor_schema.py` modified +1/-0
- Key code excerpts:

```diff
diff -- docs/contributing/model/multimodal.md
@@ -293,21 +293,22 @@ Assuming that the memory usage increases with the number of tokens, the dummy in
             self,
             seq_len: int,
             mm_counts: Mapping[str, int],
-            mm_options: Mapping[str, BaseDummyOptions] | None = None,
+            mm_options: Mapping[str, BaseDummyOptions],
         ) -> MultiModalDataDict:
             num_images = mm_counts.get("image", 0)

             target_width, target_height = \
                 self.info.get_image_size_with_most_features()

-            image_overrides = mm_options.get("image") if mm_options else None
+            image_overrides = mm_options.get("image")
diff -- tests/models/multimodal/processing/test_audioflamingo3.py
@@ -116,7 +116,7 @@ def test_dummy_data_generation(mock_ctx):
     builder = AudioFlamingo3DummyInputsBuilder(info)

     mm_counts = {"audio": 2}
-    dummy_data = builder.get_dummy_mm_data(100, mm_counts, None)
+    dummy_data = builder.get_dummy_mm_data(100, mm_counts, {})

     assert "audio" in dummy_data
     assert len(dummy_data["audio"]) == 2
```
- Reviewed files:
  - runtime: `vllm/config/multimodal.py` modified +36/-20; `vllm/model_executor/models/aria.py` modified +2/-3; `vllm/model_executor/models/audioflamingo3.py` modified +3/-6; `vllm/model_executor/models/aya_vision.py` modified +2/-3; `vllm/model_executor/models/bagel.py` modified +2/-3; `vllm/model_executor/models/bee.py` modified +2/-3; `vllm/model_executor/models/blip2.py` modified +2/-3; `vllm/model_executor/models/chameleon.py` modified +2/-3
  - tests: `tests/models/multimodal/processing/test_audioflamingo3.py` modified +1/-1; `tests/models/multimodal/processing/test_common.py` modified +2/-0; `tests/models/multimodal/processing/test_tensor_schema.py` modified +1/-0
  - docs/bench: `docs/contributing/model/multimodal.md` modified +11/-11
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #36024 - [Misc] Lazy import registered processors

- Link: https://github.com/vllm-project/vllm/pull/36024
- Status/date: merged / 2026-03-06
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 15 files, +68/-51, with 288 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Lazy import registered processors"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `tests/models/registry.py`, `vllm/model_executor/models/deepseek_vl2.py`, `vllm/model_executor/models/fireredasr2.py`.
- Key implementation:
  - `tests/models/registry.py` modified +2/-5
  - `vllm/model_executor/models/deepseek_vl2.py` modified +1/-2
  - `vllm/model_executor/models/fireredasr2.py` modified +1/-1
  - `vllm/model_executor/models/funasr.py` modified +1/-1
- Code diff details:
  - `tests/models/registry.py` modified +2/-5
  - `vllm/model_executor/models/deepseek_vl2.py` modified +1/-2
  - `vllm/model_executor/models/fireredasr2.py` modified +1/-1
  - `vllm/model_executor/models/funasr.py` modified +1/-1
- Key code excerpts:

```diff
diff -- tests/models/registry.py
@@ -1020,18 +1020,15 @@ def check_available_online(
         min_transformers_version="4.57",
     ),
     "Qwen3ASRForConditionalGeneration": _HfExamplesInfo(
-        "Qwen/Qwen3-ASR-1.7B",
+        "Qwen/Qwen3-ASR-0.6B",
         max_model_len=4096,
         min_transformers_version="4.57",
-        is_available_online=False,
     ),
     "Qwen3ASRRealtimeGeneration": _HfExamplesInfo(
-        "Qwen/Qwen3-ASR-1.7B",
+        "Qwen/Qwen3-ASR-0.6B",
         max_model_len=4096,
diff -- vllm/model_executor/models/deepseek_vl2.py
@@ -48,7 +48,6 @@
     MlpProjectorConfig,
     VisionEncoderConfig,
 )
-from vllm.transformers_utils.processors.deepseek_vl2 import DeepseekVLV2Processor
 from vllm.utils.tensor_schema import TensorSchema, TensorShape
 from vllm.utils.torch_utils import set_default_torch_dtype

@@ -160,7 +159,7 @@ def get_hf_config(self):
         return self.ctx.get_hf_config(DeepseekVLV2Config)

     def get_hf_processor(self, **kwargs: object):
-        return self.ctx.get_hf_processor(DeepseekVLV2Processor, **kwargs)
+        return self.ctx.get_hf_processor(**kwargs)
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_vl2.py` modified +1/-2; `vllm/model_executor/models/fireredasr2.py` modified +1/-1; `vllm/model_executor/models/funasr.py` modified +1/-1; `vllm/transformers_utils/processor.py` modified +31/-1; `vllm/transformers_utils/processors/__init__.py` modified +28/-10; `vllm/transformers_utils/processors/bagel.py` modified +0/-4; `vllm/transformers_utils/processors/deepseek_ocr.py` modified +1/-4; `vllm/transformers_utils/processors/deepseek_vl2.py` modified +1/-4
  - tests: `tests/models/registry.py` modified +2/-5
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #36670 - [Bugfix][Model] Fix DeepSeek-OCR TensorSchema crash on empty images_crop

- Link: https://github.com/vllm-project/vllm/pull/36670
- Status/date: merged / 2026-03-12
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 2 files, +135/-4, with 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Model] Fix DeepSeek-OCR TensorSchema crash on empty images_crop"; model line: DeepSeek OCR 2; category: bug fix; main diff: `tests/models/multimodal/processing/test_deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr.py`.
- Key implementation:
  - `tests/models/multimodal/processing/test_deepseek_ocr.py` added +134/-0; symbols: processor, TestDeepseekOCREmptyImagesCrop, test_empty_images_crop_small_image, test_populated_images_crop_large_image
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-4
- Code diff details:
  - `tests/models/multimodal/processing/test_deepseek_ocr.py` added +134/-0
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-4
- Key code excerpts:

```diff
diff -- tests/models/multimodal/processing/test_deepseek_ocr.py
@@ -0,0 +1,134 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Regression test for DeepSeek-OCR TensorSchema validation with empty images_crop.
+
+When using the Gundam preset (BASE_SIZE=1024, IMAGE_SIZE=640, CROP_MODE=True),
+images that are small enough to not require cropping produce an empty
+images_crop tensor with shape (0, 3, 640, 640). The _parse_and_validate_image_input
+method must correctly read image_size from this tensor's shape rather than
+falling back to base_size, which would cause a TensorSchema mismatch.
+
+Run with:
+  pytest tests/models/multimodal/processing/test_deepseek_ocr.py -v
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -452,10 +452,7 @@ def _parse_and_validate_image_input(
         # support arbitrary resolutions via pos-encoding interpolation,
         # so Tiny/Small/Base/Large variants all work with the same weights.
         base_size = pixel_values.shape[-1]
-        if images_crop is not None and images_crop.numel() > 0:
-            image_size = images_crop.shape[-1]
-        else:
-            image_size = base_size
+        image_size = images_crop.shape[-1] if images_crop is not None else base_size

         return DeepseekOCRImagePixelInputs(
             type="pixel_values",
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +1/-4
  - tests: `tests/models/multimodal/processing/test_deepseek_ocr.py` added +134/-0
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #37289 - [Bugfix] Standardize custom HF Processor init

- Link: https://github.com/vllm-project/vllm/pull/37289
- Status/date: merged / 2026-03-17
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 7 files, +39/-33, with 152 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Standardize custom HF Processor init"; model line: DeepSeek OCR 2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr2.py`, `vllm/model_executor/models/glm4v.py`.
- Key implementation:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +3/-1
  - `vllm/model_executor/models/deepseek_ocr2.py` modified +3/-1
  - `vllm/model_executor/models/glm4v.py` modified +11/-3; symbols: get_image_processor, get_hf_processor
  - `vllm/model_executor/models/qwen_vl.py` modified +11/-3; symbols: get_image_processor, get_hf_processor
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +3/-1
  - `vllm/model_executor/models/deepseek_ocr2.py` modified +3/-1
  - `vllm/model_executor/models/glm4v.py` modified +11/-3
  - `vllm/model_executor/models/qwen_vl.py` modified +11/-3
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -196,8 +196,10 @@ def get_hf_processor(self, **kwargs: object):
             crop_mode=CROP_MODE,
             strategy="v1",
         )
+
         return self.ctx.get_hf_processor(
-            DeepseekOCRProcessor, **{**kwargs, **v1_processor_config}
+            DeepseekOCRProcessor,
+            **{**v1_processor_config, **kwargs},
         )

     def get_supported_mm_limits(self) -> Mapping[str, int | None]:
diff -- vllm/model_executor/models/deepseek_ocr2.py
@@ -76,8 +76,10 @@ def get_hf_processor(self, **kwargs: object):
             crop_mode=CROP_MODE,
             strategy="v2",
         )
+
         return self.ctx.get_hf_processor(
-            DeepseekOCRProcessor, **{**kwargs, **v2_processor_config}
+            DeepseekOCRProcessor,
+            **{**v2_processor_config, **kwargs},
         )

     def get_supported_mm_limits(self) -> Mapping[str, int | None]:
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +3/-1; `vllm/model_executor/models/deepseek_ocr2.py` modified +3/-1; `vllm/model_executor/models/glm4v.py` modified +11/-3; `vllm/model_executor/models/qwen_vl.py` modified +11/-3; `vllm/tokenizers/qwen_vl.py` modified +4/-0; `vllm/transformers_utils/processors/glm4v.py` modified +2/-7; `vllm/transformers_utils/processors/qwen_vl.py` modified +5/-18
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #35182 - [Misc] Reorganize inputs

- Link: https://github.com/vllm-project/vllm/pull/35182
- Status/date: merged / 2026-03-25
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +1037/-959, with 4452 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Reorganize inputs"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `docs/api/README.md`, `docs/contributing/model/transcription.md`, `docs/features/multimodal_inputs.md`.
- Key implementation:
  - `docs/api/README.md` modified +3/-12
  - `docs/contributing/model/transcription.md` modified +2/-2
  - `docs/features/multimodal_inputs.md` modified +1/-1
  - `examples/pooling/token_embed/jina_embeddings_v4_offline.py` modified +1/-1
- Code diff details:
  - `docs/api/README.md` modified +3/-12
  - `docs/contributing/model/transcription.md` modified +2/-2
  - `docs/features/multimodal_inputs.md` modified +1/-1
  - `examples/pooling/token_embed/jina_embeddings_v4_offline.py` modified +1/-1
- Key code excerpts:

```diff
diff -- docs/api/README.md
@@ -27,11 +27,9 @@ LLM Class.

 - [vllm.LLM][]

-LLM Inputs.
+Prompt schema for LLM APIs.

-- [vllm.inputs.PromptType][]
-- [vllm.inputs.TextPrompt][]
-- [vllm.inputs.TokensPrompt][]
+- [vllm.inputs.llm][]

 ## vLLM Engines

diff -- docs/contributing/model/transcription.md
@@ -23,7 +23,7 @@ Declare supported languages and capabilities:
     from torch import nn

     from vllm.config import ModelConfig, SpeechToTextConfig
-    from vllm.inputs.data import PromptType
+    from vllm.inputs import PromptType
     from vllm.model_executor.models.interfaces import SupportsTranscription

     class YourASRModel(nn.Module, SupportsTranscription):
@@ -66,7 +66,7 @@ This is for controlling general behavior of the API when serving your model:

 See [Audio preprocessing and chunking](#audio-preprocessing-and-chunking) for what each field controls.

-Implement the prompt construction via [get_generation_prompt][vllm.model_executor.models.interfaces.SupportsTranscription.get_generation_prompt]. The server passes you the resampled waveform and task parameters; you return a valid [PromptType][vllm.inputs.data.PromptType]. There are two common patterns:
```
- Reviewed files:
  - runtime: `vllm/beam_search.py` modified +18/-13; `vllm/benchmarks/datasets.py` modified +1/-1; `vllm/engine/protocol.py` modified +4/-4; `vllm/entrypoints/anthropic/serving.py` modified +4/-4; `vllm/entrypoints/chat_utils.py` modified +2/-1; `vllm/entrypoints/llm.py` modified +22/-22; `vllm/entrypoints/openai/chat_completion/serving.py` modified +11/-11; `vllm/entrypoints/openai/completion/serving.py` modified +15/-17
  - tests: `tests/entrypoints/openai/chat_completion/test_chat_error.py` modified +1/-1; `tests/entrypoints/openai/chat_completion/test_serving_chat.py` modified +8/-8; `tests/entrypoints/openai/responses/test_serving_responses.py` modified +5/-5; `tests/entrypoints/serve/render/test_launch_render.py` modified +0/-14; `tests/entrypoints/test_chat_utils.py` modified +1/-1; `tests/models/multimodal/generation/test_pixtral.py` modified +1/-1; `tests/models/multimodal/processing/test_common.py` modified +6/-8; `tests/plugins/bge_m3_sparse_plugin/bge_m3_sparse_processor/sparse_embeddings_processor.py` modified +2/-4
  - docs/bench: `docs/api/README.md` modified +3/-12; `docs/contributing/model/transcription.md` modified +2/-2; `docs/features/multimodal_inputs.md` modified +1/-1; `examples/pooling/token_embed/jina_embeddings_v4_offline.py` modified +1/-1
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #36464 - [Examples] Resettle generate examples.

- Link: https://github.com/vllm-project/vllm/pull/36464
- Status/date: merged / 2026-04-27
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 36 files, +46/-50, with 267 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Examples] Resettle generate examples."; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `.buildkite/test-amd.yaml`, `.buildkite/test_areas/misc.yaml`, `.buildkite/test_areas/model_runner_v2.yaml`.
- Key implementation:
  - `.buildkite/test-amd.yaml` modified +14/-14
  - `.buildkite/test_areas/misc.yaml` modified +4/-4
  - `.buildkite/test_areas/model_runner_v2.yaml` modified +4/-4
  - `.buildkite/test_areas/models_basic.yaml` modified +5/-5
- Code diff details:
  - `.buildkite/test-amd.yaml` modified +14/-14
  - `.buildkite/test_areas/misc.yaml` modified +4/-4
  - `.buildkite/test_areas/model_runner_v2.yaml` modified +4/-4
  - `.buildkite/test_areas/models_basic.yaml` modified +5/-5
- Key code excerpts:

```diff
diff -- .buildkite/test-amd.yaml
@@ -388,10 +388,10 @@ steps:
     - python3 basic/offline_inference/embed.py
     - python3 basic/offline_inference/score.py
     # Multi-modal models
-    - python3 offline_inference/audio_language.py --seed 0
-    - python3 offline_inference/vision_language.py --seed 0
-    - python3 offline_inference/vision_language_multi_image.py --seed 0
-    - python3 offline_inference/encoder_decoder_multimodal.py --model-type whisper --seed 0
+    - python3 generate/multimodal/audio_language_offline.py --seed 0
+    - python3 generate/multimodal/vision_language_offline.py --seed 0
+    - python3 generate/multimodal/vision_language_multi_image_offline.py --seed 0
+    - python3 generate/multimodal/encoder_decoder_multimodal_offline.py --model-type whisper --seed 0
     # Pooling models
     - python3 pooling/embed/vision_embedding_offline.py --seed 0
diff -- .buildkite/test_areas/misc.yaml
@@ -113,10 +113,10 @@ steps:
     - python3 basic/offline_inference/embed.py
     - python3 basic/offline_inference/score.py
     # for multi-modal models
-    - python3 offline_inference/audio_language.py --seed 0
-    - python3 offline_inference/vision_language.py --seed 0
-    - python3 offline_inference/vision_language_multi_image.py --seed 0
-    - python3 offline_inference/encoder_decoder_multimodal.py --model-type whisper --seed 0
+    - python3 generate/multimodal/audio_language_offline.py --seed 0
+    - python3 generate/multimodal/vision_language_offline.py --seed 0
+    - python3 generate/multimodal/vision_language_multi_image_offline.py --seed 0
+    - python3 generate/multimodal/encoder_decoder_multimodal_offline.py --model-type whisper --seed 0
      # for pooling models
     - python3 pooling/embed/vision_embedding_offline.py --seed 0
```
- Reviewed files:
  - docs/bench: `.buildkite/test-amd.yaml` modified +14/-14; `.buildkite/test_areas/misc.yaml` modified +4/-4; `.buildkite/test_areas/model_runner_v2.yaml` modified +4/-4; `.buildkite/test_areas/models_basic.yaml` modified +5/-5; `docs/features/multimodal_inputs.md` modified +7/-7; `docs/features/reasoning_outputs.md` modified +1/-1; `docs/serving/openai_compatible_server.md` modified +3/-3; `examples/generate/batched_chat_completions_online.py` renamed +0/-0
  - other: `.github/mergify.yml` modified +1/-5
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #40830 - [MM][CG] Support ViT CG for Qwen2.5-VL

- Link: https://github.com/vllm-project/vllm/pull/40830
- Status/date: merged / 2026-05-02
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 5 files, +539/-22, with 669 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][CG] Support ViT CG for Qwen2.5-VL"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `docs/design/cuda_graphs_multimodal.md`, `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py`.
- Key implementation:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-0
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
  - `tests/models/multimodal/generation/test_qwen2_5_vl.py` modified +95/-0; symbols: _window_attention_regression_image, _encoder_cudagraph_config, test_qwen2_5_vl_window_attention_image, test_qwen2_5_vl_window_attention_image_batch
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-1
- Code diff details:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-0
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
  - `tests/models/multimodal/generation/test_qwen2_5_vl.py` modified +95/-0
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-1
- Key code excerpts:

```diff
diff -- docs/design/cuda_graphs_multimodal.md
@@ -86,9 +86,11 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra
 | Architecture | Models | CG for Image | CG for Video |
 | ------------ | ------ | ------------ | ------------ |
 | `Qwen3VLForConditionalGeneration` | `Qwen3-VL` | ✅︎ | ✅︎ |
+| `Qwen2_5_VLForConditionalGeneration` | `Qwen2.5-VL` | ✅︎ | ✅︎ |

 !!! note
     Encoder CUDA Graphs have currently been tested with `--mm-encoder-attn-backend=FLASH_ATTN` and `--mm-encoder-attn-backend=FLASHINFER` on Blackwell GPUs.
+    For Qwen2.5-VL only FA2 and FA3 has been tested.

 ## Configuration

diff -- examples/generate/multimodal/vision_language_offline.py
@@ -2466,6 +2466,7 @@ def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:
 MODELS_SUPPORT_VIT_CUDA_GRAPH = [
     "qwen3_vl",
     "qwen3_vl_moe",
+    "qwen2_5_vl",
 ]


```
- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +429/-21
  - tests: `tests/models/multimodal/generation/test_qwen2_5_vl.py` modified +95/-0; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-1
  - docs/bench: `docs/design/cuda_graphs_multimodal.md` modified +2/-0; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #41736 - [MM][CG] Support ViT CG for Qwen2-VL

- Link: https://github.com/vllm-project/vllm/pull/41736
- Status/date: merged / 2026-05-13
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 4 files, +315/-21, with 415 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][CG] Support ViT CG for Qwen2-VL"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `docs/design/cuda_graphs_multimodal.md`, `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`.
- Key implementation:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-1
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0
  - `vllm/model_executor/models/qwen2_vl.py` modified +300/-20; symbols: prepare_encoder_metadata, get_encoder_cudagraph_config, get_input_modality, get_max_frames_per_video
- Code diff details:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-1
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0
  - `vllm/model_executor/models/qwen2_vl.py` modified +300/-20
- Key code excerpts:

```diff
diff -- docs/design/cuda_graphs_multimodal.md
@@ -85,13 +85,14 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra

 | Architecture | Models | CG for Image | CG for Video |
 | ------------ | ------ | ------------ | ------------ |
+| `Qwen2VLForConditionalGeneration` | `Qwen2-VL` | ✅︎ | ✅︎ |
 | `Qwen2_5_VLForConditionalGeneration` | `Qwen2.5-VL` | ✅︎ | ✅︎ |
 | `Qwen3VLForConditionalGeneration` | `Qwen3-VL` | ✅︎ | ✅︎ |
 | `Qwen3_5ForConditionalGeneration` | `Qwen3.5` | ✅︎ | ✅︎ |

 !!! note
     Encoder CUDA Graphs have currently been tested with `--mm-encoder-attn-backend=FLASH_ATTN` and `--mm-encoder-attn-backend=FLASHINFER` on Blackwell GPUs.
-    For Qwen2.5-VL only FA2 and FA3 has been tested.
+    For Qwen2-VL and Qwen2.5-VL only FA2 and FA3 has been tested.

diff -- examples/generate/multimodal/vision_language_offline.py
@@ -2557,6 +2557,7 @@ def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:
     "qwen2_5_vl",
     "qwen3_vl",
     "qwen3_vl_moe",
+    "qwen2_vl",
     "qwen3_5",
     "qwen3_5_moe",
 ]
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +300/-20
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0
  - docs/bench: `docs/design/cuda_graphs_multimodal.md` modified +2/-1; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #42151 - [MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/42151
- Status/date: merged / 2026-05-13
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 4 files, +112/-5, with 187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5"; model line: DeepSeek OCR 2; category: performance/backend optimization; main diff: `docs/design/cuda_graphs_multimodal.md`, `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`.
- Key implementation:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-1
  - `examples/generate/multimodal/vision_language_offline.py` modified +93/-1; symbols: run_qwen3_5, run_qwen3_5_moe
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +15/-3
  - `vllm/model_executor/models/qwen3_5.py` modified +2/-0
- Code diff details:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-1
  - `examples/generate/multimodal/vision_language_offline.py` modified +93/-1
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +15/-3
  - `vllm/model_executor/models/qwen3_5.py` modified +2/-0
- Key code excerpts:

```diff
diff -- docs/design/cuda_graphs_multimodal.md
@@ -85,8 +85,9 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra

 | Architecture | Models | CG for Image | CG for Video |
 | ------------ | ------ | ------------ | ------------ |
-| `Qwen3VLForConditionalGeneration` | `Qwen3-VL` | ✅︎ | ✅︎ |
 | `Qwen2_5_VLForConditionalGeneration` | `Qwen2.5-VL` | ✅︎ | ✅︎ |
+| `Qwen3VLForConditionalGeneration` | `Qwen3-VL` | ✅︎ | ✅︎ |
+| `Qwen3_5ForConditionalGeneration` | `Qwen3.5` | ✅︎ | ✅︎ |

 !!! note
     Encoder CUDA Graphs have currently been tested with `--mm-encoder-attn-backend=FLASH_ATTN` and `--mm-encoder-attn-backend=FLASHINFER` on Blackwell GPUs.
diff -- examples/generate/multimodal/vision_language_offline.py
@@ -2179,6 +2179,92 @@ def run_qwen3_vl_moe(questions: list[str], modality: str) -> ModelRequestData:
     )


+# Qwen3.5-Dense
+def run_qwen3_5(questions: list[str], modality: str) -> ModelRequestData:
+    model_name = "Qwen/Qwen3.5-4B"
+
+    mm_limit = {"image": 1, "video": 1} if modality == "image+video" else {modality: 1}
+    engine_args = EngineArgs(
+        model=model_name,
+        max_model_len=4096,
+        max_num_seqs=5,
+        mm_processor_kwargs={
```
- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +2/-0
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +15/-3
  - docs/bench: `docs/design/cuda_graphs_multimodal.md` modified +2/-1; `examples/generate/multimodal/vision_language_offline.py` modified +93/-1
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #42224 - [MM][CG] Enable encoder Cudagraph for Step3VL

- Link: https://github.com/vllm-project/vllm/pull/42224
- Status/date: merged / 2026-05-18
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 8 files, +384/-22, with 534 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][CG] Enable encoder Cudagraph for Step3VL"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `docs/design/cuda_graphs_multimodal.md`, `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`.
- Key implementation:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-0
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0; symbols: step3_vl_chat_template
  - `vllm/model_executor/models/interfaces.py` modified +21/-0; symbols: postprocess_encoder_output
- Code diff details:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-0
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0
  - `vllm/model_executor/models/interfaces.py` modified +21/-0
- Key code excerpts:

```diff
diff -- docs/design/cuda_graphs_multimodal.md
@@ -77,6 +77,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra
 * `encoder_eager_forward(...)` — fallback eager forward when no graph fits.
 * `get_input_modality(...)` - return the modality of the inputs.
 * `get_max_frames_per_video()` - return model-specific max frames per video.
+* `postprocess_encoder_output(...)` - post process encoder output, directly call scatter_output_slices by default

 !!! note
     The `SupportsEncoderCudaGraph` protocol is designed to be model-agnostic. New vision encoder models can opt-in by implementing the protocol methods without modifying the manager.
@@ -89,6 +90,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra
 | `Qwen2_5_VLForConditionalGeneration` | `Qwen2.5-VL` | ✅︎ | ✅︎ |
 | `Qwen3VLForConditionalGeneration` | `Qwen3-VL` | ✅︎ | ✅︎ |
 | `Qwen3_5ForConditionalGeneration` | `Qwen3.5` | ✅︎ | ✅︎ |
+| `Step3VLForConditionalGeneration` | `Step3-VL` | ✅︎ | ❌︎ |

diff -- examples/generate/multimodal/vision_language_offline.py
@@ -2560,6 +2560,7 @@ def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:
     "qwen2_vl",
     "qwen3_5",
     "qwen3_5_moe",
+    "stepvl",
 ]


```
- Reviewed files:
  - runtime: `vllm/model_executor/models/interfaces.py` modified +21/-0; `vllm/model_executor/models/step3_vl.py` modified +323/-2; `vllm/model_executor/models/step_vl.py` modified +1/-0; `vllm/model_executor/models/utils.py` modified +16/-0; `vllm/v1/worker/encoder_cudagraph.py` modified +8/-20
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0
  - docs/bench: `docs/design/cuda_graphs_multimodal.md` modified +2/-0; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.
