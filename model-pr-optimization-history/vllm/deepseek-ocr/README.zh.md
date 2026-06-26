# vllm DeepSeek OCR 模型 PR 优化历史

## 2026-06-26 最新源码扫描

已按 vLLM 上游 `vllm-project/vllm@abc71548ef029132c3316b902207f254a246d593` 重新扫描本文下方列出的 tracked files。
文件级匹配使用 GitHub mirror 的 `git log --name-only`；PR 标题、链接和合并时间通过 GitHub GraphQL Pull Request API 批量复核。上一时效锚点：`2026-06-05`。

结果：发现 6 个额外 PR-numbered merge 触及 tracked files，但尚未提升为下方完整逐 PR diff audit card。此节只作为 freshness index；需要引用实现细节时，仍应先人工阅读 PR diff 再补完整卡片。

| 合并日期 | PR | 标题 | 命中的 tracked files |
| --- | --- | --- | --- |
| 2026-06-22 | [#45993](https://github.com/vllm-project/vllm/pull/45993) | [Model] Remove MiniMaxText01, MiniMaxVL01, MiniMaxForCausalLM | `vision_language_offline.py` |
| 2026-06-17 | [#41992](https://github.com/vllm-project/vllm/pull/41992) | [MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL | `vision_language_offline.py` |
| 2026-06-16 | [#43586](https://github.com/vllm-project/vllm/pull/43586) | [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR | `vision_language_offline.py`, `deepseek_ocr.py` |
| 2026-06-12 | [#40660](https://github.com/vllm-project/vllm/pull/40660) | [MM][Perf][CG] Support ViT full cudagraphs for mllama4 | `vision_language_offline.py` |
| 2026-06-10 | [#45131](https://github.com/vllm-project/vllm/pull/45131) | Deprecated 1st generation Qwen and QwenVL models | `vision_language_multi_image_offline.py`, `vision_language_offline.py` |
| 2026-06-09 | [#40576](https://github.com/vllm-project/vllm/pull/40576) | [MM][Perf][CG] Support ViT full CUDA graph for glm4_1v image and video inference | `vision_language_offline.py` |

## 2026-06-05 PR 补漏复核

已于 2026-06-05 按 vllm 上游 `origin/main@c66b19800` 复核；自上次时效基准（2026-05-19）以来，共有 1 个带 PR 编号的合并改动到所跟踪的实现文件，这些 PR 尚未并入下方时间线 / 逐 PR diff 审计卡，应在下次完整重生成时补齐。

| 合并日期 | PR | 标题 | 改动到的跟踪文件 |
| --- | --- | --- | --- |
| 2026-06-04 | [#41759](https://github.com/vllm-project/vllm/pull/41759) | [MM][Perf][CG] Support ViT full CUDA graph for InternVL | `vision_language_offline.py` |


## 2026-05-19 新增覆盖

按 vllm 上游 `origin/main@ef54a4d604`、模型相关文件的 `git log --name-only -- <model-files>` 以及 GitHub Pull Request files API 生成。本页用于补齐 sgl-cookbook 中 `DeepSeek OCR` 缺失的历史 PR 优化文档。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `vllm/model_executor/models/deepseek_ocr.py` | [#35182](https://github.com/vllm-project/vllm/pull/35182), [#37289](https://github.com/vllm-project/vllm/pull/37289), [#36670](https://github.com/vllm-project/vllm/pull/36670), [#35025](https://github.com/vllm-project/vllm/pull/35025), [#34085](https://github.com/vllm-project/vllm/pull/34085), [#34330](https://github.com/vllm-project/vllm/pull/34330), [#33909](https://github.com/vllm-project/vllm/pull/33909), [#33063](https://github.com/vllm-project/vllm/pull/33063), [#31972](https://github.com/vllm-project/vllm/pull/31972), [#32632](https://github.com/vllm-project/vllm/pull/32632), [#32327](https://github.com/vllm-project/vllm/pull/32327), [#32016](https://github.com/vllm-project/vllm/pull/32016), ... (25 total) |
| `vllm/transformers_utils/processors/deepseek_ocr.py` | [#36024](https://github.com/vllm-project/vllm/pull/36024), [#33909](https://github.com/vllm-project/vllm/pull/33909), [#27361](https://github.com/vllm-project/vllm/pull/27361), [#27247](https://github.com/vllm-project/vllm/pull/27247) |
| `vllm/transformers_utils/chat_templates/template_deepseek_ocr.jinja` | [#27247](https://github.com/vllm-project/vllm/pull/27247) |
| `tests/models/multimodal/processing/test_deepseek_ocr.py` | [#36670](https://github.com/vllm-project/vllm/pull/36670) |
| `examples/generate/multimodal/vision_language_offline.py` | [#42224](https://github.com/vllm-project/vllm/pull/42224), [#41736](https://github.com/vllm-project/vllm/pull/41736), [#42151](https://github.com/vllm-project/vllm/pull/42151), [#40830](https://github.com/vllm-project/vllm/pull/40830), [#36464](https://github.com/vllm-project/vllm/pull/36464) |
| `examples/generate/multimodal/vision_language_multi_image_offline.py` | [#36464](https://github.com/vllm-project/vllm/pull/36464) |

## PR 覆盖总览

- git 追溯 PR 数: 31
- 关键词/补充 PR 数: 0
- 当前文档总 PR 数: 31
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
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

## 逐 PR diff 审计卡

### PR #27247 - [Model] Upstream Deepseek-OCR model

- 链接: https://github.com/vllm-project/vllm/pull/27247
- 状态/时间: merged / 2025-10-22
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+1821/-40，可读 patch 1953 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Upstream Deepseek-OCR model」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `docs/models/supported_models.md`, `examples/offline_inference/vision_language.py`, `tests/models/registry.py`。
- 实现要点:
  - `docs/models/supported_models.md` modified +1/-0
  - `examples/offline_inference/vision_language.py` modified +69/-20；symbols: run_deepseek_ocr, run_dots_ocr
  - `tests/models/registry.py` modified +3/-0
  - `vllm/model_executor/models/deepencoder.py` added +673/-0；symbols: MLPBlock, __init__, forward, LayerNorm2d
- 代码 diff 细节:
  - `docs/models/supported_models.md` modified +1/-0
  - `examples/offline_inference/vision_language.py` modified +69/-20
  - `tests/models/registry.py` modified +3/-0
  - `vllm/model_executor/models/deepencoder.py` added +673/-0
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/deepencoder.py` added +673/-0; `vllm/model_executor/models/deepseek_ocr.py` added +594/-0; `vllm/model_executor/models/deepseek_vl2.py` modified +23/-20; `vllm/model_executor/models/registry.py` modified +1/-0; `vllm/transformers_utils/chat_templates/registry.py` modified +1/-0; `vllm/transformers_utils/chat_templates/template_deepseek_ocr.jinja` added +14/-0; `vllm/transformers_utils/processors/deepseek_ocr.py` added +442/-0
  - tests: `tests/models/registry.py` modified +3/-0
  - docs/bench: `docs/models/supported_models.md` modified +1/-0; `examples/offline_inference/vision_language.py` modified +69/-20
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #27361 - [Bugfix] Fix deepseek-ocr multi-image inference and add `merge_by_field_config=True` with tensor schema support

- 链接: https://github.com/vllm-project/vllm/pull/27361
- 状态/时间: merged / 2025-10-23
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+112/-66，可读 patch 306 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix deepseek-ocr multi-image inference and add `merge_by_field_config=True` with tensor schema support」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `examples/offline_inference/vision_language_multi_image.py`, `tests/models/multimodal/processing/test_common.py`, `vllm/model_executor/models/deepseek_ocr.py`。
- 实现要点:
  - `examples/offline_inference/vision_language_multi_image.py` modified +48/-2；symbols: load_deepseek_ocr
  - `tests/models/multimodal/processing/test_common.py` modified +1/-0
  - `vllm/model_executor/models/deepseek_ocr.py` modified +58/-55；symbols: DeepseekOCRImagePixelInputs, _parse_and_validate_image_input, _process_image_input
  - `vllm/transformers_utils/processors/deepseek_ocr.py` modified +5/-9
- 代码 diff 细节:
  - `examples/offline_inference/vision_language_multi_image.py` modified +48/-2
  - `tests/models/multimodal/processing/test_common.py` modified +1/-0
  - `vllm/model_executor/models/deepseek_ocr.py` modified +58/-55
  - `vllm/transformers_utils/processors/deepseek_ocr.py` modified +5/-9
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +58/-55; `vllm/transformers_utils/processors/deepseek_ocr.py` modified +5/-9
  - tests: `tests/models/multimodal/processing/test_common.py` modified +1/-0
  - docs/bench: `examples/offline_inference/vision_language_multi_image.py` modified +48/-2
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #27560 - [Bugfix] Validate custom logits processor xargs for online serving

- 链接: https://github.com/vllm-project/vllm/pull/27560
- 状态/时间: merged / 2025-11-05
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+232/-49，可读 patch 574 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Validate custom logits processor xargs for online serving」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `docs/design/logits_processors.md`, `docs/features/custom_arguments.md`, `docs/features/custom_logitsprocs.md`。
- 实现要点:
  - `docs/design/logits_processors.md` modified +13/-1；symbols: validate_params
  - `docs/features/custom_arguments.md` modified +3/-0
  - `docs/features/custom_logitsprocs.md` modified +33/-9；symbols: validate_params
  - `examples/offline_inference/logits_processor/custom.py` modified +17/-2；symbols: validate_params, extract_extra_arg
- 代码 diff 细节:
  - `docs/design/logits_processors.md` modified +13/-1
  - `docs/features/custom_arguments.md` modified +3/-0
  - `docs/features/custom_logitsprocs.md` modified +33/-9
  - `examples/offline_inference/logits_processor/custom.py` modified +17/-2
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/entrypoints/openai/protocol.py` modified +2/-2; `vllm/entrypoints/openai/serving_chat.py` modified +8/-0; `vllm/entrypoints/openai/serving_completion.py` modified +9/-0; `vllm/model_executor/models/deepseek_ocr.py` modified +23/-17; `vllm/transformers_utils/configs/deepseek_vl2.py` modified +6/-0; `vllm/utils/torch_utils.py` modified +28/-0; `vllm/v1/sample/logits_processor/__init__.py` modified +20/-2; `vllm/v1/sample/logits_processor/interface.py` modified +8/-0
  - tests: `tests/entrypoints/openai/test_lora_resolvers.py` modified +1/-0; `tests/entrypoints/openai/test_serving_chat.py` modified +1/-0; `tests/v1/logits_processors/test_custom_online.py` modified +29/-0; `tests/v1/logits_processors/utils.py` modified +15/-2
  - docs/bench: `docs/design/logits_processors.md` modified +13/-1; `docs/features/custom_arguments.md` modified +3/-0; `docs/features/custom_logitsprocs.md` modified +33/-9; `examples/offline_inference/logits_processor/custom.py` modified +17/-2; `examples/offline_inference/logits_processor/custom_req.py` modified +8/-7; `examples/offline_inference/logits_processor/custom_req_init.py` modified +8/-7
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #28101 - [Model] Consolidate Deepseek-MoE implementation with DeepSeek-v2

- 链接: https://github.com/vllm-project/vllm/pull/28101
- 状态/时间: merged / 2025-11-08
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+144/-548，可读 patch 825 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Consolidate Deepseek-MoE implementation with DeepSeek-v2」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `tests/models/registry.py`, `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/models/deepseek_ocr.py`。
- 实现要点:
  - `tests/models/registry.py` modified +4/-1
  - `vllm/model_executor/models/deepseek.py` removed +0/-517
  - `vllm/model_executor/models/deepseek_ocr.py` modified +0/-8
  - `vllm/model_executor/models/deepseek_v2.py` modified +139/-13；symbols: DeepseekAttention, __init__, forward, DeepseekForCausalLM
- 代码 diff 细节:
  - `tests/models/registry.py` modified +4/-1
  - `vllm/model_executor/models/deepseek.py` removed +0/-517
  - `vllm/model_executor/models/deepseek_ocr.py` modified +0/-8
  - `vllm/model_executor/models/deepseek_v2.py` modified +139/-13
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek.py` removed +0/-517; `vllm/model_executor/models/deepseek_ocr.py` modified +0/-8; `vllm/model_executor/models/deepseek_v2.py` modified +139/-13; `vllm/model_executor/models/deepseek_vl2.py` modified +0/-8; `vllm/model_executor/models/registry.py` modified +1/-1
  - tests: `tests/models/registry.py` modified +4/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #27583 - Rename clashing method names for vLLM model protocol

- 链接: https://github.com/vllm-project/vllm/pull/27583
- 状态/时间: merged / 2025-11-13
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+368/-367，可读 patch 2559 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Rename clashing method names for vLLM model protocol」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `docs/contributing/model/basic.md`, `docs/contributing/model/multimodal.md`, `vllm/model_executor/models/apertus.py`。
- 实现要点:
  - `docs/contributing/model/basic.md` modified +2/-2；symbols: embed_input_ids
  - `docs/contributing/model/multimodal.md` modified +3/-3；symbols: embed_multimodal
  - `vllm/model_executor/models/apertus.py` modified +4/-4；symbols: embed_input_ids
  - `vllm/model_executor/models/arcee.py` modified +4/-4；symbols: embed_input_ids
- 代码 diff 细节:
  - `docs/contributing/model/basic.md` modified +2/-2
  - `docs/contributing/model/multimodal.md` modified +3/-3
  - `vllm/model_executor/models/apertus.py` modified +4/-4
  - `vllm/model_executor/models/arcee.py` modified +4/-4
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/apertus.py` modified +4/-4; `vllm/model_executor/models/arcee.py` modified +4/-4; `vllm/model_executor/models/arctic.py` modified +4/-4; `vllm/model_executor/models/aria.py` modified +3/-3; `vllm/model_executor/models/aya_vision.py` modified +1/-1; `vllm/model_executor/models/baichuan.py` modified +4/-4; `vllm/model_executor/models/bailing_moe.py` modified +4/-4; `vllm/model_executor/models/bamba.py` modified +4/-4
  - docs/bench: `docs/contributing/model/basic.md` modified +2/-2; `docs/contributing/model/multimodal.md` modified +3/-3
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #28617 - [BugFix] DeepSeek-OCR: apply NoRepeatNGramLogitsProcessor to greedy path

- 链接: https://github.com/vllm-project/vllm/pull/28617
- 状态/时间: merged / 2025-11-13
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] DeepSeek-OCR: apply NoRepeatNGramLogitsProcessor to greedy path」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_ocr.py`。
- 实现要点:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-1
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-1
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +1/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #29793 - [Chore] Move tokenizer initialization methods

- 链接: https://github.com/vllm-project/vllm/pull/29793
- 状态/时间: merged / 2025-12-02
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 51 个文件，+150/-129，可读 patch 761 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Chore] Move tokenizer initialization methods」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `benchmarks/benchmark_prefix_caching.py`, `benchmarks/benchmark_serving_structured_output.py`, `tests/compile/test_dynamic_shapes_compilation.py`。
- 实现要点:
  - `benchmarks/benchmark_prefix_caching.py` modified +1/-1
  - `benchmarks/benchmark_serving_structured_output.py` modified +1/-1
  - `tests/compile/test_dynamic_shapes_compilation.py` modified +1/-1
  - `tests/entrypoints/openai/test_chat_template.py` modified +1/-1
- 代码 diff 细节:
  - `benchmarks/benchmark_prefix_caching.py` modified +1/-1
  - `benchmarks/benchmark_serving_structured_output.py` modified +1/-1
  - `tests/compile/test_dynamic_shapes_compilation.py` modified +1/-1
  - `tests/entrypoints/openai/test_chat_template.py` modified +1/-1
- 关键代码摘录:

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
- 已读文件:
  - tests: `tests/compile/test_dynamic_shapes_compilation.py` modified +1/-1; `tests/entrypoints/openai/test_chat_template.py` modified +1/-1; `tests/entrypoints/openai/test_lora_resolvers.py` modified +1/-1; `tests/entrypoints/openai/test_return_token_ids.py` modified +1/-1; `tests/entrypoints/openai/test_return_tokens_as_ids.py` modified +1/-1; `tests/entrypoints/openai/test_serving_chat.py` modified +1/-1; `tests/entrypoints/openai/test_token_in_token_out.py` modified +1/-1; `tests/entrypoints/openai/test_tokenization.py` modified +1/-1
  - docs/bench: `benchmarks/benchmark_prefix_caching.py` modified +1/-1; `benchmarks/benchmark_serving_structured_output.py` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #30035 - [Chore] Deprecate `merge_by_field_config` arg

- 链接: https://github.com/vllm-project/vllm/pull/30035
- 状态/时间: merged / 2025-12-04
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+90/-302，可读 patch 728 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Chore] Deprecate `merge_by_field_config` arg」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_glm4_1v.py`, `tests/models/multimodal/processing/test_tensor_schema.py`。
- 实现要点:
  - `tests/models/multimodal/processing/test_common.py` modified +2/-2
  - `tests/models/multimodal/processing/test_glm4_1v.py` modified +4/-3
  - `tests/models/multimodal/processing/test_tensor_schema.py` modified +2/-3
  - `tests/multimodal/test_cache.py` modified +3/-6
- 代码 diff 细节:
  - `tests/models/multimodal/processing/test_common.py` modified +2/-2
  - `tests/models/multimodal/processing/test_glm4_1v.py` modified +4/-3
  - `tests/models/multimodal/processing/test_tensor_schema.py` modified +2/-3
  - `tests/multimodal/test_cache.py` modified +3/-6
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +2/-2; `vllm/model_executor/models/interfaces.py` modified +1/-1; `vllm/model_executor/models/lightonocr.py` modified +2/-2; `vllm/model_executor/models/nano_nemotron_vl.py` modified +6/-6; `vllm/model_executor/models/opencua.py` modified +2/-2; `vllm/model_executor/models/paddleocr_vl.py` modified +2/-2; `vllm/model_executor/models/paligemma.py` modified +3/-5; `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-2
  - tests: `tests/models/multimodal/processing/test_common.py` modified +2/-2; `tests/models/multimodal/processing/test_glm4_1v.py` modified +4/-3; `tests/models/multimodal/processing/test_tensor_schema.py` modified +2/-3; `tests/multimodal/test_cache.py` modified +3/-6; `tests/multimodal/test_inputs.py` removed +0/-91
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #30170 - [Chore] Deprecate `SupportsMultiModal.merge_by_field_config`

- 链接: https://github.com/vllm-project/vllm/pull/30170
- 状态/时间: merged / 2025-12-06
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 61 个文件，+23/-110，可读 patch 568 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Chore] Deprecate `SupportsMultiModal.merge_by_field_config`」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/aya_vision.py`, `vllm/model_executor/models/blip2.py`。
- 实现要点:
  - `vllm/model_executor/models/aria.py` modified +0/-2
  - `vllm/model_executor/models/aya_vision.py` modified +0/-2
  - `vllm/model_executor/models/blip2.py` modified +0/-2
  - `vllm/model_executor/models/chameleon.py` modified +0/-2
- 代码 diff 细节:
  - `vllm/model_executor/models/aria.py` modified +0/-2
  - `vllm/model_executor/models/aya_vision.py` modified +0/-2
  - `vllm/model_executor/models/blip2.py` modified +0/-2
  - `vllm/model_executor/models/chameleon.py` modified +0/-2
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/aria.py` modified +0/-2; `vllm/model_executor/models/aya_vision.py` modified +0/-2; `vllm/model_executor/models/blip2.py` modified +0/-2; `vllm/model_executor/models/chameleon.py` modified +0/-2; `vllm/model_executor/models/clip.py` modified +0/-1; `vllm/model_executor/models/cohere2_vision.py` modified +0/-2; `vllm/model_executor/models/deepseek_ocr.py` modified +0/-2; `vllm/model_executor/models/deepseek_vl2.py` modified +0/-2
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #30145 - [Renderer] Separate out `RendererConfig` from `ModelConfig`

- 链接: https://github.com/vllm-project/vllm/pull/30145
- 状态/时间: merged / 2025-12-07
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+944/-773，可读 patch 4691 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Renderer] Separate out `RendererConfig` from `ModelConfig`」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `docs/contributing/model/transcription.md`, `tests/compile/distributed/test_sequence_parallelism.py`, `tests/compile/test_functionalization.py`。
- 实现要点:
  - `docs/contributing/model/transcription.md` modified +6/-6
  - `tests/compile/distributed/test_sequence_parallelism.py` modified +2/-0
  - `tests/compile/test_functionalization.py` modified +5/-1
  - `tests/compile/test_fusion.py` modified +5/-1
- 代码 diff 细节:
  - `docs/contributing/model/transcription.md` modified +6/-6
  - `tests/compile/distributed/test_sequence_parallelism.py` modified +2/-0
  - `tests/compile/test_functionalization.py` modified +5/-1
  - `tests/compile/test_fusion.py` modified +5/-1
- 关键代码摘录:

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
- 已读文件:
  - tests: `tests/compile/distributed/test_sequence_parallelism.py` modified +2/-0; `tests/compile/test_functionalization.py` modified +5/-1; `tests/compile/test_fusion.py` modified +5/-1; `tests/compile/test_fusion_attn.py` modified +2/-0; `tests/compile/test_pass_manager.py` modified +6/-2; `tests/compile/test_qk_norm_rope_fusion.py` modified +4/-1; `tests/distributed/test_kvlayout.py` modified +3/-0; `tests/entrypoints/openai/test_chat_template.py` modified +4/-18
  - docs/bench: `docs/contributing/model/transcription.md` modified +6/-6
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #30199 - Revert "[Renderer] Separate out `RendererConfig` from `ModelConfig` (#30145)"

- 链接: https://github.com/vllm-project/vllm/pull/30199
- 状态/时间: merged / 2025-12-07
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+773/-944，可读 patch 4691 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "[Renderer] Separate out `RendererConfig` from `ModelConfig` (#30145)"」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `docs/contributing/model/transcription.md`, `tests/compile/distributed/test_sequence_parallelism.py`, `tests/compile/test_functionalization.py`。
- 实现要点:
  - `docs/contributing/model/transcription.md` modified +6/-6
  - `tests/compile/distributed/test_sequence_parallelism.py` modified +0/-2
  - `tests/compile/test_functionalization.py` modified +1/-5
  - `tests/compile/test_fusion.py` modified +1/-5
- 代码 diff 细节:
  - `docs/contributing/model/transcription.md` modified +6/-6
  - `tests/compile/distributed/test_sequence_parallelism.py` modified +0/-2
  - `tests/compile/test_functionalization.py` modified +1/-5
  - `tests/compile/test_fusion.py` modified +1/-5
- 关键代码摘录:

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
- 已读文件:
  - tests: `tests/compile/distributed/test_sequence_parallelism.py` modified +0/-2; `tests/compile/test_functionalization.py` modified +1/-5; `tests/compile/test_fusion.py` modified +1/-5; `tests/compile/test_fusion_attn.py` modified +0/-2; `tests/compile/test_pass_manager.py` modified +2/-6; `tests/compile/test_qk_norm_rope_fusion.py` modified +1/-4; `tests/distributed/test_kvlayout.py` modified +0/-3; `tests/entrypoints/openai/test_chat_template.py` modified +18/-4
  - docs/bench: `docs/contributing/model/transcription.md` modified +6/-6
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #31569 - feat: support LoRA for DeepSeek-OCR(Language Model part)

- 链接: https://github.com/vllm-project/vllm/pull/31569
- 状态/时间: merged / 2026-01-02
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-2，可读 patch 44 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: support LoRA for DeepSeek-OCR(Language Model part)」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `docs/models/supported_models.md`, `vllm/model_executor/models/deepseek_ocr.py`。
- 实现要点:
  - `docs/models/supported_models.md` modified +1/-1
  - `vllm/model_executor/models/deepseek_ocr.py` modified +13/-1；symbols: DeepseekOCRForCausalLM, get_mm_mapping
- 代码 diff 细节:
  - `docs/models/supported_models.md` modified +1/-1
  - `vllm/model_executor/models/deepseek_ocr.py` modified +13/-1
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +13/-1
  - docs/bench: `docs/models/supported_models.md` modified +1/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #31947 - [Model] Standardize common vision encoders

- 链接: https://github.com/vllm-project/vllm/pull/31947
- 状态/时间: merged / 2026-01-08
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+254/-174，可读 patch 1287 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Standardize common vision encoders」；模型线: DeepSeek OCR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/clip.py`, `vllm/model_executor/models/deepencoder.py`, `vllm/model_executor/models/deepseek_ocr.py`。
- 实现要点:
  - `vllm/model_executor/models/clip.py` modified +57/-12
  - `vllm/model_executor/models/deepencoder.py` modified +3/-0
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-0
  - `vllm/model_executor/models/hyperclovax_vision.py` modified +9/-10；symbols: __init__
- 代码 diff 细节:
  - `vllm/model_executor/models/clip.py` modified +57/-12
  - `vllm/model_executor/models/deepencoder.py` modified +3/-0
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-0
  - `vllm/model_executor/models/hyperclovax_vision.py` modified +9/-10
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/clip.py` modified +57/-12; `vllm/model_executor/models/deepencoder.py` modified +3/-0; `vllm/model_executor/models/deepseek_ocr.py` modified +1/-0; `vllm/model_executor/models/hyperclovax_vision.py` modified +9/-10; `vllm/model_executor/models/isaac.py` modified +1/-1; `vllm/model_executor/models/keye.py` modified +2/-0; `vllm/model_executor/models/lightonocr.py` modified +3/-1; `vllm/model_executor/models/llava.py` modified +7/-2
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #32016 - [Model] Remove redundant None check in DeepSeekOCR image input processing

- 链接: https://github.com/vllm-project/vllm/pull/32016
- 状态/时间: merged / 2026-01-09
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-13，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Remove redundant None check in DeepSeekOCR image input processing」；模型线: DeepSeek OCR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepseek_ocr.py`。
- 实现要点:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +10/-13
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +10/-13
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +10/-13
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #32327 - [1/N] Reorganize multimodal processing code

- 链接: https://github.com/vllm-project/vllm/pull/32327
- 状态/时间: merged / 2026-01-14
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 76 个文件，+717/-670，可读 patch 2419 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[1/N] Reorganize multimodal processing code」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `docs/api/README.md`, `docs/contributing/model/multimodal.md`, `docs/design/mm_processing.md`。
- 实现要点:
  - `docs/api/README.md` modified +0/-4
  - `docs/contributing/model/multimodal.md` modified +4/-6
  - `docs/design/mm_processing.md` modified +1/-1
  - `tests/multimodal/test_processing.py` modified +2/-2
- 代码 diff 细节:
  - `docs/api/README.md` modified +0/-4
  - `docs/contributing/model/multimodal.md` modified +4/-6
  - `docs/design/mm_processing.md` modified +1/-1
  - `tests/multimodal/test_processing.py` modified +2/-2
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/benchmarks/mm_processor.py` modified +1/-1; `vllm/model_executor/models/aria.py` modified +1/-1; `vllm/model_executor/models/audioflamingo3.py` modified +1/-1; `vllm/model_executor/models/aya_vision.py` modified +6/-3; `vllm/model_executor/models/bagel.py` modified +1/-1; `vllm/model_executor/models/blip2.py` modified +1/-1; `vllm/model_executor/models/chameleon.py` modified +1/-1; `vllm/model_executor/models/clip.py` modified +1/-1
  - tests: `tests/multimodal/test_processing.py` modified +2/-2
  - docs/bench: `docs/api/README.md` modified +0/-4; `docs/contributing/model/multimodal.md` modified +4/-6; `docs/design/mm_processing.md` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #32632 - [1/N] Initialize MM components in context managers (A-D)

- 链接: https://github.com/vllm-project/vllm/pull/32632
- 状态/时间: merged / 2026-01-20
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+239/-267，可读 patch 721 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[1/N] Initialize MM components in context managers (A-D)」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/audioflamingo3.py`, `vllm/model_executor/models/aya_vision.py`。
- 实现要点:
  - `vllm/model_executor/models/aria.py` modified +21/-32；symbols: compute_logits
  - `vllm/model_executor/models/audioflamingo3.py` modified +13/-15
  - `vllm/model_executor/models/aya_vision.py` modified +17/-18
  - `vllm/model_executor/models/bagel.py` modified +36/-43
- 代码 diff 细节:
  - `vllm/model_executor/models/aria.py` modified +21/-32
  - `vllm/model_executor/models/audioflamingo3.py` modified +13/-15
  - `vllm/model_executor/models/aya_vision.py` modified +17/-18
  - `vllm/model_executor/models/bagel.py` modified +36/-43
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/aria.py` modified +21/-32; `vllm/model_executor/models/audioflamingo3.py` modified +13/-15; `vllm/model_executor/models/aya_vision.py` modified +17/-18; `vllm/model_executor/models/bagel.py` modified +36/-43; `vllm/model_executor/models/blip2.py` modified +24/-30; `vllm/model_executor/models/clip.py` modified +23/-24; `vllm/model_executor/models/cohere2_vision.py` modified +17/-18; `vllm/model_executor/models/deepseek_ocr.py` modified +40/-41
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #31972 - [Models]: Make Multimodal config implicit in ViT implementation

- 链接: https://github.com/vllm-project/vllm/pull/31972
- 状态/时间: merged / 2026-01-24
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 38 个文件，+118/-470，可读 patch 2781 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models]: Make Multimodal config implicit in ViT implementation」；模型线: DeepSeek OCR；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/attention/mm_encoder_attention.py`, `vllm/model_executor/models/clip.py`, `vllm/model_executor/models/deepencoder.py`。
- 实现要点:
  - `vllm/model_executor/layers/attention/mm_encoder_attention.py` modified +0/-9
  - `vllm/model_executor/models/clip.py` modified +4/-24
  - `vllm/model_executor/models/deepencoder.py` modified +0/-3
  - `vllm/model_executor/models/deepseek_ocr.py` modified +0/-1
- 代码 diff 细节:
  - `vllm/model_executor/layers/attention/mm_encoder_attention.py` modified +0/-9
  - `vllm/model_executor/models/clip.py` modified +4/-24
  - `vllm/model_executor/models/deepencoder.py` modified +0/-3
  - `vllm/model_executor/models/deepseek_ocr.py` modified +0/-1
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/layers/attention/mm_encoder_attention.py` modified +0/-9; `vllm/model_executor/models/clip.py` modified +4/-24; `vllm/model_executor/models/deepencoder.py` modified +0/-3; `vllm/model_executor/models/deepseek_ocr.py` modified +0/-1; `vllm/model_executor/models/dots_ocr.py` modified +5/-34; `vllm/model_executor/models/eagle2_5_vl.py` modified +0/-1; `vllm/model_executor/models/ernie45_vl.py` modified +1/-12; `vllm/model_executor/models/glm4_1v.py` modified +5/-30
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #33063 - [Chore] Update type annotation of `input_ids` in model forward

- 链接: https://github.com/vllm-project/vllm/pull/33063
- 状态/时间: merged / 2026-01-26
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+146/-143，可读 patch 1304 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Chore] Update type annotation of `input_ids` in model forward」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `docs/contributing/model/basic.md`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`, `vllm/model_executor/models/afmoe.py`。
- 实现要点:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +1/-1
  - `vllm/model_executor/models/afmoe.py` modified +2/-2
  - `vllm/model_executor/models/apertus.py` modified +1/-1
- 代码 diff 细节:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +1/-1
  - `vllm/model_executor/models/afmoe.py` modified +2/-2
  - `vllm/model_executor/models/apertus.py` modified +1/-1
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/afmoe.py` modified +2/-2; `vllm/model_executor/models/apertus.py` modified +1/-1; `vllm/model_executor/models/arcee.py` modified +1/-1; `vllm/model_executor/models/arctic.py` modified +2/-2; `vllm/model_executor/models/aria.py` modified +1/-1; `vllm/model_executor/models/audioflamingo3.py` modified +1/-1; `vllm/model_executor/models/aya_vision.py` modified +1/-1; `vllm/model_executor/models/bagel.py` modified +1/-1
  - tests: `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +1/-1
  - docs/bench: `docs/contributing/model/basic.md` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #33909 - [Models] Consolidate Deepseek-OCR2 processor

- 链接: https://github.com/vllm-project/vllm/pull/33909
- 状态/时间: merged / 2026-02-05
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+52/-336，可读 patch 480 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models] Consolidate Deepseek-OCR2 processor」；模型线: DeepSeek OCR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepencoder2.py`, `vllm/model_executor/models/deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr2.py`。
- 实现要点:
  - `vllm/model_executor/models/deepencoder2.py` modified +1/-1
  - `vllm/model_executor/models/deepseek_ocr.py` modified +10/-2
  - `vllm/model_executor/models/deepseek_ocr2.py` modified +12/-4
  - `vllm/transformers_utils/processors/deepseek_ocr.py` modified +29/-9
- 代码 diff 细节:
  - `vllm/model_executor/models/deepencoder2.py` modified +1/-1
  - `vllm/model_executor/models/deepseek_ocr.py` modified +10/-2
  - `vllm/model_executor/models/deepseek_ocr2.py` modified +12/-4
  - `vllm/transformers_utils/processors/deepseek_ocr.py` modified +29/-9
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/deepencoder2.py` modified +1/-1; `vllm/model_executor/models/deepseek_ocr.py` modified +10/-2; `vllm/model_executor/models/deepseek_ocr2.py` modified +12/-4; `vllm/transformers_utils/processors/deepseek_ocr.py` modified +29/-9; `vllm/transformers_utils/processors/deepseek_ocr2.py` removed +0/-320
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #34330 - [Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder`

- 链接: https://github.com/vllm-project/vllm/pull/34330
- 状态/时间: merged / 2026-02-11
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 72 个文件，+131/-27，可读 patch 784 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder`」；模型线: DeepSeek OCR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/audioflamingo3.py`, `vllm/model_executor/models/aya_vision.py`。
- 实现要点:
  - `vllm/model_executor/models/aria.py` modified +1/-0
  - `vllm/model_executor/models/audioflamingo3.py` modified +4/-1
  - `vllm/model_executor/models/aya_vision.py` modified +1/-0
  - `vllm/model_executor/models/bagel.py` modified +1/-0
- 代码 diff 细节:
  - `vllm/model_executor/models/aria.py` modified +1/-0
  - `vllm/model_executor/models/audioflamingo3.py` modified +4/-1
  - `vllm/model_executor/models/aya_vision.py` modified +1/-0
  - `vllm/model_executor/models/bagel.py` modified +1/-0
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/aria.py` modified +1/-0; `vllm/model_executor/models/audioflamingo3.py` modified +4/-1; `vllm/model_executor/models/aya_vision.py` modified +1/-0; `vllm/model_executor/models/bagel.py` modified +1/-0; `vllm/model_executor/models/bee.py` modified +1/-0; `vllm/model_executor/models/blip2.py` modified +1/-0; `vllm/model_executor/models/chameleon.py` modified +1/-0; `vllm/model_executor/models/clip.py` modified +1/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #34085 - Fix DeepSeek-OCR tensor validation for all size variants

- 链接: https://github.com/vllm-project/vllm/pull/34085
- 状态/时间: merged / 2026-02-12
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-1，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix DeepSeek-OCR tensor validation for all size variants」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_ocr.py`。
- 实现要点:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +11/-1
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +11/-1
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +11/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #35025 - [Refactor] Simplify dummy data generation

- 链接: https://github.com/vllm-project/vllm/pull/35025
- 状态/时间: merged / 2026-02-23
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 78 个文件，+282/-367，可读 patch 1791 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Simplify dummy data generation」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `docs/contributing/model/multimodal.md`, `tests/models/multimodal/processing/test_audioflamingo3.py`, `tests/models/multimodal/processing/test_common.py`。
- 实现要点:
  - `docs/contributing/model/multimodal.md` modified +11/-11
  - `tests/models/multimodal/processing/test_audioflamingo3.py` modified +1/-1
  - `tests/models/multimodal/processing/test_common.py` modified +2/-0
  - `tests/models/multimodal/processing/test_tensor_schema.py` modified +1/-0
- 代码 diff 细节:
  - `docs/contributing/model/multimodal.md` modified +11/-11
  - `tests/models/multimodal/processing/test_audioflamingo3.py` modified +1/-1
  - `tests/models/multimodal/processing/test_common.py` modified +2/-0
  - `tests/models/multimodal/processing/test_tensor_schema.py` modified +1/-0
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/config/multimodal.py` modified +36/-20; `vllm/model_executor/models/aria.py` modified +2/-3; `vllm/model_executor/models/audioflamingo3.py` modified +3/-6; `vllm/model_executor/models/aya_vision.py` modified +2/-3; `vllm/model_executor/models/bagel.py` modified +2/-3; `vllm/model_executor/models/bee.py` modified +2/-3; `vllm/model_executor/models/blip2.py` modified +2/-3; `vllm/model_executor/models/chameleon.py` modified +2/-3
  - tests: `tests/models/multimodal/processing/test_audioflamingo3.py` modified +1/-1; `tests/models/multimodal/processing/test_common.py` modified +2/-0; `tests/models/multimodal/processing/test_tensor_schema.py` modified +1/-0
  - docs/bench: `docs/contributing/model/multimodal.md` modified +11/-11
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #36024 - [Misc] Lazy import registered processors

- 链接: https://github.com/vllm-project/vllm/pull/36024
- 状态/时间: merged / 2026-03-06
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+68/-51，可读 patch 288 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Lazy import registered processors」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `tests/models/registry.py`, `vllm/model_executor/models/deepseek_vl2.py`, `vllm/model_executor/models/fireredasr2.py`。
- 实现要点:
  - `tests/models/registry.py` modified +2/-5
  - `vllm/model_executor/models/deepseek_vl2.py` modified +1/-2
  - `vllm/model_executor/models/fireredasr2.py` modified +1/-1
  - `vllm/model_executor/models/funasr.py` modified +1/-1
- 代码 diff 细节:
  - `tests/models/registry.py` modified +2/-5
  - `vllm/model_executor/models/deepseek_vl2.py` modified +1/-2
  - `vllm/model_executor/models/fireredasr2.py` modified +1/-1
  - `vllm/model_executor/models/funasr.py` modified +1/-1
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_vl2.py` modified +1/-2; `vllm/model_executor/models/fireredasr2.py` modified +1/-1; `vllm/model_executor/models/funasr.py` modified +1/-1; `vllm/transformers_utils/processor.py` modified +31/-1; `vllm/transformers_utils/processors/__init__.py` modified +28/-10; `vllm/transformers_utils/processors/bagel.py` modified +0/-4; `vllm/transformers_utils/processors/deepseek_ocr.py` modified +1/-4; `vllm/transformers_utils/processors/deepseek_vl2.py` modified +1/-4
  - tests: `tests/models/registry.py` modified +2/-5
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #36670 - [Bugfix][Model] Fix DeepSeek-OCR TensorSchema crash on empty images_crop

- 链接: https://github.com/vllm-project/vllm/pull/36670
- 状态/时间: merged / 2026-03-12
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+135/-4，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Model] Fix DeepSeek-OCR TensorSchema crash on empty images_crop」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `tests/models/multimodal/processing/test_deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr.py`。
- 实现要点:
  - `tests/models/multimodal/processing/test_deepseek_ocr.py` added +134/-0；symbols: processor, TestDeepseekOCREmptyImagesCrop, test_empty_images_crop_small_image, test_populated_images_crop_large_image
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-4
- 代码 diff 细节:
  - `tests/models/multimodal/processing/test_deepseek_ocr.py` added +134/-0
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-4
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +1/-4
  - tests: `tests/models/multimodal/processing/test_deepseek_ocr.py` added +134/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #37289 - [Bugfix] Standardize custom HF Processor init

- 链接: https://github.com/vllm-project/vllm/pull/37289
- 状态/时间: merged / 2026-03-17
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+39/-33，可读 patch 152 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Standardize custom HF Processor init」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr2.py`, `vllm/model_executor/models/glm4v.py`。
- 实现要点:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +3/-1
  - `vllm/model_executor/models/deepseek_ocr2.py` modified +3/-1
  - `vllm/model_executor/models/glm4v.py` modified +11/-3；symbols: get_image_processor, get_hf_processor
  - `vllm/model_executor/models/qwen_vl.py` modified +11/-3；symbols: get_image_processor, get_hf_processor
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +3/-1
  - `vllm/model_executor/models/deepseek_ocr2.py` modified +3/-1
  - `vllm/model_executor/models/glm4v.py` modified +11/-3
  - `vllm/model_executor/models/qwen_vl.py` modified +11/-3
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +3/-1; `vllm/model_executor/models/deepseek_ocr2.py` modified +3/-1; `vllm/model_executor/models/glm4v.py` modified +11/-3; `vllm/model_executor/models/qwen_vl.py` modified +11/-3; `vllm/tokenizers/qwen_vl.py` modified +4/-0; `vllm/transformers_utils/processors/glm4v.py` modified +2/-7; `vllm/transformers_utils/processors/qwen_vl.py` modified +5/-18
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #35182 - [Misc] Reorganize inputs

- 链接: https://github.com/vllm-project/vllm/pull/35182
- 状态/时间: merged / 2026-03-25
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+1037/-959，可读 patch 4452 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Reorganize inputs」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `docs/api/README.md`, `docs/contributing/model/transcription.md`, `docs/features/multimodal_inputs.md`。
- 实现要点:
  - `docs/api/README.md` modified +3/-12
  - `docs/contributing/model/transcription.md` modified +2/-2
  - `docs/features/multimodal_inputs.md` modified +1/-1
  - `examples/pooling/token_embed/jina_embeddings_v4_offline.py` modified +1/-1
- 代码 diff 细节:
  - `docs/api/README.md` modified +3/-12
  - `docs/contributing/model/transcription.md` modified +2/-2
  - `docs/features/multimodal_inputs.md` modified +1/-1
  - `examples/pooling/token_embed/jina_embeddings_v4_offline.py` modified +1/-1
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/beam_search.py` modified +18/-13; `vllm/benchmarks/datasets.py` modified +1/-1; `vllm/engine/protocol.py` modified +4/-4; `vllm/entrypoints/anthropic/serving.py` modified +4/-4; `vllm/entrypoints/chat_utils.py` modified +2/-1; `vllm/entrypoints/llm.py` modified +22/-22; `vllm/entrypoints/openai/chat_completion/serving.py` modified +11/-11; `vllm/entrypoints/openai/completion/serving.py` modified +15/-17
  - tests: `tests/entrypoints/openai/chat_completion/test_chat_error.py` modified +1/-1; `tests/entrypoints/openai/chat_completion/test_serving_chat.py` modified +8/-8; `tests/entrypoints/openai/responses/test_serving_responses.py` modified +5/-5; `tests/entrypoints/serve/render/test_launch_render.py` modified +0/-14; `tests/entrypoints/test_chat_utils.py` modified +1/-1; `tests/models/multimodal/generation/test_pixtral.py` modified +1/-1; `tests/models/multimodal/processing/test_common.py` modified +6/-8; `tests/plugins/bge_m3_sparse_plugin/bge_m3_sparse_processor/sparse_embeddings_processor.py` modified +2/-4
  - docs/bench: `docs/api/README.md` modified +3/-12; `docs/contributing/model/transcription.md` modified +2/-2; `docs/features/multimodal_inputs.md` modified +1/-1; `examples/pooling/token_embed/jina_embeddings_v4_offline.py` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #36464 - [Examples] Resettle generate examples.

- 链接: https://github.com/vllm-project/vllm/pull/36464
- 状态/时间: merged / 2026-04-27
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 36 个文件，+46/-50，可读 patch 267 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Examples] Resettle generate examples.」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `.buildkite/test-amd.yaml`, `.buildkite/test_areas/misc.yaml`, `.buildkite/test_areas/model_runner_v2.yaml`。
- 实现要点:
  - `.buildkite/test-amd.yaml` modified +14/-14
  - `.buildkite/test_areas/misc.yaml` modified +4/-4
  - `.buildkite/test_areas/model_runner_v2.yaml` modified +4/-4
  - `.buildkite/test_areas/models_basic.yaml` modified +5/-5
- 代码 diff 细节:
  - `.buildkite/test-amd.yaml` modified +14/-14
  - `.buildkite/test_areas/misc.yaml` modified +4/-4
  - `.buildkite/test_areas/model_runner_v2.yaml` modified +4/-4
  - `.buildkite/test_areas/models_basic.yaml` modified +5/-5
- 关键代码摘录:

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
- 已读文件:
  - docs/bench: `.buildkite/test-amd.yaml` modified +14/-14; `.buildkite/test_areas/misc.yaml` modified +4/-4; `.buildkite/test_areas/model_runner_v2.yaml` modified +4/-4; `.buildkite/test_areas/models_basic.yaml` modified +5/-5; `docs/features/multimodal_inputs.md` modified +7/-7; `docs/features/reasoning_outputs.md` modified +1/-1; `docs/serving/openai_compatible_server.md` modified +3/-3; `examples/generate/batched_chat_completions_online.py` renamed +0/-0
  - other: `.github/mergify.yml` modified +1/-5
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #40830 - [MM][CG] Support ViT CG for Qwen2.5-VL

- 链接: https://github.com/vllm-project/vllm/pull/40830
- 状态/时间: merged / 2026-05-02
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+539/-22，可读 patch 669 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][CG] Support ViT CG for Qwen2.5-VL」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `docs/design/cuda_graphs_multimodal.md`, `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py`。
- 实现要点:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-0
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
  - `tests/models/multimodal/generation/test_qwen2_5_vl.py` modified +95/-0；symbols: _window_attention_regression_image, _encoder_cudagraph_config, test_qwen2_5_vl_window_attention_image, test_qwen2_5_vl_window_attention_image_batch
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-1
- 代码 diff 细节:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-0
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
  - `tests/models/multimodal/generation/test_qwen2_5_vl.py` modified +95/-0
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-1
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +429/-21
  - tests: `tests/models/multimodal/generation/test_qwen2_5_vl.py` modified +95/-0; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-1
  - docs/bench: `docs/design/cuda_graphs_multimodal.md` modified +2/-0; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #41736 - [MM][CG] Support ViT CG for Qwen2-VL

- 链接: https://github.com/vllm-project/vllm/pull/41736
- 状态/时间: merged / 2026-05-13
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+315/-21，可读 patch 415 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][CG] Support ViT CG for Qwen2-VL」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `docs/design/cuda_graphs_multimodal.md`, `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`。
- 实现要点:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-1
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0
  - `vllm/model_executor/models/qwen2_vl.py` modified +300/-20；symbols: prepare_encoder_metadata, get_encoder_cudagraph_config, get_input_modality, get_max_frames_per_video
- 代码 diff 细节:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-1
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0
  - `vllm/model_executor/models/qwen2_vl.py` modified +300/-20
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +300/-20
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0
  - docs/bench: `docs/design/cuda_graphs_multimodal.md` modified +2/-1; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #42151 - [MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5

- 链接: https://github.com/vllm-project/vllm/pull/42151
- 状态/时间: merged / 2026-05-13
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+112/-5，可读 patch 187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5」；模型线: DeepSeek OCR；类别: 性能/后端优化；主要 diff: `docs/design/cuda_graphs_multimodal.md`, `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`。
- 实现要点:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-1
  - `examples/generate/multimodal/vision_language_offline.py` modified +93/-1；symbols: run_qwen3_5, run_qwen3_5_moe
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +15/-3
  - `vllm/model_executor/models/qwen3_5.py` modified +2/-0
- 代码 diff 细节:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-1
  - `examples/generate/multimodal/vision_language_offline.py` modified +93/-1
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +15/-3
  - `vllm/model_executor/models/qwen3_5.py` modified +2/-0
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +2/-0
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +15/-3
  - docs/bench: `docs/design/cuda_graphs_multimodal.md` modified +2/-1; `examples/generate/multimodal/vision_language_offline.py` modified +93/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #42224 - [MM][CG] Enable encoder Cudagraph for Step3VL

- 链接: https://github.com/vllm-project/vllm/pull/42224
- 状态/时间: merged / 2026-05-18
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+384/-22，可读 patch 534 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][CG] Enable encoder Cudagraph for Step3VL」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `docs/design/cuda_graphs_multimodal.md`, `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`。
- 实现要点:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-0
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0；symbols: step3_vl_chat_template
  - `vllm/model_executor/models/interfaces.py` modified +21/-0；symbols: postprocess_encoder_output
- 代码 diff 细节:
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-0
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0
  - `vllm/model_executor/models/interfaces.py` modified +21/-0
- 关键代码摘录:

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
- 已读文件:
  - runtime: `vllm/model_executor/models/interfaces.py` modified +21/-0; `vllm/model_executor/models/step3_vl.py` modified +323/-2; `vllm/model_executor/models/step_vl.py` modified +1/-0; `vllm/model_executor/models/utils.py` modified +16/-0; `vllm/v1/worker/encoder_cudagraph.py` modified +8/-20
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0
  - docs/bench: `docs/design/cuda_graphs_multimodal.md` modified +2/-0; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。
