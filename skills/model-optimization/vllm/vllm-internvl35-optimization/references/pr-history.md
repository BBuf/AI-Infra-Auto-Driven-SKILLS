# vllm InternVL 3.5 PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 1
- Rule: use this as the backing dossier for the skill, not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/models/multimodal/pooling/test_intern_vit.py` | no direct PR-number commit |
| `tests/models/multimodal/processing/test_internvl.py` | [#12553](https://github.com/vllm-project/vllm/pull/12553), [#37260](https://github.com/vllm-project/vllm/pull/37260) |
| `vllm/model_executor/models/intern_vit.py` | [#6514](https://github.com/vllm-project/vllm/pull/6514), [#7067](https://github.com/vllm-project/vllm/pull/7067), [#9528](https://github.com/vllm-project/vllm/pull/9528), [#23909](https://github.com/vllm-project/vllm/pull/23909), [#38049](https://github.com/vllm-project/vllm/pull/38049) |
| `vllm/model_executor/models/internvl.py` | [#6514](https://github.com/vllm-project/vllm/pull/6514), [#7067](https://github.com/vllm-project/vllm/pull/7067), [#7164](https://github.com/vllm-project/vllm/pull/7164), [#7860](https://github.com/vllm-project/vllm/pull/7860), [#8201](https://github.com/vllm-project/vllm/pull/8201), [#8250](https://github.com/vllm-project/vllm/pull/8250), [#8299](https://github.com/vllm-project/vllm/pull/8299), [#8375](https://github.com/vllm-project/vllm/pull/8375), [#8614](https://github.com/vllm-project/vllm/pull/8614), [#8946](https://github.com/vllm-project/vllm/pull/8946), [#9351](https://github.com/vllm-project/vllm/pull/9351), [#9528](https://github.com/vllm-project/vllm/pull/9528), ... (27 total) |
| `vllm/transformers_utils/processors/internvl.py` | [#37260](https://github.com/vllm-project/vllm/pull/37260), [#37324](https://github.com/vllm-project/vllm/pull/37324) |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2024-07-29 | [#6514](https://github.com/vllm-project/vllm/pull/6514) | merged | [Model] Initialize support for InternVL2 series models | `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/test_internvl.py` |
| 2024-08-03 | [#7067](https://github.com/vllm-project/vllm/pull/7067) | merged | [Model] Refactor and decouple weight loading logic for InternVL2 model | `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py` |
| 2024-08-07 | [#7164](https://github.com/vllm-project/vllm/pull/7164) | merged | [Bugfix] Fix input processor for InternVL2 model | `vllm/model_executor/models/internvl.py`, `tests/models/test_internvl.py` |
| 2024-09-05 | [#7860](https://github.com/vllm-project/vllm/pull/7860) | merged | Inclusion of InternVLChatModel In PP_SUPPORTED_MODELS(Pipeline Parallelism) | `vllm/model_executor/models/internvl.py` |
| 2024-09-07 | [#8201](https://github.com/vllm-project/vllm/pull/8201) | merged | [Model][VLM] Support multi-images inputs for InternVL2 models | `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py` |
| 2024-09-11 | [#8299](https://github.com/vllm-project/vllm/pull/8299) | merged | [Bugfix] Fix InternVL2 vision embeddings process with pipeline parallel | `vllm/model_executor/models/internvl.py` |
| 2024-09-12 | [#8375](https://github.com/vllm-project/vllm/pull/8375) | merged | [Bugfix] Fix InternVL2 inference with various num_patches | `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py` |
| 2024-09-25 | [#8250](https://github.com/vllm-project/vllm/pull/8250) | merged | [BugFix] Propagate 'trust_remote_code' setting in internvl and minicpmv | `vllm/model_executor/models/internvl.py` |
| 2024-09-25 | [#8614](https://github.com/vllm-project/vllm/pull/8614) | merged | [VLM][Bugfix] enable internvl running with num_scheduler_steps > 1 | `vllm/model_executor/models/internvl.py` |
| 2024-09-30 | [#8946](https://github.com/vllm-project/vllm/pull/8946) | merged | [Model] Expose InternVL2 max_dynamic_patch as a mm_processor_kwarg | `vllm/model_executor/models/internvl.py` |
| 2024-10-15 | [#9351](https://github.com/vllm-project/vllm/pull/9351) | merged | [Bugfix] Update InternVL input mapper to support image embeds | `vllm/model_executor/models/internvl.py` |
| 2024-10-22 | [#9528](https://github.com/vllm-project/vllm/pull/9528) | merged | [Model][VLM] Initialize support for Mono-InternVL model | `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/decoder_only/vision_language/test_internvl.py` |
| 2024-11-21 | [#10518](https://github.com/vllm-project/vllm/pull/10518) | merged | [Model] Expose `dynamic_image_size` as mm_processor_kwargs for InternVL2 models | `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py`, `vllm/model_executor/models/internvl.py` |
| 2024-12-13 | [#11165](https://github.com/vllm-project/vllm/pull/11165) | merged | [V1][VLM] Fix edge case bug for InternVL2 | `vllm/model_executor/models/internvl.py` |
| 2025-02-04 | [#12553](https://github.com/vllm-project/vllm/pull/12553) | merged | [VLM] Merged multi-modal processor for InternVL-based models | `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py` |
| 2025-03-13 | [#14738](https://github.com/vllm-project/vllm/pull/14738) | merged | [VLM] Support loading InternVideo2.5 models as original InternVLChatModel | `vllm/model_executor/models/internvl.py` |
| 2025-03-20 | [#15086](https://github.com/vllm-project/vllm/pull/15086) | merged | [Bugfix] Fix embedding assignment for InternVL-based models | `vllm/model_executor/models/internvl.py` |
| 2025-05-25 | [#18499](https://github.com/vllm-project/vllm/pull/18499) | merged | [VLM] Initialize video input support for InternVL models | `vllm/model_executor/models/internvl.py` |
| 2025-05-29 | [#18842](https://github.com/vllm-project/vllm/pull/18842) | merged | [LoRA] Add LoRA support for InternVL | `vllm/model_executor/models/internvl.py` |
| 2025-07-29 | [#21684](https://github.com/vllm-project/vllm/pull/21684) | merged | Migrate InternVLImageInputs and InternVLVideoInputs to TensorSchema | `vllm/model_executor/models/internvl.py` |
| 2025-08-26 | [#23658](https://github.com/vllm-project/vllm/pull/23658) | merged | [Model] Enable video support for InternVL3.5 models | `vllm/model_executor/models/internvl.py` |
| 2025-08-27 | [#23742](https://github.com/vllm-project/vllm/pull/23742) | merged | [Model] Enable native HF format InternVL support | `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`, `docs/models/supported_models.md` |
| 2025-09-10 | [#24519](https://github.com/vllm-project/vllm/pull/24519) | merged | [Model] Limit CPU threads for image transformations in InternVL to reduce cpu contention. | `vllm/model_executor/models/internvl.py` |
| 2025-09-18 | [#23909](https://github.com/vllm-project/vllm/pull/23909) | merged | [Model] enable data parallel for InternVL vision encoder | `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py` |
| 2025-10-03 | [#26153](https://github.com/vllm-project/vllm/pull/26153) | merged | [Model] Use `merge_by_field_config` for MM models (InternVL family) | `vllm/model_executor/models/internvl.py` |
| 2026-01-23 | [#32397](https://github.com/vllm-project/vllm/pull/32397) | merged | [Model] Enable LoRA support for internvl2 | `vllm/model_executor/models/internvl.py` |
| 2026-03-17 | [#37260](https://github.com/vllm-project/vllm/pull/37260) | merged | [1/2] Move InternVL-based processors | `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py` |
| 2026-03-18 | [#37324](https://github.com/vllm-project/vllm/pull/37324) | merged | [2/3] Refactor InternVL-based processors | `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py` |
| 2026-03-26 | [#38049](https://github.com/vllm-project/vllm/pull/38049) | merged | [Model] Add torch.compile support for InternVL vision encoder | `vllm/model_executor/models/intern_vit.py` |

## Per-PR Diff Audit Cards

### PR #6514 - [Model] Initialize support for InternVL2 series models

- Link: https://github.com/vllm-project/vllm/pull/6514
- Status/date: merged / 2024-07-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`; associated commits `7cbd9ec7a9bf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +1042/-6, 1164 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[Model] Initialize support for InternVL2 series models". The diff centers on `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/test_internvl.py`. PR body context: FILL IN THE PR DESCRIPTION HERE FIX #4393 FIX #6321 (*link existing issues this PR will resolve*) **This PR aims to add support for InternVL2 series models:** - [x] Port InternV...
- Key implementation: `vllm/model_executor/models/internvl.py` added +471/-0 (471 lines); hunks: -0,0 +1,471; symbols: InternVLImagePixelInputs, build_transform, find_closest_aspect_ratio, calculate_num_blocks, touching `InternVLImagePixelInputs, build_transform, find_closest_aspect_ratio`; `vllm/model_executor/models/intern_vit.py` added +270/-0 (270 lines); hunks: -0,0 +1,270; symbols: InternVisionEmbeddings, __init__, _get_pos_embed, forward, touching `InternVisionEmbeddings, __init__, _get_pos_embed`; `tests/models/test_internvl.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: InternVLProcessor, __init__, __call__, generate, touching `InternVLProcessor, __init__, __call__`; `vllm/transformers_utils/configs/internvl.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: InternVLChatConfig, __init__, touching `InternVLChatConfig, __init__`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` added +471/-0 (471 lines); hunks: -0,0 +1,471; symbols: InternVLImagePixelInputs, build_transform, find_closest_aspect_ratio, calculate_num_blocks
  - `vllm/model_executor/models/intern_vit.py` added +270/-0 (270 lines); hunks: -0,0 +1,270; symbols: InternVisionEmbeddings, __init__, _get_pos_embed, forward
  - `tests/models/test_internvl.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: InternVLProcessor, __init__, __call__, generate
  - `vllm/transformers_utils/configs/internvl.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: InternVLChatConfig, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -0,0 +1,471 @@
+# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
+# --------------------------------------------------------
+# InternVL
+# Copyright (c) 2023 OpenGVLab
+# Licensed under The MIT License [see LICENSE for details]
+# --------------------------------------------------------
diff -- vllm/model_executor/models/intern_vit.py
@@ -0,0 +1,270 @@
+# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
+# --------------------------------------------------------
+# InternVL
+# Copyright (c) 2023 OpenGVLab
+# Licensed under The MIT License [see LICENSE for details]
+# --------------------------------------------------------
diff -- tests/models/test_internvl.py
@@ -0,0 +1,201 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` added +471/-0; `vllm/model_executor/models/intern_vit.py` added +270/-0; `vllm/transformers_utils/configs/internvl.py` added +51/-0
  - tests: `tests/models/test_internvl.py` added +201/-0
- Risk and verification: The diff ships test coverage in `tests/models/test_internvl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7067 - [Model] Refactor and decouple weight loading logic for InternVL2 model

- Link: https://github.com/vllm-project/vllm/pull/7067
- Status/date: merged / 2024-08-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`; associated commits `0c25435daa0a`
- Diff scope read: GitHub Pull Request files API returned 2 files, +38/-55, 123 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR changes model-related implementation. Title: "[Model] Refactor and decouple weight loading logic for InternVL2 model". The diff centers on `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`. PR body context: FILL IN THE PR DESCRIPTION HERE - The `load_weights` method for `InternVLChatModel` is highly coupled with weight loading logic for `Qwen2`, `Llama` and `InternLM2`, which is di...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +28/-54 (82 lines); hunks: -4,6 +4,7; -414,58 +415,31 @@ def sample(; symbols: sample, load_weights, _filter_weights, touching `sample, load_weights, _filter_weights`; `vllm/model_executor/models/intern_vit.py` modified +10/-1 (11 lines); hunks: -4,7 +4,7; -16,6 +16,7; symbols: forward, load_weights, touching `forward, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +28/-54 (82 lines); hunks: -4,6 +4,7; -414,58 +415,31 @@ def sample(; symbols: sample, load_weights, _filter_weights
  - `vllm/model_executor/models/intern_vit.py` modified +10/-1 (11 lines); hunks: -4,7 +4,7; -16,6 +16,7; symbols: forward, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -4,6 +4,7 @@
+import itertools
@@ -414,58 +415,31 @@ def sample(
-    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-        stacked_params_mapping = [
-            # (param_name, shard_name, shard_id)
-            (".qkv_proj", ".q_proj", "q"),
diff -- vllm/model_executor/models/intern_vit.py
@@ -4,7 +4,7 @@
-from typing import Optional
+from typing import Iterable, Optional, Tuple
@@ -16,6 +16,7 @@
+from vllm.model_executor.model_loader.weight_utils import default_weight_loader
@@ -268,3 +269,11 @@ def forward(
+    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +28/-54; `vllm/model_executor/models/intern_vit.py` modified +10/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7164 - [Bugfix] Fix input processor for InternVL2 model

- Link: https://github.com/vllm-project/vllm/pull/7164
- Status/date: merged / 2024-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `b764547616e6`
- Diff scope read: GitHub Pull Request files API returned 2 files, +73/-34, 211 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix input processor for InternVL2 model". The diff centers on `vllm/model_executor/models/internvl.py`, `tests/models/test_internvl.py`. PR body context: FILL IN THE PR DESCRIPTION HERE FIX #7160 FIX #7272 - This PR also aims to make some small refactor to fix some hidden issues. ~~So I marked it as a draft.~~ - Since most of pro...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +54/-30 (84 lines); hunks: -38,9 +38,6; -84,11 +81,9 @@ def find_closest_aspect_ratio(aspect_ratio, target_ratios, wi...; symbols: InternVLImagePixelInputs, find_closest_aspect_ratio, calculate_num_blocks, touching `InternVLImagePixelInputs, find_closest_aspect_ratio, calculate_num_blocks`; `tests/models/test_internvl.py` modified +19/-4 (23 lines); hunks: -5,6 +5,7; -26,10 +27,15; symbols: __init__, __call__, touching `__init__, __call__`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +54/-30 (84 lines); hunks: -38,9 +38,6; -84,11 +81,9 @@ def find_closest_aspect_ratio(aspect_ratio, target_ratios, wi...; symbols: InternVLImagePixelInputs, find_closest_aspect_ratio, calculate_num_blocks
  - `tests/models/test_internvl.py` modified +19/-4 (23 lines); hunks: -5,6 +5,7; -26,10 +27,15; symbols: __init__, __call__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -38,9 +38,6 @@
-MAX_IMAGE_FEATURE_SIZE_WIDTH = 3000
-MAX_IMAGE_FEATURE_SIZE_HEIGHT = 500
@@ -84,11 +81,9 @@ def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
-def calculate_num_blocks(orig_width: int,
-                         orig_height: int,
-                         min_num=1,
diff -- tests/models/test_internvl.py
@@ -5,6 +5,7 @@
+from transformers import AutoConfig
@@ -26,10 +27,15 @@
+DOWNLOAD_PATTERN = ["*.json", "*.py", "*.safetensors", "*.txt", "*.model"]
-    snapshot_download("OpenGVLab/InternVL2-1B"),
-    snapshot_download("OpenGVLab/InternVL2-2B"),
-    # snapshot_download("OpenGVLab/InternVL2-4B"),  # broken
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +54/-30
  - tests: `tests/models/test_internvl.py` modified +19/-4
- Risk and verification: The diff ships test coverage in `tests/models/test_internvl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7860 - Inclusion of InternVLChatModel In PP_SUPPORTED_MODELS(Pipeline Parallelism)

- Link: https://github.com/vllm-project/vllm/pull/7860
- Status/date: merged / 2024-09-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `8685ba1a1ec0`
- Diff scope read: GitHub Pull Request files API returned 6 files, +90/-35, 266 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "Inclusion of InternVLChatModel In PP_SUPPORTED_MODELS(Pipeline Parallelism)". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: FILL IN THE PR DESCRIPTION HERE Hi Folks, This PR is completed based on the 7168. This @andoorve PR includes the changes needed for the Add remaining model PP support, This PR 7...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +3/-1 (4 lines); hunks: -341,6 +341,8 @@ def __init__(self,; -461,7 +463,7 @@ def forward(; symbols: __init__, pixel_shuffle, forward, touching `__init__, pixel_shuffle, forward`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +3/-1 (4 lines); hunks: -341,6 +341,8 @@ def __init__(self,; -461,7 +463,7 @@ def forward(; symbols: __init__, pixel_shuffle, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -341,6 +341,8 @@ def __init__(self,
+        self.make_empty_intermediate_tensors = (
+            self.language_model.make_empty_intermediate_tensors)
@@ -461,7 +463,7 @@ def forward(
-                                                  None,
+                                                  intermediate_tensors,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `tests/distributed/test_pipeline_parallel.py`, `tests/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8201 - [Model][VLM] Support multi-images inputs for InternVL2 models

- Link: https://github.com/vllm-project/vllm/pull/8201
- Status/date: merged / 2024-09-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `e807125936a9`
- Diff scope read: GitHub Pull Request files API returned 5 files, +199/-57, 482 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[Model][VLM] Support multi-images inputs for InternVL2 models". The diff centers on `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py`. PR body context: FILL IN THE PR DESCRIPTION HERE FIX #xxxx (*link existing issues this PR will resolve*) **TODO** - [x] Initialize multi-images inputs support for InternVL2 - [x] ~~Update `_plac...
- Key implementation: `tests/models/test_internvl.py` modified +73/-19 (92 lines); hunks: -1,5 +1,5; -9,7 +9,8; symbols: generate, run_test, __init__, __call__, touching `generate, run_test, __init__`; `vllm/model_executor/models/internvl.py` modified +46/-14 (60 lines); hunks: -5,6 +5,7; -26,6 +27,7; symbols: find_closest_aspect_ratio, calculate_num_blocks, dynamic_preprocess, input_processor_for_internvl, touching `find_closest_aspect_ratio, calculate_num_blocks, dynamic_preprocess`.
- Code diff details:
  - `tests/models/test_internvl.py` modified +73/-19 (92 lines); hunks: -1,5 +1,5; -9,7 +9,8; symbols: generate, run_test, __init__, __call__
  - `vllm/model_executor/models/internvl.py` modified +46/-14 (60 lines); hunks: -5,6 +5,7; -26,6 +27,7; symbols: find_closest_aspect_ratio, calculate_num_blocks, dynamic_preprocess, input_processor_for_internvl
- Key code excerpts:

```diff
diff -- tests/models/test_internvl.py
@@ -1,5 +1,5 @@
-from typing import List, Optional, Tuple, Type
+from typing import List, Optional, Tuple, Type, Union
@@ -9,7 +9,8 @@
-from ..conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
+from ..conftest import (IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner,
+                        _ImageAssets)
diff -- vllm/model_executor/models/internvl.py
@@ -5,6 +5,7 @@
+import re
@@ -26,6 +27,7 @@
+from vllm.utils import is_list_of
@@ -95,8 +97,8 @@ def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
-                         max_num: int,
-                         image_size: int) -> Tuple[int, int, int]:
```

- Reviewed files:
  - tests: `tests/models/test_internvl.py` modified +73/-19
  - runtime: `vllm/model_executor/models/internvl.py` modified +46/-14
- Risk and verification: The diff ships test coverage in `tests/models/test_internvl.py`, `tests/models/test_phi3v.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8299 - [Bugfix] Fix InternVL2 vision embeddings process with pipeline parallel

- Link: https://github.com/vllm-project/vllm/pull/8299
- Status/date: merged / 2024-09-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `1230263e161c`
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-3, 48 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix InternVL2 vision embeddings process with pipeline parallel". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: FILL IN THE PR DESCRIPTION HERE FIX #8275 - For InternVL2 with PP, we only need to process image input on first rank. - This PR fixed the error raised by image input processing...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +2/-1 (3 lines); hunks: -17,6 +17,7; -480,7 +481,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +2/-1 (3 lines); hunks: -17,6 +17,7; -480,7 +481,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -17,6 +17,7 @@
+from vllm.distributed import get_pp_group
@@ -480,7 +481,7 @@ def forward(
-        if image_input is not None:
+        if image_input is not None and get_pp_group().is_first_rank:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/distributed/test_pipeline_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8375 - [Bugfix] Fix InternVL2 inference with various num_patches

- Link: https://github.com/vllm-project/vllm/pull/8375
- Status/date: merged / 2024-09-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `e56bf2774158`
- Diff scope read: GitHub Pull Request files API returned 2 files, +39/-3, 73 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[Bugfix] Fix InternVL2 inference with various num_patches". The diff centers on `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py`. PR body context: FILL IN THE PR DESCRIPTION HERE FIX #8361 (*link existing issues this PR will resolve*) FIX #8369 **TODO** - [x] Add test to cover these cases. **BEFORE SUBMITTING, PLEASE READ...
- Key implementation: `tests/models/test_internvl.py` modified +35/-0 (35 lines); hunks: -331,6 +331,41 @@ def test_multi_images_models(hf_runner, vllm_runner, image_...; symbols: test_multi_images_models, test_different_num_patches, touching `test_multi_images_models, test_different_num_patches`; `vllm/model_executor/models/internvl.py` modified +4/-3 (7 lines); hunks: -270,14 +270,14 @@ def input_mapper_for_internvl(ctx: InputContext, data: obj...; -449,11 +449,12 @@ def _parse_and_validate_image_input(; symbols: input_mapper_for_internvl, _parse_and_validate_image_input, touching `input_mapper_for_internvl, _parse_and_validate_image_input`.
- Code diff details:
  - `tests/models/test_internvl.py` modified +35/-0 (35 lines); hunks: -331,6 +331,41 @@ def test_multi_images_models(hf_runner, vllm_runner, image_...; symbols: test_multi_images_models, test_different_num_patches
  - `vllm/model_executor/models/internvl.py` modified +4/-3 (7 lines); hunks: -270,14 +270,14 @@ def input_mapper_for_internvl(ctx: InputContext, data: obj...; -449,11 +449,12 @@ def _parse_and_validate_image_input(; symbols: input_mapper_for_internvl, _parse_and_validate_image_input
- Key code excerpts:

```diff
diff -- tests/models/test_internvl.py
@@ -331,6 +331,41 @@ def test_multi_images_models(hf_runner, vllm_runner, image_assets, model,
+@pytest.mark.parametrize("model", ["OpenGVLab/InternVL2-2B"])
+@pytest.mark.parametrize("size_factors", [[0.5, 1.0]])
+@pytest.mark.parametrize("dtype", [target_dtype])
+@pytest.mark.parametrize("max_tokens", [128])
+@pytest.mark.parametrize("num_logprobs", [5])
+@torch.inference_mode()
diff -- vllm/model_executor/models/internvl.py
@@ -270,14 +270,14 @@ def input_mapper_for_internvl(ctx: InputContext, data: object):
+        # we can't stack here because the images may have different num_patches
-        data = torch.stack(data)
@@ -449,11 +449,12 @@ def _parse_and_validate_image_input(
+            # We need to flatten (B, N, P) to (B*N*P),
+            # so we call flatten_bn twice.
-                    flatten_bn(pixel_values, concat=True).flatten(0, 1)),
```

- Reviewed files:
  - tests: `tests/models/test_internvl.py` modified +35/-0
  - runtime: `vllm/model_executor/models/internvl.py` modified +4/-3
- Risk and verification: The diff ships test coverage in `tests/models/test_internvl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8250 - [BugFix] Propagate 'trust_remote_code' setting in internvl and minicpmv

- Link: https://github.com/vllm-project/vllm/pull/8250
- Status/date: merged / 2024-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `e3dd0692fa2c`
- Diff scope read: GitHub Pull Request files API returned 3 files, +126/-41, 343 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[BugFix] Propagate 'trust_remote_code' setting in internvl and minicpmv". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: For MiniCPM-V model, the token ids needed for compute image bounds are piped through the preprocessing pipeline via `MiniCPMVImageInput` and `MultiModalInputs`. --- section, mar...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +9/-6 (15 lines); hunks: -230,8 +230,9 @@ def input_processor_for_internvl(ctx: InputContext, llm_inpu...; -278,8 +279,9 @@ def input_mapper_for_internvl(ctx: InputContext, data: object):; symbols: input_processor_for_internvl, input_mapper_for_internvl, dummy_data_for_internvl, touching `input_processor_for_internvl, input_mapper_for_internvl, dummy_data_for_internvl`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +9/-6 (15 lines); hunks: -230,8 +230,9 @@ def input_processor_for_internvl(ctx: InputContext, llm_inpu...; -278,8 +279,9 @@ def input_mapper_for_internvl(ctx: InputContext, data: object):; symbols: input_processor_for_internvl, input_mapper_for_internvl, dummy_data_for_internvl
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -230,8 +230,9 @@ def input_processor_for_internvl(ctx: InputContext, llm_inputs: LLMInputs):
-    tokenizer = cached_get_tokenizer(model_config.tokenizer,
-                                     trust_remote_code=True)
+    tokenizer = cached_get_tokenizer(
+        model_config.tokenizer,
+        trust_remote_code=model_config.trust_remote_code)
@@ -278,8 +279,9 @@ def input_mapper_for_internvl(ctx: InputContext, data: object):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +9/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/minicpmv.py`, `vllm/model_executor/models/qwen.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8614 - [VLM][Bugfix] enable internvl running with num_scheduler_steps > 1

- Link: https://github.com/vllm-project/vllm/pull/8614
- Status/date: merged / 2024-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `0c4d2ad5e641`
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-1, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[VLM][Bugfix] enable internvl running with num_scheduler_steps > 1". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: FILL IN THE PR DESCRIPTION HERE A small fix for InternVL while running with num_scheduler_steps > 1, before this PR, set `num_scheduler_steps > 1` for InternVL, will raise an er...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +6/-1 (7 lines); hunks: -19,7 +19,7; -376,6 +376,11 @@ def __init__(self,; symbols: __init__, pixel_shuffle, touching `__init__, pixel_shuffle`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +6/-1 (7 lines); hunks: -19,7 +19,7; -376,6 +376,11 @@ def __init__(self,; symbols: __init__, pixel_shuffle
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -19,7 +19,7 @@
-from vllm.model_executor.layers.sampler import SamplerOutput
+from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
@@ -376,6 +376,11 @@ def __init__(self,
+        if hasattr(self.language_model, "sampler"):
+            self.sampler = self.language_model.sampler
+        else:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8946 - [Model] Expose InternVL2 max_dynamic_patch as a mm_processor_kwarg

- Link: https://github.com/vllm-project/vllm/pull/8946
- Status/date: merged / 2024-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `2ae25f79cf1e`
- Diff scope read: GitHub Pull Request files API returned 2 files, +90/-61, 252 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Model] Expose InternVL2 max_dynamic_patch as a mm_processor_kwarg". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: FILL IN THE PR DESCRIPTION HERE - Fix the bug mentioned in https://github.com/vllm-project/vllm/pull/6514#issuecomment-2378785414 - Expose InternVL2 max_dynamic_patch as require...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +89/-61 (150 lines); hunks: -5,8 +5,9; -122,6 +123,20 @@ def calculate_num_blocks(orig_width: int, orig_height: int,...; symbols: calculate_num_blocks, calculate_num_blocks_wrapper, dynamic_preprocess, image_to_pixel_values, touching `calculate_num_blocks, calculate_num_blocks_wrapper, dynamic_preprocess`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +89/-61 (150 lines); hunks: -5,8 +5,9; -122,6 +123,20 @@ def calculate_num_blocks(orig_width: int, orig_height: int,...; symbols: calculate_num_blocks, calculate_num_blocks_wrapper, dynamic_preprocess, image_to_pixel_values
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -5,8 +5,9 @@
-from typing import (Iterable, List, Literal, Mapping, Optional, Tuple,
-                    TypedDict, Union)
+from functools import partial
+from typing import (Any, Dict, Iterable, List, Literal, Mapping, Optional,
+                    Tuple, TypedDict, Union)
@@ -122,6 +123,20 @@ def calculate_num_blocks(orig_width: int, orig_height: int, min_num: int,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +89/-61
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9351 - [Bugfix] Update InternVL input mapper to support image embeds

- Link: https://github.com/vllm-project/vllm/pull/9351
- Status/date: merged / 2024-10-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `55e081fbad29`
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[Bugfix] Update InternVL input mapper to support image embeds". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: The InternVL input mapper originally only outputs MultiModalInputs with pixel values. However, the model itself has support for image embedding inputs and this change is needed...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +2/-0 (2 lines); hunks: -342,6 +342,8 @@ def input_mapper(; symbols: input_mapper, touching `input_mapper`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +2/-0 (2 lines); hunks: -342,6 +342,8 @@ def input_mapper(; symbols: input_mapper
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -342,6 +342,8 @@ def input_mapper(
+        else:
+            return MultiModalInputs({"image_embeds": data})
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9528 - [Model][VLM] Initialize support for Mono-InternVL model

- Link: https://github.com/vllm-project/vllm/pull/9528
- Status/date: merged / 2024-10-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`; associated commits `bb392ea2d2bf`
- Diff scope read: GitHub Pull Request files API returned 6 files, +254/-28, 387 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[Model][VLM] Initialize support for Mono-InternVL model". The diff centers on `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/decoder_only/vision_language/test_internvl.py`. PR body context: FILL IN THE PR DESCRIPTION HERE FIX #xxxx (*link existing issues this PR will resolve*) - This PR adds support for Mono-InternVL series models: OpenGVLab/Mono-InternVL-2B, a mon...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +42/-19 (61 lines); hunks: -21,7 +21,8; -427,13 +428,9 @@ def __init__(self,; symbols: __init__, sampler, _init_vision_model, _init_mlp1, touching `__init__, sampler, _init_vision_model`; `vllm/model_executor/models/intern_vit.py` modified +31/-0 (31 lines); hunks: -97,6 +97,37 @@ def forward(self, pixel_values: torch.FloatTensor) -> torch.T...; symbols: forward, InternVisionPatchModel, __init__, get_input_embeddings, touching `forward, InternVisionPatchModel, __init__`; `tests/models/decoder_only/vision_language/test_internvl.py` modified +13/-8 (21 lines); hunks: -7,7 +7,6; -19,15 +18,20; symbols: generate, run_awq_test, touching `generate, run_awq_test`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +42/-19 (61 lines); hunks: -21,7 +21,8; -427,13 +428,9 @@ def __init__(self,; symbols: __init__, sampler, _init_vision_model, _init_mlp1
  - `vllm/model_executor/models/intern_vit.py` modified +31/-0 (31 lines); hunks: -97,6 +97,37 @@ def forward(self, pixel_values: torch.FloatTensor) -> torch.T...; symbols: forward, InternVisionPatchModel, __init__, get_input_embeddings
  - `tests/models/decoder_only/vision_language/test_internvl.py` modified +13/-8 (21 lines); hunks: -7,7 +7,6; -19,15 +18,20; symbols: generate, run_awq_test
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -21,7 +21,8 @@
-from vllm.model_executor.models.intern_vit import InternVisionModel
+from vllm.model_executor.models.intern_vit import (InternVisionModel,
+                                                   InternVisionPatchModel)
@@ -427,13 +428,9 @@ def __init__(self,
-        vision_feature_layer = self.select_layer
-        if vision_feature_layer < 0:
diff -- vllm/model_executor/models/intern_vit.py
@@ -97,6 +97,37 @@ def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
+class InternVisionPatchModel(nn.Module):
+    def __init__(self, config: PretrainedConfig):
+        super().__init__()
+        self.config = config
+        self.embeddings = InternVisionEmbeddings(config)
+    def get_input_embeddings(self):
diff -- tests/models/decoder_only/vision_language/test_internvl.py
@@ -7,7 +7,6 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +42/-19; `vllm/model_executor/models/intern_vit.py` modified +31/-0
  - tests: `tests/models/decoder_only/vision_language/test_internvl.py` modified +13/-8
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_internvl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10518 - [Model] Expose `dynamic_image_size` as mm_processor_kwargs for InternVL2 models

- Link: https://github.com/vllm-project/vllm/pull/10518
- Status/date: merged / 2024-11-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `d5ec121f95f5`
- Diff scope read: GitHub Pull Request files API returned 2 files, +255/-14, 350 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Model] Expose `dynamic_image_size` as mm_processor_kwargs for InternVL2 models". The diff centers on `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py`, `vllm/model_executor/models/internvl.py`. PR body context: FIX #9989 - https://github.com/vllm-project/vllm/issues/9989#issuecomment-2488303299 - Note that `dynamic_image_size=False` is equivalent to `max_dynamic_patch=1`, means that im...
- Key implementation: `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py` added +206/-0 (206 lines); hunks: -0,0 +1,206; symbols: input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens, test_input_mapper_override, touching `input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens`; `vllm/model_executor/models/internvl.py` modified +49/-14 (63 lines); hunks: -123,8 +123,15 @@ def calculate_num_blocks(orig_width: int, orig_height: int,...; -183,10 +190,17 @@ def image_to_pixel_values(image: Image.Image, input_size:...; symbols: calculate_num_blocks, calculate_num_blocks_wrapper, image_to_pixel_values, image_to_pixel_values_wrapper, touching `calculate_num_blocks, calculate_num_blocks_wrapper, image_to_pixel_values`.
- Code diff details:
  - `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py` added +206/-0 (206 lines); hunks: -0,0 +1,206; symbols: input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens, test_input_mapper_override
  - `vllm/model_executor/models/internvl.py` modified +49/-14 (63 lines); hunks: -123,8 +123,15 @@ def calculate_num_blocks(orig_width: int, orig_height: int,...; -183,10 +190,17 @@ def image_to_pixel_values(image: Image.Image, input_size:...; symbols: calculate_num_blocks, calculate_num_blocks_wrapper, image_to_pixel_values, image_to_pixel_values_wrapper
- Key code excerpts:

```diff
diff -- tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py
@@ -0,0 +1,206 @@
+"""Tests for InternVL's multimodal preprocessing kwargs."""
+from typing import Callable, Optional
+import pytest
+from transformers import AutoTokenizer
+from vllm.inputs import InputContext, token_inputs
+from vllm.multimodal import MultiModalRegistry
diff -- vllm/model_executor/models/internvl.py
@@ -123,8 +123,15 @@ def calculate_num_blocks(orig_width: int, orig_height: int, min_num: int,
-def calculate_num_blocks_wrapper(hf_config: PretrainedConfig,
-                                 max_dynamic_patch: Optional[int] = None):
+def calculate_num_blocks_wrapper(
+    hf_config: PretrainedConfig,
+    max_dynamic_patch: Optional[int] = None,
+    dynamic_image_size: Optional[bool] = None,
```

- Reviewed files:
  - tests: `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py` added +206/-0
  - runtime: `vllm/model_executor/models/internvl.py` modified +49/-14
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11165 - [V1][VLM] Fix edge case bug for InternVL2

- Link: https://github.com/vllm-project/vllm/pull/11165
- Status/date: merged / 2024-12-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `969da7d70bc0`
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[V1][VLM] Fix edge case bug for InternVL2". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: When there's only one image in the batch, the `image_embeds` needs to be reshaped to `[feature_size, hidden_size]` from `[num_patches, feature_size_per_patch, hidden_size]` befo...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +4/-1 (5 lines); hunks: -669,8 +669,11 @@ def _process_image_input(; symbols: _process_image_input, touching `_process_image_input`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +4/-1 (5 lines); hunks: -669,8 +669,11 @@ def _process_image_input(; symbols: _process_image_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -669,8 +669,11 @@ def _process_image_input(
+        # Only one image in the current batch
-            image_embeds = image_embeds.unsqueeze(0)
+            image_embeds = image_embeds.view(
+                -1, self.config.text_config.hidden_size).unsqueeze(0)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12553 - [VLM] Merged multi-modal processor for InternVL-based models

- Link: https://github.com/vllm-project/vllm/pull/12553
- Status/date: merged / 2025-02-04
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_internvl.py`, `vllm/model_executor/models/internvl.py`; associated commits `d1ca7df84d9f`
- Diff scope read: GitHub Pull Request files API returned 34 files, +1434/-986, 3135 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR changes model-related implementation. Title: "[VLM] Merged multi-modal processor for InternVL-based models". The diff centers on `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py`. PR body context: Update InternVL (similarly, H2OVL and NVLM-D) to use the merged multi-modal processor. **Note:** `BaseProcessingInfo.get_mm_max_tokens_per_item` now takes a `mm_counts` argument...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +503/-320 (823 lines); hunks: -6,35 +6,37; -75,22 +77,27 @@ class InternVLImageEmbeddingInputs(TypedDict):; symbols: InternVLImageEmbeddingInputs, build_transform, find_closest_aspect_ratio, touching `InternVLImageEmbeddingInputs, build_transform, find_closest_aspect_ratio`; `tests/models/multimodal/processing/test_internvl.py` modified +32/-175 (207 lines); hunks: -1,207 +1,64; symbols: input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens, test_input_mapper_override, touching `input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +503/-320 (823 lines); hunks: -6,35 +6,37; -75,22 +77,27 @@ class InternVLImageEmbeddingInputs(TypedDict):; symbols: InternVLImageEmbeddingInputs, build_transform, find_closest_aspect_ratio
  - `tests/models/multimodal/processing/test_internvl.py` modified +32/-175 (207 lines); hunks: -1,207 +1,64; symbols: input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens, test_input_mapper_override
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -6,35 +6,37 @@
-import re
-from functools import cached_property, partial
+from abc import ABC, abstractmethod
+from functools import cached_property
-                    TypedDict, Union)
+                    TypedDict, TypeVar, Union)
diff -- tests/models/multimodal/processing/test_internvl.py
@@ -1,207 +1,64 @@
-from typing import Callable, Optional
+from typing import Optional
-from transformers import AutoTokenizer
-from vllm.inputs import InputContext, token_inputs
-from vllm.multimodal import MultiModalRegistry
+from vllm.multimodal import MULTIMODAL_REGISTRY
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +503/-320
  - tests: `tests/models/multimodal/processing/test_internvl.py` modified +32/-175
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_h2ovl.py`, `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14738 - [VLM] Support loading InternVideo2.5 models as original InternVLChatModel

- Link: https://github.com/vllm-project/vllm/pull/14738
- Status/date: merged / 2025-03-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `b1cc4dfef57a`
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-3, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[VLM] Support loading InternVideo2.5 models as original InternVLChatModel". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: FIX #14180 Separate from #14688 - Note that hierarchical token compression (HiCo) technology newly added in InternVideo hasn't supported yet, so the outputs will be different fr...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +8/-1 (9 lines); hunks: -981,5 +981,12 @@ def sample(; symbols: sample, load_weights, touching `sample, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +8/-1 (9 lines); hunks: -981,5 +981,12 @@ def sample(; symbols: sample, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -981,5 +981,12 @@ def sample(
-        loader = AutoWeightsLoader(self)
+        # unused modules appear in OpenGVLab/InternVideo2_5_Chat_8B
+        skip_prefixes = [
+            "action_embed", "temporal_embed", "track_embed",
+            "track_embed_decoder", "box_token", "cg_criterion", "cg_model",
+            "loc_encoder", "loc_decoder", "sam", "temporal_token",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +8/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15086 - [Bugfix] Fix embedding assignment for InternVL-based models

- Link: https://github.com/vllm-project/vllm/pull/15086
- Status/date: merged / 2025-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `ffa443afedd3`
- Diff scope read: GitHub Pull Request files API returned 7 files, +123/-106, 488 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix embedding assignment for InternVL-based models". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: Similar to #14980, this PR fixes the mismatch between multimodal embeddings and `PlaceholderRange` indices for InternVL-based models. FIX https://github.com/vllm-project/vllm/is...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +104/-69 (173 lines); hunks: -9,14 +9,13; -36,10 +35,12; symbols: InternVLImagePixelInputs, InternVLImageEmbeddingInputs, image_token_id, get_image_repl_features, touching `InternVLImagePixelInputs, InternVLImageEmbeddingInputs, image_token_id`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +104/-69 (173 lines); hunks: -9,14 +9,13; -36,10 +35,12; symbols: InternVLImagePixelInputs, InternVLImageEmbeddingInputs, image_token_id, get_image_repl_features
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -9,14 +9,13 @@
-from typing import (List, Literal, Optional, Set, Tuple, TypedDict, TypeVar,
-                    Union)
+from typing import Literal, Optional, Set, Tuple, TypedDict, TypeVar, Union
-from transformers import BatchFeature, PretrainedConfig, TensorType
+from transformers import BatchEncoding, PretrainedConfig, TensorType
@@ -36,10 +35,12 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +104/-69
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma3_mm.py`, `vllm/model_executor/models/h2ovl.py`, `vllm/model_executor/models/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18499 - [VLM] Initialize video input support for InternVL models

- Link: https://github.com/vllm-project/vllm/pull/18499
- Status/date: merged / 2025-05-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `75f81750f3a9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +596/-62, 940 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[VLM] Initialize video input support for InternVL models". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: FIX #18381 (*link existing issues this PR will resolve*) - This PR only adds video support to InternVL with Qwen2.5 backbone, because it has unused Qwen2-VL video token in token...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +485/-26 (511 lines); hunks: -8,8 +8,9; -74,6 +75,33 @@ class InternVLImageEmbeddingInputs(TypedDict):; symbols: InternVLImageEmbeddingInputs, InternVLVideoPixelInputs, InternVLVideoEmbeddingInputs, build_transform, touching `InternVLImageEmbeddingInputs, InternVLVideoPixelInputs, InternVLVideoEmbeddingInputs`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +485/-26 (511 lines); hunks: -8,8 +8,9; -74,6 +75,33 @@ class InternVLImageEmbeddingInputs(TypedDict):; symbols: InternVLImageEmbeddingInputs, InternVLVideoPixelInputs, InternVLVideoEmbeddingInputs, build_transform
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -8,8 +8,9 @@
-from typing import Literal, Optional, TypedDict, TypeVar, Union
+from typing import Any, Literal, Optional, TypedDict, TypeVar, Union
+import numpy.typing as npt
@@ -74,6 +75,33 @@ class InternVLImageEmbeddingInputs(TypedDict):
+class InternVLVideoPixelInputs(TypedDict):
+    type: Literal["pixel_values_videos"]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +485/-26
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18842 - [LoRA] Add LoRA support for InternVL

- Link: https://github.com/vllm-project/vllm/pull/18842
- Status/date: merged / 2025-05-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `34d6c447c4b9`
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-2, 50 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[LoRA] Add LoRA support for InternVL". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: FIX https://github.com/vllm-project/vllm/issues/18820
- Key implementation: `vllm/model_executor/models/internvl.py` modified +23/-2 (25 lines); hunks: -22,6 +22,7; -36,7 +37,8; symbols: get_video_replacement_internvl, InternVLChatModel, __init__, load_weights, touching `get_video_replacement_internvl, InternVLChatModel, __init__`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +23/-2 (25 lines); hunks: -22,6 +22,7; -36,7 +37,8; symbols: get_video_replacement_internvl, InternVLChatModel, __init__, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -22,6 +22,7 @@
+from vllm.model_executor.models.module_mapping import MultiModelKeys
@@ -36,7 +37,8 @@
-from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
+from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
+                         SupportsMultiModal, SupportsPP)
@@ -1014,7 +1016,17 @@ def get_video_replacement_internvl(item_idx: int):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +23/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21684 - Migrate InternVLImageInputs and InternVLVideoInputs to TensorSchema

- Link: https://github.com/vllm-project/vllm/pull/21684
- Status/date: merged / 2025-07-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `f1e2c095ecee`
- Diff scope read: GitHub Pull Request files API returned 1 files, +49/-62, 184 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR changes model-related implementation. Title: "Migrate InternVLImageInputs and InternVLVideoInputs to TensorSchema". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: ## Purpose This PR migrates InternVLImageInputs and InternVLVideoInputs from a TypedDict-based definition to a structured TensorSchema model with runtime shape validation. This...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +49/-62 (111 lines); hunks: -9,7 +9,7; -37,6 +37,7; symbols: InternVLImagePixelInputs, InternVLImageEmbeddingInputs, InternVLVideoPixelInputs, touching `InternVLImagePixelInputs, InternVLImageEmbeddingInputs, InternVLVideoPixelInputs`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +49/-62 (111 lines); hunks: -9,7 +9,7; -37,6 +37,7; symbols: InternVLImagePixelInputs, InternVLImageEmbeddingInputs, InternVLVideoPixelInputs
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -9,7 +9,7 @@
-from typing import Any, Literal, Optional, TypedDict, TypeVar, Union
+from typing import Annotated, Any, Literal, Optional, TypeVar, Union
@@ -37,6 +37,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
@@ -51,54 +52,60 @@
-class InternVLImagePixelInputs(TypedDict):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +49/-62
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23658 - [Model] Enable video support for InternVL3.5 models

- Link: https://github.com/vllm-project/vllm/pull/23658
- Status/date: merged / 2025-08-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `9816b81f5f9f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +22/-7, 71 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[Model] Enable video support for InternVL3.5 models". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: ## Purpose - Fix https://github.com/vllm-project/vllm/issues/23638 - Update InternVL model's video token map to enable video support for InternVL3.5 backbone - Update documentat...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +7/-3 (10 lines); hunks: -855,9 +855,13 @@ def get_supported_mm_limits(self):; symbols: get_supported_mm_limits, get_video_token, get_num_frames_with_most_features, touching `get_supported_mm_limits, get_video_token, get_num_frames_with_most_features`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +7/-3 (10 lines); hunks: -855,9 +855,13 @@ def get_supported_mm_limits(self):; symbols: get_supported_mm_limits, get_video_token, get_num_frames_with_most_features
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -855,9 +855,13 @@ def get_supported_mm_limits(self):
-        if text_model_type == "qwen2":
-            return "<|video_pad|>"
-        return None
+        video_token_map = {
+            "qwen2": "<|video_pad|>",
+            "qwen3": "<|video_pad|>",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +7/-3
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23742 - [Model] Enable native HF format InternVL support

- Link: https://github.com/vllm-project/vllm/pull/23742
- Status/date: merged / 2025-08-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +18/-16, 76 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[Model] Enable native HF format InternVL support". The diff centers on `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`, `docs/models/supported_models.md`. PR body context: ## Purpose - `InternS1ForConditionalGeneration` shares same implementation with `InternVLForConditionalGeneration`, so we can support it natively. FIX #23714 FIX #23730 ## Test...
- Key implementation: `tests/models/multimodal/generation/test_common.py` modified +14/-15 (29 lines); hunks: -222,21 +222,6; -461,6 +446,20; `tests/models/registry.py` modified +2/-1 (3 lines); hunks: -429,6 +429,7 @@ def check_available_online(; -584,7 +585,7 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`; `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -629,6 +629,7 @@ These models primarily accept the [`LLM.generate`](./generat...; `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -217,6 +217,7.
- Code diff details:
  - `tests/models/multimodal/generation/test_common.py` modified +14/-15 (29 lines); hunks: -222,21 +222,6; -461,6 +446,20
  - `tests/models/registry.py` modified +2/-1 (3 lines); hunks: -429,6 +429,7 @@ def check_available_online(; -584,7 +585,7 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -629,6 +629,7 @@ These models primarily accept the [`LLM.generate`](./generat...
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -217,6 +217,7
- Key code excerpts:

```diff
diff -- tests/models/multimodal/generation/test_common.py
@@ -222,21 +222,6 @@
-    # Check "auto" with fallback to transformers
-    "internvl-transformers": VLMTestInfo(
-        models=["OpenGVLab/InternVL3-1B-hf"],
-        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
-        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n", # noqa: E501
-        img_idx_to_prompt=lambda idx: "<IMG_CONTEXT>",
diff -- tests/models/registry.py
@@ -429,6 +429,7 @@ def check_available_online(
+    "InternVLForConditionalGeneration": _HfExamplesInfo("OpenGVLab/InternVL3-1B-hf"),    # noqa: E501
@@ -584,7 +585,7 @@ def check_available_online(
-    "TransformersForMultimodalLM": _HfExamplesInfo("OpenGVLab/InternVL3-1B-hf"),
+    "TransformersForMultimodalLM": _HfExamplesInfo("BAAI/Emu3-Chat-hf"),
diff -- docs/models/supported_models.md
@@ -629,6 +629,7 @@ These models primarily accept the [`LLM.generate`](./generative_models.md#llmgen
+| `InternVLForConditionalGeneration` | InternVL 3.0 (HF format) | T + I<sup>E+</sup> + V<sup>E+</sup> | `OpenGVLab/InternVL3-1B-hf`, etc. | ✅︎ | ✅︎ | ✅︎ |
diff -- vllm/model_executor/models/registry.py
```

- Reviewed files:
  - tests: `tests/models/multimodal/generation/test_common.py` modified +14/-15; `tests/models/registry.py` modified +2/-1
  - docs: `docs/models/supported_models.md` modified +1/-0
  - runtime: `vllm/model_executor/models/registry.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24519 - [Model] Limit CPU threads for image transformations in InternVL to reduce cpu contention.

- Link: https://github.com/vllm-project/vllm/pull/24519
- Status/date: merged / 2025-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `267c80d31f6b`
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-1, 44 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR changes model-related implementation. Title: "[Model] Limit CPU threads for image transformations in InternVL to reduce cpu contention.". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: # Limit CPU threads for image transformations in InternVL to reduce cpu contention. ## Purpose Image transformation operations in Internvl model (which include tensor computatio...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +16/-1 (17 lines); hunks: -7,6 +7,7; -37,6 +38,7; symbols: InternVLVideoEmbeddingInputs, build_transform, apply, touching `InternVLVideoEmbeddingInputs, build_transform, apply`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +16/-1 (17 lines); hunks: -7,6 +7,7; -37,6 +38,7; symbols: InternVLVideoEmbeddingInputs, build_transform, apply
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -7,6 +7,7 @@
+import os
@@ -37,6 +38,7 @@
+from vllm.utils import set_default_torch_num_threads
@@ -115,13 +117,26 @@ class InternVLVideoEmbeddingInputs(TensorSchema):
-    return T.Compose([
+    transform = T.Compose([
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +16/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23909 - [Model] enable data parallel for InternVL vision encoder

- Link: https://github.com/vllm-project/vllm/pull/23909
- Status/date: merged / 2025-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`; associated commits `52bc9d5b3edb`
- Diff scope read: GitHub Pull Request files API returned 3 files, +80/-33, 262 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[Model] enable data parallel for InternVL vision encoder". The diff centers on `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`. PR body context: ## Purpose This PR enables DP for InternVL vision encoder FIX #23876 ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpos...
- Key implementation: `vllm/model_executor/models/intern_vit.py` modified +75/-32 (107 lines); hunks: -25,9 +25,11; -137,6 +139,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/internvl.py` modified +4/-1 (5 lines); hunks: -1020,6 +1020,8 @@ def get_video_replacement_internvl(item_idx: int):; -1038,6 +1040,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: get_video_replacement_internvl, InternVLChatModel, get_placeholder_str, __init__, touching `get_video_replacement_internvl, InternVLChatModel, get_placeholder_str`.
- Code diff details:
  - `vllm/model_executor/models/intern_vit.py` modified +75/-32 (107 lines); hunks: -25,9 +25,11; -137,6 +139,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/internvl.py` modified +4/-1 (5 lines); hunks: -1020,6 +1020,8 @@ def get_video_replacement_internvl(item_idx: int):; -1038,6 +1040,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: get_video_replacement_internvl, InternVLChatModel, get_placeholder_str, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/intern_vit.py
@@ -25,9 +25,11 @@
+                                               ReplicatedLinear,
+from vllm.multimodal.utils import run_dp_sharded_vision_model
@@ -137,6 +139,7 @@ def __init__(
+        use_data_parallel: bool = False,
@@ -150,23 +153,34 @@ def __init__(
-        self.tp_size = get_tensor_model_parallel_world_size()
diff -- vllm/model_executor/models/internvl.py
@@ -1020,6 +1020,8 @@ def get_video_replacement_internvl(item_idx: int):
+    supports_encoder_tp_data = True
@@ -1038,6 +1040,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
+        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
@@ -1105,7 +1108,7 @@ def _init_vision_model(
-            )
+                use_data_parallel=self.use_data_parallel)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/intern_vit.py` modified +75/-32; `vllm/model_executor/models/internvl.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26153 - [Model] Use `merge_by_field_config` for MM models (InternVL family)

- Link: https://github.com/vllm-project/vllm/pull/26153
- Status/date: merged / 2025-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `f9a8084e4879`
- Diff scope read: GitHub Pull Request files API returned 9 files, +84/-182, 785 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR extends deployment docs, tests, or CI coverage. Title: "[Model] Use `merge_by_field_config` for MM models (InternVL family)". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: ## Purpose Part of #26149 Also use Intern-S1-mini for the example scripts. ## Test Plan Model and tensor schema tests should pass. I have also run the example script on all mode...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +27/-54 (81 lines); hunks: -17,7 +17,7; -28,7 +28,7; symbols: _preprocess_image, __call__, InternVLProcessor, _preprocess_video, touching `_preprocess_image, __call__, InternVLProcessor`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +27/-54 (81 lines); hunks: -17,7 +17,7; -28,7 +28,7; symbols: _preprocess_image, __call__, InternVLProcessor, _preprocess_video
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -17,7 +17,7 @@
-from transformers import BatchEncoding, PretrainedConfig, TensorType
+from transformers import BatchFeature, PretrainedConfig, TensorType
@@ -28,7 +28,7 @@
-                                    MultiModalKwargsItems, NestedTensors)
+                                    MultiModalKwargsItems)
@@ -42,8 +42,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +27/-54
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/nano_nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32397 - [Model] Enable LoRA support for internvl2

- Link: https://github.com/vllm-project/vllm/pull/32397
- Status/date: merged / 2026-01-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`; associated commits `fec9da0af48d`
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-3, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[Model] Enable LoRA support for internvl2". The diff centers on `vllm/model_executor/models/internvl.py`. PR body context: ## Purpose Enable dynamic LoRA adapters on InternVL2 vision tower + connector (part of #31479) by implementing: - `get_num_mm_encoder_tokens()` - `get_num_mm_connector_tokens()`...
- Key implementation: `vllm/model_executor/models/internvl.py` modified +16/-3 (19 lines); hunks: -1086,9 +1086,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -1430,3 +1429,17 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: __init__, get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens, touching `__init__, get_mm_mapping, get_num_mm_encoder_tokens`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +16/-3 (19 lines); hunks: -1086,9 +1086,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -1430,3 +1429,17 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: __init__, get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -1086,9 +1086,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
-        self.num_image_token = int(
-            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
-        )
+        self.patch_tokens = (image_size // patch_size) ** 2
+        self.num_image_token = int(self.patch_tokens * (config.downsample_ratio**2))
@@ -1430,3 +1429,17 @@ def get_mm_mapping(self) -> MultiModelKeys:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +16/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37260 - [1/2] Move InternVL-based processors

- Link: https://github.com/vllm-project/vllm/pull/37260
- Status/date: merged / 2026-03-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_internvl.py`, `vllm/model_executor/models/internvl.py`, `vllm/transformers_utils/processors/internvl.py`; associated commits `f34032433573`
- Diff scope read: GitHub Pull Request files API returned 20 files, +3252/-3099, 6681 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR changes model-related implementation. Title: "[1/2] Move InternVL-based processors". The diff centers on `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py`. PR body context: ## Purpose Move the Transformers-like processing code from modeling file to `transformers_utils` in order to enhance readability. They will be further refactored in the next PR...
- Key implementation: `vllm/transformers_utils/processors/internvl.py` added +603/-0 (603 lines); hunks: -0,0 +1,603; symbols: build_transform, find_closest_aspect_ratio, resolve_internvl_min_max_num, get_internvl_target_ratios, touching `build_transform, find_closest_aspect_ratio, resolve_internvl_min_max_num`; `vllm/model_executor/models/internvl.py` modified +7/-578 (585 lines); hunks: -7,16 +7,13; -28,7 +25,6; symbols: InternVLImagePixelInputs, InternVLVideoEmbeddingInputs, build_transform, find_closest_aspect_ratio, touching `InternVLImagePixelInputs, InternVLVideoEmbeddingInputs, build_transform`; `tests/models/multimodal/processing/test_internvl.py` modified +1/-1 (2 lines); hunks: -23,7 +23,7 @@ def _get_expected_num_patches(; symbols: _get_expected_num_patches, touching `_get_expected_num_patches`.
- Code diff details:
  - `vllm/transformers_utils/processors/internvl.py` added +603/-0 (603 lines); hunks: -0,0 +1,603; symbols: build_transform, find_closest_aspect_ratio, resolve_internvl_min_max_num, get_internvl_target_ratios
  - `vllm/model_executor/models/internvl.py` modified +7/-578 (585 lines); hunks: -7,16 +7,13; -28,7 +25,6; symbols: InternVLImagePixelInputs, InternVLVideoEmbeddingInputs, build_transform, find_closest_aspect_ratio
  - `tests/models/multimodal/processing/test_internvl.py` modified +1/-1 (2 lines); hunks: -23,7 +23,7 @@ def _get_expected_num_patches(; symbols: _get_expected_num_patches
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/processors/internvl.py
@@ -0,0 +1,603 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
+# --------------------------------------------------------
+# InternVL
+# Copyright (c) 2023 OpenGVLab
diff -- vllm/model_executor/models/internvl.py
@@ -7,16 +7,13 @@
-from abc import ABC, abstractmethod
+from abc import abstractmethod
-from typing import Annotated, Any, Literal, TypeAlias, TypeVar
+from typing import Annotated, Literal, TypeAlias, TypeVar
-import numpy.typing as npt
-import torchvision.transforms as T
diff -- tests/models/multimodal/processing/test_internvl.py
@@ -23,7 +23,7 @@ def _get_expected_num_patches(
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/internvl.py` added +603/-0; `vllm/model_executor/models/internvl.py` modified +7/-578
  - tests: `tests/models/multimodal/processing/test_internvl.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_h2ovl.py`, `tests/models/multimodal/processing/test_internvl.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37324 - [2/3] Refactor InternVL-based processors

- Link: https://github.com/vllm-project/vllm/pull/37324
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/internvl.py`, `vllm/transformers_utils/processors/internvl.py`; associated commits `99267c23ca51`
- Diff scope read: GitHub Pull Request files API returned 18 files, +762/-1146, 2597 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR changes model-related implementation. Title: "[2/3] Refactor InternVL-based processors". The diff centers on `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py`. PR body context: ## Purpose Follow-up to #37289 - Split up processing logic into `*ImageProcessor` and `*VideoProcessor`. - Remove unnecessary processor definitions for Eagle2.5 and SkyworkR1V -...
- Key implementation: `vllm/transformers_utils/processors/internvl.py` modified +233/-273 (506 lines); hunks: -7,24 +7,17; -33,7 +26,7; symbols: build_transform, video_to_pixel_values_internvl, BaseInternVLProcessor, InternVLImageProcessor, touching `build_transform, video_to_pixel_values_internvl, BaseInternVLProcessor`; `vllm/model_executor/models/internvl.py` modified +85/-46 (131 lines); hunks: -9,6 +9,7; -45,8 +46,9; symbols: BaseInternVLProcessingInfo, get_hf_processor, get_supported_mm_limits, get_num_image_tokens, touching `BaseInternVLProcessingInfo, get_hf_processor, get_supported_mm_limits`.
- Code diff details:
  - `vllm/transformers_utils/processors/internvl.py` modified +233/-273 (506 lines); hunks: -7,24 +7,17; -33,7 +26,7; symbols: build_transform, video_to_pixel_values_internvl, BaseInternVLProcessor, InternVLImageProcessor
  - `vllm/model_executor/models/internvl.py` modified +85/-46 (131 lines); hunks: -9,6 +9,7; -45,8 +46,9; symbols: BaseInternVLProcessingInfo, get_hf_processor, get_supported_mm_limits, get_num_image_tokens
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/processors/internvl.py
@@ -7,24 +7,17 @@
-from abc import ABC, abstractmethod
-from typing import Any, TypeVar
-from transformers import BatchFeature, PretrainedConfig, TensorType
+from transformers import BatchFeature, TensorType
+from transformers.processing_utils import ProcessorMixin
-from vllm.tokenizers import TokenizerLike
diff -- vllm/model_executor/models/internvl.py
@@ -9,6 +9,7 @@
+from functools import cached_property
@@ -45,8 +46,9 @@
-    BaseInternVLProcessor,
+    InternVLImageProcessor,
+    InternVLVideoProcessor,
@@ -123,7 +125,7 @@ class BaseInternVLProcessingInfo(BaseProcessingInfo):
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/internvl.py` modified +233/-273; `vllm/model_executor/models/internvl.py` modified +85/-46
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38049 - [Model] Add torch.compile support for InternVL vision encoder

- Link: https://github.com/vllm-project/vllm/pull/38049
- Status/date: merged / 2026-03-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/intern_vit.py`; associated commits `38de82231023`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +20/-3, 51 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For InternVL 3.5, this PR adds or enables a model support/runtime surface. Title: "[Model] Add torch.compile support for InternVL vision encoder". The diff centers on `vllm/model_executor/models/intern_vit.py`. PR body context: # [Model] Add torch.compile support for InternVL vision encoder ## Summary ## Purpose Add `torch.compile` support for the InternVL/InternViT vision encoder using the `@support_t...
- Key implementation: `vllm/model_executor/models/intern_vit.py` modified +11/-2 (13 lines); hunks: -15,6 +15,10; -280,6 +284,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: forward, InternVisionEncoderLayer, __init__, touching `forward, InternVisionEncoderLayer, __init__`.
- Code diff details:
  - `vllm/model_executor/models/intern_vit.py` modified +11/-2 (13 lines); hunks: -15,6 +15,10; -280,6 +284,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: forward, InternVisionEncoderLayer, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/intern_vit.py
@@ -15,6 +15,10 @@
+from vllm.compilation.decorators import (
+    should_torch_compile_mm_encoder,
+    support_torch_compile,
+)
@@ -280,6 +284,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
+@support_torch_compile(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/intern_vit.py` modified +11/-2
- Risk and verification: Runtime changes concentrate in `vllm/config/utils.py`, `vllm/model_executor/models/intern_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.
