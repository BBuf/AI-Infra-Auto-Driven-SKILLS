# sglang GLM VLM/OCR Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/GLM/GLM-OCR.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/glm-ocr-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/models/glm4v.py` | [#9884](https://github.com/sgl-project/sglang/pull/9884), [#10147](https://github.com/sgl-project/sglang/pull/10147), [#17582](https://github.com/sgl-project/sglang/pull/17582), [#20033](https://github.com/sgl-project/sglang/pull/20033) |
| `python/sglang/srt/models/glm4v_moe.py` | [#20463](https://github.com/sgl-project/sglang/pull/20463), [#20740](https://github.com/sgl-project/sglang/pull/20740), [#21134](https://github.com/sgl-project/sglang/pull/21134) |
| `python/sglang/srt/models/glm_ocr.py` | [#17582](https://github.com/sgl-project/sglang/pull/17582), [#20463](https://github.com/sgl-project/sglang/pull/20463), [#20740](https://github.com/sgl-project/sglang/pull/20740), [#21134](https://github.com/sgl-project/sglang/pull/21134) |
| `python/sglang/srt/models/glm_ocr_nextn.py` | [#17582](https://github.com/sgl-project/sglang/pull/17582) |
| `python/sglang/srt/multimodal/processors/glm4v.py` | [#17582](https://github.com/sgl-project/sglang/pull/17582), [#18885](https://github.com/sgl-project/sglang/pull/18885) |

## PR Coverage Summary

- Git-traced PRs: 8
- Extra PRs preserved from existing docs: 28
- Total PRs in this document: 36
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-08-07 | [#8015](https://github.com/sgl-project/sglang/pull/8015) | closed | [WIP] Support glm4.1v | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/multimodal/processors/glm4v.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-08-09 | [#8798](https://github.com/sgl-project/sglang/pull/8798) | merged | Support glm4.1v and glm4.5v | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-08-17 | [#9245](https://github.com/sgl-project/sglang/pull/9245) | merged | Set the default attention backend for GLM-4.5v to fa3 | `python/sglang/srt/utils.py` |
| 2025-08-18 | [#9059](https://github.com/sgl-project/sglang/pull/9059) | merged | [GLM4.1V and GLM4.5V] Add vision transformer num_dummy_head support: max tp=4 -> max tp=8 | `python/sglang/srt/layers/attention/vision_utils.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/internvl.py` |
| 2025-08-19 | [#9349](https://github.com/sgl-project/sglang/pull/9349) | open | Add support for GLM 4.5V FP8 | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` |
| 2025-08-25 | [#9554](https://github.com/sgl-project/sglang/pull/9554) | merged | Fix GLM45v launch server cuda torch compile bug | `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-09-05 | [#9884](https://github.com/sgl-project/sglang/pull/9884) | merged | [Bug Fix] Fix Glm4vVisionBlock norm | `python/sglang/srt/models/glm4v.py` |
| 2025-09-08 | [#10147](https://github.com/sgl-project/sglang/pull/10147) | merged | Fix: (glm4v) Add missing field | `python/sglang/srt/models/glm4v.py` |
| 2025-09-14 | [#10228](https://github.com/sgl-project/sglang/pull/10228) | merged | Add self.capture_aux_hidden_states For GLM-4.5V | `python/sglang/srt/models/glm4v_moe.py` |
| 2025-10-03 | [#11166](https://github.com/sgl-project/sglang/pull/11166) | merged | Tiny move files to utils folder | `test/srt/test_tokenizer_manager.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/configs/model_config.py` |
| 2025-10-10 | [#11388](https://github.com/sgl-project/sglang/pull/11388) | merged | Replace pad with cat for better performance | `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-10-22 | [#11922](https://github.com/sgl-project/sglang/pull/11922) | merged | [lint] improve ruff check | `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` |
| 2025-10-27 | [#12117](https://github.com/sgl-project/sglang/pull/12117) | merged | GLM-4-0414 and GLM-4.1V Code Refactor | `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-11-16 | [#13228](https://github.com/sgl-project/sglang/pull/13228) | merged | Cleanup vision attention related codes | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2025-11-18 | [#13126](https://github.com/sgl-project/sglang/pull/13126) | merged | [VLM][feat] Support encoder DP for Qwen2.5-VL | `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/layers/attention/vision.py` |
| 2025-11-28 | [#13724](https://github.com/sgl-project/sglang/pull/13724) | merged | support qwen3_vl vision model dp | `python/sglang/srt/models/qwen3_vl.py`, `test/nightly/test_encoder_dp.py` |
| 2025-12-05 | [#14097](https://github.com/sgl-project/sglang/pull/14097) | merged | support GLM-V vision model dp | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-12-08 | [#14662](https://github.com/sgl-project/sglang/pull/14662) | open | [Glm46v] support ktransformers | `python/sglang/srt/models/glm4v_moe.py` |
| 2025-12-10 | [#14720](https://github.com/sgl-project/sglang/pull/14720) | merged | [GLM-4.6V] Support Pipeline Parallelism for GLM-4.6V & GLM-4.1V | `python/sglang/srt/models/glm4v.py`, `test/srt/test_pp_single_node.py`, `python/sglang/test/test_utils.py` |
| 2025-12-12 | [#14927](https://github.com/sgl-project/sglang/pull/14927) | merged | [CI]add nightly CI for glm4v_moe arch model | `test/nightly/test_vlms_mmmu_eval.py` |
| 2025-12-13 | [#14998](https://github.com/sgl-project/sglang/pull/14998) | merged | add transformers version validation for glm-4.6v moe models | `python/sglang/srt/configs/model_config.py` |
| 2025-12-18 | [#15205](https://github.com/sgl-project/sglang/pull/15205) | merged | [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py` |
| 2026-01-04 | [#15434](https://github.com/sgl-project/sglang/pull/15434) | merged | Convert cu_seqlens to CPU for npu_flash_attention_unpad operator | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/paddleocr_vl.py` |
| 2026-01-27 | [#17582](https://github.com/sgl-project/sglang/pull/17582) | merged | [GLM-OCR] Support GLM-OCR Model | `python/sglang/srt/models/glm_ocr.py`, `python/sglang/srt/models/glm_ocr_nextn.py`, `python/sglang/srt/multimodal/processors/glm4v.py` |
| 2026-02-01 | [#17420](https://github.com/sgl-project/sglang/pull/17420) | merged | [VLM] Optimize get_rope_index for GLM4v | `python/sglang/srt/layers/rotary_embedding.py`, `benchmark/bench_rope/benchmark_rope_index.py` |
| 2026-02-16 | [#18885](https://github.com/sgl-project/sglang/pull/18885) | merged | Fix GLM-4V processor registration when glm_ocr is unavailable | `python/sglang/srt/multimodal/processors/glm4v.py` |
| 2026-03-03 | [#19728](https://github.com/sgl-project/sglang/pull/19728) | open | Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `test/registered/moe/test_fused_moe.py` |
| 2026-03-08 | [#20033](https://github.com/sgl-project/sglang/pull/20033) | merged | [VLM] Replace conv3d proj with linear for GLM4V | `python/sglang/srt/models/glm4v.py` |
| 2026-03-14 | [#20463](https://github.com/sgl-project/sglang/pull/20463) | merged | [Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr | `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py` |
| 2026-03-15 | [#20282](https://github.com/sgl-project/sglang/pull/20282) | merged | Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug | `python/sglang/srt/layers/conv.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/pixtral.py` |
| 2026-03-18 | [#20740](https://github.com/sgl-project/sglang/pull/20740) | merged | Revert "[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr" | `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py` |
| 2026-03-23 | [#21134](https://github.com/sgl-project/sglang/pull/21134) | merged | [Bug Fix] GLM-V / GLM-OCR: field detection for transformers 5.x and MTP omission fix | `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py` |
| 2026-03-29 | [#19788](https://github.com/sgl-project/sglang/pull/19788) | closed | [Feature] Optimizations for class Qwen3VLMoeVisionModel (Conv3d to Linear) in Qwen3VL | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `test/registered/vlm/test_qwen3vl_conv3d_to_linear.py` |
| 2026-03-30 | [#18559](https://github.com/sgl-project/sglang/pull/18559) | closed | Optimizations for Qwen3VL models | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/multimodal/processors/base_processor.py` |
| 2026-04-01 | [#17122](https://github.com/sgl-project/sglang/pull/17122) | merged | [bugfix]GLM-4V model | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py` |
| 2026-04-28 | [#22961](https://github.com/sgl-project/sglang/pull/22961) | merged | [NPU] Fix issue and support GLM-4.5V | `python/sglang/srt/models/glm4_moe.py` |

## Per-PR Diff Audit Cards

### PR #8015 - [WIP] Support glm4.1v

- Link: https://github.com/sgl-project/sglang/pull/8015
- Status/date: closed / 2025-08-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +1019/-12, 1136 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[WIP] Support glm4.1v"; model line: GLM VLM/OCR; category: model support/runtime entry; main diff: `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/multimodal/processors/glm4v.py`, `python/sglang/srt/layers/rotary_embedding.py`; technical summary: Covers "[WIP] Support glm4.1v"; the main implementation surface is `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/multimodal/processors/glm4v.py`, `python/sglang/srt/layers/rotary_embedding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v.py` added +552/-0 (552 lines); hunks: -0,0 +1,552; symbols: Glm4vRMSNorm, forward, Glm4vVisionMLP, __init__, touching `Glm4vRMSNorm, forward, Glm4vVisionMLP`; `python/sglang/srt/multimodal/processors/glm4v.py` added +185/-0 (185 lines); hunks: -0,0 +1,185; symbols: Glm4vImageProcessor, __init__, preprocess_video, process_mm_data_async, touching `Glm4vImageProcessor, __init__, preprocess_video`; `python/sglang/srt/layers/rotary_embedding.py` modified +175/-1 (176 lines); hunks: -1,6 +1,7; -946,7 +947,37 @@ def __init__(; symbols: __init__, forward, get_rope_index, get_rope_index_glm4v, touching `__init__, forward, get_rope_index`; `python/sglang/srt/multimodal/processors/base_processor.py` modified +9/-3 (12 lines); hunks: -22,13 +22,19 @@ class BaseMultiModalProcessorOutput:; symbols: BaseMultiModalProcessorOutput, organize_results, touching `BaseMultiModalProcessorOutput, organize_results`.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` added +552/-0 (552 lines); hunks: -0,0 +1,552; symbols: Glm4vRMSNorm, forward, Glm4vVisionMLP, __init__
  - `python/sglang/srt/multimodal/processors/glm4v.py` added +185/-0 (185 lines); hunks: -0,0 +1,185; symbols: Glm4vImageProcessor, __init__, preprocess_video, process_mm_data_async
  - `python/sglang/srt/layers/rotary_embedding.py` modified +175/-1 (176 lines); hunks: -1,6 +1,7; -946,7 +947,37 @@ def __init__(; symbols: __init__, forward, get_rope_index, get_rope_index_glm4v
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +9/-3 (12 lines); hunks: -22,13 +22,19 @@ class BaseMultiModalProcessorOutput:; symbols: BaseMultiModalProcessorOutput, organize_results
  - `python/sglang/srt/models/glm4.py` modified +6/-0 (6 lines); hunks: -218,6 +218,12 @@ def __init__(; symbols: __init__, get_input_embeddings, dtype, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -0,0 +1,552 @@
+import logging
+from functools import lru_cache, partial
+from typing import Iterable, List, Optional, Tuple
+import torch
+import torch.nn as nn
+import torch.nn.functional as F
diff -- python/sglang/srt/multimodal/processors/glm4v.py
@@ -0,0 +1,185 @@
+import re
+from typing import List, Union
+import torch
+from decord import VideoReader
+from transformers.video_utils import VideoMetadata
+from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -1,6 +1,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v.py` added +552/-0; `python/sglang/srt/multimodal/processors/glm4v.py` added +185/-0; `python/sglang/srt/layers/rotary_embedding.py` modified +175/-1; `python/sglang/srt/multimodal/processors/base_processor.py` modified +9/-3; `python/sglang/srt/models/glm4.py` modified +6/-0; `python/sglang/srt/configs/model_config.py` modified +1/-0
  - tests: `test/srt/test_jinja_template_utils.py` modified +40/-0; `test/srt/test_vision_openai_server_c.py` added +39/-0
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_jinja_template_utils.py`, `test/srt/test_vision_openai_server_c.py`, `test/srt/test_vision_openai_server_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8798 - Support glm4.1v and glm4.5v

- Link: https://github.com/sgl-project/sglang/pull/8798
- Status/date: merged / 2025-08-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 21 files, +1584/-19, 1822 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support glm4.1v and glm4.5v"; model line: GLM VLM/OCR; category: model support/runtime entry; main diff: `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/layers/rotary_embedding.py`; technical summary: Covers "Support glm4.1v and glm4.5v"; the main implementation surface is `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/layers/rotary_embedding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v.py` added +589/-0 (589 lines); hunks: -0,0 +1,589; symbols: Glm4vRMSNorm, forward, Glm4vVisionMLP, __init__, touching `Glm4vRMSNorm, forward, Glm4vVisionMLP`; `python/sglang/srt/models/glm4v_moe.py` added +400/-0 (400 lines); hunks: -0,0 +1,400; symbols: Glm4vMoeForConditionalGeneration, __init__, determine_num_fused_shared_experts, load_weights, touching `Glm4vMoeForConditionalGeneration, __init__, determine_num_fused_shared_experts`; `python/sglang/srt/layers/rotary_embedding.py` modified +230/-1 (231 lines); hunks: -1,6 +1,7; -946,7 +947,37 @@ def __init__(; symbols: __init__, forward, get_rope_index, get_rope_index_glm4v, touching `__init__, forward, get_rope_index`; `python/sglang/srt/multimodal/processors/glm4v.py` added +132/-0 (132 lines); hunks: -0,0 +1,132; symbols: Glm4vImageProcessor, __init__, preprocess_video, process_mm_data_async, touching `Glm4vImageProcessor, __init__, preprocess_video`.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` added +589/-0 (589 lines); hunks: -0,0 +1,589; symbols: Glm4vRMSNorm, forward, Glm4vVisionMLP, __init__
  - `python/sglang/srt/models/glm4v_moe.py` added +400/-0 (400 lines); hunks: -0,0 +1,400; symbols: Glm4vMoeForConditionalGeneration, __init__, determine_num_fused_shared_experts, load_weights
  - `python/sglang/srt/layers/rotary_embedding.py` modified +230/-1 (231 lines); hunks: -1,6 +1,7; -946,7 +947,37 @@ def __init__(; symbols: __init__, forward, get_rope_index, get_rope_index_glm4v
  - `python/sglang/srt/multimodal/processors/glm4v.py` added +132/-0 (132 lines); hunks: -0,0 +1,132; symbols: Glm4vImageProcessor, __init__, preprocess_video, process_mm_data_async
  - `test/srt/test_function_call_parser.py` modified +18/-2 (20 lines); hunks: -497,6 +497,17 @@ def setUp(self):; -630,16 +641,21 @@ def test_glm45_detector_ebnf(self):; symbols: setUp, test_glm45_detector_ebnf
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -0,0 +1,589 @@
+import logging
+from functools import lru_cache, partial
+from typing import Iterable, List, Optional, Tuple
+import torch
+import torch.nn as nn
+import torch.nn.functional as F
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -0,0 +1,400 @@
+import logging
+from functools import lru_cache
+from typing import Iterable, Optional, Tuple
+import torch
+import torch.nn as nn
+from transformers.models.glm4v_moe.configuration_glm4v_moe import Glm4vMoeConfig
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -1,6 +1,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v.py` added +589/-0; `python/sglang/srt/models/glm4v_moe.py` added +400/-0; `python/sglang/srt/layers/rotary_embedding.py` modified +230/-1; `python/sglang/srt/multimodal/processors/glm4v.py` added +132/-0; `python/sglang/srt/multimodal/processors/base_processor.py` modified +9/-3; `python/sglang/srt/models/glm4.py` modified +6/-0
  - tests: `test/srt/test_function_call_parser.py` modified +18/-2
- Risk and verification: The diff ships test coverage in `test/srt/openai_server/function_call/test_openai_function_calling.py`, `test/srt/test_function_call_parser.py`, `test/srt/test_jinja_template_utils.py`, `test/srt/test_vision_openai_server_b.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9245 - Set the default attention backend for GLM-4.5v to fa3

- Link: https://github.com/sgl-project/sglang/pull/9245
- Status/date: merged / 2025-08-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Set the default attention backend for GLM-4.5v to fa3"; model line: GLM VLM/OCR; category: model implementation change; main diff: `python/sglang/srt/utils.py`; technical summary: Covers "Set the default attention backend for GLM-4.5v to fa3"; the main implementation surface is `python/sglang/srt/utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/utils.py` modified +1/-0 (1 lines); hunks: -2343,6 +2343,7 @@ def is_fa3_default_architecture(hf_config):; symbols: is_fa3_default_architecture, touching `is_fa3_default_architecture`.
- Code diff details:
  - `python/sglang/srt/utils.py` modified +1/-0 (1 lines); hunks: -2343,6 +2343,7 @@ def is_fa3_default_architecture(hf_config):; symbols: is_fa3_default_architecture
- Key code excerpts:

```diff
diff -- python/sglang/srt/utils.py
@@ -2343,6 +2343,7 @@ def is_fa3_default_architecture(hf_config):
+        "Glm4vMoeForConditionalGeneration",
```

- Reviewed files:
  - runtime: `python/sglang/srt/utils.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9059 - [GLM4.1V and GLM4.5V] Add vision transformer num_dummy_head support: max tp=4 -> max tp=8

- Link: https://github.com/sgl-project/sglang/pull/9059
- Status/date: merged / 2025-08-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +150/-102, 423 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[GLM4.1V and GLM4.5V] Add vision transformer num_dummy_head support: max tp=4 -> max tp=8"; model line: GLM VLM/OCR; category: model support/runtime entry; main diff: `python/sglang/srt/layers/attention/vision_utils.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/internvl.py`; technical summary: Covers "[GLM4.1V and GLM4.5V] Add vision transformer num_dummy_head support: max tp=4 -> max tp=8"; the main implementation surface is `python/sglang/srt/layers/attention/vision_utils.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/vision_utils.py` added +65/-0 (65 lines); hunks: -0,0 +1,65; symbols: update_vit_attn_dummy_heads_config, pad_vit_attn_dummy_heads, touching `update_vit_attn_dummy_heads_config, pad_vit_attn_dummy_heads`; `python/sglang/srt/models/glm4v.py` modified +52/-1 (53 lines); hunks: -9,6 +9,7; -91,6 +92,7 @@ def __init__(; symbols: __init__, get_video_feature, _update_hf_config, _pad_vit_attn_dummy_heads, touching `__init__, get_video_feature, _update_hf_config`; `python/sglang/srt/models/internvl.py` modified +4/-49 (53 lines); hunks: -10,7 +10,7; -412,7 +412,7 @@ def __init__(; symbols: __init__, _update_vision_config, pixel_shuffle, pad_input_ids, touching `__init__, _update_vision_config, pixel_shuffle`; `python/sglang/srt/models/interns1.py` modified +5/-46 (51 lines); hunks: -4,7 +4,7; -35,7 +35,7 @@ def __init__(; symbols: __init__, _update_hf_config, pixel_shuffle, pad_input_ids, touching `__init__, _update_hf_config, pixel_shuffle`.
- Code diff details:
  - `python/sglang/srt/layers/attention/vision_utils.py` added +65/-0 (65 lines); hunks: -0,0 +1,65; symbols: update_vit_attn_dummy_heads_config, pad_vit_attn_dummy_heads
  - `python/sglang/srt/models/glm4v.py` modified +52/-1 (53 lines); hunks: -9,6 +9,7; -91,6 +92,7 @@ def __init__(; symbols: __init__, get_video_feature, _update_hf_config, _pad_vit_attn_dummy_heads
  - `python/sglang/srt/models/internvl.py` modified +4/-49 (53 lines); hunks: -10,7 +10,7; -412,7 +412,7 @@ def __init__(; symbols: __init__, _update_vision_config, pixel_shuffle, pad_input_ids
  - `python/sglang/srt/models/interns1.py` modified +5/-46 (51 lines); hunks: -4,7 +4,7; -35,7 +35,7 @@ def __init__(; symbols: __init__, _update_hf_config, pixel_shuffle, pad_input_ids
  - `python/sglang/srt/models/glm4v_moe.py` modified +6/-0 (6 lines); hunks: -11,6 +11,7; -40,6 +41,7 @@ def __init__(; symbols: __init__, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/vision_utils.py
@@ -0,0 +1,65 @@
+"""Utility functions for vision attention layers."""
+import torch
+from sglang.srt.layers.dp_attention import get_attention_tp_size
+def update_vit_attn_dummy_heads_config(config):
+    """Update HF config to ensure vision attention num_attention_heads is divisible by tp_size"""
+    tp_size = get_attention_tp_size()
diff -- python/sglang/srt/models/glm4v.py
@@ -9,6 +9,7 @@
+from sglang.srt.layers.attention import vision_utils
@@ -91,6 +92,7 @@ def __init__(
+            num_dummy_heads=config.num_dummy_heads,
@@ -469,7 +471,7 @@ def __init__(
+        vision_utils.update_vit_attn_dummy_heads_config(self.config)
@@ -537,6 +539,51 @@ def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
diff -- python/sglang/srt/models/internvl.py
@@ -10,7 +10,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/vision_utils.py` added +65/-0; `python/sglang/srt/models/glm4v.py` modified +52/-1; `python/sglang/srt/models/internvl.py` modified +4/-49; `python/sglang/srt/models/interns1.py` modified +5/-46; `python/sglang/srt/models/glm4v_moe.py` modified +6/-0; `python/sglang/srt/models/qwen2_5_vl.py` modified +2/-0
  - other: `benchmark/mmmu/bench_hf.py` modified +6/-2; `benchmark/mmmu/bench_sglang.py` modified +6/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/vision_utils.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4v_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9349 - Add support for GLM 4.5V FP8

- Link: https://github.com/sgl-project/sglang/pull/9349
- Status/date: open / 2025-08-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +153/-4, 165 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add support for GLM 4.5V FP8"; model line: GLM VLM/OCR; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`; technical summary: Covers "Add support for GLM 4.5V FP8"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146; `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +7/-4 (11 lines); hunks: -441,10 +441,13 @@ def main(args: argparse.Namespace):; symbols: main, touching `main`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +7/-4 (11 lines); hunks: -441,10 +441,13 @@ def main(args: argparse.Namespace):; symbols: main
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
diff -- benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py
@@ -441,10 +441,13 @@ def main(args: argparse.Namespace):
-    elif config.architectures[0] in ["Glm4MoeForCausalLM"]:
-        E = config.n_routed_experts
-        topk = config.num_experts_per_tok
-        intermediate_size = config.moe_intermediate_size
+    elif config.architectures[0] in ["Glm4MoeForCausalLM", "Glm4vMoeForConditionalGeneration"]:
+        cfg_source = config
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json` added +146/-0
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +7/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9554 - Fix GLM45v launch server cuda torch compile bug

- Link: https://github.com/sgl-project/sglang/pull/9554
- Status/date: merged / 2025-08-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix GLM45v launch server cuda torch compile bug"; model line: GLM VLM/OCR; category: bug fix; main diff: `python/sglang/srt/models/qwen2_5_vl.py`; technical summary: Covers "Fix GLM45v launch server cuda torch compile bug"; the main implementation surface is `python/sglang/srt/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0 (1 lines); hunks: -526,6 +526,7 @@ def get_video_feature(self, items: List[MultimodalDataItem])...; symbols: get_video_feature, get_input_embeddings, forward, touching `get_video_feature, get_input_embeddings, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0 (1 lines); hunks: -526,6 +526,7 @@ def get_video_feature(self, items: List[MultimodalDataItem])...; symbols: get_video_feature, get_input_embeddings, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -526,6 +526,7 @@ def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
+    @torch.no_grad()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9884 - [Bug Fix] Fix Glm4vVisionBlock norm

- Link: https://github.com/sgl-project/sglang/pull/9884
- Status/date: merged / 2025-09-05
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glm4v.py`; associated commits `0f6ac5e21db0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-4, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug Fix] Fix Glm4vVisionBlock norm"; model line: GLM VLM/OCR; category: bug fix; main diff: `python/sglang/srt/models/glm4v.py`; technical summary: Covers "[Bug Fix] Fix Glm4vVisionBlock norm"; the main implementation surface is `python/sglang/srt/models/glm4v.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v.py` modified +1/-2 (3 lines); hunks: -93,9 +93,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` modified +1/-2 (3 lines); hunks: -93,9 +93,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -93,9 +93,8 @@ def __init__(
+            rms_norm_eps=config.rms_norm_eps,
-        self.norm1 = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
-        self.norm2 = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10147 - Fix: (glm4v) Add missing field

- Link: https://github.com/sgl-project/sglang/pull/10147
- Status/date: merged / 2025-09-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glm4v.py`; associated commits `8116804e4f6e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix: (glm4v) Add missing field"; model line: GLM VLM/OCR; category: bug fix; main diff: `python/sglang/srt/models/glm4v.py`; technical summary: Covers "Fix: (glm4v) Add missing field"; the main implementation surface is `python/sglang/srt/models/glm4v.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v.py` modified +3/-0 (3 lines); hunks: -497,6 +497,9 @@ def __init__(; symbols: __init__, get_image_feature, touching `__init__, get_image_feature`.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` modified +3/-0 (3 lines); hunks: -497,6 +497,9 @@ def __init__(; symbols: __init__, get_image_feature
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -497,6 +497,9 @@ def __init__(
+        # For EAGLE3 support
+        self.capture_aux_hidden_states = False
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4v.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10228 - Add self.capture_aux_hidden_states For GLM-4.5V

- Link: https://github.com/sgl-project/sglang/pull/10228
- Status/date: merged / 2025-09-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add self.capture_aux_hidden_states For GLM-4.5V"; model line: GLM VLM/OCR; category: model support/runtime entry; main diff: `python/sglang/srt/models/glm4v_moe.py`; technical summary: Covers "Add self.capture_aux_hidden_states For GLM-4.5V"; the main implementation surface is `python/sglang/srt/models/glm4v_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v_moe.py` modified +3/-0 (3 lines); hunks: -74,6 +74,9 @@ def __init__(; symbols: __init__, determine_num_fused_shared_experts, touching `__init__, determine_num_fused_shared_experts`.
- Code diff details:
  - `python/sglang/srt/models/glm4v_moe.py` modified +3/-0 (3 lines); hunks: -74,6 +74,9 @@ def __init__(; symbols: __init__, determine_num_fused_shared_experts
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -74,6 +74,9 @@ def __init__(
+        # For EAGLE3 support
+        self.capture_aux_hidden_states = False
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v_moe.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4v_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11166 - Tiny move files to utils folder

- Link: https://github.com/sgl-project/sglang/pull/11166
- Status/date: merged / 2025-10-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 66 files, +91/-79, 794 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Tiny move files to utils folder"; model line: GLM VLM/OCR; category: model implementation change; main diff: `test/srt/test_tokenizer_manager.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/configs/model_config.py`; technical summary: Covers "Tiny move files to utils folder"; the main implementation surface is `test/srt/test_tokenizer_manager.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/configs/model_config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/srt/test_tokenizer_manager.py` modified +12/-4 (16 lines); hunks: -31,7 +31,9 @@ def setUp(self):; -125,7 +127,9 @@ def setUp(self):; symbols: setUp, touching `setUp`; `python/sglang/srt/managers/tokenizer_manager.py` modified +5/-5 (10 lines); hunks: -43,11 +43,6; -99,6 +94,11; `python/sglang/srt/configs/model_config.py` modified +4/-4 (8 lines); hunks: -23,16 +23,16; `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: -29,7 +29,6; -115,7 +114,6.
- Code diff details:
  - `test/srt/test_tokenizer_manager.py` modified +12/-4 (16 lines); hunks: -31,7 +31,9 @@ def setUp(self):; -125,7 +127,9 @@ def setUp(self):; symbols: setUp
  - `python/sglang/srt/managers/tokenizer_manager.py` modified +5/-5 (10 lines); hunks: -43,11 +43,6; -99,6 +94,11
  - `python/sglang/srt/configs/model_config.py` modified +4/-4 (8 lines); hunks: -23,16 +23,16
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: -29,7 +29,6; -115,7 +114,6
  - `test/srt/test_tokenizer_batch_encode.py` modified +3/-1 (4 lines); hunks: -34,7 +34,9 @@ def setUp(self):; symbols: setUp
- Key code excerpts:

```diff
diff -- test/srt/test_tokenizer_manager.py
@@ -31,7 +31,9 @@ def setUp(self):
-        ), patch("sglang.srt.hf_transformers_utils.get_tokenizer") as mock_tokenizer:
+        ), patch(
+            "sglang.srt.utils.hf_transformers_utils.get_tokenizer"
+        ) as mock_tokenizer:
@@ -125,7 +127,9 @@ def setUp(self):
-        ), patch("sglang.srt.hf_transformers_utils.get_tokenizer") as mock_tokenizer:
diff -- python/sglang/srt/managers/tokenizer_manager.py
@@ -43,11 +43,6 @@
-from sglang.srt.hf_transformers_utils import (
-    get_processor,
-    get_tokenizer,
-    get_tokenizer_from_processor,
-)
@@ -99,6 +94,11 @@
diff -- python/sglang/srt/configs/model_config.py
@@ -23,16 +23,16 @@
```

- Reviewed files:
  - tests: `test/srt/test_tokenizer_manager.py` modified +12/-4; `test/srt/test_tokenizer_batch_encode.py` modified +3/-1
  - runtime: `python/sglang/srt/managers/tokenizer_manager.py` modified +5/-5; `python/sglang/srt/configs/model_config.py` modified +4/-4; `python/sglang/srt/model_executor/model_runner.py` modified +2/-2; `python/sglang/srt/managers/detokenizer_manager.py` modified +1/-1; `python/sglang/srt/model_executor/cpu_graph_runner.py` modified +1/-1; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/runners.py`, `python/sglang/test/test_programs.py`, `python/sglang/test/test_utils.py`, `test/srt/openai_server/basic/test_openai_server.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11388 - Replace pad with cat for better performance

- Link: https://github.com/sgl-project/sglang/pull/11388
- Status/date: merged / 2025-10-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +5/-5, 45 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Replace pad with cat for better performance"; model line: GLM VLM/OCR; category: performance/backend optimization; main diff: `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`; technical summary: Covers "Replace pad with cat for better performance"; the main implementation surface is `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/dots_vlm_vit.py` modified +1/-1 (2 lines); hunks: -323,7 +323,7 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/glm4v.py` modified +1/-1 (2 lines); hunks: -434,7 +434,7 @@ def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -...; symbols: forward, touching `forward`; `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-1 (2 lines); hunks: -436,7 +436,7 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/qwen2_vl.py` modified +1/-1 (2 lines); hunks: -407,7 +407,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/dots_vlm_vit.py` modified +1/-1 (2 lines); hunks: -323,7 +323,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/glm4v.py` modified +1/-1 (2 lines); hunks: -434,7 +434,7 @@ def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -...; symbols: forward
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-1 (2 lines); hunks: -436,7 +436,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen2_vl.py` modified +1/-1 (2 lines); hunks: -407,7 +407,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-1 (2 lines); hunks: -458,7 +458,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/dots_vlm_vit.py
@@ -323,7 +323,7 @@ def forward(
-        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
+        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
diff -- python/sglang/srt/models/glm4v.py
@@ -434,7 +434,7 @@ def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
-        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
+        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -436,7 +436,7 @@ def forward(
-        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
+        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
diff -- python/sglang/srt/models/qwen2_vl.py
@@ -407,7 +407,7 @@ def forward(
-        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
+        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -458,7 +458,7 @@ def forward(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/dots_vlm_vit.py` modified +1/-1; `python/sglang/srt/models/glm4v.py` modified +1/-1; `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-1; `python/sglang/srt/models/qwen2_vl.py` modified +1/-1; `python/sglang/srt/models/qwen3_vl.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11922 - [lint] improve ruff check

- Link: https://github.com/sgl-project/sglang/pull/11922
- Status/date: merged / 2025-10-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +73/-31, 354 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[lint] improve ruff check"; model line: GLM VLM/OCR; category: bug fix; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`; technical summary: Covers "[lint] improve ruff check"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` modified +20/-19 (39 lines); hunks: -13,7 +13,8; -119,14 +120,14 @@ def triton_kernel_fused_experts(; symbols: triton_kernel_fused_experts, triton_kernel_fused_experts_with_bias, touching `triton_kernel_fused_experts, triton_kernel_fused_experts_with_bias`; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +7/-0 (7 lines); hunks: -44,6 +44,13; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +4/-1 (5 lines); hunks: -38,6 +38,9; -66,7 +69,7 @@ class PrefillMetadata:; symbols: PrefillMetadata, FlashInferMhaChunkKVRunner, __init__, touching `PrefillMetadata, FlashInferMhaChunkKVRunner, __init__`; `python/sglang/srt/models/opt.py` modified +4/-0 (4 lines); hunks: -13,6 +13,7; -46,6 +47,9; symbols: get_activation, touching `get_activation`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` modified +20/-19 (39 lines); hunks: -13,7 +13,8; -119,14 +120,14 @@ def triton_kernel_fused_experts(; symbols: triton_kernel_fused_experts, triton_kernel_fused_experts_with_bias
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +7/-0 (7 lines); hunks: -44,6 +44,13
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +4/-1 (5 lines); hunks: -38,6 +38,9; -66,7 +69,7 @@ class PrefillMetadata:; symbols: PrefillMetadata, FlashInferMhaChunkKVRunner, __init__
  - `python/sglang/srt/models/opt.py` modified +4/-0 (4 lines); hunks: -13,6 +13,7; -46,6 +47,9; symbols: get_activation
  - `python/sglang/srt/entrypoints/openai/serving_responses.py` modified +2/-1 (3 lines); hunks: -14,6 +14,7; -1061,7 +1062,7 @@ def _send_event(event):; symbols: _send_event
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py
@@ -13,7 +13,8 @@
-from triton_kernels.numerics import InFlexData
+from triton_kernels.numerics import InFlexData, MicroscalingCtx
+from triton_kernels.quantization import downcast_to_mxfp
@@ -119,14 +120,14 @@ def triton_kernel_fused_experts(
-    assert use_fp8_w8a8 == False, "use_fp8_w8a8 is not supported"
-    assert per_channel_quant == False, "per_channel_quant is not supported"
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py
@@ -44,6 +44,13 @@
+    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_24 import (
+        CompressedTensors24,
+    )
+    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_sparse24 import (
+        W4A16SPARSE24_SUPPORTED_BITS,
+        CompressedTensorsW4A16Sparse24,
diff -- python/sglang/srt/layers/attention/flashinfer_mla_backend.py
@@ -38,6 +38,9 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` modified +20/-19; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +7/-0; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +4/-1; `python/sglang/srt/models/opt.py` modified +4/-0; `python/sglang/srt/entrypoints/openai/serving_responses.py` modified +2/-1; `python/sglang/srt/models/deepseek_v2.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/few_shot_gsm8k_engine.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12117 - GLM-4-0414 and GLM-4.1V Code Refactor

- Link: https://github.com/sgl-project/sglang/pull/12117
- Status/date: merged / 2025-10-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +679/-173, 1187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "GLM-4-0414 and GLM-4.1V Code Refactor"; model line: GLM VLM/OCR; category: model implementation change; main diff: `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/layers/rotary_embedding.py`; technical summary: Covers "GLM-4-0414 and GLM-4.1V Code Refactor"; the main implementation surface is `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/layers/rotary_embedding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4.py` modified +391/-77 (468 lines); hunks: -15,46 +15,119; -63,27 +136,30 @@ def __init__(; symbols: Glm4MLP, __init__, forward, Glm4Attention, touching `Glm4MLP, __init__, forward`; `python/sglang/srt/models/glm4v.py` modified +196/-55 (251 lines); hunks: -1,15 +1,35; -20,13 +40,14; symbols: __init__, forward, Glm4vVisionBlock, touching `__init__, forward, Glm4vVisionBlock`; `python/sglang/srt/layers/rotary_embedding.py` modified +92/-40 (132 lines); hunks: -1070,6 +1070,7 @@ def _triton_mrope_forward(; -1124,51 +1125,99 @@ def _triton_mrope_forward(; symbols: _triton_mrope_forward, triton_mrope, touching `_triton_mrope_forward, triton_mrope`; `python/sglang/srt/models/glm4v_moe.py` modified +0/-1 (1 lines); hunks: -53,7 +53,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/glm4.py` modified +391/-77 (468 lines); hunks: -15,46 +15,119; -63,27 +136,30 @@ def __init__(; symbols: Glm4MLP, __init__, forward, Glm4Attention
  - `python/sglang/srt/models/glm4v.py` modified +196/-55 (251 lines); hunks: -1,15 +1,35; -20,13 +40,14; symbols: __init__, forward, Glm4vVisionBlock
  - `python/sglang/srt/layers/rotary_embedding.py` modified +92/-40 (132 lines); hunks: -1070,6 +1070,7 @@ def _triton_mrope_forward(; -1124,51 +1125,99 @@ def _triton_mrope_forward(; symbols: _triton_mrope_forward, triton_mrope
  - `python/sglang/srt/models/glm4v_moe.py` modified +0/-1 (1 lines); hunks: -53,7 +53,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4.py
@@ -15,46 +15,119 @@
-"""Inference-only GLM4 model compatible with THUDM weights."""
+"""Inference-only GLM-4-0414 model compatible with HuggingFace weights."""
-from typing import Iterable, List, Optional, Tuple, Union
+import logging
+from typing import Any, Dict, Iterable, Optional, Tuple, Union
-from transformers import Glm4Config
diff -- python/sglang/srt/models/glm4v.py
@@ -1,15 +1,35 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -1070,6 +1070,7 @@ def _triton_mrope_forward(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4.py` modified +391/-77; `python/sglang/srt/models/glm4v.py` modified +196/-55; `python/sglang/srt/layers/rotary_embedding.py` modified +92/-40; `python/sglang/srt/models/glm4v_moe.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4v.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13228 - Cleanup vision attention related codes

- Link: https://github.com/sgl-project/sglang/pull/13228
- Status/date: merged / 2025-11-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +4/-142, 378 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Cleanup vision attention related codes"; model line: GLM VLM/OCR; category: performance/backend optimization; main diff: `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "Cleanup vision attention related codes"; the main implementation surface is `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v.py` modified +1/-26 (27 lines); hunks: -104,7 +104,6 @@ def __init__(; -114,37 +113,13 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-26 (27 lines); hunks: -111,7 +111,6 @@ def __init__(; -121,37 +120,13 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/qwen3_vl.py` modified +1/-23 (24 lines); hunks: -130,7 +130,6 @@ def __init__(; -140,33 +139,13 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/clip.py` modified +0/-13 (13 lines); hunks: -141,7 +141,6 @@ def __init__(; -150,22 +149,11 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` modified +1/-26 (27 lines); hunks: -104,7 +104,6 @@ def __init__(; -114,37 +113,13 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-26 (27 lines); hunks: -111,7 +111,6 @@ def __init__(; -121,37 +120,13 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-23 (24 lines); hunks: -130,7 +130,6 @@ def __init__(; -140,33 +139,13 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/clip.py` modified +0/-13 (13 lines); hunks: -141,7 +141,6 @@ def __init__(; -150,22 +149,11 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen2_vl.py` modified +0/-13 (13 lines); hunks: -127,7 +127,6 @@ def __init__(; -137,23 +136,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -104,7 +104,6 @@ def __init__(
-        attn_implementation: Optional[str] = None,
@@ -114,37 +113,13 @@ def __init__(
-        if attn_implementation is None:
-            softmax_in_single_precision = False
-            qkv_backend = None
-            flatten_batch = True
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -111,7 +111,6 @@ def __init__(
-        attn_implementation: Optional[str] = None,
@@ -121,37 +120,13 @@ def __init__(
-        if attn_implementation is None:
-            softmax_in_single_precision = False
-            qkv_backend = None
-            flatten_batch = True
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -130,7 +130,6 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +1/-26; `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-26; `python/sglang/srt/models/qwen3_vl.py` modified +1/-23; `python/sglang/srt/models/clip.py` modified +0/-13; `python/sglang/srt/models/qwen2_vl.py` modified +0/-13; `python/sglang/srt/models/siglip.py` modified +0/-13
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/clip.py`, `python/sglang/srt/models/deepseek_janus_pro.py`, `python/sglang/srt/models/dots_vlm_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13126 - [VLM][feat] Support encoder DP for Qwen2.5-VL

- Link: https://github.com/sgl-project/sglang/pull/13126
- Status/date: merged / 2025-11-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +595/-6, 790 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM][feat] Support encoder DP for Qwen2.5-VL"; model line: GLM VLM/OCR; category: model support/runtime entry; main diff: `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/layers/attention/vision.py`; technical summary: Covers "[VLM][feat] Support encoder DP for Qwen2.5-VL"; the main implementation surface is `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/layers/attention/vision.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/mm_utils.py` modified +268/-0 (268 lines); hunks: -28,14 +28,22; -347,3 +355,263 @@ def process_images(images, image_processor, model_cfg):; symbols: process_images, get_dp_encoder_lb_assignment, run_dp_sharded_mrope_vision_model, touching `process_images, get_dp_encoder_lb_assignment, run_dp_sharded_mrope_vision_model`; `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-2 (46 lines); hunks: -40,6 +40,10; -62,6 +66,8; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/layers/attention/vision.py` modified +3/-4 (7 lines); hunks: -486,13 +486,12 @@ def __init__(; symbols: __init__, touching `__init__`; `test/srt/nightly/test_encoder_dp.py` added +270/-0 (270 lines); hunks: -0,0 +1,270; symbols: TestVLMEncoderDP, setUpClass, run_mmmu_eval, _run_vlm_mmmu_test, touching `TestVLMEncoderDP, setUpClass, run_mmmu_eval`.
- Code diff details:
  - `python/sglang/srt/multimodal/mm_utils.py` modified +268/-0 (268 lines); hunks: -28,14 +28,22; -347,3 +355,263 @@ def process_images(images, image_processor, model_cfg):; symbols: process_images, get_dp_encoder_lb_assignment, run_dp_sharded_mrope_vision_model
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-2 (46 lines); hunks: -40,6 +40,10; -62,6 +66,8; symbols: __init__, forward
  - `python/sglang/srt/layers/attention/vision.py` modified +3/-4 (7 lines); hunks: -486,13 +486,12 @@ def __init__(; symbols: __init__
  - `test/srt/nightly/test_encoder_dp.py` added +270/-0 (270 lines); hunks: -0,0 +1,270; symbols: TestVLMEncoderDP, setUpClass, run_mmmu_eval, _run_vlm_mmmu_test
  - `python/sglang/srt/server_args.py` modified +9/-0 (9 lines); hunks: -578,6 +578,9 @@ class ServerArgs:; -3728,6 +3731,12 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/mm_utils.py
@@ -28,14 +28,22 @@
+import itertools
+from typing import Literal
+import torch
+from sglang.srt.distributed import (
+    get_tensor_model_parallel_rank,
+    get_tensor_model_parallel_world_size,
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -40,6 +40,10 @@
+from sglang.srt.distributed import (
+    get_tensor_model_parallel_rank,
+    get_tensor_model_parallel_world_size,
+)
@@ -62,6 +66,8 @@
+from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
diff -- python/sglang/srt/layers/attention/vision.py
@@ -486,13 +486,12 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/mm_utils.py` modified +268/-0; `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-2; `python/sglang/srt/layers/attention/vision.py` modified +3/-4; `python/sglang/srt/server_args.py` modified +9/-0
  - tests: `test/srt/nightly/test_encoder_dp.py` added +270/-0; `test/srt/run_suite.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/srt/nightly/test_encoder_dp.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13724 - support qwen3_vl vision model dp

- Link: https://github.com/sgl-project/sglang/pull/13724
- Status/date: merged / 2025-11-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +50/-2, 208 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support qwen3_vl vision model dp"; model line: GLM VLM/OCR; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3_vl.py`, `test/nightly/test_encoder_dp.py`; technical summary: Covers "support qwen3_vl vision model dp"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `test/nightly/test_encoder_dp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +49/-2 (51 lines); hunks: -28,6 +28,10; -47,6 +51,8; symbols: __init__, forward, touching `__init__, forward`; `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: -19,6 +19,7.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +49/-2 (51 lines); hunks: -28,6 +28,10; -47,6 +51,8; symbols: __init__, forward
  - `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: -19,6 +19,7
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -28,6 +28,10 @@
+from sglang.srt.distributed import (
+    get_tensor_model_parallel_rank,
+    get_tensor_model_parallel_world_size,
+)
@@ -47,6 +51,8 @@
+from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
diff -- test/nightly/test_encoder_dp.py
@@ -19,6 +19,7 @@
+    SimpleNamespace(model="Qwen/Qwen3-VL-32B-Instruct", mmmu_accuracy=0.55),
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +49/-2
  - tests: `test/nightly/test_encoder_dp.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/nightly/test_encoder_dp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14097 - support GLM-V vision model dp

- Link: https://github.com/sgl-project/sglang/pull/14097
- Status/date: merged / 2025-12-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +91/-52, 329 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support GLM-V vision model dp"; model line: GLM VLM/OCR; category: model support/runtime entry; main diff: `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4_moe.py`; technical summary: Covers "support GLM-V vision model dp"; the main implementation surface is `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v.py` modified +84/-50 (134 lines); hunks: -27,18 +27,24; -48,6 +54,8; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/models/glm4.py` modified +3/-1 (4 lines); hunks: -220,7 +220,9 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: -682,7 +682,9 @@ def __init__(; symbols: __init__, touching `__init__`; `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: -24,6 +24,7.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` modified +84/-50 (134 lines); hunks: -27,18 +27,24; -48,6 +54,8; symbols: __init__, forward
  - `python/sglang/srt/models/glm4.py` modified +3/-1 (4 lines); hunks: -220,7 +220,9 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: -682,7 +682,9 @@ def __init__(; symbols: __init__
  - `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: -24,6 +24,7
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -27,18 +27,24 @@
+from sglang.srt.distributed import (
+    get_tensor_model_parallel_rank,
+    get_tensor_model_parallel_world_size,
+)
+from sglang.srt.distributed.parallel_state import get_pp_group
-from sglang.srt.layers.layernorm import RMSNorm
diff -- python/sglang/srt/models/glm4.py
@@ -220,7 +220,9 @@ def __init__(
-        partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
+        partial_rotary_factor = getattr(
+            getattr(config, "rope_parameters", None), "partial_rotary_factor", None
+        ) or getattr(config, "partial_rotary_factor", 0.5)
diff -- python/sglang/srt/models/glm4_moe.py
@@ -682,7 +682,9 @@ def __init__(
-        partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
+        partial_rotary_factor = getattr(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +84/-50; `python/sglang/srt/models/glm4.py` modified +3/-1; `python/sglang/srt/models/glm4_moe.py` modified +3/-1
  - tests: `test/nightly/test_encoder_dp.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/nightly/test_encoder_dp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14662 - [Glm46v] support ktransformers

- Link: https://github.com/sgl-project/sglang/pull/14662
- Status/date: open / 2025-12-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-0, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Glm46v] support ktransformers"; model line: GLM VLM/OCR; category: model support/runtime entry; main diff: `python/sglang/srt/models/glm4v_moe.py`; technical summary: Covers "[Glm46v] support ktransformers"; the main implementation surface is `python/sglang/srt/models/glm4v_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v_moe.py` modified +8/-0 (8 lines); hunks: -16,6 +16,7; -281,5 +282,12 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, get_model_config_for_expert_location, touching `load_weights, get_model_config_for_expert_location`.
- Code diff details:
  - `python/sglang/srt/models/glm4v_moe.py` modified +8/-0 (8 lines); hunks: -16,6 +16,7; -281,5 +282,12 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, get_model_config_for_expert_location
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -16,6 +16,7 @@
+from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
@@ -281,5 +282,12 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
+    @classmethod
+    def get_model_config_for_expert_location(cls, config):
+        return ModelConfigForExpertLocation(
+            num_layers=config.text_config.num_hidden_layers,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v_moe.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4v_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14720 - [GLM-4.6V] Support Pipeline Parallelism for GLM-4.6V & GLM-4.1V

- Link: https://github.com/sgl-project/sglang/pull/14720
- Status/date: merged / 2025-12-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +66/-2, 144 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[GLM-4.6V] Support Pipeline Parallelism for GLM-4.6V & GLM-4.1V"; model line: GLM VLM/OCR; category: model support/runtime entry; main diff: `python/sglang/srt/models/glm4v.py`, `test/srt/test_pp_single_node.py`, `python/sglang/test/test_utils.py`; technical summary: Covers "[GLM-4.6V] Support Pipeline Parallelism for GLM-4.6V & GLM-4.1V"; the main implementation surface is `python/sglang/srt/models/glm4v.py`, `test/srt/test_pp_single_node.py`, `python/sglang/test/test_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v.py` modified +24/-1 (25 lines); hunks: -51,7 +51,7; -659,6 +659,7 @@ def forward(; symbols: forward, load_weights, touching `forward, load_weights`; `test/srt/test_pp_single_node.py` modified +38/-0 (38 lines); hunks: -19,6 +19,7; -318,5 +319,42 @@ def test_chunked_prefill_with_small_bs(self):; symbols: test_chunked_prefill_with_small_bs, TestGLM41VPPAccuracy, setUpClass, tearDownClass, touching `test_chunked_prefill_with_small_bs, TestGLM41VPPAccuracy, setUpClass`; `python/sglang/test/test_utils.py` modified +3/-0 (3 lines); hunks: -57,6 +57,9; `test/srt/run_suite.py` modified +1/-1 (2 lines); hunks: -150,7 +150,7.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` modified +24/-1 (25 lines); hunks: -51,7 +51,7; -659,6 +659,7 @@ def forward(; symbols: forward, load_weights
  - `test/srt/test_pp_single_node.py` modified +38/-0 (38 lines); hunks: -19,6 +19,7; -318,5 +319,42 @@ def test_chunked_prefill_with_small_bs(self):; symbols: test_chunked_prefill_with_small_bs, TestGLM41VPPAccuracy, setUpClass, tearDownClass
  - `python/sglang/test/test_utils.py` modified +3/-0 (3 lines); hunks: -57,6 +57,9
  - `test/srt/run_suite.py` modified +1/-1 (2 lines); hunks: -150,7 +150,7
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -51,7 +51,7 @@
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch
+from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
@@ -659,6 +659,7 @@ def forward(
+        pp_proxy_tensors: Optional[PPProxyTensors] = None,
@@ -691,6 +692,7 @@ def forward(
+            pp_proxy_tensors=pp_proxy_tensors,
diff -- test/srt/test_pp_single_node.py
@@ -19,6 +19,7 @@
+    DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP,
@@ -318,5 +319,42 @@ def test_chunked_prefill_with_small_bs(self):
+class TestGLM41VPPAccuracy(unittest.TestCase):
+    @classmethod
+    def setUpClass(cls):
+        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP
diff -- python/sglang/test/test_utils.py
@@ -57,6 +57,9 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +24/-1
  - tests: `test/srt/test_pp_single_node.py` modified +38/-0; `python/sglang/test/test_utils.py` modified +3/-0; `test/srt/run_suite.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_utils.py`, `test/srt/run_suite.py`, `test/srt/test_pp_single_node.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14927 - [CI]add nightly CI for glm4v_moe arch model

- Link: https://github.com/sgl-project/sglang/pull/14927
- Status/date: merged / 2025-12-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI]add nightly CI for glm4v_moe arch model"; model line: GLM VLM/OCR; category: bug fix; main diff: `test/nightly/test_vlms_mmmu_eval.py`; technical summary: Covers "[CI]add nightly CI for glm4v_moe arch model"; the main implementation surface is `test/nightly/test_vlms_mmmu_eval.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/nightly/test_vlms_mmmu_eval.py` modified +3/-0 (3 lines); hunks: -46,6 +46,9.
- Code diff details:
  - `test/nightly/test_vlms_mmmu_eval.py` modified +3/-0 (3 lines); hunks: -46,6 +46,9
- Key code excerpts:

```diff
diff -- test/nightly/test_vlms_mmmu_eval.py
@@ -46,6 +46,9 @@
+    ModelLaunchSettings(
+        "zai-org/GLM-4.5V-FP8", extra_args=["--tp=2"]
+    ): ModelEvalMetrics(0.26, 32.0),
```

- Reviewed files:
  - tests: `test/nightly/test_vlms_mmmu_eval.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `test/nightly/test_vlms_mmmu_eval.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14998 - add transformers version validation for glm-4.6v moe models

- Link: https://github.com/sgl-project/sglang/pull/14998
- Status/date: merged / 2025-12-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +37/-0, 51 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "add transformers version validation for glm-4.6v moe models"; model line: GLM VLM/OCR; category: model support/runtime entry; main diff: `python/sglang/srt/configs/model_config.py`; technical summary: Covers "add transformers version validation for glm-4.6v moe models"; the main implementation surface is `python/sglang/srt/configs/model_config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/configs/model_config.py` modified +37/-0 (37 lines); hunks: -205,6 +205,8 @@ def __init__(; -773,6 +775,41 @@ def _verify_dual_chunk_attention_config(self) -> None:; symbols: __init__, _verify_dual_chunk_attention_config, _verify_transformers_version, _get_hf_eos_token_id, touching `__init__, _verify_dual_chunk_attention_config, _verify_transformers_version`.
- Code diff details:
  - `python/sglang/srt/configs/model_config.py` modified +37/-0 (37 lines); hunks: -205,6 +205,8 @@ def __init__(; -773,6 +775,41 @@ def _verify_dual_chunk_attention_config(self) -> None:; symbols: __init__, _verify_dual_chunk_attention_config, _verify_transformers_version, _get_hf_eos_token_id
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/model_config.py
@@ -205,6 +205,8 @@ def __init__(
+        self._verify_transformers_version()
@@ -773,6 +775,41 @@ def _verify_dual_chunk_attention_config(self) -> None:
+    def _verify_transformers_version(self):
+        import transformers
+        from packaging import version
+        tf_version_str = getattr(transformers, "__version__", None)
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/model_config.py` modified +37/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15205 - [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V

- Link: https://github.com/sgl-project/sglang/pull/15205
- Status/date: merged / 2025-12-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +100/-80, 345 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V"; model line: GLM VLM/OCR; category: docs/tests/CI; main diff: `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`; technical summary: Covers "[VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V"; the main implementation surface is `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v.py` modified +34/-50 (84 lines); hunks: -44,6 +44,7; -157,7 +158,8 @@ def forward(; symbols: forward, Glm4vVisionRotaryEmbedding, __init__, touching `forward, Glm4vVisionRotaryEmbedding, __init__`; `python/sglang/srt/models/qwen3_vl.py` modified +41/-20 (61 lines); hunks: -24,9 +24,6; -39,6 +36,7; symbols: forward, __init__, dtype, device, touching `forward, __init__, dtype`; `python/sglang/srt/layers/attention/vision.py` modified +20/-10 (30 lines); hunks: -675,6 +675,8 @@ def forward(; -724,26 +726,34 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/layers/rotary_embedding.py` modified +5/-0 (5 lines); hunks: -219,6 +219,11 @@ def get_cos_sin_with_position(self, positions):; symbols: get_cos_sin_with_position, get_cos_sin, forward_native, touching `get_cos_sin_with_position, get_cos_sin, forward_native`.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` modified +34/-50 (84 lines); hunks: -44,6 +44,7; -157,7 +158,8 @@ def forward(; symbols: forward, Glm4vVisionRotaryEmbedding, __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +41/-20 (61 lines); hunks: -24,9 +24,6; -39,6 +36,7; symbols: forward, __init__, dtype, device
  - `python/sglang/srt/layers/attention/vision.py` modified +20/-10 (30 lines); hunks: -675,6 +675,8 @@ def forward(; -724,26 +726,34 @@ def forward(; symbols: forward
  - `python/sglang/srt/layers/rotary_embedding.py` modified +5/-0 (5 lines); hunks: -219,6 +219,11 @@ def get_cos_sin_with_position(self, positions):; symbols: get_cos_sin_with_position, get_cos_sin, forward_native
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -44,6 +44,7 @@
+from sglang.srt.layers.rotary_embedding import get_rope
@@ -157,7 +158,8 @@ def forward(
-        position_embeddings: torch.Tensor,
+        rotary_pos_emb_cos: torch.Tensor,
+        rotary_pos_emb_sin: torch.Tensor,
@@ -169,7 +171,8 @@ def forward(
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -24,9 +24,6 @@
-from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
-    Qwen2_5_VisionRotaryEmbedding,
-)
@@ -39,6 +36,7 @@
+from sglang.srt.layers.rotary_embedding import get_rope
@@ -188,14 +186,16 @@ def forward(
diff -- python/sglang/srt/layers/attention/vision.py
@@ -675,6 +675,8 @@ def forward(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +34/-50; `python/sglang/srt/models/qwen3_vl.py` modified +41/-20; `python/sglang/srt/layers/attention/vision.py` modified +20/-10; `python/sglang/srt/layers/rotary_embedding.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/glm4v.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15434 - Convert cu_seqlens to CPU for npu_flash_attention_unpad operator

- Link: https://github.com/sgl-project/sglang/pull/15434
- Status/date: merged / 2026-01-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +36/-13, 169 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Convert cu_seqlens to CPU for npu_flash_attention_unpad operator"; model line: GLM VLM/OCR; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/paddleocr_vl.py`; technical summary: Covers "Convert cu_seqlens to CPU for npu_flash_attention_unpad operator"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/paddleocr_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +6/-3 (9 lines); hunks: -63,7 +63,7; -446,9 +446,12 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/glm4v.py` modified +5/-1 (6 lines); hunks: -57,7 +57,7; -514,6 +514,10 @@ def forward(self, x: torch.Tensor, grid_thw: torch.Tensor)...; symbols: forward, touching `forward`; `python/sglang/srt/models/paddleocr_vl.py` modified +4/-2 (6 lines); hunks: -36,7 +36,7; -442,7 +442,9 @@ def forward(; symbols: Projector, forward, touching `Projector, forward`; `python/sglang/srt/models/qwen2_5_vl.py` modified +4/-2 (6 lines); hunks: -76,7 +76,7; -453,7 +453,9 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +6/-3 (9 lines); hunks: -63,7 +63,7; -446,9 +446,12 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/glm4v.py` modified +5/-1 (6 lines); hunks: -57,7 +57,7; -514,6 +514,10 @@ def forward(self, x: torch.Tensor, grid_thw: torch.Tensor)...; symbols: forward
  - `python/sglang/srt/models/paddleocr_vl.py` modified +4/-2 (6 lines); hunks: -36,7 +36,7; -442,7 +442,9 @@ def forward(; symbols: Projector, forward
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +4/-2 (6 lines); hunks: -76,7 +76,7; -453,7 +453,9 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/dots_vlm_vit.py` modified +4/-1 (5 lines); hunks: -12,7 +12,7; -315,6 +315,9 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -63,7 +63,7 @@
-from sglang.srt.utils import add_prefix, get_int_env_var
+from sglang.srt.utils import add_prefix, get_int_env_var, is_npu
@@ -446,9 +446,12 @@ def forward(
+        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
+        if not is_npu():
+            cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
diff -- python/sglang/srt/models/glm4v.py
@@ -57,7 +57,7 @@
-from sglang.srt.utils import add_prefix
+from sglang.srt.utils import add_prefix, is_npu
@@ -514,6 +514,10 @@ def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
+        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
+        if is_npu():
+            cu_seqlens = cu_seqlens.to("cpu")
diff -- python/sglang/srt/models/paddleocr_vl.py
@@ -36,7 +36,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +6/-3; `python/sglang/srt/models/glm4v.py` modified +5/-1; `python/sglang/srt/models/paddleocr_vl.py` modified +4/-2; `python/sglang/srt/models/qwen2_5_vl.py` modified +4/-2; `python/sglang/srt/models/dots_vlm_vit.py` modified +4/-1; `python/sglang/srt/models/idefics2.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17582 - [GLM-OCR] Support GLM-OCR Model

- Link: https://github.com/sgl-project/sglang/pull/17582
- Status/date: merged / 2026-01-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm_ocr.py`, `python/sglang/srt/models/glm_ocr_nextn.py`, `python/sglang/srt/multimodal/processors/glm4v.py`; associated commits `7106f6c8e150`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +679/-29, 851 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[GLM-OCR] Support GLM-OCR Model"; model line: GLM VLM/OCR; category: model support/runtime entry; main diff: `python/sglang/srt/models/glm_ocr.py`, `python/sglang/srt/models/glm_ocr_nextn.py`, `python/sglang/srt/multimodal/processors/glm4v.py`; technical summary: Covers "[GLM-OCR] Support GLM-OCR Model"; the main implementation surface is `python/sglang/srt/models/glm_ocr.py`, `python/sglang/srt/models/glm_ocr_nextn.py`, `python/sglang/srt/multimodal/processors/glm4v.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm_ocr.py` added +435/-0 (435 lines); hunks: -0,0 +1,435; symbols: GlmOcrRMSNorm, GlmOcrVisionMLP, GlmOcrVisionBlock, __init__, touching `GlmOcrRMSNorm, GlmOcrVisionMLP, GlmOcrVisionBlock`; `python/sglang/srt/models/glm_ocr_nextn.py` added +162/-0 (162 lines); hunks: -0,0 +1,162; symbols: GlmOcrModelNextN, __init__, forward, GlmOcrForConditionalGenerationNextN, touching `GlmOcrModelNextN, __init__, forward`; `python/sglang/srt/multimodal/processors/glm4v.py` modified +6/-1 (7 lines); hunks: -3,14 +3,19; symbols: Glm4vImageProcessor, __init__, touching `Glm4vImageProcessor, __init__`; `python/sglang/srt/models/glm4v.py` modified +0/-2 (2 lines); hunks: -758,8 +758,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/glm_ocr.py` added +435/-0 (435 lines); hunks: -0,0 +1,435; symbols: GlmOcrRMSNorm, GlmOcrVisionMLP, GlmOcrVisionBlock, __init__
  - `python/sglang/srt/models/glm_ocr_nextn.py` added +162/-0 (162 lines); hunks: -0,0 +1,162; symbols: GlmOcrModelNextN, __init__, forward, GlmOcrForConditionalGenerationNextN
  - `python/sglang/srt/multimodal/processors/glm4v.py` modified +6/-1 (7 lines); hunks: -3,14 +3,19; symbols: Glm4vImageProcessor, __init__
  - `python/sglang/srt/models/glm4v.py` modified +0/-2 (2 lines); hunks: -758,8 +758,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm_ocr.py
@@ -0,0 +1,435 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/glm_ocr_nextn.py
@@ -0,0 +1,162 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/multimodal/processors/glm4v.py
@@ -3,14 +3,19 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm_ocr.py` added +435/-0; `python/sglang/srt/models/glm_ocr_nextn.py` added +162/-0; `python/sglang/srt/multimodal/processors/glm4v.py` modified +6/-1; `python/sglang/srt/models/glm4v.py` modified +0/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/glm4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17420 - [VLM] Optimize get_rope_index for GLM4v

- Link: https://github.com/sgl-project/sglang/pull/17420
- Status/date: merged / 2026-02-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +526/-86, 758 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Optimize get_rope_index for GLM4v"; model line: GLM VLM/OCR; category: performance/backend optimization; main diff: `python/sglang/srt/layers/rotary_embedding.py`, `benchmark/bench_rope/benchmark_rope_index.py`; technical summary: Covers "[VLM] Optimize get_rope_index for GLM4v"; the main implementation surface is `python/sglang/srt/layers/rotary_embedding.py`, `benchmark/bench_rope/benchmark_rope_index.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/rotary_embedding.py` modified +101/-86 (187 lines); hunks: -1700,11 +1700,12 @@ def get_rope_index(; -1737,24 +1738,24 @@ def get_rope_index(; symbols: get_rope_index, get_rope_index_glm4v, touching `get_rope_index, get_rope_index_glm4v`; `benchmark/bench_rope/benchmark_rope_index.py` added +425/-0 (425 lines); hunks: -0,0 +1,425; symbols: DummyVisionConfig, DummyHFConfig, calculate_stats, _sync, touching `DummyVisionConfig, DummyHFConfig, calculate_stats`.
- Code diff details:
  - `python/sglang/srt/layers/rotary_embedding.py` modified +101/-86 (187 lines); hunks: -1700,11 +1700,12 @@ def get_rope_index(; -1737,24 +1738,24 @@ def get_rope_index(; symbols: get_rope_index, get_rope_index_glm4v
  - `benchmark/bench_rope/benchmark_rope_index.py` added +425/-0 (425 lines); hunks: -0,0 +1,425; symbols: DummyVisionConfig, DummyHFConfig, calculate_stats, _sync
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -1700,11 +1700,12 @@ def get_rope_index(
-                    llm_grid_t, llm_grid_h, llm_grid_w = (
-                        t.item(),
-                        h.item() // spatial_merge_size,
-                        w.item() // spatial_merge_size,
-                    )
+                    # Avoid .item() lookups in repeated context
diff -- benchmark/bench_rope/benchmark_rope_index.py
@@ -0,0 +1,425 @@
+# This script benchmarks MRotaryEmbedding.get_rope_index_glm4v (GLM4V mrope index builder).
+# It generates synthetic multimodal input_ids + attention_mask (+ optional image/video grids),
+# runs benchmarks.
+#
+# == Usage Examples ==
+#
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/rotary_embedding.py` modified +101/-86
  - other: `benchmark/bench_rope/benchmark_rope_index.py` added +425/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/rotary_embedding.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18885 - Fix GLM-4V processor registration when glm_ocr is unavailable

- Link: https://github.com/sgl-project/sglang/pull/18885
- Status/date: merged / 2026-02-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/glm4v.py`; associated commits `206accd15de3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-4, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix GLM-4V processor registration when glm_ocr is unavailable"; model line: GLM VLM/OCR; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/glm4v.py`; technical summary: Covers "Fix GLM-4V processor registration when glm_ocr is unavailable"; the main implementation surface is `python/sglang/srt/multimodal/processors/glm4v.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/glm4v.py` modified +12/-4 (16 lines); hunks: -3,20 +3,28; symbols: Glm4vImageProcessor, __init__, touching `Glm4vImageProcessor, __init__`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/glm4v.py` modified +12/-4 (16 lines); hunks: -3,20 +3,28; symbols: Glm4vImageProcessor, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/glm4v.py
@@ -3,20 +3,28 @@
-from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration
+try:
+    from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration
+except ImportError:
+    GlmOcrForConditionalGeneration = None
-        Glm4vForConditionalGeneration,
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/glm4v.py` modified +12/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/glm4v.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19728 - Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback

- Link: https://github.com/sgl-project/sglang/pull/19728
- Status/date: open / 2026-03-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +104/-4, 179 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback"; model line: GLM VLM/OCR; category: bug fix; main diff: `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `test/registered/moe/test_fused_moe.py`; technical summary: Covers "Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback"; the main implementation surface is `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `test/registered/moe/test_fused_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +21/-4 (25 lines); hunks: -1541,6 +1541,23 @@ def per_token_group_quant_mla_deep_gemm_masked_fp8(; -1549,10 +1566,10 @@ def _native_dynamic_per_token_quant_fp8(output, input, s...; symbols: per_token_group_quant_mla_deep_gemm_masked_fp8, _copy_with_optional_row_padding, _native_dynamic_per_token_quant_fp8, _native_dynamic_per_tensor_quant_fp8, touching `per_token_group_quant_mla_deep_gemm_masked_fp8, _copy_with_optional_row_padding, _native_dynamic_per_token_quant_fp8`; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +6/-0 (6 lines); hunks: -351,6 +351,12 @@ def fused_experts_impl(; symbols: fused_experts_impl, touching `fused_experts_impl`; `test/registered/moe/test_fused_moe.py` modified +66/-0 (66 lines); hunks: -1,9 +1,11; -239,6 +241,70 @@ def test_various_configurations(self):; symbols: test_various_configurations, test_fp8_unpadded_weights_with_global_moe_padding, touching `test_various_configurations, test_fp8_unpadded_weights_with_global_moe_padding`; `python/sglang/test/test_custom_ops.py` modified +11/-0 (11 lines); hunks: -3,11 +3,14; -141,6 +144,14 @@ def test_scaled_fp8_quant_with_padding(dtype) -> None:; symbols: test_scaled_fp8_quant_with_padding, touching `test_scaled_fp8_quant_with_padding`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +21/-4 (25 lines); hunks: -1541,6 +1541,23 @@ def per_token_group_quant_mla_deep_gemm_masked_fp8(; -1549,10 +1566,10 @@ def _native_dynamic_per_token_quant_fp8(output, input, s...; symbols: per_token_group_quant_mla_deep_gemm_masked_fp8, _copy_with_optional_row_padding, _native_dynamic_per_token_quant_fp8, _native_dynamic_per_tensor_quant_fp8
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +6/-0 (6 lines); hunks: -351,6 +351,12 @@ def fused_experts_impl(; symbols: fused_experts_impl
  - `test/registered/moe/test_fused_moe.py` modified +66/-0 (66 lines); hunks: -1,9 +1,11; -239,6 +241,70 @@ def test_various_configurations(self):; symbols: test_various_configurations, test_fp8_unpadded_weights_with_global_moe_padding
  - `python/sglang/test/test_custom_ops.py` modified +11/-0 (11 lines); hunks: -3,11 +3,14; -141,6 +144,14 @@ def test_scaled_fp8_quant_with_padding(dtype) -> None:; symbols: test_scaled_fp8_quant_with_padding
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -1541,6 +1541,23 @@ def per_token_group_quant_mla_deep_gemm_masked_fp8(
+    def _copy_with_optional_row_padding(
+        dst: torch.Tensor,
+        src: torch.Tensor,
+        pad_value: float = 0.0,
+    ) -> None:
+        if dst.shape == src.shape:
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
@@ -351,6 +351,12 @@ def fused_experts_impl(
+    elif hidden_states.shape[1] == w1.shape[2]:
+        # Some ROCm FP8 MoE checkpoints load unpadded expert weights even when
+        # SGLANG_MOE_PADDING is enabled globally. In that case the runtime
+        # shape already matches the true hidden size and subtracting
+        # `padding_size` would reject a valid layout.
+        padded_size = 0
diff -- test/registered/moe/test_fused_moe.py
@@ -1,9 +1,11 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +21/-4; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +6/-0
  - tests: `test/registered/moe/test_fused_moe.py` modified +66/-0; `python/sglang/test/test_custom_ops.py` modified +11/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_custom_ops.py`, `test/registered/moe/test_fused_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20033 - [VLM] Replace conv3d proj with linear for GLM4V

- Link: https://github.com/sgl-project/sglang/pull/20033
- Status/date: merged / 2026-03-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glm4v.py`; associated commits `97a2a9be0f45`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +192/-9, 228 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Replace conv3d proj with linear for GLM4V"; model line: GLM VLM/OCR; category: performance/backend optimization; main diff: `python/sglang/srt/models/glm4v.py`; technical summary: Covers "[VLM] Replace conv3d proj with linear for GLM4V"; the main implementation surface is `python/sglang/srt/models/glm4v.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v.py` modified +26/-9 (35 lines); hunks: -211,16 +211,26 @@ def __init__(; -446,10 +456,16 @@ def __init__(; symbols: __init__, forward, copy_conv3d_weight_to_linear, Glm4vPatchMerger, touching `__init__, forward, copy_conv3d_weight_to_linear`.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` modified +26/-9 (35 lines); hunks: -211,16 +211,26 @@ def __init__(; -446,10 +456,16 @@ def __init__(; symbols: __init__, forward, copy_conv3d_weight_to_linear, Glm4vPatchMerger
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -211,16 +211,26 @@ def __init__(
-    def forward(self, x: torch.Tensor) -> torch.Tensor:
-        x = x.view(
-            -1,
-            self.in_channels,
-            self.temporal_patch_size,
-            self.patch_size,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +26/-9
- Risk and verification: The diff ships test coverage in `test/registered/vlm/test_patch_embed_perf.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20463 - [Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr

- Link: https://github.com/sgl-project/sglang/pull/20463
- Status/date: merged / 2026-03-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`; associated commits `c330b687a116`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-0, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr"; model line: GLM VLM/OCR; category: bug fix; main diff: `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`; technical summary: Covers "[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr"; the main implementation surface is `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v_moe.py` modified +3/-0 (3 lines); hunks: -281,5 +281,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`; `python/sglang/srt/models/glm_ocr.py` modified +3/-0 (3 lines); hunks: -431,5 +431,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/glm4v_moe.py` modified +3/-0 (3 lines); hunks: -281,5 +281,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
  - `python/sglang/srt/models/glm_ocr.py` modified +3/-0 (3 lines); hunks: -431,5 +431,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -281,5 +281,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
+        if not is_nextn:
+            self.visual.patch_embed.copy_conv3d_weight_to_linear()
diff -- python/sglang/srt/models/glm_ocr.py
@@ -431,5 +431,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
+        if not is_nextn:
+            self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v_moe.py` modified +3/-0; `python/sglang/srt/models/glm_ocr.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20282 - Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug

- Link: https://github.com/sgl-project/sglang/pull/20282
- Status/date: merged / 2026-03-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +704/-90, 1053 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug"; model line: GLM VLM/OCR; category: bug fix; main diff: `python/sglang/srt/layers/conv.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/pixtral.py`; technical summary: Covers "Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug"; the main implementation surface is `python/sglang/srt/layers/conv.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/pixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/conv.py` added +300/-0 (300 lines); hunks: -0,0 +1,300; symbols: _tuplify, _check_enable_linear, _reverse_repeat_tuple, _compute_same_padding_for_pad, touching `_tuplify, _check_enable_linear, _reverse_repeat_tuple`; `python/sglang/srt/models/glm4v.py` modified +12/-27 (39 lines); hunks: -35,6 +35,7; -203,34 +204,25 @@ def __init__(; symbols: __init__, copy_conv3d_weight_to_linear, forward, Glm4vPatchMerger, touching `__init__, copy_conv3d_weight_to_linear, forward`; `python/sglang/srt/models/pixtral.py` modified +3/-2 (5 lines); hunks: -35,6 +35,7; -328,7 +329,7 @@ class VisionTransformer(nn.Module):; symbols: VisionTransformer, __init__, touching `VisionTransformer, __init__`; `python/sglang/srt/models/clip.py` modified +2/-1 (3 lines); hunks: -11,6 +11,7; -32,7 +33,7 @@ def __init__(self, config: CLIPVisionConfig):; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/conv.py` added +300/-0 (300 lines); hunks: -0,0 +1,300; symbols: _tuplify, _check_enable_linear, _reverse_repeat_tuple, _compute_same_padding_for_pad
  - `python/sglang/srt/models/glm4v.py` modified +12/-27 (39 lines); hunks: -35,6 +35,7; -203,34 +204,25 @@ def __init__(; symbols: __init__, copy_conv3d_weight_to_linear, forward, Glm4vPatchMerger
  - `python/sglang/srt/models/pixtral.py` modified +3/-2 (5 lines); hunks: -35,6 +35,7; -328,7 +329,7 @@ class VisionTransformer(nn.Module):; symbols: VisionTransformer, __init__
  - `python/sglang/srt/models/clip.py` modified +2/-1 (3 lines); hunks: -11,6 +11,7; -32,7 +33,7 @@ def __init__(self, config: CLIPVisionConfig):; symbols: __init__
  - `python/sglang/srt/models/dots_vlm_vit.py` modified +2/-1 (3 lines); hunks: -11,6 +11,7; -113,7 +114,7 @@ def __init__(self, config, quant_config: Optional[Quantizati...; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/conv.py
@@ -0,0 +1,300 @@
+"""
+Conv2d/Conv3d layers with unfold+linear optimization for patch embeddings.
+When kernel_size == stride, padding == 0, dilation == 1, groups == 1, the conv
+is equivalent to unfold + F.linear, which is significantly faster on CUDA and
+also avoids the PyTorch 2.9.1 + CuDNN < 9.15 Conv3d bug
+(https://github.com/pytorch/pytorch/issues/168167).
diff -- python/sglang/srt/models/glm4v.py
@@ -35,6 +35,7 @@
+from sglang.srt.layers.conv import Conv3dLayer
@@ -203,34 +204,25 @@ def __init__(
-        self.proj = nn.Conv3d(
+        self.proj = Conv3dLayer(
-        k = self.in_channels * self.temporal_patch_size * self.patch_size**2
-        self.linear = nn.Linear(
diff -- python/sglang/srt/models/pixtral.py
@@ -35,6 +35,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/conv.py` added +300/-0; `python/sglang/srt/models/glm4v.py` modified +12/-27; `python/sglang/srt/models/pixtral.py` modified +3/-2; `python/sglang/srt/models/clip.py` modified +2/-1; `python/sglang/srt/models/dots_vlm_vit.py` modified +2/-1; `python/sglang/srt/models/idefics2.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `test/unit/test_conv_layer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20740 - Revert "[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr"

- Link: https://github.com/sgl-project/sglang/pull/20740
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`; associated commits `f15b3338c9f3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +0/-6, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr""; model line: GLM VLM/OCR; category: bug fix; main diff: `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`; technical summary: Covers "Revert "[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr""; the main implementation surface is `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v_moe.py` modified +0/-3 (3 lines); hunks: -281,8 +281,5 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`; `python/sglang/srt/models/glm_ocr.py` modified +0/-3 (3 lines); hunks: -431,8 +431,5 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/glm4v_moe.py` modified +0/-3 (3 lines); hunks: -281,8 +281,5 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
  - `python/sglang/srt/models/glm_ocr.py` modified +0/-3 (3 lines); hunks: -431,8 +431,5 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -281,8 +281,5 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
-        if not is_nextn:
-            self.visual.patch_embed.copy_conv3d_weight_to_linear()
diff -- python/sglang/srt/models/glm_ocr.py
@@ -431,8 +431,5 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
-        if not is_nextn:
-            self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v_moe.py` modified +0/-3; `python/sglang/srt/models/glm_ocr.py` modified +0/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21134 - [Bug Fix] GLM-V / GLM-OCR: field detection for transformers 5.x and MTP omission fix

- Link: https://github.com/sgl-project/sglang/pull/21134
- Status/date: merged / 2026-03-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`; associated commits `fcaad42b0038`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +16/-9, 74 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug Fix] GLM-V / GLM-OCR: field detection for transformers 5.x and MTP omission fix"; model line: GLM VLM/OCR; category: bug fix; main diff: `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`; technical summary: Covers "[Bug Fix] GLM-V / GLM-OCR: field detection for transformers 5.x and MTP omission fix"; the main implementation surface is `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v_moe.py` modified +7/-7 (14 lines); hunks: -158,6 +158,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; -196,13 +203,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, touching `load_weights`; `python/sglang/srt/models/glm_ocr.py` modified +4/-1 (5 lines); hunks: -26,6 +26,7; -151,6 +152,7 @@ class GlmOcrVisionModel(Glm4vVisionModel):; symbols: GlmOcrVisionModel, __init__, touching `GlmOcrVisionModel, __init__`.
- Code diff details:
  - `python/sglang/srt/models/glm4v_moe.py` modified +7/-7 (14 lines); hunks: -158,6 +158,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; -196,13 +203,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
  - `python/sglang/srt/models/glm_ocr.py` modified +4/-1 (5 lines); hunks: -26,6 +26,7; -151,6 +152,7 @@ class GlmOcrVisionModel(Glm4vVisionModel):; symbols: GlmOcrVisionModel, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -158,6 +158,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
+            if "language_model." in name:
+                name = name.replace("language_model.", "")
+            if "model.visual." in name:
+                name = name.replace("model.visual.", "visual.")
+            if "rotary_emb.inv_freq" in name:
+                continue
diff -- python/sglang/srt/models/glm_ocr.py
@@ -26,6 +26,7 @@
+    GlmOcrTextConfig,
@@ -151,6 +152,7 @@ class GlmOcrVisionModel(Glm4vVisionModel):
+        text_config: GlmOcrTextConfig,
@@ -203,7 +205,7 @@ def __init__(
-            context_dim=vision_config.out_hidden_size * vision_config.in_channels,
+            context_dim=text_config.intermediate_size,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v_moe.py` modified +7/-7; `python/sglang/srt/models/glm_ocr.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19788 - [Feature] Optimizations for class Qwen3VLMoeVisionModel (Conv3d to Linear) in Qwen3VL

- Link: https://github.com/sgl-project/sglang/pull/19788
- Status/date: closed / 2026-03-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +155/-12, 209 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Optimizations for class Qwen3VLMoeVisionModel (Conv3d to Linear) in Qwen3VL"; model line: GLM VLM/OCR; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `test/registered/vlm/test_qwen3vl_conv3d_to_linear.py`; technical summary: Covers "[Feature] Optimizations for class Qwen3VLMoeVisionModel (Conv3d to Linear) in Qwen3VL"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `test/registered/vlm/test_qwen3vl_conv3d_to_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +26/-12 (38 lines); hunks: -147,20 +147,26 @@ def __init__(self, config) -> None:; -407,10 +413,16 @@ def __init__(; symbols: __init__, copy_conv3d_weight_to_linear, forward, Qwen3_VisionBlock, touching `__init__, copy_conv3d_weight_to_linear, forward`; `python/sglang/srt/models/qwen3_vl_moe.py` modified +2/-0 (2 lines); hunks: -355,6 +355,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`; `test/registered/vlm/test_qwen3vl_conv3d_to_linear.py` added +127/-0 (127 lines); hunks: -0,0 +1,127; symbols: _set_default_device_npu, _build_model, test_conv3d_to_linear, touching `_set_default_device_npu, _build_model, test_conv3d_to_linear`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +26/-12 (38 lines); hunks: -147,20 +147,26 @@ def __init__(self, config) -> None:; -407,10 +413,16 @@ def __init__(; symbols: __init__, copy_conv3d_weight_to_linear, forward, Qwen3_VisionBlock
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +2/-0 (2 lines); hunks: -355,6 +355,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
  - `test/registered/vlm/test_qwen3vl_conv3d_to_linear.py` added +127/-0 (127 lines); hunks: -0,0 +1,127; symbols: _set_default_device_npu, _build_model, test_conv3d_to_linear
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -147,20 +147,26 @@ def __init__(self, config) -> None:
+        k = self.in_channels * self.temporal_patch_size * self.patch_size**2
+        self.linear = nn.Linear(
+            in_features=k,
+            out_features=self.embed_dim,
+            bias=True,
+            dtype=self.proj.weight.dtype,
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -355,6 +355,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+        self.visual.patch_embed.copy_conv3d_weight_to_linear()
diff -- test/registered/vlm/test_qwen3vl_conv3d_to_linear.py
@@ -0,0 +1,127 @@
+import pytest
+import torch
+import torch.nn as nn
+from sglang.srt.configs.qwen3_vl import Qwen3VLConfig
+from sglang.srt.distributed.parallel_state import (
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +26/-12; `python/sglang/srt/models/qwen3_vl_moe.py` modified +2/-0
  - tests: `test/registered/vlm/test_qwen3vl_conv3d_to_linear.py` added +127/-0
- Risk and verification: The diff ships test coverage in `test/registered/vlm/test_qwen3vl_conv3d_to_linear.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18559 - Optimizations for Qwen3VL models

- Link: https://github.com/sgl-project/sglang/pull/18559
- Status/date: closed / 2026-03-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +512/-33, 730 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimizations for Qwen3VL models"; model line: GLM VLM/OCR; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/multimodal/processors/base_processor.py`; technical summary: Covers "Optimizations for Qwen3VL models"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/multimodal/processors/base_processor.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +445/-15 (460 lines); hunks: -16,15 +16,17; -90,8 +92,12 @@ def __init__(; symbols: __init__, forward, Qwen3VLVisionPatchEmbed, touching `__init__, forward, Qwen3VLVisionPatchEmbed`; `python/sglang/srt/layers/attention/vision.py` modified +32/-14 (46 lines); hunks: -3,6 +3,7; -755,8 +756,13 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/multimodal/processors/base_processor.py` modified +6/-1 (7 lines); hunks: -431,7 +431,12 @@ def _load_single_item(; symbols: _load_single_item, touching `_load_single_item`; `python/sglang/srt/utils/common.py` modified +20/-2 (22 lines); hunks: -101,6 +101,8; -923,8 +925,24 @@ def load_image(; symbols: load_image, touching `load_image`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +445/-15 (460 lines); hunks: -16,15 +16,17; -90,8 +92,12 @@ def __init__(; symbols: __init__, forward, Qwen3VLVisionPatchEmbed
  - `python/sglang/srt/layers/attention/vision.py` modified +32/-14 (46 lines); hunks: -3,6 +3,7; -755,8 +756,13 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +6/-1 (7 lines); hunks: -431,7 +431,12 @@ def _load_single_item(; symbols: _load_single_item
  - `python/sglang/srt/utils/common.py` modified +20/-2 (22 lines); hunks: -101,6 +101,8; -923,8 +925,24 @@ def load_image(; symbols: load_image
  - `python/sglang/srt/model_loader/loader.py` modified +8/-0 (8 lines); hunks: -683,7 +683,15 @@ def load_model(; symbols: load_model, load_weights_and_postprocess
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -16,15 +16,17 @@
+import os  # wili
-import numpy as np
+import numpy as np  # wili
+from vfly.utils.parallel import dit_sp_gather, dit_sp_split  # wili
@@ -90,8 +92,12 @@ def __init__(
-        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
diff -- python/sglang/srt/layers/attention/vision.py
@@ -3,6 +3,7 @@
+import os  # wili
@@ -755,8 +756,13 @@ def __init__(
-        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
-        self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()
+        self.enable_vfly = bool(int(os.environ.get("ENABLE_VFLY", "0")))  # wili
+        if self.enable_vfly:  # wili, reuse TP group but keep TP size as 1
diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -431,7 +431,12 @@ def _load_single_item(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +445/-15; `python/sglang/srt/layers/attention/vision.py` modified +32/-14; `python/sglang/srt/multimodal/processors/base_processor.py` modified +6/-1; `python/sglang/srt/utils/common.py` modified +20/-2; `python/sglang/srt/model_loader/loader.py` modified +8/-0; `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/model_loader/loader.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17122 - [bugfix]GLM-4V model

- Link: https://github.com/sgl-project/sglang/pull/17122
- Status/date: merged / 2026-04-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +38/-3, 70 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bugfix]GLM-4V model"; model line: GLM VLM/OCR; category: bug fix; main diff: `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py`; technical summary: Covers "[bugfix]GLM-4V model"; the main implementation surface is `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4v.py` modified +2/-2 (4 lines); hunks: -414,6 +414,7 @@ def __init__(; -553,15 +554,14 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/multimodal/processors/base_processor.py` modified +3/-1 (4 lines); hunks: -407,7 +407,9 @@ def process_mm_data(; symbols: process_mm_data, touching `process_mm_data`; `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py` added +33/-0 (33 lines); hunks: -0,0 +1,33; symbols: TestGLM4Models, test_vlm_mmmu_benchmark, touching `TestGLM4Models, test_vlm_mmmu_benchmark`.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` modified +2/-2 (4 lines); hunks: -414,6 +414,7 @@ def __init__(; -553,15 +554,14 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +3/-1 (4 lines); hunks: -407,7 +407,9 @@ def process_mm_data(; symbols: process_mm_data
  - `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py` added +33/-0 (33 lines); hunks: -0,0 +1,33; symbols: TestGLM4Models, test_vlm_mmmu_benchmark
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4v.py
@@ -414,6 +414,7 @@ def __init__(
+                    num_dummy_heads=vision_config.num_dummy_heads,
@@ -553,15 +554,14 @@ def __init__(
+        vision_utils.update_vit_attn_dummy_heads_config(self.config)
-        vision_utils.update_vit_attn_dummy_heads_config(self.config)
diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -407,7 +407,9 @@ def process_mm_data(
-            else:
+            elif processor.__class__.__name__ not in {
+                "Glm4vProcessor",
+            }:
diff -- test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py
@@ -0,0 +1,33 @@
+import unittest
+from sglang.test.ascend.vlm_utils import TestVLMModels
+from sglang.test.ci.ci_register import register_npu_ci
+register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4v.py` modified +2/-2; `python/sglang/srt/multimodal/processors/base_processor.py` modified +3/-1
  - tests: `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py` added +33/-0
- Risk and verification: The diff ships test coverage in `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22961 - [NPU] Fix issue and support GLM-4.5V

- Link: https://github.com/sgl-project/sglang/pull/22961
- Status/date: merged / 2026-04-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +17/-5, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Fix issue and support GLM-4.5V"; model line: GLM VLM/OCR; category: bug fix; main diff: `python/sglang/srt/models/glm4_moe.py`; technical summary: Covers "[NPU] Fix issue and support GLM-4.5V"; the main implementation surface is `python/sglang/srt/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +17/-5 (22 lines); hunks: -314,18 +314,30 @@ def forward_prepare(; symbols: forward_prepare, touching `forward_prepare`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +17/-5 (22 lines); hunks: -314,18 +314,30 @@ def forward_prepare(; symbols: forward_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -314,18 +314,30 @@ def forward_prepare(
+            if self.use_qk_norm:
+                eps = self.q_norm.variance_epsilon
+                q_weight = self.q_norm.weight
+                k_weight = self.k_norm.weight
+                q_bias = getattr(self.q_norm, "bias", None)
+                k_bias = getattr(self.k_norm, "bias", None)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +17/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
