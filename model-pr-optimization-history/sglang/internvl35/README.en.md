# sglang InternVL 3.5 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/InternVL/InternVL3.5.mdx` | no direct PR-number commit |
| `python/sglang/srt/configs/internvl.py` | [#5350](https://github.com/sgl-project/sglang/pull/5350), [#8067](https://github.com/sgl-project/sglang/pull/8067), [#9705](https://github.com/sgl-project/sglang/pull/9705) |
| `python/sglang/srt/models/internvl.py` | [#5350](https://github.com/sgl-project/sglang/pull/5350), [#6870](https://github.com/sgl-project/sglang/pull/6870), [#9705](https://github.com/sgl-project/sglang/pull/9705), [#13640](https://github.com/sgl-project/sglang/pull/13640), [#13925](https://github.com/sgl-project/sglang/pull/13925), [#15942](https://github.com/sgl-project/sglang/pull/15942), [#16732](https://github.com/sgl-project/sglang/pull/16732), [#19127](https://github.com/sgl-project/sglang/pull/19127) |
| `python/sglang/srt/multimodal/internvl_utils.py` | no direct PR-number commit |
| `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py` | [#16732](https://github.com/sgl-project/sglang/pull/16732) |
| `python/sglang/srt/multimodal/processors/internvl.py` | [#9381](https://github.com/sgl-project/sglang/pull/9381), [#9795](https://github.com/sgl-project/sglang/pull/9795), [#10375](https://github.com/sgl-project/sglang/pull/10375), [#15942](https://github.com/sgl-project/sglang/pull/15942), [#17040](https://github.com/sgl-project/sglang/pull/17040), [#19127](https://github.com/sgl-project/sglang/pull/19127), [#19997](https://github.com/sgl-project/sglang/pull/19997) |

## PR Coverage Summary

- Git-traced PRs: 14
- Extra PRs preserved from existing docs: 12
- Total PRs in this document: 26
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-03-21 | [#3351](https://github.com/sgl-project/sglang/pull/3351) | closed | model: Intern vl 2.5 | `python/sglang/srt/models/deepseek_janus_pro.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/tokenizers/lmtokenizer.py` |
| 2025-05-02 | [#5350](https://github.com/sgl-project/sglang/pull/5350) | merged | Support InternVL3 | `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py` |
| 2025-05-30 | [#4433](https://github.com/sgl-project/sglang/pull/4433) | closed | Support InternVL2.5 | `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/configs/model_config.py` |
| 2025-06-11 | [#6870](https://github.com/sgl-project/sglang/pull/6870) | merged | vlm: adapt internvl to VisionAttention | `python/sglang/srt/models/internvl.py` |
| 2025-07-20 | [#8067](https://github.com/sgl-project/sglang/pull/8067) | merged | fix: fix the bug of loading Internvl3 | `python/sglang/srt/configs/internvl.py` |
| 2025-08-20 | [#9381](https://github.com/sgl-project/sglang/pull/9381) | merged | fix: InternS1 don't recognize image, updates image token for InternVL processor | `python/sglang/srt/multimodal/processors/internvl.py` |
| 2025-09-02 | [#9705](https://github.com/sgl-project/sglang/pull/9705) | merged | Support the internvl3.5 family models in sglang | `python/sglang/srt/models/internvl.py`, `python/sglang/srt/configs/internvl.py` |
| 2025-09-10 | [#9795](https://github.com/sgl-project/sglang/pull/9795) | merged | refactor(InternVL): Use gpu to preprocess the input image | `python/sglang/srt/multimodal/processors/internvl.py` |
| 2025-09-15 | [#10375](https://github.com/sgl-project/sglang/pull/10375) | merged | fix(internvl): fix accuracy issue of normalization | `python/sglang/srt/multimodal/processors/internvl.py` |
| 2025-11-21 | [#13640](https://github.com/sgl-project/sglang/pull/13640) | merged | [VLM] Support Piecewise CUDA Graph for InternVL | `python/sglang/srt/models/internvl.py` |
| 2025-11-26 | [#13925](https://github.com/sgl-project/sglang/pull/13925) | merged | [VLM] Support InternVL Vision Encoder Data Parallelism | `python/sglang/srt/models/internvl.py` |
| 2025-12-30 | [#15942](https://github.com/sgl-project/sglang/pull/15942) | merged | [VLM] Support Video for InternVL3_5 | `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py` |
| 2026-01-14 | [#16732](https://github.com/sgl-project/sglang/pull/16732) | merged | [VLM] Support ViT CUDA Graph for InternVL | `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py`, `python/sglang/srt/models/internvl.py` |
| 2026-01-26 | [#17040](https://github.com/sgl-project/sglang/pull/17040) | merged | fix(processor): support InternS1 text_config in InternVL processor | `python/sglang/srt/multimodal/processors/internvl.py` |
| 2026-02-27 | [#19127](https://github.com/sgl-project/sglang/pull/19127) | merged | [vlm][internVL] Support processor and embedding inputs for InternVL | `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py` |
| 2026-03-15 | [#19997](https://github.com/sgl-project/sglang/pull/19997) | merged | Fix InternVL and vision attention for non-CUDA backends (e.g. XPU) | `python/sglang/srt/multimodal/processors/internvl.py`, `test/srt/xpu/test_internvl.py` |
| 2026-03-15 | [#20282](https://github.com/sgl-project/sglang/pull/20282) | merged | Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug | `python/sglang/srt/layers/conv.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/pixtral.py` |
| 2026-03-18 | [#17784](https://github.com/sgl-project/sglang/pull/17784) | merged | Upgrade transformers==5.3.0 | `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py` |
| 2026-03-29 | [#19749](https://github.com/sgl-project/sglang/pull/19749) | merged | [Feature] Optimizations for JPEG input on NVIDIA GPU | `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/llava.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` |
| 2026-04-03 | [#21899](https://github.com/sgl-project/sglang/pull/21899) | merged | [VLM] Enable per-image MM splitting by default and remove MULTI_IMAGES modality | `python/sglang/srt/models/llava.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/llava.py` |
| 2026-04-03 | [#21738](https://github.com/sgl-project/sglang/pull/21738) | merged | refactor: replace mm_inputs dict with MultimodalProcessorOutput | `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py` |
| 2026-04-20 | [#23001](https://github.com/sgl-project/sglang/pull/23001) | merged | Add new Mintlify documentation site (docs_new/) | `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx` |
| 2026-04-25 | [#23568](https://github.com/sgl-project/sglang/pull/23568) | merged | Parakeet nemotron encoder | `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/internvl_utils.py`, `python/sglang/srt/models/radio.py` |
| 2026-05-13 | [#25182](https://github.com/sgl-project/sglang/pull/25182) | merged | chore: add vLLM SPDX copyright headers to ported files | `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py` |
| 2026-05-22 | [#24751](https://github.com/sgl-project/sglang/pull/24751) | merged | fix(mm): make multimodal data loading non-blocking to prevent health check stalls | `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |

## Per-PR Diff Audit Cards

### PR #3351 - model: Intern vl 2.5

- Link: https://github.com/sgl-project/sglang/pull/3351
- Status/date: closed / 2025-03-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 24 files, +4538/-163, 5186 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "model: Intern vl 2.5"; model line: InternVL 3.5; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_janus_pro.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/tokenizers/lmtokenizer.py`; technical summary: Covers "model: Intern vl 2.5"; the main implementation surface is `python/sglang/srt/models/deepseek_janus_pro.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/tokenizers/lmtokenizer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_janus_pro.py` added +2174/-0 (2174 lines); hunks: -0,0 +1,2174; symbols: VQ_16, ModelArgs, _ntuple, parse, touching `VQ_16, ModelArgs, _ntuple`; `python/sglang/srt/models/internvl.py` added +622/-0 (622 lines); hunks: -0,0 +1,622; symbols: InternVisionEmbeddings, __init__, _get_pos_embed, forward, touching `InternVisionEmbeddings, __init__, _get_pos_embed`; `python/sglang/srt/tokenizers/lmtokenizer.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: InternLM2Tokenizer, __init__, no_prefix_space_tokens, vocab_size, touching `InternLM2Tokenizer, __init__, no_prefix_space_tokens`; `python/sglang/srt/configs/janus.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: DictToObject, __init__, VisionConfig, GenAlignerConfig, touching `DictToObject, __init__, VisionConfig`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_janus_pro.py` added +2174/-0 (2174 lines); hunks: -0,0 +1,2174; symbols: VQ_16, ModelArgs, _ntuple, parse
  - `python/sglang/srt/models/internvl.py` added +622/-0 (622 lines); hunks: -0,0 +1,622; symbols: InternVisionEmbeddings, __init__, _get_pos_embed, forward
  - `python/sglang/srt/tokenizers/lmtokenizer.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: InternLM2Tokenizer, __init__, no_prefix_space_tokens, vocab_size
  - `python/sglang/srt/configs/janus.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: DictToObject, __init__, VisionConfig, GenAlignerConfig
  - `python/sglang/srt/models/minicpmv.py` modified +11/-73 (84 lines); hunks: -41,7 +41,6; -51,7 +50,7; symbols: __init__, pad_input_ids
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_janus_pro.py
@@ -0,0 +1,2174 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/internvl.py
@@ -0,0 +1,622 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/tokenizers/lmtokenizer.py
@@ -0,0 +1,242 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_janus_pro.py` added +2174/-0; `python/sglang/srt/models/internvl.py` added +622/-0; `python/sglang/srt/tokenizers/lmtokenizer.py` added +242/-0; `python/sglang/srt/configs/janus.py` added +155/-0; `python/sglang/srt/models/minicpmv.py` modified +11/-73; `python/sglang/srt/models/qwen2_vl.py` modified +12/-35
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_utils.py`, `test/srt/test_vision_openai_server.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5350 - Support InternVL3

- Link: https://github.com/sgl-project/sglang/pull/5350
- Status/date: merged / 2025-05-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`; associated commits `3409aaab32c6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +1728/-9, 1901 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support InternVL3"; model line: InternVL 3.5; category: model support/runtime entry; main diff: `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py`; technical summary: Covers "Support InternVL3"; the main implementation surface is `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/configs/internvl.py` added +696/-0 (696 lines); hunks: -0,0 +1,696; symbols: InternLM2Config, to, __init__, _rope_scaling_validation, touching `InternLM2Config, to, __init__`; `python/sglang/srt/models/internvl.py` added +670/-0 (670 lines); hunks: -0,0 +1,670; symbols: FlashAttention, __init__, forward, InternAttention, touching `FlashAttention, __init__, forward`; `python/sglang/srt/managers/multimodal_processors/internvl.py` added +232/-0 (232 lines); hunks: -0,0 +1,232; symbols: InternVLImageProcessor, __init__, build_transform, resize_image, touching `InternVLImageProcessor, __init__, build_transform`.
- Code diff details:
  - `python/sglang/srt/configs/internvl.py` added +696/-0 (696 lines); hunks: -0,0 +1,696; symbols: InternLM2Config, to, __init__, _rope_scaling_validation
  - `python/sglang/srt/models/internvl.py` added +670/-0 (670 lines); hunks: -0,0 +1,670; symbols: FlashAttention, __init__, forward, InternAttention
  - `python/sglang/srt/managers/multimodal_processors/internvl.py` added +232/-0 (232 lines); hunks: -0,0 +1,232; symbols: InternVLImageProcessor, __init__, build_transform, resize_image
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/internvl.py
@@ -0,0 +1,696 @@
+import copy
+import os
+from shutil import copyfile
+from typing import Any, Dict, List, Optional, Tuple, Union
+import sentencepiece as spm
+from transformers import (
diff -- python/sglang/srt/models/internvl.py
@@ -0,0 +1,670 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/managers/multimodal_processors/internvl.py
@@ -0,0 +1,232 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/internvl.py` added +696/-0; `python/sglang/srt/models/internvl.py` added +670/-0; `python/sglang/srt/managers/multimodal_processors/internvl.py` added +232/-0
- Risk and verification: The diff ships test coverage in `test/srt/test_vision_openai_server.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4433 - Support InternVL2.5

- Link: https://github.com/sgl-project/sglang/pull/4433
- Status/date: closed / 2025-05-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +1210/-16, 1464 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support InternVL2.5"; model line: InternVL 3.5; category: model support/runtime entry; main diff: `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/configs/model_config.py`; technical summary: Covers "Support InternVL2.5"; the main implementation surface is `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/configs/model_config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/internvl.py` added +733/-0 (733 lines); hunks: -0,0 +1,733; symbols: FlashAttention, __init__, forward, InternVisionEmbeddings, touching `FlashAttention, __init__, forward`; `python/sglang/srt/managers/tokenizer_manager.py` modified +7/-2 (9 lines); hunks: -49,7 +49,11; -187,7 +191,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/configs/model_config.py` modified +5/-1 (6 lines); hunks: -318,7 +318,10 @@ def _verify_quantization(self) -> None:; -472,6 +475,7 @@ def is_generation_model(model_architectures: List[str], is_e...; symbols: _verify_quantization, get_hf_eos_token_id, is_generation_model, touching `_verify_quantization, get_hf_eos_token_id, is_generation_model`; `python/sglang/srt/models/deepseek_janus_pro.py` modified +1/-1 (2 lines); hunks: -1984,7 +1984,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/internvl.py` added +733/-0 (733 lines); hunks: -0,0 +1,733; symbols: FlashAttention, __init__, forward, InternVisionEmbeddings
  - `python/sglang/srt/managers/tokenizer_manager.py` modified +7/-2 (9 lines); hunks: -49,7 +49,11; -187,7 +191,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/configs/model_config.py` modified +5/-1 (6 lines); hunks: -318,7 +318,10 @@ def _verify_quantization(self) -> None:; -472,6 +475,7 @@ def is_generation_model(model_architectures: List[str], is_e...; symbols: _verify_quantization, get_hf_eos_token_id, is_generation_model
  - `python/sglang/srt/models/deepseek_janus_pro.py` modified +1/-1 (2 lines); hunks: -1984,7 +1984,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/internlm2.py` modified +1/-1 (2 lines); hunks: -114,7 +114,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/internvl.py
@@ -0,0 +1,733 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/managers/tokenizer_manager.py
@@ -49,7 +49,11 @@
-from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
+from sglang.srt.hf_transformers_utils import (
+    get_processor,
+    get_tokenizer,
+    get_tokenizer_from_processor,
+)
diff -- python/sglang/srt/configs/model_config.py
@@ -318,7 +318,10 @@ def _verify_quantization(self) -> None:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/internvl.py` added +733/-0; `python/sglang/srt/managers/tokenizer_manager.py` modified +7/-2; `python/sglang/srt/configs/model_config.py` modified +5/-1; `python/sglang/srt/models/deepseek_janus_pro.py` modified +1/-1; `python/sglang/srt/models/internlm2.py` modified +1/-1; `python/sglang/srt/managers/image_processors/intern_vl.py` added +230/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_utils.py`, `test/srt/test_vision_openai_server.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6870 - vlm: adapt internvl to VisionAttention

- Link: https://github.com/sgl-project/sglang/pull/6870
- Status/date: merged / 2025-06-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/internvl.py`; associated commits `83d87685c531`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +103/-126, 408 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "vlm: adapt internvl to VisionAttention"; model line: InternVL 3.5; category: model implementation change; main diff: `python/sglang/srt/models/internvl.py`; technical summary: Covers "vlm: adapt internvl to VisionAttention"; the main implementation surface is `python/sglang/srt/models/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/internvl.py` modified +46/-102 (148 lines); hunks: -11,21 +11,19; -40,83 +38,32; symbols: FlashAttention, InternAttention, __init__, forward, touching `FlashAttention, InternAttention, __init__`.
- Code diff details:
  - `python/sglang/srt/models/internvl.py` modified +46/-102 (148 lines); hunks: -11,21 +11,19; -40,83 +38,32; symbols: FlashAttention, InternAttention, __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/internvl.py
@@ -11,21 +11,19 @@
-from typing import Iterable, List, Optional, Tuple, Union
+from typing import Iterable, List, Optional, Set, Tuple, Union
-from einops import rearrange, repeat
-from sgl_kernel.flash_attn import flash_attn_varlen_func
+from sglang.srt.layers.attention.vision import SingletonCache, VisionAttention
@@ -40,83 +38,32 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/internvl.py` modified +46/-102
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8067 - fix: fix the bug of loading Internvl3

- Link: https://github.com/sgl-project/sglang/pull/8067
- Status/date: merged / 2025-07-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/internvl.py`; associated commits `750838adc4f9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: fix the bug of loading Internvl3"; model line: InternVL 3.5; category: bug fix; main diff: `python/sglang/srt/configs/internvl.py`; technical summary: Covers "fix: fix the bug of loading Internvl3"; the main implementation surface is `python/sglang/srt/configs/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/configs/internvl.py` modified +3/-0 (3 lines); hunks: -9,6 +9,7; -311,6 +312,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/configs/internvl.py` modified +3/-0 (3 lines); hunks: -9,6 +9,7; -311,6 +312,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/internvl.py
@@ -9,6 +9,7 @@
+    Qwen2Config,
@@ -311,6 +312,8 @@ def __init__(
+        elif llm_config.get("architectures")[0] == "Qwen2ForCausalLM":
+            self.llm_config = Qwen2Config(**llm_config)
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/internvl.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9381 - fix: InternS1 don't recognize image, updates image token for InternVL processor

- Link: https://github.com/sgl-project/sglang/pull/9381
- Status/date: merged / 2025-08-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/internvl.py`; associated commits `84719b527a2d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +9/-17, 60 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: InternS1 don't recognize image, updates image token for InternVL processor"; model line: InternVL 3.5; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/internvl.py`; technical summary: Covers "fix: InternS1 don't recognize image, updates image token for InternVL processor"; the main implementation surface is `python/sglang/srt/multimodal/processors/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2 (9 lines); hunks: -44,7 +44,7 @@ def __init__(self, hf_config, server_args, _image_processor, *...; -218,13 +218,18 @@ def process_image_internvl(image, input_size=448, max_num=...; symbols: __init__, process_image_internvl, touching `__init__, process_image_internvl`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2 (9 lines); hunks: -44,7 +44,7 @@ def __init__(self, hf_config, server_args, _image_processor, *...; -218,13 +218,18 @@ def process_image_internvl(image, input_size=448, max_num=...; symbols: __init__, process_image_internvl
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -44,7 +44,7 @@ def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
-            image_token="<image>",
+            image_token="<IMG_CONTEXT>",
@@ -218,13 +218,18 @@ def process_image_internvl(image, input_size=448, max_num=12):
+        original_placeholder = "<<<__IMG_CONTEXT_PLACEHOLDER__>>>"
+        input_text = input_text.replace(self.IMG_CONTEXT_TOKEN, original_placeholder)
-            input_text = input_text.replace("<image>", image_tokens, 1)
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/conversation.py`, `python/sglang/srt/multimodal/processors/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9705 - Support the internvl3.5 family models in sglang

- Link: https://github.com/sgl-project/sglang/pull/9705
- Status/date: merged / 2025-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`; associated commits `f64b8e3e4e13`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +34/-0, 84 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support the internvl3.5 family models in sglang"; model line: InternVL 3.5; category: model support/runtime entry; main diff: `python/sglang/srt/models/internvl.py`, `python/sglang/srt/configs/internvl.py`; technical summary: Covers "Support the internvl3.5 family models in sglang"; the main implementation surface is `python/sglang/srt/models/internvl.py`, `python/sglang/srt/configs/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/internvl.py` modified +28/-0 (28 lines); hunks: -26,8 +26,10; -445,6 +447,14 @@ def __init__(; symbols: __init__, load_weights, touching `__init__, load_weights`; `python/sglang/srt/configs/internvl.py` modified +6/-0 (6 lines); hunks: -6,11 +6,13; -316,7 +318,11 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/internvl.py` modified +28/-0 (28 lines); hunks: -26,8 +26,10; -445,6 +447,14 @@ def __init__(; symbols: __init__, load_weights
  - `python/sglang/srt/configs/internvl.py` modified +6/-0 (6 lines); hunks: -6,11 +6,13; -316,7 +318,11 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/internvl.py
@@ -26,8 +26,10 @@
+from sglang.srt.models.gpt_oss import GptOssForCausalLM
+from sglang.srt.models.qwen3 import Qwen3ForCausalLM
@@ -445,6 +447,14 @@ def __init__(
+        elif config.llm_config.architectures[0] == "GptOssForCausalLM":
+            self.language_model = GptOssForCausalLM(
+                config=config.llm_config, quant_config=quant_config
diff -- python/sglang/srt/configs/internvl.py
@@ -6,11 +6,13 @@
+    GptOssConfig,
+    Qwen3MoeConfig,
@@ -316,7 +318,11 @@ def __init__(
+            self.llm_config = Qwen3MoeConfig(**llm_config)
+        elif llm_config.get("architectures")[0] == "Qwen3ForCausalLM":
+        elif llm_config.get("architectures")[0] == "GptOssForCausalLM":
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/internvl.py` modified +28/-0; `python/sglang/srt/configs/internvl.py` modified +6/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9795 - refactor(InternVL): Use gpu to preprocess the input image

- Link: https://github.com/sgl-project/sglang/pull/9795
- Status/date: merged / 2025-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/internvl.py`; associated commits `15f993472c58`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +141/-129, 340 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "refactor(InternVL): Use gpu to preprocess the input image"; model line: InternVL 3.5; category: model implementation change; main diff: `python/sglang/srt/multimodal/processors/internvl.py`; technical summary: Covers "refactor(InternVL): Use gpu to preprocess the input image"; the main implementation surface is `python/sglang/srt/multimodal/processors/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/internvl.py` modified +141/-129 (270 lines); hunks: -2,8 +2,10; -48,99 +50,6 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__, build_transform, resize_image, to_tensor, touching `__init__, build_transform, resize_image`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +141/-129 (270 lines); hunks: -2,8 +2,10; -48,99 +50,6 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__, build_transform, resize_image, to_tensor
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -2,8 +2,10 @@
-from decord import VideoReader, cpu
+import torchvision.transforms as T
+from decord import VideoReader, cpu, gpu
+from torchvision.transforms import InterpolationMode
@@ -48,99 +50,6 @@ def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
-    @staticmethod
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +141/-129
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10375 - fix(internvl): fix accuracy issue of normalization

- Link: https://github.com/sgl-project/sglang/pull/10375
- Status/date: merged / 2025-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/internvl.py`; associated commits `1fcccda4b2b3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +20/-8, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(internvl): fix accuracy issue of normalization"; model line: InternVL 3.5; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/internvl.py`; technical summary: Covers "fix(internvl): fix accuracy issue of normalization"; the main implementation surface is `python/sglang/srt/multimodal/processors/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/internvl.py` modified +20/-8 (28 lines); hunks: -1,5 +1,7; -19,6 +21,20; symbols: InternVLImageProcessor, _get_normalize_tensors, __init__, load_video, touching `InternVLImageProcessor, _get_normalize_tensors, __init__`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +20/-8 (28 lines); hunks: -1,5 +1,7; -19,6 +21,20; symbols: InternVLImageProcessor, _get_normalize_tensors, __init__, load_video
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -1,5 +1,7 @@
+from functools import lru_cache
@@ -19,6 +21,20 @@
+    IMAGENET_MEAN = [0.485, 0.456, 0.406]
+    IMAGENET_STD = [0.229, 0.224, 0.225]
+    @staticmethod
+    @lru_cache(maxsize=1)
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +20/-8
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13640 - [VLM] Support Piecewise CUDA Graph for InternVL

- Link: https://github.com/sgl-project/sglang/pull/13640
- Status/date: merged / 2025-11-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/internvl.py`; associated commits `475962a139d1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +103/-13, 183 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Support Piecewise CUDA Graph for InternVL"; model line: InternVL 3.5; category: performance/backend optimization; main diff: `python/sglang/srt/models/internvl.py`; technical summary: Covers "[VLM] Support Piecewise CUDA Graph for InternVL"; the main implementation surface is `python/sglang/srt/models/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/internvl.py` modified +21/-10 (31 lines); hunks: -14,10 +14,7; -471,6 +468,12 @@ def __init__(; symbols: __init__, pixel_shuffle, forward, pad_input_ids, touching `__init__, pixel_shuffle, forward`.
- Code diff details:
  - `python/sglang/srt/models/internvl.py` modified +21/-10 (31 lines); hunks: -14,10 +14,7; -471,6 +468,12 @@ def __init__(; symbols: __init__, pixel_shuffle, forward, pad_input_ids
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/internvl.py
@@ -14,10 +14,7 @@
-from sglang.srt.managers.mm_utils import (
-    MultiModalityDataPaddingPatternTokenPairs,
-    general_mm_embed_routine,
-)
+from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternTokenPairs
@@ -471,6 +468,12 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/internvl.py` modified +21/-10
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_piecewise_cuda_graph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13925 - [VLM] Support InternVL Vision Encoder Data Parallelism

- Link: https://github.com/sgl-project/sglang/pull/13925
- Status/date: merged / 2025-11-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/internvl.py`; associated commits `ca5c8b16f67d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +118/-25, 266 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Support InternVL Vision Encoder Data Parallelism"; model line: InternVL 3.5; category: model support/runtime entry; main diff: `python/sglang/srt/models/internvl.py`; technical summary: Covers "[VLM] Support InternVL Vision Encoder Data Parallelism"; the main implementation surface is `python/sglang/srt/models/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/internvl.py` modified +83/-25 (108 lines); hunks: -7,11 +7,16; -28,6 +33,8; symbols: __init__, forward, InternMLP, touching `__init__, forward, InternMLP`.
- Code diff details:
  - `python/sglang/srt/models/internvl.py` modified +83/-25 (108 lines); hunks: -7,11 +7,16; -28,6 +33,8; symbols: __init__, forward, InternMLP
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/internvl.py
@@ -7,11 +7,16 @@
-from transformers.activations import ACT2FN
+from sglang.srt.distributed import (
+    get_tensor_model_parallel_rank,
+    get_tensor_model_parallel_world_size,
+)
+from sglang.srt.layers.activation import get_act_fn
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/internvl.py` modified +83/-25
- Risk and verification: The diff ships test coverage in `test/nightly/test_encoder_dp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15942 - [VLM] Support Video for InternVL3_5

- Link: https://github.com/sgl-project/sglang/pull/15942
- Status/date: merged / 2025-12-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py`; associated commits `94bcc19bcef6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +426/-118, 658 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Support Video for InternVL3_5"; model line: InternVL 3.5; category: model support/runtime entry; main diff: `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`; technical summary: Covers "[VLM] Support Video for InternVL3_5"; the main implementation surface is `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/internvl.py` modified +418/-118 (536 lines); hunks: -1,6 +1,8; -15,26 +17,44; symbols: InternVLImageProcessor, InternVLProcessor, _get_normalize_tensors, __init__, touching `InternVLImageProcessor, InternVLProcessor, _get_normalize_tensors`; `python/sglang/srt/models/internvl.py` modified +8/-0 (8 lines); hunks: -539,6 +539,7 @@ def __init__(; -594,6 +595,13 @@ def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: __init__, get_image_feature, get_video_feature, forward, touching `__init__, get_image_feature, get_video_feature`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +418/-118 (536 lines); hunks: -1,6 +1,8; -15,26 +17,44; symbols: InternVLImageProcessor, InternVLProcessor, _get_normalize_tensors, __init__
  - `python/sglang/srt/models/internvl.py` modified +8/-0 (8 lines); hunks: -539,6 +539,7 @@ def __init__(; -594,6 +595,13 @@ def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: __init__, get_image_feature, get_video_feature, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -1,6 +1,8 @@
+import logging
+from typing import List
@@ -15,26 +17,44 @@
+logger = logging.getLogger(__name__)
-class InternVLImageProcessor(BaseMultimodalProcessor):
+class InternVLProcessor(BaseMultimodalProcessor):
diff -- python/sglang/srt/models/internvl.py
@@ -539,6 +539,7 @@ def __init__(
+            Modality.VIDEO: self.get_video_feature,
@@ -594,6 +595,13 @@ def get_image_feature(self, items: List[MultimodalDataItem]):
+    def get_video_feature(self, items: List[MultimodalDataItem]):
+        # items: each item corresponds to one video (recommended)
+        # item.feature shape: [num_frames, 3, 448, 448]  (or [num_tiles, 3, 448, 448])
+        pixel_values = torch.cat([item.feature for item in items], dim=0)
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +418/-118; `python/sglang/srt/models/internvl.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16732 - [VLM] Support ViT CUDA Graph for InternVL

- Link: https://github.com/sgl-project/sglang/pull/16732
- Status/date: merged / 2026-01-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py`; associated commits `feae615b1146`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +219/-6, 304 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Support ViT CUDA Graph for InternVL"; model line: InternVL 3.5; category: performance/backend optimization; main diff: `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py`, `python/sglang/srt/models/internvl.py`; technical summary: Covers "[VLM] Support ViT CUDA Graph for InternVL"; the main implementation surface is `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py`, `python/sglang/srt/models/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py` added +183/-0 (183 lines); hunks: -0,0 +1,183; symbols: InternViTCudaGraphRunner, __init__, device, dtype, touching `InternViTCudaGraphRunner, __init__, device`; `python/sglang/srt/models/internvl.py` modified +27/-3 (30 lines); hunks: -13,6 +13,7; -36,6 +37,9; symbols: forward, __init__, touching `forward, __init__`.
- Code diff details:
  - `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py` added +183/-0 (183 lines); hunks: -0,0 +1,183; symbols: InternViTCudaGraphRunner, __init__, device, dtype
  - `python/sglang/srt/models/internvl.py` modified +27/-3 (30 lines); hunks: -13,6 +13,7; -36,6 +37,9; symbols: forward, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py
@@ -0,0 +1,183 @@
+# Copyright 2023-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/internvl.py
@@ -13,6 +13,7 @@
+from sglang.srt.environ import envs
@@ -36,6 +37,9 @@
+from sglang.srt.multimodal.internvl_vit_cuda_graph_runner import (
+    InternViTCudaGraphRunner,
+)
@@ -82,8 +86,9 @@ def forward(
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py` added +183/-0; `python/sglang/srt/models/internvl.py` modified +27/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/internvl.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17040 - fix(processor): support InternS1 text_config in InternVL processor

- Link: https://github.com/sgl-project/sglang/pull/17040
- Status/date: merged / 2026-01-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/internvl.py`; associated commits `539924037fbc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-4, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(processor): support InternS1 text_config in InternVL processor"; model line: InternVL 3.5; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/internvl.py`; technical summary: Covers "fix(processor): support InternS1 text_config in InternVL processor"; the main implementation surface is `python/sglang/srt/multimodal/processors/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/internvl.py` modified +12/-4 (16 lines); hunks: -72,7 +72,17 @@ def __init__(self, hf_config, server_args, _image_processor,...; -121,9 +131,7 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +12/-4 (16 lines); hunks: -72,7 +72,17 @@ def __init__(self, hf_config, server_args, _image_processor,...; -121,9 +131,7 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -72,7 +72,17 @@ def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
-        llm_arch = hf_config.llm_config.architectures[0]
+        # Support both InternVL (llm_config) and InternS1 (text_config).
+        # Different multimodal models use different field names for the text backbone:
+        # - InternVL uses: hf_config.llm_config
+        # - InternS1 uses: hf_config.text_config
+        # - Some store architectures at top-level
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +12/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19127 - [vlm][internVL] Support processor and embedding inputs for InternVL

- Link: https://github.com/sgl-project/sglang/pull/19127
- Status/date: merged / 2026-02-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py`; associated commits `f0c208959794`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +282/-7, 379 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[vlm][internVL] Support processor and embedding inputs for InternVL"; model line: InternVL 3.5; category: model support/runtime entry; main diff: `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`; technical summary: Covers "[vlm][internVL] Support processor and embedding inputs for InternVL"; the main implementation surface is `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/internvl.py` modified +109/-1 (110 lines); hunks: -9,11 +9,15; -255,9 +259,113 @@ def _resolve_video_num_frames(; symbols: _resolve_video_num_frames, _has_special_format, _process_special_format, process_and_combine_mm_data, touching `_resolve_video_num_frames, _has_special_format, _process_special_format`; `python/sglang/srt/models/internvl.py` modified +7/-0 (7 lines); hunks: -616,13 +616,20 @@ def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: get_image_feature, get_video_feature, touching `get_image_feature, get_video_feature`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +109/-1 (110 lines); hunks: -9,11 +9,15; -255,9 +259,113 @@ def _resolve_video_num_frames(; symbols: _resolve_video_num_frames, _has_special_format, _process_special_format, process_and_combine_mm_data
  - `python/sglang/srt/models/internvl.py` modified +7/-0 (7 lines); hunks: -616,13 +616,20 @@ def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: get_image_feature, get_video_feature
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -9,11 +9,15 @@
-from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
+from sglang.srt.managers.schedule_batch import (
+    Modality,
+    MultimodalDataItem,
+)
+    BaseMultiModalProcessorOutput,
diff -- python/sglang/srt/models/internvl.py
@@ -616,13 +616,20 @@ def get_image_feature(self, items: List[MultimodalDataItem]):
+        # If already precomputed embeddings (not raw pixel values), skip vision encoder.
+        # Normal pixel_values are 4D [N, C, H, W]; precomputed embeddings are 2D or 3D.
+        if pixel_values.dim() != 4:
+            return pixel_values
+        # If already precomputed embeddings, skip vision encoder.
+        if pixel_values.dim() != 4:
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +109/-1; `python/sglang/srt/models/internvl.py` modified +7/-0
- Risk and verification: The diff ships test coverage in `test/registered/vlm/test_vlm_input_format.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19997 - Fix InternVL and vision attention for non-CUDA backends (e.g. XPU)

- Link: https://github.com/sgl-project/sglang/pull/19997
- Status/date: merged / 2026-03-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/internvl.py`; associated commits `7458407437ca`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +184/-14, 324 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix InternVL and vision attention for non-CUDA backends (e.g. XPU)"; model line: InternVL 3.5; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/internvl.py`, `test/srt/xpu/test_internvl.py`; technical summary: Covers "Fix InternVL and vision attention for non-CUDA backends (e.g. XPU)"; the main implementation surface is `python/sglang/srt/multimodal/processors/internvl.py`, `test/srt/xpu/test_internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/internvl.py` modified +17/-10 (27 lines); hunks: -19,6 +19,7; -434,7 +435,7 @@ async def process_qwen_mm_data_async(; symbols: process_qwen_mm_data_async, process_internlm2_mm_data_async, touching `process_qwen_mm_data_async, process_internlm2_mm_data_async`; `test/srt/xpu/test_internvl.py` added +147/-0 (147 lines); hunks: -0,0 +1,147; symbols: InternVLXPUServerBase, setUpClass, tearDownClass, TestInternVL25Server, touching `InternVLXPUServerBase, setUpClass, tearDownClass`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +17/-10 (27 lines); hunks: -19,6 +19,7; -434,7 +435,7 @@ async def process_qwen_mm_data_async(; symbols: process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `test/srt/xpu/test_internvl.py` added +147/-0 (147 lines); hunks: -0,0 +1,147; symbols: InternVLXPUServerBase, setUpClass, tearDownClass, TestInternVL25Server
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -19,6 +19,7 @@
+from sglang.srt.utils import get_device
@@ -434,7 +435,7 @@ async def process_qwen_mm_data_async(
-        mean, std = self._get_normalize_tensors(device="cuda")
+        mean, std = self._get_normalize_tensors(device=get_device())
@@ -444,10 +445,11 @@ async def process_qwen_mm_data_async(
-                    torch.from_numpy(img_np).permute(2, 0, 1).cuda().float() / 255.0
diff -- test/srt/xpu/test_internvl.py
@@ -0,0 +1,147 @@
+"""
+XPU tests for InternVL models (InternVL2.5-2B, InternVL3.5-2B).
+Uses the same structure as test_vision_openai_server_a.py: OpenAI /v1 chat API
+and ImageOpenAITestMixin. An XPU-specific base injects --device xpu and
+--attention-backend intel_xpu.
+Usage (pick module path to match your cwd):
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +17/-10
  - tests: `test/srt/xpu/test_internvl.py` added +147/-0
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/xpu/test_internvl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20282 - Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug

- Link: https://github.com/sgl-project/sglang/pull/20282
- Status/date: merged / 2026-03-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +704/-90, 1053 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug"; model line: InternVL 3.5; category: bug fix; main diff: `python/sglang/srt/layers/conv.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/pixtral.py`; technical summary: Covers "Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug"; the main implementation surface is `python/sglang/srt/layers/conv.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/pixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #17784 - Upgrade transformers==5.3.0

- Link: https://github.com/sgl-project/sglang/pull/17784
- Status/date: merged / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 95 files, +1136/-343, 2752 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Upgrade transformers==5.3.0"; model line: InternVL 3.5; category: model implementation change; main diff: `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py`; technical summary: Covers "Upgrade transformers==5.3.0"; the main implementation surface is `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update, touching `__init__, Gemma3RotaryEmbedding, _dynamic_frequency_update`; `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope, touching `_get_rope_param, get_rope`; `python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes, touching `ModelImpl, is_deepseek_nsa, _derive_model_shapes`; `python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__, touching `compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope`.
- Code diff details:
  - `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update
  - `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope
  - `python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes
  - `python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__
  - `python/sglang/srt/models/midashenglm.py` modified +6/-14 (20 lines); hunks: -476,20 +476,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gemma3_causal.py
@@ -166,18 +166,36 @@ def __init__(
+        # In transformers v5, rope_parameters is nested per layer type:
+        #   {"sliding_attention": {"rope_theta": 10000}, "full_attention": {"rope_theta": 1000000}}
+        # In v4 it was flat: {"rope_type": "default", "rope_theta": ...}
+        rope_params = config.rope_parameters
+        is_nested = isinstance(rope_params, dict) and "full_attention" in rope_params
-            self.rope_theta = config.rope_local_base_freq
diff -- python/sglang/srt/layers/rotary_embedding/factory.py
@@ -2,6 +2,7 @@
+import logging
@@ -26,6 +27,29 @@
+logger = logging.getLogger(__name__)
+def _get_rope_param(rope_scaling, key, default, scaling_type):
+    """Get a parameter from rope_scaling dict, warn if missing.
+    In transformers v5, config.rope_scaling is an alias for rope_parameters
diff -- python/sglang/srt/configs/model_config.py
@@ -51,10 +51,20 @@ class ModelImpl(str, Enum):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14; `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13; `python/sglang/srt/configs/model_config.py` modified +38/-18; `python/sglang/srt/models/qwen3_moe.py` modified +14/-7; `python/sglang/srt/models/midashenglm.py` modified +6/-14; `python/sglang/srt/models/glm4.py` modified +3/-14
- Risk and verification: The diff ships test coverage in `python/sglang/test/runners.py`, `test/registered/core/test_score_api.py`, `test/registered/quant/test_awq.py`, `test/registered/rl/test_multi_instance_release_memory_occupation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19749 - [Feature] Optimizations for JPEG input on NVIDIA GPU

- Link: https://github.com/sgl-project/sglang/pull/19749
- Status/date: merged / 2026-03-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +114/-46, 301 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Optimizations for JPEG input on NVIDIA GPU"; model line: InternVL 3.5; category: performance/backend optimization; main diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/llava.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`; technical summary: Covers "[Feature] Optimizations for JPEG input on NVIDIA GPU"; the main implementation surface is `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/llava.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/base_processor.py` modified +14/-6 (20 lines); hunks: -173,6 +173,7 @@ def get_combined_regex(self) -> re.Pattern:; -468,8 +469,9 @@ def get_estimated_frames_list(self, image_data):; symbols: get_combined_regex, BaseMultimodalProcessor, __init__, get_estimated_frames_list, touching `get_combined_regex, BaseMultimodalProcessor, __init__`; `python/sglang/srt/multimodal/processors/llava.py` modified +2/-1 (3 lines); hunks: -33,6 +33,7 @@ class LlavaImageProcessor(BaseMultimodalProcessor):; -49,7 +50,7 @@ def _process_single_image_task(; symbols: LlavaImageProcessor, __init__, _process_single_image_task, touching `LlavaImageProcessor, __init__, _process_single_image_task`; `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +3/-0 (3 lines); hunks: -35,6 +35,9; symbols: NanoNemotronVLImageProcessor, __init__, touching `NanoNemotronVLImageProcessor, __init__`; `python/sglang/srt/multimodal/processors/internvl.py` modified +1/-0 (1 lines); hunks: -27,6 +27,7; symbols: InternVLProcessor, touching `InternVLProcessor`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +14/-6 (20 lines); hunks: -173,6 +173,7 @@ def get_combined_regex(self) -> re.Pattern:; -468,8 +469,9 @@ def get_estimated_frames_list(self, image_data):; symbols: get_combined_regex, BaseMultimodalProcessor, __init__, get_estimated_frames_list
  - `python/sglang/srt/multimodal/processors/llava.py` modified +2/-1 (3 lines); hunks: -33,6 +33,7 @@ class LlavaImageProcessor(BaseMultimodalProcessor):; -49,7 +50,7 @@ def _process_single_image_task(; symbols: LlavaImageProcessor, __init__, _process_single_image_task
  - `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +3/-0 (3 lines); hunks: -35,6 +35,9; symbols: NanoNemotronVLImageProcessor, __init__
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +1/-0 (1 lines); hunks: -27,6 +27,7; symbols: InternVLProcessor
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +1/-0 (1 lines); hunks: -16,6 +16,7; symbols: KimiK2_5VLImageProcessor, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -173,6 +173,7 @@ def get_combined_regex(self) -> re.Pattern:
+    gpu_image_decode = True  # Enable GPU decoding by default
@@ -468,8 +469,9 @@ def get_estimated_frames_list(self, image_data):
-    @staticmethod
+    @classmethod
+        cls,
@@ -481,7 +483,8 @@ def _load_single_item(
diff -- python/sglang/srt/multimodal/processors/llava.py
@@ -33,6 +33,7 @@ class LlavaImageProcessor(BaseMultimodalProcessor):
+    gpu_image_decode = False  # Llava processes loaded image as PIL image explicitly
@@ -49,7 +50,7 @@ def _process_single_image_task(
-            image, image_size = load_image(url)
+            image, image_size = load_image(url, False)
diff -- python/sglang/srt/multimodal/processors/nano_nemotron_vl.py
@@ -35,6 +35,9 @@
+    gpu_image_decode = (
+        False  # NanoNemotronVL processes loaded image as PIL image explicitly
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/base_processor.py` modified +14/-6; `python/sglang/srt/multimodal/processors/llava.py` modified +2/-1; `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +3/-0; `python/sglang/srt/multimodal/processors/internvl.py` modified +1/-0; `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +1/-0; `python/sglang/srt/multimodal/processors/kimi_vl.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21899 - [VLM] Enable per-image MM splitting by default and remove MULTI_IMAGES modality

- Link: https://github.com/sgl-project/sglang/pull/21899
- Status/date: merged / 2026-04-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +217/-136, 647 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Enable per-image MM splitting by default and remove MULTI_IMAGES modality"; model line: InternVL 3.5; category: model support/runtime entry; main diff: `python/sglang/srt/models/llava.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/llava.py`; technical summary: Covers "[VLM] Enable per-image MM splitting by default and remove MULTI_IMAGES modality"; the main implementation surface is `python/sglang/srt/models/llava.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/llava.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/llava.py` modified +47/-34 (81 lines); hunks: -55,6 +55,21; -63,13 +78,8 @@ def pad_input_ids(self, input_ids: List[int], image_inputs: M...; symbols: LlavaBaseForCausalLM, _infer_image_aspect_ratio, pad_input_ids, forward, touching `LlavaBaseForCausalLM, _infer_image_aspect_ratio, pad_input_ids`; `python/sglang/srt/multimodal/processors/internvl.py` modified +28/-8 (36 lines); hunks: -588,11 +588,21 @@ async def process_qwen_mm_data_async(; -702,11 +712,21 @@ async def process_internlm2_mm_data_async(; symbols: process_qwen_mm_data_async, process_internlm2_mm_data_async, touching `process_qwen_mm_data_async, process_internlm2_mm_data_async`; `python/sglang/srt/multimodal/processors/llava.py` modified +16/-11 (27 lines); hunks: -187,34 +187,39 @@ async def process_mm_data_async(; symbols: process_mm_data_async, touching `process_mm_data_async`; `python/sglang/srt/multimodal/processors/minicpm.py` modified +19/-7 (26 lines); hunks: -223,6 +223,8 @@ async def process_mm_data_async(; -231,6 +233,7 @@ async def process_mm_data_async(; symbols: process_mm_data_async, touching `process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/models/llava.py` modified +47/-34 (81 lines); hunks: -55,6 +55,21; -63,13 +78,8 @@ def pad_input_ids(self, input_ids: List[int], image_inputs: M...; symbols: LlavaBaseForCausalLM, _infer_image_aspect_ratio, pad_input_ids, forward
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +28/-8 (36 lines); hunks: -588,11 +588,21 @@ async def process_qwen_mm_data_async(; -702,11 +712,21 @@ async def process_internlm2_mm_data_async(; symbols: process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `python/sglang/srt/multimodal/processors/llava.py` modified +16/-11 (27 lines); hunks: -187,34 +187,39 @@ async def process_mm_data_async(; symbols: process_mm_data_async
  - `python/sglang/srt/multimodal/processors/minicpm.py` modified +19/-7 (26 lines); hunks: -223,6 +223,8 @@ async def process_mm_data_async(; -231,6 +233,7 @@ async def process_mm_data_async(; symbols: process_mm_data_async
  - `python/sglang/srt/models/minicpmv.py` modified +15/-3 (18 lines); hunks: -993,7 +993,11 @@ def pad_input_ids(self, input_ids: List[int], image_inputs:...; -1155,7 +1159,11 @@ def pad_input_ids(self, input_ids: List[int], image_input...; symbols: pad_input_ids
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llava.py
@@ -55,6 +55,21 @@
+    @staticmethod
+    def _infer_image_aspect_ratio(mm_items):
+        """Determine image_aspect_ratio from processor metadata or item count."""
+        # Check if processor stored the aspect_ratio it used
+        for item in mm_items:
+            ar = item.model_specific_data.get("image_aspect_ratio")
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -588,11 +588,21 @@ async def process_qwen_mm_data_async(
-            items.append(
-                MultimodalDataItem(
-                    feature=image_tensor, modality=Modality.IMAGE, offsets=image_offsets
-                )
+            # Split per-image for better cache granularity
+            assert len(num_patches_list) == len(image_offsets), (
diff -- python/sglang/srt/multimodal/processors/llava.py
@@ -187,34 +187,39 @@ async def process_mm_data_async(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llava.py` modified +47/-34; `python/sglang/srt/multimodal/processors/internvl.py` modified +28/-8; `python/sglang/srt/multimodal/processors/llava.py` modified +16/-11; `python/sglang/srt/multimodal/processors/minicpm.py` modified +19/-7; `python/sglang/srt/models/minicpmv.py` modified +15/-3; `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-3
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_mm_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21738 - refactor: replace mm_inputs dict with MultimodalProcessorOutput

- Link: https://github.com/sgl-project/sglang/pull/21738
- Status/date: merged / 2026-04-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 40 files, +408/-314, 1321 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "refactor: replace mm_inputs dict with MultimodalProcessorOutput"; model line: InternVL 3.5; category: model implementation change; main diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`; technical summary: Covers "refactor: replace mm_inputs dict with MultimodalProcessorOutput"; the main implementation surface is `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +27/-23 (50 lines); hunks: -12,7 +12,11; -474,17 +478,17 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, process_mm_data_async, touching `get_mm_data, process_mm_data_async`; `python/sglang/srt/multimodal/processors/internvl.py` modified +25/-24 (49 lines); hunks: -11,6 +11,7; -337,14 +338,14 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async, process_qwen_mm_data_async, process_internlm2_mm_data_async, touching `_process_special_format, process_mm_data_async, process_qwen_mm_data_async`; `python/sglang/srt/multimodal/processors/minicpm.py` modified +23/-22 (45 lines); hunks: -5,6 +5,7; -158,17 +159,17 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async, touching `_process_special_format, process_mm_data_async`; `python/sglang/srt/multimodal/processors/interns1pro.py` modified +23/-19 (42 lines); hunks: -1,7 +1,11; -26,15 +30,15 @@ def get_mm_data(self, prompt, embeddings, img_grid_thw):; symbols: get_mm_data, process_mm_data_async, touching `get_mm_data, process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +27/-23 (50 lines); hunks: -12,7 +12,11; -474,17 +478,17 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +25/-24 (49 lines); hunks: -11,6 +11,7; -337,14 +338,14 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async, process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `python/sglang/srt/multimodal/processors/minicpm.py` modified +23/-22 (45 lines); hunks: -5,6 +5,7; -158,17 +159,17 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/interns1pro.py` modified +23/-19 (42 lines); hunks: -1,7 +1,11; -26,15 +30,15 @@ def get_mm_data(self, prompt, embeddings, img_grid_thw):; symbols: get_mm_data, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/qwen_audio.py` modified +19/-15 (34 lines); hunks: -1,6 +1,10; -69,13 +73,13 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, process_mm_data_async
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -12,7 +12,11 @@
-from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
+from sglang.srt.managers.schedule_batch import (
+    Modality,
+    MultimodalDataItem,
+    MultimodalProcessorOutput,
+)
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -11,6 +11,7 @@
+    MultimodalProcessorOutput,
@@ -337,14 +338,14 @@ async def _process_special_format(
-        return {
-            "input_ids": input_ids_tensor.flatten().tolist(),
-            "mm_items": mm_items,
-            "im_start_id": self.img_start_token_id,
diff -- python/sglang/srt/multimodal/processors/minicpm.py
@@ -5,6 +5,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +27/-23; `python/sglang/srt/multimodal/processors/internvl.py` modified +25/-24; `python/sglang/srt/multimodal/processors/minicpm.py` modified +23/-22; `python/sglang/srt/multimodal/processors/interns1pro.py` modified +23/-19; `python/sglang/srt/multimodal/processors/qwen_audio.py` modified +19/-15; `python/sglang/srt/multimodal/processors/transformers_auto.py` modified +18/-14
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/disaggregation/encode_receiver.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/managers/io_struct.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- Link: https://github.com/sgl-project/sglang/pull/23001
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 330 files, +80364/-0, 68714 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add new Mintlify documentation site (docs_new/)"; model line: InternVL 3.5; category: docs/tests/CI; main diff: `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`; technical summary: Covers "Add new Mintlify documentation site (docs_new/)"; the main implementation surface is `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0 (740 lines); hunks: -0,0 +1,740; symbols: get_messages, get_current_weather, convert_dict_to_tool, in, touching `get_messages, get_current_weather, convert_dict_to_tool`; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: CapitalInfo, get_messages, touching `CapitalInfo, get_messages`; `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0 (317 lines); hunks: -0,0 +1,317; `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0 (3327 lines).
- Code diff details:
  - `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0 (740 lines); hunks: -0,0 +1,740; symbols: get_messages, get_current_weather, convert_dict_to_tool, in
  - `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: CapitalInfo, get_messages
  - `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0 (317 lines); hunks: -0,0 +1,317
  - `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0 (3327 lines)
  - `docs_new/docs/hardware-platforms/ascend-npus/Best-Practice-on-Ascend-NPU.mdx` added +2911/-0 (2911 lines)
- Key code excerpts:

```diff
diff -- docs_new/docs/advanced_features/tool_parser.mdx
@@ -0,0 +1,740 @@
+---
+title: "Tool Parser"
+metatags:
+    description: "SGLang function calling: tool parsers for DeepSeek, Llama, Qwen, Mistral, GLM, Kimi K2. OpenAI-compatible tool use API."
+---
+This guide demonstrates how to use SGLang’s [Function calling](https://platform.openai.com/docs/guides/function-calling) functionality.
diff -- docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx
@@ -0,0 +1,663 @@
+---
+title: "Structured Outputs For Reasoning Models"
+metatags:
+    description: "SGLang structured outputs for reasoning models: free-form thinking with constrained final output for DeepSeek R1, QwQ models."
+---
+When working with reasoning models that use special tokens like `&lt;think&gt;...&lt;/think&gt;` to denote reasoning sections, you might want to allow free-form text within these
diff -- docs_new/docs/advanced_features/separate_reasoning.mdx
@@ -0,0 +1,317 @@
```

- Reviewed files:
  - docs: `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0; `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0; `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0; `docs_new/docs/hardware-platforms/ascend-npus/Best-Practice-on-Ascend-NPU.mdx` added +2911/-0; `docs_new/docs/advanced_features/server_arguments.mdx` added +2871/-0
- Risk and verification: This is mostly docs/examples in `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore`, `docs_new/.mintignore`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23568 - Parakeet nemotron encoder

- Link: https://github.com/sgl-project/sglang/pull/23568
- Status/date: merged / 2026-04-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +1289/-116, 1817 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Parakeet nemotron encoder"; model line: InternVL 3.5; category: model implementation change; main diff: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/internvl_utils.py`, `python/sglang/srt/models/radio.py`; technical summary: Covers "Parakeet nemotron encoder"; the main implementation surface is `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/internvl_utils.py`, `python/sglang/srt/models/radio.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +322/-36 (358 lines); hunks: -11,23 +11,39; -63,18 +79,62 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__, preprocess_image, render_image, render_image_dynamic, touching `__init__, preprocess_image, render_image`; `python/sglang/srt/multimodal/internvl_utils.py` modified +239/-0 (239 lines); hunks: -1,4 +1,6; -113,3 +115,240 @@ def image_to_pixel_values(; symbols: image_to_pixel_values, compute_dynamic_image_size, dynamic_resize_image, resize_image_to_pixels, touching `image_to_pixel_values, compute_dynamic_image_size, dynamic_resize_image`; `python/sglang/srt/models/radio.py` modified +135/-57 (192 lines); hunks: -13,6 +13,7; -33,6 +34,8; symbols: forward, ViTPatchGenerator, __init__, touching `forward, ViTPatchGenerator, __init__`; `python/sglang/srt/models/nano_nemotron_vl.py` modified +171/-20 (191 lines); hunks: -35,8 +35,10; -66,9 +68,13 @@ def __init__(; symbols: __init__, pad_input_ids, pixel_shuffle, touching `__init__, pad_input_ids, pixel_shuffle`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +322/-36 (358 lines); hunks: -11,23 +11,39; -63,18 +79,62 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__, preprocess_image, render_image, render_image_dynamic
  - `python/sglang/srt/multimodal/internvl_utils.py` modified +239/-0 (239 lines); hunks: -1,4 +1,6; -113,3 +115,240 @@ def image_to_pixel_values(; symbols: image_to_pixel_values, compute_dynamic_image_size, dynamic_resize_image, resize_image_to_pixels
  - `python/sglang/srt/models/radio.py` modified +135/-57 (192 lines); hunks: -13,6 +13,7; -33,6 +34,8; symbols: forward, ViTPatchGenerator, __init__
  - `python/sglang/srt/models/nano_nemotron_vl.py` modified +171/-20 (191 lines); hunks: -35,8 +35,10; -66,9 +68,13 @@ def __init__(; symbols: __init__, pad_input_ids, pixel_shuffle
  - `python/sglang/srt/models/parakeet.py` added +182/-0 (182 lines); hunks: -0,0 +1,182; symbols: ParakeetProjection, __init__, forward, ProjectedParakeet
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/nano_nemotron_vl.py
@@ -11,23 +11,39 @@
+import logging
+import math
-from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
+from sglang.srt.managers.schedule_batch import (
+    Modality,
+    MultimodalDataItem,
diff -- python/sglang/srt/multimodal/internvl_utils.py
@@ -1,4 +1,6 @@
+import math
@@ -113,3 +115,240 @@ def image_to_pixel_values(
+def compute_dynamic_image_size(
+    orig_w: int,
+    orig_h: int,
+    patch_size: int,
diff -- python/sglang/srt/models/radio.py
@@ -13,6 +13,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +322/-36; `python/sglang/srt/multimodal/internvl_utils.py` modified +239/-0; `python/sglang/srt/models/radio.py` modified +135/-57; `python/sglang/srt/models/nano_nemotron_vl.py` modified +171/-20; `python/sglang/srt/models/parakeet.py` added +182/-0; `python/sglang/srt/multimodal/audio_from_video.py` added +89/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/configs/parakeet.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25182 - chore: add vLLM SPDX copyright headers to ported files

- Link: https://github.com/sgl-project/sglang/pull/25182
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 136 files, +255/-0, 872 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "chore: add vLLM SPDX copyright headers to ported files"; model line: InternVL 3.5; category: model support/runtime entry; main diff: `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`; technical summary: Covers "chore: add vLLM SPDX copyright headers to ported files"; the main implementation surface is `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7; `python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7; `python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6; `python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6.
- Code diff details:
  - `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma2.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/baichuan.py
@@ -1,3 +1,7 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/baichuan.py
diff -- python/sglang/srt/models/commandr.py
@@ -1,3 +1,7 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/commandr.py
diff -- python/sglang/srt/models/dbrx.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
diff -- python/sglang/srt/models/gemma.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/baichuan.py` modified +4/-0; `python/sglang/srt/models/commandr.py` modified +4/-0; `python/sglang/srt/models/dbrx.py` modified +3/-0; `python/sglang/srt/models/gemma.py` modified +3/-0; `python/sglang/srt/models/gemma2.py` modified +3/-0; `python/sglang/srt/models/gpt_bigcode.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_custom_ops.py`, `python/sglang/test/test_marlin_utils.py`, `sgl-kernel/tests/test_causal_conv1d.py`, `test/registered/layers/mamba/test_causal_conv1d.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24751 - fix(mm): make multimodal data loading non-blocking to prevent health check stalls

- Link: https://github.com/sgl-project/sglang/pull/24751
- Status/date: merged / 2026-05-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 35 files, +45/-44, 401 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(mm): make multimodal data loading non-blocking to prevent health check stalls"; model line: InternVL 3.5; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`; technical summary: Covers "fix(mm): make multimodal data loading non-blocking to prevent health check stalls"; the main implementation surface is `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data, touching `_process_loaded_mm_data, load_mm_data`; `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async, touching `_process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async`; `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async, touching `_process_special_format, process_mm_data_async`; `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async, touching `__init__, process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1 (2 lines); hunks: -29,7 +29,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -1,3 +1,4 @@
+import asyncio
@@ -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):
-    def load_mm_data(
+    async def load_mm_data(
@@ -772,7 +773,7 @@ def load_mm_data(
-            return self.legacy_load_mm_data(
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -310,7 +310,7 @@ async def _process_special_format(
-            base_output = self.load_mm_data(
+            base_output = await self.load_mm_data(
@@ -423,7 +423,7 @@ async def process_qwen_mm_data_async(
-        base_output = self.load_mm_data(
+        base_output = await self.load_mm_data(
@@ -644,7 +644,7 @@ async def process_internlm2_mm_data_async(
diff -- python/sglang/srt/multimodal/processors/minicpm.py
@@ -118,7 +118,7 @@ async def _process_special_format(
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7; `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3; `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2; `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_vl_v2.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/clip.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- Link: https://github.com/sgl-project/sglang/pull/28567
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 184 files, +1865/-1727, 8932 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add get_parallel(): a structured accessor for parallel-topology state"; model line: InternVL 3.5; category: model support/runtime entry; main diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Add get_parallel(): a structured accessor for parallel-topology state"; the main implementation surface is `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention, touching `ApertusMLP, __init__, forward`; `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales, touching `__init__, forward, load_kv_cache_scales`; `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__, touching `_resolve_moe_input_pad_multiple, __init__`; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention
  - `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales
  - `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__
  - `python/sglang/srt/layers/communicator.py` modified +13/-19 (32 lines); hunks: -23,8 +23,6; -44,12 +42,7; symbols: apply_aiter_all_reduce_fusion, init_context, should_fuse_mlp_allreduce_with_next_layer, is_same_group_size
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/apertus.py
@@ -1,687 +1,686 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Copyright 2025 The SwissAI Initiative
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
diff -- python/sglang/srt/models/solar.py
@@ -1,37 +1,14 @@
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
-# Copyright 2023 The vLLM team.
-# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
-#
-# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,21 +28,13 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/apertus.py` modified +686/-687; `python/sglang/srt/models/solar.py` modified +28/-27; `python/sglang/srt/models/gpt_oss.py` modified +17/-24; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23; `python/sglang/srt/layers/communicator.py` modified +13/-19; `python/sglang/srt/models/qwen3_moe.py` modified +12/-18
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dual_chunk_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
