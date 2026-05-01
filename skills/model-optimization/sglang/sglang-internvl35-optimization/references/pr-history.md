# sglang InternVL 3.5 PR Diff Audit Reference

- Rebuilt on: 2026-05-01
- Source baseline: `sgl-project/sglang` trace worktree commit `4197c55968`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 3
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/InternVL/InternVL3.5.mdx` | no direct PR-number commit |
| `python/sglang/srt/configs/internvl.py` | [#5350](https://github.com/sgl-project/sglang/pull/5350), [#8067](https://github.com/sgl-project/sglang/pull/8067), [#9705](https://github.com/sgl-project/sglang/pull/9705) |
| `python/sglang/srt/models/internvl.py` | [#5350](https://github.com/sgl-project/sglang/pull/5350), [#6870](https://github.com/sgl-project/sglang/pull/6870), [#9705](https://github.com/sgl-project/sglang/pull/9705), [#13640](https://github.com/sgl-project/sglang/pull/13640), [#13925](https://github.com/sgl-project/sglang/pull/13925), [#15942](https://github.com/sgl-project/sglang/pull/15942), [#16732](https://github.com/sgl-project/sglang/pull/16732), [#19127](https://github.com/sgl-project/sglang/pull/19127) |
| `python/sglang/srt/multimodal/internvl_utils.py` | no direct PR-number commit |
| `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py` | [#16732](https://github.com/sgl-project/sglang/pull/16732) |
| `python/sglang/srt/multimodal/processors/internvl.py` | [#9381](https://github.com/sgl-project/sglang/pull/9381), [#9795](https://github.com/sgl-project/sglang/pull/9795), [#10375](https://github.com/sgl-project/sglang/pull/10375), [#15942](https://github.com/sgl-project/sglang/pull/15942), [#17040](https://github.com/sgl-project/sglang/pull/17040), [#19127](https://github.com/sgl-project/sglang/pull/19127), [#19997](https://github.com/sgl-project/sglang/pull/19997) |

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

## Per-PR Diff Audit Cards

### PR #3351 - model: Intern vl 2.5

- Link: https://github.com/sgl-project/sglang/pull/3351
- Status/date: closed / 2025-03-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 24 files, +4538/-163, 5186 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "model: Intern vl 2.5"; model line: InternVL 3.5; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_janus_pro.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/tokenizers/lmtokenizer.py`; technical summary: Covers "model: Intern vl 2.5"; the main implementation surface is `python/sglang/srt/models/deepseek_janus_pro.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/tokenizers/lmtokenizer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Support InternVL2.5"; model line: InternVL 3.5; category: bug fix; main diff: `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/configs/model_config.py`; technical summary: Covers "Support InternVL2.5"; the main implementation surface is `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/configs/model_config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Support the internvl3.5 family models in sglang"; model line: InternVL 3.5; category: bug fix; main diff: `python/sglang/srt/models/internvl.py`, `python/sglang/srt/configs/internvl.py`; technical summary: Covers "Support the internvl3.5 family models in sglang"; the main implementation surface is `python/sglang/srt/models/internvl.py`, `python/sglang/srt/configs/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "refactor(InternVL): Use gpu to preprocess the input image"; model line: InternVL 3.5; category: performance/backend optimization; main diff: `python/sglang/srt/multimodal/processors/internvl.py`; technical summary: Covers "refactor(InternVL): Use gpu to preprocess the input image"; the main implementation surface is `python/sglang/srt/multimodal/processors/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[vlm][internVL] Support processor and embedding inputs for InternVL"; model line: InternVL 3.5; category: docs/tests/CI; main diff: `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`; technical summary: Covers "[vlm][internVL] Support processor and embedding inputs for InternVL"; the main implementation surface is `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
