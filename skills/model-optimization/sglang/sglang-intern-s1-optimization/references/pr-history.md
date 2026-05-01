# sglang Intern-S1 PR Diff Audit Reference

- Rebuilt on: 2026-05-01
- Source baseline: `sgl-project/sglang` trace worktree commit `4197c55968`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 2
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx` | no direct PR-number commit |
| `python/sglang/srt/function_call/internlm_detector.py` | [#14866](https://github.com/sgl-project/sglang/pull/14866) |
| `python/sglang/srt/models/interns1.py` | [#8350](https://github.com/sgl-project/sglang/pull/8350), [#9299](https://github.com/sgl-project/sglang/pull/9299), [#12367](https://github.com/sgl-project/sglang/pull/12367) |
| `python/sglang/srt/models/interns1pro.py` | [#18145](https://github.com/sgl-project/sglang/pull/18145) |
| `python/sglang/srt/multimodal/processors/interns1pro.py` | [#18145](https://github.com/sgl-project/sglang/pull/18145) |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-07-26 | [#8350](https://github.com/sgl-project/sglang/pull/8350) | merged | model: support intern-s1 | `python/sglang/srt/models/interns1.py` |
| 2025-08-19 | [#9299](https://github.com/sgl-project/sglang/pull/9299) | merged | support for interns1-mini | `python/sglang/srt/models/interns1.py` |
| 2025-08-20 | [#9381](https://github.com/sgl-project/sglang/pull/9381) | merged | fix: InternS1 don't recognize image, updates image token for InternVL processor | `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/conversation.py` |
| 2025-11-03 | [#12367](https://github.com/sgl-project/sglang/pull/12367) | merged | [Bug] Fix Intern-S1 model accuracy and support /generate interface with input_ids | `python/sglang/srt/models/interns1.py` |
| 2025-12-16 | [#14866](https://github.com/sgl-project/sglang/pull/14866) | merged | Adding tool calling and reasoning parser support for Intern-S1 | `python/sglang/srt/function_call/internlm_detector.py` |
| 2026-01-26 | [#17040](https://github.com/sgl-project/sglang/pull/17040) | merged | fix(processor): support InternS1 text_config in InternVL processor | `python/sglang/srt/multimodal/processors/internvl.py` |
| 2026-02-04 | [#18145](https://github.com/sgl-project/sglang/pull/18145) | merged | support interns1-pro | `python/sglang/srt/models/interns1pro.py`, `python/sglang/srt/multimodal/processors/interns1pro.py` |

## Per-PR Diff Audit Cards

### PR #8350 - model: support intern-s1

- Link: https://github.com/sgl-project/sglang/pull/8350
- Status/date: merged / 2025-07-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/interns1.py`; associated commits `b7094a5ef197`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +616/-63, 986 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "model: support intern-s1"; model line: Intern-S1; category: performance/backend optimization; main diff: `python/sglang/srt/models/interns1.py`; technical summary: Covers "model: support intern-s1"; the main implementation surface is `python/sglang/srt/models/interns1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/interns1.py` added +328/-0 (328 lines); hunks: -0,0 +1,328; symbols: InternS1ForConditionalGeneration, __init__, _update_hf_config, pixel_shuffle, touching `InternS1ForConditionalGeneration, __init__, _update_hf_config`.
- Code diff details:
  - `python/sglang/srt/models/interns1.py` added +328/-0 (328 lines); hunks: -0,0 +1,328; symbols: InternS1ForConditionalGeneration, __init__, _update_hf_config, pixel_shuffle
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/interns1.py
@@ -0,0 +1,328 @@
+from typing import Iterable, List, Optional, Set, Tuple
+import torch
+from torch import nn
+from transformers import PretrainedConfig
+from sglang.srt.distributed import parallel_state
+from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/interns1.py` added +328/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/lang/chat_template.py`, `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/configs/model_config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9299 - support for interns1-mini

- Link: https://github.com/sgl-project/sglang/pull/9299
- Status/date: merged / 2025-08-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/interns1.py`; associated commits `a31ea4482436`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-2, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support for interns1-mini"; model line: Intern-S1; category: model support/runtime entry; main diff: `python/sglang/srt/models/interns1.py`; technical summary: Covers "support for interns1-mini"; the main implementation surface is `python/sglang/srt/models/interns1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/interns1.py` modified +5/-0 (5 lines); hunks: -21,6 +21,7; -70,6 +71,10 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/interns1.py` modified +5/-0 (5 lines); hunks: -21,6 +21,7; -70,6 +71,10 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/interns1.py
@@ -21,6 +21,7 @@
+from sglang.srt.models.qwen3 import Qwen3ForCausalLM
@@ -70,6 +71,10 @@ def __init__(
+        elif config.text_config.architectures[0] == "Qwen3ForCausalLM":
+            self.language_model = Qwen3ForCausalLM(
+                config=config.text_config, quant_config=quant_config
+            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/interns1.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/interns1.py`, `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9381 - fix: InternS1 don't recognize image, updates image token for InternVL processor

- Link: https://github.com/sgl-project/sglang/pull/9381
- Status/date: merged / 2025-08-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +9/-17, 60 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: InternS1 don't recognize image, updates image token for InternVL processor"; model line: Intern-S1; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/conversation.py`; technical summary: Covers "fix: InternS1 don't recognize image, updates image token for InternVL processor"; the main implementation surface is `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/conversation.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2 (9 lines); hunks: -44,7 +44,7 @@ def __init__(self, hf_config, server_args, _image_processor, *...; -218,13 +218,18 @@ def process_image_internvl(image, input_size=448, max_num=...; symbols: __init__, process_image_internvl, touching `__init__, process_image_internvl`; `python/sglang/srt/conversation.py` modified +2/-15 (17 lines); hunks: -625,7 +625,7 @@ def generate_chat_conv(; -817,20 +817,7 @@ def generate_chat_conv(; symbols: generate_chat_conv, touching `generate_chat_conv`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2 (9 lines); hunks: -44,7 +44,7 @@ def __init__(self, hf_config, server_args, _image_processor, *...; -218,13 +218,18 @@ def process_image_internvl(image, input_size=448, max_num=...; symbols: __init__, process_image_internvl
  - `python/sglang/srt/conversation.py` modified +2/-15 (17 lines); hunks: -625,7 +625,7 @@ def generate_chat_conv(; -817,20 +817,7 @@ def generate_chat_conv(; symbols: generate_chat_conv
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
diff -- python/sglang/srt/conversation.py
@@ -625,7 +625,7 @@ def generate_chat_conv(
-                        if conv.name in ["internvl-2-5", "interns1"]:
+                        if conv.name in ["internvl-2-5"]:
@@ -817,20 +817,7 @@ def generate_chat_conv(
-        image_token="<image>",
-    )
-)
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2; `python/sglang/srt/conversation.py` modified +2/-15
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/conversation.py`, `python/sglang/srt/multimodal/processors/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12367 - [Bug] Fix Intern-S1 model accuracy and support /generate interface with input_ids

- Link: https://github.com/sgl-project/sglang/pull/12367
- Status/date: merged / 2025-11-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/interns1.py`; associated commits `65f1d065c5cf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +8/-41, 110 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix Intern-S1 model accuracy and support /generate interface with input_ids"; model line: Intern-S1; category: bug fix; main diff: `python/sglang/srt/models/interns1.py`; technical summary: Covers "[Bug] Fix Intern-S1 model accuracy and support /generate interface with input_ids"; the main implementation surface is `python/sglang/srt/models/interns1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/interns1.py` modified +3/-21 (24 lines); hunks: -1,4 +1,4; -50,16 +50,13 @@ def __init__(; symbols: __init__, pixel_shuffle, extract_feature, load_weights, touching `__init__, pixel_shuffle, extract_feature`.
- Code diff details:
  - `python/sglang/srt/models/interns1.py` modified +3/-21 (24 lines); hunks: -1,4 +1,4; -50,16 +50,13 @@ def __init__(; symbols: __init__, pixel_shuffle, extract_feature, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/interns1.py
@@ -1,4 +1,4 @@
-from typing import Iterable, List, Optional, Set, Tuple
+from typing import Iterable, List, Optional, Tuple
@@ -50,16 +50,13 @@ def __init__(
-        self.ps_version = getattr(config, "ps_version", "v1")
-        # self.template = getattr(config, 'template', 'internvl2_5')
-        logger.info(f"ps_version: {self.ps_version}")
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/interns1.py` modified +3/-21
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/interns1.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14866 - Adding tool calling and reasoning parser support for Intern-S1

- Link: https://github.com/sgl-project/sglang/pull/14866
- Status/date: merged / 2025-12-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/internlm_detector.py`; associated commits `5e96beb3e559`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +290/-14, 361 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Adding tool calling and reasoning parser support for Intern-S1"; model line: Intern-S1; category: bug fix; main diff: `python/sglang/srt/function_call/internlm_detector.py`; technical summary: Covers "Adding tool calling and reasoning parser support for Intern-S1"; the main implementation surface is `python/sglang/srt/function_call/internlm_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/internlm_detector.py` added +248/-0 (248 lines); hunks: -0,0 +1,248; symbols: InternlmDetector, __init__, has_tool_call, get_arguments, touching `InternlmDetector, __init__, has_tool_call`.
- Code diff details:
  - `python/sglang/srt/function_call/internlm_detector.py` added +248/-0 (248 lines); hunks: -0,0 +1,248; symbols: InternlmDetector, __init__, has_tool_call, get_arguments
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/internlm_detector.py
@@ -0,0 +1,248 @@
+# modified from https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/tool_parser/internlm2_parser.py
+import json
+import logging
+import re
+from typing import List
+from sglang.srt.entrypoints.openai.protocol import Tool
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/internlm_detector.py` added +248/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/constrained/base_grammar_backend.py`, `python/sglang/srt/constrained/xgrammar_backend.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17040 - fix(processor): support InternS1 text_config in InternVL processor

- Link: https://github.com/sgl-project/sglang/pull/17040
- Status/date: merged / 2026-01-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-4, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(processor): support InternS1 text_config in InternVL processor"; model line: Intern-S1; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/internvl.py`; technical summary: Covers "fix(processor): support InternS1 text_config in InternVL processor"; the main implementation surface is `python/sglang/srt/multimodal/processors/internvl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #18145 - support interns1-pro

- Link: https://github.com/sgl-project/sglang/pull/18145
- Status/date: merged / 2026-02-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/interns1pro.py`, `python/sglang/srt/multimodal/processors/interns1pro.py`; associated commits `3e7ecb78a60f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +586/-2, 647 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support interns1-pro"; model line: Intern-S1; category: model support/runtime entry; main diff: `python/sglang/srt/models/interns1pro.py`, `python/sglang/srt/multimodal/processors/interns1pro.py`; technical summary: Covers "support interns1-pro"; the main implementation surface is `python/sglang/srt/models/interns1pro.py`, `python/sglang/srt/multimodal/processors/interns1pro.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/interns1pro.py` added +252/-0 (252 lines); hunks: -0,0 +1,252; symbols: InternS1ProTextAttention, __init__, forward_prepare_npu, InternS1ProTextDecoderLayer, touching `InternS1ProTextAttention, __init__, forward_prepare_npu`; `python/sglang/srt/multimodal/processors/interns1pro.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: InternS1_1ImageProcessor, get_mm_data, process_mm_data_async, touching `InternS1_1ImageProcessor, get_mm_data, process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/models/interns1pro.py` added +252/-0 (252 lines); hunks: -0,0 +1,252; symbols: InternS1ProTextAttention, __init__, forward_prepare_npu, InternS1ProTextDecoderLayer
  - `python/sglang/srt/multimodal/processors/interns1pro.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: InternS1_1ImageProcessor, get_mm_data, process_mm_data_async
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/interns1pro.py
@@ -0,0 +1,252 @@
+import functools
+import logging
+from typing import Any, Dict, Iterable, Optional, Tuple
+import torch
+from transformers import PretrainedConfig
+from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
diff -- python/sglang/srt/multimodal/processors/interns1pro.py
@@ -0,0 +1,118 @@
+import time
+from typing import List, Union
+from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
+from sglang.srt.models.interns1pro import InternS1ProForConditionalGeneration
+from sglang.srt.multimodal.processors.qwen_vl import (
+    QwenVLImageProcessor,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/interns1pro.py` added +252/-0; `python/sglang/srt/multimodal/processors/interns1pro.py` added +118/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/entrypoints/openai/protocol.py`, `python/sglang/srt/layers/rotary_embedding.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.
