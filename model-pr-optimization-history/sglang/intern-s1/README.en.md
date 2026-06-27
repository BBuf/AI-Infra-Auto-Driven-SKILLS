# sglang Intern-S1 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/function_call/internlm_detector.py` | [#14866](https://github.com/sgl-project/sglang/pull/14866) |
| `python/sglang/srt/models/interns1.py` | [#8350](https://github.com/sgl-project/sglang/pull/8350), [#9299](https://github.com/sgl-project/sglang/pull/9299), [#12367](https://github.com/sgl-project/sglang/pull/12367), [#28629](https://github.com/sgl-project/sglang/pull/28629) |
| `python/sglang/srt/models/interns1pro.py` | [#18145](https://github.com/sgl-project/sglang/pull/18145) |
| `python/sglang/srt/multimodal/processors/interns1pro.py` | [#18145](https://github.com/sgl-project/sglang/pull/18145) |

## PR Coverage Summary

- Git-traced PRs: 6
- Extra PRs preserved from existing docs: 7
- Total PRs in this document: 13
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

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
| 2026-04-03 | [#21738](https://github.com/sgl-project/sglang/pull/21738) | merged | refactor: replace mm_inputs dict with MultimodalProcessorOutput | `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py` |
| 2026-04-20 | [#23001](https://github.com/sgl-project/sglang/pull/23001) | merged | Add new Mintlify documentation site (docs_new/) | `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx` |
| 2026-05-22 | [#24751](https://github.com/sgl-project/sglang/pull/24751) | merged | fix(mm): make multimodal data loading non-blocking to prevent health check stalls | `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-18 | [#28629](https://github.com/sgl-project/sglang/pull/28629) | merged | [Bugfix] Fix Intern-S1 FP8 expert count lookup | `python/sglang/srt/models/interns1.py` |
| 2026-06-19 | [#28697](https://github.com/sgl-project/sglang/pull/28697) | merged | [docs] Add B300 cookbook deployment options | `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` |

## Per-PR Diff Audit Cards

### PR #8350 - model: support intern-s1

- Link: https://github.com/sgl-project/sglang/pull/8350
- Status/date: merged / 2025-07-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/interns1.py`; associated commits `b7094a5ef197`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +616/-63, 986 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "model: support intern-s1"; model line: Intern-S1; category: model support/runtime entry; main diff: `python/sglang/srt/models/interns1.py`; technical summary: Covers "model: support intern-s1"; the main implementation surface is `python/sglang/srt/models/interns1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Adding tool calling and reasoning parser support for Intern-S1"; model line: Intern-S1; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/internlm_detector.py`; technical summary: Covers "Adding tool calling and reasoning parser support for Intern-S1"; the main implementation surface is `python/sglang/srt/function_call/internlm_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #21738 - refactor: replace mm_inputs dict with MultimodalProcessorOutput

- Link: https://github.com/sgl-project/sglang/pull/21738
- Status/date: merged / 2026-04-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 40 files, +408/-314, 1321 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "refactor: replace mm_inputs dict with MultimodalProcessorOutput"; model line: Intern-S1; category: model implementation change; main diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`; technical summary: Covers "refactor: replace mm_inputs dict with MultimodalProcessorOutput"; the main implementation surface is `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Add new Mintlify documentation site (docs_new/)"; model line: Intern-S1; category: docs/tests/CI; main diff: `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`; technical summary: Covers "Add new Mintlify documentation site (docs_new/)"; the main implementation surface is `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #24751 - fix(mm): make multimodal data loading non-blocking to prevent health check stalls

- Link: https://github.com/sgl-project/sglang/pull/24751
- Status/date: merged / 2026-05-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 35 files, +45/-44, 401 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(mm): make multimodal data loading non-blocking to prevent health check stalls"; model line: Intern-S1; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`; technical summary: Covers "fix(mm): make multimodal data loading non-blocking to prevent health check stalls"; the main implementation surface is `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Add get_parallel(): a structured accessor for parallel-topology state"; model line: Intern-S1; category: model support/runtime entry; main diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Add get_parallel(): a structured accessor for parallel-topology state"; the main implementation surface is `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #28629 - [Bugfix] Fix Intern-S1 FP8 expert count lookup

- Link: https://github.com/sgl-project/sglang/pull/28629
- Status/date: merged / 2026-06-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/interns1.py`; associated commits `b7d7dfb4ed5a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Intern-S1 FP8 expert count lookup"; model line: Intern-S1; category: bug fix; main diff: `python/sglang/srt/models/interns1.py`; technical summary: Covers "[Bugfix] Fix Intern-S1 FP8 expert count lookup"; the main implementation surface is `python/sglang/srt/models/interns1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/interns1.py` modified +1/-1 (2 lines); hunks: -211,7 +211,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/interns1.py` modified +1/-1 (2 lines); hunks: -211,7 +211,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/interns1.py
@@ -211,7 +211,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-                num_experts=self.config.num_experts,
+                num_experts=self.config.text_config.num_experts,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/interns1.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/interns1.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28697 - [docs] Add B300 cookbook deployment options

- Link: https://github.com/sgl-project/sglang/pull/28697
- Status/date: merged / 2026-06-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +503/-69, 1291 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] Add B300 cookbook deployment options"; model line: Intern-S1; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`; technical summary: Covers "[docs] Add B300 cookbook deployment options"; the main implementation surface is `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167
  - `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {
  - `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15 (38 lines); hunks: -8,19 +8,19 @@ export const Qwen35Deployment = () => {; -149,7 +149,7 @@ export const Qwen35Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx
@@ -0,0 +1,167 @@
+export const InternS1Deployment = () => {
+  const options = {
+    hardware: {
+      name: 'hardware',
+      title: 'Hardware Platform',
+      items: [
diff -- docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx
@@ -9,6 +9,11 @@ const lookupData = {
+      {
+        "id": "b300",
+        "label": "B300",
+        "default": false
+      },
@@ -182,6 +187,66 @@ const lookupData = {
diff -- docs_new/src/snippets/autoregressive/glm-5-deployment.jsx
@@ -4,6 +4,7 @@ export const GLM5Deployment = () => {
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10; `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15; `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx` modified +16/-13
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx`, `docs_new/src/snippets/autoregressive/deepseek-math-v2-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
