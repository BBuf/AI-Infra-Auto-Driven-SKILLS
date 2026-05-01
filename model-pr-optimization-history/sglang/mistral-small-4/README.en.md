# sglang Mistral Small 4 Model PR Optimization History

## Scope

- Rebuilt on: 2026-05-01
- Source baseline: `sgl-project/sglang` trace worktree commit `4197c55968`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/Mistral/Devstral-2.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/Mistral/Ministral-3.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/Mistral/Mistral-Medium-3.5.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/Mistral/Mistral-Small-4.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/ministral-3-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/mistral-medium-3-5-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/mistral-small-4-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/function_call/mistral_detector.py` | [#6597](https://github.com/sgl-project/sglang/pull/6597), [#14921](https://github.com/sgl-project/sglang/pull/14921), [#20708](https://github.com/sgl-project/sglang/pull/20708) |
| `python/sglang/srt/models/ministral3.py` | [#14251](https://github.com/sgl-project/sglang/pull/14251) |
| `python/sglang/srt/models/mistral.py` | [#108](https://github.com/sgl-project/sglang/pull/108), [#5099](https://github.com/sgl-project/sglang/pull/5099) |
| `python/sglang/srt/models/mistral_eagle.py` | no direct PR-number commit |
| `python/sglang/srt/models/mistral_large_3.py` | [#14213](https://github.com/sgl-project/sglang/pull/14213), [#14466](https://github.com/sgl-project/sglang/pull/14466), [#14485](https://github.com/sgl-project/sglang/pull/14485) |
| `python/sglang/srt/models/mistral_large_3_eagle.py` | [#14466](https://github.com/sgl-project/sglang/pull/14466), [#14485](https://github.com/sgl-project/sglang/pull/14485), [#20708](https://github.com/sgl-project/sglang/pull/20708) |
| `python/sglang/srt/utils/hf_transformers/mistral_utils.py` | no direct PR-number commit |
| `test/manual/models/test_mistral_large3_basic.py` | no direct PR-number commit |
| `test/registered/8-gpu-models/test_mistral_large3.py` | [#15422](https://github.com/sgl-project/sglang/pull/15422), [#18065](https://github.com/sgl-project/sglang/pull/18065), [#19402](https://github.com/sgl-project/sglang/pull/19402) |
| `test/registered/ascend/llm_models/test_npu_mistral_7b.py` | no direct PR-number commit |
| `test/registered/ascend/vlm_models/test_npu_mistral_small_3_1_24b_instruct_2503.py` | no direct PR-number commit |
| `test/registered/models/test_ministral3_models.py` | no direct PR-number commit |
| `test/registered/models/test_ministral4_models.py` | [#21620](https://github.com/sgl-project/sglang/pull/21620) |
| `test/registered/unit/function_call/test_mistral_detector.py` | [#21399](https://github.com/sgl-project/sglang/pull/21399) |

## PR Coverage Summary

- Git-traced PRs: 14
- Extra PRs preserved from existing docs: 1
- Total PRs in this document: 15
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2024-01-26 | [#108](https://github.com/sgl-project/sglang/pull/108) | merged | Fix Mistral model loading | `python/sglang/srt/models/mistral.py` |
| 2025-05-17 | [#5099](https://github.com/sgl-project/sglang/pull/5099) | merged | model(vlm): mistral 3.1 | `python/sglang/srt/models/mistral.py` |
| 2025-05-26 | [#6597](https://github.com/sgl-project/sglang/pull/6597) | merged | feat: Improve Mistral and Qwen25 function call parsing | `python/sglang/srt/function_call/mistral_detector.py` |
| 2025-12-04 | [#14213](https://github.com/sgl-project/sglang/pull/14213) | merged | Add Mistral Large 3 support. | `python/sglang/srt/models/mistral_large_3.py` |
| 2025-12-04 | [#14251](https://github.com/sgl-project/sglang/pull/14251) | merged | ministral3 | `python/sglang/srt/models/ministral3.py` |
| 2025-12-05 | [#14466](https://github.com/sgl-project/sglang/pull/14466) | merged | Add Mistral Large 3 Eagle Support | `python/sglang/srt/models/mistral_large_3_eagle.py`, `python/sglang/srt/models/mistral_large_3.py` |
| 2025-12-12 | [#14921](https://github.com/sgl-project/sglang/pull/14921) | merged | update mistral detector | `python/sglang/srt/function_call/mistral_detector.py` |
| 2025-12-13 | [#14485](https://github.com/sgl-project/sglang/pull/14485) | merged | Mistral Large 3 NVFP4 support | `python/sglang/srt/models/mistral_large_3.py`, `python/sglang/srt/models/mistral_large_3_eagle.py` |
| 2025-12-18 | [#15049](https://github.com/sgl-project/sglang/pull/15049) | merged | Mistral Large 3 NVFP4 TRTLLM MoE support | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py` |
| 2026-02-03 | [#18065](https://github.com/sgl-project/sglang/pull/18065) | merged | [Bugfix] Fix Mistral Large 3 NVFP4 TRTLLM MoE | `test/registered/8-gpu-models/test_mistral_large3.py` |
| 2026-02-25 | [#15422](https://github.com/sgl-project/sglang/pull/15422) | merged | Flashinfer MOE FP8 support for Mistral Large 3. | `test/registered/8-gpu-models/test_mistral_large3.py` |
| 2026-02-26 | [#19402](https://github.com/sgl-project/sglang/pull/19402) | merged | Fix nightly Mistral-Large-3 NVFP4 accuracy threshold | `test/registered/8-gpu-models/test_mistral_large3.py` |
| 2026-03-18 | [#20708](https://github.com/sgl-project/sglang/pull/20708) | merged | Add Mistral Small 4 (Pixtral) support | `python/sglang/srt/function_call/mistral_detector.py`, `python/sglang/srt/models/mistral_large_3_eagle.py` |
| 2026-03-30 | [#21620](https://github.com/sgl-project/sglang/pull/21620) | merged | fix: Mistral Small 4 fails to start due to config/weight format mismatch | `test/registered/models/test_ministral4_models.py` |
| 2026-04-06 | [#21399](https://github.com/sgl-project/sglang/pull/21399) | merged | [CI] Add unit tests for function_call detectors (hermes, llama32, mistral) | `test/registered/unit/function_call/test_mistral_detector.py` |

## Per-PR Diff Audit Cards

### PR #108 - Fix Mistral model loading

- Link: https://github.com/sgl-project/sglang/pull/108
- Status/date: merged / 2024-01-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mistral.py`; associated commits `cd6872334e9e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-0, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Mistral model loading"; model line: Mistral Small 4; category: bug fix; main diff: `python/sglang/srt/models/mistral.py`; technical summary: Covers "Fix Mistral model loading"; the main implementation surface is `python/sglang/srt/models/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/mistral.py` added +10/-0 (10 lines); hunks: -0,0 +1,10; symbols: MistralForCausalLM, __init__, touching `MistralForCausalLM, __init__`.
- Code diff details:
  - `python/sglang/srt/models/mistral.py` added +10/-0 (10 lines); hunks: -0,0 +1,10; symbols: MistralForCausalLM, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mistral.py
@@ -0,0 +1,10 @@
+"""Inference-only Mistral model."""
+from sglang.srt.models.llama2 import LlamaForCausalLM
+class MistralForCausalLM(LlamaForCausalLM):
+    def __init__(self, *args, **kwargs):
+        super().__init__(*args, **kwargs)
+EntryClass = MistralForCausalLM
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mistral.py` added +10/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5099 - model(vlm): mistral 3.1

- Link: https://github.com/sgl-project/sglang/pull/5099
- Status/date: merged / 2025-05-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mistral.py`; associated commits `64825b839521`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +152/-21, 272 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "model(vlm): mistral 3.1"; model line: Mistral Small 4; category: model support/runtime entry; main diff: `python/sglang/srt/models/mistral.py`; technical summary: Covers "model(vlm): mistral 3.1"; the main implementation surface is `python/sglang/srt/models/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/mistral.py` modified +71/-1 (72 lines); hunks: -13,11 +13,81; symbols: MistralForCausalLM, Mistral3ForConditionalGeneration, __init__, get_image_feature, touching `MistralForCausalLM, Mistral3ForConditionalGeneration, __init__`.
- Code diff details:
  - `python/sglang/srt/models/mistral.py` modified +71/-1 (72 lines); hunks: -13,11 +13,81; symbols: MistralForCausalLM, Mistral3ForConditionalGeneration, __init__, get_image_feature
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mistral.py
@@ -13,11 +13,81 @@
+from typing import List, Union
+import torch
+from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector
+from sglang.srt.managers.schedule_batch import MultimodalDataItem
-EntryClass = MistralForCausalLM
+class Mistral3ForConditionalGeneration:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mistral.py` modified +71/-1
- Risk and verification: The diff ships test coverage in `test/srt/test_vision_openai_server.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6597 - feat: Improve Mistral and Qwen25 function call parsing

- Link: https://github.com/sgl-project/sglang/pull/6597
- Status/date: merged / 2025-05-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/mistral_detector.py`; associated commits `16f69b1f65c6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +318/-61, 529 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: Improve Mistral and Qwen25 function call parsing"; model line: Mistral Small 4; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/mistral_detector.py`; technical summary: Covers "feat: Improve Mistral and Qwen25 function call parsing"; the main implementation surface is `python/sglang/srt/function_call/mistral_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/mistral_detector.py` modified +72/-26 (98 lines); hunks: -1,4 +1,5; -11,12 +12,14; symbols: MistralDetector, __init__, has_tool_call, _clean_text, touching `MistralDetector, __init__, has_tool_call`.
- Code diff details:
  - `python/sglang/srt/function_call/mistral_detector.py` modified +72/-26 (98 lines); hunks: -1,4 +1,5; -11,12 +12,14; symbols: MistralDetector, __init__, has_tool_call, _clean_text
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/mistral_detector.py
@@ -1,4 +1,5 @@
+import logging
@@ -11,12 +12,14 @@
+logger = logging.getLogger(__name__)
-      [TOOL_CALLS] [{"name":"xxx", "arguments":{...}}]
+      [TOOL_CALLS] [{"name":"func1", "arguments":{...}}, {"name":"func2", "arguments":{...}}]
@@ -32,21 +35,6 @@ def has_tool_call(self, text: str) -> bool:
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/mistral_detector.py` modified +72/-26
- Risk and verification: The diff ships test coverage in `test/srt/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14213 - Add Mistral Large 3 support.

- Link: https://github.com/sgl-project/sglang/pull/14213
- Status/date: merged / 2025-12-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mistral_large_3.py`; associated commits `842807843671`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +1400/-120, 2012 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Mistral Large 3 support."; model line: Mistral Small 4; category: model support/runtime entry; main diff: `python/sglang/srt/models/mistral_large_3.py`; technical summary: Covers "Add Mistral Large 3 support."; the main implementation surface is `python/sglang/srt/models/mistral_large_3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/mistral_large_3.py` added +81/-0 (81 lines); hunks: -0,0 +1,81; symbols: MistralLarge3ForCausalLM, load_weights, _iterable_remap_mistral_to_ds, touching `MistralLarge3ForCausalLM, load_weights, _iterable_remap_mistral_to_ds`.
- Code diff details:
  - `python/sglang/srt/models/mistral_large_3.py` added +81/-0 (81 lines); hunks: -0,0 +1,81; symbols: MistralLarge3ForCausalLM, load_weights, _iterable_remap_mistral_to_ds
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mistral_large_3.py
@@ -0,0 +1,81 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable
+import regex as re
+import torch
+from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mistral_large_3.py` added +81/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14251 - ministral3

- Link: https://github.com/sgl-project/sglang/pull/14251
- Status/date: merged / 2025-12-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/ministral3.py`; associated commits `6d37e7088337`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +245/-26, 405 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "ministral3"; model line: Mistral Small 4; category: docs/tests/CI; main diff: `python/sglang/srt/models/ministral3.py`; technical summary: Covers "ministral3"; the main implementation surface is `python/sglang/srt/models/ministral3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/ministral3.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: _get_llama_4_attn_scale, Ministral3Attention, __init__, forward, touching `_get_llama_4_attn_scale, Ministral3Attention, __init__`.
- Code diff details:
  - `python/sglang/srt/models/ministral3.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: _get_llama_4_attn_scale, Ministral3Attention, __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/ministral3.py
@@ -0,0 +1,157 @@
+from typing import Any, Dict, Optional
+import torch
+from transformers import PretrainedConfig
+from sglang.srt.layers.quantization.base_config import QuantizationConfig
+from sglang.srt.model_executor.forward_batch_info import ForwardBatch
+from sglang.srt.models.llama import (
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/ministral3.py` added +157/-0
- Risk and verification: The diff ships test coverage in `test/srt/models/test_ministral3_models.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14466 - Add Mistral Large 3 Eagle Support

- Link: https://github.com/sgl-project/sglang/pull/14466
- Status/date: merged / 2025-12-05
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mistral_large_3.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`; associated commits `205f041e9619`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +313/-62, 550 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Mistral Large 3 Eagle Support"; model line: Mistral Small 4; category: performance/backend optimization; main diff: `python/sglang/srt/models/mistral_large_3_eagle.py`, `python/sglang/srt/models/mistral_large_3.py`; technical summary: Covers "Add Mistral Large 3 Eagle Support"; the main implementation surface is `python/sglang/srt/models/mistral_large_3_eagle.py`, `python/sglang/srt/models/mistral_large_3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/mistral_large_3_eagle.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: MistralLarge3Model, __init__, forward, MistralLarge3ForCausalLMEagle, touching `MistralLarge3Model, __init__, forward`; `python/sglang/srt/models/mistral_large_3.py` modified +0/-3 (3 lines); hunks: -72,9 +72,6 @@ def _iterable_remap_mistral_to_ds(; symbols: _iterable_remap_mistral_to_ds, touching `_iterable_remap_mistral_to_ds`.
- Code diff details:
  - `python/sglang/srt/models/mistral_large_3_eagle.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: MistralLarge3Model, __init__, forward, MistralLarge3ForCausalLMEagle
  - `python/sglang/srt/models/mistral_large_3.py` modified +0/-3 (3 lines); hunks: -72,9 +72,6 @@ def _iterable_remap_mistral_to_ds(; symbols: _iterable_remap_mistral_to_ds
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mistral_large_3_eagle.py
@@ -0,0 +1,105 @@
+from typing import Optional
+import torch
+from torch import nn
+from transformers import PretrainedConfig
+from python.sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
+from sglang.srt.distributed import get_pp_group
diff -- python/sglang/srt/models/mistral_large_3.py
@@ -72,9 +72,6 @@ def _iterable_remap_mistral_to_ds(
-            if name.endswith(".weight_scale") and ".experts." not in name:
-                name = re.sub(r"\.weight_scale$", ".weight_scale_inv", name)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mistral_large_3_eagle.py` added +105/-0; `python/sglang/srt/models/mistral_large_3.py` modified +0/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14921 - update mistral detector

- Link: https://github.com/sgl-project/sglang/pull/14921
- Status/date: merged / 2025-12-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/mistral_detector.py`; associated commits `fd1ebbb0d614`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +274/-34, 361 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "update mistral detector"; model line: Mistral Small 4; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/mistral_detector.py`; technical summary: Covers "update mistral detector"; the main implementation surface is `python/sglang/srt/function_call/mistral_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/mistral_detector.py` modified +240/-34 (274 lines); hunks: -1,47 +1,49; -51,31 +53,235 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: MistralDetector, __init__, has_tool_call, detect_and_parse, touching `MistralDetector, __init__, has_tool_call`.
- Code diff details:
  - `python/sglang/srt/function_call/mistral_detector.py` modified +240/-34 (274 lines); hunks: -1,47 +1,49; -51,31 +53,235 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: MistralDetector, __init__, has_tool_call, detect_and_parse
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/mistral_detector.py
@@ -1,47 +1,49 @@
-import re
-from typing import List
+from typing import Any, List, Optional, Tuple
+    ToolCallItem,
+from sglang.srt.function_call.utils import _is_complete_json
-    Detector for Mistral model function call format.
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/mistral_detector.py` modified +240/-34
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14485 - Mistral Large 3 NVFP4 support

- Link: https://github.com/sgl-project/sglang/pull/14485
- Status/date: merged / 2025-12-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mistral_large_3.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`; associated commits `f6031adf0875`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +502/-36, 707 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Mistral Large 3 NVFP4 support"; model line: Mistral Small 4; category: performance/backend optimization; main diff: `python/sglang/srt/models/mistral_large_3.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`; technical summary: Covers "Mistral Large 3 NVFP4 support"; the main implementation surface is `python/sglang/srt/models/mistral_large_3.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/mistral_large_3.py` modified +1/-1 (2 lines); hunks: -1,5 +1,5; `python/sglang/srt/models/mistral_large_3_eagle.py` modified +2/-0 (2 lines); hunks: -1,3 +1,5.
- Code diff details:
  - `python/sglang/srt/models/mistral_large_3.py` modified +1/-1 (2 lines); hunks: -1,5 +1,5
  - `python/sglang/srt/models/mistral_large_3_eagle.py` modified +2/-0 (2 lines); hunks: -1,3 +1,5
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mistral_large_3.py
@@ -1,5 +1,5 @@
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mistral_large_3.py
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
diff -- python/sglang/srt/models/mistral_large_3_eagle.py
@@ -1,3 +1,5 @@
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mistral_large_3_eagle.py
+# SPDX-License-Identifier: Apache-2.0
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mistral_large_3.py` modified +1/-1; `python/sglang/srt/models/mistral_large_3_eagle.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15049 - Mistral Large 3 NVFP4 TRTLLM MoE support

- Link: https://github.com/sgl-project/sglang/pull/15049
- Status/date: merged / 2025-12-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +340/-151, 624 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Mistral Large 3 NVFP4 TRTLLM MoE support"; model line: Mistral Small 4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`; technical summary: Covers "Mistral Large 3 NVFP4 TRTLLM MoE support"; the main implementation surface is `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +193/-21 (214 lines); hunks: -11,10 +11,15; -29,10 +34,18; symbols: __init__, create_weights, process_weights_after_loading, touching `__init__, create_weights, process_weights_after_loading`; `python/sglang/srt/layers/quantization/utils.py` modified +140/-0 (140 lines); hunks: -592,3 +592,143 @@ def swizzle_blockscale(scale: torch.Tensor):; symbols: swizzle_blockscale, reorder_w1w3_to_w3w1, prepare_static_weights_for_trtllm_fp4_moe, touching `swizzle_blockscale, reorder_w1w3_to_w3w1, prepare_static_weights_for_trtllm_fp4_moe`; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +2/-125 (127 lines); hunks: -42,6 +42,7; -1398,130 +1399,6 @@ def create_weights(; symbols: create_weights, prepare_static_weights_for_kernel, process_weights_after_loading, _slice_scale, touching `create_weights, prepare_static_weights_for_kernel, process_weights_after_loading`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-1 (3 lines); hunks: -548,8 +548,9 @@ def get_moe_impl_class(quant_config: Optional[QuantizationCo...; symbols: get_moe_impl_class, touching `get_moe_impl_class`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +193/-21 (214 lines); hunks: -11,10 +11,15; -29,10 +34,18; symbols: __init__, create_weights, process_weights_after_loading
  - `python/sglang/srt/layers/quantization/utils.py` modified +140/-0 (140 lines); hunks: -592,3 +592,143 @@ def swizzle_blockscale(scale: torch.Tensor):; symbols: swizzle_blockscale, reorder_w1w3_to_w3w1, prepare_static_weights_for_trtllm_fp4_moe
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +2/-125 (127 lines); hunks: -42,6 +42,7; -1398,130 +1399,6 @@ def create_weights(; symbols: create_weights, prepare_static_weights_for_kernel, process_weights_after_loading, _slice_scale
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-1 (3 lines); hunks: -548,8 +548,9 @@ def get_moe_impl_class(quant_config: Optional[QuantizationCo...; symbols: get_moe_impl_class
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-1 (1 lines); hunks: -1093,7 +1093,6 @@ def forward(self, hidden_states: torch.Tensor, topk_output...; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -11,10 +11,15 @@
-from sglang.srt.distributed import get_tensor_model_parallel_world_size
+from sglang.srt.distributed import get_tensor_model_parallel_world_size, get_tp_group
+from sglang.srt.distributed.device_communicators.pynccl_allocator import (
+    use_symmetric_memory,
+)
+from sglang.srt.layers.dp_attention import is_allocation_symmetric
diff -- python/sglang/srt/layers/quantization/utils.py
@@ -592,3 +592,143 @@ def swizzle_blockscale(scale: torch.Tensor):
+def reorder_w1w3_to_w3w1(
+    weight: torch.Tensor, scale: torch.Tensor, dim: int = -2
+) -> tuple[torch.Tensor, torch.Tensor]:
+    """Re-order the concatenated `[w1, w3]` tensors to `[w3, w1]`"""
+    size = weight.size(dim)
+    assert size % 2 == 0, f"Expected even size in dim {dim}, got {size}"
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -42,6 +42,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +193/-21; `python/sglang/srt/layers/quantization/utils.py` modified +140/-0; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +2/-125; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-1; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-1; `python/sglang/srt/server_args.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18065 - [Bugfix] Fix Mistral Large 3 NVFP4 TRTLLM MoE

- Link: https://github.com/sgl-project/sglang/pull/18065
- Status/date: merged / 2026-02-03
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_mistral_large3.py`; associated commits `99fab2ce673e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +115/-111, 282 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Mistral Large 3 NVFP4 TRTLLM MoE"; model line: Mistral Small 4; category: bug fix; main diff: `test/registered/8-gpu-models/test_mistral_large3.py`; technical summary: Covers "[Bugfix] Fix Mistral Large 3 NVFP4 TRTLLM MoE"; the main implementation surface is `test/registered/8-gpu-models/test_mistral_large3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_mistral_large3.py` modified +21/-8 (29 lines); hunks: -9,19 +9,21; -56,22 +58,33 @@ def test_mistral_large3_all_variants(self):; symbols: TestMistralLarge3, for, test_mistral_large3_all_variants, touching `TestMistralLarge3, for, test_mistral_large3_all_variants`.
- Code diff details:
  - `test/registered/8-gpu-models/test_mistral_large3.py` modified +21/-8 (29 lines); hunks: -9,19 +9,21; -56,22 +58,33 @@ def test_mistral_large3_all_variants(self):; symbols: TestMistralLarge3, for, test_mistral_large3_all_variants
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_mistral_large3.py
@@ -9,19 +9,21 @@
-register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)
+register_cuda_ci(est_time=3000, suite="nightly-8-gpu-common", nightly=True)
-MISTRAL_LARGE3_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512"
+MISTRAL_LARGE3_FP8_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512"
+MISTRAL_LARGE3_NVFP4_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4"
-    Two variants:
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_mistral_large3.py` modified +21/-8
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_mistral_large3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15422 - Flashinfer MOE FP8 support for Mistral Large 3.

- Link: https://github.com/sgl-project/sglang/pull/15422
- Status/date: merged / 2026-02-25
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_mistral_large3.py`; associated commits `350190487be4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +60/-17, 143 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Flashinfer MOE FP8 support for Mistral Large 3."; model line: Mistral Small 4; category: performance/backend optimization; main diff: `test/registered/8-gpu-models/test_mistral_large3.py`; technical summary: Covers "Flashinfer MOE FP8 support for Mistral Large 3."; the main implementation surface is `test/registered/8-gpu-models/test_mistral_large3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_mistral_large3.py` modified +2/-5 (7 lines); hunks: -46,6 +46,7 @@ def test_mistral_large3_all_variants(self):; -58,10 +59,6 @@ def test_mistral_large3_all_variants(self):; symbols: test_mistral_large3_all_variants, touching `test_mistral_large3_all_variants`.
- Code diff details:
  - `test/registered/8-gpu-models/test_mistral_large3.py` modified +2/-5 (7 lines); hunks: -46,6 +46,7 @@ def test_mistral_large3_all_variants(self):; -58,10 +59,6 @@ def test_mistral_large3_all_variants(self):; symbols: test_mistral_large3_all_variants
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_mistral_large3.py
@@ -46,6 +46,7 @@ def test_mistral_large3_all_variants(self):
+            "--moe-runner-backend=flashinfer_trtllm",
@@ -58,10 +59,6 @@ def test_mistral_large3_all_variants(self):
-        # TODO: add this to base args when FP8 TRTLLM moe is supported
-        nvfp4_args = [
-            "--moe-runner-backend=flashinfer_trtllm",
-        ]
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_mistral_large3.py` modified +2/-5
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_mistral_large3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19402 - Fix nightly Mistral-Large-3 NVFP4 accuracy threshold

- Link: https://github.com/sgl-project/sglang/pull/19402
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_mistral_large3.py`; associated commits `e14fd4accb43`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix nightly Mistral-Large-3 NVFP4 accuracy threshold"; model line: Mistral Small 4; category: bug fix; main diff: `test/registered/8-gpu-models/test_mistral_large3.py`; technical summary: Covers "Fix nightly Mistral-Large-3 NVFP4 accuracy threshold"; the main implementation surface is `test/registered/8-gpu-models/test_mistral_large3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_mistral_large3.py` modified +1/-1 (2 lines); hunks: -88,7 +88,7 @@ def test_mistral_large3_all_variants(self):; symbols: test_mistral_large3_all_variants, touching `test_mistral_large3_all_variants`.
- Code diff details:
  - `test/registered/8-gpu-models/test_mistral_large3.py` modified +1/-1 (2 lines); hunks: -88,7 +88,7 @@ def test_mistral_large3_all_variants(self):; symbols: test_mistral_large3_all_variants
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_mistral_large3.py
@@ -88,7 +88,7 @@ def test_mistral_large3_all_variants(self):
-            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.90),
+            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.85),
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_mistral_large3.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_mistral_large3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20708 - Add Mistral Small 4 (Pixtral) support

- Link: https://github.com/sgl-project/sglang/pull/20708
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/mistral_detector.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`; associated commits `6b8a6545b231`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +360/-124, 868 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Mistral Small 4 (Pixtral) support"; model line: Mistral Small 4; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/mistral_detector.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`; technical summary: Covers "Add Mistral Small 4 (Pixtral) support"; the main implementation surface is `python/sglang/srt/function_call/mistral_detector.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/mistral_detector.py` modified +17/-9 (26 lines); hunks: -90,19 +90,27 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: detect_and_parse, parse_streaming_increment, touching `detect_and_parse, parse_streaming_increment`; `python/sglang/srt/models/mistral_large_3_eagle.py` modified +11/-3 (14 lines); hunks: -18,7 +18,10; -99,9 +102,14 @@ def __init__(; symbols: MistralLarge3Model, MistralLarge3EagleModel, __init__, touching `MistralLarge3Model, MistralLarge3EagleModel, __init__`.
- Code diff details:
  - `python/sglang/srt/function_call/mistral_detector.py` modified +17/-9 (26 lines); hunks: -90,19 +90,27 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: detect_and_parse, parse_streaming_increment
  - `python/sglang/srt/models/mistral_large_3_eagle.py` modified +11/-3 (14 lines); hunks: -18,7 +18,10; -99,9 +102,14 @@ def __init__(; symbols: MistralLarge3Model, MistralLarge3EagleModel, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/mistral_detector.py
@@ -90,19 +90,27 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult
-        parsed = self._try_parse_compact_args_format(tool_part)
-        if not parsed:
+        # Loop to extract all consecutive compact tool calls.
+        all_calls: list = []
+        remaining = tool_part
+        while remaining:
diff -- python/sglang/srt/models/mistral_large_3_eagle.py
@@ -18,7 +18,10 @@
-class MistralLarge3Model(DeepseekV2Model):
+class MistralLarge3EagleModel(DeepseekV2Model):
+    """EAGLE draft model with an fc layer that fuses token embeddings and
+    target-model hidden states before passing through transformer layers."""
@@ -99,9 +102,14 @@ def __init__(
-        config.quant_config = quant_config
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/mistral_detector.py` modified +17/-9; `python/sglang/srt/models/mistral_large_3_eagle.py` modified +11/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/configs/janus_pro.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21620 - fix: Mistral Small 4 fails to start due to config/weight format mismatch

- Link: https://github.com/sgl-project/sglang/pull/21620
- Status/date: merged / 2026-03-30
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/models/test_ministral4_models.py`; associated commits `1d6424d5ad2d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +59/-7, 83 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: Mistral Small 4 fails to start due to config/weight format mismatch"; model line: Mistral Small 4; category: bug fix; main diff: `test/registered/models/test_ministral4_models.py`; technical summary: Covers "fix: Mistral Small 4 fails to start due to config/weight format mismatch"; the main implementation surface is `test/registered/models/test_ministral4_models.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models/test_ministral4_models.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: TestMistralSmall4TextOnly, TestMistralSmall4MMMU, touching `TestMistralSmall4TextOnly, TestMistralSmall4MMMU`.
- Code diff details:
  - `test/registered/models/test_ministral4_models.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: TestMistralSmall4TextOnly, TestMistralSmall4MMMU
- Key code excerpts:

```diff
diff -- test/registered/models/test_ministral4_models.py
@@ -0,0 +1,32 @@
+import unittest
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
+from sglang.test.kits.mmmu_vlm_kit import MMMUMixin
+from sglang.test.server_fixtures.default_fixture import DefaultServerBase
+from sglang.test.server_fixtures.mmmu_fixture import MMMUServerBase
```

- Reviewed files:
  - tests: `test/registered/models/test_ministral4_models.py` added +32/-0
- Risk and verification: The diff ships test coverage in `test/registered/models/test_ministral4_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21399 - [CI] Add unit tests for function_call detectors (hermes, llama32, mistral)

- Link: https://github.com/sgl-project/sglang/pull/21399
- Status/date: merged / 2026-04-06
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/unit/function_call/test_mistral_detector.py`; associated commits `30f5b8760851`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +595/-0, 598 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Add unit tests for function_call detectors (hermes, llama32, mistral)"; model line: Mistral Small 4; category: docs/tests/CI; main diff: `test/registered/unit/function_call/test_mistral_detector.py`; technical summary: Covers "[CI] Add unit tests for function_call detectors (hermes, llama32, mistral)"; the main implementation surface is `test/registered/unit/function_call/test_mistral_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/function_call/test_mistral_detector.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: TestMistralDetector, setUp, test_has_tool_call_json_array_format, test_has_tool_call_compact_format, touching `TestMistralDetector, setUp, test_has_tool_call_json_array_format`.
- Code diff details:
  - `test/registered/unit/function_call/test_mistral_detector.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: TestMistralDetector, setUp, test_has_tool_call_json_array_format, test_has_tool_call_compact_format
- Key code excerpts:

```diff
diff -- test/registered/unit/function_call/test_mistral_detector.py
@@ -0,0 +1,224 @@
+"""Unit tests for MistralDetector — no server, no model loading."""
+import json
+from sglang.srt.entrypoints.openai.protocol import Function, Tool
+from sglang.srt.function_call.mistral_detector import MistralDetector
+from sglang.test.ci.ci_register import register_cpu_ci
+from sglang.test.test_utils import CustomTestCase
```

- Reviewed files:
  - tests: `test/registered/unit/function_call/test_mistral_detector.py` added +224/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/function_call/test_hermes_detector.py`, `test/registered/unit/function_call/test_llama32_detector.py`, `test/registered/unit/function_call/test_mistral_detector.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
