# sglang Llama 4 Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/basic_usage/llama4.md` | [#13421](https://github.com/sgl-project/sglang/pull/13421) |
| `docs_new/cookbook/autoregressive/Llama/Llama4.mdx` | no direct PR-number commit |
| `docs_new/docs/basic_usage/llama4.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/llama4-maverick-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/llama4-scout-deployment.jsx` | no direct PR-number commit |
| `examples/chat_template/tool_chat_template_llama4_pythonic.jinja` | [#6679](https://github.com/sgl-project/sglang/pull/6679) |
| `python/sglang/srt/models/llama4.py` | [#5092](https://github.com/sgl-project/sglang/pull/5092), [#5127](https://github.com/sgl-project/sglang/pull/5127), [#5128](https://github.com/sgl-project/sglang/pull/5128), [#5144](https://github.com/sgl-project/sglang/pull/5144), [#5159](https://github.com/sgl-project/sglang/pull/5159), [#5194](https://github.com/sgl-project/sglang/pull/5194), [#6752](https://github.com/sgl-project/sglang/pull/6752), [#7729](https://github.com/sgl-project/sglang/pull/7729), [#8512](https://github.com/sgl-project/sglang/pull/8512), [#8683](https://github.com/sgl-project/sglang/pull/8683), [#9101](https://github.com/sgl-project/sglang/pull/9101), [#10047](https://github.com/sgl-project/sglang/pull/10047), ... (16 total) |
| `python/sglang/srt/models/mllama4.py` | [#5092](https://github.com/sgl-project/sglang/pull/5092), [#5144](https://github.com/sgl-project/sglang/pull/5144), [#5194](https://github.com/sgl-project/sglang/pull/5194), [#6985](https://github.com/sgl-project/sglang/pull/6985), [#7129](https://github.com/sgl-project/sglang/pull/7129), [#8272](https://github.com/sgl-project/sglang/pull/8272), [#8512](https://github.com/sgl-project/sglang/pull/8512), [#10042](https://github.com/sgl-project/sglang/pull/10042), [#10047](https://github.com/sgl-project/sglang/pull/10047), [#10611](https://github.com/sgl-project/sglang/pull/10611), [#11282](https://github.com/sgl-project/sglang/pull/11282) |
| `python/sglang/srt/multimodal/processors/mllama4.py` | [#7840](https://github.com/sgl-project/sglang/pull/7840), [#8156](https://github.com/sgl-project/sglang/pull/8156) |
| `test/manual/lora/test_lora_llama4.py` | no direct PR-number commit |
| `test/manual/models/test_llama4_models.py` | no direct PR-number commit |
| `test/registered/8-gpu-models/test_llama4.py` | [#16599](https://github.com/sgl-project/sglang/pull/16599), [#16971](https://github.com/sgl-project/sglang/pull/16971) |
| `test/registered/ascend/llm_models/test_npu_llama4_scount_17b_16e.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 28
- Extra PRs preserved from existing docs: 1
- Total PRs in this document: 29
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-04-07 | [#5092](https://github.com/sgl-project/sglang/pull/5092) | merged | Add Llama4 support | `python/sglang/srt/models/llama4.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py`, `python/sglang/srt/models/mllama4.py` |
| 2025-04-08 | [#5159](https://github.com/sgl-project/sglang/pull/5159) | merged | Support 2x8xH100 for Llama 4 | `python/sglang/srt/models/llama4.py` |
| 2025-04-09 | [#5128](https://github.com/sgl-project/sglang/pull/5128) | merged | Optimize topk operation in llama4 | `python/sglang/srt/models/llama4.py` |
| 2025-04-09 | [#5194](https://github.com/sgl-project/sglang/pull/5194) | merged | Support Llama4 fp8 inference | `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py` |
| 2025-04-09 | [#5144](https://github.com/sgl-project/sglang/pull/5144) | merged | model: support mllama4 | `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py`, `python/sglang/srt/models/llama4.py` |
| 2025-04-11 | [#5127](https://github.com/sgl-project/sglang/pull/5127) | merged | Optimize attention in llama4 | `python/sglang/srt/models/llama4.py` |
| 2025-05-09 | [#6162](https://github.com/sgl-project/sglang/pull/6162) | merged | [Bugfix] Fix Llama4 gibberish output with long context and CUDA graph | `python/sglang/srt/layers/attention/flashattention_backend.py` |
| 2025-05-31 | [#6679](https://github.com/sgl-project/sglang/pull/6679) | merged | update llama4 chat template and pythonic parser | `examples/chat_template/tool_chat_template_llama4_pythonic.jinja` |
| 2025-07-01 | [#6985](https://github.com/sgl-project/sglang/pull/6985) | merged | support llama4 eagle3 | `python/sglang/srt/models/mllama4.py` |
| 2025-07-04 | [#7729](https://github.com/sgl-project/sglang/pull/7729) | merged | refactor llama4 dp attention logic | `python/sglang/srt/models/llama4.py` |
| 2025-07-08 | [#7129](https://github.com/sgl-project/sglang/pull/7129) | merged | Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang | `python/sglang/srt/models/mllama4.py` |
| 2025-07-08 | [#7840](https://github.com/sgl-project/sglang/pull/7840) | merged | Fix llama4 vision | `python/sglang/srt/multimodal/processors/mllama4.py` |
| 2025-07-23 | [#8272](https://github.com/sgl-project/sglang/pull/8272) | merged | Skip llama4 vision module loading when multimodal disabled | `python/sglang/srt/models/mllama4.py` |
| 2025-07-27 | [#8156](https://github.com/sgl-project/sglang/pull/8156) | merged | Support precomputed_embeddings for Llama 4 | `python/sglang/srt/multimodal/processors/mllama4.py` |
| 2025-08-02 | [#8512](https://github.com/sgl-project/sglang/pull/8512) | merged | model: adapt mllama4 to VisionAttention | `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py` |
| 2025-08-03 | [#8683](https://github.com/sgl-project/sglang/pull/8683) | merged | [fix] Fix divide by zero error for llama4. | `python/sglang/srt/models/llama4.py` |
| 2025-08-09 | [#6752](https://github.com/sgl-project/sglang/pull/6752) | merged | Tiny Llama4 type error in constructor | `python/sglang/srt/models/llama4.py` |
| 2025-08-14 | [#9101](https://github.com/sgl-project/sglang/pull/9101) | merged | Feature: support qwen and llama4 reducescatter for dp attention padding | `python/sglang/srt/models/llama4.py` |
| 2025-09-06 | [#10047](https://github.com/sgl-project/sglang/pull/10047) | merged | support Llama4 with non uniformed intermediate size across layers for… | `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py`, `test/srt/lora/test_lora_llama4.py` |
| 2025-09-29 | [#10611](https://github.com/sgl-project/sglang/pull/10611) | merged | fix: fp8 mllama4 without vision modules being quantized | `python/sglang/srt/models/mllama4.py` |
| 2025-10-06 | [#10042](https://github.com/sgl-project/sglang/pull/10042) | merged | [quantization] Fix scale remapping for mllama4 | `python/sglang/srt/models/mllama4.py` |
| 2025-10-07 | [#11282](https://github.com/sgl-project/sglang/pull/11282) | merged | fix: correct scale parameter remapping logic in Llama4ForConditionalGeneration | `python/sglang/srt/models/mllama4.py` |
| 2025-10-31 | [#12405](https://github.com/sgl-project/sglang/pull/12405) | merged | Fix the shared expert & routed expert overlap in Llama 4 | `python/sglang/srt/models/llama4.py` |
| 2025-11-08 | [#12811](https://github.com/sgl-project/sglang/pull/12811) | merged | use fast stream instead of torch.cuda.current_stream in llama 4 shared experts overlap | `python/sglang/srt/models/llama4.py` |
| 2025-11-25 | [#13421](https://github.com/sgl-project/sglang/pull/13421) | merged | Add Llama4 attention backend auto-selection | `docs/basic_usage/llama4.md` |
| 2026-01-07 | [#16599](https://github.com/sgl-project/sglang/pull/16599) | merged | ci: adding llama4 placeholder test to nightly | `test/registered/8-gpu-models/test_llama4.py` |
| 2026-01-14 | [#16971](https://github.com/sgl-project/sglang/pull/16971) | merged | fix: renaming test file and job names + skip blocking llama4 nightly | `test/registered/8-gpu-models/test_llama4.py` |
| 2026-01-30 | [#12813](https://github.com/sgl-project/sglang/pull/12813) | merged | add weightless qk norm to RMSNorm interface for Llama 4 | `python/sglang/srt/models/llama4.py` |
| 2026-02-27 | [#17123](https://github.com/sgl-project/sglang/pull/17123) | merged | llama4 npu adapt | `python/sglang/srt/models/llama4.py` |

## Per-PR Diff Audit Cards

### PR #5092 - Add Llama4 support

- Link: https://github.com/sgl-project/sglang/pull/5092
- Status/date: merged / 2025-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`, `python/sglang/srt/models/mllama4.py`; associated commits `f04c80dc42be`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +2213/-21, 2640 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Llama4 support"; model line: Llama 4; category: model support/runtime entry; main diff: `python/sglang/srt/models/llama4.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py`, `python/sglang/srt/models/mllama4.py`; PR body summary: - Add LM support for meta-llama/Llama4 model family. - Add conversation template support for Llama4 models @ispobock - Implement Llama4 language model @ch-wan - Implement local....
- Key implementation: `python/sglang/srt/models/llama4.py` added +420/-0 (420 lines); hunks: -0,0 +1,420; symbols: Llama4MoE, custom_routing_function, __init__, forward, touching `Llama4MoE, custom_routing_function, __init__`; `python/sglang/srt/managers/multimodal_processors/mllama4.py` added +161/-0 (161 lines); hunks: -0,0 +1,161; symbols: Mllama4ImageProcessor, __init__, process_mm_data_async, get_patch_per_chunk, touching `Mllama4ImageProcessor, __init__, process_mm_data_async`; `python/sglang/srt/models/mllama4.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: Llama4ForConditionalGeneration, __init__, forward, permute_qk_weight_for_rotary, touching `Llama4ForConditionalGeneration, __init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/llama4.py` added +420/-0 (420 lines); hunks: -0,0 +1,420; symbols: Llama4MoE, custom_routing_function, __init__, forward
  - `python/sglang/srt/managers/multimodal_processors/mllama4.py` added +161/-0 (161 lines); hunks: -0,0 +1,161; symbols: Mllama4ImageProcessor, __init__, process_mm_data_async, get_patch_per_chunk
  - `python/sglang/srt/models/mllama4.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: Llama4ForConditionalGeneration, __init__, forward, permute_qk_weight_for_rotary
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -0,0 +1,420 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/managers/multimodal_processors/mllama4.py
@@ -0,0 +1,161 @@
+from typing import List, Mapping, Optional, Tuple, Union
+import torch
+from PIL import Image
+from transformers import Llama4Processor
+from transformers.image_utils import SizeDict
+from transformers.models.llama4.image_processing_llama4 import (
diff -- python/sglang/srt/models/mllama4.py
@@ -0,0 +1,154 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama4.py` added +420/-0; `python/sglang/srt/managers/multimodal_processors/mllama4.py` added +161/-0; `python/sglang/srt/models/mllama4.py` added +154/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/lang/chat_template.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/conversation.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5159 - Support 2x8xH100 for Llama 4

- Link: https://github.com/sgl-project/sglang/pull/5159
- Status/date: merged / 2025-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`; associated commits `5039d547724c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +77/-19, 178 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support 2x8xH100 for Llama 4"; model line: Llama 4; category: docs/tests/CI; main diff: `python/sglang/srt/models/llama4.py`; PR body summary: 2x8xH100 + Maverick Comparison test: 1x8xH100 + Scout.
- Key implementation: `python/sglang/srt/models/llama4.py` modified +77/-19 (96 lines); hunks: -27,6 +27,13; -38,6 +45,7; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/llama4.py` modified +77/-19 (96 lines); hunks: -27,6 +27,13; -38,6 +45,7; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -27,6 +27,13 @@
+from sglang.srt.layers.dp_attention import (
+    dp_gather_partial,
+    dp_scatter,
+    get_attention_dp_size,
+    get_attention_tp_rank,
+    get_attention_tp_size,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama4.py` modified +77/-19
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5128 - Optimize topk operation in llama4

- Link: https://github.com/sgl-project/sglang/pull/5128
- Status/date: merged / 2025-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`; associated commits `86a876d883a7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +18/-15, 75 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimize topk operation in llama4"; model line: Llama 4; category: performance/backend optimization; main diff: `python/sglang/srt/models/llama4.py`; PR body summary: Before: * bs=1: 120.41 * bs=64: 3514 * gsm8k: 92.3 After (original PR): * bs=1: 123.14 (improved) * bs=64: 3681 (only run twice, may be randomness or improvement) * gsm8k: 92.6....
- Key implementation: `python/sglang/srt/models/llama4.py` modified +2/-2 (4 lines); hunks: -48,7 +48,7; -63,7 +63,7 @@ def custom_routing_function(; symbols: custom_routing_function, touching `custom_routing_function`.
- Code diff details:
  - `python/sglang/srt/models/llama4.py` modified +2/-2 (4 lines); hunks: -48,7 +48,7; -63,7 +63,7 @@ def custom_routing_function(; symbols: custom_routing_function
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -48,7 +48,7 @@
-from sglang.srt.utils import add_prefix, get_compiler_backend, make_layers
+from sglang.srt.utils import add_prefix, fast_topk, get_compiler_backend, make_layers
@@ -63,7 +63,7 @@ def custom_routing_function(
-        router_scores_aK, router_indices_aK = torch.topk(gating_output, topk, dim=-1)
+        router_scores_aK, router_indices_aK = fast_topk(gating_output, topk, dim=-1)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama4.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/llama4.py`, `python/sglang/srt/speculative/eagle_utils.py`, `python/sglang/srt/speculative/eagle_worker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5194 - Support Llama4 fp8 inference

- Link: https://github.com/sgl-project/sglang/pull/5194
- Status/date: merged / 2025-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`, `python/sglang/srt/models/mllama4.py`; associated commits `406524821457`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +537/-106, 1026 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support Llama4 fp8 inference"; model line: Llama 4; category: performance/backend optimization; main diff: `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py`; PR body summary: Support llama4 fp8 inference for meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 @zhyncs @ispobock @zhaochenyang20 | Dataset | Score | |-----|------| | gsm8k | 93.6 | | mmlu |....
- Key implementation: `python/sglang/srt/models/mllama4.py` modified +52/-18 (70 lines); hunks: -7,6 +7,7; -16,6 +17,7; symbols: Llama4ForConditionalGeneration, __init__, load_weights, touching `Llama4ForConditionalGeneration, __init__, load_weights`; `python/sglang/srt/models/llama4.py` modified +1/-1 (2 lines); hunks: -414,7 +414,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/mllama4.py` modified +52/-18 (70 lines); hunks: -7,6 +7,7; -16,6 +17,7; symbols: Llama4ForConditionalGeneration, __init__, load_weights
  - `python/sglang/srt/models/llama4.py` modified +1/-1 (2 lines); hunks: -414,7 +414,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -7,6 +7,7 @@
+from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
@@ -16,6 +17,7 @@
+        "gate_up_proj": ["gate_proj", "up_proj"],
@@ -96,6 +98,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
+        # Params for weights, fp8 weight scales, fp8 activation scales
+        # (param_name, weight_name, expert_id, shard_id)
diff -- python/sglang/srt/models/llama4.py
@@ -414,7 +414,7 @@ def __init__(
-            prefix="model.layers",
+            prefix=add_prefix("layers", prefix),
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +52/-18; `python/sglang/srt/models/llama4.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_int8_kernel.py`, `test/srt/test_triton_moe_channel_fp8_kernel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5144 - model: support mllama4

- Link: https://github.com/sgl-project/sglang/pull/5144
- Status/date: merged / 2025-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`, `python/sglang/srt/models/mllama4.py`; associated commits `fbebcb7aa4aa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +145/-65, 403 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "model: support mllama4"; model line: Llama 4; category: model support/runtime entry; main diff: `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py`, `python/sglang/srt/models/llama4.py`; no usable PR-body summary.
- Key implementation: `python/sglang/srt/models/mllama4.py` modified +50/-11 (61 lines); hunks: -1,14 +1,19; -30,6 +35,9 @@ def __init__(; symbols: __init__, pad_input_ids, get_image_feature, forward, touching `__init__, pad_input_ids, get_image_feature`; `python/sglang/srt/managers/multimodal_processors/mllama4.py` modified +21/-36 (57 lines); hunks: -1,10 +1,8; -15,7 +13,6; symbols: Mllama4ImageProcessor, __init__, process_mm_data_async, touching `Mllama4ImageProcessor, __init__, process_mm_data_async`; `python/sglang/srt/models/llama4.py` modified +3/-0 (3 lines); hunks: -466,6 +466,9 @@ def __init__(; symbols: __init__, get_input_embeddings, _init_model, touching `__init__, get_input_embeddings, _init_model`.
- Code diff details:
  - `python/sglang/srt/models/mllama4.py` modified +50/-11 (61 lines); hunks: -1,14 +1,19; -30,6 +35,9 @@ def __init__(; symbols: __init__, pad_input_ids, get_image_feature, forward
  - `python/sglang/srt/managers/multimodal_processors/mllama4.py` modified +21/-36 (57 lines); hunks: -1,10 +1,8; -15,7 +13,6; symbols: Mllama4ImageProcessor, __init__, process_mm_data_async
  - `python/sglang/srt/models/llama4.py` modified +3/-0 (3 lines); hunks: -466,6 +466,9 @@ def __init__(; symbols: __init__, get_input_embeddings, _init_model
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -1,14 +1,19 @@
-# TODO: add Aapted from vllm/mllama4.py
-from typing import Optional, Set, Tuple
+from typing import List, Optional, Set, Tuple
-from transformers import Llama4Config
+from transformers import Llama4Config, Llama4VisionModel
+from transformers.models.llama4.modeling_llama4 import Llama4MultiModalProjector
diff -- python/sglang/srt/managers/multimodal_processors/mllama4.py
@@ -1,10 +1,8 @@
-from typing import List, Mapping, Optional, Tuple, Union
+from typing import List, Union
-from PIL import Image
-from transformers import Llama4Processor
-from transformers.models.llama4.image_processing_llama4 import (
+from transformers.models.llama4.image_processing_llama4_fast import (
diff -- python/sglang/srt/models/llama4.py
@@ -466,6 +466,9 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +50/-11; `python/sglang/srt/managers/multimodal_processors/mllama4.py` modified +21/-36; `python/sglang/srt/models/llama4.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `test/srt/test_vision_openai_server.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5127 - Optimize attention in llama4

- Link: https://github.com/sgl-project/sglang/pull/5127
- Status/date: merged / 2025-04-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`; associated commits `cd7e32e2cb15`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +18/-12, 56 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimize attention in llama4"; model line: Llama 4; category: performance/backend optimization; main diff: `python/sglang/srt/models/llama4.py`; PR body summary: Before: * bs=1: 120.41 * bs=64: 3514 * gsm8k: 92.3 After: * bs=1: 124.12 * bs=64: 3634 * gsm8k: 92.0 Command:.
- Key implementation: `python/sglang/srt/models/llama4.py` modified +18/-12 (30 lines); hunks: -240,37 +240,43 @@ def __init__(; symbols: __init__, _get_attn_scale, _mul_attn_scale, forward, touching `__init__, _get_attn_scale, _mul_attn_scale`.
- Code diff details:
  - `python/sglang/srt/models/llama4.py` modified +18/-12 (30 lines); hunks: -240,37 +240,43 @@ def __init__(; symbols: __init__, _get_attn_scale, _mul_attn_scale, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -240,37 +240,43 @@ def __init__(
+    @torch.compile(dynamic=True, backend=get_compiler_backend())
+    def _mul_attn_scale(self, positions, q):
+        attn_scale = self._get_attn_scale(positions)
+        return (q * attn_scale).to(q.dtype)
-        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
+        qk, v = qkv.split([self.q_size + self.kv_size, self.kv_size], dim=-1)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama4.py` modified +18/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6162 - [Bugfix] Fix Llama4 gibberish output with long context and CUDA graph

- Link: https://github.com/sgl-project/sglang/pull/6162
- Status/date: merged / 2025-05-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +125/-8, 203 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Llama4 gibberish output with long context and CUDA graph"; model line: Llama 4; category: bug fix; main diff: `python/sglang/srt/layers/attention/flashattention_backend.py`; PR body summary: Llama4 is generating gibberish output with long context (larger than 8192) with cuda graph. This PR fixes the issue. This is because the local attention mechanism wasn't properl....
- Key implementation: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +125/-8 (133 lines); hunks: -913,8 +913,10 @@ def forward_decode(; -970,7 +972,7 @@ def forward_decode(; symbols: forward_decode, init_cuda_graph_state, touching `forward_decode, init_cuda_graph_state`.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +125/-8 (133 lines); hunks: -913,8 +913,10 @@ def forward_decode(; -970,7 +972,7 @@ def forward_decode(; symbols: forward_decode, init_cuda_graph_state
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/flashattention_backend.py
@@ -913,8 +913,10 @@ def forward_decode(
-        use_local_attention = (
-            self.attention_chunk_size is not None and local_attn_metadata is not None
+        use_local_attn = (
+            self.attention_chunk_size is not None
+            and local_attn_metadata is not None
+            and (hasattr(layer, "use_irope") and layer.use_irope)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +125/-8
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/flashattention_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6679 - update llama4 chat template and pythonic parser

- Link: https://github.com/sgl-project/sglang/pull/6679
- Status/date: merged / 2025-05-31
- Trace source: `git log --name-only -- <model-files>` found it through `examples/chat_template/tool_chat_template_llama4_pythonic.jinja`; associated commits `4fac524b14a0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +165/-73, 350 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "update llama4 chat template and pythonic parser"; model line: Llama 4; category: docs/tests/CI; main diff: `examples/chat_template/tool_chat_template_llama4_pythonic.jinja`; PR body summary: update llama4 chat template and pythonic parser to remove and - llama4 chat template updated in https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/ and vllm https:....
- Key implementation: `examples/chat_template/tool_chat_template_llama4_pythonic.jinja` modified +35/-63 (98 lines); hunks: -1,86 +1,52; -92,10 +58,12.
- Code diff details:
  - `examples/chat_template/tool_chat_template_llama4_pythonic.jinja` modified +35/-63 (98 lines); hunks: -1,86 +1,52; -92,10 +58,12
- Key code excerpts:

```diff
diff -- examples/chat_template/tool_chat_template_llama4_pythonic.jinja
@@ -1,86 +1,52 @@
-{# Copied from https://github.com/yeqcharlotte/vllm/blob/4fcf68a948bbe0498dc8a98feafa102cfb1dd210/examples/tool_chat_template_llama4_pythonic.jinja to enable better model response
+{# Copied from https://github.com/wukaixingxp/vllm/blob/8a32e2a6e452a03c0e8222e3876ad6086cbf581f/examples/tool_chat_template_llama4_pythonic.jinja to enable better model response.
-{%- if custom_tools is defined %}
+{%- if custom_tools is defined and custom_tools %}
-{%- if not tools_in_user_message is defined %}
-    {%- set tools_in_user_message = false %}
```

- Reviewed files:
  - docs: `examples/chat_template/tool_chat_template_llama4_pythonic.jinja` modified +35/-63
- Risk and verification: The diff ships test coverage in `test/srt/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6985 - support llama4 eagle3

- Link: https://github.com/sgl-project/sglang/pull/6985
- Status/date: merged / 2025-07-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mllama4.py`; associated commits `886d34496475`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +139/-19, 237 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support llama4 eagle3"; model line: Llama 4; category: performance/backend optimization; main diff: `python/sglang/srt/models/mllama4.py`; PR body summary: support llama4 eagle3 draft model https://huggingface.co/nvidia/Llama-4-Maverick-17B-128E-Eagle3 Test performence test batch size=1 | Output throughput -- | -- Llama4-17B-128E +....
- Key implementation: `python/sglang/srt/models/mllama4.py` modified +29/-0 (29 lines); hunks: -223,5 +223,34 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, set_eagle3_layers_to_capture, get_embed_and_head, set_embed_and_head, touching `load_weights, set_eagle3_layers_to_capture, get_embed_and_head`.
- Code diff details:
  - `python/sglang/srt/models/mllama4.py` modified +29/-0 (29 lines); hunks: -223,5 +223,34 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, set_eagle3_layers_to_capture, get_embed_and_head, set_embed_and_head
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -223,5 +223,34 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
+    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
+        if hasattr(self.language_model, "set_eagle3_layers_to_capture"):
+            self.language_model.set_eagle3_layers_to_capture(layer_ids)
+    def get_embed_and_head(self):
+        # For EAGLE3, we delegate to the language model which should have this method
+        # If the language model doesn't have lm_head (like EAGLE3), we return None for head
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +29/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/model_loader/weight_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7729 - refactor llama4 dp attention logic

- Link: https://github.com/sgl-project/sglang/pull/7729
- Status/date: merged / 2025-07-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`; associated commits `8c298031d57e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +32/-45, 115 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "refactor llama4 dp attention logic"; model line: Llama 4; category: model implementation change; main diff: `python/sglang/srt/models/llama4.py`; PR body summary: use communicator to replace dp attention logic in llama4.
- Key implementation: `python/sglang/srt/models/llama4.py` modified +32/-45 (77 lines); hunks: -27,9 +27,8; -367,7 +366,10 @@ def __init__(; symbols: __init__, _is_moe_layer, forward, touching `__init__, _is_moe_layer, forward`.
- Code diff details:
  - `python/sglang/srt/models/llama4.py` modified +32/-45 (77 lines); hunks: -27,9 +27,8; -367,7 +366,10 @@ def __init__(; symbols: __init__, _is_moe_layer, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -27,9 +27,8 @@
+from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
-    dp_gather_partial,
-    dp_scatter,
@@ -367,7 +366,10 @@ def __init__(
-        is_moe_layer = (layer_id + 1) % config.interleave_moe_layer_step == 0
+        self.config = config
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama4.py` modified +32/-45
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7129 - Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang

- Link: https://github.com/sgl-project/sglang/pull/7129
- Status/date: merged / 2025-07-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mllama4.py`; associated commits `659907e32b95`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +643/-81, 847 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang"; model line: Llama 4; category: performance/backend optimization; main diff: `python/sglang/srt/models/mllama4.py`; PR body summary: Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang, as part of our efforts to promote ModelOpt in SGLang. See https://github.com/sgl-project/sglang/issues/5251 Resolve r....
- Key implementation: `python/sglang/srt/models/mllama4.py` modified +360/-79 (439 lines); hunks: -1,3 +1,6; -19,6 +22,13; symbols: Llama4ForConditionalGeneration, __init__, _has_vision_weights, _check_vision_weights_in_index, touching `Llama4ForConditionalGeneration, __init__, _has_vision_weights`.
- Code diff details:
  - `python/sglang/srt/models/mllama4.py` modified +360/-79 (439 lines); hunks: -1,3 +1,6; -19,6 +22,13; symbols: Llama4ForConditionalGeneration, __init__, _has_vision_weights, _check_vision_weights_in_index
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -1,3 +1,6 @@
+import json as json_lib
+import logging
+import os
@@ -19,6 +22,13 @@
+from sglang.srt.model_loader.weight_utils import (
+    default_weight_loader,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +360/-79
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7840 - Fix llama4 vision

- Link: https://github.com/sgl-project/sglang/pull/7840
- Status/date: merged / 2025-07-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/mllama4.py`; associated commits `4bab50a6b580`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +82/-63, 186 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix llama4 vision"; model line: Llama 4; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/mllama4.py`; PR body summary: This pull request primarily introduces initial support for Llama 4 vision models by integrating its conversation template and updating model recognition. Additionally, it includ....
- Key implementation: `python/sglang/srt/multimodal/processors/mllama4.py` modified +62/-60 (122 lines); hunks: -60,70 +60,72 @@ async def process_mm_data_async(; symbols: process_mm_data_async, touching `process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/mllama4.py` modified +62/-60 (122 lines); hunks: -60,70 +60,72 @@ async def process_mm_data_async(; symbols: process_mm_data_async
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/mllama4.py
@@ -60,70 +60,72 @@ async def process_mm_data_async(
-        if "pixel_values" in processor_output:
-            image_processor = processor.image_processor
-            tokenizer = self._processor.tokenizer
+        if "pixel_values" not in processor_output:  # no image processed
+            return None
-            # Calculate tile size and find supported resolutions
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/mllama4.py` modified +62/-60
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/conversation.py`, `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/multimodal/processors/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8272 - Skip llama4 vision module loading when multimodal disabled

- Link: https://github.com/sgl-project/sglang/pull/8272
- Status/date: merged / 2025-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mllama4.py`; associated commits `e2d66f60c8f8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-3, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Skip llama4 vision module loading when multimodal disabled"; model line: Llama 4; category: model implementation change; main diff: `python/sglang/srt/models/mllama4.py`; PR body summary: Skip llama4 vision module loading when multimodal disabled..
- Key implementation: `python/sglang/srt/models/mllama4.py` modified +10/-3 (13 lines); hunks: -23,6 +23,7; -55,13 +56,17 @@ def __init__(; symbols: __init__, load_weights, _should_skip_weight, _transform_weight_name, touching `__init__, load_weights, _should_skip_weight`.
- Code diff details:
  - `python/sglang/srt/models/mllama4.py` modified +10/-3 (13 lines); hunks: -23,6 +23,7; -55,13 +56,17 @@ def __init__(; symbols: __init__, load_weights, _should_skip_weight, _transform_weight_name
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -23,6 +23,7 @@
+    global_server_args_dict,
@@ -55,13 +56,17 @@ def __init__(
-        self.has_vision = self._has_vision_weights(config)
-        if not self.has_vision:
+        self.has_vision_weights = self._has_vision_weights(config)
+        if not self.has_vision_weights:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +10/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8156 - Support precomputed_embeddings for Llama 4

- Link: https://github.com/sgl-project/sglang/pull/8156
- Status/date: merged / 2025-07-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/mllama4.py`; associated commits `44d600cd675f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +449/-123, 235 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support precomputed_embeddings for Llama 4"; model line: Llama 4; category: model support/runtime entry; main diff: `python/sglang/srt/multimodal/processors/mllama4.py`; PR body summary: Support precomputed_embeddings for llama 4 to close #8065 1. Simplified mllama4 processor by calling `process_and_combine_mm_data`. 2. Added an example notebook for running visi....
- Key implementation: `python/sglang/srt/multimodal/processors/mllama4.py` modified +16/-109 (125 lines); hunks: -22,12 +22,12 @@ def __init__(self, hf_config, server_args, _processor, *args...; -37,114 +37,21 @@ async def process_mm_data_async(; symbols: __init__, process_mm_data_async, touching `__init__, process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/mllama4.py` modified +16/-109 (125 lines); hunks: -22,12 +22,12 @@ def __init__(self, hf_config, server_args, _processor, *args...; -37,114 +37,21 @@ async def process_mm_data_async(; symbols: __init__, process_mm_data_async
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/mllama4.py
@@ -22,12 +22,12 @@ def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
-        self.boi_token_index = hf_config.boi_token_index
-        self.eoi_token_index = hf_config.eoi_token_index
-        self.image_token_index = hf_config.image_token_index
-        self.multimodal_tokens = MultimodalSpecialTokens(
+        self.IM_START_TOKEN_ID = hf_config.boi_token_index
+        self.IM_END_TOKEN_ID = hf_config.eoi_token_index
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/mllama4.py` modified +16/-109
- Risk and verification: The diff ships test coverage in `test/srt/test_vlm_input_format.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8512 - model: adapt mllama4 to VisionAttention

- Link: https://github.com/sgl-project/sglang/pull/8512
- Status/date: merged / 2025-08-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`, `python/sglang/srt/models/mllama4.py`; associated commits `ea93079b3038`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +518/-52, 903 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "model: adapt mllama4 to VisionAttention"; model line: Llama 4; category: docs/tests/CI; main diff: `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py`; PR body summary: To resolve https://github.com/sgl-project/sglang/issues/8487 and close #8468, adapt mllama4 to support Vision attention WIP Benchmark & Profiling WIP.
- Key implementation: `python/sglang/srt/models/mllama4.py` modified +428/-19 (447 lines); hunks: -1,17 +1,24; -26,10 +33,10; symbols: Llama4VisionMLP, __init__, forward, pixel_shuffle, touching `Llama4VisionMLP, __init__, forward`; `python/sglang/srt/models/llama4.py` modified +11/-2 (13 lines); hunks: -241,13 +241,22 @@ def __init__(; -257,7 +266,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/mllama4.py` modified +428/-19 (447 lines); hunks: -1,17 +1,24; -26,10 +33,10; symbols: Llama4VisionMLP, __init__, forward, pixel_shuffle
  - `python/sglang/srt/models/llama4.py` modified +11/-2 (13 lines); hunks: -241,13 +241,22 @@ def __init__(; -257,7 +266,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -1,17 +1,24 @@
+import math
-from transformers import Llama4Config
+from transformers import Llama4Config, Llama4VisionConfig
-    Llama4VisionModel,
+    vision_apply_rotary_emb,
+from sglang.srt.layers.attention.vision import VisionAttention
diff -- python/sglang/srt/models/llama4.py
@@ -241,13 +241,22 @@ def __init__(
+        qkv_quant_config = quant_config
+        o_quant_config = quant_config
+        if quant_config and hasattr(quant_config, "ignore") and quant_config.ignore:
+            if add_prefix("q_proj", prefix) in quant_config.ignore:
+                qkv_quant_config = None
+            if add_prefix("o_proj", prefix) in quant_config.ignore:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +428/-19; `python/sglang/srt/models/llama4.py` modified +11/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hf_transformers_utils.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/managers/tokenizer_manager.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8683 - [fix] Fix divide by zero error for llama4.

- Link: https://github.com/sgl-project/sglang/pull/8683
- Status/date: merged / 2025-08-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`; associated commits `7ed8e51bc31e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[fix] Fix divide by zero error for llama4."; model line: Llama 4; category: bug fix; main diff: `python/sglang/srt/models/llama4.py`; PR body summary: - When loading a pure dense model (similar to llama4 guard), sglang throws the following error - Llama 4 guard model has 0 experts hence the modulo operation throws an exception....
- Key implementation: `python/sglang/srt/models/llama4.py` modified +2/-0 (2 lines); hunks: -406,6 +406,8 @@ def __init__(; symbols: __init__, _is_moe_layer, forward, touching `__init__, _is_moe_layer, forward`.
- Code diff details:
  - `python/sglang/srt/models/llama4.py` modified +2/-0 (2 lines); hunks: -406,6 +406,8 @@ def __init__(; symbols: __init__, _is_moe_layer, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -406,6 +406,8 @@ def __init__(
+        if self.config.interleave_moe_layer_step == 0:
+            return self.config.num_local_experts > 0
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama4.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6752 - Tiny Llama4 type error in constructor

- Link: https://github.com/sgl-project/sglang/pull/6752
- Status/date: merged / 2025-08-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`; associated commits `4a9f3eef90a5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Tiny Llama4 type error in constructor"; model line: Llama 4; category: bug fix; main diff: `python/sglang/srt/models/llama4.py`; PR body summary: Fix the type error. self.use_rope doesn't need to be an int `self.use_rope` should be a boolean because: 1. **It's used as a boolean flag:** 2. **RadixAttention expects boolean:....
- Key implementation: `python/sglang/srt/models/llama4.py` modified +1/-1 (2 lines); hunks: -203,7 +203,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/llama4.py` modified +1/-1 (2 lines); hunks: -203,7 +203,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -203,7 +203,7 @@ def __init__(
-        self.use_rope = int((layer_id + 1) % 4 != 0)
+        self.use_rope = (layer_id + 1) % 4 != 0
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama4.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9101 - Feature: support qwen and llama4 reducescatter for dp attention padding

- Link: https://github.com/sgl-project/sglang/pull/9101
- Status/date: merged / 2025-08-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`; associated commits `4c22897a66ab`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +68/-16, 210 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Feature: support qwen and llama4 reducescatter for dp attention padding"; model line: Llama 4; category: model support/runtime entry; main diff: `python/sglang/srt/models/llama4.py`; PR body summary: Similar to #8280 and #8539 , this PR adds support for using reduce-scatter instead of all-reduce after MoE/MLP layers in Qwen2 MoE, Qwen3 MoE, and Llama4 when DP attention uses....
- Key implementation: `python/sglang/srt/models/llama4.py` modified +16/-3 (19 lines); hunks: -131,14 +131,19 @@ def __init__(; -412,6 +417,7 @@ def __init__(; symbols: __init__, forward, _is_moe_layer, touching `__init__, forward, _is_moe_layer`.
- Code diff details:
  - `python/sglang/srt/models/llama4.py` modified +16/-3 (19 lines); hunks: -131,14 +131,19 @@ def __init__(; -412,6 +417,7 @@ def __init__(; symbols: __init__, forward, _is_moe_layer
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -131,14 +131,19 @@ def __init__(
-    def forward(self, hidden_states, forward_batch: ForwardBatch):
+    def forward(
+        self,
+        hidden_states,
+        forward_batch: ForwardBatch,
+        use_reduce_scatter: bool = False,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama4.py` modified +16/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/lora/layers.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10047 - support Llama4 with non uniformed intermediate size across layers for…

- Link: https://github.com/sgl-project/sglang/pull/10047
- Status/date: merged / 2025-09-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`, `python/sglang/srt/models/mllama4.py`; associated commits `ab62b135c18a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +123/-13, 220 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support Llama4 with non uniformed intermediate size across layers for…"; model line: Llama 4; category: model support/runtime entry; main diff: `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py`, `test/srt/lora/test_lora_llama4.py`; PR body summary: … Lora Llama4 doesn't support Lora for all modules since the intermediate_sizes are different between moe layer and non-moe layers, it would fail since lora mem pool is allocate....
- Key implementation: `python/sglang/srt/models/mllama4.py` modified +25/-0 (25 lines); hunks: -961,5 +961,30 @@ def get_embed(self):; symbols: get_embed, set_embed, get_hidden_dim, touching `get_embed, set_embed, get_hidden_dim`; `python/sglang/srt/models/llama4.py` modified +9/-0 (9 lines); hunks: -423,6 +423,12 @@ def _is_moe_layer(self, layer_id: int) -> bool:; -540,6 +546,9 @@ def __init__(; symbols: _is_moe_layer, get_intermediate_size, forward, __init__, touching `_is_moe_layer, get_intermediate_size, forward`; `test/srt/lora/test_lora_llama4.py` added +61/-0 (61 lines); hunks: -0,0 +1,61; symbols: TestLlama4LoRA, setUpClass, test_bringup, touching `TestLlama4LoRA, setUpClass, test_bringup`.
- Code diff details:
  - `python/sglang/srt/models/mllama4.py` modified +25/-0 (25 lines); hunks: -961,5 +961,30 @@ def get_embed(self):; symbols: get_embed, set_embed, get_hidden_dim
  - `python/sglang/srt/models/llama4.py` modified +9/-0 (9 lines); hunks: -423,6 +423,12 @@ def _is_moe_layer(self, layer_id: int) -> bool:; -540,6 +546,9 @@ def __init__(; symbols: _is_moe_layer, get_intermediate_size, forward, __init__
  - `test/srt/lora/test_lora_llama4.py` added +61/-0 (61 lines); hunks: -0,0 +1,61; symbols: TestLlama4LoRA, setUpClass, test_bringup
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -961,5 +961,30 @@ def get_embed(self):
+    def get_hidden_dim(self, module_name, layer_idx):
+        # return input_dim, output_dim
+        if module_name == "qkv_proj":
+            return (
+                self.config.hidden_size,
+                self.config.head_dim
diff -- python/sglang/srt/models/llama4.py
@@ -423,6 +423,12 @@ def _is_moe_layer(self, layer_id: int) -> bool:
+    def get_intermediate_size(self) -> int:
+        if isinstance(self.feed_forward, Llama4MoE):
+            return self.config.intermediate_size
+        else:
+            return self.config.intermediate_size_mlp
@@ -540,6 +546,9 @@ def __init__(
diff -- test/srt/lora/test_lora_llama4.py
@@ -0,0 +1,61 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +25/-0; `python/sglang/srt/models/llama4.py` modified +9/-0
  - tests: `test/srt/lora/test_lora_llama4.py` added +61/-0
- Risk and verification: The diff ships test coverage in `test/srt/lora/test_lora_llama4.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10611 - fix: fp8 mllama4 without vision modules being quantized

- Link: https://github.com/sgl-project/sglang/pull/10611
- Status/date: merged / 2025-09-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mllama4.py`; associated commits `9de1320b637c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +14/-3, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: fp8 mllama4 without vision modules being quantized"; model line: Llama 4; category: bug fix; main diff: `python/sglang/srt/models/mllama4.py`; PR body summary: fix #9758, where vision models are not quantized in a quantized model.
- Key implementation: `python/sglang/srt/models/mllama4.py` modified +14/-3 (17 lines); hunks: -291,7 +291,7 @@ def __init__(; -446,9 +446,20 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/mllama4.py` modified +14/-3 (17 lines); hunks: -291,7 +291,7 @@ def __init__(; -446,9 +446,20 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -291,7 +291,7 @@ def __init__(
-        hidden_states = hidden_states.permute(0, 2, 1)
+        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
@@ -446,9 +446,20 @@ def __init__(
+            # TODO: make this more general
+            ignore_quant_layers = getattr(config, "quantization_config", {}).get(
+                "ignore", {}
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +14/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10042 - [quantization] Fix scale remapping for mllama4

- Link: https://github.com/sgl-project/sglang/pull/10042
- Status/date: merged / 2025-10-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mllama4.py`; associated commits `c7a104c12bda`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[quantization] Fix scale remapping for mllama4"; model line: Llama 4; category: bug fix; main diff: `python/sglang/srt/models/mllama4.py`; PR body summary: Small bugfix. Previously the method always returns False regardless of whether remap happens. This PR fix kv scale loading for ckpts that store kv_scale in naming pattern that a....
- Key implementation: `python/sglang/srt/models/mllama4.py` modified +1/-1 (2 lines); hunks: -700,7 +700,7 @@ def _handle_scale_remapping(self, name: str, params_dict: di...; symbols: _handle_scale_remapping, _handle_stacked_params, touching `_handle_scale_remapping, _handle_stacked_params`.
- Code diff details:
  - `python/sglang/srt/models/mllama4.py` modified +1/-1 (2 lines); hunks: -700,7 +700,7 @@ def _handle_scale_remapping(self, name: str, params_dict: di...; symbols: _handle_scale_remapping, _handle_stacked_params
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -700,7 +700,7 @@ def _handle_scale_remapping(self, name: str, params_dict: dict) -> bool:
-            return remapped_name is None
+            return remapped_name is not None and remapped_name != name
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11282 - fix: correct scale parameter remapping logic in Llama4ForConditionalGeneration

- Link: https://github.com/sgl-project/sglang/pull/11282
- Status/date: merged / 2025-10-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/mllama4.py`; associated commits `fd8a0b29c044`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: correct scale parameter remapping logic in Llama4ForConditionalGeneration"; model line: Llama 4; category: bug fix; main diff: `python/sglang/srt/models/mllama4.py`; PR body summary: Fix the issue below, which is caused by https://github.com/sgl-project/sglang/pull/10042 It looks like https://github.com/sgl-project/sglang/pull/10611 should be included in the....
- Key implementation: `python/sglang/srt/models/mllama4.py` modified +1/-1 (2 lines); hunks: -700,7 +700,7 @@ def _handle_scale_remapping(self, name: str, params_dict: di...; symbols: _handle_scale_remapping, _handle_stacked_params, touching `_handle_scale_remapping, _handle_stacked_params`.
- Code diff details:
  - `python/sglang/srt/models/mllama4.py` modified +1/-1 (2 lines); hunks: -700,7 +700,7 @@ def _handle_scale_remapping(self, name: str, params_dict: di...; symbols: _handle_scale_remapping, _handle_stacked_params
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -700,7 +700,7 @@ def _handle_scale_remapping(self, name: str, params_dict: dict) -> bool:
-            return remapped_name is not None and remapped_name != name
+            return remapped_name != name
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12405 - Fix the shared expert & routed expert overlap in Llama 4

- Link: https://github.com/sgl-project/sglang/pull/12405
- Status/date: merged / 2025-10-31
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`; associated commits `34c286b8115e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix the shared expert & routed expert overlap in Llama 4"; model line: Llama 4; category: bug fix; main diff: `python/sglang/srt/models/llama4.py`; PR body summary: It seems that the original PR to overlap the shared expert and routed expert computation for Llama 4 specifically was correct, but incorrectly applied: https://github.com/sgl-pr....
- Key implementation: `python/sglang/srt/models/llama4.py` modified +1/-1 (2 lines); hunks: -148,7 +148,7 @@ def forward(; symbols: forward, _forward_core, touching `forward, _forward_core`.
- Code diff details:
  - `python/sglang/srt/models/llama4.py` modified +1/-1 (2 lines); hunks: -148,7 +148,7 @@ def forward(; symbols: forward, _forward_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -148,7 +148,7 @@ def forward(
-        if hidden_states.shape[0] < 4 and _is_cuda:
+        if _is_cuda:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama4.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12811 - use fast stream instead of torch.cuda.current_stream in llama 4 shared experts overlap

- Link: https://github.com/sgl-project/sglang/pull/12811
- Status/date: merged / 2025-11-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`; associated commits `49653c88964f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-2, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "use fast stream instead of torch.cuda.current_stream in llama 4 shared experts overlap"; model line: Llama 4; category: performance/backend optimization; main diff: `python/sglang/srt/models/llama4.py`; PR body summary: Don't use `torch.cuda.current_stream`, use `get_current_device_stream_fast()`, just like #12524 | Metric | Before | After | Δ | Gain | |:--|:--:|:--:|:--:|:--:| | Request Throug....
- Key implementation: `python/sglang/srt/models/llama4.py` modified +3/-2 (5 lines); hunks: -58,6 +58,7; -164,7 +165,7 @@ def _forward_core_normal(self, hidden_states):; symbols: _forward_core_normal, _forward_core_shared_routed_overlap, touching `_forward_core_normal, _forward_core_shared_routed_overlap`.
- Code diff details:
  - `python/sglang/srt/models/llama4.py` modified +3/-2 (5 lines); hunks: -58,6 +58,7; -164,7 +165,7 @@ def _forward_core_normal(self, hidden_states):; symbols: _forward_core_normal, _forward_core_shared_routed_overlap
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -58,6 +58,7 @@
+from sglang.srt.utils.common import get_current_device_stream_fast
@@ -164,7 +165,7 @@ def _forward_core_normal(self, hidden_states):
-        alt_stream.wait_stream(self.device_module.current_stream())
+        alt_stream.wait_stream(get_current_device_stream_fast())
@@ -173,7 +174,7 @@ def _forward_core_shared_routed_overlap(self, hidden_states):
-        self.device_module.current_stream().wait_stream(alt_stream)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama4.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13421 - Add Llama4 attention backend auto-selection

- Link: https://github.com/sgl-project/sglang/pull/13421
- Status/date: merged / 2025-11-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/llama4.md`; associated commits `fcccaf9001ab`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +24/-5, 50 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Llama4 attention backend auto-selection"; model line: Llama 4; category: bug fix; main diff: `docs/basic_usage/llama4.md`; PR body summary: When using Llama4 models without explicitly specifying the `--attention-backend` parameter, user encounter an `AssertionError` during server initialization: `AssertionError: fa3....
- Key implementation: `docs/basic_usage/llama4.md` modified +9/-0 (9 lines); hunks: -21,6 +21,15 @@ python3 -m sglang.launch_server \.
- Code diff details:
  - `docs/basic_usage/llama4.md` modified +9/-0 (9 lines); hunks: -21,6 +21,15 @@ python3 -m sglang.launch_server \
- Key code excerpts:

```diff
diff -- docs/basic_usage/llama4.md
@@ -21,6 +21,15 @@ python3 -m sglang.launch_server \
+- **Attention Backend Auto-Selection**: SGLang automatically selects the optimal attention backend for Llama 4 based on your hardware. You typically don't need to specify `--atten
+  - **Blackwell GPUs (B200/GB200)**: `trtllm_mha`
+  - **Hopper GPUs (H100/H200)**: `fa3`
+  - **AMD GPUs**: `aiter`
+  - **Intel XPU**: `intel_xpu`
+  - **Other platforms**: `triton` (fallback)
```

- Reviewed files:
  - docs: `docs/basic_usage/llama4.md` modified +9/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16599 - ci: adding llama4 placeholder test to nightly

- Link: https://github.com/sgl-project/sglang/pull/16599
- Status/date: merged / 2026-01-07
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_llama4.py`; associated commits `2ff872311b4d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +55/-0, 56 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "ci: adding llama4 placeholder test to nightly"; model line: Llama 4; category: docs/tests/CI; main diff: `test/registered/8-gpu-models/test_llama4.py`; PR body summary: Adding llama4 placeholder test to nightly based on pr-test file. Pending merge from @ispobock for exact configs. Added new test-llama4.py. pending merge from ke before testing. n/a.
- Key implementation: `test/registered/8-gpu-models/test_llama4.py` added +55/-0 (55 lines); hunks: -0,0 +1,55; symbols: TestLlama4Unified, for, test_llama4, touching `TestLlama4Unified, for, test_llama4`.
- Code diff details:
  - `test/registered/8-gpu-models/test_llama4.py` added +55/-0 (55 lines); hunks: -0,0 +1,55; symbols: TestLlama4Unified, for, test_llama4
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_llama4.py
@@ -0,0 +1,55 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_llama4.py` added +55/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_llama4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16971 - fix: renaming test file and job names + skip blocking llama4 nightly

- Link: https://github.com/sgl-project/sglang/pull/16971
- Status/date: merged / 2026-01-14
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_llama4.py`; associated commits `aa2b4f766179`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +22/-21, 142 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: renaming test file and job names + skip blocking llama4 nightly"; model line: Llama 4; category: bug fix; main diff: `test/registered/8-gpu-models/test_llama4.py`; PR body summary: For clearer naming and unblocking nightly. Changed test class names for registered/8-gpu-models and adjusted nightly-test-perf-8-gpu-b200 and nightly-test-general-8-gpu-b200 n/a....
- Key implementation: `test/registered/8-gpu-models/test_llama4.py` modified +2/-1 (3 lines); hunks: -12,7 +12,8; symbols: TestLlama4Unified, TestLlama4, for, touching `TestLlama4Unified, TestLlama4, for`.
- Code diff details:
  - `test/registered/8-gpu-models/test_llama4.py` modified +2/-1 (3 lines); hunks: -12,7 +12,8; symbols: TestLlama4Unified, TestLlama4, for
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_llama4.py
@@ -12,7 +12,8 @@
-class TestLlama4Unified(unittest.TestCase):
+@unittest.skip("Blocked: Missing HF token permission for Llama 4 model")
+class TestLlama4(unittest.TestCase):
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_llama4.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_deepseek_v31.py`, `test/registered/8-gpu-models/test_deepseek_v32.py`, `test/registered/8-gpu-models/test_glm_46.py`, `test/registered/8-gpu-models/test_glm_46_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12813 - add weightless qk norm to RMSNorm interface for Llama 4

- Link: https://github.com/sgl-project/sglang/pull/12813
- Status/date: merged / 2026-01-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`; associated commits `22df62d5862b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +8/-2, 38 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "add weightless qk norm to RMSNorm interface for Llama 4"; model line: Llama 4; category: model support/runtime entry; main diff: `python/sglang/srt/models/llama4.py`; PR body summary: Otherwise with Llama 4 that has weightless qk_norm, we get these false warnings: After:.
- Key implementation: `python/sglang/srt/models/llama4.py` modified +1/-0 (1 lines); hunks: -242,6 +242,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/llama4.py` modified +1/-0 (1 lines); hunks: -242,6 +242,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -242,6 +242,7 @@ def __init__(
+                has_weight=False,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama4.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17123 - llama4 npu adapt

- Link: https://github.com/sgl-project/sglang/pull/17123
- Status/date: merged / 2026-02-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llama4.py`; associated commits `bc9190435b3f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +37/-1, 67 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "llama4 npu adapt"; model line: Llama 4; category: model support/runtime entry; main diff: `python/sglang/srt/models/llama4.py`; PR body summary: llama4 npu adapt 1. Add ascend in attention backend list for llama4 NPU GPU.
- Key implementation: `python/sglang/srt/models/llama4.py` modified +4/-0 (4 lines); hunks: -56,11 +56,13; -329,6 +331,8 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/llama4.py` modified +4/-0 (4 lines); hunks: -56,11 +56,13; -329,6 +331,8 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -56,11 +56,13 @@
+    is_npu,
+_is_npu = is_npu()
@@ -329,6 +331,8 @@ def forward(
+            if _is_npu:
+                qk = torch.cat([q_out_unused, k_out_unused], dim=-1)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama4.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `test/registered/ascend/llm_models/test_ascend_llama4_scount_17b_16e.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
