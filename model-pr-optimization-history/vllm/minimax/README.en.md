# vllm MiniMax M2 Series Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.
- Diffusion model families have been removed from this history set and are no longer part of model optimization skills.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/tool_chat_template_minimax_m1.jinja` | [#20297](https://github.com/vllm-project/vllm/pull/20297) |
| `tests/kernels/core/test_minimax_reduce_rms.py` | [#37045](https://github.com/vllm-project/vllm/pull/37045) |
| `tests/models/multimodal/processing/test_minimax_vl_01.py` | [#16328](https://github.com/vllm-project/vllm/pull/16328), [#17354](https://github.com/vllm-project/vllm/pull/17354) |
| `tests/reasoning/test_minimax_m2_append_reasoning_parser.py` | [#29882](https://github.com/vllm-project/vllm/pull/29882) |
| `tests/reasoning/test_minimax_m2_reasoning_parser.py` | [#29882](https://github.com/vllm-project/vllm/pull/29882) |
| `tests/tool_parsers/test_minimax_m2_tool_parser.py` | [#35895](https://github.com/vllm-project/vllm/pull/35895) |
| `tests/tool_parsers/test_minimax_tool_parser.py` | no direct PR-number commit |
| `vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py` | [#37045](https://github.com/vllm-project/vllm/pull/37045) |
| `vllm/model_executor/models/minimax_m2.py` | [#27535](https://github.com/vllm-project/vllm/pull/27535), [#27537](https://github.com/vllm-project/vllm/pull/27537), [#27627](https://github.com/vllm-project/vllm/pull/27627), [#30384](https://github.com/vllm-project/vllm/pull/30384), [#31493](https://github.com/vllm-project/vllm/pull/31493), [#32736](https://github.com/vllm-project/vllm/pull/32736), [#32763](https://github.com/vllm-project/vllm/pull/32763), [#36965](https://github.com/vllm-project/vllm/pull/36965), [#37045](https://github.com/vllm-project/vllm/pull/37045), [#37214](https://github.com/vllm-project/vllm/pull/37214), [#37512](https://github.com/vllm-project/vllm/pull/37512) |
| `vllm/model_executor/models/minimax_text_01.py` | [#13454](https://github.com/vllm-project/vllm/pull/13454), [#16328](https://github.com/vllm-project/vllm/pull/16328), [#19592](https://github.com/vllm-project/vllm/pull/19592), [#20211](https://github.com/vllm-project/vllm/pull/20211), [#22151](https://github.com/vllm-project/vllm/pull/22151), [#22589](https://github.com/vllm-project/vllm/pull/22589), [#22928](https://github.com/vllm-project/vllm/pull/22928), [#23831](https://github.com/vllm-project/vllm/pull/23831), [#37371](https://github.com/vllm-project/vllm/pull/37371) |
| `vllm/model_executor/models/minimax_vl_01.py` | [#16328](https://github.com/vllm-project/vllm/pull/16328), [#17354](https://github.com/vllm-project/vllm/pull/17354), [#22116](https://github.com/vllm-project/vllm/pull/22116) |
| `vllm/parser/minimax_m2_parser.py` | [#39683](https://github.com/vllm-project/vllm/pull/39683), [#39861](https://github.com/vllm-project/vllm/pull/39861) |
| `vllm/reasoning/minimax_m2_reasoning_parser.py` | [#27535](https://github.com/vllm-project/vllm/pull/27535), [#29882](https://github.com/vllm-project/vllm/pull/29882), [#35352](https://github.com/vllm-project/vllm/pull/35352) |
| `vllm/tool_parsers/minimax_m2_tool_parser.py` | [#30555](https://github.com/vllm-project/vllm/pull/30555), [#31083](https://github.com/vllm-project/vllm/pull/31083), [#32278](https://github.com/vllm-project/vllm/pull/32278), [#32342](https://github.com/vllm-project/vllm/pull/32342), [#35895](https://github.com/vllm-project/vllm/pull/35895) |
| `vllm/tool_parsers/minimax_tool_parser.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 30
- Extra PRs preserved from existing docs: 1
- Total PRs in this document: 31
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-04-01 | [#13454](https://github.com/vllm-project/vllm/pull/13454) | merged | [Model][MiniMaxText01] Support MiniMaxText01 model inference | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-04-29 | [#16328](https://github.com/vllm-project/vllm/pull/16328) | merged | [Model] support MiniMax-VL-01 model | `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/transformers_utils/configs/minimax_vl_01.py` |
| 2025-04-29 | [#17354](https://github.com/vllm-project/vllm/pull/17354) | merged | [Bugfix] Clean up MiniMax-VL and fix processing | `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py` |
| 2025-06-13 | [#19592](https://github.com/vllm-project/vllm/pull/19592) | merged | [Model] Fix minimax model cache & lm_head precision | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-06-16 | [#19677](https://github.com/vllm-project/vllm/pull/19677) | merged | [Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM) | `tests/models/registry.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/registry.py` |
| 2025-07-03 | [#20297](https://github.com/vllm-project/vllm/pull/20297) | merged | [Feature] Support MiniMax-M1 function calls features | `examples/tool_chat_template_minimax_m1.jinja` |
| 2025-07-11 | [#20211](https://github.com/vllm-project/vllm/pull/20211) | merged | [Model] Support HF format of minimax | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-08-09 | [#22151](https://github.com/vllm-project/vllm/pull/22151) | merged | [V1] [Hybrid] Support Minimax-Text-01 in V1 | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-08-15 | [#22928](https://github.com/vllm-project/vllm/pull/22928) | merged | [V1] [Hybrid] Support using float32 for state in Hybrid Models (Mamba2, Mamba1, Minimax) | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-08-19 | [#22116](https://github.com/vllm-project/vllm/pull/22116) | merged | [Bugfix] Fix broken Minimax-01-VL model | `vllm/model_executor/models/minimax_vl_01.py` |
| 2025-08-27 | [#22589](https://github.com/vllm-project/vllm/pull/22589) | merged | [V1] [Hybrid] Enable compile and piecewise CUDA graph for MiniMax-Text models | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-08-30 | [#23831](https://github.com/vllm-project/vllm/pull/23831) | merged | [V1] [Hybrid] Move MiniMaxLinearAttention into layers/mamba | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-10-26 | [#27535](https://github.com/vllm-project/vllm/pull/27535) | merged | [Model][MiniMax-M2] Support MiniMax-M2 Model | `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py` |
| 2025-10-27 | [#27537](https://github.com/vllm-project/vllm/pull/27537) | merged | Fix MiniMax-M2 copyright | `vllm/model_executor/models/minimax_m2.py` |
| 2025-10-29 | [#27627](https://github.com/vllm-project/vllm/pull/27627) | merged | Fix MiniMax-M2 rmsnorm precision and remove useless code | `vllm/model_executor/models/minimax_m2.py` |
| 2025-12-10 | [#30384](https://github.com/vllm-project/vllm/pull/30384) | merged | [BugFix] Fix minimax m2 model rotary_dim | `vllm/model_executor/models/minimax_m2.py` |
| 2025-12-11 | [#29882](https://github.com/vllm-project/vllm/pull/29882) | merged | [bugfix] fix MiniMaxM2ReasoningParser streaming output not separating reasoning_content. | `tests/reasoning/test_minimax_m2_reasoning_parser.py`, `tests/reasoning/test_minimax_m2_append_reasoning_parser.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py` |
| 2025-12-17 | [#30555](https://github.com/vllm-project/vllm/pull/30555) | merged | [Bugfix][Frontend] Prevent IndexError in MiniMax M2 tool parser during streaming extraction | `tests/tool_use/test_minimax_m2_tool_parser.py`, `vllm/tool_parsers/minimax_m2_tool_parser.py` |
| 2025-12-22 | [#31083](https://github.com/vllm-project/vllm/pull/31083) | merged | Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs | `vllm/tool_parsers/minimax_m2_tool_parser.py` |
| 2025-12-29 | [#31493](https://github.com/vllm-project/vllm/pull/31493) | merged | Optimize QKNorm for MiniMax-M2/M2.1 | `vllm/model_executor/models/minimax_m2.py` |
| 2026-01-15 | [#32342](https://github.com/vllm-project/vllm/pull/32342) | merged | Fix optional parameter parsing in MiniMax M2 tool parser #32278 | `vllm/tool_parsers/minimax_m2_tool_parser.py` |
| 2026-01-24 | [#32763](https://github.com/vllm-project/vllm/pull/32763) | merged | feat: Complete LoRA support for MiniMaxM2 Fixes #32736 | `vllm/model_executor/models/minimax_m2.py` |
| 2026-02-26 | [#35352](https://github.com/vllm-project/vllm/pull/35352) | merged | [Bug] Fix missing tag after tool call in MiniMax 2.1 | `vllm/reasoning/minimax_m2_reasoning_parser.py` |
| 2026-03-12 | [#35895](https://github.com/vllm-project/vllm/pull/35895) | merged | [Bugfix] Fix minimax_m2 tool parser when stream interval > 1 | `vllm/tool_parsers/minimax_m2_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `tests/tool_use/test_minimax_m2_tool_parser.py` |
| 2026-03-18 | [#37371](https://github.com/vllm-project/vllm/pull/37371) | merged | standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01 | `vllm/model_executor/models/minimax_text_01.py` |
| 2026-03-26 | [#37214](https://github.com/vllm-project/vllm/pull/37214) | merged | Fix minimax m2.5 nvfp4 kv scales weight loading | `vllm/model_executor/models/minimax_m2.py` |
| 2026-03-30 | [#36965](https://github.com/vllm-project/vllm/pull/36965) | merged | [Model][Quantization] Add GGUF support for MiniMax-M2.1 | `vllm/model_executor/models/minimax_m2.py` |
| 2026-04-06 | [#37512](https://github.com/vllm-project/vllm/pull/37512) | merged | MiniMax-M2: add Eagle3 speculative decoding support | `vllm/model_executor/models/minimax_m2.py` |
| 2026-04-10 | [#37045](https://github.com/vllm-project/vllm/pull/37045) | merged | [Kernel] Porting the TRTLLM minimax_allreduce_rms kernels | `vllm/model_executor/models/minimax_m2.py`, `vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py`, `tests/kernels/core/test_minimax_reduce_rms.py` |
| 2026-04-14 | [#39683](https://github.com/vllm-project/vllm/pull/39683) | merged | [Bugfix]: Fix MinimaxM2ToolParser missing tools parameter | `vllm/parser/minimax_m2_parser.py` |
| 2026-04-16 | [#39861](https://github.com/vllm-project/vllm/pull/39861) | merged | [Bugfix] Accept **kwargs in MiniMaxM2Parser.__init__() | `vllm/parser/minimax_m2_parser.py` |

## Per-PR Diff Audit Cards

### PR #13454 - [Model][MiniMaxText01] Support MiniMaxText01 model inference

- Link: https://github.com/vllm-project/vllm/pull/13454
- Status/date: merged / 2025-04-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_text_01.py`; associated commits `9ef98d527ee1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +2440/-130, 2657 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "[Model][MiniMaxText01] Support MiniMaxText01 model inference". The diff centers on `vllm/model_executor/models/minimax_text_01.py`. PR body context: # Purpose This PR is intended to support the MiniMaxText01 model inference. It can run on a single machine with 8xH800 and 8xH20, where a single H800 machine can handle a maximu...
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` added +1273/-0 (1273 lines); hunks: -0,0 +1,1273; symbols: replace_weight_name, weight_loader_with_alias, wrapper, inner_func, touching `replace_weight_name, weight_loader_with_alias, wrapper`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` added +1273/-0 (1273 lines); hunks: -0,0 +1,1273; symbols: replace_weight_name, weight_loader_with_alias, wrapper, inner_func
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -0,0 +1,1273 @@
+# SPDX-License-Identifier: Apache-2.0
+"""Inference-only MiniMaxText01 model."""
+import copy
+import math
+import re
+from typing import Dict, Iterable, List, Optional, Tuple, Union
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` added +1273/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_lightning_attn.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16328 - [Model] support MiniMax-VL-01 model

- Link: https://github.com/vllm-project/vllm/pull/16328
- Status/date: merged / 2025-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/models/minimax_vl_01.py`; associated commits `cde384cd92c8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +954/-19, 1193 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "[Model] support MiniMax-VL-01 model". The diff centers on `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/transformers_utils/configs/minimax_vl_01.py`. PR body context: CLOSES #12073 This PR is intended to support the MiniMaxVL01 model inference. It adopts the “ViT-MLP-LLM” framework, which is a commonly used technique in the field of multimoda...
- Key implementation: `vllm/model_executor/models/minimax_vl_01.py` added +615/-0 (615 lines); hunks: -0,0 +1,615; symbols: MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches, touching `MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs`; `tests/models/multimodal/processing/test_minimax_vl_01.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override, _validate_image_prompt_replacements_one, _test_image_prompt_replacements, test_processor_prompt_replacements_regression, touching `test_processor_override, _validate_image_prompt_replacements_one, _test_image_prompt_replacements`; `vllm/transformers_utils/configs/minimax_vl_01.py` added +70/-0 (70 lines); hunks: -0,0 +1,70; symbols: MiniMaxVL01Config, __init__, touching `MiniMaxVL01Config, __init__`; `vllm/transformers_utils/configs/minimax_text_01.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: MiniMaxText01Config, __init__, touching `MiniMaxText01Config, __init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_vl_01.py` added +615/-0 (615 lines); hunks: -0,0 +1,615; symbols: MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches
  - `tests/models/multimodal/processing/test_minimax_vl_01.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override, _validate_image_prompt_replacements_one, _test_image_prompt_replacements, test_processor_prompt_replacements_regression
  - `vllm/transformers_utils/configs/minimax_vl_01.py` added +70/-0 (70 lines); hunks: -0,0 +1,70; symbols: MiniMaxVL01Config, __init__
  - `vllm/transformers_utils/configs/minimax_text_01.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: MiniMaxText01Config, __init__
  - `vllm/model_executor/models/minimax_text_01.py` modified +53/-14 (67 lines); hunks: -3,7 +3,7; -110,7 +110,17 @@ def _forward(; symbols: _forward, forward, _prefill_and_mix_infer, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_vl_01.py
@@ -0,0 +1,615 @@
+# SPDX-License-Identifier: Apache-2.0
+from abc import abstractmethod
+from collections.abc import Iterable, Mapping, Sequence
+from dataclasses import dataclass
+from typing import (Final, Literal, Optional, Protocol, Set, Tuple, TypedDict,
+                    TypeVar, Union, cast)
diff -- tests/models/multimodal/processing/test_minimax_vl_01.py
@@ -0,0 +1,99 @@
+# SPDX-License-Identifier: Apache-2.0
+import pytest
+from PIL import Image
+from vllm.multimodal import MULTIMODAL_REGISTRY
+from vllm.multimodal.parse import ImageSize
+from vllm.multimodal.processing import BaseMultiModalProcessor
diff -- vllm/transformers_utils/configs/minimax_vl_01.py
@@ -0,0 +1,70 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_vl_01.py` added +615/-0; `vllm/transformers_utils/configs/minimax_vl_01.py` added +70/-0; `vllm/transformers_utils/configs/minimax_text_01.py` added +69/-0; `vllm/model_executor/models/minimax_text_01.py` modified +53/-14
  - tests: `tests/models/multimodal/processing/test_minimax_vl_01.py` added +99/-0
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17354 - [Bugfix] Clean up MiniMax-VL and fix processing

- Link: https://github.com/vllm-project/vllm/pull/17354
- Status/date: merged / 2025-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/model_executor/models/minimax_vl_01.py`; associated commits `00ee37efa236`
- Diff scope read: GitHub Pull Request files API returned 4 files, +38/-283, 442 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "[Bugfix] Clean up MiniMax-VL and fix processing". The diff centers on `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`. PR body context: Follow-up to #16328 @qscqesze can you verify this fix? Also, can you open another PR after this one to add example scripts under `examples/offline_inference/vision_language.py`...
- Key implementation: `vllm/model_executor/models/minimax_vl_01.py` modified +30/-282 (312 lines); hunks: -1,52 +1,32; -69,66 +49,8 @@ class MiniMaxVL01ImageEmbeddingInputs(TypedDict):; symbols: MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches, touching `MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs`; `tests/models/multimodal/processing/test_minimax_vl_01.py` modified +0/-1 (1 lines); hunks: -12,7 +12,6; symbols: test_processor_override, touching `test_processor_override`.
- Code diff details:
  - `vllm/model_executor/models/minimax_vl_01.py` modified +30/-282 (312 lines); hunks: -1,52 +1,32; -69,66 +49,8 @@ class MiniMaxVL01ImageEmbeddingInputs(TypedDict):; symbols: MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches
  - `tests/models/multimodal/processing/test_minimax_vl_01.py` modified +0/-1 (1 lines); hunks: -12,7 +12,6; symbols: test_processor_override
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_vl_01.py
@@ -1,52 +1,32 @@
+from collections.abc import Iterable, Mapping
+from typing import Literal, Optional, Set, Tuple, TypedDict, Union, cast
-from abc import abstractmethod
-from collections.abc import Iterable, Mapping, Sequence
-from dataclasses import dataclass
-from typing import (Final, Literal, Optional, Protocol, Set, Tuple, TypedDict,
diff -- tests/models/multimodal/processing/test_minimax_vl_01.py
@@ -12,7 +12,6 @@
-# yapf: enable
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_vl_01.py` modified +30/-282
  - tests: `tests/models/multimodal/processing/test_minimax_vl_01.py` modified +0/-1
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19592 - [Model] Fix minimax model cache & lm_head precision

- Link: https://github.com/vllm-project/vllm/pull/19592
- Status/date: merged / 2025-06-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_text_01.py`; associated commits `a24cb91600bd`
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Model] Fix minimax model cache & lm_head precision". The diff centers on `vllm/model_executor/models/minimax_text_01.py`. PR body context: Change the precision of the MiniMax model in vLLM: update the LM head and KV cache from bfloat16 (bf16) to float32 (fp32). Purpose: To improve numerical stability and output acc...
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` modified +3/-3 (6 lines); hunks: -856,7 +856,7 @@ def layer_fn(prefix):; -1021,7 +1021,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: layer_fn, __init__, forward, compute_logits, touching `layer_fn, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` modified +3/-3 (6 lines); hunks: -856,7 +856,7 @@ def layer_fn(prefix):; -1021,7 +1021,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: layer_fn, __init__, forward, compute_logits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -856,7 +856,7 @@ def layer_fn(prefix):
-        self.minimax_cache = MinimaxCacheManager(dtype=self._dtype,
+        self.minimax_cache = MinimaxCacheManager(dtype=torch.float32,
@@ -1021,7 +1021,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
+        self.lm_head.float()
@@ -1054,7 +1054,7 @@ def forward(self,
-        logits = self.logits_processor(self.lm_head, hidden_states,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/minimax_text_01.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19677 - [Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM)

- Link: https://github.com/vllm-project/vllm/pull/19677
- Status/date: merged / 2025-06-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +4/-0, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "[Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM)". The diff centers on `tests/models/registry.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/registry.py`. PR body context: We would like to propose official support for the MiniMaxM1ForCausalLM model within vLLM. This model shares the exact same architecture and inference behavior as MiniMaxText01Fo...
- Key implementation: `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -205,6 +205,8 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`; `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -370,6 +370,7 @@ Specified using `--task generate`.; `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -36,6 +36,7; symbols: name, touching `name`.
- Code diff details:
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -205,6 +205,8 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -370,6 +370,7 @@ Specified using `--task generate`.
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -36,6 +36,7; symbols: name
- Key code excerpts:

```diff
diff -- tests/models/registry.py
@@ -205,6 +205,8 @@ def check_available_online(
+    "MiniMaxM1ForCausalLM": _HfExamplesInfo("MiniMaxAI/MiniMax-M1-40k",
+                                            trust_remote_code=True),
diff -- docs/models/supported_models.md
@@ -370,6 +370,7 @@ Specified using `--task generate`.
+| `MiniMaxM1ForCausalLM`                        | MiniMax-Text                                        | `MiniMaxAI/MiniMax-M1-40k`, `MiniMaxAI/MiniMax-M1-80k`etc.
diff -- vllm/model_executor/models/registry.py
@@ -36,6 +36,7 @@
+    "MiniMaxM1ForCausalLM": ("minimax_text_01", "MiniMaxText01ForCausalLM"),
```

- Reviewed files:
  - tests: `tests/models/registry.py` modified +2/-0
  - docs: `docs/models/supported_models.md` modified +1/-0
  - runtime: `vllm/model_executor/models/registry.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20297 - [Feature] Support MiniMax-M1 function calls features

- Link: https://github.com/vllm-project/vllm/pull/20297
- Status/date: merged / 2025-07-03
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_minimax_m1.jinja`; associated commits `363528de27db`
- Diff scope read: GitHub Pull Request files API returned 5 files, +842/-1, 866 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "[Feature] Support MiniMax-M1 function calls features". The diff centers on `examples/tool_chat_template_minimax_m1.jinja`. PR body context: Support MiniMax-M1 function calls features How to use: test_scripts: Outpt: Streaming Test: Output:
- Key implementation: `examples/tool_chat_template_minimax_m1.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91.
- Code diff details:
  - `examples/tool_chat_template_minimax_m1.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91
- Key code excerpts:

```diff
diff -- examples/tool_chat_template_minimax_m1.jinja
@@ -0,0 +1,91 @@
+{{ '<begin_of_document>' -}}
+{%- if custom_tools is defined %}
+    {%- set tools = custom_tools %}
+{%- endif %}
+{%- if not tools is defined %}
+    {%- set tools = none %}
```

- Reviewed files:
  - docs: `examples/tool_chat_template_minimax_m1.jinja` added +91/-0
- Risk and verification: The diff ships test coverage in `tests/tool_use/test_minimax_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20211 - [Model] Support HF format of minimax

- Link: https://github.com/vllm-project/vllm/pull/20211
- Status/date: merged / 2025-07-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_text_01.py`; associated commits `922f316441ce`
- Diff scope read: GitHub Pull Request files API returned 3 files, +36/-11, 101 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "[Model] Support HF format of minimax". The diff centers on `vllm/model_executor/models/minimax_text_01.py`. PR body context: ## Purpose https://github.com/vllm-project/vllm/pull/20199 set us up to test the older format of `MiniMaxAI/MiniMax-Text-01` we initially implemented. This PR implements the sma...
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` modified +33/-11 (44 lines); hunks: -667,16 +667,24 @@ def __init__(; -794,6 +802,18 @@ def __init__(; symbols: __init__, which_layer, is_linear_attn_layer, touching `__init__, which_layer, is_linear_attn_layer`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` modified +33/-11 (44 lines); hunks: -667,16 +667,24 @@ def __init__(; -794,6 +802,18 @@ def __init__(; symbols: __init__, which_layer, is_linear_attn_layer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -667,16 +667,24 @@ def __init__(
-                config, 'layernorm_linear_attention_alpha', 1)
+                config, 'layernorm_linear_attention_alpha',
+                getattr(config, 'linear_attn_alpha_factor', 1))
-                config, 'layernorm_linear_attention_beta', 1)
+                config, 'layernorm_linear_attention_beta',
+                getattr(config, 'linear_attn_beta_factor', 1))
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +33/-11
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22151 - [V1] [Hybrid] Support Minimax-Text-01 in V1

- Link: https://github.com/vllm-project/vllm/pull/22151
- Status/date: merged / 2025-08-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_text_01.py`; associated commits `6ade99eafa37`
- Diff scope read: GitHub Pull Request files API returned 5 files, +234/-42, 448 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "[V1] [Hybrid] Support Minimax-Text-01 in V1". The diff centers on `vllm/model_executor/models/minimax_text_01.py`. PR body context: # Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". - [x] The test p...
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` modified +152/-40 (192 lines); hunks: -14,8 +14,9; -33,6 +34,9; symbols: jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type, get_state_shape, touching `jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` modified +152/-40 (192 lines); hunks: -14,8 +14,9; -33,6 +34,9; symbols: jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type, get_state_shape
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -14,8 +14,9 @@
+from vllm import envs
-from vllm.config import CacheConfig, VllmConfig
+from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
@@ -33,6 +34,9 @@
+from vllm.model_executor.layers.mamba.abstract import MambaBase
+from vllm.model_executor.layers.mamba.mamba_utils import (
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +152/-40
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/lightning_attn.py`, `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/models/minimax_text_01.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22928 - [V1] [Hybrid] Support using float32 for state in Hybrid Models (Mamba2, Mamba1, Minimax)

- Link: https://github.com/vllm-project/vllm/pull/22928
- Status/date: merged / 2025-08-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_text_01.py`; associated commits `75531a6c1342`
- Diff scope read: GitHub Pull Request files API returned 23 files, +467/-87, 1435 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "[V1] [Hybrid] Support using float32 for state in Hybrid Models (Mamba2, Mamba1, Minimax)". The diff centers on `vllm/model_executor/models/minimax_text_01.py`. PR body context: ## Purpose This PR adds support for customizing the dtype of the state stored for models based on mamba/mamba2/linear attention. It introduces a new CLI argument `mamba_ssm_cach...
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` modified +31/-3 (34 lines); hunks: -16,7 +16,8; -36,7 +37,7; symbols: MiniMaxText01LinearAttention, mamba_type, get_state_dtype, get_state_shape, touching `MiniMaxText01LinearAttention, mamba_type, get_state_dtype`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` modified +31/-3 (34 lines); hunks: -16,7 +16,8; -36,7 +37,7; symbols: MiniMaxText01LinearAttention, mamba_type, get_state_dtype, get_state_shape
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -16,7 +16,8 @@
-from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
+from vllm.config import (CacheConfig, ModelConfig, VllmConfig,
+                         get_current_vllm_config)
@@ -36,7 +37,7 @@
-    MambaStateShapeCalculator)
+    MambaStateDtypeCalculator, MambaStateShapeCalculator)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +31/-3
- Risk and verification: The diff ships test coverage in `tests/models/language/generation/test_hybrid.py`, `tests/v1/worker/test_gpu_model_runner.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22116 - [Bugfix] Fix broken Minimax-01-VL model

- Link: https://github.com/vllm-project/vllm/pull/22116
- Status/date: merged / 2025-08-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_vl_01.py`; associated commits `31fd3265c8b2`
- Diff scope read: GitHub Pull Request files API returned 3 files, +123/-32, 258 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix broken Minimax-01-VL model". The diff centers on `vllm/model_executor/models/minimax_vl_01.py`. PR body context: # Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". - [ ] The test p...
- Key implementation: `vllm/model_executor/models/minimax_vl_01.py` modified +89/-31 (120 lines); hunks: -1,11 +1,13; -17,6 +19,7; symbols: MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, _get_mm_fields_config, touching `MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, _get_mm_fields_config`.
- Code diff details:
  - `vllm/model_executor/models/minimax_vl_01.py` modified +89/-31 (120 lines); hunks: -1,11 +1,13; -17,6 +19,7; symbols: MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, _get_mm_fields_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_vl_01.py
@@ -1,11 +1,13 @@
-from typing import Literal, Optional, TypedDict, Union, cast
+from typing import Annotated, Literal, Optional, Union, cast
+from transformers.models.llava_next.modeling_llava_next import (
+    get_anyres_image_grid_shape, unpad_image)
@@ -17,6 +19,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_vl_01.py` modified +89/-31
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/test_tensor_schema.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22589 - [V1] [Hybrid] Enable compile and piecewise CUDA graph for MiniMax-Text models

- Link: https://github.com/vllm-project/vllm/pull/22589
- Status/date: merged / 2025-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_text_01.py`; associated commits `dd589322801e`
- Diff scope read: GitHub Pull Request files API returned 2 files, +98/-137, 387 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "[V1] [Hybrid] Enable compile and piecewise CUDA graph for MiniMax-Text models". The diff centers on `vllm/model_executor/models/minimax_text_01.py`. PR body context: # Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". - [x] The test p...
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` modified +97/-137 (234 lines); hunks: -1,7 +1,6; -19,13 +18,14; symbols: forward, MiniMaxText01RotaryEmbedding, __init__, _compute_inv_freq, touching `forward, MiniMaxText01RotaryEmbedding, __init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` modified +97/-137 (234 lines); hunks: -1,7 +1,6; -19,13 +18,14; symbols: forward, MiniMaxText01RotaryEmbedding, __init__, _compute_inv_freq
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -1,7 +1,6 @@
-import copy
@@ -19,13 +18,14 @@
+from vllm.compilation.decorators import support_torch_compile
-from vllm.forward_context import get_forward_context
+from vllm.forward_context import ForwardContext, get_forward_context
@@ -43,12 +43,15 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +97/-137
- Risk and verification: Runtime changes concentrate in `vllm/config/compilation.py`, `vllm/model_executor/models/minimax_text_01.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23831 - [V1] [Hybrid] Move MiniMaxLinearAttention into layers/mamba

- Link: https://github.com/vllm-project/vllm/pull/23831
- Status/date: merged / 2025-08-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_text_01.py`; associated commits `4071c76cf3cf`
- Diff scope read: GitHub Pull Request files API returned 2 files, +448/-410, 917 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR changes model-related implementation. Title: "[V1] [Hybrid] Move MiniMaxLinearAttention into layers/mamba". The diff centers on `vllm/model_executor/models/minimax_text_01.py`. PR body context: ## Purpose This PR just moves the `MinimaxLinearAttention` layer into the `layers/mamba` directory, to be consistent with the other mamba-like layers (e.g., mamba1, mamba2, shor...
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` modified +6/-410 (416 lines); hunks: -1,45 +1,37; -50,10 +42,7; symbols: inner_func, MiniMaxText01RMSNormTP, __init__, weight_loader, touching `inner_func, MiniMaxText01RMSNormTP, __init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` modified +6/-410 (416 lines); hunks: -1,45 +1,37; -50,10 +42,7; symbols: inner_func, MiniMaxText01RMSNormTP, __init__, weight_loader
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -1,45 +1,37 @@
-import math
-    from vllm.attention.backends.abstract import AttentionBackend
+    pass
-import torch.nn.functional as F
-from einops import rearrange
-from vllm.config import (CacheConfig, ModelConfig, VllmConfig,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +6/-410
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/models/minimax_text_01.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27535 - [Model][MiniMax-M2] Support MiniMax-M2 Model

- Link: https://github.com/vllm-project/vllm/pull/27535
- Status/date: merged / 2025-10-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`; associated commits `720af6ab7911`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +1306/-0, 1347 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "[Model][MiniMax-M2] Support MiniMax-M2 Model". The diff centers on `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`. PR body context: ## Purpose This PR is intended to support the MiniMax-M2 model inference. It has been tested on a single machine with both 8x96GB and 4x96GB GPU configurations. ## Deploy Deploy...
- Key implementation: `vllm/model_executor/models/minimax_m2.py` added +585/-0 (585 lines); hunks: -0,0 +1,585; symbols: MiniMaxM2MoE, __init__, ebias_weight_loader, forward, touching `MiniMaxM2MoE, __init__, ebias_weight_loader`; `vllm/reasoning/minimax_m2_reasoning_parser.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: MiniMaxM2ReasoningParser, start_token, end_token, MiniMaxM2AppendThinkReasoningParser, touching `MiniMaxM2ReasoningParser, start_token, end_token`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` added +585/-0 (585 lines); hunks: -0,0 +1,585; symbols: MiniMaxM2MoE, __init__, ebias_weight_loader, forward
  - `vllm/reasoning/minimax_m2_reasoning_parser.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: MiniMaxM2ReasoningParser, start_token, end_token, MiniMaxM2AppendThinkReasoningParser
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -0,0 +1,585 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from
+# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
+# Copyright 2023 The vLLM team.
+# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
diff -- vllm/reasoning/minimax_m2_reasoning_parser.py
@@ -0,0 +1,69 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Sequence
+from vllm.entrypoints.openai.protocol import (
+    ChatCompletionRequest,
+    DeltaMessage,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` added +585/-0; `vllm/reasoning/minimax_m2_reasoning_parser.py` added +69/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27537 - Fix MiniMax-M2 copyright

- Link: https://github.com/vllm-project/vllm/pull/27537
- Status/date: merged / 2025-10-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `5980604c44d8`
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-3, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix MiniMax-M2 copyright". The diff centers on `vllm/model_executor/models/minimax_m2.py`. PR body context: ## Purpose Fix MiniMax-M2 copyright. ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, such as "Fix some i...
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +2/-3 (5 lines); hunks: -1,10 +1,9.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +2/-3 (5 lines); hunks: -1,10 +1,9
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -1,10 +1,9 @@
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
+# Copyright 2025 The MiniMax AI team.
-# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +2/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27627 - Fix MiniMax-M2 rmsnorm precision and remove useless code

- Link: https://github.com/vllm-project/vllm/pull/27627
- Status/date: merged / 2025-10-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `d6704dd099b7`
- Diff scope read: GitHub Pull Request files API returned 2 files, +1/-19, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix MiniMax-M2 rmsnorm precision and remove useless code". The diff centers on `vllm/model_executor/models/minimax_m2.py`. PR body context: ## Purpose - Fix rmsnorm precision - Remove useless code ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR,...
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +0/-18 (18 lines); hunks: -263,23 +263,6 @@ def __init__(; -288,7 +271,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +0/-18 (18 lines); hunks: -263,23 +263,6 @@ def __init__(; -288,7 +271,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -263,23 +263,6 @@ def __init__(
-        # TODO: support MTP
-        attn_window_size = getattr(config, "attn_window_size", None)
-        if attn_window_size is not None:
-            if isinstance(attn_window_size, list):
-                attn_window_size = attn_window_size[layer_idx]
-            elif isinstance(attn_window_size, int):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +0/-18
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30384 - [BugFix] Fix minimax m2 model rotary_dim

- Link: https://github.com/vllm-project/vllm/pull/30384
- Status/date: merged / 2025-12-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `d017bceb08ea`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[BugFix] Fix minimax m2 model rotary_dim". The diff centers on `vllm/model_executor/models/minimax_m2.py`. PR body context: ## Purpose After #29966, get_rope always reads partial_rotary_factor from the configuration and performs the multiplication again, even if rotary_dim has already been scaled. Th...
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -201,7 +201,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -201,7 +201,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -201,7 +201,7 @@ def __init__(
-            rotary_dim=rotary_dim,
+            rotary_dim=self.head_dim,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29882 - [bugfix] fix MiniMaxM2ReasoningParser streaming output not separating reasoning_content.

- Link: https://github.com/vllm-project/vllm/pull/29882
- Status/date: merged / 2025-12-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_minimax_m2_append_reasoning_parser.py`, `tests/reasoning/test_minimax_m2_reasoning_parser.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`; associated commits `6299628d326f`
- Diff scope read: GitHub Pull Request files API returned 3 files, +468/-0, 484 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[bugfix] fix MiniMaxM2ReasoningParser streaming output not separating reasoning_content.". The diff centers on `tests/reasoning/test_minimax_m2_reasoning_parser.py`, `tests/reasoning/test_minimax_m2_append_reasoning_parser.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`. PR body context: ## Summary Fix `minimax_m2` reasoning parser not separating `content` and `reasoning_content` in streaming mode. ## Problem MiniMax-M2 models don't generate ` ` start token, onl...
- Key implementation: `tests/reasoning/test_minimax_m2_reasoning_parser.py` added +230/-0 (230 lines); hunks: -0,0 +1,230; symbols: minimax_m2_tokenizer, test_reasoning, touching `minimax_m2_tokenizer, test_reasoning`; `tests/reasoning/test_minimax_m2_append_reasoning_parser.py` added +195/-0 (195 lines); hunks: -0,0 +1,195; symbols: minimax_m2_tokenizer, test_reasoning, touching `minimax_m2_tokenizer, test_reasoning`; `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +43/-0 (43 lines); hunks: -19,6 +19,10; -31,6 +35,45 @@ def end_token(self) -> str:; symbols: MiniMaxM2ReasoningParser, end_token, extract_reasoning_streaming, MiniMaxM2AppendThinkReasoningParser, touching `MiniMaxM2ReasoningParser, end_token, extract_reasoning_streaming`.
- Code diff details:
  - `tests/reasoning/test_minimax_m2_reasoning_parser.py` added +230/-0 (230 lines); hunks: -0,0 +1,230; symbols: minimax_m2_tokenizer, test_reasoning
  - `tests/reasoning/test_minimax_m2_append_reasoning_parser.py` added +195/-0 (195 lines); hunks: -0,0 +1,195; symbols: minimax_m2_tokenizer, test_reasoning
  - `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +43/-0 (43 lines); hunks: -19,6 +19,10; -31,6 +35,45 @@ def end_token(self) -> str:; symbols: MiniMaxM2ReasoningParser, end_token, extract_reasoning_streaming, MiniMaxM2AppendThinkReasoningParser
- Key code excerpts:

```diff
diff -- tests/reasoning/test_minimax_m2_reasoning_parser.py
@@ -0,0 +1,230 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from transformers import AutoTokenizer
+from tests.reasoning.utils import run_reasoning_extraction
+from vllm.reasoning import ReasoningParser, ReasoningParserManager
diff -- tests/reasoning/test_minimax_m2_append_reasoning_parser.py
@@ -0,0 +1,195 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from transformers import AutoTokenizer
+from tests.reasoning.utils import run_reasoning_extraction
+from vllm.reasoning import ReasoningParser, ReasoningParserManager
diff -- vllm/reasoning/minimax_m2_reasoning_parser.py
@@ -19,6 +19,10 @@
```

- Reviewed files:
  - tests: `tests/reasoning/test_minimax_m2_reasoning_parser.py` added +230/-0; `tests/reasoning/test_minimax_m2_append_reasoning_parser.py` added +195/-0
  - runtime: `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +43/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_minimax_m2_append_reasoning_parser.py`, `tests/reasoning/test_minimax_m2_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30555 - [Bugfix][Frontend] Prevent IndexError in MiniMax M2 tool parser during streaming extraction

- Link: https://github.com/vllm-project/vllm/pull/30555
- Status/date: merged / 2025-12-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/minimax_m2_tool_parser.py`; associated commits `20fda431515d`
- Diff scope read: GitHub Pull Request files API returned 2 files, +137/-4, 186 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "[Bugfix][Frontend] Prevent IndexError in MiniMax M2 tool parser during streaming extraction". The diff centers on `tests/tool_use/test_minimax_m2_tool_parser.py`, `vllm/tool_parsers/minimax_m2_tool_parser.py`. PR body context: Fix https://github.com/vllm-project/vllm/issues/30554. Prevent IndexError in MiniMax M2 tool parser during streaming extraction. Add args str to streamed_args_for_tool field.
- Key implementation: `tests/tool_use/test_minimax_m2_tool_parser.py` added +119/-0 (119 lines); hunks: -0,0 +1,119; symbols: FakeTokenizer, __init__, get_vocab, minimax_m2_tool_parser, touching `FakeTokenizer, __init__, get_vocab`; `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +18/-4 (22 lines); hunks: -122,6 +122,8 @@ def _reset_streaming_state(self):; -421,9 +423,12 @@ def extract_tool_calls_streaming(; symbols: _reset_streaming_state, _extract_name, extract_tool_calls_streaming, touching `_reset_streaming_state, _extract_name, extract_tool_calls_streaming`.
- Code diff details:
  - `tests/tool_use/test_minimax_m2_tool_parser.py` added +119/-0 (119 lines); hunks: -0,0 +1,119; symbols: FakeTokenizer, __init__, get_vocab, minimax_m2_tool_parser
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +18/-4 (22 lines); hunks: -122,6 +122,8 @@ def _reset_streaming_state(self):; -421,9 +423,12 @@ def extract_tool_calls_streaming(; symbols: _reset_streaming_state, _extract_name, extract_tool_calls_streaming
- Key code excerpts:

```diff
diff -- tests/tool_use/test_minimax_m2_tool_parser.py
@@ -0,0 +1,119 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+import pytest
+from vllm.tool_parsers.minimax_m2_tool_parser import (
+    MinimaxM2ToolParser,
diff -- vllm/tool_parsers/minimax_m2_tool_parser.py
@@ -122,6 +122,8 @@ def _reset_streaming_state(self):
+        # Reset streamed args tracking
+        self.streamed_args_for_tool.clear()
@@ -421,9 +423,12 @@ def extract_tool_calls_streaming(
-                                "arguments": "{}",  # Placeholder, will be updated later
+                                "arguments": {},  # Placeholder, will be updated later
+                        # Initialize streamed_args_for_tool for this tool call
```

- Reviewed files:
  - tests: `tests/tool_use/test_minimax_m2_tool_parser.py` added +119/-0
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +18/-4
- Risk and verification: The diff ships test coverage in `tests/tool_use/test_minimax_m2_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31083 - Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs

- Link: https://github.com/vllm-project/vllm/pull/31083
- Status/date: merged / 2025-12-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/minimax_m2_tool_parser.py`; associated commits `c02a2705f9ce`
- Diff scope read: GitHub Pull Request files API returned 2 files, +167/-48, 257 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs". The diff centers on `vllm/tool_parsers/minimax_m2_tool_parser.py`. PR body context: ## Purpose The MiniMax-M2 tool call parser supports `anyOf`, `oneOf`, `allOf`, type arrays, and enum fields. And add MiniMax-M2.1 to the documentation. ## Test Plan ## Test Resu...
- Key implementation: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +166/-47 (213 lines); hunks: -138,37 +138,167 @@ def _extract_name(self, name_str: str) -> str:; -207,17 +337,11 @@ def _parse_single_invoke(; symbols: _extract_name, _convert_param_value, _extract_types_from_schema, _convert_param_value_with_types, touching `_extract_name, _convert_param_value, _extract_types_from_schema`.
- Code diff details:
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +166/-47 (213 lines); hunks: -138,37 +138,167 @@ def _extract_name(self, name_str: str) -> str:; -207,17 +337,11 @@ def _parse_single_invoke(; symbols: _extract_name, _convert_param_value, _extract_types_from_schema, _convert_param_value_with_types
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/minimax_m2_tool_parser.py
@@ -138,37 +138,167 @@ def _extract_name(self, name_str: str) -> str:
-        """Convert parameter value to the correct type."""
+        """Convert parameter value to the correct type (legacy single-type version)."""
+        return self._convert_param_value_with_types(value, [param_type])
+    def _extract_types_from_schema(self, schema: Any) -> list[str]:
+        """
+        Extract all possible types from a JSON schema definition.
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +166/-47
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/minimax_m2_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31493 - Optimize QKNorm for MiniMax-M2/M2.1

- Link: https://github.com/vllm-project/vllm/pull/31493
- Status/date: merged / 2025-12-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `5bc664110f12`
- Diff scope read: GitHub Pull Request files API returned 2 files, +25/-2, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR optimizes an inference path or backend selection. Title: "Optimize QKNorm for MiniMax-M2/M2.1". The diff centers on `vllm/model_executor/models/minimax_m2.py`. PR body context: ## Purpose Fuses QKNorm in tensor-parallel mode so that only a single all_reduce is launched, reducing communication overhead. Part of #31478. ## Test Plan Benchmarks were condu...
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +3/-2 (5 lines); hunks: -234,8 +234,9 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +3/-2 (5 lines); hunks: -234,8 +234,9 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -234,8 +234,9 @@ def forward(
-        q = self.q_norm(q)
-        k = self.k_norm(k)
+        q, k = MiniMaxText01RMSNormTP.forward_qk(
+            self.q_norm, self.k_norm, q.contiguous(), k.contiguous()
+        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32342 - Fix optional parameter parsing in MiniMax M2 tool parser #32278

- Link: https://github.com/vllm-project/vllm/pull/32342
- Status/date: merged / 2026-01-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/minimax_m2_tool_parser.py`; associated commits `19b251fe3d26`
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-5, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix optional parameter parsing in MiniMax M2 tool parser #32278". The diff centers on `vllm/tool_parsers/minimax_m2_tool_parser.py`. PR body context: ## Purpose #32278 ## Test Plan ## Test Result main branch: "logprobs":null, \"logfile\": \"/tmp/foo\"}" --- Essential Elements of an Effective PR Description Checklist - [ ] The...
- Key implementation: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-5 (7 lines); hunks: -217,16 +217,13 @@ def _convert_param_value_with_types(; symbols: _convert_param_value_with_types, touching `_convert_param_value_with_types`.
- Code diff details:
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-5 (7 lines); hunks: -217,16 +217,13 @@ def _convert_param_value_with_types(; symbols: _convert_param_value_with_types
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/minimax_m2_tool_parser.py
@@ -217,16 +217,13 @@ def _convert_param_value_with_types(
-        if value.lower() == "null":
+        # Check if the VALUE itself indicates null (not just if null is allowed)
+        if value.lower() in ("null", "none", "nil"):
-        # Try null first if it's in the list
-        if "null" in normalized_types or value.lower() in ("null", "none", "nil"):
-            return None
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-5
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/minimax_m2_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32763 - feat: Complete LoRA support for MiniMaxM2 Fixes #32736

- Link: https://github.com/vllm-project/vllm/pull/32763
- Status/date: merged / 2026-01-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `bc0d291bfebf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-3, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "feat: Complete LoRA support for MiniMaxM2 Fixes #32736". The diff centers on `vllm/model_executor/models/minimax_m2.py`. PR body context: FIX #32736 This PR builds upon #31703 by adding the missing components to enable full LoRA support for MiniMaxM2ForCausalLM. Changes: - Add SupportsLoRA interface to MiniMaxM2Fo...
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +10/-2 (12 lines); hunks: -59,7 +59,7; -487,7 +487,15 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, MiniMaxM2ForCausalLM, __init__, touching `load_weights, MiniMaxM2ForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +10/-2 (12 lines); hunks: -59,7 +59,7; -487,7 +487,15 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, MiniMaxM2ForCausalLM, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -59,7 +59,7 @@
-from .interfaces import SupportsPP
+from .interfaces import SupportsLoRA, SupportsPP
@@ -487,7 +487,15 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-class MiniMaxM2ForCausalLM(nn.Module, SupportsPP):
+class MiniMaxM2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
+    packed_modules_mapping = {
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +10/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35352 - [Bug] Fix missing tag after tool call in MiniMax 2.1

- Link: https://github.com/vllm-project/vllm/pull/35352
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/reasoning/minimax_m2_reasoning_parser.py`; associated commits `7fea7250a46c`
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-1, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bug] Fix missing tag after tool call in MiniMax 2.1". The diff centers on `vllm/reasoning/minimax_m2_reasoning_parser.py`. PR body context: Fix: #35349 ## Purpose This PR improves the reasoning end detection logic in MiniMaxM2AppendThinkReasoningParser.is_reasoning_end. After a successful tool call, the following li...
- Key implementation: `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +6/-1 (7 lines); hunks: -87,10 +87,15 @@ class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):; symbols: MiniMaxM2AppendThinkReasoningParser, __init__, is_reasoning_end, extract_content_ids, touching `MiniMaxM2AppendThinkReasoningParser, __init__, is_reasoning_end`.
- Code diff details:
  - `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +6/-1 (7 lines); hunks: -87,10 +87,15 @@ class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):; symbols: MiniMaxM2AppendThinkReasoningParser, __init__, is_reasoning_end, extract_content_ids
- Key code excerpts:

```diff
diff -- vllm/reasoning/minimax_m2_reasoning_parser.py
@@ -87,10 +87,15 @@ class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):
+        self.start_token_id = self.vocab.get("<think>")
-        return any(input_id == end_token_id for input_id in reversed(input_ids))
+        start_token_id = self.start_token_id
+        for input_id in reversed(input_ids):
+            if input_id in (end_token_id, start_token_id):
+                return input_id == end_token_id
```

- Reviewed files:
  - runtime: `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `vllm/reasoning/minimax_m2_reasoning_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35895 - [Bugfix] Fix minimax_m2 tool parser when stream interval > 1

- Link: https://github.com/vllm-project/vllm/pull/35895
- Status/date: merged / 2026-03-12
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `vllm/tool_parsers/minimax_m2_tool_parser.py`; associated commits `8647c6cf510b`
- Diff scope read: GitHub Pull Request files API returned 3 files, +534/-532, 1119 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix minimax_m2 tool parser when stream interval > 1". The diff centers on `vllm/tool_parsers/minimax_m2_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `tests/tool_use/test_minimax_m2_tool_parser.py`. PR body context: ## Purpose Fix MiniMax M2 tool parser dropping arguments when stream_interval > 1. ### Background MiniMax M2 uses XML-based tool call format: The old parser used an incremental...
- Key implementation: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +90/-413 (503 lines); hunks: -37,37 +37,10 @@ def __init__(self, tokenizer: TokenizerLike):; -103,46 +76,15 @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, type, _generate_tool_call_id, _reset_streaming_state, touching `__init__, type, _generate_tool_call_id`; `tests/tool_parsers/test_minimax_m2_tool_parser.py` added +444/-0 (444 lines); hunks: -0,0 +1,444; symbols: FakeTokenizer, __init__, get_vocab, parser, touching `FakeTokenizer, __init__, get_vocab`; `tests/tool_use/test_minimax_m2_tool_parser.py` removed +0/-119 (119 lines); hunks: -1,119 +0,0; symbols: FakeTokenizer, __init__, get_vocab, minimax_m2_tool_parser, touching `FakeTokenizer, __init__, get_vocab`.
- Code diff details:
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +90/-413 (503 lines); hunks: -37,37 +37,10 @@ def __init__(self, tokenizer: TokenizerLike):; -103,46 +76,15 @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, type, _generate_tool_call_id, _reset_streaming_state
  - `tests/tool_parsers/test_minimax_m2_tool_parser.py` added +444/-0 (444 lines); hunks: -0,0 +1,444; symbols: FakeTokenizer, __init__, get_vocab, parser
  - `tests/tool_use/test_minimax_m2_tool_parser.py` removed +0/-119 (119 lines); hunks: -1,119 +0,0; symbols: FakeTokenizer, __init__, get_vocab, minimax_m2_tool_parser
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/minimax_m2_tool_parser.py
@@ -37,37 +37,10 @@ def __init__(self, tokenizer: TokenizerLike):
-        self.invoke_start_prefix: str = "<invoke name="
-        self.invoke_end_token: str = "</invoke>"
-        self.parameter_prefix: str = "<parameter name="
-        self.parameter_end_token: str = "</parameter>"
-        # Streaming state variables
-        self.current_tool_name_sent: bool = False
diff -- tests/tool_parsers/test_minimax_m2_tool_parser.py
@@ -0,0 +1,444 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+import pytest
+from vllm.tool_parsers.minimax_m2_tool_parser import (
+    MinimaxM2ToolParser,
diff -- tests/tool_use/test_minimax_m2_tool_parser.py
@@ -1,119 +0,0 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +90/-413
  - tests: `tests/tool_parsers/test_minimax_m2_tool_parser.py` added +444/-0; `tests/tool_use/test_minimax_m2_tool_parser.py` removed +0/-119
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `tests/tool_use/test_minimax_m2_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37371 - standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01

- Link: https://github.com/vllm-project/vllm/pull/37371
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_text_01.py`; associated commits `17808394bc48`
- Diff scope read: GitHub Pull Request files API returned 2 files, +235/-219, 527 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR fixes a launch, loading, parsing, or numerical issue. Title: "standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01". The diff centers on `vllm/model_executor/models/minimax_text_01.py`. PR body context: FIX (partial) #15697 ## Test Plan Verified the refactor with a mock-weight loading script using a "Tiny Model Hack" (reducing layers to 1 for fast validation): Mock Weights: Gen...
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` modified +138/-131 (269 lines); hunks: -52,7 +52,12; -494,6 +499,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: replace_weight_name, __init__, _clear_prefill_cache, embed_input_ids, touching `replace_weight_name, __init__, _clear_prefill_cache`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` modified +138/-131 (269 lines); hunks: -52,7 +52,12; -494,6 +499,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: replace_weight_name, __init__, _clear_prefill_cache, embed_input_ids
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -52,7 +52,12 @@
-from .utils import PPMissingLayer, is_pp_missing_parameter, make_layers
+from .utils import (
+    AutoWeightsLoader,
+    PPMissingLayer,
+    is_pp_missing_parameter,
+    make_layers,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +138/-131
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/models/minimax_text_01.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37214 - Fix minimax m2.5 nvfp4 kv scales weight loading

- Link: https://github.com/vllm-project/vllm/pull/37214
- Status/date: merged / 2026-03-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `74056039b776`
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-0, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix minimax m2.5 nvfp4 kv scales weight loading". The diff centers on `vllm/model_executor/models/minimax_m2.py`. PR body context: ## Purpose Fix kv scale weight loading issue in minimax m2.5 nvfp4: ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose...
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +11/-0 (11 lines); hunks: -439,6 +439,17 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +11/-0 (11 lines); hunks: -439,6 +439,17 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -439,6 +439,17 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+                # Remap qkv_proj.[kv]_scale to attn.[kv]_scale
+                if name.endswith((".k_scale", ".v_scale")):
+                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
+                    if remapped_name is not None and remapped_name in params_dict:
+                        param = params_dict[remapped_name]
+                        weight_loader = getattr(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +11/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36965 - [Model][Quantization] Add GGUF support for MiniMax-M2.1

- Link: https://github.com/vllm-project/vllm/pull/36965
- Status/date: merged / 2026-03-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `63babd17f1b1`
- Diff scope read: GitHub Pull Request files API returned 5 files, +137/-10, 238 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "[Model][Quantization] Add GGUF support for MiniMax-M2.1". The diff centers on `vllm/model_executor/models/minimax_m2.py`. PR body context: ## Purpose FIX #28724 Add GGUF loading support for MiniMax-M2.1 (456B MoE, 45.9B active, 256 experts, 8 active per token). This enables serving GGUF-quantized MiniMax-M2.1 check...
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +5/-2 (7 lines); hunks: -331,7 +331,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -518,7 +518,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +5/-2 (7 lines); hunks: -331,7 +331,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -518,7 +518,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -331,7 +331,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-                quant_config=None,
+                quant_config=quant_config,
@@ -518,7 +518,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-                config.vocab_size, config.hidden_size, quant_config=None
+                config.vocab_size,
+                config.hidden_size,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/config/model.py`, `vllm/model_executor/layers/quantization/gguf.py`, `vllm/model_executor/model_loader/gguf_loader.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37512 - MiniMax-M2: add Eagle3 speculative decoding support

- Link: https://github.com/vllm-project/vllm/pull/37512
- Status/date: merged / 2026-04-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `f6983f01de2b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +24/-5, 99 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR adds or enables a model support/runtime surface. Title: "MiniMax-M2: add Eagle3 speculative decoding support". The diff centers on `vllm/model_executor/models/minimax_m2.py`. PR body context: - Add SupportsEagle3 interface to MiniMaxM2ForCausalLM with aux hidden state collection in MiniMaxM2Model.forward() - Register Eagle3MiniMaxM2ForCausalLM in speculative decoding...
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +16/-5 (21 lines); hunks: -24,6 +24,7; -59,7 +60,7; symbols: forward, MiniMaxM2Model, __init__, touching `forward, MiniMaxM2Model, __init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +16/-5 (21 lines); hunks: -24,6 +24,7; -59,7 +60,7; symbols: forward, MiniMaxM2Model, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -24,6 +24,7 @@
+from itertools import islice
@@ -59,7 +60,7 @@
-from .interfaces import SupportsLoRA, SupportsPP
+from .interfaces import EagleModelMixin, SupportsEagle3, SupportsLoRA, SupportsPP
@@ -313,7 +314,7 @@ def forward(
-class MiniMaxM2Model(nn.Module):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +16/-5
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37045 - [Kernel] Porting the TRTLLM minimax_allreduce_rms kernels

- Link: https://github.com/vllm-project/vllm/pull/37045
- Status/date: merged / 2026-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/core/test_minimax_reduce_rms.py`, `vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py`, `vllm/model_executor/models/minimax_m2.py`; associated commits `ecd1ea13634e`
- Diff scope read: GitHub Pull Request files API returned 14 files, +1861/-4, 1936 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR optimizes an inference path or backend selection. Title: "[Kernel] Porting the TRTLLM minimax_allreduce_rms kernels". The diff centers on `vllm/model_executor/models/minimax_m2.py`, `vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py`, `tests/kernels/core/test_minimax_reduce_rms.py`. PR body context: ## Purpose See: https://github.com/NVIDIA/TensorRT-LLM/pull/12163 ## Plan - [x] Clean up the code - [x] Verify the accuracy on GSM8K Benchmark - [x] Performance Benchmark - [x]...
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +1/-3 (4 lines); hunks: -233,9 +233,7 @@ def forward(; symbols: forward, touching `forward`; `vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py` added +340/-0 (340 lines); hunks: -0,0 +1,340; symbols: _minimax_qk_norm_fused, _minimax_qk_norm_fused_fake, MiniMaxQKNormPattern, __init__, touching `_minimax_qk_norm_fused, _minimax_qk_norm_fused_fake, MiniMaxQKNormPattern`; `tests/kernels/core/test_minimax_reduce_rms.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: _worker_forward_qk, test_minimax_reduce_rms_qk, touching `_worker_forward_qk, test_minimax_reduce_rms_qk`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +1/-3 (4 lines); hunks: -233,9 +233,7 @@ def forward(; symbols: forward
  - `vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py` added +340/-0 (340 lines); hunks: -0,0 +1,340; symbols: _minimax_qk_norm_fused, _minimax_qk_norm_fused_fake, MiniMaxQKNormPattern, __init__
  - `tests/kernels/core/test_minimax_reduce_rms.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: _worker_forward_qk, test_minimax_reduce_rms_qk
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -233,9 +233,7 @@ def forward(
-        q, k = MiniMaxText01RMSNormTP.forward_qk(
-            self.q_norm, self.k_norm, q.contiguous(), k.contiguous()
-        )
+        q, k = MiniMaxText01RMSNormTP.forward_qk(self.q_norm, self.k_norm, q, k)
diff -- vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py
@@ -0,0 +1,340 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Fusion pass: replace MiniMax QK allreduce + RMS norm with the Lamport
+fused kernel (minimax_allreduce_rms_qk) for decode-size batches.
+Pattern (inlined forward_qk in compiled graph):
diff -- tests/kernels/core/test_minimax_reduce_rms.py
@@ -0,0 +1,152 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +1/-3; `vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py` added +340/-0
  - tests: `tests/kernels/core/test_minimax_reduce_rms.py` added +152/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_minimax_reduce_rms.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39683 - [Bugfix]: Fix MinimaxM2ToolParser missing tools parameter

- Link: https://github.com/vllm-project/vllm/pull/39683
- Status/date: merged / 2026-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/parser/minimax_m2_parser.py`; associated commits `d2130a47bb97`
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-2, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix]: Fix MinimaxM2ToolParser missing tools parameter". The diff centers on `vllm/parser/minimax_m2_parser.py`. PR body context: ## Purpose [Bugfix]: Fix MinimaxM2ToolParser missing tools parameter ## Test Plan before ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The...
- Key implementation: `vllm/parser/minimax_m2_parser.py` modified +5/-2 (7 lines); hunks: -13,6 +13,9; -40,12 +43,12 @@ class MiniMaxM2Parser(DelegatingParser):; symbols: MiniMaxM2Parser, __init__, touching `MiniMaxM2Parser, __init__`.
- Code diff details:
  - `vllm/parser/minimax_m2_parser.py` modified +5/-2 (7 lines); hunks: -13,6 +13,9; -40,12 +43,12 @@ class MiniMaxM2Parser(DelegatingParser):; symbols: MiniMaxM2Parser, __init__
- Key code excerpts:

```diff
diff -- vllm/parser/minimax_m2_parser.py
@@ -13,6 +13,9 @@
+from vllm.tool_parsers.abstract_tool_parser import (
+    Tool,
+)
@@ -40,12 +43,12 @@ class MiniMaxM2Parser(DelegatingParser):
-    def __init__(self, tokenizer: TokenizerLike):
+    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
```

- Reviewed files:
  - runtime: `vllm/parser/minimax_m2_parser.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/parser/minimax_m2_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39861 - [Bugfix] Accept **kwargs in MiniMaxM2Parser.__init__()

- Link: https://github.com/vllm-project/vllm/pull/39861
- Status/date: merged / 2026-04-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/parser/minimax_m2_parser.py`; associated commits `8d7c9628337a`
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-3, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiniMax M2 Series, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Accept **kwargs in MiniMaxM2Parser.__init__()". The diff centers on `vllm/parser/minimax_m2_parser.py`. PR body context: Fixes #39847. ### Summary `MiniMaxM2Parser.__init__()` does not accept `**kwargs`, causing a `TypeError` on every streaming chat completion request when serving MiniMax M2 model...
- Key implementation: `vllm/parser/minimax_m2_parser.py` modified +9/-3 (12 lines); hunks: -43,11 +43,17 @@ class MiniMaxM2Parser(DelegatingParser):; symbols: MiniMaxM2Parser, __init__, touching `MiniMaxM2Parser, __init__`.
- Code diff details:
  - `vllm/parser/minimax_m2_parser.py` modified +9/-3 (12 lines); hunks: -43,11 +43,17 @@ class MiniMaxM2Parser(DelegatingParser):; symbols: MiniMaxM2Parser, __init__
- Key code excerpts:

```diff
diff -- vllm/parser/minimax_m2_parser.py
@@ -43,11 +43,17 @@ class MiniMaxM2Parser(DelegatingParser):
-    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
-        super().__init__(tokenizer)
+    def __init__(
+        self,
+        tokenizer: TokenizerLike,
+        tools: list[Tool] | None = None,
```

- Reviewed files:
  - runtime: `vllm/parser/minimax_m2_parser.py` modified +9/-3
- Risk and verification: Runtime changes concentrate in `vllm/parser/minimax_m2_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- This version rejects title-only PR lists; every PR must include trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
