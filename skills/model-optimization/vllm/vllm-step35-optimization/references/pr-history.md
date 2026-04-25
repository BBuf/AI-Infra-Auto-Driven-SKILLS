# vllm Step 3.5 PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 1
- Rule: use this as the backing dossier for the skill, not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/reasoning/test_step3p5_reasoning_parser.py` | [#34211](https://github.com/vllm-project/vllm/pull/34211) |
| `tests/tool_parsers/test_step3p5_tool_parser.py` | [#33690](https://github.com/vllm-project/vllm/pull/33690) |
| `vllm/model_executor/models/step3_text.py` | no direct PR-number commit |
| `vllm/model_executor/models/step3_vl.py` | no direct PR-number commit |
| `vllm/model_executor/models/step3p5.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523), [#33755](https://github.com/vllm-project/vllm/pull/33755), [#34478](https://github.com/vllm-project/vllm/pull/34478) |
| `vllm/model_executor/models/step3p5_mtp.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523) |
| `vllm/reasoning/step3p5_reasoning_parser.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523), [#34211](https://github.com/vllm-project/vllm/pull/34211) |
| `vllm/tool_parsers/step3p5_tool_parser.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523), [#33690](https://github.com/vllm-project/vllm/pull/33690) |
| `vllm/transformers_utils/configs/step3_vl.py` | no direct PR-number commit |
| `vllm/transformers_utils/configs/step3p5.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523) |
| `vllm/transformers_utils/processors/step3_vl.py` | no direct PR-number commit |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-02-02 | [#33523](https://github.com/vllm-project/vllm/pull/33523) | merged | [Models] Step-3.5-Flash | `vllm/tool_parsers/step3p5_tool_parser.py`, `vllm/model_executor/models/step3p5.py`, `vllm/model_executor/models/step3p5_mtp.py` |
| 2026-02-05 | [#33690](https://github.com/vllm-project/vllm/pull/33690) | merged | [Bugfix] Fix step3p5 parser when using mtp | `tests/tool_parsers/test_step3p5_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py` |
| 2026-02-07 | [#33755](https://github.com/vllm-project/vllm/pull/33755) | merged | [Model] Enable Step3p5ForCausalLM testing | `vllm/model_executor/models/step3p5.py` |
| 2026-02-22 | [#34478](https://github.com/vllm-project/vllm/pull/34478) | merged | [Model] Add NVFP4 quantization support for Step3.5-Flash | `vllm/model_executor/models/step3p5.py` |
| 2026-02-25 | [#34211](https://github.com/vllm-project/vllm/pull/34211) | merged | [Bugfix] Fix step3p5 reasoning with interleaved thinking | `tests/reasoning/test_step3p5_reasoning_parser.py`, `vllm/reasoning/step3p5_reasoning_parser.py` |
| 2026-03-20 | [#37579](https://github.com/vllm-project/vllm/pull/37579) | merged | [Model] Refactor Step3-VL processor to HF style | `vllm/transformers_utils/processors/step3_vl.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/transformers_utils/processors/internvl.py` |

## Per-PR Diff Audit Cards

### PR #33523 - [Models] Step-3.5-Flash

- Link: https://github.com/vllm-project/vllm/pull/33523
- Status/date: merged / 2026-02-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/step3p5.py`, `vllm/model_executor/models/step3p5_mtp.py`, `vllm/reasoning/step3p5_reasoning_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`, `vllm/transformers_utils/configs/step3p5.py`; associated commits `c3b40dc3e74d`
- Diff scope read: GitHub Pull Request files API returned 18 files, +3107/-4, 3270 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Step 3.5, this PR adds or enables a model support/runtime surface. Title: "[Models] Step-3.5-Flash". The diff centers on `vllm/tool_parsers/step3p5_tool_parser.py`, `vllm/model_executor/models/step3p5.py`, `vllm/model_executor/models/step3p5_mtp.py`. PR body context: ## Purpose Step-3.5-Flash model support. ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, such as "Fix so...
- Key implementation: `vllm/tool_parsers/step3p5_tool_parser.py` added +1511/-0 (1511 lines); hunks: -0,0 +1,1511; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks, touching `StreamingXMLToolCallParser, __init__, reset_streaming_state`; `vllm/model_executor/models/step3p5.py` added +894/-0 (894 lines); hunks: -0,0 +1,894; symbols: FP32ReplicatedLinear, forward, Step3p5MLP, __init__, touching `FP32ReplicatedLinear, forward, Step3p5MLP`; `vllm/model_executor/models/step3p5_mtp.py` added +315/-0 (315 lines); hunks: -0,0 +1,315; symbols: SharedHead, __init__, forward, Step3p5AMultiTokenPredictorLayer, touching `SharedHead, __init__, forward`; `vllm/reasoning/step3p5_reasoning_parser.py` added +153/-0 (153 lines); hunks: -0,0 +1,153; symbols: Step3p5ReasoningParser, start_token, end_token, __init__, touching `Step3p5ReasoningParser, start_token, end_token`.
- Code diff details:
  - `vllm/tool_parsers/step3p5_tool_parser.py` added +1511/-0 (1511 lines); hunks: -0,0 +1,1511; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks
  - `vllm/model_executor/models/step3p5.py` added +894/-0 (894 lines); hunks: -0,0 +1,894; symbols: FP32ReplicatedLinear, forward, Step3p5MLP, __init__
  - `vllm/model_executor/models/step3p5_mtp.py` added +315/-0 (315 lines); hunks: -0,0 +1,315; symbols: SharedHead, __init__, forward, Step3p5AMultiTokenPredictorLayer
  - `vllm/reasoning/step3p5_reasoning_parser.py` added +153/-0 (153 lines); hunks: -0,0 +1,153; symbols: Step3p5ReasoningParser, start_token, end_token, __init__
  - `vllm/transformers_utils/configs/step3p5.py` added +100/-0 (100 lines); hunks: -0,0 +1,100; symbols: Step3p5Config, __init__
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/step3p5_tool_parser.py
@@ -0,0 +1,1511 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import ast
+import json
+from collections.abc import Sequence
+from typing import Any
diff -- vllm/model_executor/models/step3p5.py
@@ -0,0 +1,894 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only Jurassic model."""
+from collections.abc import Iterable
+from typing import Any
+import torch
diff -- vllm/model_executor/models/step3p5_mtp.py
@@ -0,0 +1,315 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/step3p5_tool_parser.py` added +1511/-0; `vllm/model_executor/models/step3p5.py` added +894/-0; `vllm/model_executor/models/step3p5_mtp.py` added +315/-0; `vllm/reasoning/step3p5_reasoning_parser.py` added +153/-0; `vllm/transformers_utils/configs/step3p5.py` added +100/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_activation.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33690 - [Bugfix] Fix step3p5 parser when using mtp

- Link: https://github.com/vllm-project/vllm/pull/33690
- Status/date: merged / 2026-02-05
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_step3p5_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`; associated commits `82914d2ae8d0`
- Diff scope read: GitHub Pull Request files API returned 2 files, +1455/-5, 1508 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Step 3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix step3p5 parser when using mtp". The diff centers on `tests/tool_parsers/test_step3p5_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`. PR body context: ## Purpose Fix step3.5 parser when using mtp. If model outputs ` Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such as "Fix some issue...
- Key implementation: `tests/tool_parsers/test_step3p5_tool_parser.py` added +1435/-0 (1435 lines); hunks: -0,0 +1,1435; symbols: step3p5_tokenizer, step3p5_tool_parser, sample_tools, assert_tool_calls, touching `step3p5_tokenizer, step3p5_tool_parser, sample_tools`; `vllm/tool_parsers/step3p5_tool_parser.py` modified +20/-5 (25 lines); hunks: -97,11 +97,26 @@ def parse_single_streaming_chunks(self, xml_chunk: str) -> D...; -110,7 +125,7 @@ def parse_single_streaming_chunks(self, xml_chunk: str) -> D...; symbols: parse_single_streaming_chunks, touching `parse_single_streaming_chunks`.
- Code diff details:
  - `tests/tool_parsers/test_step3p5_tool_parser.py` added +1435/-0 (1435 lines); hunks: -0,0 +1,1435; symbols: step3p5_tokenizer, step3p5_tool_parser, sample_tools, assert_tool_calls
  - `vllm/tool_parsers/step3p5_tool_parser.py` modified +20/-5 (25 lines); hunks: -97,11 +97,26 @@ def parse_single_streaming_chunks(self, xml_chunk: str) -> D...; -110,7 +125,7 @@ def parse_single_streaming_chunks(self, xml_chunk: str) -> D...; symbols: parse_single_streaming_chunks
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_step3p5_tool_parser.py
@@ -0,0 +1,1435 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+from collections.abc import Generator
+import pytest
+from vllm.entrypoints.openai.chat_completion.protocol import (
diff -- vllm/tool_parsers/step3p5_tool_parser.py
@@ -97,11 +97,26 @@ def parse_single_streaming_chunks(self, xml_chunk: str) -> DeltaMessage:
+        entry_call_id = self.current_call_id
+        entry_tool_call_index = self.tool_call_index
+        fallback_call_id = None
+        if entry_call_id is not None:
+            if (
+                self.current_call_id == entry_call_id
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_step3p5_tool_parser.py` added +1435/-0
  - runtime: `vllm/tool_parsers/step3p5_tool_parser.py` modified +20/-5
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_step3p5_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33755 - [Model] Enable Step3p5ForCausalLM testing

- Link: https://github.com/vllm-project/vllm/pull/33755
- Status/date: merged / 2026-02-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/step3p5.py`; associated commits `db4ede974343`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +28/-32, 115 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Step 3.5, this PR adds or enables a model support/runtime surface. Title: "[Model] Enable Step3p5ForCausalLM testing". The diff centers on `vllm/model_executor/models/step3p5.py`. PR body context: ## Purpose ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, such as "Fix some issue (link existing issues...
- Key implementation: `vllm/model_executor/models/step3p5.py` modified +12/-25 (37 lines); hunks: -36,7 +36,6; -770,37 +769,17 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/step3p5.py` modified +12/-25 (37 lines); hunks: -36,7 +36,6; -770,37 +769,17 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/step3p5.py
@@ -36,7 +36,6 @@
-    DEFAULT_VOCAB_PADDING_SIZE,
@@ -770,37 +769,17 @@ def __init__(
-        lora_config = vllm_config.lora_config
-        self.config = config
-        self.vllm_config = vllm_config
-        self.moe_layers: list[FusedMoEBlock] = []
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/step3p5.py` modified +12/-25
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34478 - [Model] Add NVFP4 quantization support for Step3.5-Flash

- Link: https://github.com/vllm-project/vllm/pull/34478
- Status/date: merged / 2026-02-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/step3p5.py`; associated commits `b7892a3beff0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +204/-4, 291 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Step 3.5, this PR adds or enables a model support/runtime surface. Title: "[Model] Add NVFP4 quantization support for Step3.5-Flash". The diff centers on `vllm/model_executor/models/step3p5.py`. PR body context: ## Summary Enable NVFP4 (FP4) quantized MoE inference for stepfun-ai/Step-3.5-Flash. This model uses a `swiglustep` activation (clipped SwiGLU with `limit=7.0`) on MoE layers 43...
- Key implementation: `vllm/model_executor/models/step3p5.py` modified +71/-1 (72 lines); hunks: -2,7 +2,8; -231,6 +232,7 @@ def __init__(; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/step3p5.py` modified +71/-1 (72 lines); hunks: -2,7 +2,8; -231,6 +232,7 @@ def __init__(; symbols: __init__, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/step3p5.py
@@ -2,7 +2,8 @@
-from collections.abc import Iterable
+import typing
+from collections.abc import Callable, Iterable
@@ -231,6 +232,7 @@ def __init__(
+                quant_config=quant_config,
@@ -640,12 +642,22 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/step3p5.py` modified +71/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_nvfp4_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34211 - [Bugfix] Fix step3p5 reasoning with interleaved thinking

- Link: https://github.com/vllm-project/vllm/pull/34211
- Status/date: merged / 2026-02-25
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_step3p5_reasoning_parser.py`, `vllm/reasoning/step3p5_reasoning_parser.py`; associated commits `af5e6afa0af2`
- Diff scope read: GitHub Pull Request files API returned 2 files, +387/-14, 423 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Step 3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix step3p5 reasoning with interleaved thinking". The diff centers on `tests/reasoning/test_step3p5_reasoning_parser.py`, `vllm/reasoning/step3p5_reasoning_parser.py`. PR body context: ## Purpose When there are multiple rounds of conversation, the prompt contains ` ` from the previous round, and the step3p5 reasoning parser failed to correctly determine the en...
- Key implementation: `tests/reasoning/test_step3p5_reasoning_parser.py` added +341/-0 (341 lines); hunks: -0,0 +1,341; symbols: step3p5_tokenizer, test_reasoning, test_step3p5_streaming_drops_leading_newline, touching `step3p5_tokenizer, test_reasoning, test_step3p5_streaming_drops_leading_newline`; `vllm/reasoning/step3p5_reasoning_parser.py` modified +46/-14 (60 lines); hunks: -39,24 +39,59 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):; -136,9 +171,6 @@ def extract_reasoning_streaming(; symbols: __init__, is_reasoning_end, is_reasoning_end_streaming, _is_reasoning_end_from_ids, touching `__init__, is_reasoning_end, is_reasoning_end_streaming`.
- Code diff details:
  - `tests/reasoning/test_step3p5_reasoning_parser.py` added +341/-0 (341 lines); hunks: -0,0 +1,341; symbols: step3p5_tokenizer, test_reasoning, test_step3p5_streaming_drops_leading_newline
  - `vllm/reasoning/step3p5_reasoning_parser.py` modified +46/-14 (60 lines); hunks: -39,24 +39,59 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):; -136,9 +171,6 @@ def extract_reasoning_streaming(; symbols: __init__, is_reasoning_end, is_reasoning_end_streaming, _is_reasoning_end_from_ids
- Key code excerpts:

```diff
diff -- tests/reasoning/test_step3p5_reasoning_parser.py
@@ -0,0 +1,341 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from transformers import AutoTokenizer
+from tests.reasoning.utils import run_reasoning_extraction
+from vllm.reasoning import ReasoningParser, ReasoningParserManager
diff -- vllm/reasoning/step3p5_reasoning_parser.py
@@ -39,24 +39,59 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
-        # Used to delay the reasoning end detection.
-        # This is necessary to remove the newline appears immediately after </think>,
-        # which may cause the end detection to be delayed by one round.
-        self.end_offset = 1
+        # Tracks whether we've seen </think> but are still waiting for one more
+        # token to confirm the end.
```

- Reviewed files:
  - tests: `tests/reasoning/test_step3p5_reasoning_parser.py` added +341/-0
  - runtime: `vllm/reasoning/step3p5_reasoning_parser.py` modified +46/-14
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_step3p5_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37579 - [Model] Refactor Step3-VL processor to HF style

- Link: https://github.com/vllm-project/vllm/pull/37579
- Status/date: merged / 2026-03-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +228/-160, 511 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Step 3.5, this PR extends deployment docs, tests, or CI coverage. Title: "[Model] Refactor Step3-VL processor to HF style". The diff centers on `vllm/transformers_utils/processors/step3_vl.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/transformers_utils/processors/internvl.py`. PR body context: ## Purpose - Make Step3-VL processor contain `image_processor` in order to fit HF call semantics. - Make Step3-VL more efficient by avoiding unnecessary text/token construction...
- Key implementation: `vllm/transformers_utils/processors/step3_vl.py` modified +197/-127 (324 lines); hunks: -8,13 +8,13; -185,7 +185,7 @@ def get_num_patches(self, img_width: int, img_height: int) -...; symbols: Step3VisionProcessor, get_num_patches, __call__, touching `Step3VisionProcessor, get_num_patches, __call__`; `vllm/model_executor/models/step3_vl.py` modified +27/-29 (56 lines); hunks: -39,7 +39,11; -86,21 +90,30 @@ class Step3VLImageEmbeddingInputs(TensorSchema):; symbols: Step3VLImageEmbeddingInputs, Step3VLProcessingInfo, get_image_processor, get_hf_processor, touching `Step3VLImageEmbeddingInputs, Step3VLProcessingInfo, get_image_processor`; `vllm/transformers_utils/processors/internvl.py` modified +4/-3 (7 lines); hunks: -558,6 +558,7 @@ def __call__(; symbols: __call__, touching `__call__`; `vllm/transformers_utils/processors/kimi_k25.py` modified +0/-1 (1 lines); hunks: -19,7 +19,6 @@ def __init__(; symbols: __init__, __call__, touching `__init__, __call__`.
- Code diff details:
  - `vllm/transformers_utils/processors/step3_vl.py` modified +197/-127 (324 lines); hunks: -8,13 +8,13; -185,7 +185,7 @@ def get_num_patches(self, img_width: int, img_height: int) -...; symbols: Step3VisionProcessor, get_num_patches, __call__
  - `vllm/model_executor/models/step3_vl.py` modified +27/-29 (56 lines); hunks: -39,7 +39,11; -86,21 +90,30 @@ class Step3VLImageEmbeddingInputs(TensorSchema):; symbols: Step3VLImageEmbeddingInputs, Step3VLProcessingInfo, get_image_processor, get_hf_processor
  - `vllm/transformers_utils/processors/internvl.py` modified +4/-3 (7 lines); hunks: -558,6 +558,7 @@ def __call__(; symbols: __call__
  - `vllm/transformers_utils/processors/kimi_k25.py` modified +0/-1 (1 lines); hunks: -19,7 +19,6 @@ def __init__(; symbols: __init__, __call__
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/processors/step3_vl.py
@@ -8,13 +8,13 @@
-from transformers import BatchFeature, PretrainedConfig, TensorType
+from transformers import BatchFeature, ProcessorMixin, TensorType
-ImageWithPatches = tuple[Image.Image, list[Image.Image], list[bool] | None]
+ImageWithPatches = tuple[Image.Image, list[Image.Image], list[bool]]
@@ -185,7 +185,7 @@ def get_num_patches(self, img_width: int, img_height: int) -> tuple[int, int]:
-    ) -> tuple[Image.Image, list[Image.Image], list[bool] | None]:
diff -- vllm/model_executor/models/step3_vl.py
@@ -39,7 +39,11 @@
-from vllm.transformers_utils.processors.step3_vl import Step3VLProcessor
+from vllm.transformers_utils.processors.step3_vl import (
+    MAX_IMAGE_SIZE,
+    Step3VLImageProcessor,
+    Step3VLProcessor,
+)
diff -- vllm/transformers_utils/processors/internvl.py
@@ -558,6 +558,7 @@ def __call__(
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/step3_vl.py` modified +197/-127; `vllm/model_executor/models/step3_vl.py` modified +27/-29; `vllm/transformers_utils/processors/internvl.py` modified +4/-3; `vllm/transformers_utils/processors/kimi_k25.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/step3_vl.py`, `vllm/transformers_utils/processors/internvl.py`, `vllm/transformers_utils/processors/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.
