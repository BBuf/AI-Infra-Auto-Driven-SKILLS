# vllm DeepSeek V3.1 PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 3
- Rule: use this as the backing dossier for the skill, not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/online_serving/elastic_ep/serve_deepseek_v2.sh` | no direct PR-number commit |
| `examples/tool_chat_template_deepseekv31.jinja` | [#23454](https://github.com/vllm-project/vllm/pull/23454) |
| `tests/tool_parsers/test_deepseekv31_tool_parser.py` | no direct PR-number commit |
| `vllm/model_executor/models/deepseek_mtp.py` | no direct PR-number commit |
| `vllm/model_executor/models/deepseek_v2.py` | no direct PR-number commit |
| `vllm/tool_parsers/deepseekv31_tool_parser.py` | no direct PR-number commit |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-08-23 | [#23454](https://github.com/vllm-project/vllm/pull/23454) | merged | Support DeepSeek-V3.1 tool call | `examples/tool_chat_template_deepseekv31.jinja` |
| 2025-08-27 | [#23666](https://github.com/vllm-project/vllm/pull/23666) | merged | [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt | `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` |
| 2025-10-15 | [#25589](https://github.com/vllm-project/vllm/pull/25589) | merged | [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972) | `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py` |
| 2026-01-15 | [#32361](https://github.com/vllm-project/vllm/pull/32361) | merged | [BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes | `vllm/model_executor/layers/quantization/utils/quant_utils.py` |

## Per-PR Diff Audit Cards

### PR #23454 - Support DeepSeek-V3.1 tool call

- Link: https://github.com/vllm-project/vllm/pull/23454
- Status/date: merged / 2025-08-23
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_deepseekv31.jinja`; associated commits `b8f17f5d980e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +468/-0, 491 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.1, this PR adds or enables a model support/runtime surface. Title: "Support DeepSeek-V3.1 tool call". The diff centers on `examples/tool_chat_template_deepseekv31.jinja`. PR body context: ## Purpose Support DeepSeek-V3.1 tool call. The tool call format of DeepSeek-V3.1 is different from DeepSeek-V3/R1: DeepSeek-V3.1: tool_call_name tool_call_arguments DeepSeek-R1...
- Key implementation: `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91.
- Code diff details:
  - `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91
- Key code excerpts:

```diff
diff -- examples/tool_chat_template_deepseekv31.jinja
@@ -0,0 +1,91 @@
+{% if not add_generation_prompt is defined %}
+  {% set add_generation_prompt = false %}
+{% endif %}
+{% if not thinking is defined %}
+  {% set thinking = false %}
+{% endif %}
```

- Reviewed files:
  - docs: `examples/tool_chat_template_deepseekv31.jinja` added +91/-0
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23666 - [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt

- Link: https://github.com/vllm-project/vllm/pull/23666
- Status/date: merged / 2025-08-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +68/-53, 322 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.1, this PR adds or enables a model support/runtime surface. Title: "[Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt". The diff centers on `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py`. PR body context: ## Purpose Recently DeepGEMM has supported E8M0 scale on Hopper, and this is also required by DeepSeekV3.1 This PR adds the support for it ## Test ### Unit Test ### Accuracy `VL...
- Key implementation: `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5 (9 lines); hunks: -48,8 +48,7; -427,7 +426,7 @@ def process_weights_after_loading(self, layer: Module) -> None:; symbols: process_weights_after_loading, touching `process_weights_after_loading`; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-4 (7 lines); hunks: -40,7 +40,7; -1431,9 +1431,8 @@ def fused_experts(hidden_states: torch.Tensor,; symbols: fused_experts, touching `fused_experts`; `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` modified +3/-3 (6 lines); hunks: -10,7 +10,7; -107,7 +107,7 @@ def workspace_shapes(; symbols: TritonOrDeepGemmExperts, workspace_shapes, apply, touching `TritonOrDeepGemmExperts, workspace_shapes, apply`; `vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -174,7 +174,7 @@ def silu_mul_fp8_quant_deep_gemm(; symbols: silu_mul_fp8_quant_deep_gemm, touching `silu_mul_fp8_quant_deep_gemm`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5 (9 lines); hunks: -48,8 +48,7; -427,7 +426,7 @@ def process_weights_after_loading(self, layer: Module) -> None:; symbols: process_weights_after_loading
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-4 (7 lines); hunks: -40,7 +40,7; -1431,9 +1431,8 @@ def fused_experts(hidden_states: torch.Tensor,; symbols: fused_experts
  - `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` modified +3/-3 (6 lines); hunks: -10,7 +10,7; -107,7 +107,7 @@ def workspace_shapes(; symbols: TritonOrDeepGemmExperts, workspace_shapes, apply
  - `vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -174,7 +174,7 @@ def silu_mul_fp8_quant_deep_gemm(; symbols: silu_mul_fp8_quant_deep_gemm
  - `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +2/-2 (4 lines); hunks: -20,7 +20,7; -385,7 +385,7 @@ def per_token_group_quant_fp8(; symbols: per_token_group_quant_fp8
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/fp8.py
@@ -48,8 +48,7 @@
-from vllm.utils.deep_gemm import (is_blackwell_deep_gemm_e8m0_used,
-                                  is_deep_gemm_supported)
+from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used, is_deep_gemm_supported
@@ -427,7 +426,7 @@ def process_weights_after_loading(self, layer: Module) -> None:
-        if is_blackwell_deep_gemm_e8m0_used():
+        if is_deep_gemm_e8m0_used():
diff -- vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -40,7 +40,7 @@
-from vllm.utils.deep_gemm import is_blackwell_deep_gemm_e8m0_used
+from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
@@ -1431,9 +1431,8 @@ def fused_experts(hidden_states: torch.Tensor,
-    if (allow_deep_gemm and use_fp8_w8a8
-            and (is_blackwell_deep_gemm_e8m0_used()
-                 or _valid_deep_gemm(hidden_states, w1, w2))):
diff -- vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py
@@ -10,7 +10,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-4; `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` modified +3/-3; `vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py` modified +2/-2; `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +2/-2; `vllm/utils/deep_gemm.py` modified +24/-29
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_block_fp8.py`, `tests/kernels/moe/test_deepep_deepgemm_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25589 - [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)

- Link: https://github.com/vllm-project/vllm/pull/25589
- Status/date: merged / 2025-10-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +215/-3, 269 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.1, this PR adds or enables a model support/runtime surface. Title: "[Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)". The diff centers on `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py`. PR body context: ## Purpose This PR adds a new reasoning parser for the DeepSeek-V3.1 model, named deepseek_v3. Unlike previous models such as deepseek_r1, the reasoning parser for DeepSeek-V3.1...
- Key implementation: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic, touching `tokenizer, test_parser_selection, test_identity_reasoning_parser_basic`; `vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: DeepSeekV3ReasoningParser, __init__, is_reasoning_end, extract_content_ids, touching `DeepSeekV3ReasoningParser, __init__, is_reasoning_end`; `vllm/reasoning/identity_reasoning_parser.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: IdentityReasoningParser, __init__, is_reasoning_end, extract_content_ids, touching `IdentityReasoningParser, __init__, is_reasoning_end`; `vllm/entrypoints/openai/serving_chat.py` modified +8/-2 (10 lines); hunks: -573,7 +573,10 @@ async def chat_completion_stream_generator(; -1342,7 +1345,10 @@ async def chat_completion_full_generator(; symbols: chat_completion_stream_generator, chat_completion_full_generator, touching `chat_completion_stream_generator, chat_completion_full_generator`.
- Code diff details:
  - `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic
  - `vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: DeepSeekV3ReasoningParser, __init__, is_reasoning_end, extract_content_ids
  - `vllm/reasoning/identity_reasoning_parser.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: IdentityReasoningParser, __init__, is_reasoning_end, extract_content_ids
  - `vllm/entrypoints/openai/serving_chat.py` modified +8/-2 (10 lines); hunks: -573,7 +573,10 @@ async def chat_completion_stream_generator(; -1342,7 +1345,10 @@ async def chat_completion_full_generator(; symbols: chat_completion_stream_generator, chat_completion_full_generator
  - `docs/features/reasoning_outputs.md` modified +3/-1 (4 lines); hunks: -11,6 +11,7 @@ vLLM currently supports the following reasoning models:; -20,8 +21,9 @@ vLLM currently supports the following reasoning models:
- Key code excerpts:

```diff
diff -- tests/reasoning/test_deepseekv3_reasoning_parser.py
@@ -0,0 +1,76 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from transformers import AutoTokenizer
+from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
+from vllm.reasoning import (
diff -- vllm/reasoning/deepseek_v3_reasoning_parser.py
@@ -0,0 +1,66 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Sequence
+from transformers import PreTrainedTokenizerBase
+from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
+from vllm.logger import init_logger
diff -- vllm/reasoning/identity_reasoning_parser.py
@@ -0,0 +1,58 @@
```

- Reviewed files:
  - tests: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0
  - runtime: `vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0; `vllm/reasoning/identity_reasoning_parser.py` added +58/-0; `vllm/entrypoints/openai/serving_chat.py` modified +8/-2; `vllm/reasoning/__init__.py` modified +4/-0
  - docs: `docs/features/reasoning_outputs.md` modified +3/-1
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_deepseekv3_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #32361 - [BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes

- Link: https://github.com/vllm-project/vllm/pull/32361
- Status/date: merged / 2026-01-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.1, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes". The diff centers on `vllm/model_executor/layers/quantization/utils/quant_utils.py`. PR body context: https://github.com/vllm-project/vllm/pull/29867 broke with For DeepGEMM revert to the behavior before https://github.com/vllm-project/vllm/pull/29867 Credit to: Eldar Kurtić for...
- Key implementation: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0 (3 lines); hunks: -299,6 +299,9 @@ def get_and_maybe_dequant_weights(; symbols: get_and_maybe_dequant_weights, touching `get_and_maybe_dequant_weights`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0 (3 lines); hunks: -299,6 +299,9 @@ def get_and_maybe_dequant_weights(; symbols: get_and_maybe_dequant_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/utils/quant_utils.py
@@ -299,6 +299,9 @@ def get_and_maybe_dequant_weights(
+        # DeepGEMM transforms the scales using `transform_sf_into_required_layout` into
+        # a layout that is not compatible with `scaled_dequantize`.
+        and not layer.quant_method.use_deep_gemm
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/utils/quant_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.
