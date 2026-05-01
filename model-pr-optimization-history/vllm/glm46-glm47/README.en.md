# vllm GLM-4.6/4.7 Model PR Optimization History

## Scope

- Rebuilt on: 2026-05-01
- Source baseline: `vllm-project/vllm` trace worktree commit `7075df79b3`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/reasoning/test_glm4_moe_reasoning_parser.py` | no direct PR-number commit |
| `tests/tool_parsers/test_glm47_moe_tool_parser.py` | [#37386](https://github.com/vllm-project/vllm/pull/37386) |
| `tests/tool_parsers/test_glm4_moe_tool_parser.py` | [#37386](https://github.com/vllm-project/vllm/pull/37386) |
| `vllm/model_executor/models/glm4_moe.py` | [#30876](https://github.com/vllm-project/vllm/pull/30876) |
| `vllm/model_executor/models/glm4_moe_lite.py` | [#31386](https://github.com/vllm-project/vllm/pull/31386) |
| `vllm/model_executor/models/glm4_moe_lite_mtp.py` | [#31386](https://github.com/vllm-project/vllm/pull/31386) |
| `vllm/model_executor/models/glm4_moe_mtp.py` | [#27597](https://github.com/vllm-project/vllm/pull/27597), [#31386](https://github.com/vllm-project/vllm/pull/31386) |
| `vllm/tool_parsers/glm47_moe_tool_parser.py` | [#30876](https://github.com/vllm-project/vllm/pull/30876), [#37386](https://github.com/vllm-project/vllm/pull/37386) |
| `vllm/tool_parsers/glm4_moe_tool_parser.py` | [#31622](https://github.com/vllm-project/vllm/pull/31622), [#37386](https://github.com/vllm-project/vllm/pull/37386) |

## PR Coverage Summary

- Git-traced PRs: 5
- Extra PRs preserved from existing docs: 2
- Total PRs in this document: 7
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-10-14 | [#26818](https://github.com/vllm-project/vllm/pull/26818) | merged | [Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200 | `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` |
| 2025-11-12 | [#27597](https://github.com/vllm-project/vllm/pull/27597) | merged | [Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint. | `vllm/model_executor/models/glm4_moe_mtp.py` |
| 2025-12-09 | [#30210](https://github.com/vllm-project/vllm/pull/30210) | merged | [Bugfix]: Fix glm46 awq marlin moe wna16 compatibility | `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py` |
| 2025-12-20 | [#30876](https://github.com/vllm-project/vllm/pull/30876) | merged | GLM-4.7 Tool Parser and Doc Update | `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/model_executor/models/glm4_moe.py` |
| 2026-01-05 | [#31622](https://github.com/vllm-project/vllm/pull/31622) | merged | Fix GLM-4.6v flash tool calling in transformers 5.x | `vllm/tool_parsers/glm4_moe_tool_parser.py` |
| 2026-01-19 | [#31386](https://github.com/vllm-project/vllm/pull/31386) | merged | [GLM-4.7] GLM Model support for GLM-Lite | `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/model_executor/models/glm4_moe_mtp.py` |
| 2026-03-18 | [#37386](https://github.com/vllm-project/vllm/pull/37386) | merged | fix(glm47): improve tool call parsing and content normalization | `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py` |

## Per-PR Diff Audit Cards

### PR #26818 - [Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200

- Link: https://github.com/vllm-project/vllm/pull/26818
- Status/date: merged / 2025-10-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +441/-0, 444 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200"; model line: GLM-4.6/4.7; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json`; technical summary: Covers "[Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200"; the main implementation surface is `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147; `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147; `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json
@@ -0,0 +1,147 @@
+{
+    "triton_version": "3.4.0",
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
diff -- vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json
@@ -0,0 +1,147 @@
+{
+    "triton_version": "3.4.0",
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
diff -- vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json
@@ -0,0 +1,147 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` added +147/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27597 - [Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint.

- Link: https://github.com/vllm-project/vllm/pull/27597
- Status/date: merged / 2025-11-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_moe_mtp.py`; associated commits `d3ade61e429f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-4, 23 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint."; model line: GLM-4.6/4.7; category: bug fix; main diff: `vllm/model_executor/models/glm4_moe_mtp.py`; technical summary: Covers "[Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint."; the main implementation surface is `vllm/model_executor/models/glm4_moe_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/glm4_moe_mtp.py` modified +11/-4 (15 lines); hunks: -256,11 +256,18 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +11/-4 (15 lines); hunks: -256,11 +256,18 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glm4_moe_mtp.py
@@ -256,11 +256,18 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        spec_layer = self.model.mtp_start_layer_idx
-            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
-            if spec_layer is None:
-                continue
-            name = self._rewrite_spec_layer_name(spec_layer, name)
+            if name == "lm_head.weight":
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_moe_mtp.py` modified +11/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glm4_moe_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30210 - [Bugfix]: Fix glm46 awq marlin moe wna16 compatibility

- Link: https://github.com/vllm-project/vllm/pull/30210
- Status/date: merged / 2025-12-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +50/-4, 96 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix]: Fix glm46 awq marlin moe wna16 compatibility"; model line: GLM-4.6/4.7; category: bug fix; main diff: `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py`; technical summary: Covers "[Bugfix]: Fix glm46 awq marlin moe wna16 compatibility"; the main implementation surface is `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +45/-0 (45 lines); hunks: -895,6 +895,48 @@ def get_moe_configs(; -960,6 +1002,9 @@ def get_moe_wna16_block_config(; symbols: get_moe_configs, _ensure_block_size_k_divisible, get_moe_wna16_block_config, touching `get_moe_configs, _ensure_block_size_k_divisible, get_moe_wna16_block_config`; `vllm/model_executor/layers/quantization/moe_wna16.py` modified +5/-4 (9 lines); hunks: -60,7 +60,7 @@ def __init__(; -107,7 +107,7 @@ def from_config(cls, config: dict[str, Any]) -> "MoeWNA16Con...; symbols: __init__, from_config, get_quant_method, moe_wna16_weight_loader, touching `__init__, from_config, get_quant_method`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +45/-0 (45 lines); hunks: -895,6 +895,48 @@ def get_moe_configs(; -960,6 +1002,9 @@ def get_moe_wna16_block_config(; symbols: get_moe_configs, _ensure_block_size_k_divisible, get_moe_wna16_block_config
  - `vllm/model_executor/layers/quantization/moe_wna16.py` modified +5/-4 (9 lines); hunks: -60,7 +60,7 @@ def __init__(; -107,7 +107,7 @@ def from_config(cls, config: dict[str, Any]) -> "MoeWNA16Con...; symbols: __init__, from_config, get_quant_method, moe_wna16_weight_loader
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -895,6 +895,48 @@ def get_moe_configs(
+def _ensure_block_size_k_divisible(
+    size_k: int, block_size_k: int, group_size: int
+) -> int:
+    """Ensure block_size_k is a divisor of size_k and divisible by group_size.
+    This ensures BLOCK_SIZE_K compatibility with MoeWNA16 CUDA kernel which
+    requires size_k % BLOCK_SIZE_K == 0 and BLOCK_SIZE_K % group_size == 0.
diff -- vllm/model_executor/layers/quantization/moe_wna16.py
@@ -60,7 +60,7 @@ def __init__(
-        elif self.linear_quant_method == "awq":
+        elif self.linear_quant_method in ("awq", "awq_marlin"):
@@ -107,7 +107,7 @@ def from_config(cls, config: dict[str, Any]) -> "MoeWNA16Config":
-        elif linear_quant_method == "awq":
+        elif linear_quant_method in ("awq", "awq_marlin"):
@@ -184,7 +184,7 @@ def get_quant_method(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +45/-0; `vllm/model_executor/layers/quantization/moe_wna16.py` modified +5/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30876 - GLM-4.7 Tool Parser and Doc Update

- Link: https://github.com/vllm-project/vllm/pull/30876
- Status/date: merged / 2025-12-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_moe.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`; associated commits `8a7a41437490`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +38/-3, 73 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "GLM-4.7 Tool Parser and Doc Update"; model line: GLM-4.6/4.7; category: docs/tests/CI; main diff: `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/model_executor/models/glm4_moe.py`; technical summary: Covers "GLM-4.7 Tool Parser and Doc Update"; the main implementation surface is `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/model_executor/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/glm47_moe_tool_parser.py` added +23/-0 (23 lines); hunks: -0,0 +1,23; symbols: Glm47MoeModelToolParser, __init__, touching `Glm47MoeModelToolParser, __init__`; `vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -21,7 +21,8.
- Code diff details:
  - `vllm/tool_parsers/glm47_moe_tool_parser.py` added +23/-0 (23 lines); hunks: -0,0 +1,23; symbols: Glm47MoeModelToolParser, __init__
  - `vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -21,7 +21,8
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/glm47_moe_tool_parser.py
@@ -0,0 +1,23 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import regex as re
+from vllm.logger import init_logger
+from vllm.tokenizers import TokenizerLike
+from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser
diff -- vllm/model_executor/models/glm4_moe.py
@@ -21,7 +21,8 @@
-"""Inference-only GLM-4.5, GLM-4.6 model compatible with HuggingFace weights."""
+"""Inference-only GLM-4.5, GLM-4.6, GLM-4.7 model
+compatible with HuggingFace weights."""
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/glm47_moe_tool_parser.py` added +23/-0; `vllm/model_executor/models/glm4_moe.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glm4_moe.py`, `vllm/tool_parsers/__init__.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31622 - Fix GLM-4.6v flash tool calling in transformers 5.x

- Link: https://github.com/vllm-project/vllm/pull/31622
- Status/date: merged / 2026-01-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/glm4_moe_tool_parser.py`; associated commits `02dbb933cb28`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +68/-0, 76 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix GLM-4.6v flash tool calling in transformers 5.x"; model line: GLM-4.6/4.7; category: bug fix; main diff: `vllm/tool_parsers/glm4_moe_tool_parser.py`; technical summary: Covers "Fix GLM-4.6v flash tool calling in transformers 5.x"; the main implementation surface is `vllm/tool_parsers/glm4_moe_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +14/-0 (14 lines); hunks: -56,6 +56,20 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, adjust_request, extract_tool_calls, touching `__init__, adjust_request, extract_tool_calls`.
- Code diff details:
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +14/-0 (14 lines); hunks: -56,6 +56,20 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, adjust_request, extract_tool_calls
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/glm4_moe_tool_parser.py
@@ -56,6 +56,20 @@ def __init__(self, tokenizer: TokenizerLike):
+    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
+        """
+        Adjust request parameters to ensure tool call tokens are not skipped
+        during tokenizer decoding.
+        """
+        request = super().adjust_request(request)
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +14/-0
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/glm4_moe_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31386 - [GLM-4.7] GLM Model support for GLM-Lite

- Link: https://github.com/vllm-project/vllm/pull/31386
- Status/date: merged / 2026-01-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/model_executor/models/glm4_moe_mtp.py`; associated commits `71832ba71e77`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +1135/-1, 1208 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[GLM-4.7] GLM Model support for GLM-Lite"; model line: GLM-4.6/4.7; category: model support/runtime entry; main diff: `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/model_executor/models/glm4_moe_mtp.py`; technical summary: Covers "[GLM-4.7] GLM Model support for GLM-Lite"; the main implementation surface is `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/model_executor/models/glm4_moe_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/glm4_moe_lite.py` added +642/-0 (642 lines); hunks: -0,0 +1,642; symbols: Glm4MoeLiteMLP, Glm4MoeLite, Glm4LiteMixtureOfExperts, Glm4MoeLiteAttention, touching `Glm4MoeLiteMLP, Glm4MoeLite, Glm4LiteMixtureOfExperts`; `vllm/model_executor/models/glm4_moe_lite_mtp.py` added +464/-0 (464 lines); hunks: -0,0 +1,464; symbols: SharedHead, __init__, forward, Glm4MoeLiteMultiTokenPredictorLayer, touching `SharedHead, __init__, forward`; `vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-1 (3 lines); hunks: -21,7 +21,8.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe_lite.py` added +642/-0 (642 lines); hunks: -0,0 +1,642; symbols: Glm4MoeLiteMLP, Glm4MoeLite, Glm4LiteMixtureOfExperts, Glm4MoeLiteAttention
  - `vllm/model_executor/models/glm4_moe_lite_mtp.py` added +464/-0 (464 lines); hunks: -0,0 +1,464; symbols: SharedHead, __init__, forward, Glm4MoeLiteMultiTokenPredictorLayer
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-1 (3 lines); hunks: -21,7 +21,8
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glm4_moe_lite.py
@@ -0,0 +1,642 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The ZhipuAI Team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/model_executor/models/glm4_moe_lite_mtp.py
@@ -0,0 +1,464 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The ZhipuAI Team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/model_executor/models/glm4_moe_mtp.py
@@ -21,7 +21,8 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_moe_lite.py` added +642/-0; `vllm/model_executor/models/glm4_moe_lite_mtp.py` added +464/-0; `vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37386 - fix(glm47): improve tool call parsing and content normalization

- Link: https://github.com/vllm-project/vllm/pull/37386
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`; associated commits `fad09e8a1f51`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +193/-6, 244 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(glm47): improve tool call parsing and content normalization"; model line: GLM-4.6/4.7; category: bug fix; main diff: `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`; technical summary: Covers "fix(glm47): improve tool call parsing and content normalization"; the main implementation surface is `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_glm47_moe_tool_parser.py` added +168/-0 (168 lines); hunks: -0,0 +1,168; symbols: glm47_tokenizer, glm47_tool_parser, mock_request, TestGlm47ExtractToolCalls, touching `glm47_tokenizer, glm47_tool_parser, mock_request`; `vllm/tool_parsers/glm47_moe_tool_parser.py` modified +16/-2 (18 lines); hunks: -1,6 +1,16; -14,10 +24,14; symbols: Glm47MoeModelToolParser, __init__, touching `Glm47MoeModelToolParser, __init__`; `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +6/-1 (7 lines); hunks: -206,7 +206,12 @@ def extract_tool_calls(; symbols: extract_tool_calls, touching `extract_tool_calls`; `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +3/-3 (6 lines); hunks: -107,7 +107,7 @@ def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, m...; -152,7 +152,7 @@ def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, m...; symbols: test_extract_tool_calls_no_tools, touching `test_extract_tool_calls_no_tools`.
- Code diff details:
  - `tests/tool_parsers/test_glm47_moe_tool_parser.py` added +168/-0 (168 lines); hunks: -0,0 +1,168; symbols: glm47_tokenizer, glm47_tool_parser, mock_request, TestGlm47ExtractToolCalls
  - `vllm/tool_parsers/glm47_moe_tool_parser.py` modified +16/-2 (18 lines); hunks: -1,6 +1,16; -14,10 +24,14; symbols: Glm47MoeModelToolParser, __init__
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +6/-1 (7 lines); hunks: -206,7 +206,12 @@ def extract_tool_calls(; symbols: extract_tool_calls
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +3/-3 (6 lines); hunks: -107,7 +107,7 @@ def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, m...; -152,7 +152,7 @@ def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, m...; symbols: test_extract_tool_calls_no_tools
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_glm47_moe_tool_parser.py
@@ -0,0 +1,168 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+"""Tests for the GLM-4.7 tool call parser."""
+import json
+from unittest.mock import Mock
diff -- vllm/tool_parsers/glm47_moe_tool_parser.py
@@ -1,6 +1,16 @@
+"""
+GLM-4.7 Tool Call Parser.
+GLM-4.7 uses a slightly different tool call format compared to GLM-4.5:
+  - The function name may appear on the same line as ``<tool_call>`` without
+    a newline separator before the first ``<arg_key>``.
+  - Tool calls may have zero arguments
diff -- vllm/tool_parsers/glm4_moe_tool_parser.py
@@ -206,7 +206,12 @@ def extract_tool_calls(
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_glm47_moe_tool_parser.py` added +168/-0; `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +3/-3
  - runtime: `vllm/tool_parsers/glm47_moe_tool_parser.py` modified +16/-2; `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +6/-1
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
