# vllm GLM-4.5 PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 3
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/reasoning/test_glm4_moe_reasoning_parser.py` | no direct PR-number commit |
| `tests/tool_parsers/test_glm4_moe_tool_parser.py` | no direct PR-number commit |
| `vllm/model_executor/models/glm4_moe.py` | [#21435](https://github.com/vllm-project/vllm/pull/21435), [#22143](https://github.com/vllm-project/vllm/pull/22143), [#22203](https://github.com/vllm-project/vllm/pull/22203), [#22460](https://github.com/vllm-project/vllm/pull/22460), [#22520](https://github.com/vllm-project/vllm/pull/22520), [#22832](https://github.com/vllm-project/vllm/pull/22832), [#24849](https://github.com/vllm-project/vllm/pull/24849), [#25830](https://github.com/vllm-project/vllm/pull/25830) |
| `vllm/model_executor/models/glm4_moe_lite.py` | no direct PR-number commit |
| `vllm/model_executor/models/glm4_moe_lite_mtp.py` | no direct PR-number commit |
| `vllm/model_executor/models/glm4_moe_mtp.py` | [#27597](https://github.com/vllm-project/vllm/pull/27597), [#28805](https://github.com/vllm-project/vllm/pull/28805) |
| `vllm/tool_parsers/glm4_moe_tool_parser.py` | no direct PR-number commit |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-07-24 | [#21435](https://github.com/vllm-project/vllm/pull/21435) | merged | remove GLM-4 quantization wrong Code | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-03 | [#22143](https://github.com/vllm-project/vllm/pull/22143) | merged | fuse fp32 for GLM-4.5 e_score_correction_bias | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-04 | [#22171](https://github.com/vllm-project/vllm/pull/22171) | merged | [Misc] Modify the organization of GLM series | `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py` |
| 2025-08-05 | [#22203](https://github.com/vllm-project/vllm/pull/22203) | merged | self.gate dtype update for GLM-4.5 | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-08 | [#22460](https://github.com/vllm-project/vllm/pull/22460) | merged | not tie_word_embeddings for glm-4.5 and glm-4.5v | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-09 | [#22520](https://github.com/vllm-project/vllm/pull/22520) | merged | GLM-4.5V with new class name at transformers | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-14 | [#22832](https://github.com/vllm-project/vllm/pull/22832) | merged | [Model] Modify the gate implementation of glm4_moe | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-27 | [#23695](https://github.com/vllm-project/vllm/pull/23695) | merged | feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200 | `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` |
| 2025-09-10 | [#24589](https://github.com/vllm-project/vllm/pull/24589) | merged | [Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser | `docs/features/reasoning_outputs.md`, `docs/features/tool_calling.md` |
| 2025-09-17 | [#24849](https://github.com/vllm-project/vllm/pull/24849) | merged | [Model] Apply SharedFusedMoE to glm4_moe. | `vllm/model_executor/models/glm4_moe.py` |
| 2025-09-28 | [#25830](https://github.com/vllm-project/vllm/pull/25830) | merged | Update GLM-4.5 Doc transformers version | `vllm/model_executor/models/glm4_moe.py` |
| 2025-11-12 | [#27597](https://github.com/vllm-project/vllm/pull/27597) | merged | [Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint. | `vllm/model_executor/models/glm4_moe_mtp.py` |
| 2025-11-17 | [#28805](https://github.com/vllm-project/vllm/pull/28805) | merged | [BugFix] Fix glm4_moe_mtp load weights bug | `vllm/model_executor/models/glm4_moe_mtp.py` |

## Per-PR Diff Audit Cards

### PR #21435 - remove GLM-4 quantization wrong Code

- Link: https://github.com/vllm-project/vllm/pull/21435
- Status/date: merged / 2025-07-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_moe.py`; associated commits `85bda9e7d053`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +2/-3, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "remove GLM-4 quantization wrong Code"; model line: GLM-4.5; category: bug fix; main diff: `vllm/model_executor/models/glm4_moe.py`; PR body summary: need to remove this line.
- Key implementation: `vllm/model_executor/models/glm4_moe.py` modified +0/-1 (1 lines); hunks: -390,7 +390,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe.py` modified +0/-1 (1 lines); hunks: -390,7 +390,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -390,7 +390,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-                quant_config=quant_config,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/tool_parsers/glm4_moe_tool_parser.py`, `vllm/model_executor/models/glm4_moe.py`, `vllm/reasoning/glm4_moe_reasoning_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22143 - fuse fp32 for GLM-4.5 e_score_correction_bias

- Link: https://github.com/vllm-project/vllm/pull/22143
- Status/date: merged / 2025-08-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_moe.py`; associated commits `d3c18c9cb0b6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-3, 12 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fuse fp32 for GLM-4.5 e_score_correction_bias"; model line: GLM-4.5; category: performance/backend optimization; main diff: `vllm/model_executor/models/glm4_moe.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/glm4_moe.py` modified +2/-3 (5 lines); hunks: -125,9 +125,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe.py` modified +2/-3 (5 lines); hunks: -125,9 +125,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -125,9 +125,8 @@ def __init__(
-        # noaux_tc is not set in transformers new config now
-        self.gate.e_score_correction_bias = (nn.Parameter(
-            torch.empty(config.n_routed_experts)))
+        self.gate.e_score_correction_bias = nn.Parameter(
+            torch.empty(config.n_routed_experts, dtype=torch.float32))
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +2/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22171 - [Misc] Modify the organization of GLM series

- Link: https://github.com/vllm-project/vllm/pull/22171
- Status/date: merged / 2025-08-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +31/-31, 241 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Modify the organization of GLM series"; model line: GLM-4.5; category: model implementation change; main diff: `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py`; no usable PR-body summary.
- Key implementation: `docs/models/supported_models.md` modified +5/-5 (10 lines); hunks: -328,7 +328,7 @@ th {; -348,8 +348,8 @@ th {; `tests/models/registry.py` modified +5/-5 (10 lines); hunks: -153,7 +153,7 @@ def check_available_online(; -187,8 +187,8 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`; `tests/models/multimodal/generation/test_common.py` modified +3/-3 (6 lines); hunks: -355,7 +355,7; -374,7 +374,7; `vllm/model_executor/models/chatglm.py` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -86,10 +86,10 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `docs/models/supported_models.md` modified +5/-5 (10 lines); hunks: -328,7 +328,7 @@ th {; -348,8 +348,8 @@ th {
  - `tests/models/registry.py` modified +5/-5 (10 lines); hunks: -153,7 +153,7 @@ def check_available_online(; -187,8 +187,8 @@ def check_available_online(; symbols: check_available_online
  - `tests/models/multimodal/generation/test_common.py` modified +3/-3 (6 lines); hunks: -355,7 +355,7; -374,7 +374,7
  - `vllm/model_executor/models/chatglm.py` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -86,10 +86,10 @@ def __init__(; symbols: __init__
  - `tests/models/multimodal/processing/test_common.py` modified +2/-2 (4 lines); hunks: -271,8 +271,8 @@ def _test_processing_correctness_one(; symbols: _test_processing_correctness_one
- Key code excerpts:

```diff
diff -- docs/models/supported_models.md
@@ -328,7 +328,7 @@ th {
-| `ChatGLMModel`, `ChatGLMForConditionalGeneration` | ChatGLM | `THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, `ShieldLM-6B-chatglm3`, etc. | ✅︎ | ✅︎ | ✅︎ |
+| `ChatGLMModel`, `ChatGLMForConditionalGeneration` | ChatGLM | `zai-org/chatglm2-6b`, `zai-org/chatglm3-6b`, `ShieldLM-6B-chatglm3`, etc. | ✅︎ | ✅︎ | ✅︎ |
@@ -348,8 +348,8 @@ th {
-| `GlmForCausalLM` | GLM-4 | `THUDM/glm-4-9b-chat-hf`, etc. | ✅︎ | ✅︎ | ✅︎ |
-| `Glm4ForCausalLM` | GLM-4-0414 | `THUDM/GLM-4-32B-0414`, etc. | ✅︎ | ✅︎ | ✅︎ |
+| `GlmForCausalLM` | GLM-4 | `zai-org/glm-4-9b-chat-hf`, etc. | ✅︎ | ✅︎ | ✅︎ |
diff -- tests/models/registry.py
@@ -153,7 +153,7 @@ def check_available_online(
-    "ChatGLMModel": _HfExamplesInfo("THUDM/chatglm3-6b",
+    "ChatGLMModel": _HfExamplesInfo("zai-org/chatglm3-6b",
@@ -187,8 +187,8 @@ def check_available_online(
-    "GlmForCausalLM": _HfExamplesInfo("THUDM/glm-4-9b-chat-hf"),
-    "Glm4ForCausalLM": _HfExamplesInfo("THUDM/GLM-4-9B-0414"),
+    "GlmForCausalLM": _HfExamplesInfo("zai-org/glm-4-9b-chat-hf"),
diff -- tests/models/multimodal/generation/test_common.py
@@ -355,7 +355,7 @@
```

- Reviewed files:
  - docs: `docs/models/supported_models.md` modified +5/-5
  - tests: `tests/models/registry.py` modified +5/-5; `tests/models/multimodal/generation/test_common.py` modified +3/-3; `tests/models/multimodal/processing/test_common.py` modified +2/-2; `tests/models/language/generation/test_common.py` modified +1/-1; `tests/models/multimodal/processing/test_glm4_1v.py` modified +1/-1; `tests/tokenization/test_cached_tokenizer.py` modified +1/-1
  - runtime: `vllm/model_executor/models/chatglm.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `tests/distributed/test_pipeline_parallel.py`, `tests/lora/test_add_lora.py`, `tests/lora/test_chatglm3_tp.py`, `tests/models/language/generation/test_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22203 - self.gate dtype update for GLM-4.5

- Link: https://github.com/vllm-project/vllm/pull/22203
- Status/date: merged / 2025-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_moe.py`; associated commits `6fa41e0c32f3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +4/-3, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "self.gate dtype update for GLM-4.5"; model line: GLM-4.5; category: performance/backend optimization; main diff: `vllm/model_executor/models/glm4_moe.py`; PR body summary: The entire self.gate module needs to remain in float32 to ensure benchmark performance for GLM-4.5 and GLM-4.5V during propagation..
- Key implementation: `vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -123,6 +123,7 @@ def __init__(; -180,7 +181,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -123,6 +123,7 @@ def __init__(; -180,7 +181,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -123,6 +123,7 @@ def __init__(
+                                     params_dtype=torch.float32,
@@ -180,7 +181,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        router_logits, _ = self.gate(hidden_states)
+        router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22460 - not tie_word_embeddings for glm-4.5 and glm-4.5v

- Link: https://github.com/vllm-project/vllm/pull/22460
- Status/date: merged / 2025-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_moe.py`; associated commits `c152e2a8a0f4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-2, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "not tie_word_embeddings for glm-4.5 and glm-4.5v"; model line: GLM-4.5; category: model implementation change; main diff: `vllm/model_executor/models/glm4_moe.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/glm4_moe.py` modified +0/-2 (2 lines); hunks: -601,8 +601,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe.py` modified +0/-2 (2 lines); hunks: -601,8 +601,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -601,8 +601,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        if self.config.tie_word_embeddings:
-            self.lm_head.weight = self.model.embed_tokens.weight
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +0/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22520 - GLM-4.5V with new class name at transformers

- Link: https://github.com/vllm-project/vllm/pull/22520
- Status/date: merged / 2025-08-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_moe.py`; associated commits `a6022e6fbcbd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +13/-6, 61 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "GLM-4.5V with new class name at transformers"; model line: GLM-4.5; category: model implementation change; main diff: `vllm/model_executor/models/glm4_moe.py`; PR body summary: As this PR mention: https://github.com/huggingface/transformers/pull/39805.
- Key implementation: `vllm/model_executor/models/glm4_moe.py` modified +7/-1 (8 lines); hunks: -372,7 +372,13 @@ def forward(; symbols: forward, Glm4MoeModel, __init__, touching `forward, Glm4MoeModel, __init__`.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe.py` modified +7/-1 (8 lines); hunks: -372,7 +372,13 @@ def forward(; symbols: forward, Glm4MoeModel, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -372,7 +372,13 @@ def forward(
-@support_torch_compile
+@support_torch_compile(
+    dynamic_arg_dims={
+        "input_ids": 0,
+        "positions": -1,
+        "intermediate_tensors": 0,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +7/-1
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22832 - [Model] Modify the gate implementation of glm4_moe

- Link: https://github.com/vllm-project/vllm/pull/22832
- Status/date: merged / 2025-08-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_moe.py`; associated commits `92ff41abea13`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-11, 50 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Modify the gate implementation of glm4_moe"; model line: GLM-4.5; category: model implementation change; main diff: `vllm/model_executor/models/glm4_moe.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/glm4_moe.py` modified +10/-10 (20 lines); hunks: -41,7 +41,6; -118,14 +117,15 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe.py` modified +10/-10 (20 lines); hunks: -41,7 +41,6; -118,14 +117,15 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -41,7 +41,6 @@
-                                               ReplicatedLinear,
@@ -118,14 +117,15 @@ def __init__(
-        self.gate = ReplicatedLinear(config.hidden_size,
-                                     config.n_routed_experts,
-                                     bias=False,
-                                     quant_config=None,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +10/-10
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23695 - feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200

- Link: https://github.com/vllm-project/vllm/pull/23695
- Status/date: merged / 2025-08-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +146/-0, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200"; model line: GLM-4.5; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`; PR body summary: Add fused MoE config for GLM-4.5-Air-FP8 on NVIDIA B200, tp = 2 With config: Without config:.
- Key implementation: `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +146/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24589 - [Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser

- Link: https://github.com/vllm-project/vllm/pull/24589
- Status/date: merged / 2025-09-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-0, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser"; model line: GLM-4.5; category: docs/tests/CI; main diff: `docs/features/reasoning_outputs.md`, `docs/features/tool_calling.md`; PR body summary: Add documentation for GLM-4.5 series models: tool-calling and reasoning parser..
- Key implementation: `docs/features/reasoning_outputs.md` modified +1/-0 (1 lines); hunks: -15,6 +15,7 @@ vLLM currently supports the following reasoning models:; `docs/features/tool_calling.md` modified +9/-0 (9 lines); hunks: -311,6 +311,15 @@ Flags:.
- Code diff details:
  - `docs/features/reasoning_outputs.md` modified +1/-0 (1 lines); hunks: -15,6 +15,7 @@ vLLM currently supports the following reasoning models:
  - `docs/features/tool_calling.md` modified +9/-0 (9 lines); hunks: -311,6 +311,15 @@ Flags:
- Key code excerpts:

```diff
diff -- docs/features/reasoning_outputs.md
@@ -15,6 +15,7 @@ vLLM currently supports the following reasoning models:
+| [GLM-4.5 series](https://huggingface.co/collections/zai-org/glm-45-687c621d34bda8c9e4bf503b) | `glm45` | `guided_json`, `guided_regex` | ✅ |
diff -- docs/features/tool_calling.md
@@ -311,6 +311,15 @@ Flags:
+### GLM-4.5 Models (`glm45`)
+Supported models:
+* `ZhipuAI/GLM-4.5`
+* `ZhipuAI/GLM-4.5-Air`
+Flags: `--tool-call-parser glm45`
```

- Reviewed files:
  - docs: `docs/features/reasoning_outputs.md` modified +1/-0; `docs/features/tool_calling.md` modified +9/-0
- Risk and verification: This is mostly docs/examples in `docs/features/reasoning_outputs.md`, `docs/features/tool_calling.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #24849 - [Model] Apply SharedFusedMoE to glm4_moe.

- Link: https://github.com/vllm-project/vllm/pull/24849
- Status/date: merged / 2025-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_moe.py`; associated commits `c15309a730fa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +55/-30, 114 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Apply SharedFusedMoE to glm4_moe."; model line: GLM-4.5; category: performance/backend optimization; main diff: `vllm/model_executor/models/glm4_moe.py`; PR body summary: The class `SharedFusedMoE` was proposed by @bnellnm in PR #23273. The model glm4_moe has shared experts but we don't use `SharedFusedMoE` for glm4_moe, I'm not sure why, please....
- Key implementation: `vllm/model_executor/models/glm4_moe.py` modified +55/-30 (85 lines); hunks: -46,6 +46,7; -146,25 +147,6 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe.py` modified +55/-30 (85 lines); hunks: -46,6 +46,7; -146,25 +147,6 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -46,6 +46,7 @@
+from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE
@@ -146,25 +147,6 @@ def __init__(
-        self.experts = FusedMoE(
-            num_experts=config.n_routed_experts,
-            top_k=config.num_experts_per_tok,
-            hidden_size=config.hidden_size,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +55/-30
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25830 - Update GLM-4.5 Doc transformers version

- Link: https://github.com/vllm-project/vllm/pull/25830
- Status/date: merged / 2025-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_moe.py`; associated commits `b1ded114b976`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +7/-5, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update GLM-4.5 Doc transformers version"; model line: GLM-4.5; category: docs/tests/CI; main diff: `vllm/model_executor/models/glm4_moe.py`; PR body summary: Update for GLM-4.5 Doc transformers version.
- Key implementation: `vllm/model_executor/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -21,7 +21,7 @@
-"""Inference-only GLM-4.5 model compatible with HuggingFace weights."""
+"""Inference-only GLM-4.5, GLM-4.6 model compatible with HuggingFace weights."""
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27597 - [Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint.

- Link: https://github.com/vllm-project/vllm/pull/27597
- Status/date: merged / 2025-11-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_moe_mtp.py`; associated commits `d3ade61e429f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-4, 23 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint."; model line: GLM-4.5; category: bug fix; main diff: `vllm/model_executor/models/glm4_moe_mtp.py`; PR body summary: As described in issue#25993, when serving GLM-4.6 with mtp using the following command: It raises the following error: The root cause is that the GLM-4.6 checkpoint doesn't incl....
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

### PR #28805 - [BugFix] Fix glm4_moe_mtp load weights bug

- Link: https://github.com/vllm-project/vllm/pull/28805
- Status/date: merged / 2025-11-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_moe_mtp.py`; associated commits `ab01cd14e5e2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-4, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Fix glm4_moe_mtp load weights bug"; model line: GLM-4.5; category: bug fix; main diff: `vllm/model_executor/models/glm4_moe_mtp.py`; PR body summary: As comment in issue#25993,PR#27597 introduced bugs causing GLM-4.5/GLM-4.6 serving with mtp error because: 1. spec_layer is overwritten by later None assignment in "else" branch....
- Key implementation: `vllm/model_executor/models/glm4_moe_mtp.py` modified +3/-4 (7 lines); hunks: -256,13 +256,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +3/-4 (7 lines); hunks: -256,13 +256,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glm4_moe_mtp.py
@@ -256,13 +256,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-        spec_layer = self.model.mtp_start_layer_idx
-                name = f"model.layers.{spec_layer}.shard_head.head.weight"
+                spec_layer = self.model.mtp_start_layer_idx
+                name = f"model.layers.{spec_layer}.shared_head.head.weight"
-                # This name is same with local model, rewriting is not needed.
-                pass
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_moe_mtp.py` modified +3/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glm4_moe_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.
