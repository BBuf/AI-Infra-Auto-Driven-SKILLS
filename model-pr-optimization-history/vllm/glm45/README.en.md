# vllm GLM-4.5 Model PR Optimization History

## 2026-06-26 Latest Source Scan

Rechecked vLLM upstream `vllm-project/vllm@abc71548ef029132c3316b902207f254a246d593` against the tracked files listed below.
The file-level match used a GitHub mirror `git log --name-only`; PR titles, links, and merge times were batch-verified through the GitHub GraphQL Pull Request API. Previous freshness anchor: `2026-06-05`.

Result: 3 additional PR-numbered merge(s) touched tracked files and are not yet promoted into full per-PR diff audit cards below. Treat this section as a freshness index; promote any row into a full card only after manual diff review.

| Merged | PR | Title | Tracked files touched |
| --- | --- | --- | --- |
| 2026-06-25 | [#46651](https://github.com/vllm-project/vllm/pull/46651) | [Perf] Remove redundant clone for GLM, Deepseek etc | `glm4_moe_lite.py` |
| 2026-06-18 | [#45915](https://github.com/vllm-project/vllm/pull/45915) | [Frontend] Add Streaming Parser Engine and new GLM4.7/GLM5.1/GLM5.2 Parser | `test_glm4_moe_reasoning_parser.py`, `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-06-08 | [#41184](https://github.com/vllm-project/vllm/pull/41184) | [MoE Refactor] FusedMoE/MoERunner inversion refactor | `glm4_moe_lite_mtp.py`, `glm4_moe_mtp.py` |

## 2026-06-05 PR Backfill Audit

Rechecked vllm upstream `origin/main@c66b19800` on 2026-06-05; 39 additional PR-numbered merge(s) touched the tracked implementation files after the previous freshness cutoff (2025-11-17). These are not yet reflected in the timeline / diff-audit cards below and should be folded in on the next full regeneration.

| Merged | PR | Title | Tracked files touched |
| --- | --- | --- | --- |
| 2026-06-03 | [#44346](https://github.com/vllm-project/vllm/pull/44346) | [Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers | `glm4_moe_tool_parser.py` |
| 2026-05-20 | [#39601](https://github.com/vllm-project/vllm/pull/39601) | [Bugfix] Fix glm4_moe_tool_parser._is_string_type for /v1/responses FunctionTool format | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-05-08 | [#42026](https://github.com/vllm-project/vllm/pull/42026) | [Bugfix] Preserve leading/trailing whitespace in GLM non-streaming tool parser | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-05-07 | [#41755](https://github.com/vllm-project/vllm/pull/41755) | [Bugfix] Fix GLM4-MoE weight loading for NVFP4 quantized checkpoints | `glm4_moe.py` |
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `glm4_moe.py`, `glm4_moe_lite.py`, `glm4_moe_lite_mtp.py`, … (+1) |
| 2026-04-21 | [#35782](https://github.com/vllm-project/vllm/pull/35782) | [MoE Refactor] Remove SharedFusedMoE class | `glm4_moe.py`, `glm4_moe_lite.py`, `glm4_moe_lite_mtp.py` |
| 2026-04-20 | [#35949](https://github.com/vllm-project/vllm/pull/35949) | [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase | `glm4_moe.py` |
| 2026-04-17 | [#39870](https://github.com/vllm-project/vllm/pull/39870) | [BugFix] Support custom tool parsers when tool_choice is `required` and named function | `glm4_moe_tool_parser.py` |
| 2026-04-13 | [#39253](https://github.com/vllm-project/vllm/pull/39253) | [Bugfix] Fix GLM tool parser streaming with MTP or stream interval | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-04-01 | [#38172](https://github.com/vllm-project/vllm/pull/38172) | [Misc] Add 20 regression tests for 11 tool parser bug fixes | `test_glm4_moe_tool_parser.py` |
| 2026-03-31 | [#38264](https://github.com/vllm-project/vllm/pull/38264) | [Mypy] Fix adjust_request typing | `glm4_moe_tool_parser.py` |
| 2026-03-31 | [#38189](https://github.com/vllm-project/vllm/pull/38189) | [Tool Parser][2/3] Use self.tools instead of request.tools in tool parsers | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-03-26 | [#38029](https://github.com/vllm-project/vllm/pull/38029) | [Tool Parser][1/3] Pass tools to ToolParser constructor | `glm4_moe_tool_parser.py` |
| 2026-03-18 | [#37386](https://github.com/vllm-project/vllm/pull/37386) | fix(glm47): improve tool call parsing and content normalization | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-03-16 | [#35208](https://github.com/vllm-project/vllm/pull/35208) | GLM4 tool parser: fix streaming mode | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-03-04 | [#35640](https://github.com/vllm-project/vllm/pull/35640) | [MISC] fixed tool_parser mypy errors | `glm4_moe_tool_parser.py` |
| 2026-02-24 | [#34905](https://github.com/vllm-project/vllm/pull/34905) | Fix GLM4 parser tests | `test_glm4_moe_tool_parser.py` |
| 2026-02-02 | [#33525](https://github.com/vllm-project/vllm/pull/33525) | Update get_expert_mapping to include self parameter | `glm4_moe_lite.py` |
| 2026-02-02 | [#33218](https://github.com/vllm-project/vllm/pull/33218) | [Bugfix] GLM-4 tool parser: incremental string streaming | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-01-27 | [#32064](https://github.com/vllm-project/vllm/pull/32064) | [5/N][Attention] Finish eliminating `vllm/attention` folder | `glm4_moe.py` |
| 2026-01-26 | [#33063](https://github.com/vllm-project/vllm/pull/33063) | [Chore] Update type annotation of `input_ids` in model forward | `glm4_moe.py`, `glm4_moe_lite.py`, `glm4_moe_lite_mtp.py`, … (+1) |
| 2026-01-19 | [#31386](https://github.com/vllm-project/vllm/pull/31386) | [GLM-4.7] GLM Model support for GLM-Lite | `glm4_moe_lite.py`, `glm4_moe_lite_mtp.py`, `glm4_moe_mtp.py` |
| 2026-01-15 | [#32321](https://github.com/vllm-project/vllm/pull/32321) | fix: avoid crash on zero-arg tool calls in glm4 parser | `glm4_moe_tool_parser.py` |
| 2026-01-13 | [#32240](https://github.com/vllm-project/vllm/pull/32240) | [Refactor] [6/N] to simplify the vLLM openai chat_completion serving architecture | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-01-12 | [#32150](https://github.com/vllm-project/vllm/pull/32150) | [Model] Remove incorrect `SupportsPP` from MTP models | `glm4_moe_mtp.py` |
| 2026-01-10 | [#32101](https://github.com/vllm-project/vllm/pull/32101) | [MTP][GLM][Bugfix] Fixed .weight_scale loading logic that dropped MTP prediction accuracy with fp8+mtp | `glm4_moe_mtp.py` |
| 2026-01-07 | [#31869](https://github.com/vllm-project/vllm/pull/31869) | [Model] Cleanup: Remove redundant manual definition of `make_empty_intermediate_tensors` in GLM-4-MoE | `glm4_moe.py` |
| 2026-01-07 | [#31757](https://github.com/vllm-project/vllm/pull/31757) | [Bugfix][MTP] Fix GLM4 MoE fp8 loading with MTP on | `glm4_moe_mtp.py` |
| 2026-01-07 | [#31104](https://github.com/vllm-project/vllm/pull/31104) | [BugFix] LoRA: Support loading base_layer of experts | `glm4_moe.py`, `glm4_moe_mtp.py` |
| 2026-01-06 | [#31055](https://github.com/vllm-project/vllm/pull/31055) | [Bugfix] Fix GLM-4 MoE router logits dtype for data parallel chunking | `glm4_moe.py` |
| 2026-01-05 | [#31622](https://github.com/vllm-project/vllm/pull/31622) | Fix GLM-4.6v flash tool calling in transformers 5.x | `glm4_moe_tool_parser.py` |
| 2025-12-20 | [#30876](https://github.com/vllm-project/vllm/pull/30876) | GLM-4.7 Tool Parser and Doc Update | `glm4_moe.py` |
| 2025-12-18 | [#30920](https://github.com/vllm-project/vllm/pull/30920) | [Bugfix] Fix Unicode issues in GLM-4 tool calling | `glm4_moe_tool_parser.py` |
| 2025-12-15 | [#30693](https://github.com/vllm-project/vllm/pull/30693) | [Refactor] [3/N] Move tool parser tests and run on CPU | `test_glm4_moe_tool_parser.py` |
| 2025-12-15 | [#30675](https://github.com/vllm-project/vllm/pull/30675) | [Refactor] [2/N] Move tool parsers into the vLLM main directory | `glm4_moe_tool_parser.py` |
| 2025-12-11 | [#30389](https://github.com/vllm-project/vllm/pull/30389) | Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim` | `glm4_moe.py` |
| 2025-12-04 | [#29966](https://github.com/vllm-project/vllm/pull/29966) | Access `partial_rotary_factor` from `rope_parameters` | `glm4_moe.py` |
| 2025-11-26 | [#29342](https://github.com/vllm-project/vllm/pull/29342) | [Attention] Remove imports from `vllm/attention/__init__.py` | `glm4_moe.py` |
| 2025-11-19 | [#28542](https://github.com/vllm-project/vllm/pull/28542) | Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5 | `glm4_moe.py` |


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

## PR Coverage Summary

- Git-traced PRs: 10
- Extra PRs preserved from existing docs: 3
- Total PRs in this document: 13
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

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
- Motivation: Title: "remove GLM-4 quantization wrong Code"; model line: GLM-4.5; category: bug fix; main diff: `vllm/model_executor/models/glm4_moe.py`; technical summary: Covers "remove GLM-4 quantization wrong Code"; the main implementation surface is `vllm/model_executor/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "fuse fp32 for GLM-4.5 e_score_correction_bias"; model line: GLM-4.5; category: performance/backend optimization; main diff: `vllm/model_executor/models/glm4_moe.py`; technical summary: Covers "fuse fp32 for GLM-4.5 e_score_correction_bias"; the main implementation surface is `vllm/model_executor/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Misc] Modify the organization of GLM series"; model line: GLM-4.5; category: model implementation change; main diff: `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py`; technical summary: Covers "[Misc] Modify the organization of GLM series"; the main implementation surface is `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "self.gate dtype update for GLM-4.5"; model line: GLM-4.5; category: performance/backend optimization; main diff: `vllm/model_executor/models/glm4_moe.py`; technical summary: Covers "self.gate dtype update for GLM-4.5"; the main implementation surface is `vllm/model_executor/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "not tie_word_embeddings for glm-4.5 and glm-4.5v"; model line: GLM-4.5; category: model implementation change; main diff: `vllm/model_executor/models/glm4_moe.py`; technical summary: Covers "not tie_word_embeddings for glm-4.5 and glm-4.5v"; the main implementation surface is `vllm/model_executor/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "GLM-4.5V with new class name at transformers"; model line: GLM-4.5; category: model implementation change; main diff: `vllm/model_executor/models/glm4_moe.py`; technical summary: Covers "GLM-4.5V with new class name at transformers"; the main implementation surface is `vllm/model_executor/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Model] Modify the gate implementation of glm4_moe"; model line: GLM-4.5; category: model implementation change; main diff: `vllm/model_executor/models/glm4_moe.py`; technical summary: Covers "[Model] Modify the gate implementation of glm4_moe"; the main implementation surface is `vllm/model_executor/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200"; model line: GLM-4.5; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`; technical summary: Covers "feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200"; the main implementation surface is `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser"; model line: GLM-4.5; category: docs/tests/CI; main diff: `docs/features/reasoning_outputs.md`, `docs/features/tool_calling.md`; technical summary: Covers "[Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser"; the main implementation surface is `docs/features/reasoning_outputs.md`, `docs/features/tool_calling.md`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Model] Apply SharedFusedMoE to glm4_moe."; model line: GLM-4.5; category: performance/backend optimization; main diff: `vllm/model_executor/models/glm4_moe.py`; technical summary: Covers "[Model] Apply SharedFusedMoE to glm4_moe."; the main implementation surface is `vllm/model_executor/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Update GLM-4.5 Doc transformers version"; model line: GLM-4.5; category: docs/tests/CI; main diff: `vllm/model_executor/models/glm4_moe.py`; technical summary: Covers "Update GLM-4.5 Doc transformers version"; the main implementation surface is `vllm/model_executor/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint."; model line: GLM-4.5; category: bug fix; main diff: `vllm/model_executor/models/glm4_moe_mtp.py`; technical summary: Covers "[Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint."; the main implementation surface is `vllm/model_executor/models/glm4_moe_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[BugFix] Fix glm4_moe_mtp load weights bug"; model line: GLM-4.5; category: bug fix; main diff: `vllm/model_executor/models/glm4_moe_mtp.py`; technical summary: Covers "[BugFix] Fix glm4_moe_mtp load weights bug"; the main implementation surface is `vllm/model_executor/models/glm4_moe_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
