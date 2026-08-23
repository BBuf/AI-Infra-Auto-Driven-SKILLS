# vllm Qwen3 Core Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/models/multimodal/pooling/test_colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398), [#34574](https://github.com/vllm-project/vllm/pull/34574) |
| `tests/parser/engine/test_qwen3.py` | [#45413](https://github.com/vllm-project/vllm/pull/45413), [#46047](https://github.com/vllm-project/vllm/pull/46047), [#46351](https://github.com/vllm-project/vllm/pull/46351), [#48846](https://github.com/vllm-project/vllm/pull/48846) |
| `vllm/model_executor/models/colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398), [#34574](https://github.com/vllm-project/vllm/pull/34574) |
| `vllm/model_executor/models/qwen3.py` | [#15289](https://github.com/vllm-project/vllm/pull/15289), [#17735](https://github.com/vllm-project/vllm/pull/17735), [#19260](https://github.com/vllm-project/vllm/pull/19260), [#21924](https://github.com/vllm-project/vllm/pull/21924), [#29816](https://github.com/vllm-project/vllm/pull/29816) |
| `vllm/model_executor/models/qwen3_dflash.py` | [#52560](https://github.com/vllm-project/vllm/pull/52560) |
| `vllm/model_executor/models/qwen3_moe.py` | [#15289](https://github.com/vllm-project/vllm/pull/15289), [#16203](https://github.com/vllm-project/vllm/pull/16203), [#17735](https://github.com/vllm-project/vllm/pull/17735), [#18118](https://github.com/vllm-project/vllm/pull/18118), [#19598](https://github.com/vllm-project/vllm/pull/19598), [#19860](https://github.com/vllm-project/vllm/pull/19860), [#20101](https://github.com/vllm-project/vllm/pull/20101), [#20815](https://github.com/vllm-project/vllm/pull/20815), [#21924](https://github.com/vllm-project/vllm/pull/21924), [#22017](https://github.com/vllm-project/vllm/pull/22017), [#22785](https://github.com/vllm-project/vllm/pull/22785), [#23169](https://github.com/vllm-project/vllm/pull/23169), ... (24 total) |
| `vllm/parser/qwen3.py` | [#45413](https://github.com/vllm-project/vllm/pull/45413), [#45763](https://github.com/vllm-project/vllm/pull/45763), [#46047](https://github.com/vllm-project/vllm/pull/46047), [#46314](https://github.com/vllm-project/vllm/pull/46314), [#46351](https://github.com/vllm-project/vllm/pull/46351), [#48846](https://github.com/vllm-project/vllm/pull/48846) |
| `vllm/transformers_utils/configs/colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398) |

## PR Coverage Summary

- Git-traced PRs: 33
- Extra PRs preserved from existing docs: 4
- Total PRs in this document: 37
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-04-07 | [#15289](https://github.com/vllm-project/vllm/pull/15289) | merged | [Model] Add Qwen3 and Qwen3MoE | `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py` |
| 2025-04-08 | [#16203](https://github.com/vllm-project/vllm/pull/16203) | merged | [Model] use AutoWeightsLoader for phimoe,qwen2_moe,qwen3_moe | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-05-07 | [#17735](https://github.com/vllm-project/vllm/pull/17735) | merged | [Kernel] Use fused rmsnorm for some models like qwen3 series | `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py` |
| 2025-05-14 | [#18118](https://github.com/vllm-project/vllm/pull/18118) | merged | [Model] Add packed_modules_mapping for Qwen3-MOE | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-06-11 | [#19260](https://github.com/vllm-project/vllm/pull/19260) | merged | [New Model]: Support Qwen3 Embedding & Reranker | `vllm/model_executor/models/qwen3.py` |
| 2025-06-20 | [#19860](https://github.com/vllm-project/vllm/pull/19860) | merged | [Chore]: qwen3-moe-type-hints-mistake | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-06-30 | [#19598](https://github.com/vllm-project/vllm/pull/19598) | merged | [Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-07-30 | [#20815](https://github.com/vllm-project/vllm/pull/20815) | merged | [Feature][EPLB] Add eplb support for Qwen3 | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-07 | [#21924](https://github.com/vllm-project/vllm/pull/21924) | merged | [Qwen3] Enable dual-chunk-attention support for Qwen3 models. | `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-08 | [#20101](https://github.com/vllm-project/vllm/pull/20101) | merged | Add ModelOpt Qwen3 nvfp4 support | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-11 | [#22017](https://github.com/vllm-project/vllm/pull/22017) | merged | [BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-13 | [#22785](https://github.com/vllm-project/vllm/pull/22785) | merged | Fix GGUF loader for Qwen3 MoE. | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-19 | [#23169](https://github.com/vllm-project/vllm/pull/23169) | merged | [Model] Removes redundant all-reduce operation in Qwen3MoeSparseMoeBlock | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-25 | [#23490](https://github.com/vllm-project/vllm/pull/23490) | merged | [Bugfix] Fix Qwen3 MoE GPTQ inference | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-09-01 | [#23994](https://github.com/vllm-project/vllm/pull/23994) | merged | [BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ) | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-09-17 | [#24727](https://github.com/vllm-project/vllm/pull/24727) | merged | [Model] Support Qwen3-VL Model Series | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-09-27 | [#24982](https://github.com/vllm-project/vllm/pull/24982) | merged | [Bugfix][WideEP] Apply TP Attn + EP MoE fix to other models | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-09-28 | [#25814](https://github.com/vllm-project/vllm/pull/25814) | merged | [Bugfix] Fix Qwen3-VL regression from #24982 | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-10-11 | [#26485](https://github.com/vllm-project/vllm/pull/26485) | merged | Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-11-10 | [#27492](https://github.com/vllm-project/vllm/pull/27492) | merged | [Performance] Support FP8 flashinfer TRTLLM MOE on Qwen3 and Qwen-3next | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-12-10 | [#30308](https://github.com/vllm-project/vllm/pull/30308) | merged | [bugfix][quantization] fix quark qwen3 kv_cache quantization | `vllm/model_executor/models/qwen3_moe.py` |
| 2026-01-24 | [#32082](https://github.com/vllm-project/vllm/pull/32082) | merged | [Models] Add `SharedFusedMoE` support to Qwen3MoE | `vllm/model_executor/models/qwen3_moe.py` |
| 2026-02-06 | [#29816](https://github.com/vllm-project/vllm/pull/29816) | merged | [Bugfix][Model] Support LoRA on Qwen3 Output Embedding | `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py` |
| 2026-02-14 | [#34398](https://github.com/vllm-project/vllm/pull/34398) | merged | [new model] add COLQwen3 code & Inference | `vllm/model_executor/models/colqwen3.py`, `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/transformers_utils/configs/colqwen3.py` |
| 2026-02-21 | [#34574](https://github.com/vllm-project/vllm/pull/34574) | merged | [Frontend] Support multimodal inputs for late-interaction scoring (ColQwen3) + NewModel: nvidia/nemotron-colembed | `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py` |
| 2026-03-04 | [#35656](https://github.com/vllm-project/vllm/pull/35656) | merged | [Bugfix][Model] Fix FP8 k_scale/v_scale not loaded for Qwen3-MoE | `vllm/model_executor/models/qwen3_moe.py` |
| 2026-04-23 | [#40664](https://github.com/vllm-project/vllm/pull/40664) | merged | [BugFix]fix Qwen3 MoE call gate twice | `vllm/model_executor/models/qwen3_moe.py` |
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | merged | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py` |
| 2026-05-11 | [#42280](https://github.com/vllm-project/vllm/pull/42280) | merged | [Model] Fix missing `maybe_prefix` | `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/cohere_asr.py`, `vllm/model_executor/models/hunyuan_v1.py` |
| 2026-06-05 | [#43167](https://github.com/vllm-project/vllm/pull/43167) | merged | Remove KV cache scale boilerplate from model weight loading methods | `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py` |
| 2026-06-10 | [#39419](https://github.com/vllm-project/vllm/pull/39419) | merged | [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding | `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py` |
| 2026-06-15 | [#45413](https://github.com/vllm-project/vllm/pull/45413) | merged | [Frontend] Add Streaming Parser Engine and new Qwen3 Parser | `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py` |
| 2026-06-16 | [#45763](https://github.com/vllm-project/vllm/pull/45763) | merged | [Bugfix] Fix Qwen3 prompt tool-call reasoning false positive | `vllm/parser/qwen3.py` |
| 2026-06-18 | [#46047](https://github.com/vllm-project/vllm/pull/46047) | merged | [Bugfix] [Parser] Fix Qwen3 latent bug in partial params dropping values containing `<` | `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py` |
| 2026-06-23 | [#46351](https://github.com/vllm-project/vllm/pull/46351) | merged | fix: stream Qwen3 tool call string arguments | `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py` |
| 2026-06-25 | [#46314](https://github.com/vllm-project/vllm/pull/46314) | merged | [Frontend] Port seed_oss to the streaming parser engine as a Qwen3 subclass | `vllm/parser/qwen3.py` |
| 2026-07-17 | [#48846](https://github.com/vllm-project/vllm/pull/48846) | merged | [Bugfix][Tool Parser] Preserve whitespace in parameter values (MiniMax M2, Qwen3, MiniCPM5 XML) | `vllm/parser/qwen3.py`, `tests/parser/engine/test_qwen3.py` |

## Per-PR Diff Audit Cards

### PR #15289 - [Model] Add Qwen3 and Qwen3MoE

- Link: https://github.com/vllm-project/vllm/pull/15289
- Status/date: merged / 2025-04-07
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/15289 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`; associated commits `7699258ef013`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +893/-5, 937 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add Qwen3 and Qwen3MoE"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py`; technical summary: Covers "[Model] Add Qwen3 and Qwen3MoE"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: Qwen3MoeMLP, __init__, forward, Qwen3MoeSparseMoeBlock, touching `Qwen3MoeMLP, __init__, forward`; `vllm/model_executor/models/qwen3.py` added +329/-0 (329 lines); hunks: -0,0 +1,329; symbols: Qwen3Attention, __init__, forward, Qwen3DecoderLayer, touching `Qwen3Attention, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: Qwen3MoeMLP, __init__, forward, Qwen3MoeSparseMoeBlock
  - `vllm/model_executor/models/qwen3.py` added +329/-0 (329 lines); hunks: -0,0 +1,329; symbols: Qwen3Attention, __init__, forward, Qwen3DecoderLayer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -0,0 +1,531 @@
+# SPDX-License-Identifier: Apache-2.0
+# Copyright 2024 The Qwen team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
+# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
diff -- vllm/model_executor/models/qwen3.py
@@ -0,0 +1,329 @@
+# SPDX-License-Identifier: Apache-2.0
+# Copyright 2024 The Qwen team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
+# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` added +531/-0; `vllm/model_executor/models/qwen3.py` added +329/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16203 - [Model] use AutoWeightsLoader for phimoe,qwen2_moe,qwen3_moe

- Link: https://github.com/vllm-project/vllm/pull/16203
- Status/date: merged / 2025-04-08
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/16203 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `5a1e1c8353b9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +220/-198, 514 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] use AutoWeightsLoader for phimoe,qwen2_moe,qwen3_moe"; model line: Qwen3 Core; category: model implementation change; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Model] use AutoWeightsLoader for phimoe,qwen2_moe,qwen3_moe"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +65/-58 (123 lines); hunks: -52,7 +52,8; -326,7 +327,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, Qwen3MoeForCausalLM, get_input_embeddings, touching `__init__, forward, Qwen3MoeForCausalLM`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +65/-58 (123 lines); hunks: -52,7 +52,8; -326,7 +327,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, Qwen3MoeForCausalLM, get_input_embeddings
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -52,7 +52,8 @@
-from .utils import (extract_layer_index, is_pp_missing_parameter,
+from .utils import (AutoWeightsLoader, extract_layer_index,
+                    is_pp_missing_parameter,
@@ -326,7 +327,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        self.config = config
@@ -375,60 +376,6 @@ def forward(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +65/-58
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/phimoe.py`, `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17735 - [Kernel] Use fused rmsnorm for some models like qwen3 series

- Link: https://github.com/vllm-project/vllm/pull/17735
- Status/date: merged / 2025-05-07
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/17735 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`; associated commits `f80ae5bdcfa7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +19/-15, 97 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Use fused rmsnorm for some models like qwen3 series"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Kernel] Use fused rmsnorm for some models like qwen3 series"; the main implementation surface is `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3.py` modified +2/-2 (4 lines); hunks: -133,11 +133,11 @@ def forward(; symbols: forward, touching `forward`; `vllm/model_executor/models/qwen3_moe.py` modified +2/-2 (4 lines); hunks: -225,12 +225,12 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3.py` modified +2/-2 (4 lines); hunks: -133,11 +133,11 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/qwen3_moe.py` modified +2/-2 (4 lines); hunks: -225,12 +225,12 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3.py
@@ -133,11 +133,11 @@ def forward(
-        q_by_head = self.q_norm.forward_native(q_by_head)
+        q_by_head = self.q_norm(q_by_head)
-        k_by_head = self.k_norm.forward_native(k_by_head)
+        k_by_head = self.k_norm(k_by_head)
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -225,12 +225,12 @@ def forward(
-        q_by_head = self.q_norm.forward_native(q_by_head)
+        q_by_head = self.q_norm(q_by_head)
-        k_by_head = self.k_norm.forward_native(k_by_head)
+        k_by_head = self.k_norm(k_by_head)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3.py` modified +2/-2; `vllm/model_executor/models/qwen3_moe.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/_custom_ops.py`, `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/molmo.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18118 - [Model] Add packed_modules_mapping for Qwen3-MOE

- Link: https://github.com/vllm-project/vllm/pull/18118
- Status/date: merged / 2025-05-14
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/18118 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `63dc3426e078`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-0, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add packed_modules_mapping for Qwen3-MOE"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Model] Add packed_modules_mapping for Qwen3-MOE"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +11/-0 (11 lines); hunks: -475,6 +475,17 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: load_weights, Qwen3MoeForCausalLM, touching `load_weights, Qwen3MoeForCausalLM`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +11/-0 (11 lines); hunks: -475,6 +475,17 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: load_weights, Qwen3MoeForCausalLM
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -475,6 +475,17 @@ def load_weights(self, weights: Iterable[Tuple[str,
+    packed_modules_mapping = {
+        "qkv_proj": [
+            "q_proj",
+            "k_proj",
+            "v_proj",
+        ],
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +11/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19260 - [New Model]: Support Qwen3 Embedding & Reranker

- Link: https://github.com/vllm-project/vllm/pull/19260
- Status/date: merged / 2025-06-11
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/19260 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3.py`; associated commits `3952731e8f25`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +396/-19, 470 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[New Model]: Support Qwen3 Embedding & Reranker"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3.py`; technical summary: Covers "[New Model]: Support Qwen3 Embedding & Reranker"; the main implementation surface is `vllm/model_executor/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3.py` modified +123/-2 (125 lines); hunks: -38,13 +38,15; -319,3 +321,122 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, Qwen3ForSequenceClassification, __init__, forward, touching `load_weights, Qwen3ForSequenceClassification, __init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3.py` modified +123/-2 (125 lines); hunks: -38,13 +38,15; -319,3 +321,122 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, Qwen3ForSequenceClassification, __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3.py
@@ -38,13 +38,15 @@
+from vllm.model_executor.layers.pooler import Pooler, PoolingType
+from vllm.model_executor.pooling_metadata import PoolingMetadata
-from vllm.sequence import IntermediateTensors
+from vllm.sequence import IntermediateTensors, PoolerOutput
-from .interfaces import SupportsLoRA, SupportsPP
+from .interfaces import SupportsCrossEncoding, SupportsLoRA, SupportsPP
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3.py` modified +123/-2
- Risk and verification: The diff ships test coverage in `tests/models/language/pooling/test_gte.py`, `tests/models/language/pooling/test_qwen3_reranker.py`, `tests/models/language/pooling/test_qwen3_reranker_seq_cls.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19860 - [Chore]: qwen3-moe-type-hints-mistake

- Link: https://github.com/vllm-project/vllm/pull/19860
- Status/date: merged / 2025-06-20
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/19860 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `e41bf15cd04e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Chore]: qwen3-moe-type-hints-mistake"; model line: Qwen3 Core; category: model implementation change; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Chore]: qwen3-moe-type-hints-mistake"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -294,7 +294,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -294,7 +294,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -294,7 +294,7 @@ def forward(
-    ) -> torch.Tensor:
+    ) -> tuple[torch.Tensor, torch.Tensor]:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19598 - [Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model

- Link: https://github.com/vllm-project/vllm/pull/19598
- Status/date: merged / 2025-06-30
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/19598 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `f5dfa0753163`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +15/-9, 53 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model"; model line: Qwen3 Core; category: bug fix; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +15/-9 (24 lines); hunks: -386,6 +386,11 @@ def load_weights(self, weights: Iterable[tuple[str,; -410,10 +415,11 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +15/-9 (24 lines); hunks: -386,6 +386,11 @@ def load_weights(self, weights: Iterable[tuple[str,; -410,10 +415,11 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -386,6 +386,11 @@ def load_weights(self, weights: Iterable[tuple[str,
+        # Skip loading extra parameters for GPTQ/modelopt models.
+        ignore_suffixes = (".bias", "_bias", ".k_scale", "_k_scale",
+                           ".v_scale", "_v_scale", ".weight_scale",
+                           "_weight_scale", ".input_scale", "_input_scale")
@@ -410,10 +415,11 @@ def load_weights(self, weights: Iterable[tuple[str,
-                # Skip loading extra bias for GPTQ models.
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +15/-9
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20815 - [Feature][EPLB] Add eplb support for Qwen3

- Link: https://github.com/vllm-project/vllm/pull/20815
- Status/date: merged / 2025-07-30
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/20815 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `d979dd6bebb1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +142/-24, 273 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature][EPLB] Add eplb support for Qwen3"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Feature][EPLB] Add eplb support for Qwen3"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +142/-24 (166 lines); hunks: -22,7 +22,8; -31,8 +32,9; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +142/-24 (166 lines); hunks: -22,7 +22,8; -31,8 +32,9; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -22,7 +22,8 @@
-from collections.abc import Iterable
+import typing
+from collections.abc import Callable, Iterable
@@ -31,8 +32,9 @@
-from vllm.config import CacheConfig, VllmConfig
-from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +142/-24
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21924 - [Qwen3] Enable dual-chunk-attention support for Qwen3 models.

- Link: https://github.com/vllm-project/vllm/pull/21924
- Status/date: merged / 2025-08-07
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/21924 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`; associated commits `7377131a2ccb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +60/-31, 176 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Qwen3] Enable dual-chunk-attention support for Qwen3 models."; model line: Qwen3 Core; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Qwen3] Enable dual-chunk-attention support for Qwen3 models."; the main implementation surface is `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3.py` modified +40/-24 (64 lines); hunks: -23,7 +23,7; -47,27 +47,31; symbols: Qwen3Attention, __init__, touching `Qwen3Attention, __init__`; `vllm/model_executor/models/qwen3_moe.py` modified +20/-7 (27 lines); hunks: -159,6 +159,7 @@ def __init__(; -182,6 +183,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3.py` modified +40/-24 (64 lines); hunks: -23,7 +23,7; -47,27 +47,31; symbols: Qwen3Attention, __init__
  - `vllm/model_executor/models/qwen3_moe.py` modified +20/-7 (27 lines); hunks: -159,6 +159,7 @@ def __init__(; -182,6 +183,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3.py
@@ -23,7 +23,7 @@
-from typing import Optional, Union
+from typing import Any, Optional, Union
@@ -47,27 +47,31 @@
-from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix
+from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
+                    maybe_prefix)
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -159,6 +159,7 @@ def __init__(
+        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
@@ -182,6 +183,7 @@ def __init__(
+        self.dual_chunk_attention_config = dual_chunk_attention_config
@@ -203,14 +205,21 @@ def __init__(
+            dual_chunk_attention_config=dual_chunk_attention_config,
+        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3.py` modified +40/-24; `vllm/model_executor/models/qwen3_moe.py` modified +20/-7
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20101 - Add ModelOpt Qwen3 nvfp4 support

- Link: https://github.com/vllm-project/vllm/pull/20101
- Status/date: merged / 2025-08-08
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/20101 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `d57dc2364e88`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +58/-37, 129 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add ModelOpt Qwen3 nvfp4 support"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "Add ModelOpt Qwen3 nvfp4 support"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +13/-3 (16 lines); hunks: -48,7 +48,8; -471,12 +472,21 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +13/-3 (16 lines); hunks: -48,7 +48,8; -471,12 +472,21 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -48,7 +48,8 @@
-from vllm.model_executor.model_loader.weight_utils import default_weight_loader
+from vllm.model_executor.model_loader.weight_utils import (
+    default_weight_loader, maybe_remap_kv_scale_name)
@@ -471,12 +472,21 @@ def load_weights(self, weights: Iterable[tuple[str,
+                if name.endswith("scale"):
+                    # Remapping the name of FP8 kv-scale.
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +13/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/qwen2.py`, `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22017 - [BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm

- Link: https://github.com/vllm-project/vllm/pull/22017
- Status/date: merged / 2025-08-11
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/22017 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `1e55dfa7e552`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm"; model line: Qwen3 Core; category: bug fix; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -122,7 +122,7 @@ def __init__(
-                                     quant_config=None,
+                                     quant_config=quant_config,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22785 - Fix GGUF loader for Qwen3 MoE.

- Link: https://github.com/vllm-project/vllm/pull/22785
- Status/date: merged / 2025-08-13
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/22785 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `b159c0a67aaa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +12/-0, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix GGUF loader for Qwen3 MoE."; model line: Qwen3 Core; category: bug fix; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "Fix GGUF loader for Qwen3 MoE."; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -375,6 +375,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -375,6 +375,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -375,6 +375,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+            quant_config=quant_config,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/model_loader/gguf_loader.py`, `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23169 - [Model] Removes redundant all-reduce operation in Qwen3MoeSparseMoeBlock

- Link: https://github.com/vllm-project/vllm/pull/23169
- Status/date: merged / 2025-08-19
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/23169 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `4f510bc2a175`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-5, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Removes redundant all-reduce operation in Qwen3MoeSparseMoeBlock"; model line: Qwen3 Core; category: model implementation change; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Model] Removes redundant all-reduce operation in Qwen3MoeSparseMoeBlock"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +1/-5 (6 lines); hunks: -139,7 +139,7 @@ def __init__(; -163,10 +163,6 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-5 (6 lines); hunks: -139,7 +139,7 @@ def __init__(; -163,10 +163,6 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -139,7 +139,7 @@ def __init__(
-                                reduce_results=False,
+                                reduce_results=True,
@@ -163,10 +163,6 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        if self.tp_size > 1:
-            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
-                final_hidden_states)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +1/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23490 - [Bugfix] Fix Qwen3 MoE GPTQ inference

- Link: https://github.com/vllm-project/vllm/pull/23490
- Status/date: merged / 2025-08-25
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/23490 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `a9082a4d144e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +18/-6, 43 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen3 MoE GPTQ inference"; model line: Qwen3 Core; category: bug fix; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Bugfix] Fix Qwen3 MoE GPTQ inference"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +18/-6 (24 lines); hunks: -45,6 +45,9; -146,11 +149,20 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, forward, load_weights, touching `__init__, _maybe_ignore_quant_config, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +18/-6 (24 lines); hunks: -45,6 +45,9; -146,11 +149,20 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, forward, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -45,6 +45,9 @@
+from vllm.model_executor.layers.quantization.gptq import GPTQConfig
+from vllm.model_executor.layers.quantization.gptq_marlin import (
+    GPTQMarlinConfig)
@@ -146,11 +149,20 @@ def __init__(
-        self.gate = ReplicatedLinear(config.hidden_size,
-                                     config.num_experts,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +18/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23994 - [BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ)

- Link: https://github.com/vllm-project/vllm/pull/23994
- Status/date: merged / 2025-09-01
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/23994 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `183a70967a90`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +17/-4, 57 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ)"; model line: Qwen3 Core; category: bug fix; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ)"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +7/-3 (10 lines); hunks: -159,9 +159,13 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, touching `__init__, _maybe_ignore_quant_config`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +7/-3 (10 lines); hunks: -159,9 +159,13 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -159,9 +159,13 @@ def __init__(
-        # seems to avoid gate quantization.
-        # See: https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4
-        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
+        # seems to avoid gate quantization while AutoRound does.
+        # See: https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4,
+        # and https://huggingface.co/jart25/Qwen3-Coder-30B-A3B-Instruct-Int4-gptq
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +7/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/gptq.py`, `vllm/model_executor/layers/quantization/gptq_marlin.py`, `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24727 - [Model] Support Qwen3-VL Model Series

- Link: https://github.com/vllm-project/vllm/pull/24727
- Status/date: merged / 2025-09-17
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/24727 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `0f7acdd73ca6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +2084/-17, 2262 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support Qwen3-VL Model Series"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Model] Support Qwen3-VL Model Series"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -378,7 +378,7 @@ class Qwen3MoeModel(nn.Module):; symbols: Qwen3MoeModel, __init__, touching `Qwen3MoeModel, __init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -378,7 +378,7 @@ class Qwen3MoeModel(nn.Module):; symbols: Qwen3MoeModel, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -378,7 +378,7 @@ class Qwen3MoeModel(nn.Module):
-        config = vllm_config.model_config.hf_config
+        config = vllm_config.model_config.hf_config.get_text_config()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24982 - [Bugfix][WideEP] Apply TP Attn + EP MoE fix to other models

- Link: https://github.com/vllm-project/vllm/pull/24982
- Status/date: merged / 2025-09-27
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/24982 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `614475401466`, `a5354b3ed247`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 23 files, +541/-376, 1804 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][WideEP] Apply TP Attn + EP MoE fix to other models"; model line: Qwen3 Core; category: bug fix; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Bugfix][WideEP] Apply TP Attn + EP MoE fix to other models"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +33/-27 (60 lines); hunks: -29,13 +29,13; -51,6 +51,7; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, touching `Qwen3MoeSparseMoeBlock, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +33/-27 (60 lines); hunks: -29,13 +29,13; -51,6 +51,7; symbols: Qwen3MoeSparseMoeBlock, __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -29,13 +29,13 @@
-from transformers import Qwen3MoeConfig
-                              get_tensor_model_parallel_world_size)
+                              get_tensor_model_parallel_world_size,
+                              tensor_model_parallel_all_gather)
@@ -51,6 +51,7 @@
+from vllm.model_executor.models.utils import sequence_parallel_chunk
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +33/-27
- Risk and verification: Runtime changes concentrate in `vllm/config/parallel.py`, `vllm/distributed/device_communicators/all2all.py`, `vllm/distributed/device_communicators/base_device_communicator.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25814 - [Bugfix] Fix Qwen3-VL regression from #24982

- Link: https://github.com/vllm-project/vllm/pull/25814
- Status/date: merged / 2025-09-28
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/25814 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `614475401466`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-4, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen3-VL regression from #24982"; model line: Qwen3 Core; category: bug fix; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Bugfix] Fix Qwen3-VL regression from #24982"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +4/-4 (8 lines); hunks: -107,7 +107,7 @@ def __init__(; -293,7 +293,7 @@ class Qwen3MoeDecoderLayer(nn.Module):; symbols: __init__, Qwen3MoeDecoderLayer, Qwen3MoeModel, touching `__init__, Qwen3MoeDecoderLayer, Qwen3MoeModel`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +4/-4 (8 lines); hunks: -107,7 +107,7 @@ def __init__(; -293,7 +293,7 @@ class Qwen3MoeDecoderLayer(nn.Module):; symbols: __init__, Qwen3MoeDecoderLayer, Qwen3MoeModel
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -107,7 +107,7 @@ def __init__(
-        config = vllm_config.model_config.hf_config
+        config = vllm_config.model_config.hf_text_config
@@ -293,7 +293,7 @@ class Qwen3MoeDecoderLayer(nn.Module):
-        config = vllm_config.model_config.hf_config
+        config = vllm_config.model_config.hf_text_config
@@ -372,7 +372,7 @@ class Qwen3MoeModel(nn.Module):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +4/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26485 - Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE

- Link: https://github.com/vllm-project/vllm/pull/26485
- Status/date: merged / 2025-10-11
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/26485 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `d2a71530c159`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +33/-4, 85 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +33/-4 (37 lines); hunks: -64,7 +64,7; -422,6 +422,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, get_input_embeddings, forward, get_expert_mapping, touching `__init__, get_input_embeddings, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +33/-4 (37 lines); hunks: -64,7 +64,7; -422,6 +422,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, get_input_embeddings, forward, get_expert_mapping
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -64,7 +64,7 @@
-from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
+from .interfaces import MixtureOfExperts, SupportsEagle3, SupportsLoRA, SupportsPP
@@ -422,6 +422,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        # Track layers for auxiliary hidden state outputs (EAGLE3)
+        self.aux_hidden_state_layers: tuple[int, ...] = ()
@@ -432,7 +434,9 @@ def forward(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +33/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27492 - [Performance] Support FP8 flashinfer TRTLLM MOE on Qwen3 and Qwen-3next

- Link: https://github.com/vllm-project/vllm/pull/27492
- Status/date: merged / 2025-11-10
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/27492 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `34553b9d2702`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +78/-30, 251 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Performance] Support FP8 flashinfer TRTLLM MOE on Qwen3 and Qwen-3next"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Performance] Support FP8 flashinfer TRTLLM MOE on Qwen3 and Qwen-3next"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -43,6 +43,7; -171,6 +172,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -43,6 +43,7; -171,6 +172,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -43,6 +43,7 @@
+from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
@@ -171,6 +172,7 @@ def __init__(
+            routing_method_type=RoutingMethodType.Renormalize,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30308 - [bugfix][quantization] fix quark qwen3 kv_cache quantization

- Link: https://github.com/vllm-project/vllm/pull/30308
- Status/date: merged / 2025-12-10
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/30308 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `06462392e40f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +14/-0, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bugfix][quantization] fix quark qwen3 kv_cache quantization"; model line: Qwen3 Core; category: bug fix; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[bugfix][quantization] fix quark qwen3 kv_cache quantization"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +14/-0 (14 lines); hunks: -403,6 +403,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -505,6 +506,19 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +14/-0 (14 lines); hunks: -403,6 +403,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -505,6 +506,19 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: __init__, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -403,6 +403,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        self.quant_config = quant_config
@@ -505,6 +506,19 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            if self.quant_config is not None and (
+                scale_name := self.quant_config.get_cache_scale(name)
+            ):
+                # Loading kv cache quantization scales
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +14/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32082 - [Models] Add `SharedFusedMoE` support to Qwen3MoE

- Link: https://github.com/vllm-project/vllm/pull/32082
- Status/date: merged / 2026-01-24
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/32082 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `8edaf3857027`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +56/-16, 143 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Add `SharedFusedMoE` support to Qwen3MoE"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Models] Add `SharedFusedMoE` support to Qwen3MoE"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +56/-16 (72 lines); hunks: -29,6 +29,7; -42,7 +43,7; symbols: __init__, forward, Qwen3MoeSparseMoeBlock, touching `__init__, forward, Qwen3MoeSparseMoeBlock`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +56/-16 (72 lines); hunks: -29,6 +29,7; -42,7 +43,7; symbols: __init__, forward, Qwen3MoeSparseMoeBlock
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -29,6 +29,7 @@
+import torch.nn.functional as F
@@ -42,7 +43,7 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import SharedFusedMoE
@@ -86,6 +87,7 @@ def __init__(
+        expert_gate: torch.nn.Linear | None = None,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +56/-16
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29816 - [Bugfix][Model] Support LoRA on Qwen3 Output Embedding

- Link: https://github.com/vllm-project/vllm/pull/29816
- Status/date: merged / 2026-02-06
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/29816 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`; associated commits `2991dd3d2241`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +132/-13, 188 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Model] Support LoRA on Qwen3 Output Embedding"; model line: Qwen3 Core; category: bug fix; main diff: `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Bugfix][Model] Support LoRA on Qwen3 Output Embedding"; the main implementation surface is `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3.py` modified +5/-0 (5 lines); hunks: -263,6 +263,11 @@ class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP,...; symbols: Qwen3ForCausalLM, __init__, touching `Qwen3ForCausalLM, __init__`; `vllm/model_executor/models/qwen3_moe.py` modified +5/-0 (5 lines); hunks: -689,6 +689,11 @@ class Qwen3MoeForCausalLM(; symbols: Qwen3MoeForCausalLM, __init__, touching `Qwen3MoeForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3.py` modified +5/-0 (5 lines); hunks: -263,6 +263,11 @@ class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP,...; symbols: Qwen3ForCausalLM, __init__
  - `vllm/model_executor/models/qwen3_moe.py` modified +5/-0 (5 lines); hunks: -689,6 +689,11 @@ class Qwen3MoeForCausalLM(; symbols: Qwen3MoeForCausalLM, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3.py
@@ -263,6 +263,11 @@ class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
+    embedding_modules = {
+        "embed_tokens": "input_embeddings",
+        "lm_head": "output_embeddings",
+    }
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -689,6 +689,11 @@ class Qwen3MoeForCausalLM(
+    embedding_modules = {
+        "embed_tokens": "input_embeddings",
+        "lm_head": "output_embeddings",
+    }
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3.py` modified +5/-0; `vllm/model_executor/models/qwen3_moe.py` modified +5/-0
- Risk and verification: The diff ships test coverage in `tests/lora/test_qwen3_unembed.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34398 - [new model] add COLQwen3 code & Inference

- Link: https://github.com/vllm-project/vllm/pull/34398
- Status/date: merged / 2026-02-14
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/34398 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py`, `vllm/transformers_utils/configs/colqwen3.py`; associated commits `d1ea65d0a1c6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +935/-0, 982 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[new model] add COLQwen3 code & Inference"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `vllm/model_executor/models/colqwen3.py`, `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/transformers_utils/configs/colqwen3.py`; technical summary: Covers "[new model] add COLQwen3 code & Inference"; the main implementation surface is `vllm/model_executor/models/colqwen3.py`, `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/transformers_utils/configs/colqwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/colqwen3.py` added +306/-0 (306 lines); hunks: -0,0 +1,306; symbols: ColQwen3ProcessingInfo, get_hf_config, get_hf_processor, _supports_video, touching `ColQwen3ProcessingInfo, get_hf_config, get_hf_processor`; `tests/models/multimodal/pooling/test_colqwen3.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: _run_token_embed_test, _run_late_interaction_test, _run_relevance_test, test_colqwen3_token_embed, touching `_run_token_embed_test, _run_late_interaction_test, _run_relevance_test`; `vllm/transformers_utils/configs/colqwen3.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: that, ColQwen3Config, for, __init__, touching `that, ColQwen3Config, for`.
- Code diff details:
  - `vllm/model_executor/models/colqwen3.py` added +306/-0 (306 lines); hunks: -0,0 +1,306; symbols: ColQwen3ProcessingInfo, get_hf_config, get_hf_processor, _supports_video
  - `tests/models/multimodal/pooling/test_colqwen3.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: _run_token_embed_test, _run_late_interaction_test, _run_relevance_test, test_colqwen3_token_embed
  - `vllm/transformers_utils/configs/colqwen3.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: that, ColQwen3Config, for, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/colqwen3.py
@@ -0,0 +1,306 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+ColQwen3 late interaction model for multi-modal retrieval and reranking.
+ColQwen3 extends Qwen3-VL with a ColBERT-style late interaction head,
+producing per-token embeddings for both text and image inputs. It uses
diff -- tests/models/multimodal/pooling/test_colqwen3.py
@@ -0,0 +1,156 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tests for ColQwen3 late interaction model for multi-modal retrieval.
+ColQwen3 is a multi-vector retrieval model based on Qwen3-VL backbone with
+ColBERT-style late interaction scoring (MaxSim). It produces per-token
+embeddings for both text and image inputs.
diff -- vllm/transformers_utils/configs/colqwen3.py
@@ -0,0 +1,58 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/colqwen3.py` added +306/-0; `vllm/transformers_utils/configs/colqwen3.py` added +58/-0
  - tests: `tests/models/multimodal/pooling/test_colqwen3.py` added +156/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/pooling/test_colqwen3.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34574 - [Frontend] Support multimodal inputs for late-interaction scoring (ColQwen3) + NewModel: nvidia/nemotron-colembed

- Link: https://github.com/vllm-project/vllm/pull/34574
- Status/date: merged / 2026-02-21
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/34574 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py`; associated commits `5719a4e4e601`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +532/-66, 843 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] Support multimodal inputs for late-interaction scoring (ColQwen3) + NewModel: nvidia/nemotron-colembed"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py`; technical summary: Covers "[Frontend] Support multimodal inputs for late-interaction scoring (ColQwen3) + NewModel: nvidia/nemotron-colembed"; the main implementation surface is `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/multimodal/pooling/test_colqwen3.py` modified +191/-0 (191 lines); hunks: -7,19 +7,31; -33,6 +45,43; symbols: _make_base64_image, _make_image_mm_param, _make_text_mm_param, _run_token_embed_test, touching `_make_base64_image, _make_image_mm_param, _make_text_mm_param`; `vllm/model_executor/models/colqwen3.py` modified +8/-6 (14 lines); hunks: -16,6 +16,7; -229,13 +230,14 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `tests/models/multimodal/pooling/test_colqwen3.py` modified +191/-0 (191 lines); hunks: -7,19 +7,31; -33,6 +45,43; symbols: _make_base64_image, _make_image_mm_param, _make_text_mm_param, _run_token_embed_test
  - `vllm/model_executor/models/colqwen3.py` modified +8/-6 (14 lines); hunks: -16,6 +16,7; -229,13 +230,14 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- tests/models/multimodal/pooling/test_colqwen3.py
@@ -7,19 +7,31 @@
+import base64
+from io import BytesIO
+from PIL import Image
+from vllm.entrypoints.chat_utils import (
+    ChatCompletionContentPartImageParam,
+    ChatCompletionContentPartTextParam,
diff -- vllm/model_executor/models/colqwen3.py
@@ -16,6 +16,7 @@
+- nvidia/nemotron-colembed-vl-4b-v2
@@ -229,13 +230,14 @@ def forward(
-        proj_dtype = self.custom_text_proj.weight.dtype  # type: ignore
-        if hidden_states.dtype != proj_dtype:
-            hidden_states = hidden_states.to(proj_dtype)
+        if self.custom_text_proj is not None:
```

- Reviewed files:
  - tests: `tests/models/multimodal/pooling/test_colqwen3.py` modified +191/-0
  - runtime: `vllm/model_executor/models/colqwen3.py` modified +8/-6
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/pooling/test_colqwen3.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35656 - [Bugfix][Model] Fix FP8 k_scale/v_scale not loaded for Qwen3-MoE

- Link: https://github.com/vllm-project/vllm/pull/35656
- Status/date: merged / 2026-03-04
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/35656 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `c8c3935b7013`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +129/-36, 221 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Model] Fix FP8 k_scale/v_scale not loaded for Qwen3-MoE"; model line: Qwen3 Core; category: bug fix; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[Bugfix][Model] Fix FP8 k_scale/v_scale not loaded for Qwen3-MoE"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +6/-18 (24 lines); hunks: -535,10 +535,6 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -562,6 +558,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +6/-18 (24 lines); hunks: -535,10 +535,6 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -562,6 +558,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -535,10 +535,6 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-            ".k_scale",
-            "_k_scale",
-            ".v_scale",
-            "_v_scale",
@@ -562,6 +558,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            if "scale" in name or "zero_point" in name:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +6/-18
- Risk and verification: The diff ships test coverage in `tests/model_executor/test_weight_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40664 - [BugFix]fix Qwen3 MoE call gate twice

- Link: https://github.com/vllm-project/vllm/pull/40664
- Status/date: merged / 2026-04-23
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/40664 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `342c58bc548f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +13/-5, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix]fix Qwen3 MoE call gate twice"; model line: Qwen3 Core; category: bug fix; main diff: `vllm/model_executor/models/qwen3_moe.py`; technical summary: Covers "[BugFix]fix Qwen3 MoE call gate twice"; the main implementation surface is `vllm/model_executor/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_moe.py` modified +13/-5 (18 lines); hunks: -231,11 +231,19 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_moe.py` modified +13/-5 (18 lines); hunks: -231,11 +231,19 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -231,11 +231,19 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        # router_logits: (num_tokens, n_experts)
-        router_logits, _ = self.gate(hidden_states)
-        final_hidden_states = self.experts(
-            hidden_states=hidden_states, router_logits=router_logits
-        )
+        if self.experts.is_internal_router:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +13/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- Link: https://github.com/vllm-project/vllm/pull/40671
- Status/date: merged / 2026-04-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 53 files, +254/-98, 1073 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`; technical summary: Covers "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; the main implementation surface is `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping, touching `extra_repr, fused_moe_make_expert_params_mapping`; `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights, touching `load_moe_expert_weights, load_weights`; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits, touching `make_empty_intermediate_tensors, get_expert_mapping, load_weights`; `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights, touching `compute_logits, get_expert_mapping, load_weights`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping
  - `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits
  - `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/afmoe.py` modified +5/-2 (7 lines); hunks: -18,7 +18,10; -479,7 +482,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -1618,6 +1618,25 @@ def extra_repr(self) -> str:
+# This is a temporary forwarding method which will be removed/modified layer.
+def fused_moe_make_expert_params_mapping(
+    model: torch.nn.Module,
+    ckpt_gate_proj_name: str,
+    ckpt_down_proj_name: str,
+    ckpt_up_proj_name: str,
diff -- vllm/model_executor/models/llama4.py
@@ -36,7 +36,10 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import (
+    FusedMoE,
+    fused_moe_make_expert_params_mapping,
+)
@@ -414,7 +417,7 @@ def load_moe_expert_weights(
diff -- vllm/model_executor/models/glm4_moe_lite.py
@@ -41,7 +41,9 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0; `vllm/model_executor/models/llama4.py` modified +7/-4; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4; `vllm/model_executor/models/AXK1.py` modified +6/-3; `vllm/model_executor/models/afmoe.py` modified +5/-2; `vllm/model_executor/models/bailing_moe.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42280 - [Model] Fix missing `maybe_prefix`

- Link: https://github.com/vllm-project/vllm/pull/42280
- Status/date: merged / 2026-05-11
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/42280 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 25 files, +49/-29, 302 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Fix missing `maybe_prefix`"; model line: Qwen3 Core; category: bug fix; main diff: `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/cohere_asr.py`, `vllm/model_executor/models/hunyuan_v1.py`; technical summary: Covers "[Model] Fix missing `maybe_prefix`"; the main implementation surface is `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/cohere_asr.py`, `vllm/model_executor/models/hunyuan_v1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/arcee.py` modified +6/-2 (8 lines); hunks: -45,6 +45,7; -367,7 +368,10 @@ def __init__(self, *, vllm_config, prefix: str = "") -> None:; symbols: __init__, touching `__init__`; `vllm/model_executor/models/cohere_asr.py` modified +3/-2 (5 lines); hunks: -64,7 +64,7; -1717,7 +1717,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`; `vllm/model_executor/models/hunyuan_v1.py` modified +4/-1 (5 lines); hunks: -930,7 +930,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`; `vllm/model_executor/models/deepseek_eagle.py` modified +3/-1 (4 lines); hunks: -198,7 +198,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/arcee.py` modified +6/-2 (8 lines); hunks: -45,6 +45,7; -367,7 +368,10 @@ def __init__(self, *, vllm_config, prefix: str = "") -> None:; symbols: __init__
  - `vllm/model_executor/models/cohere_asr.py` modified +3/-2 (5 lines); hunks: -64,7 +64,7; -1717,7 +1717,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/hunyuan_v1.py` modified +4/-1 (5 lines); hunks: -930,7 +930,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/deepseek_eagle.py` modified +3/-1 (4 lines); hunks: -198,7 +198,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
  - `vllm/model_executor/models/deepseek_eagle3.py` modified +3/-1 (4 lines); hunks: -318,7 +318,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/arcee.py
@@ -45,6 +45,7 @@
+    maybe_prefix,
@@ -367,7 +368,10 @@ def __init__(self, *, vllm_config, prefix: str = "") -> None:
-        self.model = ArceeModel(vllm_config=vllm_config, prefix=f"{prefix}.model")
+        self.model = ArceeModel(
+            vllm_config=vllm_config,
+            prefix=maybe_prefix(prefix, "model"),
diff -- vllm/model_executor/models/cohere_asr.py
@@ -64,7 +64,7 @@
-from .utils import AutoWeightsLoader, WeightsMapper, make_layers
+from .utils import AutoWeightsLoader, WeightsMapper, make_layers, maybe_prefix
@@ -1717,7 +1717,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-            vllm_config=vllm_config, prefix=f"{prefix}.decoder"
+            vllm_config=vllm_config,
+            prefix=maybe_prefix(prefix, "decoder"),
diff -- vllm/model_executor/models/hunyuan_v1.py
@@ -930,7 +930,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/arcee.py` modified +6/-2; `vllm/model_executor/models/cohere_asr.py` modified +3/-2; `vllm/model_executor/models/hunyuan_v1.py` modified +4/-1; `vllm/model_executor/models/deepseek_eagle.py` modified +3/-1; `vllm/model_executor/models/deepseek_eagle3.py` modified +3/-1; `vllm/model_executor/models/granite_speech.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/blip2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43167 - Remove KV cache scale boilerplate from model weight loading methods

- Link: https://github.com/vllm-project/vllm/pull/43167
- Status/date: merged / 2026-06-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 56 files, +88/-731, 1251 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove KV cache scale boilerplate from model weight loading methods"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`; technical summary: Covers "Remove KV cache scale boilerplate from model weight loading methods"; the main implementation surface is `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name, touching `test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale`; `vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader, touching `_get_moe_weight_dtype, kv_cache_scale_loader`; `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod, touching `KVCacheScaleParameter, __new__, weight_loader`; `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter, touching `get_quant_method, get_cache_scale, get_cache_scale_mapper`.
- Code diff details:
  - `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name
  - `vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader
  - `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod
  - `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20 (30 lines); hunks: -646,26 +646,16 @@ def get_scheme(; symbols: get_scheme, get_cache_scale, get_cache_scale_mapper, QuarkLinearMethod
- Key code excerpts:

```diff
diff -- tests/model_executor/test_eagle_quantization.py
@@ -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, dist_init, device) ->
-def test_kv_cache_scale_name_handling():
-    # Mock a quant config that supports cache scales
-    mock_quant_config = Mock()
-    mock_quant_config.get_cache_scale = Mock(return_value="layers.0.self_attn.kv_scale")
-    # Condition check in load_weights
-    name = "layers.0.self_attn.k_proj.weight"
diff -- vllm/model_executor/models/gpt_oss.py
@@ -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:
-            def kv_cache_scale_loader(
-                quant_config: QuantizationConfig,
-                name: str,
-                params_dict: dict[str, typing.Any],
-                weight: torch.Tensor,
-                default_weight_loader: Callable[..., None],
diff -- vllm/model_executor/layers/quantization/kv_cache.py
@@ -15,6 +15,30 @@
```

- Reviewed files:
  - tests: `tests/model_executor/test_eagle_quantization.py` modified +0/-56
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +0/-46; `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4; `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19; `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20; `vllm/model_executor/models/llama4.py` modified +3/-18; `vllm/model_executor/models/glm_ocr_mtp.py` modified +4/-13
- Risk and verification: The diff ships test coverage in `tests/model_executor/test_eagle_quantization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39419 - [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding

- Link: https://github.com/vllm-project/vllm/pull/39419
- Status/date: merged / 2026-06-10
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/39419 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +53/-39, 169 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding"; model line: Qwen3 Core; category: model implementation change; main diff: `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py`; technical summary: Covers "[SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding"; the main implementation surface is `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interfaces.py` modified +35/-0 (35 lines); hunks: -1282,6 +1282,41 @@ def supports_any_eagle(; symbols: supports_any_eagle, LocalArgmaxMixin, get_top_tokens, EagleModelMixin, touching `supports_any_eagle, LocalArgmaxMixin, get_top_tokens`; `vllm/model_executor/models/llama4_eagle.py` modified +0/-17 (17 lines); hunks: -208,23 +208,6 @@ def forward(; symbols: forward, get_top_tokens, load_weights, transform, touching `forward, get_top_tokens, load_weights`; `vllm/model_executor/models/qwen3.py` modified +8/-2 (10 lines); hunks: -48,7 +48,13; -259,7 +265,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, Qwen3ForCausalLM, touching `__init__, Qwen3ForCausalLM`; `vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1 (3 lines); hunks: -31,6 +31,7; -309,7 +310,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, Eagle3DeepseekV2ForCausalLM, __init__, touching `load_weights, Eagle3DeepseekV2ForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/interfaces.py` modified +35/-0 (35 lines); hunks: -1282,6 +1282,41 @@ def supports_any_eagle(; symbols: supports_any_eagle, LocalArgmaxMixin, get_top_tokens, EagleModelMixin
  - `vllm/model_executor/models/llama4_eagle.py` modified +0/-17 (17 lines); hunks: -208,23 +208,6 @@ def forward(; symbols: forward, get_top_tokens, load_weights, transform
  - `vllm/model_executor/models/qwen3.py` modified +8/-2 (10 lines); hunks: -48,7 +48,13; -259,7 +265,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, Qwen3ForCausalLM
  - `vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1 (3 lines); hunks: -31,6 +31,7; -309,7 +310,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, Eagle3DeepseekV2ForCausalLM, __init__
  - `vllm/model_executor/models/llama.py` modified +2/-1 (3 lines); hunks: -62,6 +62,7; -487,7 +488,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, LlamaForCausalLM
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interfaces.py
@@ -1282,6 +1282,41 @@ def supports_any_eagle(
+class LocalArgmaxMixin:
+    """Mixin for draft model heads in speculative decoding.
+    Provides a D2T-aware ``get_top_tokens`` that preserves the
+    local-argmax communication reduction even when the draft vocabulary
+    is smaller than the target vocabulary.
+    When ``draft_id_to_target_id`` is present (shape ``(draft_vocab_size,)``,
diff -- vllm/model_executor/models/llama4_eagle.py
@@ -208,23 +208,6 @@ def forward(
-    def get_top_tokens(
-        self,
-        hidden_states: torch.Tensor,
-    ) -> torch.Tensor:
-        """Vocab-parallel argmax without all-gathering full logits.
-        Falls back to full logits when draft_id_to_target_id remapping is
diff -- vllm/model_executor/models/qwen3.py
@@ -48,7 +48,13 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interfaces.py` modified +35/-0; `vllm/model_executor/models/llama4_eagle.py` modified +0/-17; `vllm/model_executor/models/qwen3.py` modified +8/-2; `vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1; `vllm/model_executor/models/llama.py` modified +2/-1; `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_eagle3.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45413 - [Frontend] Add Streaming Parser Engine and new Qwen3 Parser

- Link: https://github.com/vllm-project/vllm/pull/45413
- Status/date: merged / 2026-06-15
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/45413 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`; associated commits `c4a3f9d13709`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 33 files, +8492/-902, 9786 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] Add Streaming Parser Engine and new Qwen3 Parser"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`; technical summary: Covers "[Frontend] Add Streaming Parser Engine and new Qwen3 Parser"; the main implementation surface is `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/parser/engine/test_qwen3.py` added +1095/-0 (1095 lines); hunks: -0,0 +1,1095; symbols: mock_tokenizer, parser, TestNonStreaming, test_no_tool_calls, touching `mock_tokenizer, parser, TestNonStreaming`; `vllm/parser/qwen3.py` added +218/-0 (218 lines); hunks: -0,0 +1,218; symbols: _qwen3_arg_converter, qwen3_config, Qwen3Parser, __init__, touching `_qwen3_arg_converter, qwen3_config, Qwen3Parser`.
- Code diff details:
  - `tests/parser/engine/test_qwen3.py` added +1095/-0 (1095 lines); hunks: -0,0 +1,1095; symbols: mock_tokenizer, parser, TestNonStreaming, test_no_tool_calls
  - `vllm/parser/qwen3.py` added +218/-0 (218 lines); hunks: -0,0 +1,218; symbols: _qwen3_arg_converter, qwen3_config, Qwen3Parser, __init__
- Key code excerpts:

```diff
diff -- tests/parser/engine/test_qwen3.py
@@ -0,0 +1,1095 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tests for the engine-based Qwen3 tool call parser.
+These validate that the engine-driven parser correctly handles
+Qwen3 XML-style tool calls.
+"""
diff -- vllm/parser/qwen3.py
@@ -0,0 +1,218 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Qwen3 parser for tool calls and reasoning.
+Qwen3 XML tool call format::
+    <tool_call>
+    <function=func_name>
```

- Reviewed files:
  - tests: `tests/parser/engine/test_qwen3.py` added +1095/-0
  - runtime: `vllm/parser/qwen3.py` added +218/-0
- Risk and verification: The diff ships test coverage in `tests/parser/engine/__init__.py`, `tests/parser/engine/conftest.py`, `tests/parser/engine/replay_harness.py`, `tests/parser/engine/streaming_helpers.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45763 - [Bugfix] Fix Qwen3 prompt tool-call reasoning false positive

- Link: https://github.com/vllm-project/vllm/pull/45763
- Status/date: merged / 2026-06-16
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/45763 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/parser/qwen3.py`; associated commits `7d567172fcb7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +40/-0, 78 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen3 prompt tool-call reasoning false positive"; model line: Qwen3 Core; category: bug fix; main diff: `vllm/parser/qwen3.py`; technical summary: Covers "[Bugfix] Fix Qwen3 prompt tool-call reasoning false positive"; the main implementation surface is `vllm/parser/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/parser/qwen3.py` modified +6/-0 (6 lines); hunks: -206,8 +206,14 @@ def is_reasoning_end(self, input_ids: list[int]) -> bool:; symbols: is_reasoning_end, touching `is_reasoning_end`.
- Code diff details:
  - `vllm/parser/qwen3.py` modified +6/-0 (6 lines); hunks: -206,8 +206,14 @@ def is_reasoning_end(self, input_ids: list[int]) -> bool:; symbols: is_reasoning_end
- Key code excerpts:

```diff
diff -- vllm/parser/qwen3.py
@@ -206,8 +206,14 @@ def is_reasoning_end(self, input_ids: list[int]) -> bool:
+        reasoning_start_id = self._reasoning_start_token_id
+                if (
+                    reasoning_start_id is not None
+                    and input_ids[i] == reasoning_start_id
+                ):
+                    return False
```

- Reviewed files:
  - runtime: `vllm/parser/qwen3.py` modified +6/-0
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_qwen3_reasoning.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #46047 - [Bugfix] [Parser] Fix Qwen3 latent bug in partial params dropping values containing `<`

- Link: https://github.com/vllm-project/vllm/pull/46047
- Status/date: merged / 2026-06-18
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/46047 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`; associated commits `09f3cd5c1080`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +19/-1, 34 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] [Parser] Fix Qwen3 latent bug in partial params dropping values containing `<`"; model line: Qwen3 Core; category: bug fix; main diff: `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`; technical summary: Covers "[Bugfix] [Parser] Fix Qwen3 latent bug in partial params dropping values containing `<`"; the main implementation surface is `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/parser/engine/test_qwen3.py` modified +18/-0 (18 lines); hunks: -615,6 +615,24 @@ def test_partial_multiline(self):; symbols: test_partial_multiline, test_partial_value_with_angle_bracket, test_partial_value_with_angle_bracket_and_complete_param, TestSchemaAwareTypeCoercion, touching `test_partial_multiline, test_partial_value_with_angle_bracket, test_partial_value_with_angle_bracket_and_complete_param`; `vllm/parser/qwen3.py` modified +1/-1 (2 lines); hunks: -49,7 +49,7; symbols: _qwen3_arg_converter, touching `_qwen3_arg_converter`.
- Code diff details:
  - `tests/parser/engine/test_qwen3.py` modified +18/-0 (18 lines); hunks: -615,6 +615,24 @@ def test_partial_multiline(self):; symbols: test_partial_multiline, test_partial_value_with_angle_bracket, test_partial_value_with_angle_bracket_and_complete_param, TestSchemaAwareTypeCoercion
  - `vllm/parser/qwen3.py` modified +1/-1 (2 lines); hunks: -49,7 +49,7; symbols: _qwen3_arg_converter
- Key code excerpts:

```diff
diff -- tests/parser/engine/test_qwen3.py
@@ -615,6 +615,24 @@ def test_partial_multiline(self):
+    def test_partial_value_with_angle_bracket(self):
+        from vllm.parser.qwen3 import (
+            _qwen3_arg_converter,
+        )
+        raw = "<parameter=expr>x<5"
+        result = json.loads(_qwen3_arg_converter(raw, partial=True))
diff -- vllm/parser/qwen3.py
@@ -49,7 +49,7 @@
-_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>([^<]*)$", re.DOTALL)
+_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)
```

- Reviewed files:
  - tests: `tests/parser/engine/test_qwen3.py` modified +18/-0
  - runtime: `vllm/parser/qwen3.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_qwen3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #46351 - fix: stream Qwen3 tool call string arguments

- Link: https://github.com/vllm-project/vllm/pull/46351
- Status/date: merged / 2026-06-23
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/46351 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`; associated commits `8db12169a474`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +177/-9, 326 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: stream Qwen3 tool call string arguments"; model line: Qwen3 Core; category: bug fix; main diff: `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`; technical summary: Covers "fix: stream Qwen3 tool call string arguments"; the main implementation surface is `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/parser/engine/test_qwen3.py` modified +65/-1 (66 lines); hunks: -346,6 +346,49 @@ def test_streaming_args_arrive_incrementally(self, parser,...; -395,6 +438,27 @@ def test_streaming_split_parameter_tag(self, parser, mock_r...; symbols: test_streaming_args_arrive_incrementally, test_streaming_long_string_arg_before_parameter_end, test_streaming_text_before_tool, test_streaming_split_parameter_tag, touching `test_streaming_args_arrive_incrementally, test_streaming_long_string_arg_before_parameter_end, test_streaming_text_before_tool`; `vllm/parser/qwen3.py` modified +13/-1 (14 lines); hunks: -42,6 +42,8; -67,7 +69,7 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:; symbols: _qwen3_arg_converter, qwen3_config, touching `_qwen3_arg_converter, qwen3_config`.
- Code diff details:
  - `tests/parser/engine/test_qwen3.py` modified +65/-1 (66 lines); hunks: -346,6 +346,49 @@ def test_streaming_args_arrive_incrementally(self, parser,...; -395,6 +438,27 @@ def test_streaming_split_parameter_tag(self, parser, mock_r...; symbols: test_streaming_args_arrive_incrementally, test_streaming_long_string_arg_before_parameter_end, test_streaming_text_before_tool, test_streaming_split_parameter_tag
  - `vllm/parser/qwen3.py` modified +13/-1 (14 lines); hunks: -42,6 +42,8; -67,7 +69,7 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:; symbols: _qwen3_arg_converter, qwen3_config
- Key code excerpts:

```diff
diff -- tests/parser/engine/test_qwen3.py
@@ -346,6 +346,49 @@ def test_streaming_args_arrive_incrementally(self, parser, mock_request):
+    def test_streaming_long_string_arg_before_parameter_end(self, parser, mock_request):
+        """Long string arguments should stream before the closing parameter tag."""
+        chunks = [
+            "<tool_call>\n",
+            "<function=write_report>\n",
+            "<parameter=content>",
diff -- vllm/parser/qwen3.py
@@ -42,6 +42,8 @@
+PARAM_START = "<parameter="
+PARAM_END = "</parameter>"
@@ -67,7 +69,7 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:
-                params[name] = value
+                params[name] = value.strip()
@@ -86,6 +88,8 @@ def qwen3_config(thinking: bool = True) -> ParserEngineConfig:
```

- Reviewed files:
  - tests: `tests/parser/engine/test_qwen3.py` modified +65/-1
  - runtime: `vllm/parser/qwen3.py` modified +13/-1
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_parser_engine.py`, `tests/parser/engine/test_qwen3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #46314 - [Frontend] Port seed_oss to the streaming parser engine as a Qwen3 subclass

- Link: https://github.com/vllm-project/vllm/pull/46314
- Status/date: merged / 2026-06-25
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/46314 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/parser/qwen3.py`; associated commits `cd347298e86c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +338/-1435, 1879 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] Port seed_oss to the streaming parser engine as a Qwen3 subclass"; model line: Qwen3 Core; category: model implementation change; main diff: `vllm/parser/qwen3.py`; technical summary: Covers "[Frontend] Port seed_oss to the streaming parser engine as a Qwen3 subclass"; the main implementation surface is `vllm/parser/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/parser/qwen3.py` modified +40/-13 (53 lines); hunks: -38,6 +38,8; -75,28 +77,36 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:; symbols: _qwen3_arg_converter, qwen3_config, Qwen3Parser, attributes, touching `_qwen3_arg_converter, qwen3_config, Qwen3Parser`.
- Code diff details:
  - `vllm/parser/qwen3.py` modified +40/-13 (53 lines); hunks: -38,6 +38,8; -75,28 +77,36 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:; symbols: _qwen3_arg_converter, qwen3_config, Qwen3Parser, attributes
- Key code excerpts:

```diff
diff -- vllm/parser/qwen3.py
@@ -38,6 +38,8 @@
+THINK_START = "<think>"
+THINK_END = "</think>"
@@ -75,28 +77,36 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:
-def qwen3_config(thinking: bool = True) -> ParserEngineConfig:
+def qwen3_config(
+    thinking: bool = True,
```

- Reviewed files:
  - runtime: `vllm/parser/qwen3.py` modified +40/-13
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_seed_oss.py`, `tests/parser/engine/trace_builder.py`, `tests/reasoning/test_seedoss_reasoning_parser.py`, `tests/tool_parsers/test_seed_oss_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #48846 - [Bugfix][Tool Parser] Preserve whitespace in parameter values (MiniMax M2, Qwen3, MiniCPM5 XML)

- Link: https://github.com/vllm-project/vllm/pull/48846
- Status/date: merged / 2026-07-17
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/48846 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`; associated commits `11d291511a35`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +164/-9, 249 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Tool Parser] Preserve whitespace in parameter values (MiniMax M2, Qwen3, MiniCPM5 XML)"; model line: Qwen3 Core; category: bug fix; main diff: `vllm/parser/qwen3.py`, `tests/parser/engine/test_qwen3.py`; technical summary: Covers "[Bugfix][Tool Parser] Preserve whitespace in parameter values (MiniMax M2, Qwen3, MiniCPM5 XML)"; the main implementation surface is `vllm/parser/qwen3.py`, `tests/parser/engine/test_qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/parser/qwen3.py` modified +11/-2 (13 lines); hunks: -56,13 +56,22; -71,7 +80,7 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:; symbols: _trim_wrapping_newlines, _qwen3_arg_converter, touching `_trim_wrapping_newlines, _qwen3_arg_converter`; `tests/parser/engine/test_qwen3.py` modified +2/-2 (4 lines); hunks: -454,10 +454,10 @@ def test_streaming_split_next_parameter_tag_is_buffered(se...; symbols: test_streaming_split_next_parameter_tag_is_buffered, test_streaming_numeric_values, touching `test_streaming_split_next_parameter_tag_is_buffered, test_streaming_numeric_values`.
- Code diff details:
  - `vllm/parser/qwen3.py` modified +11/-2 (13 lines); hunks: -56,13 +56,22; -71,7 +80,7 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:; symbols: _trim_wrapping_newlines, _qwen3_arg_converter
  - `tests/parser/engine/test_qwen3.py` modified +2/-2 (4 lines); hunks: -454,10 +454,10 @@ def test_streaming_split_next_parameter_tag_is_buffered(se...; symbols: test_streaming_split_next_parameter_tag_is_buffered, test_streaming_numeric_values
- Key code excerpts:

```diff
diff -- vllm/parser/qwen3.py
@@ -56,13 +56,22 @@
+def _trim_wrapping_newlines(value: str) -> str:
+    """Strip one leading and one trailing newline (the Qwen3 template markup)."""
+    if value.startswith("\n"):
+        value = value[1:]
+    if value.endswith("\n"):
+        value = value[:-1]
diff -- tests/parser/engine/test_qwen3.py
@@ -454,10 +454,10 @@ def test_streaming_split_next_parameter_tag_is_buffered(self, parser, mock_reque
-        assert args_after_partial_tag == '{"query": "hello'
+        assert args_after_partial_tag == '{"query": "hello '
-        assert json.loads(args_text) == {"query": "hello", "limit": "10"}
+        assert json.loads(args_text) == {"query": "hello ", "limit": "10"}
```

- Reviewed files:
  - runtime: `vllm/parser/qwen3.py` modified +11/-2
  - tests: `tests/parser/engine/test_qwen3.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_qwen3.py`, `tests/tool_parsers/test_minicpm5xml_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `tests/tool_parsers/test_qwen3coder_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
