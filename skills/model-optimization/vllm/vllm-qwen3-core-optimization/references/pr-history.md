# vllm Qwen3 Core PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 0
- Rule: use this as the backing dossier for the skill, not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/models/multimodal/pooling/test_colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398), [#34574](https://github.com/vllm-project/vllm/pull/34574) |
| `vllm/model_executor/models/colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398), [#34574](https://github.com/vllm-project/vllm/pull/34574) |
| `vllm/model_executor/models/qwen3.py` | [#15289](https://github.com/vllm-project/vllm/pull/15289), [#17735](https://github.com/vllm-project/vllm/pull/17735), [#19260](https://github.com/vllm-project/vllm/pull/19260), [#21924](https://github.com/vllm-project/vllm/pull/21924), [#29816](https://github.com/vllm-project/vllm/pull/29816) |
| `vllm/model_executor/models/qwen3_dflash.py` | no direct PR-number commit |
| `vllm/model_executor/models/qwen3_moe.py` | [#15289](https://github.com/vllm-project/vllm/pull/15289), [#16203](https://github.com/vllm-project/vllm/pull/16203), [#17735](https://github.com/vllm-project/vllm/pull/17735), [#18118](https://github.com/vllm-project/vllm/pull/18118), [#19598](https://github.com/vllm-project/vllm/pull/19598), [#19860](https://github.com/vllm-project/vllm/pull/19860), [#20101](https://github.com/vllm-project/vllm/pull/20101), [#20815](https://github.com/vllm-project/vllm/pull/20815), [#21924](https://github.com/vllm-project/vllm/pull/21924), [#22017](https://github.com/vllm-project/vllm/pull/22017), [#22785](https://github.com/vllm-project/vllm/pull/22785), [#23169](https://github.com/vllm-project/vllm/pull/23169), ... (24 total) |
| `vllm/transformers_utils/configs/colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398) |

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

## Per-PR Diff Audit Cards

### PR #15289 - [Model] Add Qwen3 and Qwen3MoE

- Link: https://github.com/vllm-project/vllm/pull/15289
- Status/date: merged / 2025-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`; associated commits `7699258ef013`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +893/-5, 937 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "[Model] Add Qwen3 and Qwen3MoE". The diff centers on `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py`. PR body context: ## Description Recently, I have submitted a pull request to Hugging Face Transformers containing the implementation of the Qwen3 and Qwen3MoE model. I would also like to contrib...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `5a1e1c8353b9`
- Diff scope read: GitHub Pull Request files API returned 3 files, +220/-198, 514 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Model] use AutoWeightsLoader for phimoe,qwen2_moe,qwen3_moe". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: FIX https://github.com/vllm-project/vllm/issues/15697 - Qwen3MoeForCausalLM Because can't download Qwen/Qwen3-MoE-15B-A2B this model, so not test. - Qwen2MoeForCausalLM - PhiMoE...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`; associated commits `f80ae5bdcfa7`
- Diff scope read: GitHub Pull Request files API returned 7 files, +19/-15, 97 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR optimizes an inference path or backend selection. Title: "[Kernel] Use fused rmsnorm for some models like qwen3 series". The diff centers on `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`. PR body context: Some models like qwen3 series still use forward_native method when q and k need to be normalized. The reason why fused rmsnorm kernel is not being used is that before q norm and...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `63dc3426e078`
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-0, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "[Model] Add packed_modules_mapping for Qwen3-MOE". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. The PR body has no extra context, so this assessment comes from the title, file list, and patch.
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3.py`; associated commits `3952731e8f25`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +396/-19, 470 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "[New Model]: Support Qwen3 Embedding & Reranker". The diff centers on `vllm/model_executor/models/qwen3.py`. PR body context: # Summary - Qwen3 Embedding - Qwen/Qwen3-Embedding-0.6B - Qwen/Qwen3-Embedding-4B - Qwen/Qwen3-Embedding-8B - Qwen3 Reranker - Qwen/Qwen3-Reranker-0.6B - Qwen/Qwen3-Reranker-4B...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `e41bf15cd04e`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Chore]: qwen3-moe-type-hints-mistake". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". - [ ] The test...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `f5dfa0753163`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +15/-9, 53 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". - [x] The test...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `d979dd6bebb1`
- Diff scope read: GitHub Pull Request files API returned 1 files, +142/-24, 273 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "[Feature][EPLB] Add eplb support for Qwen3". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". - [x] The test...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`; associated commits `7377131a2ccb`
- Diff scope read: GitHub Pull Request files API returned 2 files, +60/-31, 176 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "[Qwen3] Enable dual-chunk-attention support for Qwen3 models.". The diff centers on `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`. PR body context: # Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". - [ ] The test p...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `d57dc2364e88`
- Diff scope read: GitHub Pull Request files API returned 3 files, +58/-37, 129 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "Add ModelOpt Qwen3 nvfp4 support". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". - [x] The test...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `1e55dfa7e552`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: Description: This PR fixes a KeyError for 'layers.14.mlp.gate.g_idx' that was popping up when loading GPTQ-quantized Qwen3-MoE models, particularly in ROCm setups. The problem s...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `b159c0a67aaa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +12/-0, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "Fix GGUF loader for Qwen3 MoE.". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Purpose Despite upstream repositories (gguf-py, transformers) having added support for Qwen3 MoE GGUF quantization, vLLM GGUF loading is still broken. This PR aims to fix the...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `4f510bc2a175`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-5, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR optimizes an inference path or backend selection. Title: "[Model] Removes redundant all-reduce operation in Qwen3MoeSparseMoeBlock". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Purpose Remove redundant all-reduce operation and reuse the all-reduce implementation provided by `FusedMoE` (see `vllm/model_executor/layers/fused_moe/layer.py#L1672`). ## T...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `a9082a4d144e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +18/-6, 43 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix Qwen3 MoE GPTQ inference". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Purpose - Fix #23467 ## Test Plan ## Test Result ## (Optional) Documentation Update --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the P...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `183a70967a90`
- Diff scope read: GitHub Pull Request files API returned 3 files, +17/-4, 57 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ)". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: The following PR attempts to make the quantized MOE model chart compatible with AutoGPTQ and Autoround-GPTQ. The PR: https://github.com/vllm-project/vllm/issues/23467 attempted...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `0f7acdd73ca6`
- Diff scope read: GitHub Pull Request files API returned 13 files, +2084/-17, 2262 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "[Model] Support Qwen3-VL Model Series". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Purpose This PR adds model support for the upcoming Qwen3-VL models, including both dense and MoE variants. Reference HF implementation - https://github.com/huggingface/trans...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `614475401466`, `a5354b3ed247`
- Diff scope read: GitHub Pull Request files API returned 23 files, +541/-376, 1804 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "[Bugfix][WideEP] Apply TP Attn + EP MoE fix to other models". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Purpose Prior to this PR, in many cases, using TP Attn and EP MoEs with `--tensor-parallel-size N --data-parallel-size M --enable-expert-parallel` would result in factor `N`...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `614475401466`
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-4, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix Qwen3-VL regression from #24982". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Purpose #24982 caused a regression since the Qwen3-VL needs to access the text config of Qwen3-MoE via `get_text_config()` ## Test Plan ## Test Result Tested locally to make...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `d2a71530c159`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +33/-4, 85 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: This PR adds support for EAGLE-3 speculative decoding to the `Qwen3MoeForCausalLM` model, enabling faster inference with draft models like `nm-testing/Mockup-qwen235-eagle3-fp16...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `34553b9d2702`
- Diff scope read: GitHub Pull Request files API returned 7 files, +78/-30, 251 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "[Performance] Support FP8 flashinfer TRTLLM MOE on Qwen3 and Qwen-3next". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Purpose - Integrate multiple routing methods for FP8 flashinfer trtllm MOE, currently only DS and Llama4 - Add FP8 flashinfer trtllm MOE support on Qwen3 and Qwen3-next ## Te...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `06462392e40f`
- Diff scope read: GitHub Pull Request files API returned 1 files, +14/-0, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[bugfix][quantization] fix quark qwen3 kv_cache quantization". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: Qwen3moe cannot effectively recognize the kv_cache scale. We need to call the get_cache_scale method in the base class to identify it. test scripts:
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `8edaf3857027`
- Diff scope read: GitHub Pull Request files API returned 1 files, +56/-16, 143 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "[Models] Add `SharedFusedMoE` support to Qwen3MoE". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Purpose - Fix https://github.com/vllm-project/vllm-omni/pull/560#discussion_r2655368800 - Qwen3-omni's MoE talker has share experts in its sparse moe block, while vLLM's Qwen...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`; associated commits `2991dd3d2241`
- Diff scope read: GitHub Pull Request files API returned 6 files, +132/-13, 188 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "[Bugfix][Model] Support LoRA on Qwen3 Output Embedding". The diff centers on `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Purpose This PR adds support for LoRA on the embed/unembed layers for Qwen3 dense/MoE models. It is a simplified version of #26115 that removes the changes for supporting zer...
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
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py`, `vllm/transformers_utils/configs/colqwen3.py`; associated commits `d1ea65d0a1c6`
- Diff scope read: GitHub Pull Request files API returned 10 files, +935/-0, 982 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "[new model] add COLQwen3 code & Inference". The diff centers on `vllm/model_executor/models/colqwen3.py`, `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/transformers_utils/configs/colqwen3.py`. PR body context: ## Purpose Add native support for **ColQwen3** multi-modal late interaction models in vLLM. ColPali (arXiv:2407.01449) introduces a ColBERT-style multi-vector retrieval approach...
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
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py`; associated commits `5719a4e4e601`
- Diff scope read: GitHub Pull Request files API returned 10 files, +532/-66, 843 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR adds or enables a model support/runtime surface. Title: "[Frontend] Support multimodal inputs for late-interaction scoring (ColQwen3) + NewModel: nvidia/nemotron-colembed". The diff centers on `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py`. PR body context: ## Purpose Follow-up to #34398 (`[new model] add COLQwen3 code & Inference`). PR #34398 added native support for ColQwen3 multi-modal late interaction models, but the `/score` a...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `c8c3935b7013`
- Diff scope read: GitHub Pull Request files API returned 3 files, +129/-36, 221 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix][Model] Fix FP8 k_scale/v_scale not loaded for Qwen3-MoE". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: FP8 KV cache scales from llm-compressor checkpoints (e.g. `qkv_proj.k_scale`) were silently dropped during weight loading in Qwen3MoeModel, causing fallback to scale=1.0 and acc...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_moe.py`; associated commits `342c58bc548f`
- Diff scope read: GitHub Pull Request files API returned 1 files, +13/-5, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Core, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[BugFix]fix Qwen3 MoE call gate twice". The diff centers on `vllm/model_executor/models/qwen3_moe.py`. PR body context: ## Purpose Qwen3 MoE model will call gate gemm twice, we find this in xpu kernel profiling. thanks @zufangzhu raise this. see discussion here https://github.com/vllm-project/vll...
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
