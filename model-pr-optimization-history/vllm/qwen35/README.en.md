# vllm Qwen3.5 Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.
- Diffusion model families have been removed from this history set and are no longer part of model optimization skills.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/pooling/score/colqwen3_5_rerank_online.py` | [#36887](https://github.com/vllm-project/vllm/pull/36887) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml` | [#38083](https://github.com/vllm-project/vllm/pull/38083) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml` | [#38083](https://github.com/vllm-project/vllm/pull/38083) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml` | [#38664](https://github.com/vllm-project/vllm/pull/38664) |
| `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` | [#38083](https://github.com/vllm-project/vllm/pull/38083), [#38632](https://github.com/vllm-project/vllm/pull/38632) |
| `tests/evals/gsm8k/configs/models-qwen35-blackwell.txt` | [#38083](https://github.com/vllm-project/vllm/pull/38083) |
| `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` | [#38155](https://github.com/vllm-project/vllm/pull/38155), [#38664](https://github.com/vllm-project/vllm/pull/38664) |
| `tests/lora/test_qwen35_densemodel_lora.py` | [#37816](https://github.com/vllm-project/vllm/pull/37816) |
| `tests/models/multimodal/pooling/test_colqwen3_5.py` | [#36887](https://github.com/vllm-project/vllm/pull/36887) |
| `vllm/model_executor/models/colqwen3_5.py` | [#36887](https://github.com/vllm-project/vllm/pull/36887) |
| `vllm/model_executor/models/qwen3_5.py` | [#34110](https://github.com/vllm-project/vllm/pull/34110), [#34198](https://github.com/vllm-project/vllm/pull/34198), [#34200](https://github.com/vllm-project/vllm/pull/34200), [#34313](https://github.com/vllm-project/vllm/pull/34313), [#34489](https://github.com/vllm-project/vllm/pull/34489), [#34492](https://github.com/vllm-project/vllm/pull/34492), [#34512](https://github.com/vllm-project/vllm/pull/34512), [#34683](https://github.com/vllm-project/vllm/pull/34683), [#34697](https://github.com/vllm-project/vllm/pull/34697), [#34719](https://github.com/vllm-project/vllm/pull/34719), [#34723](https://github.com/vllm-project/vllm/pull/34723), [#35617](https://github.com/vllm-project/vllm/pull/35617), ... (18 total) |
| `vllm/model_executor/models/qwen3_5_mtp.py` | [#34110](https://github.com/vllm-project/vllm/pull/34110), [#34512](https://github.com/vllm-project/vllm/pull/34512), [#35581](https://github.com/vllm-project/vllm/pull/35581), [#37114](https://github.com/vllm-project/vllm/pull/37114), [#38832](https://github.com/vllm-project/vllm/pull/38832) |
| `vllm/transformers_utils/configs/qwen3_5.py` | [#34512](https://github.com/vllm-project/vllm/pull/34512), [#34554](https://github.com/vllm-project/vllm/pull/34554), [#34604](https://github.com/vllm-project/vllm/pull/34604), [#34610](https://github.com/vllm-project/vllm/pull/34610) |
| `vllm/transformers_utils/configs/qwen3_5_moe.py` | [#34512](https://github.com/vllm-project/vllm/pull/34512), [#34554](https://github.com/vllm-project/vllm/pull/34554), [#34604](https://github.com/vllm-project/vllm/pull/34604), [#34610](https://github.com/vllm-project/vllm/pull/34610) |

## PR Coverage Summary

- Git-traced PRs: 29
- Extra PRs preserved from existing docs: 2
- Total PRs in this document: 31
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-02-09 | [#34110](https://github.com/vllm-project/vllm/pull/34110) | merged | [MODEL] Adding Support for Qwen3.5 Models | `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py` |
| 2026-02-10 | [#34198](https://github.com/vllm-project/vllm/pull/34198) | merged | [Bugfix] Adopt `ChunkGatedDeltaRule` for Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-10 | [#34200](https://github.com/vllm-project/vllm/pull/34200) | merged | [Bugfix] Fix mamba cache dtype for Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-11 | [#34313](https://github.com/vllm-project/vllm/pull/34313) | merged | [Bugfix] Fix weight naming in Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-13 | [#34489](https://github.com/vllm-project/vllm/pull/34489) | merged | [Bugfix] Fix mamba state dtype setting for Qwen3-Next and Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-13 | [#34512](https://github.com/vllm-project/vllm/pull/34512) | merged | [Misc] Port Qwen3.5 Configs | `vllm/transformers_utils/configs/qwen3_5_moe.py`, `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-14 | [#34554](https://github.com/vllm-project/vllm/pull/34554) | merged | [Bugfix] Fix Qwen3.5 config loading | `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py` |
| 2026-02-16 | [#34604](https://github.com/vllm-project/vllm/pull/34604) | merged | [Misc] fix qwen3.5 config | `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py` |
| 2026-02-16 | [#34610](https://github.com/vllm-project/vllm/pull/34610) | merged | Revert "[Misc] fix qwen3.5 config" | `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py` |
| 2026-02-16 | [#34492](https://github.com/vllm-project/vllm/pull/34492) | merged | [Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-17 | [#34683](https://github.com/vllm-project/vllm/pull/34683) | merged | Revert "[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj" | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-18 | [#34723](https://github.com/vllm-project/vllm/pull/34723) | merged | [Bugfix] Fix prefix creation for Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-18 | [#34697](https://github.com/vllm-project/vllm/pull/34697) | merged | [Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-19 | [#34719](https://github.com/vllm-project/vllm/pull/34719) | merged | [Bugfix] Qwen3.5 kv-scale weight remapping | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-26 | [#35289](https://github.com/vllm-project/vllm/pull/35289) | merged | [Bugfix] [Qwen3.5]Fix Qwen3.5 FP8 quantization: tuple shard_id weight loading | `vllm/model_executor/layers/linear.py` |
| 2026-02-28 | [#35581](https://github.com/vllm-project/vllm/pull/35581) | merged | Fix Qwen3_5MTP packed_modules_mapping for gate_up_proj | `vllm/model_executor/models/qwen3_5_mtp.py` |
| 2026-03-01 | [#35617](https://github.com/vllm-project/vllm/pull/35617) | merged | [Bugfix][Model] Fix Qwen3.5/Qwen3Next ignoring --dtype flag on older GPUs | `vllm/model_executor/models/qwen3_5.py` |
| 2026-03-11 | [#36658](https://github.com/vllm-project/vllm/pull/36658) | merged | Add: Eagle3 support for Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-03-17 | [#36887](https://github.com/vllm-project/vllm/pull/36887) | merged | [Model] Add ColQwen3.5 4.5B support | `vllm/model_executor/models/colqwen3_5.py`, `tests/models/multimodal/pooling/test_colqwen3_5.py`, `examples/pooling/score/colqwen3_5_rerank_online.py` |
| 2026-03-19 | [#37448](https://github.com/vllm-project/vllm/pull/37448) | merged | Fix AttributeError in Qwen3.5 GDN layers with quantized models | `vllm/model_executor/models/qwen3_5.py` |
| 2026-03-20 | [#36976](https://github.com/vllm-project/vllm/pull/36976) | merged | [Bugfix][LoRA] Fix Qwen35 LoRA | `vllm/model_executor/models/qwen3_5.py` |
| 2026-03-23 | [#37816](https://github.com/vllm-project/vllm/pull/37816) | merged | [CI/Build][LoRA] Update Qwen35 LoRA testing | `tests/lora/test_qwen35_densemodel_lora.py` |
| 2026-03-26 | [#38083](https://github.com/vllm-project/vllm/pull/38083) | merged | [Bugfix] Fix DeepGemm E8M0 accuracy degradation for Qwen3.5 FP8 on Blackwell | `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml` |
| 2026-03-26 | [#38155](https://github.com/vllm-project/vllm/pull/38155) | merged | [ROCm][CI] Add LM Eval Qwen3.5 Models test for MI355 | `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` |
| 2026-03-27 | [#37975](https://github.com/vllm-project/vllm/pull/37975) | merged | [Model] Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-03-31 | [#38632](https://github.com/vllm-project/vllm/pull/38632) | merged | [CI] fix LM Eval Qwen3.5 Models (B200) | `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` |
| 2026-04-03 | [#38832](https://github.com/vllm-project/vllm/pull/38832) | merged | [Bugfix] Fix NVFP4+MTP crash: force unquantized mtp.fc for Qwen3.5 | `vllm/model_executor/models/qwen3_5_mtp.py` |
| 2026-04-03 | [#38664](https://github.com/vllm-project/vllm/pull/38664) | merged | [CI][ROCm] Add Qwen3.5-35B-A3B-MXFP4 model eval into CI | `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml`, `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` |
| 2026-04-03 | [#38927](https://github.com/vllm-project/vllm/pull/38927) | merged | [Bugfix][LoRA] Fix missing in_proj_z in Qwen3_5ForConditionalGenerati… | `vllm/model_executor/models/qwen3_5.py` |
| 2026-04-08 | [#39181](https://github.com/vllm-project/vllm/pull/39181) | merged | [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next | `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py` |
| 2026-04-21 | [#37114](https://github.com/vllm-project/vllm/pull/37114) | merged | [Bugfix] LoRA: extend expert base_layer loading to Qwen3.5 and Step3.x | `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py` |

## Per-PR Diff Audit Cards

### PR #34110 - [MODEL] Adding Support for Qwen3.5 Models

- Link: https://github.com/vllm-project/vllm/pull/34110
- Status/date: merged / 2026-02-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`; associated commits `9562912cead1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +1501/-9, 1631 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[MODEL] Adding Support for Qwen3.5 Models". The diff centers on `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`. PR body context: ## Purpose This PR adds model support for the upcoming Qwen3.5 models, including both dense and MoE variants. 🫡 Many thanks to @wulipc and @sighingnow for model verification and...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` added +993/-0 (993 lines); hunks: -0,0 +1,993; symbols: Qwen3_5ProcessingInfo, get_hf_config, Qwen3_5MoeProcessingInfo, Qwen3_5GatedDeltaNet, touching `Qwen3_5ProcessingInfo, get_hf_config, Qwen3_5MoeProcessingInfo`; `vllm/model_executor/models/qwen3_5_mtp.py` added +447/-0 (447 lines); hunks: -0,0 +1,447; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward, touching `Qwen3_5MultiTokenPredictor, __init__, embed_input_ids`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` added +993/-0 (993 lines); hunks: -0,0 +1,993; symbols: Qwen3_5ProcessingInfo, get_hf_config, Qwen3_5MoeProcessingInfo, Qwen3_5GatedDeltaNet
  - `vllm/model_executor/models/qwen3_5_mtp.py` added +447/-0 (447 lines); hunks: -0,0 +1,447; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -0,0 +1,993 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The vLLM team.
+# Copyright 2025 The Qwen Team.
+# Copyright 2025 The HuggingFace Inc. team.
+# All rights reserved.
diff -- vllm/model_executor/models/qwen3_5_mtp.py
@@ -0,0 +1,447 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only Qwen3_5 MTP model."""
+import typing
+from collections.abc import Callable, Iterable
+import torch
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` added +993/-0; `vllm/model_executor/models/qwen3_5_mtp.py` added +447/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34198 - [Bugfix] Adopt `ChunkGatedDeltaRule` for Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/34198
- Status/date: merged / 2026-02-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `047a457fa4af`
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[Bugfix] Adopt `ChunkGatedDeltaRule` for Qwen3.5". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Purpose `ChunkGatedDeltaRule` was recently added in #32846. Qwen3.5 missed this in its initialization which causes an error since its `_forward_core` inherits from Qwen3-Next...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +3/-0 (3 lines); hunks: -99,6 +99,7; -268,6 +269,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +3/-0 (3 lines); hunks: -99,6 +99,7; -268,6 +269,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -99,6 +99,7 @@
+    ChunkGatedDeltaRule,
@@ -268,6 +269,8 @@ def __init__(
+        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34200 - [Bugfix] Fix mamba cache dtype for Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/34200
- Status/date: merged / 2026-02-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `9615575afc0d`
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix mamba cache dtype for Qwen3.5". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Purpose Qwen3.5 uses float32 for mamba cache dtype and it's rather inconvenient to ask users to pass ` --mamba-cache-dtype float32` every single time. Since it's not part of...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +2/-1 (3 lines); hunks: -867,8 +867,9 @@ def get_mamba_state_dtype_from_config(; symbols: get_mamba_state_dtype_from_config, touching `get_mamba_state_dtype_from_config`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +2/-1 (3 lines); hunks: -867,8 +867,9 @@ def get_mamba_state_dtype_from_config(; symbols: get_mamba_state_dtype_from_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -867,8 +867,9 @@ def get_mamba_state_dtype_from_config(
+        mamba_ssm_dtype = vllm_config.model_config.hf_text_config.mamba_ssm_dtype
-            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
+            vllm_config.model_config.dtype, mamba_ssm_dtype
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34313 - [Bugfix] Fix weight naming in Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/34313
- Status/date: merged / 2026-02-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `0b20469c627e`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix weight naming in Qwen3.5". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Purpose ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, such as "Fix some issue (link existing issues...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +1/-1 (2 lines); hunks: -206,7 +206,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +1/-1 (2 lines); hunks: -206,7 +206,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -206,7 +206,7 @@ def __init__(
-            prefix=f"{prefix}.in_proj_ba",
+            prefix=f"{prefix}.in_proj_b",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34489 - [Bugfix] Fix mamba state dtype setting for Qwen3-Next and Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/34489
- Status/date: merged / 2026-02-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `eea3024f43e0`
- Diff scope read: GitHub Pull Request files API returned 4 files, +42/-6, 91 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix mamba state dtype setting for Qwen3-Next and Qwen3.5". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Purpose Previously conv and ssm state dtypes are coupled for Qwen3-Next, and therefore affected Qwen3.5 which inherits from it. This PR fixes the dtype setting for both model...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +3/-2 (5 lines); hunks: -870,9 +870,10 @@ def get_mamba_state_dtype_from_config(; symbols: get_mamba_state_dtype_from_config, touching `get_mamba_state_dtype_from_config`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +3/-2 (5 lines); hunks: -870,9 +870,10 @@ def get_mamba_state_dtype_from_config(; symbols: get_mamba_state_dtype_from_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -870,9 +870,10 @@ def get_mamba_state_dtype_from_config(
-        mamba_ssm_dtype = vllm_config.model_config.hf_text_config.mamba_ssm_dtype
-            vllm_config.model_config.dtype, mamba_ssm_dtype
+            vllm_config.model_config.dtype,
+            vllm_config.cache_config.mamba_cache_dtype,
+            vllm_config.cache_config.mamba_ssm_cache_dtype,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/models/config.py`, `vllm/model_executor/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34512 - [Misc] Port Qwen3.5 Configs

- Link: https://github.com/vllm-project/vllm/pull/34512
- Status/date: merged / 2026-02-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`, `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`; associated commits `5885e330efea`
- Diff scope read: GitHub Pull Request files API returned 6 files, +410/-12, 473 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR extends deployment docs, tests, or CI coverage. Title: "[Misc] Port Qwen3.5 Configs". The diff centers on `vllm/transformers_utils/configs/qwen3_5_moe.py`, `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Purpose so that users don't need to install transformers ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the...
- Key implementation: `vllm/transformers_utils/configs/qwen3_5_moe.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: Qwen3_5MoeTextConfig, __init__, Qwen3_5MoeVisionConfig, Qwen3_5MoeConfig, touching `Qwen3_5MoeTextConfig, __init__, Qwen3_5MoeVisionConfig`; `vllm/transformers_utils/configs/qwen3_5.py` added +189/-0 (189 lines); hunks: -0,0 +1,189; symbols: Qwen3_5TextConfig, __init__, Qwen3_5VisionConfig, Qwen3_5Config, touching `Qwen3_5TextConfig, __init__, Qwen3_5VisionConfig`; `vllm/model_executor/models/qwen3_5.py` modified +8/-8 (16 lines); hunks: -31,14 +31,6; -87,6 +79,14; `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-4 (6 lines); hunks: -7,10 +7,6; -27,6 +23,8.
- Code diff details:
  - `vllm/transformers_utils/configs/qwen3_5_moe.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: Qwen3_5MoeTextConfig, __init__, Qwen3_5MoeVisionConfig, Qwen3_5MoeConfig
  - `vllm/transformers_utils/configs/qwen3_5.py` added +189/-0 (189 lines); hunks: -0,0 +1,189; symbols: Qwen3_5TextConfig, __init__, Qwen3_5VisionConfig, Qwen3_5Config
  - `vllm/model_executor/models/qwen3_5.py` modified +8/-8 (16 lines); hunks: -31,14 +31,6; -87,6 +79,14
  - `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-4 (6 lines); hunks: -7,10 +7,6; -27,6 +23,8
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/qwen3_5_moe.py
@@ -0,0 +1,201 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
+# All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
diff -- vllm/transformers_utils/configs/qwen3_5.py
@@ -0,0 +1,189 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
+# All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
diff -- vllm/model_executor/models/qwen3_5.py
@@ -31,14 +31,6 @@
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/qwen3_5_moe.py` added +201/-0; `vllm/transformers_utils/configs/qwen3_5.py` added +189/-0; `vllm/model_executor/models/qwen3_5.py` modified +8/-8; `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`, `vllm/transformers_utils/config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34554 - [Bugfix] Fix Qwen3.5 config loading

- Link: https://github.com/vllm-project/vllm/pull/34554
- Status/date: merged / 2026-02-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`; associated commits `2f186635cbcb`
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-10, 72 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix Qwen3.5 config loading". The diff centers on `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`. PR body context: ## Purpose #34512 actually didn't port it completely correctly and resulted in degradation - this PR fixes it. ## Test Plan ## Test Result --- Essential Elements of an Effective...
- Key implementation: `vllm/transformers_utils/configs/qwen3_5.py` modified +9/-5 (14 lines); hunks: -72,10 +72,6 @@ def __init__(; -111,6 +107,13 @@ def __init__(; symbols: __init__, Qwen3_5VisionConfig, touching `__init__, Qwen3_5VisionConfig`; `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +9/-5 (14 lines); hunks: -79,10 +79,6 @@ def __init__(; -123,6 +119,13 @@ def __init__(; symbols: __init__, Qwen3_5MoeVisionConfig, touching `__init__, Qwen3_5MoeVisionConfig`.
- Code diff details:
  - `vllm/transformers_utils/configs/qwen3_5.py` modified +9/-5 (14 lines); hunks: -72,10 +72,6 @@ def __init__(; -111,6 +107,13 @@ def __init__(; symbols: __init__, Qwen3_5VisionConfig
  - `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +9/-5 (14 lines); hunks: -79,10 +79,6 @@ def __init__(; -123,6 +119,13 @@ def __init__(; symbols: __init__, Qwen3_5MoeVisionConfig
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/qwen3_5.py
@@ -72,10 +72,6 @@ def __init__(
-        self.pad_token_id = pad_token_id
-        self.bos_token_id = bos_token_id
-        self.eos_token_id = eos_token_id
-        self.tie_word_embeddings = tie_word_embeddings
@@ -111,6 +107,13 @@ def __init__(
+        # Set these AFTER super().__init__() because transformers v4's
diff -- vllm/transformers_utils/configs/qwen3_5_moe.py
@@ -79,10 +79,6 @@ def __init__(
-        self.pad_token_id = pad_token_id
-        self.bos_token_id = bos_token_id
-        self.eos_token_id = eos_token_id
-        self.tie_word_embeddings = tie_word_embeddings
@@ -123,6 +119,13 @@ def __init__(
+        # Set these AFTER super().__init__() because transformers v4's
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/qwen3_5.py` modified +9/-5; `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +9/-5
- Risk and verification: Runtime changes concentrate in `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34604 - [Misc] fix qwen3.5 config

- Link: https://github.com/vllm-project/vllm/pull/34604
- Status/date: merged / 2026-02-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`; associated commits `9521002f0ace`
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-4, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Misc] fix qwen3.5 config". The diff centers on `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`. PR body context: ## Purpose fix qwen3.5 config loading @ywang96 before:
- Key implementation: `vllm/transformers_utils/configs/qwen3_5.py` modified +2/-2 (4 lines); hunks: -68,10 +68,10 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +2/-2 (4 lines); hunks: -75,10 +75,10 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/transformers_utils/configs/qwen3_5.py` modified +2/-2 (4 lines); hunks: -68,10 +68,10 @@ def __init__(; symbols: __init__
  - `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +2/-2 (4 lines); hunks: -75,10 +75,10 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/qwen3_5.py
@@ -68,10 +68,10 @@ def __init__(
-        kwargs["ignore_keys_at_rope_validation"] = [
+        kwargs["ignore_keys_at_rope_validation"] = {
-        ]
+        }
diff -- vllm/transformers_utils/configs/qwen3_5_moe.py
@@ -75,10 +75,10 @@ def __init__(
-        kwargs["ignore_keys_at_rope_validation"] = [
+        kwargs["ignore_keys_at_rope_validation"] = {
-        ]
+        }
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/qwen3_5.py` modified +2/-2; `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34610 - Revert "[Misc] fix qwen3.5 config"

- Link: https://github.com/vllm-project/vllm/pull/34610
- Status/date: merged / 2026-02-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`; associated commits `b5475d053442`
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-4, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Revert "[Misc] fix qwen3.5 config"". The diff centers on `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`. PR body context: Reverts vllm-project/vllm#34604 It was meant for transformers v5
- Key implementation: `vllm/transformers_utils/configs/qwen3_5.py` modified +2/-2 (4 lines); hunks: -68,10 +68,10 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +2/-2 (4 lines); hunks: -75,10 +75,10 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/transformers_utils/configs/qwen3_5.py` modified +2/-2 (4 lines); hunks: -68,10 +68,10 @@ def __init__(; symbols: __init__
  - `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +2/-2 (4 lines); hunks: -75,10 +75,10 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/qwen3_5.py
@@ -68,10 +68,10 @@ def __init__(
-        kwargs["ignore_keys_at_rope_validation"] = {
+        kwargs["ignore_keys_at_rope_validation"] = [
-        }
+        ]
diff -- vllm/transformers_utils/configs/qwen3_5_moe.py
@@ -75,10 +75,10 @@ def __init__(
-        kwargs["ignore_keys_at_rope_validation"] = {
+        kwargs["ignore_keys_at_rope_validation"] = [
-        }
+        ]
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/qwen3_5.py` modified +2/-2; `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34492 - [Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj

- Link: https://github.com/vllm-project/vllm/pull/34492
- Status/date: merged / 2026-02-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `3bb4e4311c6d`
- Diff scope read: GitHub Pull Request files API returned 3 files, +87/-182, 404 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR optimizes an inference path or backend selection. Title: "[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Purpose - Fuse Qwen3.5's GDN qkvz proj and ba proj to get better performance gain. ## Test Plan ## Test Result **Main branch** **PR** --- Essential Elements of an Effective P...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +32/-166 (198 lines); hunks: -30,36 +30,20; -73,11 +57,8; symbols: get_hf_config, Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering, touching `get_hf_config, Qwen3_5GatedDeltaNet, __init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +32/-166 (198 lines); hunks: -30,36 +30,20; -73,11 +57,8; symbols: get_hf_config, Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -30,36 +30,20 @@
-from transformers.activations import ACT2FN
-    CacheConfig,
-    ModelConfig,
-    SpeculativeConfig,
-    get_current_vllm_config,
-    divide,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +32/-166
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34683 - Revert "[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj"

- Link: https://github.com/vllm-project/vllm/pull/34683
- Status/date: merged / 2026-02-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `1d65283e95f4`
- Diff scope read: GitHub Pull Request files API returned 3 files, +182/-87, 402 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Revert "[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj"". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: FIX https://github.com/vllm-project/vllm/issues/34640 Reverts vllm-project/vllm#34492 It makes the output of qwen3 next as random symbols local-chat-completions (model=Qwen/Qwen...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +166/-32 (198 lines); hunks: -30,20 +30,36; -57,8 +73,11; symbols: get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering, __init__, touching `get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +166/-32 (198 lines); hunks: -30,20 +30,36; -57,8 +73,11; symbols: get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -30,20 +30,36 @@
+from transformers.activations import ACT2FN
+    CacheConfig,
+    ModelConfig,
+    SpeculativeConfig,
+    get_current_vllm_config,
+    divide,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +166/-32
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34723 - [Bugfix] Fix prefix creation for Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/34723
- Status/date: merged / 2026-02-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `909b14719725`
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-5, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix prefix creation for Qwen3.5". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Purpose It seems that the prefix creation for Qwen3.5 is incorrect since it adds a "model" module after the "language_model" module in the HF checkpoint. This will cause issu...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +6/-5 (11 lines); hunks: -542,9 +542,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -620,7 +621,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: __init__, get_expert_mapping, Qwen3_5ForConditionalGeneration, touching `__init__, get_expert_mapping, Qwen3_5ForConditionalGeneration`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +6/-5 (11 lines); hunks: -542,9 +542,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -620,7 +621,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: __init__, get_expert_mapping, Qwen3_5ForConditionalGeneration
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -542,9 +542,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        self.model = Qwen3_5Model(
-            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
-        )
+        # Deal with the case where the prefix is already "language_model" since
+        # Qwen/Qwen3.5-397B-A17B has naming like: model.language_model.layers.0
+        model_prefix = prefix if "model" in prefix else "model"
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +6/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34697 - [Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion

- Link: https://github.com/vllm-project/vllm/pull/34697
- Status/date: merged / 2026-02-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `c0bd8b13da36`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +102/-192, 477 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Purpose - Redo https://github.com/vllm-project/vllm/pull/34683 and fix the root issue - Actualy, Qwen3-Next's qkvz_proj output_sizes should be `output_sizes=[sum((key_dim, ke...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +43/-170 (213 lines); hunks: -30,36 +30,20; -73,11 +57,8; symbols: get_hf_config, Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering, touching `get_hf_config, Qwen3_5GatedDeltaNet, __init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +43/-170 (213 lines); hunks: -30,36 +30,20; -73,11 +57,8; symbols: get_hf_config, Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -30,36 +30,20 @@
-from transformers.activations import ACT2FN
-    CacheConfig,
-    ModelConfig,
-    SpeculativeConfig,
-    get_current_vllm_config,
-    divide,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +43/-170
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34719 - [Bugfix] Qwen3.5 kv-scale weight remapping

- Link: https://github.com/vllm-project/vllm/pull/34719
- Status/date: merged / 2026-02-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `6fff24f30fe2`
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-0, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Qwen3.5 kv-scale weight remapping". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Purpose - Fix for loading KV cache scaling factors by remapping the names, similar to how it is done in `models/qwen3_next.py` https://github.com/vllm-project/vllm/pull/32199...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +7/-0 (7 lines); hunks: -57,6 +57,7; -397,6 +398,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +7/-0 (7 lines); hunks: -57,6 +57,7; -397,6 +398,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -57,6 +57,7 @@
+    maybe_remap_kv_scale_name,
@@ -397,6 +398,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            # Remapping the name of FP8 kv-scale.
+            if name.endswith("scale"):
+                name = maybe_remap_kv_scale_name(name, params_dict)
+                if name is None:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35289 - [Bugfix] [Qwen3.5]Fix Qwen3.5 FP8 quantization: tuple shard_id weight loading

- Link: https://github.com/vllm-project/vllm/pull/35289
- Status/date: merged / 2026-02-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +19/-8, 55 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] [Qwen3.5]Fix Qwen3.5 FP8 quantization: tuple shard_id weight loading". The diff centers on `vllm/model_executor/layers/linear.py`. PR body context: ## Purpose Fix #35287 Fix online FP8 quantization (`--quantization fp8`) for Qwen3.5 models. ### Bug — Weight loading crash: `NotImplementedError` for tuple `shard_id` Qwen3.5's...
- Key implementation: `vllm/model_executor/layers/linear.py` modified +19/-8 (27 lines); hunks: -731,16 +731,16 @@ def weight_loader(; -768,7 +768,7 @@ def weight_loader(; symbols: weight_loader, touching `weight_loader`.
- Code diff details:
  - `vllm/model_executor/layers/linear.py` modified +19/-8 (27 lines); hunks: -731,16 +731,16 @@ def weight_loader(; -768,7 +768,7 @@ def weight_loader(; symbols: weight_loader
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/linear.py
@@ -731,16 +731,16 @@ def weight_loader(
-        # FIXME(Isotr0py): Enable tuple shard_id for BNB quantization.
-        if isinstance(loaded_shard_id, tuple):
-            raise NotImplementedError(
-                "Shard id with multiple indices is not supported in weight_loader, "
-                "please use weight_loader_v2 instead."
-            )
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/linear.py` modified +19/-8
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35581 - Fix Qwen3_5MTP packed_modules_mapping for gate_up_proj

- Link: https://github.com/vllm-project/vllm/pull/35581
- Status/date: merged / 2026-02-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5_mtp.py`; associated commits `63d7972f13d1`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix Qwen3_5MTP packed_modules_mapping for gate_up_proj". The diff centers on `vllm/model_executor/models/qwen3_5_mtp.py`. PR body context: ### What does this PR do? Fixes the `packed_modules_mapping` in `Qwen3_5MTP` class where `gate_up_proj` incorrectly included `down_proj` instead of `gate_proj`. ### Why is this...
- Key implementation: `vllm/model_executor/models/qwen3_5_mtp.py` modified +1/-1 (2 lines); hunks: -339,7 +339,7 @@ class Qwen3_5MTP(nn.Module, SupportsMultiModal):; symbols: Qwen3_5MTP, __init__, touching `Qwen3_5MTP, __init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5_mtp.py` modified +1/-1 (2 lines); hunks: -339,7 +339,7 @@ class Qwen3_5MTP(nn.Module, SupportsMultiModal):; symbols: Qwen3_5MTP, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5_mtp.py
@@ -339,7 +339,7 @@ class Qwen3_5MTP(nn.Module, SupportsMultiModal):
-        "gate_up_proj": ["up_proj", "down_proj"],
+        "gate_up_proj": ["gate_proj", "up_proj"],
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5_mtp.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35617 - [Bugfix][Model] Fix Qwen3.5/Qwen3Next ignoring --dtype flag on older GPUs

- Link: https://github.com/vllm-project/vllm/pull/35617
- Status/date: merged / 2026-03-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `afd089f231d7`
- Diff scope read: GitHub Pull Request files API returned 2 files, +0/-5, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[Bugfix][Model] Fix Qwen3.5/Qwen3Next ignoring --dtype flag on older GPUs". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Summary - **Bug**: Qwen3.5 and Qwen3Next models ignore `--dtype float16`, creating bfloat16 parameters that crash on GPUs without bfloat16 support (e.g. 2080 Ti) - **Root cau...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +0/-2 (2 lines); hunks: -274,15 +274,13 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +0/-2 (2 lines); hunks: -274,15 +274,13 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -274,15 +274,13 @@ def __init__(
-                    dtype=config.dtype,
-                    dtype=config.dtype,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +0/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36658 - Add: Eagle3 support for Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/36658
- Status/date: merged / 2026-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `9d07a3d6e472`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +25/-2, 83 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "Add: Eagle3 support for Qwen3.5". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: This PR adds support for EAGLE-3 speculative decoding to `Qwen3.5`, enabling faster inference with draft models like `BLR2/Qwen3.5-9B-Eagle3-ShareGPT`. ## Changes ### Modified F...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +11/-0 (11 lines); hunks: -75,6 +75,7; -353,6 +354,8 @@ def get_layer(prefix: str):; symbols: get_layer, load_fused_expert_weights, load_weights, Qwen3_5ForCausalLMBase, touching `get_layer, load_fused_expert_weights, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +11/-0 (11 lines); hunks: -75,6 +75,7; -353,6 +354,8 @@ def get_layer(prefix: str):; symbols: get_layer, load_fused_expert_weights, load_weights, Qwen3_5ForCausalLMBase
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -75,6 +75,7 @@
+    SupportsEagle3,
@@ -353,6 +354,8 @@ def get_layer(prefix: str):
+        self.aux_hidden_state_layers: tuple[int, ...] = ()
@@ -536,6 +539,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+    SupportsEagle3,
@@ -592,6 +596,13 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +11/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36887 - [Model] Add ColQwen3.5 4.5B support

- Link: https://github.com/vllm-project/vllm/pull/36887
- Status/date: merged / 2026-03-17
- Trace source: `git log --name-only -- <model-files>` found it through `examples/pooling/score/colqwen3_5_rerank_online.py`, `tests/models/multimodal/pooling/test_colqwen3_5.py`, `vllm/model_executor/models/colqwen3_5.py`; associated commits `c0745a851a4f`
- Diff scope read: GitHub Pull Request files API returned 8 files, +579/-0, 619 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[Model] Add ColQwen3.5 4.5B support". The diff centers on `vllm/model_executor/models/colqwen3_5.py`, `tests/models/multimodal/pooling/test_colqwen3_5.py`, `examples/pooling/score/colqwen3_5_rerank_online.py`. PR body context: ## Purpose Add support for **ColQwen3.5** late interaction model for multi-modal document retrieval and reranking. ColQwen3.5 extends the ColPali architecture using the Qwen3.5...
- Key implementation: `vllm/model_executor/models/colqwen3_5.py` added +246/-0 (246 lines); hunks: -0,0 +1,246; symbols: ColQwen3_5ProcessingInfo, get_hf_config, get_hf_processor, _supports_video, touching `ColQwen3_5ProcessingInfo, get_hf_config, get_hf_processor`; `tests/models/multimodal/pooling/test_colqwen3_5.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: _run_token_embed_test, _run_late_interaction_test, _run_relevance_test, test_colqwen3_5_token_embed, touching `_run_token_embed_test, _run_late_interaction_test, _run_relevance_test`; `examples/pooling/score/colqwen3_5_rerank_online.py` added +130/-0 (130 lines); hunks: -0,0 +1,130; symbols: rerank_text, score_text, score_text_top_n, main, touching `rerank_text, score_text, score_text_top_n`.
- Code diff details:
  - `vllm/model_executor/models/colqwen3_5.py` added +246/-0 (246 lines); hunks: -0,0 +1,246; symbols: ColQwen3_5ProcessingInfo, get_hf_config, get_hf_processor, _supports_video
  - `tests/models/multimodal/pooling/test_colqwen3_5.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: _run_token_embed_test, _run_late_interaction_test, _run_relevance_test, test_colqwen3_5_token_embed
  - `examples/pooling/score/colqwen3_5_rerank_online.py` added +130/-0 (130 lines); hunks: -0,0 +1,130; symbols: rerank_text, score_text, score_text_top_n, main
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/colqwen3_5.py
@@ -0,0 +1,246 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+ColQwen3.5 late interaction model for multi-modal retrieval and reranking.
+ColQwen3.5 extends Qwen3.5 with a ColBERT-style late interaction head,
+producing per-token embeddings for both text and image inputs. It uses
diff -- tests/models/multimodal/pooling/test_colqwen3_5.py
@@ -0,0 +1,154 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tests for ColQwen3.5 late interaction model for multi-modal retrieval.
+ColQwen3.5 is a multi-vector retrieval model based on Qwen3.5 backbone with
+ColBERT-style late interaction scoring (MaxSim). It produces per-token
+embeddings for both text and image inputs.
diff -- examples/pooling/score/colqwen3_5_rerank_online.py
@@ -0,0 +1,130 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/colqwen3_5.py` added +246/-0
  - tests: `tests/models/multimodal/pooling/test_colqwen3_5.py` added +154/-0
  - docs: `examples/pooling/score/colqwen3_5_rerank_online.py` added +130/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/pooling/test_colqwen3_5.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37448 - Fix AttributeError in Qwen3.5 GDN layers with quantized models

- Link: https://github.com/vllm-project/vllm/pull/37448
- Status/date: merged / 2026-03-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `4120a05ff1d0`
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-4, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix AttributeError in Qwen3.5 GDN layers with quantized models". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Summary - Replace `self.in_proj_qkvz.weight.shape[0]` and `self.in_proj_ba.weight.shape[0]` with `sum(self.in_proj_qkvz.output_sizes)` and `sum(self.in_proj_ba.output_sizes)`...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +2/-2 (4 lines); hunks: -182,8 +182,8 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +2/-2 (4 lines); hunks: -182,8 +182,8 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -182,8 +182,8 @@ def forward(
-            self.in_proj_qkvz.weight.shape[0],
-            self.in_proj_ba.weight.shape[0],
+            sum(self.in_proj_qkvz.output_sizes) // self.tp_size,
+            sum(self.in_proj_ba.output_sizes) // self.tp_size,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36976 - [Bugfix][LoRA] Fix Qwen35 LoRA

- Link: https://github.com/vllm-project/vllm/pull/36976
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `8fbe3f303fbf`
- Diff scope read: GitHub Pull Request files API returned 5 files, +257/-46, 464 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix][LoRA] Fix Qwen35 LoRA". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Purpose ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, such as "Fix some issue (link existing issues...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +100/-23 (123 lines); hunks: -32,17 +32,18; -130,6 +131,40 @@ def fix_query_key_value_ordering(; symbols: fix_query_key_value_ordering, __init__, create_qkvz_proj, forward, touching `fix_query_key_value_ordering, __init__, create_qkvz_proj`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +100/-23 (123 lines); hunks: -32,17 +32,18; -130,6 +131,40 @@ def fix_query_key_value_ordering(; symbols: fix_query_key_value_ordering, __init__, create_qkvz_proj, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -32,17 +32,18 @@
-from vllm.config import (
-    VllmConfig,
-)
+from vllm.config import VllmConfig
-from vllm.model_executor.layers.linear import MergedColumnParallelLinear
+from vllm.model_executor.layers.linear import (
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +100/-23
- Risk and verification: The diff ships test coverage in `tests/lora/conftest.py`, `tests/lora/test_qwen35_densemoel_lora.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37816 - [CI/Build][LoRA] Update Qwen35 LoRA testing

- Link: https://github.com/vllm-project/vllm/pull/37816
- Status/date: merged / 2026-03-23
- Trace source: `git log --name-only -- <model-files>` found it through `tests/lora/test_qwen35_densemodel_lora.py`; associated commits `1f0d21064137`
- Diff scope read: GitHub Pull Request files API returned 4 files, +369/-135, 529 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[CI/Build][LoRA] Update Qwen35 LoRA testing". The diff centers on `tests/lora/test_qwen35_densemodel_lora.py`. PR body context: ## Purpose ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, such as "Fix some issue (link existing issues...
- Key implementation: `tests/lora/test_qwen35_densemodel_lora.py` added +361/-0 (361 lines); hunks: -0,0 +1,361; symbols: _assert_exact_outputs, _assert_prefix_outputs, _run_text_lora_sample, _run_vl_lora_sample, touching `_assert_exact_outputs, _assert_prefix_outputs, _run_text_lora_sample`.
- Code diff details:
  - `tests/lora/test_qwen35_densemodel_lora.py` added +361/-0 (361 lines); hunks: -0,0 +1,361; symbols: _assert_exact_outputs, _assert_prefix_outputs, _run_text_lora_sample, _run_vl_lora_sample
- Key code excerpts:

```diff
diff -- tests/lora/test_qwen35_densemodel_lora.py
@@ -0,0 +1,361 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from transformers import AutoTokenizer
+import vllm
+import vllm.config
+from vllm.assets.image import ImageAsset
```

- Reviewed files:
  - tests: `tests/lora/test_qwen35_densemodel_lora.py` added +361/-0
- Risk and verification: The diff ships test coverage in `tests/lora/conftest.py`, `tests/lora/test_qwen35_densemodel_lora.py`, `tests/lora/test_qwen35_densemoel_lora.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38083 - [Bugfix] Fix DeepGemm E8M0 accuracy degradation for Qwen3.5 FP8 on Blackwell

- Link: https://github.com/vllm-project/vllm/pull/38083
- Status/date: merged / 2026-03-26
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`, `tests/evals/gsm8k/configs/models-qwen35-blackwell.txt`; associated commits `52069012fe53`
- Diff scope read: GitHub Pull Request files API returned 10 files, +69/-11, 177 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix DeepGemm E8M0 accuracy degradation for Qwen3.5 FP8 on Blackwell". The diff centers on `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml`. PR body context: ## Summary Fixes #37804 — `Qwen/Qwen3.5-35B-A3B-FP8` accuracy drops ~12pp on Blackwell (B200) when DeepGemm is active. DeepGemm's mandatory E8M0 (power-of-2 ceiling) scale forma...
- Key implementation: `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` added +9/-0 (9 lines); hunks: -0,0 +1,9; `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml` modified +2/-1 (3 lines); hunks: -1,5 +1,6; `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml` modified +2/-1 (3 lines); hunks: -1,5 +1,6; `tests/evals/gsm8k/configs/models-qwen35-blackwell.txt` modified +2/-0 (2 lines); hunks: -1 +1,3.
- Code diff details:
  - `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` added +9/-0 (9 lines); hunks: -0,0 +1,9
  - `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml` modified +2/-1 (3 lines); hunks: -1,5 +1,6
  - `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml` modified +2/-1 (3 lines); hunks: -1,5 +1,6
  - `tests/evals/gsm8k/configs/models-qwen35-blackwell.txt` modified +2/-0 (2 lines); hunks: -1 +1,3
- Key code excerpts:

```diff
diff -- tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml
@@ -0,0 +1,9 @@
+model_name: "nvidia/Qwen3.5-397B-A17B-NVFP4"
+accuracy_threshold: 0.88
+tolerance: 0.03
+num_questions: 1319
+num_fewshot: 5
+server_args: >-
diff -- tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml
@@ -1,5 +1,6 @@
-accuracy_threshold: 0.86
+accuracy_threshold: 0.84
+tolerance: 0.03
diff -- tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml
@@ -1,5 +1,6 @@
-accuracy_threshold: 0.86
+accuracy_threshold: 0.79
+tolerance: 0.03
```

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` added +9/-0; `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml` modified +2/-1; `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml` modified +2/-1; `tests/evals/gsm8k/configs/models-qwen35-blackwell.txt` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`, `tests/evals/gsm8k/configs/models-qwen35-blackwell.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38155 - [ROCm][CI] Add LM Eval Qwen3.5 Models test for MI355

- Link: https://github.com/vllm-project/vllm/pull/38155
- Status/date: merged / 2026-03-26
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`; associated commits `9c3ae04bfe65`
- Diff scope read: GitHub Pull Request files API returned 2 files, +25/-0, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[ROCm][CI] Add LM Eval Qwen3.5 Models test for MI355". The diff centers on `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`. PR body context: This PR adds a new CI entry for running Qwen3.5 model evaluation on MI355 GPUs: - Added `LM Eval Qwen3.5 Models (B200-MI355)` entry in `test-amd.yaml` - Created `tests/evals/gsm...
- Key implementation: `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` added +1/-0 (1 lines); hunks: -0,0 +1.
- Code diff details:
  - `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` added +1/-0 (1 lines); hunks: -0,0 +1
- Key code excerpts:

```diff
diff -- tests/evals/gsm8k/configs/models-qwen35-mi355.txt
@@ -0,0 +1 @@
+Qwen3.5-35B-A3B-DEP2.yaml
```

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` added +1/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37975 - [Model] Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/37975
- Status/date: merged / 2026-03-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `a8eab8f30dda`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +1053/-1126, 2304 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR changes model-related implementation. Title: "[Model] Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Purpose Move the GDN (Gated Delta Net) layer implementation from `qwen3_next.py` into a dedicated `gdn_linear_attn.py`, and unify Qwen3Next and Qwen3.5 under a single `GatedD...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +4/-151 (155 lines); hunks: -28,7 +28,6; -40,18 +39,14; symbols: get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering, __init__, touching `get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +4/-151 (155 lines); hunks: -28,7 +28,6; -40,18 +39,14; symbols: get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -28,7 +28,6 @@
-from einops import rearrange
@@ -40,18 +39,14 @@
-from vllm.model_executor.layers.linear import (
-    ColumnParallelLinear,
-    MergedColumnParallelLinear,
-)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +4/-151
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38632 - [CI] fix LM Eval Qwen3.5 Models (B200)

- Link: https://github.com/vllm-project/vllm/pull/38632
- Status/date: merged / 2026-03-31
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`; associated commits `ea7bfde6e40d`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 5 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[CI] fix LM Eval Qwen3.5 Models (B200)". The diff centers on `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`. PR body context: https://buildkite.com/vllm/ci/builds/58942/steps/canvas?sid=019d427b-900f-4987-9ed9-07480b73fc65&tab=output This was introduced in https://github.com/vllm-project/vllm/pull/3827...
- Key implementation: `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` modified +1/-0 (1 lines); hunks: -7,3 +7,4 @@ server_args: >-.
- Code diff details:
  - `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` modified +1/-0 (1 lines); hunks: -7,3 +7,4 @@ server_args: >-
- Key code excerpts:

```diff
diff -- tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml
@@ -7,3 +7,4 @@ server_args: >-
+  --max-num-seqs 512
```

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38832 - [Bugfix] Fix NVFP4+MTP crash: force unquantized mtp.fc for Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/38832
- Status/date: merged / 2026-04-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5_mtp.py`; associated commits `771913e4a024`
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-1, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix NVFP4+MTP crash: force unquantized mtp.fc for Qwen3.5". The diff centers on `vllm/model_executor/models/qwen3_5_mtp.py`. PR body context: ## Description Fix `AssertionError` when loading `nvidia/Qwen3.5-397B-A17B-NVFP4` with `method="mtp"`. The NVFP4 checkpoint stores the entire MTP branch in BF16, but `hf_quant_c...
- Key implementation: `vllm/model_executor/models/qwen3_5_mtp.py` modified +10/-1 (11 lines); hunks: -75,13 +75,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5_mtp.py` modified +10/-1 (11 lines); hunks: -75,13 +75,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5_mtp.py
@@ -75,13 +75,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        # Workaround: mtp.fc is stored as BF16 in NVFP4 checkpoints but is
+        # missing from hf_quant_config.json exclude_modules. Force unquantized.
+        # Ref: https://github.com/vllm-project/vllm/pull/38650
+        # Ref: https://github.com/NVIDIA/Model-Optimizer/pull/1124
+        fc_quant = (
+            None
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5_mtp.py` modified +10/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38664 - [CI][ROCm] Add Qwen3.5-35B-A3B-MXFP4 model eval into CI

- Link: https://github.com/vllm-project/vllm/pull/38664
- Status/date: merged / 2026-04-03
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml`, `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`; associated commits `201d2ea5bfb9`
- Diff scope read: GitHub Pull Request files API returned 2 files, +9/-0, 12 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[CI][ROCm] Add Qwen3.5-35B-A3B-MXFP4 model eval into CI". The diff centers on `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml`, `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`. PR body context: As title, TP2 validated to work locally with https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4. Will mark this PR as ready once the mxfp4 model is made public.
- Key implementation: `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8; `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` modified +1/-0 (1 lines); hunks: -1 +1,2.
- Code diff details:
  - `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8
  - `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` modified +1/-0 (1 lines); hunks: -1 +1,2
- Key code excerpts:

```diff
diff -- tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml
@@ -0,0 +1,8 @@
+model_name: "amd/Qwen3.5-35B-A3B-MXFP4"
+accuracy_threshold: 0.82
+tolerance: 0.03
+num_questions: 1319
+num_fewshot: 5
+server_args: >-
diff -- tests/evals/gsm8k/configs/models-qwen35-mi355.txt
@@ -1 +1,2 @@
+Qwen3.5-35B-A3B-MXFP4-TP2.yaml
```

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml` added +8/-0; `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml`, `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38927 - [Bugfix][LoRA] Fix missing in_proj_z in Qwen3_5ForConditionalGenerati…

- Link: https://github.com/vllm-project/vllm/pull/38927
- Status/date: merged / 2026-04-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `81994e1d0ea6`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[Bugfix][LoRA] Fix missing in_proj_z in Qwen3_5ForConditionalGenerati…". The diff centers on `vllm/model_executor/models/qwen3_5.py`. PR body context: ## Purpose Fix missing `in_proj_z` entry in `Qwen3_5ForConditionalGeneration.update_packed_mapping()` when LoRA is enabled. When LoRA is enabled, GDN layers use separate `in_pro...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +1/-0 (1 lines); hunks: -620,6 +620,7 @@ def update_packed_mapping(self, enable_lora: bool):; symbols: update_packed_mapping, embed_input_ids, touching `update_packed_mapping, embed_input_ids`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +1/-0 (1 lines); hunks: -620,6 +620,7 @@ def update_packed_mapping(self, enable_lora: bool):; symbols: update_packed_mapping, embed_input_ids
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -620,6 +620,7 @@ def update_packed_mapping(self, enable_lora: bool):
+            self.packed_modules_mapping["in_proj_z"] = ["in_proj_z"]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39181 - [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next

- Link: https://github.com/vllm-project/vllm/pull/39181
- Status/date: merged / 2026-04-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-0, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR adds or enables a model support/runtime surface. Title: "[Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next". The diff centers on `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py`. PR body context: ## Purpose Do not shard shared experts weights when sequence parallel is enabled to fix precision issue for Qwen3.5/Qwen3-Next with EP. At present, when sequence_parallel is ena...
- Key implementation: `vllm/model_executor/models/qwen2_moe.py` modified +3/-0 (3 lines); hunks: -80,6 +80,7 @@ def __init__(; -88,6 +89,7 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/model_executor/models/qwen3_next.py` modified +1/-0 (1 lines); hunks: -140,6 +140,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_moe.py` modified +3/-0 (3 lines); hunks: -80,6 +80,7 @@ def __init__(; -88,6 +89,7 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/qwen3_next.py` modified +1/-0 (1 lines); hunks: -140,6 +140,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_moe.py
@@ -80,6 +80,7 @@ def __init__(
+        is_sequence_parallel: bool = False,
@@ -88,6 +89,7 @@ def __init__(
+            disable_tp=is_sequence_parallel,
@@ -96,6 +98,7 @@ def __init__(
+            disable_tp=is_sequence_parallel,
diff -- vllm/model_executor/models/qwen3_next.py
@@ -140,6 +140,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
+                is_sequence_parallel=self.is_sequence_parallel,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_moe.py` modified +3/-0; `vllm/model_executor/models/qwen3_next.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37114 - [Bugfix] LoRA: extend expert base_layer loading to Qwen3.5 and Step3.x

- Link: https://github.com/vllm-project/vllm/pull/37114
- Status/date: merged / 2026-04-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`; associated commits `908a713488db`
- Diff scope read: GitHub Pull Request files API returned 6 files, +34/-16, 104 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] LoRA: extend expert base_layer loading to Qwen3.5 and Step3.x". The diff centers on `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`. PR body context: ## Purpose This PR extends https://github.com/vllm-project/vllm/pull/31104 to the remaining model-specific MoE loaders that still hardcode expert parameter names without `.base_...
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +5/-2 (7 lines); hunks: -306,9 +306,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, touching `load_weights`; `vllm/model_executor/models/qwen3_5_mtp.py` modified +5/-2 (7 lines); hunks: -207,9 +207,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +5/-2 (7 lines); hunks: -306,9 +306,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
  - `vllm/model_executor/models/qwen3_5_mtp.py` modified +5/-2 (7 lines); hunks: -207,9 +207,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -306,9 +306,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        base_layer = (
+            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
+        )
-            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
-            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
+            (f"experts.{base_layer}w13_weight", "experts.gate_up_proj", 0, "w1"),
diff -- vllm/model_executor/models/qwen3_5_mtp.py
@@ -207,9 +207,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        base_layer = (
+            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
+        )
-            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
-            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
+            (f"experts.{base_layer}w13_weight", "experts.gate_up_proj", 0, "w1"),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +5/-2; `vllm/model_executor/models/qwen3_5_mtp.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`, `vllm/model_executor/models/qwen3_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- This version rejects title-only PR lists; every PR must include trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
