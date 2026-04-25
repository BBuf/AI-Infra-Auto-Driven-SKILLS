# vllm GLM-5/5.1 Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.
- Diffusion model families have been removed from this history set and are no longer part of model optimization skills.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| - | No matching implementation file on current main |

## PR Coverage Summary

- Git-traced PRs: 0
- Extra PRs preserved from existing docs: 2
- Total PRs in this document: 2
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-02-09 | [#34124](https://github.com/vllm-project/vllm/pull/34124) | merged | [Model] GLM adaptation | `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py` |
| 2026-02-12 | [#34385](https://github.com/vllm-project/vllm/pull/34385) | merged | [Bugfix] Fix MTP accuracy for GLM-5 | `vllm/v1/spec_decode/eagle.py` |

## Per-PR Diff Audit Cards

### PR #34124 - [Model] GLM adaptation

- Link: https://github.com/vllm-project/vllm/pull/34124
- Status/date: merged / 2026-02-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +13/-3, 72 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Model] GLM adaptation". The diff centers on `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py`. PR body context: ## Purpose ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, such as "Fix some issue (link existing issues...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -836,7 +836,7 @@ def __init__(; -1499,6 +1499,10 @@ class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):; symbols: __init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM, get_spec_layer_idx_from_weight_name, touching `__init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM`; `tests/models/registry.py` modified +3/-0 (3 lines); hunks: -275,6 +275,9 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`; `tests/models/test_initialization.py` modified +1/-1 (2 lines); hunks: -97,7 +97,7 @@ def _initialize_kv_caches_v1(self, vllm_config):; symbols: _initialize_kv_caches_v1, touching `_initialize_kv_caches_v1`; `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -114,6 +114,7.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -836,7 +836,7 @@ def __init__(; -1499,6 +1499,10 @@ class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):; symbols: __init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM, get_spec_layer_idx_from_weight_name
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunks: -275,6 +275,9 @@ def check_available_online(; symbols: check_available_online
  - `tests/models/test_initialization.py` modified +1/-1 (2 lines); hunks: -97,7 +97,7 @@ def _initialize_kv_caches_v1(self, vllm_config):; symbols: _initialize_kv_caches_v1
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -114,6 +114,7
  - `vllm/config/speculative.py` modified +1/-1 (2 lines); hunks: -181,7 +181,7 @@ def compute_hash(self) -> str:; symbols: compute_hash, hf_config_override
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -836,7 +836,7 @@ def __init__(
-                is_neox_style=True,
+                is_neox_style=not getattr(config, "indexer_rope_interleave", True),
@@ -1499,6 +1499,10 @@ class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
+class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
+    pass
diff -- tests/models/registry.py
@@ -275,6 +275,9 @@ def check_available_online(
+    "GlmMoeDsaForCausalLM": _HfExamplesInfo(
+        "zai-org/GLM-5", min_transformers_version="5.0.1", is_available_online=False
+    ),
diff -- tests/models/test_initialization.py
@@ -97,7 +97,7 @@ def _initialize_kv_caches_v1(self, vllm_config):
-    if model_arch == "DeepseekV32ForCausalLM":
+    if model_arch in ["DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"]:
diff -- vllm/model_executor/models/registry.py
@@ -114,6 +114,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1; `vllm/model_executor/models/registry.py` modified +1/-0; `vllm/config/speculative.py` modified +1/-1; `vllm/transformers_utils/model_arch_config_convertor.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +3/-0; `tests/models/test_initialization.py` modified +1/-1
  - other: `benchmarks/kernels/benchmark_moe.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/models/test_initialization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34385 - [Bugfix] Fix MTP accuracy for GLM-5

- Link: https://github.com/vllm-project/vllm/pull/34385
- Status/date: merged / 2026-02-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +18/-0, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix MTP accuracy for GLM-5". The diff centers on `vllm/v1/spec_decode/eagle.py`. PR body context: ## Purpose Fix MTP producing NaN logits for models (e.g. GLM-5) whose checkpoints don't store a duplicate `shared_head.head` weight in the MTP layer (like DeepSeek V3.2). The ex...
- Key implementation: `vllm/v1/spec_decode/eagle.py` modified +18/-0 (18 lines); hunks: -1506,6 +1506,24 @@ def _maybe_share_lm_head(self, target_language_model: nn....; symbols: _maybe_share_lm_head, dummy_run, touching `_maybe_share_lm_head, dummy_run`.
- Code diff details:
  - `vllm/v1/spec_decode/eagle.py` modified +18/-0 (18 lines); hunks: -1506,6 +1506,24 @@ def _maybe_share_lm_head(self, target_language_model: nn....; symbols: _maybe_share_lm_head, dummy_run
- Key code excerpts:

```diff
diff -- vllm/v1/spec_decode/eagle.py
@@ -1506,6 +1506,24 @@ def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
+            # MTP models call compute_logits via shared_head.head (a
+            # ParallelLMHead inside each MTP layer), not self.model.lm_head.
+            # If the checkpoint omits a copy of the lm_head weights at the
+            # MTP layer path, shared_head.head stays uninitialised and
+            # produces NaN logits. Always share it explicitly.
+            inner = getattr(self.model, "model", None)
```

- Reviewed files:
  - runtime: `vllm/v1/spec_decode/eagle.py` modified +18/-0
- Risk and verification: Runtime changes concentrate in `vllm/v1/spec_decode/eagle.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- This version rejects title-only PR lists; every PR must include trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
