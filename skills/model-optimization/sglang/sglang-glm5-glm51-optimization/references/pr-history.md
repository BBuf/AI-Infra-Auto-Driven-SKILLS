# sglang GLM-5/5.1 PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 11
- Rule: use this as the backing dossier for the skill, not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/platforms/ascend/ascend_npu_glm5_examples.md` | [#22712](https://github.com/sgl-project/sglang/pull/22712) |
| `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx` | no direct PR-number commit |
| `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5_examples.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` | [#23540](https://github.com/sgl-project/sglang/pull/23540) |
| `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` | [#22336](https://github.com/sgl-project/sglang/pull/22336) |
| `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` | [#18911](https://github.com/sgl-project/sglang/pull/18911), [#21710](https://github.com/sgl-project/sglang/pull/21710) |
| `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` | [#22336](https://github.com/sgl-project/sglang/pull/22336) |
| `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` | [#18911](https://github.com/sgl-project/sglang/pull/18911), [#21710](https://github.com/sgl-project/sglang/pull/21710) |
| `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` | [#21773](https://github.com/sgl-project/sglang/pull/21773) |
| `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` | [#22336](https://github.com/sgl-project/sglang/pull/22336) |
| `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` | [#21710](https://github.com/sgl-project/sglang/pull/21710) |
| `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` | [#22336](https://github.com/sgl-project/sglang/pull/22336) |
| `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` | [#21773](https://github.com/sgl-project/sglang/pull/21773) |
| `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` | [#21710](https://github.com/sgl-project/sglang/pull/21710) |
| `test/registered/gb300/test_glm5_fp8.py` | [#22399](https://github.com/sgl-project/sglang/pull/22399) |
| `test/registered/gb300/test_glm5_nvfp4.py` | no direct PR-number commit |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-02-10 | [#18521](https://github.com/sgl-project/sglang/pull/18521) | merged | Support GlmMoeDsaForCausalLM | `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py` |
| 2026-02-16 | [#18804](https://github.com/sgl-project/sglang/pull/18804) | merged | Fix GLM-5 fused shared expert | `python/sglang/srt/models/glm4_moe.py` |
| 2026-02-25 | [#18911](https://github.com/sgl-project/sglang/pull/18911) | merged | [AMD] [GLM-5 Day 0] Add GLM-5 nightly test | `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` |
| 2026-03-09 | [#20062](https://github.com/sgl-project/sglang/pull/20062) | merged | [V32/GLM5] Control the threshold of applying dense attention with an environ | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` |
| 2026-04-06 | [#22179](https://github.com/sgl-project/sglang/pull/22179) | merged | [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation | `docs/basic_usage/deepseek_v32.md` |
| 2026-04-08 | [#22314](https://github.com/sgl-project/sglang/pull/22314) | merged | [AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300 | `python/sglang/srt/mem_cache/memory_pool.py` |
| 2026-04-08 | [#21710](https://github.com/sgl-project/sglang/pull/21710) | merged | [AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x | `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` |
| 2026-04-08 | [#22285](https://github.com/sgl-project/sglang/pull/22285) | merged | Add CI tests for GLM-5 | `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py` |
| 2026-04-09 | [#22399](https://github.com/sgl-project/sglang/pull/22399) | merged | [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model | `test/registered/gb300/test_glm5_fp8.py` |
| 2026-04-09 | [#22336](https://github.com/sgl-project/sglang/pull/22336) | merged | [AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x | `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` |
| 2026-04-13 | [#22712](https://github.com/sgl-project/sglang/pull/22712) | merged | [NPU] update glm5 running guide | `docs/platforms/ascend/ascend_npu_glm5_examples.md` |
| 2026-04-14 | [#22543](https://github.com/sgl-project/sglang/pull/22543) | merged | GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/server_args.py` |
| 2026-04-15 | [#21773](https://github.com/sgl-project/sglang/pull/21773) | merged | [AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x | `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` |
| 2026-04-16 | [#22595](https://github.com/sgl-project/sglang/pull/22595) | merged | fix: normalize tool message content for GLM5.1 chat template | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py` |
| 2026-04-19 | [#22850](https://github.com/sgl-project/sglang/pull/22850) | merged | [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion) | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-20 | [#23219](https://github.com/sgl-project/sglang/pull/23219) | merged | [AMD] Enable MTP for GLM-5-mxfp4 model | `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-04-23 | [#23060](https://github.com/sgl-project/sglang/pull/23060) | merged | [fix] Fix dynamic chunking profiling crash on GLM-5 models | `python/sglang/srt/managers/scheduler_pp_mixin.py` |
| 2026-04-23 | [#23540](https://github.com/sgl-project/sglang/pull/23540) | merged | docs: split MI300X and MI325X options in GLM-5.1 generator | `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` |

## Per-PR Diff Audit Cards

### PR #18521 - Support GlmMoeDsaForCausalLM

- Link: https://github.com/sgl-project/sglang/pull/18521
- Status/date: merged / 2026-02-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +22/-7, 98 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR adds or enables a model support/runtime surface. Title: "Support GlmMoeDsaForCausalLM". The diff centers on `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `python/sglang/srt/configs/model_config.py` modified +6/-5 (11 lines); hunks: -61,6 +61,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; -271,10 +272,10 @@ def from_server_args(; symbols: is_deepseek_nsa, from_server_args, _config_draft_model, _derive_model_shapes, touching `is_deepseek_nsa, from_server_args, _config_draft_model`; `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: -79,6 +79,7; -1279,4 +1280,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, touching `set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM`; `python/sglang/srt/server_args.py` modified +10/-1 (11 lines); hunks: -1194,9 +1194,15 @@ def _handle_model_specific_adjustments(self):; -2323,6 +2329,7 @@ def _handle_speculative_decoding(self):; symbols: _handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference, auto_choose_speculative_params, touching `_handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference`.
- Code diff details:
  - `python/sglang/srt/configs/model_config.py` modified +6/-5 (11 lines); hunks: -61,6 +61,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; -271,10 +272,10 @@ def from_server_args(; symbols: is_deepseek_nsa, from_server_args, _config_draft_model, _derive_model_shapes
  - `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: -79,6 +79,7; -1279,4 +1280,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM
  - `python/sglang/srt/server_args.py` modified +10/-1 (11 lines); hunks: -1194,9 +1194,15 @@ def _handle_model_specific_adjustments(self):; -2323,6 +2329,7 @@ def _handle_speculative_decoding(self):; symbols: _handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference, auto_choose_speculative_params
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/model_config.py
@@ -61,6 +61,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:
+            "GlmMoeDsaForCausalLM",
@@ -271,10 +272,10 @@ def from_server_args(
-        if (
-            is_draft_model
-            and self.hf_config.architectures[0] == "DeepseekV3ForCausalLM"
-        ):
diff -- python/sglang/srt/models/glm4_moe.py
@@ -79,6 +79,7 @@
+from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
@@ -1279,4 +1280,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
-EntryClass = [Glm4MoeForCausalLM]
+class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
+    pass
+EntryClass = [Glm4MoeForCausalLM, GlmMoeDsaForCausalLM]
diff -- python/sglang/srt/server_args.py
@@ -1194,9 +1194,15 @@ def _handle_model_specific_adjustments(self):
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/model_config.py` modified +6/-5; `python/sglang/srt/models/glm4_moe.py` modified +6/-1; `python/sglang/srt/server_args.py` modified +10/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18804 - Fix GLM-5 fused shared expert

- Link: https://github.com/sgl-project/sglang/pull/18804
- Status/date: merged / 2026-02-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix GLM-5 fused shared expert". The diff centers on `python/sglang/srt/models/glm4_moe.py`. PR body context: ## Motivation The MoE parts of GLM-5 consists of 256 routing experts and 1 shared expert, but currently the code fully inherits from `DeepseekV2ForCausalLM`, which causes the fu...
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -1281,7 +1281,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts, touching `set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -1281,7 +1281,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -1281,7 +1281,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
-    pass
+    def determine_num_fused_shared_experts(self):
+        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18911 - [AMD] [GLM-5 Day 0] Add GLM-5 nightly test

- Link: https://github.com/sgl-project/sglang/pull/18911
- Status/date: merged / 2026-02-25
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`; associated commits `23adb50751d5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +635/-1, 725 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR adds or enables a model support/runtime surface. Title: "[AMD] [GLM-5 Day 0] Add GLM-5 nightly test". The diff centers on `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`. PR body context: ## Motivation Add nightly CI accuracy tests for GLM-5 on AMD MI325/MI300X and MI35x runners. GLM-5 is a 744B parameter (40B active) MoE model that uses DeepSeek Sparse Attention...
- Key implementation: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples, touching `ModelConfig, get_display_name, get_one_example`; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples, touching `ModelConfig, get_display_name, get_one_example`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py
@@ -0,0 +1,249 @@
+"""MI35x GLM-5 GSM8K Completion Evaluation Test (8-GPU)
+Tests GLM-5 with NSA attention backend using few-shot completion
+benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-glm5 suite
+"""
+import ast
diff -- test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py
@@ -0,0 +1,244 @@
+"""AMD GLM-5 GSM8K Completion Evaluation Test (8-GPU)
+Tests GLM-5 with NSA attention backend using few-shot completion
+benchmark on MI325/MI300X.
+Registry: nightly-amd-accuracy-8-gpu-glm5 suite
+"""
+import ast
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20062 - [V32/GLM5] Control the threshold of applying dense attention with an environ

- Link: https://github.com/sgl-project/sglang/pull/20062
- Status/date: merged / 2026-03-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +32/-59, 200 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR adds or enables a model support/runtime surface. Title: "[V32/GLM5] Control the threshold of applying dense attention with an environ". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`. PR body context: ## Motivation - Add an environ `SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD`, for controlling whether to use dense MHA or sparse MLA kernel. It's set to index.topk by default, thus n...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46 (49 lines); hunks: -16,10 +16,6; -71,15 +67,10; symbols: NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed, set_nsa_prefill_impl, touching `NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed`; `python/sglang/srt/server_args.py` modified +26/-3 (29 lines); hunks: -1353,12 +1353,35 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`; `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4 (4 lines); hunks: -34,8 +34,6 @@ def setUpClass(cls):; -103,8 +101,6 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4 (4 lines); hunks: -39,8 +39,6 @@ def setUpClass(cls):; -131,8 +129,6 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46 (49 lines); hunks: -16,10 +16,6; -71,15 +67,10; symbols: NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed, set_nsa_prefill_impl
  - `python/sglang/srt/server_args.py` modified +26/-3 (29 lines); hunks: -1353,12 +1353,35 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4 (4 lines); hunks: -34,8 +34,6 @@ def setUpClass(cls):; -103,8 +101,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4 (4 lines); hunks: -39,8 +39,6 @@ def setUpClass(cls):; -131,8 +129,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `python/sglang/srt/environ.py` modified +1/-2 (3 lines); hunks: -377,8 +377,7 @@ class Envs:; symbols: Envs
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -16,10 +16,6 @@
-from sglang.srt.layers.attention.nsa.nsa_mtp_verification import (
-    verify_multi_backend_fused_metadata_copy,
-    verify_single_backend_fused_metadata_copy,
-)
@@ -71,15 +67,10 @@
-# Control whether to use fused metadata copy kernel (default: enabled)
diff -- python/sglang/srt/server_args.py
@@ -1353,12 +1353,35 @@ def _handle_model_specific_adjustments(self):
-            if is_deepseek_nsa(hf_config):  # DeepSeek 3.2, GlmMoeDsaForCausalLM
+            if is_deepseek_nsa(hf_config):  # DeepSeek 3.2/GLM 5
-                    envs.SGLANG_NSA_FORCE_MLA.set(True)
+                    envs.SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.set(0)
-                        "Force NSA prefill to use MLA (i.e. disable MHA_ONE_SHOT) for GlmMoeDsaForCausalLM on Blackwell."
+                        "Force NSA prefill to use sparse MLA (i.e. disable MHA_ONE_SHOT) for GlmMoeDsaForCausalLM on Blackwell."
diff -- test/registered/quant/test_deepseek_v32_fp4_4gpu.py
@@ -34,8 +34,6 @@ def setUpClass(cls):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46; `python/sglang/srt/server_args.py` modified +26/-3; `python/sglang/srt/environ.py` modified +1/-2
  - tests: `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4; `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4
  - docs: `docs/references/environment_variables.md` modified +2/-0
- Risk and verification: The diff ships test coverage in `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22179 - [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation

- Link: https://github.com/sgl-project/sglang/pull/22179
- Status/date: merged / 2026-04-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-12, 91 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Summary Remove skip-softmax section (I think it's for dense attention only, not DSA, per flashinfer constraint below) and improve docs https://github.com/flashinfer-ai/flashi...
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunks: -3,7 +3,7; -56,13 +56,13 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-....
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunks: -3,7 +3,7; -56,13 +56,13 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-...
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -3,7 +3,7 @@
-Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://hu
+Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://hu
@@ -56,13 +56,13 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8
-To server GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
+To serve GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
-- **MHA prefill threshold relax** To apply MHA attention to requests longer than 2048 tokens, please set flag `SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` to a value larger th
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +11/-12
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #22314 - [AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300

- Link: https://github.com/sgl-project/sglang/pull/22314
- Status/date: merged / 2026-04-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +27/-31, 73 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300". The diff centers on `python/sglang/srt/mem_cache/memory_pool.py`. PR body context: ## Motivation On MI300, running GLM-5-fp8 with FP8 KV cache can fail (see CI log). The root cause is that the quant path does not dispatch the correct kernel (`set_mla_kv_buffer...
- Key implementation: `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31 (58 lines); hunks: -45,7 +45,7; -1575,37 +1575,33 @@ def set_mla_kv_buffer(; symbols: set_mla_kv_buffer, touching `set_mla_kv_buffer`.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31 (58 lines); hunks: -45,7 +45,7; -1575,37 +1575,33 @@ def set_mla_kv_buffer(; symbols: set_mla_kv_buffer
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/memory_pool.py
@@ -45,7 +45,7 @@
-from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
+from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
@@ -1575,37 +1575,33 @@ def set_mla_kv_buffer(
-        if self.nsa_kv_cache_store_fp8:
-            if _is_hip:
-                # HIP FP8 path uses raw MLA KV layout (nope + rope) without per-block scales.
```

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/memory_pool.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21710 - [AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x

- Link: https://github.com/sgl-project/sglang/pull/21710
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`; associated commits `db60a620dbf1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +345/-5, 448 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR adds or enables a model support/runtime surface. Title: "[AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x". The diff centers on `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`. PR body context: ## Summary Add GLM-5-FP8 nightly perf benchmarks (`bench_one_batch`) for MI30x and MI35x. Both accuracy and perf use `zai-org/GLM-5-FP8` with NSA tilelang backend, TP=8, FP8 KV...
- Key implementation: `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0 (143 lines); hunks: -0,0 +1,143; symbols: generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass, test_glm5_perf, touching `generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass`; `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass, test_bench_glm5, touching `generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass`; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2 (8 lines); hunks: -59,13 +59,17 @@ def get_display_name(self) -> str:; -77,7 +81,7 @@ def get_display_name(self) -> str:; symbols: get_display_name, touching `get_display_name`; `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` modified +6/-2 (8 lines); hunks: -64,13 +64,17 @@ def get_display_name(self) -> str:; -82,7 +86,7 @@ def get_display_name(self) -> str:; symbols: get_display_name, touching `get_display_name`.
- Code diff details:
  - `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0 (143 lines); hunks: -0,0 +1,143; symbols: generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass, test_glm5_perf
  - `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass, test_bench_glm5
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2 (8 lines); hunks: -59,13 +59,17 @@ def get_display_name(self) -> str:; -77,7 +81,7 @@ def get_display_name(self) -> str:; symbols: get_display_name
  - `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` modified +6/-2 (8 lines); hunks: -64,13 +64,17 @@ def get_display_name(self) -> str:; -82,7 +86,7 @@ def get_display_name(self) -> str:; symbols: get_display_name
- Key code excerpts:

```diff
diff -- test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py
@@ -0,0 +1,143 @@
+"""MI35x Nightly performance benchmark for GLM-5.
+Tests GLM-5 with NSA attention backend using bench_one_batch on 8 GPUs.
+Registry: nightly-perf-8-gpu-mi35x-glm5 suite
+"""
+import os
+os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
diff -- test/registered/amd/perf/mi30x/test_glm5_perf_amd.py
@@ -0,0 +1,140 @@
+"""Nightly performance benchmark for GLM-5 on MI30x.
+Tests GLM-5 with NSA attention backend using bench_one_batch on 8 GPUs.
+Model paths can be configured via environment variables:
+- GLM5_MODEL_PATH: Path to GLM-5 model (default: zai-org/GLM-5-FP8)
+Example usage:
+    python -m pytest test_glm5_perf_amd.py -v
diff -- test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py
@@ -59,13 +59,17 @@ def get_display_name(self) -> str:
```

- Reviewed files:
  - tests: `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0; `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2; `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` modified +6/-2
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22285 - Add CI tests for GLM-5

- Link: https://github.com/sgl-project/sglang/pull/22285
- Status/date: merged / 2026-04-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +153/-30, 301 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR adds or enables a model support/runtime surface. Title: "Add CI tests for GLM-5". The diff centers on `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Speed Tests and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit te...
- Key implementation: `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1 (122 lines); hunks: -14,9 +14,10; -138,5 +139,124 @@ def test_bs_1_speed(self):; symbols: TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP, setUpClass, touching `TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP`; `test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29 (61 lines); hunks: -20,6 +20,7; -47,12 +48,13 @@ def setUpClass(cls):; symbols: TestDeepseekV32DPMTP, setUpClass, tearDownClass, test_bs_1_speed, touching `TestDeepseekV32DPMTP, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1 (122 lines); hunks: -14,9 +14,10; -138,5 +139,124 @@ def test_bs_1_speed(self):; symbols: TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP, setUpClass
  - `test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29 (61 lines); hunks: -20,6 +20,7; -47,12 +48,13 @@ def setUpClass(cls):; symbols: TestDeepseekV32DPMTP, setUpClass, tearDownClass, test_bs_1_speed
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_dsa_models_basic.py
@@ -14,9 +14,10 @@
-register_cuda_ci(est_time=360, suite="stage-c-test-8-gpu-h200")
+register_cuda_ci(est_time=720, suite="stage-c-test-8-gpu-h200")
+GLM5_MODEL_PATH = "zai-org/GLM-5-FP8"
@@ -138,5 +139,124 @@ def test_bs_1_speed(self):
+class TestGLM5DP(CustomTestCase):
+    @classmethod
diff -- test/registered/8-gpu-models/test_dsa_models_mtp.py
@@ -20,6 +20,7 @@
+GLM5_MODEL_PATH = "zai-org/GLM-5-FP8"
@@ -47,12 +48,13 @@ def setUpClass(cls):
-        cls.process = popen_launch_server(
-            cls.model,
-            cls.base_url,
-            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1; `test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22399 - [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model

- Link: https://github.com/sgl-project/sglang/pull/22399
- Status/date: merged / 2026-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/gb300/test_glm5_fp8.py`; associated commits `46c2b7762765`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +82/-6, 131 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR adds or enables a model support/runtime surface. Title: "[CI] Add GLM-5.1 nightly tests and update Qwen3.5 model". The diff centers on `test/registered/gb300/test_glm5_fp8.py`. PR body context: ## Summary - Add GLM-5.1 FP8 nightly test for H200/B200 (`nightly-8-gpu-common` suite) with TP8, TP8+DP8, and TP8+DP8+MTP variants - Update GB300 GLM-5 tests to GLM-5.1 model na...
- Key implementation: `test/registered/gb300/test_glm5_fp8.py` modified +3/-3 (6 lines); hunks: -8,7 +8,7; -27,7 +27,7; symbols: TestGlm5Fp8, test_glm5_fp8, touching `TestGlm5Fp8, test_glm5_fp8`.
- Code diff details:
  - `test/registered/gb300/test_glm5_fp8.py` modified +3/-3 (6 lines); hunks: -8,7 +8,7; -27,7 +27,7; symbols: TestGlm5Fp8, test_glm5_fp8
- Key code excerpts:

```diff
diff -- test/registered/gb300/test_glm5_fp8.py
@@ -8,7 +8,7 @@
-MODEL_PATH = "zai-org/GLM-5-FP8"
+MODEL_PATH = "zai-org/GLM-5.1-FP8"
@@ -27,7 +27,7 @@
-    """GLM-5 FP8 on GB300 (4x B200 NVL4, tp=4)."""
+    """GLM-5.1 FP8 on GB300 (4x B200 NVL4, tp=4)."""
@@ -56,7 +56,7 @@ def test_glm5_fp8(self):
```

- Reviewed files:
  - tests: `test/registered/gb300/test_glm5_fp8.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/8-gpu-models/test_qwen35.py`, `test/registered/gb300/test_glm5_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22336 - [AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x

- Link: https://github.com/sgl-project/sglang/pull/22336
- Status/date: merged / 2026-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`; associated commits `ef6bfc1197ab`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +918/-25, 1064 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR adds or enables a model support/runtime surface. Title: "[AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x". The diff centers on `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`. PR body context: ## Summary Add GLM-5.1-FP8 nightly accuracy + perf benchmarks (`bench_one_batch`) for MI30x and MI35x. Both GLM-5-FP8 and GLM-5.1-FP8 share identical architecture (`GlmMoeDsaFor...
- Key implementation: `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples, touching `ModelConfig, get_display_name, get_one_example`; `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples, touching `ModelConfig, get_display_name, get_one_example`; `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass, test_glm51_perf, touching `generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass`; `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass, test_bench_glm51, touching `generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass, test_glm51_perf
  - `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass, test_bench_glm51
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py
@@ -0,0 +1,242 @@
+"""MI35x GLM-5.1 GSM8K Completion Evaluation Test (8-GPU)
+Tests GLM-5.1-FP8 with NSA attention backend using few-shot
+completion benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-glm51 suite
+"""
+import ast
diff -- test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py
@@ -0,0 +1,238 @@
+"""AMD GLM-5.1 GSM8K Completion Evaluation Test (8-GPU)
+Tests GLM-5.1-FP8 with NSA attention backend using few-shot
+completion benchmark on MI325/MI300X.
+Registry: nightly-amd-accuracy-8-gpu-glm51 suite
+"""
+import ast
diff -- test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py
@@ -0,0 +1,146 @@
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0; `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0; `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0; `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22712 - [NPU] update glm5 running guide

- Link: https://github.com/sgl-project/sglang/pull/22712
- Status/date: merged / 2026-04-13
- Trace source: `git log --name-only -- <model-files>` found it through `docs/platforms/ascend/ascend_npu_glm5_examples.md`; associated commits `13a4aafdbe69`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-2, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR adds or enables a model support/runtime surface. Title: "[NPU] update glm5 running guide". The diff centers on `docs/platforms/ascend/ascend_npu_glm5_examples.md`. PR body context: ## Motivation Update NPU document, add best practice of GLM5 supported on ascend npu ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [x] Format y...
- Key implementation: `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2 (10 lines); hunks: -53,10 +53,16 @@ docker run -itd --shm-size=16g --privileged=true --name ${NA....
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2 (10 lines); hunks: -53,10 +53,16 @@ docker run -itd --shm-size=16g --privileged=true --name ${NA...
- Key code excerpts:

```diff
diff -- docs/platforms/ascend/ascend_npu_glm5_examples.md
@@ -53,10 +53,16 @@ docker run -itd --shm-size=16g --privileged=true --name ${NAME} \
-Note: Using this image, you need to update transformers to main branch
+### Best Practices
+Note: Using this image for **best practices**, you need to update transformers to version 5.3.0
-pip install git+https://github.com/huggingface/transformers.git
+# Install transformers version 5.3.0 from PyPI
+pip install transformers==5.3.0
```

- Reviewed files:
  - docs: `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2
- Risk and verification: This is mostly docs/examples in `docs/platforms/ascend/ascend_npu_glm5_examples.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #22543 - GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix

- Link: https://github.com/sgl-project/sglang/pull/22543
- Status/date: merged / 2026-04-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +8/-0, 29 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR fixes a launch, loading, parsing, or numerical issue. Title: "GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix". The diff centers on `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/server_args.py`. PR body context: ## Motivation Addresses this issue regarding AMD Quark-quantized GLM-5 and GLM-5.1 MXFP4 checkpoints when using with SGLang (Exclude-layer names don't match SGLang internal name...
- Key implementation: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0 (3 lines); hunks: -560,6 +560,9 @@ def post_load_weights(; symbols: post_load_weights, touching `post_load_weights`; `python/sglang/srt/model_loader/loader.py` modified +3/-0 (3 lines); hunks: -198,6 +198,9 @@ def _get_quantization_config(; symbols: _get_quantization_config, touching `_get_quantization_config`; `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -1016,6 +1016,8 @@ def _handle_missing_default_values(self):; symbols: _handle_missing_default_values, touching `_handle_missing_default_values`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0 (3 lines); hunks: -560,6 +560,9 @@ def post_load_weights(; symbols: post_load_weights
  - `python/sglang/srt/model_loader/loader.py` modified +3/-0 (3 lines); hunks: -198,6 +198,9 @@ def _get_quantization_config(; symbols: _get_quantization_config
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -1016,6 +1016,8 @@ def _handle_missing_default_values(self):; symbols: _handle_missing_default_values
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -560,6 +560,9 @@ def post_load_weights(
+                and self.config.architectures
+                and self.config.architectures[0]
+                == "DeepseekV3ForCausalLM"  # Avoid processing other models like GlmMoeDsaForCausalLM
diff -- python/sglang/srt/model_loader/loader.py
@@ -198,6 +198,9 @@ def _get_quantization_config(
+    if model_config.quantization == "quark":
+        packed_modules_mapping.update({"gate_up_proj": ["gate_proj", "up_proj"]})
diff -- python/sglang/srt/server_args.py
@@ -1016,6 +1016,8 @@ def _handle_missing_default_values(self):
+        # strip device index from user if any (e.g. "cuda:0" -> "cuda")
+        self.device = self.device.split(":")[0]
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0; `python/sglang/srt/model_loader/loader.py` modified +3/-0; `python/sglang/srt/server_args.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21773 - [AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x

- Link: https://github.com/sgl-project/sglang/pull/21773
- Status/date: merged / 2026-04-15
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`; associated commits `39c6bf730c41`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +528/-130, 821 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR adds or enables a model support/runtime surface. Title: "[AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x". The diff centers on `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`. PR body context: ## Summary - Add nightly accuracy test (GSM8K 5-shot) and perf benchmark (\`bench_one_batch\`) for \`amd/GLM-5-MXFP4\` on MI35x 8-GPU - Remove obsolete base GLM-5 (BF16 NSA) CI...
- Key implementation: `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: get_model_path, ModelConfig, __post_init__, get_display_name, touching `get_model_path, ModelConfig, __post_init__`; `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0 (187 lines); hunks: -0,0 +1,187; symbols: generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x, setUpClass, touching `generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: get_model_path, ModelConfig, __post_init__, get_display_name
  - `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0 (187 lines); hunks: -0,0 +1,187; symbols: generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x, setUpClass
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py
@@ -0,0 +1,281 @@
+"""MI35x GLM-5-MXFP4 GSM8K Completion Evaluation Test (8-GPU)
+Tests the AMD Quark MXFP4-quantized GLM-5 model using few-shot
+completion benchmark on MI35x.
+Model: amd/GLM-5-MXFP4 (MOE-only MXFP4 quantization of zai-org/GLM-5)
+Reference: https://huggingface.co/amd/GLM-5-MXFP4
+Registry: nightly-amd-8-gpu-mi35x-glm5-mxfp4 suite
diff -- test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py
@@ -0,0 +1,187 @@
+"""MI35x Nightly performance benchmark for GLM-5-MXFP4 model.
+Benchmarks the AMD Quark MXFP4-quantized GLM-5 model on MI35x with 8 GPUs.
+Model: amd/GLM-5-MXFP4 (MOE-only MXFP4 quantization of zai-org/GLM-5)
+Reference: https://huggingface.co/amd/GLM-5-MXFP4
+Registry: nightly-perf-8-gpu-mi35x-glm5-mxfp4 suite
+"""
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0; `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22595 - fix: normalize tool message content for GLM5.1 chat template

- Link: https://github.com/sgl-project/sglang/pull/22595
- Status/date: merged / 2026-04-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +67/-1, 95 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR fixes a launch, loading, parsing, or numerical issue. Title: "fix: normalize tool message content for GLM5.1 chat template". The diff centers on `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py`. PR body context: ## Fix: Normalize tool message `content` from array format to string before applying chat template ### Problem Per the OpenAI API specification%20chat.completions%20%3E%20(model...
- Key implementation: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0 (26 lines); hunks: -60,6 +60,28; -457,6 +479,10 @@ def _apply_jinja_template(; symbols: normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template, touching `normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template`; `test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1 (42 lines); hunks: -19,7 +19,10; -894,5 +897,42 @@ def test_required_without_parser_invalid_json_returns_none(...; symbols: test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened, test_multiple_text_parts_joined, touching `test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0 (26 lines); hunks: -60,6 +60,28; -457,6 +479,10 @@ def _apply_jinja_template(; symbols: normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template
  - `test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1 (42 lines); hunks: -19,7 +19,10; -894,5 +897,42 @@ def test_required_without_parser_invalid_json_returns_none(...; symbols: test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened, test_multiple_text_parts_joined
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -60,6 +60,28 @@
+def normalize_tool_content(role: str, content):
+    """Normalize tool message content from OpenAI array format to plain string.
+    OpenAI clients may send tool content as a list of content parts
+    (e.g. [{"type":"text","text":"..."}]) but most chat templates expect
+    a plain string for tool messages. Only flatten when ALL items are
+    pure OpenAI text parts; preserve lists containing non-text-type items
diff -- test/registered/openai_server/basic/test_serving_chat.py
@@ -19,7 +19,10 @@
-from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
+from sglang.srt.entrypoints.openai.serving_chat import (
+    OpenAIServingChat,
+    normalize_tool_content,
+)
@@ -894,5 +897,42 @@ def test_required_without_parser_invalid_json_returns_none(self):
```

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0
  - tests: `test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1
- Risk and verification: The diff ships test coverage in `test/registered/openai_server/basic/test_serving_chat.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22850 - [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)

- Link: https://github.com/sgl-project/sglang/pull/22850
- Status/date: merged / 2026-04-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +24/-5, 72 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR changes model-related implementation. Title: "[AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation Redundant kernels in the NSA indexer on HIP: weights_proj: - The ReplicatedLinear layer uses fp32 params_dtype, preventing tgemm from dispatching to the tuned bf16...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunks: -14,7 +14,7; -32,14 +32,16; symbols: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache, touching `__init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunks: -14,7 +14,7; -32,14 +32,16; symbols: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -14,7 +14,7 @@
-from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
+from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
@@ -32,14 +32,16 @@
-_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
+if _use_aiter:
+    from aiter.ops.cache import indexer_k_quant_and_cache
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23219 - [AMD] Enable MTP for GLM-5-mxfp4 model

- Link: https://github.com/sgl-project/sglang/pull/23219
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +41/-15, 87 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR adds or enables a model support/runtime surface. Title: "[AMD] Enable MTP for GLM-5-mxfp4 model". The diff centers on `python/sglang/srt/models/deepseek_nextn.py`. PR body context: ## Motivation Fix https://github.com/sgl-project/sglang/issues/23142. Quark-quantized GLM-5-MXFP4 checkpoints store MTP (NextN) weights — including `eh_proj` — in FP4-packed for...
- Key implementation: `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunks: -42,6 +42,7; -99,7 +100,18 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunks: -42,6 +42,7; -99,7 +100,18 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -42,6 +42,7 @@
+from sglang.srt.layers.linear import ReplicatedLinear
@@ -99,7 +100,18 @@ def __init__(
-        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
+        if quant_config is not None and quant_config.get_name() == "quark":
+            self.eh_proj = ReplicatedLinear(
+                2 * config.hidden_size,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23060 - [fix] Fix dynamic chunking profiling crash on GLM-5 models

- Link: https://github.com/sgl-project/sglang/pull/23060
- Status/date: merged / 2026-04-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR adds or enables a model support/runtime surface. Title: "[fix] Fix dynamic chunking profiling crash on GLM-5 models". The diff centers on `python/sglang/srt/managers/scheduler_pp_mixin.py`. PR body context: ## Motivation Fixes #23057 When `--enable-dynamic-chunking` is used with GLM-5 (have DeepEP ), the profiling phase crashes with `AttributeError: _is_extend_in_batch`. This silen...
- Key implementation: `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0 (3 lines); hunks: -20,6 +20,7; -631,6 +632,8 @@ def profile_and_init_predictor(self: Scheduler):; symbols: profile_and_init_predictor, touching `profile_and_init_predictor`.
- Code diff details:
  - `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0 (3 lines); hunks: -20,6 +20,7; -631,6 +632,8 @@ def profile_and_init_predictor(self: Scheduler):; symbols: profile_and_init_predictor
- Key code excerpts:

```diff
diff -- python/sglang/srt/managers/scheduler_pp_mixin.py
@@ -20,6 +20,7 @@
+    set_is_extend_in_batch,
@@ -631,6 +632,8 @@ def profile_and_init_predictor(self: Scheduler):
+                set_is_extend_in_batch(batch.forward_mode.is_extend())
```

- Reviewed files:
  - runtime: `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/scheduler_pp_mixin.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23540 - docs: split MI300X and MI325X options in GLM-5.1 generator

- Link: https://github.com/sgl-project/sglang/pull/23540
- Status/date: merged / 2026-04-23
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`; associated commits `9b2f7f8a91d4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +15/-13, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-5/5.1, this PR optimizes an inference path or backend selection. Title: "docs: split MI300X and MI325X options in GLM-5.1 generator". The diff centers on `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`. PR body context: ## Summary - split the GLM-5.1 hardware selector button `MI300X/MI325X` into separate `MI300X` and `MI325X` options - map `MI325X` to the same AMD config path and generated comm...
- Key implementation: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4 (10 lines); hunks: -14,7 +14,8 @@ export const GLM51Deployment = () => {; -23,7 +24,7 @@ export const GLM51Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4 (10 lines); hunks: -14,7 +14,8 @@ export const GLM51Deployment = () => {; -23,7 +24,7 @@ export const GLM51Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/glm-51-deployment.jsx
@@ -14,7 +14,8 @@ export const GLM51Deployment = () => {
-        { id: 'mi300x', label: 'MI300X/MI325X', default: false },
+        { id: 'mi300x', label: 'MI300X',        default: false },
+        { id: 'mi325x', label: 'MI325X',        default: false },
@@ -23,7 +24,7 @@ export const GLM51Deployment = () => {
-        const isAMD = hw === 'mi300x' || hw === 'mi355x';
+        const isAMD = ['mi300x', 'mi325x', 'mi355x'].includes(hw);
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/docs.json`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.
