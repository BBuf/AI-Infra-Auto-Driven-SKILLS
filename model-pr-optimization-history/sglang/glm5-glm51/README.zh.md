# sglang GLM-5/5.1 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/platforms/ascend/ascend_npu_glm5_examples.md` | [#22712](https://github.com/sgl-project/sglang/pull/22712) |
| `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx` | 无直接 PR 号提交 |
| `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5_examples.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` | 无直接 PR 号提交 |
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
| `test/registered/gb300/test_glm5_nvfp4.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 7
- 原文档显式引用补充 PR 数: 11
- 当前文档总 PR 数: 18
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
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

## 逐 PR diff 审计卡

### PR #18521 - Support GlmMoeDsaForCausalLM

- 链接: https://github.com/sgl-project/sglang/pull/18521
- 状态/时间: merged / 2026-02-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+22/-7，可读 patch 98 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support GlmMoeDsaForCausalLM」；模型线: GLM-5/5.1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/configs/model_config.py` modified +6/-5 (11 lines); hunks: -61,6 +61,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; -271,10 +272,10 @@ def from_server_args(; symbols: is_deepseek_nsa, from_server_args, _config_draft_model, _derive_model_shapes，涉及 `is_deepseek_nsa, from_server_args, _config_draft_model`；`python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: -79,6 +79,7; -1279,4 +1280,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM，涉及 `set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM`；`python/sglang/srt/server_args.py` modified +10/-1 (11 lines); hunks: -1194,9 +1194,15 @@ def _handle_model_specific_adjustments(self):; -2323,6 +2329,7 @@ def _handle_speculative_decoding(self):; symbols: _handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference, auto_choose_speculative_params，涉及 `_handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/model_config.py` modified +6/-5 (11 lines); hunks: -61,6 +61,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; -271,10 +272,10 @@ def from_server_args(; symbols: is_deepseek_nsa, from_server_args, _config_draft_model, _derive_model_shapes
  - `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: -79,6 +79,7; -1279,4 +1280,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM
  - `python/sglang/srt/server_args.py` modified +10/-1 (11 lines); hunks: -1194,9 +1194,15 @@ def _handle_model_specific_adjustments(self):; -2323,6 +2329,7 @@ def _handle_speculative_decoding(self):; symbols: _handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference, auto_choose_speculative_params
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/configs/model_config.py` modified +6/-5; `python/sglang/srt/models/glm4_moe.py` modified +6/-1; `python/sglang/srt/server_args.py` modified +10/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18804 - Fix GLM-5 fused shared expert

- 链接: https://github.com/sgl-project/sglang/pull/18804
- 状态/时间: merged / 2026-02-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-1，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GLM-5 fused shared expert」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe.py`；PR 正文摘要: The MoE parts of GLM-5 consists of 256 routing experts and 1 shared expert, but currently the code fully inherits from `DeepseekV2ForCausalLM`, which causes the fused shared exp...。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -1281,7 +1281,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts，涉及 `set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -1281,7 +1281,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -1281,7 +1281,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
-    pass
+    def determine_num_fused_shared_experts(self):
+        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18911 - [AMD] [GLM-5 Day 0] Add GLM-5 nightly test

- 链接: https://github.com/sgl-project/sglang/pull/18911
- 状态/时间: merged / 2026-02-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`；关联提交 `23adb50751d5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+635/-1，可读 patch 725 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [GLM-5 Day 0] Add GLM-5 nightly test」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`；PR 正文摘要: Add nightly CI accuracy tests for GLM-5 on AMD MI325/MI300X and MI35x runners. GLM-5 is a 744B parameter (40B active) MoE model that uses DeepSeek Sparse Attention (DSA/NSA), sh...。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples，涉及 `ModelConfig, get_display_name, get_one_example`；`test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples，涉及 `ModelConfig, get_display_name, get_one_example`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20062 - [V32/GLM5] Control the threshold of applying dense attention with an environ

- 链接: https://github.com/sgl-project/sglang/pull/20062
- 状态/时间: merged / 2026-03-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+32/-59，可读 patch 200 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V32/GLM5] Control the threshold of applying dense attention with an environ」；模型线: GLM-5/5.1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`；PR 正文摘要: - Add an environ `SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD`, for controlling whether to use dense MHA or sparse MLA kernel. It's set to index.topk by default, thus not breaking th...。
- 实现要点: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46 (49 lines); hunks: -16,10 +16,6; -71,15 +67,10; symbols: NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed, set_nsa_prefill_impl，涉及 `NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed`；`python/sglang/srt/server_args.py` modified +26/-3 (29 lines); hunks: -1353,12 +1353,35 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments，涉及 `_handle_model_specific_adjustments`；`test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4 (4 lines); hunks: -34,8 +34,6 @@ def setUpClass(cls):; -103,8 +101,6 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`；`test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4 (4 lines); hunks: -39,8 +39,6 @@ def setUpClass(cls):; -131,8 +129,6 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46 (49 lines); hunks: -16,10 +16,6; -71,15 +67,10; symbols: NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed, set_nsa_prefill_impl
  - `python/sglang/srt/server_args.py` modified +26/-3 (29 lines); hunks: -1353,12 +1353,35 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4 (4 lines); hunks: -34,8 +34,6 @@ def setUpClass(cls):; -103,8 +101,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4 (4 lines); hunks: -39,8 +39,6 @@ def setUpClass(cls):; -131,8 +129,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `python/sglang/srt/environ.py` modified +1/-2 (3 lines); hunks: -377,8 +377,7 @@ class Envs:; symbols: Envs
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46; `python/sglang/srt/server_args.py` modified +26/-3; `python/sglang/srt/environ.py` modified +1/-2
  - tests: `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4; `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4
  - docs: `docs/references/environment_variables.md` modified +2/-0
- 验证与风险: diff 自带测试面 `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22179 - [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation

- 链接: https://github.com/sgl-project/sglang/pull/22179
- 状态/时间: merged / 2026-04-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-12，可读 patch 91 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `docs/basic_usage/deepseek_v32.md`；PR 正文摘要: Remove skip-softmax section (I think it's for dense attention only, not DSA, per flashinfer constraint below) and improve docs https://github.com/flashinfer-ai/flashinfer/blob/v...。
- 实现要点: `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunks: -3,7 +3,7; -56,13 +56,13 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-...。
- 代码 diff 细节:
  - `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunks: -3,7 +3,7; -56,13 +56,13 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-...
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +11/-12
- 验证与风险: 该 PR 主要落在文档/示例 `docs/basic_usage/deepseek_v32.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #22314 - [AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300

- 链接: https://github.com/sgl-project/sglang/pull/22314
- 状态/时间: merged / 2026-04-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+27/-31，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/mem_cache/memory_pool.py`；PR 正文摘要: On MI300, running GLM-5-fp8 with FP8 KV cache can fail (see CI log). The root cause is that the quant path does not dispatch the correct kernel (`set_mla_kv_buffer_triton_fp8_qu...。
- 实现要点: `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31 (58 lines); hunks: -45,7 +45,7; -1575,37 +1575,33 @@ def set_mla_kv_buffer(; symbols: set_mla_kv_buffer，涉及 `set_mla_kv_buffer`。
- 代码 diff 细节:
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31 (58 lines); hunks: -45,7 +45,7; -1575,37 +1575,33 @@ def set_mla_kv_buffer(; symbols: set_mla_kv_buffer
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/mem_cache/memory_pool.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21710 - [AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x

- 链接: https://github.com/sgl-project/sglang/pull/21710
- 状态/时间: merged / 2026-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`；关联提交 `db60a620dbf1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+345/-5，可读 patch 448 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`；PR 正文摘要: Add GLM-5-FP8 nightly perf benchmarks (`bench_one_batch`) for MI30x and MI35x. Both accuracy and perf use `zai-org/GLM-5-FP8` with NSA tilelang backend, TP=8, FP8 KV cache, and...。
- 实现要点: `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0 (143 lines); hunks: -0,0 +1,143; symbols: generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass, test_glm5_perf，涉及 `generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass`；`test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass, test_bench_glm5，涉及 `generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass`；`test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2 (8 lines); hunks: -59,13 +59,17 @@ def get_display_name(self) -> str:; -77,7 +81,7 @@ def get_display_name(self) -> str:; symbols: get_display_name，涉及 `get_display_name`；`test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` modified +6/-2 (8 lines); hunks: -64,13 +64,17 @@ def get_display_name(self) -> str:; -82,7 +86,7 @@ def get_display_name(self) -> str:; symbols: get_display_name，涉及 `get_display_name`。
- 代码 diff 细节:
  - `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0 (143 lines); hunks: -0,0 +1,143; symbols: generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass, test_glm5_perf
  - `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass, test_bench_glm5
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2 (8 lines); hunks: -59,13 +59,17 @@ def get_display_name(self) -> str:; -77,7 +81,7 @@ def get_display_name(self) -> str:; symbols: get_display_name
  - `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` modified +6/-2 (8 lines); hunks: -64,13 +64,17 @@ def get_display_name(self) -> str:; -82,7 +86,7 @@ def get_display_name(self) -> str:; symbols: get_display_name
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0; `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2; `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` modified +6/-2
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22285 - Add CI tests for GLM-5

- 链接: https://github.com/sgl-project/sglang/pull/22285
- 状态/时间: merged / 2026-04-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+153/-30，可读 patch 301 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add CI tests for GLM-5」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`；PR 正文未提供可用摘要。
- 实现要点: `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1 (122 lines); hunks: -14,9 +14,10; -138,5 +139,124 @@ def test_bs_1_speed(self):; symbols: TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP, setUpClass，涉及 `TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP`；`test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29 (61 lines); hunks: -20,6 +20,7; -47,12 +48,13 @@ def setUpClass(cls):; symbols: TestDeepseekV32DPMTP, setUpClass, tearDownClass, test_bs_1_speed，涉及 `TestDeepseekV32DPMTP, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1 (122 lines); hunks: -14,9 +14,10; -138,5 +139,124 @@ def test_bs_1_speed(self):; symbols: TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP, setUpClass
  - `test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29 (61 lines); hunks: -20,6 +20,7; -47,12 +48,13 @@ def setUpClass(cls):; symbols: TestDeepseekV32DPMTP, setUpClass, tearDownClass, test_bs_1_speed
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1; `test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22399 - [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model

- 链接: https://github.com/sgl-project/sglang/pull/22399
- 状态/时间: merged / 2026-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/gb300/test_glm5_fp8.py`；关联提交 `46c2b7762765`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+82/-6，可读 patch 131 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add GLM-5.1 nightly tests and update Qwen3.5 model」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/gb300/test_glm5_fp8.py`；PR 正文摘要: - Add GLM-5.1 FP8 nightly test for H200/B200 (`nightly-8-gpu-common` suite) with TP8, TP8+DP8, and TP8+DP8+MTP variants - Update GB300 GLM-5 tests to GLM-5.1 model names (`zai-o...。
- 实现要点: `test/registered/gb300/test_glm5_fp8.py` modified +3/-3 (6 lines); hunks: -8,7 +8,7; -27,7 +27,7; symbols: TestGlm5Fp8, test_glm5_fp8，涉及 `TestGlm5Fp8, test_glm5_fp8`。
- 代码 diff 细节:
  - `test/registered/gb300/test_glm5_fp8.py` modified +3/-3 (6 lines); hunks: -8,7 +8,7; -27,7 +27,7; symbols: TestGlm5Fp8, test_glm5_fp8
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/gb300/test_glm5_fp8.py` modified +3/-3
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/8-gpu-models/test_qwen35.py`, `test/registered/gb300/test_glm5_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22336 - [AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x

- 链接: https://github.com/sgl-project/sglang/pull/22336
- 状态/时间: merged / 2026-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`；关联提交 `ef6bfc1197ab`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+918/-25，可读 patch 1064 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`；PR 正文摘要: Add GLM-5.1-FP8 nightly accuracy + perf benchmarks (`bench_one_batch`) for MI30x and MI35x. Both GLM-5-FP8 and GLM-5.1-FP8 share identical architecture (`GlmMoeDsaForCausalLM`,...。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples，涉及 `ModelConfig, get_display_name, get_one_example`；`test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples，涉及 `ModelConfig, get_display_name, get_one_example`；`test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass, test_glm51_perf，涉及 `generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass`；`test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass, test_bench_glm51，涉及 `generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass, test_glm51_perf
  - `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass, test_bench_glm51
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0; `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0; `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0; `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22712 - [NPU] update glm5 running guide

- 链接: https://github.com/sgl-project/sglang/pull/22712
- 状态/时间: merged / 2026-04-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/platforms/ascend/ascend_npu_glm5_examples.md`；关联提交 `13a4aafdbe69`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-2，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] update glm5 running guide」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `docs/platforms/ascend/ascend_npu_glm5_examples.md`；PR 正文摘要: Update NPU document, add best practice of GLM5 supported on ascend npu。
- 实现要点: `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2 (10 lines); hunks: -53,10 +53,16 @@ docker run -itd --shm-size=16g --privileged=true --name ${NA...。
- 代码 diff 细节:
  - `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2 (10 lines); hunks: -53,10 +53,16 @@ docker run -itd --shm-size=16g --privileged=true --name ${NA...
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs/platforms/ascend/ascend_npu_glm5_examples.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #22543 - GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix

- 链接: https://github.com/sgl-project/sglang/pull/22543
- 状态/时间: merged / 2026-04-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+8/-0，可读 patch 29 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/server_args.py`；PR 正文摘要: Addresses this issue regarding AMD Quark-quantized GLM-5 and GLM-5.1 MXFP4 checkpoints when using with SGLang (Exclude-layer names don't match SGLang internal names & Weight sha...。
- 实现要点: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0 (3 lines); hunks: -560,6 +560,9 @@ def post_load_weights(; symbols: post_load_weights，涉及 `post_load_weights`；`python/sglang/srt/model_loader/loader.py` modified +3/-0 (3 lines); hunks: -198,6 +198,9 @@ def _get_quantization_config(; symbols: _get_quantization_config，涉及 `_get_quantization_config`；`python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -1016,6 +1016,8 @@ def _handle_missing_default_values(self):; symbols: _handle_missing_default_values，涉及 `_handle_missing_default_values`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0 (3 lines); hunks: -560,6 +560,9 @@ def post_load_weights(; symbols: post_load_weights
  - `python/sglang/srt/model_loader/loader.py` modified +3/-0 (3 lines); hunks: -198,6 +198,9 @@ def _get_quantization_config(; symbols: _get_quantization_config
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -1016,6 +1016,8 @@ def _handle_missing_default_values(self):; symbols: _handle_missing_default_values
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0; `python/sglang/srt/model_loader/loader.py` modified +3/-0; `python/sglang/srt/server_args.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21773 - [AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x

- 链接: https://github.com/sgl-project/sglang/pull/21773
- 状态/时间: merged / 2026-04-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`；关联提交 `39c6bf730c41`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+528/-130，可读 patch 821 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`；PR 正文摘要: - Add nightly accuracy test (GSM8K 5-shot) and perf benchmark (\`bench_one_batch\`) for \`amd/GLM-5-MXFP4\` on MI35x 8-GPU - Remove obsolete base GLM-5 (BF16 NSA) CI jobs supers...。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: get_model_path, ModelConfig, __post_init__, get_display_name，涉及 `get_model_path, ModelConfig, __post_init__`；`test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0 (187 lines); hunks: -0,0 +1,187; symbols: generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x, setUpClass，涉及 `generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: get_model_path, ModelConfig, __post_init__, get_display_name
  - `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0 (187 lines); hunks: -0,0 +1,187; symbols: generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x, setUpClass
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0; `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22595 - fix: normalize tool message content for GLM5.1 chat template

- 链接: https://github.com/sgl-project/sglang/pull/22595
- 状态/时间: merged / 2026-04-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+67/-1，可读 patch 95 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: normalize tool message content for GLM5.1 chat template」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py`；PR 正文摘要: Fix: Normalize tool message `content` from array format to string before applying chat template Problem Per the OpenAI API specification%20chat.completions%20%3E%20(model)%20cha...。
- 实现要点: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0 (26 lines); hunks: -60,6 +60,28; -457,6 +479,10 @@ def _apply_jinja_template(; symbols: normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template，涉及 `normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template`；`test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1 (42 lines); hunks: -19,7 +19,10; -894,5 +897,42 @@ def test_required_without_parser_invalid_json_returns_none(...; symbols: test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened, test_multiple_text_parts_joined，涉及 `test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened`。
- 代码 diff 细节:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0 (26 lines); hunks: -60,6 +60,28; -457,6 +479,10 @@ def _apply_jinja_template(; symbols: normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template
  - `test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1 (42 lines); hunks: -19,7 +19,10; -894,5 +897,42 @@ def test_required_without_parser_invalid_json_returns_none(...; symbols: test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened, test_multiple_text_parts_joined
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0
  - tests: `test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1
- 验证与风险: diff 自带测试面 `test/registered/openai_server/basic/test_serving_chat.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22850 - [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)

- 链接: https://github.com/sgl-project/sglang/pull/22850
- 状态/时间: merged / 2026-04-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+24/-5，可读 patch 72 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；PR 正文摘要: Redundant kernels in the NSA indexer on HIP: weights_proj: - The ReplicatedLinear layer uses fp32 params_dtype, preventing tgemm from dispatching to the tuned bf16 fused kernel...。
- 实现要点: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunks: -14,7 +14,7; -32,14 +32,16; symbols: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache，涉及 `__init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunks: -14,7 +14,7; -32,14 +32,16; symbols: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23219 - [AMD] Enable MTP for GLM-5-mxfp4 model

- 链接: https://github.com/sgl-project/sglang/pull/23219
- 状态/时间: merged / 2026-04-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+41/-15，可读 patch 87 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Enable MTP for GLM-5-mxfp4 model」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_nextn.py`；PR 正文摘要: Fix https://github.com/sgl-project/sglang/issues/23142. Quark-quantized GLM-5-MXFP4 checkpoints store MTP (NextN) weights — including `eh_proj` — in FP4-packed format. The exist...。
- 实现要点: `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunks: -42,6 +42,7; -99,7 +100,18 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunks: -42,6 +42,7; -99,7 +100,18 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23060 - [fix] Fix dynamic chunking profiling crash on GLM-5 models

- 链接: https://github.com/sgl-project/sglang/pull/23060
- 状态/时间: merged / 2026-04-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[fix] Fix dynamic chunking profiling crash on GLM-5 models」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/managers/scheduler_pp_mixin.py`；PR 正文摘要: Fixes #23057 When `--enable-dynamic-chunking` is used with GLM-5 (have DeepEP ), the profiling phase crashes with `AttributeError: _is_extend_in_batch`. This silently disables d...。
- 实现要点: `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0 (3 lines); hunks: -20,6 +20,7; -631,6 +632,8 @@ def profile_and_init_predictor(self: Scheduler):; symbols: profile_and_init_predictor，涉及 `profile_and_init_predictor`。
- 代码 diff 细节:
  - `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0 (3 lines); hunks: -20,6 +20,7; -631,6 +632,8 @@ def profile_and_init_predictor(self: Scheduler):; symbols: profile_and_init_predictor
- 关键代码摘录:

```diff
diff -- python/sglang/srt/managers/scheduler_pp_mixin.py
@@ -20,6 +20,7 @@
+    set_is_extend_in_batch,
@@ -631,6 +632,8 @@ def profile_and_init_predictor(self: Scheduler):
+                set_is_extend_in_batch(batch.forward_mode.is_extend())
```

- 已读文件:
  - runtime: `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/scheduler_pp_mixin.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23540 - docs: split MI300X and MI325X options in GLM-5.1 generator

- 链接: https://github.com/sgl-project/sglang/pull/23540
- 状态/时间: merged / 2026-04-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；关联提交 `9b2f7f8a91d4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+15/-13，可读 patch 79 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: split MI300X and MI325X options in GLM-5.1 generator」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；PR 正文摘要: - split the GLM-5.1 hardware selector button `MI300X/MI325X` into separate `MI300X` and `MI325X` options - map `MI325X` to the same AMD config path and generated command that th...。
- 实现要点: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4 (10 lines); hunks: -14,7 +14,8 @@ export const GLM51Deployment = () => {; -23,7 +24,7 @@ export const GLM51Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4 (10 lines); hunks: -14,7 +14,8 @@ export const GLM51Deployment = () => {; -23,7 +24,7 @@ export const GLM51Deployment = () => {
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/docs.json`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
