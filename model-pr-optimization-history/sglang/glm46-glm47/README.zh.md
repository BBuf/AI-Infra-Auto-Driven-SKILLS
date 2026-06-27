# sglang GLM-4.6/4.7 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/GLM/GLM-4.6.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/GLM/GLM-4.7-Flash.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` | [#26384](https://github.com/sgl-project/sglang/pull/26384) |
| `python/sglang/srt/function_call/glm47_moe_detector.py` | [#15333](https://github.com/sgl-project/sglang/pull/15333), [#15753](https://github.com/sgl-project/sglang/pull/15753), [#28149](https://github.com/sgl-project/sglang/pull/28149) |
| `python/sglang/srt/function_call/glm4_moe_detector.py` | [#13989](https://github.com/sgl-project/sglang/pull/13989), [#15333](https://github.com/sgl-project/sglang/pull/15333), [#15753](https://github.com/sgl-project/sglang/pull/15753) |
| `python/sglang/srt/models/glm4_moe.py` | [#13873](https://github.com/sgl-project/sglang/pull/13873), [#14585](https://github.com/sgl-project/sglang/pull/14585), [#15333](https://github.com/sgl-project/sglang/pull/15333), [#17166](https://github.com/sgl-project/sglang/pull/17166), [#21403](https://github.com/sgl-project/sglang/pull/21403), [#21660](https://github.com/sgl-project/sglang/pull/21660), [#21851](https://github.com/sgl-project/sglang/pull/21851) |
| `python/sglang/srt/models/glm4_moe_lite.py` | [#21851](https://github.com/sgl-project/sglang/pull/21851), [#22509](https://github.com/sgl-project/sglang/pull/22509), [#26088](https://github.com/sgl-project/sglang/pull/26088), [#28516](https://github.com/sgl-project/sglang/pull/28516) |
| `python/sglang/srt/models/glm4_moe_lite_nextn.py` | [#26088](https://github.com/sgl-project/sglang/pull/26088) |
| `python/sglang/srt/models/glm4_moe_nextn.py` | [#13873](https://github.com/sgl-project/sglang/pull/13873) |
| `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py` | [#21534](https://github.com/sgl-project/sglang/pull/21534) |
| `test/registered/moe/test_glm4_moe_models.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 15
- 原文档显式引用补充 PR 数: 38
- 当前文档总 PR 数: 53
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-10-22 | [#11951](https://github.com/sgl-project/sglang/pull/11951) | open | WIP: Fix glm-4.6 tool call streaming parse | `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `sgl-router/tests/tool_parser_glm4_moe.rs` |
| 2025-11-05 | [#12456](https://github.com/sgl-project/sglang/pull/12456) | merged | [fix] Handle escaped characters in GLM tool call parser to prevent double serialization | `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-11-25 | [#13786](https://github.com/sgl-project/sglang/pull/13786) | merged | Overlap glm moe gemms in two cuda streams | `python/sglang/srt/models/glm4_moe.py` |
| 2025-12-01 | [#13873](https://github.com/sgl-project/sglang/pull/13873) | merged | Feat: GLM-4.6 supports shared experts fusion | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2025-12-08 | [#14568](https://github.com/sgl-project/sglang/pull/14568) | closed | update custom_logit_processor.py for. GLM-4.5 and GLM-4.6 support | `python/sglang/srt/models/glm4v_moe.py`, `docs/basic_usage/glmv.md`, `docs/basic_usage/glm45.md` |
| 2025-12-08 | [#14585](https://github.com/sgl-project/sglang/pull/14585) | merged | [Glm46v] Bug fix for accuracy drop and unable to launch server | `python/sglang/srt/models/glm4_moe.py` |
| 2025-12-13 | [#13989](https://github.com/sgl-project/sglang/pull/13989) | merged | Fix GLM-4.6 tool calls don't support streaming output for arguments i… | `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-12-20 | [#15333](https://github.com/sgl-project/sglang/pull/15333) | merged | [GLM-4.7] GLM-4.7 Tool Parser and Doc Update | `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-12-21 | [#15520](https://github.com/sgl-project/sglang/pull/15520) | merged | [model-gateway]: Tool parser for glm47 | `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs` |
| 2025-12-30 | [#15754](https://github.com/sgl-project/sglang/pull/15754) | merged | Fix: Handle empty func_name and None values in GLM MoE detectors | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-01-09 | [#15753](https://github.com/sgl-project/sglang/pull/15753) | merged | Fix GLM-4.7 MoE Detector complex JSON Schema type parsing | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-01-20 | [#17247](https://github.com/sgl-project/sglang/pull/17247) | merged | [New Model] GLM4.7-Flash | `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` |
| 2026-01-21 | [#17166](https://github.com/sgl-project/sglang/pull/17166) | merged | [Fix] GLM 4.7 + NVFP4 + MTP | `python/sglang/srt/models/glm4_moe.py` |
| 2026-01-24 | [#14668](https://github.com/sgl-project/sglang/pull/14668) | merged | [NVIDIA] Add flashinfer all-to-all MOE dispatcher | `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py` |
| 2026-02-17 | [#18930](https://github.com/sgl-project/sglang/pull/18930) | open | [AMD] Unit tests for mtp in GLM-4.7 | `python/sglang/srt/layers/attention/aiter_backend.py`, `test/registered/amd/test_glm4v_fp8_mtp.py` |
| 2026-02-20 | [#19040](https://github.com/sgl-project/sglang/pull/19040) | open | feat: add Glm4MoeLiteConfig and fix enable_a2a_moe for GLM-4.7-Flash | `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-02-21 | [#19106](https://github.com/sgl-project/sglang/pull/19106) | open | Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` |
| 2026-03-26 | [#21135](https://github.com/sgl-project/sglang/pull/21135) | merged | fix: use get_rope_config() to support models without rope_parameters | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py` |
| 2026-03-28 | [#21534](https://github.com/sgl-project/sglang/pull/21534) | merged | [AMD] Add GLM-4.7-FP8 accuracy CI test for MI35x | `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py` |
| 2026-03-30 | [#21660](https://github.com/sgl-project/sglang/pull/21660) | merged | [GLM-V and GLM-4.7] Cast to FP32 before gate projection for GLM model. | `python/sglang/srt/models/glm4_moe.py` |
| 2026-04-03 | [#19246](https://github.com/sgl-project/sglang/pull/19246) | merged | [NPU] optimize glm4.7 | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py` |
| 2026-04-04 | [#21851](https://github.com/sgl-project/sglang/pull/21851) | merged | GLM-4.7 and GLM-4.7-Flash Loading and import format | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-04-09 | [#20543](https://github.com/sgl-project/sglang/pull/20543) | merged | fix: do not strip whitespace from GLM tool call values | `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-04-11 | [#21403](https://github.com/sgl-project/sglang/pull/21403) | merged | [AMD] Fuse RMSNorm + FP8 per-token quant for GLM-4.7-FP8 | `python/sglang/srt/models/glm4_moe.py` |
| 2026-04-13 | [#22720](https://github.com/sgl-project/sglang/pull/22720) | merged | fix[glm4.7 flash]: properly detect `gfx95_quant_format` | `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-04-14 | [#22801](https://github.com/sgl-project/sglang/pull/22801) | open | [NPU]add dual-stream and deepep support for GLM-4.7-Flash | `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` |
| 2026-04-15 | [#22823](https://github.com/sgl-project/sglang/pull/22823) | merged | [Bugfix] Preserve auto-detected quant_config for GLM NextN draft model | `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2026-04-17 | [#23067](https://github.com/sgl-project/sglang/pull/23067) | open | Fix: forward continue_final_message kwargs in Glm45Detector | `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2026-04-22 | [#22509](https://github.com/sgl-project/sglang/pull/22509) | merged | [NPU]Fix GLM-4.7-Flash failed on NPU | `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-04-26 | [#23732](https://github.com/sgl-project/sglang/pull/23732) | merged | Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731) | `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py` |
| 2026-04-27 | [#23785](https://github.com/sgl-project/sglang/pull/23785) | merged | chore: update CI test est_time values | `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba2_mixer.py`, `test/registered/layers/mamba/test_mamba_ssm.py` |
| 2026-04-27 | [#23748](https://github.com/sgl-project/sglang/pull/23748) | merged | refactor(moe): centralize post-experts all-reduce skip predicate | `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-28 | [#22961](https://github.com/sgl-project/sglang/pull/22961) | merged | [NPU] Fix issue and support GLM-4.5V | `python/sglang/srt/models/glm4_moe.py` |
| 2026-05-14 | [#25197](https://github.com/sgl-project/sglang/pull/25197) | merged | ci: decouple stage and runner for cuda registry | `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py` |
| 2026-05-16 | [#25420](https://github.com/sgl-project/sglang/pull/25420) | merged | [CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI | `.github/workflows/pr-test-multimodal-gen.yml`, `test/registered/bench_fn/test_bench_serving_reasoning_stream.py`, `test/registered/function_call/test_kimik2_detector.py` |
| 2026-05-18 | [#22822](https://github.com/sgl-project/sglang/pull/22822) | merged | [Refactor] Refactor DeepEP dispatcher | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2026-05-18 | [#17869](https://github.com/sgl-project/sglang/pull/17869) | closed | [NPU]Support model GLM-4.7-Flash for npu, accuracy 81% | `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-05-19 | [#25524](https://github.com/sgl-project/sglang/pull/25524) | merged | [Bug Fix] Align glm4_moe_nextn NPU MTP loading with qwen3 MTP | `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2026-05-20 | [#25825](https://github.com/sgl-project/sglang/pull/25825) | merged | [Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool | `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/qwen2.py` |
| 2026-05-20 | [#25821](https://github.com/sgl-project/sglang/pull/25821) | merged | [Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-05-25 | [#22315](https://github.com/sgl-project/sglang/pull/22315) | closed | [Bugfix] Fix GLM-4.7-FP8 EAGLE accept_len=1.00 due to draft model loading with incorrect quant_config | `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2026-05-26 | [#26088](https://github.com/sgl-project/sglang/pull/26088) | merged | GLM-4.7-Flash: standalone MLA impl and MLA NextN/MTP | `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/glm4_moe_lite_nextn.py` |
| 2026-05-29 | [#26673](https://github.com/sgl-project/sglang/pull/26673) | merged | [refactor] remove unused op_mlp | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-06-02 | [#25813](https://github.com/sgl-project/sglang/pull/25813) | merged | docs(cookbook): port popular model usage guides into cookbook pages | `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` |
| 2026-06-02 | [#26384](https://github.com/sgl-project/sglang/pull/26384) | merged | [Docs] GLM-4.7 cookbook: add NVIDIA Blackwell (B200, GB200) + NVFP4 sections | `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` |
| 2026-06-03 | [#27001](https://github.com/sgl-project/sglang/pull/27001) | merged | [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests | `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` |
| 2026-06-10 | [#23906](https://github.com/sgl-project/sglang/pull/23906) | merged | [Refactor] Cuda Graph Runner/Backend Refactor | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` |
| 2026-06-11 | [#27964](https://github.com/sgl-project/sglang/pull/27964) | merged | [Spec] Retire Spec V1 | `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py` |
| 2026-06-12 | [#18383](https://github.com/sgl-project/sglang/pull/18383) | closed | [Bug Fix] Add missing use_mla guard in aiter_backend draft_extend CUD… | `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2026-06-14 | [#28149](https://github.com/sgl-project/sglang/pull/28149) | merged | Support GLM-4.7 function calling via structural tags | `python/sglang/srt/function_call/glm47_moe_detector.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-18 | [#28516](https://github.com/sgl-project/sglang/pull/28516) | merged | [NPU] Add MTP support for GLM-4.7-Flash | `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-06-25 | [#29261](https://github.com/sgl-project/sglang/pull/29261) | merged | [Docs] Fix broken links in cookbook | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` |

## 逐 PR diff 审计卡

### PR #11951 - WIP: Fix glm-4.6 tool call streaming parse

- 链接: https://github.com/sgl-project/sglang/pull/11951
- 状态/时间: open / 2025-10-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+450/-105，可读 patch 660 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「WIP: Fix glm-4.6 tool call streaming parse」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `sgl-router/tests/tool_parser_glm4_moe.rs`；技术摘要: 覆盖「WIP: Fix glm-4.6 tool call streaming parse」；主要实现面是 `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `sgl-router/tests/tool_parser_glm4_moe.rs`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs` modified +198/-86 (284 lines); hunks: -41,6 +41,9 @@ pub struct Glm4MoeParser {; -67,6 +70,7 @@ impl Glm4MoeParser {；`python/sglang/srt/function_call/glm4_moe_detector.py` modified +180/-19 (199 lines); hunks: -6,7 +6,11; -99,6 +103,7 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, _parse_partial_tool_call, _find_common_prefix, supports_structural_tag，涉及 `parse_streaming_increment, _parse_partial_tool_call, _find_common_prefix`；`sgl-router/tests/tool_parser_glm4_moe.rs` modified +72/-0 (72 lines); hunks: -167,3 +167,75 @@ async fn test_glm4_nested_json_in_arg_values() {。
- 代码 diff 细节:
  - `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs` modified +198/-86 (284 lines); hunks: -41,6 +41,9 @@ pub struct Glm4MoeParser {; -67,6 +70,7 @@ impl Glm4MoeParser {
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +180/-19 (199 lines); hunks: -6,7 +6,11; -99,6 +103,7 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, _parse_partial_tool_call, _find_common_prefix, supports_structural_tag
  - `sgl-router/tests/tool_parser_glm4_moe.rs` modified +72/-0 (72 lines); hunks: -167,3 +167,75 @@ async fn test_glm4_nested_json_in_arg_values() {
- 关键代码摘录:

```diff
diff -- sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs
@@ -41,6 +41,9 @@ pub struct Glm4MoeParser {
+    /// Whether the current tool's name has been sent (for streaming)
+    current_tool_name_sent: bool,
@@ -67,6 +70,7 @@ impl Glm4MoeParser {
+            current_tool_name_sent: false,
@@ -154,6 +158,79 @@ impl Glm4MoeParser {
+    /// Parse partial tool call from buffer (for streaming)
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -6,7 +6,11 @@
-from sglang.srt.function_call.core_types import StreamingParseResult, _GetInfoFunc
+from sglang.srt.function_call.core_types import (
+    StreamingParseResult,
+    ToolCallItem,
+    _GetInfoFunc,
+)
diff -- sgl-router/tests/tool_parser_glm4_moe.rs
@@ -167,3 +167,75 @@ async fn test_glm4_nested_json_in_arg_values() {
```

- 已读文件:
  - runtime: `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs` modified +198/-86; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +180/-19
  - tests: `sgl-router/tests/tool_parser_glm4_moe.rs` modified +72/-0
- 验证与风险: diff 自带测试面 `sgl-router/tests/tool_parser_glm4_moe.rs`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12456 - [fix] Handle escaped characters in GLM tool call parser to prevent double serialization

- 链接: https://github.com/sgl-project/sglang/pull/12456
- 状态/时间: merged / 2025-11-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+127/-13，可读 patch 172 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[fix] Handle escaped characters in GLM tool call parser to prevent double serialization」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；技术摘要: 覆盖「[fix] Handle escaped characters in GLM tool call parser to prevent double serialization」；主要实现面是 `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/srt/test_function_call_parser.py` modified +103/-0 (103 lines); hunks: -2191,6 +2191,109 @@ def test_partial_tool_call(self):; symbols: test_partial_tool_call, test_array_argument_with_escaped_json, check_params, check_single_todos，涉及 `test_partial_tool_call, test_array_argument_with_escaped_json, check_params`；`python/sglang/srt/function_call/glm4_moe_detector.py` modified +24/-13 (37 lines); hunks: -24,13 +24,23 @@ def get_argument_type(func_name: str, arg_key: str, defined_...; -45,8 +55,13 @@ def __init__(self):; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__，涉及 `get_argument_type, parse_arguments, Glm4MoeDetector`。
- 代码 diff 细节:
  - `test/srt/test_function_call_parser.py` modified +103/-0 (103 lines); hunks: -2191,6 +2191,109 @@ def test_partial_tool_call(self):; symbols: test_partial_tool_call, test_array_argument_with_escaped_json, check_params, check_single_todos
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +24/-13 (37 lines); hunks: -24,13 +24,23 @@ def get_argument_type(func_name: str, arg_key: str, defined_...; -45,8 +55,13 @@ def __init__(self):; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__
- 关键代码摘录:

```diff
diff -- test/srt/test_function_call_parser.py
@@ -2191,6 +2191,109 @@ def test_partial_tool_call(self):
+    def test_array_argument_with_escaped_json(self):
+        """Test that array arguments with escaped JSON are properly handled without double-escaping."""
+        # Add a tool with array parameter
+        tools_with_array = [
+            Tool(
+                type="function",
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -24,13 +24,23 @@ def get_argument_type(func_name: str, arg_key: str, defined_tools: list):
-        try:
-            parsed_value = json.loads(json_value)
-        except:
-            parsed_value = ast.literal_eval(json_value)
+        parsed_value = json.loads(json_value)
-        return json_value, False
```

- 已读文件:
  - tests: `test/srt/test_function_call_parser.py` modified +103/-0
  - runtime: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +24/-13
- 验证与风险: diff 自带测试面 `test/srt/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13786 - Overlap glm moe gemms in two cuda streams

- 链接: https://github.com/sgl-project/sglang/pull/13786
- 状态/时间: merged / 2025-11-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+47/-3，可读 patch 60 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Overlap glm moe gemms in two cuda streams」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「Overlap glm moe gemms in two cuda streams」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +47/-3 (50 lines); hunks: -448,12 +448,56 @@ def forward(; symbols: forward, forward_normal_dual_stream, forward_normal，涉及 `forward, forward_normal_dual_stream, forward_normal`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +47/-3 (50 lines); hunks: -448,12 +448,56 @@ def forward(; symbols: forward, forward_normal_dual_stream, forward_normal
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -448,12 +448,56 @@ def forward(
-            return self.forward_normal(
-                hidden_states, should_allreduce_fusion, use_reduce_scatter
-            )
+            if (
+                self.alt_stream is not None
+                and hidden_states.shape[0] > 0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +47/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13873 - Feat: GLM-4.6 supports shared experts fusion

- 链接: https://github.com/sgl-project/sglang/pull/13873
- 状态/时间: merged / 2025-12-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`；关联提交 `982db4ebac26`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+252/-24，可读 patch 431 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Feat: GLM-4.6 supports shared experts fusion」；模型线: GLM-4.6/4.7；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`；技术摘要: 覆盖「Feat: GLM-4.6 supports shared experts fusion」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +74/-19 (93 lines); hunks: -85,6 +85,7; -352,8 +353,14 @@ def __init__(; symbols: __init__, forward, forward_normal_dual_stream，涉及 `__init__, forward, forward_normal_dual_stream`；`python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-0 (4 lines); hunks: -139,6 +139,10 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +74/-19 (93 lines); hunks: -85,6 +85,7; -352,8 +353,14 @@ def __init__(; symbols: __init__, forward, forward_normal_dual_stream
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-0 (4 lines); hunks: -139,6 +139,10 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -85,6 +85,7 @@
+    log_info_on_rank0,
@@ -352,8 +353,14 @@ def __init__(
+        self.moe_ep_size = get_moe_expert_parallel_world_size()
+        self.num_fused_shared_experts = (
+            0
+            if get_global_server_args().disable_shared_experts_fusion
diff -- python/sglang/srt/models/glm4_moe_nextn.py
@@ -139,6 +139,10 @@ def __init__(
+        self.num_fused_shared_experts = (
+            0 if get_global_server_args().disable_shared_experts_fusion else 1
+        )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +74/-19; `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14568 - update custom_logit_processor.py for. GLM-4.5 and GLM-4.6 support

- 链接: https://github.com/sgl-project/sglang/pull/14568
- 状态/时间: closed / 2025-12-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+222/-1，可读 patch 269 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「update custom_logit_processor.py for. GLM-4.5 and GLM-4.6 support」；模型线: GLM-4.6/4.7；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/glm4v_moe.py`, `docs/basic_usage/glmv.md`, `docs/basic_usage/glm45.md`；技术摘要: 覆盖「update custom_logit_processor.py for. GLM-4.5 and GLM-4.6 support」；主要实现面是 `python/sglang/srt/models/glm4v_moe.py`, `docs/basic_usage/glmv.md`, `docs/basic_usage/glm45.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4v_moe.py` modified +4/-0 (4 lines); hunks: -7,6 +7,7; -36,7 +37,9 @@ def __init__(; symbols: __init__，涉及 `__init__`；`docs/basic_usage/glmv.md` added +136/-0 (136 lines); hunks: -0,0 +1,136；`docs/basic_usage/glm45.md` added +70/-0 (70 lines); hunks: -0,0 +1,70；`python/sglang/srt/sampling/custom_logit_processor.py` modified +8/-0 (8 lines); hunks: -112,6 +112,14 @@ def __call__(self, logits, custom_param_list: list[dict[str...; symbols: __call__, Glm4MoeThinkingBudgetLogitProcessor, Qwen3ThinkingBudgetLogitProcessor，涉及 `__call__, Glm4MoeThinkingBudgetLogitProcessor, Qwen3ThinkingBudgetLogitProcessor`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v_moe.py` modified +4/-0 (4 lines); hunks: -7,6 +7,7; -36,7 +37,9 @@ def __init__(; symbols: __init__
  - `docs/basic_usage/glmv.md` added +136/-0 (136 lines); hunks: -0,0 +1,136
  - `docs/basic_usage/glm45.md` added +70/-0 (70 lines); hunks: -0,0 +1,70
  - `python/sglang/srt/sampling/custom_logit_processor.py` modified +8/-0 (8 lines); hunks: -112,6 +112,14 @@ def __call__(self, logits, custom_param_list: list[dict[str...; symbols: __call__, Glm4MoeThinkingBudgetLogitProcessor, Qwen3ThinkingBudgetLogitProcessor
  - `docs/basic_usage/popular_model_usage.rst` modified +3/-1 (4 lines); hunks: -1,11 +1,13
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -7,6 +7,7 @@
+from sglang.srt.distributed.parallel_state import get_pp_group
@@ -36,7 +37,9 @@ def __init__(
+        self.pp_group = get_pp_group()
+        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
@@ -55,6 +58,7 @@ def __init__(
+            use_data_parallel=self.use_data_parallel,
diff -- docs/basic_usage/glmv.md
@@ -0,0 +1,136 @@
+# GLM-4.6V / GLM-4.5V Usage
+## Launch commands for SGLang
+Below are suggested launch commands tailored for different hardware / precision modes
+### FP8 (quantised) mode
+For high memory-efficiency and latency optimized deployments (e.g., on H100, H200) where FP8 checkpoint is supported:
+'''bash
diff -- docs/basic_usage/glm45.md
@@ -0,0 +1,70 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v_moe.py` modified +4/-0; `python/sglang/srt/sampling/custom_logit_processor.py` modified +8/-0
  - docs: `docs/basic_usage/glmv.md` added +136/-0; `docs/basic_usage/glm45.md` added +70/-0; `docs/basic_usage/popular_model_usage.rst` modified +3/-1; `docs/advanced_features/dp_for_multi_modal_encoder.md` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/sampling/custom_logit_processor.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14585 - [Glm46v] Bug fix for accuracy drop and unable to launch server

- 链接: https://github.com/sgl-project/sglang/pull/14585
- 状态/时间: merged / 2025-12-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe.py`；关联提交 `cf0478d602ce`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+308/-29，可读 patch 530 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Glm46v] Bug fix for accuracy drop and unable to launch server」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「[Glm46v] Bug fix for accuracy drop and unable to launch server」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +1/-0 (1 lines); hunks: -361,6 +361,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-0 (1 lines); hunks: -361,6 +361,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -361,6 +361,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/pyproject.toml`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/configs/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13989 - Fix GLM-4.6 tool calls don't support streaming output for arguments i…

- 链接: https://github.com/sgl-project/sglang/pull/13989
- 状态/时间: merged / 2025-12-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/glm4_moe_detector.py`；关联提交 `80554598d33b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+527/-81，可读 patch 700 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GLM-4.6 tool calls don't support streaming output for arguments i…」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/glm4_moe_detector.py`；技术摘要: 覆盖「Fix GLM-4.6 tool calls don't support streaming output for arguments i…」；主要实现面是 `python/sglang/srt/function_call/glm4_moe_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +498/-66 (564 lines); hunks: -2,16 +2,43; -21,32 +48,90 @@ def get_argument_type(func_name: str, arg_key: str, defined_...; symbols: get_argument_type, StreamState, parse_arguments，涉及 `get_argument_type, StreamState, parse_arguments`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +498/-66 (564 lines); hunks: -2,16 +2,43; -21,32 +48,90 @@ def get_argument_type(func_name: str, arg_key: str, defined_...; symbols: get_argument_type, StreamState, parse_arguments
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -2,16 +2,43 @@
-from typing import List
+from enum import Enum
+from typing import Any, Dict, List, Optional, Tuple
-from sglang.srt.function_call.core_types import StreamingParseResult, _GetInfoFunc
+from sglang.srt.function_call.core_types import (
+    StreamingParseResult,
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +498/-66
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15333 - [GLM-4.7] GLM-4.7 Tool Parser and Doc Update

- 链接: https://github.com/sgl-project/sglang/pull/15333
- 状态/时间: merged / 2025-12-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`；关联提交 `b82c7a0ae744`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+809/-394，可读 patch 1356 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[GLM-4.7] GLM-4.7 Tool Parser and Doc Update」；模型线: GLM-4.6/4.7；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「[GLM-4.7] GLM-4.7 Tool Parser and Doc Update」；主要实现面是 `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/glm47_moe_detector.py` added +584/-0 (584 lines); hunks: -0,0 +1,584; symbols: StreamState, get_argument_type, _convert_to_number, parse_arguments，涉及 `StreamState, get_argument_type, _convert_to_number`；`python/sglang/srt/function_call/glm4_moe_detector.py` modified +5/-2 (7 lines); hunks: -43,9 +43,12 @@ def get_argument_type(; symbols: get_argument_type, _convert_to_number，涉及 `get_argument_type, _convert_to_number`；`python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -12,7 +12,7。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/glm47_moe_detector.py` added +584/-0 (584 lines); hunks: -0,0 +1,584; symbols: StreamState, get_argument_type, _convert_to_number, parse_arguments
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +5/-2 (7 lines); hunks: -43,9 +43,12 @@ def get_argument_type(; symbols: get_argument_type, _convert_to_number
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -12,7 +12,7
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/glm47_moe_detector.py
@@ -0,0 +1,584 @@
+import ast
+import json
+import logging
+import re
+from enum import Enum
+from typing import Any, Dict, List, Optional, Tuple
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -43,9 +43,12 @@ def get_argument_type(
-    if arg_key not in tool.function.parameters["properties"]:
+    properties = (tool.function.parameters or {}).get("properties", {})
+    if not isinstance(properties, dict):
+        properties = {}
+    if arg_key not in properties:
-    return tool.function.parameters["properties"][arg_key].get("type", None)
diff -- python/sglang/srt/models/glm4_moe.py
@@ -12,7 +12,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/glm47_moe_detector.py` added +584/-0; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +5/-2; `python/sglang/srt/models/glm4_moe.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15520 - [model-gateway]: Tool parser for glm47

- 链接: https://github.com/sgl-project/sglang/pull/15520
- 状态/时间: merged / 2025-12-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+179/-26，可读 patch 392 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[model-gateway]: Tool parser for glm47」；模型线: GLM-4.6/4.7；类别: 文档/测试/CI；主要 diff: `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs`；技术摘要: 覆盖「[model-gateway]: Tool parser for glm47」；主要实现面是 `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `sgl-model-gateway/tests/tool_parser_glm47_moe.rs` added +132/-0 (132 lines); hunks: -0,0 +1,132；`sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs` modified +22/-8 (30 lines); hunks: -14,8 +14,9 @@ use crate::{; -47,13 +48,17 @@ pub struct Glm4MoeParser {；`sgl-model-gateway/tests/tool_parser_glm4_moe.rs` modified +7/-7 (14 lines); hunks: -7,7 +7,7 @@ use common::create_test_tools;; -30,7 +30,7 @@ The weather will be..."#;；`sgl-model-gateway/src/tool_parser/factory.rs` modified +5/-3 (8 lines); hunks: -239,7 +239,8 @@ impl ParserFactory {; -281,8 +282,9 @@ impl ParserFactory {。
- 代码 diff 细节:
  - `sgl-model-gateway/tests/tool_parser_glm47_moe.rs` added +132/-0 (132 lines); hunks: -0,0 +1,132
  - `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs` modified +22/-8 (30 lines); hunks: -14,8 +14,9 @@ use crate::{; -47,13 +48,17 @@ pub struct Glm4MoeParser {
  - `sgl-model-gateway/tests/tool_parser_glm4_moe.rs` modified +7/-7 (14 lines); hunks: -7,7 +7,7 @@ use common::create_test_tools;; -30,7 +30,7 @@ The weather will be..."#;
  - `sgl-model-gateway/src/tool_parser/factory.rs` modified +5/-3 (8 lines); hunks: -239,7 +239,8 @@ impl ParserFactory {; -281,8 +282,9 @@ impl ParserFactory {
  - `sgl-model-gateway/benches/tool_parser_benchmark.rs` modified +5/-2 (7 lines); hunks: -80,7 +80,7 @@ Let me examine the scan results and provide recommendations."#;; -94,6 +94,8 @@ analyze_customer_behavior
- 关键代码摘录:

```diff
diff -- sgl-model-gateway/tests/tool_parser_glm47_moe.rs
@@ -0,0 +1,132 @@
+//! GLM-4.7 MoE Parser Integration Tests
+use sgl_model_gateway::tool_parser::{Glm4MoeParser, ToolParser};
+mod common;
+use common::create_test_tools;
+#[tokio::test]
+async fn test_glm47_complete_parsing() {
diff -- sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs
@@ -14,8 +14,9 @@ use crate::{
-/// Handles the GLM-4 MoE specific format:
-/// `<tool_call>{name}\n<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n</tool_call>`
+/// Handles both GLM-4 MoE and GLM-4.7 MoE formats:
+/// - GLM-4: `<tool_call>{name}\n<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n</tool_call>`
+/// - GLM-4.7: `<tool_call>{name}<arg_key>{key}</arg_key><arg_value>{value}</arg_value></tool_call>`
@@ -47,13 +48,17 @@ pub struct Glm4MoeParser {
diff -- sgl-model-gateway/tests/tool_parser_glm4_moe.rs
@@ -7,7 +7,7 @@ use common::create_test_tools;
```

- 已读文件:
  - tests: `sgl-model-gateway/tests/tool_parser_glm47_moe.rs` added +132/-0; `sgl-model-gateway/tests/tool_parser_glm4_moe.rs` modified +7/-7
  - runtime: `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs` modified +22/-8; `sgl-model-gateway/src/tool_parser/factory.rs` modified +5/-3; `sgl-model-gateway/benches/tool_parser_benchmark.rs` modified +5/-2; `sgl-model-gateway/src/reasoning_parser/README.md` modified +4/-3; `sgl-model-gateway/src/reasoning_parser/factory.rs` modified +1/-0
  - other: `sgl-model-gateway/README.md` modified +3/-3
- 验证与风险: diff 自带测试面 `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15754 - Fix: Handle empty func_name and None values in GLM MoE detectors

- 链接: https://github.com/sgl-project/sglang/pull/15754
- 状态/时间: merged / 2025-12-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+1513/-140，可读 patch 1786 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix: Handle empty func_name and None values in GLM MoE detectors」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；技术摘要: 覆盖「Fix: Handle empty func_name and None values in GLM MoE detectors」；主要实现面是 `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/function_call/test_glm47_moe_detector.py` added +1176/-0 (1176 lines); hunks: -0,0 +1,1176; symbols: TestGlm47MoeDetector, setUp, test_single_tool_call, test_multiple_tool_calls，涉及 `TestGlm47MoeDetector, setUp, test_single_tool_call`；`python/sglang/srt/function_call/glm47_moe_detector.py` modified +303/-132 (435 lines); hunks: -40,15 +40,27 @@ def get_argument_type(; -143,6 +155,10 @@ def __init__(self):; symbols: get_argument_type, _convert_to_number, __init__, _reset_streaming_state，涉及 `get_argument_type, _convert_to_number, __init__`；`python/sglang/srt/function_call/glm4_moe_detector.py` modified +19/-8 (27 lines); hunks: -189,8 +189,10 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; -426,10 +428,19 @@ def parse_streaming_increment(; symbols: detect_and_parse, parse_streaming_increment，涉及 `detect_and_parse, parse_streaming_increment`；`test/registered/function_call/test_function_call_parser.py` modified +15/-0 (15 lines); hunks: -2257,6 +2257,21 @@ def check_single_todos(tool_result, expected):; symbols: check_single_todos, test_empty_function_name_handling, TestGlm47MoeDetector, setUp，涉及 `check_single_todos, test_empty_function_name_handling, TestGlm47MoeDetector`。
- 代码 diff 细节:
  - `test/registered/function_call/test_glm47_moe_detector.py` added +1176/-0 (1176 lines); hunks: -0,0 +1,1176; symbols: TestGlm47MoeDetector, setUp, test_single_tool_call, test_multiple_tool_calls
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +303/-132 (435 lines); hunks: -40,15 +40,27 @@ def get_argument_type(; -143,6 +155,10 @@ def __init__(self):; symbols: get_argument_type, _convert_to_number, __init__, _reset_streaming_state
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +19/-8 (27 lines); hunks: -189,8 +189,10 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; -426,10 +428,19 @@ def parse_streaming_increment(; symbols: detect_and_parse, parse_streaming_increment
  - `test/registered/function_call/test_function_call_parser.py` modified +15/-0 (15 lines); hunks: -2257,6 +2257,21 @@ def check_single_todos(tool_result, expected):; symbols: check_single_todos, test_empty_function_name_handling, TestGlm47MoeDetector, setUp
- 关键代码摘录:

```diff
diff -- test/registered/function_call/test_glm47_moe_detector.py
@@ -0,0 +1,1176 @@
+import json
+import unittest
+from sglang.srt.entrypoints.openai.protocol import Function, Tool
+from sglang.srt.function_call.core_types import StreamingParseResult
+from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
+from sglang.test.ci.ci_register import register_cpu_ci
diff -- python/sglang/srt/function_call/glm47_moe_detector.py
@@ -40,15 +40,27 @@ def get_argument_type(
-    if func_name not in name2tool:
+    # Check if function exists
+    tool = name2tool.get(func_name)
+    if not tool:
+        return None
+    # Get parameters safely using getattr
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -189,8 +189,10 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult
```

- 已读文件:
  - tests: `test/registered/function_call/test_glm47_moe_detector.py` added +1176/-0; `test/registered/function_call/test_function_call_parser.py` modified +15/-0
  - runtime: `python/sglang/srt/function_call/glm47_moe_detector.py` modified +303/-132; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +19/-8
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_function_call_parser.py`, `test/registered/function_call/test_glm47_moe_detector.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15753 - Fix GLM-4.7 MoE Detector complex JSON Schema type parsing

- 链接: https://github.com/sgl-project/sglang/pull/15753
- 状态/时间: merged / 2026-01-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；关联提交 `8ef5b9052825`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+869/-20，可读 patch 989 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GLM-4.7 MoE Detector complex JSON Schema type parsing」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；技术摘要: 覆盖「Fix GLM-4.7 MoE Detector complex JSON Schema type parsing」；主要实现面是 `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/function_call/test_glm47_moe_detector.py` modified +678/-3 (681 lines); hunks: -3,7 +3,11; -1172,5 +1176,676 @@ def test_streamed_raw_length_multiple_empty_returns(self):; symbols: test_streamed_raw_length_multiple_empty_returns, TestGlm4ComplexJsonSchema, setUp, test_get_argument_type_simple_type，涉及 `test_streamed_raw_length_multiple_empty_returns, TestGlm4ComplexJsonSchema, setUp`；`python/sglang/srt/function_call/glm47_moe_detector.py` modified +43/-10 (53 lines); hunks: -12,6 +12,7; -31,6 +32,14 @@ def get_argument_type(; symbols: get_argument_type, _get_value_type, _format_value_complete，涉及 `get_argument_type, _get_value_type, _format_value_complete`；`python/sglang/srt/function_call/glm4_moe_detector.py` modified +44/-6 (50 lines); hunks: -12,6 +12,7; -31,6 +32,14 @@ def get_argument_type(; symbols: get_argument_type, _convert_to_number, _get_value_type, _format_value_complete，涉及 `get_argument_type, _convert_to_number, _get_value_type`。
- 代码 diff 细节:
  - `test/registered/function_call/test_glm47_moe_detector.py` modified +678/-3 (681 lines); hunks: -3,7 +3,11; -1172,5 +1176,676 @@ def test_streamed_raw_length_multiple_empty_returns(self):; symbols: test_streamed_raw_length_multiple_empty_returns, TestGlm4ComplexJsonSchema, setUp, test_get_argument_type_simple_type
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +43/-10 (53 lines); hunks: -12,6 +12,7; -31,6 +32,14 @@ def get_argument_type(; symbols: get_argument_type, _get_value_type, _format_value_complete
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +44/-6 (50 lines); hunks: -12,6 +12,7; -31,6 +32,14 @@ def get_argument_type(; symbols: get_argument_type, _convert_to_number, _get_value_type, _format_value_complete
- 关键代码摘录:

```diff
diff -- test/registered/function_call/test_glm47_moe_detector.py
@@ -3,7 +3,11 @@
-from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
+from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
+from sglang.srt.function_call.glm47_moe_detector import (
+    Glm47MoeDetector,
+    get_argument_type,
+)
diff -- python/sglang/srt/function_call/glm47_moe_detector.py
@@ -12,6 +12,7 @@
+from sglang.srt.function_call.utils import infer_type_from_json_schema
@@ -31,6 +32,14 @@ def get_argument_type(
+    Supports complex JSON Schema definitions including:
+    - Direct type field (including type arrays)
+    - anyOf/oneOf: parameter can be any of multiple types
+    - enum: parameter must be one of enum values
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -12,6 +12,7 @@
```

- 已读文件:
  - tests: `test/registered/function_call/test_glm47_moe_detector.py` modified +678/-3
  - runtime: `python/sglang/srt/function_call/glm47_moe_detector.py` modified +43/-10; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +44/-6
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_glm47_moe_detector.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17247 - [New Model] GLM4.7-Flash

- 链接: https://github.com/sgl-project/sglang/pull/17247
- 状态/时间: merged / 2026-01-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+842/-12，可读 patch 940 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[New Model] GLM4.7-Flash」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`；技术摘要: 覆盖「[New Model] GLM4.7-Flash」；主要实现面是 `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_lite.py` added +808/-0 (808 lines); hunks: -0,0 +1,808; symbols: Glm4MoeLiteMLP, __init__, forward, Glm4MoeLiteGate，涉及 `Glm4MoeLiteMLP, __init__, forward`；`python/sglang/srt/configs/model_config.py` modified +19/-9 (28 lines); hunks: -269,7 +269,10 @@ def _config_draft_model(self):; -375,6 +378,7 @@ def _derive_model_shapes(self):; symbols: _config_draft_model, _derive_model_shapes，涉及 `_config_draft_model, _derive_model_shapes`；`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +7/-2 (9 lines); hunks: -17,7 +17,7; -472,7 +472,12 @@ def _concat_and_cast_mha_k(; symbols: _concat_and_cast_mha_k，涉及 `_concat_and_cast_mha_k`；`python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: -685,6 +685,8 @@ def __init__(; -699,7 +701,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_lite.py` added +808/-0 (808 lines); hunks: -0,0 +1,808; symbols: Glm4MoeLiteMLP, __init__, forward, Glm4MoeLiteGate
  - `python/sglang/srt/configs/model_config.py` modified +19/-9 (28 lines); hunks: -269,7 +269,10 @@ def _config_draft_model(self):; -375,6 +378,7 @@ def _derive_model_shapes(self):; symbols: _config_draft_model, _derive_model_shapes
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +7/-2 (9 lines); hunks: -17,7 +17,7; -472,7 +472,12 @@ def _concat_and_cast_mha_k(; symbols: _concat_and_cast_mha_k
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: -685,6 +685,8 @@ def __init__(; -699,7 +701,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +2/-0 (2 lines); hunks: -454,6 +454,7 @@ def _apply_jinja_template(; -476,6 +477,7 @@ def _apply_jinja_template(; symbols: _apply_jinja_template
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -0,0 +1,808 @@
+# Copyright 2025-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/model_config.py
@@ -269,7 +269,10 @@ def _config_draft_model(self):
-        if is_draft_model and self.hf_config.architectures[0] == "Glm4MoeForCausalLM":
+        if is_draft_model and self.hf_config.architectures[0] in [
+            "Glm4MoeForCausalLM",
+            "Glm4MoeLiteForCausalLM",
+        ]:
@@ -375,6 +378,7 @@ def _derive_model_shapes(self):
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py
@@ -17,7 +17,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_lite.py` added +808/-0; `python/sglang/srt/configs/model_config.py` modified +19/-9; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +7/-2; `python/sglang/srt/models/glm4_moe.py` modified +3/-1; `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +2/-0; `python/sglang/srt/server_args.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17166 - [Fix] GLM 4.7 + NVFP4 + MTP

- 链接: https://github.com/sgl-project/sglang/pull/17166
- 状态/时间: merged / 2026-01-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe.py`；关联提交 `2ff0880a0ed1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+114/-9，可读 patch 206 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] GLM 4.7 + NVFP4 + MTP」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「[Fix] GLM 4.7 + NVFP4 + MTP」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +5/-1 (6 lines); hunks: -63,7 +63,10; -376,6 +379,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +5/-1 (6 lines); hunks: -63,7 +63,10; -376,6 +379,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -63,7 +63,10 @@
-from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert
+from sglang.srt.layers.moe.utils import (
+    RoutingMethodType,
+    filter_moe_weight_param_global_expert,
+)
@@ -376,6 +379,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/model_loader/weight_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14668 - [NVIDIA] Add flashinfer all-to-all MOE dispatcher

- 链接: https://github.com/sgl-project/sglang/pull/14668
- 状态/时间: merged / 2026-01-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+723/-16，可读 patch 935 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NVIDIA] Add flashinfer all-to-all MOE dispatcher」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`；技术摘要: 覆盖「[NVIDIA] Add flashinfer all-to-all MOE dispatcher」；主要实现面是 `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: FlashinferDispatchOutput, format, FlashinferCombineInput, FlashinferDispatcher，涉及 `FlashinferDispatchOutput, format, FlashinferCombineInput`；`python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: CommBackend, when, TorchDistributedCommBackend, __init__，涉及 `CommBackend, when, TorchDistributedCommBackend`；`python/sglang/srt/layers/quantization/modelopt_quant.py` modified +23/-14 (37 lines); hunks: -18,6 +18,7; -1479,6 +1480,7 @@ def _slice_scale(w):; symbols: _slice_scale, apply，涉及 `_slice_scale, apply`；`python/sglang/srt/layers/moe/token_dispatcher/base.py` modified +19/-0 (19 lines); hunks: -25,6 +25,8; -149,12 +151,19 @@ def format_is_deepep(; symbols: format_is_deepep, format_is_flashinfer, DispatchOutputFormat, is_standard，涉及 `format_is_deepep, format_is_flashinfer, DispatchOutputFormat`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: FlashinferDispatchOutput, format, FlashinferCombineInput, FlashinferDispatcher
  - `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: CommBackend, when, TorchDistributedCommBackend, __init__
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +23/-14 (37 lines); hunks: -18,6 +18,7; -1479,6 +1480,7 @@ def _slice_scale(w):; symbols: _slice_scale, apply
  - `python/sglang/srt/layers/moe/token_dispatcher/base.py` modified +19/-0 (19 lines); hunks: -25,6 +25,8; -149,12 +151,19 @@ def format_is_deepep(; symbols: format_is_deepep, format_is_flashinfer, DispatchOutputFormat, is_standard
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-0 (9 lines); hunks: -37,6 +37,7; -117,6 +118,14 @@ def create_moe_dispatcher(moe_runner_config: MoeRunnerConfi...; symbols: create_moe_dispatcher
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py
@@ -0,0 +1,263 @@
+from __future__ import annotations
+import logging
+from typing import NamedTuple, Optional
+import torch
+from sglang.srt.environ import envs
+from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
diff -- python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py
@@ -0,0 +1,47 @@
+import torch.distributed as dist
+from sglang.srt.utils import is_flashinfer_available
+if is_flashinfer_available():
+    from flashinfer.comm.mnnvl import CommBackend
+else:
+    class CommBackend:
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -18,6 +18,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` added +263/-0; `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` added +47/-0; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +23/-14; `python/sglang/srt/layers/moe/token_dispatcher/base.py` modified +19/-0; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-0; `python/sglang/srt/layers/moe/token_dispatcher/__init__.py` modified +6/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/test_flashinfer_dispatcher.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18930 - [AMD] Unit tests for mtp in GLM-4.7

- 链接: https://github.com/sgl-project/sglang/pull/18930
- 状态/时间: open / 2026-02-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+120/-1，可读 patch 129 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Unit tests for mtp in GLM-4.7」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/aiter_backend.py`, `test/registered/amd/test_glm4v_fp8_mtp.py`；技术摘要: 覆盖「[AMD] Unit tests for mtp in GLM-4.7」；主要实现面是 `python/sglang/srt/layers/attention/aiter_backend.py`, `test/registered/amd/test_glm4v_fp8_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-1 (3 lines); hunks: -999,7 +999,8 @@ def init_forward_metadata_capture_cuda_graph(; symbols: init_forward_metadata_capture_cuda_graph，涉及 `init_forward_metadata_capture_cuda_graph`；`test/registered/amd/test_glm4v_fp8_mtp.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: TestGLM47FP8TPMTP, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestGLM47FP8TPMTP, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-1 (3 lines); hunks: -999,7 +999,8 @@ def init_forward_metadata_capture_cuda_graph(; symbols: init_forward_metadata_capture_cuda_graph
  - `test/registered/amd/test_glm4v_fp8_mtp.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: TestGLM47FP8TPMTP, setUpClass, tearDownClass, test_a_gsm8k
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -999,7 +999,8 @@ def init_forward_metadata_capture_cuda_graph(
-                if _use_mla_ps_kernel:
+                # https://github.com/sgl-project/sglang/pull/18383/changes
+                if self.use_mla and _use_mla_ps_kernel:
diff -- test/registered/amd/test_glm4v_fp8_mtp.py
@@ -0,0 +1,118 @@
+import unittest
+from types import SimpleNamespace
+import requests
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_amd_ci
+from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-1
  - tests: `test/registered/amd/test_glm4v_fp8_mtp.py` added +118/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_glm4v_fp8_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19040 - feat: add Glm4MoeLiteConfig and fix enable_a2a_moe for GLM-4.7-Flash

- 链接: https://github.com/sgl-project/sglang/pull/19040
- 状态/时间: open / 2026-02-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+52/-0，可读 patch 88 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: add Glm4MoeLiteConfig and fix enable_a2a_moe for GLM-4.7-Flash」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/models/glm4_moe_lite.py`；技术摘要: 覆盖「feat: add Glm4MoeLiteConfig and fix enable_a2a_moe for GLM-4.7-Flash」；主要实现面是 `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/glm4_moe_lite.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: Glm4MoeLiteConfig, with, __init__，涉及 `Glm4MoeLiteConfig, with, __init__`；`python/sglang/srt/configs/__init__.py` modified +2/-0 (2 lines); hunks: -7,6 +7,7; -53,6 +54,7；`python/sglang/srt/models/glm4_moe_lite.py` modified +1/-0 (1 lines); hunks: -435,6 +435,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0 (2 lines); hunks: -53,6 +53,7; -93,6 +94,7。
- 代码 diff 细节:
  - `python/sglang/srt/configs/glm4_moe_lite.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: Glm4MoeLiteConfig, with, __init__
  - `python/sglang/srt/configs/__init__.py` modified +2/-0 (2 lines); hunks: -7,6 +7,7; -53,6 +54,7
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +1/-0 (1 lines); hunks: -435,6 +435,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0 (2 lines); hunks: -53,6 +53,7; -93,6 +94,7
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/glm4_moe_lite.py
@@ -0,0 +1,47 @@
+# Copyright 2025-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/__init__.py
@@ -7,6 +7,7 @@
+from sglang.srt.configs.glm4_moe_lite import Glm4MoeLiteConfig
@@ -53,6 +54,7 @@
+    "Glm4MoeLiteConfig",
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -435,6 +435,7 @@ def __init__(
+        self.enable_a2a_moe = False  # Glm4MoeLite does not use all-to-all MoE dispatch
diff -- python/sglang/srt/utils/hf_transformers_utils.py
@@ -53,6 +53,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/glm4_moe_lite.py` added +47/-0; `python/sglang/srt/configs/__init__.py` modified +2/-0; `python/sglang/srt/models/glm4_moe_lite.py` modified +1/-0; `python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19106 - Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks

- 链接: https://github.com/sgl-project/sglang/pull/19106
- 状态/时间: open / 2026-02-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+505/-37，可读 patch 677 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`；技术摘要: 覆盖「Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks」；主要实现面是 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +52/-27 (79 lines); hunks: -1275,40 +1275,66 @@ def __init__(; -2791,8 +2817,18 @@ def forward(; symbols: __init__, forward, DeepseekV2ForCausalLM，涉及 `__init__, forward, DeepseekV2ForCausalLM`；`python/sglang/srt/models/glm4_moe_lite.py` modified +52/-8 (60 lines); hunks: -132,16 +132,13 @@ def forward(; -467,6 +464,17 @@ def __init__(; symbols: forward, __init__, Glm4MoeLiteForCausalLM, determine_num_fused_shared_experts，涉及 `forward, __init__, Glm4MoeLiteForCausalLM`；`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +54/-0 (54 lines); hunks: -35,6 +35,7; -93,6 +94,55 @@ class DeepseekV2WeightLoaderMixin:; symbols: DeepseekV2WeightLoaderMixin, _dequantize_ct_wna16_weight, do_load_weights, post_load_weights，涉及 `DeepseekV2WeightLoaderMixin, _dequantize_ct_wna16_weight, do_load_weights`；`python/sglang/srt/models/glm4_moe.py` modified +16/-0 (16 lines); hunks: -1001,6 +1001,13 @@ def forward(; -1047,6 +1054,15 @@ def determine_num_fused_shared_experts(self):; symbols: forward, Glm4MoeForCausalLM, __init__, determine_num_fused_shared_experts，涉及 `forward, Glm4MoeForCausalLM, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +52/-27 (79 lines); hunks: -1275,40 +1275,66 @@ def __init__(; -2791,8 +2817,18 @@ def forward(; symbols: __init__, forward, DeepseekV2ForCausalLM
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +52/-8 (60 lines); hunks: -132,16 +132,13 @@ def forward(; -467,6 +464,17 @@ def __init__(; symbols: forward, __init__, Glm4MoeLiteForCausalLM, determine_num_fused_shared_experts
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +54/-0 (54 lines); hunks: -35,6 +35,7; -93,6 +94,55 @@ class DeepseekV2WeightLoaderMixin:; symbols: DeepseekV2WeightLoaderMixin, _dequantize_ct_wna16_weight, do_load_weights, post_load_weights
  - `python/sglang/srt/models/glm4_moe.py` modified +16/-0 (16 lines); hunks: -1001,6 +1001,13 @@ def forward(; -1047,6 +1054,15 @@ def determine_num_fused_shared_experts(self):; symbols: forward, Glm4MoeForCausalLM, __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/configs/model_config.py` modified +14/-1 (15 lines); hunks: -1009,7 +1009,20 @@ def _verify_transformers_version(self):; symbols: _verify_transformers_version
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1275,40 +1275,66 @@ def __init__(
-        # If we have self.fused_qkv_a_proj_with_mqa and we're running on CPU, we will choose the torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight kernel
-        # which requires self.w_kc and self.w_vc to be packed.
-        # If not, we will use torch.bmm and weight shouldn't be packed in this case
-        has_fused_proj = hasattr(self, "fused_qkv_a_proj_with_mqa")
+        # If we have self.fused_qkv_a_proj_with_mqa and we're running on CPU,
+        # we will choose the torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -132,16 +132,13 @@ def forward(
-        # Some quantization wrappers store the underlying parameter as `weight_packed`.
-        if not hasattr(self.gate_up_proj, "weight"):
-            self.gate_up_proj.weight = getattr(self.gate_up_proj, "weight_packed")
-        if not hasattr(self.down_proj, "weight"):
-            self.down_proj.weight = getattr(self.down_proj, "weight_packed")
+        gate_up_proj_weight = getattr(self.gate_up_proj, "weight", None)
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -35,6 +35,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +52/-27; `python/sglang/srt/models/glm4_moe_lite.py` modified +52/-8; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +54/-0; `python/sglang/srt/models/glm4_moe.py` modified +16/-0; `python/sglang/srt/configs/model_config.py` modified +14/-1; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +6/-1
  - tests: `test/registered/core/test_deepseek_weight_loader.py` added +86/-0; `test/registered/core/test_model_config_transformers_version.py` added +84/-0
- 验证与风险: diff 自带测试面 `test/registered/core/test_deepseek_attention_backend_handler.py`, `test/registered/core/test_deepseek_packed_modules_mapping.py`, `test/registered/core/test_deepseek_weight_loader.py`, `test/registered/core/test_glm4_moe_lite_shared_experts_fusion.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21135 - fix: use get_rope_config() to support models without rope_parameters

- 链接: https://github.com/sgl-project/sglang/pull/21135
- 状态/时间: merged / 2026-03-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+44/-42，可读 patch 342 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: use get_rope_config() to support models without rope_parameters」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py`；技术摘要: 覆盖「fix: use get_rope_config() to support models without rope_parameters」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +5/-5 (10 lines); hunks: -94,6 +94,7; -684,11 +685,10 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/glm4.py` modified +5/-3 (8 lines); hunks: -52,6 +52,7; -217,9 +218,10 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/grok.py` modified +2/-5 (7 lines); hunks: -61,6 +61,7; -477,11 +478,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/llada2.py` modified +4/-2 (6 lines); hunks: -84,6 +84,7; -486,12 +487,13 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +5/-5 (10 lines); hunks: -94,6 +94,7; -684,11 +685,10 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4.py` modified +5/-3 (8 lines); hunks: -52,6 +52,7; -217,9 +218,10 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/grok.py` modified +2/-5 (7 lines); hunks: -61,6 +61,7; -477,11 +478,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/llada2.py` modified +4/-2 (6 lines); hunks: -84,6 +84,7; -486,12 +487,13 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/deepseek.py` modified +2/-2 (4 lines); hunks: -49,6 +49,7; -310,8 +311,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -94,6 +94,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_rope_config
@@ -684,11 +685,10 @@ def __init__(
-        rope_theta = config.rope_parameters["rope_theta"]
-        rope_scaling = config.rope_parameters
-        partial_rotary_factor = getattr(
-            getattr(config, "rope_parameters", None), "partial_rotary_factor", None
diff -- python/sglang/srt/models/glm4.py
@@ -52,6 +52,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_rope_config
@@ -217,9 +218,10 @@ def __init__(
-        rope_theta = config.rope_parameters["rope_theta"]
-        rope_scaling = config.rope_parameters
-        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 0.5)
+        rope_theta, rope_scaling = get_rope_config(config)
diff -- python/sglang/srt/models/grok.py
@@ -61,6 +61,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +5/-5; `python/sglang/srt/models/glm4.py` modified +5/-3; `python/sglang/srt/models/grok.py` modified +2/-5; `python/sglang/srt/models/llada2.py` modified +4/-2; `python/sglang/srt/models/deepseek.py` modified +2/-2; `python/sglang/srt/models/ernie4.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/ernie4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21534 - [AMD] Add GLM-4.7-FP8 accuracy CI test for MI35x

- 链接: https://github.com/sgl-project/sglang/pull/21534
- 状态/时间: merged / 2026-03-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`；关联提交 `7078e385ea13`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+96/-0，可读 patch 118 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add GLM-4.7-FP8 accuracy CI test for MI35x」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`；技术摘要: 覆盖「[AMD] Add GLM-4.7-FP8 accuracy CI test for MI35x」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py` added +61/-0 (61 lines); hunks: -0,0 +1,61; symbols: TestGLM47FP8EvalMI35x, test_glm_47_fp8，涉及 `TestGLM47FP8EvalMI35x, test_glm_47_fp8`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py` added +61/-0 (61 lines); hunks: -0,0 +1,61; symbols: TestGLM47FP8EvalMI35x, test_glm_47_fp8
- 关键代码摘录:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py
@@ -0,0 +1,61 @@
+"""MI35x GLM-4.7-FP8 GSM8K Accuracy Evaluation Test (8-GPU)
+Tests GLM-4.7-FP8 accuracy using GSM8K benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-glm47-fp8 suite
+"""
+import os
+# Set HF cache for MI35x
```

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py` added +61/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21660 - [GLM-V and GLM-4.7] Cast to FP32 before gate projection for GLM model.

- 链接: https://github.com/sgl-project/sglang/pull/21660
- 状态/时间: merged / 2026-03-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe.py`；关联提交 `ad064c2f4e33`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-1，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[GLM-V and GLM-4.7] Cast to FP32 before gate projection for GLM model.」；模型线: GLM-4.6/4.7；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「[GLM-V and GLM-4.7] Cast to FP32 before gate projection for GLM model.」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: -327,9 +327,14 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: -327,9 +327,14 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -327,9 +327,14 @@ def __init__(
+        # GLM requires FP32 gate projection; cache to avoid per-forward cast.
+        # FIXME: if gate weight is updated at runtime (e.g. expert rebalancing), _weight_fp32 must be invalidated.
+        self.register_buffer("_weight_fp32", None, persistent=False)
-        logits = F.linear(hidden_states, self.weight, None)
+        if self._weight_fp32 is None:
+            self._weight_fp32 = self.weight.data.to(torch.float32)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19246 - [NPU] optimize glm4.7

- 链接: https://github.com/sgl-project/sglang/pull/19246
- 状态/时间: merged / 2026-04-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+146/-15，可读 patch 259 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] optimize glm4.7」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`；技术摘要: 覆盖「[NPU] optimize glm4.7」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +61/-11 (72 lines); hunks: -34,6 +34,7; -91,6 +92,7; symbols: Glm4MoeMLP, __init__, forward_prepare, forward_deepep，涉及 `Glm4MoeMLP, __init__, forward_prepare`；`python/sglang/srt/models/glm4_moe_nextn.py` modified +19/-2 (21 lines); hunks: -14,6 +14,7; -22,6 +23,7; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +2/-2 (4 lines); hunks: -58,13 +58,13 @@ def _rmsnorm_forward_oot(; symbols: _rmsnorm_forward_oot，涉及 `_rmsnorm_forward_oot`；`python/sglang/srt/hardware_backend/npu/utils.py` modified +64/-0 (64 lines); hunks: -178,3 +178,67 @@ def get_indexer_weight_stream():; symbols: get_indexer_weight_stream, get_share_stream, set_share_stream, get_routed_stream，涉及 `get_indexer_weight_stream, get_share_stream, set_share_stream`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +61/-11 (72 lines); hunks: -34,6 +34,7; -91,6 +92,7; symbols: Glm4MoeMLP, __init__, forward_prepare, forward_deepep
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +19/-2 (21 lines); hunks: -14,6 +14,7; -22,6 +23,7; symbols: __init__, forward
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +2/-2 (4 lines); hunks: -58,13 +58,13 @@ def _rmsnorm_forward_oot(; symbols: _rmsnorm_forward_oot
  - `python/sglang/srt/hardware_backend/npu/utils.py` modified +64/-0 (64 lines); hunks: -178,3 +178,67 @@ def get_indexer_weight_stream():; symbols: get_indexer_weight_stream, get_share_stream, set_share_stream, get_routed_stream
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -34,6 +34,7 @@
+from sglang.srt.environ import envs
@@ -91,6 +92,7 @@
+    is_npu,
@@ -102,10 +104,19 @@
+_is_npu = is_npu()
+if _is_npu:
diff -- python/sglang/srt/models/glm4_moe_nextn.py
@@ -14,6 +14,7 @@
+import contextlib
@@ -22,6 +23,7 @@
+from sglang.srt.environ import temp_set_env
@@ -126,7 +128,10 @@ def __init__(
-        self.quant_config = quant_config
+        self.needs_quant_draft = (
diff -- python/sglang/srt/layers/quantization/modelslim/modelslim.py
@@ -58,13 +58,13 @@ def _rmsnorm_forward_oot(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +61/-11; `python/sglang/srt/models/glm4_moe_nextn.py` modified +19/-2; `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +2/-2; `python/sglang/srt/hardware_backend/npu/utils.py` modified +64/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/hardware_backend/npu/utils.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21851 - GLM-4.7 and GLM-4.7-Flash Loading and import format

- 链接: https://github.com/sgl-project/sglang/pull/21851
- 状态/时间: merged / 2026-04-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`；关联提交 `b7ae3b5a9a57`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+139/-86，可读 patch 486 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「GLM-4.7 and GLM-4.7-Flash Loading and import format」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`；技术摘要: 覆盖「GLM-4.7 and GLM-4.7-Flash Loading and import format」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +130/-57 (187 lines); hunks: -15,13 +15,15; -62,6 +64,7; symbols: __init__, get_moe_weights，涉及 `__init__, get_moe_weights`；`python/sglang/srt/models/glm4_moe_lite.py` modified +9/-29 (38 lines); hunks: -12,13 +12,15; -29,12 +31,14; symbols: forward, __init__, load_weights，涉及 `forward, __init__, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +130/-57 (187 lines); hunks: -15,13 +15,15; -62,6 +64,7; symbols: __init__, get_moe_weights
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +9/-29 (38 lines); hunks: -12,13 +12,15; -29,12 +31,14; symbols: forward, __init__, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -15,13 +15,15 @@
+import re
+from sglang.srt.batch_overlap.single_batch_overlap import SboFlags
@@ -62,6 +64,7 @@
+from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
@@ -172,7 +175,7 @@ def __init__(
-        rope_theta: float = 10000,
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -12,13 +12,15 @@
-"""Inference-only GLM-Lite model compatible with HuggingFace weights"""
+"""Inference-only GLM-4.7-Flash model compatible with HuggingFace weights"""
+import re
+from sgl_kernel import dsv3_router_gemm
@@ -29,12 +31,14 @@
+from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +130/-57; `python/sglang/srt/models/glm4_moe_lite.py` modified +9/-29
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20543 - fix: do not strip whitespace from GLM tool call values

- 链接: https://github.com/sgl-project/sglang/pull/20543
- 状态/时间: merged / 2026-04-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+66/-2，可读 patch 96 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: do not strip whitespace from GLM tool call values」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；技术摘要: 覆盖「fix: do not strip whitespace from GLM tool call values」；主要实现面是 `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/function_call/test_function_call_parser.py` modified +66/-0 (66 lines); hunks: -2270,6 +2270,39 @@ def test_empty_function_name_handling(self):; -2546,6 +2579,39 @@ def check_single_todos(tool_result, expected):; symbols: test_empty_function_name_handling, test_whitespace_preserved_in_arg_values, TestGlm47MoeDetector, setUp，涉及 `test_empty_function_name_handling, test_whitespace_preserved_in_arg_values, TestGlm47MoeDetector`；`python/sglang/srt/function_call/glm47_moe_detector.py` modified +0/-1 (1 lines); hunks: -759,7 +759,6 @@ def _parse_argument_pairs(; symbols: _parse_argument_pairs，涉及 `_parse_argument_pairs`；`python/sglang/srt/function_call/glm4_moe_detector.py` modified +0/-1 (1 lines); hunks: -613,7 +613,6 @@ def _parse_argument_pairs(; symbols: _parse_argument_pairs，涉及 `_parse_argument_pairs`。
- 代码 diff 细节:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +66/-0 (66 lines); hunks: -2270,6 +2270,39 @@ def test_empty_function_name_handling(self):; -2546,6 +2579,39 @@ def check_single_todos(tool_result, expected):; symbols: test_empty_function_name_handling, test_whitespace_preserved_in_arg_values, TestGlm47MoeDetector, setUp
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +0/-1 (1 lines); hunks: -759,7 +759,6 @@ def _parse_argument_pairs(; symbols: _parse_argument_pairs
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +0/-1 (1 lines); hunks: -613,7 +613,6 @@ def _parse_argument_pairs(; symbols: _parse_argument_pairs
- 关键代码摘录:

```diff
diff -- test/registered/unit/function_call/test_function_call_parser.py
@@ -2270,6 +2270,39 @@ def test_empty_function_name_handling(self):
+    def test_whitespace_preserved_in_arg_values(self):
+        """Test that leading/trailing whitespace in arg values is not stripped."""
+        tools_with_string = [
+            Tool(
+                type="function",
+                function=Function(
diff -- python/sglang/srt/function_call/glm47_moe_detector.py
@@ -759,7 +759,6 @@ def _parse_argument_pairs(
-            arg_value = arg_value.strip()
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -613,7 +613,6 @@ def _parse_argument_pairs(
-            arg_value = arg_value.strip()
```

- 已读文件:
  - tests: `test/registered/unit/function_call/test_function_call_parser.py` modified +66/-0
  - runtime: `python/sglang/srt/function_call/glm47_moe_detector.py` modified +0/-1; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +0/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21403 - [AMD] Fuse RMSNorm + FP8 per-token quant for GLM-4.7-FP8

- 链接: https://github.com/sgl-project/sglang/pull/21403
- 状态/时间: merged / 2026-04-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe.py`；关联提交 `7e4e1dcd7ac8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+149/-13，可读 patch 269 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fuse RMSNorm + FP8 per-token quant for GLM-4.7-FP8」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「[AMD] Fuse RMSNorm + FP8 per-token quant for GLM-4.7-FP8」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +58/-3 (61 lines); hunks: -289,7 +289,9 @@ def forward_prepare(; -865,6 +867,51 @@ def __init__(; symbols: forward_prepare, __init__, _detect_fp8_per_token_quant, _detect_attn_quant_format，涉及 `forward_prepare, __init__, _detect_fp8_per_token_quant`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +58/-3 (61 lines); hunks: -289,7 +289,9 @@ def forward_prepare(; -865,6 +867,51 @@ def __init__(; symbols: forward_prepare, __init__, _detect_fp8_per_token_quant, _detect_attn_quant_format
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -289,7 +289,9 @@ def forward_prepare(
-        if hidden_states.shape[0] == 0:
+        # hidden_states can be a (fp8_tensor, scale) tuple from fused RMSNorm+Quant
+        hs = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
+        if hs.shape[0] == 0:
@@ -865,6 +867,51 @@ def __init__(
+        # Detect if QKV uses aiter FP8 per-token quant so we can fuse
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +58/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22720 - fix[glm4.7 flash]: properly detect `gfx95_quant_format`

- 链接: https://github.com/sgl-project/sglang/pull/22720
- 状态/时间: merged / 2026-04-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix[glm4.7 flash]: properly detect `gfx95_quant_format`」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe_lite.py`；技术摘要: 覆盖「fix[glm4.7 flash]: properly detect `gfx95_quant_format`」；主要实现面是 `python/sglang/srt/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_lite.py` modified +2/-0 (2 lines); hunks: -403,6 +403,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +2/-0 (2 lines); hunks: -403,6 +403,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -403,6 +403,8 @@ def __init__(
+        self._gfx95_quant_format = self._detect_gfx95_quant_format()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_lite.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22801 - [NPU]add dual-stream and deepep support for GLM-4.7-Flash

- 链接: https://github.com/sgl-project/sglang/pull/22801
- 状态/时间: open / 2026-04-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-3，可读 patch 52 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]add dual-stream and deepep support for GLM-4.7-Flash」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`；技术摘要: 覆盖「[NPU]add dual-stream and deepep support for GLM-4.7-Flash」；主要实现面是 `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_lite.py` modified +13/-2 (15 lines); hunks: -30,6 +30,7; -58,6 +59,7; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunks: -609,7 +609,7 @@ def _dispatch_core(; symbols: _dispatch_core，涉及 `_dispatch_core`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +13/-2 (15 lines); hunks: -30,6 +30,7; -58,6 +59,7; symbols: __init__, forward
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunks: -609,7 +609,7 @@ def _dispatch_core(; symbols: _dispatch_core
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -30,6 +30,7 @@
+from sglang.srt.environ import envs
@@ -58,6 +59,7 @@
+from sglang.srt.model_executor.forward_batch_info import ForwardBatch
@@ -178,7 +180,12 @@ def __init__(
-    def forward(self, hidden_states, gemm_output_zero_allocator: BumpAllocator = None):
+    def forward(
diff -- python/sglang/srt/layers/moe/token_dispatcher/deepep.py
@@ -609,7 +609,7 @@ def _dispatch_core(
-        else:
+        elif not envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_lite.py` modified +13/-2; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22823 - [Bugfix] Preserve auto-detected quant_config for GLM NextN draft model

- 链接: https://github.com/sgl-project/sglang/pull/22823
- 状态/时间: merged / 2026-04-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-1，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Preserve auto-detected quant_config for GLM NextN draft model」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe_nextn.py`；技术摘要: 覆盖「[Bugfix] Preserve auto-detected quant_config for GLM NextN draft model」；主要实现面是 `python/sglang/srt/models/glm4_moe_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_nextn.py` modified +2/-1 (3 lines); hunks: -129,7 +129,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +2/-1 (3 lines); hunks: -129,7 +129,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_nextn.py
@@ -129,7 +129,8 @@ def __init__(
-            get_global_server_args().speculative_draft_model_quantization
+            get_global_server_args().speculative_draft_model_quantization is not None
+            or quant_config is not None
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_nextn.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23067 - Fix: forward continue_final_message kwargs in Glm45Detector

- 链接: https://github.com/sgl-project/sglang/pull/23067
- 状态/时间: open / 2026-04-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+66/-1，可读 patch 94 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix: forward continue_final_message kwargs in Glm45Detector」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`；技术摘要: 覆盖「Fix: forward continue_final_message kwargs in Glm45Detector」；主要实现面是 `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/parser/test_reasoning_parser.py` modified +57/-0 (57 lines); hunks: -518,6 +518,39 @@ def test_forced_reasoning_mode(self):; -1248,6 +1281,30 @@ def test_continue_final_message_with_request(self):; symbols: test_forced_reasoning_mode, test_continue_final_message_accepts_kwargs, test_continue_final_message_think_start_in_previous, test_continue_final_message_think_end_in_previous，涉及 `test_forced_reasoning_mode, test_continue_final_message_accepts_kwargs, test_continue_final_message_think_start_in_previous`；`python/sglang/srt/parser/reasoning_parser.py` modified +9/-1 (10 lines); hunks: -314,13 +314,21 @@ class Glm45Detector(BaseReasoningFormatDetector):; symbols: Glm45Detector, __init__，涉及 `Glm45Detector, __init__`。
- 代码 diff 细节:
  - `test/registered/unit/parser/test_reasoning_parser.py` modified +57/-0 (57 lines); hunks: -518,6 +518,39 @@ def test_forced_reasoning_mode(self):; -1248,6 +1281,30 @@ def test_continue_final_message_with_request(self):; symbols: test_forced_reasoning_mode, test_continue_final_message_accepts_kwargs, test_continue_final_message_think_start_in_previous, test_continue_final_message_think_end_in_previous
  - `python/sglang/srt/parser/reasoning_parser.py` modified +9/-1 (10 lines); hunks: -314,13 +314,21 @@ class Glm45Detector(BaseReasoningFormatDetector):; symbols: Glm45Detector, __init__
- 关键代码摘录:

```diff
diff -- test/registered/unit/parser/test_reasoning_parser.py
@@ -518,6 +518,39 @@ def test_forced_reasoning_mode(self):
+    def test_continue_final_message_accepts_kwargs(self):
+        """Regression: Glm45Detector must accept continue_final_message and
+        previous_content kwargs (forwarded by ReasoningParser when the request
+        sets continue_final_message=True with a trailing assistant message)."""
+        detector = Glm45Detector(
+            continue_final_message=True,
diff -- python/sglang/srt/parser/reasoning_parser.py
@@ -314,13 +314,21 @@ class Glm45Detector(BaseReasoningFormatDetector):
-    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
+    def __init__(
+        self,
+        stream_reasoning: bool = True,
+        force_reasoning: bool = False,
+        continue_final_message: bool = False,
```

- 已读文件:
  - tests: `test/registered/unit/parser/test_reasoning_parser.py` modified +57/-0
  - runtime: `python/sglang/srt/parser/reasoning_parser.py` modified +9/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/parser/test_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22509 - [NPU]Fix GLM-4.7-Flash failed on NPU

- 链接: https://github.com/sgl-project/sglang/pull/22509
- 状态/时间: merged / 2026-04-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe_lite.py`；关联提交 `92f28e9ba80b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-2，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]Fix GLM-4.7-Flash failed on NPU」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe_lite.py`；技术摘要: 覆盖「[NPU]Fix GLM-4.7-Flash failed on NPU」；主要实现面是 `python/sglang/srt/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_lite.py` modified +3/-1 (4 lines); hunks: -20,7 +20,6; -81,6 +80,9。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +3/-1 (4 lines); hunks: -20,7 +20,6; -81,6 +80,9
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -20,7 +20,6 @@
-from sgl_kernel import dsv3_router_gemm
@@ -81,6 +80,9 @@
+if _is_cuda:
+    from sgl_kernel import dsv3_router_gemm
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_lite.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23732 - Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)

- 链接: https://github.com/sgl-project/sglang/pull/23732
- 状态/时间: merged / 2026-04-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+59/-12，可读 patch 290 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)」；模型线: GLM-4.6/4.7；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py`；技术摘要: 覆盖「Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)」；主要实现面是 `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llada2.py` modified +10/-2 (12 lines); hunks: -55,7 +55,11; -379,7 +383,11 @@ def forward_normal(; symbols: forward_normal，涉及 `forward_normal`；`python/sglang/srt/models/hunyuan_v3.py` modified +7/-4 (11 lines); hunks: -34,6 +34,7; -191,10 +192,11 @@ def _forward_single_stream(self, hidden_states: torch.Tens...; symbols: _forward_single_stream, _forward_dual_stream，涉及 `_forward_single_stream, _forward_dual_stream`；`python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1 (8 lines); hunks: -34,6 +34,7; -347,7 +348,12 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/exaone_moe.py` modified +6/-2 (8 lines); hunks: -47,7 +47,7; -300,7 +300,11 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llada2.py` modified +10/-2 (12 lines); hunks: -55,7 +55,11; -379,7 +383,11 @@ def forward_normal(; symbols: forward_normal
  - `python/sglang/srt/models/hunyuan_v3.py` modified +7/-4 (11 lines); hunks: -34,6 +34,7; -191,10 +192,11 @@ def _forward_single_stream(self, hidden_states: torch.Tens...; symbols: _forward_single_stream, _forward_dual_stream
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1 (8 lines); hunks: -34,6 +34,7; -347,7 +348,12 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/exaone_moe.py` modified +6/-2 (8 lines); hunks: -47,7 +47,7; -300,7 +300,11 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/llama4.py` modified +6/-1 (7 lines); hunks: -39,6 +39,7; -145,7 +146,11 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llada2.py
@@ -55,7 +55,11 @@
-from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
+from sglang.srt.layers.moe import (
+    get_deepep_mode,
+    get_moe_a2a_backend,
+    should_use_dp_reduce_scatterv,
+)
diff -- python/sglang/srt/models/hunyuan_v3.py
@@ -34,6 +34,7 @@
+from sglang.srt.layers.moe import should_use_dp_reduce_scatterv
@@ -191,10 +192,11 @@ def _forward_single_stream(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        if self.ep_size > 1:
+        skip_post_reduce = should_use_dp_reduce_scatterv()
+        if self.ep_size > 1 and not skip_post_reduce:
-        if self.tp_size > 1:
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -34,6 +34,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llada2.py` modified +10/-2; `python/sglang/srt/models/hunyuan_v3.py` modified +7/-4; `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1; `python/sglang/srt/models/exaone_moe.py` modified +6/-2; `python/sglang/srt/models/llama4.py` modified +6/-1; `python/sglang/srt/models/sarvam_moe.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23785 - chore: update CI test est_time values

- 链接: https://github.com/sgl-project/sglang/pull/23785
- 状态/时间: merged / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 268 个文件，+269/-269，可读 patch 2404 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「chore: update CI test est_time values」；模型线: GLM-4.6/4.7；类别: 文档/测试/CI；主要 diff: `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba2_mixer.py`, `test/registered/layers/mamba/test_mamba_ssm.py`；技术摘要: 覆盖「chore: update CI test est_time values」；主要实现面是 `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba2_mixer.py`, `test/registered/layers/mamba/test_mamba_ssm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6；`test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7；`test/registered/layers/mamba/test_mamba_ssm.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6；`test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6。
- 代码 diff 细节:
  - `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6
  - `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7
  - `test/registered/layers/mamba/test_mamba_ssm.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6
  - `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6
  - `test/registered/models/test_compressed_tensors_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7
- 关键代码摘录:

```diff
diff -- test/registered/layers/mamba/test_causal_conv1d.py
@@ -1,6 +1,6 @@
-register_cuda_ci(est_time=13, suite="stage-b-test-1-gpu-small")
+register_cuda_ci(est_time=11, suite="stage-b-test-1-gpu-small")
diff -- test/registered/layers/mamba/test_mamba2_mixer.py
@@ -15,7 +15,7 @@
-register_cuda_ci(est_time=28, suite="stage-b-test-2-gpu-large")
+register_cuda_ci(est_time=32, suite="stage-b-test-2-gpu-large")
diff -- test/registered/layers/mamba/test_mamba_ssm.py
@@ -1,6 +1,6 @@
-register_cuda_ci(est_time=9, suite="stage-b-test-1-gpu-small")
+register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")
diff -- test/registered/layers/mamba/test_mamba_ssm_ssd.py
@@ -1,6 +1,6 @@
-register_cuda_ci(est_time=9, suite="stage-b-test-1-gpu-small")
+register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")
diff -- test/registered/models/test_compressed_tensors_models.py
@@ -13,7 +13,7 @@
```

- 已读文件:
  - tests: `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1; `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1; `test/registered/layers/mamba/test_mamba_ssm.py` modified +1/-1; `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +1/-1; `test/registered/models/test_compressed_tensors_models.py` modified +1/-1; `test/registered/models/test_generation_models.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23748 - refactor(moe): centralize post-experts all-reduce skip predicate

- 链接: https://github.com/sgl-project/sglang/pull/23748
- 状态/时间: merged / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+134/-132，可读 patch 532 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「refactor(moe): centralize post-experts all-reduce skip predicate」；模型线: GLM-4.6/4.7；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py`；技术摘要: 覆盖「refactor(moe): centralize post-experts all-reduce skip predicate」；主要实现面是 `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/utils.py` modified +33/-0 (33 lines); hunks: -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():; symbols: should_use_dp_reduce_scatterv, should_skip_post_experts_all_reduce, speculative_moe_backend_context，涉及 `should_use_dp_reduce_scatterv, should_skip_post_experts_all_reduce, speculative_moe_backend_context`；`python/sglang/srt/models/sarvam_moe.py` modified +9/-16 (25 lines); hunks: -39,10 +39,7; -373,12 +370,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal，涉及 `forward_normal_dual_stream, forward_normal`；`python/sglang/srt/models/deepseek_v2.py` modified +9/-13 (22 lines); hunks: -85,7 +85,7; -651,12 +651,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, _post_combine_hook，涉及 `forward_normal_dual_stream, _post_combine_hook`；`python/sglang/srt/models/glm4_moe.py` modified +9/-13 (22 lines); hunks: -61,7 +61,7; -594,12 +594,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal，涉及 `forward_normal_dual_stream, forward_normal`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/utils.py` modified +33/-0 (33 lines); hunks: -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():; symbols: should_use_dp_reduce_scatterv, should_skip_post_experts_all_reduce, speculative_moe_backend_context
  - `python/sglang/srt/models/sarvam_moe.py` modified +9/-16 (25 lines); hunks: -39,10 +39,7; -373,12 +370,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-13 (22 lines); hunks: -85,7 +85,7; -651,12 +651,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, _post_combine_hook
  - `python/sglang/srt/models/glm4_moe.py` modified +9/-13 (22 lines); hunks: -61,7 +61,7; -594,12 +594,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-13 (22 lines); hunks: -50,8 +50,7; -332,20 +331,17 @@ def forward_normal(; symbols: forward_normal
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/utils.py
@@ -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():
+def should_skip_post_experts_all_reduce(
+    *,
+    is_tp_path: bool,
+    use_reduce_scatter: bool = False,
+    should_allreduce_fusion: bool = False,
+) -> bool:
diff -- python/sglang/srt/models/sarvam_moe.py
@@ -39,10 +39,7 @@
-from sglang.srt.layers.moe import (
-    should_use_dp_reduce_scatterv,
-    should_use_flashinfer_cutlass_moe_fp4_allgather,
-)
+from sglang.srt.layers.moe import should_skip_post_experts_all_reduce
@@ -373,12 +370,10 @@ def forward_normal_dual_stream(
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -85,7 +85,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/utils.py` modified +33/-0; `python/sglang/srt/models/sarvam_moe.py` modified +9/-16; `python/sglang/srt/models/deepseek_v2.py` modified +9/-13; `python/sglang/srt/models/glm4_moe.py` modified +9/-13; `python/sglang/srt/models/qwen3_moe.py` modified +9/-13; `python/sglang/srt/models/hunyuan_v3.py` modified +13/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/__init__.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22961 - [NPU] Fix issue and support GLM-4.5V

- 链接: https://github.com/sgl-project/sglang/pull/22961
- 状态/时间: merged / 2026-04-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+17/-5，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Fix issue and support GLM-4.5V」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「[NPU] Fix issue and support GLM-4.5V」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +17/-5 (22 lines); hunks: -314,18 +314,30 @@ def forward_prepare(; symbols: forward_prepare，涉及 `forward_prepare`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +17/-5 (22 lines); hunks: -314,18 +314,30 @@ def forward_prepare(; symbols: forward_prepare
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -314,18 +314,30 @@ def forward_prepare(
+            if self.use_qk_norm:
+                eps = self.q_norm.variance_epsilon
+                q_weight = self.q_norm.weight
+                k_weight = self.k_norm.weight
+                q_bias = getattr(self.q_norm, "bias", None)
+                k_bias = getattr(self.k_norm, "bias", None)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +17/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25197 - ci: decouple stage and runner for cuda registry

- 链接: https://github.com/sgl-project/sglang/pull/25197
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 261 个文件，+388/-293，可读 patch 2625 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: decouple stage and runner for cuda registry」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py`；技术摘要: 覆盖「ci: decouple stage and runner for cuda registry」；主要实现面是 `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1 (3 lines); hunks: -19,7 +19,8；`test/registered/models/test_dummy_grok_models.py` modified +2/-1 (3 lines); hunks: -5,7 +5,8；`test/registered/models/test_ministral3_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8；`test/registered/models/test_ministral4_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8。
- 代码 diff 细节:
  - `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1 (3 lines); hunks: -19,7 +19,8
  - `test/registered/models/test_dummy_grok_models.py` modified +2/-1 (3 lines); hunks: -5,7 +5,8
  - `test/registered/models/test_ministral3_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8
  - `test/registered/models/test_ministral4_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8
  - `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +2/-1 (3 lines); hunks: -6,7 +6,8
- 关键代码摘录:

```diff
diff -- test/registered/layers/test_fla_layernorm_guard.py
@@ -19,7 +19,8 @@
-    suite="stage-b-test-2-gpu-large",
+    stage="stage-b",
+    runner_config="2-gpu-large",
diff -- test/registered/models/test_dummy_grok_models.py
@@ -5,7 +5,8 @@
-    suite="stage-b-test-2-gpu-large",
+    stage="stage-b",
+    runner_config="2-gpu-large",
diff -- test/registered/models/test_ministral3_models.py
@@ -8,7 +8,8 @@
-    suite="stage-b-test-1-gpu-small",
+    stage="stage-b",
+    runner_config="1-gpu-small",
diff -- test/registered/models/test_ministral4_models.py
@@ -8,7 +8,8 @@
-    suite="stage-b-test-2-gpu-large",
```

- 已读文件:
  - tests: `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1; `test/registered/models/test_dummy_grok_models.py` modified +2/-1; `test/registered/models/test_ministral3_models.py` modified +2/-1; `test/registered/models/test_ministral4_models.py` modified +2/-1; `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +2/-1; `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/ci/ci_register.py`, `test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py`, `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25420 - [CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI

- 链接: https://github.com/sgl-project/sglang/pull/25420
- 状态/时间: merged / 2026-05-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 473 个文件，+746/-747，可读 patch 5614 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI」；模型线: GLM-4.6/4.7；类别: 文档/测试/CI；主要 diff: `.github/workflows/pr-test-multimodal-gen.yml`, `test/registered/bench_fn/test_bench_serving_reasoning_stream.py`, `test/registered/function_call/test_kimik2_detector.py`；技术摘要: 覆盖「[CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI」；主要实现面是 `.github/workflows/pr-test-multimodal-gen.yml`, `test/registered/bench_fn/test_bench_serving_reasoning_stream.py`, `test/registered/function_call/test_kimik2_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.github/workflows/pr-test-multimodal-gen.yml` modified +7/-7 (14 lines); hunks: -31,7 +31,7 @@ on:; -42,7 +42,7 @@ env:；`test/registered/bench_fn/test_bench_serving_reasoning_stream.py` modified +1/-1 (2 lines); hunks: -24,7 +24,7; symbols: _free_port，涉及 `_free_port`；`test/registered/function_call/test_kimik2_detector.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7; symbols: _make_tool，涉及 `_make_tool`；`test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1 (2 lines); hunks: -2,7 +2,7。
- 代码 diff 细节:
  - `.github/workflows/pr-test-multimodal-gen.yml` modified +7/-7 (14 lines); hunks: -31,7 +31,7 @@ on:; -42,7 +42,7 @@ env:
  - `test/registered/bench_fn/test_bench_serving_reasoning_stream.py` modified +1/-1 (2 lines); hunks: -24,7 +24,7; symbols: _free_port
  - `test/registered/function_call/test_kimik2_detector.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7; symbols: _make_tool
  - `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1 (2 lines); hunks: -2,7 +2,7
  - `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1 (2 lines); hunks: -18,7 +18,7
- 关键代码摘录:

```diff
diff -- .github/workflows/pr-test-multimodal-gen.yml
@@ -31,7 +31,7 @@ on:
-      skip_stage_health_check:
+      skip_pr_test_health_check:
@@ -42,7 +42,7 @@ env:
-  SKIP_STAGE_HEALTH_CHECK: ${{ inputs.skip_stage_health_check == 'true' }}
+  SKIP_PR_TEST_HEALTH_CHECK: ${{ inputs.skip_pr_test_health_check == 'true' }}
@@ -90,7 +90,7 @@ jobs:
diff -- test/registered/bench_fn/test_bench_serving_reasoning_stream.py
@@ -24,7 +24,7 @@
-register_cpu_ci(est_time=10, suite="stage-a-test-cpu")
+register_cpu_ci(est_time=10, suite="base-a-test-cpu")
diff -- test/registered/function_call/test_kimik2_detector.py
@@ -11,7 +11,7 @@
-register_cpu_ci(5, "stage-a-test-cpu")
+register_cpu_ci(5, "base-a-test-cpu")
diff -- test/registered/layers/mamba/test_causal_conv1d.py
@@ -2,7 +2,7 @@
```

- 已读文件:
  - runtime: `.github/workflows/pr-test-multimodal-gen.yml` modified +7/-7
  - tests: `test/registered/bench_fn/test_bench_serving_reasoning_stream.py` modified +1/-1; `test/registered/function_call/test_kimik2_detector.py` modified +1/-1; `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1; `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1; `test/registered/layers/mamba/test_mamba_ssm.py` modified +1/-1; `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +1/-1
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/deepseek_v4/test_c128_v2.py`, `python/sglang/jit_kernel/tests/deepseek_v4/test_c4_v2.py`, `python/sglang/jit_kernel/tests/test_activation.py`, `python/sglang/jit_kernel/tests/test_add_constant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22822 - [Refactor] Refactor DeepEP dispatcher

- 链接: https://github.com/sgl-project/sglang/pull/22822
- 状态/时间: merged / 2026-05-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 30 个文件，+302/-182，可读 patch 1332 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Refactor DeepEP dispatcher」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`；技术摘要: 覆盖「[Refactor] Refactor DeepEP dispatcher」；主要实现面是 `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +77/-31 (108 lines); hunks: -1,6 +1,7; -22,8 +23,9; symbols: __init__, dispatch_a, _get_buffer, set_quant_config，涉及 `__init__, dispatch_a, _get_buffer`；`python/sglang/srt/layers/moe/utils.py` modified +76/-0 (76 lines); hunks: -9,14 +9,20; -161,6 +167,76 @@ def is_auto(self) -> bool:; symbols: is_auto, DeepEPOutputDtype, get_deepep_output_dtype，涉及 `is_auto, DeepEPOutputDtype, get_deepep_output_dtype`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +6/-21 (27 lines); hunks: -25,12 +25,6; -48,9 +42,6; symbols: __init__, process_weights_after_loading, forward_npu，涉及 `__init__, process_weights_after_loading, forward_npu`；`python/sglang/srt/models/qwen3_5_mtp.py` modified +0/-15 (15 lines); hunks: -15,15 +15,13; -138,17 +136,6 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +77/-31 (108 lines); hunks: -1,6 +1,7; -22,8 +23,9; symbols: __init__, dispatch_a, _get_buffer, set_quant_config
  - `python/sglang/srt/layers/moe/utils.py` modified +76/-0 (76 lines); hunks: -9,14 +9,20; -161,6 +167,76 @@ def is_auto(self) -> bool:; symbols: is_auto, DeepEPOutputDtype, get_deepep_output_dtype
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +6/-21 (27 lines); hunks: -25,12 +25,6; -48,9 +42,6; symbols: __init__, process_weights_after_loading, forward_npu
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +0/-15 (15 lines); hunks: -15,15 +15,13; -138,17 +136,6 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +0/-15 (15 lines); hunks: -15,15 +15,13; -93,17 +91,6 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/token_dispatcher/deepep.py
@@ -1,6 +1,7 @@
+import os
@@ -22,8 +23,9 @@
+    DeepEPOutputDtype,
-    get_moe_runner_backend,
+    get_deepep_output_dtype,
@@ -344,6 +346,8 @@ def __init__(
diff -- python/sglang/srt/layers/moe/utils.py
@@ -9,14 +9,20 @@
+from sglang.srt.environ import envs
+from sglang.srt.utils import is_npu
+_is_npu = is_npu()
+from sglang.srt.server_args import get_global_server_args
@@ -161,6 +167,76 @@ def is_auto(self) -> bool:
+class DeepEPOutputDtype(Enum):
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -25,12 +25,6 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +77/-31; `python/sglang/srt/layers/moe/utils.py` modified +76/-0; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +6/-21; `python/sglang/srt/models/qwen3_5_mtp.py` modified +0/-15; `python/sglang/srt/models/qwen3_next_mtp.py` modified +0/-15; `python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a4_int4_moe.py` modified +11/-3
- 验证与风险: diff 自带测试面 `test/manual/layers/moe/test_moe_runners_4gpu.py`, `test/manual/test_w4a8_deepseek_v3.py`, `test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17869 - [NPU]Support model GLM-4.7-Flash for npu, accuracy 81%

- 链接: https://github.com/sgl-project/sglang/pull/17869
- 状态/时间: closed / 2026-05-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+86/-5，可读 patch 113 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]Support model GLM-4.7-Flash for npu, accuracy 81%」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`；技术摘要: 覆盖「[NPU]Support model GLM-4.7-Flash for npu, accuracy 81%」；主要实现面是 `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py` added +54/-0 (54 lines); hunks: -0,0 +1,54; symbols: TestGLM47Flash，涉及 `TestGLM47Flash`；`python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +30/-4 (34 lines); hunks: -1001,10 +1001,36 @@ def forward_extend(; symbols: forward_extend，涉及 `forward_extend`；`python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +1/-1 (2 lines); hunks: -103,7 +103,7 @@ def forward_mha_prepare_npu(; symbols: forward_mha_prepare_npu，涉及 `forward_mha_prepare_npu`；`python/sglang/test/ascend/test_ascend_utils.py` modified +1/-0 (1 lines); hunks: -110,6 +110,7。
- 代码 diff 细节:
  - `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py` added +54/-0 (54 lines); hunks: -0,0 +1,54; symbols: TestGLM47Flash
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +30/-4 (34 lines); hunks: -1001,10 +1001,36 @@ def forward_extend(; symbols: forward_extend
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +1/-1 (2 lines); hunks: -103,7 +103,7 @@ def forward_mha_prepare_npu(; symbols: forward_mha_prepare_npu
  - `python/sglang/test/ascend/test_ascend_utils.py` modified +1/-0 (1 lines); hunks: -110,6 +110,7
- 关键代码摘录:

```diff
diff -- test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py
@@ -0,0 +1,54 @@
+import os
+import unittest
+from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
+from sglang.test.ascend.test_ascend_utils import GLM_4_7_FLASH_WEIGHTS_PATH
+from sglang.test.ci.ci_register import register_npu_ci
+from sglang.test.test_utils import CustomTestCase
diff -- python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py
@@ -1001,10 +1001,36 @@ def forward_extend(
-            assert (
-                layer.qk_head_dim != layer.v_head_dim
-            ), "FIA only supports qk_head_dim != v_head_dim"
-            if layer.v_head_dim in [256]:
+            if layer.qk_head_dim == layer.v_head_dim:
+                """FIA only supports qk_head_dim != v_head_dim"""
diff -- python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
@@ -103,7 +103,7 @@ def forward_mha_prepare_npu(
```

- 已读文件:
  - tests: `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py` added +54/-0; `python/sglang/test/ascend/test_ascend_utils.py` modified +1/-0
  - runtime: `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +30/-4; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +1/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/ascend/test_ascend_utils.py`, `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25524 - [Bug Fix] Align glm4_moe_nextn NPU MTP loading with qwen3 MTP

- 链接: https://github.com/sgl-project/sglang/pull/25524
- 状态/时间: merged / 2026-05-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-25，可读 patch 79 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug Fix] Align glm4_moe_nextn NPU MTP loading with qwen3 MTP」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe_nextn.py`；技术摘要: 覆盖「[Bug Fix] Align glm4_moe_nextn NPU MTP loading with qwen3 MTP」；主要实现面是 `python/sglang/srt/models/glm4_moe_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_nextn.py` modified +11/-25 (36 lines); hunks: -12,9 +12,8; -23,7 +22,6; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +11/-25 (36 lines); hunks: -12,9 +12,8; -23,7 +22,6; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_nextn.py
@@ -12,9 +12,8 @@
-"""Inference-only GLM-4.5, GLM-4.6 Speculative Decoding."""
+"""Inference-only GLM-4.5, GLM-4.6 and GLM-4.7 Speculative Decoding."""
-import contextlib
@@ -23,7 +22,6 @@
-from sglang.srt.environ import temp_set_env
@@ -36,7 +34,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_nextn.py` modified +11/-25
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25825 - [Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool

- 链接: https://github.com/sgl-project/sglang/pull/25825
- 状态/时间: merged / 2026-05-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+59/-8，可读 patch 326 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool」；模型线: GLM-4.6/4.7；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/qwen2.py`；技术摘要: 覆盖「[Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool」；主要实现面是 `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/qwen2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama.py` modified +16/-2 (18 lines); hunks: -27,6 +27,7; -131,6 +132,7 @@ def __init__(; symbols: __init__, forward_prepare_native, forward_prepare_npu，涉及 `__init__, forward_prepare_native, forward_prepare_npu`；`python/sglang/srt/models/glm4_moe.py` modified +12/-1 (13 lines); hunks: -28,6 +28,7; -187,6 +188,7 @@ def __init__(; symbols: __init__, forward_prepare，涉及 `__init__, forward_prepare`；`python/sglang/srt/models/qwen2.py` modified +9/-0 (9 lines); hunks: -24,6 +24,7; -200,12 +201,14 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunks: -32,6 +32,7; -600,13 +601,15 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama.py` modified +16/-2 (18 lines); hunks: -27,6 +27,7; -131,6 +132,7 @@ def __init__(; symbols: __init__, forward_prepare_native, forward_prepare_npu
  - `python/sglang/srt/models/glm4_moe.py` modified +12/-1 (13 lines); hunks: -28,6 +28,7; -187,6 +188,7 @@ def __init__(; symbols: __init__, forward_prepare
  - `python/sglang/srt/models/qwen2.py` modified +9/-0 (9 lines); hunks: -24,6 +24,7; -200,12 +201,14 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunks: -32,6 +32,7; -600,13 +601,15 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3.py` modified +5/-1 (6 lines); hunks: -64,6 +64,7 @@ def __init__(; -76,6 +77,7 @@ def __init__(; symbols: __init__, forward_prepare_native, forward_prepare_npu
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama.py
@@ -27,6 +27,7 @@
+    get_pp_indices,
@@ -131,6 +132,7 @@ def __init__(
+        start_layer: int = 0,
@@ -141,6 +143,7 @@ def __init__(
+        self.start_layer = start_layer
@@ -210,7 +213,7 @@ def forward_prepare_native(self, positions, hidden_states):
diff -- python/sglang/srt/models/glm4_moe.py
@@ -28,6 +28,7 @@
+    get_pp_indices,
@@ -187,6 +188,7 @@ def __init__(
+        start_layer: int = 0,
@@ -201,6 +203,7 @@ def __init__(
+        self.start_layer = start_layer
@@ -312,7 +315,7 @@ def forward_prepare(
diff -- python/sglang/srt/models/qwen2.py
@@ -24,6 +24,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama.py` modified +16/-2; `python/sglang/srt/models/glm4_moe.py` modified +12/-1; `python/sglang/srt/models/qwen2.py` modified +9/-0; `python/sglang/srt/models/qwen2_moe.py` modified +9/-0; `python/sglang/srt/models/qwen3.py` modified +5/-1; `python/sglang/srt/models/qwen3_moe.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/llama_eagle.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25821 - [Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename

- 链接: https://github.com/sgl-project/sglang/pull/25821
- 状态/时间: merged / 2026-05-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 162 个文件，+11303/-10745，可读 patch 15980 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename」；模型线: GLM-4.6/4.7；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`；技术摘要: 覆盖「[Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename」；主要实现面是 `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587 (2595 lines)；`python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0 (2589 lines)；`python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518 (2539 lines)；`python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0 (2528 lines)。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587 (2595 lines)
  - `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0 (2589 lines)
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518 (2539 lines)
  - `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0 (2528 lines)
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +8/-1744 (1752 lines); hunks: -1,1746 +1,10; symbols: BaseIndexerMetadata, get_seqlens_int32, get_page_table_64, get_page_table_1
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -1,1746 +1,10 @@
-from __future__ import annotations
+# [Deprecated] Re-export shim for backward compatibility. Use dsa.dsa_indexer instead.
+import warnings
-import contextlib
-import logging
-from abc import ABC, abstractmethod
diff -- python/sglang/srt/layers/attention/dsa/dsa_indexer.py
@@ -0,0 +1,1746 @@
+from __future__ import annotations
+import contextlib
+import logging
+from abc import ABC, abstractmethod
+from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
+import torch
diff -- python/sglang/srt/layers/attention/nsa/index_buf_accessor.py
@@ -1,814 +1,10 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587; `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0; `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518; `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +8/-1744; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` added +1746/-0
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py`, `python/sglang/jit_kernel/tests/test_fused_store_index_cache.py`, `python/sglang/jit_kernel/tests/test_set_mla_kv_buffer.py`, `python/sglang/test/nightly_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22315 - [Bugfix] Fix GLM-4.7-FP8 EAGLE accept_len=1.00 due to draft model loading with incorrect quant_config

- 链接: https://github.com/sgl-project/sglang/pull/22315
- 状态/时间: closed / 2026-05-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-5，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix GLM-4.7-FP8 EAGLE accept_len=1.00 due to draft model loading with incorrect quant_config」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe_nextn.py`；技术摘要: 覆盖「[Bugfix] Fix GLM-4.7-FP8 EAGLE accept_len=1.00 due to draft model loading with incorrect quant_config」；主要实现面是 `python/sglang/srt/models/glm4_moe_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_nextn.py` modified +7/-5 (12 lines); hunks: -36,7 +36,7; -128,10 +128,12 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +7/-5 (12 lines); hunks: -36,7 +36,7; -128,10 +128,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_nextn.py
@@ -36,7 +36,7 @@
-from sglang.srt.utils import add_prefix
+from sglang.srt.utils import add_prefix, is_npu
@@ -128,10 +128,12 @@ def __init__(
-        self.needs_quant_draft = (
-            get_global_server_args().speculative_draft_model_quantization
-        )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_nextn.py` modified +7/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26088 - GLM-4.7-Flash: standalone MLA impl and MLA NextN/MTP

- 链接: https://github.com/sgl-project/sglang/pull/26088
- 状态/时间: merged / 2026-05-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/glm4_moe_lite_nextn.py`；关联提交 `7ef06bfc06ec`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+799/-86，可读 patch 1076 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「GLM-4.7-Flash: standalone MLA impl and MLA NextN/MTP」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/glm4_moe_lite_nextn.py`；技术摘要: 覆盖「GLM-4.7-Flash: standalone MLA impl and MLA NextN/MTP」；主要实现面是 `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/glm4_moe_lite_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_lite.py` modified +603/-81 (684 lines); hunks: -1,4 +1,4; -12,77 +12,81; symbols: forward, __init__, Glm4MoeLiteSparseMoeBlock，涉及 `forward, __init__, Glm4MoeLiteSparseMoeBlock`；`python/sglang/srt/models/glm4_moe_lite_nextn.py` added +182/-0 (182 lines); hunks: -0,0 +1,182; symbols: Glm4MoeLiteModelNextN, __init__, forward, Glm4MoeLiteForCausalLMNextN，涉及 `Glm4MoeLiteModelNextN, __init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +603/-81 (684 lines); hunks: -1,4 +1,4; -12,77 +12,81; symbols: forward, __init__, Glm4MoeLiteSparseMoeBlock
  - `python/sglang/srt/models/glm4_moe_lite_nextn.py` added +182/-0 (182 lines); hunks: -0,0 +1,182; symbols: Glm4MoeLiteModelNextN, __init__, forward, Glm4MoeLiteForCausalLMNextN
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -1,4 +1,4 @@
-# Copyright 2025-2026 SGLang Team
+# Copyright 2026-2027 SGLang Team
@@ -12,77 +12,81 @@
-"""Inference-only GLM-4.7-Flash model compatible with HuggingFace weights"""
+"""Inference-only GLM-4.7-Flash model compatible with HuggingFace weights."""
-from typing import Iterable, Optional, Tuple
diff -- python/sglang/srt/models/glm4_moe_lite_nextn.py
@@ -0,0 +1,182 @@
+# Copyright 2026-2027 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_lite.py` modified +603/-81; `python/sglang/srt/models/glm4_moe_lite_nextn.py` added +182/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26673 - [refactor] remove unused op_mlp

- 链接: https://github.com/sgl-project/sglang/pull/26673
- 状态/时间: merged / 2026-05-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+0/-53，可读 patch 95 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[refactor] remove unused op_mlp」；模型线: GLM-4.6/4.7；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`；技术摘要: 覆盖「[refactor] remove unused op_mlp」；主要实现面是 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +0/-13 (13 lines); hunks: -2114,19 +2114,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer，涉及 `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`；`python/sglang/srt/models/glm4_moe.py` modified +0/-13 (13 lines); hunks: -1017,19 +1017,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer，涉及 `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`；`python/sglang/srt/models/glm4_moe_lite.py` modified +0/-13 (13 lines); hunks: -737,19 +737,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer，涉及 `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`；`python/sglang/srt/models/minimax_m2.py` modified +0/-6 (6 lines); hunks: -1069,12 +1069,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer，涉及 `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +0/-13 (13 lines); hunks: -2114,19 +2114,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/glm4_moe.py` modified +0/-13 (13 lines); hunks: -1017,19 +1017,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +0/-13 (13 lines); hunks: -737,19 +737,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/minimax_m2.py` modified +0/-6 (6 lines); hunks: -1069,12 +1069,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/mimo_v2.py` modified +0/-4 (4 lines); hunks: -808,10 +808,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2114,19 +2114,6 @@ def op_comm_prepare_mlp(self, state):
-    def op_mlp(self, state):
-        hidden_states = state.pop("hidden_states_mlp_input")
-        if not (
-            enable_moe_dense_fully_dp()
-            and (not self.is_layer_sparse)
-            and hidden_states.shape[0] == 0
diff -- python/sglang/srt/models/glm4_moe.py
@@ -1017,19 +1017,6 @@ def op_comm_prepare_mlp(self, state):
-    def op_mlp(self, state):
-        hidden_states = state.pop("hidden_states_mlp_input")
-        if not (
-            enable_moe_dense_fully_dp()
-            and (not self.is_layer_sparse)
-            and hidden_states.shape[0] == 0
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -737,19 +737,6 @@ def op_comm_prepare_mlp(self, state):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +0/-13; `python/sglang/srt/models/glm4_moe.py` modified +0/-13; `python/sglang/srt/models/glm4_moe_lite.py` modified +0/-13; `python/sglang/srt/models/minimax_m2.py` modified +0/-6; `python/sglang/srt/models/mimo_v2.py` modified +0/-4; `python/sglang/srt/models/qwen3_moe.py` modified +0/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25813 - docs(cookbook): port popular model usage guides into cookbook pages

- 链接: https://github.com/sgl-project/sglang/pull/25813
- 状态/时间: merged / 2026-06-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 47 个文件，+1262/-2154，可读 patch 4187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): port popular model usage guides into cookbook pages」；模型线: GLM-4.6/4.7；类别: 文档/测试/CI；主要 diff: `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`；技术摘要: 覆盖「docs(cookbook): port popular model usage guides into cookbook pages」；主要实现面是 `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0；`docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...；`docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64，涉及 `image_to_base64`。
- 代码 diff 细节:
  - `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0
  - `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...
  - `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64
  - `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181 (181 lines); hunks: -1,181 +0,0
- 关键代码摘录:

```diff
diff -- docs_new/docs/basic_usage/deepseek_v32.mdx
@@ -1,601 +0,0 @@
-title: "DeepSeek V3.2/GLM-5 Usage"
-metatags:
-    description: "Deploy DeepSeek V3.2/GLM-5 with SGLang: DeepSeek Sparse Attention (DSA), long-context optimization, MTP speculative decoding, function calling. Supports H200, B2
-DeepSeek-V3.2 model family equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attention mechanism power
-Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://hu
-## Installation
diff -- docs_new/docs/basic_usage/deepseek_v3.mdx
@@ -1,375 +0,0 @@
-title: "DeepSeek V3/V3.1/R1 Usage"
-metatags:
-    description: "Deploy DeepSeek V3/R1 with SGLang: MLA optimization, FP8 quantization, multi-node TP, DP attention, MTP speculative decoding. Supports H200, B200, MI300X, A100."
-SGLang provides many optimizations specifically designed for the DeepSeek models, making it the inference engine recommended by the official [DeepSeek team](https://github.com/dee
-This document outlines current optimizations for DeepSeek.
-For an overview of the implemented features see the completed [Roadmap](https://github.com/sgl-project/sglang/issues/2591).
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx
@@ -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose the most suitable in
```

- 已读文件:
  - docs: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601; `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3; `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26; `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181; `docs_new/docs/basic_usage/glmv.mdx` removed +0/-139
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #26384 - [Docs] GLM-4.7 cookbook: add NVIDIA Blackwell (B200, GB200) + NVFP4 sections

- 链接: https://github.com/sgl-project/sglang/pull/26384
- 状态/时间: merged / 2026-06-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`；关联提交 `1c0019da7579`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+529/-45，可读 patch 746 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] GLM-4.7 cookbook: add NVIDIA Blackwell (B200, GB200) + NVFP4 sections」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`；技术摘要: 覆盖「[Docs] GLM-4.7 cookbook: add NVIDIA Blackwell (B200, GB200) + NVFP4 sections」；主要实现面是 `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +404/-14 (418 lines); hunks: -1,14 +1,14; -21,14 +21,15 @@ For more details, please refer to the [official GLM-4.7 docu...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +404/-14 (418 lines); hunks: -1,14 +1,14; -21,14 +21,15 @@ For more details, please refer to the [official GLM-4.7 docu...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx
@@ -1,14 +1,14 @@
-    description: "Deploy GLM-4.7 with SGLang on AMD GPUs - state-of-the-art reasoning, enhanced coding, and robust tool calling capabilities."
+    description: "Deploy GLM-4.7 with SGLang on NVIDIA Blackwell (B200, GB200) and AMD GPUs - state-of-the-art reasoning, robust tool calling, and NVFP4 weights for Blackwell."
-[GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) is the latest and most powerful language model in the GLM series developed by Zhipu AI, featuring state-of-the-art capabilities i
+[GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) is a powerful language model developed by Zhipu AI, featuring advanced capabilities in reasoning, function calling, and agent wor
-As the newest iteration in the GLM series, GLM-4.7 achieves significant improvements across all domains:
+GLM-4.7 brings improvements across all major domains:
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +404/-14
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/src/snippets/autoregressive/glm-47-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27001 - [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests

- 链接: https://github.com/sgl-project/sglang/pull/27001
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+11/-471，可读 patch 936 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`；技术摘要: 覆盖「[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests」；主要实现面是 `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45 (46 lines); hunks: -2,19 +2,10; -60,26 +51,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x, setUpClass，涉及 `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x`；`test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x, setUpClass，涉及 `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x`；`test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass，涉及 `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x`；`test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -41,21 +36,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models，涉及 `get_model_path, ModelConfig, get_display_name`。
- 代码 diff 细节:
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45 (46 lines); hunks: -2,19 +2,10; -60,26 +51,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x, setUpClass
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x, setUpClass
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -41,21 +36,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -39,21 +34,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models
- 关键代码摘录:

```diff
diff -- test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py
@@ -2,19 +2,10 @@
-The model path can be configured via DEEPSEEK_R1_MXFP4_MODEL_PATH environment variable.
-Example usage:
-    DEEPSEEK_R1_MXFP4_MODEL_PATH=/data2/models/amd-DeepSeek-R1-MXFP4-Preview python -m pytest test_deepseek_r1_mxfp4_perf_mi35x.py -v
-# Set HF cache to /data2/models/ for MI35x so HF models download there
-os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
-os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")
diff -- test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py
@@ -3,19 +3,10 @@
-The model path can be configured via DEEPSEEK_R1_MXFP4_MODEL_PATH environment variable.
-Example usage:
-    DEEPSEEK_R1_MXFP4_MODEL_PATH=/data2/models/amd-DeepSeek-R1-MXFP4-Preview python -m pytest test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py -v
-# Set HF cache to /data2/models/ for MI35x so HF models download there
-os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
-os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")
diff -- test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py
@@ -3,19 +3,10 @@
```

- 已读文件:
  - tests: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +1/-35; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py` modified +1/-35
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23906 - [Refactor] Cuda Graph Runner/Backend Refactor

- 链接: https://github.com/sgl-project/sglang/pull/23906
- 状态/时间: merged / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 160 个文件，+5197/-3068，可读 patch 12233 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Cuda Graph Runner/Backend Refactor」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`；技术摘要: 覆盖「[Refactor] Cuda Graph Runner/Backend Refactor」；主要实现面是 `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool，涉及 `freeze_gc, _to_torch, patch_model`；`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype，涉及 `PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled`；`python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode，涉及 `_make_graph_key, build_replay_fb_view, _allocate_decode_buffers`；`python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers，涉及 `BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool
  - `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype
  - `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode
  - `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers
  - `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0 (442 lines); hunks: -0,0 +1,442; symbols: _grouped_foreach_copy_, foreach_copy, DecodeInputBuffers, create
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -1,860 +0,0 @@
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
-# You may obtain a copy of the License at
-#
-#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py
@@ -0,0 +1,846 @@
+# Copyright 2023-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
@@ -1,4 +1,4 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860; `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0; `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463; `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541; `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0; `python/sglang/srt/model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py` added +225/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/doc_patch.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27964 - [Spec] Retire Spec V1

- 链接: https://github.com/sgl-project/sglang/pull/27964
- 状态/时间: merged / 2026-06-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 46 个文件，+111/-252，可读 patch 1422 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Spec] Retire Spec V1」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`；技术摘要: 覆盖「[Spec] Retire Spec V1」；主要实现面是 `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass，涉及 `TestDeepseekMTP, setUpClass, tearDownClass`；`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do；`python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family，涉及 `handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp`；`docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu...。
- 代码 diff 细节:
  - `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do
  - `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family
  - `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu...
  - `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10 (21 lines); hunks: -1,6 +1,5; -31,7 +30,8 @@ class TestEagleConstrainedDecoding(; symbols: TestEagleConstrainedDecoding, setUpClass, tearDownClass, TestEagleConstrainedDecodingV2
- 关键代码摘录:

```diff
diff -- test/registered/ep/test_deepep_large.py
@@ -3,7 +3,6 @@
-from sglang.srt.environ import envs
@@ -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):
-        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
-            cls.process = popen_launch_server(
-                cls.model,
-                cls.base_url,
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx
@@ -1108,7 +1108,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1227,7 +1226,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1351,7 +1349,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1476,7 +1473,6 @@ do
diff -- python/sglang/srt/arg_groups/speculative_hook.py
@@ -1,9 +1,8 @@
```

- 已读文件:
  - tests: `test/registered/ep/test_deepep_large.py` modified +43/-44; `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10; `python/sglang/test/server_fixtures/standalone_fixture.py` modified +7/-8; `python/sglang/test/server_fixtures/spec_eagle_fixture.py` modified +6/-6
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64; `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_optimization.mdx` modified +3/-8
  - runtime: `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26
- 验证与风险: diff 自带测试面 `python/sglang/test/server_fixtures/spec_eagle_fixture.py`, `python/sglang/test/server_fixtures/standalone_fixture.py`, `test/manual/dsv4/test_dsv4_flash_mtp_tp8.py`, `test/manual/dsv4/test_dsv4_pro_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18383 - [Bug Fix] Add missing use_mla guard in aiter_backend draft_extend CUD…

- 链接: https://github.com/sgl-project/sglang/pull/18383
- 状态/时间: closed / 2026-06-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug Fix] Add missing use_mla guard in aiter_backend draft_extend CUD…」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/aiter_backend.py`；技术摘要: 覆盖「[Bug Fix] Add missing use_mla guard in aiter_backend draft_extend CUD…」；主要实现面是 `python/sglang/srt/layers/attention/aiter_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/aiter_backend.py` modified +1/-1 (2 lines); hunks: -865,7 +865,7 @@ def init_forward_metadata_capture_cuda_graph(; symbols: init_forward_metadata_capture_cuda_graph，涉及 `init_forward_metadata_capture_cuda_graph`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +1/-1 (2 lines); hunks: -865,7 +865,7 @@ def init_forward_metadata_capture_cuda_graph(; symbols: init_forward_metadata_capture_cuda_graph
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -865,7 +865,7 @@ def init_forward_metadata_capture_cuda_graph(
-            if _use_mla_ps_kernel:
+            if self.use_mla and _use_mla_ps_kernel:
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/aiter_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28149 - Support GLM-4.7 function calling via structural tags

- 链接: https://github.com/sgl-project/sglang/pull/28149
- 状态/时间: merged / 2026-06-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/glm47_moe_detector.py`；关联提交 `f79a6b5c3355`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+104/-6，可读 patch 137 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support GLM-4.7 function calling via structural tags」；模型线: GLM-4.6/4.7；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/glm47_moe_detector.py`；技术摘要: 覆盖「Support GLM-4.7 function calling via structural tags」；主要实现面是 `python/sglang/srt/function_call/glm47_moe_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/glm47_moe_detector.py` modified +41/-6 (47 lines); hunks: -3,10 +3,15; -17,6 +22,21; symbols: _glm47_native_structural_tag_available, StreamState, _parse_argument_pairs, supports_structural_tag，涉及 `_glm47_native_structural_tag_available, StreamState, _parse_argument_pairs`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +41/-6 (47 lines); hunks: -3,10 +3,15; -17,6 +22,21; symbols: _glm47_native_structural_tag_available, StreamState, _parse_argument_pairs, supports_structural_tag
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/glm47_moe_detector.py
@@ -3,10 +3,15 @@
-from typing import Any, Dict, List, Optional, Tuple
-from sglang.srt.entrypoints.openai.protocol import Tool
-from sglang.srt.function_call.base_format_detector import BaseFormatDetector
+from functools import lru_cache
+from typing import Any, Dict, List, Literal, Optional, Tuple, Union
+from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/glm47_moe_detector.py` modified +41/-6
- 验证与风险: diff 自带测试面 `test/registered/unit/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- 链接: https://github.com/sgl-project/sglang/pull/28567
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 184 个文件，+1865/-1727，可读 patch 8932 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add get_parallel(): a structured accessor for parallel-topology state」；模型线: GLM-4.6/4.7；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Add get_parallel(): a structured accessor for parallel-topology state」；主要实现面是 `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention，涉及 `ApertusMLP, __init__, forward`；`python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales，涉及 `__init__, forward, load_kv_cache_scales`；`python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__，涉及 `_resolve_moe_input_pad_multiple, __init__`；`python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention
  - `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales
  - `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__
  - `python/sglang/srt/layers/communicator.py` modified +13/-19 (32 lines); hunks: -23,8 +23,6; -44,12 +42,7; symbols: apply_aiter_all_reduce_fusion, init_context, should_fuse_mlp_allreduce_with_next_layer, is_same_group_size
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/apertus.py
@@ -1,687 +1,686 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Copyright 2025 The SwissAI Initiative
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
diff -- python/sglang/srt/models/solar.py
@@ -1,37 +1,14 @@
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
-# Copyright 2023 The vLLM team.
-# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
-#
-# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,21 +28,13 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/apertus.py` modified +686/-687; `python/sglang/srt/models/solar.py` modified +28/-27; `python/sglang/srt/models/gpt_oss.py` modified +17/-24; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23; `python/sglang/srt/layers/communicator.py` modified +13/-19; `python/sglang/srt/models/qwen3_moe.py` modified +12/-18
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dual_chunk_attention.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28516 - [NPU] Add MTP support for GLM-4.7-Flash

- 链接: https://github.com/sgl-project/sglang/pull/28516
- 状态/时间: merged / 2026-06-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe_lite.py`；关联提交 `2a9cce5d2757`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+38/-2，可读 patch 74 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Add MTP support for GLM-4.7-Flash」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe_lite.py`；技术摘要: 覆盖「[NPU] Add MTP support for GLM-4.7-Flash」；主要实现面是 `python/sglang/srt/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_lite.py` modified +2/-0 (2 lines); hunks: -548,6 +548,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +2/-0 (2 lines); hunks: -548,6 +548,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -548,6 +548,8 @@ def __init__(
+        # Required for MTP: Glm4MoeLiteModelNextN bypasses Glm4MoeLiteForCausalLM.__init__
+        config.moe_layer_freq = 1
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_lite.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29261 - [Docs] Fix broken links in cookbook

- 链接: https://github.com/sgl-project/sglang/pull/29261
- 状态/时间: merged / 2026-06-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+3/-3，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Fix broken links in cookbook」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；技术摘要: 覆盖「[Docs] Fix broken links in cookbook」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -89,7 +89,7 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai/...；`docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +1/-1 (2 lines); hunks: -127,7 +127,7 @@ Pick a weight format by hardware: **NVFP4** on NVIDIA Blackw...；`docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -89,7 +89,7 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai/...
  - `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +1/-1 (2 lines); hunks: -127,7 +127,7 @@ Pick a weight format by hardware: **NVFP4** on NVIDIA Blackw...
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -89,7 +89,7 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai/deepseek-v4-benchm
-  For a runnable end-to-end example, see the [DeepSeek-V4-Flash demo notebook](/demo/deepseek_v4_flash).
+  For a runnable end-to-end example, see the [DeepSeek-V4-Flash demo notebook](https://github.com/sgl-project/sglang/blob/main/docs_new/demo/deepseek_v4_flash.ipynb).
diff -- docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx
@@ -127,7 +127,7 @@ Pick a weight format by hardware: **NVFP4** on NVIDIA Blackwell (B200, GB200), *
-For general GLM-4.x family launch guidance (AMD ROCm notes and more), see [Launch GLM-4.5 / GLM-4.6 / GLM-4.7 with SGLang](../../../docs/basic_usage/glm45). Per-hardware bench com
+For general GLM-4.x family launch guidance (AMD ROCm notes and more), see [Launch GLM-4.5 / GLM-4.6 / GLM-4.7 with SGLang](/cookbook/autoregressive/GLM/GLM-4.5). Per-hardware benc
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx
@@ -52,7 +52,7 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=pytho
-For the full Docker setup and other installation methods, refer to the [official SGLang installation guide](../../../docs/get-started/installation).
+For the full Docker setup and other installation methods, refer to the [official SGLang installation guide](../../../docs/get-started/install).
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
