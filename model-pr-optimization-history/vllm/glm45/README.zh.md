# vllm GLM-4.5 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/reasoning/test_glm4_moe_reasoning_parser.py` | 无直接 PR 号提交 |
| `tests/tool_parsers/test_glm4_moe_tool_parser.py` | [#39601](https://github.com/vllm-project/vllm/pull/39601) |
| `vllm/model_executor/models/glm4_moe.py` | [#21435](https://github.com/vllm-project/vllm/pull/21435), [#22143](https://github.com/vllm-project/vllm/pull/22143), [#22203](https://github.com/vllm-project/vllm/pull/22203), [#22460](https://github.com/vllm-project/vllm/pull/22460), [#22520](https://github.com/vllm-project/vllm/pull/22520), [#22832](https://github.com/vllm-project/vllm/pull/22832), [#24849](https://github.com/vllm-project/vllm/pull/24849), [#25830](https://github.com/vllm-project/vllm/pull/25830), [#41755](https://github.com/vllm-project/vllm/pull/41755) |
| `vllm/model_executor/models/glm4_moe_lite.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/glm4_moe_lite_mtp.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/glm4_moe_mtp.py` | [#27597](https://github.com/vllm-project/vllm/pull/27597), [#28805](https://github.com/vllm-project/vllm/pull/28805) |

## PR 覆盖总览

- git 追溯 PR 数: 12
- 原文档显式引用补充 PR 数: 43
- 当前文档总 PR 数: 55
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
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
| 2025-11-19 | [#28542](https://github.com/vllm-project/vllm/pull/28542) | merged | Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5 | `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/deepseek_v2.py` |
| 2025-11-26 | [#29342](https://github.com/vllm-project/vllm/pull/29342) | merged | [Attention] Remove imports from `vllm/attention/__init__.py` | `vllm/model_executor/models/whisper.py`, `vllm/model_executor/model_loader/utils.py`, `vllm/model_executor/models/afmoe.py` |
| 2025-12-04 | [#29966](https://github.com/vllm-project/vllm/pull/29966) | merged | Access `partial_rotary_factor` from `rope_parameters` | `vllm/transformers_utils/configs/nemotron.py`, `vllm/transformers_utils/configs/qwen3_next.py`, `vllm/model_executor/models/gpt_neox.py` |
| 2025-12-11 | [#30389](https://github.com/vllm-project/vllm/pull/30389) | merged | Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim` | `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/model_executor/models/config.py`, `vllm/model_executor/models/phi.py` |
| 2025-12-15 | [#30675](https://github.com/vllm-project/vllm/pull/30675) | merged | [Refactor] [2/N] Move tool parsers into the vLLM main directory | `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/tool_parsers/__init__.py`, `tests/tool_use/test_qwen3coder_tool_parser.py` |
| 2025-12-15 | [#30693](https://github.com/vllm-project/vllm/pull/30693) | merged | [Refactor] [3/N] Move tool parser tests and run on CPU | `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_jamba_tool_parser.py`, `tests/tool_parsers/test_kimi_k2_tool_parser.py` |
| 2025-12-18 | [#30920](https://github.com/vllm-project/vllm/pull/30920) | merged | [Bugfix] Fix Unicode issues in GLM-4 tool calling | `vllm/tool_parsers/glm4_moe_tool_parser.py` |
| 2025-12-20 | [#30876](https://github.com/vllm-project/vllm/pull/30876) | merged | GLM-4.7 Tool Parser and Doc Update | `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/__init__.py`, `vllm/model_executor/models/glm4_moe.py` |
| 2026-01-05 | [#31622](https://github.com/vllm-project/vllm/pull/31622) | merged | Fix GLM-4.6v flash tool calling in transformers 5.x | `vllm/tool_parsers/glm4_moe_tool_parser.py`, `examples/tool_chat_template_glm4.jinja` |
| 2026-01-06 | [#31055](https://github.com/vllm-project/vllm/pull/31055) | merged | [Bugfix] Fix GLM-4 MoE router logits dtype for data parallel chunking | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/models/glm4_moe.py` |
| 2026-01-07 | [#31104](https://github.com/vllm-project/vllm/pull/31104) | merged | [BugFix] LoRA: Support loading base_layer of experts | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/llama4.py` |
| 2026-01-07 | [#31869](https://github.com/vllm-project/vllm/pull/31869) | merged | [Model] Cleanup: Remove redundant manual definition of `make_empty_intermediate_tensors` in GLM-4-MoE | `vllm/model_executor/models/glm4_moe.py` |
| 2026-01-07 | [#31757](https://github.com/vllm-project/vllm/pull/31757) | merged | [Bugfix][MTP] Fix GLM4 MoE fp8 loading with MTP on | `vllm/model_executor/models/glm4_moe_mtp.py` |
| 2026-01-11 | [#32101](https://github.com/vllm-project/vllm/pull/32101) | merged | [MTP][GLM][Bugfix] Fixed .weight_scale loading logic that dropped MTP prediction accuracy with fp8+mtp | `vllm/model_executor/models/glm4_moe_mtp.py` |
| 2026-01-12 | [#32150](https://github.com/vllm-project/vllm/pull/32150) | merged | [Model] Remove incorrect `SupportsPP` from MTP models | `vllm/model_executor/models/qwen3_next_mtp.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/ernie_mtp.py` |
| 2026-01-13 | [#32240](https://github.com/vllm-project/vllm/pull/32240) | merged | [Refactor] [6/N] to simplify the vLLM openai chat_completion serving architecture | `vllm/entrypoints/openai/engine/protocol.py`, `vllm/entrypoints/openai/chat_completion/protocol.py`, `vllm/entrypoints/serve/tokenize/protocol.py` |
| 2026-01-15 | [#32321](https://github.com/vllm-project/vllm/pull/32321) | merged | fix: avoid crash on zero-arg tool calls in glm4 parser | `vllm/tool_parsers/glm4_moe_tool_parser.py` |
| 2026-01-19 | [#31386](https://github.com/vllm-project/vllm/pull/31386) | merged | [GLM-4.7] GLM Model support for GLM-Lite | `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `tests/models/registry.py` |
| 2026-01-26 | [#33063](https://github.com/vllm-project/vllm/pull/33063) | merged | [Chore] Update type annotation of `input_ids` in model forward | `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py` |
| 2026-01-27 | [#32064](https://github.com/vllm-project/vllm/pull/32064) | merged | [5/N][Attention] Finish eliminating `vllm/attention` folder | `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/__init__.py` |
| 2026-02-02 | [#33218](https://github.com/vllm-project/vllm/pull/33218) | merged | [Bugfix] GLM-4 tool parser: incremental string streaming | `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py` |
| 2026-02-02 | [#33525](https://github.com/vllm-project/vllm/pull/33525) | merged | Update get_expert_mapping to include self parameter | `vllm/model_executor/models/glm4_moe_lite.py` |
| 2026-02-24 | [#34905](https://github.com/vllm-project/vllm/pull/34905) | merged | Fix GLM4 parser tests | `tests/tool_parsers/test_glm4_moe_tool_parser.py` |
| 2026-03-04 | [#35640](https://github.com/vllm-project/vllm/pull/35640) | merged | [MISC] fixed tool_parser mypy errors | `vllm/tool_parsers/glm4_moe_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`, `vllm/tool_parsers/functiongemma_tool_parser.py` |
| 2026-03-16 | [#35208](https://github.com/vllm-project/vllm/pull/35208) | merged | GLM4 tool parser: fix streaming mode | `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py` |
| 2026-03-18 | [#37386](https://github.com/vllm-project/vllm/pull/37386) | merged | fix(glm47): improve tool call parsing and content normalization | `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py` |
| 2026-03-26 | [#38029](https://github.com/vllm-project/vllm/pull/38029) | merged | [Tool Parser][1/3] Pass tools to ToolParser constructor | `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py` |
| 2026-03-31 | [#38264](https://github.com/vllm-project/vllm/pull/38264) | merged | [Mypy] Fix adjust_request typing | `vllm/tool_parsers/deepseekv32_tool_parser.py`, `vllm/tool_parsers/functiongemma_tool_parser.py`, `vllm/tool_parsers/gigachat3_tool_parser.py` |
| 2026-03-31 | [#38189](https://github.com/vllm-project/vllm/pull/38189) | merged | [Tool Parser][2/3] Use self.tools instead of request.tools in tool parsers | `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_qwen3coder_tool_parser.py` |
| 2026-04-01 | [#38172](https://github.com/vllm-project/vllm/pull/38172) | merged | [Misc] Add 20 regression tests for 11 tool parser bug fixes | `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `tests/tool_parsers/test_step3p5_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py` |
| 2026-04-13 | [#39253](https://github.com/vllm-project/vllm/pull/39253) | merged | [Bugfix] Fix GLM tool parser streaming with MTP or stream interval | `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py` |
| 2026-04-17 | [#39870](https://github.com/vllm-project/vllm/pull/39870) | merged | [BugFix] Support custom tool parsers when tool_choice is `required` and named function | `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py` |
| 2026-04-20 | [#35949](https://github.com/vllm-project/vllm/pull/35949) | merged | [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase | `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py` |
| 2026-04-21 | [#35782](https://github.com/vllm-project/vllm/pull/35782) | merged | [MoE Refactor] Remove SharedFusedMoE class | `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py` |
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | merged | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py` |
| 2026-05-07 | [#41755](https://github.com/vllm-project/vllm/pull/41755) | merged | [Bugfix] Fix GLM4-MoE weight loading for NVFP4 quantized checkpoints | `vllm/model_executor/models/glm4_moe.py` |
| 2026-05-09 | [#42026](https://github.com/vllm-project/vllm/pull/42026) | merged | [Bugfix] Preserve leading/trailing whitespace in GLM non-streaming tool parser | `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py` |
| 2026-05-21 | [#39601](https://github.com/vllm-project/vllm/pull/39601) | merged | [Bugfix] Fix glm4_moe_tool_parser._is_string_type for /v1/responses FunctionTool format | `tests/tool_parsers/test_glm4_moe_tool_parser.py` |
| 2026-06-03 | [#44346](https://github.com/vllm-project/vllm/pull/44346) | merged | [Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers | `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py` |
| 2026-06-08 | [#41184](https://github.com/vllm-project/vllm/pull/41184) | merged | [MoE Refactor] FusedMoE/MoERunner inversion refactor | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` |
| 2026-06-18 | [#45915](https://github.com/vllm-project/vllm/pull/45915) | merged | [Frontend] Add Streaming Parser Engine and new GLM4.7/GLM5.1/GLM5.2 Parser | `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/reasoning/test_glm4_moe_reasoning_parser.py` |
| 2026-06-25 | [#46651](https://github.com/vllm-project/vllm/pull/46651) | merged | [Perf] Remove redundant clone for GLM, Deepseek etc | `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py` |

## 逐 PR diff 审计卡

### PR #21435 - remove GLM-4 quantization wrong Code

- 链接: https://github.com/vllm-project/vllm/pull/21435
- 状态/时间: merged / 2025-07-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `85bda9e7d053`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+2/-3，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「remove GLM-4 quantization wrong Code」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「remove GLM-4 quantization wrong Code」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +0/-1 (1 lines); hunks: -390,7 +390,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +0/-1 (1 lines); hunks: -390,7 +390,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -390,7 +390,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-                quant_config=quant_config,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/tool_parsers/glm4_moe_tool_parser.py`, `vllm/model_executor/models/glm4_moe.py`, `vllm/reasoning/glm4_moe_reasoning_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22143 - fuse fp32 for GLM-4.5 e_score_correction_bias

- 链接: https://github.com/vllm-project/vllm/pull/22143
- 状态/时间: merged / 2025-08-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `d3c18c9cb0b6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-3，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fuse fp32 for GLM-4.5 e_score_correction_bias」；模型线: GLM-4.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「fuse fp32 for GLM-4.5 e_score_correction_bias」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +2/-3 (5 lines); hunks: -125,9 +125,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +2/-3 (5 lines); hunks: -125,9 +125,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -125,9 +125,8 @@ def __init__(
-        # noaux_tc is not set in transformers new config now
-        self.gate.e_score_correction_bias = (nn.Parameter(
-            torch.empty(config.n_routed_experts)))
+        self.gate.e_score_correction_bias = nn.Parameter(
+            torch.empty(config.n_routed_experts, dtype=torch.float32))
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +2/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22171 - [Misc] Modify the organization of GLM series

- 链接: https://github.com/vllm-project/vllm/pull/22171
- 状态/时间: merged / 2025-08-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+31/-31，可读 patch 241 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Modify the organization of GLM series」；模型线: GLM-4.5；类别: 文档/测试/CI；主要 diff: `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py`；技术摘要: 覆盖「[Misc] Modify the organization of GLM series」；主要实现面是 `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/models/supported_models.md` modified +5/-5 (10 lines); hunks: -328,7 +328,7 @@ th {; -348,8 +348,8 @@ th {；`tests/models/registry.py` modified +5/-5 (10 lines); hunks: -153,7 +153,7 @@ def check_available_online(; -187,8 +187,8 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`tests/models/multimodal/generation/test_common.py` modified +3/-3 (6 lines); hunks: -355,7 +355,7; -374,7 +374,7；`vllm/model_executor/models/chatglm.py` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -86,10 +86,10 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `docs/models/supported_models.md` modified +5/-5 (10 lines); hunks: -328,7 +328,7 @@ th {; -348,8 +348,8 @@ th {
  - `tests/models/registry.py` modified +5/-5 (10 lines); hunks: -153,7 +153,7 @@ def check_available_online(; -187,8 +187,8 @@ def check_available_online(; symbols: check_available_online
  - `tests/models/multimodal/generation/test_common.py` modified +3/-3 (6 lines); hunks: -355,7 +355,7; -374,7 +374,7
  - `vllm/model_executor/models/chatglm.py` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -86,10 +86,10 @@ def __init__(; symbols: __init__
  - `tests/models/multimodal/processing/test_common.py` modified +2/-2 (4 lines); hunks: -271,8 +271,8 @@ def _test_processing_correctness_one(; symbols: _test_processing_correctness_one
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs/models/supported_models.md` modified +5/-5
  - tests: `tests/models/registry.py` modified +5/-5; `tests/models/multimodal/generation/test_common.py` modified +3/-3; `tests/models/multimodal/processing/test_common.py` modified +2/-2; `tests/models/language/generation/test_common.py` modified +1/-1; `tests/models/multimodal/processing/test_glm4_1v.py` modified +1/-1; `tests/tokenization/test_cached_tokenizer.py` modified +1/-1
  - runtime: `vllm/model_executor/models/chatglm.py` modified +3/-3
- 验证与风险: diff 自带测试面 `tests/distributed/test_pipeline_parallel.py`, `tests/lora/test_add_lora.py`, `tests/lora/test_chatglm3_tp.py`, `tests/models/language/generation/test_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22203 - self.gate dtype update for GLM-4.5

- 链接: https://github.com/vllm-project/vllm/pull/22203
- 状态/时间: merged / 2025-08-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `6fa41e0c32f3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+4/-3，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「self.gate dtype update for GLM-4.5」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「self.gate dtype update for GLM-4.5」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -123,6 +123,7 @@ def __init__(; -180,7 +181,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -123,6 +123,7 @@ def __init__(; -180,7 +181,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -123,6 +123,7 @@ def __init__(
+                                     params_dtype=torch.float32,
@@ -180,7 +181,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        router_logits, _ = self.gate(hidden_states)
+        router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22460 - not tie_word_embeddings for glm-4.5 and glm-4.5v

- 链接: https://github.com/vllm-project/vllm/pull/22460
- 状态/时间: merged / 2025-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `c152e2a8a0f4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-2，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「not tie_word_embeddings for glm-4.5 and glm-4.5v」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「not tie_word_embeddings for glm-4.5 and glm-4.5v」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +0/-2 (2 lines); hunks: -601,8 +601,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +0/-2 (2 lines); hunks: -601,8 +601,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -601,8 +601,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        if self.config.tie_word_embeddings:
-            self.lm_head.weight = self.model.embed_tokens.weight
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +0/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22520 - GLM-4.5V with new class name at transformers

- 链接: https://github.com/vllm-project/vllm/pull/22520
- 状态/时间: merged / 2025-08-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `a6022e6fbcbd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+13/-6，可读 patch 61 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「GLM-4.5V with new class name at transformers」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「GLM-4.5V with new class name at transformers」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +7/-1 (8 lines); hunks: -372,7 +372,13 @@ def forward(; symbols: forward, Glm4MoeModel, __init__，涉及 `forward, Glm4MoeModel, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +7/-1 (8 lines); hunks: -372,7 +372,13 @@ def forward(; symbols: forward, Glm4MoeModel, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +7/-1
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22832 - [Model] Modify the gate implementation of glm4_moe

- 链接: https://github.com/vllm-project/vllm/pull/22832
- 状态/时间: merged / 2025-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `92ff41abea13`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+11/-11，可读 patch 50 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Modify the gate implementation of glm4_moe」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「[Model] Modify the gate implementation of glm4_moe」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +10/-10 (20 lines); hunks: -41,7 +41,6; -118,14 +117,15 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +10/-10 (20 lines); hunks: -41,7 +41,6; -118,14 +117,15 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +10/-10
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23695 - feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200

- 链接: https://github.com/vllm-project/vllm/pull/23695
- 状态/时间: merged / 2025-08-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+146/-0，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200」；模型线: GLM-4.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`；技术摘要: 覆盖「feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200」；主要实现面是 `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +146/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24589 - [Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser

- 链接: https://github.com/vllm-project/vllm/pull/24589
- 状态/时间: merged / 2025-09-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-0，可读 patch 24 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser」；模型线: GLM-4.5；类别: 文档/测试/CI；主要 diff: `docs/features/reasoning_outputs.md`, `docs/features/tool_calling.md`；技术摘要: 覆盖「[Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser」；主要实现面是 `docs/features/reasoning_outputs.md`, `docs/features/tool_calling.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/features/reasoning_outputs.md` modified +1/-0 (1 lines); hunks: -15,6 +15,7 @@ vLLM currently supports the following reasoning models:；`docs/features/tool_calling.md` modified +9/-0 (9 lines); hunks: -311,6 +311,15 @@ Flags:。
- 代码 diff 细节:
  - `docs/features/reasoning_outputs.md` modified +1/-0 (1 lines); hunks: -15,6 +15,7 @@ vLLM currently supports the following reasoning models:
  - `docs/features/tool_calling.md` modified +9/-0 (9 lines); hunks: -311,6 +311,15 @@ Flags:
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs/features/reasoning_outputs.md` modified +1/-0; `docs/features/tool_calling.md` modified +9/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/features/reasoning_outputs.md`, `docs/features/tool_calling.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #24849 - [Model] Apply SharedFusedMoE to glm4_moe.

- 链接: https://github.com/vllm-project/vllm/pull/24849
- 状态/时间: merged / 2025-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `c15309a730fa`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+55/-30，可读 patch 114 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Apply SharedFusedMoE to glm4_moe.」；模型线: GLM-4.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「[Model] Apply SharedFusedMoE to glm4_moe.」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +55/-30 (85 lines); hunks: -46,6 +46,7; -146,25 +147,6 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +55/-30 (85 lines); hunks: -46,6 +46,7; -146,25 +147,6 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +55/-30
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25830 - Update GLM-4.5 Doc transformers version

- 链接: https://github.com/vllm-project/vllm/pull/25830
- 状态/时间: merged / 2025-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `b1ded114b976`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+7/-5，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update GLM-4.5 Doc transformers version」；模型线: GLM-4.5；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「Update GLM-4.5 Doc transformers version」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -21,7 +21,7 @@
-"""Inference-only GLM-4.5 model compatible with HuggingFace weights."""
+"""Inference-only GLM-4.5, GLM-4.6 model compatible with HuggingFace weights."""
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27597 - [Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint.

- 链接: https://github.com/vllm-project/vllm/pull/27597
- 状态/时间: merged / 2025-11-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe_mtp.py`；关联提交 `d3ade61e429f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-4，可读 patch 23 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint.」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_moe_mtp.py`；技术摘要: 覆盖「[Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint.」；主要实现面是 `vllm/model_executor/models/glm4_moe_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe_mtp.py` modified +11/-4 (15 lines); hunks: -256,11 +256,18 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +11/-4 (15 lines); hunks: -256,11 +256,18 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe_mtp.py` modified +11/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28805 - [BugFix] Fix glm4_moe_mtp load weights bug

- 链接: https://github.com/vllm-project/vllm/pull/28805
- 状态/时间: merged / 2025-11-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe_mtp.py`；关联提交 `ab01cd14e5e2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-4，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Fix glm4_moe_mtp load weights bug」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_moe_mtp.py`；技术摘要: 覆盖「[BugFix] Fix glm4_moe_mtp load weights bug」；主要实现面是 `vllm/model_executor/models/glm4_moe_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe_mtp.py` modified +3/-4 (7 lines); hunks: -256,13 +256,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +3/-4 (7 lines); hunks: -256,13 +256,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe_mtp.py` modified +3/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28542 - Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5

- 链接: https://github.com/vllm-project/vllm/pull/28542
- 状态/时间: merged / 2025-11-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 104 个文件，+544/-912，可读 patch 4603 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5」；主要实现面是 `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38 (76 lines); hunks: -26,23 +26,23 @@ def get_rope(; -60,15 +60,15 @@ def get_rope(; symbols: get_rope，涉及 `get_rope`；`vllm/transformers_utils/configs/nemotron.py` modified +31/-29 (60 lines); hunks: -88,8 +88,8 @@ class NemotronConfig(PretrainedConfig):; -132,8 +132,7 @@ def __init__(; symbols: NemotronConfig, __init__, _rope_scaling_validation，涉及 `NemotronConfig, __init__, _rope_scaling_validation`；`vllm/model_executor/models/deepseek_v2.py` modified +13/-30 (43 lines); hunks: -27,7 +27,6; -111,8 +110,6 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/chameleon.py` modified +4/-25 (29 lines); hunks: -264,8 +264,7 @@ def __init__(; -292,7 +291,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38 (76 lines); hunks: -26,23 +26,23 @@ def get_rope(; -60,15 +60,15 @@ def get_rope(; symbols: get_rope
  - `vllm/transformers_utils/configs/nemotron.py` modified +31/-29 (60 lines); hunks: -88,8 +88,8 @@ class NemotronConfig(PretrainedConfig):; -132,8 +132,7 @@ def __init__(; symbols: NemotronConfig, __init__, _rope_scaling_validation
  - `vllm/model_executor/models/deepseek_v2.py` modified +13/-30 (43 lines); hunks: -27,7 +27,6; -111,8 +110,6 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/chameleon.py` modified +4/-25 (29 lines); hunks: -264,8 +264,7 @@ def __init__(; -292,7 +291,6 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/openpangu.py` modified +7/-19 (26 lines); hunks: -77,6 +77,7; -259,7 +260,6 @@ def __init__(; symbols: check_ffn_act_fn, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/__init__.py
@@ -26,23 +26,23 @@ def get_rope(
-    base: float,
-    rope_scaling: dict[str, Any] | None = None,
+    rope_parameters: dict[str, Any] | None = None,
-    if rope_scaling is not None:
+    if rope_parameters is not None:
-        rope_scaling_tuple = {
diff -- vllm/transformers_utils/configs/nemotron.py
@@ -88,8 +88,8 @@ class NemotronConfig(PretrainedConfig):
-        rope_theta (`float`, *optional*, defaults to 10000.0):
-            The base period of the RoPE embeddings.
+        rope_parameters (`dict`, *optional*):
+            The parameters of the RoPE embeddings.
@@ -132,8 +132,7 @@ def __init__(
-        rope_theta=10000.0,
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -27,7 +27,6 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38; `vllm/transformers_utils/configs/nemotron.py` modified +31/-29; `vllm/model_executor/models/deepseek_v2.py` modified +13/-30; `vllm/model_executor/models/chameleon.py` modified +4/-25; `vllm/model_executor/models/openpangu.py` modified +7/-19; `vllm/model_executor/models/hunyuan_v1.py` modified +2/-23
- 验证与风险: diff 自带测试面 `tests/compile/test_functionalization.py`, `tests/kernels/core/test_mrope.py`, `tests/kernels/core/test_pos_encoding.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29342 - [Attention] Remove imports from `vllm/attention/__init__.py`

- 链接: https://github.com/vllm-project/vllm/pull/29342
- 状态/时间: merged / 2025-11-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 96 个文件，+120/-121，可读 patch 923 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Attention] Remove imports from `vllm/attention/__init__.py`」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/whisper.py`, `vllm/model_executor/model_loader/utils.py`, `vllm/model_executor/models/afmoe.py`；技术摘要: 覆盖「[Attention] Remove imports from `vllm/attention/__init__.py`」；主要实现面是 `vllm/model_executor/models/whisper.py`, `vllm/model_executor/model_loader/utils.py`, `vllm/model_executor/models/afmoe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/whisper.py` modified +2/-2 (4 lines); hunks: -16,8 +16,8；`vllm/model_executor/model_loader/utils.py` modified +1/-2 (3 lines); hunks: -11,8 +11,7；`vllm/model_executor/models/afmoe.py` modified +2/-1 (3 lines); hunks: -9,7 +9,8；`vllm/model_executor/models/apertus.py` modified +2/-1 (3 lines); hunks: -32,7 +32,8。
- 代码 diff 细节:
  - `vllm/model_executor/models/whisper.py` modified +2/-2 (4 lines); hunks: -16,8 +16,8
  - `vllm/model_executor/model_loader/utils.py` modified +1/-2 (3 lines); hunks: -11,8 +11,7
  - `vllm/model_executor/models/afmoe.py` modified +2/-1 (3 lines); hunks: -9,7 +9,8
  - `vllm/model_executor/models/apertus.py` modified +2/-1 (3 lines); hunks: -32,7 +32,8
  - `vllm/model_executor/models/clip.py` modified +1/-2 (3 lines); hunks: -14,8 +14,7
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/whisper.py
@@ -16,8 +16,8 @@
-from vllm.attention import Attention, AttentionType
-from vllm.attention.layer import MultiHeadAttention
+from vllm.attention.backends.abstract import AttentionType
+from vllm.attention.layer import Attention, MultiHeadAttention
diff -- vllm/model_executor/model_loader/utils.py
@@ -11,8 +11,7 @@
-from vllm.attention import Attention
-from vllm.attention.layer import MLAAttention
+from vllm.attention.layer import Attention, MLAAttention
diff -- vllm/model_executor/models/afmoe.py
@@ -9,7 +9,8 @@
-from vllm.attention import Attention, AttentionType
+from vllm.attention.backends.abstract import AttentionType
+from vllm.attention.layer import Attention
diff -- vllm/model_executor/models/apertus.py
@@ -32,7 +32,8 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/whisper.py` modified +2/-2; `vllm/model_executor/model_loader/utils.py` modified +1/-2; `vllm/model_executor/models/afmoe.py` modified +2/-1; `vllm/model_executor/models/apertus.py` modified +2/-1; `vllm/model_executor/models/clip.py` modified +1/-2; `vllm/model_executor/models/gemma3.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/compile/test_fusion_attn.py`, `tests/compile/test_qk_norm_rope_fusion.py`, `tests/kernels/utils.py`, `tests/v1/worker/test_gpu_model_runner.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29966 - Access `partial_rotary_factor` from `rope_parameters`

- 链接: https://github.com/vllm-project/vllm/pull/29966
- 状态/时间: merged / 2025-12-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 21 个文件，+43/-62，可读 patch 396 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Access `partial_rotary_factor` from `rope_parameters`」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/transformers_utils/configs/nemotron.py`, `vllm/transformers_utils/configs/qwen3_next.py`, `vllm/model_executor/models/gpt_neox.py`；技术摘要: 覆盖「Access `partial_rotary_factor` from `rope_parameters`」；主要实现面是 `vllm/transformers_utils/configs/nemotron.py`, `vllm/transformers_utils/configs/qwen3_next.py`, `vllm/model_executor/models/gpt_neox.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/nemotron.py` modified +13/-7 (20 lines); hunks: -89,9 +89,14 @@ class NemotronConfig(PretrainedConfig):; -133,7 +138,6 @@ def __init__(; symbols: NemotronConfig, __init__，涉及 `NemotronConfig, __init__`；`vllm/transformers_utils/configs/qwen3_next.py` modified +5/-3 (8 lines); hunks: -103,8 +103,8 @@ class Qwen3NextConfig(PretrainedConfig):; -198,7 +198,6 @@ def __init__(; symbols: Qwen3NextConfig, __init__，涉及 `Qwen3NextConfig, __init__`；`vllm/model_executor/models/gpt_neox.py` modified +2/-4 (6 lines); hunks: -89,16 +89,14 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/model_executor/layers/rotary_embedding/__init__.py` modified +4/-1 (5 lines); hunks: -30,7 +30,6 @@ def get_rope(; -55,6 +54,10 @@ def get_rope(; symbols: get_rope，涉及 `get_rope`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/nemotron.py` modified +13/-7 (20 lines); hunks: -89,9 +89,14 @@ class NemotronConfig(PretrainedConfig):; -133,7 +138,6 @@ def __init__(; symbols: NemotronConfig, __init__
  - `vllm/transformers_utils/configs/qwen3_next.py` modified +5/-3 (8 lines); hunks: -103,8 +103,8 @@ class Qwen3NextConfig(PretrainedConfig):; -198,7 +198,6 @@ def __init__(; symbols: Qwen3NextConfig, __init__
  - `vllm/model_executor/models/gpt_neox.py` modified +2/-4 (6 lines); hunks: -89,16 +89,14 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +4/-1 (5 lines); hunks: -30,7 +30,6 @@ def get_rope(; -55,6 +54,10 @@ def get_rope(; symbols: get_rope
  - `vllm/model_executor/models/apertus.py` modified +1/-4 (5 lines); hunks: -148,8 +148,6 @@ def __init__(; -228,11 +226,10 @@ def _init_rotary_emb(; symbols: __init__, _init_rotary_emb
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/configs/nemotron.py
@@ -89,9 +89,14 @@ class NemotronConfig(PretrainedConfig):
-            The parameters of the RoPE embeddings.
-        partial_rotary_factor (`float`, *optional*, defaults to 0.5):
-            Percentage of the query and keys which will have rotary embedding.
+            The parameters of the RoPE embeddings. Expected contents:
+                `rope_theta` (`float`): The base period of the RoPE embeddings.
+                `rope_type` (`str`):
diff -- vllm/transformers_utils/configs/qwen3_next.py
@@ -103,8 +103,8 @@ class Qwen3NextConfig(PretrainedConfig):
-        partial_rotary_factor (`float`, *optional*, defaults to 0.25):
-            Percentage of the query and keys which will have rotary embedding.
+                `partial_rotary_factor` (`float`, *optional*, defaults to 0.25):
+                    Percentage of the query and keys which will have rotary embedding.
@@ -198,7 +198,6 @@ def __init__(
-        partial_rotary_factor=0.25,
diff -- vllm/model_executor/models/gpt_neox.py
@@ -89,16 +89,14 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/nemotron.py` modified +13/-7; `vllm/transformers_utils/configs/qwen3_next.py` modified +5/-3; `vllm/model_executor/models/gpt_neox.py` modified +2/-4; `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +4/-1; `vllm/model_executor/models/apertus.py` modified +1/-4; `vllm/model_executor/models/config.py` modified +0/-5
- 验证与风险: diff 自带测试面 `tests/kernels/core/test_mrope.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30389 - Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim`

- 链接: https://github.com/vllm-project/vllm/pull/30389
- 状态/时间: merged / 2025-12-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 83 个文件，+238/-292，可读 patch 1379 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim`」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/model_executor/models/config.py`, `vllm/model_executor/models/phi.py`；技术摘要: 覆盖「Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim`」；主要实现面是 `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/model_executor/models/config.py`, `vllm/model_executor/models/phi.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +160/-166 (326 lines); hunks: -25,7 +25,6; -54,12 +53,15 @@ def get_rope(; symbols: get_rope，涉及 `get_rope`；`vllm/model_executor/models/config.py` modified +7/-5 (12 lines); hunks: -42,9 +42,10 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:; -77,9 +78,11 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:; symbols: verify_and_update_config，涉及 `verify_and_update_config`；`vllm/model_executor/models/phi.py` modified +4/-8 (12 lines); hunks: -84,19 +84,18 @@ def __init__(; -109,13 +108,10 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/bamba.py` modified +2/-5 (7 lines); hunks: -178,14 +178,11 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +160/-166 (326 lines); hunks: -25,7 +25,6; -54,12 +53,15 @@ def get_rope(; symbols: get_rope
  - `vllm/model_executor/models/config.py` modified +7/-5 (12 lines); hunks: -42,9 +42,10 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:; -77,9 +78,11 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:; symbols: verify_and_update_config
  - `vllm/model_executor/models/phi.py` modified +4/-8 (12 lines); hunks: -84,19 +84,18 @@ def __init__(; -109,13 +108,10 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/bamba.py` modified +2/-5 (7 lines); hunks: -178,14 +178,11 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/chatglm.py` modified +5/-2 (7 lines); hunks: -99,13 +99,16 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/__init__.py
@@ -25,7 +25,6 @@
-    rotary_dim: int,
@@ -54,12 +53,15 @@ def get_rope(
-    partial_rotary_factor = 1.0
-    if rope_parameters is not None:
-        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
+    rope_parameters = rope_parameters or {}
diff -- vllm/model_executor/models/config.py
@@ -42,9 +42,10 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:
+        rotary_dim = getattr(config, "rotary_emb_dim", head_dim)
+        config.rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim
-            "rotary_dim": getattr(config, "rotary_emb_dim", head_dim),
@@ -77,9 +78,11 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:
+            rotary_dim = getattr(config, "rotary_emb_dim", head_dim)
+            config.rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim
diff -- vllm/model_executor/models/phi.py
@@ -84,19 +84,18 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +160/-166; `vllm/model_executor/models/config.py` modified +7/-5; `vllm/model_executor/models/phi.py` modified +4/-8; `vllm/model_executor/models/bamba.py` modified +2/-5; `vllm/model_executor/models/chatglm.py` modified +5/-2; `vllm/model_executor/models/falcon_h1.py` modified +2/-5
- 验证与风险: diff 自带测试面 `tests/compile/test_functionalization.py`, `tests/kernels/core/test_mrope.py`, `tests/kernels/core/test_pos_encoding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30675 - [Refactor] [2/N] Move tool parsers into the vLLM main directory

- 链接: https://github.com/vllm-project/vllm/pull/30675
- 状态/时间: merged / 2025-12-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 61 个文件，+288/-257，可读 patch 1115 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] [2/N] Move tool parsers into the vLLM main directory」；模型线: GLM-4.5；类别: 文档/测试/CI；主要 diff: `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/tool_parsers/__init__.py`, `tests/tool_use/test_qwen3coder_tool_parser.py`；技术摘要: 覆盖「[Refactor] [2/N] Move tool parsers into the vLLM main directory」；主要实现面是 `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/tool_parsers/__init__.py`, `tests/tool_use/test_qwen3coder_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +23/-140 (163 lines); hunks: -1,150 +1,33; symbols: __getattr__, register_lazy_tool_parsers，涉及 `__getattr__, register_lazy_tool_parsers`；`vllm/tool_parsers/__init__.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: register_lazy_tool_parsers，涉及 `register_lazy_tool_parsers`；`tests/tool_use/test_qwen3coder_tool_parser.py` modified +4/-4 (8 lines); hunks: -13,12 +13,12；`vllm/tool_parsers/granite_20b_fc_tool_parser.py` renamed +4/-4 (8 lines); hunks: -19,17 +19,17。
- 代码 diff 细节:
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +23/-140 (163 lines); hunks: -1,150 +1,33; symbols: __getattr__, register_lazy_tool_parsers
  - `vllm/tool_parsers/__init__.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: register_lazy_tool_parsers
  - `tests/tool_use/test_qwen3coder_tool_parser.py` modified +4/-4 (8 lines); hunks: -13,12 +13,12
  - `vllm/tool_parsers/granite_20b_fc_tool_parser.py` renamed +4/-4 (8 lines); hunks: -19,17 +19,17
  - `vllm/tool_parsers/granite_tool_parser.py` renamed +4/-4 (8 lines); hunks: -17,17 +17,17
- 关键代码摘录:

```diff
diff -- vllm/entrypoints/openai/tool_parsers/__init__.py
@@ -1,150 +1,33 @@
-from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
-    ToolParser,
-    ToolParserManager,
-)
+import warnings
-__all__ = ["ToolParser", "ToolParserManager"]
diff -- vllm/tool_parsers/__init__.py
@@ -0,0 +1,150 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from vllm.tool_parsers.abstract_tool_parser import (
+    ToolParser,
+    ToolParserManager,
+)
diff -- tests/tool_use/test_qwen3coder_tool_parser.py
@@ -13,12 +13,12 @@
```

- 已读文件:
  - runtime: `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +23/-140; `vllm/tool_parsers/__init__.py` added +150/-0; `vllm/tool_parsers/granite_20b_fc_tool_parser.py` renamed +4/-4; `vllm/tool_parsers/granite_tool_parser.py` renamed +4/-4; `vllm/tool_parsers/hunyuan_a13b_tool_parser.py` renamed +4/-4; `vllm/tool_parsers/internlm2_tool_parser.py` renamed +4/-4
  - tests: `tests/tool_use/test_qwen3coder_tool_parser.py` modified +4/-4
- 验证与风险: diff 自带测试面 `tests/entrypoints/openai/test_serving_chat.py`, `tests/entrypoints/openai/tool_parsers/test_gigachat3_tool_parser.py`, `tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py`, `tests/entrypoints/openai/tool_parsers/test_hunyuan_a13b_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30693 - [Refactor] [3/N] Move tool parser tests and run on CPU

- 链接: https://github.com/vllm-project/vllm/pull/30693
- 状态/时间: merged / 2025-12-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+14/-53，可读 patch 197 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] [3/N] Move tool parser tests and run on CPU」；模型线: GLM-4.5；类别: 文档/测试/CI；主要 diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_jamba_tool_parser.py`, `tests/tool_parsers/test_kimi_k2_tool_parser.py`；技术摘要: 覆盖「[Refactor] [3/N] Move tool parser tests and run on CPU」；主要实现面是 `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_jamba_tool_parser.py`, `tests/tool_parsers/test_kimi_k2_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_glm4_moe_tool_parser.py` renamed +0/-2 (2 lines); hunks: -12,8 +12,6；`tests/tool_parsers/test_jamba_tool_parser.py` renamed +0/-2 (2 lines); hunks: -13,8 +13,6；`tests/tool_parsers/test_kimi_k2_tool_parser.py` renamed +0/-2 (2 lines); hunks: -10,8 +10,6；`tests/tool_parsers/test_minimax_tool_parser.py` renamed +0/-2 (2 lines); hunks: -15,8 +15,6。
- 代码 diff 细节:
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` renamed +0/-2 (2 lines); hunks: -12,8 +12,6
  - `tests/tool_parsers/test_jamba_tool_parser.py` renamed +0/-2 (2 lines); hunks: -13,8 +13,6
  - `tests/tool_parsers/test_kimi_k2_tool_parser.py` renamed +0/-2 (2 lines); hunks: -10,8 +10,6
  - `tests/tool_parsers/test_minimax_tool_parser.py` renamed +0/-2 (2 lines); hunks: -15,8 +15,6
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` renamed +0/-2 (2 lines); hunks: -20,8 +20,6
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_glm4_moe_tool_parser.py
@@ -12,8 +12,6 @@
-pytestmark = pytest.mark.cpu_test
diff -- tests/tool_parsers/test_jamba_tool_parser.py
@@ -13,8 +13,6 @@
-pytestmark = pytest.mark.cpu_test
diff -- tests/tool_parsers/test_kimi_k2_tool_parser.py
@@ -10,8 +10,6 @@
-pytestmark = pytest.mark.cpu_test
diff -- tests/tool_parsers/test_minimax_tool_parser.py
@@ -15,8 +15,6 @@
-pytestmark = pytest.mark.cpu_test
diff -- tests/tool_parsers/test_qwen3coder_tool_parser.py
@@ -20,8 +20,6 @@
-pytestmark = pytest.mark.cpu_test
diff -- tests/tool_parsers/test_seed_oss_tool_parser.py
@@ -18,8 +18,6 @@
-pytestmark = pytest.mark.cpu_test
```

- 已读文件:
  - tests: `tests/tool_parsers/test_glm4_moe_tool_parser.py` renamed +0/-2; `tests/tool_parsers/test_jamba_tool_parser.py` renamed +0/-2; `tests/tool_parsers/test_kimi_k2_tool_parser.py` renamed +0/-2; `tests/tool_parsers/test_minimax_tool_parser.py` renamed +0/-2; `tests/tool_parsers/test_qwen3coder_tool_parser.py` renamed +0/-2; `tests/tool_parsers/test_seed_oss_tool_parser.py` renamed +0/-2
- 验证与风险: diff 自带测试面 `tests/tool_parsers/__init__.py`, `tests/tool_parsers/test_deepseekv31_tool_parser.py`, `tests/tool_parsers/test_ernie45_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30920 - [Bugfix] Fix Unicode issues in GLM-4 tool calling

- 链接: https://github.com/vllm-project/vllm/pull/30920
- 状态/时间: merged / 2025-12-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-1，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Unicode issues in GLM-4 tool calling」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/glm4_moe_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Fix Unicode issues in GLM-4 tool calling」；主要实现面是 `vllm/tool_parsers/glm4_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +2/-1 (3 lines); hunks: -114,7 +114,8 @@ def _deserialize(value: str) -> Any:; symbols: _deserialize，涉及 `_deserialize`。
- 代码 diff 细节:
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +2/-1 (3 lines); hunks: -114,7 +114,8 @@ def _deserialize(value: str) -> Any:; symbols: _deserialize
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/glm4_moe_tool_parser.py
@@ -114,7 +114,8 @@ def _deserialize(value: str) -> Any:
-                            name=tc_name, arguments=json.dumps(arg_dct)
+                            name=tc_name,
+                            arguments=json.dumps(arg_dct, ensure_ascii=False),
```

- 已读文件:
  - runtime: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `vllm/tool_parsers/glm4_moe_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30876 - GLM-4.7 Tool Parser and Doc Update

- 链接: https://github.com/vllm-project/vllm/pull/30876
- 状态/时间: merged / 2025-12-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+38/-3，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「GLM-4.7 Tool Parser and Doc Update」；模型线: GLM-4.5；类别: 文档/测试/CI；主要 diff: `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/__init__.py`, `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「GLM-4.7 Tool Parser and Doc Update」；主要实现面是 `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/__init__.py`, `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/glm47_moe_tool_parser.py` added +23/-0 (23 lines); hunks: -0,0 +1,23; symbols: Glm47MoeModelToolParser, __init__，涉及 `Glm47MoeModelToolParser, __init__`；`vllm/tool_parsers/__init__.py` modified +4/-0 (4 lines); hunks: -42,6 +42,10；`vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -21,7 +21,8；`docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: -387,7 +387,7 @@ th {。
- 代码 diff 细节:
  - `vllm/tool_parsers/glm47_moe_tool_parser.py` added +23/-0 (23 lines); hunks: -0,0 +1,23; symbols: Glm47MoeModelToolParser, __init__
  - `vllm/tool_parsers/__init__.py` modified +4/-0 (4 lines); hunks: -42,6 +42,10
  - `vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -21,7 +21,8
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: -387,7 +387,7 @@ th {
  - `docs/features/tool_calling.md` modified +8/-1 (9 lines); hunks: -352,10 +352,17 @@ Supported models:
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/glm47_moe_tool_parser.py
@@ -0,0 +1,23 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import regex as re
+from vllm.logger import init_logger
+from vllm.tokenizers import TokenizerLike
+from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser
diff -- vllm/tool_parsers/__init__.py
@@ -42,6 +42,10 @@
+    "glm47": (
+        "glm47_moe_tool_parser",
+        "Glm47MoeModelToolParser",
+    ),
diff -- vllm/model_executor/models/glm4_moe.py
@@ -21,7 +21,8 @@
-"""Inference-only GLM-4.5, GLM-4.6 model compatible with HuggingFace weights."""
+"""Inference-only GLM-4.5, GLM-4.6, GLM-4.7 model
```

- 已读文件:
  - runtime: `vllm/tool_parsers/glm47_moe_tool_parser.py` added +23/-0; `vllm/tool_parsers/__init__.py` modified +4/-0; `vllm/model_executor/models/glm4_moe.py` modified +2/-1
  - docs: `docs/models/supported_models.md` modified +1/-1; `docs/features/tool_calling.md` modified +8/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe.py`, `vllm/tool_parsers/__init__.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31622 - Fix GLM-4.6v flash tool calling in transformers 5.x

- 链接: https://github.com/vllm-project/vllm/pull/31622
- 状态/时间: merged / 2026-01-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+68/-0，可读 patch 76 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GLM-4.6v flash tool calling in transformers 5.x」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/glm4_moe_tool_parser.py`, `examples/tool_chat_template_glm4.jinja`；技术摘要: 覆盖「Fix GLM-4.6v flash tool calling in transformers 5.x」；主要实现面是 `vllm/tool_parsers/glm4_moe_tool_parser.py`, `examples/tool_chat_template_glm4.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +14/-0 (14 lines); hunks: -56,6 +56,20 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, adjust_request, extract_tool_calls，涉及 `__init__, adjust_request, extract_tool_calls`；`examples/tool_chat_template_glm4.jinja` added +54/-0 (54 lines); hunks: -0,0 +1,54。
- 代码 diff 细节:
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +14/-0 (14 lines); hunks: -56,6 +56,20 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, adjust_request, extract_tool_calls
  - `examples/tool_chat_template_glm4.jinja` added +54/-0 (54 lines); hunks: -0,0 +1,54
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/glm4_moe_tool_parser.py
@@ -56,6 +56,20 @@ def __init__(self, tokenizer: TokenizerLike):
+    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
+        """
+        Adjust request parameters to ensure tool call tokens are not skipped
+        during tokenizer decoding.
+        """
+        request = super().adjust_request(request)
diff -- examples/tool_chat_template_glm4.jinja
@@ -0,0 +1,54 @@
+{%- set counter = namespace(index=0) -%}
+{%- if not tools is defined %}
+    {%- set tools = none %}
+{%- endif %}
+{%- if messages and messages[0]['role'] == 'system' %}
+    {%- set system_message = messages[0]['content']|trim %}
```

- 已读文件:
  - runtime: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +14/-0
  - docs: `examples/tool_chat_template_glm4.jinja` added +54/-0
- 验证与风险: runtime 路径改动集中在 `vllm/tool_parsers/glm4_moe_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31055 - [Bugfix] Fix GLM-4 MoE router logits dtype for data parallel chunking

- 链接: https://github.com/vllm-project/vllm/pull/31055
- 状态/时间: merged / 2026-01-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+13/-1，可读 patch 63 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix GLM-4 MoE router logits dtype for data parallel chunking」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「[Bugfix] Fix GLM-4 MoE router logits dtype for data parallel chunking」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/layer.py` modified +6/-1 (7 lines); hunks: -314,6 +314,7 @@ class FusedMoE(CustomOp):; -348,6 +349,7 @@ def __init__(; symbols: FusedMoE, __init__, ensure_dp_chunking_init，涉及 `FusedMoE, __init__, ensure_dp_chunking_init`；`vllm/model_executor/layers/fused_moe/config.py` modified +6/-0 (6 lines); hunks: -1006,6 +1006,9 @@ class FusedMoEConfig:; -1022,6 +1025,9 @@ def __post_init__(self):; symbols: FusedMoEConfig, __post_init__, tp_size，涉及 `FusedMoEConfig, __post_init__, tp_size`；`vllm/model_executor/models/glm4_moe.py` modified +1/-0 (1 lines); hunks: -197,6 +197,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +6/-1 (7 lines); hunks: -314,6 +314,7 @@ class FusedMoE(CustomOp):; -348,6 +349,7 @@ def __init__(; symbols: FusedMoE, __init__, ensure_dp_chunking_init
  - `vllm/model_executor/layers/fused_moe/config.py` modified +6/-0 (6 lines); hunks: -1006,6 +1006,9 @@ class FusedMoEConfig:; -1022,6 +1025,9 @@ def __post_init__(self):; symbols: FusedMoEConfig, __post_init__, tp_size
  - `vllm/model_executor/models/glm4_moe.py` modified +1/-0 (1 lines); hunks: -197,6 +197,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -314,6 +314,7 @@ class FusedMoE(CustomOp):
+        router_logits_dtype: Data type for router logits buffers.
@@ -348,6 +349,7 @@ def __init__(
+        router_logits_dtype: torch.dtype | None = None,
@@ -559,6 +561,7 @@ def __init__(
+            router_logits_dtype=router_logits_dtype,
@@ -1509,7 +1512,9 @@ def ensure_dp_chunking_init(self):
diff -- vllm/model_executor/layers/fused_moe/config.py
@@ -1006,6 +1006,9 @@ class FusedMoEConfig:
+    # Defaults to in_dtype if not specified.
+    router_logits_dtype: torch.dtype | None = None
@@ -1022,6 +1025,9 @@ def __post_init__(self):
+        if self.router_logits_dtype is None:
+            self.router_logits_dtype = self.in_dtype
diff -- vllm/model_executor/models/glm4_moe.py
@@ -197,6 +197,7 @@ def __init__(
+            router_logits_dtype=torch.float32,
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +6/-1; `vllm/model_executor/layers/fused_moe/config.py` modified +6/-0; `vllm/model_executor/models/glm4_moe.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31104 - [BugFix] LoRA: Support loading base_layer of experts

- 链接: https://github.com/vllm-project/vllm/pull/31104
- 状态/时间: merged / 2026-01-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+46/-3，可读 patch 319 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] LoRA: Support loading base_layer of experts」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/llama4.py`；技术摘要: 覆盖「[BugFix] LoRA: Support loading base_layer of experts」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3 (13 lines); hunks: -2007,6 +2007,7 @@ def combine_output(states: torch.Tensor) -> torch.Tensor:; -2025,13 +2026,19 @@ def make_expert_params_mapping(; symbols: combine_output, make_expert_params_mapping，涉及 `combine_output, make_expert_params_mapping`；`vllm/model_executor/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -1486,6 +1486,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int,...; -1519,6 +1520,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: get_expert_mapping, load_weights，涉及 `get_expert_mapping, load_weights`；`vllm/model_executor/models/llama4.py` modified +2/-0 (2 lines); hunks: -539,6 +539,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; -548,6 +549,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`；`vllm/model_executor/models/afmoe.py` modified +1/-0 (1 lines); hunks: -475,6 +475,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: get_expert_mapping，涉及 `get_expert_mapping`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3 (13 lines); hunks: -2007,6 +2007,7 @@ def combine_output(states: torch.Tensor) -> torch.Tensor:; -2025,13 +2026,19 @@ def make_expert_params_mapping(; symbols: combine_output, make_expert_params_mapping
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -1486,6 +1486,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int,...; -1519,6 +1520,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: get_expert_mapping, load_weights
  - `vllm/model_executor/models/llama4.py` modified +2/-0 (2 lines); hunks: -539,6 +539,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; -548,6 +549,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
  - `vllm/model_executor/models/afmoe.py` modified +1/-0 (1 lines); hunks: -475,6 +475,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: get_expert_mapping
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-0 (1 lines); hunks: -476,6 +476,7 @@ def forward(; symbols: forward, get_expert_mapping
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -2007,6 +2007,7 @@ def combine_output(states: torch.Tensor) -> torch.Tensor:
+        model: torch.nn.Module,
@@ -2025,13 +2026,19 @@ def make_expert_params_mapping(
+        base_layer = (
+            "base_layer."
+            if any(".base_layer." in name for name, _ in model.named_parameters())
+            else ""
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -1486,6 +1486,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
+            self,
@@ -1519,6 +1520,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            self,
diff -- vllm/model_executor/models/llama4.py
@@ -539,6 +539,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            self,
@@ -548,6 +549,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            self,
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3; `vllm/model_executor/models/deepseek_v2.py` modified +2/-0; `vllm/model_executor/models/llama4.py` modified +2/-0; `vllm/model_executor/models/afmoe.py` modified +1/-0; `vllm/model_executor/models/bailing_moe.py` modified +1/-0; `vllm/model_executor/models/deepseek_eagle.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31869 - [Model] Cleanup: Remove redundant manual definition of `make_empty_intermediate_tensors` in GLM-4-MoE

- 链接: https://github.com/vllm-project/vllm/pull/31869
- 状态/时间: merged / 2026-01-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-14，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Cleanup: Remove redundant manual definition of `make_empty_intermediate_tensors` in GLM-4-MoE」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「[Model] Cleanup: Remove redundant manual definition of `make_empty_intermediate_tensors` in GLM-4-MoE」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +0/-14 (14 lines); hunks: -478,20 +478,6 @@ def forward(; symbols: forward, make_empty_intermediate_tensors, get_expert_mapping，涉及 `forward, make_empty_intermediate_tensors, get_expert_mapping`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +0/-14 (14 lines); hunks: -478,20 +478,6 @@ def forward(; symbols: forward, make_empty_intermediate_tensors, get_expert_mapping
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -478,20 +478,6 @@ def forward(
-    def make_empty_intermediate_tensors(
-        self, batch_size: int, dtype: torch.dtype, device: torch.device
-    ) -> IntermediateTensors:
-        return IntermediateTensors(
-            {
-                "hidden_states": torch.zeros(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +0/-14
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31757 - [Bugfix][MTP] Fix GLM4 MoE fp8 loading with MTP on

- 链接: https://github.com/vllm-project/vllm/pull/31757
- 状态/时间: merged / 2026-01-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-1，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][MTP] Fix GLM4 MoE fp8 loading with MTP on」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_moe_mtp.py`；技术摘要: 覆盖「[Bugfix][MTP] Fix GLM4 MoE fp8 loading with MTP on」；主要实现面是 `vllm/model_executor/models/glm4_moe_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe_mtp.py` modified +6/-1 (7 lines); hunks: -106,7 +106,7 @@ def forward(; -267,6 +267,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: forward, load_weights，涉及 `forward, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +6/-1 (7 lines); hunks: -106,7 +106,7 @@ def forward(; -267,6 +267,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: forward, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe_mtp.py
@@ -106,7 +106,7 @@ def forward(
-        inputs_embeds[positions == 0] = 0
+        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
@@ -267,6 +267,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            # Some checkpoints include weight scale tensors for the LM head even
+            # when the quantized head isn't built. Skip them if the model does
+            # not expose a matching parameter to avoid KeyError during load.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe_mtp.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32101 - [MTP][GLM][Bugfix] Fixed .weight_scale loading logic that dropped MTP prediction accuracy with fp8+mtp

- 链接: https://github.com/vllm-project/vllm/pull/32101
- 状态/时间: merged / 2026-01-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-5，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MTP][GLM][Bugfix] Fixed .weight_scale loading logic that dropped MTP prediction accuracy with fp8+mtp」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_moe_mtp.py`；技术摘要: 覆盖「[MTP][GLM][Bugfix] Fixed .weight_scale loading logic that dropped MTP prediction accuracy with fp8+mtp」；主要实现面是 `vllm/model_executor/models/glm4_moe_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe_mtp.py` modified +6/-5 (11 lines); hunks: -268,11 +268,6 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -315,6 +310,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +6/-5 (11 lines); hunks: -268,11 +268,6 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -315,6 +310,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe_mtp.py
@@ -268,11 +268,6 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-            # Some checkpoints include weight scale tensors for the LM head even
-            # when the quantized head isn't built. Skip them if the model does
-            # not expose a matching parameter to avoid KeyError during load.
-            if name.endswith(".weight_scale") and name not in params_dict:
-                continue
@@ -315,6 +310,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe_mtp.py` modified +6/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32150 - [Model] Remove incorrect `SupportsPP` from MTP models

- 链接: https://github.com/vllm-project/vllm/pull/32150
- 状态/时间: merged / 2026-01-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+6/-15，可读 patch 112 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Remove incorrect `SupportsPP` from MTP models」；模型线: GLM-4.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_next_mtp.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/ernie_mtp.py`；技术摘要: 覆盖「[Model] Remove incorrect `SupportsPP` from MTP models」；主要实现面是 `vllm/model_executor/models/qwen3_next_mtp.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/ernie_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_next_mtp.py` modified +1/-5 (6 lines); hunks: -27,7 +27,6; -221,7 +220,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, Qwen3NextMTP, __init__, embed_input_ids，涉及 `load_weights, Qwen3NextMTP, __init__`；`vllm/model_executor/models/deepseek_mtp.py` modified +1/-2 (3 lines); hunks: -32,7 +32,6; -181,7 +180,7 @@ def compute_logits(; symbols: compute_logits, DeepSeekMTP, __init__，涉及 `compute_logits, DeepSeekMTP, __init__`；`vllm/model_executor/models/ernie_mtp.py` modified +1/-2 (3 lines); hunks: -39,7 +39,6; -143,7 +142,7 @@ def compute_logits(; symbols: compute_logits, ErnieMTP, __init__，涉及 `compute_logits, ErnieMTP, __init__`；`vllm/model_executor/models/glm4_moe_mtp.py` modified +1/-2 (3 lines); hunks: -47,7 +47,6; -184,7 +183,7 @@ def compute_logits(; symbols: compute_logits, Glm4MoeMTP, __init__，涉及 `compute_logits, Glm4MoeMTP, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_next_mtp.py` modified +1/-5 (6 lines); hunks: -27,7 +27,6; -221,7 +220,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, Qwen3NextMTP, __init__, embed_input_ids
  - `vllm/model_executor/models/deepseek_mtp.py` modified +1/-2 (3 lines); hunks: -32,7 +32,6; -181,7 +180,7 @@ def compute_logits(; symbols: compute_logits, DeepSeekMTP, __init__
  - `vllm/model_executor/models/ernie_mtp.py` modified +1/-2 (3 lines); hunks: -39,7 +39,6; -143,7 +142,7 @@ def compute_logits(; symbols: compute_logits, ErnieMTP, __init__
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +1/-2 (3 lines); hunks: -47,7 +47,6; -184,7 +183,7 @@ def compute_logits(; symbols: compute_logits, Glm4MoeMTP, __init__
  - `vllm/model_executor/models/longcat_flash_mtp.py` modified +1/-2 (3 lines); hunks: -24,7 +24,6; -124,7 +123,7 @@ def forward(; symbols: forward, LongCatFlashMTP, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_next_mtp.py
@@ -27,7 +27,6 @@
-from .interfaces import SupportsPP
@@ -221,7 +220,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-class Qwen3NextMTP(nn.Module, SupportsPP, QwenNextMixtureOfExperts):
+class Qwen3NextMTP(nn.Module, QwenNextMixtureOfExperts):
@@ -253,9 +252,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        self.make_empty_intermediate_tensors = (
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -32,7 +32,6 @@
-from .interfaces import SupportsPP
@@ -181,7 +180,7 @@ def compute_logits(
-class DeepSeekMTP(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts):
+class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):
diff -- vllm/model_executor/models/ernie_mtp.py
@@ -39,7 +39,6 @@
-from .interfaces import SupportsPP
@@ -143,7 +142,7 @@ def compute_logits(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_next_mtp.py` modified +1/-5; `vllm/model_executor/models/deepseek_mtp.py` modified +1/-2; `vllm/model_executor/models/ernie_mtp.py` modified +1/-2; `vllm/model_executor/models/glm4_moe_mtp.py` modified +1/-2; `vllm/model_executor/models/longcat_flash_mtp.py` modified +1/-2; `vllm/model_executor/models/openpangu_mtp.py` modified +1/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/ernie_mtp.py`, `vllm/model_executor/models/glm4_moe_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32240 - [Refactor] [6/N] to simplify the vLLM openai chat_completion serving architecture

- 链接: https://github.com/vllm-project/vllm/pull/32240
- 状态/时间: merged / 2026-01-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 128 个文件，+1310/-1097，可读 patch 3603 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] [6/N] to simplify the vLLM openai chat_completion serving architecture」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/entrypoints/openai/engine/protocol.py`, `vllm/entrypoints/openai/chat_completion/protocol.py`, `vllm/entrypoints/serve/tokenize/protocol.py`；技术摘要: 覆盖「[Refactor] [6/N] to simplify the vLLM openai chat_completion serving architecture」；主要实现面是 `vllm/entrypoints/openai/engine/protocol.py`, `vllm/entrypoints/openai/chat_completion/protocol.py`, `vllm/entrypoints/serve/tokenize/protocol.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/entrypoints/openai/engine/protocol.py` renamed +89/-847 (936 lines); hunks: -11,10 +11,6; -234,20 +230,6 @@ class FunctionDefinition(OpenAIBaseModel):; symbols: FunctionDefinition, ChatCompletionToolsParam, ChatCompletionNamedFunction, ChatCompletionNamedToolChoiceParam，涉及 `FunctionDefinition, ChatCompletionToolsParam, ChatCompletionNamedFunction`；`vllm/entrypoints/openai/chat_completion/protocol.py` added +654/-0 (654 lines); hunks: -0,0 +1,654; symbols: ChatMessage, handle_deprecated_reasoning_content, ChatCompletionLogProb, ChatCompletionLogProbsContent，涉及 `ChatMessage, handle_deprecated_reasoning_content, ChatCompletionLogProb`；`vllm/entrypoints/serve/tokenize/protocol.py` added +139/-0 (139 lines); hunks: -0,0 +1,139; symbols: TokenizeCompletionRequest, TokenizeChatRequest, check_generation_prompt, TokenizeResponse，涉及 `TokenizeCompletionRequest, TokenizeChatRequest, check_generation_prompt`；`vllm/entrypoints/openai/chat_completion/api_router.py` added +77/-0 (77 lines); hunks: -0,0 +1,77; symbols: chat, create_chat_completion, attach_router，涉及 `chat, create_chat_completion, attach_router`。
- 代码 diff 细节:
  - `vllm/entrypoints/openai/engine/protocol.py` renamed +89/-847 (936 lines); hunks: -11,10 +11,6; -234,20 +230,6 @@ class FunctionDefinition(OpenAIBaseModel):; symbols: FunctionDefinition, ChatCompletionToolsParam, ChatCompletionNamedFunction, ChatCompletionNamedToolChoiceParam
  - `vllm/entrypoints/openai/chat_completion/protocol.py` added +654/-0 (654 lines); hunks: -0,0 +1,654; symbols: ChatMessage, handle_deprecated_reasoning_content, ChatCompletionLogProb, ChatCompletionLogProbsContent
  - `vllm/entrypoints/serve/tokenize/protocol.py` added +139/-0 (139 lines); hunks: -0,0 +1,139; symbols: TokenizeCompletionRequest, TokenizeChatRequest, check_generation_prompt, TokenizeResponse
  - `vllm/entrypoints/openai/chat_completion/api_router.py` added +77/-0 (77 lines); hunks: -0,0 +1,77; symbols: chat, create_chat_completion, attach_router
  - `vllm/entrypoints/openai/api_server.py` modified +11/-48 (59 lines); hunks: -42,11 +42,9; -59,9 +57,9; symbols: translate_error_response, create_chat_completion, send_with_request_id, _extract_content_from_chunk
- 关键代码摘录:

```diff
diff -- vllm/entrypoints/openai/engine/protocol.py
@@ -11,10 +11,6 @@
-from openai.types.chat.chat_completion_audio import (
-    ChatCompletionAudio as OpenAIChatCompletionAudio,
-)
-from openai.types.chat.chat_completion_message import Annotation as OpenAIAnnotation
@@ -234,20 +230,6 @@ class FunctionDefinition(OpenAIBaseModel):
-class ChatCompletionToolsParam(OpenAIBaseModel):
diff -- vllm/entrypoints/openai/chat_completion/protocol.py
@@ -0,0 +1,654 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from
+# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
+import json
+import time
diff -- vllm/entrypoints/serve/tokenize/protocol.py
@@ -0,0 +1,139 @@
```

- 已读文件:
  - runtime: `vllm/entrypoints/openai/engine/protocol.py` renamed +89/-847; `vllm/entrypoints/openai/chat_completion/protocol.py` added +654/-0; `vllm/entrypoints/serve/tokenize/protocol.py` added +139/-0; `vllm/entrypoints/openai/chat_completion/api_router.py` added +77/-0; `vllm/entrypoints/openai/api_server.py` modified +11/-48; `vllm/entrypoints/openai/chat_completion/serving.py` renamed +16/-14
  - tests: `tests/entrypoints/openai/test_serving_chat.py` modified +13/-11
- 验证与风险: diff 自带测试面 `tests/entrypoints/openai/responses/test_errors.py`, `tests/entrypoints/openai/responses/test_function_call_parsing.py`, `tests/entrypoints/openai/test_chat_error.py`, `tests/entrypoints/openai/test_chat_template.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32321 - fix: avoid crash on zero-arg tool calls in glm4 parser

- 链接: https://github.com/vllm-project/vllm/pull/32321
- 状态/时间: merged / 2026-01-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-1，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: avoid crash on zero-arg tool calls in glm4 parser」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/glm4_moe_tool_parser.py`；技术摘要: 覆盖「fix: avoid crash on zero-arg tool calls in glm4 parser」；主要实现面是 `vllm/tool_parsers/glm4_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +7/-1 (8 lines); hunks: -115,9 +115,15 @@ def _deserialize(value: str) -> Any:; symbols: _deserialize，涉及 `_deserialize`。
- 代码 diff 细节:
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +7/-1 (8 lines); hunks: -115,9 +115,15 @@ def _deserialize(value: str) -> Any:; symbols: _deserialize
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/glm4_moe_tool_parser.py
@@ -115,9 +115,15 @@ def _deserialize(value: str) -> Any:
+                if not tc_detail:
+                    logger.warning(
+                        "Failed to parse tool call details from: %s",
+                        match,
+                    )
+                    continue
```

- 已读文件:
  - runtime: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +7/-1
- 验证与风险: runtime 路径改动集中在 `vllm/tool_parsers/glm4_moe_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31386 - [GLM-4.7] GLM Model support for GLM-Lite

- 链接: https://github.com/vllm-project/vllm/pull/31386
- 状态/时间: merged / 2026-01-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+1135/-1，可读 patch 1208 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[GLM-4.7] GLM Model support for GLM-Lite」；模型线: GLM-4.5；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `tests/models/registry.py`；技术摘要: 覆盖「[GLM-4.7] GLM Model support for GLM-Lite」；主要实现面是 `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `tests/models/registry.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe_lite.py` added +642/-0 (642 lines); hunks: -0,0 +1,642; symbols: Glm4MoeLiteMLP, Glm4MoeLite, Glm4LiteMixtureOfExperts, Glm4MoeLiteAttention，涉及 `Glm4MoeLiteMLP, Glm4MoeLite, Glm4LiteMixtureOfExperts`；`vllm/model_executor/models/glm4_moe_lite_mtp.py` added +464/-0 (464 lines); hunks: -0,0 +1,464; symbols: SharedHead, __init__, forward, Glm4MoeLiteMultiTokenPredictorLayer，涉及 `SharedHead, __init__, forward`；`tests/models/registry.py` modified +10/-0 (10 lines); hunks: -271,6 +271,11 @@ def check_available_online(; -1040,6 +1045,11 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-1 (3 lines); hunks: -21,7 +21,8。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe_lite.py` added +642/-0 (642 lines); hunks: -0,0 +1,642; symbols: Glm4MoeLiteMLP, Glm4MoeLite, Glm4LiteMixtureOfExperts, Glm4MoeLiteAttention
  - `vllm/model_executor/models/glm4_moe_lite_mtp.py` added +464/-0 (464 lines); hunks: -0,0 +1,464; symbols: SharedHead, __init__, forward, Glm4MoeLiteMultiTokenPredictorLayer
  - `tests/models/registry.py` modified +10/-0 (10 lines); hunks: -271,6 +271,11 @@ def check_available_online(; -1040,6 +1045,11 @@ def check_available_online(; symbols: check_available_online
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-1 (3 lines); hunks: -21,7 +21,8
  - `vllm/model_executor/models/registry.py` modified +2/-0 (2 lines); hunks: -112,6 +112,7; -465,6 +466,7
- 关键代码摘录:

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
diff -- tests/models/registry.py
@@ -271,6 +271,11 @@ def check_available_online(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe_lite.py` added +642/-0; `vllm/model_executor/models/glm4_moe_lite_mtp.py` added +464/-0; `vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-1; `vllm/model_executor/models/registry.py` modified +2/-0; `vllm/config/speculative.py` modified +12/-0
  - tests: `tests/models/registry.py` modified +10/-0
  - other: `benchmarks/kernels/benchmark_moe.py` modified +1/-0; `benchmarks/kernels/benchmark_moe_permute_unpermute.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33063 - [Chore] Update type annotation of `input_ids` in model forward

- 链接: https://github.com/vllm-project/vllm/pull/33063
- 状态/时间: merged / 2026-01-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 164 个文件，+243/-241，可读 patch 2158 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Chore] Update type annotation of `input_ids` in model forward」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py`；技术摘要: 覆盖「[Chore] Update type annotation of `input_ids` in model forward」；主要实现面是 `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/modernbert.py` modified +4/-5 (9 lines); hunks: -54,12 +54,11 @@ def forward(; symbols: forward, ModernBertAttention，涉及 `forward, ModernBertAttention`；`vllm/model_executor/models/gemma3n.py` modified +4/-4 (8 lines); hunks: -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torc...; -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: altup_embed, forward, embed_input_ids, fast_prefill_forward，涉及 `altup_embed, forward, embed_input_ids`；`vllm/model_executor/models/gpt2.py` modified +3/-3 (6 lines); hunks: -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -298,7 +298,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, load_weights，涉及 `embed_input_ids, forward, load_weights`；`vllm/model_executor/models/internlm2.py` modified +3/-3 (6 lines); hunks: -284,7 +284,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -350,7 +350,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, __init__，涉及 `embed_input_ids, forward, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/modernbert.py` modified +4/-5 (9 lines); hunks: -54,12 +54,11 @@ def forward(; symbols: forward, ModernBertAttention
  - `vllm/model_executor/models/gemma3n.py` modified +4/-4 (8 lines); hunks: -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torc...; -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: altup_embed, forward, embed_input_ids, fast_prefill_forward
  - `vllm/model_executor/models/gpt2.py` modified +3/-3 (6 lines); hunks: -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -298,7 +298,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, load_weights
  - `vllm/model_executor/models/internlm2.py` modified +3/-3 (6 lines); hunks: -284,7 +284,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -350,7 +350,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, __init__
  - `vllm/model_executor/models/opt.py` modified +3/-3 (6 lines); hunks: -267,7 +267,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -316,7 +316,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/modernbert.py
@@ -54,12 +54,11 @@ def forward(
-        if inputs_embeds is not None:
-            return self.norm(inputs_embeds)
-        else:
+        if inputs_embeds is None:
-            embeddings = self.norm(inputs_embeds)
-            return embeddings
diff -- vllm/model_executor/models/gemma3n.py
@@ -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torch.Tensor:
-        input_ids: torch.Tensor,
+        input_ids: torch.Tensor | None,
@@ -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
-        input_ids: torch.Tensor,
+        input_ids: torch.Tensor | None,
@@ -964,7 +964,7 @@ def fast_prefill_forward(
diff -- vllm/model_executor/models/gpt2.py
@@ -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/modernbert.py` modified +4/-5; `vllm/model_executor/models/gemma3n.py` modified +4/-4; `vllm/model_executor/models/gpt2.py` modified +3/-3; `vllm/model_executor/models/internlm2.py` modified +3/-3; `vllm/model_executor/models/opt.py` modified +3/-3; `vllm/model_executor/models/afmoe.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32064 - [5/N][Attention] Finish eliminating `vllm/attention` folder

- 链接: https://github.com/vllm-project/vllm/pull/32064
- 状态/时间: merged / 2026-01-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 151 个文件，+585/-527，可读 patch 2850 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[5/N][Attention] Finish eliminating `vllm/attention` folder」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/__init__.py`；技术摘要: 覆盖「[5/N][Attention] Finish eliminating `vllm/attention` folder」；主要实现面是 `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/__init__.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/attention/mla_attention.py` modified +354/-17 (371 lines); hunks: -191,24 +191,38; -217,11 +231,16; symbols: MLAAttention, takes, does, __init__，涉及 `MLAAttention, takes, does`；`vllm/model_executor/layers/attention/attention.py` renamed +42/-315 (357 lines); hunks: -1,23 +1,22; -33,20 +32,54; symbols: validate_kv_sharing_target, should_load_quant_weights, get_kv_cache_spec, MLAAttention，涉及 `validate_kv_sharing_target, should_load_quant_weights, get_kv_cache_spec`；`vllm/model_executor/layers/attention/__init__.py` modified +26/-0 (26 lines); hunks: -0,0 +1,26；`vllm/model_executor/models/whisper.py` modified +5/-3 (8 lines); hunks: -17,16 +17,18。
- 代码 diff 细节:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +354/-17 (371 lines); hunks: -191,24 +191,38; -217,11 +231,16; symbols: MLAAttention, takes, does, __init__
  - `vllm/model_executor/layers/attention/attention.py` renamed +42/-315 (357 lines); hunks: -1,23 +1,22; -33,20 +32,54; symbols: validate_kv_sharing_target, should_load_quant_weights, get_kv_cache_spec, MLAAttention
  - `vllm/model_executor/layers/attention/__init__.py` modified +26/-0 (26 lines); hunks: -0,0 +1,26
  - `vllm/model_executor/models/whisper.py` modified +5/-3 (8 lines); hunks: -17,16 +17,18
  - `vllm/model_executor/models/openpangu.py` modified +3/-2 (5 lines); hunks: -29,7 +29,6; -41,7 +40,8
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -191,24 +191,38 @@
-from typing import ClassVar, Generic, TypeVar
+from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast
+if TYPE_CHECKING:
+    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
+import torch.nn as nn
+import vllm.envs as envs
diff -- vllm/model_executor/layers/attention/attention.py
@@ -1,23 +1,22 @@
-"""Attention layer."""
-from typing import cast
+from typing import TYPE_CHECKING
-from vllm.attention.utils.kv_sharing_utils import validate_kv_sharing_target
-from vllm.attention.utils.kv_transfer_utils import maybe_transfer_kv_layer
+from vllm.model_executor.layers.attention.kv_transfer_utils import (
diff -- vllm/model_executor/layers/attention/__init__.py
@@ -0,0 +1,26 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/attention/mla_attention.py` modified +354/-17; `vllm/model_executor/layers/attention/attention.py` renamed +42/-315; `vllm/model_executor/layers/attention/__init__.py` modified +26/-0; `vllm/model_executor/models/whisper.py` modified +5/-3; `vllm/model_executor/models/openpangu.py` modified +3/-2; `vllm/model_executor/models/apertus.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/compile/test_fusion_attn.py`, `tests/compile/test_qk_norm_rope_fusion.py`, `tests/kernels/attention/test_attention.py`, `tests/kernels/attention/test_mha_attn.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33218 - [Bugfix] GLM-4 tool parser: incremental string streaming

- 链接: https://github.com/vllm-project/vllm/pull/33218
- 状态/时间: merged / 2026-02-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+725/-96，可读 patch 903 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] GLM-4 tool parser: incremental string streaming」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`；技术摘要: 覆盖「[Bugfix] GLM-4 tool parser: incremental string streaming」；主要实现面是 `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +389/-96 (485 lines); hunks: -1,5 +1,15; -8,6 +18,7; symbols: Glm4MoeModelToolParser, __init__, adjust_request，涉及 `Glm4MoeModelToolParser, __init__, adjust_request`；`tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +336/-0 (336 lines); hunks: -6,6 +6,7; -447,3 +448,338 @@ def test_extract_tool_calls_incomplete_tool_call(glm4_moe_...; symbols: test_extract_tool_calls_incomplete_tool_call, _reset_streaming_state, test_streaming_incremental_string_value, test_streaming_empty_tool_call，涉及 `test_extract_tool_calls_incomplete_tool_call, _reset_streaming_state, test_streaming_incremental_string_value`。
- 代码 diff 细节:
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +389/-96 (485 lines); hunks: -1,5 +1,15; -8,6 +18,7; symbols: Glm4MoeModelToolParser, __init__, adjust_request
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +336/-0 (336 lines); hunks: -6,6 +6,7; -447,3 +448,338 @@ def test_extract_tool_calls_incomplete_tool_call(glm4_moe_...; symbols: test_extract_tool_calls_incomplete_tool_call, _reset_streaming_state, test_streaming_incremental_string_value, test_streaming_empty_tool_call
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/glm4_moe_tool_parser.py
@@ -1,5 +1,15 @@
+"""
+GLM-4 Tool Call Parser with incremental string streaming support.
+This parser fixes the streaming issue reported in Issue #32829 where long string
+parameters (e.g., file content with 4000+ characters of code) are buffered until
+complete, causing multi-second delays before the user sees any content.
+The fix streams string values incrementally as they arrive, providing a true
diff -- tests/tool_parsers/test_glm4_moe_tool_parser.py
@@ -6,6 +6,7 @@
+from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
@@ -447,3 +448,338 @@ def test_extract_tool_calls_incomplete_tool_call(glm4_moe_tool_parser):
+def _reset_streaming_state(parser):
+    """Helper to reset parser streaming state."""
+    parser._buffer = ""
+    parser._in_tool_call = False
```

- 已读文件:
  - runtime: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +389/-96
  - tests: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +336/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_glm4_moe_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33525 - Update get_expert_mapping to include self parameter

- 链接: https://github.com/vllm-project/vllm/pull/33525
- 状态/时间: merged / 2026-02-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update get_expert_mapping to include self parameter」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/glm4_moe_lite.py`；技术摘要: 覆盖「Update get_expert_mapping to include self parameter」；主要实现面是 `vllm/model_executor/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe_lite.py` modified +1/-0 (1 lines); hunks: -617,6 +617,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: get_expert_mapping，涉及 `get_expert_mapping`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +1/-0 (1 lines); hunks: -617,6 +617,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: get_expert_mapping
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe_lite.py
@@ -617,6 +617,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
+            self,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe_lite.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34905 - Fix GLM4 parser tests

- 链接: https://github.com/vllm-project/vllm/pull/34905
- 状态/时间: merged / 2026-02-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+67/-45，可读 patch 402 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GLM4 parser tests」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`；技术摘要: 覆盖「Fix GLM4 parser tests」；主要实现面是 `tests/tool_parsers/test_glm4_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +67/-45 (112 lines); hunks: -1,19 +1,22; -28,6 +31,20 @@ def glm4_moe_tool_parser(glm4_moe_tokenizer):; symbols: glm4_moe_tool_parser, mock_request, assert_tool_calls, test_extract_tool_calls_no_tools，涉及 `glm4_moe_tool_parser, mock_request, assert_tool_calls`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +67/-45 (112 lines); hunks: -1,19 +1,22; -28,6 +31,20 @@ def glm4_moe_tool_parser(glm4_moe_tokenizer):; symbols: glm4_moe_tool_parser, mock_request, assert_tool_calls, test_extract_tool_calls_no_tools
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_glm4_moe_tool_parser.py
@@ -1,19 +1,22 @@
-# ruff: noqa: E501
+from unittest.mock import Mock
-from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionRequest,
+    ChatCompletionToolsParam,
```

- 已读文件:
  - tests: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +67/-45
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_glm4_moe_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35640 - [MISC] fixed tool_parser mypy errors

- 链接: https://github.com/vllm-project/vllm/pull/35640
- 状态/时间: merged / 2026-03-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+9/-15，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MISC] fixed tool_parser mypy errors」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/glm4_moe_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`, `vllm/tool_parsers/functiongemma_tool_parser.py`；技术摘要: 覆盖「[MISC] fixed tool_parser mypy errors」；主要实现面是 `vllm/tool_parsers/glm4_moe_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`, `vllm/tool_parsers/functiongemma_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +7/-8 (15 lines); hunks: -355,20 +355,17 @@ def extract_tool_calls_streaming(; -387,7 +384,9 @@ def extract_tool_calls_streaming(; symbols: extract_tool_calls_streaming，涉及 `extract_tool_calls_streaming`；`vllm/tool_parsers/step3p5_tool_parser.py` modified +1/-5 (6 lines); hunks: -23,10 +23,7; -1367,7 +1364,6 @@ def _reset_xml_parser_after_tool_call(self):; symbols: _reset_xml_parser_after_tool_call, Step3p5ToolParser, __init__，涉及 `_reset_xml_parser_after_tool_call, Step3p5ToolParser, __init__`；`vllm/tool_parsers/functiongemma_tool_parser.py` modified +1/-1 (2 lines); hunks: -72,7 +72,7 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, _parse_arguments，涉及 `__init__, _parse_arguments`；`tools/pre_commit/mypy.py` modified +0/-1 (1 lines); hunks: -42,7 +42,6。
- 代码 diff 细节:
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +7/-8 (15 lines); hunks: -355,20 +355,17 @@ def extract_tool_calls_streaming(; -387,7 +384,9 @@ def extract_tool_calls_streaming(; symbols: extract_tool_calls_streaming
  - `vllm/tool_parsers/step3p5_tool_parser.py` modified +1/-5 (6 lines); hunks: -23,10 +23,7; -1367,7 +1364,6 @@ def _reset_xml_parser_after_tool_call(self):; symbols: _reset_xml_parser_after_tool_call, Step3p5ToolParser, __init__
  - `vllm/tool_parsers/functiongemma_tool_parser.py` modified +1/-1 (2 lines); hunks: -72,7 +72,7 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, _parse_arguments
  - `tools/pre_commit/mypy.py` modified +0/-1 (1 lines); hunks: -42,7 +42,6
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/glm4_moe_tool_parser.py
@@ -355,20 +355,17 @@ def extract_tool_calls_streaming(
-                    frag = self._append_arg_fragment(
-                        key=key,
-                        raw_val=raw_val,
-                    )
-                    if frag:
-                        return self._emit_tool_args_delta(frag)
diff -- vllm/tool_parsers/step3p5_tool_parser.py
@@ -23,10 +23,7 @@
-from vllm.tool_parsers.abstract_tool_parser import (
-    ToolParser,
-    ToolParserManager,
-)
+from vllm.tool_parsers.abstract_tool_parser import ToolParser
@@ -1367,7 +1364,6 @@ def _reset_xml_parser_after_tool_call(self):
diff -- vllm/tool_parsers/functiongemma_tool_parser.py
@@ -72,7 +72,7 @@ def __init__(self, tokenizer: TokenizerLike):
```

- 已读文件:
  - runtime: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +7/-8; `vllm/tool_parsers/step3p5_tool_parser.py` modified +1/-5; `vllm/tool_parsers/functiongemma_tool_parser.py` modified +1/-1
  - other: `tools/pre_commit/mypy.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `vllm/tool_parsers/functiongemma_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35208 - GLM4 tool parser: fix streaming mode

- 链接: https://github.com/vllm-project/vllm/pull/35208
- 状态/时间: merged / 2026-03-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+26/-10，可读 patch 89 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「GLM4 tool parser: fix streaming mode」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`；技术摘要: 覆盖「GLM4 tool parser: fix streaming mode」；主要实现面是 `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +18/-6 (24 lines); hunks: -560,19 +560,23 @@ def test_streaming_empty_tool_call(glm4_moe_tool_parser, m...; -582,6 +586,8 @@ def test_streaming_prev_tool_call_arr_finalization(glm4_moe_...; symbols: test_streaming_empty_tool_call, test_streaming_prev_tool_call_arr_finalization, test_streaming_prev_tool_call_arr_updates，涉及 `test_streaming_empty_tool_call, test_streaming_prev_tool_call_arr_finalization, test_streaming_prev_tool_call_arr_updates`；`vllm/tool_parsers/glm4_moe_tool_parser.py` modified +8/-4 (12 lines); hunks: -337,10 +337,10 @@ def extract_tool_calls_streaming(; -447,6 +447,10 @@ def _revert_last_tool_call_state(self) -> None:; symbols: extract_tool_calls_streaming, _revert_last_tool_call_state, _emit_tool_name_delta, _append_arg_fragment，涉及 `extract_tool_calls_streaming, _revert_last_tool_call_state, _emit_tool_name_delta`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +18/-6 (24 lines); hunks: -560,19 +560,23 @@ def test_streaming_empty_tool_call(glm4_moe_tool_parser, m...; -582,6 +586,8 @@ def test_streaming_prev_tool_call_arr_finalization(glm4_moe_...; symbols: test_streaming_empty_tool_call, test_streaming_prev_tool_call_arr_finalization, test_streaming_prev_tool_call_arr_updates
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +8/-4 (12 lines); hunks: -337,10 +337,10 @@ def extract_tool_calls_streaming(; -447,6 +447,10 @@ def _revert_last_tool_call_state(self) -> None:; symbols: extract_tool_calls_streaming, _revert_last_tool_call_state, _emit_tool_name_delta, _append_arg_fragment
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_glm4_moe_tool_parser.py
@@ -560,19 +560,23 @@ def test_streaming_empty_tool_call(glm4_moe_tool_parser, mock_request):
-def test_streaming_prev_tool_call_arr_finalization(glm4_moe_tool_parser, mock_request):
+def test_streaming_prev_tool_call_arr_updates(glm4_moe_tool_parser, mock_request):
+    name_only = {"name": "get_weather", "arguments": {}}
+    name_and_args = {"name": "get_weather", "arguments": {"city": "Beijing"}}
-        "<tool_call>get_weather\n",
-        "<arg_key>city</arg_key>",
diff -- vllm/tool_parsers/glm4_moe_tool_parser.py
@@ -337,10 +337,10 @@ def extract_tool_calls_streaming(
-                        frag = "{" + key_json + ':"'
+                        frag = "{" + key_json + ': "'
-                        frag = "," + key_json + ':"'
+                        frag = ", " + key_json + ': "'
@@ -447,6 +447,10 @@ def _revert_last_tool_call_state(self) -> None:
+        self.prev_tool_call_arr[self.current_tool_id] = {
```

- 已读文件:
  - tests: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +18/-6
  - runtime: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +8/-4
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_glm4_moe_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37386 - fix(glm47): improve tool call parsing and content normalization

- 链接: https://github.com/vllm-project/vllm/pull/37386
- 状态/时间: merged / 2026-03-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+193/-6，可读 patch 244 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(glm47): improve tool call parsing and content normalization」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`；技术摘要: 覆盖「fix(glm47): improve tool call parsing and content normalization」；主要实现面是 `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_glm47_moe_tool_parser.py` added +168/-0 (168 lines); hunks: -0,0 +1,168; symbols: glm47_tokenizer, glm47_tool_parser, mock_request, TestGlm47ExtractToolCalls，涉及 `glm47_tokenizer, glm47_tool_parser, mock_request`；`vllm/tool_parsers/glm47_moe_tool_parser.py` modified +16/-2 (18 lines); hunks: -1,6 +1,16; -14,10 +24,14; symbols: Glm47MoeModelToolParser, __init__，涉及 `Glm47MoeModelToolParser, __init__`；`vllm/tool_parsers/glm4_moe_tool_parser.py` modified +6/-1 (7 lines); hunks: -206,7 +206,12 @@ def extract_tool_calls(; symbols: extract_tool_calls，涉及 `extract_tool_calls`；`tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +3/-3 (6 lines); hunks: -107,7 +107,7 @@ def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, m...; -152,7 +152,7 @@ def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, m...; symbols: test_extract_tool_calls_no_tools，涉及 `test_extract_tool_calls_no_tools`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_glm47_moe_tool_parser.py` added +168/-0 (168 lines); hunks: -0,0 +1,168; symbols: glm47_tokenizer, glm47_tool_parser, mock_request, TestGlm47ExtractToolCalls
  - `vllm/tool_parsers/glm47_moe_tool_parser.py` modified +16/-2 (18 lines); hunks: -1,6 +1,16; -14,10 +24,14; symbols: Glm47MoeModelToolParser, __init__
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +6/-1 (7 lines); hunks: -206,7 +206,12 @@ def extract_tool_calls(; symbols: extract_tool_calls
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +3/-3 (6 lines); hunks: -107,7 +107,7 @@ def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, m...; -152,7 +152,7 @@ def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, m...; symbols: test_extract_tool_calls_no_tools
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/tool_parsers/test_glm47_moe_tool_parser.py` added +168/-0; `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +3/-3
  - runtime: `vllm/tool_parsers/glm47_moe_tool_parser.py` modified +16/-2; `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +6/-1
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38029 - [Tool Parser][1/3] Pass tools to ToolParser constructor

- 链接: https://github.com/vllm-project/vllm/pull/38029
- 状态/时间: merged / 2026-03-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 38 个文件，+147/-92，可读 patch 858 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Tool Parser][1/3] Pass tools to ToolParser constructor」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`；技术摘要: 覆盖「[Tool Parser][1/3] Pass tools to ToolParser constructor」；主要实现面是 `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2 (16 lines); hunks: -5,13 +5,18; -30,6 +35,8; symbols: ToolParser, __init__, vocab，涉及 `ToolParser, __init__, vocab`；`vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7 (12 lines); hunks: -10,7 +10,6; -23,15 +22,16; symbols: Qwen3CoderToolParser, __init__, _reset_streaming_state, _get_arguments_config，涉及 `Qwen3CoderToolParser, __init__, _reset_streaming_state`；`vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6 (11 lines); hunks: -11,7 +11,6; -23,7 +22,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call，涉及 `__init__, setup_parser, set_tools`；`vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5 (10 lines); hunks: -11,7 +11,6; -24,6 +23,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call，涉及 `__init__, setup_parser, set_tools`。
- 代码 diff 细节:
  - `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2 (16 lines); hunks: -5,13 +5,18; -30,6 +35,8; symbols: ToolParser, __init__, vocab
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7 (12 lines); hunks: -10,7 +10,6; -23,15 +22,16; symbols: Qwen3CoderToolParser, __init__, _reset_streaming_state, _get_arguments_config
  - `vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6 (11 lines); hunks: -11,7 +11,6; -23,7 +22,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5 (10 lines); hunks: -11,7 +11,6; -24,6 +23,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call
  - `vllm/tool_parsers/llama4_pythonic_tool_parser.py` modified +7/-2 (9 lines); hunks: -17,6 +17,7; -47,8 +48,12 @@ class Llama4PythonicToolParser(ToolParser):; symbols: Llama4PythonicToolParser, __init__
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/abstract_tool_parser.py
@@ -5,13 +5,18 @@
+from typing import TypeAlias
+from openai.types.responses.tool import Tool as ResponsesTool
-from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionRequest,
+    ChatCompletionToolsParam,
diff -- vllm/tool_parsers/qwen3coder_tool_parser.py
@@ -10,7 +10,6 @@
-    ChatCompletionToolsParam,
@@ -23,15 +22,16 @@
+    Tool,
-    def __init__(self, tokenizer: TokenizerLike):
-        super().__init__(tokenizer)
+    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
diff -- vllm/tool_parsers/step3p5_tool_parser.py
@@ -11,7 +11,6 @@
```

- 已读文件:
  - runtime: `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2; `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7; `vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5; `vllm/tool_parsers/llama4_pythonic_tool_parser.py` modified +7/-2; `vllm/tool_parsers/llama_tool_parser.py` modified +7/-2
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/entrypoints/openai/parser/responses_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38264 - [Mypy] Fix adjust_request typing

- 链接: https://github.com/vllm-project/vllm/pull/38264
- 状态/时间: merged / 2026-03-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+49/-17，可读 patch 241 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Mypy] Fix adjust_request typing」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/deepseekv32_tool_parser.py`, `vllm/tool_parsers/functiongemma_tool_parser.py`, `vllm/tool_parsers/gigachat3_tool_parser.py`；技术摘要: 覆盖「[Mypy] Fix adjust_request typing」；主要实现面是 `vllm/tool_parsers/deepseekv32_tool_parser.py`, `vllm/tool_parsers/functiongemma_tool_parser.py`, `vllm/tool_parsers/gigachat3_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +4/-1 (5 lines); hunks: -19,6 +19,7; -78,7 +79,9 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, adjust_request，涉及 `__init__, adjust_request`；`vllm/tool_parsers/functiongemma_tool_parser.py` modified +4/-1 (5 lines); hunks: -18,6 +18,7; -86,7 +87,9 @@ def _parse_arguments(self, args_str: str) -> dict:; symbols: _parse_arguments, adjust_request，涉及 `_parse_arguments, adjust_request`；`vllm/tool_parsers/gigachat3_tool_parser.py` modified +4/-1 (5 lines); hunks: -18,6 +18,7; -55,7 +56,9 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, adjust_request，涉及 `__init__, adjust_request`；`vllm/tool_parsers/glm4_moe_tool_parser.py` modified +4/-1 (5 lines); hunks: -30,6 +30,7; -151,7 +152,9 @@ def _tools_enabled(request: ChatCompletionRequest) -> bool:; symbols: _tools_enabled, adjust_request，涉及 `_tools_enabled, adjust_request`。
- 代码 diff 细节:
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +4/-1 (5 lines); hunks: -19,6 +19,7; -78,7 +79,9 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, adjust_request
  - `vllm/tool_parsers/functiongemma_tool_parser.py` modified +4/-1 (5 lines); hunks: -18,6 +18,7; -86,7 +87,9 @@ def _parse_arguments(self, args_str: str) -> dict:; symbols: _parse_arguments, adjust_request
  - `vllm/tool_parsers/gigachat3_tool_parser.py` modified +4/-1 (5 lines); hunks: -18,6 +18,7; -55,7 +56,9 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, adjust_request
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +4/-1 (5 lines); hunks: -30,6 +30,7; -151,7 +152,9 @@ def _tools_enabled(request: ChatCompletionRequest) -> bool:; symbols: _tools_enabled, adjust_request
  - `vllm/tool_parsers/granite4_tool_parser.py` modified +4/-1 (5 lines); hunks: -19,6 +19,7; -59,7 +60,9 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, adjust_request
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -19,6 +19,7 @@
+from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
@@ -78,7 +79,9 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
-    def adjust_request(self, request):
+    def adjust_request(
+        self, request: ChatCompletionRequest | ResponsesRequest
+    ) -> ChatCompletionRequest | ResponsesRequest:
diff -- vllm/tool_parsers/functiongemma_tool_parser.py
@@ -18,6 +18,7 @@
+from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
@@ -86,7 +87,9 @@ def _parse_arguments(self, args_str: str) -> dict:
-    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
+    def adjust_request(
+        self, request: ChatCompletionRequest | ResponsesRequest
+    ) -> ChatCompletionRequest | ResponsesRequest:
diff -- vllm/tool_parsers/gigachat3_tool_parser.py
@@ -18,6 +18,7 @@
```

- 已读文件:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +4/-1; `vllm/tool_parsers/functiongemma_tool_parser.py` modified +4/-1; `vllm/tool_parsers/gigachat3_tool_parser.py` modified +4/-1; `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +4/-1; `vllm/tool_parsers/granite4_tool_parser.py` modified +4/-1; `vllm/tool_parsers/hermes_tool_parser.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/serve/render/serving.py`, `vllm/parser/abstract_parser.py`, `vllm/tool_parsers/abstract_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38189 - [Tool Parser][2/3] Use self.tools instead of request.tools in tool parsers

- 链接: https://github.com/vllm-project/vllm/pull/38189
- 状态/时间: merged / 2026-03-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+113/-105，可读 patch 532 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Tool Parser][2/3] Use self.tools instead of request.tools in tool parsers」；模型线: GLM-4.5；类别: 文档/测试/CI；主要 diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_qwen3coder_tool_parser.py`；技术摘要: 覆盖「[Tool Parser][2/3] Use self.tools instead of request.tools in tool parsers」；主要实现面是 `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_qwen3coder_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +32/-27 (59 lines); hunks: -27,21 +27,26 @@ def glm4_moe_tokenizer():; -671,14 +676,13 @@ def test_streaming_json_escape_in_string(glm4_moe_tool_par...; symbols: glm4_moe_tokenizer, glm4_moe_tool_parser, mock_request, sample_tools，涉及 `glm4_moe_tokenizer, glm4_moe_tool_parser, mock_request`；`tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +19/-13 (32 lines); hunks: -11,6 +11,10; -24,8 +28,8; symbols: make_parser, make_tool_param, test_content_before_tool_call_streaming, test_type_conversion_in_streaming，涉及 `make_parser, make_tool_param, test_content_before_tool_call_streaming`；`tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +10/-12 (22 lines); hunks: -31,13 +31,13 @@ def qwen3_tokenizer():; -376,7 +376,7 @@ def test_extract_tool_calls_fallback_no_tags(; symbols: qwen3_tokenizer, qwen3_tool_parser, qwen3_xml_tool_parser，涉及 `qwen3_tokenizer, qwen3_tool_parser, qwen3_xml_tool_parser`；`tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +13/-8 (21 lines); hunks: -25,14 +25,8 @@ def glm47_tokenizer():; -49,6 +43,17 @@ def mock_request() -> ChatCompletionRequest:; symbols: glm47_tokenizer, glm47_tool_parser, mock_request, sample_tools，涉及 `glm47_tokenizer, glm47_tool_parser, mock_request`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +32/-27 (59 lines); hunks: -27,21 +27,26 @@ def glm4_moe_tokenizer():; -671,14 +676,13 @@ def test_streaming_json_escape_in_string(glm4_moe_tool_par...; symbols: glm4_moe_tokenizer, glm4_moe_tool_parser, mock_request, sample_tools
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +19/-13 (32 lines); hunks: -11,6 +11,10; -24,8 +28,8; symbols: make_parser, make_tool_param, test_content_before_tool_call_streaming, test_type_conversion_in_streaming
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +10/-12 (22 lines); hunks: -31,13 +31,13 @@ def qwen3_tokenizer():; -376,7 +376,7 @@ def test_extract_tool_calls_fallback_no_tags(; symbols: qwen3_tokenizer, qwen3_tool_parser, qwen3_xml_tool_parser
  - `tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +13/-8 (21 lines); hunks: -25,14 +25,8 @@ def glm47_tokenizer():; -49,6 +43,17 @@ def mock_request() -> ChatCompletionRequest:; symbols: glm47_tokenizer, glm47_tool_parser, mock_request, sample_tools
  - `tests/tool_parsers/test_step3p5_tool_parser.py` modified +8/-10 (18 lines); hunks: -28,8 +28,8 @@ def step3p5_tokenizer():; -386,7 +386,7 @@ def test_extract_tool_calls_fallback_no_tags(step3p5_tool_pa...; symbols: step3p5_tokenizer, step3p5_tool_parser, test_extract_tool_calls_fallback_no_tags, test_extract_tool_calls_type_conversion
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_glm4_moe_tool_parser.py
@@ -27,21 +27,26 @@ def glm4_moe_tokenizer():
-def glm4_moe_tool_parser(glm4_moe_tokenizer):
-    return Glm4MoeModelToolParser(glm4_moe_tokenizer)
-@pytest.fixture
-def mock_request() -> ChatCompletionRequest:
-    request = Mock(spec=ChatCompletionRequest)
-    request.tools = [  # GLM45 parser needs this attribute to enable tool parsing.
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -11,6 +11,10 @@
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionToolsParam,
+    FunctionDefinition,
+)
@@ -24,8 +28,8 @@
-def make_parser() -> DeepSeekV32ToolParser:
diff -- tests/tool_parsers/test_qwen3coder_tool_parser.py
@@ -31,13 +31,13 @@ def qwen3_tokenizer():
```

- 已读文件:
  - tests: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +32/-27; `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +19/-13; `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +10/-12; `tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +13/-8; `tests/tool_parsers/test_step3p5_tool_parser.py` modified +8/-10
  - runtime: `vllm/tool_parsers/abstract_tool_parser.py` modified +10/-1; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +3/-6; `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +3/-5
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_qwen3coder_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38172 - [Misc] Add 20 regression tests for 11 tool parser bug fixes

- 链接: https://github.com/vllm-project/vllm/pull/38172
- 状态/时间: merged / 2026-04-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+700/-0，可读 patch 755 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Add 20 regression tests for 11 tool parser bug fixes」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `tests/tool_parsers/test_step3p5_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`；技术摘要: 覆盖「[Misc] Add 20 regression tests for 11 tool parser bug fixes」；主要实现面是 `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `tests/tool_parsers/test_step3p5_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +154/-0 (154 lines); hunks: -974,3 +974,157 @@ def test_extract_tool_calls_streaming_missing_opening_tag(; symbols: test_extract_tool_calls_streaming_missing_opening_tag, test_malformed_xml_no_gt_delimiter, test_none_tool_calls_filtered, test_anyof_parameter_not_double_encoded，涉及 `test_extract_tool_calls_streaming_missing_opening_tag, test_malformed_xml_no_gt_delimiter, test_none_tool_calls_filtered`；`tests/tool_parsers/test_step3p5_tool_parser.py` modified +137/-0 (137 lines); hunks: -1431,3 +1431,140 @@ def test_extract_tool_calls_non_streaming_multiple_tool_...; symbols: test_extract_tool_calls_non_streaming_multiple_tool_calls_no_content_between, _accumulate_tool_states, test_streaming_mtp_variable_chunks, test_streaming_multi_token_per_step，涉及 `test_extract_tool_calls_non_streaming_multiple_tool_calls_no_content_between, _accumulate_tool_states, test_streaming_mtp_variable_chunks`；`tests/tool_parsers/test_minimax_m2_tool_parser.py` modified +106/-0 (106 lines); hunks: -5,6 +5,10; -442,3 +446,105 @@ def test_header_and_params_in_separate_chunks(self, parser):; symbols: test_header_and_params_in_separate_chunks, TestAnyOfNullableParam, test_anyof_nullable_param_non_null_value, test_anyof_nullable_param_null_value，涉及 `test_header_and_params_in_separate_chunks, TestAnyOfNullableParam, test_anyof_nullable_param_non_null_value`；`tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +105/-0 (105 lines); hunks: -822,3 +822,108 @@ def test_extract_tool_calls_numeric_deserialization(glm4_m...; symbols: test_extract_tool_calls_numeric_deserialization, test_zero_argument_tool_call, test_malformed_tool_call_no_regex_match, test_delimiter_preserved_transformers_5x，涉及 `test_extract_tool_calls_numeric_deserialization, test_zero_argument_tool_call, test_malformed_tool_call_no_regex_match`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +154/-0 (154 lines); hunks: -974,3 +974,157 @@ def test_extract_tool_calls_streaming_missing_opening_tag(; symbols: test_extract_tool_calls_streaming_missing_opening_tag, test_malformed_xml_no_gt_delimiter, test_none_tool_calls_filtered, test_anyof_parameter_not_double_encoded
  - `tests/tool_parsers/test_step3p5_tool_parser.py` modified +137/-0 (137 lines); hunks: -1431,3 +1431,140 @@ def test_extract_tool_calls_non_streaming_multiple_tool_...; symbols: test_extract_tool_calls_non_streaming_multiple_tool_calls_no_content_between, _accumulate_tool_states, test_streaming_mtp_variable_chunks, test_streaming_multi_token_per_step
  - `tests/tool_parsers/test_minimax_m2_tool_parser.py` modified +106/-0 (106 lines); hunks: -5,6 +5,10; -442,3 +446,105 @@ def test_header_and_params_in_separate_chunks(self, parser):; symbols: test_header_and_params_in_separate_chunks, TestAnyOfNullableParam, test_anyof_nullable_param_non_null_value, test_anyof_nullable_param_null_value
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +105/-0 (105 lines); hunks: -822,3 +822,108 @@ def test_extract_tool_calls_numeric_deserialization(glm4_m...; symbols: test_extract_tool_calls_numeric_deserialization, test_zero_argument_tool_call, test_malformed_tool_call_no_regex_match, test_delimiter_preserved_transformers_5x
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +84/-0 (84 lines); hunks: -11,6 +11,7; -26,6 +27,7; symbols: make_parser, test_no_emission_while_incomplete, TestDelimiterPreservation, parser
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_qwen3coder_tool_parser.py
@@ -974,3 +974,157 @@ def test_extract_tool_calls_streaming_missing_opening_tag(
+def test_malformed_xml_no_gt_delimiter(qwen3_tool_parser, sample_tools):
+    """Regression: malformed XML without '>' must not crash (PR #36774)."""
+    model_output = (
+        "<tool_call>\n"
+        "<function=get_current_weather\n"
+        "<parameter=city>Dallas</parameter>\n"
diff -- tests/tool_parsers/test_step3p5_tool_parser.py
@@ -1431,3 +1431,140 @@ def test_extract_tool_calls_non_streaming_multiple_tool_calls_no_content_between
+def _accumulate_tool_states(delta_messages):
+    """Accumulate tool call state from a stream of DeltaMessage objects."""
+    content = ""
+    tool_states = {}
+    for delta_message in delta_messages:
+        if delta_message.content:
diff -- tests/tool_parsers/test_minimax_m2_tool_parser.py
@@ -5,6 +5,10 @@
```

- 已读文件:
  - tests: `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +154/-0; `tests/tool_parsers/test_step3p5_tool_parser.py` modified +137/-0; `tests/tool_parsers/test_minimax_m2_tool_parser.py` modified +106/-0; `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +105/-0; `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +84/-0; `tests/tool_parsers/test_mistral_tool_parser.py` modified +61/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39253 - [Bugfix] Fix GLM tool parser streaming with MTP or stream interval

- 链接: https://github.com/vllm-project/vllm/pull/39253
- 状态/时间: merged / 2026-04-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+788/-416，可读 patch 1480 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix GLM tool parser streaming with MTP or stream interval」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Fix GLM tool parser streaming with MTP or stream interval」；主要实现面是 `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +509/-103 (612 lines); hunks: -357,81 +357,69 @@ def test_extract_tool_calls_mixed_content(glm4_moe_tool_pa...; -479,26 +467,19 @@ def test_extract_tool_calls_incomplete_tool_call(glm4_moe_...; symbols: test_extract_tool_calls_mixed_content, test_streaming_basic_functionality, test_streaming_no_tool_calls, test_streaming_with_content_before_tool_calls，涉及 `test_extract_tool_calls_mixed_content, test_streaming_basic_functionality, test_streaming_no_tool_calls`；`vllm/tool_parsers/glm4_moe_tool_parser.py` modified +252/-296 (548 lines); hunks: -37,16 +37,17; -82,17 +83,17 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Too...; symbols: Glm4MoeModelToolParser, __init__, _deserialize, extract_tool_calls，涉及 `Glm4MoeModelToolParser, __init__, _deserialize`；`tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +14/-17 (31 lines); hunks: -117,28 +117,24 @@ def test_whitespace_content_none(self, glm47_tool_parser,...; -149,25 +145,26 @@ def test_no_args(self, glm47_tool_parser, mock_request):; symbols: test_whitespace_content_none, _reset, TestGlm47Streaming, test_no_args，涉及 `test_whitespace_content_none, _reset, TestGlm47Streaming`；`vllm/tool_parsers/utils.py` modified +13/-0 (13 lines); hunks: -31,6 +31,19; symbols: partial_tag_overlap, find_common_prefix，涉及 `partial_tag_overlap, find_common_prefix`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +509/-103 (612 lines); hunks: -357,81 +357,69 @@ def test_extract_tool_calls_mixed_content(glm4_moe_tool_pa...; -479,26 +467,19 @@ def test_extract_tool_calls_incomplete_tool_call(glm4_moe_...; symbols: test_extract_tool_calls_mixed_content, test_streaming_basic_functionality, test_streaming_no_tool_calls, test_streaming_with_content_before_tool_calls
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +252/-296 (548 lines); hunks: -37,16 +37,17; -82,17 +83,17 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Too...; symbols: Glm4MoeModelToolParser, __init__, _deserialize, extract_tool_calls
  - `tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +14/-17 (31 lines); hunks: -117,28 +117,24 @@ def test_whitespace_content_none(self, glm47_tool_parser,...; -149,25 +145,26 @@ def test_no_args(self, glm47_tool_parser, mock_request):; symbols: test_whitespace_content_none, _reset, TestGlm47Streaming, test_no_args
  - `vllm/tool_parsers/utils.py` modified +13/-0 (13 lines); hunks: -31,6 +31,19; symbols: partial_tag_overlap, find_common_prefix
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_glm4_moe_tool_parser.py
@@ -357,81 +357,69 @@ def test_extract_tool_calls_mixed_content(glm4_moe_tool_parser, mock_request):
-    # Reset streaming state
-    glm4_moe_tool_parser.current_tool_name_sent = False
-    glm4_moe_tool_parser.prev_tool_call_arr = []
-    glm4_moe_tool_parser.current_tool_id = -1
-    glm4_moe_tool_parser.streamed_args_for_tool = []
+    _reset_streaming_state(glm4_moe_tool_parser)
diff -- vllm/tool_parsers/glm4_moe_tool_parser.py
@@ -37,16 +37,17 @@
+from vllm.tool_parsers.utils import partial_tag_overlap
-    This parser emits tool-call deltas incrementally as arguments arrive.
-    For string-type parameters, content is streamed character-by-character
-    rather than waiting for the complete </arg_value> tag.
+    On every streaming call the parser re-parses ``current_text`` to find
+    ``<tool_call>`` regions, builds the JSON arguments string for each tool
diff -- tests/tool_parsers/test_glm47_moe_tool_parser.py
@@ -117,28 +117,24 @@ def test_whitespace_content_none(self, glm47_tool_parser, mock_request):
```

- 已读文件:
  - tests: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +509/-103; `tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +14/-17
  - runtime: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +252/-296; `vllm/tool_parsers/utils.py` modified +13/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39870 - [BugFix] Support custom tool parsers when tool_choice is `required` and named function

- 链接: https://github.com/vllm-project/vllm/pull/39870
- 状态/时间: merged / 2026-04-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+100/-12，可读 patch 230 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Support custom tool parsers when tool_choice is `required` and named function」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`；技术摘要: 覆盖「[BugFix] Support custom tool parsers when tool_choice is `required` and named function」；主要实现面是 `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/entrypoints/openai/chat_completion/serving.py` modified +39/-6 (45 lines); hunks: -557,6 +557,20 @@ async def chat_completion_stream_generator(; -569,7 +583,12 @@ async def chat_completion_stream_generator(; symbols: chat_completion_stream_generator，涉及 `chat_completion_stream_generator`；`vllm/entrypoints/openai/engine/serving.py` modified +26/-5 (31 lines); hunks: -627,7 +627,7 @@ def _parse_tool_calls_from_content(; -636,14 +636,20 @@ def _parse_tool_calls_from_content(; symbols: _parse_tool_calls_from_content，涉及 `_parse_tool_calls_from_content`；`vllm/tool_parsers/glm4_moe_tool_parser.py` modified +22/-1 (23 lines); hunks: -20,6 +20,7; -50,6 +51,8 @@ class Glm4MoeModelToolParser(ToolParser):; symbols: Glm4MoeModelToolParser, __init__, _tools_enabled, adjust_request，涉及 `Glm4MoeModelToolParser, __init__, _tools_enabled`；`vllm/tool_parsers/abstract_tool_parser.py` modified +11/-0 (11 lines); hunks: -44,6 +44,17 @@ class ToolParser:; symbols: ToolParser, __init__，涉及 `ToolParser, __init__`。
- 代码 diff 细节:
  - `vllm/entrypoints/openai/chat_completion/serving.py` modified +39/-6 (45 lines); hunks: -557,6 +557,20 @@ async def chat_completion_stream_generator(; -569,7 +583,12 @@ async def chat_completion_stream_generator(; symbols: chat_completion_stream_generator
  - `vllm/entrypoints/openai/engine/serving.py` modified +26/-5 (31 lines); hunks: -627,7 +627,7 @@ def _parse_tool_calls_from_content(; -636,14 +636,20 @@ def _parse_tool_calls_from_content(; symbols: _parse_tool_calls_from_content
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +22/-1 (23 lines); hunks: -20,6 +20,7; -50,6 +51,8 @@ class Glm4MoeModelToolParser(ToolParser):; symbols: Glm4MoeModelToolParser, __init__, _tools_enabled, adjust_request
  - `vllm/tool_parsers/abstract_tool_parser.py` modified +11/-0 (11 lines); hunks: -44,6 +44,17 @@ class ToolParser:; symbols: ToolParser, __init__
  - `vllm/tool_parsers/glm47_moe_tool_parser.py` modified +2/-0 (2 lines); hunks: -23,6 +23,8; symbols: Glm47MoeModelToolParser, __init__
- 关键代码摘录:

```diff
diff -- vllm/entrypoints/openai/chat_completion/serving.py
@@ -557,6 +557,20 @@ async def chat_completion_stream_generator(
+        # Determine whether required/named tool_choice should fall back to
+        # the auto tool_parser path instead of the standard JSON-based parsing.
+        # This happens when the parser declares supports_required_and_named=False
+        # (e.g. GLM models that output XML instead of JSON).
+        tool_choice_uses_parser = (
+            self.tool_parser is not None
diff -- vllm/entrypoints/openai/engine/serving.py
@@ -627,7 +627,7 @@ def _parse_tool_calls_from_content(
-            # Forced Function Call
+            # Forced Function Call (Responses API)
@@ -636,14 +636,20 @@ def _parse_tool_calls_from_content(
+            and (tool_parser_cls is None or tool_parser_cls.supports_required_and_named)
+            # Named function with standard JSON-based parsing
-            # Forced Function Call
diff -- vllm/tool_parsers/glm4_moe_tool_parser.py
@@ -20,6 +20,7 @@
```

- 已读文件:
  - runtime: `vllm/entrypoints/openai/chat_completion/serving.py` modified +39/-6; `vllm/entrypoints/openai/engine/serving.py` modified +26/-5; `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +22/-1; `vllm/tool_parsers/abstract_tool_parser.py` modified +11/-0; `vllm/tool_parsers/glm47_moe_tool_parser.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/tool_parsers/abstract_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35949 - [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase

- 链接: https://github.com/vllm-project/vllm/pull/35949
- 状态/时间: merged / 2026-04-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 53 个文件，+325/-702，可读 patch 2430 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase」；模型线: GLM-4.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`；技术摘要: 覆盖「[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase」；主要实现面是 `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake，涉及 `_resolve_layer_name, _moe_forward, _moe_forward_shared`；`vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__，涉及 `FusedMoE, __init__`；`vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__
  - `vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/AXK1.py` modified +5/-30 (35 lines); hunks: -100,7 +100,7 @@ def __init__(; -170,7 +170,6 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py
@@ -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:
-# the runner's 'forward_dispatch' method.
+# the runner's '_forward_dispatch' method.
+# These functions should never be called directly since they do not
+# include all the functionality of the MoE layer.
-    return layer.runner.forward_dispatch(
+    return layer.runner._forward_dispatch(
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -230,11 +230,18 @@ class FusedMoE(PluggableLayer):
-        reduce_results: Whether to all_reduce on the output of the layer
+        routed_scaling_factor: A scaling factor that is applied to the topk_weights
+                               by the router or the output of the layer depending
+                               on the value of `apply_routed_scale_to_output`
+        apply_routed_scale_to_output: Determine whether or not `routed_scaling_factor`
+                                      is applied to the topk_weights or to the experts
diff -- vllm/model_executor/models/exaone_moe.py
@@ -31,6 +31,7 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32; `vllm/model_executor/models/exaone_moe.py` modified +18/-28; `vllm/model_executor/models/kimi_linear.py` modified +20/-26; `vllm/model_executor/models/AXK1.py` modified +5/-30; `vllm/model_executor/models/ernie45_vl_moe.py` modified +5/-30
- 验证与风险: diff 自带测试面 `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py`, `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35782 - [MoE Refactor] Remove SharedFusedMoE class

- 链接: https://github.com/vllm-project/vllm/pull/35782
- 状态/时间: merged / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 33 个文件，+112/-141，可读 patch 926 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Remove SharedFusedMoE class」；模型线: GLM-4.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`；技术摘要: 覆盖「[MoE Refactor] Remove SharedFusedMoE class」；主要实现面是 `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25 (25 lines); hunks: -1,25 +0,0; symbols: SharedFusedMoE, forward，涉及 `SharedFusedMoE, forward`；`vllm/model_executor/models/afmoe.py` modified +5/-5 (10 lines); hunks: -18,7 +18,7; -124,8 +124,8 @@ def __init__(; symbols: __init__, make_empty_intermediate_tensors, get_expert_mapping，涉及 `__init__, make_empty_intermediate_tensors, get_expert_mapping`；`vllm/model_executor/models/llama4.py` modified +5/-5 (10 lines); hunks: -36,7 +36,7; -127,7 +127,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, load_moe_expert_weights, load_weights，涉及 `__init__, load_moe_expert_weights, load_weights`；`vllm/model_executor/models/AXK1.py` modified +4/-4 (8 lines); hunks: -42,7 +42,7; -163,7 +163,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights，涉及 `__init__, compute_logits, get_expert_mapping`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25 (25 lines); hunks: -1,25 +0,0; symbols: SharedFusedMoE, forward
  - `vllm/model_executor/models/afmoe.py` modified +5/-5 (10 lines); hunks: -18,7 +18,7; -124,8 +124,8 @@ def __init__(; symbols: __init__, make_empty_intermediate_tensors, get_expert_mapping
  - `vllm/model_executor/models/llama4.py` modified +5/-5 (10 lines); hunks: -36,7 +36,7; -127,7 +127,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/AXK1.py` modified +4/-4 (8 lines); hunks: -42,7 +42,7; -163,7 +163,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/deepseek_v2.py` modified +4/-4 (8 lines); hunks: -48,9 +48,9; -311,7 +311,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/shared_fused_moe.py
@@ -1,25 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-import torch
-from vllm.model_executor.layers.fused_moe.layer import FusedMoE
-# TODO(bnell): Remove this entirely
-class SharedFusedMoE(FusedMoE):
diff -- vllm/model_executor/models/afmoe.py
@@ -18,7 +18,7 @@
-from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
+from vllm.model_executor.layers.fused_moe import FusedMoE
@@ -124,8 +124,8 @@ def __init__(
-        # Routed experts using SharedFusedMoE
-        self.experts = SharedFusedMoE(
+        # Routed experts using FusedMoE
diff -- vllm/model_executor/models/llama4.py
@@ -36,7 +36,7 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25; `vllm/model_executor/models/afmoe.py` modified +5/-5; `vllm/model_executor/models/llama4.py` modified +5/-5; `vllm/model_executor/models/AXK1.py` modified +4/-4; `vllm/model_executor/models/deepseek_v2.py` modified +4/-4; `vllm/model_executor/models/ernie45_moe.py` modified +4/-4
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- 链接: https://github.com/vllm-project/vllm/pull/40671
- 状态/时间: merged / 2026-04-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 53 个文件，+254/-98，可读 patch 1073 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；模型线: GLM-4.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`；技术摘要: 覆盖「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping，涉及 `extra_repr, fused_moe_make_expert_params_mapping`；`vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights，涉及 `load_moe_expert_weights, load_weights`；`vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits，涉及 `make_empty_intermediate_tensors, get_expert_mapping, load_weights`；`vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights，涉及 `compute_logits, get_expert_mapping, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping
  - `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits
  - `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/afmoe.py` modified +5/-2 (7 lines); hunks: -18,7 +18,10; -479,7 +482,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0; `vllm/model_executor/models/llama4.py` modified +7/-4; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4; `vllm/model_executor/models/AXK1.py` modified +6/-3; `vllm/model_executor/models/afmoe.py` modified +5/-2; `vllm/model_executor/models/bailing_moe.py` modified +5/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41755 - [Bugfix] Fix GLM4-MoE weight loading for NVFP4 quantized checkpoints

- 链接: https://github.com/vllm-project/vllm/pull/41755
- 状态/时间: merged / 2026-05-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `75f0d516c43c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-2，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix GLM4-MoE weight loading for NVFP4 quantized checkpoints」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「[Bugfix] Fix GLM4-MoE weight loading for NVFP4 quantized checkpoints」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +10/-2 (12 lines); hunks: -506,16 +506,24 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +10/-2 (12 lines); hunks: -506,16 +506,24 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -506,16 +506,24 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+                name = maybe_remap_kv_scale_name(name, params_dict)
+                if name is None:
+                    continue
-                weight_loader = param.weight_loader
-                weight_loader(param, loaded_weight, shard_id)
+                weight_loader = getattr(param, "weight_loader", default_weight_loader)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +10/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42026 - [Bugfix] Preserve leading/trailing whitespace in GLM non-streaming tool parser

- 链接: https://github.com/vllm-project/vllm/pull/42026
- 状态/时间: merged / 2026-05-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+40/-3，可读 patch 64 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Preserve leading/trailing whitespace in GLM non-streaming tool parser」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Preserve leading/trailing whitespace in GLM non-streaming tool parser」；主要实现面是 `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +30/-0 (30 lines); hunks: -801,6 +801,36 @@ def test_extract_tool_calls_numeric_deserialization(glm4_mo...; symbols: test_extract_tool_calls_numeric_deserialization, test_whitespace_preserved_in_arg_values, test_zero_argument_tool_call，涉及 `test_extract_tool_calls_numeric_deserialization, test_whitespace_preserved_in_arg_values, test_zero_argument_tool_call`；`vllm/tool_parsers/glm4_moe_tool_parser.py` modified +4/-3 (7 lines); hunks: -210,9 +210,10 @@ def extract_tool_calls(; symbols: extract_tool_calls，涉及 `extract_tool_calls`；`tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +6/-0 (6 lines); hunks: -91,6 +91,12 @@ def test_args_with_newlines(self, glm47_tool_parser, mock_req...; symbols: test_args_with_newlines, test_whitespace_preserved_in_arg_values, test_content_before，涉及 `test_args_with_newlines, test_whitespace_preserved_in_arg_values, test_content_before`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +30/-0 (30 lines); hunks: -801,6 +801,36 @@ def test_extract_tool_calls_numeric_deserialization(glm4_mo...; symbols: test_extract_tool_calls_numeric_deserialization, test_whitespace_preserved_in_arg_values, test_zero_argument_tool_call
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +4/-3 (7 lines); hunks: -210,9 +210,10 @@ def extract_tool_calls(; symbols: extract_tool_calls
  - `tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +6/-0 (6 lines); hunks: -91,6 +91,12 @@ def test_args_with_newlines(self, glm47_tool_parser, mock_req...; symbols: test_args_with_newlines, test_whitespace_preserved_in_arg_values, test_content_before
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_glm4_moe_tool_parser.py
@@ -801,6 +801,36 @@ def test_extract_tool_calls_numeric_deserialization(glm4_moe_tool_parser, mock_r
+def test_whitespace_preserved_in_arg_values(glm4_moe_tokenizer):
+    """Test that string arguments preserve leading and trailing whitespace."""
+    tools = [
+        ChatCompletionToolsParam(
+            function=FunctionDefinition(
+                name="apply_diff",
diff -- vllm/tool_parsers/glm4_moe_tool_parser.py
@@ -210,9 +210,10 @@ def extract_tool_calls(
-                    arg_val = value.strip()
-                    if not self._is_string_type(tc_name, arg_key, self.tools):
-                        arg_val = self._deserialize(arg_val)
+                    if self._is_string_type(tc_name, arg_key, self.tools):
+                        arg_val = value
+                    else:
diff -- tests/tool_parsers/test_glm47_moe_tool_parser.py
@@ -91,6 +91,12 @@ def test_args_with_newlines(self, glm47_tool_parser, mock_request):
```

- 已读文件:
  - tests: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +30/-0; `tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +6/-0
  - runtime: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +4/-3
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39601 - [Bugfix] Fix glm4_moe_tool_parser._is_string_type for /v1/responses FunctionTool format

- 链接: https://github.com/vllm-project/vllm/pull/39601
- 状态/时间: merged / 2026-05-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_glm4_moe_tool_parser.py`；关联提交 `050611a3dd19`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+135/-25，可读 patch 214 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix glm4_moe_tool_parser._is_string_type for /v1/responses FunctionTool format」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Fix glm4_moe_tool_parser._is_string_type for /v1/responses FunctionTool format」；主要实现面是 `tests/tool_parsers/test_glm4_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +120/-0 (120 lines); hunks: -5,6 +5,7; -1363,3 +1364,122 @@ def test_stream_interval_content_between_tool_calls(; symbols: test_stream_interval_content_between_tool_calls, function_tools, glm4_moe_parser_function_tools, mock_request_function_tools，涉及 `test_stream_interval_content_between_tool_calls, function_tools, glm4_moe_parser_function_tools`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +120/-0 (120 lines); hunks: -5,6 +5,7; -1363,3 +1364,122 @@ def test_stream_interval_content_between_tool_calls(; symbols: test_stream_interval_content_between_tool_calls, function_tools, glm4_moe_parser_function_tools, mock_request_function_tools
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_glm4_moe_tool_parser.py
@@ -5,6 +5,7 @@
+from openai.types.responses import FunctionTool
@@ -1363,3 +1364,122 @@ def test_stream_interval_content_between_tool_calls(
+# ── FunctionTool (Responses API) tests ──────────────────────────────
+@pytest.fixture
+def function_tools():
+    return [
```

- 已读文件:
  - tests: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +120/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_glm4_moe_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #44346 - [Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers

- 链接: https://github.com/vllm-project/vllm/pull/44346
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+20/-15，可读 patch 178 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py`；技术摘要: 覆盖「[Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers」；主要实现面是 `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/utils.py` modified +7/-0 (7 lines); hunks: -3,6 +3,7; -31,6 +32,12; symbols: safe_literal_eval, partial_tag_overlap，涉及 `safe_literal_eval, partial_tag_overlap`；`vllm/tool_parsers/hy_v3_tool_parser.py` modified +3/-3 (6 lines); hunks: -1,7 +1,6; -27,6 +26,7; symbols: _try_parse_wildcard_number, _deserialize，涉及 `_try_parse_wildcard_number, _deserialize`；`vllm/tool_parsers/minicpm5xml_tool_parser.py` modified +2/-3 (5 lines); hunks: -1,7 +1,6; -28,7 +27,7; symbols: _parse_arguments，涉及 `_parse_arguments`；`vllm/tool_parsers/qwen3xml_tool_parser.py` modified +2/-3 (5 lines); hunks: -1,6 +1,5; -26,7 +25,7; symbols: _end_element，涉及 `_end_element`。
- 代码 diff 细节:
  - `vllm/tool_parsers/utils.py` modified +7/-0 (7 lines); hunks: -3,6 +3,7; -31,6 +32,12; symbols: safe_literal_eval, partial_tag_overlap
  - `vllm/tool_parsers/hy_v3_tool_parser.py` modified +3/-3 (6 lines); hunks: -1,7 +1,6; -27,6 +26,7; symbols: _try_parse_wildcard_number, _deserialize
  - `vllm/tool_parsers/minicpm5xml_tool_parser.py` modified +2/-3 (5 lines); hunks: -1,7 +1,6; -28,7 +27,7; symbols: _parse_arguments
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +2/-3 (5 lines); hunks: -1,6 +1,5; -26,7 +25,7; symbols: _end_element
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +2/-2 (4 lines); hunks: -11,7 +11,6; -42,6 +41,7; symbols: _deserialize
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/utils.py
@@ -3,6 +3,7 @@
+import warnings
@@ -31,6 +32,12 @@
+def safe_literal_eval(text: str):
+    with warnings.catch_warnings():
+        warnings.simplefilter("ignore", SyntaxWarning)
+        return ast.literal_eval(text)
diff -- vllm/tool_parsers/hy_v3_tool_parser.py
@@ -1,7 +1,6 @@
-import ast
@@ -27,6 +26,7 @@
+from vllm.tool_parsers.utils import safe_literal_eval
@@ -183,13 +183,13 @@ def _try_parse_wildcard_number(value: str) -> int | float | None:
-        """Deserialize a string value using json.loads then ast.literal_eval."""
+        """Deserialize a string value using json.loads then safe_literal_eval."""
diff -- vllm/tool_parsers/minicpm5xml_tool_parser.py
@@ -1,7 +1,6 @@
```

- 已读文件:
  - runtime: `vllm/tool_parsers/utils.py` modified +7/-0; `vllm/tool_parsers/hy_v3_tool_parser.py` modified +3/-3; `vllm/tool_parsers/minicpm5xml_tool_parser.py` modified +2/-3; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +2/-3; `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +2/-2; `vllm/tool_parsers/poolside_v1_tool_parser.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/tool_parsers/glm4_moe_tool_parser.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41184 - [MoE Refactor] FusedMoE/MoERunner inversion refactor

- 链接: https://github.com/vllm-project/vllm/pull/41184
- 状态/时间: merged / 2026-06-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 90 个文件，+2734/-2027，可读 patch 7329 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] FusedMoE/MoERunner inversion refactor」；模型线: GLM-4.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`；技术摘要: 覆盖「[MoE Refactor] FusedMoE/MoERunner inversion refactor」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334 (1648 lines); hunks: -1,1424 +1,404; symbols: FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE, determine_expert_counts，涉及 `FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE`；`vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0 (1144 lines); hunks: -0,0 +1,1144; symbols: FusedMoeWeightScaleSupported, RoutedExperts, __init__, _replace_quant_method，涉及 `FusedMoeWeightScaleSupported, RoutedExperts, __init__`；`vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82 (339 lines); hunks: -1,28 +1,39; -43,8 +54,23; symbols: register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward，涉及 `register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward`；`vllm/lora/layers/fused_moe.py` modified +76/-43 (119 lines); hunks: -10,7 +10,7; -25,15 +25,24; symbols: FusedMoEWithLoRA, __init__，涉及 `FusedMoEWithLoRA, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334 (1648 lines); hunks: -1,1424 +1,404; symbols: FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE, determine_expert_counts
  - `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0 (1144 lines); hunks: -0,0 +1,1144; symbols: FusedMoeWeightScaleSupported, RoutedExperts, __init__, _replace_quant_method
  - `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82 (339 lines); hunks: -1,28 +1,39; -43,8 +54,23; symbols: register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward
  - `vllm/lora/layers/fused_moe.py` modified +76/-43 (119 lines); hunks: -10,7 +10,7; -25,15 +25,24; symbols: FusedMoEWithLoRA, __init__
  - `vllm/model_executor/model_loader/weight_utils.py` modified +106/-1 (107 lines); hunks: -13,7 +13,7; -1633,3 +1633,108 @@ def maybe_remap_kv_scale_name(name: str, params_dict: di...; symbols: maybe_remap_kv_scale_name, maybe_remap_moe_expert_param_name, remap_moe_expert_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -1,1424 +1,404 @@
-from collections.abc import Callable, Iterable
-from enum import Enum
-from typing import Literal, cast, overload
+from collections.abc import Callable
+from typing import Any
-from torch.nn.parameter import UninitializedParameter
diff -- vllm/model_executor/layers/fused_moe/routed_experts.py
@@ -0,0 +1,1144 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Callable, Iterable
+from enum import Enum
+from typing import TYPE_CHECKING, Any, Literal, cast, overload
+import torch
diff -- vllm/model_executor/layers/fused_moe/runner/moe_runner.py
@@ -1,28 +1,39 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334; `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0; `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82; `vllm/lora/layers/fused_moe.py` modified +76/-43; `vllm/model_executor/model_loader/weight_utils.py` modified +106/-1; `vllm/model_executor/layers/fused_moe/runner/moe_runner_interface.py` modified +102/-2
- 验证与风险: diff 自带测试面 `tests/distributed/test_eplb_fused_moe_layer.py`, `tests/distributed/test_eplb_fused_moe_layer_dep_nvfp4.py`, `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/parallel_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45915 - [Frontend] Add Streaming Parser Engine and new GLM4.7/GLM5.1/GLM5.2 Parser

- 链接: https://github.com/vllm-project/vllm/pull/45915
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+534/-1948，可读 patch 2693 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend] Add Streaming Parser Engine and new GLM4.7/GLM5.1/GLM5.2 Parser」；模型线: GLM-4.5；类别: 文档/测试/CI；主要 diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/reasoning/test_glm4_moe_reasoning_parser.py`；技术摘要: 覆盖「[Frontend] Add Streaming Parser Engine and new GLM4.7/GLM5.1/GLM5.2 Parser」；主要实现面是 `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/reasoning/test_glm4_moe_reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +147/-1404 (1551 lines); hunks: -1,1067 +1,57; -1071,415 +61,168 @@ def test_streaming_multi_token_with_multiple_args(glm4_m...; symbols: glm4_moe_tokenizer, sample_tools, glm4_moe_tool_parser, mock_request，涉及 `glm4_moe_tokenizer, sample_tools, glm4_moe_tool_parser`；`vllm/tool_parsers/glm4_moe_tool_parser.py` removed +0/-495 (495 lines); hunks: -1,495 +0,0; symbols: Glm4MoeModelToolParser, __init__, _deserialize, _json_escape_string_content，涉及 `Glm4MoeModelToolParser, __init__, _deserialize`；`tests/reasoning/test_glm4_moe_reasoning_parser.py` modified +31/-7 (38 lines); hunks: -11,7 +11,7; -35,18 +35,32 @@ def glm45_tokenizer():; symbols: glm45_tokenizer，涉及 `glm45_tokenizer`；`tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +31/-5 (36 lines); hunks: -16,7 +16,7; -136,9 +136,10 @@ def test_no_args(self, glm47_tool_parser, mock_request):; symbols: test_no_args, test_with_args，涉及 `test_no_args, test_with_args`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +147/-1404 (1551 lines); hunks: -1,1067 +1,57; -1071,415 +61,168 @@ def test_streaming_multi_token_with_multiple_args(glm4_m...; symbols: glm4_moe_tokenizer, sample_tools, glm4_moe_tool_parser, mock_request
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` removed +0/-495 (495 lines); hunks: -1,495 +0,0; symbols: Glm4MoeModelToolParser, __init__, _deserialize, _json_escape_string_content
  - `tests/reasoning/test_glm4_moe_reasoning_parser.py` modified +31/-7 (38 lines); hunks: -11,7 +11,7; -35,18 +35,32 @@ def glm45_tokenizer():; symbols: glm45_tokenizer
  - `tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +31/-5 (36 lines); hunks: -16,7 +16,7; -136,9 +136,10 @@ def test_no_args(self, glm47_tool_parser, mock_request):; symbols: test_no_args, test_with_args
  - `vllm/tool_parsers/glm47_moe_tool_parser.py` modified +3/-33 (36 lines); hunks: -1,41 +1,11; symbols: Glm47MoeModelToolParser, __init__
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_glm4_moe_tool_parser.py
@@ -1,1067 +1,57 @@
+"""Compatibility tests for GLM-4.5 using the shared GLM XML parser."""
-from unittest.mock import Mock
-import pytest
-from openai.types.responses import FunctionTool
+from typing import Any, TypedDict
+from tests.parser.engine.replay_harness import MockTokenizer
diff -- vllm/tool_parsers/glm4_moe_tool_parser.py
@@ -1,495 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-"""
-GLM-4 Tool Call Parser with incremental string streaming support.
-This parser fixes the streaming issue reported in Issue #32829 where long string
-parameters (e.g., file content with 4000+ characters of code) are buffered until
diff -- tests/reasoning/test_glm4_moe_reasoning_parser.py
@@ -11,7 +11,7 @@
```

- 已读文件:
  - tests: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +147/-1404; `tests/reasoning/test_glm4_moe_reasoning_parser.py` modified +31/-7; `tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +31/-5
  - runtime: `vllm/tool_parsers/glm4_moe_tool_parser.py` removed +0/-495; `vllm/tool_parsers/glm47_moe_tool_parser.py` modified +3/-33; `vllm/reasoning/__init__.py` modified +6/-2; `vllm/reasoning/glm47_moe_reasoning_parser.py` added +6/-0; `vllm/tool_parsers/__init__.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/parser/engine/trace_builder.py`, `tests/reasoning/test_glm4_moe_reasoning_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #46651 - [Perf] Remove redundant clone for GLM, Deepseek etc

- 链接: https://github.com/vllm-project/vllm/pull/46651
- 状态/时间: merged / 2026-06-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+4/-4，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf] Remove redundant clone for GLM, Deepseek etc」；模型线: GLM-4.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`；技术摘要: 覆盖「[Perf] Remove redundant clone for GLM, Deepseek etc」；主要实现面是 `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/AXK1.py` modified +1/-1 (2 lines); hunks: -649,7 +649,7 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1186,7 +1186,7 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/model_executor/models/glm4_moe_lite.py` modified +1/-1 (2 lines); hunks: -184,7 +184,7 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/model_executor/models/openpangu.py` modified +1/-1 (2 lines); hunks: -935,7 +935,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/AXK1.py` modified +1/-1 (2 lines); hunks: -649,7 +649,7 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1186,7 +1186,7 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +1/-1 (2 lines); hunks: -184,7 +184,7 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/openpangu.py` modified +1/-1 (2 lines); hunks: -935,7 +935,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/AXK1.py
@@ -649,7 +649,7 @@ def forward(
-            residual = hidden_states.clone()
+            residual = hidden_states
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -1186,7 +1186,7 @@ def forward(
-            residual = hidden_states.clone()
+            residual = hidden_states
diff -- vllm/model_executor/models/glm4_moe_lite.py
@@ -184,7 +184,7 @@ def forward(
-            residual = hidden_states.clone()
+            residual = hidden_states
diff -- vllm/model_executor/models/openpangu.py
@@ -935,7 +935,7 @@ def forward(
-            residual = hidden_states.clone()
+            residual = hidden_states
```

- 已读文件:
  - runtime: `vllm/model_executor/models/AXK1.py` modified +1/-1; `vllm/model_executor/models/deepseek_v2.py` modified +1/-1; `vllm/model_executor/models/glm4_moe_lite.py` modified +1/-1; `vllm/model_executor/models/openpangu.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
