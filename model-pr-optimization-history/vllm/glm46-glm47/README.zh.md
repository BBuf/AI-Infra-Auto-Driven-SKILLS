# vllm GLM-4.6/4.7 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/reasoning/test_glm4_moe_reasoning_parser.py` | 无直接 PR 号提交 |
| `tests/tool_parsers/test_glm47_moe_tool_parser.py` | [#37386](https://github.com/vllm-project/vllm/pull/37386) |
| `tests/tool_parsers/test_glm4_moe_tool_parser.py` | [#37386](https://github.com/vllm-project/vllm/pull/37386) |
| `vllm/model_executor/models/glm4_moe.py` | [#30876](https://github.com/vllm-project/vllm/pull/30876) |
| `vllm/model_executor/models/glm4_moe_lite.py` | [#31386](https://github.com/vllm-project/vllm/pull/31386) |
| `vllm/model_executor/models/glm4_moe_lite_mtp.py` | [#31386](https://github.com/vllm-project/vllm/pull/31386) |
| `vllm/model_executor/models/glm4_moe_mtp.py` | [#27597](https://github.com/vllm-project/vllm/pull/27597), [#31386](https://github.com/vllm-project/vllm/pull/31386) |
| `vllm/parser/glm47_moe.py` | 无直接 PR 号提交 |
| `vllm/reasoning/glm47_moe_reasoning_parser.py` | 无直接 PR 号提交 |
| `vllm/tool_parsers/glm47_moe_tool_parser.py` | [#30876](https://github.com/vllm-project/vllm/pull/30876), [#37386](https://github.com/vllm-project/vllm/pull/37386) |

## PR 覆盖总览

- git 追溯 PR 数: 4
- 原文档显式引用补充 PR 数: 20
- 当前文档总 PR 数: 24
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-10-14 | [#26818](https://github.com/vllm-project/vllm/pull/26818) | merged | [Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200 | `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` |
| 2025-11-12 | [#27597](https://github.com/vllm-project/vllm/pull/27597) | merged | [Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint. | `vllm/model_executor/models/glm4_moe_mtp.py` |
| 2025-12-09 | [#30210](https://github.com/vllm-project/vllm/pull/30210) | merged | [Bugfix]: Fix glm46 awq marlin moe wna16 compatibility | `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py` |
| 2025-12-20 | [#30876](https://github.com/vllm-project/vllm/pull/30876) | merged | GLM-4.7 Tool Parser and Doc Update | `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/model_executor/models/glm4_moe.py` |
| 2026-01-05 | [#31622](https://github.com/vllm-project/vllm/pull/31622) | merged | Fix GLM-4.6v flash tool calling in transformers 5.x | `vllm/tool_parsers/glm4_moe_tool_parser.py`, `examples/tool_chat_template_glm4.jinja` |
| 2026-01-19 | [#31386](https://github.com/vllm-project/vllm/pull/31386) | merged | [GLM-4.7] GLM Model support for GLM-Lite | `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/model_executor/models/glm4_moe_mtp.py` |
| 2026-03-18 | [#37386](https://github.com/vllm-project/vllm/pull/37386) | merged | fix(glm47): improve tool call parsing and content normalization | `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py` |
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
| 2026-05-21 | [#39601](https://github.com/vllm-project/vllm/pull/39601) | merged | [Bugfix] Fix glm4_moe_tool_parser._is_string_type for /v1/responses FunctionTool format | `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py` |
| 2026-06-03 | [#44346](https://github.com/vllm-project/vllm/pull/44346) | merged | [Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers | `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py` |
| 2026-06-08 | [#41184](https://github.com/vllm-project/vllm/pull/41184) | merged | [MoE Refactor] FusedMoE/MoERunner inversion refactor | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` |
| 2026-06-12 | [#45003](https://github.com/vllm-project/vllm/pull/45003) | merged | [Frontend] Support strict mode for tool calling | `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py` |
| 2026-06-18 | [#45915](https://github.com/vllm-project/vllm/pull/45915) | merged | [Frontend] Add Streaming Parser Engine and new GLM4.7/GLM5.1/GLM5.2 Parser | `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/reasoning/test_glm4_moe_reasoning_parser.py` |
| 2026-06-25 | [#46651](https://github.com/vllm-project/vllm/pull/46651) | merged | [Perf] Remove redundant clone for GLM, Deepseek etc | `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py` |

## 逐 PR diff 审计卡

### PR #26818 - [Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200

- 链接: https://github.com/vllm-project/vllm/pull/26818
- 状态/时间: merged / 2025-10-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+441/-0，可读 patch 444 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json`；技术摘要: 覆盖「[Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200」；主要实现面是 `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147；`vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147；`vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json
@@ -0,0 +1,147 @@
+{
+    "triton_version": "3.4.0",
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
diff -- vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json
@@ -0,0 +1,147 @@
+{
+    "triton_version": "3.4.0",
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
diff -- vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json
@@ -0,0 +1,147 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` added +147/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27597 - [Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint.

- 链接: https://github.com/vllm-project/vllm/pull/27597
- 状态/时间: merged / 2025-11-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe_mtp.py`；关联提交 `d3ade61e429f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-4，可读 patch 23 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint.」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_moe_mtp.py`；技术摘要: 覆盖「[Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint.」；主要实现面是 `vllm/model_executor/models/glm4_moe_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #30210 - [Bugfix]: Fix glm46 awq marlin moe wna16 compatibility

- 链接: https://github.com/vllm-project/vllm/pull/30210
- 状态/时间: merged / 2025-12-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+50/-4，可读 patch 96 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix]: Fix glm46 awq marlin moe wna16 compatibility」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py`；技术摘要: 覆盖「[Bugfix]: Fix glm46 awq marlin moe wna16 compatibility」；主要实现面是 `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +45/-0 (45 lines); hunks: -895,6 +895,48 @@ def get_moe_configs(; -960,6 +1002,9 @@ def get_moe_wna16_block_config(; symbols: get_moe_configs, _ensure_block_size_k_divisible, get_moe_wna16_block_config，涉及 `get_moe_configs, _ensure_block_size_k_divisible, get_moe_wna16_block_config`；`vllm/model_executor/layers/quantization/moe_wna16.py` modified +5/-4 (9 lines); hunks: -60,7 +60,7 @@ def __init__(; -107,7 +107,7 @@ def from_config(cls, config: dict[str, Any]) -> "MoeWNA16Con...; symbols: __init__, from_config, get_quant_method, moe_wna16_weight_loader，涉及 `__init__, from_config, get_quant_method`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +45/-0 (45 lines); hunks: -895,6 +895,48 @@ def get_moe_configs(; -960,6 +1002,9 @@ def get_moe_wna16_block_config(; symbols: get_moe_configs, _ensure_block_size_k_divisible, get_moe_wna16_block_config
  - `vllm/model_executor/layers/quantization/moe_wna16.py` modified +5/-4 (9 lines); hunks: -60,7 +60,7 @@ def __init__(; -107,7 +107,7 @@ def from_config(cls, config: dict[str, Any]) -> "MoeWNA16Con...; symbols: __init__, from_config, get_quant_method, moe_wna16_weight_loader
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -895,6 +895,48 @@ def get_moe_configs(
+def _ensure_block_size_k_divisible(
+    size_k: int, block_size_k: int, group_size: int
+) -> int:
+    """Ensure block_size_k is a divisor of size_k and divisible by group_size.
+    This ensures BLOCK_SIZE_K compatibility with MoeWNA16 CUDA kernel which
+    requires size_k % BLOCK_SIZE_K == 0 and BLOCK_SIZE_K % group_size == 0.
diff -- vllm/model_executor/layers/quantization/moe_wna16.py
@@ -60,7 +60,7 @@ def __init__(
-        elif self.linear_quant_method == "awq":
+        elif self.linear_quant_method in ("awq", "awq_marlin"):
@@ -107,7 +107,7 @@ def from_config(cls, config: dict[str, Any]) -> "MoeWNA16Config":
-        elif linear_quant_method == "awq":
+        elif linear_quant_method in ("awq", "awq_marlin"):
@@ -184,7 +184,7 @@ def get_quant_method(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +45/-0; `vllm/model_executor/layers/quantization/moe_wna16.py` modified +5/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30876 - GLM-4.7 Tool Parser and Doc Update

- 链接: https://github.com/vllm-project/vllm/pull/30876
- 状态/时间: merged / 2025-12-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`；关联提交 `8a7a41437490`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+38/-3，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「GLM-4.7 Tool Parser and Doc Update」；模型线: GLM-4.6/4.7；类别: 文档/测试/CI；主要 diff: `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「GLM-4.7 Tool Parser and Doc Update」；主要实现面是 `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/glm47_moe_tool_parser.py` added +23/-0 (23 lines); hunks: -0,0 +1,23; symbols: Glm47MoeModelToolParser, __init__，涉及 `Glm47MoeModelToolParser, __init__`；`vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -21,7 +21,8。
- 代码 diff 细节:
  - `vllm/tool_parsers/glm47_moe_tool_parser.py` added +23/-0 (23 lines); hunks: -0,0 +1,23; symbols: Glm47MoeModelToolParser, __init__
  - `vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -21,7 +21,8
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
diff -- vllm/model_executor/models/glm4_moe.py
@@ -21,7 +21,8 @@
-"""Inference-only GLM-4.5, GLM-4.6 model compatible with HuggingFace weights."""
+"""Inference-only GLM-4.5, GLM-4.6, GLM-4.7 model
+compatible with HuggingFace weights."""
```

- 已读文件:
  - runtime: `vllm/tool_parsers/glm47_moe_tool_parser.py` added +23/-0; `vllm/model_executor/models/glm4_moe.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe.py`, `vllm/tool_parsers/__init__.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31622 - Fix GLM-4.6v flash tool calling in transformers 5.x

- 链接: https://github.com/vllm-project/vllm/pull/31622
- 状态/时间: merged / 2026-01-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+68/-0，可读 patch 76 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GLM-4.6v flash tool calling in transformers 5.x」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/glm4_moe_tool_parser.py`, `examples/tool_chat_template_glm4.jinja`；技术摘要: 覆盖「Fix GLM-4.6v flash tool calling in transformers 5.x」；主要实现面是 `vllm/tool_parsers/glm4_moe_tool_parser.py`, `examples/tool_chat_template_glm4.jinja`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #31386 - [GLM-4.7] GLM Model support for GLM-Lite

- 链接: https://github.com/vllm-project/vllm/pull/31386
- 状态/时间: merged / 2026-01-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/model_executor/models/glm4_moe_mtp.py`；关联提交 `71832ba71e77`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+1135/-1，可读 patch 1208 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[GLM-4.7] GLM Model support for GLM-Lite」；模型线: GLM-4.6/4.7；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/model_executor/models/glm4_moe_mtp.py`；技术摘要: 覆盖「[GLM-4.7] GLM Model support for GLM-Lite」；主要实现面是 `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/model_executor/models/glm4_moe_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe_lite.py` added +642/-0 (642 lines); hunks: -0,0 +1,642; symbols: Glm4MoeLiteMLP, Glm4MoeLite, Glm4LiteMixtureOfExperts, Glm4MoeLiteAttention，涉及 `Glm4MoeLiteMLP, Glm4MoeLite, Glm4LiteMixtureOfExperts`；`vllm/model_executor/models/glm4_moe_lite_mtp.py` added +464/-0 (464 lines); hunks: -0,0 +1,464; symbols: SharedHead, __init__, forward, Glm4MoeLiteMultiTokenPredictorLayer，涉及 `SharedHead, __init__, forward`；`vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-1 (3 lines); hunks: -21,7 +21,8。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe_lite.py` added +642/-0 (642 lines); hunks: -0,0 +1,642; symbols: Glm4MoeLiteMLP, Glm4MoeLite, Glm4LiteMixtureOfExperts, Glm4MoeLiteAttention
  - `vllm/model_executor/models/glm4_moe_lite_mtp.py` added +464/-0 (464 lines); hunks: -0,0 +1,464; symbols: SharedHead, __init__, forward, Glm4MoeLiteMultiTokenPredictorLayer
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-1 (3 lines); hunks: -21,7 +21,8
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
diff -- vllm/model_executor/models/glm4_moe_mtp.py
@@ -21,7 +21,8 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe_lite.py` added +642/-0; `vllm/model_executor/models/glm4_moe_lite_mtp.py` added +464/-0; `vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37386 - fix(glm47): improve tool call parsing and content normalization

- 链接: https://github.com/vllm-project/vllm/pull/37386
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`；关联提交 `fad09e8a1f51`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+193/-6，可读 patch 244 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(glm47): improve tool call parsing and content normalization」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`；技术摘要: 覆盖「fix(glm47): improve tool call parsing and content normalization」；主要实现面是 `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_glm47_moe_tool_parser.py` added +168/-0 (168 lines); hunks: -0,0 +1,168; symbols: glm47_tokenizer, glm47_tool_parser, mock_request, TestGlm47ExtractToolCalls，涉及 `glm47_tokenizer, glm47_tool_parser, mock_request`；`vllm/tool_parsers/glm47_moe_tool_parser.py` modified +16/-2 (18 lines); hunks: -1,6 +1,16; -14,10 +24,14; symbols: Glm47MoeModelToolParser, __init__，涉及 `Glm47MoeModelToolParser, __init__`；`tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +3/-3 (6 lines); hunks: -107,7 +107,7 @@ def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, m...; -152,7 +152,7 @@ def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, m...; symbols: test_extract_tool_calls_no_tools，涉及 `test_extract_tool_calls_no_tools`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_glm47_moe_tool_parser.py` added +168/-0 (168 lines); hunks: -0,0 +1,168; symbols: glm47_tokenizer, glm47_tool_parser, mock_request, TestGlm47ExtractToolCalls
  - `vllm/tool_parsers/glm47_moe_tool_parser.py` modified +16/-2 (18 lines); hunks: -1,6 +1,16; -14,10 +24,14; symbols: Glm47MoeModelToolParser, __init__
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
diff -- tests/tool_parsers/test_glm4_moe_tool_parser.py
@@ -107,7 +107,7 @@ def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, mock_request):
```

- 已读文件:
  - tests: `tests/tool_parsers/test_glm47_moe_tool_parser.py` added +168/-0; `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +3/-3
  - runtime: `vllm/tool_parsers/glm47_moe_tool_parser.py` modified +16/-2
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38029 - [Tool Parser][1/3] Pass tools to ToolParser constructor

- 链接: https://github.com/vllm-project/vllm/pull/38029
- 状态/时间: merged / 2026-03-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 38 个文件，+147/-92，可读 patch 858 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Tool Parser][1/3] Pass tools to ToolParser constructor」；模型线: GLM-4.6/4.7；类别: 模型实现调整；主要 diff: `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`；技术摘要: 覆盖「[Tool Parser][1/3] Pass tools to ToolParser constructor」；主要实现面是 `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Mypy] Fix adjust_request typing」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/deepseekv32_tool_parser.py`, `vllm/tool_parsers/functiongemma_tool_parser.py`, `vllm/tool_parsers/gigachat3_tool_parser.py`；技术摘要: 覆盖「[Mypy] Fix adjust_request typing」；主要实现面是 `vllm/tool_parsers/deepseekv32_tool_parser.py`, `vllm/tool_parsers/functiongemma_tool_parser.py`, `vllm/tool_parsers/gigachat3_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Tool Parser][2/3] Use self.tools instead of request.tools in tool parsers」；模型线: GLM-4.6/4.7；类别: 文档/测试/CI；主要 diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_qwen3coder_tool_parser.py`；技术摘要: 覆盖「[Tool Parser][2/3] Use self.tools instead of request.tools in tool parsers」；主要实现面是 `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_qwen3coder_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Misc] Add 20 regression tests for 11 tool parser bug fixes」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `tests/tool_parsers/test_step3p5_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`；技术摘要: 覆盖「[Misc] Add 20 regression tests for 11 tool parser bug fixes」；主要实现面是 `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `tests/tool_parsers/test_step3p5_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Bugfix] Fix GLM tool parser streaming with MTP or stream interval」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Fix GLM tool parser streaming with MTP or stream interval」；主要实现面是 `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[BugFix] Support custom tool parsers when tool_choice is `required` and named function」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`；技术摘要: 覆盖「[BugFix] Support custom tool parsers when tool_choice is `required` and named function」；主要实现面是 `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`；技术摘要: 覆盖「[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase」；主要实现面是 `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[MoE Refactor] Remove SharedFusedMoE class」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`；技术摘要: 覆盖「[MoE Refactor] Remove SharedFusedMoE class」；主要实现面是 `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`；技术摘要: 覆盖「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-2，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix GLM4-MoE weight loading for NVFP4 quantized checkpoints」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「[Bugfix] Fix GLM4-MoE weight loading for NVFP4 quantized checkpoints」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Bugfix] Preserve leading/trailing whitespace in GLM non-streaming tool parser」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Preserve leading/trailing whitespace in GLM non-streaming tool parser」；主要实现面是 `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+135/-25，可读 patch 214 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix glm4_moe_tool_parser._is_string_type for /v1/responses FunctionTool format」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Fix glm4_moe_tool_parser._is_string_type for /v1/responses FunctionTool format」；主要实现面是 `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +120/-0 (120 lines); hunks: -5,6 +5,7; -1363,3 +1364,122 @@ def test_stream_interval_content_between_tool_calls(; symbols: test_stream_interval_content_between_tool_calls, function_tools, glm4_moe_parser_function_tools, mock_request_function_tools，涉及 `test_stream_interval_content_between_tool_calls, function_tools, glm4_moe_parser_function_tools`；`vllm/tool_parsers/glm4_moe_tool_parser.py` modified +15/-25 (40 lines); hunks: -38,7 +38,11; -123,27 +127,13 @@ def _json_escape_string_content(s: str) -> str:; symbols: _json_escape_string_content, _is_string_type, _tools_enabled, extract_tool_calls，涉及 `_json_escape_string_content, _is_string_type, _tools_enabled`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +120/-0 (120 lines); hunks: -5,6 +5,7; -1363,3 +1364,122 @@ def test_stream_interval_content_between_tool_calls(; symbols: test_stream_interval_content_between_tool_calls, function_tools, glm4_moe_parser_function_tools, mock_request_function_tools
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +15/-25 (40 lines); hunks: -38,7 +38,11; -123,27 +127,13 @@ def _json_escape_string_content(s: str) -> str:; symbols: _json_escape_string_content, _is_string_type, _tools_enabled, extract_tool_calls
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
diff -- vllm/tool_parsers/glm4_moe_tool_parser.py
@@ -38,7 +38,11 @@
-from vllm.tool_parsers.utils import partial_tag_overlap
+from vllm.tool_parsers.utils import (
+    extract_types_from_schema,
+    find_tool_properties,
+    partial_tag_overlap,
+)
```

- 已读文件:
  - tests: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +120/-0
  - runtime: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +15/-25
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_glm4_moe_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #44346 - [Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers

- 链接: https://github.com/vllm-project/vllm/pull/44346
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+20/-15，可读 patch 178 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers」；模型线: GLM-4.6/4.7；类别: 模型实现调整；主要 diff: `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py`；技术摘要: 覆盖「[Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers」；主要实现面是 `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[MoE Refactor] FusedMoE/MoERunner inversion refactor」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`；技术摘要: 覆盖「[MoE Refactor] FusedMoE/MoERunner inversion refactor」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #45003 - [Frontend] Support strict mode for tool calling

- 链接: https://github.com/vllm-project/vllm/pull/45003
- 状态/时间: merged / 2026-06-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 29 个文件，+672/-1936，可读 patch 3162 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend] Support strict mode for tool calling」；模型线: GLM-4.6/4.7；类别: 文档/测试/CI；主要 diff: `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`；技术摘要: 覆盖「[Frontend] Support strict mode for tool calling」；主要实现面是 `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300 (1300 lines); hunks: -1,1300 +0,0; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks，涉及 `StreamingXMLToolCallParser, __init__, reset_streaming_state`；`vllm/tool_parsers/structural_tag_registry.py` modified +174/-240 (414 lines); hunks: -1,14 +1,15; -24,23 +25,51; symbols: register_model_structural_tag, register_vllm_structural_tag, decorator, get_model_structural_tag，涉及 `register_model_structural_tag, register_vllm_structural_tag, decorator`；`tests/tool_parsers/test_structural_tag_registry.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins, test_get_model_structural_tag_supports_vllm_hermes，涉及 `sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins`；`tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190 (203 lines); hunks: -3,6 +3,7; -19,15 +20,12; symbols: qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized, assert_tool_calls，涉及 `qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized`。
- 代码 diff 细节:
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300 (1300 lines); hunks: -1,1300 +0,0; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks
  - `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240 (414 lines); hunks: -1,14 +1,15; -24,23 +25,51; symbols: register_model_structural_tag, register_vllm_structural_tag, decorator, get_model_structural_tag
  - `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins, test_get_model_structural_tag_supports_vllm_hermes
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190 (203 lines); hunks: -3,6 +3,7; -19,15 +20,12; symbols: qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized, assert_tool_calls
  - `tests/tool_parsers/test_qwen3xml_tool_parser.py` removed +0/-72 (72 lines); hunks: -1,72 +0,0; symbols: TestQwen3xmlToolParser, test_config
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/qwen3xml_tool_parser.py
@@ -1,1300 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-import json
-from collections.abc import Sequence
-from typing import Any
-from xml.parsers.expat import ParserCreate
diff -- vllm/tool_parsers/structural_tag_registry.py
@@ -1,14 +1,15 @@
-# Model-specific structural tag builders adapted from XGrammar's
-# builtin structural tag implementations:
-# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/builtin_structural_tag.py
-from xgrammar import StructuralTag
+from xgrammar import StructuralTag, normalize_tool_choice
+from xgrammar import get_model_structural_tag as get_xgrammar_model_structural_tag
diff -- tests/tool_parsers/test_structural_tag_registry.py
@@ -0,0 +1,314 @@
```

- 已读文件:
  - runtime: `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300; `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240; `vllm/tool_parsers/abstract_tool_parser.py` modified +36/-28; `vllm/entrypoints/serve/render/serving.py` modified +24/-28; `vllm/tool_parsers/deepseekv4_tool_parser.py` modified +1/-15
  - tests: `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0; `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190; `tests/tool_parsers/test_qwen3xml_tool_parser.py` removed +0/-72
- 验证与风险: diff 自带测试面 `requirements/test/rocm.txt`, `tests/entrypoints/openai/chat_completion/test_completion_with_function_calling.py`, `tests/entrypoints/openai/responses/conftest.py`, `tests/parser/test_parse.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45915 - [Frontend] Add Streaming Parser Engine and new GLM4.7/GLM5.1/GLM5.2 Parser

- 链接: https://github.com/vllm-project/vllm/pull/45915
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+534/-1948，可读 patch 2693 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend] Add Streaming Parser Engine and new GLM4.7/GLM5.1/GLM5.2 Parser」；模型线: GLM-4.6/4.7；类别: 文档/测试/CI；主要 diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/reasoning/test_glm4_moe_reasoning_parser.py`；技术摘要: 覆盖「[Frontend] Add Streaming Parser Engine and new GLM4.7/GLM5.1/GLM5.2 Parser」；主要实现面是 `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`, `tests/reasoning/test_glm4_moe_reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Perf] Remove redundant clone for GLM, Deepseek etc」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`；技术摘要: 覆盖「[Perf] Remove redundant clone for GLM, Deepseek etc」；主要实现面是 `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
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
