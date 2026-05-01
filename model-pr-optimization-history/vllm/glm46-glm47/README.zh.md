# vllm GLM-4.6/4.7 模型 PR 优化历史

## 覆盖范围

- 重做日期: 2026-05-01
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `7075df79b3`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

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
| `vllm/tool_parsers/glm47_moe_tool_parser.py` | [#30876](https://github.com/vllm-project/vllm/pull/30876), [#37386](https://github.com/vllm-project/vllm/pull/37386) |
| `vllm/tool_parsers/glm4_moe_tool_parser.py` | [#31622](https://github.com/vllm-project/vllm/pull/31622), [#37386](https://github.com/vllm-project/vllm/pull/37386) |

## PR 覆盖总览

- git 追溯 PR 数: 5
- 原文档显式引用补充 PR 数: 2
- 当前文档总 PR 数: 7
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-10-14 | [#26818](https://github.com/vllm-project/vllm/pull/26818) | merged | [Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200 | `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` |
| 2025-11-12 | [#27597](https://github.com/vllm-project/vllm/pull/27597) | merged | [Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint. | `vllm/model_executor/models/glm4_moe_mtp.py` |
| 2025-12-09 | [#30210](https://github.com/vllm-project/vllm/pull/30210) | merged | [Bugfix]: Fix glm46 awq marlin moe wna16 compatibility | `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py` |
| 2025-12-20 | [#30876](https://github.com/vllm-project/vllm/pull/30876) | merged | GLM-4.7 Tool Parser and Doc Update | `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/model_executor/models/glm4_moe.py` |
| 2026-01-05 | [#31622](https://github.com/vllm-project/vllm/pull/31622) | merged | Fix GLM-4.6v flash tool calling in transformers 5.x | `vllm/tool_parsers/glm4_moe_tool_parser.py` |
| 2026-01-19 | [#31386](https://github.com/vllm-project/vllm/pull/31386) | merged | [GLM-4.7] GLM Model support for GLM-Lite | `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/model_executor/models/glm4_moe_mtp.py` |
| 2026-03-18 | [#37386](https://github.com/vllm-project/vllm/pull/37386) | merged | fix(glm47): improve tool call parsing and content normalization | `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py` |

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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tool_parsers/glm4_moe_tool_parser.py`；关联提交 `02dbb933cb28`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+68/-0，可读 patch 76 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GLM-4.6v flash tool calling in transformers 5.x」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/glm4_moe_tool_parser.py`；技术摘要: 覆盖「Fix GLM-4.6v flash tool calling in transformers 5.x」；主要实现面是 `vllm/tool_parsers/glm4_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +14/-0 (14 lines); hunks: -56,6 +56,20 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, adjust_request, extract_tool_calls，涉及 `__init__, adjust_request, extract_tool_calls`。
- 代码 diff 细节:
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +14/-0 (14 lines); hunks: -56,6 +56,20 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, adjust_request, extract_tool_calls
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
```

- 已读文件:
  - runtime: `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +14/-0
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`；关联提交 `fad09e8a1f51`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+193/-6，可读 patch 244 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(glm47): improve tool call parsing and content normalization」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`；技术摘要: 覆盖「fix(glm47): improve tool call parsing and content normalization」；主要实现面是 `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
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

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
