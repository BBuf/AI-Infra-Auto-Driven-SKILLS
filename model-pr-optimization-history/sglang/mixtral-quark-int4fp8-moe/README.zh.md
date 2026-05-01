# sglang Mixtral Quark INT4/FP8 MoE 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/sglang/srt/layers/quantization/quark/__init__.py` | 无直接 PR 号提交 |
| `python/sglang/srt/layers/quantization/quark/quark.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485), [#13147](https://github.com/sgl-project/sglang/pull/13147), [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/schemes/__init__.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485), [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py` | [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485), [#18252](https://github.com/sgl-project/sglang/pull/18252), [#19422](https://github.com/sgl-project/sglang/pull/19422) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` | [#18252](https://github.com/sgl-project/sglang/pull/18252), [#18684](https://github.com/sgl-project/sglang/pull/18684), [#21040](https://github.com/sgl-project/sglang/pull/21040), [#21067](https://github.com/sgl-project/sglang/pull/21067), [#21097](https://github.com/sgl-project/sglang/pull/21097), [#23585](https://github.com/sgl-project/sglang/pull/23585), [#23597](https://github.com/sgl-project/sglang/pull/23597) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485), [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py` | [#18252](https://github.com/sgl-project/sglang/pull/18252) |
| `python/sglang/srt/layers/quantization/quark/utils.py` | [#10485](https://github.com/sgl-project/sglang/pull/10485) |
| `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` | [#7392](https://github.com/sgl-project/sglang/pull/7392), [#23597](https://github.com/sgl-project/sglang/pull/23597) |
| `python/sglang/srt/models/mixtral.py` | [#460](https://github.com/sgl-project/sglang/pull/460), [#1081](https://github.com/sgl-project/sglang/pull/1081), [#1290](https://github.com/sgl-project/sglang/pull/1290), [#1418](https://github.com/sgl-project/sglang/pull/1418), [#1835](https://github.com/sgl-project/sglang/pull/1835), [#2156](https://github.com/sgl-project/sglang/pull/2156), [#2163](https://github.com/sgl-project/sglang/pull/2163), [#2300](https://github.com/sgl-project/sglang/pull/2300), [#2371](https://github.com/sgl-project/sglang/pull/2371), [#2563](https://github.com/sgl-project/sglang/pull/2563), [#6223](https://github.com/sgl-project/sglang/pull/6223), [#7966](https://github.com/sgl-project/sglang/pull/7966), ... (17 total) |
| `python/sglang/srt/models/mixtral_quant.py` | [#460](https://github.com/sgl-project/sglang/pull/460), [#1081](https://github.com/sgl-project/sglang/pull/1081) |

## PR 覆盖总览

- git 追溯 PR 数: 28
- 原文档显式引用补充 PR 数: 5
- 当前文档总 PR 数: 33
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2024-05-21 | [#460](https://github.com/sgl-project/sglang/pull/460) | merged | port fp8 mixtral | `python/sglang/srt/models/mixtral_quant.py`, `python/sglang/srt/models/mixtral.py` |
| 2024-08-13 | [#1081](https://github.com/sgl-project/sglang/pull/1081) | merged | Update the mixtral to use the better FusedMoE layer | `python/sglang/srt/models/mixtral.py`, `python/sglang/srt/models/mixtral_quant.py` |
| 2024-09-01 | [#1290](https://github.com/sgl-project/sglang/pull/1290) | merged | fix: resolve fp8 for mixtral | `python/sglang/srt/models/mixtral.py` |
| 2024-09-09 | [#1341](https://github.com/sgl-project/sglang/pull/1341) | merged | Add torchao quant (int4/int8/fp8) to llama models | `python/sglang/srt/layers/torchao_utils.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2024-09-14 | [#1418](https://github.com/sgl-project/sglang/pull/1418) | merged | Add torchao quant for mixtral and qwen_moe | `python/sglang/srt/models/mixtral.py` |
| 2024-10-29 | [#1835](https://github.com/sgl-project/sglang/pull/1835) | merged | [FP8 KV Cache, Mixtral] Avoid KeyError at loading pre-quantized FP8 m… | `python/sglang/srt/models/mixtral.py` |
| 2024-11-24 | [#2156](https://github.com/sgl-project/sglang/pull/2156) | merged | feat: update other MoE models deps | `python/sglang/srt/models/mixtral.py` |
| 2024-11-24 | [#2163](https://github.com/sgl-project/sglang/pull/2163) | merged | Rename triton_fused_moe -> fused_moe_triton | `python/sglang/srt/models/mixtral.py` |
| 2024-12-03 | [#2300](https://github.com/sgl-project/sglang/pull/2300) | merged | Fix gptq for moe layers | `python/sglang/srt/models/mixtral.py` |
| 2024-12-05 | [#2203](https://github.com/sgl-project/sglang/pull/2203) | merged | MoE Expert Parallel Impl | `python/sglang/srt/layers/ep_moe/layer.py`, `python/sglang/srt/layers/ep_moe/kernels.py`, `python/sglang/srt/models/mixtral.py` |
| 2024-12-06 | [#2371](https://github.com/sgl-project/sglang/pull/2371) | merged | MoE Expert Parallel | `python/sglang/srt/models/mixtral.py` |
| 2024-12-23 | [#2563](https://github.com/sgl-project/sglang/pull/2563) | merged | Reorg moe code | `python/sglang/srt/models/mixtral.py` |
| 2025-05-12 | [#6238](https://github.com/sgl-project/sglang/pull/6238) | open | [Feature][ROCM] add online int4_fp8_moe quant feature | `python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py`, `python/sglang/srt/layers/quark_utils.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2025-05-12 | [#6223](https://github.com/sgl-project/sglang/pull/6223) | merged | [PP] Fix init_memory_pool desync & add PP for mixtral | `python/sglang/srt/models/mixtral.py` |
| 2025-07-19 | [#7966](https://github.com/sgl-project/sglang/pull/7966) | merged | [1/N] MoE Refactor: refactor `select_experts` | `python/sglang/srt/models/mixtral.py` |
| 2025-07-29 | [#8448](https://github.com/sgl-project/sglang/pull/8448) | merged | Support EPLB in FusedMoE | `python/sglang/srt/models/mixtral.py` |
| 2025-08-01 | [#8658](https://github.com/sgl-project/sglang/pull/8658) | merged | [5/N] MoE Refactor: Update MoE parallelism arguments | `python/sglang/srt/models/mixtral.py` |
| 2025-08-15 | [#8849](https://github.com/sgl-project/sglang/pull/8849) | merged | [6/N] MoE Refactor: Cleanup MoE-related configs | `python/sglang/srt/models/mixtral.py` |
| 2025-10-08 | [#11211](https://github.com/sgl-project/sglang/pull/11211) | merged | [8/N] MoE Refactor: deprecate `EPMoE` | `python/sglang/srt/models/mixtral.py` |
| 2025-11-13 | [#10485](https://github.com/sgl-project/sglang/pull/10485) | merged | [Quantization] Support Quark Dense + MoE FP8 & FP8 PTPC | `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2025-11-20 | [#13667](https://github.com/sgl-project/sglang/pull/13667) | merged | [Piecewise CUDA Graph] Fix recompile issue for Mixtral and Grok2 | `python/sglang/srt/models/mixtral.py` |
| 2025-12-09 | [#13147](https://github.com/sgl-project/sglang/pull/13147) | merged | Aiter fp8 kv cache | `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-01-14 | [#7392](https://github.com/sgl-project/sglang/pull/7392) | merged | [AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm | `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` |
| 2026-01-19 | [#17116](https://github.com/sgl-project/sglang/pull/17116) | merged | [AMD CI] Migrate and Add More Testcases | `.github/workflows/pr-test-amd.yml`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_v3_basic.py` |
| 2026-02-16 | [#17503](https://github.com/sgl-project/sglang/pull/17503) | merged | [2/N] Quantization Refactor: Compressed tensors MoE schemes | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` |
| 2026-02-18 | [#18252](https://github.com/sgl-project/sglang/pull/18252) | merged | [4/N] Quantization Refactor: Quark MoE schemes | `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-02-26 | [#19422](https://github.com/sgl-project/sglang/pull/19422) | merged | [AMD] Use fused GEMM with FP8 cast for FP8 prefill | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` |
| 2026-03-20 | [#18684](https://github.com/sgl-project/sglang/pull/18684) | merged | [AMD] Add MoE weights and scales padding | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-03-21 | [#21067](https://github.com/sgl-project/sglang/pull/21067) | merged | Revert "[AMD] Add MoE weights and scales padding" | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-03-25 | [#21040](https://github.com/sgl-project/sglang/pull/21040) | merged | [AMD][MoRI] Auto-select dispatch quantization type from MoE weight dtype. | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-04-13 | [#21097](https://github.com/sgl-project/sglang/pull/21097) | merged | [AMD] Add MoE weights and scales padding | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-04-24 | [#23585](https://github.com/sgl-project/sglang/pull/23585) | merged | Move expert_mask_gpu from FusedMoE layer to StandardDispatcher | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` |
| 2026-05-01 | [#23597](https://github.com/sgl-project/sglang/pull/23597) | merged | [MoE] Add Aiter MoE runner backend and purge aiter.fused_moe from quant methods | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` |

## 逐 PR diff 审计卡

### PR #460 - port fp8 mixtral

- 链接: https://github.com/sgl-project/sglang/pull/460
- 状态/时间: merged / 2024-05-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`, `python/sglang/srt/models/mixtral_quant.py`；关联提交 `0fafc5606b0d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+636/-121，可读 patch 921 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「port fp8 mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/mixtral_quant.py`, `python/sglang/srt/models/mixtral.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/models/mixtral_quant.py` added +371/-0 (371 lines); hunks: -0,0 +1,371; symbols: MixtralMLP, __init__, forward, MixtralMoE，涉及 `MixtralMLP, __init__, forward`；`python/sglang/srt/models/mixtral.py` modified +240/-101 (341 lines); hunks: -1,5 +1,5; -8,131 +8,226; symbols: MixtralMLP, __init__, forward, MixtralMoE，涉及 `MixtralMLP, __init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral_quant.py` added +371/-0 (371 lines); hunks: -0,0 +1,371; symbols: MixtralMLP, __init__, forward, MixtralMoE
  - `python/sglang/srt/models/mixtral.py` modified +240/-101 (341 lines); hunks: -1,5 +1,5; -8,131 +8,226; symbols: MixtralMLP, __init__, forward, MixtralMoE
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral_quant.py
@@ -0,0 +1,371 @@
+# Adapted from
+# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral_quant.py#L1
+"""Inference-only Mixtral model."""
+from typing import Iterable, Optional, Tuple
+import numpy as np
+import torch
diff -- python/sglang/srt/models/mixtral.py
@@ -1,5 +1,5 @@
-# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral_quant.py#L1
+# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral.py#L1
@@ -8,131 +8,226 @@
+from vllm import _custom_ops as ops
+from vllm.model_executor.layers.fused_moe import fused_moe
+from vllm.model_executor.layers.quantization.fp8 import Fp8Config
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral_quant.py` added +371/-0; `python/sglang/srt/models/mixtral.py` modified +240/-101
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/router/model_rpc.py`, `python/sglang/srt/managers/router/model_runner.py`, `python/sglang/srt/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1081 - Update the mixtral to use the better FusedMoE layer

- 链接: https://github.com/sgl-project/sglang/pull/1081
- 状态/时间: merged / 2024-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`, `python/sglang/srt/models/mixtral_quant.py`；关联提交 `ad3e4f16199a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+57/-258，可读 patch 515 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update the mixtral to use the better FusedMoE layer」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/mixtral.py`, `python/sglang/srt/models/mixtral_quant.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +55/-253 (308 lines); hunks: -18,34 +18,25; -69,216 +60,44 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading, forward，涉及 `__init__, weight_loader, process_weights_after_loading`；`python/sglang/srt/models/mixtral_quant.py` modified +0/-3 (3 lines); hunks: -160,7 +160,6 @@ def __init__(; -183,7 +182,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +55/-253 (308 lines); hunks: -18,34 +18,25; -69,216 +60,44 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading, forward
  - `python/sglang/srt/models/mixtral_quant.py` modified +0/-3 (3 lines); hunks: -160,7 +160,6 @@ def __init__(; -183,7 +182,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -18,34 +18,25 @@
-import numpy as np
-import torch.nn.functional as F
-from vllm import _custom_ops as ops
-from vllm.distributed import (
-    get_tensor_model_parallel_rank,
-    get_tensor_model_parallel_world_size,
diff -- python/sglang/srt/models/mixtral_quant.py
@@ -160,7 +160,6 @@ def __init__(
-        sliding_window: Optional[int] = None,
@@ -183,7 +182,6 @@ def __init__(
-        self.sliding_window = sliding_window
@@ -246,7 +244,6 @@ def __init__(
-            sliding_window=config.sliding_window,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +55/-253; `python/sglang/srt/models/mixtral_quant.py` modified +0/-3
- 验证与风险: diff 自带测试面 `test/srt/test_moe_serving_throughput.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1290 - fix: resolve fp8 for mixtral

- 链接: https://github.com/sgl-project/sglang/pull/1290
- 状态/时间: merged / 2024-09-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `9b0805242eea`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: resolve fp8 for mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/mixtral.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +1/-1 (2 lines); hunks: -362,7 +362,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +1/-1 (2 lines); hunks: -362,7 +362,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -362,7 +362,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-                        weight_name,
+                        name,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1341 - Add torchao quant (int4/int8/fp8) to llama models

- 链接: https://github.com/sgl-project/sglang/pull/1341
- 状态/时间: merged / 2024-09-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+151/-12，可读 patch 275 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add torchao quant (int4/int8/fp8) to llama models」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/torchao_utils.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/model_executor/model_runner.py`；技术摘要: 覆盖「Add torchao quant (int4/int8/fp8) to llama models」；主要实现面是 `python/sglang/srt/layers/torchao_utils.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/model_executor/model_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/torchao_utils.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: torchao_quantize_param_data，涉及 `torchao_quantize_param_data`；`python/sglang/srt/models/llama.py` modified +22/-0 (22 lines); hunks: -42,6 +42,8; -299,6 +301,7 @@ def __init__(; symbols: __init__, load_weights, Phi3ForCausalLM，涉及 `__init__, load_weights, Phi3ForCausalLM`；`python/sglang/srt/model_executor/model_runner.py` modified +1/-0 (1 lines); hunks: -97,6 +97,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/srt/test_torchao.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: TestTorchCompile, setUpClass, tearDownClass, test_mmlu，涉及 `TestTorchCompile, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/torchao_utils.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: torchao_quantize_param_data
  - `python/sglang/srt/models/llama.py` modified +22/-0 (22 lines); hunks: -42,6 +42,8; -299,6 +301,7 @@ def __init__(; symbols: __init__, load_weights, Phi3ForCausalLM
  - `python/sglang/srt/model_executor/model_runner.py` modified +1/-0 (1 lines); hunks: -97,6 +97,7 @@ def __init__(; symbols: __init__
  - `test/srt/test_torchao.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: TestTorchCompile, setUpClass, tearDownClass, test_mmlu
  - `python/sglang/srt/server_args.py` modified +8/-1 (9 lines); hunks: -95,6 +95,7 @@ class ServerArgs:; -443,7 +444,13 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/torchao_utils.py
@@ -0,0 +1,36 @@
+"""
+Common utilities for torchao.
+"""
+import torch
+from torchao.quantization import (
+    int4_weight_only,
diff -- python/sglang/srt/models/llama.py
@@ -42,6 +42,8 @@
+from sglang.srt.layers.torchao_utils import torchao_quantize_param_data
+from sglang.srt.managers.schedule_batch import global_server_args_dict
@@ -299,6 +301,7 @@ def __init__(
+        self.torchao_config = global_server_args_dict["torchao_config"]
@@ -361,6 +364,25 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                if self.torchao_config:
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -97,6 +97,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/torchao_utils.py` added +36/-0; `python/sglang/srt/models/llama.py` modified +22/-0; `python/sglang/srt/model_executor/model_runner.py` modified +1/-0; `python/sglang/srt/server_args.py` modified +8/-1
  - tests: `test/srt/test_torchao.py` added +73/-0; `test/srt/test_moe_eval_accuracy_large.py` modified +3/-3; `test/srt/test_torch_compile.py` modified +3/-3; `test/srt/test_eval_accuracy_mini.py` modified +2/-2
- 验证与风险: diff 自带测试面 `test/srt/test_eval_accuracy_mini.py`, `test/srt/test_moe_eval_accuracy_large.py`, `test/srt/test_torch_compile.py`, `test/srt/test_torchao.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1418 - Add torchao quant for mixtral and qwen_moe

- 链接: https://github.com/sgl-project/sglang/pull/1418
- 状态/时间: merged / 2024-09-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `30b404ce72b5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+50/-20，可读 patch 138 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add torchao quant for mixtral and qwen_moe」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/mixtral.py`；技术摘要: 覆盖「Add torchao quant for mixtral and qwen_moe」；主要实现面是 `python/sglang/srt/models/mixtral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +5/-0 (5 lines); hunks: -41,6 +41,8; -296,6 +298,7 @@ def __init__(; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +5/-0 (5 lines); hunks: -41,6 +41,8; -296,6 +298,7 @@ def __init__(; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -41,6 +41,8 @@
+from sglang.srt.layers.torchao_utils import apply_torchao_config_
+from sglang.srt.managers.schedule_batch import global_server_args_dict
@@ -296,6 +298,7 @@ def __init__(
+        self.torchao_config = global_server_args_dict["torchao_config"]
@@ -376,5 +379,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+        apply_torchao_config_(self, params_dict, set(["proj.weight"]))
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +5/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/torchao_utils.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1835 - [FP8 KV Cache, Mixtral] Avoid KeyError at loading pre-quantized FP8 m…

- 链接: https://github.com/sgl-project/sglang/pull/1835
- 状态/时间: merged / 2024-10-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `54dd3ea12277`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[FP8 KV Cache, Mixtral] Avoid KeyError at loading pre-quantized FP8 m…」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/mixtral.py`；技术摘要: 覆盖「[FP8 KV Cache, Mixtral] Avoid KeyError at loading pre-quantized FP8 m…」；主要实现面是 `python/sglang/srt/models/mixtral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +3/-0 (3 lines); hunks: -369,6 +369,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +3/-0 (3 lines); hunks: -369,6 +369,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -369,6 +369,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                    # Skip loading kv_scale from ckpts towards new design.
+                    if name.endswith(".kv_scale") and name not in params_dict:
+                        continue
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #2156 - feat: update other MoE models deps

- 链接: https://github.com/sgl-project/sglang/pull/2156
- 状态/时间: merged / 2024-11-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `e3938b2f9c96`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+28/-14，可读 patch 162 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: update other MoE models deps」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/mixtral.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +1/-1 (2 lines); hunks: -22,7 +22,6; -36,6 +35,7。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +1/-1 (2 lines); hunks: -22,7 +22,6; -36,6 +35,7
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -22,7 +22,6 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
@@ -36,6 +35,7 @@
+from sglang.srt.layers.triton_fused_moe import FusedMoE
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/fused_moe/layer.py`, `python/sglang/srt/layers/triton_fused_moe/fused_moe.py`, `python/sglang/srt/layers/triton_fused_moe/layer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #2163 - Rename triton_fused_moe -> fused_moe_triton

- 链接: https://github.com/sgl-project/sglang/pull/2163
- 状态/时间: merged / 2024-11-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `be0124bda09d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 76 个文件，+19/-19，可读 patch 199 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Rename triton_fused_moe -> fused_moe_triton」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/mixtral.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +1/-1 (2 lines); hunks: -25,6 +25,7; -35,7 +36,6。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +1/-1 (2 lines); hunks: -25,6 +25,7; -35,7 +36,6
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -25,6 +25,7 @@
+from sglang.srt.layers.fused_moe_triton import FusedMoE
@@ -35,7 +36,6 @@
-from sglang.srt.layers.triton_fused_moe import FusedMoE
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/fused_moe/__init__.py`, `python/sglang/srt/layers/fused_moe_grok/__init__.py`, `python/sglang/srt/layers/fused_moe_grok/configs/E=8,N=4096,device_name=AMD_Instinct_MI300X,dtype=float8.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #2300 - Fix gptq for moe layers

- 链接: https://github.com/sgl-project/sglang/pull/2300
- 状态/时间: merged / 2024-12-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `1228f7ca69e6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+44/-2，可读 patch 78 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix gptq for moe layers」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/mixtral.py`；技术摘要: 覆盖「Fix gptq for moe layers」；主要实现面是 `python/sglang/srt/models/mixtral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +10/-2 (12 lines); hunks: -339,7 +339,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; -353,6 +355,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +10/-2 (12 lines); hunks: -339,7 +339,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; -353,6 +355,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -339,7 +339,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-                if name.endswith(".bias") and name not in params_dict:
+                if (
+                    name.endswith(".bias") or name.endswith("_bias")
+                ) and name not in params_dict:
@@ -353,6 +355,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                    if (
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +10/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/__init__.py`, `python/sglang/srt/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #2203 - MoE Expert Parallel Impl

- 链接: https://github.com/sgl-project/sglang/pull/2203
- 状态/时间: merged / 2024-12-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+1172/-8，可读 patch 1320 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「MoE Expert Parallel Impl」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/ep_moe/layer.py`, `python/sglang/srt/layers/ep_moe/kernels.py`, `python/sglang/srt/models/mixtral.py`；技术摘要: 覆盖「MoE Expert Parallel Impl」；主要实现面是 `python/sglang/srt/layers/ep_moe/layer.py`, `python/sglang/srt/layers/ep_moe/kernels.py`, `python/sglang/srt/models/mixtral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/ep_moe/layer.py` added +661/-0 (661 lines); hunks: -0,0 +1,661; symbols: GroupedGemmRunner, __init__, _init_flashinfer_wrapper, forward，涉及 `GroupedGemmRunner, __init__, _init_flashinfer_wrapper`；`python/sglang/srt/layers/ep_moe/kernels.py` added +349/-0 (349 lines); hunks: -0,0 +1,349; symbols: compute_seg_indptr_triton_kernel, compute_src2dst_triton_kernel, run_moe_ep_preproess, pre_reorder_triton_kernel，涉及 `compute_seg_indptr_triton_kernel, compute_src2dst_triton_kernel, run_moe_ep_preproess`；`python/sglang/srt/models/mixtral.py` modified +13/-5 (18 lines); hunks: -21,9 +21,13; -38,6 +42,7; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`；`python/sglang/srt/models/deepseek_v2.py` modified +5/-3 (8 lines); hunks: -31,6 +31,7; -113,12 +114,12 @@ def __init__(; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/ep_moe/layer.py` added +661/-0 (661 lines); hunks: -0,0 +1,661; symbols: GroupedGemmRunner, __init__, _init_flashinfer_wrapper, forward
  - `python/sglang/srt/layers/ep_moe/kernels.py` added +349/-0 (349 lines); hunks: -0,0 +1,349; symbols: compute_seg_indptr_triton_kernel, compute_src2dst_triton_kernel, run_moe_ep_preproess, pre_reorder_triton_kernel
  - `python/sglang/srt/models/mixtral.py` modified +13/-5 (18 lines); hunks: -21,9 +21,13; -38,6 +42,7; symbols: __init__, forward, load_weights
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-3 (8 lines); hunks: -31,6 +31,7; -113,12 +114,12 @@ def __init__(; symbols: __init__, load_weights
  - `python/sglang/srt/model_executor/model_runner.py` modified +1/-0 (1 lines); hunks: -141,6 +141,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/ep_moe/layer.py
@@ -0,0 +1,661 @@
+import logging
+from typing import Callable, List, Optional, Tuple
+import torch
+from torch.nn import Module
+from vllm import _custom_ops as ops
+from vllm.distributed import (
diff -- python/sglang/srt/layers/ep_moe/kernels.py
@@ -0,0 +1,349 @@
+import logging
+from typing import Optional
+import torch
+import triton
+import triton.language as tl
+logger = logging.getLogger(__name__)
diff -- python/sglang/srt/models/mixtral.py
@@ -21,9 +21,13 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/ep_moe/layer.py` added +661/-0; `python/sglang/srt/layers/ep_moe/kernels.py` added +349/-0; `python/sglang/srt/models/mixtral.py` modified +13/-5; `python/sglang/srt/models/deepseek_v2.py` modified +5/-3; `python/sglang/srt/model_executor/model_runner.py` modified +1/-0; `python/sglang/srt/layers/ep_moe/__init__.py` added +0/-0
  - tests: `test/srt/test_moe_ep.py` added +113/-0
- 验证与风险: diff 自带测试面 `test/srt/test_moe_ep.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #2371 - MoE Expert Parallel

- 链接: https://github.com/sgl-project/sglang/pull/2371
- 状态/时间: merged / 2024-12-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `3d32e4a32c4c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+1172/-8，可读 patch 1320 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「MoE Expert Parallel」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/mixtral.py`；技术摘要: 覆盖「MoE Expert Parallel」；主要实现面是 `python/sglang/srt/models/mixtral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +13/-5 (18 lines); hunks: -21,9 +21,13; -38,6 +42,7; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +13/-5 (18 lines); hunks: -21,9 +21,13; -38,6 +42,7; symbols: __init__, forward, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -21,9 +21,13 @@
-from vllm.distributed import get_tensor_model_parallel_world_size
+from vllm.distributed import (
+    get_tensor_model_parallel_world_size,
+    tensor_model_parallel_all_reduce,
+)
+from sglang.srt.layers.ep_moe.layer import EPMoE
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +13/-5
- 验证与风险: diff 自带测试面 `test/srt/test_moe_ep.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #2563 - Reorg moe code

- 链接: https://github.com/sgl-project/sglang/pull/2563
- 状态/时间: merged / 2024-12-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `e835a50021e0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 88 个文件，+338/-344，可读 patch 1108 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Reorg moe code」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/mixtral.py`；技术摘要: 覆盖「Reorg moe code」；主要实现面是 `python/sglang/srt/models/mixtral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +2/-2 (4 lines); hunks: -27,15 +27,15。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +2/-2 (4 lines); hunks: -27,15 +27,15
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -27,15 +27,15 @@
-from sglang.srt.layers.ep_moe.layer import EPMoE
-from sglang.srt.layers.fused_moe_triton import FusedMoE
+from sglang.srt.layers.moe.ep_moe.layer import EPMoE
+from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +2/-2
- 验证与风险: diff 自带测试面 `test/srt/test_fused_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6238 - [Feature][ROCM] add online int4_fp8_moe quant feature

- 链接: https://github.com/sgl-project/sglang/pull/6238
- 状态/时间: open / 2025-05-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+651/-3，可读 patch 752 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature][ROCM] add online int4_fp8_moe quant feature」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py`, `python/sglang/srt/layers/quark_utils.py`, `python/sglang/srt/model_executor/model_runner.py`；技术摘要: 覆盖「[Feature][ROCM] add online int4_fp8_moe quant feature」；主要实现面是 `python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py`, `python/sglang/srt/layers/quark_utils.py`, `python/sglang/srt/model_executor/model_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py` added +524/-0 (524 lines); hunks: -0,0 +1,524; symbols: dummy_func, QuarkInt4Fp8Config, for, __init__，涉及 `dummy_func, QuarkInt4Fp8Config, for`；`python/sglang/srt/layers/quark_utils.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: apply_quark_quant_config_to_model, online_quant, quantize_fp8_scale_tensorwise, quantize_int4_scale_columnwise，涉及 `apply_quark_quant_config_to_model, online_quant, quantize_fp8_scale_tensorwise`；`python/sglang/srt/model_executor/model_runner.py` modified +6/-0 (6 lines); hunks: -49,6 +49,7; -168,6 +169,7 @@ def __init__(; symbols: __init__, initialize，涉及 `__init__, initialize`；`python/sglang/srt/layers/quantization/__init__.py` modified +2/-0 (2 lines); hunks: -66,6 +66,7 @@ def override_quantization_method(self, *args, **kwargs):; -77,6 +78,7 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method，涉及 `override_quantization_method`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py` added +524/-0 (524 lines); hunks: -0,0 +1,524; symbols: dummy_func, QuarkInt4Fp8Config, for, __init__
  - `python/sglang/srt/layers/quark_utils.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: apply_quark_quant_config_to_model, online_quant, quantize_fp8_scale_tensorwise, quantize_int4_scale_columnwise
  - `python/sglang/srt/model_executor/model_runner.py` modified +6/-0 (6 lines); hunks: -49,6 +49,7; -168,6 +169,7 @@ def __init__(; symbols: __init__, initialize
  - `python/sglang/srt/layers/quantization/__init__.py` modified +2/-0 (2 lines); hunks: -66,6 +66,7 @@ def override_quantization_method(self, *args, **kwargs):; -77,6 +78,7 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method
  - `python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunks: -317,6 +317,7 @@ def _verify_quantization(self) -> None:; symbols: _verify_quantization
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py
@@ -0,0 +1,524 @@
+import logging
+from typing import Any, Callable, Dict, List, Optional
+import torch
+from torch.nn.parameter import Parameter
+try:
+    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
diff -- python/sglang/srt/layers/quark_utils.py
@@ -0,0 +1,104 @@
+"""
+Common utilities for quark.
+"""
+import logging
+from tqdm.auto import tqdm
+import torch
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -49,6 +49,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/quark_w4a8_int4fp8.py` added +524/-0; `python/sglang/srt/layers/quark_utils.py` added +104/-0; `python/sglang/srt/model_executor/model_runner.py` modified +6/-0; `python/sglang/srt/layers/quantization/__init__.py` modified +2/-0; `python/sglang/srt/configs/model_config.py` modified +1/-0; `python/sglang/srt/layers/linear.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/quantization/__init__.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6223 - [PP] Fix init_memory_pool desync & add PP for mixtral

- 链接: https://github.com/sgl-project/sglang/pull/6223
- 状态/时间: merged / 2025-05-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `bad7c26fdc7f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+179/-47，可读 patch 391 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[PP] Fix init_memory_pool desync & add PP for mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/mixtral.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +98/-34 (132 lines); hunks: -16,13 +16,15; -38,14 +40,17; symbols: MixtralMoE, __init__, forward，涉及 `MixtralMoE, __init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +98/-34 (132 lines); hunks: -16,13 +16,15; -38,14 +40,17; symbols: MixtralMoE, __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -16,13 +16,15 @@
-from typing import Iterable, Optional, Tuple
+import logging
+from typing import Iterable, Optional, Tuple, Union
+    get_pp_group,
@@ -38,14 +40,17 @@
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +98/-34
- 验证与风险: diff 自带测试面 `test/srt/test_bench_serving.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7966 - [1/N] MoE Refactor: refactor `select_experts`

- 链接: https://github.com/sgl-project/sglang/pull/7966
- 状态/时间: merged / 2025-07-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `15ad6c908670`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 39 个文件，+557/-872，可读 patch 2848 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[1/N] MoE Refactor: refactor `select_experts`」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/mixtral.py`；技术摘要: 覆盖「[1/N] MoE Refactor: refactor `select_experts`」；主要实现面是 `python/sglang/srt/models/mixtral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +9/-2 (11 lines); hunks: -37,6 +37,7; -86,14 +87,19 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +9/-2 (11 lines); hunks: -37,6 +37,7; -86,14 +87,19 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -37,6 +37,7 @@
+from sglang.srt.layers.moe.topk import TopK
@@ -86,14 +87,19 @@ def __init__(
+        self.topk = TopK(
+            top_k=top_k,
+            renormalize=True,
+        )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +9/-2
- 验证与风险: diff 自带测试面 `python/sglang/test/test_block_fp8.py`, `python/sglang/test/test_block_fp8_ep.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`, `python/sglang/test/test_fp4_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8448 - Support EPLB in FusedMoE

- 链接: https://github.com/sgl-project/sglang/pull/8448
- 状态/时间: merged / 2025-07-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `9effeb5bddf2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+107/-11，可读 patch 407 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support EPLB in FusedMoE」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/mixtral.py`；技术摘要: 覆盖「Support EPLB in FusedMoE」；主要实现面是 `python/sglang/srt/models/mixtral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +3/-0 (3 lines); hunks: -69,6 +69,7 @@ def __init__(; -97,6 +98,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +3/-0 (3 lines); hunks: -69,6 +69,7 @@ def __init__(; -97,6 +98,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -69,6 +69,7 @@ def __init__(
+        layer_id: int,
@@ -97,6 +98,7 @@ def __init__(
+            layer_id=layer_id,
@@ -226,6 +228,7 @@ def __init__(
+            layer_id=layer_id,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/eplb/expert_distribution.py`, `python/sglang/srt/eplb/expert_location.py`, `python/sglang/srt/eplb/expert_location_dispatch.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8658 - [5/N] MoE Refactor: Update MoE parallelism arguments

- 链接: https://github.com/sgl-project/sglang/pull/8658
- 状态/时间: merged / 2025-08-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `6c88f6c8d908`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 38 个文件，+342/-299，可读 patch 1748 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[5/N] MoE Refactor: Update MoE parallelism arguments」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/mixtral.py`；技术摘要: 覆盖「[5/N] MoE Refactor: Update MoE parallelism arguments」；主要实现面是 `python/sglang/srt/models/mixtral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +3/-3 (6 lines); hunks: -24,6 +24,7; -94,7 +95,7 @@ def __init__(; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +3/-3 (6 lines); hunks: -24,6 +24,7; -94,7 +95,7 @@ def __init__(; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -24,6 +24,7 @@
+    get_moe_expert_parallel_world_size,
@@ -94,7 +95,7 @@ def __init__(
-        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
+        MoEImpl = EPMoE if get_moe_expert_parallel_world_size() > 1 else FusedMoE
@@ -398,8 +399,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +3/-3
- 验证与风险: diff 自带测试面 `python/sglang/test/runners.py`, `test/srt/test_deepep_large.py`, `test/srt/test_deepep_small.py`, `test/srt/test_eplb.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8849 - [6/N] MoE Refactor: Cleanup MoE-related configs

- 链接: https://github.com/sgl-project/sglang/pull/8849
- 状态/时间: merged / 2025-08-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `295895120df4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 69 个文件，+958/-1039，可读 patch 4640 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[6/N] MoE Refactor: Cleanup MoE-related configs」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/mixtral.py`；技术摘要: 覆盖「[6/N] MoE Refactor: Cleanup MoE-related configs」；主要实现面是 `python/sglang/srt/models/mixtral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +0/-2 (2 lines); hunks: -47,7 +47,6; -104,7 +103,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +0/-2 (2 lines); hunks: -47,7 +47,6; -104,7 +103,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -47,7 +47,6 @@
-from sglang.srt.managers.schedule_batch import global_server_args_dict
@@ -104,7 +103,6 @@ def __init__(
-            tp_size=tp_size,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +0/-2
- 验证与风险: diff 自带测试面 `python/sglang/test/test_block_fp8.py`, `python/sglang/test/test_block_fp8_ep.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`, `python/sglang/test/test_fp4_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11211 - [8/N] MoE Refactor: deprecate `EPMoE`

- 链接: https://github.com/sgl-project/sglang/pull/11211
- 状态/时间: merged / 2025-10-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `3c06b673aff9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+496/-1778，可读 patch 2897 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[8/N] MoE Refactor: deprecate `EPMoE`」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/mixtral.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +1/-3 (4 lines); hunks: -36,7 +36,6; -94,8 +93,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +1/-3 (4 lines); hunks: -36,7 +36,6; -94,8 +93,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -36,7 +36,6 @@
-from sglang.srt.layers.moe.ep_moe.layer import EPMoE
@@ -94,8 +93,7 @@ def __init__(
-        MoEImpl = EPMoE if get_moe_expert_parallel_world_size() > 1 else FusedMoE
-        self.experts = MoEImpl(
+        self.experts = FusedMoE(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +1/-3
- 验证与风险: diff 自带测试面 `python/sglang/test/test_block_fp8_ep.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`, `test/srt/ep/test_moe_ep.py`, `test/srt/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10485 - [Quantization] Support Quark Dense + MoE FP8 & FP8 PTPC

- 链接: https://github.com/sgl-project/sglang/pull/10485
- 状态/时间: merged / 2025-11-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/__init__.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `python/sglang/srt/layers/quantization/quark/utils.py`；关联提交 `67e9d287eea0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+666/-243，可读 patch 1041 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] Support Quark Dense + MoE FP8 & FP8 PTPC」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `python/sglang/srt/layers/quantization/quark/quark.py`；技术摘要: 覆盖「[Quantization] Support Quark Dense + MoE FP8 & FP8 PTPC」；主要实现面是 `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `python/sglang/srt/layers/quantization/quark/quark.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +116/-220 (336 lines); hunks: -604,158 +604,16 @@ def apply_fp8_linear(; -783,87 +641,125 @@ def apply_fp8_linear(; symbols: apply_fp8_linear, can_auto_enable_marlin_fp8，涉及 `apply_fp8_linear, can_auto_enable_marlin_fp8`；`python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py` added +186/-0 (186 lines); hunks: -0,0 +1,186; symbols: QuarkW8A8Fp8, __init__, get_min_capability, process_weights_after_loading，涉及 `QuarkW8A8Fp8, __init__, get_min_capability`；`python/sglang/srt/layers/quantization/quark/quark.py` modified +42/-1 (43 lines); hunks: -14,7 +14,11; -173,6 +177,37 @@ def _check_scheme_supported(self, min_capability: int, erro...; symbols: _check_scheme_supported, _is_fp8_w8a8, _is_mx_fp4, _get_scheme_from_config，涉及 `_check_scheme_supported, _is_fp8_w8a8, _is_mx_fp4`；`python/sglang/srt/layers/quantization/__init__.py` modified +3/-11 (14 lines); hunks: -35,6 +35,7 @@ def override_quantization_method(self, *args, **kwargs):; -65,23 +66,14 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method，涉及 `override_quantization_method`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +116/-220 (336 lines); hunks: -604,158 +604,16 @@ def apply_fp8_linear(; -783,87 +641,125 @@ def apply_fp8_linear(; symbols: apply_fp8_linear, can_auto_enable_marlin_fp8
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py` added +186/-0 (186 lines); hunks: -0,0 +1,186; symbols: QuarkW8A8Fp8, __init__, get_min_capability, process_weights_after_loading
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +42/-1 (43 lines); hunks: -14,7 +14,11; -173,6 +177,37 @@ def _check_scheme_supported(self, min_capability: int, erro...; symbols: _check_scheme_supported, _is_fp8_w8a8, _is_mx_fp4, _get_scheme_from_config
  - `python/sglang/srt/layers/quantization/__init__.py` modified +3/-11 (14 lines); hunks: -35,6 +35,7 @@ def override_quantization_method(self, *args, **kwargs):; -65,23 +66,14 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method
  - `python/sglang/srt/layers/quantization/quark/utils.py` modified +11/-1 (12 lines); hunks: -6,7 +6,17; symbols: raise_aiter_import_error
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_utils.py
@@ -604,158 +604,16 @@ def apply_fp8_linear(
-        # cutlass_scaled_mm supports per tensor/channel W and per tensor/token A
-        # for sgl-kernel fp8_scaled_mm, it support per channel W now
+        # Maybe apply padding to output, see comment in __init__
+        num_token_padding = output_padding
-            qinput, x_scale = scaled_fp8_quant(
-                input_2d,
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py
@@ -0,0 +1,186 @@
+# SPDX-License-Identifier: Apache-2.0
+from typing import Any, Callable, Optional, cast
+import torch
+from torch.nn import Parameter
+from sglang.srt.layers.parameter import (
+    ChannelQuantScaleParameter,
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -14,7 +14,11 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +116/-220; `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py` added +186/-0; `python/sglang/srt/layers/quantization/quark/quark.py` modified +42/-1; `python/sglang/srt/layers/quantization/__init__.py` modified +3/-11; `python/sglang/srt/layers/quantization/quark/utils.py` modified +11/-1; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +8/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/quantization/__init__.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13667 - [Piecewise CUDA Graph] Fix recompile issue for Mixtral and Grok2

- 链接: https://github.com/sgl-project/sglang/pull/13667
- 状态/时间: merged / 2025-11-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mixtral.py`；关联提交 `b5344b31b8f1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+24/-160，可读 patch 315 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Piecewise CUDA Graph] Fix recompile issue for Mixtral and Grok2」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/mixtral.py`；技术摘要: 覆盖「[Piecewise CUDA Graph] Fix recompile issue for Mixtral and Grok2」；主要实现面是 `python/sglang/srt/models/mixtral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mixtral.py` modified +1/-0 (1 lines); hunks: -353,6 +353,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mixtral.py` modified +1/-0 (1 lines); hunks: -353,6 +353,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mixtral.py
@@ -353,6 +353,7 @@ def __init__(
+    @torch.no_grad()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mixtral.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/grok.py`, `python/sglang/srt/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13147 - Aiter fp8 kv cache

- 链接: https://github.com/sgl-project/sglang/pull/13147
- 状态/时间: merged / 2025-12-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/quantization/quark/quark.py`；关联提交 `c106b54b57d8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+594/-96，可读 patch 1032 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Aiter fp8 kv cache」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/quark/quark.py`；技术摘要: 覆盖「Aiter fp8 kv cache」；主要实现面是 `python/sglang/srt/layers/quantization/quark/quark.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/quark/quark.py` modified +2/-0 (2 lines); hunks: -71,6 +71,8 @@ def get_quant_method(; symbols: get_quant_method，涉及 `get_quant_method`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +2/-0 (2 lines); hunks: -71,6 +71,8 @@ def get_quant_method(; symbols: get_quant_method
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -71,6 +71,8 @@ def get_quant_method(
+            elif isinstance(layer, RadixAttention):
+                return QuarkKVCacheMethod(self)
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/quark/quark.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/layers/quantization/quark/quark.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7392 - [AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm

- 链接: https://github.com/sgl-project/sglang/pull/7392
- 状态/时间: merged / 2026-01-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`；关联提交 `5af84c8af554`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+615/-15，可读 patch 759 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`；技术摘要: 覆盖「[AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm」；主要实现面是 `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` added +443/-0 (443 lines); hunks: -0,0 +1,443; symbols: tqdm_reset_no_print, QuarkInt4Fp8Config, for, __init__，涉及 `tqdm_reset_no_print, QuarkInt4Fp8Config, for`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` added +443/-0 (443 lines); hunks: -0,0 +1,443; symbols: tqdm_reset_no_print, QuarkInt4Fp8Config, for, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/quark_int4fp8_moe.py
@@ -0,0 +1,443 @@
+import logging
+from typing import TYPE_CHECKING, Any, Dict, List, Optional
+import torch
+from tqdm import tqdm
+from tqdm.std import EMA
+from sglang.srt.distributed import get_tensor_model_parallel_rank
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` added +443/-0
- 验证与风险: diff 自带测试面 `test/srt/run_suite.py`, `test/srt/test_int4fp8_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17116 - [AMD CI] Migrate and Add More Testcases

- 链接: https://github.com/sgl-project/sglang/pull/17116
- 状态/时间: merged / 2026-01-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+310/-66，可读 patch 596 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD CI] Migrate and Add More Testcases」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 文档/测试/CI；主要 diff: `.github/workflows/pr-test-amd.yml`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_v3_basic.py`；技术摘要: 覆盖「[AMD CI] Migrate and Add More Testcases」；主要实现面是 `.github/workflows/pr-test-amd.yml`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_v3_basic.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.github/workflows/pr-test-amd.yml` modified +81/-47 (128 lines); hunks: -149,7 +149,10 @@ jobs:; -190,7 +193,7 @@ jobs:；`test/registered/amd/test_deepseek_v3_mtp.py` added +116/-0 (116 lines); hunks: -0,0 +1,116; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV3MTP, setUpClass, tearDownClass`；`test/registered/amd/test_deepseek_v3_basic.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestDeepseekV3Basic, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV3Basic, setUpClass, tearDownClass`；`test/srt/run_suite.py` modified +0/-8 (8 lines); hunks: -91,10 +91,7; -103,15 +100,10。
- 代码 diff 细节:
  - `.github/workflows/pr-test-amd.yml` modified +81/-47 (128 lines); hunks: -149,7 +149,10 @@ jobs:; -190,7 +193,7 @@ jobs:
  - `test/registered/amd/test_deepseek_v3_mtp.py` added +116/-0 (116 lines); hunks: -0,0 +1,116; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/test_deepseek_v3_basic.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestDeepseekV3Basic, setUpClass, tearDownClass, test_a_gsm8k
  - `test/srt/run_suite.py` modified +0/-8 (8 lines); hunks: -91,10 +91,7; -103,15 +100,10
  - `test/registered/core/test_deterministic.py` modified +5/-1 (6 lines); hunks: -9,15 +9,18; -32,6 +35,7 @@ def get_server_args(cls):; symbols: TestFlashinferDeterministic, get_server_args, TestFa3Deterministic
- 关键代码摘录:

```diff
diff -- .github/workflows/pr-test-amd.yml
@@ -149,7 +149,10 @@ jobs:
+          docker exec -w /sglang-checkout/sgl-kernel/tests ci_sglang python3 -m pytest test_moe_topk_sigmoid.py
+          docker exec -w /sglang-checkout/sgl-kernel/tests ci_sglang python3 -m pytest test_torch_defaults_reset.py
+          docker exec -w /sglang-checkout/sgl-kernel/tests ci_sglang python3 -m pytest test_amd_deterministic_custom_allreduce.py
+          docker exec -w /sglang-checkout/sgl-kernel/tests ci_sglang python3 -m pytest test_amd_nccl_allreduce_determinism.py
@@ -190,7 +193,7 @@ jobs:
-          bash scripts/ci/amd_ci_exec.sh -w "/sglang-checkout/test" python3 run_suite.py --hw amd --suite stage-a-test-1
diff -- test/registered/amd/test_deepseek_v3_mtp.py
@@ -0,0 +1,116 @@
+import unittest
+from types import SimpleNamespace
+import requests
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_amd_ci
+from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
diff -- test/registered/amd/test_deepseek_v3_basic.py
@@ -0,0 +1,84 @@
```

- 已读文件:
  - ci: `.github/workflows/pr-test-amd.yml` modified +81/-47
  - tests: `test/registered/amd/test_deepseek_v3_mtp.py` added +116/-0; `test/registered/amd/test_deepseek_v3_basic.py` added +84/-0; `test/srt/run_suite.py` modified +0/-8; `test/registered/core/test_deterministic.py` modified +5/-1; `test/registered/hicache/test_hicache_storage_3fs_backend.py` modified +2/-1; `test/registered/hicache/test_hicache_storage_file_backend.py` modified +2/-1
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py`, `test/registered/amd/test_deepseek_v3_basic.py`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/attention/test_wave_attention_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17503 - [2/N] Quantization Refactor: Compressed tensors MoE schemes

- 链接: https://github.com/sgl-project/sglang/pull/17503
- 状态/时间: merged / 2026-02-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+2643/-2237，可读 patch 5144 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[2/N] Quantization Refactor: Compressed tensors MoE schemes」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`；技术摘要: 覆盖「[2/N] Quantization Refactor: Compressed tensors MoE schemes」；主要实现面是 `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` removed +0/-2190 (2190 lines); hunks: -1,2190 +0,0; symbols: GPTQMarlinState, CompressedTensorsMoEMethod, __new__, get_moe_method，涉及 `GPTQMarlinState, CompressedTensorsMoEMethod, __new__`；`python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` added +621/-0 (621 lines); hunks: -0,0 +1,621; symbols: GPTQMarlinState, CompressedTensorsWNA16MoE, __init__, get_min_capability，涉及 `GPTQMarlinState, CompressedTensorsWNA16MoE, __init__`；`python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` added +421/-0 (421 lines); hunks: -0,0 +1,421; symbols: CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability, create_weights，涉及 `CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability`；`python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py` added +384/-0 (384 lines); hunks: -0,0 +1,384; symbols: CompressedTensorsW8A8Fp8MoE, __init__, get_min_capability, create_weights，涉及 `CompressedTensorsW8A8Fp8MoE, __init__, get_min_capability`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` removed +0/-2190 (2190 lines); hunks: -1,2190 +0,0; symbols: GPTQMarlinState, CompressedTensorsMoEMethod, __new__, get_moe_method
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` added +621/-0 (621 lines); hunks: -0,0 +1,621; symbols: GPTQMarlinState, CompressedTensorsWNA16MoE, __init__, get_min_capability
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` added +421/-0 (421 lines); hunks: -0,0 +1,421; symbols: CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability, create_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py` added +384/-0 (384 lines); hunks: -0,0 +1,384; symbols: CompressedTensorsW8A8Fp8MoE, __init__, get_min_capability, create_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py` added +358/-0 (358 lines); hunks: -0,0 +1,358; symbols: CompressedTensorsMxInt4MoE, __init__, get_min_capability, create_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -1,2190 +0,0 @@
-# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
-# SPDX-License-Identifier: Apache-2.0
-from __future__ import annotations
-import enum
-import logging
-from enum import Enum
diff -- python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py
@@ -0,0 +1,621 @@
+from __future__ import annotations
+import enum
+import logging
+from enum import Enum
+from typing import TYPE_CHECKING
+import torch
diff -- python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py
@@ -0,0 +1,421 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` removed +0/-2190; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` added +621/-0; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` added +421/-0; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py` added +384/-0; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py` added +358/-0; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int8_moe.py` added +293/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/kt_ep_wrapper.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18252 - [4/N] Quantization Refactor: Quark MoE schemes

- 链接: https://github.com/sgl-project/sglang/pull/18252
- 状态/时间: merged / 2026-02-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/__init__.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` 等 7 个文件；关联提交 `150ed881be2c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+396/-243，可读 patch 835 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[4/N] Quantization Refactor: Quark MoE schemes」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/quark.py`；技术摘要: 覆盖「[4/N] Quantization Refactor: Quark MoE schemes」；主要实现面是 `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark/quark.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py` renamed +5/-221 (226 lines); hunks: -9,248 +9,32; -479,7 +263,7 @@ def create_moe_runner(; symbols: QuarkMoEMethod, __init__, get_moe_method, QuarkW4A4MXFp4MoEMethod，涉及 `QuarkMoEMethod, __init__, get_moe_method`；`python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` added +213/-0 (213 lines); hunks: -0,0 +1,213; symbols: QuarkW4A4MXFp4MoE, __init__, get_min_capability, create_weights，涉及 `QuarkW4A4MXFp4MoE, __init__, get_min_capability`；`python/sglang/srt/layers/quantization/quark/quark.py` modified +96/-10 (106 lines); hunks: -2,29 +2,36; -77,7 +84,7 @@ def get_quant_method(; symbols: get_quant_method, _find_matched_config, _get_scheme_from_config，涉及 `get_quant_method, _find_matched_config, _get_scheme_from_config`；`python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py` modified +65/-4 (69 lines); hunks: -1,14 +1,20; -30,6 +36,14 @@ def create_weights(self, *args, **kwargs):; symbols: QuarkScheme, QuarkLinearScheme, used, create_weights，涉及 `QuarkScheme, QuarkLinearScheme, used`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py` renamed +5/-221 (226 lines); hunks: -9,248 +9,32; -479,7 +263,7 @@ def create_moe_runner(; symbols: QuarkMoEMethod, __init__, get_moe_method, QuarkW4A4MXFp4MoEMethod
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` added +213/-0 (213 lines); hunks: -0,0 +1,213; symbols: QuarkW4A4MXFp4MoE, __init__, get_min_capability, create_weights
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +96/-10 (106 lines); hunks: -2,29 +2,36; -77,7 +84,7 @@ def get_quant_method(; symbols: get_quant_method, _find_matched_config, _get_scheme_from_config
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py` modified +65/-4 (69 lines); hunks: -1,14 +1,20; -30,6 +36,14 @@ def create_weights(self, *args, **kwargs):; symbols: QuarkScheme, QuarkLinearScheme, used, create_weights
  - `python/sglang/srt/layers/quantization/quark/schemes/__init__.py` modified +11/-2 (13 lines); hunks: -1,7 +1,16
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py
@@ -9,248 +9,32 @@
-from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
+from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme
-from sglang.srt.utils import (
-    get_bool_env_var,
-    is_gfx95_supported,
-    is_hip,
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -0,0 +1,213 @@
+# SPDX-License-Identifier: Apache-2.0
+from __future__ import annotations
+import logging
+from typing import TYPE_CHECKING, Any
+import torch
+from sglang.srt.layers.moe import MoeRunnerConfig
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -2,29 +2,36 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py` renamed +5/-221; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` added +213/-0; `python/sglang/srt/layers/quantization/quark/quark.py` modified +96/-10; `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py` modified +65/-4; `python/sglang/srt/layers/quantization/quark/schemes/__init__.py` modified +11/-2; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/layers/quantization/quark/schemes/__init__.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19422 - [AMD] Use fused GEMM with FP8 cast for FP8 prefill

- 链接: https://github.com/sgl-project/sglang/pull/19422
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`；关联提交 `5172c378456f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+73/-20，可读 patch 194 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Use fused GEMM with FP8 cast for FP8 prefill」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`；技术摘要: 覆盖「[AMD] Use fused GEMM with FP8 cast for FP8 prefill」；主要实现面是 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +30/-5 (35 lines); hunks: -10,6 +10,9; -87,20 +90,29 @@ def apply_weights(; symbols: apply_weights，涉及 `apply_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +30/-5 (35 lines); hunks: -10,6 +10,9; -87,20 +90,29 @@ def apply_weights(; symbols: apply_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py
@@ -10,6 +10,9 @@
+    from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import (
+        fused_gemm_afp4wfp4_split_cat,
+    )
@@ -87,20 +90,29 @@ def apply_weights(
+        fused_gemm_split_cat = False
-            ], "For tuple input, only (x, x_s) or (x, x_s, y) formats are accepted"
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +30/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18684 - [AMD] Add MoE weights and scales padding

- 链接: https://github.com/sgl-project/sglang/pull/18684
- 状态/时间: merged / 2026-03-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`；关联提交 `941945371314`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+131/-36，可读 patch 388 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add MoE weights and scales padding」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`；技术摘要: 覆盖「[AMD] Add MoE weights and scales padding」；主要实现面是 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +23/-5 (28 lines); hunks: -8,6 +8,7; -73,10 +74,20 @@ def create_weights(; symbols: create_weights，涉及 `create_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +23/-5 (28 lines); hunks: -8,6 +8,7; -73,10 +74,20 @@ def create_weights(; symbols: create_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -8,6 +8,7 @@
+from sglang.srt.layers.moe.utils import get_moe_weight_sizes
@@ -73,10 +74,20 @@ def create_weights(
+        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
+            intermediate_size_per_partition,
+            is_aiter_moe=True,
+            is_concat=True,
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +23/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21067 - Revert "[AMD] Add MoE weights and scales padding"

- 链接: https://github.com/sgl-project/sglang/pull/21067
- 状态/时间: merged / 2026-03-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`；关联提交 `048d90e1651a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+36/-131，可读 patch 388 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "[AMD] Add MoE weights and scales padding"」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`；技术摘要: 覆盖「Revert "[AMD] Add MoE weights and scales padding"」；主要实现面是 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +5/-23 (28 lines); hunks: -8,7 +8,6; -74,20 +73,10 @@ def create_weights(; symbols: create_weights，涉及 `create_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +5/-23 (28 lines); hunks: -8,7 +8,6; -74,20 +73,10 @@ def create_weights(; symbols: create_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -8,7 +8,6 @@
-from sglang.srt.layers.moe.utils import get_moe_weight_sizes
@@ -74,20 +73,10 @@ def create_weights(
-        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
-            intermediate_size_per_partition,
-            is_aiter_moe=True,
-            is_concat=True,
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +5/-23
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21040 - [AMD][MoRI] Auto-select dispatch quantization type from MoE weight dtype.

- 链接: https://github.com/sgl-project/sglang/pull/21040
- 状态/时间: merged / 2026-03-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`；关联提交 `61a902ce88ea`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+90/-54，可读 patch 331 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][MoRI] Auto-select dispatch quantization type from MoE weight dtype.」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`；技术摘要: 覆盖「[AMD][MoRI] Auto-select dispatch quantization type from MoE weight dtype.」；主要实现面是 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +4/-0 (4 lines); hunks: -160,6 +160,10 @@ def process_weights_after_loading(self, layer: torch.nn.Mod...; symbols: process_weights_after_loading, create_moe_runner，涉及 `process_weights_after_loading, create_moe_runner`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +4/-0 (4 lines); hunks: -160,6 +160,10 @@ def process_weights_after_loading(self, layer: torch.nn.Mod...; symbols: process_weights_after_loading, create_moe_runner
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -160,6 +160,10 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
+        if hasattr(layer, "dispatcher"):
+            # Weights are stored as torch.uint8 but semantically MXFP4
+            layer.dispatcher.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +4/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_moriep_small.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21097 - [AMD] Add MoE weights and scales padding

- 链接: https://github.com/sgl-project/sglang/pull/21097
- 状态/时间: merged / 2026-04-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`；关联提交 `f4f9e6818916`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+153/-46，可读 patch 432 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add MoE weights and scales padding」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`；技术摘要: 覆盖「[AMD] Add MoE weights and scales padding」；主要实现面是 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +20/-5 (25 lines); hunks: -8,6 +8,7; -73,10 +74,20 @@ def create_weights(; symbols: create_weights，涉及 `create_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +20/-5 (25 lines); hunks: -8,6 +8,7; -73,10 +74,20 @@ def create_weights(; symbols: create_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -8,6 +8,7 @@
+from sglang.srt.layers.moe.utils import get_moe_weight_sizes
@@ -73,10 +74,20 @@ def create_weights(
+        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
+            intermediate_size_per_partition,
+            is_aiter_moe=_use_aiter,
+            is_concat=True,
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +20/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23585 - Move expert_mask_gpu from FusedMoE layer to StandardDispatcher

- 链接: https://github.com/sgl-project/sglang/pull/23585
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`；关联提交 `000a2525e196`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+25/-25，可读 patch 124 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Move expert_mask_gpu from FusedMoE layer to StandardDispatcher」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`；技术摘要: 覆盖「Move expert_mask_gpu from FusedMoE layer to StandardDispatcher」；主要实现面是 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +1/-1 (2 lines); hunks: -227,6 +227,6 @@ def apply_weights(; symbols: apply_weights，涉及 `apply_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +1/-1 (2 lines); hunks: -227,6 +227,6 @@ def apply_weights(; symbols: apply_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -227,6 +227,6 @@ def apply_weights(
-            expert_mask=layer.expert_mask_gpu,
+            expert_mask=layer.dispatcher.expert_mask_gpu,
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/token_dispatcher/standard.py`, `python/sglang/srt/layers/quantization/fp8.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23597 - [MoE] Add Aiter MoE runner backend and purge aiter.fused_moe from quant methods

- 链接: https://github.com/sgl-project/sglang/pull/23597
- 状态/时间: merged / 2026-05-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`；关联提交 `108bfd8b6a0d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+316/-251，可读 patch 873 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE] Add Aiter MoE runner backend and purge aiter.fused_moe from quant methods」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`；技术摘要: 覆盖「[MoE] Add Aiter MoE runner backend and purge aiter.fused_moe from quant methods」；主要实现面是 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +25/-29 (54 lines); hunks: -7,7 +7,7; -32,8 +32,6; symbols: process_weights_after_loading, create_moe_runner, apply_weights，涉及 `process_weights_after_loading, create_moe_runner, apply_weights`；`python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` modified +25/-22 (47 lines); hunks: -11,7 +11,7; -27,8 +27,6; symbols: process_weights_after_loading, create_moe_runner, apply，涉及 `process_weights_after_loading, create_moe_runner, apply`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +25/-29 (54 lines); hunks: -7,7 +7,7; -32,8 +32,6; symbols: process_weights_after_loading, create_moe_runner, apply_weights
  - `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` modified +25/-22 (47 lines); hunks: -11,7 +11,7; -27,8 +27,6; symbols: process_weights_after_loading, create_moe_runner, apply
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -7,7 +7,7 @@
-from sglang.srt.layers.moe import MoeRunnerConfig
+from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
@@ -32,8 +32,6 @@
-    from aiter import ActivationType, QuantType
-    from aiter.fused_moe import fused_moe
@@ -182,24 +180,31 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
diff -- python/sglang/srt/layers/quantization/quark_int4fp8_moe.py
@@ -11,7 +11,7 @@
-from sglang.srt.layers.moe import MoeRunnerConfig
+from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
@@ -27,8 +27,6 @@
-    from aiter import ActivationType, QuantType
-    from aiter.fused_moe import fused_moe
@@ -405,39 +403,44 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +25/-29; `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` modified +25/-22
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/moe_runner/aiter.py`, `python/sglang/srt/layers/moe/moe_runner/runner.py`, `python/sglang/srt/layers/moe/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
