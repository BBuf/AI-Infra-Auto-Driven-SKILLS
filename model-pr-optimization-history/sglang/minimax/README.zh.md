# sglang MiniMax M2 Series 模型 PR 优化历史

## 覆盖范围

- 重做日期: 2026-05-01
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `4197c55968`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/basic_usage/minimax_m2.md` | [#15538](https://github.com/sgl-project/sglang/pull/15538), [#19443](https://github.com/sgl-project/sglang/pull/19443) |
| `docs_new/cookbook/autoregressive/MiniMax/MiniMax-M2.5.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/MiniMax/MiniMax-M2.7.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/MiniMax/MiniMax-M2.mdx` | 无直接 PR 号提交 |
| `docs_new/docs/basic_usage/minimax_m2.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/minimax-m2-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/minimax-m25-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/minimax-m27-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/function_call/minimax_m2.py` | [#12129](https://github.com/sgl-project/sglang/pull/12129), [#15538](https://github.com/sgl-project/sglang/pull/15538) |
| `python/sglang/srt/models/minimax_m2.py` | [#12129](https://github.com/sgl-project/sglang/pull/12129), [#12798](https://github.com/sgl-project/sglang/pull/12798), [#13297](https://github.com/sgl-project/sglang/pull/13297), [#13659](https://github.com/sgl-project/sglang/pull/13659), [#13892](https://github.com/sgl-project/sglang/pull/13892), [#14047](https://github.com/sgl-project/sglang/pull/14047), [#14416](https://github.com/sgl-project/sglang/pull/14416), [#16483](https://github.com/sgl-project/sglang/pull/16483), [#18217](https://github.com/sgl-project/sglang/pull/18217), [#19577](https://github.com/sgl-project/sglang/pull/19577), [#19995](https://github.com/sgl-project/sglang/pull/19995), [#20067](https://github.com/sgl-project/sglang/pull/20067), ... (20 total) |
| `test/registered/8-gpu-models/test_minimax_m25.py` | [#20067](https://github.com/sgl-project/sglang/pull/20067), [#20083](https://github.com/sgl-project/sglang/pull/20083) |
| `test/registered/8-gpu-models/test_minimax_m25_basic.py` | [#21792](https://github.com/sgl-project/sglang/pull/21792) |
| `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py` | [#19443](https://github.com/sgl-project/sglang/pull/19443) |
| `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py` | [#22722](https://github.com/sgl-project/sglang/pull/22722) |
| `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py` | [#19443](https://github.com/sgl-project/sglang/pull/19443) |
| `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py` | [#22722](https://github.com/sgl-project/sglang/pull/22722) |
| `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py` | [#21524](https://github.com/sgl-project/sglang/pull/21524) |
| `test/registered/amd/perf/mi30x/test_minimax_m27_perf_amd.py` | [#22722](https://github.com/sgl-project/sglang/pull/22722) |
| `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py` | [#21524](https://github.com/sgl-project/sglang/pull/21524) |
| `test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py` | [#22722](https://github.com/sgl-project/sglang/pull/22722) |
| `test/registered/ascend/llm_models/test_ascend_minimax_m2.py` | [#17695](https://github.com/sgl-project/sglang/pull/17695) |

## PR 覆盖总览

- git 追溯 PR 数: 26
- 原文档显式引用补充 PR 数: 15
- 当前文档总 PR 数: 41
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-10-26 | [#12129](https://github.com/sgl-project/sglang/pull/12129) | merged | Support MiniMax M2 model | `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py` |
| 2025-10-27 | [#12186](https://github.com/sgl-project/sglang/pull/12186) | merged | improve mimax-m2 rmsnorm precision | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-08 | [#12798](https://github.com/sgl-project/sglang/pull/12798) | merged | Support capturing aux_hidden_states for minimax m2. | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-15 | [#13297](https://github.com/sgl-project/sglang/pull/13297) | merged | Fix: add missing get_embed_and_head in MiniMax M2 for Eagle3 | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-20 | [#13659](https://github.com/sgl-project/sglang/pull/13659) | merged | Super tiny remove unused MiniMaxM2MLP class | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-26 | [#13892](https://github.com/sgl-project/sglang/pull/13892) | merged | fix: correct usage of minimax-m2 deepep moe forward | `python/sglang/srt/models/minimax_m2.py` |
| 2025-12-02 | [#14047](https://github.com/sgl-project/sglang/pull/14047) | merged | Optimize topk sigmoid in minimax_m2 | `python/sglang/srt/models/minimax_m2.py` |
| 2025-12-23 | [#15538](https://github.com/sgl-project/sglang/pull/15538) | merged | Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs | `python/sglang/srt/function_call/minimax_m2.py`, `docs/basic_usage/minimax_m2.md` |
| 2025-12-30 | [#14416](https://github.com/sgl-project/sglang/pull/14416) | merged | Fusing RMSNormTP in minimax_m2 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-01-27 | [#17826](https://github.com/sgl-project/sglang/pull/17826) | open | Support Pipeline and Data Parallelism for MiniMax-M2 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-02-01 | [#16483](https://github.com/sgl-project/sglang/pull/16483) | merged | Optimizing all_reduce in RMSNormTP in minimax_m2 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-02-05 | [#18217](https://github.com/sgl-project/sglang/pull/18217) | merged | [piecewise graph]: support MiniMax-M2 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-02-05 | [#18310](https://github.com/sgl-project/sglang/pull/18310) | open | [Fix] MiniMax-M2.1 CUDA Graph + torch.compile crashes due to outplace_all_reduce being traced by Dynamo | `python/sglang/srt/distributed/parallel_state.py` |
| 2026-02-27 | [#19468](https://github.com/sgl-project/sglang/pull/19468) | open | fix[minimax]: support deepep with minimax models | `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh` |
| 2026-02-27 | [#19443](https://github.com/sgl-project/sglang/pull/19443) | merged | [AMD] [MiniMax-M2.5 Day 0] Add MiniMax-M2.5 nightly accuracy test | `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py`, `docs/basic_usage/minimax_m2.md` |
| 2026-03-02 | [#19577](https://github.com/sgl-project/sglang/pull/19577) | merged | [Feat] add PP Support for minimax-m2 series | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-06 | [#20031](https://github.com/sgl-project/sglang/pull/20031) | open | fix(minimax): support loading merged expert weights (w13) for awq | `tests/registered/models/test_minimax_m2_weights.py`, `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-07 | [#20083](https://github.com/sgl-project/sglang/pull/20083) | merged | [Nightly] Replace MiniMax-M2 with MiniMax-M2.5 | `test/registered/8-gpu-models/test_minimax_m25.py` |
| 2026-03-13 | [#20489](https://github.com/sgl-project/sglang/pull/20489) | open | fix(dp-attn): fix issues with dp-attention for MiniMax M2 and general… | `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/rotary_embedding/base.py` |
| 2026-03-18 | [#20873](https://github.com/sgl-project/sglang/pull/20873) | open | docs: add MiniMax-M2.7 and M2.7-highspeed model support | `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md` |
| 2026-03-18 | [#19995](https://github.com/sgl-project/sglang/pull/19995) | merged | Add packed_modules_mapping for MiniMax-M2 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-18 | [#20870](https://github.com/sgl-project/sglang/pull/20870) | merged | [MiniMax M2] Fix KV cache scale loading | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-20 | [#20975](https://github.com/sgl-project/sglang/pull/20975) | open | fix(dp-attn): fix issues with dp-attention for MiniMax M2 | `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2026-03-20 | [#20931](https://github.com/sgl-project/sglang/pull/20931) | merged | [Bugifx] qwen3 rope parameter compatibility | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-23 | [#17695](https://github.com/sgl-project/sglang/pull/17695) | merged | [NPU] enhance accuracy for model minimaxm2 from 16.5% to 95.5% | `test/registered/ascend/llm_models/test_ascend_minimax_m2.py` |
| 2026-03-24 | [#20905](https://github.com/sgl-project/sglang/pull/20905) | merged | [NPU][ModelSlim] adapt w2 quant layer for Minimax2.5 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-31 | [#21241](https://github.com/sgl-project/sglang/pull/21241) | merged | [bugfix] Fix rope theta config for MiniMax after transformers v5 update | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-03 | [#19652](https://github.com/sgl-project/sglang/pull/19652) | merged | [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+) | `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` |
| 2026-04-03 | [#21524](https://github.com/sgl-project/sglang/pull/21524) | merged | [AMD] Add MiniMax-M2.5 nightly perf benchmarks for MI30x and MI35x | `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py` |
| 2026-04-06 | [#21792](https://github.com/sgl-project/sglang/pull/21792) | merged | [CI] Add basic unit test for Minimax-M2.5 | `test/registered/8-gpu-models/test_minimax_m25_basic.py` |
| 2026-04-07 | [#20919](https://github.com/sgl-project/sglang/pull/20919) | merged | [NPU] Support dp-attention for MiniMax2.5 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-08 | [#22300](https://github.com/sgl-project/sglang/pull/22300) | open | [NVIDIA] Fix FP8 gemm performance with fp16 models (MInimax-M2.5) | `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/model_loader/utils.py` |
| 2026-04-10 | [#20967](https://github.com/sgl-project/sglang/pull/20967) | merged | 【BugFix】fix the bug of minimax_m2.5 model that causes repeated outputs when using tp16 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-10 | [#20067](https://github.com/sgl-project/sglang/pull/20067) | merged | MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn | `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py` |
| 2026-04-13 | [#20673](https://github.com/sgl-project/sglang/pull/20673) | merged | [Feature][JIT Kernel] Fused TP QK norm For Minimax | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-14 | [#22744](https://github.com/sgl-project/sglang/pull/22744) | open | [NVIDIA] Support TF32 matmul to improve MiniMax gate gemm performance | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md` |
| 2026-04-14 | [#22722](https://github.com/sgl-project/sglang/pull/22722) | merged | [AMD] Add MiniMax-M2.7 accuracy and performance nightly tests | `python/sglang/srt/models/minimax_m2.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py` |
| 2026-04-16 | [#22934](https://github.com/sgl-project/sglang/pull/22934) | open | Minimax eplb bugfix | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-21 | [#23301](https://github.com/sgl-project/sglang/pull/23301) | open | [sgl] Stream MiniMax M2 string parameters token-by-token | `python/sglang/srt/function_call/minimax_m2.py` |
| 2026-04-27 | [#22432](https://github.com/sgl-project/sglang/pull/22432) | closed | [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-30 | [#23190](https://github.com/sgl-project/sglang/pull/23190) | merged | [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode | `python/sglang/srt/models/minimax_m2.py` |

## 逐 PR diff 审计卡

### PR #12129 - Support MiniMax M2 model

- 链接: https://github.com/sgl-project/sglang/pull/12129
- 状态/时间: merged / 2025-10-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/minimax_m2.py`, `python/sglang/srt/models/minimax_m2.py`；关联提交 `7ebc28f5d657`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+1320/-1，可读 patch 1365 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support MiniMax M2 model」；模型线: MiniMax M2 Series；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py`；技术摘要: 覆盖「Support MiniMax M2 model」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` added +922/-0 (922 lines); hunks: -0,0 +1,922; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, forward，涉及 `MiniMaxM2RMSNormTP, __init__, weight_loader`；`python/sglang/srt/function_call/minimax_m2.py` added +367/-0 (367 lines); hunks: -0,0 +1,367; symbols: _safe_val, MinimaxM2Detector, __init__, has_tool_call，涉及 `_safe_val, MinimaxM2Detector, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` added +922/-0 (922 lines); hunks: -0,0 +1,922; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, forward
  - `python/sglang/srt/function_call/minimax_m2.py` added +367/-0 (367 lines); hunks: -0,0 +1,367; symbols: _safe_val, MinimaxM2Detector, __init__, has_tool_call
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -0,0 +1,922 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/function_call/minimax_m2.py
@@ -0,0 +1,367 @@
+import ast
+import html
+import json
+import logging
+import re
+from typing import Any, Dict, List, Tuple
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` added +922/-0; `python/sglang/srt/function_call/minimax_m2.py` added +367/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/function_call/minimax_m2.py`, `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12186 - improve mimax-m2 rmsnorm precision

- 链接: https://github.com/sgl-project/sglang/pull/12186
- 状态/时间: merged / 2025-10-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「improve mimax-m2 rmsnorm precision」；模型线: MiniMax M2 Series；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「improve mimax-m2 rmsnorm precision」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -122,7 +122,7 @@ def forward(
-        x = x.to(orig_dtype) * self.weight
+        x = (x * self.weight).to(orig_dtype)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12798 - Support capturing aux_hidden_states for minimax m2.

- 链接: https://github.com/sgl-project/sglang/pull/12798
- 状态/时间: merged / 2025-11-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `f1a9c72de3c1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+34/-3，可读 patch 90 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support capturing aux_hidden_states for minimax m2.」；模型线: MiniMax M2 Series；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「Support capturing aux_hidden_states for minimax m2.」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +34/-3 (37 lines); hunks: -706,6 +706,9 @@ def layer_fn(idx, prefix: str) -> nn.Module:; -716,7 +719,7 @@ def forward(; symbols: layer_fn, get_input_embeddings, forward，涉及 `layer_fn, get_input_embeddings, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +34/-3 (37 lines); hunks: -706,6 +706,9 @@ def layer_fn(idx, prefix: str) -> nn.Module:; -716,7 +719,7 @@ def forward(; symbols: layer_fn, get_input_embeddings, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -706,6 +706,9 @@ def layer_fn(idx, prefix: str) -> nn.Module:
+        # For EAGLE3 support
+        self.layers_to_capture = []
@@ -716,7 +719,7 @@ def forward(
-    ) -> Union[torch.Tensor, PPProxyTensors]:
+    ) -> Union[torch.Tensor, PPProxyTensors, Tuple[torch.Tensor, list[torch.Tensor]]]:
@@ -728,6 +731,7 @@ def forward(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +34/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13297 - Fix: add missing get_embed_and_head in MiniMax M2 for Eagle3

- 链接: https://github.com/sgl-project/sglang/pull/13297
- 状态/时间: merged / 2025-11-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `b051d76dabb8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix: add missing get_embed_and_head in MiniMax M2 for Eagle3」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「Fix: add missing get_embed_and_head in MiniMax M2 for Eagle3」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +3/-0 (3 lines); hunks: -821,6 +821,9 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[l...; symbols: set_eagle3_layers_to_capture, get_embed_and_head, forward，涉及 `set_eagle3_layers_to_capture, get_embed_and_head, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +3/-0 (3 lines); hunks: -821,6 +821,9 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[l...; symbols: set_eagle3_layers_to_capture, get_embed_and_head, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -821,6 +821,9 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
+    def get_embed_and_head(self):
+        return self.model.embed_tokens.weight, self.lm_head.weight
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13659 - Super tiny remove unused MiniMaxM2MLP class

- 链接: https://github.com/sgl-project/sglang/pull/13659
- 状态/时间: merged / 2025-11-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `3f1cfd87b6fd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-36，可读 patch 57 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Super tiny remove unused MiniMaxM2MLP class」；模型线: MiniMax M2 Series；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/minimax_m2.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +0/-36 (36 lines); hunks: -31,15 +31,13; -127,40 +125,6 @@ def forward(; symbols: forward, MiniMaxM2MLP, __init__, MiniMaxM2MoE，涉及 `forward, MiniMaxM2MLP, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +0/-36 (36 lines); hunks: -31,15 +31,13; -127,40 +125,6 @@ def forward(; symbols: forward, MiniMaxM2MLP, __init__, MiniMaxM2MoE
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -31,15 +31,13 @@
-from sglang.srt.layers.activation import SiluAndMul
-    MergedColumnParallelLinear,
@@ -127,40 +125,6 @@ def forward(
-class MiniMaxM2MLP(nn.Module):
-    def __init__(
-        self,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +0/-36
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13892 - fix: correct usage of minimax-m2 deepep moe forward

- 链接: https://github.com/sgl-project/sglang/pull/13892
- 状态/时间: merged / 2025-11-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `e0e8a9963043`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-7，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: correct usage of minimax-m2 deepep moe forward」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/minimax_m2.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +3/-7 (10 lines); hunks: -222,7 +222,7 @@ def forward_deepep(; -231,14 +231,10 @@ def forward_deepep(; symbols: forward_deepep，涉及 `forward_deepep`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +3/-7 (10 lines); hunks: -222,7 +222,7 @@ def forward_deepep(; -231,14 +231,10 @@ def forward_deepep(; symbols: forward_deepep
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -222,7 +222,7 @@ def forward_deepep(
-            topk_weights, topk_idx, _ = self.topk(
+            topk_output = self.topk(
@@ -231,14 +231,10 @@ def forward_deepep(
-            topk_weights, topk_idx, _ = self.topk.empty_topk_output(
-                hidden_states.shape[0], self.top_k
-            )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +3/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14047 - Optimize topk sigmoid in minimax_m2

- 链接: https://github.com/sgl-project/sglang/pull/14047
- 状态/时间: merged / 2025-12-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `3dabd609fb03`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+38/-13，可读 patch 149 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Optimize topk sigmoid in minimax_m2」；模型线: MiniMax M2 Series；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「Optimize topk sigmoid in minimax_m2」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +0/-3 (3 lines); hunks: -167,9 +167,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +0/-3 (3 lines); hunks: -167,9 +167,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -167,9 +167,6 @@ def __init__(
-            use_grouped_topk=True,  # TODO: Use "grouped top-k" flag only for hardcoded sigmoid scoring
-            num_expert_group=1,
-            topk_group=1,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +0/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15538 - Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs

- 链接: https://github.com/sgl-project/sglang/pull/15538
- 状态/时间: merged / 2025-12-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/basic_usage/minimax_m2.md`, `python/sglang/srt/function_call/minimax_m2.py`；关联提交 `5c64a20da7dd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+254/-19，可读 patch 345 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs」；模型线: MiniMax M2 Series；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/function_call/minimax_m2.py`, `docs/basic_usage/minimax_m2.md`；技术摘要: 覆盖「Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs」；主要实现面是 `python/sglang/srt/function_call/minimax_m2.py`, `docs/basic_usage/minimax_m2.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/minimax_m2.py` modified +185/-17 (202 lines); hunks: -1,5 +1,3; -16,17 +14,6; symbols: _safe_val, MinimaxM2Detector, detect_and_parse, _convert_param_value，涉及 `_safe_val, MinimaxM2Detector, detect_and_parse`；`docs/basic_usage/minimax_m2.md` added +66/-0 (66 lines); hunks: -0,0 +1,66。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/minimax_m2.py` modified +185/-17 (202 lines); hunks: -1,5 +1,3; -16,17 +14,6; symbols: _safe_val, MinimaxM2Detector, detect_and_parse, _convert_param_value
  - `docs/basic_usage/minimax_m2.md` added +66/-0 (66 lines); hunks: -0,0 +1,66
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/minimax_m2.py
@@ -1,5 +1,3 @@
-import ast
-import html
@@ -16,17 +14,6 @@
-def _safe_val(raw: str) -> Any:
-    raw = html.unescape(raw.strip())
-    try:
diff -- docs/basic_usage/minimax_m2.md
@@ -0,0 +1,66 @@
+# MiniMax M2.1/M2 Usage
+[MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) and [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) are advanced large language models created by [MiniMax]
+MiniMax-M2 series redefines efficiency for agents. It's a compact, fast, and cost-effective MoE model (230 billion total parameters with 10 billion active parameters) built for el
+## Supported Models
+This guide applies to the following models. You only need to update the model name during deployment. The following examples use **MiniMax-M2**:
+- [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/minimax_m2.py` modified +185/-17
  - docs: `docs/basic_usage/minimax_m2.md` added +66/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14416 - Fusing RMSNormTP in minimax_m2

- 链接: https://github.com/sgl-project/sglang/pull/14416
- 状态/时间: merged / 2025-12-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `d17b9e639224`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+189/-2，可读 patch 219 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fusing RMSNormTP in minimax_m2」；模型线: MiniMax M2 Series；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「Fusing RMSNormTP in minimax_m2」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +189/-2 (191 lines); hunks: -19,6 +19,8; -73,6 +75,164; symbols: rmsnorm_sumsq_kernel_serial, rmsnorm_apply_kernel_serial, rms_sumsq_serial, rms_apply_serial，涉及 `rmsnorm_sumsq_kernel_serial, rmsnorm_apply_kernel_serial, rms_sumsq_serial`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +189/-2 (191 lines); hunks: -19,6 +19,8; -73,6 +75,164; symbols: rmsnorm_sumsq_kernel_serial, rmsnorm_apply_kernel_serial, rms_sumsq_serial, rms_apply_serial
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -19,6 +19,8 @@
+import triton
+import triton.language as tl
@@ -73,6 +75,164 @@
+@triton.jit
+def rmsnorm_sumsq_kernel_serial(
+    x1_ptr,  # T* [B, D]
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +189/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17826 - Support Pipeline and Data Parallelism for MiniMax-M2

- 链接: https://github.com/sgl-project/sglang/pull/17826
- 状态/时间: open / 2026-01-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+167/-70，可读 patch 479 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support Pipeline and Data Parallelism for MiniMax-M2」；模型线: MiniMax M2 Series；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「Support Pipeline and Data Parallelism for MiniMax-M2」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +167/-70 (237 lines); hunks: -16,7 +16,8; -28,7 +29,6; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, ebias_weight_loader，涉及 `MiniMaxM2RMSNormTP, __init__, weight_loader`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +167/-70 (237 lines); hunks: -16,7 +16,8; -28,7 +29,6; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, ebias_weight_loader
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -16,7 +16,8 @@
-from typing import Iterable, Optional, Set, Tuple, Union
+from contextlib import nullcontext
+from typing import Iterable, List, Optional, Set, Tuple, Union
@@ -28,7 +29,6 @@
-    get_tensor_model_parallel_rank,
@@ -39,6 +39,11 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +167/-70
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16483 - Optimizing all_reduce in RMSNormTP in minimax_m2

- 链接: https://github.com/sgl-project/sglang/pull/16483
- 状态/时间: merged / 2026-02-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `486c7de39f5c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-2，可读 patch 24 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Optimizing all_reduce in RMSNormTP in minimax_m2」；模型线: MiniMax M2 Series；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「Optimizing all_reduce in RMSNormTP in minimax_m2」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +8/-2 (10 lines); hunks: -166,7 +166,14 @@ def rms_sumsq_serial(x1: torch.Tensor, x2: torch.Tensor) ->...; -285,7 +292,6 @@ def forward(; symbols: rms_sumsq_serial, forward, forward_qk，涉及 `rms_sumsq_serial, forward, forward_qk`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +8/-2 (10 lines); hunks: -166,7 +166,14 @@ def rms_sumsq_serial(x1: torch.Tensor, x2: torch.Tensor) ->...; -285,7 +292,6 @@ def forward(; symbols: rms_sumsq_serial, forward, forward_qk
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -166,7 +166,14 @@ def rms_sumsq_serial(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
-    sum_sq = torch.empty(B + B2, device=x1.device, dtype=torch.float32)
+    # We found that custom all-reduce `sglang::cross_device_reduce_1stage`
+    # is much faster than the nccl all-reduce in torch.
+    # However, `should_custom_ar` checks if the reduced buffer is 16-byte aligned.
+    # RMSNormTP reduces a [B, 2] fp32 tensor, so we pad the total element count to
+    # satisfy the alignment requirement.
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +8/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18217 - [piecewise graph]: support MiniMax-M2

- 链接: https://github.com/sgl-project/sglang/pull/18217
- 状态/时间: merged / 2026-02-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `079fc8f3c591`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+28/-7，可读 patch 70 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[piecewise graph]: support MiniMax-M2」；模型线: MiniMax M2 Series；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「[piecewise graph]: support MiniMax-M2」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +23/-7 (30 lines); hunks: -16,6 +16,7; -442,9 +443,14 @@ def op_select_experts(self, state):; symbols: op_select_experts, op_dispatch_a, op_dispatch_b, forward，涉及 `op_select_experts, op_dispatch_a, op_dispatch_b`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +23/-7 (30 lines); hunks: -16,6 +16,7; -442,9 +443,14 @@ def op_select_experts(self, state):; symbols: op_select_experts, op_dispatch_a, op_dispatch_b, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -16,6 +16,7 @@
+from contextlib import nullcontext
@@ -442,9 +443,14 @@ def op_select_experts(self, state):
-            with get_global_expert_distribution_recorder().with_current_layer(
-                self.layer_id
-            ):
+            ctx = (
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +23/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18310 - [Fix] MiniMax-M2.1 CUDA Graph + torch.compile crashes due to outplace_all_reduce being traced by Dynamo

- 链接: https://github.com/sgl-project/sglang/pull/18310
- 状态/时间: open / 2026-02-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-0，可读 patch 15 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] MiniMax-M2.1 CUDA Graph + torch.compile crashes due to outplace_all_reduce being traced by Dynamo」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/distributed/parallel_state.py`；技术摘要: 覆盖「[Fix] MiniMax-M2.1 CUDA Graph + torch.compile crashes due to outplace_all_reduce being traced by Dynamo」；主要实现面是 `python/sglang/srt/distributed/parallel_state.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/distributed/parallel_state.py` modified +8/-0 (8 lines); hunks: -586,6 +586,14 @@ def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:; symbols: all_reduce，涉及 `all_reduce`。
- 代码 diff 细节:
  - `python/sglang/srt/distributed/parallel_state.py` modified +8/-0 (8 lines); hunks: -586,6 +586,14 @@ def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:; symbols: all_reduce
- 关键代码摘录:

```diff
diff -- python/sglang/srt/distributed/parallel_state.py
@@ -586,6 +586,14 @@ def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
+        # IMPORTANT:
+        # Never allow Dynamo/Inductor to trace the out-of-place all-reduce path.
+        # If it gets traced, it will appear in compiled code and break CUDA graph replay.
+        # Reference: https://github.com/sgl-project/sglang/issues/16102
+        if hasattr(torch, "_dynamo") and torch._dynamo.is_compiling():
+            torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
```

- 已读文件:
  - runtime: `python/sglang/srt/distributed/parallel_state.py` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/distributed/parallel_state.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19468 - fix[minimax]: support deepep with minimax models

- 链接: https://github.com/sgl-project/sglang/pull/19468
- 状态/时间: open / 2026-02-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+10/-2，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix[minimax]: support deepep with minimax models」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh`；技术摘要: 覆盖「fix[minimax]: support deepep with minimax models」；主要实现面是 `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: -2117,6 +2117,12 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments，涉及 `_handle_model_specific_adjustments`；`docker/Dockerfile` modified +2/-1 (3 lines); hunks: -9,7 +9,8 @@ ARG HOPPER_SBO=0；`scripts/ci/cuda/ci_install_deepep.sh` modified +2/-1 (3 lines); hunks: -88,9 +88,10 @@ if [ "$GRACE_BLACKWELL" = "1" ]; then。
- 代码 diff 细节:
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: -2117,6 +2117,12 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `docker/Dockerfile` modified +2/-1 (3 lines); hunks: -9,7 +9,8 @@ ARG HOPPER_SBO=0
  - `scripts/ci/cuda/ci_install_deepep.sh` modified +2/-1 (3 lines); hunks: -88,9 +88,10 @@ if [ "$GRACE_BLACKWELL" = "1" ]; then
- 关键代码摘录:

```diff
diff -- python/sglang/srt/server_args.py
@@ -2117,6 +2117,12 @@ def _handle_model_specific_adjustments(self):
+        elif model_arch in ["MiniMaxM2ForCausalLM"]:
+            if self.moe_a2a_backend == "deepep":
+                # When using DeepEP, we need to make sure activation dtype is bf16 and not float16
+                # otherwise DeepEP will error due to activation dtype mismatch.
+                self.dtype = "bfloat16"
diff -- docker/Dockerfile
@@ -9,7 +9,8 @@ ARG HOPPER_SBO=0
-ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
+# https://github.com/deepseek-ai/DeepEP/pull/458
+ARG DEEPEP_COMMIT=73b6ea4a439ba03a695563f9fd242c8e4b02b37c
diff -- scripts/ci/cuda/ci_install_deepep.sh
@@ -88,9 +88,10 @@ if [ "$GRACE_BLACKWELL" = "1" ]; then
+    DEEPEP_COMMIT=73b6ea4a439ba03a695563f9fd242c8e4b02b37c
-    git checkout 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee && \
+    git checkout ${DEEPEP_COMMIT} && \
```

- 已读文件:
  - runtime: `python/sglang/srt/server_args.py` modified +6/-0
  - other: `docker/Dockerfile` modified +2/-1; `scripts/ci/cuda/ci_install_deepep.sh` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19443 - [AMD] [MiniMax-M2.5 Day 0] Add MiniMax-M2.5 nightly accuracy test

- 链接: https://github.com/sgl-project/sglang/pull/19443
- 状态/时间: merged / 2026-02-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/basic_usage/minimax_m2.md`, `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py`；关联提交 `403195d59de0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+653/-4，可读 patch 766 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [MiniMax-M2.5 Day 0] Add MiniMax-M2.5 nightly accuracy test」；模型线: MiniMax M2 Series；类别: 性能/后端优化；主要 diff: `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py`, `docs/basic_usage/minimax_m2.md`；技术摘要: 覆盖「[AMD] [MiniMax-M2.5 Day 0] Add MiniMax-M2.5 nightly accuracy test」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py`, `docs/basic_usage/minimax_m2.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, __post_init__, get_display_name, get_one_example，涉及 `ModelConfig, __post_init__, get_display_name`；`test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: ModelConfig, __post_init__, get_display_name, get_one_example，涉及 `ModelConfig, __post_init__, get_display_name`；`docs/basic_usage/minimax_m2.md` modified +22/-3 (25 lines); hunks: -1,13 +1,14; -49,6 +50,24 @@ python -m sglang.launch_server \。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, __post_init__, get_display_name, get_one_example
  - `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: ModelConfig, __post_init__, get_display_name, get_one_example
  - `docs/basic_usage/minimax_m2.md` modified +22/-3 (25 lines); hunks: -1,13 +1,14; -49,6 +50,24 @@ python -m sglang.launch_server \
- 关键代码摘录:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py
@@ -0,0 +1,249 @@
+"""MI35x MiniMax-M2.5 GSM8K Completion Evaluation Test (8-GPU)
+Tests MiniMax-M2.5 with TP=8 + EP=8 configuration using few-shot completion
+benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-minimax-m25 suite
+"""
+import ast
diff -- test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py
@@ -0,0 +1,245 @@
+"""AMD MiniMax-M2.5 GSM8K Completion Evaluation Test (8-GPU)
+Tests MiniMax-M2.5 with TP=8 + EP=8 configuration using few-shot completion
+benchmark on MI325/MI300X.
+Registry: nightly-amd-accuracy-8-gpu-minimax-m25 suite
+"""
+import ast
diff -- docs/basic_usage/minimax_m2.md
@@ -1,13 +1,14 @@
```

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py` added +249/-0; `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py` added +245/-0
  - docs: `docs/basic_usage/minimax_m2.md` modified +22/-3
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19577 - [Feat] add PP Support for minimax-m2 series

- 链接: https://github.com/sgl-project/sglang/pull/19577
- 状态/时间: merged / 2026-03-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `2d183c4e6d32`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+35/-7，可读 patch 97 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feat] add PP Support for minimax-m2 series」；模型线: MiniMax M2 Series；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「[Feat] add PP Support for minimax-m2 series」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +35/-7 (42 lines); hunks: -54,7 +54,7; -967,6 +967,7 @@ def __init__(; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +35/-7 (42 lines); hunks: -54,7 +54,7; -967,6 +967,7 @@ def __init__(; symbols: __init__, forward, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -54,7 +54,7 @@
-from sglang.srt.layers.utils import PPMissingLayer
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
@@ -967,6 +967,7 @@ def __init__(
+        self.pp_group = get_pp_group()
@@ -999,17 +1000,26 @@ def forward(
+        pp_proxy_tensors: Optional[PPProxyTensors] = None,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +35/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20031 - fix(minimax): support loading merged expert weights (w13) for awq

- 链接: https://github.com/sgl-project/sglang/pull/20031
- 状态/时间: open / 2026-03-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+203/-9，可读 patch 236 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(minimax): support loading merged expert weights (w13) for awq」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `tests/registered/models/test_minimax_m2_weights.py`, `python/sglang/srt/models/minimax_m2.py`；未提供可用技术摘要。
- 实现要点: `tests/registered/models/test_minimax_m2_weights.py` added +145/-0 (145 lines); hunks: -0,0 +1,145; symbols: TestMiniMaxM2WeightLoading, setUp, test_load_weights_merged_w13，涉及 `TestMiniMaxM2WeightLoading, setUp, test_load_weights_merged_w13`；`python/sglang/srt/models/minimax_m2.py` modified +58/-9 (67 lines); hunks: -1058,6 +1058,14 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; -1112,7 +1120,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `tests/registered/models/test_minimax_m2_weights.py` added +145/-0 (145 lines); hunks: -0,0 +1,145; symbols: TestMiniMaxM2WeightLoading, setUp, test_load_weights_merged_w13
  - `python/sglang/srt/models/minimax_m2.py` modified +58/-9 (67 lines); hunks: -1058,6 +1058,14 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; -1112,7 +1120,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights
- 关键代码摘录:

```diff
diff -- tests/registered/models/test_minimax_m2_weights.py
@@ -0,0 +1,145 @@
+import unittest
+from unittest.mock import MagicMock, patch
+import torch
+from transformers import PretrainedConfig
+from sglang.srt.models.minimax_m2 import MiniMaxM2ForCausalLM
+class TestMiniMaxM2WeightLoading(unittest.TestCase):
diff -- python/sglang/srt/models/minimax_m2.py
@@ -1058,6 +1058,14 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+        expert_params_mapping_fused = FusedMoE.make_expert_params_mapping_fused(
+            ckpt_gate_up_proj_name="w13",
+            ckpt_down_proj_name="w2",
+            ckpt_gate_up_proj_bias_name="w13_bias",
+            ckpt_down_proj_bias_name="w2_bias",
+            num_experts=self.config.num_local_experts,
```

- 已读文件:
  - tests: `tests/registered/models/test_minimax_m2_weights.py` added +145/-0
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +58/-9
- 验证与风险: diff 自带测试面 `tests/registered/models/test_minimax_m2_weights.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20083 - [Nightly] Replace MiniMax-M2 with MiniMax-M2.5

- 链接: https://github.com/sgl-project/sglang/pull/20083
- 状态/时间: merged / 2026-03-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_minimax_m25.py`；关联提交 `1aa6ab41deb5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-14，可读 patch 56 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Nightly] Replace MiniMax-M2 with MiniMax-M2.5」；模型线: MiniMax M2 Series；类别: 模型支持/运行时入口；主要 diff: `test/registered/8-gpu-models/test_minimax_m25.py`；技术摘要: 覆盖「[Nightly] Replace MiniMax-M2 with MiniMax-M2.5」；主要实现面是 `test/registered/8-gpu-models/test_minimax_m25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_minimax_m25.py` renamed +12/-14 (26 lines); hunks: -9,32 +9,30; -43,10 +41,10 @@ def test_minimax_m2(self):; symbols: TestMiniMaxM2, for, TestMiniMaxM25, test_minimax_m2，涉及 `TestMiniMaxM2, for, TestMiniMaxM25`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_minimax_m25.py` renamed +12/-14 (26 lines); hunks: -9,32 +9,30; -43,10 +41,10 @@ def test_minimax_m2(self):; symbols: TestMiniMaxM2, for, TestMiniMaxM25, test_minimax_m2
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_minimax_m25.py
@@ -9,32 +9,30 @@
-MINIMAX_M2_MODEL_PATH = "MiniMaxAI/MiniMax-M2"
+MINIMAX_M25_MODEL_PATH = "MiniMaxAI/MiniMax-M2.5"
-class TestMiniMaxM2(unittest.TestCase):
-    """Unified test class for MiniMax-M2 performance and accuracy.
+class TestMiniMaxM25(unittest.TestCase):
+    """Unified test class for MiniMax-M2.5 performance and accuracy.
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_minimax_m25.py` renamed +12/-14
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_minimax_m25.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20489 - fix(dp-attn): fix issues with dp-attention for MiniMax M2 and general…

- 链接: https://github.com/sgl-project/sglang/pull/20489
- 状态/时间: open / 2026-03-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+118/-20，可读 patch 247 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(dp-attn): fix issues with dp-attention for MiniMax M2 and general…」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/rotary_embedding/base.py`；技术摘要: 覆盖「fix(dp-attn): fix issues with dp-attention for MiniMax M2 and general…」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/rotary_embedding/base.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunks: -27,10 +27,14; -244,10 +248,16 @@ def rms_apply_serial(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, weight_loader，涉及 `rms_apply_serial, MiniMaxM2RMSNormTP, __init__`；`python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: -1976,14 +1976,14 @@ def _dummy_run(self, batch_size: int, run_ctx=None):; symbols: _dummy_run，涉及 `_dummy_run`；`python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0 (2 lines); hunks: -291,6 +291,8 @@ def forward_cuda(; symbols: forward_cuda，涉及 `forward_cuda`；`PR_DESCRIPTION.md` added +78/-0 (78 lines); hunks: -0,0 +1,78。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunks: -27,10 +27,14; -244,10 +248,16 @@ def rms_apply_serial(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, weight_loader
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: -1976,14 +1976,14 @@ def _dummy_run(self, batch_size: int, run_ctx=None):; symbols: _dummy_run
  - `python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0 (2 lines); hunks: -291,6 +291,8 @@ def forward_cuda(; symbols: forward_cuda
  - `PR_DESCRIPTION.md` added +78/-0 (78 lines); hunks: -0,0 +1,78
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-2 (5 lines); hunks: -100,9 +100,10 @@ def _set_kv_buffer_impl(; symbols: _set_kv_buffer_impl
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -27,10 +27,14 @@
+    GroupCoordinator,
+    get_attn_tp_group,
+    get_attn_tensor_model_parallel_world_size,
+    get_tp_group,
@@ -244,10 +248,16 @@ def rms_apply_serial(
-    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -1976,14 +1976,14 @@ def _dummy_run(self, batch_size: int, run_ctx=None):
-                    [num_tokens],
+                    [num_tokens] * self.server_args.dp_size,
-                    [num_tokens],
+                    [num_tokens] * self.server_args.dp_size,
diff -- python/sglang/srt/layers/rotary_embedding/base.py
@@ -291,6 +291,8 @@ def forward_cuda(
+            if batch_size == 0:
+                return query, key
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +33/-16; `python/sglang/srt/model_executor/model_runner.py` modified +2/-2; `python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0; `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-2
  - other: `PR_DESCRIPTION.md` added +78/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/rotary_embedding/base.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/model_executor/model_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20873 - docs: add MiniMax-M2.7 and M2.7-highspeed model support

- 链接: https://github.com/sgl-project/sglang/pull/20873
- 状态/时间: open / 2026-03-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+15/-3，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: add MiniMax-M2.7 and M2.7-highspeed model support」；模型线: MiniMax M2 Series；类别: 文档/测试/CI；主要 diff: `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md`；技术摘要: 覆盖「docs: add MiniMax-M2.7 and M2.7-highspeed model support」；主要实现面是 `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/basic_usage/minimax_m2.md` modified +14/-2 (16 lines); hunks: -1,13 +1,14; -83,3 +84,14 @@ curl http://localhost:8000/v1/chat/completions \；`docs/supported_models/text_generation/generative_models.md` modified +1/-1 (2 lines); hunks: -37,7 +37,7 @@ in the GitHub search bar.。
- 代码 diff 细节:
  - `docs/basic_usage/minimax_m2.md` modified +14/-2 (16 lines); hunks: -1,13 +1,14; -83,3 +84,14 @@ curl http://localhost:8000/v1/chat/completions \
  - `docs/supported_models/text_generation/generative_models.md` modified +1/-1 (2 lines); hunks: -37,7 +37,7 @@ in the GitHub search bar.
- 关键代码摘录:

```diff
diff -- docs/basic_usage/minimax_m2.md
@@ -1,13 +1,14 @@
-# MiniMax M2.5/M2.1/M2 Usage
+# MiniMax M2.7/M2.5/M2.1/M2 Usage
-[MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5), [MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1), and [MiniMax-M2](https://huggingface.co/MiniMaxAI/Min
+[MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7), [MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5), [MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniM
+- [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)
@@ -83,3 +84,14 @@ curl http://localhost:8000/v1/chat/completions \
diff -- docs/supported_models/text_generation/generative_models.md
@@ -37,7 +37,7 @@ in the GitHub search bar.
-| **MiniMax-M2** (M2, M2.1, M2.5)               | `MiniMaxAI/MiniMax-M2.5`, `MiniMaxAI/MiniMax-M2.1`, `MiniMaxAI/MiniMax-M2` | MiniMax's SOTA LLM for coding & agentic workflows. |
+| **MiniMax-M2** (M2, M2.1, M2.5, M2.7)               | `MiniMaxAI/MiniMax-M2.7`, `MiniMaxAI/MiniMax-M2.5`, `MiniMaxAI/MiniMax-M2.1`, `MiniMaxAI/MiniMax-M2` | MiniMax's SOTA LLM f
```

- 已读文件:
  - docs: `docs/basic_usage/minimax_m2.md` modified +14/-2; `docs/supported_models/text_generation/generative_models.md` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #19995 - Add packed_modules_mapping for MiniMax-M2

- 链接: https://github.com/sgl-project/sglang/pull/19995
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `df1d046de2a1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-0，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add packed_modules_mapping for MiniMax-M2」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「Add packed_modules_mapping for MiniMax-M2」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +12/-0 (12 lines); hunks: -941,6 +941,18 @@ def forward(; symbols: forward, MiniMaxM2ForCausalLM, __init__，涉及 `forward, MiniMaxM2ForCausalLM, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +12/-0 (12 lines); hunks: -941,6 +941,18 @@ def forward(; symbols: forward, MiniMaxM2ForCausalLM, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -941,6 +941,18 @@ def forward(
+    packed_modules_mapping = {
+        "qkv_proj": [
+            "q_proj",
+            "k_proj",
+            "v_proj",
+        ],
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +12/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20870 - [MiniMax M2] Fix KV cache scale loading

- 链接: https://github.com/sgl-project/sglang/pull/20870
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `a3196d08b8f6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-0，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MiniMax M2] Fix KV cache scale loading」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「[MiniMax M2] Fix KV cache scale loading」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +8/-0 (8 lines); hunks: -1063,10 +1063,18 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +8/-0 (8 lines); hunks: -1063,10 +1063,18 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -1063,10 +1063,18 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+            _is_kv_scale = name.endswith(".k_scale") or name.endswith(".v_scale")
+                # Skip kv cache scales - maybe_remap_kv_scale_name expects the
+                # original checkpoint name (e.g. self_attn.k_proj.k_scale) to
+                # remap it to self_attn.attn.k_scale. Renaming k_proj -> qkv_proj
+                # here would break that pattern match.
+                if _is_kv_scale:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20975 - fix(dp-attn): fix issues with dp-attention for MiniMax M2

- 链接: https://github.com/sgl-project/sglang/pull/20975
- 状态/时间: open / 2026-03-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+122/-20，可读 patch 258 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(dp-attn): fix issues with dp-attention for MiniMax M2」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/model_executor/model_runner.py`；技术摘要: 覆盖「fix(dp-attn): fix issues with dp-attention for MiniMax M2」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/model_executor/model_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunks: -28,10 +28,14; -247,10 +251,16 @@ def rms_apply_serial(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, weight_loader，涉及 `rms_apply_serial, MiniMaxM2RMSNormTP, __init__`；`python/sglang/srt/layers/dp_attention.py` modified +4/-0 (4 lines); hunks: -328,6 +328,10 @@ def get_attention_tp_size() -> int:; symbols: get_attention_tp_size, get_attention_tp_world_size, get_attention_cp_group，涉及 `get_attention_tp_size, get_attention_tp_world_size, get_attention_cp_group`；`python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: -2121,14 +2121,14 @@ def _dummy_run(self, batch_size: int, run_ctx=None):; symbols: _dummy_run，涉及 `_dummy_run`；`python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0 (2 lines); hunks: -291,6 +291,8 @@ def forward_cuda(; symbols: forward_cuda，涉及 `forward_cuda`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunks: -28,10 +28,14; -247,10 +251,16 @@ def rms_apply_serial(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, weight_loader
  - `python/sglang/srt/layers/dp_attention.py` modified +4/-0 (4 lines); hunks: -328,6 +328,10 @@ def get_attention_tp_size() -> int:; symbols: get_attention_tp_size, get_attention_tp_world_size, get_attention_cp_group
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: -2121,14 +2121,14 @@ def _dummy_run(self, batch_size: int, run_ctx=None):; symbols: _dummy_run
  - `python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0 (2 lines); hunks: -291,6 +291,8 @@ def forward_cuda(; symbols: forward_cuda
  - `PR_DESCRIPTION.md` added +78/-0 (78 lines); hunks: -0,0 +1,78
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -28,10 +28,14 @@
+    GroupCoordinator,
+    get_attention_tp_group,
+    get_attention_tp_world_size,
+    get_tp_group,
@@ -247,10 +251,16 @@ def rms_apply_serial(
-    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
diff -- python/sglang/srt/layers/dp_attention.py
@@ -328,6 +328,10 @@ def get_attention_tp_size() -> int:
+def get_attention_tp_world_size() -> int:
+    return get_attn_tensor_model_parallel_world_size()
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -2121,14 +2121,14 @@ def _dummy_run(self, batch_size: int, run_ctx=None):
-                    [num_tokens],
+                    [num_tokens] * self.server_args.dp_size,
-                    [num_tokens],
+                    [num_tokens] * self.server_args.dp_size,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +33/-16; `python/sglang/srt/layers/dp_attention.py` modified +4/-0; `python/sglang/srt/model_executor/model_runner.py` modified +2/-2; `python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0; `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-2
  - other: `PR_DESCRIPTION.md` added +78/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/rotary_embedding/base.py`, `python/sglang/srt/mem_cache/memory_pool.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20931 - [Bugifx] qwen3 rope parameter compatibility

- 链接: https://github.com/sgl-project/sglang/pull/20931
- 状态/时间: merged / 2026-03-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-3，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugifx] qwen3 rope parameter compatibility」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；技术摘要: 覆盖「[Bugifx] qwen3 rope parameter compatibility」；主要实现面是 `python/sglang/srt/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -78,6 +78,7; -566,7 +567,7 @@ def forward_prepare_native(; symbols: forward_prepare_native, apply_qk_norm_rope, __init__，涉及 `forward_prepare_native, apply_qk_norm_rope, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -78,6 +78,7; -566,7 +567,7 @@ def forward_prepare_native(; symbols: forward_prepare_native, apply_qk_norm_rope, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -78,6 +78,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_rope_config
@@ -566,7 +567,7 @@ def forward_prepare_native(
-            theta = self.config.rope_parameters["rope_theta"]
+            theta = self.rope_theta
@@ -691,8 +692,8 @@ def __init__(
-        rope_theta = config.rope_parameters["rope_theta"]
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +4/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17695 - [NPU] enhance accuracy for model minimaxm2 from 16.5% to 95.5%

- 链接: https://github.com/sgl-project/sglang/pull/17695
- 状态/时间: merged / 2026-03-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/ascend/llm_models/test_ascend_minimax_m2.py`；关联提交 `4641e5a3d2bb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+45/-1，可读 patch 61 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] enhance accuracy for model minimaxm2 from 16.5% to 95.5%」；模型线: MiniMax M2 Series；类别: 性能/后端优化；主要 diff: `test/registered/ascend/llm_models/test_ascend_minimax_m2.py`；技术摘要: 覆盖「[NPU] enhance accuracy for model minimaxm2 from 16.5% to 95.5%」；主要实现面是 `test/registered/ascend/llm_models/test_ascend_minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/ascend/llm_models/test_ascend_minimax_m2.py` added +43/-0 (43 lines); hunks: -0,0 +1,43; symbols: TestMiniMaxM2，涉及 `TestMiniMaxM2`。
- 代码 diff 细节:
  - `test/registered/ascend/llm_models/test_ascend_minimax_m2.py` added +43/-0 (43 lines); hunks: -0,0 +1,43; symbols: TestMiniMaxM2
- 关键代码摘录:

```diff
diff -- test/registered/ascend/llm_models/test_ascend_minimax_m2.py
@@ -0,0 +1,43 @@
+import unittest
+from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
+from sglang.test.ascend.test_ascend_utils import MINIMAX_M2_WEIGHTS_PATH
+from sglang.test.ci.ci_register import register_npu_ci
+from sglang.test.test_utils import CustomTestCase
+register_npu_ci(
```

- 已读文件:
  - tests: `test/registered/ascend/llm_models/test_ascend_minimax_m2.py` added +43/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/ascend/test_ascend_utils.py`, `test/registered/ascend/llm_models/test_ascend_minimax_m2.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20905 - [NPU][ModelSlim] adapt w2 quant layer for Minimax2.5

- 链接: https://github.com/sgl-project/sglang/pull/20905
- 状态/时间: merged / 2026-03-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `1b4933d45d93`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+22/-30，可读 patch 67 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU][ModelSlim] adapt w2 quant layer for Minimax2.5」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「[NPU][ModelSlim] adapt w2 quant layer for Minimax2.5」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -713,7 +713,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -713,7 +713,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -713,7 +713,7 @@ def __init__(
-            prefix=add_prefix("mlp", prefix),
+            prefix=add_prefix("block_sparse_moe", prefix),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21241 - [bugfix] Fix rope theta config for MiniMax after transformers v5 update

- 链接: https://github.com/sgl-project/sglang/pull/21241
- 状态/时间: merged / 2026-03-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `b91f78d255d8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-3，可读 patch 32 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bugfix] Fix rope theta config for MiniMax after transformers v5 update」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「[bugfix] Fix rope theta config for MiniMax after transformers v5 update」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +3/-3 (6 lines); hunks: -73,6 +73,7; -570,7 +571,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +3/-3 (6 lines); hunks: -73,6 +73,7; -570,7 +571,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -73,6 +73,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_rope_config
@@ -570,7 +571,7 @@ def __init__(
-        self.rope_theta = config.rope_theta
+        self.rope_theta, self.rope_scaling = get_rope_config(config)
@@ -600,13 +601,12 @@ def __init__(
-        rope_scaling = getattr(config, "rope_scaling", None)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +3/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19652 - [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+)

- 链接: https://github.com/sgl-project/sglang/pull/19652
- 状态/时间: merged / 2026-04-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+1410/-95，可读 patch 1875 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+)」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`；技术摘要: 覆盖「[Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+)」；主要实现面是 `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` added +320/-0 (320 lines); hunks: -0,0 +1,320; symbols: is_fp4_marlin_supported, should_use_fp4_marlin_fallback, nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale，涉及 `is_fp4_marlin_supported, should_use_fp4_marlin_fallback, nvfp4_marlin_process_scales`；`python/sglang/srt/layers/quantization/modelopt_quant.py` modified +82/-7 (89 lines); hunks: -40,6 +40,11; -1128,7 +1133,7 @@ def get_supported_act_dtypes(cls) -> List[torch.dtype]:; symbols: get_supported_act_dtypes, get_min_capability, common_group_size, create_weights，涉及 `get_supported_act_dtypes, get_min_capability, common_group_size`；`python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` modified +66/-8 (74 lines); hunks: -17,6 +17,10; -38,19 +42,27; symbols: CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability, create_weights，涉及 `CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability`；`python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +33/-10 (43 lines); hunks: -23,6 +23,13 @@ def get_scalar_type(num_bits: int, has_zp: bool):; -46,6 +53,8 @@ def fused_marlin_moe(; symbols: get_scalar_type, _get_fp4_scalar_type, fused_marlin_moe，涉及 `get_scalar_type, _get_fp4_scalar_type, fused_marlin_moe`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` added +320/-0 (320 lines); hunks: -0,0 +1,320; symbols: is_fp4_marlin_supported, should_use_fp4_marlin_fallback, nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +82/-7 (89 lines); hunks: -40,6 +40,11; -1128,7 +1133,7 @@ def get_supported_act_dtypes(cls) -> List[torch.dtype]:; symbols: get_supported_act_dtypes, get_min_capability, common_group_size, create_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` modified +66/-8 (74 lines); hunks: -17,6 +17,10; -38,19 +42,27; symbols: CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability, create_weights
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +33/-10 (43 lines); hunks: -23,6 +23,13 @@ def get_scalar_type(num_bits: int, has_zp: bool):; -46,6 +53,8 @@ def fused_marlin_moe(; symbols: get_scalar_type, _get_fp4_scalar_type, fused_marlin_moe
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` modified +32/-1 (33 lines); hunks: -16,6 +16,10; -34,7 +38,7 @@ def __init__(self):; symbols: __init__, get_min_capability, create_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/marlin_utils_fp4.py
@@ -0,0 +1,320 @@
+# SPDX-License-Identifier: Apache-2.0
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
+"""NVFP4 Marlin fallback: run FP4-quantized models on non-Blackwell GPUs via Marlin kernel."""
+import logging
+from typing import Optional
+import torch
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -40,6 +40,11 @@
+from sglang.srt.layers.quantization.marlin_utils_fp4 import (
+    prepare_fp4_layer_for_marlin,
+    prepare_moe_fp4_layer_for_marlin,
+    should_use_fp4_marlin_fallback,
+)
@@ -1128,7 +1133,7 @@ def get_supported_act_dtypes(cls) -> List[torch.dtype]:
diff -- python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py
@@ -17,6 +17,10 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` added +320/-0; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +82/-7; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` modified +66/-8; `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +33/-10; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` modified +32/-1; `python/sglang/srt/layers/moe/moe_runner/marlin.py` modified +9/-1
  - tests: `test/registered/quant/test_nvfp4_marlin_fallback.py` added +788/-0
- 验证与风险: diff 自带测试面 `test/registered/quant/test_nvfp4_marlin_fallback.py`, `test/registered/unit/model_loader/test_modelopt_loader.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21524 - [AMD] Add MiniMax-M2.5 nightly perf benchmarks for MI30x and MI35x

- 链接: https://github.com/sgl-project/sglang/pull/21524
- 状态/时间: merged / 2026-04-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py`, `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py`；关联提交 `d07d0a15ceb8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+338/-4，可读 patch 400 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add MiniMax-M2.5 nightly perf benchmarks for MI30x and MI35x」；模型线: MiniMax M2 Series；类别: 性能/后端优化；主要 diff: `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py`；技术摘要: 覆盖「[AMD] Add MiniMax-M2.5 nightly perf benchmarks for MI30x and MI35x」；主要实现面是 `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM25PerformanceMI35x, setUpClass, test_bench_minimax_m25，涉及 `generate_simple_markdown_report, TestNightlyMiniMaxM25PerformanceMI35x, setUpClass`；`test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM25Performance, setUpClass, test_bench_minimax_m25，涉及 `generate_simple_markdown_report, TestNightlyMiniMaxM25Performance, setUpClass`。
- 代码 diff 细节:
  - `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM25PerformanceMI35x, setUpClass, test_bench_minimax_m25
  - `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM25Performance, setUpClass, test_bench_minimax_m25
- 关键代码摘录:

```diff
diff -- test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py
@@ -0,0 +1,146 @@
+"""MI35x Nightly performance benchmark for MiniMax-M2.5 (8-GPU).
+This test benchmarks MiniMax-M2.5 with TP=8 + EP=8 configuration on MI35x.
+The model path can be configured via MINIMAX_M25_MODEL_PATH environment variable.
+Registry: nightly-perf-8-gpu-mi35x-minimax-m25 suite
+Example usage:
+    python -m pytest test_minimax_m25_perf_mi35x.py -v
diff -- test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py
@@ -0,0 +1,140 @@
+"""Nightly performance benchmark for MiniMax-M2.5 on MI325/MI300X (8-GPU).
+This test benchmarks MiniMax-M2.5 with TP=8 + EP=8 configuration.
+The model path can be configured via MINIMAX_M25_MODEL_PATH environment variable.
+Registry: nightly-perf-8-gpu-minimax-m25 suite
+Example usage:
+    python -m pytest test_minimax_m25_perf_amd.py -v
```

- 已读文件:
  - tests: `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py` added +146/-0; `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py` added +140/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py`, `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21792 - [CI] Add basic unit test for Minimax-M2.5

- 链接: https://github.com/sgl-project/sglang/pull/21792
- 状态/时间: merged / 2026-04-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_minimax_m25_basic.py`；关联提交 `56266de624f9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+84/-0，可读 patch 85 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add basic unit test for Minimax-M2.5」；模型线: MiniMax M2 Series；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_minimax_m25_basic.py`；技术摘要: 覆盖「[CI] Add basic unit test for Minimax-M2.5」；主要实现面是 `test/registered/8-gpu-models/test_minimax_m25_basic.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_minimax_m25_basic.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestMiniMaxM25Basic, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestMiniMaxM25Basic, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_minimax_m25_basic.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestMiniMaxM25Basic, setUpClass, tearDownClass, test_a_gsm8k
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_minimax_m25_basic.py
@@ -0,0 +1,84 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
+from sglang.test.send_one import BenchArgs, send_one_prompt
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_minimax_m25_basic.py` added +84/-0
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_minimax_m25_basic.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20919 - [NPU] Support dp-attention for MiniMax2.5

- 链接: https://github.com/sgl-project/sglang/pull/20919
- 状态/时间: merged / 2026-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `ae38b24cc358`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+104/-40，可读 patch 298 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Support dp-attention for MiniMax2.5」；模型线: MiniMax M2 Series；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「[NPU] Support dp-attention for MiniMax2.5」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +82/-39 (121 lines); hunks: -30,7 +30,6; -41,6 +40,12; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, forward，涉及 `MiniMaxM2RMSNormTP, __init__, weight_loader`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +82/-39 (121 lines); hunks: -30,7 +30,6; -41,6 +40,12; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -30,7 +30,6 @@
-    get_tensor_model_parallel_rank,
@@ -41,6 +40,12 @@
+from sglang.srt.layers.dp_attention import (
+    attn_tp_all_reduce,
+    get_attention_tp_rank,
+    get_attention_tp_size,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +82/-39
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/hardware_backend/npu/moe/topk.py`, `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22300 - [NVIDIA] Fix FP8 gemm performance with fp16 models (MInimax-M2.5)

- 链接: https://github.com/sgl-project/sglang/pull/22300
- 状态/时间: open / 2026-04-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+30/-6，可读 patch 65 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NVIDIA] Fix FP8 gemm performance with fp16 models (MInimax-M2.5)」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/model_loader/utils.py`；技术摘要: 覆盖「[NVIDIA] Fix FP8 gemm performance with fp16 models (MInimax-M2.5)」；主要实现面是 `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/model_loader/utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +5/-2 (7 lines); hunks: -488,8 +488,11 @@ def flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(; symbols: flashinfer_gemm_w8a8_block_fp8_linear_with_fallback，涉及 `flashinfer_gemm_w8a8_block_fp8_linear_with_fallback`；`python/sglang/srt/layers/quantization/fp8.py` modified +5/-0 (5 lines); hunks: -474,11 +474,16 @@ def process_weights_after_loading_block_quant(self, layer:...; symbols: process_weights_after_loading_block_quant，涉及 `process_weights_after_loading_block_quant`；`python/sglang/srt/model_loader/utils.py` modified +20/-4 (24 lines); hunks: -259,13 +259,29 @@ def post_load_weights(model: nn.Module, model_config: Mode...; symbols: post_load_weights, should_deepgemm_weight_requant_ue8m0, should_async_load，涉及 `post_load_weights, should_deepgemm_weight_requant_ue8m0, should_async_load`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +5/-2 (7 lines); hunks: -488,8 +488,11 @@ def flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(; symbols: flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
  - `python/sglang/srt/layers/quantization/fp8.py` modified +5/-0 (5 lines); hunks: -474,11 +474,16 @@ def process_weights_after_loading_block_quant(self, layer:...; symbols: process_weights_after_loading_block_quant
  - `python/sglang/srt/model_loader/utils.py` modified +20/-4 (24 lines); hunks: -259,13 +259,29 @@ def post_load_weights(model: nn.Module, model_config: Mode...; symbols: post_load_weights, should_deepgemm_weight_requant_ue8m0, should_async_load
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_utils.py
@@ -488,8 +488,11 @@ def flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(
-    # TRTLLM backend requires K dimension >= 256.
-    if backend == "trtllm" and input_2d.shape[1] < 256:
+    # TRTLLM backend requires K >= 256 and weight scales in UE8M0/R128c4
+    # packed format. Fall back to triton when scales are plain float32.
+    if backend == "trtllm" and (
+        input_2d.shape[1] < 256 or not getattr(weight_scale, "format_ue8m0", False)
diff -- python/sglang/srt/layers/quantization/fp8.py
@@ -474,11 +474,16 @@ def process_weights_after_loading_block_quant(self, layer: Module) -> None:
+            # Only requantize to UE8M0 if DeepGEMM can actually run
+            # this layer. If the dtype or shape is unsupported, the GEMM
+            # falls back to triton at runtime, which needs float32 scales.
+                    output_dtype=getattr(layer, "orig_dtype", None),
+                    weight_shape=layer.weight.shape,
diff -- python/sglang/srt/model_loader/utils.py
@@ -259,13 +259,29 @@ def post_load_weights(model: nn.Module, model_config: ModelConfig):
-def should_deepgemm_weight_requant_ue8m0(weight_block_size):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +5/-2; `python/sglang/srt/layers/quantization/fp8.py` modified +5/-0; `python/sglang/srt/model_loader/utils.py` modified +20/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/model_loader/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20967 - 【BugFix】fix the bug of minimax_m2.5 model that causes repeated outputs when using tp16

- 链接: https://github.com/sgl-project/sglang/pull/20967
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `84194c25c1cd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+34/-10，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「【BugFix】fix the bug of minimax_m2.5 model that causes repeated outputs when using tp16」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「【BugFix】fix the bug of minimax_m2.5 model that causes repeated outputs when using tp16」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +34/-10 (44 lines); hunks: -253,27 +253,47 @@ def rms_apply_serial(; -641,10 +661,14 @@ def __init__(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, weight_loader，涉及 `rms_apply_serial, MiniMaxM2RMSNormTP, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +34/-10 (44 lines); hunks: -253,27 +253,47 @@ def rms_apply_serial(; -641,10 +661,14 @@ def __init__(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, weight_loader
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -253,27 +253,47 @@ def rms_apply_serial(
-    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
+    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-6) -> None:
+        # Align with QKVParallelLinear pattern
+        if self.attn_tp_size >= num_heads:
+            assert (
+                self.attn_tp_size % num_heads == 0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +34/-10
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20067 - MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn

- 链接: https://github.com/sgl-project/sglang/pull/20067
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`；关联提交 `7dbd0dd9f01a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+39/-6，可读 patch 106 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn」；模型线: MiniMax M2 Series；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`；技术摘要: 覆盖「MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +25/-6 (31 lines); hunks: -53,10 +53,13; -417,12 +420,20 @@ def forward_normal(; symbols: forward_normal, forward_prepare, forward_core, __init__，涉及 `forward_normal, forward_prepare, forward_core`；`test/registered/8-gpu-models/test_minimax_m25.py` modified +10/-0 (10 lines); hunks: -29,6 +29,10 @@ def test_minimax_m25(self):; -37,6 +41,12 @@ def test_minimax_m25(self):; symbols: test_minimax_m25，涉及 `test_minimax_m25`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +25/-6 (31 lines); hunks: -53,10 +53,13; -417,12 +420,20 @@ def forward_normal(; symbols: forward_normal, forward_prepare, forward_core, __init__
  - `test/registered/8-gpu-models/test_minimax_m25.py` modified +10/-0 (10 lines); hunks: -29,6 +29,10 @@ def test_minimax_m25(self):; -37,6 +41,12 @@ def test_minimax_m25(self):; symbols: test_minimax_m25
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -53,10 +53,13 @@
+from sglang.srt.layers.moe import (
+    get_moe_a2a_backend,
+    should_use_flashinfer_cutlass_moe_fp4_allgather,
+)
-from sglang.srt.layers.moe.utils import get_moe_a2a_backend
@@ -417,12 +420,20 @@ def forward_normal(
diff -- test/registered/8-gpu-models/test_minimax_m25.py
@@ -29,6 +29,10 @@ def test_minimax_m25(self):
+        dp_attn_args = base_args + [
+            "--enable-dp-attention",
+            "--dp=8",
+        ]
@@ -37,6 +41,12 @@ def test_minimax_m25(self):
+            ModelLaunchSettings(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +25/-6
  - tests: `test/registered/8-gpu-models/test_minimax_m25.py` modified +10/-0
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_minimax_m25.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20673 - [Feature][JIT Kernel] Fused TP QK norm For Minimax

- 链接: https://github.com/sgl-project/sglang/pull/20673
- 状态/时间: merged / 2026-04-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `314d6ecf0880`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+923/-82，可读 patch 1277 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature][JIT Kernel] Fused TP QK norm For Minimax」；模型线: MiniMax M2 Series；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「[Feature][JIT Kernel] Fused TP QK norm For Minimax」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +113/-21 (134 lines); hunks: -17,17 +17,23; -42,6 +48,7; symbols: forward, fused_tp_qknorm, MiniMaxM2QKRMSNorm, __init__，涉及 `forward, fused_tp_qknorm, MiniMaxM2QKRMSNorm`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +113/-21 (134 lines); hunks: -17,17 +17,23; -42,6 +48,7; symbols: forward, fused_tp_qknorm, MiniMaxM2QKRMSNorm, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -17,17 +17,23 @@
-from typing import Iterable, Optional, Set, Tuple, Union
+from functools import lru_cache
+from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union
+from sglang.jit_kernel.all_reduce import (
+    fused_parallel_qknorm,
+    get_fused_parallel_qknorm_max_occupancy,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +113/-21
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_custom_all_reduce.py`, `python/sglang/jit_kernel/tests/test_tp_qknorm.py`, `python/sglang/jit_kernel/tests/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22744 - [NVIDIA] Support TF32 matmul to improve MiniMax gate gemm performance

- 链接: https://github.com/sgl-project/sglang/pull/22744
- 状态/时间: open / 2026-04-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+11/-0，可读 patch 39 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NVIDIA] Support TF32 matmul to improve MiniMax gate gemm performance」；模型线: MiniMax M2 Series；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md`；技术摘要: 覆盖「[NVIDIA] Support TF32 matmul to improve MiniMax gate gemm performance」；主要实现面是 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunks: -449,6 +449,10 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: -687,6 +687,7 @@ class ServerArgs:; -6074,6 +6075,11 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args，涉及 `ServerArgs, add_cli_args`；`docs/advanced_features/server_arguments.md` modified +1/-0 (1 lines); hunks: -470,6 +470,7 @@ Please consult the documentation below and [server_args.py](...。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunks: -449,6 +449,10 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: -687,6 +687,7 @@ class ServerArgs:; -6074,6 +6075,11 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args
  - `docs/advanced_features/server_arguments.md` modified +1/-0 (1 lines); hunks: -470,6 +470,7 @@ Please consult the documentation below and [server_args.py](...
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -449,6 +449,10 @@ def __init__(
+        # Set float32 matmul precision
+        if server_args.enable_tf32_matmul:
+            torch.set_float32_matmul_precision("high")
diff -- python/sglang/srt/server_args.py
@@ -687,6 +687,7 @@ class ServerArgs:
+    enable_tf32_matmul: bool = False
@@ -6074,6 +6075,11 @@ def add_cli_args(parser: argparse.ArgumentParser):
+        parser.add_argument(
+            "--enable-tf32-matmul",
+            action="store_true",
+            help="Enable float32 matmuls to use TensorFloat32 precision for better performance.",
diff -- docs/advanced_features/server_arguments.md
@@ -470,6 +470,7 @@ Please consult the documentation below and [server_args.py](https://github.com/s
+| `--enable-tf32-matmul` | Enables float32 matmuls to use TensorFloat32 precision for better performance. | `False` | bool flag (set to enable) |
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +4/-0; `python/sglang/srt/server_args.py` modified +6/-0
  - docs: `docs/advanced_features/server_arguments.md` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22722 - [AMD] Add MiniMax-M2.7 accuracy and performance nightly tests

- 链接: https://github.com/sgl-project/sglang/pull/22722
- 状态/时间: merged / 2026-04-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_minimax_m27_perf_amd.py`, `test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py`；关联提交 `eab045b2b74e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+805/-113，可读 patch 1069 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add MiniMax-M2.7 accuracy and performance nightly tests」；模型线: MiniMax M2 Series；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/minimax_m2.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py`；技术摘要: 覆盖「[AMD] Add MiniMax-M2.7 accuracy and performance nightly tests」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +7/-1 (8 lines); hunks: -33,7 +33,6; -81,9 +80,16；`test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, __post_init__, get_display_name, get_one_example，涉及 `ModelConfig, __post_init__, get_display_name`；`test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: ModelConfig, __post_init__, get_display_name, get_one_example，涉及 `ModelConfig, __post_init__, get_display_name`；`test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM27PerformanceMI35x, setUpClass, test_bench_minimax_m27，涉及 `generate_simple_markdown_report, TestNightlyMiniMaxM27PerformanceMI35x, setUpClass`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +7/-1 (8 lines); hunks: -33,7 +33,6; -81,9 +80,16
  - `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, __post_init__, get_display_name, get_one_example
  - `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: ModelConfig, __post_init__, get_display_name, get_one_example
  - `test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM27PerformanceMI35x, setUpClass, test_bench_minimax_m27
  - `test/registered/amd/perf/mi30x/test_minimax_m27_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyMiniMaxM27Performance, setUpClass, test_bench_minimax_m27
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -33,7 +33,6 @@
-    get_bool_env_var,
@@ -81,9 +80,16 @@
+# get_bool_env_var is defined in sglang.srt.utils.common, not sglang.srt.distributed.
+# Importing from the wrong module causes this file to fail import, which prevents the
+# native MiniMaxM2ForCausalLM from registering in ModelRegistry. The fallback to the
+# transformers wrapper then crashes on config.rope_parameters (transformers v5 issue).
diff -- test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py
@@ -0,0 +1,249 @@
+"""MI35x MiniMax-M2.7 GSM8K Completion Evaluation Test (8-GPU)
+Tests MiniMax-M2.7 with TP=8 + EP=8 configuration using few-shot completion
+benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-minimax-m27 suite
+"""
+import ast
diff -- test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py
@@ -0,0 +1,245 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +7/-1
  - tests: `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py` added +249/-0; `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py` added +245/-0; `test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py` added +146/-0; `test/registered/amd/perf/mi30x/test_minimax_m27_perf_amd.py` added +140/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_minimax_m27_perf_amd.py`, `test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22934 - Minimax eplb bugfix

- 链接: https://github.com/sgl-project/sglang/pull/22934
- 状态/时间: open / 2026-04-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+25/-0，可读 patch 53 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Minimax eplb bugfix」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「Minimax eplb bugfix」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +25/-0 (25 lines); hunks: -66,6 +66,7; -88,6 +89,7; symbols: op_output, get_moe_weights, MiniMaxM2Attention, __init__，涉及 `op_output, get_moe_weights, MiniMaxM2Attention`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +25/-0 (25 lines); hunks: -66,6 +66,7; -88,6 +89,7; symbols: op_output, get_moe_weights, MiniMaxM2Attention, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -66,6 +66,7 @@
+from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert
@@ -88,6 +89,7 @@
+    LazyValue,
@@ -683,6 +685,16 @@ def op_output(self, state):
+    def get_moe_weights(self):
+        return [
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +25/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23301 - [sgl] Stream MiniMax M2 string parameters token-by-token

- 链接: https://github.com/sgl-project/sglang/pull/23301
- 状态/时间: open / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+332/-280，可读 patch 742 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[sgl] Stream MiniMax M2 string parameters token-by-token」；模型线: MiniMax M2 Series；类别: 模型实现调整；主要 diff: `python/sglang/srt/function_call/minimax_m2.py`；技术摘要: 覆盖「[sgl] Stream MiniMax M2 string parameters token-by-token」；主要实现面是 `python/sglang/srt/function_call/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/minimax_m2.py` modified +332/-280 (612 lines); hunks: -13,6 +13,11; -24,6 +29,9 @@ class MinimaxM2Detector(BaseFormatDetector):; symbols: MinimaxM2Detector, __init__，涉及 `MinimaxM2Detector, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/minimax_m2.py` modified +332/-280 (612 lines); hunks: -13,6 +13,11; -24,6 +29,9 @@ class MinimaxM2Detector(BaseFormatDetector):; symbols: MinimaxM2Detector, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/minimax_m2.py
@@ -13,6 +13,11 @@
+_PARAM_END_TAG = "</parameter>"
+_PARAM_END_TAG_LEN = len(_PARAM_END_TAG)
+# Hold back this many chars while streaming to avoid emitting a partial end tag
+_STREAM_HOLD_BACK = _PARAM_END_TAG_LEN - 1  # 11
@@ -24,6 +29,9 @@ class MinimaxM2Detector(BaseFormatDetector):
+    String-typed parameters are streamed token-by-token.
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/minimax_m2.py` modified +332/-280
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22432 - [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2

- 链接: https://github.com/sgl-project/sglang/pull/22432
- 状态/时间: closed / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+69/-11，可读 patch 154 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「[NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +69/-11 (80 lines); hunks: -17,7 +17,7; -42,6 +42,7; symbols: forward_prepare, forward_prepare_npu, forward_core, forward，涉及 `forward_prepare, forward_prepare_npu, forward_core`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +69/-11 (80 lines); hunks: -17,7 +17,7; -42,6 +42,7; symbols: forward_prepare, forward_prepare_npu, forward_core, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -17,7 +17,7 @@
-from typing import Iterable, Optional, Set, Tuple, Union
+from typing import Iterable, List, Optional, Set, Tuple, Union
@@ -42,6 +42,7 @@
+    get_attention_tp_group,
@@ -76,10 +77,16 @@
+    is_npu,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +69/-11
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23190 - [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode

- 链接: https://github.com/sgl-project/sglang/pull/23190
- 状态/时间: merged / 2026-04-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/minimax_m2.py`；关联提交 `3553fd032251`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+66/-10，可读 patch 133 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/minimax_m2.py`；技术摘要: 覆盖「[NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode」；主要实现面是 `python/sglang/srt/models/minimax_m2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minimax_m2.py` modified +66/-10 (76 lines); hunks: -18,7 +18,7; -93,13 +93,18; symbols: forward_prepare, forward_prepare_npu, forward_core, forward，涉及 `forward_prepare, forward_prepare_npu, forward_core`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minimax_m2.py` modified +66/-10 (76 lines); hunks: -18,7 +18,7; -93,13 +93,18; symbols: forward_prepare, forward_prepare_npu, forward_core, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minimax_m2.py
@@ -18,7 +18,7 @@
-from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union
+from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
@@ -93,13 +93,18 @@
+    is_npu,
+_is_npu = is_npu()
+if _is_npu:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minimax_m2.py` modified +66/-10
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
