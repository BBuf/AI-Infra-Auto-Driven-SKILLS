# sglang GPT-OSS 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `benchmark/gpt_oss/README.md` | [#9728](https://github.com/sgl-project/sglang/pull/9728) |
| `docs/basic_usage/gpt_oss.md` | [#9497](https://github.com/sgl-project/sglang/pull/9497), [#9613](https://github.com/sgl-project/sglang/pull/9613), [#9626](https://github.com/sgl-project/sglang/pull/9626) |
| `docs_new/cookbook/autoregressive/OpenAI/GPT-OSS.mdx` | 无直接 PR 号提交 |
| `docs_new/docs/basic_usage/gpt_oss.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/gpt-oss-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/function_call/gpt_oss_detector.py` | [#9043](https://github.com/sgl-project/sglang/pull/9043), [#9190](https://github.com/sgl-project/sglang/pull/9190), [#9657](https://github.com/sgl-project/sglang/pull/9657) |
| `python/sglang/srt/models/gpt_oss.py` | [#8824](https://github.com/sgl-project/sglang/pull/8824), [#8843](https://github.com/sgl-project/sglang/pull/8843), [#8944](https://github.com/sgl-project/sglang/pull/8944), [#9028](https://github.com/sgl-project/sglang/pull/9028), [#9146](https://github.com/sgl-project/sglang/pull/9146), [#9161](https://github.com/sgl-project/sglang/pull/9161), [#9359](https://github.com/sgl-project/sglang/pull/9359), [#9433](https://github.com/sgl-project/sglang/pull/9433), [#9469](https://github.com/sgl-project/sglang/pull/9469), [#9783](https://github.com/sgl-project/sglang/pull/9783), [#14197](https://github.com/sgl-project/sglang/pull/14197), [#17553](https://github.com/sgl-project/sglang/pull/17553), ... (15 total) |
| `python/sglang/test/gpt_oss_common.py` | [#16426](https://github.com/sgl-project/sglang/pull/16426) |
| `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` | [#18869](https://github.com/sgl-project/sglang/pull/18869), [#22237](https://github.com/sgl-project/sglang/pull/22237) |
| `test/registered/8-gpu-models/test_gpt_oss_120b.py` | [#18134](https://github.com/sgl-project/sglang/pull/18134) |
| `test/registered/amd/accuracy/mi30x/test_gpt_oss_eval_amd.py` | 无直接 PR 号提交 |
| `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` | 无直接 PR 号提交 |
| `test/registered/core/test_gpt_oss_1gpu.py` | [#16426](https://github.com/sgl-project/sglang/pull/16426) |
| `test/registered/core/test_gpt_oss_sm120.py` | [#20056](https://github.com/sgl-project/sglang/pull/20056) |
| `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` | [#21570](https://github.com/sgl-project/sglang/pull/21570) |
| `test/registered/perf/test_gpt_oss_4gpu_perf.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 27
- 原文档显式引用补充 PR 数: 2
- 当前文档总 PR 数: 29
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-08-05 | [#8824](https://github.com/sgl-project/sglang/pull/8824) | merged | Add initial support for gpt-oss | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-06 | [#8843](https://github.com/sgl-project/sglang/pull/8843) | merged | Support mxfp4 for GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-08 | [#8944](https://github.com/sgl-project/sglang/pull/8944) | merged | Expert Parallelism for GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-12 | [#9043](https://github.com/sgl-project/sglang/pull/9043) | merged | (gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support | `python/sglang/srt/function_call/gpt_oss_detector.py` |
| 2025-08-13 | [#9146](https://github.com/sgl-project/sglang/pull/9146) | merged | Fix gpt-oss ~2x memory consumption issue | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-13 | [#9028](https://github.com/sgl-project/sglang/pull/9028) | merged | Support FA3 backend for gpt-oss | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-13 | [#9161](https://github.com/sgl-project/sglang/pull/9161) | merged | Fix broken trtllm_mha attn backend with gpt-oss | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-20 | [#9359](https://github.com/sgl-project/sglang/pull/9359) | merged | Support DP attention with GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-21 | [#9433](https://github.com/sgl-project/sglang/pull/9433) | merged | [fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-22 | [#9469](https://github.com/sgl-project/sglang/pull/9469) | merged | fix: tmp revert gpt oss tp sharding on hopper | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-22 | [#9497](https://github.com/sgl-project/sglang/pull/9497) | merged | [Docs] Add doc and quick demo for gpt-oss responses api & buildin tools | `docs/basic_usage/gpt_oss.md` |
| 2025-08-25 | [#9190](https://github.com/sgl-project/sglang/pull/9190) | merged | Fix Harmony reasoning parser for and auto-separation for gpt-oss models | `python/sglang/srt/function_call/gpt_oss_detector.py` |
| 2025-08-25 | [#9613](https://github.com/sgl-project/sglang/pull/9613) | merged | [docs] Refactor, remove compiled results and add gpt-oss | `docs/basic_usage/gpt_oss.md` |
| 2025-08-28 | [#9728](https://github.com/sgl-project/sglang/pull/9728) | merged | gpt-oss blog reproduction document | `benchmark/gpt_oss/README.md` |
| 2025-09-01 | [#9783](https://github.com/sgl-project/sglang/pull/9783) | merged | support fp8 kvcache for hybrid attn backend on GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-09-15 | [#9626](https://github.com/sgl-project/sglang/pull/9626) | merged | Add reasoning examples for GPT-OSS in Markdown examples | `docs/basic_usage/gpt_oss.md` |
| 2025-09-15 | [#9657](https://github.com/sgl-project/sglang/pull/9657) | merged | fix: gpt-oss streaming dropping normal content when tools are provided but not used | `python/sglang/srt/function_call/gpt_oss_detector.py` |
| 2025-12-30 | [#14920](https://github.com/sgl-project/sglang/pull/14920) | merged | Eagle: GPT-OSS Eagle v2 support | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_worker.py` |
| 2026-01-07 | [#16426](https://github.com/sgl-project/sglang/pull/16426) | merged | Fix gpt_oss_common import path and migrate core tests | `test/registered/core/test_gpt_oss_1gpu.py`, `python/sglang/test/gpt_oss_common.py` |
| 2026-01-18 | [#14197](https://github.com/sgl-project/sglang/pull/14197) | merged | [NPU]Support GPT-OSS for NPU | `python/sglang/srt/models/gpt_oss.py` |
| 2026-01-22 | [#17553](https://github.com/sgl-project/sglang/pull/17553) | merged | [NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py | `python/sglang/srt/models/gpt_oss.py` |
| 2026-02-03 | [#18134](https://github.com/sgl-project/sglang/pull/18134) | merged | feature: adding gpt-oss 120b nightly test | `test/registered/8-gpu-models/test_gpt_oss_120b.py` |
| 2026-02-12 | [#18405](https://github.com/sgl-project/sglang/pull/18405) | merged | [PCG] GPT OSS Triton Kernel Support | `python/sglang/srt/models/gpt_oss.py` |
| 2026-02-16 | [#18869](https://github.com/sgl-project/sglang/pull/18869) | merged | [CI] Remove `--mem-fraction-static 0.93` from gpt-oss test | `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` |
| 2026-02-20 | [#18988](https://github.com/sgl-project/sglang/pull/18988) | merged | [GPT-OSS] support fp8 online quantization for gpt-oss bf16 | `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py` |
| 2026-03-06 | [#20056](https://github.com/sgl-project/sglang/pull/20056) | merged | [CI] Add GPT-OSS test for SM120 | `test/registered/core/test_gpt_oss_sm120.py` |
| 2026-03-24 | [#20755](https://github.com/sgl-project/sglang/pull/20755) | merged | Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+ | `python/sglang/srt/models/gpt_oss.py` |
| 2026-04-02 | [#21570](https://github.com/sgl-project/sglang/pull/21570) | merged | [4/n] Support gpt oss 20b lora | `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` |
| 2026-04-08 | [#22237](https://github.com/sgl-project/sglang/pull/22237) | merged | [CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58 | `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` |

## 逐 PR diff 审计卡

### PR #8824 - Add initial support for gpt-oss

- 链接: https://github.com/sgl-project/sglang/pull/8824
- 状态/时间: merged / 2025-08-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `c1d2061f97ae`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+1595/-47，可读 patch 2185 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add initial support for gpt-oss」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文摘要: Future progress will be tracked here: https://github.com/sgl-project/sglang/issues/8833 **This PR only works for FP8/BF16 ckpt. The FP8/BF16 ckpt has been uploaded to:** `lmsys/...。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` added +923/-0 (923 lines); hunks: -0,0 +1,923; symbols: GptOssConfig, __init__, get_attention_sliding_window_size, GptOssSparseMoeBlock，涉及 `GptOssConfig, __init__, get_attention_sliding_window_size`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` added +923/-0 (923 lines); hunks: -0,0 +1,923; symbols: GptOssConfig, __init__, get_attention_sliding_window_size, GptOssSparseMoeBlock
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -0,0 +1,923 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` added +923/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`, `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8843 - Support mxfp4 for GPT-OSS

- 链接: https://github.com/sgl-project/sglang/pull/8843
- 状态/时间: merged / 2025-08-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `168033d5fb1e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+791/-325，可读 patch 1320 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support mxfp4 for GPT-OSS」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +209/-9 (218 lines); hunks: -25,6 +25,8; -108,11 +110,15 @@ def __init__(; symbols: __init__, _get_default_weight_mapping, load_weights，涉及 `__init__, _get_default_weight_mapping, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +209/-9 (218 lines); hunks: -25,6 +25,8; -108,11 +110,15 @@ def __init__(; symbols: __init__, _get_default_weight_mapping, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -25,6 +25,8 @@
+    get_moe_expert_parallel_rank,
+    get_moe_expert_parallel_world_size,
@@ -108,11 +110,15 @@ def __init__(
+            quant_config_name = (
+                quant_config.get_name() if quant_config is not None else None
+            )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +209/-9
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/layers/quantization/__init__.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8944 - Expert Parallelism for GPT-OSS

- 链接: https://github.com/sgl-project/sglang/pull/8944
- 状态/时间: merged / 2025-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `1d24db834803`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+269/-119，可读 patch 956 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Expert Parallelism for GPT-OSS」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文摘要: - What's in this PR: - Enable GPT-OSS launch without triton-kernels - Support expert parallelism for GPT-OSS Example: - TODO: Benchmark & Profiling。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +54/-47 (101 lines); hunks: -28,6 +28,7; -96,11 +97,6 @@ def __init__(; symbols: __init__, _load_mxfp4_experts_weights，涉及 `__init__, _load_mxfp4_experts_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +54/-47 (101 lines); hunks: -28,6 +28,7; -96,11 +97,6 @@ def __init__(; symbols: __init__, _load_mxfp4_experts_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,6 +28,7 @@
+    get_moe_tensor_parallel_world_size,
@@ -96,11 +97,6 @@ def __init__(
-        if self.tp_size > config.num_local_experts:
-            raise ValueError(
-                f"Tensor parallel size {self.tp_size} is greater than "
-                f"the number of experts {config.num_local_experts}."
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +54/-47
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9043 - (gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support

- 链接: https://github.com/sgl-project/sglang/pull/9043
- 状态/时间: merged / 2025-08-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/gpt_oss_detector.py`；关联提交 `a21849013607`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+717/-409，可读 patch 1293 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「(gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support」；模型线: GPT-OSS；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/gpt_oss_detector.py`；PR 正文摘要: Why remove Harmony Harmony integration was removed due to two critical limitations: 1. Missing output token ID support: Harmony requires output token IDs for proper functioning,...。
- 实现要点: `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0 (331 lines); hunks: -0,0 +1,331; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse，涉及 `GptOssDetector, __init__, has_tool_call`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0 (331 lines); hunks: -0,0 +1,331; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/gpt_oss_detector.py
@@ -0,0 +1,331 @@
+import json
+import logging
+import re
+from typing import List
+from sglang.srt.entrypoints.openai.protocol import Tool
+from sglang.srt.function_call.base_format_detector import BaseFormatDetector
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/entrypoints/harmony_utils.py`, `python/sglang/srt/entrypoints/http_server.py`, `python/sglang/srt/entrypoints/openai/protocol.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9146 - Fix gpt-oss ~2x memory consumption issue

- 链接: https://github.com/sgl-project/sglang/pull/9146
- 状态/时间: merged / 2025-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `9394ed63867d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+19/-7，可读 patch 47 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix gpt-oss ~2x memory consumption issue」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +19/-7 (26 lines); hunks: -64,7 +64,13; -655,6 +661,18 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, forward, _load_normal_weights，涉及 `__init__, routed_experts_weights_of_layer, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +19/-7 (26 lines); hunks: -64,7 +64,13; -655,6 +661,18 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, forward, _load_normal_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -64,7 +64,13 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_flashinfer_available, make_layers
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
+    is_cuda,
+    is_flashinfer_available,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +19/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9028 - Support FA3 backend for gpt-oss

- 链接: https://github.com/sgl-project/sglang/pull/9028
- 状态/时间: merged / 2025-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `0ff6d1fce122`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+24/-6，可读 patch 121 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support FA3 backend for gpt-oss」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文摘要: Apply changes of https://github.com/sgl-project/sgl-attn/pull/4. `openai/gpt-oss-20b` mmlu 4k: Benchmark & Profiling `openai/gpt-oss-20b` TP1 4k in 1k out Triton: FA3: `openai/g...。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -294,7 +294,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -294,7 +294,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -294,7 +294,7 @@ def __init__(
-            torch.empty(self.num_heads, dtype=torch.float32), requires_grad=False
+            torch.empty(self.num_heads, dtype=torch.bfloat16), requires_grad=False
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/pyproject.toml`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9161 - Fix broken trtllm_mha attn backend with gpt-oss

- 链接: https://github.com/sgl-project/sglang/pull/9161
- 状态/时间: merged / 2025-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `6b7c24712cda`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix broken trtllm_mha attn backend with gpt-oss」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文摘要: FA3 PR #9028 broke trtllm_mha attn backend.。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +5/-1 (6 lines); hunks: -293,8 +293,12 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +5/-1 (6 lines); hunks: -293,8 +293,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -293,8 +293,12 @@ def __init__(
+        # Choose dtype of sinks based on attention backend: trtllm_mha requires float32,
+        # others can use bfloat16
+        attn_backend = global_server_args_dict.get("attention_backend")
+        sinks_dtype = torch.float32 if attn_backend == "trtllm_mha" else torch.bfloat16
-            torch.empty(self.num_heads, dtype=torch.bfloat16), requires_grad=False
+            torch.empty(self.num_heads, dtype=sinks_dtype), requires_grad=False
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9359 - Support DP attention with GPT-OSS

- 链接: https://github.com/sgl-project/sglang/pull/9359
- 状态/时间: merged / 2025-08-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `c10b8e6a0f2a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-5，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support DP attention with GPT-OSS」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文摘要: Tested on 4x B200 with DP4 Attn + EP4 MoE Total output: `66,941.40 tok/s` Total output per GPU: `16,735.35 tok/s/gpu` Cmds:。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -1123,7 +1123,7 @@ def _load_normal_weights(; symbols: _load_normal_weights，涉及 `_load_normal_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -1123,7 +1123,7 @@ def _load_normal_weights(; symbols: _load_normal_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -1123,7 +1123,7 @@ def _load_normal_weights(
-                            start = tp_rank * param.numel()
+                            start = get_attention_tp_rank() * param.numel()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9433 - [fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS

- 链接: https://github.com/sgl-project/sglang/pull/9433
- 状态/时间: merged / 2025-08-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `dae9a80f43e8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+11/-3，可读 patch 46 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文摘要: gpqa and mmlu with tp=4 were incorrect. The tp=1 and tep=4 scores are correct though. The bug is due to incorrect TP sharding of mxfp4 MoE weights, which missed loading the last...。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +9/-1 (10 lines); hunks: -16,6 +16,7; -788,18 +789,25 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights，涉及 `_load_mxfp4_experts_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +9/-1 (10 lines); hunks: -16,6 +16,7; -788,18 +789,25 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -16,6 +16,7 @@
+import math
@@ -788,18 +789,25 @@ def _load_mxfp4_experts_weights(self, weights):
+        assert (
+            intermediate_size % mxfp4_block == 0
+        ), f"{intermediate_size=} must be divisible by {mxfp4_block=}"
-        per_rank_intermediate_size_block = intermediate_size_block // moe_tp_size
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +9/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/entrypoints/openai/protocol.py`, `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9469 - fix: tmp revert gpt oss tp sharding on hopper

- 链接: https://github.com/sgl-project/sglang/pull/9469
- 状态/时间: merged / 2025-08-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `849957bc76c3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-3，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: tmp revert gpt oss tp sharding on hopper」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +6/-3 (9 lines); hunks: -793,9 +793,12 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights，涉及 `_load_mxfp4_experts_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +6/-3 (9 lines); hunks: -793,9 +793,12 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -793,9 +793,12 @@ def _load_mxfp4_experts_weights(self, weights):
-        per_rank_intermediate_size_block = math.ceil(
-            intermediate_size_block / moe_tp_size
-        )
+        if _is_sm100_supported:
+            per_rank_intermediate_size_block = math.ceil(
+                intermediate_size_block / moe_tp_size
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +6/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9497 - [Docs] Add doc and quick demo for gpt-oss responses api & buildin tools

- 链接: https://github.com/sgl-project/sglang/pull/9497
- 状态/时间: merged / 2025-08-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/basic_usage/gpt_oss.md`；关联提交 `fedfe91c1a6e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+106/-0，可读 patch 110 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add doc and quick demo for gpt-oss responses api & buildin tools」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `docs/basic_usage/gpt_oss.md`；PR 正文未提供可用摘要。
- 实现要点: `docs/basic_usage/gpt_oss.md` modified +106/-0 (106 lines); hunks: -1,3 +1,109。
- 代码 diff 细节:
  - `docs/basic_usage/gpt_oss.md` modified +106/-0 (106 lines); hunks: -1,3 +1,109
- 关键代码摘录:

```diff
diff -- docs/basic_usage/gpt_oss.md
@@ -1,3 +1,109 @@
+## Responses API & Built-in Tools
+### Responses API
+GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use.
+### Built-in Tools
+GPT‑OSS can call built‑in tools for web search and Python execution. You can use the demo tool server or connect to external MCP tool servers.
+#### Python Tool
```

- 已读文件:
  - docs: `docs/basic_usage/gpt_oss.md` modified +106/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/basic_usage/gpt_oss.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #9190 - Fix Harmony reasoning parser for and auto-separation for gpt-oss models

- 链接: https://github.com/sgl-project/sglang/pull/9190
- 状态/时间: merged / 2025-08-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/gpt_oss_detector.py`；关联提交 `a0a77d937b99`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+1681/-556，可读 patch 2406 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Harmony reasoning parser for and auto-separation for gpt-oss models」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/gpt_oss_detector.py`；PR 正文摘要: This PR fixes critical regressions in Harmony reasoning output parsing that were introduced in PR #9043. After that change, gpt-oss models were incorrectly concatenating analysi...。
- 实现要点: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +144/-256 (400 lines); hunks: -1,7 +1,7; -10,60 +10,31; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse，涉及 `GptOssDetector, __init__, has_tool_call`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/gpt_oss_detector.py` modified +144/-256 (400 lines); hunks: -1,7 +1,7; -10,60 +10,31; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/gpt_oss_detector.py
@@ -1,7 +1,7 @@
-from typing import List
+from typing import List, Optional
@@ -10,60 +10,31 @@
+from sglang.srt.harmony_parser import HarmonyParser
-    Detector for T4-style function calls with channel format.
+    Detector for T4-style function calls using HarmonyParser.
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +144/-256
- 验证与风险: diff 自带测试面 `test/srt/run_suite.py`, `test/srt/test_harmony_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #9613 - [docs] Refactor, remove compiled results and add gpt-oss

- 链接: https://github.com/sgl-project/sglang/pull/9613
- 状态/时间: merged / 2025-08-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/basic_usage/gpt_oss.md`；关联提交 `9b08d975a0a5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+166/-611，可读 patch 638 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Refactor, remove compiled results and add gpt-oss」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `docs/basic_usage/gpt_oss.md`；PR 正文摘要: 1. I refactored `docs/advanced_features/function_calling.ipynb`, making sure the servers are turned off as expected. I moved `Tool Choice Mode` to the end of the docs, and I fin...。
- 实现要点: `docs/basic_usage/gpt_oss.md` modified +5/-0 (5 lines); hunks: -23,6 +23,11 @@ GPT‑OSS can call built‑in tools for web search and Python exe...。
- 代码 diff 细节:
  - `docs/basic_usage/gpt_oss.md` modified +5/-0 (5 lines); hunks: -23,6 +23,11 @@ GPT‑OSS can call built‑in tools for web search and Python exe...
- 关键代码摘录:

```diff
diff -- docs/basic_usage/gpt_oss.md
@@ -23,6 +23,11 @@ GPT‑OSS can call built‑in tools for web search and Python execution. You can
+### Tool & Reasoning Parser
+- We support OpenAI Reasoning and Tool Call parser, as well as our SGLang native api for tool call and reasoning. Refer to [reasoning parser](../advanced_features/separate_reasoni
```

- 已读文件:
  - docs: `docs/basic_usage/gpt_oss.md` modified +5/-0
- 验证与风险: runtime 路径改动集中在 `scripts/playground/frontend_reasoning.ipynb`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9728 - gpt-oss blog reproduction document

- 链接: https://github.com/sgl-project/sglang/pull/9728
- 状态/时间: merged / 2025-08-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `benchmark/gpt_oss/README.md`；关联提交 `d0934a519257`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+163/-0，可读 patch 164 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「gpt-oss blog reproduction document」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `benchmark/gpt_oss/README.md`；PR 正文未提供可用摘要。
- 实现要点: `benchmark/gpt_oss/README.md` added +163/-0 (163 lines); hunks: -0,0 +1,163。
- 代码 diff 细节:
  - `benchmark/gpt_oss/README.md` added +163/-0 (163 lines); hunks: -0,0 +1,163
- 关键代码摘录:

```diff
diff -- benchmark/gpt_oss/README.md
@@ -0,0 +1,163 @@
+# How to reproduce the result of GPT-OSS with SGLang
+### Install the latest SGLang
+'''bash
+git clone https://github.com/sgl-project/sglang.git
+cd sglang
+git checkout v0.5.1.post3
```

- 已读文件:
  - other: `benchmark/gpt_oss/README.md` added +163/-0
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #9783 - support fp8 kvcache for hybrid attn backend on GPT-OSS

- 链接: https://github.com/sgl-project/sglang/pull/9783
- 状态/时间: merged / 2025-09-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `9db8025376b2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-4，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support fp8 kvcache for hybrid attn backend on GPT-OSS」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文摘要: 9782 On B200/GB200 kv cache volume actually blocks the batch size which is the bottleneck for the GPT-OSS performance. And the trtllm-mha cuda kernel could not support Q(bf16),...。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +5/-4 (9 lines); hunks: -193,8 +193,9 @@ def forward_normal(; -341,7 +342,7 @@ def forward_prepare(; symbols: forward_normal, _enable_fused_set_kv_buffer, forward_prepare, forward_core，涉及 `forward_normal, _enable_fused_set_kv_buffer, forward_prepare`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +5/-4 (9 lines); hunks: -193,8 +193,9 @@ def forward_normal(; -341,7 +342,7 @@ def forward_prepare(; symbols: forward_normal, _enable_fused_set_kv_buffer, forward_prepare, forward_core
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -193,8 +193,9 @@ def forward_normal(
-def _enable_fused_set_kv_buffer():
-    return _is_cuda
+def _enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
+    """Enable fused set_kv_buffer only on CUDA with bfloat16 KV cache."""
+    return _is_cuda and forward_batch.token_to_kv_pool.dtype == torch.bfloat16
@@ -341,7 +342,7 @@ def forward_prepare(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +5/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9626 - Add reasoning examples for GPT-OSS in Markdown examples

- 链接: https://github.com/sgl-project/sglang/pull/9626
- 状态/时间: merged / 2025-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/basic_usage/gpt_oss.md`；关联提交 `0b14159fc4e0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+12/-2，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add reasoning examples for GPT-OSS in Markdown examples」；模型线: GPT-OSS；类别: 模型支持/运行时入口；主要 diff: `docs/basic_usage/gpt_oss.md`；PR 正文摘要: A useful parameter, and it's not mentioned here already. The information is already in Hugging Face, but it's nice to have it in the examples as well.。
- 实现要点: `docs/basic_usage/gpt_oss.md` modified +11/-1 (12 lines); hunks: -6,7 +6,7 @@ Please refer to [https://github.com/sgl-project/sglang/issues/88...; -69,6 +69,16 @@ tools = [。
- 代码 diff 细节:
  - `docs/basic_usage/gpt_oss.md` modified +11/-1 (12 lines); hunks: -6,7 +6,7 @@ Please refer to [https://github.com/sgl-project/sglang/issues/88...; -69,6 +69,16 @@ tools = [
- 关键代码摘录:

```diff
diff -- docs/basic_usage/gpt_oss.md
@@ -6,7 +6,7 @@ Please refer to [https://github.com/sgl-project/sglang/issues/8833](https://gith
-GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use.
+GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use. Yo
@@ -69,6 +69,16 @@ tools = [
+# Reasoning level example
+response = client.responses.create(
+    model="openai/gpt-oss-120b",
```

- 已读文件:
  - docs: `docs/basic_usage/gpt_oss.md` modified +11/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/entrypoints/openai/protocol.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9657 - fix: gpt-oss streaming dropping normal content when tools are provided but not used

- 链接: https://github.com/sgl-project/sglang/pull/9657
- 状态/时间: merged / 2025-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/gpt_oss_detector.py`；关联提交 `28c79dc84ab8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+23/-0，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: gpt-oss streaming dropping normal content when tools are provided but not used」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/gpt_oss_detector.py`；PR 正文摘要: Streaming chat responses returned empty content when tools were provided but not actually invoked. For gpt-oss, the tool-call streaming path uses the Harmony parser, which filte...。
- 实现要点: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +23/-0 (23 lines); hunks: -81,6 +81,29 @@ def parse_streaming_increment(; symbols: parse_streaming_increment，涉及 `parse_streaming_increment`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/gpt_oss_detector.py` modified +23/-0 (23 lines); hunks: -81,6 +81,29 @@ def parse_streaming_increment(; symbols: parse_streaming_increment
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/gpt_oss_detector.py
@@ -81,6 +81,29 @@ def parse_streaming_increment(
+        # If there are no parsed events and the chunk contains no Harmony structural
+        # markers, treat it as plain text and pass it through. This fixes a bug where
+        # normal content was held in the buffer when tools were provided but not used.
+        if not events:
+            has_harmony_markers = any(
+                marker in self._buffer
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +23/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/gpt_oss_detector.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14920 - Eagle: GPT-OSS Eagle v2 support

- 链接: https://github.com/sgl-project/sglang/pull/14920
- 状态/时间: merged / 2025-12-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+48/-25，可读 patch 124 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Eagle: GPT-OSS Eagle v2 support」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_worker.py`；PR 正文摘要: EAGLE v2/v3 models from GPT-OSS introduce an optional auxiliary hidden-state mechanism that improves speculative decoding quality by exposing additional per-layer representation...。
- 实现要点: `python/sglang/srt/model_executor/model_runner.py` modified +30/-23 (53 lines); hunks: -345,6 +345,32 @@ def __init__(; -593,30 +619,11 @@ def initialize(self, min_per_gpu_memory: float):; symbols: __init__, initialize, _dummy_run，涉及 `__init__, initialize, _dummy_run`；`python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -349,7 +349,10 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__，涉及 `__init__`；`python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunks: -186,6 +186,15 @@ def __init__(; -897,6 +906,7 @@ def forward_draft_extend_after_decode(self, batch: ScheduleB...; symbols: __init__, forward_draft_extend_after_decode，涉及 `__init__, forward_draft_extend_after_decode`；`python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -100,7 +100,10 @@ def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/model_runner.py` modified +30/-23 (53 lines); hunks: -345,6 +345,32 @@ def __init__(; -593,30 +619,11 @@ def initialize(self, min_per_gpu_memory: float):; symbols: __init__, initialize, _dummy_run
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -349,7 +349,10 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__
  - `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunks: -186,6 +186,15 @@ def __init__(; -897,6 +906,7 @@ def forward_draft_extend_after_decode(self, batch: ScheduleB...; symbols: __init__, forward_draft_extend_after_decode
  - `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -100,7 +100,10 @@ def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -345,6 +345,32 @@ def __init__(
+        # auxiliary hidden capture mode. TODO: expose this to server args?
+        self.eagle_use_aux_hidden_state = False
+        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
+            # load draft config
+            draft_model_config = ModelConfig.from_server_args(
+                server_args,
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -349,7 +349,10 @@ def __init__(self, model_runner: ModelRunner):
-        if model_runner.spec_algorithm.is_eagle3():
+        if (
+            model_runner.spec_algorithm.is_eagle3()
+            and model_runner.eagle_use_aux_hidden_state
+        ):
diff -- python/sglang/srt/speculative/eagle_worker.py
@@ -186,6 +186,15 @@ def __init__(
+        self.eagle_use_aux_hidden_state = False
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +30/-23; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1; `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0; `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16426 - Fix gpt_oss_common import path and migrate core tests

- 链接: https://github.com/sgl-project/sglang/pull/16426
- 状态/时间: merged / 2026-01-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/test/gpt_oss_common.py`, `test/registered/core/test_gpt_oss_1gpu.py`；关联提交 `0c474273c514`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+48/-26，可读 patch 255 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix gpt_oss_common import path and migrate core tests」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `test/registered/core/test_gpt_oss_1gpu.py`, `python/sglang/test/gpt_oss_common.py`；PR 正文摘要: Fix sys.path hack by moving gpt_oss_common.py to python/sglang/test/ and migrate core tests to test/registered/core/.。
- 实现要点: `test/registered/core/test_gpt_oss_1gpu.py` renamed +5/-1 (6 lines); hunks: -1,6 +1,10; symbols: TestGptOss1Gpu，涉及 `TestGptOss1Gpu`；`python/sglang/test/gpt_oss_common.py` renamed +0/-0 (0 lines)。
- 代码 diff 细节:
  - `test/registered/core/test_gpt_oss_1gpu.py` renamed +5/-1 (6 lines); hunks: -1,6 +1,10; symbols: TestGptOss1Gpu
  - `python/sglang/test/gpt_oss_common.py` renamed +0/-0 (0 lines)
- 关键代码摘录:

```diff
diff -- test/registered/core/test_gpt_oss_1gpu.py
@@ -1,6 +1,10 @@
-from test_gpt_oss_common import BaseTestGptOss
+from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
+from sglang.test.gpt_oss_common import BaseTestGptOss
+register_cuda_ci(est_time=402, suite="stage-b-test-small-1-gpu")
+register_amd_ci(est_time=750, suite="stage-b-test-small-1-gpu-amd")
```

- 已读文件:
  - tests: `test/registered/core/test_gpt_oss_1gpu.py` renamed +5/-1; `python/sglang/test/gpt_oss_common.py` renamed +0/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/gpt_oss_common.py`, `test/registered/core/test_deterministic.py`, `test/registered/core/test_gpt_oss_1gpu.py`, `test/registered/core/test_hidden_states.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14197 - [NPU]Support GPT-OSS for NPU

- 链接: https://github.com/sgl-project/sglang/pull/14197
- 状态/时间: merged / 2026-01-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `733de6be31e2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+96/-17，可读 patch 244 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]Support GPT-OSS for NPU」；模型线: GPT-OSS；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文摘要: Co-author: @mczywu Adapting GPT-OSS model for NPU. 1. Operators capable of handling sinks and sliding windows have been added to the Ascend backend for attention. 2. The swiglu...。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +20/-15 (35 lines); hunks: -71,9 +71,10; -129,6 +130,7 @@ def __init__(; symbols: __init__, forward_prepare，涉及 `__init__, forward_prepare`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +20/-15 (35 lines); hunks: -71,9 +71,10; -129,6 +130,7 @@ def __init__(; symbols: __init__, forward_prepare
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -71,9 +71,10 @@
-from sglang.srt.utils import LazyValue, add_prefix, is_cuda, make_layers
+from sglang.srt.utils import LazyValue, add_prefix, is_cuda, is_npu, make_layers
+_is_npu = is_npu()
@@ -129,6 +130,7 @@ def __init__(
@@ -305,20 +307,20 @@ def forward_prepare(
-        q, k = self.rotary_emb(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +20/-15
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17553 - [NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py

- 链接: https://github.com/sgl-project/sglang/pull/17553
- 状态/时间: merged / 2026-01-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `61abff66c150`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文摘要: NPU support for gpt-oss enabled through the following PR https://github.com/sgl-project/sglang/pull/14197 has a small typo in npu device check for activation. That needs to be f...。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -492,7 +492,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -492,7 +492,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -492,7 +492,7 @@ def __init__(
-        if is_npu:
+        if _is_npu:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18134 - feature: adding gpt-oss 120b nightly test

- 链接: https://github.com/sgl-project/sglang/pull/18134
- 状态/时间: merged / 2026-02-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_gpt_oss_120b.py`；关联提交 `c8da307d7e63`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+88/-4，可读 patch 121 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feature: adding gpt-oss 120b nightly test」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_gpt_oss_120b.py`；PR 正文摘要: Adding gpt-oss nightly tests to h200 and b200 nightly suites, testing key configs as listed on SGL cookbook here: https://cookbook.sglang.io/docs/autoregressive/OpenAI/GPT-OSS b...。
- 实现要点: `test/registered/8-gpu-models/test_gpt_oss_120b.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestGptOss120B, for, test_gpt_oss_120b_all_variants，涉及 `TestGptOss120B, for, test_gpt_oss_120b_all_variants`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_gpt_oss_120b.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestGptOss120B, for, test_gpt_oss_120b_all_variants
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_gpt_oss_120b.py
@@ -0,0 +1,84 @@
+import unittest
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
+# Runs on both H200 and B200 via nightly-8-gpu-common suite
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_gpt_oss_120b.py` added +84/-0
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_gpt_oss_120b.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18405 - [PCG] GPT OSS Triton Kernel Support

- 链接: https://github.com/sgl-project/sglang/pull/18405
- 状态/时间: merged / 2026-02-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `2bd8363486e4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+68/-32，可读 patch 228 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[PCG] GPT OSS Triton Kernel Support」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文摘要: Support Backend for GPT-OSS。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +21/-4 (25 lines); hunks: -25,6 +25,10; -72,6 +76,7; symbols: forward_normal, moe_impl, GptOssAttention, __init__，涉及 `forward_normal, moe_impl, GptOssAttention`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +21/-4 (25 lines); hunks: -25,6 +25,10; -72,6 +76,7; symbols: forward_normal, moe_impl, GptOssAttention, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -25,6 +25,10 @@
+from sglang.srt.compilation.piecewise_context_manager import (
+    get_forward_context,
+    is_in_piecewise_cuda_graph,
+)
@@ -72,6 +76,7 @@
+from sglang.srt.utils.custom_op import register_custom_op
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +21/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/compilation/piecewise_context_manager.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/model_executor/model_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18869 - [CI] Remove `--mem-fraction-static 0.93` from gpt-oss test

- 链接: https://github.com/sgl-project/sglang/pull/18869
- 状态/时间: merged / 2026-02-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`；关联提交 `8290171f5247`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-2，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Remove `--mem-fraction-static 0.93` from gpt-oss test」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`；PR 正文摘要: Remove `--mem-fraction-static 0.93` from gpt-oss test to avoid OOMs.。
- 实现要点: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +0/-2 (2 lines); hunks: -30,8 +30,6 @@ def test_mxfp4_120b(self):; symbols: test_mxfp4_120b，涉及 `test_mxfp4_120b`。
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +0/-2 (2 lines); hunks: -30,8 +30,6 @@ def test_mxfp4_120b(self):; symbols: test_mxfp4_120b
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_gpt_oss_4gpu.py
@@ -30,8 +30,6 @@ def test_mxfp4_120b(self):
-                "--mem-fraction-static",
-                "0.93",
```

- 已读文件:
  - tests: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +0/-2
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18988 - [GPT-OSS] support fp8 online quantization for gpt-oss bf16

- 链接: https://github.com/sgl-project/sglang/pull/18988
- 状态/时间: merged / 2026-02-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+31/-1，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[GPT-OSS] support fp8 online quantization for gpt-oss bf16」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py`；PR 正文摘要: 1. Keep `moe_runner_backend` as `auto` when launch gpt-oss bf16 with online quantization (e.g. fp8) to pick up either `deep_gemm` or `triton` moe backend, since `triton_kernels`...。
- 实现要点: `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0 (26 lines); hunks: -677,6 +677,7 @@ def __init__(self, quant_config: Fp8Config):; -706,8 +707,10 @@ def create_weights(; symbols: __init__, create_weights, apply，涉及 `__init__, create_weights, apply`；`python/sglang/srt/server_args.py` modified +5/-1 (6 lines); hunks: -1386,7 +1386,11 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments，涉及 `_handle_model_specific_adjustments`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0 (26 lines); hunks: -677,6 +677,7 @@ def __init__(self, quant_config: Fp8Config):; -706,8 +707,10 @@ def create_weights(; symbols: __init__, create_weights, apply
  - `python/sglang/srt/server_args.py` modified +5/-1 (6 lines); hunks: -1386,7 +1386,11 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/fp8.py
@@ -677,6 +677,7 @@ def __init__(self, quant_config: Fp8Config):
+        self.with_bias = False
@@ -706,8 +707,10 @@ def create_weights(
+        with_bias: bool = False,
+        self.with_bias = with_bias
@@ -782,6 +785,27 @@ def create_weights(
+        # BIAS (optional, e.g. GPT-OSS)
diff -- python/sglang/srt/server_args.py
@@ -1386,7 +1386,11 @@ def _handle_model_specific_adjustments(self):
-                elif self.ep_size == 1 and is_triton_kernels_available():
+                elif (
+                    self.ep_size == 1
+                    and is_triton_kernels_available()
+                    and self.quantization is None
+                ):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0; `python/sglang/srt/server_args.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20056 - [CI] Add GPT-OSS test for SM120

- 链接: https://github.com/sgl-project/sglang/pull/20056
- 状态/时间: merged / 2026-03-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/core/test_gpt_oss_sm120.py`；关联提交 `8cdb7e1fd453`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+34/-0，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add GPT-OSS test for SM120」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `test/registered/core/test_gpt_oss_sm120.py`；PR 正文摘要: Adds an SM120-specific CI test for GPT-OSS to prevent regressions on consumer Blackwell hardware. This follows up on #20040.。
- 实现要点: `test/registered/core/test_gpt_oss_sm120.py` added +34/-0 (34 lines); hunks: -0,0 +1,34; symbols: TestGptOssSm120, setUpClass, test_mxfp4_20b，涉及 `TestGptOssSm120, setUpClass, test_mxfp4_20b`。
- 代码 diff 细节:
  - `test/registered/core/test_gpt_oss_sm120.py` added +34/-0 (34 lines); hunks: -0,0 +1,34; symbols: TestGptOssSm120, setUpClass, test_mxfp4_20b
- 关键代码摘录:

```diff
diff -- test/registered/core/test_gpt_oss_sm120.py
@@ -0,0 +1,34 @@
+import unittest
+import torch
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.gpt_oss_common import BaseTestGptOss
+register_cuda_ci(est_time=500, suite="stage-b-test-small-1-gpu")
+@unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
```

- 已读文件:
  - tests: `test/registered/core/test_gpt_oss_sm120.py` added +34/-0
- 验证与风险: diff 自带测试面 `test/registered/core/test_gpt_oss_sm120.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20755 - Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+

- 链接: https://github.com/sgl-project/sglang/pull/20755
- 状态/时间: merged / 2026-03-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `bbe25b24126d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+65/-2，可读 patch 91 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`；PR 正文摘要: FlashInfer 0.6.6 (actually 0.6.5) added `tinygemm_bf16`, a faster kernel for small GEMMs. This applies it to the GPT-OSS MoE router on SM90+. Accuracy Test (GPQA) Before: After:...。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +65/-2 (67 lines); hunks: -75,10 +75,34; -97,6 +121,45 @@ def get_attention_sliding_window_size(config):; symbols: GptOssConfig, get_attention_sliding_window_size, TinyGemmLinear, __init__，涉及 `GptOssConfig, get_attention_sliding_window_size, TinyGemmLinear`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +65/-2 (67 lines); hunks: -75,10 +75,34; -97,6 +121,45 @@ def get_attention_sliding_window_size(config):; symbols: GptOssConfig, get_attention_sliding_window_size, TinyGemmLinear, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -75,10 +75,34 @@
-from sglang.srt.utils import LazyValue, add_prefix, is_npu, make_layers
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
+    is_blackwell_supported,
+    is_cuda,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +65/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21570 - [4/n] Support gpt oss 20b lora

- 链接: https://github.com/sgl-project/sglang/pull/21570
- 状态/时间: merged / 2026-04-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`；关联提交 `566b4a4f1ccc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+195/-24，可读 patch 328 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[4/n] Support gpt oss 20b lora」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`；PR 正文摘要: 1. Support gpt oss 20b lora 2. Tune some ci thresholds - more strictly。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +8/-0 (8 lines); hunks: -17,6 +17,7; -651,6 +652,13 @@ def forward(; symbols: forward, GptOssForCausalLM, should_apply_lora, __init__，涉及 `forward, GptOssForCausalLM, should_apply_lora`；`test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: kl_v2, get_prompt_logprobs, TestLoRAGptOss20BLogprobDiff, test_lora_gpt_oss_20b_logprob_accuracy，涉及 `kl_v2, get_prompt_logprobs, TestLoRAGptOss20BLogprobDiff`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +8/-0 (8 lines); hunks: -17,6 +17,7; -651,6 +652,13 @@ def forward(; symbols: forward, GptOssForCausalLM, should_apply_lora, __init__
  - `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: kl_v2, get_prompt_logprobs, TestLoRAGptOss20BLogprobDiff, test_lora_gpt_oss_20b_logprob_accuracy
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -17,6 +17,7 @@
+import re
@@ -651,6 +652,13 @@ def forward(
+    _lora_pattern_moe = re.compile(
+        r"^(?:model\.layers\.\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.experts)|lm_head|model\.embed_tokens)$"
+    )
+    def should_apply_lora(self, module_name: str) -> bool:
diff -- test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py
@@ -0,0 +1,151 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +8/-0
  - tests: `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` added +151/-0
- 验证与风险: diff 自带测试面 `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_8b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22237 - [CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58

- 链接: https://github.com/sgl-project/sglang/pull/22237
- 状态/时间: merged / 2026-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`；关联提交 `2ad5e6df12d3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`；PR 正文摘要: - Lower `expected_score` for both `test_bf16_120b` and `test_mxfp4_120b` from 0.60 to 0.58 - 21% failure rate on both B200 and H100 runners at the 0.60 threshold Score Trend Dat...。
- 实现要点: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2 (4 lines); hunks: -13,7 +13,7 @@ def test_bf16_120b(self):; -23,7 +23,7 @@ def test_mxfp4_120b(self):; symbols: test_bf16_120b, test_mxfp4_120b，涉及 `test_bf16_120b, test_mxfp4_120b`。
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2 (4 lines); hunks: -13,7 +13,7 @@ def test_bf16_120b(self):; -23,7 +23,7 @@ def test_mxfp4_120b(self):; symbols: test_bf16_120b, test_mxfp4_120b
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_gpt_oss_4gpu.py
@@ -13,7 +13,7 @@ def test_bf16_120b(self):
-                "low": 0.60,
+                "low": 0.58,
@@ -23,7 +23,7 @@ def test_mxfp4_120b(self):
-                "low": 0.60,
+                "low": 0.58,
```

- 已读文件:
  - tests: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
