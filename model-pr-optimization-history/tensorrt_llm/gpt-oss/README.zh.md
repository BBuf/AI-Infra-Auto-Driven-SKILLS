# TensorRT-LLM GPT-OSS 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/source/blogs/tech_blog/blog09_Deploying_GPT_OSS_on_TRTLLM.md` | [#17228](https://github.com/NVIDIA/TensorRT-LLM/pull/17228) |
| `docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md` | [#7140](https://github.com/NVIDIA/TensorRT-LLM/pull/7140) |
| `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md` | [#10283](https://github.com/NVIDIA/TensorRT-LLM/pull/10283), [#11074](https://github.com/NVIDIA/TensorRT-LLM/pull/11074) |
| `examples/configs/curated/gpt-oss-120b-latency.yaml` | [#10283](https://github.com/NVIDIA/TensorRT-LLM/pull/10283) |
| `examples/configs/curated/gpt-oss-120b-throughput.yaml` | [#10283](https://github.com/NVIDIA/TensorRT-LLM/pull/10283) |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp1_conc256.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp2_conc256.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc128.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc1280.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc1536.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc256.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc8.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc1.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc16.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc1792.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc2.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc2048.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc32.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc384.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc4.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc512.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc64.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc640.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc768.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc896.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc256.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc4.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc10.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc128.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc384.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc64.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc640.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc896.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc1.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc1024.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc1280.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc16.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc1792.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc2.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc2048.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc32.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc768.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc8.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc128.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc16.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc256.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc32.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc4.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc64.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc8.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc1280.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc768.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc1.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc10.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc1536.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc1792.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc2.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc256.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc32.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc4.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc64.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc128.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc16.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc2048.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc384.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc640.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc8.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp1_conc64.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp1_conc8.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc32.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc4.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc1024.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc128.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc16.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc32.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc384.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc4.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc8.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc16.yaml` | 无直接 PR 号提交 |
| `examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc2048.yaml` | 无直接 PR 号提交 |
| ... | 67 more files omitted from table; all were used for git tracing. |

## PR 覆盖总览

- git 追溯 PR 数: 22
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 22
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-08-07 | [#6645](https://github.com/NVIDIA/TensorRT-LLM/pull/6645) | merged | [None] [feat] Add model gpt-oss | `tensorrt_llm/_torch/models/modeling_gpt_oss.py`, `examples/models/core/gpt_oss/openai_chat_client_function_calling.py`, `examples/models/core/gpt_oss/README.md` |
| 2025-08-15 | [#6908](https://github.com/NVIDIA/TensorRT-LLM/pull/6908) | merged | [None][doc] Update gpt-oss doc on MoE support matrix | `examples/models/core/gpt_oss/README.md` |
| 2025-08-28 | [#7261](https://github.com/NVIDIA/TensorRT-LLM/pull/7261) | merged | [TRTLLM-7207][feat] Chat completions API for gpt-oss | `examples/models/core/gpt_oss/openai_chat_client_function_calling.py`, `examples/models/core/gpt_oss/README.md`, `tensorrt_llm/serve/harmony_adapter.py` |
| 2025-09-03 | [#7140](https://github.com/NVIDIA/TensorRT-LLM/pull/7140) | merged | [None][doc] add GPT OSS Eagle3 blog | `docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md` |
| 2025-09-14 | [#7612](https://github.com/NVIDIA/TensorRT-LLM/pull/7612) | merged | [None][feat] support gpt-oss with fp8 kv cache | `examples/models/core/gpt_oss/README.md`, `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/kernelMetaInfo.h`, `cpp/tensorrt_llm/common/attentionOp.cpp` |
| 2025-09-25 | [#7911](https://github.com/NVIDIA/TensorRT-LLM/pull/7911) | merged | [https://nvbugs/5525951][fix] Clarify that PP is not supported for GPTOSS | `tensorrt_llm/_torch/models/modeling_gpt_oss.py` |
| 2025-10-02 | [#7916](https://github.com/NVIDIA/TensorRT-LLM/pull/7916) | merged | [TRTLLM-7775][feat] Integrate tinygemm2 for gpt-oss | `tensorrt_llm/_torch/models/modeling_gpt_oss.py` |
| 2025-10-06 | [#7937](https://github.com/NVIDIA/TensorRT-LLM/pull/7937) | merged | [None][feat] GPT-OSS Sm120/Sm121 Support | `tensorrt_llm/_torch/models/modeling_gpt_oss.py` |
| 2025-12-03 | [#8253](https://github.com/NVIDIA/TensorRT-LLM/pull/8253) | merged | [https://nvbugs/5552132][fix] Enable LoRa for GPT OSS Torch | `tensorrt_llm/_torch/models/modeling_gpt_oss.py` |
| 2025-12-25 | [#10283](https://github.com/NVIDIA/TensorRT-LLM/pull/10283) | merged | [None] [doc] Update IFB performance guide & GPTOSS deployment guide | `examples/configs/curated/gpt-oss-120b-latency.yaml`, `examples/configs/curated/gpt-oss-120b-throughput.yaml`, `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md` |
| 2026-01-04 | [#8956](https://github.com/NVIDIA/TensorRT-LLM/pull/8956) | merged | [TRTLLM-7138][feat] Support nvfp4 for gptoss | `tensorrt_llm/_torch/models/modeling_gpt_oss.py` |
| 2026-01-30 | [#11074](https://github.com/NVIDIA/TensorRT-LLM/pull/11074) | merged | [TRTLLM-10733][feat] Make TRTLLM MOE the default one for GPTOSS on Blackwell | `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md`, `tensorrt_llm/_torch/model_config.py`, `tensorrt_llm/llmapi/llm_args.py` |
| 2026-02-26 | [#11668](https://github.com/NVIDIA/TensorRT-LLM/pull/11668) | merged | [https://nvbugs/5914691][fix] WAR F.linear perf regression for GPTOSS | `tensorrt_llm/_torch/models/modeling_gpt_oss.py` |
| 2026-04-28 | [#12796](https://github.com/NVIDIA/TensorRT-LLM/pull/12796) | merged | [None][test] add unit test and e2e test for gpt_oss_20b MHA kernel | `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` |
| 2026-05-07 | [#13743](https://github.com/NVIDIA/TensorRT-LLM/pull/13743) | merged | [https://nvbugs/6115290][fix] Fix GPT OSS 120B GB200 Test Regression | `tests/scripts/perf-sanity/aggregated/gpt_oss_120b_fp4_grace_blackwell.yaml` |
| 2026-05-14 | [#13166](https://github.com/NVIDIA/TensorRT-LLM/pull/13166) | merged | [None][fix] Raise clear error when GPT-OSS is used with non-TRTLLM attention backend | `tensorrt_llm/_torch/models/modeling_gpt_oss.py` |
| 2026-07-07 | [#15765](https://github.com/NVIDIA/TensorRT-LLM/pull/15765) | merged | [None][perf] Validate GPT-OSS transceiver v2 performance | `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_triton.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_trtllm.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_tllm.yaml` |
| 2026-07-17 | [#16378](https://github.com/NVIDIA/TensorRT-LLM/pull/16378) | merged | [None][fix] Fix unfused RoPE for yarn models: double rotation and GPT-OSS pairing | `tensorrt_llm/_torch/models/modeling_gpt_oss.py` |
| 2026-07-20 | [#16479](https://github.com/NVIDIA/TensorRT-LLM/pull/16479) | merged | [None][feat] Default GPT-OSS to the Python KV-cache transceiver | `tensorrt_llm/_torch/models/modeling_gpt_oss.py`, `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` |
| 2026-08-04 | [#17228](https://github.com/NVIDIA/TensorRT-LLM/pull/17228) | merged | [https://nvbugs/6533913][docs] Update pinned container for GPT OSS | `docs/source/blogs/tech_blog/blog09_Deploying_GPT_OSS_on_TRTLLM.md` |
| 2026-08-10 | [#16942](https://github.com/NVIDIA/TensorRT-LLM/pull/16942) | merged | [None][feat] Opt GPT-OSS in to KV cache manager V2 by default | `tensorrt_llm/_torch/models/modeling_gpt_oss.py`, `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` |
| 2026-08-13 | [#17597](https://github.com/NVIDIA/TensorRT-LLM/pull/17597) | merged | [None][fix] Remove obsolete GPT-OSS two-model Eagle3 tests | `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` |

## 逐 PR diff 审计卡

### PR #6645 - [None] [feat] Add model gpt-oss

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/6645
- 状态/时间: merged / 2025-08-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/models/core/gpt_oss/README.md`, `examples/models/core/gpt_oss/openai_chat_client_function_calling.py`, `tensorrt_llm/_torch/models/modeling_gpt_oss.py`, `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py`；关联提交 `8207d5fd3996`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2102 个文件，+34656/-8844，可读 patch 34289 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` added +912/-0 (912 lines); hunks: -0,0 +1,912; symbols: AttentionBlock, __init__, forward, load_weights，涉及 `AttentionBlock, __init__, forward`；`examples/models/core/gpt_oss/openai_chat_client_function_calling.py` added +191/-0 (191 lines); hunks: -0,0 +1,191; symbols: get_current_weather, get_multiple_weathers, main，涉及 `get_current_weather, get_multiple_weathers, main`；`examples/models/core/gpt_oss/README.md` added +145/-0 (145 lines); hunks: -0,0 +1,145；`tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` added +89/-0 (89 lines); hunks: -0,0 +1,89; symbols: dump_config_json, test_gpt_oss_trtllmgen，涉及 `dump_config_json, test_gpt_oss_trtllmgen`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gpt_oss.py` added +912/-0 (912 lines); hunks: -0,0 +1,912; symbols: AttentionBlock, __init__, forward, load_weights
  - `examples/models/core/gpt_oss/openai_chat_client_function_calling.py` added +191/-0 (191 lines); hunks: -0,0 +1,191; symbols: get_current_weather, get_multiple_weathers, main
  - `examples/models/core/gpt_oss/README.md` added +145/-0 (145 lines); hunks: -0,0 +1,145
  - `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` added +89/-0 (89 lines); hunks: -0,0 +1,89; symbols: dump_config_json, test_gpt_oss_trtllmgen
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gpt_oss.py
@@ -0,0 +1,912 @@
+import os
+from typing import Dict, Optional, Tuple
+import torch
+from torch import nn
+from torch.nn.parameter import Parameter
+from tqdm import tqdm
diff -- examples/models/core/gpt_oss/openai_chat_client_function_calling.py
@@ -0,0 +1,191 @@
+import argparse
+import json
+import re
+from openai import OpenAI
+system_prompt = """You are ChatGPT, a large language model trained by OpenAI.
+Knowledge cutoff: 2024-06
diff -- examples/models/core/gpt_oss/README.md
@@ -0,0 +1,145 @@
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` added +912/-0
  - docs: `examples/models/core/gpt_oss/openai_chat_client_function_calling.py` added +191/-0; `examples/models/core/gpt_oss/README.md` added +145/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` added +89/-0
- 验证与风险: diff 自带测试面 `cpp/kernels/xqa/test/refAttention.cpp`, `cpp/kernels/xqa/test/refAttention.h`, `cpp/kernels/xqa/test/test.cpp`, `cpp/tests/unit_tests/kernels/allReduce/allReduceFusionTest.cu`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6908 - [None][doc] Update gpt-oss doc on MoE support matrix

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/6908
- 状态/时间: merged / 2025-08-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/models/core/gpt_oss/README.md`；关联提交 `5346eb7bc5fa`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+10/-6，可读 patch 23 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `examples/models/core/gpt_oss/README.md` modified +10/-6 (16 lines); hunks: -6,12 +6,16 @@ GPT-OSS is a reasoning model with MoE weights quantized with m...。
- 代码 diff 细节:
  - `examples/models/core/gpt_oss/README.md` modified +10/-6 (16 lines); hunks: -6,12 +6,16 @@ GPT-OSS is a reasoning model with MoE weights quantized with m...
- 关键代码摘录:

```diff
diff -- examples/models/core/gpt_oss/README.md
@@ -6,12 +6,16 @@ GPT-OSS is a reasoning model with MoE weights quantized with mxfp4. All the othe
-In MoE, the weights are pre-quantized to mxfp4. The activation can be in either bf16 (Hopper) or mxfp8 (Blackwell), with similar accuracy.
-| device | Activation | Weight | Supported moe_backend |
-|----------|----------|----------|----------|
-| Hopper | bf16 | mxfp4 | **TRITON**, CUTLASS |
-| Blackwell | mxfp8 | mxfp4 | CUTLASS, TRTLLM |
+In MoE, the weights are pre-quantized to mxfp4. The activation can be in either bf16 (Hopper) or mxfp8 (Blackwell), with similar accuracy. FP8 activation with per-tensor scaling f
```

- 提取文件（未人工审阅）:
  - docs: `examples/models/core/gpt_oss/README.md` modified +10/-6
- 验证与风险: 该 PR 主要落在文档/示例 `examples/models/core/gpt_oss/README.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #7261 - [TRTLLM-7207][feat] Chat completions API for gpt-oss

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7261
- 状态/时间: merged / 2025-08-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/models/core/gpt_oss/README.md`, `examples/models/core/gpt_oss/openai_chat_client_function_calling.py`；关联提交 `c1e7fb9042b2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+2050/-168，可读 patch 2416 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `examples/models/core/gpt_oss/openai_chat_client_function_calling.py` modified +69/-125 (194 lines); hunks: -1,82 +1,58; -103,14 +79,6 @@ def main():; symbols: main，涉及 `main`；`examples/models/core/gpt_oss/README.md` modified +12/-23 (35 lines); hunks: -35,13 +35,7 @@ OpenAI MoE models support function calling. Here is an exampl...; -68,14 +62,9 @@ The output would look similar to:；`tensorrt_llm/serve/harmony_adapter.py` added +1598/-0 (1598 lines); hunks: -0,0 +1,1598; symbols: HarmonyStreamState, __init__, process_token_batch, _create_closing_token_delta，涉及 `HarmonyStreamState, __init__, process_token_batch`；`tensorrt_llm/serve/openai_server.py` modified +80/-2 (82 lines); hunks: -49,6 +49,9; -98,6 +101,10 @@ def __init__(self,; symbols: __init__, lifespan, create_error_response, register_routes，涉及 `__init__, lifespan, create_error_response`。
- 代码 diff 细节:
  - `examples/models/core/gpt_oss/openai_chat_client_function_calling.py` modified +69/-125 (194 lines); hunks: -1,82 +1,58; -103,14 +79,6 @@ def main():; symbols: main
  - `examples/models/core/gpt_oss/README.md` modified +12/-23 (35 lines); hunks: -35,13 +35,7 @@ OpenAI MoE models support function calling. Here is an exampl...; -68,14 +62,9 @@ The output would look similar to:
  - `tensorrt_llm/serve/harmony_adapter.py` added +1598/-0 (1598 lines); hunks: -0,0 +1,1598; symbols: HarmonyStreamState, __init__, process_token_batch, _create_closing_token_delta
  - `tensorrt_llm/serve/openai_server.py` modified +80/-2 (82 lines); hunks: -49,6 +49,9; -98,6 +101,10 @@ def __init__(self,; symbols: __init__, lifespan, create_error_response, register_routes
  - `tensorrt_llm/serve/openai_protocol.py` modified +39/-8 (47 lines); hunks: -6,10 +6,12; -327,17 +329,30 @@ class FunctionCall(OpenAIBaseModel):; symbols: FunctionCall, DeltaFunctionCall, ToolCall, DeltaToolCall
- 关键代码摘录:

```diff
diff -- examples/models/core/gpt_oss/openai_chat_client_function_calling.py
@@ -1,82 +1,58 @@
-import re
-system_prompt = """You are ChatGPT, a large language model trained by OpenAI.
-Knowledge cutoff: 2024-06
-Current date: 2025-06-28
-Reasoning: high
-# Valid channels: analysis, commentary, final. Channel must be included for every message.
diff -- examples/models/core/gpt_oss/README.md
@@ -35,13 +35,7 @@ OpenAI MoE models support function calling. Here is an example based on [XGramma
-cat > ./extra_llm_api_options.yaml <<EOF
-guided_decoding_backend: xgrammar
-EOF
-trtllm-serve <model> \
-    --backend pytorch \
-    --extra_llm_api_options extra_llm_api_options.yaml
diff -- tensorrt_llm/serve/harmony_adapter.py
@@ -0,0 +1,1598 @@
```

- 提取文件（未人工审阅）:
  - docs: `examples/models/core/gpt_oss/openai_chat_client_function_calling.py` modified +69/-125; `examples/models/core/gpt_oss/README.md` modified +12/-23
  - runtime: `tensorrt_llm/serve/harmony_adapter.py` added +1598/-0; `tensorrt_llm/serve/openai_server.py` modified +80/-2; `tensorrt_llm/serve/openai_protocol.py` modified +39/-8; `tensorrt_llm/executor/worker.py` modified +26/-10
- 验证与风险: diff 自带测试面 `tests/integration/defs/test_e2e.py`, `tests/integration/test_lists/test-db/l0_h100.yml`, `tests/unittest/llmapi/apps/_test_openai_chat_harmony.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7140 - [None][doc] add GPT OSS Eagle3 blog

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7140
- 状态/时间: merged / 2025-09-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md`；关联提交 `f156221c2756`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+140/-0，可读 patch 141 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md` added +140/-0 (140 lines); hunks: -0,0 +1,140。
- 代码 diff 细节:
  - `docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md` added +140/-0 (140 lines); hunks: -0,0 +1,140
- 关键代码摘录:

```diff
diff -- docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md
@@ -0,0 +1,140 @@
+## Running GPT-OSS-120B with Eagle3 Speculative Decoding on GB200/B200 (TensorRT-LLM)
+This guide sets up a production endpoint that uses Eagle3 speculative decoding on NVIDIA GB200 or B200 GPUs only. It replaces the low‑latency flow from the previous guide and inte
+### Prerequisites
+- NVIDIA GB200 or B200 GPUs (example below assumes 8 GPUs; adjust flags for your setup)
+- Fast SSD storage for model weights
+- Base model weights available under a directory named `gpt-oss-120b` (example path)
```

- 提取文件（未人工审阅）:
  - docs: `docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md` added +140/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #7612 - [None][feat] support gpt-oss with fp8 kv cache

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7612
- 状态/时间: merged / 2025-09-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/models/core/gpt_oss/README.md`；关联提交 `1b29c2e7314f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1460 个文件，+4469/-4391，可读 patch 9280 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `examples/models/core/gpt_oss/README.md` modified +9/-0 (9 lines); hunks: -26,6 +26,15 @@ In MoE, the weights are pre-quantized to mxfp4. The activatio...；`cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/kernelMetaInfo.h` modified +1445/-1445 (2890 lines)；`cpp/tensorrt_llm/common/attentionOp.cpp` modified +14/-14 (28 lines); hunks: -281,9 +281,8 @@ bool AttentionOp::convertMMHAParamsToXQAParams(tensorrt_llm:...; -379,7 +378,7 @@ int AttentionOp::ulyssesContextPostprocess(T* input, T* outp...；`cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaReduction.cu` modified +12/-7 (19 lines); hunks: -202,24 +202,29 @@ __global__ void __launch_bounds__(NumThreadsPerCta, 2) fmh...。
- 代码 diff 细节:
  - `examples/models/core/gpt_oss/README.md` modified +9/-0 (9 lines); hunks: -26,6 +26,15 @@ In MoE, the weights are pre-quantized to mxfp4. The activatio...
  - `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/kernelMetaInfo.h` modified +1445/-1445 (2890 lines)
  - `cpp/tensorrt_llm/common/attentionOp.cpp` modified +14/-14 (28 lines); hunks: -281,9 +281,8 @@ bool AttentionOp::convertMMHAParamsToXQAParams(tensorrt_llm:...; -379,7 +378,7 @@ int AttentionOp::ulyssesContextPostprocess(T* input, T* outp...
  - `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaReduction.cu` modified +12/-7 (19 lines); hunks: -202,24 +202,29 @@ __global__ void __launch_bounds__(NumThreadsPerCta, 2) fmh...
  - `cpp/tensorrt_llm/common/attentionOp.h` modified +8/-7 (15 lines); hunks: -411,6 +411,7 @@ class AttentionOp; -466,13 +467,13 @@ class AttentionOp; symbols: AttentionOp
- 关键代码摘录:

```diff
diff -- examples/models/core/gpt_oss/README.md
@@ -26,6 +26,15 @@ In MoE, the weights are pre-quantized to mxfp4. The activation can be in either
+## KV Cache Support Matrix
+|   device  | bf16 kv cache dtype | fp8 kv cache dtype |
+|:---------:|:-------------------:|--------------------|
+|   Hopper  | yes                 | no                 |
+| Blackwell | yes                 | yes                |
+On Blackwell GPUs, support for FP8 KV cache is available, allowing the key-value cache to be stored in FP8 format. This reduces memory usage and bandwidth requirements, which can
diff -- cpp/tensorrt_llm/common/attentionOp.cpp
@@ -281,9 +281,8 @@ bool AttentionOp::convertMMHAParamsToXQAParams(tensorrt_llm::kernels::XQAParams&
-    xqaParams.is_fp8_output = mFP8ContextFMHA;
-    xqaParams.fp8_out_scale
-        = ((mFP8ContextFMHA || mFP8ContextMLA) ? generationsParams.attention_output_orig_quant : nullptr);
+    xqaParams.is_fp8_output = mFP8AttenOutput;
+    xqaParams.fp8_out_scale = ((mFP8AttenOutput) ? generationsParams.attention_output_orig_quant : nullptr);
@@ -379,7 +378,7 @@ int AttentionOp::ulyssesContextPostprocess(T* input, T* output, T* buffer, Enque
diff -- cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaReduction.cu
@@ -202,24 +202,29 @@ __global__ void __launch_bounds__(NumThreadsPerCta, 2) fmhaReductionKernel(
```

- 提取文件（未人工审阅）:
  - docs: `examples/models/core/gpt_oss/README.md` modified +9/-0
  - runtime: `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/kernelMetaInfo.h` modified +1445/-1445; `cpp/tensorrt_llm/common/attentionOp.cpp` modified +14/-14; `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaReduction.cu` modified +12/-7; `cpp/tensorrt_llm/common/attentionOp.h` modified +8/-7; `cpp/tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.cpp` modified +9/-5; `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext_cubin.cpp` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/qa/llm_function_core_sanity.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7911 - [https://nvbugs/5525951][fix] Clarify that PP is not supported for GPTOSS

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7911
- 状态/时间: merged / 2025-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gpt_oss.py`；关联提交 `1eb653146a6a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 9 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +2/-0 (2 lines); hunks: -562,6 +562,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +2/-0 (2 lines); hunks: -562,6 +562,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gpt_oss.py
@@ -562,6 +562,8 @@ def __init__(
+        assert model_config.mapping.pp_size == 1, "Pipeline parallelism is not supported."
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7916 - [TRTLLM-7775][feat] Integrate tinygemm2 for gpt-oss

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7916
- 状态/时间: merged / 2025-10-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gpt_oss.py`；关联提交 `6568e565dbae`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+690/-3，可读 patch 737 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +14/-2 (16 lines); hunks: -36,6 +36,9; -219,6 +222,15 @@ def _create_ideal_expert_load_balanced_logits(; symbols: AttentionBlock, _create_ideal_expert_load_balanced_logits, compute_gate_output, forward_normal，涉及 `AttentionBlock, _create_ideal_expert_load_balanced_logits, compute_gate_output`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +14/-2 (16 lines); hunks: -36,6 +36,9; -219,6 +222,15 @@ def _create_ideal_expert_load_balanced_logits(; symbols: AttentionBlock, _create_ideal_expert_load_balanced_logits, compute_gate_output, forward_normal
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gpt_oss.py
@@ -36,6 +36,9 @@
+# Use TinyGEMM when the number of tokens is not larger than this threshold
+MIN_LATENCY_TINYGEMM_NUM_TOKENS = 128
@@ -219,6 +222,15 @@ def _create_ideal_expert_load_balanced_logits(
+    def compute_gate_output(self, x: torch.Tensor) -> torch.Tensor:
+        if x.shape[0] <= MIN_LATENCY_TINYGEMM_NUM_TOKENS:
+            weight = self.gate.weight
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +14/-2
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/thop/parallel/test_tinygemm2.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7937 - [None][feat] GPT-OSS Sm120/Sm121 Support

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7937
- 状态/时间: merged / 2025-10-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gpt_oss.py`；关联提交 `27a5091fcb94`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 69 个文件，+352/-264，可读 patch 1199 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +17/-4 (21 lines); hunks: -43,6 +43,7 @@ def __init__(; -75,6 +76,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +17/-4 (21 lines); hunks: -43,6 +43,7 @@ def __init__(; -75,6 +76,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gpt_oss.py
@@ -43,6 +43,7 @@ def __init__(
+        use_custom_cublas_mm: bool = False,
@@ -75,6 +76,7 @@ def __init__(
+            use_custom_cublas_mm=use_custom_cublas_mm,
@@ -127,6 +129,7 @@ def __init__(
+        use_custom_cublas_mm: bool = False,
@@ -146,8 +149,7 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +17/-4
- 验证与风险: diff 自带测试面 `cpp/tests/unit_tests/kernels/mixtureOfExpertsTest.cu`, `tests/integration/defs/test_e2e.py`, `tests/integration/test_lists/test-db/l0_rtx_pro_6000.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8253 - [https://nvbugs/5552132][fix] Enable LoRa for GPT OSS Torch

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/8253
- 状态/时间: merged / 2025-12-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gpt_oss.py`；关联提交 `4e5b10da487f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+115/-14，可读 patch 281 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +36/-14 (50 lines); hunks: -230,21 +230,27 @@ def _create_ideal_expert_load_balanced_logits(; -253,7 +259,7 @@ def forward_normal(; symbols: _create_ideal_expert_load_balanced_logits, compute_gate_output, forward_normal，涉及 `_create_ideal_expert_load_balanced_logits, compute_gate_output, forward_normal`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +36/-14 (50 lines); hunks: -230,21 +230,27 @@ def _create_ideal_expert_load_balanced_logits(; -253,7 +259,7 @@ def forward_normal(; symbols: _create_ideal_expert_load_balanced_logits, compute_gate_output, forward_normal
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gpt_oss.py
@@ -230,21 +230,27 @@ def _create_ideal_expert_load_balanced_logits(
-    def compute_gate_output(self, x: torch.Tensor) -> torch.Tensor:
-        if get_sm_version() in [
-                90, 100, 103
-        ] and x.shape[0] <= MIN_LATENCY_TINYGEMM_NUM_TOKENS:
+    def compute_gate_output(self,
+                            x: torch.Tensor,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +36/-14
- 验证与风险: diff 自带测试面 `tests/integration/defs/conftest.py`, `tests/integration/defs/examples/test_gpt.py`, `tests/integration/test_lists/test-db/l0_h100.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10283 - [None] [doc] Update IFB performance guide & GPTOSS deployment guide

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/10283
- 状态/时间: merged / 2025-12-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md`, `examples/configs/curated/gpt-oss-120b-latency.yaml`, `examples/configs/curated/gpt-oss-120b-throughput.yaml`；关联提交 `97b38ac403b0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+19/-19，可读 patch 113 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `examples/configs/curated/gpt-oss-120b-latency.yaml` modified +3/-3 (6 lines); hunks: -1,13 +1,13；`examples/configs/curated/gpt-oss-120b-throughput.yaml` modified +3/-3 (6 lines); hunks: -1,8 +1,8；`docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md` modified +6/-5 (11 lines); hunks: -26,9 +26,10 @@ There are multiple MOE backends inside TensorRT LLM. Here are...; -139,11 +140,11 @@ These options provide control over TensorRT LLM's behavior...。
- 代码 diff 细节:
  - `examples/configs/curated/gpt-oss-120b-latency.yaml` modified +3/-3 (6 lines); hunks: -1,13 +1,13
  - `examples/configs/curated/gpt-oss-120b-throughput.yaml` modified +3/-3 (6 lines); hunks: -1,8 +1,8
  - `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md` modified +6/-5 (11 lines); hunks: -26,9 +26,10 @@ There are multiple MOE backends inside TensorRT LLM. Here are...; -139,11 +140,11 @@ These options provide control over TensorRT LLM's behavior...
- 关键代码摘录:

```diff
diff -- examples/configs/curated/gpt-oss-120b-latency.yaml
@@ -1,13 +1,13 @@
-max_batch_size: 720
+max_batch_size: 64
-moe_expert_parallel_size: 8
+moe_expert_parallel_size: 1
-  max_batch_size: 720
+  max_batch_size: 64
diff -- examples/configs/curated/gpt-oss-120b-throughput.yaml
@@ -1,8 +1,8 @@
-max_batch_size: 720
+max_batch_size: 720 # Depends on max_sequence_length
-tensor_parallel_size: 8
-moe_expert_parallel_size: 8
+tensor_parallel_size: 2
+moe_expert_parallel_size: 2
diff -- docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md
@@ -26,9 +26,10 @@ There are multiple MOE backends inside TensorRT LLM. Here are the support matrix
```

- 提取文件（未人工审阅）:
  - docs: `examples/configs/curated/gpt-oss-120b-latency.yaml` modified +3/-3; `examples/configs/curated/gpt-oss-120b-throughput.yaml` modified +3/-3; `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md` modified +6/-5
- 验证与风险: 该 PR 主要落在文档/示例 `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md`, `docs/source/features/paged-attention-ifb-scheduler.md`, `examples/configs/curated/gpt-oss-120b-latency.yaml`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #8956 - [TRTLLM-7138][feat] Support nvfp4 for gptoss

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/8956
- 状态/时间: merged / 2026-01-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gpt_oss.py`；关联提交 `afc533193d05`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+961/-255，可读 patch 1680 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +92/-156 (248 lines); hunks: -34,8 +34,7; -639,6 +638,15 @@ def __post_init__(self):; symbols: __post_init__, load_weights, load_hf_weights, load_ori_weights，涉及 `__post_init__, load_weights, load_hf_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +92/-156 (248 lines); hunks: -34,8 +34,7; -639,6 +638,15 @@ def __post_init__(self):; symbols: __post_init__, load_weights, load_hf_weights, load_ori_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gpt_oss.py
@@ -34,8 +34,7 @@
-from .modeling_utils import (DecoderModel, duplicate_kv_weight, filter_weights,
-                             register_auto_model)
+from .modeling_utils import DecoderModel, filter_weights, register_auto_model
@@ -639,6 +638,15 @@ def __post_init__(self):
+            if quant_config.quant_algo == "NVFP4":
+                quant_config.exclude_modules = [
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +92/-156
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/unittest/_torch/modules/test_fused_moe.py`, `tests/unittest/_torch/thop/serial/test_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11074 - [TRTLLM-10733][feat] Make TRTLLM MOE the default one for GPTOSS on Blackwell

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11074
- 状态/时间: merged / 2026-01-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md`；关联提交 `4f0c1b2489bf`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+51/-14，可读 patch 100 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md` modified +1/-2 (3 lines); hunks: -28,8 +28,7 @@ There are multiple MOE backends inside TensorRT LLM. Here are...；`tensorrt_llm/_torch/model_config.py` modified +32/-1 (33 lines); hunks: -229,6 +229,32 @@ def is_generation_model(model_architectures: Optional[List[...; -566,7 +592,12 @@ def _recursive_update_config(config: transformers.Pretraine...; symbols: is_generation_model, resolve_moe_backend, load_modelopt_quant_config, _recursive_update_config，涉及 `is_generation_model, resolve_moe_backend, load_modelopt_quant_config`；`tensorrt_llm/llmapi/llm_args.py` modified +7/-4 (11 lines); hunks: -444,10 +444,13 @@ class MoeConfig(StrictBaseModel):; symbols: MoeConfig，涉及 `MoeConfig`。
- 代码 diff 细节:
  - `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md` modified +1/-2 (3 lines); hunks: -28,8 +28,7 @@ There are multiple MOE backends inside TensorRT LLM. Here are...
  - `tensorrt_llm/_torch/model_config.py` modified +32/-1 (33 lines); hunks: -229,6 +229,32 @@ def is_generation_model(model_architectures: Optional[List[...; -566,7 +592,12 @@ def _recursive_update_config(config: transformers.Pretraine...; symbols: is_generation_model, resolve_moe_backend, load_modelopt_quant_config, _recursive_update_config
  - `tensorrt_llm/llmapi/llm_args.py` modified +7/-4 (11 lines); hunks: -444,10 +444,13 @@ class MoeConfig(StrictBaseModel):; symbols: MoeConfig
- 关键代码摘录:

```diff
diff -- docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md
@@ -28,8 +28,7 @@ There are multiple MOE backends inside TensorRT LLM. Here are the support matrix
-The default moe backend is `CUTLASS`, so for the best possible perf, one must set the `moe_config.backend` explicitly to run the model.
-For Blackwell, `CUTLASS` was better for max throughput at first but now we have optimized `TRTLLM` moe to be universally faster. For Hopper, Triton is the faster backend.
+For Blackwell, the default MoE backend is `TRTLLM`. For Hopper, the default MoE backend is `TRITON`. They are recommended for the best perf. Users don't need to explicitly set `mo
diff -- tensorrt_llm/_torch/model_config.py
@@ -229,6 +229,32 @@ def is_generation_model(model_architectures: Optional[List[str]],
+    @staticmethod
+    def resolve_moe_backend(moe_backend: str, architecture: str) -> str:
+        """Resolve AUTO moe_backend to a specific backend based on model architecture.
+        Args:
+            moe_backend: The configured moe_backend (may be "AUTO")
+            architecture: The model architecture name (e.g., "GptOssForCausalLM")
diff -- tensorrt_llm/llmapi/llm_args.py
@@ -444,10 +444,13 @@ class MoeConfig(StrictBaseModel):
-    backend: Literal["CUTLASS", "CUTEDSL", "WIDEEP", "TRTLLM", "DEEPGEMM",
-                     "VANILLA",
-                     "TRITON"] = Field(default='CUTLASS',
```

- 提取文件（未人工审阅）:
  - docs: `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md` modified +1/-2
  - runtime: `tensorrt_llm/_torch/model_config.py` modified +32/-1; `tensorrt_llm/llmapi/llm_args.py` modified +7/-4
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/model_config.py`, `tensorrt_llm/llmapi/llm_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11668 - [https://nvbugs/5914691][fix] WAR F.linear perf regression for GPTOSS

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11668
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gpt_oss.py`；关联提交 `b103625ebff8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+2/-1，可读 patch 10 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +2/-1 (3 lines); hunks: -513,7 +513,8 @@ def __init__(self, model_config: ModelConfig[GptOssConfig]):; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +2/-1 (3 lines); hunks: -513,7 +513,8 @@ def __init__(self, model_config: ModelConfig[GptOssConfig]):; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gpt_oss.py
@@ -513,7 +513,8 @@ def __init__(self, model_config: ModelConfig[GptOssConfig]):
-        self.use_custom_cublas_mm = sm_version == 121
+        # Use custom cublas to bypass F.linear's additional memory copy for biases on SM 100+
+        self.use_custom_cublas_mm = sm_version >= 100
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12796 - [None][test] add unit test and e2e test for gpt_oss_20b MHA kernel

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12796
- 状态/时间: merged / 2026-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py`；关联提交 `a76136803605`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+171/-1，可读 patch 195 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` modified +148/-1 (149 lines); hunks: -3,12 +3,23; -89,3 +100,139 @@ def test_gpt_oss_trtllmgen(moe_backend):; symbols: test_gpt_oss_trtllmgen, test_gpt_oss_xqa_kernel_selection，涉及 `test_gpt_oss_trtllmgen, test_gpt_oss_xqa_kernel_selection`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` modified +148/-1 (149 lines); hunks: -3,12 +3,23; -89,3 +100,139 @@ def test_gpt_oss_trtllmgen(moe_backend):; symbols: test_gpt_oss_trtllmgen, test_gpt_oss_xqa_kernel_selection
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_gpt_oss.py
@@ -3,12 +3,23 @@
-from transformers import AutoTokenizer
+import torch
+from torch.profiler import ProfilerActivity
+from transformers import AutoTokenizer, GptOssConfig
+import tensorrt_llm
+from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` modified +148/-1
- 验证与风险: diff 自带测试面 `tests/integration/defs/perf/pytorch_model_config.py`, `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13743 - [https://nvbugs/6115290][fix] Fix GPT OSS 120B GB200 Test Regression

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13743
- 状态/时间: merged / 2026-05-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/scripts/perf-sanity/aggregated/gpt_oss_120b_fp4_grace_blackwell.yaml`；关联提交 `84c69deaa070`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/scripts/perf-sanity/aggregated/gpt_oss_120b_fp4_grace_blackwell.yaml` modified +1/-0 (1 lines); hunks: -21,6 +21,7 @@ server_configs:。
- 代码 diff 细节:
  - `tests/scripts/perf-sanity/aggregated/gpt_oss_120b_fp4_grace_blackwell.yaml` modified +1/-0 (1 lines); hunks: -21,6 +21,7 @@ server_configs:
- 关键代码摘录:

```diff
diff -- tests/scripts/perf-sanity/aggregated/gpt_oss_120b_fp4_grace_blackwell.yaml
@@ -21,6 +21,7 @@ server_configs:
+      batch_sizes: [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 256, 512, 640]
```

- 提取文件（未人工审阅）:
  - tests: `tests/scripts/perf-sanity/aggregated/gpt_oss_120b_fp4_grace_blackwell.yaml` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/scripts/perf-sanity/aggregated/gpt_oss_120b_fp4_grace_blackwell.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13166 - [None][fix] Raise clear error when GPT-OSS is used with non-TRTLLM attention backend

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13166
- 状态/时间: merged / 2026-05-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gpt_oss.py`；关联提交 `f5b0bdea06fd`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+9/-1，可读 patch 24 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +6/-0 (6 lines); hunks: -80,6 +80,12 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +6/-0 (6 lines); hunks: -80,6 +80,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gpt_oss.py
@@ -80,6 +80,12 @@ def __init__(
+        if self.attn_backend != "TRTLLM":
+            raise ValueError(
+                f"GPT-OSS model uses attention sinks, which are only supported "
+                f"with attn_backend='TRTLLM'. Current backend: {self.attn_backend}."
+            )
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +6/-0
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_gpt_oss.py`, `tensorrt_llm/_torch/modules/attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15765 - [None][perf] Validate GPT-OSS transceiver v2 performance

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15765
- 状态/时间: merged / 2026-07-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_triton.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_trtllm.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_tllm.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_triton.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_gpt-oss-120b-fp4_1k1k_con2048_ctx1_tp1_gen1_dep2_eplb0_mtp0_ccb-NIXL.yaml` 等 16 个文件；关联提交 `ce67288c926f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 20 个文件，+80/-10，可读 patch 547 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_triton.yaml` modified +4/-2 (6 lines); hunks: -26,7 +26,8 @@ context_servers:; -63,5 +64,6 @@ generation_servers:；`tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_trtllm.yaml` modified +4/-2 (6 lines); hunks: -26,7 +26,8 @@ context_servers:; -63,5 +64,6 @@ generation_servers:；`tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_tllm.yaml` modified +4/-2 (6 lines); hunks: -22,7 +22,8 @@ context_servers:; -59,5 +60,6 @@ generation_servers:；`tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_triton.yaml` modified +4/-2 (6 lines); hunks: -21,7 +21,8 @@ context_servers:; -53,5 +54,6 @@ generation_servers:。
- 代码 diff 细节:
  - `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_triton.yaml` modified +4/-2 (6 lines); hunks: -26,7 +26,8 @@ context_servers:; -63,5 +64,6 @@ generation_servers:
  - `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_trtllm.yaml` modified +4/-2 (6 lines); hunks: -26,7 +26,8 @@ context_servers:; -63,5 +64,6 @@ generation_servers:
  - `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_tllm.yaml` modified +4/-2 (6 lines); hunks: -22,7 +22,8 @@ context_servers:; -59,5 +60,6 @@ generation_servers:
  - `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_triton.yaml` modified +4/-2 (6 lines); hunks: -21,7 +21,8 @@ context_servers:; -53,5 +54,6 @@ generation_servers:
  - `tests/scripts/perf-sanity/disaggregated/gb200_gpt-oss-120b-fp4_1k1k_con2048_ctx1_tp1_gen1_dep2_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0 (4 lines); hunks: -57,6 +57,7 @@ worker_config:; -65,6 +66,7 @@ worker_config:
- 关键代码摘录:

```diff
diff -- tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_triton.yaml
@@ -26,7 +26,8 @@ context_servers:
-    backend: DEFAULT
+    backend: NIXL
+    transceiver_runtime: PYTHON
@@ -63,5 +64,6 @@ generation_servers:
-    backend: DEFAULT
+    backend: NIXL
diff -- tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_trtllm.yaml
@@ -26,7 +26,8 @@ context_servers:
-    backend: DEFAULT
+    backend: NIXL
+    transceiver_runtime: PYTHON
@@ -63,5 +64,6 @@ generation_servers:
-    backend: DEFAULT
+    backend: NIXL
diff -- tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_tllm.yaml
@@ -22,7 +22,8 @@ context_servers:
```

- 提取文件（未人工审阅）:
  - tests: `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_triton.yaml` modified +4/-2; `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_trtllm.yaml` modified +4/-2; `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_tllm.yaml` modified +4/-2; `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_triton.yaml` modified +4/-2; `tests/scripts/perf-sanity/disaggregated/gb200_gpt-oss-120b-fp4_1k1k_con2048_ctx1_tp1_gen1_dep2_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0; `tests/scripts/perf-sanity/disaggregated/gb200_gpt-oss-120b-fp4_1k1k_con512_ctx1_tp1_gen1_dep2_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_disaggregated_serving.py`, `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_triton.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_trtllm.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_tllm.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16378 - [None][fix] Fix unfused RoPE for yarn models: double rotation and GPT-OSS pairing

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16378
- 状态/时间: merged / 2026-07-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gpt_oss.py`；关联提交 `29f4552bfc67`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+86/-2，可读 patch 106 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +4/-1 (5 lines); hunks: -60,7 +60,10 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +4/-1 (5 lines); hunks: -60,7 +60,10 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gpt_oss.py
@@ -60,7 +60,10 @@ def __init__(
-            is_neox=False,
+            # GPT-OSS applies NeoX-style (rotate-half) RoPE, matching the HF
+            # reference. The fused kernel ignores this flag for yarn (which
+            # masked the wrong value); the unfused path honors it.
+            is_neox=True,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +4/-1
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modules/test_rotary_embedding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16479 - [None][feat] Default GPT-OSS to the Python KV-cache transceiver

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16479
- 状态/时间: merged / 2026-07-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gpt_oss.py`, `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py`；关联提交 `4e38fb823c11`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+12/-1，可读 patch 31 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +8/-1 (9 lines); hunks: -1,4 +1,4; -552,6 +552,13 @@ def forward(; symbols: forward, GptOssForCausalLM, get_preferred_transceiver_runtime，涉及 `forward, GptOssForCausalLM, get_preferred_transceiver_runtime`；`tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` modified +4/-0 (4 lines); hunks: -47,6 +47,10; symbols: test_gpt_oss_prefers_python_transceiver, dump_config_json，涉及 `test_gpt_oss_prefers_python_transceiver, dump_config_json`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +8/-1 (9 lines); hunks: -1,4 +1,4; -552,6 +552,13 @@ def forward(; symbols: forward, GptOssForCausalLM, get_preferred_transceiver_runtime
  - `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` modified +4/-0 (4 lines); hunks: -47,6 +47,10; symbols: test_gpt_oss_prefers_python_transceiver, dump_config_json
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gpt_oss.py
@@ -1,4 +1,4 @@
-from typing import Dict, Optional
+from typing import Any, Dict, Literal, Optional
@@ -552,6 +552,13 @@ def forward(
+    @classmethod
+    def get_preferred_transceiver_runtime(
+        cls,
diff -- tests/unittest/_torch/modeling/test_modeling_gpt_oss.py
@@ -47,6 +47,10 @@
+def test_gpt_oss_prefers_python_transceiver() -> None:
+    assert GptOssForCausalLM.get_preferred_transceiver_runtime() == "PYTHON"
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +8/-1
  - tests: `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` modified +4/-0
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17228 - [https://nvbugs/6533913][docs] Update pinned container for GPT OSS

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/17228
- 状态/时间: merged / 2026-08-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/source/blogs/tech_blog/blog09_Deploying_GPT_OSS_on_TRTLLM.md`；关联提交 `b469fb3bfef9`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/source/blogs/tech_blog/blog09_Deploying_GPT_OSS_on_TRTLLM.md` modified +2/-2 (4 lines); hunks: -19,7 +19,7 @@ We have a forthcoming guide for getting great performance on H...; -31,7 +31,7 @@ docker run --rm --ipc=host -it \。
- 代码 diff 细节:
  - `docs/source/blogs/tech_blog/blog09_Deploying_GPT_OSS_on_TRTLLM.md` modified +2/-2 (4 lines); hunks: -19,7 +19,7 @@ We have a forthcoming guide for getting great performance on H...; -31,7 +31,7 @@ docker run --rm --ipc=host -it \
- 关键代码摘录:

```diff
diff -- docs/source/blogs/tech_blog/blog09_Deploying_GPT_OSS_on_TRTLLM.md
@@ -19,7 +19,7 @@ We have a forthcoming guide for getting great performance on H100, however this
-The container image that you will use will be pulled from NVIDIA's NGC. This container is multi-platform and will run on both x64 and arm64 architectures: `nvcr.io/nvidia/tensorrt
+The container image that you will use will be pulled from NVIDIA's NGC. This container is multi-platform and will run on both x64 and arm64 architectures: `nvcr.io/nvidia/tensorrt
@@ -31,7 +31,7 @@ docker run --rm --ipc=host -it \
-  nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc1 \
+  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14 \
```

- 提取文件（未人工审阅）:
  - docs: `docs/source/blogs/tech_blog/blog09_Deploying_GPT_OSS_on_TRTLLM.md` modified +2/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs/source/blogs/tech_blog/blog09_Deploying_GPT_OSS_on_TRTLLM.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #16942 - [None][feat] Opt GPT-OSS in to KV cache manager V2 by default

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16942
- 状态/时间: merged / 2026-08-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gpt_oss.py`, `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py`；关联提交 `67dd1b7ed8ca`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+176/-11，可读 patch 265 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +25/-1 (26 lines); hunks: -1,4 +1,4; -32,6 +32,9; symbols: forward, GptOssForCausalLM, get_model_defaults, get_preferred_transceiver_runtime，涉及 `forward, GptOssForCausalLM, get_model_defaults`；`tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` modified +62/-1 (63 lines); hunks: -18,7 +18,11; -51,6 +55,63 @@ def test_gpt_oss_prefers_python_transceiver() -> None:; symbols: test_gpt_oss_prefers_python_transceiver, _resolve_gpt_oss_kv_cache_manager_v2, test_gpt_oss_model_defaults_select_v2, test_gpt_oss_explicit_setting_wins，涉及 `test_gpt_oss_prefers_python_transceiver, _resolve_gpt_oss_kv_cache_manager_v2, test_gpt_oss_model_defaults_select_v2`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +25/-1 (26 lines); hunks: -1,4 +1,4; -32,6 +32,9; symbols: forward, GptOssForCausalLM, get_model_defaults, get_preferred_transceiver_runtime
  - `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` modified +62/-1 (63 lines); hunks: -18,7 +18,11; -51,6 +55,63 @@ def test_gpt_oss_prefers_python_transceiver() -> None:; symbols: test_gpt_oss_prefers_python_transceiver, _resolve_gpt_oss_kv_cache_manager_v2, test_gpt_oss_model_defaults_select_v2, test_gpt_oss_explicit_setting_wins
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gpt_oss.py
@@ -1,4 +1,4 @@
-from typing import Any, Dict, Literal, Optional
+from typing import TYPE_CHECKING, Any, Dict, Literal, Optional
@@ -32,6 +32,9 @@
+if TYPE_CHECKING:
+    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
@@ -552,6 +555,27 @@ def forward(
diff -- tests/unittest/_torch/modeling/test_modeling_gpt_oss.py
@@ -18,7 +18,11 @@
-from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig
+from tensorrt_llm.llmapi import (CudaGraphConfig, Eagle3DecodingConfig,
+                                 KvCacheConfig, MoeConfig)
+from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
+from tensorrt_llm.llmapi.llm_utils import (_resolve_kv_cache_manager_v2_auto,
+                                           apply_model_defaults_to_llm_args)
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gpt_oss.py` modified +25/-1
  - tests: `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` modified +62/-1
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17597 - [None][fix] Remove obsolete GPT-OSS two-model Eagle3 tests

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/17597
- 状态/时间: merged / 2026-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py`；关联提交 `839819640311`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+0/-25，可读 patch 32 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` modified +0/-25 (25 lines); hunks: -75,31 +75,6 @@ def test_gpt_oss_explicit_setting_wins(user_setting):; symbols: test_gpt_oss_explicit_setting_wins, test_gpt_oss_two_model_eagle3_falls_back_to_v1, test_gpt_oss_explicit_v2_rejects_two_model_eagle3, test_gpt_oss_one_model_eagle3_keeps_v2，涉及 `test_gpt_oss_explicit_setting_wins, test_gpt_oss_two_model_eagle3_falls_back_to_v1, test_gpt_oss_explicit_v2_rejects_two_model_eagle3`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` modified +0/-25 (25 lines); hunks: -75,31 +75,6 @@ def test_gpt_oss_explicit_setting_wins(user_setting):; symbols: test_gpt_oss_explicit_setting_wins, test_gpt_oss_two_model_eagle3_falls_back_to_v1, test_gpt_oss_explicit_v2_rejects_two_model_eagle3, test_gpt_oss_one_model_eagle3_keeps_v2
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_gpt_oss.py
@@ -75,31 +75,6 @@ def test_gpt_oss_explicit_setting_wins(user_setting):
-def test_gpt_oss_two_model_eagle3_falls_back_to_v1():
-    """Two-model Eagle3 builds a separate draft engine with its own KV cache
-    manager, and V2 sizes both from the full budget, so the model preference is
-    demoted to V1."""
-    assert _resolve_gpt_oss_kv_cache_manager_v2(
-        speculative_config=Eagle3DecodingConfig(
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py` modified +0/-25
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_gpt_oss.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
