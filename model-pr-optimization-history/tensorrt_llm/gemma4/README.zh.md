# TensorRT-LLM Gemma 4 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tensorrt_llm/_torch/configs/gemma4.py` | [#15833](https://github.com/NVIDIA/TensorRT-LLM/pull/15833) |
| `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` | [#12932](https://github.com/NVIDIA/TensorRT-LLM/pull/12932), [#15768](https://github.com/NVIDIA/TensorRT-LLM/pull/15768), [#15833](https://github.com/NVIDIA/TensorRT-LLM/pull/15833), [#16108](https://github.com/NVIDIA/TensorRT-LLM/pull/16108), [#16797](https://github.com/NVIDIA/TensorRT-LLM/pull/16797) |
| `tensorrt_llm/_torch/models/modeling_gemma4.py` | [#12932](https://github.com/NVIDIA/TensorRT-LLM/pull/12932), [#14134](https://github.com/NVIDIA/TensorRT-LLM/pull/14134), [#15768](https://github.com/NVIDIA/TensorRT-LLM/pull/15768), [#15833](https://github.com/NVIDIA/TensorRT-LLM/pull/15833), [#15848](https://github.com/NVIDIA/TensorRT-LLM/pull/15848), [#16074](https://github.com/NVIDIA/TensorRT-LLM/pull/16074), [#17396](https://github.com/NVIDIA/TensorRT-LLM/pull/17396), [#17557](https://github.com/NVIDIA/TensorRT-LLM/pull/17557), [#17837](https://github.com/NVIDIA/TensorRT-LLM/pull/17837), [#18002](https://github.com/NVIDIA/TensorRT-LLM/pull/18002) |
| `tensorrt_llm/_torch/models/modeling_gemma4_audio.py` | [#14300](https://github.com/NVIDIA/TensorRT-LLM/pull/14300) |
| `tensorrt_llm/_torch/models/modeling_gemma4_unified.py` | [#15768](https://github.com/NVIDIA/TensorRT-LLM/pull/15768), [#15833](https://github.com/NVIDIA/TensorRT-LLM/pull/15833), [#16662](https://github.com/NVIDIA/TensorRT-LLM/pull/16662) |
| `tensorrt_llm/_torch/models/modeling_gemma4_vision.py` | [#14300](https://github.com/NVIDIA/TensorRT-LLM/pull/14300), [#15566](https://github.com/NVIDIA/TensorRT-LLM/pull/15566), [#15613](https://github.com/NVIDIA/TensorRT-LLM/pull/15613), [#16509](https://github.com/NVIDIA/TensorRT-LLM/pull/16509), [#19571](https://github.com/NVIDIA/TensorRT-LLM/pull/19571) |
| `tensorrt_llm/_torch/models/modeling_gemma4mm.py` | [#12932](https://github.com/NVIDIA/TensorRT-LLM/pull/12932), [#14134](https://github.com/NVIDIA/TensorRT-LLM/pull/14134), [#14300](https://github.com/NVIDIA/TensorRT-LLM/pull/14300), [#15566](https://github.com/NVIDIA/TensorRT-LLM/pull/15566), [#15613](https://github.com/NVIDIA/TensorRT-LLM/pull/15613), [#15833](https://github.com/NVIDIA/TensorRT-LLM/pull/15833), [#15848](https://github.com/NVIDIA/TensorRT-LLM/pull/15848), [#16662](https://github.com/NVIDIA/TensorRT-LLM/pull/16662), [#17231](https://github.com/NVIDIA/TensorRT-LLM/pull/17231), [#17396](https://github.com/NVIDIA/TensorRT-LLM/pull/17396), [#17837](https://github.com/NVIDIA/TensorRT-LLM/pull/17837), [#18274](https://github.com/NVIDIA/TensorRT-LLM/pull/18274) |
| `tensorrt_llm/_torch/modules/gemma4/__init__.py` | [#16074](https://github.com/NVIDIA/TensorRT-LLM/pull/16074) |
| `tensorrt_llm/_torch/modules/gemma4/fused_qkv.py` | [#16074](https://github.com/NVIDIA/TensorRT-LLM/pull/16074) |
| `tensorrt_llm/serve/tool_parser/gemma4_parser.py` | [#12932](https://github.com/NVIDIA/TensorRT-LLM/pull/12932), [#13248](https://github.com/NVIDIA/TensorRT-LLM/pull/13248) |
| `tests/scripts/perf-sanity/aggregated/gemma4_26b_a4b_nvfp4_blackwell.yaml` | 无直接 PR 号提交 |
| `tests/unittest/_torch/modeling/test_gemma4_e2e_dummy.py` | [#12932](https://github.com/NVIDIA/TensorRT-LLM/pull/12932), [#16099](https://github.com/NVIDIA/TensorRT-LLM/pull/16099) |
| `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` | [#12932](https://github.com/NVIDIA/TensorRT-LLM/pull/12932), [#14300](https://github.com/NVIDIA/TensorRT-LLM/pull/14300), [#15848](https://github.com/NVIDIA/TensorRT-LLM/pull/15848), [#16662](https://github.com/NVIDIA/TensorRT-LLM/pull/16662), [#17231](https://github.com/NVIDIA/TensorRT-LLM/pull/17231), [#18274](https://github.com/NVIDIA/TensorRT-LLM/pull/18274) |
| `tests/unittest/_torch/modeling/test_modeling_gemma4.py` | [#12932](https://github.com/NVIDIA/TensorRT-LLM/pull/12932), [#14082](https://github.com/NVIDIA/TensorRT-LLM/pull/14082), [#14300](https://github.com/NVIDIA/TensorRT-LLM/pull/14300), [#15833](https://github.com/NVIDIA/TensorRT-LLM/pull/15833), [#15848](https://github.com/NVIDIA/TensorRT-LLM/pull/15848), [#16108](https://github.com/NVIDIA/TensorRT-LLM/pull/16108), [#16509](https://github.com/NVIDIA/TensorRT-LLM/pull/16509), [#16797](https://github.com/NVIDIA/TensorRT-LLM/pull/16797), [#17557](https://github.com/NVIDIA/TensorRT-LLM/pull/17557), [#18002](https://github.com/NVIDIA/TensorRT-LLM/pull/18002), [#19571](https://github.com/NVIDIA/TensorRT-LLM/pull/19571) |
| `tests/unittest/_torch/modeling/test_modeling_gemma4_unified.py` | [#15768](https://github.com/NVIDIA/TensorRT-LLM/pull/15768), [#16662](https://github.com/NVIDIA/TensorRT-LLM/pull/16662) |
| `tests/unittest/_torch/modules/test_gemma4_fused_qkv_prep.py` | [#16074](https://github.com/NVIDIA/TensorRT-LLM/pull/16074) |

## PR 覆盖总览

- git 追溯 PR 数: 22
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 22
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-30 | [#13248](https://github.com/NVIDIA/TensorRT-LLM/pull/13248) | merged | [None][feat] AutoDeploy: add Gemma 4 reasoning and tool-call parsers | `tensorrt_llm/serve/tool_parser/gemma4_parser.py` |
| 2026-05-09 | [#12932](https://github.com/NVIDIA/TensorRT-LLM/pull/12932) | merged | [None][feat] Add Gemma4 multimodal model support (text + vision + audio) | `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py`, `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` |
| 2026-05-14 | [#14082](https://github.com/NVIDIA/TensorRT-LLM/pull/14082) | merged | [None][fix] Gemma4 CUDA-graph test KV pre-alloc + L0 registration | `tests/unittest/_torch/modeling/test_modeling_gemma4.py` |
| 2026-05-19 | [#14134](https://github.com/NVIDIA/TensorRT-LLM/pull/14134) | merged | [None][feat] Add chunked prefill support for Gemma4 (text + vision multimodal) | `tensorrt_llm/_torch/models/modeling_gemma4mm.py`, `tensorrt_llm/_torch/models/modeling_gemma4.py` |
| 2026-05-21 | [#14300](https://github.com/NVIDIA/TensorRT-LLM/pull/14300) | merged | [None][feat] Gemma4 MM: native vision + audio towers | `tests/unittest/_torch/modeling/test_gemma4_multimodal.py`, `tensorrt_llm/_torch/models/modeling_gemma4_vision.py`, `tensorrt_llm/_torch/models/modeling_gemma4_audio.py` |
| 2026-06-26 | [#15566](https://github.com/NVIDIA/TensorRT-LLM/pull/15566) | merged | [#15613][fix] Gemma4 multimodal: fix vision TP and xgrammar startup crashes | `tensorrt_llm/_torch/models/modeling_gemma4_vision.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py` |
| 2026-07-04 | [#15768](https://github.com/NVIDIA/TensorRT-LLM/pull/15768) | merged | [None][feat] Add Gemma 4 12B Unified (encoder-free multimodal) support | `tensorrt_llm/_torch/models/modeling_gemma4_unified.py`, `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` |
| 2026-07-06 | [#15848](https://github.com/NVIDIA/TensorRT-LLM/pull/15848) | merged | [None][perf] Improve inference correctness and perf for Gemma4 | `tensorrt_llm/_torch/models/modeling_gemma4mm.py`, `tests/unittest/_torch/modeling/test_gemma4_multimodal.py`, `tensorrt_llm/_torch/models/modeling_gemma4.py` |
| 2026-07-10 | [#16108](https://github.com/NVIDIA/TensorRT-LLM/pull/16108) | merged | [https://nvbugs/6379636][fix] Fix Gemma4 MoE weight loading | `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py` |
| 2026-07-11 | [#16074](https://github.com/NVIDIA/TensorRT-LLM/pull/16074) | merged | [TRTLLM-14138][perf] Add fused kernels for Gemma4 serving | `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tensorrt_llm/_torch/modules/gemma4/fused_qkv.py`, `tests/unittest/_torch/modules/test_gemma4_fused_qkv_prep.py` |
| 2026-07-15 | [#16099](https://github.com/NVIDIA/TensorRT-LLM/pull/16099) | merged | [None][fix] Fix Gemma4 illegal memory access when max_seq_len is at most the sliding window size | `tests/unittest/_torch/modeling/test_gemma4_e2e_dummy.py`, `tensorrt_llm/_torch/attention_backend/flashinfer.py`, `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` |
| 2026-07-17 | [#16509](https://github.com/NVIDIA/TensorRT-LLM/pull/16509) | merged | [None][perf] Various Gemma4 related perf fixes | `tensorrt_llm/_torch/models/modeling_gemma4_vision.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py` |
| 2026-07-24 | [#16797](https://github.com/NVIDIA/TensorRT-LLM/pull/16797) | merged | [NVBUG-6379624][fix] Enable W4A8 checkpoint loading for Gemma4 K=V layers | `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py` |
| 2026-07-31 | [#16662](https://github.com/NVIDIA/TensorRT-LLM/pull/16662) | merged | [None][feat] Enable MM encoder cache on Qwen3.x and Gemma4 VLMs | `tensorrt_llm/_torch/models/modeling_gemma4mm.py`, `tensorrt_llm/_torch/models/modeling_gemma4_unified.py`, `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` |
| 2026-08-04 | [#15833](https://github.com/NVIDIA/TensorRT-LLM/pull/15833) | merged | [None][feat] Add Gemma4 MTP assistant support | `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tensorrt_llm/_torch/configs/gemma4.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py` |
| 2026-08-06 | [#17231](https://github.com/NVIDIA/TensorRT-LLM/pull/17231) | merged | [https://nvbugs/6550127][fix] Support Gemma4 multimodal cache partial hits | `tests/unittest/_torch/modeling/test_gemma4_multimodal.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py` |
| 2026-08-13 | [#17557](https://github.com/NVIDIA/TensorRT-LLM/pull/17557) | merged | [https://nvbugs/6566891][fix] Use FlashInfer FA2 for Gemma4 on SM120 and SM121 | `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py` |
| 2026-08-18 | [#17396](https://github.com/NVIDIA/TensorRT-LLM/pull/17396) | merged | [None][feat] Enable KVCacheManagerV2 by default for Gemma3 and Gemma4 | `tensorrt_llm/_torch/models/modeling_gemma4mm.py`, `tensorrt_llm/_torch/models/modeling_gemma4.py` |
| 2026-08-19 | [#17837](https://github.com/NVIDIA/TensorRT-LLM/pull/17837) | merged | [None][fix] Restore Gemma4 shared-KV draft loading | `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py` |
| 2026-08-26 | [#18002](https://github.com/NVIDIA/TensorRT-LLM/pull/18002) | merged | [None][fix] Stabilize Gemma4 FA2 CUDA Graph decode on Hopper | `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py` |
| 2026-08-29 | [#18274](https://github.com/NVIDIA/TensorRT-LLM/pull/18274) | merged | [https://nvbugs/6663281][fix] Fix Gemma4 video token counting | `tensorrt_llm/_torch/models/modeling_gemma4mm.py`, `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` |
| 2026-09-23 | [#19571](https://github.com/NVIDIA/TensorRT-LLM/pull/19571) | merged | [None][models] Avoid GEMM in Gemma4 vision RoPE | `tensorrt_llm/_torch/models/modeling_gemma4_vision.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py` |

## 逐 PR diff 审计卡

### PR #13248 - [None][feat] AutoDeploy: add Gemma 4 reasoning and tool-call parsers

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13248
- 状态/时间: merged / 2026-04-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/serve/tool_parser/gemma4_parser.py`；关联提交 `eaed16fb4bb5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+991/-292，可读 patch 1344 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/serve/tool_parser/gemma4_parser.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: _gemma4_args_to_json, _parse_gemma4_args, Gemma4ToolParser, __init__，涉及 `_gemma4_args_to_json, _parse_gemma4_args, Gemma4ToolParser`。
- 代码 diff 细节:
  - `tensorrt_llm/serve/tool_parser/gemma4_parser.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: _gemma4_args_to_json, _parse_gemma4_args, Gemma4ToolParser, __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/serve/tool_parser/gemma4_parser.py
@@ -0,0 +1,220 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+import json
+import re
+from typing import Any, Dict, List
+from tensorrt_llm.logger import logger
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/serve/tool_parser/gemma4_parser.py` added +220/-0
- 验证与风险: diff 自带测试面 `tests/unittest/llmapi/apps/test_tool_parsers.py`, `tests/unittest/llmapi/test_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12932 - [None][feat] Add Gemma4 multimodal model support (text + vision + audio)

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12932
- 状态/时间: merged / 2026-05-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py`, `tensorrt_llm/serve/tool_parser/gemma4_parser.py`, `tests/unittest/_torch/modeling/test_gemma4_e2e_dummy.py` 等 7 个文件；关联提交 `c163c728676e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 42 个文件，+11837/-434，可读 patch 10025 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4.py` added +1087/-0 (1087 lines); hunks: -0,0 +1,1087; symbols: Gemma4TextScaledWordEmbedding, __init__, forward, gelu_tanh，涉及 `Gemma4TextScaledWordEmbedding, __init__, forward`；`tensorrt_llm/_torch/models/modeling_gemma4mm.py` added +918/-0 (918 lines); hunks: -0,0 +1,918; symbols: _is_disagg, RMSNormNoScale, __init__, forward，涉及 `_is_disagg, RMSNormNoScale, __init__`；`tests/unittest/_torch/modeling/test_gemma4_multimodal.py` added +714/-0 (714 lines); hunks: -0,0 +1,714; symbols: _get_model_path, _model_available, TestGemma4InputProcessor, setUpClass，涉及 `_get_model_path, _model_available, TestGemma4InputProcessor`；`tensorrt_llm/serve/tool_parser/gemma4_parser.py` modified +485/-161 (646 lines); hunks: -1,9 +1,20; -15,206 +26,519; symbols: _gemma4_args_to_json, _find_matching_brace, _parse_gemma4_value, _parse_gemma4_array，涉及 `_gemma4_args_to_json, _find_matching_brace, _parse_gemma4_value`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4.py` added +1087/-0 (1087 lines); hunks: -0,0 +1,1087; symbols: Gemma4TextScaledWordEmbedding, __init__, forward, gelu_tanh
  - `tensorrt_llm/_torch/models/modeling_gemma4mm.py` added +918/-0 (918 lines); hunks: -0,0 +1,918; symbols: _is_disagg, RMSNormNoScale, __init__, forward
  - `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` added +714/-0 (714 lines); hunks: -0,0 +1,714; symbols: _get_model_path, _model_available, TestGemma4InputProcessor, setUpClass
  - `tensorrt_llm/serve/tool_parser/gemma4_parser.py` modified +485/-161 (646 lines); hunks: -1,9 +1,20; -15,206 +26,519; symbols: _gemma4_args_to_json, _find_matching_brace, _parse_gemma4_value, _parse_gemma4_array
  - `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` added +342/-0 (342 lines); hunks: -0,0 +1,342; symbols: Gemma4HfWeightMapper, _is_vlm, apply_callbacks, _resolve_layer_head_dim
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4.py
@@ -0,0 +1,1087 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/models/modeling_gemma4mm.py
@@ -0,0 +1,918 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tests/unittest/_torch/modeling/test_gemma4_multimodal.py
@@ -0,0 +1,714 @@
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4.py` added +1087/-0; `tensorrt_llm/_torch/models/modeling_gemma4mm.py` added +918/-0; `tensorrt_llm/serve/tool_parser/gemma4_parser.py` modified +485/-161; `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` added +342/-0
  - tests: `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` added +714/-0; `tests/unittest/_torch/modeling/test_modeling_gemma4.py` added +3009/-0; `tests/unittest/_torch/modeling/test_gemma4_e2e_dummy.py` added +291/-0
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/attention/test_flashinfer_attention.py`, `tests/unittest/_torch/attention_backend/test_triton_prefill.py`, `tests/unittest/_torch/executor/test_kv_cache_estimation.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14082 - [None][fix] Gemma4 CUDA-graph test KV pre-alloc + L0 registration

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14082
- 状态/时间: merged / 2026-05-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；关联提交 `95204b780259`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+33/-16，可读 patch 85 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +31/-16 (47 lines); hunks: -25,24 +25,35; -2278,7 +2289,9 @@ def test_cuda_graph_multi_step_decode(self):; symbols: test_cuda_graph_multi_step_decode, test_cuda_graph_decode_high_gqa，涉及 `test_cuda_graph_multi_step_decode, test_cuda_graph_decode_high_gqa`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +31/-16 (47 lines); hunks: -25,24 +25,35; -2278,7 +2289,9 @@ def test_cuda_graph_multi_step_decode(self):; symbols: test_cuda_graph_multi_step_decode, test_cuda_graph_decode_high_gqa
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_gemma4.py
@@ -25,24 +25,35 @@
+import transformers
+from packaging.version import Version
-pytest.importorskip(
-    "transformers", minversion="5.5.0", reason="Gemma4 requires transformers>=5.5.0"
+# Use a module-level pytestmark.skipif (not pytest.importorskip) so collection
+# still picks up the test cases when the version gate fires. importorskip
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +31/-16
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14134 - [None][feat] Add chunked prefill support for Gemma4 (text + vision multimodal)

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14134
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py`；关联提交 `025086fc1a37`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+592/-16，可读 patch 822 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +64/-1 (65 lines); hunks: -48,7 +48,7; -199,6 +199,13 @@ class Gemma4InputProcessor(BaseMultimodalInputProcessor, Ba...; symbols: Gemma4InputProcessor, __init__, get_vocab_size, get_mm_token_ids，涉及 `Gemma4InputProcessor, __init__, get_vocab_size`；`tensorrt_llm/_torch/models/modeling_gemma4.py` modified +26/-7 (33 lines); hunks: -980,23 +980,41 @@ def get_context_mask(; -1014,13 +1032,14 @@ def get_flashinfer_attention_mask(; symbols: get_context_mask, get_flashinfer_attention_mask，涉及 `get_context_mask, get_flashinfer_attention_mask`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +64/-1 (65 lines); hunks: -48,7 +48,7; -199,6 +199,13 @@ class Gemma4InputProcessor(BaseMultimodalInputProcessor, Ba...; symbols: Gemma4InputProcessor, __init__, get_vocab_size, get_mm_token_ids
  - `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +26/-7 (33 lines); hunks: -980,23 +980,41 @@ def get_context_mask(; -1014,13 +1032,14 @@ def get_flashinfer_attention_mask(; symbols: get_context_mask, get_flashinfer_attention_mask
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4mm.py
@@ -48,7 +48,7 @@
-from .modeling_multimodal_utils import fuse_input_embeds
+from .modeling_multimodal_utils import find_input_mm_embeds, fuse_input_embeds
@@ -199,6 +199,13 @@ class Gemma4InputProcessor(BaseMultimodalInputProcessor, BaseMultimodalDummyInpu
+    # Default class-level fallback. Real value computed per-instance below
+    # from text_config.use_bidirectional_attention. Only 26B/31B set
+    # use_bidirectional_attention="vision" — their image blocks need intact
diff -- tensorrt_llm/_torch/models/modeling_gemma4.py
@@ -980,23 +980,41 @@ def get_context_mask(
+        prefix_len: int = 0,
-        """Build context mask with causal + bidirectional for MM tokens."""
+        """Build context mask with causal + bidirectional for MM tokens.
+        Returns a [extend_len, prefix_len + extend_len] mask where:
+        - The first `prefix_len` columns (cached/paged history) are True for
+          all rows. SWA window enforcement is delegated to the kernel's
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +64/-1; `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +26/-7
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/executor/test_kv_cache_v2_scheduler.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14300 - [None][feat] Gemma4 MM: native vision + audio towers

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14300
- 状态/时间: merged / 2026-05-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4_audio.py`, `tensorrt_llm/_torch/models/modeling_gemma4_vision.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py`, `tests/unittest/_torch/modeling/test_gemma4_multimodal.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；关联提交 `4c5500ea44ff`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+3023/-270，可读 patch 3612 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +891/-237 (1128 lines); hunks: -12,13 +12,32; -28,28 +47,44; symbols: _get_model_path, _make_dummy_pixel_input, _build_trt_vision_tower, TestGemma4VisionTower，涉及 `_get_model_path, _make_dummy_pixel_input, _build_trt_vision_tower`；`tensorrt_llm/_torch/models/modeling_gemma4_vision.py` added +1001/-0 (1001 lines); hunks: -0,0 +1,1001; symbols: Gemma4VisionRMSNorm, __init__, forward, VisionOutput，涉及 `Gemma4VisionRMSNorm, __init__, forward`；`tensorrt_llm/_torch/models/modeling_gemma4_audio.py` added +720/-0 (720 lines); hunks: -0,0 +1,720; symbols: AudioOutput, Gemma4AudioRMSNorm, __init__, forward，涉及 `AudioOutput, Gemma4AudioRMSNorm, __init__`；`tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +106/-33 (139 lines); hunks: -15,8 +15,9; -48,6 +49,8; symbols: __call__, get_model_defaults, _check_and_adjust_experts_implementation, __init__，涉及 `__call__, get_model_defaults, _check_and_adjust_experts_implementation`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +891/-237 (1128 lines); hunks: -12,13 +12,32; -28,28 +47,44; symbols: _get_model_path, _make_dummy_pixel_input, _build_trt_vision_tower, TestGemma4VisionTower
  - `tensorrt_llm/_torch/models/modeling_gemma4_vision.py` added +1001/-0 (1001 lines); hunks: -0,0 +1,1001; symbols: Gemma4VisionRMSNorm, __init__, forward, VisionOutput
  - `tensorrt_llm/_torch/models/modeling_gemma4_audio.py` added +720/-0 (720 lines); hunks: -0,0 +1,720; symbols: AudioOutput, Gemma4AudioRMSNorm, __init__, forward
  - `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +106/-33 (139 lines); hunks: -15,8 +15,9; -48,6 +49,8; symbols: __call__, get_model_defaults, _check_and_adjust_experts_implementation, __init__
  - `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +304/-0 (304 lines); hunks: -3020,5 +3020,309 @@ def test_forward_accepts_input_ids_and_inputs_embeds_tog...; symbols: test_forward_accepts_input_ids_and_inputs_embeds_together, TestGemma4MMTowerRMSNormConvention, _rms_normalize, test_audio_rmsnorm_does_not_import_hf_class
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_gemma4_multimodal.py
@@ -12,13 +12,32 @@
-"""Unit tests for the Gemma4 multimodal model components.
-Tests Gemma4MultimodalEmbedder, Gemma4ForConditionalGeneration,
-and Gemma4InputProcessor with HF reference comparison.
+"""Unit + E2E tests for the Gemma4 multimodal model components.
+Tier 0/1 (no LLM_MODELS_ROOT needed, GPU only):
+  - ``TestGemma4VisionTower`` — refactored TRT-LLM ``Gemma4VisionModel``
diff -- tensorrt_llm/_torch/models/modeling_gemma4_vision.py
@@ -0,0 +1,1001 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/models/modeling_gemma4_audio.py
@@ -0,0 +1,720 @@
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +891/-237; `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +304/-0
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4_vision.py` added +1001/-0; `tensorrt_llm/_torch/models/modeling_gemma4_audio.py` added +720/-0; `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +106/-33
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modeling/test_gemma4_multimodal.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15566 - [#15613][fix] Gemma4 multimodal: fix vision TP and xgrammar startup crashes

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15566
- 状态/时间: merged / 2026-06-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4_vision.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py`；关联提交 `0425801b33ab`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+10/-2，可读 patch 26 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4_vision.py` modified +6/-2 (8 lines); hunks: -949,8 +949,12 @@ def _pad_attention_head_dim(self, weights: Dict[str, torch....; symbols: _pad_attention_head_dim，涉及 `_pad_attention_head_dim`；`tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +4/-0 (4 lines); hunks: -768,6 +768,10 @@ def post_config(self):; symbols: post_config, infer_max_seq_len, vocab_size_padded, multimodal_data_device_paths，涉及 `post_config, infer_max_seq_len, vocab_size_padded`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4_vision.py` modified +6/-2 (8 lines); hunks: -949,8 +949,12 @@ def _pad_attention_head_dim(self, weights: Dict[str, torch....; symbols: _pad_attention_head_dim
  - `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +4/-0 (4 lines); hunks: -768,6 +768,10 @@ def post_config(self):; symbols: post_config, infer_max_seq_len, vocab_size_padded, multimodal_data_device_paths
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4_vision.py
@@ -949,8 +949,12 @@ def _pad_attention_head_dim(self, weights: Dict[str, torch.Tensor]) -> Dict[str,
-        nh = first_attn.num_heads
-        nkv = first_attn.num_key_value_heads
+        # first_attn.num_heads / num_key_value_heads are already divided by
+        # tp_size, but these weights are still unsharded here, so read the full
+        # head counts from the vision config (no-op at tp1).
+        vc = first_attn.vision_config
diff -- tensorrt_llm/_torch/models/modeling_gemma4mm.py
@@ -768,6 +768,10 @@ def post_config(self):
+    @property
+    def vocab_size_padded(self) -> int:
+        return self.llm.vocab_size_padded
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4_vision.py` modified +6/-2; `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_gemma4_vision.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15768 - [None][feat] Add Gemma 4 12B Unified (encoder-free multimodal) support

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15768
- 状态/时间: merged / 2026-07-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tensorrt_llm/_torch/models/modeling_gemma4_unified.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4_unified.py`；关联提交 `c50ac2a0f79a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+1930/-21，可读 patch 2063 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4_unified.py` added +1481/-0 (1481 lines); hunks: -0,0 +1,1481; symbols: Gemma4UnifiedVisionEmbedder, __init__, forward, load_weights，涉及 `Gemma4UnifiedVisionEmbedder, __init__, forward`；`tensorrt_llm/_torch/models/modeling_gemma4.py` modified +8/-5 (13 lines); hunks: -954,14 +954,17 @@ def get_model_defaults(cls, llm_args) -> dict:; symbols: get_model_defaults, _get_token_type_mask，涉及 `get_model_defaults, _get_token_type_mask`；`tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` modified +1/-0 (1 lines); hunks: -27,6 +27,7; symbols: Gemma4HfWeightMapper, _is_vlm，涉及 `Gemma4HfWeightMapper, _is_vlm`；`tests/unittest/_torch/modeling/test_modeling_gemma4_unified.py` added +257/-0 (257 lines); hunks: -0,0 +1,257; symbols: test_config_registered_with_transformers, test_config_shim_parses_checkpoint_dict, test_config_shim_handles_absent_modalities, test_sub_config_shims_standalone，涉及 `test_config_registered_with_transformers, test_config_shim_parses_checkpoint_dict, test_config_shim_handles_absent_modalities`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4_unified.py` added +1481/-0 (1481 lines); hunks: -0,0 +1,1481; symbols: Gemma4UnifiedVisionEmbedder, __init__, forward, load_weights
  - `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +8/-5 (13 lines); hunks: -954,14 +954,17 @@ def get_model_defaults(cls, llm_args) -> dict:; symbols: get_model_defaults, _get_token_type_mask
  - `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` modified +1/-0 (1 lines); hunks: -27,6 +27,7; symbols: Gemma4HfWeightMapper, _is_vlm
  - `tests/unittest/_torch/modeling/test_modeling_gemma4_unified.py` added +257/-0 (257 lines); hunks: -0,0 +1,257; symbols: test_config_registered_with_transformers, test_config_shim_parses_checkpoint_dict, test_config_shim_handles_absent_modalities, test_sub_config_shims_standalone
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4_unified.py
@@ -0,0 +1,1481 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/models/modeling_gemma4.py
@@ -954,14 +954,17 @@ def get_model_defaults(cls, llm_args) -> dict:
-        mm_token_type_ids: 0=text, 1=image, 2=video (or any positive int for
-        a modality blob). Tokens within the same contiguous blob of the same
-        modality attend bidirectionally to each other.
+        mm_token_type_ids: 0=text, 1=image, 2=video, 3=audio. Only VISION
+        tokens (image/video) attend bidirectionally within their contiguous
+        blob; text and audio stay causal. Matches HF Gemma4, where
diff -- tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py
@@ -27,6 +27,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4_unified.py` added +1481/-0; `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +8/-5; `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` modified +1/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_gemma4_unified.py` added +257/-0
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modeling/test_modeling_gemma4_unified.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15848 - [None][perf] Improve inference correctness and perf for Gemma4

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15848
- 状态/时间: merged / 2026-07-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py`, `tests/unittest/_torch/modeling/test_gemma4_multimodal.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；关联提交 `c5bdcaa49aaa`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+648/-195，可读 patch 1174 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +165/-120 (285 lines); hunks: -23,6 +23,7; -43,6 +44,7; symbols: _get_audio_features, forward, _has_active_multimodal_tokens, _forward_multimodal_encoder，涉及 `_get_audio_features, forward, _has_active_multimodal_tokens`；`tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +138/-0 (138 lines); hunks: -62,6 +62,11; -651,6 +656,22 @@ def test_pipeline_matches_hf(self):; symbols: test_pipeline_matches_hf, TestGemma4ForConditionalGeneration, _make_model, test_instantiation_with_vision，涉及 `test_pipeline_matches_hf, TestGemma4ForConditionalGeneration, _make_model`；`tensorrt_llm/_torch/models/modeling_gemma4.py` modified +38/-38 (76 lines); hunks: -449,17 +449,14 @@ def apply(; -865,7 +862,6 @@ def forward(; symbols: apply, Gemma4MoE, forward, _get_token_type_mask，涉及 `apply, Gemma4MoE, forward`；`tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +90/-7 (97 lines); hunks: -50,6 +50,7; -786,6 +787,34 @@ def test_logit_softcapping_matches_hf(self):; symbols: test_logit_softcapping_matches_hf, _stabilize_moe_routing, add_expert_offsets, _run_full_model_comparison，涉及 `test_logit_softcapping_matches_hf, _stabilize_moe_routing, add_expert_offsets`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +165/-120 (285 lines); hunks: -23,6 +23,7; -43,6 +44,7; symbols: _get_audio_features, forward, _has_active_multimodal_tokens, _forward_multimodal_encoder
  - `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +138/-0 (138 lines); hunks: -62,6 +62,11; -651,6 +656,22 @@ def test_pipeline_matches_hf(self):; symbols: test_pipeline_matches_hf, TestGemma4ForConditionalGeneration, _make_model, test_instantiation_with_vision
  - `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +38/-38 (76 lines); hunks: -449,17 +449,14 @@ def apply(; -865,7 +862,6 @@ def forward(; symbols: apply, Gemma4MoE, forward, _get_token_type_mask
  - `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +90/-7 (97 lines); hunks: -50,6 +50,7; -786,6 +787,34 @@ def test_logit_softcapping_matches_hf(self):; symbols: test_logit_softcapping_matches_hf, _stabilize_moe_routing, add_expert_offsets, _run_full_model_comparison
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4mm.py
@@ -23,6 +23,7 @@
+from itertools import groupby
@@ -43,6 +44,7 @@
+from ...inputs.multimodal import MultimodalParams
@@ -55,6 +57,7 @@
+    get_multimodal_embeddings,
@@ -876,141 +879,188 @@ def _get_audio_features(
diff -- tests/unittest/_torch/modeling/test_gemma4_multimodal.py
@@ -62,6 +62,11 @@
+from tensorrt_llm._torch.models.modeling_multimodal_utils import (  # noqa: E402
+    find_input_mm_embeds,
+    get_multimodal_embeddings,
+)
+from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalRuntimeData  # noqa: E402
@@ -651,6 +656,22 @@ def test_pipeline_matches_hf(self):
diff -- tensorrt_llm/_torch/models/modeling_gemma4.py
@@ -449,17 +449,14 @@ def apply(
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +165/-120; `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +38/-38
  - tests: `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +138/-0; `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +90/-7
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/executor/test_per_layer_head_dim.py`, `tests/unittest/_torch/executor/test_resource_manager.py`, `tests/unittest/_torch/modeling/test_gemma4_multimodal.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16108 - [https://nvbugs/6379636][fix] Fix Gemma4 MoE weight loading

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16108
- 状态/时间: merged / 2026-07-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；关联提交 `fd9166c0a762`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+156/-31，可读 patch 251 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` modified +22/-0 (22 lines); hunks: -205,12 +205,19 @@ def _remap_moe_keys(self, weights: dict) -> dict:; -220,6 +227,18 @@ def _remap_moe_keys(self, weights: dict) -> dict:; symbols: _remap_moe_keys，涉及 `_remap_moe_keys`；`tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +91/-30 (121 lines); hunks: -23,38 +23,21; -375,6 +358,84 @@ def test_num_kv_heads_per_layer_type(self):; symbols: test_num_kv_heads_per_layer_type, TestGemma4HfWeightMapper, test_remap_modelopt_nvfp4_per_expert_weights, test_remap_modelopt_nvfp4_vlm_prefix，涉及 `test_num_kv_heads_per_layer_type, TestGemma4HfWeightMapper, test_remap_modelopt_nvfp4_per_expert_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` modified +22/-0 (22 lines); hunks: -205,12 +205,19 @@ def _remap_moe_keys(self, weights: dict) -> dict:; -220,6 +227,18 @@ def _remap_moe_keys(self, weights: dict) -> dict:; symbols: _remap_moe_keys
  - `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +91/-30 (121 lines); hunks: -23,38 +23,21; -375,6 +358,84 @@ def test_num_kv_heads_per_layer_type(self):; symbols: test_num_kv_heads_per_layer_type, TestGemma4HfWeightMapper, test_remap_modelopt_nvfp4_per_expert_weights, test_remap_modelopt_nvfp4_vlm_prefix
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py
@@ -205,12 +205,19 @@ def _remap_moe_keys(self, weights: dict) -> dict:
+        ModelOpt NVFP4 checkpoints store already-split per-expert tensors:
+          experts.{id}.{gate,up,down}_proj.{field} → moe.experts.{id}.{w1,w3,w2}.{field}
+        expert_projection_map = {
+            "gate_proj": "w1",
+            "up_proj": "w3",
+            "down_proj": "w2",
diff -- tests/unittest/_torch/modeling/test_modeling_gemma4.py
@@ -23,38 +23,21 @@
-import pytest
-import transformers
-from packaging.version import Version
-# Gemma4 requires transformers>=5.5.0 (native Gemma4 config/model classes).
-# Use a module-level pytestmark.skipif (not pytest.importorskip) so collection
-# still picks up the test cases when the version gate fires. importorskip
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` modified +22/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +91/-30
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`, `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16074 - [TRTLLM-14138][perf] Add fused kernels for Gemma4 serving

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16074
- 状态/时间: merged / 2026-07-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tensorrt_llm/_torch/modules/gemma4/__init__.py`, `tensorrt_llm/_torch/modules/gemma4/fused_qkv.py`, `tests/unittest/_torch/modules/test_gemma4_fused_qkv_prep.py`；关联提交 `cf82c5d90220`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 13 个文件，+1702/-10，可读 patch 1870 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +325/-10 (335 lines); hunks: -15,7 +15,7; -29,6 +29,7; symbols: gelu_tanh, _Gemma4GeluQuantMLP, __init__, _fused_gelu_quant_enabled，涉及 `gelu_tanh, _Gemma4GeluQuantMLP, __init__`；`tensorrt_llm/_torch/modules/gemma4/fused_qkv.py` added +198/-0 (198 lines); hunks: -0,0 +1,198; symbols: _gemma4_qkv_norm_rope_quant_kernel, gemma4_fused_qkv_norm_rope_quant，涉及 `_gemma4_qkv_norm_rope_quant_kernel, gemma4_fused_qkv_norm_rope_quant`；`tests/unittest/_torch/modules/test_gemma4_fused_qkv_prep.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: _make_cos_sin, _ref_chain, norm, rope，涉及 `_make_cos_sin, _ref_chain, norm`；`tensorrt_llm/_torch/modules/fused_ops/__init__.py` added +10/-0 (10 lines); hunks: -0,0 +1,10。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +325/-10 (335 lines); hunks: -15,7 +15,7; -29,6 +29,7; symbols: gelu_tanh, _Gemma4GeluQuantMLP, __init__, _fused_gelu_quant_enabled
  - `tensorrt_llm/_torch/modules/gemma4/fused_qkv.py` added +198/-0 (198 lines); hunks: -0,0 +1,198; symbols: _gemma4_qkv_norm_rope_quant_kernel, gemma4_fused_qkv_norm_rope_quant
  - `tests/unittest/_torch/modules/test_gemma4_fused_qkv_prep.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: _make_cos_sin, _ref_chain, norm, rope
  - `tensorrt_llm/_torch/modules/fused_ops/__init__.py` added +10/-0 (10 lines); hunks: -0,0 +1,10
  - `tensorrt_llm/_torch/modules/gemma4/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4.py
@@ -15,7 +15,7 @@
-from typing import Dict, Optional, Tuple
+from typing import Dict, Optional, Tuple, Union
@@ -29,6 +29,7 @@
+from tensorrt_llm.logger import logger
@@ -43,10 +44,17 @@
+from ..modules.fused_ops.gelu_tanh_mul_fp4_quant import gelu_tanh_mul_fp4_quant
diff -- tensorrt_llm/_torch/modules/gemma4/fused_qkv.py
@@ -0,0 +1,198 @@
+# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tests/unittest/_torch/modules/test_gemma4_fused_qkv_prep.py
@@ -0,0 +1,157 @@
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +325/-10; `tensorrt_llm/_torch/modules/gemma4/fused_qkv.py` added +198/-0; `tensorrt_llm/_torch/modules/fused_ops/__init__.py` added +10/-0; `tensorrt_llm/_torch/modules/gemma4/__init__.py` added +2/-0
  - tests: `tests/unittest/_torch/modules/test_gemma4_fused_qkv_prep.py` added +157/-0
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modules/fused_ops/test_gelu_tanh_mul_fp4_quant.py`, `tests/unittest/_torch/modules/fused_ops/test_rmsnorm_fp4_quant.py`, `tests/unittest/_torch/modules/fused_ops/test_rmsnorm_residual_add.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16099 - [None][fix] Fix Gemma4 illegal memory access when max_seq_len is at most the sliding window size

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16099
- 状态/时间: merged / 2026-07-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_gemma4_e2e_dummy.py`；关联提交 `9115e4839128`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+139/-39，可读 patch 325 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_gemma4_e2e_dummy.py` modified +99/-27 (126 lines); hunks: -42,7 +42,19; -53,11 +65,6; symbols: _model_available, _make_dummy_config_dir, test_e2e_text_26b_dummy，涉及 `_model_available, _make_dummy_config_dir, test_e2e_text_26b_dummy`；`tensorrt_llm/_torch/attention_backend/flashinfer.py` modified +20/-9 (29 lines); hunks: -663,17 +663,28 @@ def _post_init_with_buffers(self, buffers) -> None:; symbols: _post_init_with_buffers，涉及 `_post_init_with_buffers`；`tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` modified +18/-3 (21 lines); hunks: -2796,6 +2796,12 @@ def free_resources(self, request: LlmRequest, pin_on_rele...; -2804,13 +2810,16 @@ def get_batch_cache_indices(; symbols: free_resources, get_layer_page_index_scale, get_batch_cache_indices, _get_batch_cache_indices_by_pool_id，涉及 `free_resources, get_layer_page_index_scale, get_batch_cache_indices`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_gemma4_e2e_dummy.py` modified +99/-27 (126 lines); hunks: -42,7 +42,19; -53,11 +65,6; symbols: _model_available, _make_dummy_config_dir, test_e2e_text_26b_dummy
  - `tensorrt_llm/_torch/attention_backend/flashinfer.py` modified +20/-9 (29 lines); hunks: -663,17 +663,28 @@ def _post_init_with_buffers(self, buffers) -> None:; symbols: _post_init_with_buffers
  - `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` modified +18/-3 (21 lines); hunks: -2796,6 +2796,12 @@ def free_resources(self, request: LlmRequest, pin_on_rele...; -2804,13 +2810,16 @@ def get_batch_cache_indices(; symbols: free_resources, get_layer_page_index_scale, get_batch_cache_indices, _get_batch_cache_indices_by_pool_id
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_gemma4_e2e_dummy.py
@@ -42,7 +42,19 @@
-_GEMMA4_MODELS = os.path.join(_LLM_MODELS_ROOT, "gemma4")
+# Canonical model root subdir is "gemma" (see tests/test_common/llm_data.py:
+# "google/gemma-4-E2B-it" -> "gemma/gemma-4-E2B-it"), not "gemma4".
+_GEMMA4_MODELS = os.path.join(_LLM_MODELS_ROOT, "gemma")
+# Imported after the module-level skip guard so that collecting this module on
+# a machine without LLM_MODELS_ROOT does not pull in the runtime import.
diff -- tensorrt_llm/_torch/attention_backend/flashinfer.py
@@ -663,17 +663,28 @@ def _post_init_with_buffers(self, buffers) -> None:
-            # Detect VSWA: check if the manager has multiple pools.
-            # Guard on layer_to_pool_mapping_dict which is V2-specific — V1
-            # managers also expose is_vswa but lack the per-pool infrastructure.
-            if (getattr(self.kv_cache_manager, 'is_vswa', False) and hasattr(
-                    self.kv_cache_manager, 'layer_to_pool_mapping_dict')):
-                mgr = self.kv_cache_manager
diff -- tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py
@@ -2796,6 +2796,12 @@ def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_gemma4_e2e_dummy.py` modified +99/-27
  - runtime: `tensorrt_llm/_torch/attention_backend/flashinfer.py` modified +20/-9; `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` modified +18/-3
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modeling/test_gemma4_e2e_dummy.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16509 - [None][perf] Various Gemma4 related perf fixes

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16509
- 状态/时间: merged / 2026-07-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4_vision.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；关联提交 `66f76b607d6c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+423/-44，可读 patch 571 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4_vision.py` modified +5/-5 (10 lines); hunks: -629,11 +629,11 @@ def _position_embeddings(; symbols: _position_embeddings, forward，涉及 `_position_embeddings, forward`；`tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +331/-0 (331 lines); hunks: -22,10 +22,13; -39,6 +42,12; symbols: TestGemma4CUDAGraph, _make_trtllm_gen_decode_case, _prepare_decode_page_counts, _expected_decode_block_table，涉及 `TestGemma4CUDAGraph, _make_trtllm_gen_decode_case, _prepare_decode_page_counts`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4_vision.py` modified +5/-5 (10 lines); hunks: -629,11 +629,11 @@ def _position_embeddings(; symbols: _position_embeddings, forward
  - `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +331/-0 (331 lines); hunks: -22,10 +22,13; -39,6 +42,12; symbols: TestGemma4CUDAGraph, _make_trtllm_gen_decode_case, _prepare_decode_page_counts, _expected_decode_block_table
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4_vision.py
@@ -629,11 +629,11 @@ def _position_embeddings(
-        one_hot = F.one_hot(clamped, num_classes=self.position_embedding_size)
-        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
-        position_embeddings = one_hot @ self.position_embedding_table
-        position_embeddings = position_embeddings.sum(dim=1)
-        return torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)
+        position_embeddings = (
diff -- tests/unittest/_torch/modeling/test_modeling_gemma4.py
@@ -22,10 +22,13 @@
+from typing import TYPE_CHECKING
+from tensorrt_llm._torch.attention_backend import FlashInferAttention, FlashInferAttentionMetadata
+from tensorrt_llm._torch.metadata import KVCacheParams
@@ -39,6 +42,12 @@
+if TYPE_CHECKING:
+    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4_vision.py` modified +5/-5
  - tests: `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +331/-0
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16797 - [NVBUG-6379624][fix] Enable W4A8 checkpoint loading for Gemma4 K=V layers

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16797
- 状态/时间: merged / 2026-07-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；关联提交 `2e2ed4ed1c4d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+102/-18，可读 patch 164 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` modified +16/-18 (34 lines); hunks: -23,6 +23,10; -273,9 +277,6 @@ def _remap_moe_keys(self, weights: dict) -> dict:; symbols: _remap_moe_keys, _handle_buffers_and_kvdup, get_layer，涉及 `_remap_moe_keys, _handle_buffers_and_kvdup, get_layer`；`tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +86/-0 (86 lines); hunks: -22,6 +22,7; -370,6 +371,91 @@ def test_num_kv_heads_per_layer_type(self):; symbols: test_num_kv_heads_per_layer_type, TestGemma4HfWeightMapper, test_duplicate_full_attention_kv_projection_tensors, test_remap_modelopt_nvfp4_per_expert_weights，涉及 `test_num_kv_heads_per_layer_type, TestGemma4HfWeightMapper, test_duplicate_full_attention_kv_projection_tensors`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` modified +16/-18 (34 lines); hunks: -23,6 +23,10; -273,9 +277,6 @@ def _remap_moe_keys(self, weights: dict) -> dict:; symbols: _remap_moe_keys, _handle_buffers_and_kvdup, get_layer
  - `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +86/-0 (86 lines); hunks: -22,6 +22,7; -370,6 +371,91 @@ def test_num_kv_heads_per_layer_type(self):; symbols: test_num_kv_heads_per_layer_type, TestGemma4HfWeightMapper, test_duplicate_full_attention_kv_projection_tensors, test_remap_modelopt_nvfp4_per_expert_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py
@@ -23,6 +23,10 @@
+_LAYER_SCALAR_KEY_RE = re.compile(r"^(?:language_model\.)?model\.layers\.(\d+)\.layer_scalar$")
+_K_PROJ_KEY_RE = re.compile(
+    r"^((?:language_model\.)?model\.layers\.(\d+)\.self_attn\.)k_proj(\..+)$"
+)
@@ -273,9 +277,6 @@ def _remap_moe_keys(self, weights: dict) -> dict:
-        # Determine the layer scalar key pattern and accessor based on
diff -- tests/unittest/_torch/modeling/test_modeling_gemma4.py
@@ -22,6 +22,7 @@
+from types import SimpleNamespace
@@ -370,6 +371,91 @@ def test_num_kv_heads_per_layer_type(self):
+    def test_duplicate_full_attention_kv_projection_tensors(self):
+        """Full K=V layers should duplicate all missing k_proj tensors to v_proj."""
+        fields = ("weight", "weight_scale", "input_scale", "weight_scale_2", "pre_quant_scale")
+        for is_vlm in (False, True):
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` modified +16/-18
  - tests: `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +86/-0
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16662 - [None][feat] Enable MM encoder cache on Qwen3.x and Gemma4 VLMs

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16662
- 状态/时间: merged / 2026-07-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4_unified.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py`, `tests/unittest/_torch/modeling/test_gemma4_multimodal.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4_unified.py`；关联提交 `a2df74eef602`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 15 个文件，+826/-744，可读 patch 2024 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +248/-303 (551 lines); hunks: -47,18 +47,13; -550,6 +545,250 @@ def call_with_text_prompt(; symbols: call_with_text_prompt, Gemma4MultimodalModelBase, get_model_defaults, _check_and_adjust_experts_implementation，涉及 `call_with_text_prompt, Gemma4MultimodalModelBase, get_model_defaults`；`tensorrt_llm/_torch/models/modeling_gemma4_unified.py` modified +16/-190 (206 lines); hunks: -33,15 +33,14; -55,13 +54,12; symbols: __init__, Gemma4UnifiedForConditionalGeneration, _get_audio_features，涉及 `__init__, Gemma4UnifiedForConditionalGeneration, _get_audio_features`；`tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +85/-15 (100 lines); hunks: -62,11 +62,17; -125,6 +131,59; symbols: _Gemma4EncoderCacheHarness, __init__, embedding_dim, embedding_dtype，涉及 `_Gemma4EncoderCacheHarness, __init__, embedding_dim`；`tests/unittest/_torch/modeling/test_modeling_gemma4_unified.py` modified +10/-0 (10 lines); hunks: -26,6 +26,7; -43,6 +44,7; symbols: test_get_model_architecture_resolves_wrapper, test_wrapper_rejects_missing_image_token_id, test_weight_mapper_registered，涉及 `test_get_model_architecture_resolves_wrapper, test_wrapper_rejects_missing_image_token_id, test_weight_mapper_registered`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +248/-303 (551 lines); hunks: -47,18 +47,13; -550,6 +545,250 @@ def call_with_text_prompt(; symbols: call_with_text_prompt, Gemma4MultimodalModelBase, get_model_defaults, _check_and_adjust_experts_implementation
  - `tensorrt_llm/_torch/models/modeling_gemma4_unified.py` modified +16/-190 (206 lines); hunks: -33,15 +33,14; -55,13 +54,12; symbols: __init__, Gemma4UnifiedForConditionalGeneration, _get_audio_features
  - `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +85/-15 (100 lines); hunks: -62,11 +62,17; -125,6 +131,59; symbols: _Gemma4EncoderCacheHarness, __init__, embedding_dim, embedding_dtype
  - `tests/unittest/_torch/modeling/test_modeling_gemma4_unified.py` modified +10/-0 (10 lines); hunks: -26,6 +26,7; -43,6 +44,7; symbols: test_get_model_architecture_resolves_wrapper, test_wrapper_rejects_missing_image_token_id, test_weight_mapper_registered
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4mm.py
@@ -47,18 +47,13 @@
-from ..attention_backend import AttentionMetadata
+from ..modules.embedding import Embedding
-from .modeling_multimodal_utils import (
-    _MULTIMODAL_ENV_NAME,
-    _is_mm_disagg,
-    find_input_mm_embeds,
diff -- tensorrt_llm/_torch/models/modeling_gemma4_unified.py
@@ -33,15 +33,14 @@
-This module reuses the existing Gemma 4 multimodal wrapper
-(:class:`Gemma4ForConditionalGeneration`) for all engine plumbing
+This module inherits the shared Gemma 4 multimodal wrapper base
+(:class:`Gemma4MultimodalModelBase`) for all engine plumbing
-keys match). It overrides `__init__` / `forward` / `_get_image_features` /
-`_get_audio_features` / `load_weights` to drop the encoder towers and use the
diff -- tests/unittest/_torch/modeling/test_gemma4_multimodal.py
@@ -62,11 +62,17 @@
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +248/-303; `tensorrt_llm/_torch/models/modeling_gemma4_unified.py` modified +16/-190
  - tests: `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +85/-15; `tests/unittest/_torch/modeling/test_modeling_gemma4_unified.py` modified +10/-0
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/executor/test_pytorch_model_engine.py`, `tests/unittest/_torch/modeling/test_gemma4_multimodal.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4_unified.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15833 - [None][feat] Add Gemma4 MTP assistant support

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15833
- 状态/时间: merged / 2026-08-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/configs/gemma4.py`, `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tensorrt_llm/_torch/models/modeling_gemma4_unified.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py` 等 6 个文件；关联提交 `048ae4acde91`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 21 个文件，+1285/-128，可读 patch 2042 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +268/-12 (280 lines); hunks: -14,8 +14,9; -40,6 +41,7; symbols: __init__, forward, Gemma4ForCausalLM，涉及 `__init__, forward, Gemma4ForCausalLM`；`tensorrt_llm/_torch/configs/gemma4.py` renamed +82/-14 (96 lines); hunks: -12,20 +12,88; -39,7 +107,7 @@ class Gemma4UnifiedTextConfig(Gemma4TextConfig):; symbols: Gemma4AssistantConfig, once, __init__, hidden_size，涉及 `Gemma4AssistantConfig, once, __init__`；`tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +16/-0 (16 lines); hunks: -732,6 +732,17 @@ def post_config(self):; -743,6 +754,8 @@ def get_language_model_extra_forward_kwargs(; symbols: post_config, draft_config, draft_model, load_draft_weights，涉及 `post_config, draft_config, draft_model`；`tensorrt_llm/_torch/models/modeling_gemma4_unified.py` modified +1/-1 (2 lines); hunks: -43,7 +43,7。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +268/-12 (280 lines); hunks: -14,8 +14,9; -40,6 +41,7; symbols: __init__, forward, Gemma4ForCausalLM
  - `tensorrt_llm/_torch/configs/gemma4.py` renamed +82/-14 (96 lines); hunks: -12,20 +12,88; -39,7 +107,7 @@ class Gemma4UnifiedTextConfig(Gemma4TextConfig):; symbols: Gemma4AssistantConfig, once, __init__, hidden_size
  - `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +16/-0 (16 lines); hunks: -732,6 +732,17 @@ def post_config(self):; -743,6 +754,8 @@ def get_language_model_extra_forward_kwargs(; symbols: post_config, draft_config, draft_model, load_draft_weights
  - `tensorrt_llm/_torch/models/modeling_gemma4_unified.py` modified +1/-1 (2 lines); hunks: -43,7 +43,7
  - `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` modified +1/-0 (1 lines); hunks: -32,6 +32,7; symbols: Gemma4HfWeightMapper, _is_vlm
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4.py
@@ -14,8 +14,9 @@
+import dataclasses
-from typing import Dict, Optional, Tuple, Union
+from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
@@ -40,6 +41,7 @@
+from ..distributed import AllReduce
@@ -54,9 +56,14 @@
diff -- tensorrt_llm/_torch/configs/gemma4.py
@@ -12,20 +12,88 @@
-"""Config classes for Gemma 4 12B Unified (encoder-free multimodal).
+"""Compatibility configs for Gemma 4 assistant and Unified checkpoints."""
-Registered with the transformers CONFIG_MAPPING (see `_torch/configs/__init__.py`)
-so `AutoConfig.from_pretrained` can parse a Gemma 4 12B checkpoint whenever the
-installed transformers does not ship the `gemma4_unified` model_types natively.
+from transformers import Gemma4TextConfig, PreTrainedConfig
diff -- tensorrt_llm/_torch/models/modeling_gemma4mm.py
@@ -732,6 +732,17 @@ def post_config(self):
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +268/-12; `tensorrt_llm/_torch/configs/gemma4.py` renamed +82/-14; `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +16/-0; `tensorrt_llm/_torch/models/modeling_gemma4_unified.py` modified +1/-1; `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` modified +1/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +166/-8
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/attention/test_flashinfer_attention.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`, `tests/unittest/_torch/speculative/hw_agnostic/test_mtp.py`, `tests/unittest/_torch/speculative/hw_agnostic/test_sa.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17231 - [https://nvbugs/6550127][fix] Support Gemma4 multimodal cache partial hits

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/17231
- 状态/时间: merged / 2026-08-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4mm.py`, `tests/unittest/_torch/modeling/test_gemma4_multimodal.py`；关联提交 `05d7947480d3`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+321/-10，可读 patch 412 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +202/-8 (210 lines); hunks: -61,6 +61,7; -131,7 +132,7; symbols: _Gemma4EncoderCacheHarness, __init__, embedding_dim, _get_image_features，涉及 `_Gemma4EncoderCacheHarness, __init__, embedding_dim`；`tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +119/-1 (120 lines); hunks: -23,6 +23,7; -32,6 +33,7; symbols: multimodal_data_device_paths, partition_encoder_cache, build_multimodal_encoder_input, encode_multimodal_inputs，涉及 `multimodal_data_device_paths, partition_encoder_cache, build_multimodal_encoder_input`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +202/-8 (210 lines); hunks: -61,6 +61,7; -131,7 +132,7; symbols: _Gemma4EncoderCacheHarness, __init__, embedding_dim, _get_image_features
  - `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +119/-1 (120 lines); hunks: -23,6 +23,7; -32,6 +33,7; symbols: multimodal_data_device_paths, partition_encoder_cache, build_multimodal_encoder_input, encode_multimodal_inputs
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_gemma4_multimodal.py
@@ -61,6 +61,7 @@
+    Gemma4MultimodalModelBase,
@@ -131,7 +132,7 @@
-class _Gemma4EncoderCacheHarness(MultimodalModelMixin):
+class _Gemma4EncoderCacheHarness(Gemma4MultimodalModelBase):
@@ -144,6 +145,7 @@ def __init__(self, embedding_dim: int = 12) -> None:
+        self.embed_audio = object()
diff -- tensorrt_llm/_torch/models/modeling_gemma4mm.py
@@ -23,6 +23,7 @@
+from collections.abc import Sequence
@@ -32,6 +33,7 @@
+from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
@@ -52,7 +54,11 @@
-from .modeling_multimodal_mixin import MultimodalModelMixin, PreparedLlmInputs
+from .modeling_multimodal_mixin import (
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +202/-8
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +119/-1
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/waives.txt`, `tests/unittest/_torch/modeling/test_gemma4_multimodal.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17557 - [https://nvbugs/6566891][fix] Use FlashInfer FA2 for Gemma4 on SM120 and SM121

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/17557
- 状态/时间: merged / 2026-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；关联提交 `e91b9f8717b2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+32/-13，可读 patch 93 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +5/-8 (13 lines); hunks: -29,6 +29,7; -328,14 +329,10 @@ def __init__(; symbols: __init__，涉及 `__init__`；`tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +27/-5 (32 lines); hunks: -45,6 +45,7; -2410,8 +2411,9 @@ def test_attn_backend_dispatches_to_flashinfer(self):; symbols: test_attn_backend_dispatches_to_flashinfer, test_all_layers_use_trtllm_gen, test_all_layers_use_trtllm_gen_on_sm100f, test_non_sm100f_layers_use_fa2，涉及 `test_attn_backend_dispatches_to_flashinfer, test_all_layers_use_trtllm_gen, test_all_layers_use_trtllm_gen_on_sm100f`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +5/-8 (13 lines); hunks: -29,6 +29,7; -328,14 +329,10 @@ def __init__(; symbols: __init__
  - `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +27/-5 (32 lines); hunks: -45,6 +45,7; -2410,8 +2411,9 @@ def test_attn_backend_dispatches_to_flashinfer(self):; symbols: test_attn_backend_dispatches_to_flashinfer, test_all_layers_use_trtllm_gen, test_all_layers_use_trtllm_gen_on_sm100f, test_non_sm100f_layers_use_fa2
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4.py
@@ -29,6 +29,7 @@
+from tensorrt_llm._utils import is_sm_100f
@@ -328,14 +329,10 @@ def __init__(
-        # Use trtllm-gen for ALL layers.  trtllm-gen has pre-compiled cubins
-        # for both H256+SWA and H512 across all supported dtypes.
-        # For FP8 KV cache (NVFP4), Q is also cast to FP8 in the FlashInfer
-        # backend so that QkvE4m3OBfloat16 context cubins can be used
diff -- tests/unittest/_torch/modeling/test_modeling_gemma4.py
@@ -45,6 +45,7 @@
+from tensorrt_llm._utils import is_sm_100f
@@ -2410,8 +2411,9 @@ def test_attn_backend_dispatches_to_flashinfer(self):
-    def test_all_layers_use_trtllm_gen(self):
-        """All Gemma4 layers use trtllm-gen backend uniformly.
+    @unittest.mock.patch("tensorrt_llm._torch.models.modeling_gemma4.is_sm_100f", return_value=True)
+    def test_all_layers_use_trtllm_gen_on_sm100f(self, _mock_is_sm_100f):
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +5/-8
  - tests: `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +27/-5
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17396 - [None][feat] Enable KVCacheManagerV2 by default for Gemma3 and Gemma4

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/17396
- 状态/时间: merged / 2026-08-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py`；关联提交 `92c0b91ad7d0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+165/-6，可读 patch 286 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +23/-2 (25 lines); hunks: -25,7 +25,7; -62,6 +62,9; symbols: Gemma4MultimodalModelBase, get_model_defaults, get_preferred_kv_cache_manager_version, get_preferred_transceiver_runtime，涉及 `Gemma4MultimodalModelBase, get_model_defaults, get_preferred_kv_cache_manager_version`；`tensorrt_llm/_torch/models/modeling_gemma4.py` modified +21/-2 (23 lines); hunks: -16,7 +16,7; -63,6 +63,8; symbols: __init__, get_model_defaults, get_preferred_kv_cache_manager_version，涉及 `__init__, get_model_defaults, get_preferred_kv_cache_manager_version`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +23/-2 (25 lines); hunks: -25,7 +25,7; -62,6 +62,9; symbols: Gemma4MultimodalModelBase, get_model_defaults, get_preferred_kv_cache_manager_version, get_preferred_transceiver_runtime
  - `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +21/-2 (23 lines); hunks: -16,7 +16,7; -63,6 +63,8; symbols: __init__, get_model_defaults, get_preferred_kv_cache_manager_version
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4mm.py
@@ -25,7 +25,7 @@
-from typing import Dict, List, Optional, Tuple
+from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple
@@ -62,6 +62,9 @@
+if TYPE_CHECKING:
+    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
@@ -557,12 +560,30 @@ class Gemma4MultimodalModelBase(MultimodalModelMixin, PreTrainedModel):
diff -- tensorrt_llm/_torch/models/modeling_gemma4.py
@@ -16,7 +16,7 @@
-from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
+from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, Union
@@ -63,6 +63,8 @@
+    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
@@ -1247,7 +1249,7 @@ def __init__(
-    def get_model_defaults(cls, llm_args) -> dict:
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +23/-2; `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +21/-2
- 验证与风险: diff 自带测试面 `tests/unittest/llmapi/test_llm_args.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17837 - [None][fix] Restore Gemma4 shared-KV draft loading

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/17837
- 状态/时间: merged / 2026-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tensorrt_llm/_torch/models/modeling_gemma4mm.py`；关联提交 `e9b0b08a260e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+204/-76，可读 patch 556 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +2/-0 (2 lines); hunks: -1225,6 +1225,8 @@ def forward(; symbols: forward, Gemma4ForCausalLM, __init__，涉及 `forward, Gemma4ForCausalLM, __init__`；`tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +2/-0 (2 lines); hunks: -963,6 +963,8 @@ class Gemma4ForConditionalGeneration(Gemma4MultimodalModelBa...; symbols: Gemma4ForConditionalGeneration, __init__，涉及 `Gemma4ForConditionalGeneration, __init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +2/-0 (2 lines); hunks: -1225,6 +1225,8 @@ def forward(; symbols: forward, Gemma4ForCausalLM, __init__
  - `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +2/-0 (2 lines); hunks: -963,6 +963,8 @@ class Gemma4ForConditionalGeneration(Gemma4MultimodalModelBa...; symbols: Gemma4ForConditionalGeneration, __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4.py
@@ -1225,6 +1225,8 @@ def forward(
+    build_mtp_draft_model_from_config = True
diff -- tensorrt_llm/_torch/models/modeling_gemma4mm.py
@@ -963,6 +963,8 @@ class Gemma4ForConditionalGeneration(Gemma4MultimodalModelBase):
+    build_mtp_draft_model_from_config = True
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +2/-0; `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +2/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`, `tests/unittest/_torch/speculative/hw_agnostic/test_mtp.py`, `tests/unittest/_torch/speculative/hw_agnostic/test_mtp_separate_checkpoint.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18002 - [None][fix] Stabilize Gemma4 FA2 CUDA Graph decode on Hopper

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/18002
- 状态/时间: merged / 2026-08-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；关联提交 `1b19a8197f64`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+268/-100，可读 patch 645 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +17/-8 (25 lines); hunks: -1339,11 +1339,10 @@ def get_context_mask(; -1360,9 +1359,19 @@ def get_context_mask(; symbols: get_context_mask，涉及 `get_context_mask`；`tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +176/-75 (251 lines); hunks: -48,6 +48,7; -896,6 +897,17 @@ def test_assistant_uses_target_kv_sources(self):; symbols: test_assistant_uses_target_kv_sources, _build_gemma4_kv_cache_manager, test_vswa_pool_cache_not_aliased, test_vswa_no_eviction_with_long_sequence，涉及 `test_assistant_uses_target_kv_sources, _build_gemma4_kv_cache_manager, test_vswa_pool_cache_not_aliased`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +17/-8 (25 lines); hunks: -1339,11 +1339,10 @@ def get_context_mask(; -1360,9 +1359,19 @@ def get_context_mask(; symbols: get_context_mask
  - `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +176/-75 (251 lines); hunks: -48,6 +48,7; -896,6 +897,17 @@ def test_assistant_uses_target_kv_sources(self):; symbols: test_assistant_uses_target_kv_sources, _build_gemma4_kv_cache_manager, test_vswa_pool_cache_not_aliased, test_vswa_no_eviction_with_long_sequence
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4.py
@@ -1339,11 +1339,10 @@ def get_context_mask(
-        - The first `prefix_len` columns (cached/paged history) are True for
-          all rows. SWA window enforcement is delegated to the kernel's
-          window_left clip. Bidirectional MM across the prefix/extend
-          boundary is NOT supported here; callers must ensure chunk
-          boundaries do not split a multimodal block.
+        - The first `prefix_len` columns (cached/paged history) apply the
diff -- tests/unittest/_torch/modeling/test_modeling_gemma4.py
@@ -48,6 +48,7 @@
+from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX
@@ -896,6 +897,17 @@ def test_assistant_uses_target_kv_sources(self):
+# 12B-real-dims: GQA=2 sliding (16/8), GQA=16 full K=V (16/1),
+# hd=256/512.
+GEMMA4_12B_REAL_DIMS_CONFIG = {
+    **GEMMA4_E2B_REAL_DIMS_CONFIG,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4.py` modified +17/-8
  - tests: `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +176/-75
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_h100.yml`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18274 - [https://nvbugs/6663281][fix] Fix Gemma4 video token counting

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/18274
- 状态/时间: merged / 2026-08-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4mm.py`, `tests/unittest/_torch/modeling/test_gemma4_multimodal.py`；关联提交 `7b60a569a989`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+85/-12，可读 patch 173 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +47/-3 (50 lines); hunks: -27,6 +27,7; -79,6 +80,9; symbols: RMSNormNoScale, _normalize_audio_inputs, get_num_tokens_per_audio, get_num_tokens_per_video，涉及 `RMSNormNoScale, _normalize_audio_inputs, get_num_tokens_per_audio`；`tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +38/-9 (47 lines); hunks: -38,6 +38,7; -1422,19 +1423,13 @@ def test_video_processing_returns_pixel_values(self):; symbols: test_video_processing_returns_pixel_values, test_video_token_prediction_for_non_square_pil_frames, _video_token_id，涉及 `test_video_processing_returns_pixel_values, test_video_token_prediction_for_non_square_pil_frames, _video_token_id`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +47/-3 (50 lines); hunks: -27,6 +27,7; -79,6 +80,9; symbols: RMSNormNoScale, _normalize_audio_inputs, get_num_tokens_per_audio, get_num_tokens_per_video
  - `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +38/-9 (47 lines); hunks: -38,6 +38,7; -1422,19 +1423,13 @@ def test_video_processing_returns_pixel_values(self):; symbols: test_video_processing_returns_pixel_values, test_video_token_prediction_for_non_square_pil_frames, _video_token_id
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4mm.py
@@ -27,6 +27,7 @@
+import numpy as np
@@ -79,6 +80,9 @@
+from transformers.models.gemma4.image_processing_gemma4 import (  # noqa: E402
+    get_aspect_ratio_preserving_size,
+)
@@ -152,8 +156,6 @@ def _normalize_audio_inputs(audios, target_sr: int = 16000):
diff -- tests/unittest/_torch/modeling/test_gemma4_multimodal.py
@@ -38,6 +38,7 @@
+from unittest import mock
@@ -1422,19 +1423,13 @@ def test_video_processing_returns_pixel_values(self):
-        import numpy as np
-        from PIL import Image
-        frames = [
-            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4mm.py` modified +47/-3
  - tests: `tests/unittest/_torch/modeling/test_gemma4_multimodal.py` modified +38/-9
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_gemma4_multimodal.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19571 - [None][models] Avoid GEMM in Gemma4 vision RoPE

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19571
- 状态/时间: merged / 2026-09-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_gemma4_vision.py`, `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；关联提交 `b91158667191`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+39/-4，可读 patch 64 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_gemma4_vision.py` modified +3/-3 (6 lines); hunks: -199,9 +199,9 @@ def forward(; symbols: forward，涉及 `forward`；`tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +36/-1 (37 lines); hunks: -27,7 +27,8; -4268,6 +4269,40 @@ def test_vision_attention_uses_native_rmsnorm_adapter(self):; symbols: test_vision_attention_uses_native_rmsnorm_adapter, _RejectMatrixMultiply, __torch_dispatch__, TestGemma4VisionRotaryEmbedding，涉及 `test_vision_attention_uses_native_rmsnorm_adapter, _RejectMatrixMultiply, __torch_dispatch__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_gemma4_vision.py` modified +3/-3 (6 lines); hunks: -199,9 +199,9 @@ def forward(; symbols: forward
  - `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +36/-1 (37 lines); hunks: -27,7 +27,8; -4268,6 +4269,40 @@ def test_vision_attention_uses_native_rmsnorm_adapter(self):; symbols: test_vision_attention_uses_native_rmsnorm_adapter, _RejectMatrixMultiply, __torch_dispatch__, TestGemma4VisionRotaryEmbedding
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_gemma4_vision.py
@@ -199,9 +199,9 @@ def forward(
-                freqs = (inv_freq_expanded.float() @ dim_position_ids_expanded.float()).transpose(
-                    1, 2
-                )
+                # This is an outer product. Keep it out of GEMM so TF32 matmul settings cannot
+                # reduce rotary phase precision.
+                freqs = (inv_freq_expanded * dim_position_ids_expanded).transpose(1, 2)
diff -- tests/unittest/_torch/modeling/test_modeling_gemma4.py
@@ -27,7 +27,8 @@
-from transformers import AutoConfig, Gemma4Config, Gemma4TextConfig
+from torch.utils._python_dispatch import TorchDispatchMode
+from transformers import AutoConfig, Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig
@@ -4268,6 +4269,40 @@ def test_vision_attention_uses_native_rmsnorm_adapter(self):
+class _RejectMatrixMultiply(TorchDispatchMode):
+    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_gemma4_vision.py` modified +3/-3
  - tests: `tests/unittest/_torch/modeling/test_modeling_gemma4.py` modified +36/-1
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_gemma4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
