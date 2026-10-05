# TensorRT-LLM Qwen3.5 模型 PR 优化历史

## 2026-08-23 源码 head 刷新

已复核 TensorRT-LLM 上游 main：
`NVIDIA/TensorRT-LLM@da38c1d2e0dffd073b7dfb6d69e15ee7b45d84a9`。
已读完上一记录 head `1b4ffc0291d75a21ad20118e8f44de6e3831f786`
之后的 7-commit 完整增量；其中没有新的 Qwen3.5 专属实现提交，最新 PR #16677
仅涉及 VisualGen/Wan 的 Attention2D + TP，以下 4 个此前提升的 runtime PR
继续作为当前模型证据。

结果：Qwen3.5 的 MoE 与 Dense VLM 路径先后落地，随后加入注意力预处理融合以及 AllReduce + Gemma RMSNorm 融合。只改 waiver 的测试 PR 不混入 runtime 证据集。

| 合并日期 | PR | Runtime 信号 |
| --- | --- | --- |
| 2026-07-04 | [#14599](https://github.com/NVIDIA/TensorRT-LLM/pull/14599) | Qwen3.5 MoE VLM + MTP |
| 2026-07-07 | [#15249](https://github.com/NVIDIA/TensorRT-LLM/pull/15249) | Qwen3.5 Dense VLM |
| 2026-07-21 | [#16469](https://github.com/NVIDIA/TensorRT-LLM/pull/16469) | 融合 QK norm + RoPE + gate |
| 2026-07-24 | [#15194](https://github.com/NVIDIA/TensorRT-LLM/pull/15194) | 融合 AllReduce + Gemma RMSNorm |

## 2026-06-27 PR 补漏复核

本文的逐 PR diff 审计卡基于 TensorRT-LLM 上游
`HEAD@4164b932c6c8a14d1be85d0fd62e44b7d0171980` 生成；根目录 TensorRT-LLM
history index 已跟踪 2026-06-27 runtime refresh
`aaffa2f9fef3025e0f698d978385a73460344e0b`。本文覆盖 Qwen3.5 相关 merged
PR，并采用 SGLang 风格的模型实现覆盖、PR 时间线和逐 PR diff 审计卡。

本轮筛选规则：标题/文件命中 `Qwen3.5`、`Qwen3_5`、`qwen3_5`、`AutoDeploy`、`NVFP4`、`FP8`、`DFlash`、`reasoning_parser`、`EPLB`、`MoE backend`、`model_registry` 的 merged PR；过滤纯重排和不触碰模型/loader/test lane 的基础设施 PR。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/source/deployment-guide/deployment-guide-for-qwen3.8-qwen3.5-on-trtllm.md` | 无直接 PR 号提交 |
| `examples/configs/curated/qwen3.5.yaml` | [#15111](https://github.com/NVIDIA/TensorRT-LLM/pull/15111) |
| `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` | [#12242](https://github.com/NVIDIA/TensorRT-LLM/pull/12242), [#13090](https://github.com/NVIDIA/TensorRT-LLM/pull/13090), [#13716](https://github.com/NVIDIA/TensorRT-LLM/pull/13716), [#14164](https://github.com/NVIDIA/TensorRT-LLM/pull/14164), [#14465](https://github.com/NVIDIA/TensorRT-LLM/pull/14465), [#14599](https://github.com/NVIDIA/TensorRT-LLM/pull/14599), [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067), [#15249](https://github.com/NVIDIA/TensorRT-LLM/pull/15249), [#16642](https://github.com/NVIDIA/TensorRT-LLM/pull/16642), [#16936](https://github.com/NVIDIA/TensorRT-LLM/pull/16936), [#17433](https://github.com/NVIDIA/TensorRT-LLM/pull/17433), [#19519](https://github.com/NVIDIA/TensorRT-LLM/pull/19519) |
| `tensorrt_llm/_torch/models/modeling_qwen3_5.py` | [#12242](https://github.com/NVIDIA/TensorRT-LLM/pull/12242), [#12646](https://github.com/NVIDIA/TensorRT-LLM/pull/12646), [#14164](https://github.com/NVIDIA/TensorRT-LLM/pull/14164), [#14465](https://github.com/NVIDIA/TensorRT-LLM/pull/14465), [#14599](https://github.com/NVIDIA/TensorRT-LLM/pull/14599), [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067), [#15249](https://github.com/NVIDIA/TensorRT-LLM/pull/15249), [#16353](https://github.com/NVIDIA/TensorRT-LLM/pull/16353), [#16642](https://github.com/NVIDIA/TensorRT-LLM/pull/16642), [#17293](https://github.com/NVIDIA/TensorRT-LLM/pull/17293), [#17700](https://github.com/NVIDIA/TensorRT-LLM/pull/17700) |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp1_gentp1_qwen3_5_4b_fp8_tllm.yaml` | 无直接 PR 号提交 |
| `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml` | [#15650](https://github.com/NVIDIA/TensorRT-LLM/pull/15650) |
| `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py` | [#15249](https://github.com/NVIDIA/TensorRT-LLM/pull/15249), [#16065](https://github.com/NVIDIA/TensorRT-LLM/pull/16065), [#16264](https://github.com/NVIDIA/TensorRT-LLM/pull/16264) |
| `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` | [#14164](https://github.com/NVIDIA/TensorRT-LLM/pull/14164), [#14465](https://github.com/NVIDIA/TensorRT-LLM/pull/14465), [#14599](https://github.com/NVIDIA/TensorRT-LLM/pull/14599), [#16065](https://github.com/NVIDIA/TensorRT-LLM/pull/16065), [#17293](https://github.com/NVIDIA/TensorRT-LLM/pull/17293), [#17700](https://github.com/NVIDIA/TensorRT-LLM/pull/17700) |
| `tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py` | [#16642](https://github.com/NVIDIA/TensorRT-LLM/pull/16642) |
| `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` | [#17433](https://github.com/NVIDIA/TensorRT-LLM/pull/17433), [#19519](https://github.com/NVIDIA/TensorRT-LLM/pull/19519) |

## PR 覆盖总览

- git 追溯 PR 数: 20
- 原文档显式引用补充 PR 数: 13
- 当前文档总 PR 数: 33
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-02-26 | [#11728](https://github.com/NVIDIA/TensorRT-LLM/pull/11728) | merged | Added Qwen3.5 Cookbook | AutoDeploy cookbook notebook |
| 2026-03-20 | [#12242](https://github.com/NVIDIA/TensorRT-LLM/pull/12242) | merged | [None][feat] Initial Qwen3.5 text model support for PyT backend (BF16/FP8) | `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_5.py` |
| 2026-03-24 | [#12302](https://github.com/NVIDIA/TensorRT-LLM/pull/12302) | merged | Add Qwen 3.5 supporting (NVFP4) | model wrapper, weight mapper, tests |
| 2026-03-25 | [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114) | merged | Qwen 3.5 fix 3d position ID handling | AutoDeploy Qwen3.5 MoE, mRoPE cache, registry configs |
| 2026-04-30 | [#13090](https://github.com/NVIDIA/TensorRT-LLM/pull/13090) | merged | Qwen3.5 dense weight loading | weight mapper, dense tests |
| 2026-05-04 | [#13716](https://github.com/NVIDIA/TensorRT-LLM/pull/13716) | merged | Fix Qwen3.5 NVFP4 weight loading by preserving weight_scales | HF mapper |
| 2026-05-12 | [#13782](https://github.com/NVIDIA/TensorRT-LLM/pull/13782) | merged | Qwen3.5 DFlash | speculative/DFlash runtime |
| 2026-05-16 | [#13996](https://github.com/NVIDIA/TensorRT-LLM/pull/13996) | merged | Perf optimizations for DFlash | DFlash model engine and speculative code |
| 2026-05-21 | [#12646](https://github.com/NVIDIA/TensorRT-LLM/pull/12646) | merged | [TRTLLM-11547][feat] Add Qwen3.5 MTP support. | `tensorrt_llm/_torch/models/modeling_qwen3_5.py` |
| 2026-05-21 | [#14164](https://github.com/NVIDIA/TensorRT-LLM/pull/14164) | merged | [TRTLLM-12500][feat] Add support for Qwen3.5 VL MoE - REVERTED by #14599 | `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` |
| 2026-05-23 | [#14465](https://github.com/NVIDIA/TensorRT-LLM/pull/14465) | merged | [None][feat] Revert Add support for Qwen3.5 VL MoE (#14164) | `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` |
| 2026-05-29 | [#14659](https://github.com/NVIDIA/TensorRT-LLM/pull/14659) | merged | Add a reasoning parser for qwen3_5 | `reasoning_parser.py` |
| 2026-06-02 | [#14667](https://github.com/NVIDIA/TensorRT-LLM/pull/14667) | merged | AutoDeploy: Qwen3.5 400B NVFP4 accuracy regression fix | shared expert sharding, SwiGLU fusion |
| 2026-06-05 | [#15001](https://github.com/NVIDIA/TensorRT-LLM/pull/15001) | merged | Uncomment Qwen3.5 from model registry | `models.yaml` |
| 2026-06-09 | [#15081](https://github.com/NVIDIA/TensorRT-LLM/pull/15081) | merged | Select CUTLASS MoE backend on non-Blackwell SMs | accuracy test backend selection |
| 2026-06-09 | [#15111](https://github.com/NVIDIA/TensorRT-LLM/pull/15111) | merged | [TRTLLM-11548][doc] Add Qwen3.5 deployment guide doc | `examples/configs/curated/qwen3.5.yaml` |
| 2026-06-11 | [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067) | merged | Generalize FP8 checkpoint loading for Qwen3.5 | weight mapper, modeling |
| 2026-06-13 | [#15185](https://github.com/NVIDIA/TensorRT-LLM/pull/15185) | merged | Qwen3.5 whitelist sharding and lm_head sharding | AutoDeploy sharding IR/tests |
| 2026-06-26 | [#15543](https://github.com/NVIDIA/TensorRT-LLM/pull/15543) | merged | Add EPLB support for Qwen3.5 | MoE load balancer and B200/GB200 tests |
| 2026-06-26 | [#15650](https://github.com/NVIDIA/TensorRT-LLM/pull/15650) | merged | [None][test] Add Qwen3.5-397B-A17B-NVFP4 B200 aggregated perf-sanity tests | `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml` |
| 2026-07-04 | [#14599](https://github.com/NVIDIA/TensorRT-LLM/pull/14599) | Qwen3.5 MoE VLM + MTP |
| 2026-07-07 | [#15249](https://github.com/NVIDIA/TensorRT-LLM/pull/15249) | Qwen3.5 Dense VLM |
| 2026-07-08 | [#16065](https://github.com/NVIDIA/TensorRT-LLM/pull/16065) | merged | [https://nvbugs/6422332][fix] Keep SSM cache in weights dtype when ma… | `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`, `tensorrt_llm/_torch/pyexecutor/config_utils.py`, `tensorrt_llm/_torch/pyexecutor/model_loader.py` |
| 2026-07-14 | [#16264](https://github.com/NVIDIA/TensorRT-LLM/pull/16264) | merged | [None][fix] Align dense Qwen3.5-VL SSM cache dtype test with #16065 semantics | `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py` |
| 2026-07-14 | [#16353](https://github.com/NVIDIA/TensorRT-LLM/pull/16353) | merged | [TRTLLM-14054][perf] Qwen3.5-VL: pass the inner LM's normalized model_config to the weight mapper | `tensorrt_llm/_torch/models/modeling_qwen3_5.py` |
| 2026-07-21 | [#16469](https://github.com/NVIDIA/TensorRT-LLM/pull/16469) | 融合 QK norm + RoPE + gate |
| 2026-07-24 | [#15194](https://github.com/NVIDIA/TensorRT-LLM/pull/15194) | 融合 AllReduce + Gemma RMSNorm |
| 2026-07-29 | [#16936](https://github.com/NVIDIA/TensorRT-LLM/pull/16936) | merged | [None][fix] Fix Qwen3.5 weight-load memory growth and MTP CUTLASS fallback | `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` |
| 2026-07-31 | [#16642](https://github.com/NVIDIA/TensorRT-LLM/pull/16642) | merged | [TRTLLM-14497][feat] Add BF16/FP8 refit for qwen3.5_397b | `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py` |
| 2026-08-10 | [#17293](https://github.com/NVIDIA/TensorRT-LLM/pull/17293) | merged | [https://nvbugs/6434512][fix] Select Marlin for Qwen3.5 MoE on Hopper | `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` |
| 2026-08-11 | [#17433](https://github.com/NVIDIA/TensorRT-LLM/pull/17433) | merged | [None][fix] Qwen3.5 weight mapper for FP8 per-channel checkpoints | `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` |
| 2026-08-21 | [#17700](https://github.com/NVIDIA/TensorRT-LLM/pull/17700) | merged | [None][perf] Qwen3.5/3.8 wave-2: MoE, attention-DP, GDN replay, weight loading | `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` |
| 2026-09-23 | [#19519](https://github.com/NVIDIA/TensorRT-LLM/pull/19519) | merged | [https://nvbugs/6771102][fix] Support Qwen3.5 global FP8 checkpoints | `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` |

## 逐 PR diff 审计卡

### PR #11728 - Added Qwen3.5 Cookbook

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11728
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+385/-0，本地 patch 402 行。
- 动机: 给 `Qwen/Qwen3.5-397B-A17B` 和 `nvidia/Qwen3.5-397B-A17B-NVFP4` 提供 AutoDeploy cookbook，明确 TensorRT-LLM 的服务命令和 B200 资源假设。
- 实现要点: notebook 写入 `trtllm-serve` 命令、AutoDeploy registry 配置、NVFP4 4xB200 说明和示例 OpenAI 请求。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+trtllm-serve "nvidia/Qwen3.5-397B-A17B-NVFP4" \
+MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
```

- 已读文件: `examples/auto_deploy/cookbooks/qwen_3.5_trtllm_cookbook.ipynb`
- 验证与风险: 这是公平 benchmark 的 TensorRT-LLM deployment 证据；需要和 PyTorch backend / AutoDeploy registry 配置一起记录。

### PR #12242 - [None][feat] Initial Qwen3.5 text model support for PyT backend (BF16/FP8)

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12242
- 状态/时间: merged / 2026-03-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_5.py`；关联提交 `95db1d895e87`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 22 个文件，+854/-92，可读 patch 1349 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` added +269/-0 (269 lines); hunks: -0,0 +1,269; symbols: Qwen3_5MoeHfWeightMapper, _normalize_weight_names, handle_special_instance_module, _pack_projection_tensor，涉及 `Qwen3_5MoeHfWeightMapper, _normalize_weight_names, handle_special_instance_module`；`tensorrt_llm/_torch/models/modeling_qwen3_5.py` added +27/-0 (27 lines); hunks: -0,0 +1,27; symbols: Qwen3_5MoeForCausalLM, exists, that，涉及 `Qwen3_5MoeForCausalLM, exists, that`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` added +269/-0 (269 lines); hunks: -0,0 +1,269; symbols: Qwen3_5MoeHfWeightMapper, _normalize_weight_names, handle_special_instance_module, _pack_projection_tensor
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` added +27/-0 (27 lines); hunks: -0,0 +1,27; symbols: Qwen3_5MoeForCausalLM, exists, that
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -0,0 +1,269 @@
+import math
+import re
+from collections import defaultdict
+import torch
+from torch import nn
+from tensorrt_llm._torch.models.checkpoints.hf.qwen3_next_weight_mapper import (
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -0,0 +1,27 @@
+from .modeling_qwen3_next import Qwen3NextForCausalLM
+from .modeling_utils import register_auto_model
+@register_auto_model("Qwen3_5MoeForCausalLM")
+class Qwen3_5MoeForCausalLM(Qwen3NextForCausalLM):
+    """Thin wrapper that registers the Qwen3.5 MoE text architecture.
+    Qwen3.5 text reuses the same model internals as Qwen3Next
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` added +269/-0; `tensorrt_llm/_torch/models/modeling_qwen3_5.py` added +27/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/qa/llm_function_core_sanity.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12302 - Add Qwen 3.5 supporting (NVFP4)

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12302
- 状态/时间: merged / 2026-03-24
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 9 个文件，+225/-31，本地 patch 436 行。
- 动机: TensorRT-LLM PyTorch backend 需要支持 Qwen3.5 dense/MoE 和官方 NVFP4 checkpoint。
- 实现要点: 注册 `Qwen3_5ForCausalLM` 和 `Qwen3_5MoeForCausalLM`，扩展 HF mapper，normalize exclude modules，并新增 397B A17B NVFP4 accuracy tests。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+@register_auto_model("Qwen3_5ForCausalLM")
+class Qwen3_5ForCausalLM(Qwen3NextForCausalLM):
```

- 已读文件: `modeling_qwen3_5.py`, `qwen3_5_weight_mapper.py`, `config_utils.py`, accuracy refs/tests
- 验证与风险: 与 SGLang 对比时要区分 dense 和 MoE wrapper，以及 NVFP4 exclude module 规则。

### PR #12114 - Qwen 3.5 fix 3D position ID handling

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12114
- 状态/时间: merged / 2026-03-25
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 15 个文件，+3448/-275，本地 patch 7,822 行。
- 动机: Qwen3.5 VLM/mRoPE 需要 3D position IDs、chunked multimodal positions、video grid normalization 和 mRoPE delta cache，原 AutoDeploy path 不完整。
- 实现要点: 重写/扩展 `modeling_qwen3_5_moe.py` 的 multimodal input processor、mRoPE delta cache transform、registry configs 和单元测试。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+@TransformRegistry.register("initialize_mrope_delta_cache")
+mm_token_positions: torch.Tensor
```

- 已读文件: `modeling_qwen3_5_moe.py`, `mrope_delta_cache.py`, registry YAML, `test_qwen3_5_moe.py`, serving utils tests
- 验证与风险: 多模态 Qwen3.5 不能只比较 decode kernel；position construction 和 cache resource 命名也会影响 correctness。

### PR #13090 - Qwen3.5 dense weight loading

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13090
- 状态/时间: merged / 2026-04-30
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+85/-1，本地 patch 225 行。
- 动机: Qwen3.5 dense 4B/FP8 weight loading 需要独立覆盖，不能只靠 MoE mapper 过关。
- 实现要点: 扩展 `qwen3_5_weight_mapper.py`，新增 `Qwen/Qwen3.5-4B` accuracy refs 和 `TestQwen3_5_4B`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+class TestQwen3_5_4B(LlmapiAccuracyTestHarness):
+MODEL_NAME = "Qwen/Qwen3.5-4B"
```

- 已读文件: HF mapper, accuracy refs, `test_llm_api_pytorch.py`, test lists
- 验证与风险: SOTA loop 如果选 dense Qwen3.5，不能直接套 397B MoE 的 mapper 风险结论。

### PR #13716 - Preserve Qwen3.5 NVFP4 weight_scales

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13716
- 状态/时间: merged / 2026-05-04
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+9/-3，本地 patch 45 行。
- 动机: mapper 把 `weight_scales` 归一成 `weight_scale_inv` 的 FP8 逻辑会破坏 NVFP4 loader。
- 实现要点: 在 HF mapper 中识别 NVFP4 prefix，保留 `weight_scales` 给 `NVFP4LinearMethod.load_weight_scales`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+        nvfp4_prefixes = {
+            key[: -len(".weight_scale_2")] for key in weights if key.endswith(".weight_scale_2")
+        }
+                if prefix not in nvfp4_prefixes:
```

- 已读文件: `qwen3_5_weight_mapper.py`
- 验证与风险: Qwen3.5 NVFP4 对 scale 名称敏感；SGLang 对比 weight loader 时要检查 scale key remap。

### PR #13782 - Qwen3.5 DFlash

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13782
- 状态/时间: merged / 2026-05-12
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+144/-55，本地 patch 413 行。
- 动机: Qwen3.5 speculative/DFlash 需要把 GDN/Mamba cache 和 pyexecutor/model engine 接到 DFlash runtime。
- 实现要点: 修改 `gdn_mixer.py`、`mamba_cache_manager.py`、`model_engine.py` 和 `speculative/dflash.py`，让 hybrid linear-attention 模型能参与 DFlash。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+from tensorrt_llm._torch.speculative import dflash
+mamba_cache_manager
```

- 已读文件: `gdn_mixer.py`, `pyexecutor/_util.py`, `mamba_cache_manager.py`, `model_engine.py`, `speculative/dflash.py`
- 验证与风险: 对 SGLang MTP/speculative 对比时，必须记录 DFlash 是否启用；它会改变 decode state update 形态。

### PR #13996 - Perf optimizations for DFlash

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13996
- 状态/时间: merged / 2026-05-16
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+455/-285，本地 patch 1,606 行。
- 动机: DFlash 初始支持后需要减少状态搬运和模型 engine overhead。
- 实现要点: 调整 speculative modeling、GDN mixer、model engine 和 `llm_args.py`，把 DFlash path 做成更稳定的 perf path。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+    def _build_fused_kv_buffers(self) -> None:
+        """Stack per-layer KV projection + k_norm weights for a single fused GEMM.
+        return self.max_draft_len + 1
```

- 已读文件: `modeling_speculative.py`, `gdn_mixer.py`, `model_engine.py`, `speculative/dflash.py`, `llm_args.py`
- 验证与风险: 如果 TensorRT-LLM 领先来自 DFlash，需要和 SGLang MTP/SpecV2 分开归因。

### PR #12646 - [TRTLLM-11547][feat] Add Qwen3.5 MTP support.

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12646
- 状态/时间: merged / 2026-05-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_5.py`；关联提交 `5d19712ae7c1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+303/-42，可读 patch 564 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +64/-5 (69 lines); hunks: -6,25 +6,84; symbols: _translate_mtp_pattern, _normalize_qwen35_exclude_modules，涉及 `_translate_mtp_pattern, _normalize_qwen35_exclude_modules`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +64/-5 (69 lines); hunks: -6,25 +6,84; symbols: _translate_mtp_pattern, _normalize_qwen35_exclude_modules
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -6,25 +6,84 @@
+_MTP_TOP_TO_TRTLLM = {
+    "fc": "fc",
+    "norm": "shared_head.norm",
+    "pre_fc_norm_embedding": "pre_fc_norm_embedding",
+    "pre_fc_norm_hidden": "pre_fc_norm_hidden",
+}
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +64/-5
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_b200.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14164 - [TRTLLM-12500][feat] Add support for Qwen3.5 VL MoE - REVERTED by #14599

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14164
- 状态/时间: merged / 2026-05-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`；关联提交 `751be5d9b516`, `96a4a0937e37`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 14 个文件，+1037/-173，可读 patch 1420 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +312/-2 (314 lines); hunks: -1,7 +1,29; -51,6 +73,248 @@ def _translate_mtp_pattern(name, n_hidden_layers):; symbols: _translate_mtp_pattern, Qwen35ConfigCompat, and, normalize，涉及 `_translate_mtp_pattern, Qwen35ConfigCompat, and`；`tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +1/-0 (1 lines); hunks: -13,6 +13,7; symbols: Qwen3_5MoeHfWeightMapper，涉及 `Qwen3_5MoeHfWeightMapper`；`tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` added +450/-0 (450 lines); hunks: -0,0 +1,450; symbols: _write_qwen35_moe_vl_config, test_qwen35_moe_vl_config_preserves_vlm_architecture, test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper，涉及 `_write_qwen35_moe_vl_config, test_qwen35_moe_vl_config_preserves_vlm_architecture, test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +312/-2 (314 lines); hunks: -1,7 +1,29; -51,6 +73,248 @@ def _translate_mtp_pattern(name, n_hidden_layers):; symbols: _translate_mtp_pattern, Qwen35ConfigCompat, and, normalize
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +1/-0 (1 lines); hunks: -13,6 +13,7; symbols: Qwen3_5MoeHfWeightMapper
  - `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` added +450/-0 (450 lines); hunks: -0,0 +1,450; symbols: _write_qwen35_moe_vl_config, test_qwen35_moe_vl_config_preserves_vlm_architecture, test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -1,7 +1,29 @@
+from types import SimpleNamespace
+from typing import Dict, List
+import torch
+from transformers import PretrainedConfig
+from ...inputs import (
+    ContentFormat,
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -13,6 +13,7 @@
+@register_mapper("HF", "Qwen3_5MoeForConditionalGeneration")
diff -- tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py
@@ -0,0 +1,450 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+import json
+import os
+from copy import deepcopy
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +312/-2; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +1/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` added +450/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_l40s.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14465 - [None][feat] Revert Add support for Qwen3.5 VL MoE (#14164)

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14465
- 状态/时间: merged / 2026-05-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`；关联提交 `751be5d9b516`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 14 个文件，+173/-1037，可读 patch 1420 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +2/-312 (314 lines); hunks: -1,29 +1,7; -73,248 +51,6 @@ def _translate_mtp_pattern(name, n_hidden_layers):; symbols: _translate_mtp_pattern, Qwen35ConfigCompat, and, normalize，涉及 `_translate_mtp_pattern, Qwen35ConfigCompat, and`；`tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +0/-1 (1 lines); hunks: -13,7 +13,6; symbols: Qwen3_5MoeHfWeightMapper，涉及 `Qwen3_5MoeHfWeightMapper`；`tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` removed +0/-450 (450 lines); hunks: -1,450 +0,0; symbols: _write_qwen35_moe_vl_config, test_qwen35_moe_vl_config_preserves_vlm_architecture, test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper，涉及 `_write_qwen35_moe_vl_config, test_qwen35_moe_vl_config_preserves_vlm_architecture, test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +2/-312 (314 lines); hunks: -1,29 +1,7; -73,248 +51,6 @@ def _translate_mtp_pattern(name, n_hidden_layers):; symbols: _translate_mtp_pattern, Qwen35ConfigCompat, and, normalize
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +0/-1 (1 lines); hunks: -13,7 +13,6; symbols: Qwen3_5MoeHfWeightMapper
  - `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` removed +0/-450 (450 lines); hunks: -1,450 +0,0; symbols: _write_qwen35_moe_vl_config, test_qwen35_moe_vl_config_preserves_vlm_architecture, test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -1,29 +1,7 @@
-from types import SimpleNamespace
-from typing import Dict, List
-import torch
-from transformers import PretrainedConfig
-from ...inputs import (
-    ContentFormat,
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -13,7 +13,6 @@
-@register_mapper("HF", "Qwen3_5MoeForConditionalGeneration")
diff -- tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py
@@ -1,450 +0,0 @@
-# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-# SPDX-License-Identifier: Apache-2.0
-import json
-import os
-from copy import deepcopy
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +2/-312; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +0/-1
  - tests: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` removed +0/-450
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_l40s.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14659 - Add a reasoning parser for qwen3_5

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14659
- 状态/时间: merged / 2026-05-29
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+9/-0，本地 patch 30 行。
- 动机: Qwen3.5 forced-thinking 模板输出一开始就在 reasoning block 内，旧 `qwen3` parser 的 `reasoning_at_start=False` 不匹配。
- 实现要点: 注册 `qwen3_5` parser，并设置 `reasoning_at_start=True`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+@register_reasoning_parser("qwen3_5", reasoning_at_start=True)
```

- 已读文件: `tensorrt_llm/llmapi/reasoning_parser.py`
- 验证与风险: benchmark 输出清洗/CoT 解析不一致会影响评测分数，不能只看吞吐。

### PR #14667 - AutoDeploy Qwen3.5 400B NVFP4 accuracy regression fix

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14667
- 状态/时间: merged / 2026-06-02
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+72/-35，本地 patch 464 行。
- 动机: AutoDeploy Qwen3.5 400B NVFP4 出现 accuracy regression，shared expert sharding 与 SwiGLU fusion 需要修正。
- 实现要点: 将 shared expert 复制而不是 TP sharding，加入 whitelist sharding hints，并扩展 SwiGLU fusion pattern。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+# The shared expert is replicated
+apply_sharding_hints:
```

- 已读文件: `qwen3.5_moe_400b.yaml`, `modeling_qwen3_5_moe.py`, `custom_ops/linear/swiglu.py`, `fuse_swiglu.py`, waives
- 验证与风险: SGLang 若遇到 Qwen3.5 MoE 精度差异，应检查 shared expert 并行策略和 SwiGLU fusion，不要只比较 MoE matmul。

### PR #15001 - Uncomment Qwen3.5 from model registry

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15001
- 状态/时间: merged / 2026-06-05
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+9/-12，本地 patch 50 行。
- 动机: Qwen3.5 AutoDeploy registry entry 从注释状态变成默认可发现。
- 实现要点: 在 `models.yaml` 启用 `Qwen/Qwen3.5-35B-A3B` 和 `Qwen/Qwen3.5-397B-A17B`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+- name: Qwen/Qwen3.5-397B-A17B
+  config_id: qwen3_5_moe_400b
```

- 已读文件: `examples/auto_deploy/model_registry/models.yaml`
- 验证与风险: SOTA loop 可以把 registry entry 视作 TensorRT-LLM 官方 AutoDeploy lane，而不是临时命令。

### PR #15081 - Select CUTLASS MoE backend on non-Blackwell SMs

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15081
- 状态/时间: merged / 2026-06-09
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 2 个文件，+8/-2，本地 patch 52 行。
- 动机: Qwen3.5 FP8 test 在非 Blackwell SM 上不应默认使用 DeepGEMM，需要 fallback 到 CUTLASS MoE backend。
- 实现要点: accuracy test 按 SM 版本选择 `DEEPGEMM` 或 `CUTLASS`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+moe_backend = "DEEPGEMM" if get_sm_version() in (100, 103) else "CUTLASS"
```

- 已读文件: `test_llm_api_pytorch.py`, `waives.txt`
- 验证与风险: 跨 GPU 比较必须记录 MoE backend；否则 H100/B200 结果不可直接混用。

### PR #15111 - [TRTLLM-11548][doc] Add Qwen3.5 deployment guide doc

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15111
- 状态/时间: merged / 2026-06-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/configs/curated/qwen3.5.yaml`；关联提交 `09ebc592e535`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+97/-29，可读 patch 273 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `examples/configs/curated/qwen3.5.yaml` added +15/-0 (15 lines); hunks: -0,0 +1,15。
- 代码 diff 细节:
  - `examples/configs/curated/qwen3.5.yaml` added +15/-0 (15 lines); hunks: -0,0 +1,15
- 关键代码摘录:

```diff
diff -- examples/configs/curated/qwen3.5.yaml
@@ -0,0 +1,15 @@
+max_batch_size: 512
+max_num_tokens: 2048
+tensor_parallel_size: 4
+moe_expert_parallel_size: 4
+trust_remote_code: true
+enable_attention_dp: true
```

- 提取文件（未人工审阅）:
  - docs: `examples/configs/curated/qwen3.5.yaml` added +15/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/source/_static/config_db.json`, `docs/source/deployment-guide/deployment-guide-for-qwen3.5-on-trtllm.md`, `docs/source/deployment-guide/index.rst`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #15067 - Generalize FP8 checkpoint loading for Qwen3.5

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15067
- 状态/时间: merged / 2026-06-11
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 2 个文件，+68/-48，本地 patch 220 行。
- 动机: Qwen3.5 FP8 checkpoint 的 scale/exclude module 命名需要更通用的 mapper 处理。
- 实现要点: 重构 `qwen3_5_weight_mapper.py` 与 `modeling_qwen3_5.py` 中的 FP8/NVFP4 normalization。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+    # gdn_mixer uses Linear module for weight management of depthwise conv1d
+    # but conv1d is not a proper linear module and should be excluded from quant
+    normalized.add("*linear_attn.conv1d")
```

- 已读文件: `qwen3_5_weight_mapper.py`, `modeling_qwen3_5.py`
- 验证与风险: 对 FP8 checkpoint 的 load failure 或精度差异，先查 mapper normalization，而不是 kernel。

### PR #15185 - Qwen3.5 whitelist sharding and lm_head sharding

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15185
- 状态/时间: merged / 2026-06-13
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+193/-118，本地 patch 735 行。
- 动机: AutoDeploy Qwen3.5 需要白名单式 sharding，并让 `lm_head` 参与 sharding，以减少单 rank 负载和保持图变换可控。
- 实现要点: 更新 registry YAML、`modeling_qwen3_5_moe.py` sharding hints、`fuse_swiglu.py` 和 `sharding_ir.py` 测试。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+lm_head:
+apply_sharding_hints
```

- 已读文件: `qwen3.5_moe_400b.yaml`, `modeling_qwen3_5_moe.py`, `fuse_swiglu.py`, `sharding_ir.py`, tests
- 验证与风险: SGLang 对标时应单独观察 lm_head 和 shared expert 的 sharding/communication，而不是只看 MoE expert GEMM。

### PR #15543 - Add EPLB support for Qwen3.5

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15543
- 状态/时间: merged / 2026-06-26
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 3 个文件，+73/-0，本地 patch 130 行。
- 动机: Qwen3.5 MoE 需要 EPLB，尤其是 B200/GB200 perf sanity 和 PyTorch backend test coverage。
- 实现要点: 在 `moe_load_balancer.py` 增加 Qwen3.5 支持，并把 B200/GB200 test-db lane 接入。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+    'Qwen2MoeForCausalLM',
+    'Qwen3MoeForCausalLM',
+    'Qwen3_5MoeForCausalLM',
```

- 已读文件: `moe_load_balancer.py`, `test_llm_api_pytorch.py`, B200/GB200 test-db YAML
- 验证与风险: 与 SGLang 的 EPLB/DeepEP/EP 对比时，必须记录 load-balancer 是否启用和测试集群拓扑。


### PR #15650 - [None][test] Add Qwen3.5-397B-A17B-NVFP4 B200 aggregated perf-sanity tests

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15650
- 状态/时间: merged / 2026-06-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml`；关联提交 `d419595e40d3`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+440/-5，可读 patch 460 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml` added +425/-0 (425 lines); hunks: -0,0 +1,425。
- 代码 diff 细节:
  - `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml` added +425/-0 (425 lines); hunks: -0,0 +1,425
- 关键代码摘录:

```diff
diff -- tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml
@@ -0,0 +1,425 @@
+metadata:
+  model_name: qwen3.5_397b_a17b_fp4
+  supported_gpus:
+  - B200
+hardware:
+  gpus_per_node: 8
```

- 提取文件（未人工审阅）:
  - tests: `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml` added +425/-0
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_b200_multi_gpus_perf_sanity.yml`, `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14599 - 支持 Qwen3.5 VL MoE 并修复 MTP

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14599
- 状态/时间: merged / 2026-07-04
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 1,734 行 diff，16 个文件，+1140/-256。
- 动机: TensorRT-LLM 已有可复用的 Qwen3Next 文本 runtime 和 Qwen3-VL vision tower，但缺少 Qwen3.5-35B-A3B 的复合多模态架构、原生配置归一化、权重映射以及 speculative decoding 的 token 传递。
- 实现要点: 注册 `Qwen3_5MoeForConditionalGeneration`，保留 HF text/vision 子配置并归一化 runtime alias，把 `Qwen3VisionModel` 与 MoE decoder 组合起来，映射语言模型权重；VLM wrapper 构造 embedding 后，MTP/Eagle 从 `orig_input_ids` 取回 prompt token。
- 代码 diff 细节: 新 VLM 类负责多模态 placeholder 元数据与 device path，同时复用 Qwen3Next LM；测试覆盖配置路由、权重加载、模态 parity、MTP 和 MMMU accuracy。
- 关键代码摘录:

```diff
+@register_auto_model("Qwen3_5MoeForConditionalGeneration")
+class Qwen3_5MoeVLModel(Qwen3VLModelBase):
+    """VLM wrapper composing Qwen3 vision encoder with Qwen3.5 MoE text decoder."""
+    kwargs["vision_model_class"] = Qwen3VisionModel
```

- 已读文件: runtime：`modeling_qwen3_5.py`、`qwen3_5_weight_mapper.py`、`modeling_speculative.py`、model loader/config utilities；测试/文档：`test_modeling_qwen3_5_vl_moe.py`、MMMU references、supported-model matrix。
- 验证与风险: benchmark 必须区分 text-only Qwen3.5 与 MoE VLM；需验证 mRoPE、图片/视频 placeholder、FP8 exclude-module 归一化和 MTP prompt-token 恢复。

### PR #15249 - 支持 Qwen3.5 VL Dense

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15249
- 状态/时间: merged / 2026-07-07
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 776 行 diff，11 个文件，+594/-26。
- 动机: MoE VLM wrapper 无法覆盖 Dense Qwen3.5-27B；后者使用 Dense `GatedMLP`，并有不同的 architecture 注册。
- 实现要点: 泛化共享 Qwen3.5 VLM 基类，注册 `Qwen3_5ForConditionalGeneration`，在保留 Qwen3 vision/mRoPE 处理的同时选择 Dense decoder，并把同一 HF mapper 扩展到 Dense 架构。
- 代码 diff 细节: 生产配置归一化保留 Dense `intermediate_size` 与空 deepstack index；新 parity suite 覆盖 image、multi-image、video、chunked-prefill position slicing 和模型构造。
- 关键代码摘录:

```diff
+@register_auto_model("Qwen3_5ForConditionalGeneration")
+class Qwen3_5VLModel(_Qwen3_5VLModel):
+    """VLM wrapper composing Qwen3 vision encoder with dense Qwen3.5 text decoder."""
```

- 已读文件: runtime：`modeling_qwen3_5.py`、`qwen3_5_weight_mapper.py`、model/config registries；测试/文档：`test_modeling_qwen3_5_vl.py`、MMMU references、supported-model matrix。
- 验证与风险: Dense 与 MoE checkpoint 需要独立的 accuracy/performance 行；需验证 composite config alias、mRoPE chunk slicing、SSM-cache dtype 和多模态 forward parity。

### PR #16065 - [https://nvbugs/6422332][fix] Keep SSM cache in weights dtype when ma…

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16065
- 状态/时间: merged / 2026-07-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`；关联提交 `7d7c364ae997`, `d163e74407cd`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+36/-18，可读 patch 110 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +11/-2 (13 lines); hunks: -125,15 +125,24 @@ def test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype(; symbols: test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper，涉及 `test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper`；`tensorrt_llm/_torch/pyexecutor/config_utils.py` modified +20/-13 (33 lines); hunks: -52,25 +52,32 @@ def resolve_hf_torch_dtype(config):; -307,7 +314,7 @@ def extract_mamba_kv_cache_params(; symbols: resolve_hf_torch_dtype, resolve_mamba_ssm_cache_dtype, resolve_ssm_cache_dtype, extract_mamba_kv_cache_params，涉及 `resolve_hf_torch_dtype, resolve_mamba_ssm_cache_dtype, resolve_ssm_cache_dtype`；`tensorrt_llm/_torch/pyexecutor/model_loader.py` modified +2/-2 (4 lines); hunks: -35,7 +35,7; -52,7 +52,7 @@ def validate_and_set_mamba_ssm_cache_dtype(; symbols: validate_and_set_mamba_ssm_cache_dtype，涉及 `validate_and_set_mamba_ssm_cache_dtype`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +11/-2 (13 lines); hunks: -125,15 +125,24 @@ def test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype(; symbols: test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper
  - `tensorrt_llm/_torch/pyexecutor/config_utils.py` modified +20/-13 (33 lines); hunks: -52,25 +52,32 @@ def resolve_hf_torch_dtype(config):; -307,7 +314,7 @@ def extract_mamba_kv_cache_params(; symbols: resolve_hf_torch_dtype, resolve_mamba_ssm_cache_dtype, resolve_ssm_cache_dtype, extract_mamba_kv_cache_params
  - `tensorrt_llm/_torch/pyexecutor/model_loader.py` modified +2/-2 (4 lines); hunks: -35,7 +35,7; -52,7 +52,7 @@ def validate_and_set_mamba_ssm_cache_dtype(; symbols: validate_and_set_mamba_ssm_cache_dtype
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py
@@ -125,15 +125,24 @@ def test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype(
+    # "auto" keeps the SSM cache in the model weights dtype for performance:
+    # the checkpoint's mamba_ssm_dtype=float32 expresses SSM compute intent,
+    # and honoring it for cache allocation disables the FlashInfer bf16-state
+    # GDN decode kernel and doubles state memory traffic.
-    assert model_config.quant_config.mamba_ssm_cache_dtype is torch.float32
+    assert model_config.quant_config.mamba_ssm_cache_dtype is torch.bfloat16
diff -- tensorrt_llm/_torch/pyexecutor/config_utils.py
@@ -52,25 +52,32 @@ def resolve_hf_torch_dtype(config):
-def resolve_mamba_ssm_cache_dtype(config):
-    """Return the dtype to use for hybrid Mamba/SSM cache allocations.
-    Qwen3.5-style configs may store this field on the top-level config or the
-    nested text_config, and may call it either mamba_ssm_cache_dtype or
-    mamba_ssm_dtype. This helper centralizes that lookup so cache creation does
-    not fail later with a missing dtype. An "auto" value in any field is
diff -- tensorrt_llm/_torch/pyexecutor/model_loader.py
@@ -35,7 +35,7 @@
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +11/-2
  - runtime: `tensorrt_llm/_torch/pyexecutor/config_utils.py` modified +20/-13; `tensorrt_llm/_torch/pyexecutor/model_loader.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16264 - [None][fix] Align dense Qwen3.5-VL SSM cache dtype test with #16065 semantics

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16264
- 状态/时间: merged / 2026-07-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py`；关联提交 `7d7c364ae997`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+11/-2，可读 patch 27 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py` modified +11/-2 (13 lines); hunks: -143,15 +143,24 @@ def test_qwen35_dense_vl_resolves_mamba_ssm_cache_dtype(; symbols: test_qwen35_dense_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_dense_vl_resolves_model_and_mapper，涉及 `test_qwen35_dense_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_dense_vl_resolves_model_and_mapper`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py` modified +11/-2 (13 lines); hunks: -143,15 +143,24 @@ def test_qwen35_dense_vl_resolves_mamba_ssm_cache_dtype(; symbols: test_qwen35_dense_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_dense_vl_resolves_model_and_mapper
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py
@@ -143,15 +143,24 @@ def test_qwen35_dense_vl_resolves_mamba_ssm_cache_dtype(
+    # "auto" keeps the SSM cache in the model weights dtype for performance:
+    # the checkpoint's mamba_ssm_dtype=float32 expresses SSM compute intent,
+    # and honoring it for cache allocation disables the FlashInfer bf16-state
+    # GDN decode kernel and doubles state memory traffic.
-    assert model_config.quant_config.mamba_ssm_cache_dtype is torch.float32
+    assert model_config.quant_config.mamba_ssm_cache_dtype is torch.bfloat16
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py` modified +11/-2
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16353 - [TRTLLM-14054][perf] Qwen3.5-VL: pass the inner LM's normalized model_config to the weight mapper

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16353
- 状态/时间: merged / 2026-07-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_5.py`；关联提交 `924978f7a0d0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+9/-1，可读 patch 17 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +9/-1 (10 lines); hunks: -695,7 +695,15 @@ def load_weights(self, weights: Dict[str, torch.Tensor], we...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +9/-1 (10 lines); hunks: -695,7 +695,15 @@ def load_weights(self, weights: Dict[str, torch.Tensor], we...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -695,7 +695,15 @@ def load_weights(self, weights: Dict[str, torch.Tensor], weight_mapper: BaseWeig
-        weight_mapper.init_model_and_config(self.llm, self.model_config)
+        # Hand the mapper the inner LM's model_config, not the VLM wrapper's:
+        # only the inner config went through the Qwen3.5 quant-dict
+        # normalization applied in the inner LM's __init__ (HF->TRT-LLM key
+        # translation + synthesis of the fused in_proj_qkvz FP8 entry). With
+        # the wrapper's un-normalized copy the mapper misses that FP8 entry,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +9/-1
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16469 - 融合 Qwen3.5/3.6 注意力预处理

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16469
- 状态/时间: merged / 2026-07-21
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 913 行 diff，6 个文件，+775/-19。
- 动机: full-attention layer 过去分别执行 Q/gate 解交错、Q/K 归一化、RoPE、V 拷贝，并在输出端另启 sigmoid/multiply gate。
- 实现要点: 新增 Triton fast path，一次读取 interleaved projection，完成 Gemma RMSNorm、普通/交错 mRoPE 并输出 packed QKV 与 gate；输出 gate 用 in-place fused sigmoid-multiply，不支持的布局回退到通用路径。
- 代码 diff 细节: `QKNormRoPEAttention.preprocess_qkv` 按 dtype/layout/RoPE contract 控制融合，使权重加载、LoRA、编译、HIP 和不支持的 scaling 与优化解耦。
- 关键代码摘录:

```diff
+qkv, gate = fused_qkv_gemma_rmsnorm_rope_gate(
+    qkv, self.q_norm.weight, self.k_norm.weight,
+    self.rotary_emb.rotary_cos_sin, positions.contiguous(), ...)
+return qkv, None, None, gate
```

- 已读文件: runtime：`attention.py`、`qk_norm_attention.py`、`modeling_qwen3_next.py`、`fused_qk_norm_rope_gate.py`；测试：`test_fused_qk_norm_rope_gate.py`、B200 test database。
- 验证与风险: 需验证 BF16/FP16、partial/full rotary dim、mRoPE section、zero-token 与非连续输入，并同时对照 Python reference 和生产 THOP 路径。

### PR #15194 - 为 Qwen3-Next/Qwen3.5 融合 Gemma RMSNorm 与 AllReduce

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15194
- 状态/时间: merged / 2026-07-24
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 418 行 diff，3 个文件，+278/-29。
- 动机: TP Qwen3.5 decoder layer 需要分别做 collective 和 Gemma RMSNorm，而已有融合 collective 读取普通 norm weight，不包含 Gemma 的 `(1 + weight)` 语义。
- 实现要点: 对非 attention-DP 的 TP 默认开启 eager fusion，把 attention/MoE collective 延迟到 `RESIDUAL_RMS_NORM`，权重加载后一次性预计算 Gemma-adjusted norm weight；MTP 没有下一层 norm 时禁用 post-MoE fusion。
- 代码 diff 细节: 同一 PR 还为 FlashInfer GDN decode 增加仅在标量 slice 未满足 CuTe-DSL 32-byte 对齐时 clone 的保护，避免拖慢已对齐 shape。
- 关键代码摘录:

```diff
+norm._fused_norm_weight = (w.float() + 1.0).to(w.dtype)
+fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
+norm_weight=_fused_norm_weight(self.post_attention_layernorm),
```

- 已读文件: runtime：`modeling_qwen3_next.py`、`fused_sigmoid_gating_recurrent.py`；测试：`test_qwen3_next_eager_fusion.py`。
- 验证与风险: 需确认 collective 只有一个 owner、attention-DP 保持非融合、加载后 derived weight 会刷新、Gemma 数值使用 `(1 + weight)`，未对齐 Qwen3.6 head slice 走受控 copy。

### PR #16936 - [None][fix] Fix Qwen3.5 weight-load memory growth and MTP CUTLASS fallback

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16936
- 状态/时间: merged / 2026-07-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`；关联提交 `2341c704a6e8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+89/-9，可读 patch 185 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +6/-1 (7 lines); hunks: -5,6 +5,7; -556,6 +557,7 @@ def _remap_dense_mlp_weights(self, weights: dict) -> dict:; symbols: _remap_dense_mlp_weights, preprocess_weights，涉及 `_remap_dense_mlp_weights, preprocess_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +6/-1 (7 lines); hunks: -5,6 +5,7; -556,6 +557,7 @@ def _remap_dense_mlp_weights(self, weights: dict) -> dict:; symbols: _remap_dense_mlp_weights, preprocess_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -5,6 +5,7 @@
+from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
@@ -556,6 +557,7 @@ def _remap_dense_mlp_weights(self, weights: dict) -> dict:
+        is_consumable = isinstance(weights, ConsumableWeightsDict)
@@ -584,4 +586,7 @@ def preprocess_weights(self, weights: dict) -> dict:
-        return super().preprocess_weights(packed_weights)
+        processed_weights = super().preprocess_weights(packed_weights)
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/checkpoints/base_weight_loader.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16642 - [TRTLLM-14497][feat] Add BF16/FP8 refit for qwen3.5_397b

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16642
- 状态/时间: merged / 2026-07-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py`；关联提交 `1c797cf7ab69`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 17 个文件，+1062/-159，可读 patch 1560 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +118/-4 (122 lines); hunks: -1,3 +1,6; -62,6 +65,109 @@ class Qwen3_5MoeHfWeightMapper(Qwen3NextHfWeightMapper):; symbols: Qwen3_5MoeHfWeightMapper, __init__, begin_update_weights, finalize_update_weights，涉及 `Qwen3_5MoeHfWeightMapper, __init__, begin_update_weights`；`tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +20/-6 (26 lines); hunks: -695,26 +695,40 @@ def multimodal_data_device_paths(self) -> List[str]:; symbols: multimodal_data_device_paths, load_weights，涉及 `multimodal_data_device_paths, load_weights`；`tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: test_qwen35_vl_propagates_partial_loading_to_vision_encoder, _VisualStub, __init__, test_qwen3_vision_loader_propagates_partial_loading，涉及 `test_qwen35_vl_propagates_partial_loading_to_vision_encoder, _VisualStub, __init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +118/-4 (122 lines); hunks: -1,3 +1,6; -62,6 +65,109 @@ class Qwen3_5MoeHfWeightMapper(Qwen3NextHfWeightMapper):; symbols: Qwen3_5MoeHfWeightMapper, __init__, begin_update_weights, finalize_update_weights
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +20/-6 (26 lines); hunks: -695,26 +695,40 @@ def multimodal_data_device_paths(self) -> List[str]:; symbols: multimodal_data_device_paths, load_weights
  - `tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: test_qwen35_vl_propagates_partial_loading_to_vision_encoder, _VisualStub, __init__, test_qwen3_vision_loader_propagates_partial_loading
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
@@ -62,6 +65,109 @@ class Qwen3_5MoeHfWeightMapper(Qwen3NextHfWeightMapper):
+    def __init__(self) -> None:
+        super().__init__()
+        self._partial_split_weights: dict[str, object] = {}
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -695,26 +695,40 @@ def multimodal_data_device_paths(self) -> List[str]:
-    def load_weights(self, weights: Dict[str, torch.Tensor], weight_mapper: BaseWeightMapper):
+    def load_weights(
+        self,
+        weights: Dict[str, torch.Tensor],
+        weight_mapper: BaseWeightMapper,
+        allow_partial_loading: bool = False,
diff -- tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py
@@ -0,0 +1,114 @@
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +118/-4; `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +20/-6
  - tests: `tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py` added +114/-0
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_a10.yml`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`, `tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py`, `tests/unittest/_torch/ray_orchestrator/multi_gpu/test_llm_update_weights_multi_gpu.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17293 - [https://nvbugs/6434512][fix] Select Marlin for Qwen3.5 MoE on Hopper

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/17293
- 状态/时间: merged / 2026-08-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`；关联提交 `ec044a2e1ac3`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+74/-2，可读 patch 143 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +35/-1 (36 lines); hunks: -15,11 +15,14; -32,6 +35,7; symbols: _get_qwen35_moe_model_defaults, _translate_mtp_pattern, Qwen3_5MoeForCausalLM, that，涉及 `_get_qwen35_moe_model_defaults, _translate_mtp_pattern, Qwen3_5MoeForCausalLM`；`tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +39/-1 (40 lines); hunks: -7,6 +7,7; -15,7 +16,7; symbols: _write_qwen35_moe_vl_config, test_qwen35_moe_vl_resolves_model_and_mapper, test_qwen35_moe_model_defaults, test_qwen35_moe_vl_placeholder_metadata_registered，涉及 `_write_qwen35_moe_vl_config, test_qwen35_moe_vl_resolves_model_and_mapper, test_qwen35_moe_model_defaults`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +35/-1 (36 lines); hunks: -15,11 +15,14; -32,6 +35,7; symbols: _get_qwen35_moe_model_defaults, _translate_mtp_pattern, Qwen3_5MoeForCausalLM, that
  - `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +39/-1 (40 lines); hunks: -7,6 +7,7; -15,7 +16,7; symbols: _write_qwen35_moe_vl_config, test_qwen35_moe_vl_resolves_model_and_mapper, test_qwen35_moe_model_defaults, test_qwen35_moe_vl_placeholder_metadata_registered
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -15,11 +15,14 @@
-from typing import Dict, List, Literal
+from typing import TYPE_CHECKING, Dict, List, Literal
+if TYPE_CHECKING:
+    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
@@ -32,6 +35,7 @@
+from ..utils import is_nvfp4_marlin_supported_sm
diff -- tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py
@@ -7,6 +7,7 @@
+import pytest
@@ -15,7 +16,7 @@
-from tensorrt_llm._torch.models import Qwen3_5MoeVLModel
+from tensorrt_llm._torch.models import Qwen3_5MoeForCausalLM, Qwen3_5MoeVLModel
@@ -27,6 +28,10 @@
+from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +35/-1
  - tests: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +39/-1
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17433 - [None][fix] Qwen3.5 weight mapper for FP8 per-channel checkpoints

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/17433
- 状态/时间: merged / 2026-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py`；关联提交 `28e03befe6c2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+155/-23，可读 patch 216 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` added +141/-0 (141 lines); hunks: -0,0 +1,141; symbols: _make_mapper, _fp8, _scale, _bf16，涉及 `_make_mapper, _fp8, _scale`；`tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +14/-23 (37 lines); hunks: -178,24 +178,19 @@ def _normalize_weight_names(self, weights: dict) -> dict:; -204,12 +199,6 @@ def _normalize_scale_names(self, weights: dict, quant_algo)...; symbols: _normalize_weight_names, _normalize_scale_names, _normalize_fp8_block_scale_names, preprocess_weights，涉及 `_normalize_weight_names, _normalize_scale_names, _normalize_fp8_block_scale_names`。
- 代码 diff 细节:
  - `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` added +141/-0 (141 lines); hunks: -0,0 +1,141; symbols: _make_mapper, _fp8, _scale, _bf16
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +14/-23 (37 lines); hunks: -178,24 +178,19 @@ def _normalize_weight_names(self, weights: dict) -> dict:; -204,12 +199,6 @@ def _normalize_scale_names(self, weights: dict, quant_algo)...; symbols: _normalize_weight_names, _normalize_scale_names, _normalize_fp8_block_scale_names, preprocess_weights
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py
@@ -0,0 +1,141 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -178,24 +178,19 @@ def _normalize_weight_names(self, weights: dict) -> dict:
-    def _normalize_scale_names(self, weights: dict, quant_algo) -> tuple[dict, bool]:
-        # Canonicalize FP8 weight_scale layout so the Linear loader sees one
-        # shape per quant algo:
-        #   - FP8_BLOCK_SCALES: modelopt fp8_pb_wo stores weight_scale shaped
-        #     [blocks_out, 1, blocks_in, 1]; squeeze to [blocks_out, blocks_in]
-        #     and rename to weight_scale_inv. Returns is_modelopt_pb_wo=True
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` added +141/-0
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +14/-23
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17700 - [None][perf] Qwen3.5/3.8 wave-2: MoE, attention-DP, GDN replay, weight loading

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/17700
- 状态/时间: merged / 2026-08-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`；关联提交 `2f17320eb253`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 15 个文件，+875/-71，可读 patch 1257 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +14/-1 (15 lines); hunks: -36,6 +36,7; -80,6 +81,18 @@ def _get_qwen35_moe_model_defaults(llm_args: "TorchLlmArgs")...; symbols: _get_qwen35_moe_model_defaults, _filter_language_model_weights, _translate_mtp_pattern, load_weights，涉及 `_get_qwen35_moe_model_defaults, _filter_language_model_weights, _translate_mtp_pattern`；`tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +22/-1 (23 lines); hunks: -18,9 +18,13; -198,6 +202,23 @@ def test_qwen35_moe_model_defaults(; symbols: test_qwen35_moe_model_defaults, test_qwen35_vl_filter_preserves_consumable_weights, test_qwen35_moe_vl_placeholder_metadata_registered，涉及 `test_qwen35_moe_model_defaults, test_qwen35_vl_filter_preserves_consumable_weights, test_qwen35_moe_vl_placeholder_metadata_registered`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +14/-1 (15 lines); hunks: -36,6 +36,7; -80,6 +81,18 @@ def _get_qwen35_moe_model_defaults(llm_args: "TorchLlmArgs")...; symbols: _get_qwen35_moe_model_defaults, _filter_language_model_weights, _translate_mtp_pattern, load_weights
  - `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +22/-1 (23 lines); hunks: -18,9 +18,13; -198,6 +202,23 @@ def test_qwen35_moe_model_defaults(; symbols: test_qwen35_moe_model_defaults, test_qwen35_vl_filter_preserves_consumable_weights, test_qwen35_moe_vl_placeholder_metadata_registered
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -36,6 +36,7 @@
+from .checkpoints.base_weight_loader import ConsumableWeightsDict
@@ -80,6 +81,18 @@ def _get_qwen35_moe_model_defaults(llm_args: "TorchLlmArgs") -> dict:
+def _filter_language_model_weights(weights: Dict[str, torch.Tensor]):
+    """Drop vision weights without disabling incremental weight consumption.
+    Ownership: a ConsumableWeightsDict input is emptied, since the returned
+    mapping aliases its tensors. The caller must use only the return value.
diff -- tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py
@@ -18,9 +18,13 @@
+from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
-from tensorrt_llm._torch.models.modeling_qwen3_5 import _normalize_qwen35_moe_vl_config
+from tensorrt_llm._torch.models.modeling_qwen3_5 import (
+    _filter_language_model_weights,
+    _normalize_qwen35_moe_vl_config,
+)
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +14/-1
  - tests: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +22/-1
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/executor/test_py_executor.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`, `tests/unittest/_torch/models/checkpoints/test_consumable_weights_dict.py`, `tests/unittest/_torch/modules/fused_moe/test_deepgemm_fused_expand_quant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19519 - [https://nvbugs/6771102][fix] Support Qwen3.5 global FP8 checkpoints

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19519
- 状态/时间: merged / 2026-09-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py`；关联提交 `ef9a3340a314`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+115/-19，可读 patch 182 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` modified +92/-2 (94 lines); hunks: -19,6 +19,7; -45,10 +46,12; symbols: _make_mapper, test_fp8_rowwise_full, test_modelopt_fp8_per_tensor_linear_attention, test_modelopt_fp8_excluded_linear_attention_falls_back_to_bf16，涉及 `_make_mapper, test_fp8_rowwise_full, test_modelopt_fp8_per_tensor_linear_attention`；`tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +23/-17 (40 lines); hunks: -47,7 +47,7 @@ class Qwen3_5MoeHfWeightMapper(Qwen3NextHfWeightMapper):; -339,13 +339,13 @@ def _dequantize_linear_attn_fp8_qkvz(self, weights: dict)...; symbols: Qwen3_5MoeHfWeightMapper, _dequantize_linear_attn_fp8_qkvz, _requantize_linear_attn_fp8_qkvz, preprocess_weights，涉及 `Qwen3_5MoeHfWeightMapper, _dequantize_linear_attn_fp8_qkvz, _requantize_linear_attn_fp8_qkvz`。
- 代码 diff 细节:
  - `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` modified +92/-2 (94 lines); hunks: -19,6 +19,7; -45,10 +46,12; symbols: _make_mapper, test_fp8_rowwise_full, test_modelopt_fp8_per_tensor_linear_attention, test_modelopt_fp8_excluded_linear_attention_falls_back_to_bf16
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +23/-17 (40 lines); hunks: -47,7 +47,7 @@ class Qwen3_5MoeHfWeightMapper(Qwen3NextHfWeightMapper):; -339,13 +339,13 @@ def _dequantize_linear_attn_fp8_qkvz(self, weights: dict)...; symbols: Qwen3_5MoeHfWeightMapper, _dequantize_linear_attn_fp8_qkvz, _requantize_linear_attn_fp8_qkvz, preprocess_weights
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py
@@ -19,6 +19,7 @@
+from tensorrt_llm.models.modeling_utils import QuantConfig
@@ -45,10 +46,12 @@
-def _make_mapper(quant_algo: QuantAlgo) -> Qwen3_5MoeHfWeightMapper:
+def _make_mapper(
+    quant_algo: QuantAlgo, exclude_modules: list[str] | None = None
+) -> Qwen3_5MoeHfWeightMapper:
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -47,7 +47,7 @@ class Qwen3_5MoeHfWeightMapper(Qwen3NextHfWeightMapper):
-       checkpoints with per-tensor FP8 in_proj, the projections are
+       checkpoints and global FP8 checkpoints with per-tensor FP8 in_proj, the projections are
@@ -339,13 +339,13 @@ def _dequantize_linear_attn_fp8_qkvz(self, weights: dict) -> dict:
-        When _normalize_qwen35_quant_config_dict synthesized an FP8 entry for
-        the fused ``in_proj_qkvz`` module (i.e. the checkpoint quantizes every
-        split projection per-tensor FP8), keep the weights FP8 instead of
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` modified +92/-2
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +23/-17
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
