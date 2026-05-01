# sglang Nemotron Super 模型 PR 优化历史

## 覆盖范围

- 重做日期: 2026-05-01
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `4197c55968`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` | [#23907](https://github.com/sgl-project/sglang/pull/23907), [#23968](https://github.com/sgl-project/sglang/pull/23968) |
| `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Super.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/nemotron3-nano-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` | [#23907](https://github.com/sgl-project/sglang/pull/23907) |
| `docs_new/src/snippets/autoregressive/nemotron3-super-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/configs/jet_nemotron.py` | [#12448](https://github.com/sgl-project/sglang/pull/12448) |
| `python/sglang/srt/configs/nano_nemotron_vl.py` | [#12277](https://github.com/sgl-project/sglang/pull/12277), [#23568](https://github.com/sgl-project/sglang/pull/23568), [#23857](https://github.com/sgl-project/sglang/pull/23857) |
| `python/sglang/srt/configs/nemotron_h.py` | [#10909](https://github.com/sgl-project/sglang/pull/10909), [#12690](https://github.com/sgl-project/sglang/pull/12690), [#16227](https://github.com/sgl-project/sglang/pull/16227), [#19950](https://github.com/sgl-project/sglang/pull/19950), [#20458](https://github.com/sgl-project/sglang/pull/20458) |
| `python/sglang/srt/models/jet_nemotron.py` | [#12448](https://github.com/sgl-project/sglang/pull/12448) |
| `python/sglang/srt/models/nano_nemotron_vl.py` | [#12277](https://github.com/sgl-project/sglang/pull/12277), [#14051](https://github.com/sgl-project/sglang/pull/14051), [#23568](https://github.com/sgl-project/sglang/pull/23568), [#23857](https://github.com/sgl-project/sglang/pull/23857) |
| `python/sglang/srt/models/nemotron_h.py` | [#10909](https://github.com/sgl-project/sglang/pull/10909), [#11866](https://github.com/sgl-project/sglang/pull/11866), [#12015](https://github.com/sgl-project/sglang/pull/12015), [#12277](https://github.com/sgl-project/sglang/pull/12277), [#12690](https://github.com/sgl-project/sglang/pull/12690), [#16172](https://github.com/sgl-project/sglang/pull/16172), [#16227](https://github.com/sgl-project/sglang/pull/16227), [#16569](https://github.com/sgl-project/sglang/pull/16569), [#17013](https://github.com/sgl-project/sglang/pull/17013), [#18546](https://github.com/sgl-project/sglang/pull/18546), [#19903](https://github.com/sgl-project/sglang/pull/19903), [#20580](https://github.com/sgl-project/sglang/pull/20580), ... (15 total) |
| `python/sglang/srt/models/nemotron_h_mtp.py` | [#17013](https://github.com/sgl-project/sglang/pull/17013), [#19433](https://github.com/sgl-project/sglang/pull/19433) |
| `python/sglang/srt/models/nemotron_nas.py` | [#9067](https://github.com/sgl-project/sglang/pull/9067) |
| `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` | [#12277](https://github.com/sgl-project/sglang/pull/12277), [#14051](https://github.com/sgl-project/sglang/pull/14051), [#23568](https://github.com/sgl-project/sglang/pull/23568), [#23857](https://github.com/sgl-project/sglang/pull/23857) |
| `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml` | [#18119](https://github.com/sgl-project/sglang/pull/18119) |
| `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml` | [#18119](https://github.com/sgl-project/sglang/pull/18119) |
| `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` | [#20575](https://github.com/sgl-project/sglang/pull/20575), [#20616](https://github.com/sgl-project/sglang/pull/20616), [#21516](https://github.com/sgl-project/sglang/pull/21516) |
| `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` | [#20575](https://github.com/sgl-project/sglang/pull/20575), [#20616](https://github.com/sgl-project/sglang/pull/20616) |
| `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` | [#20616](https://github.com/sgl-project/sglang/pull/20616) |
| `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` | [#23594](https://github.com/sgl-project/sglang/pull/23594) |
| `test/registered/models/test_nvidia_nemotron_3_nano.py` | [#18119](https://github.com/sgl-project/sglang/pull/18119), [#23874](https://github.com/sgl-project/sglang/pull/23874) |
| `test/registered/models/test_nvidia_nemotron_nano_v2.py` | 无直接 PR 号提交 |
| `test/registered/models/test_nvidia_nemotron_nano_v2_vl.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 29
- 原文档显式引用补充 PR 数: 2
- 当前文档总 PR 数: 31
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-08-11 | [#5073](https://github.com/sgl-project/sglang/pull/5073) | closed | [Model] Add support for nvidia/Llama-3_3-Nemotron-Super-49B-v1 | `python/sglang/srt/models/nemotron_nas.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/utils.py` |
| 2025-08-17 | [#9067](https://github.com/sgl-project/sglang/pull/9067) | merged | model: support nvidia/Llama-3_3-Nemotron-Super-49B-v1 | `python/sglang/srt/models/nemotron_nas.py` |
| 2025-10-08 | [#10909](https://github.com/sgl-project/sglang/pull/10909) | merged | model: Support Hybrid Mamba2 NemotronHForCausalLM (nvidia/NVIDIA-Nemotron-Nano-9B-v2) | `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py` |
| 2025-10-23 | [#11866](https://github.com/sgl-project/sglang/pull/11866) | merged | Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4 | `python/sglang/srt/models/nemotron_h.py` |
| 2025-10-23 | [#12015](https://github.com/sgl-project/sglang/pull/12015) | merged | Revert "Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4" | `python/sglang/srt/models/nemotron_h.py` |
| 2025-11-09 | [#12448](https://github.com/sgl-project/sglang/pull/12448) | merged | Add Jet-Nemotron | `python/sglang/srt/models/jet_nemotron.py`, `python/sglang/srt/configs/jet_nemotron.py` |
| 2025-11-21 | [#12690](https://github.com/sgl-project/sglang/pull/12690) | merged | Feat/nemotron nano v3 support | `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py` |
| 2025-11-26 | [#12277](https://github.com/sgl-project/sglang/pull/12277) | merged | Support nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 (and nvidia/C-RADIOv2-H) | `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py` |
| 2025-12-31 | [#16172](https://github.com/sgl-project/sglang/pull/16172) | merged | [NemotronH] PP support | `python/sglang/srt/models/nemotron_h.py` |
| 2026-01-02 | [#16227](https://github.com/sgl-project/sglang/pull/16227) | merged | [NemotronH] Add latent MoE support | `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py` |
| 2026-01-05 | [#14051](https://github.com/sgl-project/sglang/pull/14051) | merged | EVS Framework: Support NemotronH_Nano_VL_V2 | `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py` |
| 2026-01-14 | [#17013](https://github.com/sgl-project/sglang/pull/17013) | merged | Feat/support nemotron h mtp | `python/sglang/srt/models/nemotron_h_mtp.py`, `python/sglang/srt/models/nemotron_h.py` |
| 2026-01-14 | [#16569](https://github.com/sgl-project/sglang/pull/16569) | merged | [NemotronH] Use ReplicatedLinear for fc1_latent_proj | `python/sglang/srt/models/nemotron_h.py` |
| 2026-02-06 | [#18119](https://github.com/sgl-project/sglang/pull/18119) | merged | Add Nemotron 3 Nano tests | `test/registered/models/test_nvidia_nemotron_3_nano.py`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml` |
| 2026-02-21 | [#18546](https://github.com/sgl-project/sglang/pull/18546) | merged | [Quantization] Support config.json quantization_config format, fix exclude_modules matching, and fix KV cache scale loading for Nemotron | `python/sglang/srt/models/nemotron_h.py` |
| 2026-03-03 | [#19433](https://github.com/sgl-project/sglang/pull/19433) | merged | Fix/nemotron mtp quantaized | `python/sglang/srt/models/nemotron_h_mtp.py` |
| 2026-03-07 | [#19950](https://github.com/sgl-project/sglang/pull/19950) | merged | Refactor NemotronHConfig to canonical layers_block_type and add MTP block-type support | `python/sglang/srt/configs/nemotron_h.py` |
| 2026-03-12 | [#19903](https://github.com/sgl-project/sglang/pull/19903) | merged | Enable Piecewise CUDA Graph for NemotronH Hybrid (Mamba+Attention) Models | `python/sglang/srt/models/nemotron_h.py` |
| 2026-03-14 | [#20407](https://github.com/sgl-project/sglang/pull/20407) | merged | [Model] Support Nemotron 3 Super NVFP4 | `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/quantization/__init__.py` |
| 2026-03-14 | [#20575](https://github.com/sgl-project/sglang/pull/20575) | merged | [CI] Add Nemotron 3 Super 120B CI tests for BF16 and NVFP4 | `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` |
| 2026-03-15 | [#20458](https://github.com/sgl-project/sglang/pull/20458) | merged | fix: Nemotron chunk size alias | `python/sglang/srt/configs/nemotron_h.py` |
| 2026-03-16 | [#20616](https://github.com/sgl-project/sglang/pull/20616) | merged | [CI] Add Nemotron 3 Super 120B nightly 8-GPU tests | `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` |
| 2026-03-17 | [#20580](https://github.com/sgl-project/sglang/pull/20580) | merged | [Model] Fix NemotronH OOM on unified-mem systems: stream weights | `python/sglang/srt/models/nemotron_h.py` |
| 2026-03-27 | [#21516](https://github.com/sgl-project/sglang/pull/21516) | merged | [CI] Fix nemotron nvfp4 test estimated time | `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` |
| 2026-04-25 | [#23568](https://github.com/sgl-project/sglang/pull/23568) | merged | Parakeet nemotron encoder | `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py` |
| 2026-04-28 | [#23907](https://github.com/sgl-project/sglang/pull/23907) | merged | [Docs] add Nemotron 3 Nano Omni cookbook | `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` |
| 2026-04-28 | [#23874](https://github.com/sgl-project/sglang/pull/23874) | merged | Fix failing `test_nvidia_nemotron_3_nano` by fixing `test_grouped_topk` | `python/sglang/srt/models/nemotron_h.py`, `test/registered/models/test_nvidia_nemotron_3_nano.py` |
| 2026-04-28 | [#23968](https://github.com/sgl-project/sglang/pull/23968) | merged | [Docs] update Docker image for Nemotron 3 Nano Omni | `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` |
| 2026-04-29 | [#23857](https://github.com/sgl-project/sglang/pull/23857) | merged | Nemotron-omni-v3-alias | `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py` |
| 2026-04-29 | [#21321](https://github.com/sgl-project/sglang/pull/21321) | merged | [Kernel] Support FlashInfer TRTLLM-Gen fused MoE for non-gated FP4 & FP8 (Nemotron) | `python/sglang/srt/models/nemotron_h.py` |
| 2026-04-30 | [#23594](https://github.com/sgl-project/sglang/pull/23594) | merged | LoRA support for qwen3.5 and nemotron3 | `python/sglang/srt/models/nemotron_h.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` |

## 逐 PR diff 审计卡

### PR #5073 - [Model] Add support for nvidia/Llama-3_3-Nemotron-Super-49B-v1

- 链接: https://github.com/sgl-project/sglang/pull/5073
- 状态/时间: closed / 2025-08-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+898/-1，可读 patch 929 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add support for nvidia/Llama-3_3-Nemotron-Super-49B-v1」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nemotron_nas.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/utils.py`；技术摘要: 覆盖「[Model] Add support for nvidia/Llama-3_3-Nemotron-Super-49B-v1」；主要实现面是 `python/sglang/srt/models/nemotron_nas.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_nas.py` added +516/-0 (516 lines); hunks: -0,0 +1,516; symbols: _ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer, __init__，涉及 `_ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer`；`python/sglang/srt/configs/model_config.py` modified +8/-0 (8 lines); hunks: -320,6 +320,14 @@ def get_total_num_kv_heads(self) -> int:; symbols: get_total_num_kv_heads，涉及 `get_total_num_kv_heads`；`python/sglang/srt/utils.py` modified +374/-1 (375 lines); hunks: -55,6 +55,8; -439,8 +441,10 @@ def set_cpu_offload_max_bytes(max_bytes: int) -> None:; symbols: set_cpu_offload_max_bytes, maybe_offload_to_cpu, LayerFn, __call__，涉及 `set_cpu_offload_max_bytes, maybe_offload_to_cpu, LayerFn`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_nas.py` added +516/-0 (516 lines); hunks: -0,0 +1,516; symbols: _ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer, __init__
  - `python/sglang/srt/configs/model_config.py` modified +8/-0 (8 lines); hunks: -320,6 +320,14 @@ def get_total_num_kv_heads(self) -> int:; symbols: get_total_num_kv_heads
  - `python/sglang/srt/utils.py` modified +374/-1 (375 lines); hunks: -55,6 +55,8; -439,8 +441,10 @@ def set_cpu_offload_max_bytes(max_bytes: int) -> None:; symbols: set_cpu_offload_max_bytes, maybe_offload_to_cpu, LayerFn, __call__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_nas.py
@@ -0,0 +1,516 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/model_config.py
@@ -320,6 +320,14 @@ def get_total_num_kv_heads(self) -> int:
+        if self.hf_config.model_type in ["nemotron-nas"]:
+            for block in self.hf_config.block_configs:
+                if not block.attention.no_op:
+                    return (
+                        self.hf_config.num_attention_heads
+                        // block.attention.n_heads_in_group
diff -- python/sglang/srt/utils.py
@@ -55,6 +55,8 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_nas.py` added +516/-0; `python/sglang/srt/configs/model_config.py` modified +8/-0; `python/sglang/srt/utils.py` modified +374/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/nemotron_nas.py`, `python/sglang/srt/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9067 - model: support nvidia/Llama-3_3-Nemotron-Super-49B-v1

- 链接: https://github.com/sgl-project/sglang/pull/9067
- 状态/时间: merged / 2025-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_nas.py`；关联提交 `845d12a979fb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+465/-5，可读 patch 505 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: support nvidia/Llama-3_3-Nemotron-Super-49B-v1」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/nemotron_nas.py`；技术摘要: 覆盖「model: support nvidia/Llama-3_3-Nemotron-Super-49B-v1」；主要实现面是 `python/sglang/srt/models/nemotron_nas.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_nas.py` added +435/-0 (435 lines); hunks: -0,0 +1,435; symbols: _ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer, __init__，涉及 `_ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_nas.py` added +435/-0 (435 lines); hunks: -0,0 +1,435; symbols: _ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_nas.py
@@ -0,0 +1,435 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_nas.py` added +435/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/runners.py`, `test/srt/models/test_generation_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10909 - model: Support Hybrid Mamba2 NemotronHForCausalLM (nvidia/NVIDIA-Nemotron-Nano-9B-v2)

- 链接: https://github.com/sgl-project/sglang/pull/10909
- 状态/时间: merged / 2025-10-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/models/nemotron_h.py`；关联提交 `d6837aea4d2c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+3279/-853，可读 patch 4929 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: Support Hybrid Mamba2 NemotronHForCausalLM (nvidia/NVIDIA-Nemotron-Nano-9B-v2)」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`；技术摘要: 覆盖「model: Support Hybrid Mamba2 NemotronHForCausalLM (nvidia/NVIDIA-Nemotron-Nano-9B-v2)」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` added +514/-0 (514 lines); hunks: -0,0 +1,514; symbols: NemotronHMLP, __init__, forward, NemotronHMLPDecoderLayer，涉及 `NemotronHMLP, __init__, forward`；`python/sglang/srt/configs/nemotron_h.py` added +286/-0 (286 lines); hunks: -0,0 +1,286; symbols: NemotronHConfig, to, __init__, mamba_layer_ids，涉及 `NemotronHConfig, to, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` added +514/-0 (514 lines); hunks: -0,0 +1,514; symbols: NemotronHMLP, __init__, forward, NemotronHMLPDecoderLayer
  - `python/sglang/srt/configs/nemotron_h.py` added +286/-0 (286 lines); hunks: -0,0 +1,286; symbols: NemotronHConfig, to, __init__, mamba_layer_ids
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -0,0 +1,514 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/nemotron_h.py
@@ -0,0 +1,286 @@
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` added +514/-0; `python/sglang/srt/configs/nemotron_h.py` added +286/-0
- 验证与风险: diff 自带测试面 `test/srt/layers/attention/mamba/test_causal_conv1d.py`, `test/srt/layers/attention/mamba/test_mamba2_mixer.py`, `test/srt/layers/attention/mamba/test_mamba_ssm.py`, `test/srt/layers/attention/mamba/test_mamba_ssm_ssd.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11866 - Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4

- 链接: https://github.com/sgl-project/sglang/pull/11866
- 状态/时间: merged / 2025-10-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `d6fee73d1f59`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+207/-127，可读 patch 628 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +19/-22 (41 lines); hunks: -48,6 +48,8; -155,6 +157,7 @@ def __init__(; symbols: __init__, forward, NemotronHForCausalLM, _init_model，涉及 `__init__, forward, NemotronHForCausalLM`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +19/-22 (41 lines); hunks: -48,6 +48,8; -155,6 +157,7 @@ def __init__(; symbols: __init__, forward, NemotronHForCausalLM, _init_model
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -48,6 +48,8 @@
+    replace_prefix,
+    replace_substrings,
@@ -155,6 +157,7 @@ def __init__(
+            prefix=f"{prefix}.mixer",
@@ -381,16 +384,19 @@ def forward(
+    stacked_params_mapping = [
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +19/-22
- 验证与风险: diff 自带测试面 `test/srt/layers/attention/mamba/test_causal_conv1d.py`, `test/srt/layers/attention/mamba/test_mamba2_mixer.py`, `test/srt/layers/attention/mamba/test_mamba_ssm.py`, `test/srt/layers/attention/mamba/test_mamba_ssm_ssd.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12015 - Revert "Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4"

- 链接: https://github.com/sgl-project/sglang/pull/12015
- 状态/时间: merged / 2025-10-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `6c18addb6f53`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+127/-207，可读 patch 628 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4"」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「Revert "Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4"」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +22/-19 (41 lines); hunks: -48,8 +48,6; -157,7 +155,6 @@ def __init__(; symbols: __init__, forward, NemotronHForCausalLM, _init_model，涉及 `__init__, forward, NemotronHForCausalLM`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +22/-19 (41 lines); hunks: -48,8 +48,6; -157,7 +155,6 @@ def __init__(; symbols: __init__, forward, NemotronHForCausalLM, _init_model
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -48,8 +48,6 @@
-    replace_prefix,
-    replace_substrings,
@@ -157,7 +155,6 @@ def __init__(
-            prefix=f"{prefix}.mixer",
@@ -384,19 +381,16 @@ def forward(
-    stacked_params_mapping = [
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +22/-19
- 验证与风险: diff 自带测试面 `test/srt/layers/attention/mamba/test_causal_conv1d.py`, `test/srt/layers/attention/mamba/test_mamba2_mixer.py`, `test/srt/layers/attention/mamba/test_mamba_ssm.py`, `test/srt/layers/attention/mamba/test_mamba_ssm_ssd.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12448 - Add Jet-Nemotron

- 链接: https://github.com/sgl-project/sglang/pull/12448
- 状态/时间: merged / 2025-11-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/jet_nemotron.py`, `python/sglang/srt/models/jet_nemotron.py`；关联提交 `3633f8b0cfef`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+678/-2，可读 patch 733 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Jet-Nemotron」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/jet_nemotron.py`, `python/sglang/srt/configs/jet_nemotron.py`；技术摘要: 覆盖「Add Jet-Nemotron」；主要实现面是 `python/sglang/srt/models/jet_nemotron.py`, `python/sglang/srt/configs/jet_nemotron.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/jet_nemotron.py` added +596/-0 (596 lines); hunks: -0,0 +1,596; symbols: DynamicShortConvolutionKernelGenerator, __init__, forward, DynamicShortConvolution，涉及 `DynamicShortConvolutionKernelGenerator, __init__, forward`；`python/sglang/srt/configs/jet_nemotron.py` added +74/-0 (74 lines); hunks: -0,0 +1,74; symbols: JetBlockConfig, JetNemotronConfig, full_attention_layer_ids, linear_layer_ids，涉及 `JetBlockConfig, JetNemotronConfig, full_attention_layer_ids`。
- 代码 diff 细节:
  - `python/sglang/srt/models/jet_nemotron.py` added +596/-0 (596 lines); hunks: -0,0 +1,596; symbols: DynamicShortConvolutionKernelGenerator, __init__, forward, DynamicShortConvolution
  - `python/sglang/srt/configs/jet_nemotron.py` added +74/-0 (74 lines); hunks: -0,0 +1,74; symbols: JetBlockConfig, JetNemotronConfig, full_attention_layer_ids, linear_layer_ids
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/jet_nemotron.py
@@ -0,0 +1,596 @@
+from collections.abc import Iterable
+from typing import cast
+import einops
+import torch
+import torch.nn as nn
+from sglang.srt.configs.jet_nemotron import JetBlockConfig, JetNemotronConfig
diff -- python/sglang/srt/configs/jet_nemotron.py
@@ -0,0 +1,74 @@
+from dataclasses import dataclass
+from typing import Any
+from transformers.configuration_utils import PretrainedConfig
+from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
+@dataclass
+class JetBlockConfig:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/jet_nemotron.py` added +596/-0; `python/sglang/srt/configs/jet_nemotron.py` added +74/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/test_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12690 - Feat/nemotron nano v3 support

- 链接: https://github.com/sgl-project/sglang/pull/12690
- 状态/时间: merged / 2025-11-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/models/nemotron_h.py`；关联提交 `1b48e1b97484`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+775/-67，可读 patch 1291 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Feat/nemotron nano v3 support」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`；技术摘要: 覆盖「Feat/nemotron nano v3 support」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +259/-28 (287 lines); hunks: -22,8 +22,13; -34,9 +39,13; symbols: NemotronHMLP, __init__, forward, _get_or_create_alt_stream，涉及 `NemotronHMLP, __init__, forward`；`python/sglang/srt/configs/nemotron_h.py` modified +25/-6 (31 lines); hunks: -26,6 +26,7; -189,6 +190,15 @@ def __init__(; symbols: NemotronHConfig, __init__，涉及 `NemotronHConfig, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +259/-28 (287 lines); hunks: -22,8 +22,13; -34,9 +39,13; symbols: NemotronHMLP, __init__, forward, _get_or_create_alt_stream
  - `python/sglang/srt/configs/nemotron_h.py` modified +25/-6 (31 lines); hunks: -26,6 +26,7; -189,6 +190,15 @@ def __init__(; symbols: NemotronHConfig, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -22,8 +22,13 @@
-from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA, MLP
-from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
+from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA, MLP, MOE
+from sglang.srt.distributed import (
+    get_moe_ep_group,
+    get_pp_group,
diff -- python/sglang/srt/configs/nemotron_h.py
@@ -26,6 +26,7 @@
+MOE = "E"
@@ -189,6 +190,15 @@ def __init__(
+        n_routed_experts=8,
+        n_shared_experts=1,
+        moe_intermediate_size=7688,
+        moe_shared_expert_intermediate_size=7688,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +259/-28; `python/sglang/srt/configs/nemotron_h.py` modified +25/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=1856,device_name=NVIDIA_H100_80GB_HBM3.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=1856,device_name=NVIDIA_L40S.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12277 - Support nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 (and nvidia/C-RADIOv2-H)

- 链接: https://github.com/sgl-project/sglang/pull/12277
- 状态/时间: merged / 2025-11-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`；关联提交 `082b54c6890a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+1334/-17，可读 patch 1528 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 (and nvidia/C-RADIOv2-H)」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`；技术摘要: 覆盖「Support nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 (and nvidia/C-RADIOv2-H)」；主要实现面是 `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nano_nemotron_vl.py` added +219/-0 (219 lines); hunks: -0,0 +1,219; symbols: NemotronH_Nano_VL_V2, __init__, pad_input_ids, pixel_shuffle，涉及 `NemotronH_Nano_VL_V2, __init__, pad_input_ids`；`python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` added +197/-0 (197 lines); hunks: -0,0 +1,197; symbols: NanoNemotronVLImageProcessor, __init__, preprocess_image, render_image，涉及 `NanoNemotronVLImageProcessor, __init__, preprocess_image`；`python/sglang/srt/configs/nano_nemotron_vl.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: float_triplet, NemotronH_Nano_VL_V2_Config, __init__, create_radio_config，涉及 `float_triplet, NemotronH_Nano_VL_V2_Config, __init__`；`python/sglang/srt/models/nemotron_h.py` modified +3/-6 (9 lines); hunks: -542,9 +542,6 @@ def get_layer(idx: int, prefix: str):; -557,7 +554,7 @@ def forward(; symbols: get_layer, get_input_embeddings, forward, _init_model，涉及 `get_layer, get_input_embeddings, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nano_nemotron_vl.py` added +219/-0 (219 lines); hunks: -0,0 +1,219; symbols: NemotronH_Nano_VL_V2, __init__, pad_input_ids, pixel_shuffle
  - `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` added +197/-0 (197 lines); hunks: -0,0 +1,197; symbols: NanoNemotronVLImageProcessor, __init__, preprocess_image, render_image
  - `python/sglang/srt/configs/nano_nemotron_vl.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: float_triplet, NemotronH_Nano_VL_V2_Config, __init__, create_radio_config
  - `python/sglang/srt/models/nemotron_h.py` modified +3/-6 (9 lines); hunks: -542,9 +542,6 @@ def get_layer(idx: int, prefix: str):; -557,7 +554,7 @@ def forward(; symbols: get_layer, get_input_embeddings, forward, _init_model
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nano_nemotron_vl.py
@@ -0,0 +1,219 @@
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/multimodal/processors/nano_nemotron_vl.py
@@ -0,0 +1,197 @@
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/nano_nemotron_vl.py
@@ -0,0 +1,114 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nano_nemotron_vl.py` added +219/-0; `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` added +197/-0; `python/sglang/srt/configs/nano_nemotron_vl.py` added +114/-0; `python/sglang/srt/models/nemotron_h.py` modified +3/-6
- 验证与风险: diff 自带测试面 `test/srt/models/test_nvidia_nemotron_nano_v2_vl.py`, `test/srt/run_suite.py`, `test/srt/test_video_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16172 - [NemotronH] PP support

- 链接: https://github.com/sgl-project/sglang/pull/16172
- 状态/时间: merged / 2025-12-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `47a660d5b925`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+94/-35，可读 patch 207 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NemotronH] PP support」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「[NemotronH] PP support」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +88/-35 (123 lines); hunks: -48,6 +48,7; -65,7 +66,7; symbols: __init__, get_layer, forward，涉及 `__init__, get_layer, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +88/-35 (123 lines); hunks: -48,6 +48,7; -65,7 +66,7; symbols: __init__, get_layer, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -48,6 +48,7 @@
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
@@ -65,7 +66,7 @@
-    make_layers_non_pp,
+    make_layers,
@@ -526,21 +527,32 @@ def __init__(
+        self.pp_group = get_pp_group()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +88/-35
- 验证与风险: diff 自带测试面 `test/srt/models/test_nvidia_nemotron_nano_v2.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16227 - [NemotronH] Add latent MoE support

- 链接: https://github.com/sgl-project/sglang/pull/16227
- 状态/时间: merged / 2026-01-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/models/nemotron_h.py`；关联提交 `b0213323397c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 23 个文件，+2957/-2，可读 patch 3056 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NemotronH] Add latent MoE support」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`；技术摘要: 覆盖「[NemotronH] Add latent MoE support」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +32/-1 (33 lines); hunks: -138,6 +138,10 @@ def __init__(; -165,7 +169,7 @@ def __init__(; symbols: __init__, _forward_core, _forward_core_normal，涉及 `__init__, _forward_core, _forward_core_normal`；`python/sglang/srt/configs/nemotron_h.py` modified +2/-0 (2 lines); hunks: -194,6 +194,7 @@ def __init__(; -259,6 +260,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +32/-1 (33 lines); hunks: -138,6 +138,10 @@ def __init__(; -165,7 +169,7 @@ def __init__(; symbols: __init__, _forward_core, _forward_core_normal
  - `python/sglang/srt/configs/nemotron_h.py` modified +2/-0 (2 lines); hunks: -194,6 +194,7 @@ def __init__(; -259,6 +260,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -138,6 +138,10 @@ def __init__(
+        self.use_latent_moe = getattr(config, "moe_latent_size", None) is not None
+        self.moe_hidden_size = (
+            config.moe_latent_size if self.use_latent_moe else config.hidden_size
+        )
@@ -165,7 +169,7 @@ def __init__(
-            hidden_size=config.hidden_size,
diff -- python/sglang/srt/configs/nemotron_h.py
@@ -194,6 +194,7 @@ def __init__(
+        moe_latent_size=None,
@@ -259,6 +260,7 @@ def __init__(
+        self.moe_latent_size = moe_latent_size
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +32/-1; `python/sglang/srt/configs/nemotron_h.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_B200.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14051 - EVS Framework: Support NemotronH_Nano_VL_V2

- 链接: https://github.com/sgl-project/sglang/pull/14051
- 状态/时间: merged / 2026-01-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`；关联提交 `bebd625ba145`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+821/-56，可读 patch 1171 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「EVS Framework: Support NemotronH_Nano_VL_V2」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`；技术摘要: 覆盖「EVS Framework: Support NemotronH_Nano_VL_V2」；主要实现面是 `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +35/-22 (57 lines); hunks: -11,14 +11,16; -40,6 +42,9 @@ class NanoNemotronVLImageProcessor(BaseMultimodalProcessor):; symbols: NanoNemotronVLImageProcessor, __init__, preprocess_image, render_image，涉及 `NanoNemotronVLImageProcessor, __init__, preprocess_image`；`python/sglang/srt/models/nano_nemotron_vl.py` modified +7/-2 (9 lines); hunks: -36,19 +36,24; symbols: NemotronH_Nano_VL_V2, create_evs_config, __init__，涉及 `NemotronH_Nano_VL_V2, create_evs_config, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +35/-22 (57 lines); hunks: -11,14 +11,16; -40,6 +42,9 @@ class NanoNemotronVLImageProcessor(BaseMultimodalProcessor):; symbols: NanoNemotronVLImageProcessor, __init__, preprocess_image, render_image
  - `python/sglang/srt/models/nano_nemotron_vl.py` modified +7/-2 (9 lines); hunks: -36,19 +36,24; symbols: NemotronH_Nano_VL_V2, create_evs_config, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/nano_nemotron_vl.py
@@ -11,14 +11,16 @@
+from math import sqrt
-from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
+from sglang.srt.configs.nano_nemotron_vl import NemotronH_Nano_VL_V2_Config
+from sglang.srt.multimodal.evs import EVSProcessor
@@ -40,6 +42,9 @@ class NanoNemotronVLImageProcessor(BaseMultimodalProcessor):
+        self.evs = EVSProcessor(
diff -- python/sglang/srt/models/nano_nemotron_vl.py
@@ -36,19 +36,24 @@
+from sglang.srt.multimodal.evs import EVS, EVSConfig
-class NemotronH_Nano_VL_V2(nn.Module):
+class NemotronH_Nano_VL_V2(EVS):
+    @staticmethod
+    def create_evs_config(config: NemotronH_Nano_VL_V2_Config):
+        return EVSConfig(video_pruning_rate=config.video_pruning_rate)
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +35/-22; `python/sglang/srt/models/nano_nemotron_vl.py` modified +7/-2
- 验证与风险: diff 自带测试面 `python/sglang/test/test_utils.py`, `test/srt/run_suite.py`, `test/srt/test_evs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17013 - Feat/support nemotron h mtp

- 链接: https://github.com/sgl-project/sglang/pull/17013
- 状态/时间: merged / 2026-01-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_mtp.py`；关联提交 `ba625c2d908a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+378/-1，可读 patch 408 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Feat/support nemotron h mtp」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nemotron_h_mtp.py`, `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「Feat/support nemotron h mtp」；主要实现面是 `python/sglang/srt/models/nemotron_h_mtp.py`, `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h_mtp.py` added +340/-0 (340 lines); hunks: -0,0 +1,340; symbols: NemotronHMTPAttentionDecoderLayer, __init__, forward, NemotronHMTPMoEDecoderLayer，涉及 `NemotronHMTPAttentionDecoderLayer, __init__, forward`；`python/sglang/srt/models/nemotron_h.py` modified +28/-1 (29 lines); hunks: -728,7 +728,20 @@ def copy_inputs_before_cuda_graphs(self, input_buffers, **k...; -749,6 +762,20 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: copy_inputs_before_cuda_graphs, get_seqlen_agnostic_capture_inputs, load_weights, get_embed_and_head，涉及 `copy_inputs_before_cuda_graphs, get_seqlen_agnostic_capture_inputs, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h_mtp.py` added +340/-0 (340 lines); hunks: -0,0 +1,340; symbols: NemotronHMTPAttentionDecoderLayer, __init__, forward, NemotronHMTPMoEDecoderLayer
  - `python/sglang/srt/models/nemotron_h.py` modified +28/-1 (29 lines); hunks: -728,7 +728,20 @@ def copy_inputs_before_cuda_graphs(self, input_buffers, **k...; -749,6 +762,20 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: copy_inputs_before_cuda_graphs, get_seqlen_agnostic_capture_inputs, load_weights, get_embed_and_head
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h_mtp.py
@@ -0,0 +1,340 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/nemotron_h.py
@@ -728,7 +728,20 @@ def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
-    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
+    def get_embed_and_head(self):
+        return self.model.embed_tokens.weight, self.lm_head.weight
+    def set_embed_and_head(self, embed, head):
+        del self.model.embed_tokens.weight
+        del self.lm_head.weight
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h_mtp.py` added +340/-0; `python/sglang/srt/models/nemotron_h.py` modified +28/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16569 - [NemotronH] Use ReplicatedLinear for fc1_latent_proj

- 链接: https://github.com/sgl-project/sglang/pull/16569
- 状态/时间: merged / 2026-01-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `72bacc88c8a0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NemotronH] Use ReplicatedLinear for fc1_latent_proj」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「[NemotronH] Use ReplicatedLinear for fc1_latent_proj」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +1/-2 (3 lines); hunks: -191,12 +191,11 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +1/-2 (3 lines); hunks: -191,12 +191,11 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -191,12 +191,11 @@ def __init__(
-            self.fc1_latent_proj = ColumnParallelLinear(
+            self.fc1_latent_proj = ReplicatedLinear(
-                gather_output=True,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +1/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18119 - Add Nemotron 3 Nano tests

- 链接: https://github.com/sgl-project/sglang/pull/18119
- 状态/时间: merged / 2026-02-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml`, `test/registered/models/test_nvidia_nemotron_3_nano.py`；关联提交 `c6aa1863be84`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+177/-0，可读 patch 188 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Nemotron 3 Nano tests」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `test/registered/models/test_nvidia_nemotron_3_nano.py`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml`；技术摘要: 覆盖「Add Nemotron 3 Nano tests」；主要实现面是 `test/registered/models/test_nvidia_nemotron_3_nano.py`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models/test_nvidia_nemotron_3_nano.py` added +41/-0 (41 lines); hunks: -0,0 +1,41; symbols: TestNvidiaNemotron3Nano30BBF16, TestNvidiaNemotron3Nano30BFP8，涉及 `TestNvidiaNemotron3Nano30BBF16, TestNvidiaNemotron3Nano30BFP8`；`test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml` added +13/-0 (13 lines); hunks: -0,0 +1,13；`test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml` added +13/-0 (13 lines); hunks: -0,0 +1,13。
- 代码 diff 细节:
  - `test/registered/models/test_nvidia_nemotron_3_nano.py` added +41/-0 (41 lines); hunks: -0,0 +1,41; symbols: TestNvidiaNemotron3Nano30BBF16, TestNvidiaNemotron3Nano30BFP8
  - `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml` added +13/-0 (13 lines); hunks: -0,0 +1,13
  - `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml` added +13/-0 (13 lines); hunks: -0,0 +1,13
- 关键代码摘录:

```diff
diff -- test/registered/models/test_nvidia_nemotron_3_nano.py
@@ -0,0 +1,41 @@
+import unittest
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.kits.lm_eval_kit import LMEvalMixin
+from sglang.test.server_fixtures.default_fixture import DefaultServerBase
+register_cuda_ci(est_time=180, suite="stage-b-test-large-2-gpu")
+NEMOTRON_3_NANO_THINKING_ARGS = [
diff -- test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml
@@ -0,0 +1,13 @@
+model_name: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
+tasks:
+- name: "gsm8k"
+  metrics:
+  - name: "exact_match,strict-match"
+    value: 0.847
diff -- test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml
@@ -0,0 +1,13 @@
```

- 已读文件:
  - tests: `test/registered/models/test_nvidia_nemotron_3_nano.py` added +41/-0; `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml` added +13/-0; `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml` added +13/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/lm_eval_kit.py`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml`, `test/registered/models/test_nvidia_nemotron_3_nano.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18546 - [Quantization] Support config.json quantization_config format, fix exclude_modules matching, and fix KV cache scale loading for Nemotron

- 链接: https://github.com/sgl-project/sglang/pull/18546
- 状态/时间: merged / 2026-02-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `33c33a7de9bb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+100/-71，可读 patch 251 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] Support config.json quantization_config format, fix exclude_modules matching, and fix KV cache scale loading for Nemotron」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「[Quantization] Support config.json quantization_config format, fix exclude_modules matching, and fix KV cache scale loading for Nemotron」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +7/-0 (7 lines); hunks: -61,6 +61,7; -640,6 +641,12 @@ class NemotronHForCausalLM(nn.Module):; symbols: NemotronHForCausalLM, __init__，涉及 `NemotronHForCausalLM, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +7/-0 (7 lines); hunks: -61,6 +61,7; -640,6 +641,12 @@ class NemotronHForCausalLM(nn.Module):; symbols: NemotronHForCausalLM, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -61,6 +61,7 @@
+from sglang.srt.models.utils import WeightsMapper
@@ -640,6 +641,12 @@ class NemotronHForCausalLM(nn.Module):
+    hf_to_sglang_mapper = WeightsMapper(
+        orig_to_new_prefix={
+            "backbone.": "model.",
+        }
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19433 - Fix/nemotron mtp quantaized

- 链接: https://github.com/sgl-project/sglang/pull/19433
- 状态/时间: merged / 2026-03-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h_mtp.py`；关联提交 `4c95953b7733`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+73/-3，可读 patch 117 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix/nemotron mtp quantaized」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/nemotron_h_mtp.py`；技术摘要: 覆盖「Fix/nemotron mtp quantaized」；主要实现面是 `python/sglang/srt/models/nemotron_h_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h_mtp.py` modified +1/-1 (2 lines); hunks: -297,7 +297,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h_mtp.py` modified +1/-1 (2 lines); hunks: -297,7 +297,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h_mtp.py
@@ -297,7 +297,7 @@ def __init__(
-            prefix=add_prefix("model", prefix),
+            prefix=add_prefix("mtp", prefix),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h_mtp.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/model_loading/test_modelopt_loader.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19950 - Refactor NemotronHConfig to canonical layers_block_type and add MTP block-type support

- 链接: https://github.com/sgl-project/sglang/pull/19950
- 状态/时间: merged / 2026-03-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`；关联提交 `f8bbf56de7b2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+182/-17，可读 patch 281 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Refactor NemotronHConfig to canonical layers_block_type and add MTP block-type support」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/configs/nemotron_h.py`；技术摘要: 覆盖「Refactor NemotronHConfig to canonical layers_block_type and add MTP block-type support」；主要实现面是 `python/sglang/srt/configs/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/nemotron_h.py` modified +182/-17 (199 lines); hunks: -15,7 +15,6; -31,6 +30,8; symbols: NemotronHConfig, _validate_layers_block_type, _resolve_layers_block_type，涉及 `NemotronHConfig, _validate_layers_block_type, _resolve_layers_block_type`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/nemotron_h.py` modified +182/-17 (199 lines); hunks: -15,7 +15,6; -31,6 +30,8; symbols: NemotronHConfig, _validate_layers_block_type, _resolve_layers_block_type
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/nemotron_h.py
@@ -15,7 +15,6 @@
-import regex as re
@@ -31,6 +30,8 @@
+DEFAULT_LAYERS_BLOCK_TYPE = ["mamba", "moe", "attention", "moe"]
+DEFAULT_MTP_LAYERS_BLOCK_TYPE = ["attention", "moe"]
@@ -53,13 +54,17 @@ class NemotronHConfig(PretrainedConfig):
-        num_hidden_layers (`int`, *optional*, defaults to 52):
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/nemotron_h.py` modified +182/-17
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19903 - Enable Piecewise CUDA Graph for NemotronH Hybrid (Mamba+Attention) Models

- 链接: https://github.com/sgl-project/sglang/pull/19903
- 状态/时间: merged / 2026-03-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `25bd83033d09`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+91/-24，可读 patch 188 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable Piecewise CUDA Graph for NemotronH Hybrid (Mamba+Attention) Models」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「Enable Piecewise CUDA Graph for NemotronH Hybrid (Mamba+Attention) Models」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +70/-18 (88 lines); hunks: -21,6 +21,11; -69,6 +74,7; symbols: _forward_core, __init__, _forward_mamba, forward，涉及 `_forward_core, __init__, _forward_mamba`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +70/-18 (88 lines); hunks: -21,6 +21,11; -69,6 +74,7; symbols: _forward_core, __init__, _forward_mamba, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -21,6 +21,11 @@
+from sglang.srt.compilation.compilation_config import register_split_op
+from sglang.srt.compilation.piecewise_context_manager import (
+    get_forward_context,
+    is_in_piecewise_cuda_graph,
+)
@@ -69,6 +74,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +70/-18
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20407 - [Model] Support Nemotron 3 Super NVFP4

- 链接: https://github.com/sgl-project/sglang/pull/20407
- 状态/时间: merged / 2026-03-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+277/-11，可读 patch 413 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support Nemotron 3 Super NVFP4」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/quantization/__init__.py`；技术摘要: 覆盖「[Model] Support Nemotron 3 Super NVFP4」；主要实现面是 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/quantization/__init__.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +177/-0 (177 lines); hunks: -591,6 +591,183 @@ def __init__(self, quant_config: ModelOptFp8Config):; symbols: __init__, ModelOptMixedPrecisionConfig, override_quantization_method, get_name，涉及 `__init__, ModelOptMixedPrecisionConfig, override_quantization_method`；`python/sglang/srt/configs/model_config.py` modified +12/-0 (12 lines); hunks: -793,6 +793,11 @@ def _parse_modelopt_quant_config(self, quant_config_dict: d...; -842,6 +847,10 @@ def _get_modelopt_quant_type(self) -> str:; symbols: _parse_modelopt_quant_config, _get_modelopt_quant_type, _validate_quantize_and_serve_config, _verify_quantization，涉及 `_parse_modelopt_quant_config, _get_modelopt_quant_type, _validate_quantize_and_serve_config`；`python/sglang/srt/layers/quantization/__init__.py` modified +2/-0 (2 lines); hunks: -31,6 +31,7 @@ def override_quantization_method(self, *args, **kwargs):; -57,6 +58,7 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method，涉及 `override_quantization_method`；`test/registered/model_loading/test_modelopt_loader.py` modified +65/-0 (65 lines); hunks: -14,7 +14,11; -620,5 +624,66 @@ def test_non_modelopt_quant_method_unchanged(self):; symbols: test_non_modelopt_quant_method_unchanged, TestModelOptMixedPrecisionConfig, test_nemotron_mixed_precision_uses_modelopt_mixed, test_mixed_precision_override_does_not_hijack_w4afp8，涉及 `test_non_modelopt_quant_method_unchanged, TestModelOptMixedPrecisionConfig, test_nemotron_mixed_precision_uses_modelopt_mixed`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +177/-0 (177 lines); hunks: -591,6 +591,183 @@ def __init__(self, quant_config: ModelOptFp8Config):; symbols: __init__, ModelOptMixedPrecisionConfig, override_quantization_method, get_name
  - `python/sglang/srt/configs/model_config.py` modified +12/-0 (12 lines); hunks: -793,6 +793,11 @@ def _parse_modelopt_quant_config(self, quant_config_dict: d...; -842,6 +847,10 @@ def _get_modelopt_quant_type(self) -> str:; symbols: _parse_modelopt_quant_config, _get_modelopt_quant_type, _validate_quantize_and_serve_config, _verify_quantization
  - `python/sglang/srt/layers/quantization/__init__.py` modified +2/-0 (2 lines); hunks: -31,6 +31,7 @@ def override_quantization_method(self, *args, **kwargs):; -57,6 +58,7 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method
  - `test/registered/model_loading/test_modelopt_loader.py` modified +65/-0 (65 lines); hunks: -14,7 +14,11; -620,5 +624,66 @@ def test_non_modelopt_quant_method_unchanged(self):; symbols: test_non_modelopt_quant_method_unchanged, TestModelOptMixedPrecisionConfig, test_nemotron_mixed_precision_uses_modelopt_mixed, test_mixed_precision_override_does_not_hijack_w4afp8
  - `python/sglang/srt/server_args.py` modified +17/-9 (26 lines); hunks: -105,6 +105,7; -1546,7 +1547,8 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, _handle_moe_kernel_config
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -591,6 +591,183 @@ def __init__(self, quant_config: ModelOptFp8Config):
+class ModelOptMixedPrecisionConfig(ModelOptQuantConfig):
+    """Configuration for ModelOpt MIXED_PRECISION checkpoints."""
+    def __init__(
+        self,
+        kv_cache_quant_algo: Optional[str],
+        exclude_modules: Optional[List[str]],
diff -- python/sglang/srt/configs/model_config.py
@@ -793,6 +793,11 @@ def _parse_modelopt_quant_config(self, quant_config_dict: dict) -> Optional[dict
+            architectures = getattr(self.hf_config, "architectures", []) or []
+            if getattr(self.hf_config, "model_type", None) == "nemotron_h" or any(
+                arch.startswith("NemotronH") for arch in architectures
+            ):
+                return {"quant_method": "modelopt_mixed", "quant_algo": quant_algo}
@@ -842,6 +847,10 @@ def _get_modelopt_quant_type(self) -> str:
diff -- python/sglang/srt/layers/quantization/__init__.py
@@ -31,6 +31,7 @@ def override_quantization_method(self, *args, **kwargs):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +177/-0; `python/sglang/srt/configs/model_config.py` modified +12/-0; `python/sglang/srt/layers/quantization/__init__.py` modified +2/-0; `python/sglang/srt/server_args.py` modified +17/-9; `python/sglang/srt/model_loader/loader.py` modified +4/-2
  - tests: `test/registered/model_loading/test_modelopt_loader.py` modified +65/-0
- 验证与风险: diff 自带测试面 `test/registered/model_loading/test_modelopt_loader.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20575 - [CI] Add Nemotron 3 Super 120B CI tests for BF16 and NVFP4

- 链接: https://github.com/sgl-project/sglang/pull/20575
- 状态/时间: merged / 2026-03-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`；关联提交 `3e643967e6d7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+212/-0，可读 patch 214 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add Nemotron 3 Super 120B CI tests for BF16 and NVFP4」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`；技术摘要: 覆盖「[CI] Add Nemotron 3 Super 120B CI tests for BF16 and NVFP4」；主要实现面是 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: _run_gsm8k, TestNvidiaNemotron3SuperNVFP4, setUpClass, tearDownClass，涉及 `_run_gsm8k, TestNvidiaNemotron3SuperNVFP4, setUpClass`；`test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: _run_gsm8k, TestNvidiaNemotron3SuperBF16, setUpClass, tearDownClass，涉及 `_run_gsm8k, TestNvidiaNemotron3SuperBF16, setUpClass`。
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: _run_gsm8k, TestNvidiaNemotron3SuperNVFP4, setUpClass, tearDownClass
  - `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: _run_gsm8k, TestNvidiaNemotron3SuperBF16, setUpClass, tearDownClass
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py
@@ -0,0 +1,106 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.run_eval import run_eval
+from sglang.test.test_utils import (
diff -- test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py
@@ -0,0 +1,106 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.run_eval import run_eval
+from sglang.test.test_utils import (
```

- 已读文件:
  - tests: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` added +106/-0; `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` added +106/-0
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20458 - fix: Nemotron chunk size alias

- 链接: https://github.com/sgl-project/sglang/pull/20458
- 状态/时间: merged / 2026-03-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`；关联提交 `1ac6a2646437`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+26/-1，可读 patch 55 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: Nemotron chunk size alias」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/configs/nemotron_h.py`；技术摘要: 覆盖「fix: Nemotron chunk size alias」；主要实现面是 `python/sglang/srt/configs/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/nemotron_h.py` modified +26/-1 (27 lines); hunks: -32,6 +32,7; -213,6 +214,28 @@ def _resolve_mtp_layers_block_type(mtp_layers_block_type, k...; symbols: NemotronHConfig, _resolve_mtp_layers_block_type, _resolve_mamba_chunk_size, __init__，涉及 `NemotronHConfig, _resolve_mtp_layers_block_type, _resolve_mamba_chunk_size`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/nemotron_h.py` modified +26/-1 (27 lines); hunks: -32,6 +32,7; -213,6 +214,28 @@ def _resolve_mtp_layers_block_type(mtp_layers_block_type, k...; symbols: NemotronHConfig, _resolve_mtp_layers_block_type, _resolve_mamba_chunk_size, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/nemotron_h.py
@@ -32,6 +32,7 @@
+DEFAULT_MAMBA_CHUNK_SIZE = 256
@@ -213,6 +214,28 @@ def _resolve_mtp_layers_block_type(mtp_layers_block_type, kwargs) -> list[str]:
+    @staticmethod
+    def _resolve_mamba_chunk_size(mamba_chunk_size, kwargs) -> int:
+        """Resolve canonical mamba_chunk_size from new and legacy config fields."""
+        chunk_size = kwargs.pop("chunk_size", None)
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/nemotron_h.py` modified +26/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20616 - [CI] Add Nemotron 3 Super 120B nightly 8-GPU tests

- 链接: https://github.com/sgl-project/sglang/pull/20616
- 状态/时间: merged / 2026-03-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`；关联提交 `3879c466b432`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+145/-6，可读 patch 180 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add Nemotron 3 Super 120B nightly 8-GPU tests」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`；技术摘要: 覆盖「[CI] Add Nemotron 3 Super 120B nightly 8-GPU tests」；主要实现面是 `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` added +135/-0 (135 lines); hunks: -0,0 +1,135; symbols: TestNvidiaNemotron3SuperNightly, for, test_nemotron_3_super_bf16, test_nemotron_3_super_nvfp4，涉及 `TestNvidiaNemotron3SuperNightly, for, test_nemotron_3_super_bf16`；`test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +5/-3 (8 lines); hunks: -37,6 +37,10; -89,9 +93,7 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`；`test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` modified +5/-3 (8 lines); hunks: -37,6 +37,10; -89,9 +93,7 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` added +135/-0 (135 lines); hunks: -0,0 +1,135; symbols: TestNvidiaNemotron3SuperNightly, for, test_nemotron_3_super_bf16, test_nemotron_3_super_nvfp4
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +5/-3 (8 lines); hunks: -37,6 +37,10; -89,9 +93,7 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` modified +5/-3 (8 lines); hunks: -37,6 +37,10; -89,9 +93,7 @@ def setUpClass(cls):; symbols: setUpClass
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py
@@ -0,0 +1,135 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system
diff -- test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py
@@ -37,6 +37,10 @@
+    "--max-running-requests",
+    "200",
+    "--mem-fraction-static",
+    "0.75",
@@ -89,9 +93,7 @@ def setUpClass(cls):
-            other_args=NEMOTRON_3_SUPER_NVFP4_ARGS
diff -- test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py
@@ -37,6 +37,10 @@
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` added +135/-0; `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +5/-3; `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` modified +5/-3
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20580 - [Model] Fix NemotronH OOM on unified-mem systems: stream weights

- 链接: https://github.com/sgl-project/sglang/pull/20580
- 状态/时间: merged / 2026-03-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `466ff20e5148`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-7，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Fix NemotronH OOM on unified-mem systems: stream weights」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「[Model] Fix NemotronH OOM on unified-mem systems: stream weights」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +7/-7 (14 lines); hunks: -774,12 +774,6 @@ def set_embed_and_head(self, embed, head):; -793,7 +787,13 @@ def load_weights(; symbols: set_embed_and_head, load_weights，涉及 `set_embed_and_head, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +7/-7 (14 lines); hunks: -774,12 +774,6 @@ def set_embed_and_head(self, embed, head):; -793,7 +787,13 @@ def load_weights(; symbols: set_embed_and_head, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -774,12 +774,6 @@ def set_embed_and_head(self, embed, head):
-        updated_weights = []
-        for name, loaded_weight in weights:
-            name = replace_prefix(name, self.remap_prefix)
-            name = replace_substrings(name, self.remap_substr)
-            updated_weights.append((name, loaded_weight))
@@ -793,7 +787,13 @@ def load_weights(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +7/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21516 - [CI] Fix nemotron nvfp4 test estimated time

- 链接: https://github.com/sgl-project/sglang/pull/21516
- 状态/时间: merged / 2026-03-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；关联提交 `0138129d3cfc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Fix nemotron nvfp4 test estimated time」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；技术摘要: 覆盖「[CI] Fix nemotron nvfp4 test estimated time」；主要实现面是 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7。
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py
@@ -11,7 +11,7 @@
-register_cuda_ci(est_time=300, suite="stage-c-test-4-gpu-b200")
+register_cuda_ci(est_time=600, suite="stage-c-test-4-gpu-b200")
```

- 已读文件:
  - tests: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23568 - Parakeet nemotron encoder

- 链接: https://github.com/sgl-project/sglang/pull/23568
- 状态/时间: merged / 2026-04-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`；关联提交 `4a3fe2a0913c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+1289/-116，可读 patch 1817 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Parakeet nemotron encoder」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +322/-36 (358 lines); hunks: -11,23 +11,39; -63,18 +79,62 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__, preprocess_image, render_image, render_image_dynamic，涉及 `__init__, preprocess_image, render_image`；`python/sglang/srt/models/nano_nemotron_vl.py` modified +171/-20 (191 lines); hunks: -35,8 +35,10; -66,9 +68,13 @@ def __init__(; symbols: __init__, pad_input_ids, pixel_shuffle，涉及 `__init__, pad_input_ids, pixel_shuffle`；`python/sglang/srt/configs/nano_nemotron_vl.py` modified +38/-0 (38 lines); hunks: -38,6 +38,7 @@ def __init__(; -51,6 +52,9 @@ def __init__(; symbols: __init__, create_radio_config，涉及 `__init__, create_radio_config`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +322/-36 (358 lines); hunks: -11,23 +11,39; -63,18 +79,62 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__, preprocess_image, render_image, render_image_dynamic
  - `python/sglang/srt/models/nano_nemotron_vl.py` modified +171/-20 (191 lines); hunks: -35,8 +35,10; -66,9 +68,13 @@ def __init__(; symbols: __init__, pad_input_ids, pixel_shuffle
  - `python/sglang/srt/configs/nano_nemotron_vl.py` modified +38/-0 (38 lines); hunks: -38,6 +38,7 @@ def __init__(; -51,6 +52,9 @@ def __init__(; symbols: __init__, create_radio_config
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/nano_nemotron_vl.py
@@ -11,23 +11,39 @@
+import logging
+import math
-from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
+from sglang.srt.managers.schedule_batch import (
+    Modality,
+    MultimodalDataItem,
diff -- python/sglang/srt/models/nano_nemotron_vl.py
@@ -35,8 +35,10 @@
+from sglang.srt.models.parakeet import ProjectedParakeet
+from sglang.srt.multimodal.evs.evs_module import VideoEVSDataItem
@@ -66,9 +68,13 @@ def __init__(
-        self.rmsnorm_hidden_size = vit_hidden_size * int(1 / self.downsample_ratio) ** 2
+        self.rmsnorm_hidden_size = (
+            vit_hidden_size * int(round(1 / self.downsample_ratio)) ** 2
diff -- python/sglang/srt/configs/nano_nemotron_vl.py
@@ -38,6 +38,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +322/-36; `python/sglang/srt/models/nano_nemotron_vl.py` modified +171/-20; `python/sglang/srt/configs/nano_nemotron_vl.py` modified +38/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/configs/parakeet.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23907 - [Docs] add Nemotron 3 Nano Omni cookbook

- 链接: https://github.com/sgl-project/sglang/pull/23907
- 状态/时间: merged / 2026-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx`；关联提交 `ad785a229911`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+746/-1，可读 patch 771 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] add Nemotron 3 Nano Omni cookbook」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx`；技术摘要: 覆盖「[Docs] add Nemotron 3 Nano Omni cookbook」；主要实现面是 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` added +542/-0 (542 lines); hunks: -0,0 +1,542；`docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` added +200/-0 (200 lines); hunks: -0,0 +1,200。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` added +542/-0 (542 lines); hunks: -0,0 +1,542
  - `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` added +200/-0 (200 lines); hunks: -0,0 +1,200
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx
@@ -0,0 +1,542 @@
+---
+title: Nemotron 3 Nano Omni
+metatags:
+    description: "Deploy NVIDIA Nemotron 3 Nano Omni multimodal MoE model with SGLang - text, image, video, and audio inputs with reasoning and tool calling."
+tag:
+    NEW
diff -- docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx
@@ -0,0 +1,200 @@
+export const Nemotron3NanoOmniDeployment = () => {
+  const MODEL_PATHS = {
+    reasoning: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning',
+    bf16: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-BF16',
+    fp8: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-FP8',
+    nvfp4: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-NVFP4',
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` added +542/-0; `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` added +200/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/cookbook/intro copy.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23874 - Fix failing `test_nvidia_nemotron_3_nano` by fixing `test_grouped_topk`

- 链接: https://github.com/sgl-project/sglang/pull/23874
- 状态/时间: merged / 2026-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`, `test/registered/models/test_nvidia_nemotron_3_nano.py`；关联提交 `ddcacaf1bd4e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+223/-19，可读 patch 282 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix failing `test_nvidia_nemotron_3_nano` by fixing `test_grouped_topk`」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/nemotron_h.py`, `test/registered/models/test_nvidia_nemotron_3_nano.py`；技术摘要: 覆盖「Fix failing `test_nvidia_nemotron_3_nano` by fixing `test_grouped_topk`」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`, `test/registered/models/test_nvidia_nemotron_3_nano.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +2/-0 (2 lines); hunks: -923,6 +923,8 @@ def nemotron_mamba2_with_output(; symbols: nemotron_mamba2_with_output，涉及 `nemotron_mamba2_with_output`；`test/registered/models/test_nvidia_nemotron_3_nano.py` modified +0/-1 (1 lines); hunks: -7,7 +7,6。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +2/-0 (2 lines); hunks: -923,6 +923,8 @@ def nemotron_mamba2_with_output(; symbols: nemotron_mamba2_with_output
  - `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +0/-1 (1 lines); hunks: -7,7 +7,6
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -923,6 +923,8 @@ def nemotron_mamba2_with_output(
+    if output.shape[0] != num_actual_tokens:
+        output[num_actual_tokens:].zero_()
diff -- test/registered/models/test_nvidia_nemotron_3_nano.py
@@ -7,7 +7,6 @@
-    disabled="Temporarily disabled; failing on main.",
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +2/-0
  - tests: `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +0/-1
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_grouped_topk.py`, `test/registered/models/test_nvidia_nemotron_3_nano.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23968 - [Docs] update Docker image for Nemotron 3 Nano Omni

- 链接: https://github.com/sgl-project/sglang/pull/23968
- 状态/时间: merged / 2026-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；关联提交 `387c932dfc88`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] update Docker image for Nemotron 3 Nano Omni」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；未提供可用技术摘要。
- 实现要点: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ pip install sglang。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ pip install sglang
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx
@@ -52,7 +52,7 @@ pip install sglang
-docker pull lmsysorg/sglang:nightly
+docker pull lmsysorg/sglang:dev-cu13-nemotronh-nano-omni-reasoning-v3
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23857 - Nemotron-omni-v3-alias

- 链接: https://github.com/sgl-project/sglang/pull/23857
- 状态/时间: merged / 2026-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`；关联提交 `b437f6be48a1`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+36/-6，可读 patch 111 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Nemotron-omni-v3-alias」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`；技术摘要: 覆盖「Nemotron-omni-v3-alias」；主要实现面是 `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +14/-4 (18 lines); hunks: -19,13 +19,19; -51,15 +57,19; symbols: NanoNemotronVLImageProcessor, __init__，涉及 `NanoNemotronVLImageProcessor, __init__`；`python/sglang/srt/configs/nano_nemotron_vl.py` modified +9/-0 (9 lines); hunks: -150,3 +150,12 @@ def create_radio_config(self):; symbols: create_radio_config, NemotronH_Nano_Omni_Reasoning_V3_Config, __init__，涉及 `create_radio_config, NemotronH_Nano_Omni_Reasoning_V3_Config, __init__`；`python/sglang/srt/models/nano_nemotron_vl.py` modified +5/-1 (6 lines); hunks: -372,4 +372,8 @@ def is_sound_weights(name: str) -> bool:; symbols: is_sound_weights, NemotronH_Nano_Omni_Reasoning_V3，涉及 `is_sound_weights, NemotronH_Nano_Omni_Reasoning_V3`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +14/-4 (18 lines); hunks: -19,13 +19,19; -51,15 +57,19; symbols: NanoNemotronVLImageProcessor, __init__
  - `python/sglang/srt/configs/nano_nemotron_vl.py` modified +9/-0 (9 lines); hunks: -150,3 +150,12 @@ def create_radio_config(self):; symbols: create_radio_config, NemotronH_Nano_Omni_Reasoning_V3_Config, __init__
  - `python/sglang/srt/models/nano_nemotron_vl.py` modified +5/-1 (6 lines); hunks: -372,4 +372,8 @@ def is_sound_weights(name: str) -> bool:; symbols: is_sound_weights, NemotronH_Nano_Omni_Reasoning_V3
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/nano_nemotron_vl.py
@@ -19,13 +19,19 @@
-from sglang.srt.configs.nano_nemotron_vl import NemotronH_Nano_VL_V2_Config
+from sglang.srt.configs.nano_nemotron_vl import (
+    NemotronH_Nano_Omni_Reasoning_V3_Config,
+    NemotronH_Nano_VL_V2_Config,
+)
-from sglang.srt.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
diff -- python/sglang/srt/configs/nano_nemotron_vl.py
@@ -150,3 +150,12 @@ def create_radio_config(self):
+class NemotronH_Nano_Omni_Reasoning_V3_Config(NemotronH_Nano_VL_V2_Config):
+    model_type = "NemotronH_Nano_Omni_Reasoning_V3"
+    def __init__(self, *args, **kwargs):
+        # Explicit __init__ prevents PretrainedConfig.__init_subclass__ from
+        # replacing the parent's custom __init__ with a dataclass-generated one.
+        super().__init__(*args, **kwargs)
diff -- python/sglang/srt/models/nano_nemotron_vl.py
@@ -372,4 +372,8 @@ def is_sound_weights(name: str) -> bool:
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +14/-4; `python/sglang/srt/configs/nano_nemotron_vl.py` modified +9/-0; `python/sglang/srt/models/nano_nemotron_vl.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21321 - [Kernel] Support FlashInfer TRTLLM-Gen fused MoE for non-gated FP4 & FP8 (Nemotron)

- 链接: https://github.com/sgl-project/sglang/pull/21321
- 状态/时间: merged / 2026-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `8327270c7263`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+341/-53，可读 patch 758 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Support FlashInfer TRTLLM-Gen fused MoE for non-gated FP4 & FP8 (Nemotron)」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「[Kernel] Support FlashInfer TRTLLM-Gen fused MoE for non-gated FP4 & FP8 (Nemotron)」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +2/-0 (2 lines); hunks: -51,6 +51,7; -190,6 +191,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +2/-0 (2 lines); hunks: -51,6 +51,7; -190,6 +191,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -51,6 +51,7 @@
+from sglang.srt.layers.moe.utils import RoutingMethodType
@@ -190,6 +191,7 @@ def __init__(
+            routing_method_type=RoutingMethodType.DeepSeekV3,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/flashinfer_trtllm_moe.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23594 - LoRA support for qwen3.5 and nemotron3

- 链接: https://github.com/sgl-project/sglang/pull/23594
- 状态/时间: merged / 2026-04-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py`；关联提交 `c8c1c9261d72`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 21 个文件，+1131/-127，可读 patch 1734 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「LoRA support for qwen3.5 and nemotron3」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/nemotron_h.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py`；技术摘要: 覆盖「LoRA support for qwen3.5 and nemotron3」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +105/-0 (105 lines); hunks: -665,6 +665,17 @@ class NemotronHForCausalLM(nn.Module):; -748,6 +759,100 @@ def _init_model(; symbols: NemotronHForCausalLM, _init_model, get_input_embeddings, get_stacked_multiply，涉及 `NemotronHForCausalLM, _init_model, get_input_embeddings`；`test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: kl_v2, get_prompt_logprobs, TestLoRANemotron3Super120B_A12B_LogprobDiff, test_lora_nemotron_3_super_120b_a12b_logprob_accuracy，涉及 `kl_v2, get_prompt_logprobs, TestLoRANemotron3Super120B_A12B_LogprobDiff`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +105/-0 (105 lines); hunks: -665,6 +665,17 @@ class NemotronHForCausalLM(nn.Module):; -748,6 +759,100 @@ def _init_model(; symbols: NemotronHForCausalLM, _init_model, get_input_embeddings, get_stacked_multiply
  - `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: kl_v2, get_prompt_logprobs, TestLoRANemotron3Super120B_A12B_LogprobDiff, test_lora_nemotron_3_super_120b_a12b_logprob_accuracy
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -665,6 +665,17 @@ class NemotronHForCausalLM(nn.Module):
+    supported_lora_modules = [
+        "qkv_proj",
+        "o_proj",
+        "out_proj",
+        "in_proj",
+        "up_proj",
diff -- test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py
@@ -0,0 +1,154 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +105/-0
  - tests: `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` added +154/-0
- 验证与风险: diff 自带测试面 `test/registered/lora/test_chunked_sgmv_backend.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
