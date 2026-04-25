# sglang Nemotron Super 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Super.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/nemotron3-nano-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/nemotron3-super-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/configs/jet_nemotron.py` | [#12448](https://github.com/sgl-project/sglang/pull/12448) |
| `python/sglang/srt/configs/nano_nemotron_vl.py` | [#12277](https://github.com/sgl-project/sglang/pull/12277), [#23568](https://github.com/sgl-project/sglang/pull/23568) |
| `python/sglang/srt/configs/nemotron_h.py` | [#10909](https://github.com/sgl-project/sglang/pull/10909), [#12690](https://github.com/sgl-project/sglang/pull/12690), [#16227](https://github.com/sgl-project/sglang/pull/16227), [#19950](https://github.com/sgl-project/sglang/pull/19950), [#20458](https://github.com/sgl-project/sglang/pull/20458) |
| `python/sglang/srt/models/jet_nemotron.py` | [#12448](https://github.com/sgl-project/sglang/pull/12448) |
| `python/sglang/srt/models/nano_nemotron_vl.py` | [#12277](https://github.com/sgl-project/sglang/pull/12277), [#14051](https://github.com/sgl-project/sglang/pull/14051), [#23568](https://github.com/sgl-project/sglang/pull/23568) |
| `python/sglang/srt/models/nemotron_h.py` | [#10909](https://github.com/sgl-project/sglang/pull/10909), [#11866](https://github.com/sgl-project/sglang/pull/11866), [#12015](https://github.com/sgl-project/sglang/pull/12015), [#12277](https://github.com/sgl-project/sglang/pull/12277), [#12690](https://github.com/sgl-project/sglang/pull/12690), [#16172](https://github.com/sgl-project/sglang/pull/16172), [#16227](https://github.com/sgl-project/sglang/pull/16227), [#16569](https://github.com/sgl-project/sglang/pull/16569), [#17013](https://github.com/sgl-project/sglang/pull/17013), [#18546](https://github.com/sgl-project/sglang/pull/18546), [#19903](https://github.com/sgl-project/sglang/pull/19903), [#20580](https://github.com/sgl-project/sglang/pull/20580) |
| `python/sglang/srt/models/nemotron_h_mtp.py` | [#17013](https://github.com/sgl-project/sglang/pull/17013), [#19433](https://github.com/sgl-project/sglang/pull/19433) |
| `python/sglang/srt/models/nemotron_nas.py` | [#9067](https://github.com/sgl-project/sglang/pull/9067) |
| `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` | [#12277](https://github.com/sgl-project/sglang/pull/12277), [#14051](https://github.com/sgl-project/sglang/pull/14051), [#23568](https://github.com/sgl-project/sglang/pull/23568) |
| `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml` | [#18119](https://github.com/sgl-project/sglang/pull/18119) |
| `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml` | [#18119](https://github.com/sgl-project/sglang/pull/18119) |
| `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` | [#20575](https://github.com/sgl-project/sglang/pull/20575), [#20616](https://github.com/sgl-project/sglang/pull/20616), [#21516](https://github.com/sgl-project/sglang/pull/21516) |
| `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` | [#20575](https://github.com/sgl-project/sglang/pull/20575), [#20616](https://github.com/sgl-project/sglang/pull/20616) |
| `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` | [#20616](https://github.com/sgl-project/sglang/pull/20616) |
| `test/registered/models/test_nvidia_nemotron_3_nano.py` | [#18119](https://github.com/sgl-project/sglang/pull/18119) |
| `test/registered/models/test_nvidia_nemotron_nano_v2.py` | 无直接 PR 号提交 |
| `test/registered/models/test_nvidia_nemotron_nano_v2_vl.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 23
- 原文档显式引用补充 PR 数: 1
- 当前文档总 PR 数: 24
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
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

## 逐 PR diff 审计卡

### PR #9067 - model: support nvidia/Llama-3_3-Nemotron-Super-49B-v1

- 链接: https://github.com/sgl-project/sglang/pull/9067
- 状态/时间: merged / 2025-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_nas.py`；关联提交 `845d12a979fb`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+465/-5，可读 patch 505 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「model: support nvidia/Llama-3_3-Nemotron-Super-49B-v1」，变更集中在 `python/sglang/srt/models/nemotron_nas.py`。PR 描述补充为：Based on: https://github.com/sgl-project/sglang/pull/5073 ### Motivation Add support for nvidia/Llama-3_3-Nemotron-Super-49B-v1 and v1.5. ### Modifications Porting nemotron_nas...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/models/nemotron_h.py`；关联提交 `d6837aea4d2c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+3279/-853，可读 patch 4929 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「model: Support Hybrid Mamba2 NemotronHForCausalLM (nvidia/NVIDIA-Nemotron-Nano-9B-v2)」，变更集中在 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`。PR 描述补充为：Support the `NemotronHForCausalLM` architecture, which can include any combination of *Mamba2*, MLP and normal self-attention layers. ## Motivation Support the `NemotronHForCaus...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `d6fee73d1f59`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+207/-127，可读 patch 628 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4」，变更集中在 `python/sglang/srt/models/nemotron_h.py`。PR 描述补充为：## Motivation Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8 and nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4 variants of https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2. Fol...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `6c18addb6f53`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+127/-207，可读 patch 628 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「Revert "Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4"」，变更集中在 `python/sglang/srt/models/nemotron_h.py`。PR 描述补充为：Reverts sgl-project/sglang#11866
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/jet_nemotron.py`, `python/sglang/srt/models/jet_nemotron.py`；关联提交 `3633f8b0cfef`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+678/-2，可读 patch 733 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「Add Jet-Nemotron」，变更集中在 `python/sglang/srt/models/jet_nemotron.py`, `python/sglang/srt/configs/jet_nemotron.py`。PR 描述补充为：## Motivation To add support for Jet-Nemotron. ## Modifications - Added Jet-Nemotron implementation. - Registered Jet-Nemotron as hybrid GDN attention model. - Added Jet-Nemotro...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/models/nemotron_h.py`；关联提交 `1b48e1b97484`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+775/-67，可读 patch 1291 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「Feat/nemotron nano v3 support」，变更集中在 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`。PR 描述补充为：## Motivation Add support for upcoming NVIDIA Nemotron v3 models. ## Modifications Add an MoE layer to the NemotronH modeling code. Add support for un-gated MoE in the triton co...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`；关联提交 `082b54c6890a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+1334/-17，可读 patch 1528 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「Support nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 (and nvidia/C-RADIOv2-H)」，变更集中在 `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`。PR 描述补充为：## Motivation Support Multimodal nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16. > NVIDIA Nemotron Nano v2 12B VL model enables multi-image reasoning and video understanding, along...
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
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「[NemotronH] PP support」，变更集中在 `python/sglang/srt/models/nemotron_h.py`。PR 描述补充为：## Motivation NemotronH models currently do not support running with `--pp-size` that's larger than 1. This PR fixes this. ## Modifications Switch `make_layers_non_pp` with `mak...
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
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「[NemotronH] Add latent MoE support」，变更集中在 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`。PR 描述补充为：## Motivation Future NemotronH models will (conditionally) have a linear layer before (and after) the MoE layer, letting the MoE operate in a smaller hidden size. This PR enable...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`；关联提交 `bebd625ba145`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+821/-56，可读 patch 1171 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「EVS Framework: Support NemotronH_Nano_VL_V2」，变更集中在 `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`。PR 描述补充为：## Motivation > https://arxiv.org/abs/2510.14624: Efficient Video Sampling: Pruning Temporally Redundant Tokens for Faster VLM Inference Add support for EVS pruning to `Nemotron...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_mtp.py`；关联提交 `ba625c2d908a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+378/-1，可读 patch 408 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「Feat/support nemotron h mtp」，变更集中在 `python/sglang/srt/models/nemotron_h_mtp.py`, `python/sglang/srt/models/nemotron_h.py`。PR 描述补充为：## Motivation This PR adds Multi-Token Prediction (MTP) speculative decoding support for NemotronH hybrid Mamba2/Attention models. ## Modifications #### New NemotronH MTP Model...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `72bacc88c8a0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 调整模型相关实现，标题为「[NemotronH] Use ReplicatedLinear for fc1_latent_proj」，变更集中在 `python/sglang/srt/models/nemotron_h.py`。PR 描述补充为：## Motivation The `fc1_latent_proj` linear layer is relatively small, similar in size to the `gate` and `fc2_latent_proj` layers in the same MoE module, and we don't get a lot f...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml`, `test/registered/models/test_nvidia_nemotron_3_nano.py`；关联提交 `c6aa1863be84`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+177/-0，可读 patch 188 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「Add Nemotron 3 Nano tests」，变更集中在 `test/registered/models/test_nvidia_nemotron_3_nano.py`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml`。PR 描述补充为：## Motivation Add CI test coverage for the NVIDIA Nemotron-3-Nano-30B models (BF16 and FP8 variants) to validate GSM8K accuracy. ## Accuracy Tests Tests were run locally and pas...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `33c33a7de9bb`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+100/-71，可读 patch 251 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「[Quantization] Support config.json quantization_config format, fix exclude_modules matching, and fix KV cache scale loading for Nemotron」，变更集中在 `python/sglang/srt/models/nemotron_h.py`。PR 描述补充为：## Summary Fixes several issues with ModelOpt quantization config loading for `NemotronH` (and future models that move away from `hf_quant_config.json`). Ensures models can load...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h_mtp.py`；关联提交 `4c95953b7733`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+73/-3，可读 patch 117 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 修复已暴露的启动、加载、解析或数值问题，标题为「Fix/nemotron mtp quantaized」，变更集中在 `python/sglang/srt/models/nemotron_h_mtp.py`。PR 描述补充为：## Motivation Fix code so nemotron+mtp works for quantized checkpoints ## Modifications * mtp layer prefix should be mtp * fused_moe_triton should handle non gated moe correctly...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`；关联提交 `f8bbf56de7b2`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+182/-17，可读 patch 281 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「Refactor NemotronHConfig to canonical layers_block_type and add MTP block-type support」，变更集中在 `python/sglang/srt/configs/nemotron_h.py`。PR 描述补充为：This PR updates python/sglang/srt/configs/nemotron_h.py to make layers_block_type the source of truth for layer layout and deprecates num_hidden_layers/hybrid_override_pattern a...
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
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「Enable Piecewise CUDA Graph for NemotronH Hybrid (Mamba+Attention) Models」，变更集中在 `python/sglang/srt/models/nemotron_h.py`。PR 描述补充为：### Motivation Piecewise CUDA graph (PCG) was previously disabled for NemotronH models because the layer detection logic required all layers to use standard GQA attention. Nemot...
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
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「[Model] Support Nemotron 3 Super NVFP4」，变更集中在 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/quantization/__init__.py`。PR 描述补充为：## Summary Support `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` via `modelopt_mixed` Fix #20472 ## Accuracy Tests ### Without MTP ### With MTP
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
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「[CI] Add Nemotron 3 Super 120B CI tests for BF16 and NVFP4」，变更集中在 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`。PR 描述补充为：## Summary Add per-PR CI tests for Nemotron-3-Super-120B model: - BF16 variant on Hopper (`stage-c-test-8-gpu-h200`) with TP=8 - NVFP4 variant on Blackwell (`stage-c-test-4-gpu-...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`；关联提交 `1ac6a2646437`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+26/-1，可读 patch 55 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 修复已暴露的启动、加载、解析或数值问题，标题为「fix: Nemotron chunk size alias」，变更集中在 `python/sglang/srt/configs/nemotron_h.py`。PR 描述补充为：## Motivation NemotronH silently ignored Hugging Face chunk_size and fell back to mamba_chunk_size=256, causing NVIDIA Nemotron-3-Super to run with the wrong Mamba chunking para...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`；关联提交 `3879c466b432`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+145/-6，可读 patch 180 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「[CI] Add Nemotron 3 Super 120B nightly 8-GPU tests」，变更集中在 `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`。PR 描述补充为：## Summary Add nightly 8-GPU CI tests for Nemotron-3-Super-120B model with full gsm8k evaluation (1314 questions): - BF16 variant on Hopper + Blackwell with TP=8 and TP=8+MTP -...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `466ff20e5148`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-7，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 修复已暴露的启动、加载、解析或数值问题，标题为「[Model] Fix NemotronH OOM on unified-mem systems: stream weights」，变更集中在 `python/sglang/srt/models/nemotron_h.py`。PR 描述补充为：## Motivation On unified-memory systems like DGX Spark GB10 (119 GB shared CPU+GPU pool), the Nemotron-3-Super-120B-A12B-NVFP4 model (75 GB on disk) cannot load with the current...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；关联提交 `0138129d3cfc`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 修复已暴露的启动、加载、解析或数值问题，标题为「[CI] Fix nemotron nvfp4 test estimated time」，变更集中在 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`。PR 描述补充为：## Motivation https://github.com/sgl-project/sglang/actions/runs/23623717495/job/68827833526?pr=20904 Fix the timeout here ## Modifications ## Accuracy Tests ## Benchmarking and...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`；关联提交 `4a3fe2a0913c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+1289/-116，可读 patch 1817 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Nemotron Super 补齐模型支持入口或运行时能力，标题为「Parakeet nemotron encoder」，变更集中在 `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`。PR 描述补充为：## Motivation ## Modifications ## Accuracy Tests ## Speed Tests and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit te...
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

## 补漏结论

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
