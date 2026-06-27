# vllm Ling 2.5 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| - | 当前主线没有匹配到实现文件 |

## PR 覆盖总览

- git 追溯 PR 数: 0
- 原文档显式引用补充 PR 数: 38
- 当前文档总 PR 数: 38
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-14 | [#20680](https://github.com/vllm-project/vllm/pull/20680) | merged | [Model] Add Ling implementation | `vllm/model_executor/models/bailing_moe.py`, `tests/models/registry.py`, `docs/models/supported_models.md` |
| 2025-07-16 | [#21059](https://github.com/vllm-project/vllm/pull/21059) | merged | [Model] Remove model sampler | `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/hunyuan_v1_moe.py` |
| 2025-07-19 | [#21100](https://github.com/vllm-project/vllm/pull/21100) | merged | [Quantization] Enable BNB support for more MoE models | `vllm/model_executor/models/hunyuan_v1_moe.py`, `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/grok1.py` |
| 2025-07-26 | [#21664](https://github.com/vllm-project/vllm/pull/21664) | merged | support `torch.compile` for bailing moe | `vllm/model_executor/models/bailing_moe.py` |
| 2025-08-29 | [#19497](https://github.com/vllm-project/vllm/pull/19497) | merged | [Models] Improve iteration over layers | `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/models/gpt_j.py`, `vllm/model_executor/models/lfm2.py` |
| 2025-09-15 | [#24627](https://github.com/vllm-project/vllm/pull/24627) | merged | [Model]: support Ling2.0 | `vllm/model_executor/models/bailing_moe.py`, `tests/models/registry.py`, `docs/models/supported_models.md` |
| 2025-09-21 | [#25345](https://github.com/vllm-project/vllm/pull/25345) | merged | [V0 Deprecation] Remove V0 sampling metadata | `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/glm4_moe_mtp.py`, `vllm/model_executor/models/granite.py` |
| 2025-09-30 | [#25271](https://github.com/vllm-project/vllm/pull/25271) | merged | Move`VllmConfig` from `config/__init__.py` to `config/vllm.py` | `vllm/model_executor/layers/quantization/utils/gptq_utils.py`, `vllm/model_executor/layers/quantization/gptq.py`, `vllm/model_executor/layers/quantization/auto_round.py` |
| 2025-10-05 | [#26247](https://github.com/vllm-project/vllm/pull/26247) | merged | Convert formatting to use `ruff` instead of `yapf` + `isort` | `tests/entrypoints/test_chat_utils.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py` |
| 2025-10-06 | [#26262](https://github.com/vllm-project/vllm/pull/26262) | merged | Fix per file ruff ignores related to line length | `tests/models/multimodal/generation/test_common.py`, `tests/entrypoints/test_chat_utils.py`, `vllm/model_executor/models/longcat_flash_mtp.py` |
| 2025-10-09 | [#26145](https://github.com/vllm-project/vllm/pull/26145) | merged | [Model] Apply shared experts overlap optimization to all models with shared experts | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe.py`, `vllm/model_executor/models/ernie45_vl_moe.py` |
| 2025-10-12 | [#26633](https://github.com/vllm-project/vllm/pull/26633) | merged | Update `Optional[x]` -> `x \| None` and `Union[x, y]` to `x \| y` | `vllm/entrypoints/openai/protocol.py`, `vllm/entrypoints/llm.py`, `vllm/model_executor/layers/fused_moe/layer.py` |
| 2025-10-15 | [#26876](https://github.com/vllm-project/vllm/pull/26876) | merged | [Fix] Remove divisibility requirement between num_kv_heads and tp_size in bailing_moe | `vllm/model_executor/models/bailing_moe.py` |
| 2025-11-11 | [#28382](https://github.com/vllm-project/vllm/pull/28382) | merged | [LoRA][1/N]Remove LoRA extra vocab | `vllm/model_executor/models/phimoe.py`, `vllm/model_executor/models/lfm2_moe.py`, `vllm/model_executor/models/falcon_h1.py` |
| 2025-11-13 | [#27583](https://github.com/vllm-project/vllm/pull/27583) | merged | Rename clashing method names for vLLM model protocol | `vllm/model_executor/models/interfaces_base.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/bert.py` |
| 2025-11-14 | [#28277](https://github.com/vllm-project/vllm/pull/28277) | merged | [Model] Fix bailing_moe accuracy problem | `vllm/model_executor/models/bailing_moe.py` |
| 2025-11-15 | [#28777](https://github.com/vllm-project/vllm/pull/28777) | merged | [Model] Fix lmhead init bug of bailing_moe | `vllm/model_executor/models/bailing_moe.py` |
| 2025-11-19 | [#28542](https://github.com/vllm-project/vllm/pull/28542) | merged | Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5 | `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/deepseek_v2.py` |
| 2025-11-26 | [#29342](https://github.com/vllm-project/vllm/pull/29342) | merged | [Attention] Remove imports from `vllm/attention/__init__.py` | `vllm/model_executor/models/whisper.py`, `vllm/model_executor/model_loader/utils.py`, `vllm/model_executor/models/afmoe.py` |
| 2025-12-04 | [#29966](https://github.com/vllm-project/vllm/pull/29966) | merged | Access `partial_rotary_factor` from `rope_parameters` | `vllm/transformers_utils/configs/nemotron.py`, `vllm/transformers_utils/configs/qwen3_next.py`, `vllm/model_executor/models/gpt_neox.py` |
| 2025-12-11 | [#30389](https://github.com/vllm-project/vllm/pull/30389) | merged | Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim` | `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/model_executor/models/config.py`, `vllm/model_executor/models/phi.py` |
| 2026-01-07 | [#31104](https://github.com/vllm-project/vllm/pull/31104) | merged | [BugFix] LoRA: Support loading base_layer of experts | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/llama4.py` |
| 2026-01-26 | [#33063](https://github.com/vllm-project/vllm/pull/33063) | merged | [Chore] Update type annotation of `input_ids` in model forward | `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py` |
| 2026-01-27 | [#32064](https://github.com/vllm-project/vllm/pull/32064) | merged | [5/N][Attention] Finish eliminating `vllm/attention` folder | `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/__init__.py` |
| 2026-02-04 | [#33737](https://github.com/vllm-project/vllm/pull/33737) | merged | [Bugfix] Define router_logits_dtype for remaining MoE models | `vllm/model_executor/models/longcat_flash.py`, `vllm/model_executor/models/flex_olmo.py`, `vllm/model_executor/models/afmoe.py` |
| 2026-02-26 | [#35102](https://github.com/vllm-project/vllm/pull/35102) | merged | [Model] Ring 2.5 | `vllm/model_executor/models/bailing_moe_linear.py`, `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/layers/fla/ops/layernorm_guard.py` |
| 2026-03-18 | [#37195](https://github.com/vllm-project/vllm/pull/37195) | merged | [V0 Deprecation] Deprecate virtual engine | `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/mla_attention.py` |
| 2026-03-24 | [#37487](https://github.com/vllm-project/vllm/pull/37487) | merged | [V0 Deprecation] Refactor kv cache from list to element | `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/models/extract_hidden_states.py` |
| 2026-04-20 | [#35949](https://github.com/vllm-project/vllm/pull/35949) | merged | [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase | `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py` |
| 2026-04-21 | [#35782](https://github.com/vllm-project/vllm/pull/35782) | merged | [MoE Refactor] Remove SharedFusedMoE class | `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py` |
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | merged | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py` |
| 2026-04-28 | [#40859](https://github.com/vllm-project/vllm/pull/40859) | merged | [Bugfix ] fix bailing_moe_linear | `vllm/model_executor/models/bailing_moe_linear.py`, `vllm/model_executor/layers/mamba/mamba_utils.py` |
| 2026-04-29 | [#41185](https://github.com/vllm-project/vllm/pull/41185) | merged | [Bugfix] BailingMoeV2.5: rotate full qk_rope_head_dim in MLA RoPE | `vllm/model_executor/models/bailing_moe_linear.py` |
| 2026-05-11 | [#41188](https://github.com/vllm-project/vllm/pull/41188) | merged | [Misc] Replace mamba_type string literals with MambaAttentionBackendEnum | `vllm/model_executor/layers/kda.py`, `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/linear_attn.py` |
| 2026-05-26 | [#43410](https://github.com/vllm-project/vllm/pull/43410) | merged | [Kernel] Porting fuse_minimax_qk_norm to manual fusion | `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`, `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/layers/minimax_rms_norm/__init__.py` |
| 2026-06-01 | [#43770](https://github.com/vllm-project/vllm/pull/43770) | merged | [Bugfix] fix wrong partial_rotary_factor calculation for bailing_moe model. | `vllm/model_executor/models/bailing_moe.py` |
| 2026-06-04 | [#43556](https://github.com/vllm-project/vllm/pull/43556) | merged | [Attention] Mamba attention module refactor - LINEAR | `vllm/model_executor/models/bailing_moe_linear.py`, `vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py`, `vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py` |
| 2026-06-08 | [#41184](https://github.com/vllm-project/vllm/pull/41184) | merged | [MoE Refactor] FusedMoE/MoERunner inversion refactor | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` |

## 逐 PR diff 审计卡

### PR #20680 - [Model] Add Ling implementation

- 链接: https://github.com/vllm-project/vllm/pull/20680
- 状态/时间: merged / 2025-07-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+534/-0，可读 patch 556 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add Ling implementation」；模型线: Ling 2.5；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/bailing_moe.py`, `tests/models/registry.py`, `docs/models/supported_models.md`；技术摘要: 覆盖「[Model] Add Ling implementation」；主要实现面是 `vllm/model_executor/models/bailing_moe.py`, `tests/models/registry.py`, `docs/models/supported_models.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/bailing_moe.py` added +530/-0 (530 lines); hunks: -0,0 +1,530; symbols: BailingAttention, __init__, forward, BailingMLP，涉及 `BailingAttention, __init__, forward`；`tests/models/registry.py` modified +2/-0 (2 lines); hunks: -141,6 +141,8 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -316,6 +316,7 @@ Specified using `--task generate`.；`vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -41,6 +41,7; symbols: name，涉及 `name`。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe.py` added +530/-0 (530 lines); hunks: -0,0 +1,530; symbols: BailingAttention, __init__, forward, BailingMLP
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -141,6 +141,8 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -316,6 +316,7 @@ Specified using `--task generate`.
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -41,6 +41,7; symbols: name
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe.py
@@ -0,0 +1,530 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from
+# https://github.com/inclusionAI/Ling/blob/master/models/modeling_bailing_moe.py
+# Copyright 2023 The vLLM team.
+# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
diff -- tests/models/registry.py
@@ -141,6 +141,8 @@ def check_available_online(
+    "BailingMoeForCausalLM": _HfExamplesInfo("inclusionAI/Ling-lite-1.5",
+                                         trust_remote_code=True),
diff -- docs/models/supported_models.md
@@ -316,6 +316,7 @@ Specified using `--task generate`.
+| `BailingMoeForCausalLM` | Ling | `inclusionAI/Ling-lite-1.5`, `inclusionAI/Ling-plus`, etc. | | ✅︎ | ✅︎ |
diff -- vllm/model_executor/models/registry.py
@@ -41,6 +41,7 @@
+    "BailingMoeForCausalLM": ("bailing_moe", "BailingMoeForCausalLM"),
```

- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` added +530/-0; `vllm/model_executor/models/registry.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +2/-0
  - docs: `docs/models/supported_models.md` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21059 - [Model] Remove model sampler

- 链接: https://github.com/vllm-project/vllm/pull/21059
- 状态/时间: merged / 2025-07-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+0/-45，可读 patch 157 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Remove model sampler」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/hunyuan_v1_moe.py`；技术摘要: 覆盖「[Model] Remove model sampler」；主要实现面是 `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/hunyuan_v1_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mimo_mtp.py` modified +0/-11 (11 lines); hunks: -30,7 +30,6; -161,8 +160,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, compute_logits, sample，涉及 `__init__, forward, compute_logits`；`vllm/model_executor/models/bailing_moe.py` modified +0/-10 (10 lines); hunks: -47,7 +47,6; -485,7 +484,6 @@ def __init__(; symbols: __init__, compute_logits, sample, load_weights，涉及 `__init__, compute_logits, sample`；`vllm/model_executor/models/hunyuan_v1_moe.py` modified +0/-10 (10 lines); hunks: -49,7 +49,6; -661,7 +660,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, compute_logits, sample, make_empty_intermediate_tensors，涉及 `__init__, compute_logits, sample`；`vllm/model_executor/models/phi4flash.py` modified +0/-10 (10 lines); hunks: -23,7 +23,6; -641,7 +640,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, compute_logits, sample，涉及 `__init__, forward, compute_logits`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_mtp.py` modified +0/-11 (11 lines); hunks: -30,7 +30,6; -161,8 +160,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, compute_logits, sample
  - `vllm/model_executor/models/bailing_moe.py` modified +0/-10 (10 lines); hunks: -47,7 +47,6; -485,7 +484,6 @@ def __init__(; symbols: __init__, compute_logits, sample, load_weights
  - `vllm/model_executor/models/hunyuan_v1_moe.py` modified +0/-10 (10 lines); hunks: -49,7 +49,6; -661,7 +660,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, compute_logits, sample, make_empty_intermediate_tensors
  - `vllm/model_executor/models/phi4flash.py` modified +0/-10 (10 lines); hunks: -23,7 +23,6; -641,7 +640,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, compute_logits, sample
  - `vllm/model_executor/models/granite_speech.py` modified +0/-2 (2 lines); hunks: -36,7 +36,6; -549,7 +548,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str):; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_mtp.py
@@ -30,7 +30,6 @@
-from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
@@ -161,8 +160,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        self.sampler = get_sampler()
@@ -187,14 +184,6 @@ def compute_logits(
-    def sample(
-        self,
diff -- vllm/model_executor/models/bailing_moe.py
@@ -47,7 +47,6 @@
-from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
@@ -485,7 +484,6 @@ def __init__(
-        self.sampler = get_sampler()
@@ -512,14 +510,6 @@ def compute_logits(
-    def sample(
-        self,
diff -- vllm/model_executor/models/hunyuan_v1_moe.py
@@ -49,7 +49,6 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_mtp.py` modified +0/-11; `vllm/model_executor/models/bailing_moe.py` modified +0/-10; `vllm/model_executor/models/hunyuan_v1_moe.py` modified +0/-10; `vllm/model_executor/models/phi4flash.py` modified +0/-10; `vllm/model_executor/models/granite_speech.py` modified +0/-2; `vllm/model_executor/models/mimo.py` modified +0/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/granite_speech.py`, `vllm/model_executor/models/hunyuan_v1_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21100 - [Quantization] Enable BNB support for more MoE models

- 链接: https://github.com/vllm-project/vllm/pull/21100
- 状态/时间: merged / 2025-07-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+223/-181，可读 patch 548 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] Enable BNB support for more MoE models」；模型线: Ling 2.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/hunyuan_v1_moe.py`, `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/grok1.py`；技术摘要: 覆盖「[Quantization] Enable BNB support for more MoE models」；主要实现面是 `vllm/model_executor/models/hunyuan_v1_moe.py`, `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/grok1.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/hunyuan_v1_moe.py` modified +107/-91 (198 lines); hunks: -56,7 +56,9; -617,86 +619,6 @@ def forward(; symbols: _get_cla_factor, forward, HunYuanMoEV1ForCausalLM, __init__，涉及 `_get_cla_factor, forward, HunYuanMoEV1ForCausalLM`；`vllm/model_executor/models/ernie45_moe.py` modified +84/-69 (153 lines); hunks: -51,8 +51,8; -427,66 +427,15 @@ def forward(; symbols: forward, get_expert_mapping, Ernie4_5_MoeForCausalLM, __init__，涉及 `forward, get_expert_mapping, Ernie4_5_MoeForCausalLM`；`vllm/model_executor/models/grok1.py` modified +14/-10 (24 lines); hunks: -360,6 +360,16 @@ def forward(; -369,18 +379,9 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: forward, get_expert_mapping, load_weights，涉及 `forward, get_expert_mapping, load_weights`；`vllm/model_executor/models/bailing_moe.py` modified +14/-7 (21 lines); hunks: -53,7 +53,7; -374,21 +374,25 @@ def forward(; symbols: forward, get_expert_mapping, load_weights, BailingMoeForCausalLM，涉及 `forward, get_expert_mapping, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/hunyuan_v1_moe.py` modified +107/-91 (198 lines); hunks: -56,7 +56,9; -617,86 +619,6 @@ def forward(; symbols: _get_cla_factor, forward, HunYuanMoEV1ForCausalLM, __init__
  - `vllm/model_executor/models/ernie45_moe.py` modified +84/-69 (153 lines); hunks: -51,8 +51,8; -427,66 +427,15 @@ def forward(; symbols: forward, get_expert_mapping, Ernie4_5_MoeForCausalLM, __init__
  - `vllm/model_executor/models/grok1.py` modified +14/-10 (24 lines); hunks: -360,6 +360,16 @@ def forward(; -369,18 +379,9 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: forward, get_expert_mapping, load_weights
  - `vllm/model_executor/models/bailing_moe.py` modified +14/-7 (21 lines); hunks: -53,7 +53,7; -374,21 +374,25 @@ def forward(; symbols: forward, get_expert_mapping, load_weights, BailingMoeForCausalLM
  - `docs/models/supported_models.md` modified +4/-4 (8 lines); hunks: -316,7 +316,7 @@ Specified using `--task generate`.; -328,8 +328,8 @@ Specified using `--task generate`.
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/hunyuan_v1_moe.py
@@ -56,7 +56,9 @@
-from .utils import PPMissingLayer, is_pp_missing_parameter, make_layers
+from .interfaces import SupportsLoRA
+from .utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
+                    make_layers)
@@ -617,86 +619,6 @@ def forward(
-class HunYuanMoEV1ForCausalLM(nn.Module):
diff -- vllm/model_executor/models/ernie45_moe.py
@@ -51,8 +51,8 @@
-from .interfaces import SupportsPP
-from .utils import (PPMissingLayer, extract_layer_index,
+from .interfaces import SupportsLoRA, SupportsPP
+from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
@@ -427,66 +427,15 @@ def forward(
+    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
diff -- vllm/model_executor/models/grok1.py
@@ -360,6 +360,16 @@ def forward(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/hunyuan_v1_moe.py` modified +107/-91; `vllm/model_executor/models/ernie45_moe.py` modified +84/-69; `vllm/model_executor/models/grok1.py` modified +14/-10; `vllm/model_executor/models/bailing_moe.py` modified +14/-7
  - docs: `docs/models/supported_models.md` modified +4/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/grok1.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21664 - support `torch.compile` for bailing moe

- 链接: https://github.com/vllm-project/vllm/pull/21664
- 状态/时间: merged / 2025-07-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support `torch.compile` for bailing moe」；模型线: Ling 2.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/bailing_moe.py`；技术摘要: 覆盖「support `torch.compile` for bailing moe」；主要实现面是 `vllm/model_executor/models/bailing_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/bailing_moe.py` modified +2/-0 (2 lines); hunks: -32,6 +32,7; -291,6 +292,7 @@ def forward(; symbols: forward, BailingMoeModel, __init__，涉及 `forward, BailingMoeModel, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe.py` modified +2/-0 (2 lines); hunks: -32,6 +32,7; -291,6 +292,7 @@ def forward(; symbols: forward, BailingMoeModel, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe.py
@@ -32,6 +32,7 @@
+from vllm.compilation.decorators import support_torch_compile
@@ -291,6 +292,7 @@ def forward(
+@support_torch_compile
```

- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19497 - [Models] Improve iteration over layers

- 链接: https://github.com/vllm-project/vllm/pull/19497
- 状态/时间: merged / 2025-08-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 65 个文件，+129/-83，可读 patch 1109 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models] Improve iteration over layers」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/models/gpt_j.py`, `vllm/model_executor/models/lfm2.py`；技术摘要: 覆盖「[Models] Improve iteration over layers」；主要实现面是 `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/models/gpt_j.py`, `vllm/model_executor/models/lfm2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek.py` modified +3/-2 (5 lines); hunks: -24,6 +24,7; -377,7 +378,7 @@ def forward(; symbols: forward, compute_logits, load_weights，涉及 `forward, compute_logits, load_weights`；`vllm/model_executor/models/gpt_j.py` modified +3/-2 (5 lines); hunks: -19,6 +19,7; -223,7 +224,7 @@ def forward(; symbols: forward, compute_logits, load_weights，涉及 `forward, compute_logits, load_weights`；`vllm/model_executor/models/lfm2.py` modified +3/-2 (5 lines); hunks: -1,6 +1,7; -374,7 +375,7 @@ def forward(; symbols: forward, load_weights，涉及 `forward, load_weights`；`vllm/model_executor/models/bailing_moe.py` modified +2/-2 (4 lines); hunks: -24,6 +24,7; -359,8 +360,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek.py` modified +3/-2 (5 lines); hunks: -24,6 +24,7; -377,7 +378,7 @@ def forward(; symbols: forward, compute_logits, load_weights
  - `vllm/model_executor/models/gpt_j.py` modified +3/-2 (5 lines); hunks: -19,6 +19,7; -223,7 +224,7 @@ def forward(; symbols: forward, compute_logits, load_weights
  - `vllm/model_executor/models/lfm2.py` modified +3/-2 (5 lines); hunks: -1,6 +1,7; -374,7 +375,7 @@ def forward(; symbols: forward, load_weights
  - `vllm/model_executor/models/bailing_moe.py` modified +2/-2 (4 lines); hunks: -24,6 +24,7; -359,8 +360,7 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/ernie45_moe.py` modified +2/-2 (4 lines); hunks: -23,6 +23,7; -419,8 +420,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek.py
@@ -24,6 +24,7 @@
+from itertools import islice
@@ -377,7 +378,7 @@ def forward(
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
@@ -483,4 +484,4 @@ def compute_logits(
-        return loader.load_weights(weights)
diff -- vllm/model_executor/models/gpt_j.py
@@ -19,6 +19,7 @@
+from itertools import islice
@@ -223,7 +224,7 @@ def forward(
-        for layer in self.h[self.start_layer:self.end_layer]:
+        for layer in islice(self.h, self.start_layer, self.end_layer):
@@ -336,4 +337,4 @@ def compute_logits(
-        return loader.load_weights(weights)
diff -- vllm/model_executor/models/lfm2.py
@@ -1,6 +1,7 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek.py` modified +3/-2; `vllm/model_executor/models/gpt_j.py` modified +3/-2; `vllm/model_executor/models/lfm2.py` modified +3/-2; `vllm/model_executor/models/bailing_moe.py` modified +2/-2; `vllm/model_executor/models/ernie45_moe.py` modified +2/-2; `vllm/model_executor/models/ernie45_vl_moe.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/arctic.py`, `vllm/model_executor/models/baichuan.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24627 - [Model]: support Ling2.0

- 链接: https://github.com/vllm-project/vllm/pull/24627
- 状态/时间: merged / 2025-09-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+170/-50，可读 patch 388 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model]: support Ling2.0」；模型线: Ling 2.5；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/bailing_moe.py`, `tests/models/registry.py`, `docs/models/supported_models.md`；技术摘要: 覆盖「[Model]: support Ling2.0」；主要实现面是 `vllm/model_executor/models/bailing_moe.py`, `tests/models/registry.py`, `docs/models/supported_models.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/bailing_moe.py` modified +166/-50 (216 lines); hunks: -43,7 +43,6; -68,6 +67,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`tests/models/registry.py` modified +2/-0 (2 lines); hunks: -180,6 +180,8 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -328,6 +328,7 @@ th {；`vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -52,6 +52,7; symbols: name，涉及 `name`。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe.py` modified +166/-50 (216 lines); hunks: -43,7 +43,6; -68,6 +67,7 @@ def __init__(; symbols: __init__, forward
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -180,6 +180,8 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -328,6 +328,7 @@ th {
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -52,6 +52,7; symbols: name
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe.py
@@ -43,7 +43,6 @@
-                                               ReplicatedLinear,
@@ -68,6 +67,7 @@ def __init__(
+        reduce_results: bool = True,
@@ -84,10 +84,11 @@ def __init__(
+        self.use_qk_norm = getattr(config, "use_qk_norm", False)
+        self.use_rmsnorm = getattr(config, "use_rmsnorm", False)
diff -- tests/models/registry.py
@@ -180,6 +180,8 @@ def check_available_online(
+    "BailingMoeV2ForCausalLM": _HfExamplesInfo("inclusionAI/Ling-mini-2.0",
+                                         trust_remote_code=True),
diff -- docs/models/supported_models.md
@@ -328,6 +328,7 @@ th {
+| `BailingMoeV2ForCausalLM` | Ling | `inclusionAI/Ling-mini-2.0`, etc. | ✅︎ | ✅︎ | ✅︎ |
diff -- vllm/model_executor/models/registry.py
@@ -52,6 +52,7 @@
+    "BailingMoeV2ForCausalLM": ("bailing_moe", "BailingMoeV2ForCausalLM"),
```

- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +166/-50; `vllm/model_executor/models/registry.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +2/-0
  - docs: `docs/models/supported_models.md` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25345 - [V0 Deprecation] Remove V0 sampling metadata

- 链接: https://github.com/vllm-project/vllm/pull/25345
- 状态/时间: merged / 2025-09-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 141 个文件，+172/-583，可读 patch 2888 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V0 Deprecation] Remove V0 sampling metadata」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/glm4_moe_mtp.py`, `vllm/model_executor/models/granite.py`；技术摘要: 覆盖「[V0 Deprecation] Remove V0 sampling metadata」；主要实现面是 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/glm4_moe_mtp.py`, `vllm/model_executor/models/granite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_mtp.py` modified +2/-7 (9 lines); hunks: -15,7 +15,6; -124,15 +123,13 @@ def forward(; symbols: forward, compute_logits, load_weights，涉及 `forward, compute_logits, load_weights`；`vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-7 (9 lines); hunks: -38,7 +38,6; -155,15 +154,13 @@ def forward(; symbols: forward, compute_logits, load_weights，涉及 `forward, compute_logits, load_weights`；`vllm/model_executor/models/granite.py` modified +3/-6 (9 lines); hunks: -48,7 +48,6; -463,11 +462,9 @@ def forward(; symbols: forward, compute_logits, make_empty_intermediate_tensors，涉及 `forward, compute_logits, make_empty_intermediate_tensors`；`vllm/model_executor/models/granitemoe.py` modified +3/-6 (9 lines); hunks: -48,7 +48,6; -511,11 +510,9 @@ def forward(; symbols: forward, compute_logits, make_empty_intermediate_tensors，涉及 `forward, compute_logits, make_empty_intermediate_tensors`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_mtp.py` modified +2/-7 (9 lines); hunks: -15,7 +15,6; -124,15 +123,13 @@ def forward(; symbols: forward, compute_logits, load_weights
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-7 (9 lines); hunks: -38,7 +38,6; -155,15 +154,13 @@ def forward(; symbols: forward, compute_logits, load_weights
  - `vllm/model_executor/models/granite.py` modified +3/-6 (9 lines); hunks: -48,7 +48,6; -463,11 +462,9 @@ def forward(; symbols: forward, compute_logits, make_empty_intermediate_tensors
  - `vllm/model_executor/models/granitemoe.py` modified +3/-6 (9 lines); hunks: -48,7 +48,6; -511,11 +510,9 @@ def forward(; symbols: forward, compute_logits, make_empty_intermediate_tensors
  - `vllm/model_executor/models/granitemoeshared.py` modified +3/-6 (9 lines); hunks: -25,7 +25,6; -311,11 +310,9 @@ def forward(; symbols: forward, compute_logits, make_empty_intermediate_tensors
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -15,7 +15,6 @@
-from vllm.model_executor.sampling_metadata import SamplingMetadata
@@ -124,15 +123,13 @@ def forward(
-        sampling_metadata: SamplingMetadata,
-                                       mtp_layer.shared_head(hidden_states),
-                                       sampling_metadata)
+                                       mtp_layer.shared_head(hidden_states))
diff -- vllm/model_executor/models/glm4_moe_mtp.py
@@ -38,7 +38,6 @@
-from vllm.model_executor.sampling_metadata import SamplingMetadata
@@ -155,15 +154,13 @@ def forward(
-        sampling_metadata: SamplingMetadata,
-                                       mtp_layer.shared_head(hidden_states),
-                                       sampling_metadata)
+                                       mtp_layer.shared_head(hidden_states))
diff -- vllm/model_executor/models/granite.py
@@ -48,7 +48,6 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_mtp.py` modified +2/-7; `vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-7; `vllm/model_executor/models/granite.py` modified +3/-6; `vllm/model_executor/models/granitemoe.py` modified +3/-6; `vllm/model_executor/models/granitemoeshared.py` modified +3/-6; `vllm/model_executor/models/ernie_mtp.py` modified +2/-6
- 验证与风险: diff 自带测试面 `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_opt.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25271 - Move`VllmConfig` from `config/__init__.py` to `config/vllm.py`

- 链接: https://github.com/vllm-project/vllm/pull/25271
- 状态/时间: merged / 2025-09-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 36 个文件，+964/-905，可读 patch 2200 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Move`VllmConfig` from `config/__init__.py` to `config/vllm.py`」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/quantization/utils/gptq_utils.py`, `vllm/model_executor/layers/quantization/gptq.py`, `vllm/model_executor/layers/quantization/auto_round.py`；技术摘要: 覆盖「Move`VllmConfig` from `config/__init__.py` to `config/vllm.py`」；主要实现面是 `vllm/model_executor/layers/quantization/utils/gptq_utils.py`, `vllm/model_executor/layers/quantization/gptq.py`, `vllm/model_executor/layers/quantization/auto_round.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/quantization/utils/gptq_utils.py` modified +13/-5 (18 lines); hunks: -4,21 +4,27; -34,6 +40,7 @@ def override_config(config: QuantizationConfig, prefix: str):; symbols: override_config, get_dynamic_override，涉及 `override_config, get_dynamic_override`；`vllm/model_executor/layers/quantization/gptq.py` modified +6/-2 (8 lines); hunks: -4,7 +4,7; -13,7 +13,6; symbols: GPTQConfig, for，涉及 `GPTQConfig, for`；`vllm/model_executor/layers/quantization/auto_round.py` modified +2/-3 (5 lines); hunks: -9,9 +9,8；`vllm/model_executor/layers/quantization/bitblas.py` modified +2/-3 (5 lines); hunks: -7,9 +7,8。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/utils/gptq_utils.py` modified +13/-5 (18 lines); hunks: -4,21 +4,27; -34,6 +40,7 @@ def override_config(config: QuantizationConfig, prefix: str):; symbols: override_config, get_dynamic_override
  - `vllm/model_executor/layers/quantization/gptq.py` modified +6/-2 (8 lines); hunks: -4,7 +4,7; -13,7 +13,6; symbols: GPTQConfig, for
  - `vllm/model_executor/layers/quantization/auto_round.py` modified +2/-3 (5 lines); hunks: -9,9 +9,8
  - `vllm/model_executor/layers/quantization/bitblas.py` modified +2/-3 (5 lines); hunks: -7,9 +7,8
  - `vllm/model_executor/layers/quantization/bitsandbytes.py` modified +2/-3 (5 lines); hunks: -13,9 +13,8
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/utils/gptq_utils.py
@@ -4,21 +4,27 @@
-from typing import Optional, Union
+from typing import TYPE_CHECKING, Optional, Union
-from vllm.config import QuantizationConfig
+if TYPE_CHECKING:
+    from ..gptq import GPTQConfig
+    from ..gptq_marlin import GPTQMarlinConfig
diff -- vllm/model_executor/layers/quantization/gptq.py
@@ -4,7 +4,7 @@
-from typing import Any, Optional, Union
+from typing import TYPE_CHECKING, Any, Optional, Union
@@ -13,7 +13,6 @@
-from vllm.model_executor.layers.quantization import QuantizationMethods
@@ -26,6 +25,11 @@
+if TYPE_CHECKING:
diff -- vllm/model_executor/layers/quantization/auto_round.py
@@ -9,9 +9,8 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/utils/gptq_utils.py` modified +13/-5; `vllm/model_executor/layers/quantization/gptq.py` modified +6/-2; `vllm/model_executor/layers/quantization/auto_round.py` modified +2/-3; `vllm/model_executor/layers/quantization/bitblas.py` modified +2/-3; `vllm/model_executor/layers/quantization/bitsandbytes.py` modified +2/-3; `vllm/model_executor/layers/quantization/deepspeedfp.py` modified +2/-3
- 验证与风险: runtime 路径改动集中在 `vllm/attention/layer.py`, `vllm/attention/layers/chunked_local_attention.py`, `vllm/config/__init__.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26247 - Convert formatting to use `ruff` instead of `yapf` + `isort`

- 链接: https://github.com/vllm-project/vllm/pull/26247
- 状态/时间: merged / 2025-10-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1508 个文件，+83935/-68959，可读 patch 272044 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Convert formatting to use `ruff` instead of `yapf` + `isort`」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `tests/entrypoints/test_chat_utils.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`；技术摘要: 覆盖「Convert formatting to use `ruff` instead of `yapf` + `isort`」；主要实现面是 `tests/entrypoints/test_chat_utils.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/entrypoints/test_chat_utils.py` modified +699/-1186 (1885 lines); hunks: -6,24 +6,28; -177,8 +181,7 @@ def _assert_mm_uuids(; symbols: _assert_mm_uuids, _assert_mm_data_inputs, test_parse_chat_messages_single_image，涉及 `_assert_mm_uuids, _assert_mm_data_inputs, test_parse_chat_messages_single_image`；`vllm/model_executor/layers/fused_moe/layer.py` modified +683/-484 (1167 lines); hunks: -14,66 +14,85; -92,7 +111,6 @@ class FusedMoeWeightScaleSupported(Enum):; symbols: _eplb_map_to_physical_and_record, FusedMoeWeightScaleSupported, FusedMoEMethodBase, __init__，涉及 `_eplb_map_to_physical_and_record, FusedMoeWeightScaleSupported, FusedMoEMethodBase`；`vllm/model_executor/layers/fused_moe/fused_moe.py` modified +623/-433 (1056 lines); hunks: -1,6 +1,7; -13,25 +14,38; symbols: write_zeros_to_output, fused_moe_kernel_gptq_awq，涉及 `write_zeros_to_output, fused_moe_kernel_gptq_awq`；`vllm/entrypoints/openai/serving_chat.py` modified +591/-422 (1013 lines); hunks: -17,29 +17,48; -48,16 +67,17; symbols: OpenAIServingChat, __init__, create_chat_completion，涉及 `OpenAIServingChat, __init__, create_chat_completion`。
- 代码 diff 细节:
  - `tests/entrypoints/test_chat_utils.py` modified +699/-1186 (1885 lines); hunks: -6,24 +6,28; -177,8 +181,7 @@ def _assert_mm_uuids(; symbols: _assert_mm_uuids, _assert_mm_data_inputs, test_parse_chat_messages_single_image
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +683/-484 (1167 lines); hunks: -14,66 +14,85; -92,7 +111,6 @@ class FusedMoeWeightScaleSupported(Enum):; symbols: _eplb_map_to_physical_and_record, FusedMoeWeightScaleSupported, FusedMoEMethodBase, __init__
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +623/-433 (1056 lines); hunks: -1,6 +1,7; -13,25 +14,38; symbols: write_zeros_to_output, fused_moe_kernel_gptq_awq
  - `vllm/entrypoints/openai/serving_chat.py` modified +591/-422 (1013 lines); hunks: -17,29 +17,48; -48,16 +67,17; symbols: OpenAIServingChat, __init__, create_chat_completion
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +522/-353 (875 lines); hunks: -12,40 +12,70; -70,8 +100,10 @@ def __init__(; symbols: __init__, get_name, get_config_filenames, apply_vllm_mapper
- 关键代码摘录:

```diff
diff -- tests/entrypoints/test_chat_utils.py
@@ -6,24 +6,28 @@
-from mistral_common.tokens.tokenizers.base import (SpecialTokenPolicy,
-                                                   SpecialTokens)
-from mistral_common.tokens.tokenizers.tekken import (SpecialTokenInfo,
-                                                     Tekkenizer)
+from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens
+from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -14,66 +14,85 @@
-from vllm.distributed import (get_dp_group, get_ep_group,
-                              get_tensor_model_parallel_world_size,
-                              tensor_model_parallel_all_reduce)
+from vllm.distributed import (
+    get_dp_group,
+    get_ep_group,
diff -- vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -1,6 +1,7 @@
```

- 已读文件:
  - tests: `tests/entrypoints/test_chat_utils.py` modified +699/-1186; `tests/tool_use/test_minimax_tool_parser.py` modified +396/-387
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +683/-484; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +623/-433; `vllm/entrypoints/openai/serving_chat.py` modified +591/-422; `vllm/model_executor/layers/quantization/modelopt.py` modified +522/-353; `vllm/entrypoints/openai/protocol.py` modified +499/-372; `vllm/model_executor/layers/linear.py` modified +462/-385
- 验证与风险: diff 自带测试面 `tests/basic_correctness/test_basic_correctness.py`, `tests/basic_correctness/test_cpu_offload.py`, `tests/basic_correctness/test_cumem.py`, `tests/benchmarks/test_latency_cli.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26262 - Fix per file ruff ignores related to line length

- 链接: https://github.com/vllm-project/vllm/pull/26262
- 状态/时间: merged / 2025-10-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 65 个文件，+301/-291，可读 patch 1525 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix per file ruff ignores related to line length」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `tests/models/multimodal/generation/test_common.py`, `tests/entrypoints/test_chat_utils.py`, `vllm/model_executor/models/longcat_flash_mtp.py`；技术摘要: 覆盖「Fix per file ruff ignores related to line length」；主要实现面是 `tests/models/multimodal/generation/test_common.py`, `tests/entrypoints/test_chat_utils.py`, `vllm/model_executor/models/longcat_flash_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/generation/test_common.py` modified +38/-38 (76 lines); hunks: -130,14 +130,14; -149,8 +149,8；`tests/entrypoints/test_chat_utils.py` modified +26/-23 (49 lines); hunks: -947,7 +947,8 @@ def test_parse_chat_messages_placeholder_one_already_in_prompt(; -960,8 +961,8 @@ def test_parse_chat_messages_placeholder_one_already_in_prompt(; symbols: test_parse_chat_messages_placeholder_one_already_in_prompt, test_parse_chat_messages_multiple_images_multiple_messages_interleave, test_parse_chat_messages_multiple_images_with_uuids_multiple_messages_interleave，涉及 `test_parse_chat_messages_placeholder_one_already_in_prompt, test_parse_chat_messages_multiple_images_multiple_messages_interleave, test_parse_chat_messages_multiple_images_with_uuids_multiple_messages_interleave`；`vllm/model_executor/models/longcat_flash_mtp.py` modified +20/-20 (40 lines); hunks: -186,26 +186,26 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights，涉及 `load_weights`；`tests/models/language/generation/test_mistral.py` modified +10/-6 (16 lines); hunks: -46,12 +46,13; -85,7 +86,8。
- 代码 diff 细节:
  - `tests/models/multimodal/generation/test_common.py` modified +38/-38 (76 lines); hunks: -130,14 +130,14; -149,8 +149,8
  - `tests/entrypoints/test_chat_utils.py` modified +26/-23 (49 lines); hunks: -947,7 +947,8 @@ def test_parse_chat_messages_placeholder_one_already_in_prompt(; -960,8 +961,8 @@ def test_parse_chat_messages_placeholder_one_already_in_prompt(; symbols: test_parse_chat_messages_placeholder_one_already_in_prompt, test_parse_chat_messages_multiple_images_multiple_messages_interleave, test_parse_chat_messages_multiple_images_with_uuids_multiple_messages_interleave
  - `vllm/model_executor/models/longcat_flash_mtp.py` modified +20/-20 (40 lines); hunks: -186,26 +186,26 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
  - `tests/models/language/generation/test_mistral.py` modified +10/-6 (16 lines); hunks: -46,12 +46,13; -85,7 +86,8
  - `tests/entrypoints/openai/test_chat.py` modified +7/-6 (13 lines); hunks: -835,17 +835,18 @@ async def test_extra_fields_allowed(client: openai.AsyncOp...; symbols: test_extra_fields_allowed, test_complex_message_content
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/generation/test_common.py
@@ -130,14 +130,14 @@
-        ],  # noqa: E501
+        ],
-        img_idx_to_prompt=lambda idx: "<|vision_start|><|image_pad|><|vision_end|>",  # noqa: E501
-        video_idx_to_prompt=lambda idx: "<|vision_start|><|video_pad|><|vision_end|>",  # noqa: E501
+        img_idx_to_prompt=lambda idx: "<|vision_start|><|image_pad|><|vision_end|>",
+        video_idx_to_prompt=lambda idx: "<|vision_start|><|video_pad|><|vision_end|>",
diff -- tests/entrypoints/test_chat_utils.py
@@ -947,7 +947,8 @@ def test_parse_chat_messages_placeholder_one_already_in_prompt(
-                        "text": "What's in <|image_1|> and how does it compare to the other one?",  # noqa: E501
+                        "text": "What's in <|image_1|> and how does it compare to "
+                        "the other one?",
@@ -960,8 +961,8 @@ def test_parse_chat_messages_placeholder_one_already_in_prompt(
-            "content": "<|image_2|>\nWhat's in <|image_1|> and how does it compare to the "
-            "other one?",
diff -- vllm/model_executor/models/longcat_flash_mtp.py
@@ -186,26 +186,26 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- 已读文件:
  - tests: `tests/models/multimodal/generation/test_common.py` modified +38/-38; `tests/entrypoints/test_chat_utils.py` modified +26/-23; `tests/models/language/generation/test_mistral.py` modified +10/-6; `tests/entrypoints/openai/test_chat.py` modified +7/-6; `tests/entrypoints/openai/test_completion_with_function_calling.py` modified +8/-4; `tests/entrypoints/openai/test_chat_with_tool_reasoning.py` modified +6/-4
  - runtime: `vllm/model_executor/models/longcat_flash_mtp.py` modified +20/-20
- 验证与风险: diff 自带测试面 `tests/compile/piecewise/test_simple.py`, `tests/compile/piecewise/test_toy_llama.py`, `tests/compile/test_functionalization.py`, `tests/compile/test_fusion_attn.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26145 - [Model] Apply shared experts overlap optimization to all models with shared experts

- 链接: https://github.com/vllm-project/vllm/pull/26145
- 状态/时间: merged / 2025-10-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+271/-283，可读 patch 1118 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Apply shared experts overlap optimization to all models with shared experts」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe.py`, `vllm/model_executor/models/ernie45_vl_moe.py`；技术摘要: 覆盖「[Model] Apply shared experts overlap optimization to all models with shared experts」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe.py`, `vllm/model_executor/models/ernie45_vl_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +24/-45 (69 lines); hunks: -49,7 +49,7; -64,7 +64,6; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`；`vllm/model_executor/models/glm4_moe.py` modified +25/-43 (68 lines); hunks: -42,7 +42,7; -52,7 +52,6; symbols: __init__, forward, make_empty_intermediate_tensors, get_expert_mapping，涉及 `__init__, forward, make_empty_intermediate_tensors`；`vllm/model_executor/models/ernie45_vl_moe.py` modified +34/-23 (57 lines); hunks: -37,7 +37,7; -74,7 +74,15; symbols: Ernie4_5_VLMoeMLP, __init__, forward, Ernie4_5_VLMoeAttention，涉及 `Ernie4_5_VLMoeMLP, __init__, forward`；`vllm/model_executor/models/qwen2_moe.py` modified +30/-27 (57 lines); hunks: -40,7 +40,7; -79,6 +79,7 @@ def __init__(; symbols: __init__, forward, Qwen2MoeSparseMoeBlock，涉及 `__init__, forward, Qwen2MoeSparseMoeBlock`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +24/-45 (69 lines); hunks: -49,7 +49,7; -64,7 +64,6; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/glm4_moe.py` modified +25/-43 (68 lines); hunks: -42,7 +42,7; -52,7 +52,6; symbols: __init__, forward, make_empty_intermediate_tensors, get_expert_mapping
  - `vllm/model_executor/models/ernie45_vl_moe.py` modified +34/-23 (57 lines); hunks: -37,7 +37,7; -74,7 +74,15; symbols: Ernie4_5_VLMoeMLP, __init__, forward, Ernie4_5_VLMoeAttention
  - `vllm/model_executor/models/qwen2_moe.py` modified +30/-27 (57 lines); hunks: -40,7 +40,7; -79,6 +79,7 @@ def __init__(; symbols: __init__, forward, Qwen2MoeSparseMoeBlock
  - `vllm/model_executor/models/qwen3_next.py` modified +24/-30 (54 lines); hunks: -7,7 +7,6; -36,7 +35,7; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -49,7 +49,7 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import SharedFusedMoE
@@ -64,7 +64,6 @@
-from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE
@@ -205,26 +204,6 @@ def __init__(
-            self.experts = FusedMoE(
diff -- vllm/model_executor/models/glm4_moe.py
@@ -42,7 +42,7 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import SharedFusedMoE
@@ -52,7 +52,6 @@
-from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE
@@ -176,46 +175,29 @@ def __init__(
-            self.experts = SharedFusedMoE(
diff -- vllm/model_executor/models/ernie45_vl_moe.py
@@ -37,7 +37,7 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +24/-45; `vllm/model_executor/models/glm4_moe.py` modified +25/-43; `vllm/model_executor/models/ernie45_vl_moe.py` modified +34/-23; `vllm/model_executor/models/qwen2_moe.py` modified +30/-27; `vllm/model_executor/models/qwen3_next.py` modified +24/-30; `vllm/model_executor/models/bailing_moe.py` modified +26/-21
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/layers/quantization/fp8.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26633 - Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y`

- 链接: https://github.com/vllm-project/vllm/pull/26633
- 状态/时间: merged / 2025-10-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 944 个文件，+9491/-10122，可读 patch 61484 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y`」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `vllm/entrypoints/openai/protocol.py`, `vllm/entrypoints/llm.py`, `vllm/model_executor/layers/fused_moe/layer.py`；技术摘要: 覆盖「Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y`」；主要实现面是 `vllm/entrypoints/openai/protocol.py`, `vllm/entrypoints/llm.py`, `vllm/model_executor/layers/fused_moe/layer.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/entrypoints/openai/protocol.py` modified +339/-339 (678 lines); hunks: -6,7 +6,7; -54,6 +54,7; symbols: OpenAIBaseModel, field, __log_extra_fields__, ErrorInfo，涉及 `OpenAIBaseModel, field, __log_extra_fields__`；`vllm/entrypoints/llm.py` modified +113/-121 (234 lines); hunks: -2,8 +2,8; -191,36 +191,34 @@ def __init__(; symbols: __init__, get_default_sampling_params, generate，涉及 `__init__, get_default_sampling_params, generate`；`vllm/model_executor/layers/fused_moe/layer.py` modified +108/-108 (216 lines); hunks: -2,10 +2,10; -70,15 +70,15; symbols: _eplb_map_to_physical_and_record, FusedMoEMethodBase, __init__, uses_weight_scale_2_pattern，涉及 `_eplb_map_to_physical_and_record, FusedMoEMethodBase, __init__`；`vllm/entrypoints/chat_utils.py` modified +104/-107 (211 lines); hunks: -5,10 +5,10; -40,7 +40,7; symbols: ChatCompletionContentPartAudioParam, ChatCompletionContentPartImageEmbedsParam, CustomChatCompletionContentPILImageParam, CustomChatCompletionContentSimpleImageParam，涉及 `ChatCompletionContentPartAudioParam, ChatCompletionContentPartImageEmbedsParam, CustomChatCompletionContentPILImageParam`。
- 代码 diff 细节:
  - `vllm/entrypoints/openai/protocol.py` modified +339/-339 (678 lines); hunks: -6,7 +6,7; -54,6 +54,7; symbols: OpenAIBaseModel, field, __log_extra_fields__, ErrorInfo
  - `vllm/entrypoints/llm.py` modified +113/-121 (234 lines); hunks: -2,8 +2,8; -191,36 +191,34 @@ def __init__(; symbols: __init__, get_default_sampling_params, generate
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +108/-108 (216 lines); hunks: -2,10 +2,10; -70,15 +70,15; symbols: _eplb_map_to_physical_and_record, FusedMoEMethodBase, __init__, uses_weight_scale_2_pattern
  - `vllm/entrypoints/chat_utils.py` modified +104/-107 (211 lines); hunks: -5,10 +5,10; -40,7 +40,7; symbols: ChatCompletionContentPartAudioParam, ChatCompletionContentPartImageEmbedsParam, CustomChatCompletionContentPILImageParam, CustomChatCompletionContentSimpleImageParam
  - `vllm/entrypoints/openai/serving_engine.py` modified +94/-96 (190 lines); hunks: -5,10 +5,10; -102,38 +102,38; symbols: TextTokensPrompt, EmbedsPrompt, is_text_tokens_prompt, RequestProcessingMixin
- 关键代码摘录:

```diff
diff -- vllm/entrypoints/openai/protocol.py
@@ -6,7 +6,7 @@
-from typing import Annotated, Any, ClassVar, Generic, Literal, Optional, TypeVar, Union
+from typing import Annotated, Any, ClassVar, Generic, Literal, TypeAlias, TypeVar
@@ -54,6 +54,7 @@
@@ -67,7 +68,6 @@
-from typing_extensions import TypeAlias
@@ -93,7 +93,7 @@ class OpenAIBaseModel(BaseModel):
diff -- vllm/entrypoints/llm.py
@@ -2,8 +2,8 @@
-from collections.abc import Sequence
-from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast
+from collections.abc import Callable, Sequence
+from typing import TYPE_CHECKING, Any, cast
@@ -191,36 +191,34 @@ def __init__(
-        tokenizer: Optional[str] = None,
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -2,10 +2,10 @@
```

- 已读文件:
  - runtime: `vllm/entrypoints/openai/protocol.py` modified +339/-339; `vllm/entrypoints/llm.py` modified +113/-121; `vllm/model_executor/layers/fused_moe/layer.py` modified +108/-108; `vllm/entrypoints/chat_utils.py` modified +104/-107; `vllm/entrypoints/openai/serving_engine.py` modified +94/-96; `vllm/model_executor/models/keye.py` modified +76/-100
- 验证与风险: diff 自带测试面 `tests/benchmarks/test_random_dataset.py`, `tests/ci_envs.py`, `tests/compile/backend.py`, `tests/compile/piecewise/test_toy_llama.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26876 - [Fix] Remove divisibility requirement between num_kv_heads and tp_size in bailing_moe

- 链接: https://github.com/vllm-project/vllm/pull/26876
- 状态/时间: merged / 2025-10-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 15 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Remove divisibility requirement between num_kv_heads and tp_size in bailing_moe」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/bailing_moe.py`；技术摘要: 覆盖「[Fix] Remove divisibility requirement between num_kv_heads and tp_size in bailing_moe」；主要实现面是 `vllm/model_executor/models/bailing_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/bailing_moe.py` modified +1/-2 (3 lines); hunks: -86,13 +86,12 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-2 (3 lines); hunks: -86,13 +86,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe.py
@@ -86,13 +86,12 @@ def __init__(
-        assert self.total_kv_heads % tp_size == 0
-        self.num_kv_heads = self.total_kv_heads // tp_size
+        self.num_kv_heads = max(1, self.total_kv_heads // tp_size)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +1/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28382 - [LoRA][1/N]Remove LoRA extra vocab

- 链接: https://github.com/vllm-project/vllm/pull/28382
- 状态/时间: merged / 2025-11-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 65 个文件，+197/-754，可读 patch 2645 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[LoRA][1/N]Remove LoRA extra vocab」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/phimoe.py`, `vllm/model_executor/models/lfm2_moe.py`, `vllm/model_executor/models/falcon_h1.py`；技术摘要: 覆盖「[LoRA][1/N]Remove LoRA extra vocab」；主要实现面是 `vllm/model_executor/models/phimoe.py`, `vllm/model_executor/models/lfm2_moe.py`, `vllm/model_executor/models/falcon_h1.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/phimoe.py` modified +7/-27 (34 lines); hunks: -45,7 +45,6; -458,22 +457,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, PhiMoEForCausalLM，涉及 `__init__, PhiMoEForCausalLM`；`vllm/model_executor/models/lfm2_moe.py` modified +6/-26 (32 lines); hunks: -33,7 +33,6; -423,20 +422,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/falcon_h1.py` modified +7/-24 (31 lines); hunks: -30,7 +30,6; -424,21 +423,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/lfm2.py` modified +5/-26 (31 lines); hunks: -28,7 +28,6; -316,16 +315,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/phimoe.py` modified +7/-27 (34 lines); hunks: -45,7 +45,6; -458,22 +457,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, PhiMoEForCausalLM
  - `vllm/model_executor/models/lfm2_moe.py` modified +6/-26 (32 lines); hunks: -33,7 +33,6; -423,20 +422,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/falcon_h1.py` modified +7/-24 (31 lines); hunks: -30,7 +30,6; -424,21 +423,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/lfm2.py` modified +5/-26 (31 lines); hunks: -28,7 +28,6; -316,16 +315,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/nemotron_nas.py` modified +6/-25 (31 lines); hunks: -41,7 +41,6; -250,25 +249,19 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/phimoe.py
@@ -45,7 +45,6 @@
-    DEFAULT_VOCAB_PADDING_SIZE,
@@ -458,22 +457,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        lora_config = vllm_config.lora_config
-        lora_vocab = (
-            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
-            if lora_config
diff -- vllm/model_executor/models/lfm2_moe.py
@@ -33,7 +33,6 @@
-    DEFAULT_VOCAB_PADDING_SIZE,
@@ -423,20 +422,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        lora_config = vllm_config.lora_config
-        lora_vocab = (
-            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
-            if lora_config
diff -- vllm/model_executor/models/falcon_h1.py
@@ -30,7 +30,6 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/phimoe.py` modified +7/-27; `vllm/model_executor/models/lfm2_moe.py` modified +6/-26; `vllm/model_executor/models/falcon_h1.py` modified +7/-24; `vllm/model_executor/models/lfm2.py` modified +5/-26; `vllm/model_executor/models/nemotron_nas.py` modified +6/-25; `vllm/model_executor/models/apertus.py` modified +5/-25
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/apertus.py`, `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/arctic.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27583 - Rename clashing method names for vLLM model protocol

- 链接: https://github.com/vllm-project/vllm/pull/27583
- 状态/时间: merged / 2025-11-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 164 个文件，+574/-583，可读 patch 4116 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Rename clashing method names for vLLM model protocol」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/interfaces_base.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/bert.py`；技术摘要: 覆盖「Rename clashing method names for vLLM model protocol」；主要实现面是 `vllm/model_executor/models/interfaces_base.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/bert.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/interfaces_base.py` modified +23/-20 (43 lines); hunks: -41,36 +41,39; -110,7 +113,7 @@ def is_vllm_model(; symbols: VllmModel, __init__, get_input_embeddings, embed_input_ids，涉及 `VllmModel, __init__, get_input_embeddings`；`vllm/model_executor/models/interfaces.py` modified +19/-13 (32 lines); hunks: -94,7 +94,7 @@ def get_placeholder_str(cls, modality: str, i: int) -> str | N...; -104,7 +104,13 @@ def get_multimodal_embeddings(self, **kwargs: object) -> Mu...; symbols: get_placeholder_str, get_multimodal_embeddings, embed_multimodal, get_language_model，涉及 `get_placeholder_str, get_multimodal_embeddings, embed_multimodal`；`vllm/model_executor/models/bert.py` modified +7/-7 (14 lines); hunks: -375,7 +375,7 @@ def __init__(; -486,8 +486,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, get_input_embeddings, embed_input_ids, forward，涉及 `__init__, get_input_embeddings, embed_input_ids`；`vllm/model_executor/models/modernbert.py` modified +7/-7 (14 lines); hunks: -46,7 +46,7 @@ def __init__(self, config: ModernBertConfig):; -225,8 +225,8 @@ def __init__(; symbols: __init__, get_input_embeddings, embed_input_ids, forward，涉及 `__init__, get_input_embeddings, embed_input_ids`。
- 代码 diff 细节:
  - `vllm/model_executor/models/interfaces_base.py` modified +23/-20 (43 lines); hunks: -41,36 +41,39; -110,7 +113,7 @@ def is_vllm_model(; symbols: VllmModel, __init__, get_input_embeddings, embed_input_ids
  - `vllm/model_executor/models/interfaces.py` modified +19/-13 (32 lines); hunks: -94,7 +94,7 @@ def get_placeholder_str(cls, modality: str, i: int) -> str | N...; -104,7 +104,13 @@ def get_multimodal_embeddings(self, **kwargs: object) -> Mu...; symbols: get_placeholder_str, get_multimodal_embeddings, embed_multimodal, get_language_model
  - `vllm/model_executor/models/bert.py` modified +7/-7 (14 lines); hunks: -375,7 +375,7 @@ def __init__(; -486,8 +486,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, get_input_embeddings, embed_input_ids, forward
  - `vllm/model_executor/models/modernbert.py` modified +7/-7 (14 lines); hunks: -46,7 +46,7 @@ def __init__(self, config: ModernBertConfig):; -225,8 +225,8 @@ def __init__(; symbols: __init__, get_input_embeddings, embed_input_ids, forward
  - `vllm/model_executor/models/qwen3_vl.py` modified +6/-8 (14 lines); hunks: -1100,7 +1100,7 @@ def forward(; -1493,9 +1493,7 @@ def get_mrope_input_positions(; symbols: forward, get_mrope_input_positions, get_language_model, get_multimodal_embeddings
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/interfaces_base.py
@@ -41,36 +41,39 @@
-    def __init__(
-        self,
-        vllm_config: VllmConfig,
-        prefix: str = "",
-    ) -> None: ...
+    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
diff -- vllm/model_executor/models/interfaces.py
@@ -94,7 +94,7 @@ def get_placeholder_str(cls, modality: str, i: int) -> str | None:
-    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
+    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
@@ -104,7 +104,13 @@ def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
-        ...
+        if hasattr(self, "get_multimodal_embeddings"):
+            logger.warning_once(
diff -- vllm/model_executor/models/bert.py
@@ -375,7 +375,7 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/interfaces_base.py` modified +23/-20; `vllm/model_executor/models/interfaces.py` modified +19/-13; `vllm/model_executor/models/bert.py` modified +7/-7; `vllm/model_executor/models/modernbert.py` modified +7/-7; `vllm/model_executor/models/qwen3_vl.py` modified +6/-8; `vllm/model_executor/models/clip.py` modified +6/-6
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/apertus.py`, `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/arctic.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28277 - [Model] Fix bailing_moe accuracy problem

- 链接: https://github.com/vllm-project/vllm/pull/28277
- 状态/时间: merged / 2025-11-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-2，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Fix bailing_moe accuracy problem」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/bailing_moe.py`；技术摘要: 覆盖「[Model] Fix bailing_moe accuracy problem」；主要实现面是 `vllm/model_executor/models/bailing_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/bailing_moe.py` modified +3/-2 (5 lines); hunks: -39,7 +39,6; -330,7 +329,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe.py` modified +3/-2 (5 lines); hunks: -39,7 +39,6; -330,7 +329,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe.py
@@ -39,7 +39,6 @@
-    tensor_model_parallel_all_reduce,
@@ -330,7 +329,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
+            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
+                final_hidden_states
+            )
```

- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +3/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28777 - [Model] Fix lmhead init bug of bailing_moe

- 链接: https://github.com/vllm-project/vllm/pull/28777
- 状态/时间: merged / 2025-11-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Fix lmhead init bug of bailing_moe」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/bailing_moe.py`；技术摘要: 覆盖「[Model] Fix lmhead init bug of bailing_moe」；主要实现面是 `vllm/model_executor/models/bailing_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/bailing_moe.py` modified +1/-1 (2 lines); hunks: -599,7 +599,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-1 (2 lines); hunks: -599,7 +599,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe.py
@@ -599,7 +599,7 @@ def __init__(
-                    prefix=f"{prefix}.lm_head",
+                    prefix=maybe_prefix(prefix, "lm_head"),
```

- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28542 - Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5

- 链接: https://github.com/vllm-project/vllm/pull/28542
- 状态/时间: merged / 2025-11-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 104 个文件，+544/-912，可读 patch 4603 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5」；主要实现面是 `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38 (76 lines); hunks: -26,23 +26,23 @@ def get_rope(; -60,15 +60,15 @@ def get_rope(; symbols: get_rope，涉及 `get_rope`；`vllm/transformers_utils/configs/nemotron.py` modified +31/-29 (60 lines); hunks: -88,8 +88,8 @@ class NemotronConfig(PretrainedConfig):; -132,8 +132,7 @@ def __init__(; symbols: NemotronConfig, __init__, _rope_scaling_validation，涉及 `NemotronConfig, __init__, _rope_scaling_validation`；`vllm/model_executor/models/deepseek_v2.py` modified +13/-30 (43 lines); hunks: -27,7 +27,6; -111,8 +110,6 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/chameleon.py` modified +4/-25 (29 lines); hunks: -264,8 +264,7 @@ def __init__(; -292,7 +291,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38 (76 lines); hunks: -26,23 +26,23 @@ def get_rope(; -60,15 +60,15 @@ def get_rope(; symbols: get_rope
  - `vllm/transformers_utils/configs/nemotron.py` modified +31/-29 (60 lines); hunks: -88,8 +88,8 @@ class NemotronConfig(PretrainedConfig):; -132,8 +132,7 @@ def __init__(; symbols: NemotronConfig, __init__, _rope_scaling_validation
  - `vllm/model_executor/models/deepseek_v2.py` modified +13/-30 (43 lines); hunks: -27,7 +27,6; -111,8 +110,6 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/chameleon.py` modified +4/-25 (29 lines); hunks: -264,8 +264,7 @@ def __init__(; -292,7 +291,6 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/openpangu.py` modified +7/-19 (26 lines); hunks: -77,6 +77,7; -259,7 +260,6 @@ def __init__(; symbols: check_ffn_act_fn, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/__init__.py
@@ -26,23 +26,23 @@ def get_rope(
-    base: float,
-    rope_scaling: dict[str, Any] | None = None,
+    rope_parameters: dict[str, Any] | None = None,
-    if rope_scaling is not None:
+    if rope_parameters is not None:
-        rope_scaling_tuple = {
diff -- vllm/transformers_utils/configs/nemotron.py
@@ -88,8 +88,8 @@ class NemotronConfig(PretrainedConfig):
-        rope_theta (`float`, *optional*, defaults to 10000.0):
-            The base period of the RoPE embeddings.
+        rope_parameters (`dict`, *optional*):
+            The parameters of the RoPE embeddings.
@@ -132,8 +132,7 @@ def __init__(
-        rope_theta=10000.0,
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -27,7 +27,6 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38; `vllm/transformers_utils/configs/nemotron.py` modified +31/-29; `vllm/model_executor/models/deepseek_v2.py` modified +13/-30; `vllm/model_executor/models/chameleon.py` modified +4/-25; `vllm/model_executor/models/openpangu.py` modified +7/-19; `vllm/model_executor/models/hunyuan_v1.py` modified +2/-23
- 验证与风险: diff 自带测试面 `tests/compile/test_functionalization.py`, `tests/kernels/core/test_mrope.py`, `tests/kernels/core/test_pos_encoding.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29342 - [Attention] Remove imports from `vllm/attention/__init__.py`

- 链接: https://github.com/vllm-project/vllm/pull/29342
- 状态/时间: merged / 2025-11-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 96 个文件，+120/-121，可读 patch 923 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Attention] Remove imports from `vllm/attention/__init__.py`」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/whisper.py`, `vllm/model_executor/model_loader/utils.py`, `vllm/model_executor/models/afmoe.py`；技术摘要: 覆盖「[Attention] Remove imports from `vllm/attention/__init__.py`」；主要实现面是 `vllm/model_executor/models/whisper.py`, `vllm/model_executor/model_loader/utils.py`, `vllm/model_executor/models/afmoe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/whisper.py` modified +2/-2 (4 lines); hunks: -16,8 +16,8；`vllm/model_executor/model_loader/utils.py` modified +1/-2 (3 lines); hunks: -11,8 +11,7；`vllm/model_executor/models/afmoe.py` modified +2/-1 (3 lines); hunks: -9,7 +9,8；`vllm/model_executor/models/apertus.py` modified +2/-1 (3 lines); hunks: -32,7 +32,8。
- 代码 diff 细节:
  - `vllm/model_executor/models/whisper.py` modified +2/-2 (4 lines); hunks: -16,8 +16,8
  - `vllm/model_executor/model_loader/utils.py` modified +1/-2 (3 lines); hunks: -11,8 +11,7
  - `vllm/model_executor/models/afmoe.py` modified +2/-1 (3 lines); hunks: -9,7 +9,8
  - `vllm/model_executor/models/apertus.py` modified +2/-1 (3 lines); hunks: -32,7 +32,8
  - `vllm/model_executor/models/clip.py` modified +1/-2 (3 lines); hunks: -14,8 +14,7
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/whisper.py
@@ -16,8 +16,8 @@
-from vllm.attention import Attention, AttentionType
-from vllm.attention.layer import MultiHeadAttention
+from vllm.attention.backends.abstract import AttentionType
+from vllm.attention.layer import Attention, MultiHeadAttention
diff -- vllm/model_executor/model_loader/utils.py
@@ -11,8 +11,7 @@
-from vllm.attention import Attention
-from vllm.attention.layer import MLAAttention
+from vllm.attention.layer import Attention, MLAAttention
diff -- vllm/model_executor/models/afmoe.py
@@ -9,7 +9,8 @@
-from vllm.attention import Attention, AttentionType
+from vllm.attention.backends.abstract import AttentionType
+from vllm.attention.layer import Attention
diff -- vllm/model_executor/models/apertus.py
@@ -32,7 +32,8 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/whisper.py` modified +2/-2; `vllm/model_executor/model_loader/utils.py` modified +1/-2; `vllm/model_executor/models/afmoe.py` modified +2/-1; `vllm/model_executor/models/apertus.py` modified +2/-1; `vllm/model_executor/models/clip.py` modified +1/-2; `vllm/model_executor/models/gemma3.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/compile/test_fusion_attn.py`, `tests/compile/test_qk_norm_rope_fusion.py`, `tests/kernels/utils.py`, `tests/v1/worker/test_gpu_model_runner.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29966 - Access `partial_rotary_factor` from `rope_parameters`

- 链接: https://github.com/vllm-project/vllm/pull/29966
- 状态/时间: merged / 2025-12-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 21 个文件，+43/-62，可读 patch 396 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Access `partial_rotary_factor` from `rope_parameters`」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/transformers_utils/configs/nemotron.py`, `vllm/transformers_utils/configs/qwen3_next.py`, `vllm/model_executor/models/gpt_neox.py`；技术摘要: 覆盖「Access `partial_rotary_factor` from `rope_parameters`」；主要实现面是 `vllm/transformers_utils/configs/nemotron.py`, `vllm/transformers_utils/configs/qwen3_next.py`, `vllm/model_executor/models/gpt_neox.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/nemotron.py` modified +13/-7 (20 lines); hunks: -89,9 +89,14 @@ class NemotronConfig(PretrainedConfig):; -133,7 +138,6 @@ def __init__(; symbols: NemotronConfig, __init__，涉及 `NemotronConfig, __init__`；`vllm/transformers_utils/configs/qwen3_next.py` modified +5/-3 (8 lines); hunks: -103,8 +103,8 @@ class Qwen3NextConfig(PretrainedConfig):; -198,7 +198,6 @@ def __init__(; symbols: Qwen3NextConfig, __init__，涉及 `Qwen3NextConfig, __init__`；`vllm/model_executor/models/gpt_neox.py` modified +2/-4 (6 lines); hunks: -89,16 +89,14 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/model_executor/layers/rotary_embedding/__init__.py` modified +4/-1 (5 lines); hunks: -30,7 +30,6 @@ def get_rope(; -55,6 +54,10 @@ def get_rope(; symbols: get_rope，涉及 `get_rope`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/nemotron.py` modified +13/-7 (20 lines); hunks: -89,9 +89,14 @@ class NemotronConfig(PretrainedConfig):; -133,7 +138,6 @@ def __init__(; symbols: NemotronConfig, __init__
  - `vllm/transformers_utils/configs/qwen3_next.py` modified +5/-3 (8 lines); hunks: -103,8 +103,8 @@ class Qwen3NextConfig(PretrainedConfig):; -198,7 +198,6 @@ def __init__(; symbols: Qwen3NextConfig, __init__
  - `vllm/model_executor/models/gpt_neox.py` modified +2/-4 (6 lines); hunks: -89,16 +89,14 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +4/-1 (5 lines); hunks: -30,7 +30,6 @@ def get_rope(; -55,6 +54,10 @@ def get_rope(; symbols: get_rope
  - `vllm/model_executor/models/apertus.py` modified +1/-4 (5 lines); hunks: -148,8 +148,6 @@ def __init__(; -228,11 +226,10 @@ def _init_rotary_emb(; symbols: __init__, _init_rotary_emb
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/configs/nemotron.py
@@ -89,9 +89,14 @@ class NemotronConfig(PretrainedConfig):
-            The parameters of the RoPE embeddings.
-        partial_rotary_factor (`float`, *optional*, defaults to 0.5):
-            Percentage of the query and keys which will have rotary embedding.
+            The parameters of the RoPE embeddings. Expected contents:
+                `rope_theta` (`float`): The base period of the RoPE embeddings.
+                `rope_type` (`str`):
diff -- vllm/transformers_utils/configs/qwen3_next.py
@@ -103,8 +103,8 @@ class Qwen3NextConfig(PretrainedConfig):
-        partial_rotary_factor (`float`, *optional*, defaults to 0.25):
-            Percentage of the query and keys which will have rotary embedding.
+                `partial_rotary_factor` (`float`, *optional*, defaults to 0.25):
+                    Percentage of the query and keys which will have rotary embedding.
@@ -198,7 +198,6 @@ def __init__(
-        partial_rotary_factor=0.25,
diff -- vllm/model_executor/models/gpt_neox.py
@@ -89,16 +89,14 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/nemotron.py` modified +13/-7; `vllm/transformers_utils/configs/qwen3_next.py` modified +5/-3; `vllm/model_executor/models/gpt_neox.py` modified +2/-4; `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +4/-1; `vllm/model_executor/models/apertus.py` modified +1/-4; `vllm/model_executor/models/config.py` modified +0/-5
- 验证与风险: diff 自带测试面 `tests/kernels/core/test_mrope.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30389 - Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim`

- 链接: https://github.com/vllm-project/vllm/pull/30389
- 状态/时间: merged / 2025-12-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 83 个文件，+238/-292，可读 patch 1379 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim`」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/model_executor/models/config.py`, `vllm/model_executor/models/phi.py`；技术摘要: 覆盖「Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim`」；主要实现面是 `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/model_executor/models/config.py`, `vllm/model_executor/models/phi.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +160/-166 (326 lines); hunks: -25,7 +25,6; -54,12 +53,15 @@ def get_rope(; symbols: get_rope，涉及 `get_rope`；`vllm/model_executor/models/config.py` modified +7/-5 (12 lines); hunks: -42,9 +42,10 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:; -77,9 +78,11 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:; symbols: verify_and_update_config，涉及 `verify_and_update_config`；`vllm/model_executor/models/phi.py` modified +4/-8 (12 lines); hunks: -84,19 +84,18 @@ def __init__(; -109,13 +108,10 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/bamba.py` modified +2/-5 (7 lines); hunks: -178,14 +178,11 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +160/-166 (326 lines); hunks: -25,7 +25,6; -54,12 +53,15 @@ def get_rope(; symbols: get_rope
  - `vllm/model_executor/models/config.py` modified +7/-5 (12 lines); hunks: -42,9 +42,10 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:; -77,9 +78,11 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:; symbols: verify_and_update_config
  - `vllm/model_executor/models/phi.py` modified +4/-8 (12 lines); hunks: -84,19 +84,18 @@ def __init__(; -109,13 +108,10 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/bamba.py` modified +2/-5 (7 lines); hunks: -178,14 +178,11 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/chatglm.py` modified +5/-2 (7 lines); hunks: -99,13 +99,16 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/__init__.py
@@ -25,7 +25,6 @@
-    rotary_dim: int,
@@ -54,12 +53,15 @@ def get_rope(
-    partial_rotary_factor = 1.0
-    if rope_parameters is not None:
-        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
+    rope_parameters = rope_parameters or {}
diff -- vllm/model_executor/models/config.py
@@ -42,9 +42,10 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:
+        rotary_dim = getattr(config, "rotary_emb_dim", head_dim)
+        config.rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim
-            "rotary_dim": getattr(config, "rotary_emb_dim", head_dim),
@@ -77,9 +78,11 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:
+            rotary_dim = getattr(config, "rotary_emb_dim", head_dim)
+            config.rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim
diff -- vllm/model_executor/models/phi.py
@@ -84,19 +84,18 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +160/-166; `vllm/model_executor/models/config.py` modified +7/-5; `vllm/model_executor/models/phi.py` modified +4/-8; `vllm/model_executor/models/bamba.py` modified +2/-5; `vllm/model_executor/models/chatglm.py` modified +5/-2; `vllm/model_executor/models/falcon_h1.py` modified +2/-5
- 验证与风险: diff 自带测试面 `tests/compile/test_functionalization.py`, `tests/kernels/core/test_mrope.py`, `tests/kernels/core/test_pos_encoding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31104 - [BugFix] LoRA: Support loading base_layer of experts

- 链接: https://github.com/vllm-project/vllm/pull/31104
- 状态/时间: merged / 2026-01-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+46/-3，可读 patch 319 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] LoRA: Support loading base_layer of experts」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/llama4.py`；技术摘要: 覆盖「[BugFix] LoRA: Support loading base_layer of experts」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3 (13 lines); hunks: -2007,6 +2007,7 @@ def combine_output(states: torch.Tensor) -> torch.Tensor:; -2025,13 +2026,19 @@ def make_expert_params_mapping(; symbols: combine_output, make_expert_params_mapping，涉及 `combine_output, make_expert_params_mapping`；`vllm/model_executor/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -1486,6 +1486,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int,...; -1519,6 +1520,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: get_expert_mapping, load_weights，涉及 `get_expert_mapping, load_weights`；`vllm/model_executor/models/llama4.py` modified +2/-0 (2 lines); hunks: -539,6 +539,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; -548,6 +549,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`；`vllm/model_executor/models/afmoe.py` modified +1/-0 (1 lines); hunks: -475,6 +475,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: get_expert_mapping，涉及 `get_expert_mapping`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3 (13 lines); hunks: -2007,6 +2007,7 @@ def combine_output(states: torch.Tensor) -> torch.Tensor:; -2025,13 +2026,19 @@ def make_expert_params_mapping(; symbols: combine_output, make_expert_params_mapping
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -1486,6 +1486,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int,...; -1519,6 +1520,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: get_expert_mapping, load_weights
  - `vllm/model_executor/models/llama4.py` modified +2/-0 (2 lines); hunks: -539,6 +539,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; -548,6 +549,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
  - `vllm/model_executor/models/afmoe.py` modified +1/-0 (1 lines); hunks: -475,6 +475,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: get_expert_mapping
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-0 (1 lines); hunks: -476,6 +476,7 @@ def forward(; symbols: forward, get_expert_mapping
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -2007,6 +2007,7 @@ def combine_output(states: torch.Tensor) -> torch.Tensor:
+        model: torch.nn.Module,
@@ -2025,13 +2026,19 @@ def make_expert_params_mapping(
+        base_layer = (
+            "base_layer."
+            if any(".base_layer." in name for name, _ in model.named_parameters())
+            else ""
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -1486,6 +1486,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
+            self,
@@ -1519,6 +1520,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            self,
diff -- vllm/model_executor/models/llama4.py
@@ -539,6 +539,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            self,
@@ -548,6 +549,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            self,
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3; `vllm/model_executor/models/deepseek_v2.py` modified +2/-0; `vllm/model_executor/models/llama4.py` modified +2/-0; `vllm/model_executor/models/afmoe.py` modified +1/-0; `vllm/model_executor/models/bailing_moe.py` modified +1/-0; `vllm/model_executor/models/deepseek_eagle.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33063 - [Chore] Update type annotation of `input_ids` in model forward

- 链接: https://github.com/vllm-project/vllm/pull/33063
- 状态/时间: merged / 2026-01-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 164 个文件，+243/-241，可读 patch 2158 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Chore] Update type annotation of `input_ids` in model forward」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py`；技术摘要: 覆盖「[Chore] Update type annotation of `input_ids` in model forward」；主要实现面是 `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/modernbert.py` modified +4/-5 (9 lines); hunks: -54,12 +54,11 @@ def forward(; symbols: forward, ModernBertAttention，涉及 `forward, ModernBertAttention`；`vllm/model_executor/models/gemma3n.py` modified +4/-4 (8 lines); hunks: -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torc...; -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: altup_embed, forward, embed_input_ids, fast_prefill_forward，涉及 `altup_embed, forward, embed_input_ids`；`vllm/model_executor/models/gpt2.py` modified +3/-3 (6 lines); hunks: -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -298,7 +298,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, load_weights，涉及 `embed_input_ids, forward, load_weights`；`vllm/model_executor/models/internlm2.py` modified +3/-3 (6 lines); hunks: -284,7 +284,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -350,7 +350,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, __init__，涉及 `embed_input_ids, forward, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/modernbert.py` modified +4/-5 (9 lines); hunks: -54,12 +54,11 @@ def forward(; symbols: forward, ModernBertAttention
  - `vllm/model_executor/models/gemma3n.py` modified +4/-4 (8 lines); hunks: -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torc...; -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: altup_embed, forward, embed_input_ids, fast_prefill_forward
  - `vllm/model_executor/models/gpt2.py` modified +3/-3 (6 lines); hunks: -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -298,7 +298,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, load_weights
  - `vllm/model_executor/models/internlm2.py` modified +3/-3 (6 lines); hunks: -284,7 +284,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -350,7 +350,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, __init__
  - `vllm/model_executor/models/opt.py` modified +3/-3 (6 lines); hunks: -267,7 +267,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -316,7 +316,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/modernbert.py
@@ -54,12 +54,11 @@ def forward(
-        if inputs_embeds is not None:
-            return self.norm(inputs_embeds)
-        else:
+        if inputs_embeds is None:
-            embeddings = self.norm(inputs_embeds)
-            return embeddings
diff -- vllm/model_executor/models/gemma3n.py
@@ -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torch.Tensor:
-        input_ids: torch.Tensor,
+        input_ids: torch.Tensor | None,
@@ -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
-        input_ids: torch.Tensor,
+        input_ids: torch.Tensor | None,
@@ -964,7 +964,7 @@ def fast_prefill_forward(
diff -- vllm/model_executor/models/gpt2.py
@@ -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/modernbert.py` modified +4/-5; `vllm/model_executor/models/gemma3n.py` modified +4/-4; `vllm/model_executor/models/gpt2.py` modified +3/-3; `vllm/model_executor/models/internlm2.py` modified +3/-3; `vllm/model_executor/models/opt.py` modified +3/-3; `vllm/model_executor/models/afmoe.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32064 - [5/N][Attention] Finish eliminating `vllm/attention` folder

- 链接: https://github.com/vllm-project/vllm/pull/32064
- 状态/时间: merged / 2026-01-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 151 个文件，+585/-527，可读 patch 2850 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[5/N][Attention] Finish eliminating `vllm/attention` folder」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/__init__.py`；技术摘要: 覆盖「[5/N][Attention] Finish eliminating `vllm/attention` folder」；主要实现面是 `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/__init__.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/attention/mla_attention.py` modified +354/-17 (371 lines); hunks: -191,24 +191,38; -217,11 +231,16; symbols: MLAAttention, takes, does, __init__，涉及 `MLAAttention, takes, does`；`vllm/model_executor/layers/attention/attention.py` renamed +42/-315 (357 lines); hunks: -1,23 +1,22; -33,20 +32,54; symbols: validate_kv_sharing_target, should_load_quant_weights, get_kv_cache_spec, MLAAttention，涉及 `validate_kv_sharing_target, should_load_quant_weights, get_kv_cache_spec`；`vllm/model_executor/layers/attention/__init__.py` modified +26/-0 (26 lines); hunks: -0,0 +1,26；`vllm/model_executor/models/whisper.py` modified +5/-3 (8 lines); hunks: -17,16 +17,18。
- 代码 diff 细节:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +354/-17 (371 lines); hunks: -191,24 +191,38; -217,11 +231,16; symbols: MLAAttention, takes, does, __init__
  - `vllm/model_executor/layers/attention/attention.py` renamed +42/-315 (357 lines); hunks: -1,23 +1,22; -33,20 +32,54; symbols: validate_kv_sharing_target, should_load_quant_weights, get_kv_cache_spec, MLAAttention
  - `vllm/model_executor/layers/attention/__init__.py` modified +26/-0 (26 lines); hunks: -0,0 +1,26
  - `vllm/model_executor/models/whisper.py` modified +5/-3 (8 lines); hunks: -17,16 +17,18
  - `vllm/model_executor/models/openpangu.py` modified +3/-2 (5 lines); hunks: -29,7 +29,6; -41,7 +40,8
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -191,24 +191,38 @@
-from typing import ClassVar, Generic, TypeVar
+from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast
+if TYPE_CHECKING:
+    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
+import torch.nn as nn
+import vllm.envs as envs
diff -- vllm/model_executor/layers/attention/attention.py
@@ -1,23 +1,22 @@
-"""Attention layer."""
-from typing import cast
+from typing import TYPE_CHECKING
-from vllm.attention.utils.kv_sharing_utils import validate_kv_sharing_target
-from vllm.attention.utils.kv_transfer_utils import maybe_transfer_kv_layer
+from vllm.model_executor.layers.attention.kv_transfer_utils import (
diff -- vllm/model_executor/layers/attention/__init__.py
@@ -0,0 +1,26 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/attention/mla_attention.py` modified +354/-17; `vllm/model_executor/layers/attention/attention.py` renamed +42/-315; `vllm/model_executor/layers/attention/__init__.py` modified +26/-0; `vllm/model_executor/models/whisper.py` modified +5/-3; `vllm/model_executor/models/openpangu.py` modified +3/-2; `vllm/model_executor/models/apertus.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/compile/test_fusion_attn.py`, `tests/compile/test_qk_norm_rope_fusion.py`, `tests/kernels/attention/test_attention.py`, `tests/kernels/attention/test_mha_attn.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33737 - [Bugfix] Define router_logits_dtype for remaining MoE models

- 链接: https://github.com/vllm-project/vllm/pull/33737
- 状态/时间: merged / 2026-02-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+9/-4，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Define router_logits_dtype for remaining MoE models」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/longcat_flash.py`, `vllm/model_executor/models/flex_olmo.py`, `vllm/model_executor/models/afmoe.py`；技术摘要: 覆盖「[Bugfix] Define router_logits_dtype for remaining MoE models」；主要实现面是 `vllm/model_executor/models/longcat_flash.py`, `vllm/model_executor/models/flex_olmo.py`, `vllm/model_executor/models/afmoe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/longcat_flash.py` modified +4/-3 (7 lines); hunks: -236,9 +236,9 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; -309,6 +309,7 @@ def __init__(; symbols: forward, LongcatRouter, __init__，涉及 `forward, LongcatRouter, __init__`；`vllm/model_executor/models/flex_olmo.py` modified +1/-1 (2 lines); hunks: -71,7 +71,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; -82,6 +81,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/afmoe.py` modified +1/-0 (1 lines); hunks: -142,6 +142,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/bailing_moe.py` modified +1/-0 (1 lines); hunks: -300,6 +300,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/longcat_flash.py` modified +4/-3 (7 lines); hunks: -236,9 +236,9 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; -309,6 +309,7 @@ def __init__(; symbols: forward, LongcatRouter, __init__
  - `vllm/model_executor/models/flex_olmo.py` modified +1/-1 (2 lines); hunks: -71,7 +71,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; -82,6 +81,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__
  - `vllm/model_executor/models/afmoe.py` modified +1/-0 (1 lines); hunks: -142,6 +142,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-0 (1 lines); hunks: -300,6 +300,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/mimo_v2_flash.py` modified +1/-0 (1 lines); hunks: -174,6 +174,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/longcat_flash.py
@@ -236,9 +236,9 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:
-        config,
-        zero_expert_num=0,
-        rounter_params_dtype=torch.bfloat16,
+        config: FlashConfig,
+        zero_expert_num: int,
+        rounter_params_dtype: torch.dtype,
diff -- vllm/model_executor/models/flex_olmo.py
@@ -71,7 +71,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        # Gate always runs at half / full precision for now.
@@ -82,6 +81,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+            router_logits_dtype=torch.float32,
diff -- vllm/model_executor/models/afmoe.py
@@ -142,6 +142,7 @@ def __init__(
+            router_logits_dtype=torch.float32,
diff -- vllm/model_executor/models/bailing_moe.py
@@ -300,6 +300,7 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/longcat_flash.py` modified +4/-3; `vllm/model_executor/models/flex_olmo.py` modified +1/-1; `vllm/model_executor/models/afmoe.py` modified +1/-0; `vllm/model_executor/models/bailing_moe.py` modified +1/-0; `vllm/model_executor/models/mimo_v2_flash.py` modified +1/-0; `vllm/model_executor/models/step3p5.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/flex_olmo.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35102 - [Model] Ring 2.5

- 链接: https://github.com/vllm-project/vllm/pull/35102
- 状态/时间: merged / 2026-02-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+1407/-70，可读 patch 1650 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Ring 2.5」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/bailing_moe_linear.py`, `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/layers/fla/ops/layernorm_guard.py`；技术摘要: 覆盖「[Model] Ring 2.5」；主要实现面是 `vllm/model_executor/models/bailing_moe_linear.py`, `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/layers/fla/ops/layernorm_guard.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/bailing_moe_linear.py` added +1246/-0 (1246 lines); hunks: -0,0 +1,1246; symbols: is_linear_layer, _build_rope_parameters, BailingMoeV25MLAAttention, __init__，涉及 `is_linear_layer, _build_rope_parameters, BailingMoeV25MLAAttention`；`vllm/model_executor/layers/mamba/linear_attn.py` modified +124/-65 (189 lines); hunks: -2,6 +2,7; -43,7 +44,6 @@ def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:; symbols: __init__, weight_loader, _forward, forward_qk，涉及 `__init__, weight_loader, _forward`；`vllm/model_executor/layers/fla/ops/layernorm_guard.py` modified +30/-5 (35 lines); hunks: -84,6 +84,7 @@ def layer_norm_fwd_kernel(; -112,7 +113,10 @@ def layer_norm_fwd_kernel(; symbols: layer_norm_fwd_kernel, layer_norm_fwd，涉及 `layer_norm_fwd_kernel, layer_norm_fwd`；`tests/models/registry.py` modified +3/-0 (3 lines); hunks: -206,6 +206,9 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe_linear.py` added +1246/-0 (1246 lines); hunks: -0,0 +1,1246; symbols: is_linear_layer, _build_rope_parameters, BailingMoeV25MLAAttention, __init__
  - `vllm/model_executor/layers/mamba/linear_attn.py` modified +124/-65 (189 lines); hunks: -2,6 +2,7; -43,7 +44,6 @@ def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:; symbols: __init__, weight_loader, _forward, forward_qk
  - `vllm/model_executor/layers/fla/ops/layernorm_guard.py` modified +30/-5 (35 lines); hunks: -84,6 +84,7 @@ def layer_norm_fwd_kernel(; -112,7 +113,10 @@ def layer_norm_fwd_kernel(; symbols: layer_norm_fwd_kernel, layer_norm_fwd
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunks: -206,6 +206,9 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -372,6 +372,7 @@ th {
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe_linear.py
@@ -0,0 +1,1246 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import copy
+from collections.abc import Iterable
+import torch
+import torch.nn as nn
diff -- vllm/model_executor/layers/mamba/linear_attn.py
@@ -2,6 +2,7 @@
+from collections.abc import Callable
@@ -43,7 +44,6 @@ def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
-        return
@@ -56,7 +56,6 @@ def weight_loader(
-        return
@@ -102,6 +101,101 @@ def forward_qk(
diff -- vllm/model_executor/layers/fla/ops/layernorm_guard.py
@@ -84,6 +84,7 @@ def layer_norm_fwd_kernel(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe_linear.py` added +1246/-0; `vllm/model_executor/layers/mamba/linear_attn.py` modified +124/-65; `vllm/model_executor/layers/fla/ops/layernorm_guard.py` modified +30/-5; `vllm/model_executor/layers/layernorm.py` modified +1/-0; `vllm/model_executor/models/registry.py` modified +1/-0; `vllm/transformers_utils/model_arch_config_convertor.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +3/-0
  - docs: `docs/models/supported_models.md` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37195 - [V0 Deprecation] Deprecate virtual engine

- 链接: https://github.com/vllm-project/vllm/pull/37195
- 状态/时间: merged / 2026-03-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 23 个文件，+23/-45，可读 patch 353 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V0 Deprecation] Deprecate virtual engine」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/mla_attention.py`；技术摘要: 覆盖「[V0 Deprecation] Deprecate virtual engine」；主要实现面是 `vllm/model_executor/models/qwen3_next.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/mla_attention.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_next.py` modified +2/-4 (6 lines); hunks: -815,7 +815,6 @@ def _forward_core(; -826,7 +825,7 @@ def _forward_core(; symbols: _forward_core, _forward_core_decode_non_spec，涉及 `_forward_core, _forward_core_decode_non_spec`；`vllm/model_executor/layers/attention/attention.py` modified +2/-2 (4 lines); hunks: -589,7 +589,7 @@ def get_attention_context(; -600,7 +600,7 @@ def get_attention_context(; symbols: get_attention_context，涉及 `get_attention_context`；`vllm/model_executor/layers/attention/mla_attention.py` modified +2/-2 (4 lines); hunks: -480,7 +480,7 @@ def forward(; -940,7 +940,7 @@ def unified_mla_kv_cache_update(; symbols: forward, unified_mla_kv_cache_update，涉及 `forward, unified_mla_kv_cache_update`；`vllm/model_executor/layers/attention/static_sink_attention.py` modified +1/-2 (3 lines); hunks: -168,8 +168,7 @@ def forward_native(; symbols: forward_native，涉及 `forward_native`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_next.py` modified +2/-4 (6 lines); hunks: -815,7 +815,6 @@ def _forward_core(; -826,7 +825,7 @@ def _forward_core(; symbols: _forward_core, _forward_core_decode_non_spec
  - `vllm/model_executor/layers/attention/attention.py` modified +2/-2 (4 lines); hunks: -589,7 +589,7 @@ def get_attention_context(; -600,7 +600,7 @@ def get_attention_context(; symbols: get_attention_context
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +2/-2 (4 lines); hunks: -480,7 +480,7 @@ def forward(; -940,7 +940,7 @@ def unified_mla_kv_cache_update(; symbols: forward, unified_mla_kv_cache_update
  - `vllm/model_executor/layers/attention/static_sink_attention.py` modified +1/-2 (3 lines); hunks: -168,8 +168,7 @@ def forward_native(; symbols: forward_native
  - `vllm/model_executor/layers/kda.py` modified +1/-1 (2 lines); hunks: -306,7 +306,7 @@ def _forward(; symbols: _forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -815,7 +815,6 @@ def _forward_core(
-                virtual_engine=forward_context.virtual_engine,
@@ -826,7 +825,7 @@ def _forward_core(
-        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
+        self_kv_cache = self.kv_cache[0]
@@ -1009,13 +1008,12 @@ def _forward_core_decode_non_spec(
-        virtual_engine: int,
diff -- vllm/model_executor/layers/attention/attention.py
@@ -589,7 +589,7 @@ def get_attention_context(
-        - kv_cache: The KV cache tensor for current virtual engine
+        - kv_cache: The KV cache tensor for current forward pass
@@ -600,7 +600,7 @@ def get_attention_context(
-    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
+    kv_cache = attn_layer.kv_cache[0]
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -480,7 +480,7 @@ def forward(
-            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +2/-4; `vllm/model_executor/layers/attention/attention.py` modified +2/-2; `vllm/model_executor/layers/attention/mla_attention.py` modified +2/-2; `vllm/model_executor/layers/attention/static_sink_attention.py` modified +1/-2; `vllm/model_executor/layers/kda.py` modified +1/-1; `vllm/model_executor/layers/mamba/linear_attn.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/compile/passes/test_rope_kvcache_fusion.py`, `tests/v1/kv_connector/unit/test_decode_bench_connector.py`, `tests/v1/kv_connector/unit/test_lmcache_integration.py`, `tests/v1/kv_connector/unit/test_nixl_connector.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37487 - [V0 Deprecation] Refactor kv cache from list to element

- 链接: https://github.com/vllm-project/vllm/pull/37487
- 状态/时间: merged / 2026-03-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+70/-85，可读 patch 478 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V0 Deprecation] Refactor kv cache from list to element」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/models/extract_hidden_states.py`；技术摘要: 覆盖「[V0 Deprecation] Refactor kv cache from list to element」；主要实现面是 `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/models/extract_hidden_states.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/attention/mla_attention.py` modified +3/-8 (11 lines); hunks: -416,12 +416,7 @@ def __init__(; -480,7 +475,7 @@ def forward(; symbols: __init__, forward, unified_mla_kv_cache_update，涉及 `__init__, forward, unified_mla_kv_cache_update`；`vllm/model_executor/layers/attention/attention.py` modified +2/-5 (7 lines); hunks: -350,10 +350,7 @@ def __init__(; -600,7 +597,7 @@ def get_attention_context(; symbols: __init__, get_attention_context，涉及 `__init__, get_attention_context`；`vllm/model_executor/models/extract_hidden_states.py` modified +2/-5 (7 lines); hunks: -51,7 +51,7 @@ def unified_kv_cache_update(; -288,10 +288,7 @@ def __init__(; symbols: unified_kv_cache_update, __init__，涉及 `unified_kv_cache_update, __init__`；`vllm/model_executor/layers/sparse_attn_indexer.py` modified +2/-2 (4 lines); hunks: -365,7 +365,7 @@ def forward_cuda(; -389,7 +389,7 @@ def forward_hip(; symbols: forward_cuda, forward_hip，涉及 `forward_cuda, forward_hip`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +3/-8 (11 lines); hunks: -416,12 +416,7 @@ def __init__(; -480,7 +475,7 @@ def forward(; symbols: __init__, forward, unified_mla_kv_cache_update
  - `vllm/model_executor/layers/attention/attention.py` modified +2/-5 (7 lines); hunks: -350,10 +350,7 @@ def __init__(; -600,7 +597,7 @@ def get_attention_context(; symbols: __init__, get_attention_context
  - `vllm/model_executor/models/extract_hidden_states.py` modified +2/-5 (7 lines); hunks: -51,7 +51,7 @@ def unified_kv_cache_update(; -288,10 +288,7 @@ def __init__(; symbols: unified_kv_cache_update, __init__
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +2/-2 (4 lines); hunks: -365,7 +365,7 @@ def forward_cuda(; -389,7 +389,7 @@ def forward_hip(; symbols: forward_cuda, forward_hip
  - `vllm/model_executor/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -858,7 +858,7 @@ def _forward_core(; -1046,7 +1046,7 @@ def _forward_core_decode_non_spec(; symbols: _forward_core, _forward_core_decode_non_spec
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -416,12 +416,7 @@ def __init__(
-        self.kv_cache = [
-            torch.tensor([])
-            for _ in range(
-                get_current_vllm_config().parallel_config.pipeline_parallel_size
-            )
-        ]
diff -- vllm/model_executor/layers/attention/attention.py
@@ -350,10 +350,7 @@ def __init__(
-        self.kv_cache = [
-            torch.tensor([])
-            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
-        ]
+        self.kv_cache = torch.tensor([])
@@ -600,7 +597,7 @@ def get_attention_context(
diff -- vllm/model_executor/models/extract_hidden_states.py
@@ -51,7 +51,7 @@ def unified_kv_cache_update(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/attention/mla_attention.py` modified +3/-8; `vllm/model_executor/layers/attention/attention.py` modified +2/-5; `vllm/model_executor/models/extract_hidden_states.py` modified +2/-5; `vllm/model_executor/layers/sparse_attn_indexer.py` modified +2/-2; `vllm/model_executor/models/qwen3_next.py` modified +2/-2; `vllm/model_executor/layers/attention/static_sink_attention.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/compile/passes/test_fusion_attn.py`, `tests/compile/passes/test_rope_kvcache_fusion.py`, `tests/v1/e2e/general/test_mamba_prefix_cache.py`, `tests/v1/worker/test_gpu_model_runner.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35949 - [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase

- 链接: https://github.com/vllm-project/vllm/pull/35949
- 状态/时间: merged / 2026-04-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 53 个文件，+325/-702，可读 patch 2430 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`；技术摘要: 覆盖「[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase」；主要实现面是 `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[MoE Refactor] Remove SharedFusedMoE class」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`；技术摘要: 覆盖「[MoE Refactor] Remove SharedFusedMoE class」；主要实现面是 `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`；技术摘要: 覆盖「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #40859 - [Bugfix ] fix bailing_moe_linear

- 链接: https://github.com/vllm-project/vllm/pull/40859
- 状态/时间: merged / 2026-04-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+15/-16，可读 patch 90 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix ] fix bailing_moe_linear」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/bailing_moe_linear.py`, `vllm/model_executor/layers/mamba/mamba_utils.py`；技术摘要: 覆盖「[Bugfix ] fix bailing_moe_linear」；主要实现面是 `vllm/model_executor/models/bailing_moe_linear.py`, `vllm/model_executor/layers/mamba/mamba_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/bailing_moe_linear.py` modified +15/-13 (28 lines); hunks: -17,6 +17,7; -211,7 +212,6 @@ def __init__(; symbols: __init__, _weight_loader, BailingMoELinearAttention, mamba_type，涉及 `__init__, _weight_loader, BailingMoELinearAttention`；`vllm/model_executor/layers/mamba/mamba_utils.py` modified +0/-3 (3 lines); hunks: -55,9 +55,6 @@ def linear_attention_state_dtype(; symbols: linear_attention_state_dtype，涉及 `linear_attention_state_dtype`。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe_linear.py` modified +15/-13 (28 lines); hunks: -17,6 +17,7; -211,7 +212,6 @@ def __init__(; symbols: __init__, _weight_loader, BailingMoELinearAttention, mamba_type
  - `vllm/model_executor/layers/mamba/mamba_utils.py` modified +0/-3 (3 lines); hunks: -55,9 +55,6 @@ def linear_attention_state_dtype(; symbols: linear_attention_state_dtype
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe_linear.py
@@ -17,6 +17,7 @@
+from vllm.model_executor.custom_op import PluggableLayer
@@ -211,7 +212,6 @@ def __init__(
-            dtype=torch.float32,
@@ -425,14 +425,18 @@ def _weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> No
-class BailingMoELinearAttention(nn.Module, MambaBase):
-    """
diff -- vllm/model_executor/layers/mamba/mamba_utils.py
@@ -55,9 +55,6 @@ def linear_attention_state_dtype(
-        # TODO (tdoublep) requires testing
-        if mamba_cache_dtype == "float32":
-            raise ValueError("fp32 state for minimax is not yet supported")
```

- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe_linear.py` modified +15/-13; `vllm/model_executor/layers/mamba/mamba_utils.py` modified +0/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/models/bailing_moe_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41185 - [Bugfix] BailingMoeV2.5: rotate full qk_rope_head_dim in MLA RoPE

- 链接: https://github.com/vllm-project/vllm/pull/41185
- 状态/时间: merged / 2026-04-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-2，可读 patch 22 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] BailingMoeV2.5: rotate full qk_rope_head_dim in MLA RoPE」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/bailing_moe_linear.py`；技术摘要: 覆盖「[Bugfix] BailingMoeV2.5: rotate full qk_rope_head_dim in MLA RoPE」；主要实现面是 `vllm/model_executor/models/bailing_moe_linear.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/bailing_moe_linear.py` modified +8/-2 (10 lines); hunks: -205,13 +205,19 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe_linear.py` modified +8/-2 (10 lines); hunks: -205,13 +205,19 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe_linear.py
@@ -205,13 +205,19 @@ def __init__(
-        rope_parameters = _build_rope_parameters(config)
+        rope_parameters = _build_rope_parameters(config) or {}
+        # MLA rotates the full qk_rope_head_dim,
+        # partial_rotary_factor is for the linear-attn head only.
+        rope_parameters = {
+            k: v for k, v in rope_parameters.items() if k != "partial_rotary_factor"
```

- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe_linear.py` modified +8/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/bailing_moe_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41188 - [Misc] Replace mamba_type string literals with MambaAttentionBackendEnum

- 链接: https://github.com/vllm-project/vllm/pull/41188
- 状态/时间: merged / 2026-05-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+64/-58，可读 patch 404 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Replace mamba_type string literals with MambaAttentionBackendEnum」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/kda.py`, `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/linear_attn.py`；技术摘要: 覆盖「[Misc] Replace mamba_type string literals with MambaAttentionBackendEnum」；主要实现面是 `vllm/model_executor/layers/kda.py`, `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/linear_attn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/kda.py` modified +3/-2 (5 lines); hunks: -17,6 +17,7; -84,8 +85,8 @@ def kda_attention_fake(; symbols: kda_attention_fake, KimiDeltaAttention, mamba_type, get_state_dtype，涉及 `kda_attention_fake, KimiDeltaAttention, mamba_type`；`vllm/model_executor/layers/mamba/gdn_linear_attn.py` modified +3/-2 (5 lines); hunks: -64,6 +64,7; -237,8 +238,8 @@ def forward_native(; symbols: forward_native, GatedDeltaNetAttention, mamba_type, get_state_dtype，涉及 `forward_native, GatedDeltaNetAttention, mamba_type`；`vllm/model_executor/layers/mamba/linear_attn.py` modified +3/-2 (5 lines); hunks: -32,6 +32,7; -246,8 +247,8 @@ def jit_linear_forward_prefix(; symbols: jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type, get_state_dtype，涉及 `jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type`；`vllm/model_executor/layers/mamba/mamba_mixer.py` modified +3/-2 (5 lines); hunks: -42,6 +42,7; -476,8 +477,8 @@ def get_state_shape(self) -> tuple[tuple[int, ...], tuple[in...; symbols: get_state_shape, mamba_type, _time_proj_bias，涉及 `get_state_shape, mamba_type, _time_proj_bias`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/kda.py` modified +3/-2 (5 lines); hunks: -17,6 +17,7; -84,8 +85,8 @@ def kda_attention_fake(; symbols: kda_attention_fake, KimiDeltaAttention, mamba_type, get_state_dtype
  - `vllm/model_executor/layers/mamba/gdn_linear_attn.py` modified +3/-2 (5 lines); hunks: -64,6 +64,7; -237,8 +238,8 @@ def forward_native(; symbols: forward_native, GatedDeltaNetAttention, mamba_type, get_state_dtype
  - `vllm/model_executor/layers/mamba/linear_attn.py` modified +3/-2 (5 lines); hunks: -32,6 +32,7; -246,8 +247,8 @@ def jit_linear_forward_prefix(; symbols: jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type, get_state_dtype
  - `vllm/model_executor/layers/mamba/mamba_mixer.py` modified +3/-2 (5 lines); hunks: -42,6 +42,7; -476,8 +477,8 @@ def get_state_shape(self) -> tuple[tuple[int, ...], tuple[in...; symbols: get_state_shape, mamba_type, _time_proj_bias
  - `vllm/model_executor/layers/mamba/mamba_mixer2.py` modified +3/-2 (5 lines); hunks: -52,6 +52,7; -935,8 +936,8 @@ def get_state_shape(self) -> tuple[tuple[int, ...], tuple[in...; symbols: get_state_shape, mamba_type, mamba_mixer2
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/kda.py
@@ -17,6 +17,7 @@
+from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
@@ -84,8 +85,8 @@ def kda_attention_fake(
-    def mamba_type(self) -> str:
-        return "gdn_attention"
+    def mamba_type(self) -> MambaAttentionBackendEnum:
+        return MambaAttentionBackendEnum.GDN_ATTN
diff -- vllm/model_executor/layers/mamba/gdn_linear_attn.py
@@ -64,6 +64,7 @@
+from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
@@ -237,8 +238,8 @@ def forward_native(
-    def mamba_type(self) -> str:
-        return "gdn_attention"
+    def mamba_type(self) -> MambaAttentionBackendEnum:
+        return MambaAttentionBackendEnum.GDN_ATTN
diff -- vllm/model_executor/layers/mamba/linear_attn.py
@@ -32,6 +32,7 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/kda.py` modified +3/-2; `vllm/model_executor/layers/mamba/gdn_linear_attn.py` modified +3/-2; `vllm/model_executor/layers/mamba/linear_attn.py` modified +3/-2; `vllm/model_executor/layers/mamba/mamba_mixer.py` modified +3/-2; `vllm/model_executor/layers/mamba/mamba_mixer2.py` modified +3/-2; `vllm/model_executor/layers/mamba/short_conv.py` modified +3/-2
- 验证与风险: diff 自带测试面 `tests/kernels/mamba/test_ssu_dispatch.py`, `tests/v1/attention/test_attention_backends_selection.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43410 - [Kernel] Porting fuse_minimax_qk_norm to manual fusion

- 链接: https://github.com/vllm-project/vllm/pull/43410
- 状态/时间: merged / 2026-05-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+262/-490，可读 patch 893 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Porting fuse_minimax_qk_norm to manual fusion」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`, `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/layers/minimax_rms_norm/__init__.py`；技术摘要: 覆盖「[Kernel] Porting fuse_minimax_qk_norm to manual fusion」；主要实现面是 `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`, `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/layers/minimax_rms_norm/__init__.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` added +234/-0 (234 lines); hunks: -0,0 +1,234; symbols: _minimax_qk_norm_fallback, _minimax_qk_norm_fusion, _minimax_qk_norm_fusion_fake, MiniMaxText01RMSNormTP，涉及 `_minimax_qk_norm_fallback, _minimax_qk_norm_fusion, _minimax_qk_norm_fusion_fake`；`vllm/model_executor/layers/mamba/linear_attn.py` modified +1/-89 (90 lines); hunks: -3,21 +3,18; -28,99 +25,14; symbols: MiniMaxText01RMSNormTP, __init__, weight_loader, _forward，涉及 `MiniMaxText01RMSNormTP, __init__, weight_loader`；`vllm/model_executor/layers/minimax_rms_norm/__init__.py` added +10/-0 (10 lines); hunks: -0,0 +1,10；`vllm/model_executor/models/minimax_m2.py` modified +4/-3 (7 lines); hunks: -50,7 +50,7; -243,8 +243,9 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` added +234/-0 (234 lines); hunks: -0,0 +1,234; symbols: _minimax_qk_norm_fallback, _minimax_qk_norm_fusion, _minimax_qk_norm_fusion_fake, MiniMaxText01RMSNormTP
  - `vllm/model_executor/layers/mamba/linear_attn.py` modified +1/-89 (90 lines); hunks: -3,21 +3,18; -28,99 +25,14; symbols: MiniMaxText01RMSNormTP, __init__, weight_loader, _forward
  - `vllm/model_executor/layers/minimax_rms_norm/__init__.py` added +10/-0 (10 lines); hunks: -0,0 +1,10
  - `vllm/model_executor/models/minimax_m2.py` modified +4/-3 (7 lines); hunks: -50,7 +50,7; -243,8 +243,9 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/bailing_moe_linear.py` modified +1/-1 (2 lines); hunks: -39,7 +39,6; -49,6 +48,7
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py
@@ -0,0 +1,234 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from functools import partial
+import torch
+from torch import nn
+from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
diff -- vllm/model_executor/layers/mamba/linear_attn.py
@@ -3,21 +3,18 @@
-from functools import partial
-from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
-from vllm.model_executor.custom_op import CustomOp
@@ -28,99 +25,14 @@
+from vllm.model_executor.layers.minimax_rms_norm import MiniMaxText01RMSNormTP
-@CustomOp.register("minimax_text01_rmsnorm_tp")
diff -- vllm/model_executor/layers/minimax_rms_norm/__init__.py
@@ -0,0 +1,10 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` added +234/-0; `vllm/model_executor/layers/mamba/linear_attn.py` modified +1/-89; `vllm/model_executor/layers/minimax_rms_norm/__init__.py` added +10/-0; `vllm/model_executor/models/minimax_m2.py` modified +4/-3; `vllm/model_executor/models/bailing_moe_linear.py` modified +1/-1; `vllm/model_executor/layers/minimax_rms_norm/lamport_workspace.py` renamed +0/-0
  - docs: `docs/design/fusions.md` modified +0/-34
- 验证与风险: diff 自带测试面 `tests/kernels/core/test_minimax_reduce_rms.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43770 - [Bugfix] fix wrong partial_rotary_factor calculation for bailing_moe model.

- 链接: https://github.com/vllm-project/vllm/pull/43770
- 状态/时间: merged / 2026-06-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-1，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] fix wrong partial_rotary_factor calculation for bailing_moe model.」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/bailing_moe.py`；技术摘要: 覆盖「[Bugfix] fix wrong partial_rotary_factor calculation for bailing_moe model.」；主要实现面是 `vllm/model_executor/models/bailing_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/bailing_moe.py` modified +6/-1 (7 lines); hunks: -130,7 +130,12 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe.py` modified +6/-1 (7 lines); hunks: -130,7 +130,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe.py
@@ -130,7 +130,12 @@ def __init__(
-        rotary_dim = getattr(config, "rotary_dim", self.head_dim)
+        rotary_dim = getattr(config, "rotary_dim", None)
+        if rotary_dim is None:
+            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
+            rotary_dim = int(self.head_dim * partial_rotary_factor)
+        if rotary_dim is None:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43556 - [Attention] Mamba attention module refactor - LINEAR

- 链接: https://github.com/vllm-project/vllm/pull/43556
- 状态/时间: merged / 2026-06-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+505/-551，可读 patch 1309 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Attention] Mamba attention module refactor - LINEAR」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/bailing_moe_linear.py`, `vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py`, `vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py`；技术摘要: 覆盖「[Attention] Mamba attention module refactor - LINEAR」；主要实现面是 `vllm/model_executor/models/bailing_moe_linear.py`, `vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py`, `vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/bailing_moe_linear.py` modified +13/-439 (452 lines); hunks: -9,19 +9,14; -30,25 +25,19; symbols: is_linear_layer, _build_rope_parameters, BailingMoeV25MLAAttention, forward，涉及 `is_linear_layer, _build_rope_parameters, BailingMoeV25MLAAttention`；`vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py` added +384/-0 (384 lines); hunks: -0,0 +1,384; symbols: _build_rope_parameters, BailingGroupRMSNormGate, __init__, _weight_loader，涉及 `_build_rope_parameters, BailingGroupRMSNormGate, __init__`；`vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py` renamed +18/-68 (86 lines); hunks: -7,30 +7,20; -157,79 +147,39 @@ def jit_linear_forward_prefix(; symbols: clear_linear_attention_cache_for_new_sequences, jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type，涉及 `clear_linear_attention_cache_for_new_sequences, jit_linear_forward_prefix, MiniMaxText01LinearAttention`；`vllm/model_executor/layers/mamba/linear/base.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: LinearAttention, for, __init__, mamba_type，涉及 `LinearAttention, for, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe_linear.py` modified +13/-439 (452 lines); hunks: -9,19 +9,14; -30,25 +25,19; symbols: is_linear_layer, _build_rope_parameters, BailingMoeV25MLAAttention, forward
  - `vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py` added +384/-0 (384 lines); hunks: -0,0 +1,384; symbols: _build_rope_parameters, BailingGroupRMSNormGate, __init__, _weight_loader
  - `vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py` renamed +18/-68 (86 lines); hunks: -7,30 +7,20; -157,79 +147,39 @@ def jit_linear_forward_prefix(; symbols: clear_linear_attention_cache_for_new_sequences, jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type
  - `vllm/model_executor/layers/mamba/linear/base.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: LinearAttention, for, __init__, mamba_type
  - `vllm/model_executor/models/minimax_text_01.py` modified +14/-35 (49 lines); hunks: -15,7 +15,7; -35,7 +35,9; symbols: MiniMaxText01DecoderLayer, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe_linear.py
@@ -9,19 +9,14 @@
-from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
+from vllm.config import CacheConfig, VllmConfig
-from vllm.model_executor.custom_op import PluggableLayer
-from vllm.model_executor.layers.fla.ops.layernorm_guard import (
-    RMSNormGated,
-    layernorm_fn,
diff -- vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py
@@ -0,0 +1,384 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import copy
+import torch
+import torch.nn.functional as F
+from transformers.configuration_utils import PretrainedConfig
diff -- vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py
@@ -7,30 +7,20 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe_linear.py` modified +13/-439; `vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py` added +384/-0; `vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py` renamed +18/-68; `vllm/model_executor/layers/mamba/linear/base.py` added +66/-0; `vllm/model_executor/models/minimax_text_01.py` modified +14/-35; `vllm/model_executor/layers/mamba/linear/__init__.py` added +0/-0
  - tests: `tests/v1/attention/test_attention_backends_selection.py` modified +10/-9
- 验证与风险: diff 自带测试面 `tests/v1/attention/test_attention_backends_selection.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41184 - [MoE Refactor] FusedMoE/MoERunner inversion refactor

- 链接: https://github.com/vllm-project/vllm/pull/41184
- 状态/时间: merged / 2026-06-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 90 个文件，+2734/-2027，可读 patch 7329 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] FusedMoE/MoERunner inversion refactor」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`；技术摘要: 覆盖「[MoE Refactor] FusedMoE/MoERunner inversion refactor」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`。下方保留文件级证据、代码摘录和验证风险。
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

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
