# vllm Qwen3.5 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/pooling/score/colqwen3_5_rerank_online.py` | [#36887](https://github.com/vllm-project/vllm/pull/36887), [#46108](https://github.com/vllm-project/vllm/pull/46108) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml` | [#38083](https://github.com/vllm-project/vllm/pull/38083), [#45002](https://github.com/vllm-project/vllm/pull/45002) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml` | [#38083](https://github.com/vllm-project/vllm/pull/38083) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-AITER-TP2.yaml` | [#46520](https://github.com/vllm-project/vllm/pull/46520) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-EMU-TP2.yaml` | 无直接 PR 号提交 |
| `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2-MTP.yaml` | [#44700](https://github.com/vllm-project/vllm/pull/44700) |
| `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` | [#38083](https://github.com/vllm-project/vllm/pull/38083), [#38632](https://github.com/vllm-project/vllm/pull/38632) |
| `tests/evals/gsm8k/configs/models-qwen35-blackwell.txt` | [#38083](https://github.com/vllm-project/vllm/pull/38083), [#44700](https://github.com/vllm-project/vllm/pull/44700) |
| `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` | [#38155](https://github.com/vllm-project/vllm/pull/38155), [#38664](https://github.com/vllm-project/vllm/pull/38664) |
| `tests/evals/mrcr/configs/Qwen3.5-4B.yaml` | 无直接 PR 号提交 |
| `tests/lora/test_qwen35_densemodel_lora.py` | [#37816](https://github.com/vllm-project/vllm/pull/37816) |
| `tests/model_executor/test_qwen3_5_quantization.py` | 无直接 PR 号提交 |
| `tests/models/multimodal/pooling/test_colqwen3_5.py` | [#36887](https://github.com/vllm-project/vllm/pull/36887), [#46108](https://github.com/vllm-project/vllm/pull/46108) |
| `vllm/model_executor/models/colqwen3_5.py` | [#36887](https://github.com/vllm-project/vllm/pull/36887), [#46108](https://github.com/vllm-project/vllm/pull/46108) |
| `vllm/model_executor/models/qwen3_5.py` | [#34110](https://github.com/vllm-project/vllm/pull/34110), [#34198](https://github.com/vllm-project/vllm/pull/34198), [#34200](https://github.com/vllm-project/vllm/pull/34200), [#34313](https://github.com/vllm-project/vllm/pull/34313), [#34489](https://github.com/vllm-project/vllm/pull/34489), [#34492](https://github.com/vllm-project/vllm/pull/34492), [#34512](https://github.com/vllm-project/vllm/pull/34512), [#34683](https://github.com/vllm-project/vllm/pull/34683), [#34697](https://github.com/vllm-project/vllm/pull/34697), [#34719](https://github.com/vllm-project/vllm/pull/34719), [#34723](https://github.com/vllm-project/vllm/pull/34723), [#35617](https://github.com/vllm-project/vllm/pull/35617), ... (23 total) |
| `vllm/model_executor/models/qwen3_5_mtp.py` | [#34110](https://github.com/vllm-project/vllm/pull/34110), [#34512](https://github.com/vllm-project/vllm/pull/34512), [#35581](https://github.com/vllm-project/vllm/pull/35581), [#37114](https://github.com/vllm-project/vllm/pull/37114), [#38832](https://github.com/vllm-project/vllm/pull/38832), [#42716](https://github.com/vllm-project/vllm/pull/42716), [#45002](https://github.com/vllm-project/vllm/pull/45002) |
| `vllm/transformers_utils/configs/qwen3_5.py` | [#34512](https://github.com/vllm-project/vllm/pull/34512), [#34554](https://github.com/vllm-project/vllm/pull/34554), [#34604](https://github.com/vllm-project/vllm/pull/34604), [#34610](https://github.com/vllm-project/vllm/pull/34610) |
| `vllm/transformers_utils/configs/qwen3_5_moe.py` | [#34512](https://github.com/vllm-project/vllm/pull/34512), [#34554](https://github.com/vllm-project/vllm/pull/34554), [#34604](https://github.com/vllm-project/vllm/pull/34604), [#34610](https://github.com/vllm-project/vllm/pull/34610) |

## PR 覆盖总览

- git 追溯 PR 数: 2
- 原文档显式引用补充 PR 数: 11
- 当前文档总 PR 数: 8
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-01-07 | [#31104](https://github.com/vllm-project/vllm/pull/31104) | merged | [BugFix] LoRA: Support loading base_layer of experts | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/llama4.py` |
| 2026-05-13 | [#42151](https://github.com/vllm-project/vllm/pull/42151) | merged | [MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-05-17 | [#42716](https://github.com/vllm-project/vllm/pull/42716) | merged | Fix Weight loading for Qwen3.5-MTP and Qwen3-VL using runai_streamer | `vllm/model_executor/models/qwen3_5_mtp.py` |
| 2026-05-18 | [#41436](https://github.com/vllm-project/vllm/pull/41436) | merged | [ROCm][Quantization][3/N] Refactor quark_moe w4a4 w/ oracle | `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` |
| 2026-05-22 | [#41126](https://github.com/vllm-project/vllm/pull/41126) | merged | [Attention] Mamba attention module refactor | `vllm/model_executor/models/olmo_hybrid.py`, `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` |
| 2026-05-26 | [#42124](https://github.com/vllm-project/vllm/pull/42124) | merged | Add LM head quantization support for ModelOpt | `tests/model_executor/test_qwen3_5_quantization.py`, `tests/model_executor/test_nemotron_h_quantization.py`, `vllm/model_executor/layers/quantization/modelopt.py` |
| 2026-06-10 | [#39419](https://github.com/vllm-project/vllm/pull/39419) | merged | [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding | `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py` |
| 2026-06-11 | [#45161](https://github.com/vllm-project/vllm/pull/45161) | merged | Deprecate Transformers v4 support | `vllm/model_executor/models/transformers/base.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/model_loader/weight_utils.py` |

## 逐 PR diff 审计卡

### PR #31104 - [BugFix] LoRA: Support loading base_layer of experts

- 链接: https://github.com/vllm-project/vllm/pull/31104
- 状态/时间: merged / 2026-01-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+46/-3，可读 patch 319 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] LoRA: Support loading base_layer of experts」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/llama4.py`；技术摘要: 覆盖「[BugFix] LoRA: Support loading base_layer of experts」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #42151 - [MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5

- 链接: https://github.com/vllm-project/vllm/pull/42151
- 状态/时间: merged / 2026-05-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `92def124bcb7`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+112/-5，可读 patch 187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「[MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -565,6 +565,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -778,6 +779,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -565,6 +565,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -778,6 +779,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -565,6 +565,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
+        self.model_config = vllm_config.model_config
@@ -778,6 +779,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
+        self.model_config = vllm_config.model_config
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +2/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_vit_cudagraph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42716 - Fix Weight loading for Qwen3.5-MTP and Qwen3-VL using runai_streamer

- 链接: https://github.com/vllm-project/vllm/pull/42716
- 状态/时间: merged / 2026-05-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5_mtp.py`；关联提交 `a94189295b8b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-4，可读 patch 22 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Weight loading for Qwen3.5-MTP and Qwen3-VL using runai_streamer」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5_mtp.py`；技术摘要: 覆盖「Fix Weight loading for Qwen3.5-MTP and Qwen3-VL using runai_streamer」；主要实现面是 `vllm/model_executor/models/qwen3_5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-2 (4 lines); hunks: -175,8 +175,8 @@ def load_fused_expert_weights(; symbols: load_fused_expert_weights，涉及 `load_fused_expert_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-2 (4 lines); hunks: -175,8 +175,8 @@ def load_fused_expert_weights(; symbols: load_fused_expert_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5_mtp.py
@@ -175,8 +175,8 @@ def load_fused_expert_weights(
-                shard_id,
-                expert_id,
+                shard_id=shard_id,
+                expert_id=expert_id,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5_mtp.py`, `vllm/model_executor/models/qwen3_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41436 - [ROCm][Quantization][3/N] Refactor quark_moe w4a4 w/ oracle

- 链接: https://github.com/vllm-project/vllm/pull/41436
- 状态/时间: merged / 2026-05-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+224/-158，可读 patch 564 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Quantization][3/N] Refactor quark_moe w4a4 w/ oracle」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py`；技术摘要: 覆盖「[ROCm][Quantization][3/N] Refactor quark_moe w4a4 w/ oracle」；主要实现面是 `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +26/-151 (177 lines); hunks: -55,6 +55,7; -1040,6 +1041,11 @@ def __init__(; symbols: __init__, maybe_roundup_sizes, create_weights, process_weights_after_loading，涉及 `__init__, maybe_roundup_sizes, create_weights`；`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +76/-2 (78 lines); hunks: -31,6 +31,7; -74,6 +75,7 @@ class Mxfp4MoeBackend(Enum):; symbols: Mxfp4MoeBackend, backend_to_kernel_cls, map_mxfp4_backend, _get_priority_backends_for_gpt_oss，涉及 `Mxfp4MoeBackend, backend_to_kernel_cls, map_mxfp4_backend`；`vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` modified +17/-0 (17 lines); hunks: -26,6 +26,7; -377,6 +378,21 @@ def expects_unquantized_inputs(self) -> bool:; symbols: expects_unquantized_inputs, activation_format, is_supported_config, _supports_current_device，涉及 `expects_unquantized_inputs, activation_format, is_supported_config`；`tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-AITER-TP2.yaml` added +12/-0 (12 lines); hunks: -0,0 +1,12。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +26/-151 (177 lines); hunks: -55,6 +55,7; -1040,6 +1041,11 @@ def __init__(; symbols: __init__, maybe_roundup_sizes, create_weights, process_weights_after_loading
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +76/-2 (78 lines); hunks: -31,6 +31,7; -74,6 +75,7 @@ class Mxfp4MoeBackend(Enum):; symbols: Mxfp4MoeBackend, backend_to_kernel_cls, map_mxfp4_backend, _get_priority_backends_for_gpt_oss
  - `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` modified +17/-0 (17 lines); hunks: -26,6 +26,7; -377,6 +378,21 @@ def expects_unquantized_inputs(self) -> bool:; symbols: expects_unquantized_inputs, activation_format, is_supported_config, _supports_current_device
  - `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-AITER-TP2.yaml` added +12/-0 (12 lines); hunks: -0,0 +1,12
  - `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-EMU-TP2.yaml` renamed +3/-1 (4 lines); hunks: -1,8 +1,10
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -55,6 +55,7 @@
+    kMxfp4Dynamic,
@@ -1040,6 +1041,11 @@ def __init__(
+        elif self.ocp_mx_scheme == "w_mxfp4_a_mxfp4":
+            # W4A4: MXFP4 weights + MXFP4 activations
+            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(
+                moe, activation_key=kMxfp4Dynamic
diff -- vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
@@ -31,6 +31,7 @@
+    kMxfp4Dynamic,
@@ -74,6 +75,7 @@ class Mxfp4MoeBackend(Enum):
+    AITER_MXFP4_MXFP4 = "AITER_MXFP4_MXFP4"  # W4A4: CK kernel
@@ -89,6 +91,7 @@ class Mxfp4MoeBackend(Enum):
+    Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
@@ -193,6 +196,13 @@ def backend_to_kernel_cls(
diff -- vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py
@@ -26,6 +26,7 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +26/-151; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +76/-2; `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` modified +17/-0
  - tests: `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-AITER-TP2.yaml` added +12/-0; `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-EMU-TP2.yaml` renamed +3/-1; `tests/evals/gsm8k/configs/models-mi3xx.txt` modified +2/-1; `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` modified +2/-1; `tests/quantization/test_gfx950_moe.py` modified +86/-2
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-AITER-TP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-EMU-TP2.yaml`, `tests/evals/gsm8k/configs/models-mi3xx.txt`, `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41126 - [Attention] Mamba attention module refactor

- 链接: https://github.com/vllm-project/vllm/pull/41126
- 状态/时间: merged / 2026-05-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+765/-774，可读 patch 1913 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Attention] Mamba attention module refactor」；模型线: Qwen3.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/olmo_hybrid.py`, `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`；技术摘要: 覆盖「[Attention] Mamba attention module refactor」；主要实现面是 `vllm/model_executor/models/olmo_hybrid.py`, `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/olmo_hybrid.py` modified +6/-645 (651 lines); hunks: -26,73 +26,47; -107,502 +81,6; symbols: _make_fused_conv1d_weight_loader, weight_loader, OlmoHybridGatedDeltaNet, mamba_type，涉及 `_make_fused_conv1d_weight_loader, weight_loader, OlmoHybridGatedDeltaNet`；`vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` added +634/-0 (634 lines); hunks: -0,0 +1,634; symbols: OlmoHybridGatedDeltaNetAttention, get_state_shape, __init__, rearrange_mixed_qkv，涉及 `OlmoHybridGatedDeltaNetAttention, get_state_shape, __init__`；`vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` renamed +26/-45 (71 lines); hunks: -5,39 +5,37; -83,11 +81,8 @@ def kda_attention_fake(; symbols: kda_attention_fake, KimiDeltaAttention, mamba_type, KimiGatedDeltaNetAttention，涉及 `kda_attention_fake, KimiDeltaAttention, mamba_type`；`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` renamed +19/-52 (71 lines); hunks: -5,7 +5,6; -15,8 +14,6; symbols: forward_native, GatedDeltaNetAttention, mamba_type, get_state_dtype，涉及 `forward_native, GatedDeltaNetAttention, mamba_type`。
- 代码 diff 细节:
  - `vllm/model_executor/models/olmo_hybrid.py` modified +6/-645 (651 lines); hunks: -26,73 +26,47; -107,502 +81,6; symbols: _make_fused_conv1d_weight_loader, weight_loader, OlmoHybridGatedDeltaNet, mamba_type
  - `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` added +634/-0 (634 lines); hunks: -0,0 +1,634; symbols: OlmoHybridGatedDeltaNetAttention, get_state_shape, __init__, rearrange_mixed_qkv
  - `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` renamed +26/-45 (71 lines); hunks: -5,39 +5,37; -83,11 +81,8 @@ def kda_attention_fake(; symbols: kda_attention_fake, KimiDeltaAttention, mamba_type, KimiGatedDeltaNetAttention
  - `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` renamed +19/-52 (71 lines); hunks: -5,7 +5,6; -15,8 +14,6; symbols: forward_native, GatedDeltaNetAttention, mamba_type, get_state_dtype
  - `vllm/model_executor/layers/mamba/gdn/base.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: GatedDeltaNetAttention, for, __init__, mamba_type
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/olmo_hybrid.py
@@ -26,73 +26,47 @@
-from einops import rearrange
-from transformers.activations import ACT2FN
-    CacheConfig,
-    ModelConfig,
-    SpeculativeConfig,
-    get_current_vllm_config,
diff -- vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py
@@ -0,0 +1,634 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+from einops import rearrange
+from torch import nn
+from vllm.config import (
diff -- vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py
@@ -5,39 +5,37 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/olmo_hybrid.py` modified +6/-645; `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` added +634/-0; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` renamed +26/-45; `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` renamed +19/-52; `vllm/model_executor/layers/mamba/gdn/base.py` added +58/-0; `vllm/model_executor/models/kimi_linear.py` modified +13/-27
- 验证与风险: runtime 路径改动集中在 `vllm/config/compilation.py`, `vllm/model_executor/layers/mamba/gdn/__init__.py`, `vllm/model_executor/layers/mamba/gdn/base.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42124 - Add LM head quantization support for ModelOpt

- 链接: https://github.com/vllm-project/vllm/pull/42124
- 状态/时间: merged / 2026-05-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+220/-5，可读 patch 315 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add LM head quantization support for ModelOpt」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `tests/model_executor/test_qwen3_5_quantization.py`, `tests/model_executor/test_nemotron_h_quantization.py`, `vllm/model_executor/layers/quantization/modelopt.py`；技术摘要: 覆盖「Add LM head quantization support for ModelOpt」；主要实现面是 `tests/model_executor/test_qwen3_5_quantization.py`, `tests/model_executor/test_nemotron_h_quantization.py`, `vllm/model_executor/layers/quantization/modelopt.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/model_executor/test_qwen3_5_quantization.py` added +78/-0 (78 lines); hunks: -0,0 +1,78; symbols: test_qwen3_5_lm_head_receives_quant_config, test_qwen3_5_mtp_lm_head_receives_quant_config，涉及 `test_qwen3_5_lm_head_receives_quant_config, test_qwen3_5_mtp_lm_head_receives_quant_config`；`tests/model_executor/test_nemotron_h_quantization.py` added +34/-0 (34 lines); hunks: -0,0 +1,34; symbols: test_nemotron_h_lm_head_receives_quant_config，涉及 `test_nemotron_h_lm_head_receives_quant_config`；`vllm/model_executor/layers/quantization/modelopt.py` modified +5/-4 (9 lines); hunks: -85,6 +85,7; -187,7 +188,7 @@ def get_quant_method(; symbols: get_quant_method，涉及 `get_quant_method`；`vllm/model_executor/layers/vocab_parallel_embedding.py` modified +7/-0 (7 lines); hunks: -290,6 +290,7 @@ def __init__(; -438,6 +439,12 @@ def weight_loader(self, param: Parameter, loaded_weight: to...; symbols: __init__, weight_loader，涉及 `__init__, weight_loader`。
- 代码 diff 细节:
  - `tests/model_executor/test_qwen3_5_quantization.py` added +78/-0 (78 lines); hunks: -0,0 +1,78; symbols: test_qwen3_5_lm_head_receives_quant_config, test_qwen3_5_mtp_lm_head_receives_quant_config
  - `tests/model_executor/test_nemotron_h_quantization.py` added +34/-0 (34 lines); hunks: -0,0 +1,34; symbols: test_nemotron_h_lm_head_receives_quant_config
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +5/-4 (9 lines); hunks: -85,6 +85,7; -187,7 +188,7 @@ def get_quant_method(; symbols: get_quant_method
  - `vllm/model_executor/layers/vocab_parallel_embedding.py` modified +7/-0 (7 lines); hunks: -290,6 +290,7 @@ def __init__(; -438,6 +439,12 @@ def weight_loader(self, param: Parameter, loaded_weight: to...; symbols: __init__, weight_loader
  - `vllm/model_executor/models/nemotron_h.py` modified +1/-0 (1 lines); hunks: -875,6 +875,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- 关键代码摘录:

```diff
diff -- tests/model_executor/test_qwen3_5_quantization.py
@@ -0,0 +1,78 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from unittest.mock import Mock, patch
+def test_qwen3_5_lm_head_receives_quant_config():
+    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase
+    mock_quant_config = Mock()
diff -- tests/model_executor/test_nemotron_h_quantization.py
@@ -0,0 +1,34 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from unittest.mock import Mock, patch
+def test_nemotron_h_lm_head_receives_quant_config():
+    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
+    mock_quant_config = Mock()
diff -- vllm/model_executor/layers/quantization/modelopt.py
@@ -85,6 +85,7 @@
```

- 已读文件:
  - tests: `tests/model_executor/test_qwen3_5_quantization.py` added +78/-0; `tests/model_executor/test_nemotron_h_quantization.py` added +34/-0; `tests/quantization/test_modelopt.py` modified +93/-1
  - runtime: `vllm/model_executor/layers/quantization/modelopt.py` modified +5/-4; `vllm/model_executor/layers/vocab_parallel_embedding.py` modified +7/-0; `vllm/model_executor/models/nemotron_h.py` modified +1/-0; `vllm/model_executor/models/qwen3_5.py` modified +1/-0; `vllm/model_executor/models/qwen3_5_mtp.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/model_executor/test_nemotron_h_quantization.py`, `tests/model_executor/test_qwen3_5_quantization.py`, `tests/quantization/test_modelopt.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39419 - [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding

- 链接: https://github.com/vllm-project/vllm/pull/39419
- 状态/时间: merged / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+53/-39，可读 patch 169 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding」；模型线: Qwen3.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py`；技术摘要: 覆盖「[SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding」；主要实现面是 `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/interfaces.py` modified +35/-0 (35 lines); hunks: -1282,6 +1282,41 @@ def supports_any_eagle(; symbols: supports_any_eagle, LocalArgmaxMixin, get_top_tokens, EagleModelMixin，涉及 `supports_any_eagle, LocalArgmaxMixin, get_top_tokens`；`vllm/model_executor/models/llama4_eagle.py` modified +0/-17 (17 lines); hunks: -208,23 +208,6 @@ def forward(; symbols: forward, get_top_tokens, load_weights, transform，涉及 `forward, get_top_tokens, load_weights`；`vllm/model_executor/models/qwen3.py` modified +8/-2 (10 lines); hunks: -48,7 +48,13; -259,7 +265,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, Qwen3ForCausalLM，涉及 `__init__, Qwen3ForCausalLM`；`vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1 (3 lines); hunks: -31,6 +31,7; -309,7 +310,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, Eagle3DeepseekV2ForCausalLM, __init__，涉及 `load_weights, Eagle3DeepseekV2ForCausalLM, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/interfaces.py` modified +35/-0 (35 lines); hunks: -1282,6 +1282,41 @@ def supports_any_eagle(; symbols: supports_any_eagle, LocalArgmaxMixin, get_top_tokens, EagleModelMixin
  - `vllm/model_executor/models/llama4_eagle.py` modified +0/-17 (17 lines); hunks: -208,23 +208,6 @@ def forward(; symbols: forward, get_top_tokens, load_weights, transform
  - `vllm/model_executor/models/qwen3.py` modified +8/-2 (10 lines); hunks: -48,7 +48,13; -259,7 +265,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, Qwen3ForCausalLM
  - `vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1 (3 lines); hunks: -31,6 +31,7; -309,7 +310,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, Eagle3DeepseekV2ForCausalLM, __init__
  - `vllm/model_executor/models/llama.py` modified +2/-1 (3 lines); hunks: -62,6 +62,7; -487,7 +488,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, LlamaForCausalLM
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/interfaces.py
@@ -1282,6 +1282,41 @@ def supports_any_eagle(
+class LocalArgmaxMixin:
+    """Mixin for draft model heads in speculative decoding.
+    Provides a D2T-aware ``get_top_tokens`` that preserves the
+    local-argmax communication reduction even when the draft vocabulary
+    is smaller than the target vocabulary.
+    When ``draft_id_to_target_id`` is present (shape ``(draft_vocab_size,)``,
diff -- vllm/model_executor/models/llama4_eagle.py
@@ -208,23 +208,6 @@ def forward(
-    def get_top_tokens(
-        self,
-        hidden_states: torch.Tensor,
-    ) -> torch.Tensor:
-        """Vocab-parallel argmax without all-gathering full logits.
-        Falls back to full logits when draft_id_to_target_id remapping is
diff -- vllm/model_executor/models/qwen3.py
@@ -48,7 +48,13 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/interfaces.py` modified +35/-0; `vllm/model_executor/models/llama4_eagle.py` modified +0/-17; `vllm/model_executor/models/qwen3.py` modified +8/-2; `vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1; `vllm/model_executor/models/llama.py` modified +2/-1; `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_eagle3.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #45161 - Deprecate Transformers v4 support

- 链接: https://github.com/vllm-project/vllm/pull/45161
- 状态/时间: merged / 2026-06-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+62/-268，可读 patch 612 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Deprecate Transformers v4 support」；模型线: Qwen3.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/transformers/base.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/model_loader/weight_utils.py`；技术摘要: 覆盖「Deprecate Transformers v4 support」；主要实现面是 `vllm/model_executor/models/transformers/base.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/model_loader/weight_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/transformers/base.py` modified +16/-42 (58 lines); hunks: -27,6 +27,10; -212,16 +216,9 @@ def _patch_config(self):; symbols: _patch_config, _get_decoder_cls, _create_hf_to_vllm_mapper, _get_tie_word_embeddings，涉及 `_patch_config, _get_decoder_cls, _create_hf_to_vllm_mapper`；`vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-36 (36 lines); hunks: -30,9 +30,7; -1261,40 +1259,6 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) ->...; symbols: pad_to_hop_length，涉及 `pad_to_hop_length`；`vllm/model_executor/model_loader/weight_utils.py` modified +1/-18 (19 lines); hunks: -77,30 +77,13; symbols: enable_hf_transfer, enable_xet_high_performance, DisabledTqdm，涉及 `enable_hf_transfer, enable_xet_high_performance, DisabledTqdm`；`vllm/transformers_utils/configs/qwen3_5.py` modified +5/-12 (17 lines); hunks: -94,18 +94,11 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/transformers/base.py` modified +16/-42 (58 lines); hunks: -27,6 +27,10; -212,16 +216,9 @@ def _patch_config(self):; symbols: _patch_config, _get_decoder_cls, _create_hf_to_vllm_mapper, _get_tie_word_embeddings
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-36 (36 lines); hunks: -30,9 +30,7; -1261,40 +1259,6 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) ->...; symbols: pad_to_hop_length
  - `vllm/model_executor/model_loader/weight_utils.py` modified +1/-18 (19 lines); hunks: -77,30 +77,13; symbols: enable_hf_transfer, enable_xet_high_performance, DisabledTqdm
  - `vllm/transformers_utils/configs/qwen3_5.py` modified +5/-12 (17 lines); hunks: -94,18 +94,11 @@ def __init__(; symbols: __init__
  - `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +5/-12 (17 lines); hunks: -100,18 +100,11 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/transformers/base.py
@@ -27,6 +27,10 @@
+from transformers.conversion_mapping import (
+    WeightRenaming,
+    get_model_conversion_mapping,
+)
@@ -212,16 +216,9 @@ def _patch_config(self):
-        - Propagates this dtype to any sub-configs because Transformers model
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -30,9 +30,7 @@
-from packaging.version import Version
-from transformers import __version__ as TRANSFORMERS_VERSION
@@ -1261,40 +1259,6 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np.ndarray:
-            if Version(TRANSFORMERS_VERSION) < Version("4.58.0"):
-                # Extract audio_sample_rate before restructuring
-                audio_sample_rate = mm_kwargs.pop("audio_sample_rate", None)
diff -- vllm/model_executor/model_loader/weight_utils.py
@@ -77,30 +77,13 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/transformers/base.py` modified +16/-42; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-36; `vllm/model_executor/model_loader/weight_utils.py` modified +1/-18; `vllm/transformers_utils/configs/qwen3_5.py` modified +5/-12; `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +5/-12; `vllm/model_executor/models/ultravox.py` modified +0/-15
- 验证与风险: runtime 路径改动集中在 `vllm/config/vllm.py`, `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/gemma3n_mm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
