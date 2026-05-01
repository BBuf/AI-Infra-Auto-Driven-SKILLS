# vllm Qwen3.5 模型 PR 优化历史

## 覆盖范围

- 重做日期: 2026-05-01
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `7075df79b3`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/pooling/score/colqwen3_5_rerank_online.py` | [#36887](https://github.com/vllm-project/vllm/pull/36887) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml` | [#38083](https://github.com/vllm-project/vllm/pull/38083) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml` | [#38083](https://github.com/vllm-project/vllm/pull/38083) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml` | [#38664](https://github.com/vllm-project/vllm/pull/38664) |
| `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` | [#38083](https://github.com/vllm-project/vllm/pull/38083), [#38632](https://github.com/vllm-project/vllm/pull/38632) |
| `tests/evals/gsm8k/configs/models-qwen35-blackwell.txt` | [#38083](https://github.com/vllm-project/vllm/pull/38083) |
| `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` | [#38155](https://github.com/vllm-project/vllm/pull/38155), [#38664](https://github.com/vllm-project/vllm/pull/38664) |
| `tests/lora/test_qwen35_densemodel_lora.py` | [#37816](https://github.com/vllm-project/vllm/pull/37816) |
| `tests/models/multimodal/pooling/test_colqwen3_5.py` | [#36887](https://github.com/vllm-project/vllm/pull/36887) |
| `vllm/model_executor/models/colqwen3_5.py` | [#36887](https://github.com/vllm-project/vllm/pull/36887) |
| `vllm/model_executor/models/qwen3_5.py` | [#34110](https://github.com/vllm-project/vllm/pull/34110), [#34198](https://github.com/vllm-project/vllm/pull/34198), [#34200](https://github.com/vllm-project/vllm/pull/34200), [#34313](https://github.com/vllm-project/vllm/pull/34313), [#34489](https://github.com/vllm-project/vllm/pull/34489), [#34492](https://github.com/vllm-project/vllm/pull/34492), [#34512](https://github.com/vllm-project/vllm/pull/34512), [#34683](https://github.com/vllm-project/vllm/pull/34683), [#34697](https://github.com/vllm-project/vllm/pull/34697), [#34719](https://github.com/vllm-project/vllm/pull/34719), [#34723](https://github.com/vllm-project/vllm/pull/34723), [#35617](https://github.com/vllm-project/vllm/pull/35617), ... (18 total) |
| `vllm/model_executor/models/qwen3_5_mtp.py` | [#34110](https://github.com/vllm-project/vllm/pull/34110), [#34512](https://github.com/vllm-project/vllm/pull/34512), [#35581](https://github.com/vllm-project/vllm/pull/35581), [#37114](https://github.com/vllm-project/vllm/pull/37114), [#38832](https://github.com/vllm-project/vllm/pull/38832) |
| `vllm/transformers_utils/configs/qwen3_5.py` | [#34512](https://github.com/vllm-project/vllm/pull/34512), [#34554](https://github.com/vllm-project/vllm/pull/34554), [#34604](https://github.com/vllm-project/vllm/pull/34604), [#34610](https://github.com/vllm-project/vllm/pull/34610) |
| `vllm/transformers_utils/configs/qwen3_5_moe.py` | [#34512](https://github.com/vllm-project/vllm/pull/34512), [#34554](https://github.com/vllm-project/vllm/pull/34554), [#34604](https://github.com/vllm-project/vllm/pull/34604), [#34610](https://github.com/vllm-project/vllm/pull/34610) |

## PR 覆盖总览

- git 追溯 PR 数: 29
- 原文档显式引用补充 PR 数: 6
- 当前文档总 PR 数: 34
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-01-07 | [#31104](https://github.com/vllm-project/vllm/pull/31104) | merged | [BugFix] LoRA: Support loading base_layer of experts | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/llama4.py` |
| 2026-01-12 | [#32199](https://github.com/vllm-project/vllm/pull/32199) | merged | [BUGFIX] Add missed remaping of the names of fp8 kv-scale | `vllm/model_executor/models/qwen3_next.py` |
| 2026-02-09 | [#34110](https://github.com/vllm-project/vllm/pull/34110) | merged | [MODEL] Adding Support for Qwen3.5 Models | `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py` |
| 2026-02-10 | [#34198](https://github.com/vllm-project/vllm/pull/34198) | merged | [Bugfix] Adopt `ChunkGatedDeltaRule` for Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-10 | [#34200](https://github.com/vllm-project/vllm/pull/34200) | merged | [Bugfix] Fix mamba cache dtype for Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-11 | [#34313](https://github.com/vllm-project/vllm/pull/34313) | merged | [Bugfix] Fix weight naming in Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-13 | [#34489](https://github.com/vllm-project/vllm/pull/34489) | merged | [Bugfix] Fix mamba state dtype setting for Qwen3-Next and Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-13 | [#34512](https://github.com/vllm-project/vllm/pull/34512) | merged | [Misc] Port Qwen3.5 Configs | `vllm/transformers_utils/configs/qwen3_5_moe.py`, `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-14 | [#34554](https://github.com/vllm-project/vllm/pull/34554) | merged | [Bugfix] Fix Qwen3.5 config loading | `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py` |
| 2026-02-16 | [#34604](https://github.com/vllm-project/vllm/pull/34604) | merged | [Misc] fix qwen3.5 config | `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py` |
| 2026-02-16 | [#34610](https://github.com/vllm-project/vllm/pull/34610) | merged | Revert "[Misc] fix qwen3.5 config" | `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py` |
| 2026-02-16 | [#34492](https://github.com/vllm-project/vllm/pull/34492) | merged | [Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-17 | [#34683](https://github.com/vllm-project/vllm/pull/34683) | merged | Revert "[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj" | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-18 | [#34723](https://github.com/vllm-project/vllm/pull/34723) | merged | [Bugfix] Fix prefix creation for Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-18 | [#34697](https://github.com/vllm-project/vllm/pull/34697) | merged | [Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-19 | [#34719](https://github.com/vllm-project/vllm/pull/34719) | merged | [Bugfix] Qwen3.5 kv-scale weight remapping | `vllm/model_executor/models/qwen3_5.py` |
| 2026-02-26 | [#35289](https://github.com/vllm-project/vllm/pull/35289) | merged | [Bugfix] [Qwen3.5]Fix Qwen3.5 FP8 quantization: tuple shard_id weight loading | `vllm/model_executor/layers/linear.py` |
| 2026-02-28 | [#35581](https://github.com/vllm-project/vllm/pull/35581) | merged | Fix Qwen3_5MTP packed_modules_mapping for gate_up_proj | `vllm/model_executor/models/qwen3_5_mtp.py` |
| 2026-03-01 | [#35617](https://github.com/vllm-project/vllm/pull/35617) | merged | [Bugfix][Model] Fix Qwen3.5/Qwen3Next ignoring --dtype flag on older GPUs | `vllm/model_executor/models/qwen3_5.py` |
| 2026-03-11 | [#36658](https://github.com/vllm-project/vllm/pull/36658) | merged | Add: Eagle3 support for Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-03-17 | [#36887](https://github.com/vllm-project/vllm/pull/36887) | merged | [Model] Add ColQwen3.5 4.5B support | `vllm/model_executor/models/colqwen3_5.py`, `tests/models/multimodal/pooling/test_colqwen3_5.py`, `examples/pooling/score/colqwen3_5_rerank_online.py` |
| 2026-03-19 | [#37448](https://github.com/vllm-project/vllm/pull/37448) | merged | Fix AttributeError in Qwen3.5 GDN layers with quantized models | `vllm/model_executor/models/qwen3_5.py` |
| 2026-03-20 | [#36976](https://github.com/vllm-project/vllm/pull/36976) | merged | [Bugfix][LoRA] Fix Qwen35 LoRA | `vllm/model_executor/models/qwen3_5.py` |
| 2026-03-23 | [#37816](https://github.com/vllm-project/vllm/pull/37816) | merged | [CI/Build][LoRA] Update Qwen35 LoRA testing | `tests/lora/test_qwen35_densemodel_lora.py` |
| 2026-03-26 | [#38083](https://github.com/vllm-project/vllm/pull/38083) | merged | [Bugfix] Fix DeepGemm E8M0 accuracy degradation for Qwen3.5 FP8 on Blackwell | `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml` |
| 2026-03-26 | [#38155](https://github.com/vllm-project/vllm/pull/38155) | merged | [ROCm][CI] Add LM Eval Qwen3.5 Models test for MI355 | `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` |
| 2026-03-27 | [#37975](https://github.com/vllm-project/vllm/pull/37975) | merged | [Model] Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-03-31 | [#38632](https://github.com/vllm-project/vllm/pull/38632) | merged | [CI] fix LM Eval Qwen3.5 Models (B200) | `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` |
| 2026-04-02 | [#38650](https://github.com/vllm-project/vllm/pull/38650) | closed | [Bugfix] Enable MTP for the official Qwen3.5 NVFP4 checkpoint | `vllm/model_executor/models/qwen3_5_mtp.py` |
| 2026-04-03 | [#38832](https://github.com/vllm-project/vllm/pull/38832) | merged | [Bugfix] Fix NVFP4+MTP crash: force unquantized mtp.fc for Qwen3.5 | `vllm/model_executor/models/qwen3_5_mtp.py` |
| 2026-04-03 | [#38664](https://github.com/vllm-project/vllm/pull/38664) | merged | [CI][ROCm] Add Qwen3.5-35B-A3B-MXFP4 model eval into CI | `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml`, `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` |
| 2026-04-03 | [#38927](https://github.com/vllm-project/vllm/pull/38927) | merged | [Bugfix][LoRA] Fix missing in_proj_z in Qwen3_5ForConditionalGenerati… | `vllm/model_executor/models/qwen3_5.py` |
| 2026-04-08 | [#39181](https://github.com/vllm-project/vllm/pull/39181) | merged | [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next | `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py` |
| 2026-04-21 | [#37114](https://github.com/vllm-project/vllm/pull/37114) | merged | [Bugfix] LoRA: extend expert base_layer loading to Qwen3.5 and Step3.x | `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py` |

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

### PR #32199 - [BUGFIX] Add missed remaping of the names of fp8 kv-scale

- 链接: https://github.com/vllm-project/vllm/pull/32199
- 状态/时间: merged / 2026-01-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-0，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BUGFIX] Add missed remaping of the names of fp8 kv-scale」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_next.py`；技术摘要: 覆盖「[BUGFIX] Add missed remaping of the names of fp8 kv-scale」；主要实现面是 `vllm/model_executor/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_next.py` modified +7/-0 (7 lines); hunks: -64,6 +64,7; -1065,6 +1066,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_next.py` modified +7/-0 (7 lines); hunks: -64,6 +64,7; -1065,6 +1066,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_next.py
@@ -64,6 +64,7 @@
+    maybe_remap_kv_scale_name,
@@ -1065,6 +1066,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            # Remapping the name of FP8 kv-scale.
+            if name.endswith("scale"):
+                name = maybe_remap_kv_scale_name(name, params_dict)
+                if name is None:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_next.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34110 - [MODEL] Adding Support for Qwen3.5 Models

- 链接: https://github.com/vllm-project/vllm/pull/34110
- 状态/时间: merged / 2026-02-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`；关联提交 `9562912cead1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+1501/-9，可读 patch 1631 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MODEL] Adding Support for Qwen3.5 Models」；模型线: Qwen3.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`；技术摘要: 覆盖「[MODEL] Adding Support for Qwen3.5 Models」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` added +993/-0 (993 lines); hunks: -0,0 +1,993; symbols: Qwen3_5ProcessingInfo, get_hf_config, Qwen3_5MoeProcessingInfo, Qwen3_5GatedDeltaNet，涉及 `Qwen3_5ProcessingInfo, get_hf_config, Qwen3_5MoeProcessingInfo`；`vllm/model_executor/models/qwen3_5_mtp.py` added +447/-0 (447 lines); hunks: -0,0 +1,447; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward，涉及 `Qwen3_5MultiTokenPredictor, __init__, embed_input_ids`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` added +993/-0 (993 lines); hunks: -0,0 +1,993; symbols: Qwen3_5ProcessingInfo, get_hf_config, Qwen3_5MoeProcessingInfo, Qwen3_5GatedDeltaNet
  - `vllm/model_executor/models/qwen3_5_mtp.py` added +447/-0 (447 lines); hunks: -0,0 +1,447; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -0,0 +1,993 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The vLLM team.
+# Copyright 2025 The Qwen Team.
+# Copyright 2025 The HuggingFace Inc. team.
+# All rights reserved.
diff -- vllm/model_executor/models/qwen3_5_mtp.py
@@ -0,0 +1,447 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only Qwen3_5 MTP model."""
+import typing
+from collections.abc import Callable, Iterable
+import torch
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` added +993/-0; `vllm/model_executor/models/qwen3_5_mtp.py` added +447/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34198 - [Bugfix] Adopt `ChunkGatedDeltaRule` for Qwen3.5

- 链接: https://github.com/vllm-project/vllm/pull/34198
- 状态/时间: merged / 2026-02-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `047a457fa4af`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Adopt `ChunkGatedDeltaRule` for Qwen3.5」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「[Bugfix] Adopt `ChunkGatedDeltaRule` for Qwen3.5」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +3/-0 (3 lines); hunks: -99,6 +99,7; -268,6 +269,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +3/-0 (3 lines); hunks: -99,6 +99,7; -268,6 +269,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -99,6 +99,7 @@
+    ChunkGatedDeltaRule,
@@ -268,6 +269,8 @@ def __init__(
+        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34200 - [Bugfix] Fix mamba cache dtype for Qwen3.5

- 链接: https://github.com/vllm-project/vllm/pull/34200
- 状态/时间: merged / 2026-02-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `9615575afc0d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-1，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix mamba cache dtype for Qwen3.5」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「[Bugfix] Fix mamba cache dtype for Qwen3.5」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +2/-1 (3 lines); hunks: -867,8 +867,9 @@ def get_mamba_state_dtype_from_config(; symbols: get_mamba_state_dtype_from_config，涉及 `get_mamba_state_dtype_from_config`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +2/-1 (3 lines); hunks: -867,8 +867,9 @@ def get_mamba_state_dtype_from_config(; symbols: get_mamba_state_dtype_from_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -867,8 +867,9 @@ def get_mamba_state_dtype_from_config(
+        mamba_ssm_dtype = vllm_config.model_config.hf_text_config.mamba_ssm_dtype
-            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
+            vllm_config.model_config.dtype, mamba_ssm_dtype
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34313 - [Bugfix] Fix weight naming in Qwen3.5

- 链接: https://github.com/vllm-project/vllm/pull/34313
- 状态/时间: merged / 2026-02-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `0b20469c627e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix weight naming in Qwen3.5」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +1/-1 (2 lines); hunks: -206,7 +206,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +1/-1 (2 lines); hunks: -206,7 +206,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -206,7 +206,7 @@ def __init__(
-            prefix=f"{prefix}.in_proj_ba",
+            prefix=f"{prefix}.in_proj_b",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34489 - [Bugfix] Fix mamba state dtype setting for Qwen3-Next and Qwen3.5

- 链接: https://github.com/vllm-project/vllm/pull/34489
- 状态/时间: merged / 2026-02-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `eea3024f43e0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+42/-6，可读 patch 91 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix mamba state dtype setting for Qwen3-Next and Qwen3.5」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「[Bugfix] Fix mamba state dtype setting for Qwen3-Next and Qwen3.5」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +3/-2 (5 lines); hunks: -870,9 +870,10 @@ def get_mamba_state_dtype_from_config(; symbols: get_mamba_state_dtype_from_config，涉及 `get_mamba_state_dtype_from_config`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +3/-2 (5 lines); hunks: -870,9 +870,10 @@ def get_mamba_state_dtype_from_config(; symbols: get_mamba_state_dtype_from_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -870,9 +870,10 @@ def get_mamba_state_dtype_from_config(
-        mamba_ssm_dtype = vllm_config.model_config.hf_text_config.mamba_ssm_dtype
-            vllm_config.model_config.dtype, mamba_ssm_dtype
+            vllm_config.model_config.dtype,
+            vllm_config.cache_config.mamba_cache_dtype,
+            vllm_config.cache_config.mamba_ssm_cache_dtype,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +3/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/models/config.py`, `vllm/model_executor/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34512 - [Misc] Port Qwen3.5 Configs

- 链接: https://github.com/vllm-project/vllm/pull/34512
- 状态/时间: merged / 2026-02-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`, `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`；关联提交 `5885e330efea`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+410/-12，可读 patch 473 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Port Qwen3.5 Configs」；模型线: Qwen3.5；类别: 模型实现调整；主要 diff: `vllm/transformers_utils/configs/qwen3_5_moe.py`, `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「[Misc] Port Qwen3.5 Configs」；主要实现面是 `vllm/transformers_utils/configs/qwen3_5_moe.py`, `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/qwen3_5_moe.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: Qwen3_5MoeTextConfig, __init__, Qwen3_5MoeVisionConfig, Qwen3_5MoeConfig，涉及 `Qwen3_5MoeTextConfig, __init__, Qwen3_5MoeVisionConfig`；`vllm/transformers_utils/configs/qwen3_5.py` added +189/-0 (189 lines); hunks: -0,0 +1,189; symbols: Qwen3_5TextConfig, __init__, Qwen3_5VisionConfig, Qwen3_5Config，涉及 `Qwen3_5TextConfig, __init__, Qwen3_5VisionConfig`；`vllm/model_executor/models/qwen3_5.py` modified +8/-8 (16 lines); hunks: -31,14 +31,6; -87,6 +79,14；`vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-4 (6 lines); hunks: -7,10 +7,6; -27,6 +23,8。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/qwen3_5_moe.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: Qwen3_5MoeTextConfig, __init__, Qwen3_5MoeVisionConfig, Qwen3_5MoeConfig
  - `vllm/transformers_utils/configs/qwen3_5.py` added +189/-0 (189 lines); hunks: -0,0 +1,189; symbols: Qwen3_5TextConfig, __init__, Qwen3_5VisionConfig, Qwen3_5Config
  - `vllm/model_executor/models/qwen3_5.py` modified +8/-8 (16 lines); hunks: -31,14 +31,6; -87,6 +79,14
  - `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-4 (6 lines); hunks: -7,10 +7,6; -27,6 +23,8
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/configs/qwen3_5_moe.py
@@ -0,0 +1,201 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
+# All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
diff -- vllm/transformers_utils/configs/qwen3_5.py
@@ -0,0 +1,189 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
+# All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
diff -- vllm/model_executor/models/qwen3_5.py
@@ -31,14 +31,6 @@
```

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/qwen3_5_moe.py` added +201/-0; `vllm/transformers_utils/configs/qwen3_5.py` added +189/-0; `vllm/model_executor/models/qwen3_5.py` modified +8/-8; `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`, `vllm/transformers_utils/config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34554 - [Bugfix] Fix Qwen3.5 config loading

- 链接: https://github.com/vllm-project/vllm/pull/34554
- 状态/时间: merged / 2026-02-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`；关联提交 `2f186635cbcb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+18/-10，可读 patch 72 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen3.5 config loading」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen3.5 config loading」；主要实现面是 `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/qwen3_5.py` modified +9/-5 (14 lines); hunks: -72,10 +72,6 @@ def __init__(; -111,6 +107,13 @@ def __init__(; symbols: __init__, Qwen3_5VisionConfig，涉及 `__init__, Qwen3_5VisionConfig`；`vllm/transformers_utils/configs/qwen3_5_moe.py` modified +9/-5 (14 lines); hunks: -79,10 +79,6 @@ def __init__(; -123,6 +119,13 @@ def __init__(; symbols: __init__, Qwen3_5MoeVisionConfig，涉及 `__init__, Qwen3_5MoeVisionConfig`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/qwen3_5.py` modified +9/-5 (14 lines); hunks: -72,10 +72,6 @@ def __init__(; -111,6 +107,13 @@ def __init__(; symbols: __init__, Qwen3_5VisionConfig
  - `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +9/-5 (14 lines); hunks: -79,10 +79,6 @@ def __init__(; -123,6 +119,13 @@ def __init__(; symbols: __init__, Qwen3_5MoeVisionConfig
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/configs/qwen3_5.py
@@ -72,10 +72,6 @@ def __init__(
-        self.pad_token_id = pad_token_id
-        self.bos_token_id = bos_token_id
-        self.eos_token_id = eos_token_id
-        self.tie_word_embeddings = tie_word_embeddings
@@ -111,6 +107,13 @@ def __init__(
+        # Set these AFTER super().__init__() because transformers v4's
diff -- vllm/transformers_utils/configs/qwen3_5_moe.py
@@ -79,10 +79,6 @@ def __init__(
-        self.pad_token_id = pad_token_id
-        self.bos_token_id = bos_token_id
-        self.eos_token_id = eos_token_id
-        self.tie_word_embeddings = tie_word_embeddings
@@ -123,6 +119,13 @@ def __init__(
+        # Set these AFTER super().__init__() because transformers v4's
```

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/qwen3_5.py` modified +9/-5; `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +9/-5
- 验证与风险: runtime 路径改动集中在 `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34604 - [Misc] fix qwen3.5 config

- 链接: https://github.com/vllm-project/vllm/pull/34604
- 状态/时间: merged / 2026-02-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`；关联提交 `9521002f0ace`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-4，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] fix qwen3.5 config」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`；技术摘要: 覆盖「[Misc] fix qwen3.5 config」；主要实现面是 `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/qwen3_5.py` modified +2/-2 (4 lines); hunks: -68,10 +68,10 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/transformers_utils/configs/qwen3_5_moe.py` modified +2/-2 (4 lines); hunks: -75,10 +75,10 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/qwen3_5.py` modified +2/-2 (4 lines); hunks: -68,10 +68,10 @@ def __init__(; symbols: __init__
  - `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +2/-2 (4 lines); hunks: -75,10 +75,10 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/configs/qwen3_5.py
@@ -68,10 +68,10 @@ def __init__(
-        kwargs["ignore_keys_at_rope_validation"] = [
+        kwargs["ignore_keys_at_rope_validation"] = {
-        ]
+        }
diff -- vllm/transformers_utils/configs/qwen3_5_moe.py
@@ -75,10 +75,10 @@ def __init__(
-        kwargs["ignore_keys_at_rope_validation"] = [
+        kwargs["ignore_keys_at_rope_validation"] = {
-        ]
+        }
```

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/qwen3_5.py` modified +2/-2; `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34610 - Revert "[Misc] fix qwen3.5 config"

- 链接: https://github.com/vllm-project/vllm/pull/34610
- 状态/时间: merged / 2026-02-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`；关联提交 `b5475d053442`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-4，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "[Misc] fix qwen3.5 config"」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`；技术摘要: 覆盖「Revert "[Misc] fix qwen3.5 config"」；主要实现面是 `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/qwen3_5.py` modified +2/-2 (4 lines); hunks: -68,10 +68,10 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/transformers_utils/configs/qwen3_5_moe.py` modified +2/-2 (4 lines); hunks: -75,10 +75,10 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/qwen3_5.py` modified +2/-2 (4 lines); hunks: -68,10 +68,10 @@ def __init__(; symbols: __init__
  - `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +2/-2 (4 lines); hunks: -75,10 +75,10 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/configs/qwen3_5.py
@@ -68,10 +68,10 @@ def __init__(
-        kwargs["ignore_keys_at_rope_validation"] = {
+        kwargs["ignore_keys_at_rope_validation"] = [
-        }
+        ]
diff -- vllm/transformers_utils/configs/qwen3_5_moe.py
@@ -75,10 +75,10 @@ def __init__(
-        kwargs["ignore_keys_at_rope_validation"] = {
+        kwargs["ignore_keys_at_rope_validation"] = [
-        }
+        ]
```

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/qwen3_5.py` modified +2/-2; `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/transformers_utils/configs/qwen3_5.py`, `vllm/transformers_utils/configs/qwen3_5_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34492 - [Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj

- 链接: https://github.com/vllm-project/vllm/pull/34492
- 状态/时间: merged / 2026-02-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `3bb4e4311c6d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+87/-182，可读 patch 404 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +32/-166 (198 lines); hunks: -30,36 +30,20; -73,11 +57,8; symbols: get_hf_config, Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering，涉及 `get_hf_config, Qwen3_5GatedDeltaNet, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +32/-166 (198 lines); hunks: -30,36 +30,20; -73,11 +57,8; symbols: get_hf_config, Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -30,36 +30,20 @@
-from transformers.activations import ACT2FN
-    CacheConfig,
-    ModelConfig,
-    SpeculativeConfig,
-    get_current_vllm_config,
-    divide,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +32/-166
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34683 - Revert "[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj"

- 链接: https://github.com/vllm-project/vllm/pull/34683
- 状态/时间: merged / 2026-02-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `1d65283e95f4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+182/-87，可读 patch 402 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj"」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「Revert "[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj"」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +166/-32 (198 lines); hunks: -30,20 +30,36; -57,8 +73,11; symbols: get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering, __init__，涉及 `get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +166/-32 (198 lines); hunks: -30,20 +30,36; -57,8 +73,11; symbols: get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -30,20 +30,36 @@
+from transformers.activations import ACT2FN
+    CacheConfig,
+    ModelConfig,
+    SpeculativeConfig,
+    get_current_vllm_config,
+    divide,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +166/-32
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34723 - [Bugfix] Fix prefix creation for Qwen3.5

- 链接: https://github.com/vllm-project/vllm/pull/34723
- 状态/时间: merged / 2026-02-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `909b14719725`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-5，可读 patch 32 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix prefix creation for Qwen3.5」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「[Bugfix] Fix prefix creation for Qwen3.5」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +6/-5 (11 lines); hunks: -542,9 +542,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -620,7 +621,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: __init__, get_expert_mapping, Qwen3_5ForConditionalGeneration，涉及 `__init__, get_expert_mapping, Qwen3_5ForConditionalGeneration`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +6/-5 (11 lines); hunks: -542,9 +542,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -620,7 +621,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: __init__, get_expert_mapping, Qwen3_5ForConditionalGeneration
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -542,9 +542,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        self.model = Qwen3_5Model(
-            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
-        )
+        # Deal with the case where the prefix is already "language_model" since
+        # Qwen/Qwen3.5-397B-A17B has naming like: model.language_model.layers.0
+        model_prefix = prefix if "model" in prefix else "model"
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +6/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34697 - [Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion

- 链接: https://github.com/vllm-project/vllm/pull/34697
- 状态/时间: merged / 2026-02-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `c0bd8b13da36`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+102/-192，可读 patch 477 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「[Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +43/-170 (213 lines); hunks: -30,36 +30,20; -73,11 +57,8; symbols: get_hf_config, Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering，涉及 `get_hf_config, Qwen3_5GatedDeltaNet, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +43/-170 (213 lines); hunks: -30,36 +30,20; -73,11 +57,8; symbols: get_hf_config, Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -30,36 +30,20 @@
-from transformers.activations import ACT2FN
-    CacheConfig,
-    ModelConfig,
-    SpeculativeConfig,
-    get_current_vllm_config,
-    divide,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +43/-170
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34719 - [Bugfix] Qwen3.5 kv-scale weight remapping

- 链接: https://github.com/vllm-project/vllm/pull/34719
- 状态/时间: merged / 2026-02-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `6fff24f30fe2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-0，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Qwen3.5 kv-scale weight remapping」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「[Bugfix] Qwen3.5 kv-scale weight remapping」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +7/-0 (7 lines); hunks: -57,6 +57,7; -397,6 +398,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +7/-0 (7 lines); hunks: -57,6 +57,7; -397,6 +398,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -57,6 +57,7 @@
+    maybe_remap_kv_scale_name,
@@ -397,6 +398,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            # Remapping the name of FP8 kv-scale.
+            if name.endswith("scale"):
+                name = maybe_remap_kv_scale_name(name, params_dict)
+                if name is None:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35289 - [Bugfix] [Qwen3.5]Fix Qwen3.5 FP8 quantization: tuple shard_id weight loading

- 链接: https://github.com/vllm-project/vllm/pull/35289
- 状态/时间: merged / 2026-02-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+19/-8，可读 patch 55 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] [Qwen3.5]Fix Qwen3.5 FP8 quantization: tuple shard_id weight loading」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/linear.py`；技术摘要: 覆盖「[Bugfix] [Qwen3.5]Fix Qwen3.5 FP8 quantization: tuple shard_id weight loading」；主要实现面是 `vllm/model_executor/layers/linear.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/linear.py` modified +19/-8 (27 lines); hunks: -731,16 +731,16 @@ def weight_loader(; -768,7 +768,7 @@ def weight_loader(; symbols: weight_loader，涉及 `weight_loader`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/linear.py` modified +19/-8 (27 lines); hunks: -731,16 +731,16 @@ def weight_loader(; -768,7 +768,7 @@ def weight_loader(; symbols: weight_loader
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/linear.py
@@ -731,16 +731,16 @@ def weight_loader(
-        # FIXME(Isotr0py): Enable tuple shard_id for BNB quantization.
-        if isinstance(loaded_shard_id, tuple):
-            raise NotImplementedError(
-                "Shard id with multiple indices is not supported in weight_loader, "
-                "please use weight_loader_v2 instead."
-            )
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/linear.py` modified +19/-8
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35581 - Fix Qwen3_5MTP packed_modules_mapping for gate_up_proj

- 链接: https://github.com/vllm-project/vllm/pull/35581
- 状态/时间: merged / 2026-02-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5_mtp.py`；关联提交 `63d7972f13d1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Qwen3_5MTP packed_modules_mapping for gate_up_proj」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5_mtp.py`；技术摘要: 覆盖「Fix Qwen3_5MTP packed_modules_mapping for gate_up_proj」；主要实现面是 `vllm/model_executor/models/qwen3_5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5_mtp.py` modified +1/-1 (2 lines); hunks: -339,7 +339,7 @@ class Qwen3_5MTP(nn.Module, SupportsMultiModal):; symbols: Qwen3_5MTP, __init__，涉及 `Qwen3_5MTP, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5_mtp.py` modified +1/-1 (2 lines); hunks: -339,7 +339,7 @@ class Qwen3_5MTP(nn.Module, SupportsMultiModal):; symbols: Qwen3_5MTP, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5_mtp.py
@@ -339,7 +339,7 @@ class Qwen3_5MTP(nn.Module, SupportsMultiModal):
-        "gate_up_proj": ["up_proj", "down_proj"],
+        "gate_up_proj": ["gate_proj", "up_proj"],
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5_mtp.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35617 - [Bugfix][Model] Fix Qwen3.5/Qwen3Next ignoring --dtype flag on older GPUs

- 链接: https://github.com/vllm-project/vllm/pull/35617
- 状态/时间: merged / 2026-03-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `afd089f231d7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+0/-5，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Model] Fix Qwen3.5/Qwen3Next ignoring --dtype flag on older GPUs」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「[Bugfix][Model] Fix Qwen3.5/Qwen3Next ignoring --dtype flag on older GPUs」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +0/-2 (2 lines); hunks: -274,15 +274,13 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +0/-2 (2 lines); hunks: -274,15 +274,13 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -274,15 +274,13 @@ def __init__(
-                    dtype=config.dtype,
-                    dtype=config.dtype,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +0/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36658 - Add: Eagle3 support for Qwen3.5

- 链接: https://github.com/vllm-project/vllm/pull/36658
- 状态/时间: merged / 2026-03-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `9d07a3d6e472`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+25/-2，可读 patch 83 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add: Eagle3 support for Qwen3.5」；模型线: Qwen3.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「Add: Eagle3 support for Qwen3.5」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +11/-0 (11 lines); hunks: -75,6 +75,7; -353,6 +354,8 @@ def get_layer(prefix: str):; symbols: get_layer, load_fused_expert_weights, load_weights, Qwen3_5ForCausalLMBase，涉及 `get_layer, load_fused_expert_weights, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +11/-0 (11 lines); hunks: -75,6 +75,7; -353,6 +354,8 @@ def get_layer(prefix: str):; symbols: get_layer, load_fused_expert_weights, load_weights, Qwen3_5ForCausalLMBase
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -75,6 +75,7 @@
+    SupportsEagle3,
@@ -353,6 +354,8 @@ def get_layer(prefix: str):
+        self.aux_hidden_state_layers: tuple[int, ...] = ()
@@ -536,6 +539,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+    SupportsEagle3,
@@ -592,6 +596,13 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +11/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36887 - [Model] Add ColQwen3.5 4.5B support

- 链接: https://github.com/vllm-project/vllm/pull/36887
- 状态/时间: merged / 2026-03-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/pooling/score/colqwen3_5_rerank_online.py`, `tests/models/multimodal/pooling/test_colqwen3_5.py`, `vllm/model_executor/models/colqwen3_5.py`；关联提交 `c0745a851a4f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+579/-0，可读 patch 619 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add ColQwen3.5 4.5B support」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/colqwen3_5.py`, `tests/models/multimodal/pooling/test_colqwen3_5.py`, `examples/pooling/score/colqwen3_5_rerank_online.py`；技术摘要: 覆盖「[Model] Add ColQwen3.5 4.5B support」；主要实现面是 `vllm/model_executor/models/colqwen3_5.py`, `tests/models/multimodal/pooling/test_colqwen3_5.py`, `examples/pooling/score/colqwen3_5_rerank_online.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/colqwen3_5.py` added +246/-0 (246 lines); hunks: -0,0 +1,246; symbols: ColQwen3_5ProcessingInfo, get_hf_config, get_hf_processor, _supports_video，涉及 `ColQwen3_5ProcessingInfo, get_hf_config, get_hf_processor`；`tests/models/multimodal/pooling/test_colqwen3_5.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: _run_token_embed_test, _run_late_interaction_test, _run_relevance_test, test_colqwen3_5_token_embed，涉及 `_run_token_embed_test, _run_late_interaction_test, _run_relevance_test`；`examples/pooling/score/colqwen3_5_rerank_online.py` added +130/-0 (130 lines); hunks: -0,0 +1,130; symbols: rerank_text, score_text, score_text_top_n, main，涉及 `rerank_text, score_text, score_text_top_n`。
- 代码 diff 细节:
  - `vllm/model_executor/models/colqwen3_5.py` added +246/-0 (246 lines); hunks: -0,0 +1,246; symbols: ColQwen3_5ProcessingInfo, get_hf_config, get_hf_processor, _supports_video
  - `tests/models/multimodal/pooling/test_colqwen3_5.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: _run_token_embed_test, _run_late_interaction_test, _run_relevance_test, test_colqwen3_5_token_embed
  - `examples/pooling/score/colqwen3_5_rerank_online.py` added +130/-0 (130 lines); hunks: -0,0 +1,130; symbols: rerank_text, score_text, score_text_top_n, main
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/colqwen3_5.py
@@ -0,0 +1,246 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+ColQwen3.5 late interaction model for multi-modal retrieval and reranking.
+ColQwen3.5 extends Qwen3.5 with a ColBERT-style late interaction head,
+producing per-token embeddings for both text and image inputs. It uses
diff -- tests/models/multimodal/pooling/test_colqwen3_5.py
@@ -0,0 +1,154 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tests for ColQwen3.5 late interaction model for multi-modal retrieval.
+ColQwen3.5 is a multi-vector retrieval model based on Qwen3.5 backbone with
+ColBERT-style late interaction scoring (MaxSim). It produces per-token
+embeddings for both text and image inputs.
diff -- examples/pooling/score/colqwen3_5_rerank_online.py
@@ -0,0 +1,130 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/colqwen3_5.py` added +246/-0
  - tests: `tests/models/multimodal/pooling/test_colqwen3_5.py` added +154/-0
  - docs: `examples/pooling/score/colqwen3_5_rerank_online.py` added +130/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/pooling/test_colqwen3_5.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37448 - Fix AttributeError in Qwen3.5 GDN layers with quantized models

- 链接: https://github.com/vllm-project/vllm/pull/37448
- 状态/时间: merged / 2026-03-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `4120a05ff1d0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-4，可读 patch 22 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix AttributeError in Qwen3.5 GDN layers with quantized models」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「Fix AttributeError in Qwen3.5 GDN layers with quantized models」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +2/-2 (4 lines); hunks: -182,8 +182,8 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +2/-2 (4 lines); hunks: -182,8 +182,8 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -182,8 +182,8 @@ def forward(
-            self.in_proj_qkvz.weight.shape[0],
-            self.in_proj_ba.weight.shape[0],
+            sum(self.in_proj_qkvz.output_sizes) // self.tp_size,
+            sum(self.in_proj_ba.output_sizes) // self.tp_size,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36976 - [Bugfix][LoRA] Fix Qwen35 LoRA

- 链接: https://github.com/vllm-project/vllm/pull/36976
- 状态/时间: merged / 2026-03-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `8fbe3f303fbf`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+257/-46，可读 patch 464 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][LoRA] Fix Qwen35 LoRA」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +100/-23 (123 lines); hunks: -32,17 +32,18; -130,6 +131,40 @@ def fix_query_key_value_ordering(; symbols: fix_query_key_value_ordering, __init__, create_qkvz_proj, forward，涉及 `fix_query_key_value_ordering, __init__, create_qkvz_proj`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +100/-23 (123 lines); hunks: -32,17 +32,18; -130,6 +131,40 @@ def fix_query_key_value_ordering(; symbols: fix_query_key_value_ordering, __init__, create_qkvz_proj, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -32,17 +32,18 @@
-from vllm.config import (
-    VllmConfig,
-)
+from vllm.config import VllmConfig
-from vllm.model_executor.layers.linear import MergedColumnParallelLinear
+from vllm.model_executor.layers.linear import (
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +100/-23
- 验证与风险: diff 自带测试面 `tests/lora/conftest.py`, `tests/lora/test_qwen35_densemoel_lora.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37816 - [CI/Build][LoRA] Update Qwen35 LoRA testing

- 链接: https://github.com/vllm-project/vllm/pull/37816
- 状态/时间: merged / 2026-03-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/lora/test_qwen35_densemodel_lora.py`；关联提交 `1f0d21064137`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+369/-135，可读 patch 529 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI/Build][LoRA] Update Qwen35 LoRA testing」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `tests/lora/test_qwen35_densemodel_lora.py`；未提供可用技术摘要。
- 实现要点: `tests/lora/test_qwen35_densemodel_lora.py` added +361/-0 (361 lines); hunks: -0,0 +1,361; symbols: _assert_exact_outputs, _assert_prefix_outputs, _run_text_lora_sample, _run_vl_lora_sample，涉及 `_assert_exact_outputs, _assert_prefix_outputs, _run_text_lora_sample`。
- 代码 diff 细节:
  - `tests/lora/test_qwen35_densemodel_lora.py` added +361/-0 (361 lines); hunks: -0,0 +1,361; symbols: _assert_exact_outputs, _assert_prefix_outputs, _run_text_lora_sample, _run_vl_lora_sample
- 关键代码摘录:

```diff
diff -- tests/lora/test_qwen35_densemodel_lora.py
@@ -0,0 +1,361 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from transformers import AutoTokenizer
+import vllm
+import vllm.config
+from vllm.assets.image import ImageAsset
```

- 已读文件:
  - tests: `tests/lora/test_qwen35_densemodel_lora.py` added +361/-0
- 验证与风险: diff 自带测试面 `tests/lora/conftest.py`, `tests/lora/test_qwen35_densemodel_lora.py`, `tests/lora/test_qwen35_densemoel_lora.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38083 - [Bugfix] Fix DeepGemm E8M0 accuracy degradation for Qwen3.5 FP8 on Blackwell

- 链接: https://github.com/vllm-project/vllm/pull/38083
- 状态/时间: merged / 2026-03-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`, `tests/evals/gsm8k/configs/models-qwen35-blackwell.txt`；关联提交 `52069012fe53`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+69/-11，可读 patch 177 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix DeepGemm E8M0 accuracy degradation for Qwen3.5 FP8 on Blackwell」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml`；技术摘要: 覆盖「[Bugfix] Fix DeepGemm E8M0 accuracy degradation for Qwen3.5 FP8 on Blackwell」；主要实现面是 `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` added +9/-0 (9 lines); hunks: -0,0 +1,9；`tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml` modified +2/-1 (3 lines); hunks: -1,5 +1,6；`tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml` modified +2/-1 (3 lines); hunks: -1,5 +1,6；`tests/evals/gsm8k/configs/models-qwen35-blackwell.txt` modified +2/-0 (2 lines); hunks: -1 +1,3。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` added +9/-0 (9 lines); hunks: -0,0 +1,9
  - `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml` modified +2/-1 (3 lines); hunks: -1,5 +1,6
  - `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml` modified +2/-1 (3 lines); hunks: -1,5 +1,6
  - `tests/evals/gsm8k/configs/models-qwen35-blackwell.txt` modified +2/-0 (2 lines); hunks: -1 +1,3
- 关键代码摘录:

```diff
diff -- tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml
@@ -0,0 +1,9 @@
+model_name: "nvidia/Qwen3.5-397B-A17B-NVFP4"
+accuracy_threshold: 0.88
+tolerance: 0.03
+num_questions: 1319
+num_fewshot: 5
+server_args: >-
diff -- tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml
@@ -1,5 +1,6 @@
-accuracy_threshold: 0.86
+accuracy_threshold: 0.84
+tolerance: 0.03
diff -- tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml
@@ -1,5 +1,6 @@
-accuracy_threshold: 0.86
+accuracy_threshold: 0.79
+tolerance: 0.03
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` added +9/-0; `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml` modified +2/-1; `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml` modified +2/-1; `tests/evals/gsm8k/configs/models-qwen35-blackwell.txt` modified +2/-0
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`, `tests/evals/gsm8k/configs/models-qwen35-blackwell.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38155 - [ROCm][CI] Add LM Eval Qwen3.5 Models test for MI355

- 链接: https://github.com/vllm-project/vllm/pull/38155
- 状态/时间: merged / 2026-03-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`；关联提交 `9c3ae04bfe65`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+25/-0，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][CI] Add LM Eval Qwen3.5 Models test for MI355」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`；技术摘要: 覆盖「[ROCm][CI] Add LM Eval Qwen3.5 Models test for MI355」；主要实现面是 `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` added +1/-0 (1 lines); hunks: -0,0 +1。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` added +1/-0 (1 lines); hunks: -0,0 +1
- 关键代码摘录:

```diff
diff -- tests/evals/gsm8k/configs/models-qwen35-mi355.txt
@@ -0,0 +1 @@
+Qwen3.5-35B-A3B-DEP2.yaml
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` added +1/-0
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37975 - [Model] Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5

- 链接: https://github.com/vllm-project/vllm/pull/37975
- 状态/时间: merged / 2026-03-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `a8eab8f30dda`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+1053/-1126，可读 patch 2304 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5」；模型线: Qwen3.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「[Model] Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +4/-151 (155 lines); hunks: -28,7 +28,6; -40,18 +39,14; symbols: get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering, __init__，涉及 `get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +4/-151 (155 lines); hunks: -28,7 +28,6; -40,18 +39,14; symbols: get_hf_config, Qwen3_5GatedDeltaNet, fix_query_key_value_ordering, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -28,7 +28,6 @@
-from einops import rearrange
@@ -40,18 +39,14 @@
-from vllm.model_executor.layers.linear import (
-    ColumnParallelLinear,
-    MergedColumnParallelLinear,
-)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +4/-151
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38632 - [CI] fix LM Eval Qwen3.5 Models (B200)

- 链接: https://github.com/vllm-project/vllm/pull/38632
- 状态/时间: merged / 2026-03-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`；关联提交 `ea7bfde6e40d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 5 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] fix LM Eval Qwen3.5 Models (B200)」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`；技术摘要: 覆盖「[CI] fix LM Eval Qwen3.5 Models (B200)」；主要实现面是 `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` modified +1/-0 (1 lines); hunks: -7,3 +7,4 @@ server_args: >-。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` modified +1/-0 (1 lines); hunks: -7,3 +7,4 @@ server_args: >-
- 关键代码摘录:

```diff
diff -- tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml
@@ -7,3 +7,4 @@ server_args: >-
+  --max-num-seqs 512
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38650 - [Bugfix] Enable MTP for the official Qwen3.5 NVFP4 checkpoint

- 链接: https://github.com/vllm-project/vllm/pull/38650
- 状态/时间: closed / 2026-04-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+25/-9，可读 patch 70 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Enable MTP for the official Qwen3.5 NVFP4 checkpoint」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5_mtp.py`；技术摘要: 覆盖「[Bugfix] Enable MTP for the official Qwen3.5 NVFP4 checkpoint」；主要实现面是 `vllm/model_executor/models/qwen3_5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5_mtp.py` modified +25/-9 (34 lines); hunks: -15,6 +15,7; -43,6 +44,15; symbols: _get_qwen3_5_mtp_quant_config, __init__，涉及 `_get_qwen3_5_mtp_quant_config, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5_mtp.py` modified +25/-9 (34 lines); hunks: -15,6 +15,7; -43,6 +44,15; symbols: _get_qwen3_5_mtp_quant_config, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5_mtp.py
@@ -15,6 +15,7 @@
+from vllm.model_executor.layers.quantization import QuantizationConfig
@@ -43,6 +44,15 @@
+def _get_qwen3_5_mtp_quant_config(
+    quant_config: QuantizationConfig | None,
+) -> QuantizationConfig | None:
+    # Qwen3.5 NVFP4 checkpoints keep the entire MTP branch in bf16 weights.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5_mtp.py` modified +25/-9
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38832 - [Bugfix] Fix NVFP4+MTP crash: force unquantized mtp.fc for Qwen3.5

- 链接: https://github.com/vllm-project/vllm/pull/38832
- 状态/时间: merged / 2026-04-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5_mtp.py`；关联提交 `771913e4a024`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-1，可读 patch 24 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix NVFP4+MTP crash: force unquantized mtp.fc for Qwen3.5」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5_mtp.py`；技术摘要: 覆盖「[Bugfix] Fix NVFP4+MTP crash: force unquantized mtp.fc for Qwen3.5」；主要实现面是 `vllm/model_executor/models/qwen3_5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5_mtp.py` modified +10/-1 (11 lines); hunks: -75,13 +75,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5_mtp.py` modified +10/-1 (11 lines); hunks: -75,13 +75,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5_mtp.py
@@ -75,13 +75,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        # Workaround: mtp.fc is stored as BF16 in NVFP4 checkpoints but is
+        # missing from hf_quant_config.json exclude_modules. Force unquantized.
+        # Ref: https://github.com/vllm-project/vllm/pull/38650
+        # Ref: https://github.com/NVIDIA/Model-Optimizer/pull/1124
+        fc_quant = (
+            None
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5_mtp.py` modified +10/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38664 - [CI][ROCm] Add Qwen3.5-35B-A3B-MXFP4 model eval into CI

- 链接: https://github.com/vllm-project/vllm/pull/38664
- 状态/时间: merged / 2026-04-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml`, `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`；关联提交 `201d2ea5bfb9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+9/-0，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI][ROCm] Add Qwen3.5-35B-A3B-MXFP4 model eval into CI」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml`, `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`；技术摘要: 覆盖「[CI][ROCm] Add Qwen3.5-35B-A3B-MXFP4 model eval into CI」；主要实现面是 `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml`, `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8；`tests/evals/gsm8k/configs/models-qwen35-mi355.txt` modified +1/-0 (1 lines); hunks: -1 +1,2。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8
  - `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` modified +1/-0 (1 lines); hunks: -1 +1,2
- 关键代码摘录:

```diff
diff -- tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml
@@ -0,0 +1,8 @@
+model_name: "amd/Qwen3.5-35B-A3B-MXFP4"
+accuracy_threshold: 0.82
+tolerance: 0.03
+num_questions: 1319
+num_fewshot: 5
+server_args: >-
diff -- tests/evals/gsm8k/configs/models-qwen35-mi355.txt
@@ -1 +1,2 @@
+Qwen3.5-35B-A3B-MXFP4-TP2.yaml
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml` added +8/-0; `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml`, `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38927 - [Bugfix][LoRA] Fix missing in_proj_z in Qwen3_5ForConditionalGenerati…

- 链接: https://github.com/vllm-project/vllm/pull/38927
- 状态/时间: merged / 2026-04-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`；关联提交 `81994e1d0ea6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][LoRA] Fix missing in_proj_z in Qwen3_5ForConditionalGenerati…」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5.py`；技术摘要: 覆盖「[Bugfix][LoRA] Fix missing in_proj_z in Qwen3_5ForConditionalGenerati…」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +1/-0 (1 lines); hunks: -620,6 +620,7 @@ def update_packed_mapping(self, enable_lora: bool):; symbols: update_packed_mapping, embed_input_ids，涉及 `update_packed_mapping, embed_input_ids`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +1/-0 (1 lines); hunks: -620,6 +620,7 @@ def update_packed_mapping(self, enable_lora: bool):; symbols: update_packed_mapping, embed_input_ids
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -620,6 +620,7 @@ def update_packed_mapping(self, enable_lora: bool):
+            self.packed_modules_mapping["in_proj_z"] = ["in_proj_z"]
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39181 - [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next

- 链接: https://github.com/vllm-project/vllm/pull/39181
- 状态/时间: merged / 2026-04-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-0，可读 patch 32 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py`；技术摘要: 覆盖「[Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next」；主要实现面是 `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_moe.py` modified +3/-0 (3 lines); hunks: -80,6 +80,7 @@ def __init__(; -88,6 +89,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/qwen3_next.py` modified +1/-0 (1 lines); hunks: -140,6 +140,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_moe.py` modified +3/-0 (3 lines); hunks: -80,6 +80,7 @@ def __init__(; -88,6 +89,7 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/qwen3_next.py` modified +1/-0 (1 lines); hunks: -140,6 +140,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen2_moe.py
@@ -80,6 +80,7 @@ def __init__(
+        is_sequence_parallel: bool = False,
@@ -88,6 +89,7 @@ def __init__(
+            disable_tp=is_sequence_parallel,
@@ -96,6 +98,7 @@ def __init__(
+            disable_tp=is_sequence_parallel,
diff -- vllm/model_executor/models/qwen3_next.py
@@ -140,6 +140,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
+                is_sequence_parallel=self.is_sequence_parallel,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_moe.py` modified +3/-0; `vllm/model_executor/models/qwen3_next.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37114 - [Bugfix] LoRA: extend expert base_layer loading to Qwen3.5 and Step3.x

- 链接: https://github.com/vllm-project/vllm/pull/37114
- 状态/时间: merged / 2026-04-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`；关联提交 `908a713488db`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+34/-16，可读 patch 104 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] LoRA: extend expert base_layer loading to Qwen3.5 and Step3.x」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`；技术摘要: 覆盖「[Bugfix] LoRA: extend expert base_layer loading to Qwen3.5 and Step3.x」；主要实现面是 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_5.py` modified +5/-2 (7 lines); hunks: -306,9 +306,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`；`vllm/model_executor/models/qwen3_5_mtp.py` modified +5/-2 (7 lines); hunks: -207,9 +207,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_5.py` modified +5/-2 (7 lines); hunks: -306,9 +306,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
  - `vllm/model_executor/models/qwen3_5_mtp.py` modified +5/-2 (7 lines); hunks: -207,9 +207,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -306,9 +306,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        base_layer = (
+            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
+        )
-            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
-            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
+            (f"experts.{base_layer}w13_weight", "experts.gate_up_proj", 0, "w1"),
diff -- vllm/model_executor/models/qwen3_5_mtp.py
@@ -207,9 +207,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        base_layer = (
+            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
+        )
-            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
-            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
+            (f"experts.{base_layer}w13_weight", "experts.gate_up_proj", 0, "w1"),
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +5/-2; `vllm/model_executor/models/qwen3_5_mtp.py` modified +5/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py`, `vllm/model_executor/models/qwen3_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
