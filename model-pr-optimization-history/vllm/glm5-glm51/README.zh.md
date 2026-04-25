# vllm GLM-5/5.1 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| - | 当前主线没有匹配到实现文件 |

## PR 覆盖总览

- git 追溯 PR 数: 0
- 原文档显式引用补充 PR 数: 2
- 当前文档总 PR 数: 2
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-02-09 | [#34124](https://github.com/vllm-project/vllm/pull/34124) | merged | [Model] GLM adaptation | `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py` |
| 2026-02-12 | [#34385](https://github.com/vllm-project/vllm/pull/34385) | merged | [Bugfix] Fix MTP accuracy for GLM-5 | `vllm/v1/spec_decode/eagle.py` |

## 逐 PR diff 审计卡

### PR #34124 - [Model] GLM adaptation

- 链接: https://github.com/vllm-project/vllm/pull/34124
- 状态/时间: merged / 2026-02-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+13/-3，可读 patch 72 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] GLM adaptation」；模型线: GLM-5/5.1；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -836,7 +836,7 @@ def __init__(; -1499,6 +1499,10 @@ class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):; symbols: __init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM, get_spec_layer_idx_from_weight_name，涉及 `__init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM`；`tests/models/registry.py` modified +3/-0 (3 lines); hunks: -275,6 +275,9 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`tests/models/test_initialization.py` modified +1/-1 (2 lines); hunks: -97,7 +97,7 @@ def _initialize_kv_caches_v1(self, vllm_config):; symbols: _initialize_kv_caches_v1，涉及 `_initialize_kv_caches_v1`；`vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -114,6 +114,7。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -836,7 +836,7 @@ def __init__(; -1499,6 +1499,10 @@ class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):; symbols: __init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM, get_spec_layer_idx_from_weight_name
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunks: -275,6 +275,9 @@ def check_available_online(; symbols: check_available_online
  - `tests/models/test_initialization.py` modified +1/-1 (2 lines); hunks: -97,7 +97,7 @@ def _initialize_kv_caches_v1(self, vllm_config):; symbols: _initialize_kv_caches_v1
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -114,6 +114,7
  - `vllm/config/speculative.py` modified +1/-1 (2 lines); hunks: -181,7 +181,7 @@ def compute_hash(self) -> str:; symbols: compute_hash, hf_config_override
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -836,7 +836,7 @@ def __init__(
-                is_neox_style=True,
+                is_neox_style=not getattr(config, "indexer_rope_interleave", True),
@@ -1499,6 +1499,10 @@ class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
+class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
+    pass
diff -- tests/models/registry.py
@@ -275,6 +275,9 @@ def check_available_online(
+    "GlmMoeDsaForCausalLM": _HfExamplesInfo(
+        "zai-org/GLM-5", min_transformers_version="5.0.1", is_available_online=False
+    ),
diff -- tests/models/test_initialization.py
@@ -97,7 +97,7 @@ def _initialize_kv_caches_v1(self, vllm_config):
-    if model_arch == "DeepseekV32ForCausalLM":
+    if model_arch in ["DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"]:
diff -- vllm/model_executor/models/registry.py
@@ -114,6 +114,7 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1; `vllm/model_executor/models/registry.py` modified +1/-0; `vllm/config/speculative.py` modified +1/-1; `vllm/transformers_utils/model_arch_config_convertor.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +3/-0; `tests/models/test_initialization.py` modified +1/-1
  - other: `benchmarks/kernels/benchmark_moe.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`, `tests/models/test_initialization.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34385 - [Bugfix] Fix MTP accuracy for GLM-5

- 链接: https://github.com/vllm-project/vllm/pull/34385
- 状态/时间: merged / 2026-02-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+18/-0，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix MTP accuracy for GLM-5」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `vllm/v1/spec_decode/eagle.py`；PR 正文摘要: Fix MTP producing NaN logits for models (e.g. GLM-5) whose checkpoints don't store a duplicate `shared_head.head` weight in the MTP layer (like DeepSeek V3.2). The existing `_ma...。
- 实现要点: `vllm/v1/spec_decode/eagle.py` modified +18/-0 (18 lines); hunks: -1506,6 +1506,24 @@ def _maybe_share_lm_head(self, target_language_model: nn....; symbols: _maybe_share_lm_head, dummy_run，涉及 `_maybe_share_lm_head, dummy_run`。
- 代码 diff 细节:
  - `vllm/v1/spec_decode/eagle.py` modified +18/-0 (18 lines); hunks: -1506,6 +1506,24 @@ def _maybe_share_lm_head(self, target_language_model: nn....; symbols: _maybe_share_lm_head, dummy_run
- 关键代码摘录:

```diff
diff -- vllm/v1/spec_decode/eagle.py
@@ -1506,6 +1506,24 @@ def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
+            # MTP models call compute_logits via shared_head.head (a
+            # ParallelLMHead inside each MTP layer), not self.model.lm_head.
+            # If the checkpoint omits a copy of the lm_head weights at the
+            # MTP layer path, shared_head.head stays uninitialised and
+            # produces NaN logits. Always share it explicitly.
+            inner = getattr(self.model, "model", None)
```

- 已读文件:
  - runtime: `vllm/v1/spec_decode/eagle.py` modified +18/-0
- 验证与风险: runtime 路径改动集中在 `vllm/v1/spec_decode/eagle.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
