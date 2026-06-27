# vllm Qwen3 Core 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/models/multimodal/pooling/test_colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398), [#34574](https://github.com/vllm-project/vllm/pull/34574) |
| `tests/parser/engine/test_qwen3.py` | [#45413](https://github.com/vllm-project/vllm/pull/45413), [#46047](https://github.com/vllm-project/vllm/pull/46047), [#46351](https://github.com/vllm-project/vllm/pull/46351) |
| `vllm/model_executor/models/colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398), [#34574](https://github.com/vllm-project/vllm/pull/34574) |
| `vllm/model_executor/models/qwen3.py` | [#15289](https://github.com/vllm-project/vllm/pull/15289), [#17735](https://github.com/vllm-project/vllm/pull/17735), [#19260](https://github.com/vllm-project/vllm/pull/19260), [#21924](https://github.com/vllm-project/vllm/pull/21924), [#29816](https://github.com/vllm-project/vllm/pull/29816) |
| `vllm/model_executor/models/qwen3_dflash.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/qwen3_moe.py` | [#15289](https://github.com/vllm-project/vllm/pull/15289), [#16203](https://github.com/vllm-project/vllm/pull/16203), [#17735](https://github.com/vllm-project/vllm/pull/17735), [#18118](https://github.com/vllm-project/vllm/pull/18118), [#19598](https://github.com/vllm-project/vllm/pull/19598), [#19860](https://github.com/vllm-project/vllm/pull/19860), [#20101](https://github.com/vllm-project/vllm/pull/20101), [#20815](https://github.com/vllm-project/vllm/pull/20815), [#21924](https://github.com/vllm-project/vllm/pull/21924), [#22017](https://github.com/vllm-project/vllm/pull/22017), [#22785](https://github.com/vllm-project/vllm/pull/22785), [#23169](https://github.com/vllm-project/vllm/pull/23169), ... (24 total) |
| `vllm/parser/qwen3.py` | [#45413](https://github.com/vllm-project/vllm/pull/45413), [#45763](https://github.com/vllm-project/vllm/pull/45763), [#46047](https://github.com/vllm-project/vllm/pull/46047), [#46314](https://github.com/vllm-project/vllm/pull/46314), [#46351](https://github.com/vllm-project/vllm/pull/46351) |
| `vllm/transformers_utils/configs/colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398) |

## PR 覆盖总览

- git 追溯 PR 数: 1
- 原文档显式引用补充 PR 数: 8
- 当前文档总 PR 数: 5
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-09-17 | [#24727](https://github.com/vllm-project/vllm/pull/24727) | merged | [Model] Support Qwen3-VL Model Series | `vllm/model_executor/models/qwen3_moe.py` |
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | merged | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py` |
| 2026-05-11 | [#42280](https://github.com/vllm-project/vllm/pull/42280) | merged | [Model] Fix missing `maybe_prefix` | `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/cohere_asr.py`, `vllm/model_executor/models/hunyuan_v1.py` |
| 2026-06-05 | [#43167](https://github.com/vllm-project/vllm/pull/43167) | merged | Remove KV cache scale boilerplate from model weight loading methods | `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py` |
| 2026-06-10 | [#39419](https://github.com/vllm-project/vllm/pull/39419) | merged | [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding | `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py` |

## 逐 PR diff 审计卡

### PR #24727 - [Model] Support Qwen3-VL Model Series

- 链接: https://github.com/vllm-project/vllm/pull/24727
- 状态/时间: merged / 2025-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `0f7acdd73ca6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+2084/-17，可读 patch 2262 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support Qwen3-VL Model Series」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Model] Support Qwen3-VL Model Series」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -378,7 +378,7 @@ class Qwen3MoeModel(nn.Module):; symbols: Qwen3MoeModel, __init__，涉及 `Qwen3MoeModel, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -378,7 +378,7 @@ class Qwen3MoeModel(nn.Module):; symbols: Qwen3MoeModel, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -378,7 +378,7 @@ class Qwen3MoeModel(nn.Module):
-        config = vllm_config.model_config.hf_config
+        config = vllm_config.model_config.hf_config.get_text_config()
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- 链接: https://github.com/vllm-project/vllm/pull/40671
- 状态/时间: merged / 2026-04-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 53 个文件，+254/-98，可读 patch 1073 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`；技术摘要: 覆盖「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #42280 - [Model] Fix missing `maybe_prefix`

- 链接: https://github.com/vllm-project/vllm/pull/42280
- 状态/时间: merged / 2026-05-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 25 个文件，+49/-29，可读 patch 302 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Fix missing `maybe_prefix`」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/cohere_asr.py`, `vllm/model_executor/models/hunyuan_v1.py`；技术摘要: 覆盖「[Model] Fix missing `maybe_prefix`」；主要实现面是 `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/cohere_asr.py`, `vllm/model_executor/models/hunyuan_v1.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/arcee.py` modified +6/-2 (8 lines); hunks: -45,6 +45,7; -367,7 +368,10 @@ def __init__(self, *, vllm_config, prefix: str = "") -> None:; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/cohere_asr.py` modified +3/-2 (5 lines); hunks: -64,7 +64,7; -1717,7 +1717,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/hunyuan_v1.py` modified +4/-1 (5 lines); hunks: -930,7 +930,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/deepseek_eagle.py` modified +3/-1 (4 lines); hunks: -198,7 +198,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/arcee.py` modified +6/-2 (8 lines); hunks: -45,6 +45,7; -367,7 +368,10 @@ def __init__(self, *, vllm_config, prefix: str = "") -> None:; symbols: __init__
  - `vllm/model_executor/models/cohere_asr.py` modified +3/-2 (5 lines); hunks: -64,7 +64,7; -1717,7 +1717,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/hunyuan_v1.py` modified +4/-1 (5 lines); hunks: -930,7 +930,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/deepseek_eagle.py` modified +3/-1 (4 lines); hunks: -198,7 +198,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
  - `vllm/model_executor/models/deepseek_eagle3.py` modified +3/-1 (4 lines); hunks: -318,7 +318,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/arcee.py
@@ -45,6 +45,7 @@
+    maybe_prefix,
@@ -367,7 +368,10 @@ def __init__(self, *, vllm_config, prefix: str = "") -> None:
-        self.model = ArceeModel(vllm_config=vllm_config, prefix=f"{prefix}.model")
+        self.model = ArceeModel(
+            vllm_config=vllm_config,
+            prefix=maybe_prefix(prefix, "model"),
diff -- vllm/model_executor/models/cohere_asr.py
@@ -64,7 +64,7 @@
-from .utils import AutoWeightsLoader, WeightsMapper, make_layers
+from .utils import AutoWeightsLoader, WeightsMapper, make_layers, maybe_prefix
@@ -1717,7 +1717,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-            vllm_config=vllm_config, prefix=f"{prefix}.decoder"
+            vllm_config=vllm_config,
+            prefix=maybe_prefix(prefix, "decoder"),
diff -- vllm/model_executor/models/hunyuan_v1.py
@@ -930,7 +930,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/arcee.py` modified +6/-2; `vllm/model_executor/models/cohere_asr.py` modified +3/-2; `vllm/model_executor/models/hunyuan_v1.py` modified +4/-1; `vllm/model_executor/models/deepseek_eagle.py` modified +3/-1; `vllm/model_executor/models/deepseek_eagle3.py` modified +3/-1; `vllm/model_executor/models/granite_speech.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/blip2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43167 - Remove KV cache scale boilerplate from model weight loading methods

- 链接: https://github.com/vllm-project/vllm/pull/43167
- 状态/时间: merged / 2026-06-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 56 个文件，+88/-731，可读 patch 1251 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Remove KV cache scale boilerplate from model weight loading methods」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`；技术摘要: 覆盖「Remove KV cache scale boilerplate from model weight loading methods」；主要实现面是 `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name，涉及 `test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale`；`vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader，涉及 `_get_moe_weight_dtype, kv_cache_scale_loader`；`vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod，涉及 `KVCacheScaleParameter, __new__, weight_loader`；`vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter，涉及 `get_quant_method, get_cache_scale, get_cache_scale_mapper`。
- 代码 diff 细节:
  - `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name
  - `vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader
  - `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod
  - `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20 (30 lines); hunks: -646,26 +646,16 @@ def get_scheme(; symbols: get_scheme, get_cache_scale, get_cache_scale_mapper, QuarkLinearMethod
- 关键代码摘录:

```diff
diff -- tests/model_executor/test_eagle_quantization.py
@@ -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, dist_init, device) ->
-def test_kv_cache_scale_name_handling():
-    # Mock a quant config that supports cache scales
-    mock_quant_config = Mock()
-    mock_quant_config.get_cache_scale = Mock(return_value="layers.0.self_attn.kv_scale")
-    # Condition check in load_weights
-    name = "layers.0.self_attn.k_proj.weight"
diff -- vllm/model_executor/models/gpt_oss.py
@@ -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:
-            def kv_cache_scale_loader(
-                quant_config: QuantizationConfig,
-                name: str,
-                params_dict: dict[str, typing.Any],
-                weight: torch.Tensor,
-                default_weight_loader: Callable[..., None],
diff -- vllm/model_executor/layers/quantization/kv_cache.py
@@ -15,6 +15,30 @@
```

- 已读文件:
  - tests: `tests/model_executor/test_eagle_quantization.py` modified +0/-56
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +0/-46; `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4; `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19; `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20; `vllm/model_executor/models/llama4.py` modified +3/-18; `vllm/model_executor/models/glm_ocr_mtp.py` modified +4/-13
- 验证与风险: diff 自带测试面 `tests/model_executor/test_eagle_quantization.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39419 - [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding

- 链接: https://github.com/vllm-project/vllm/pull/39419
- 状态/时间: merged / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+53/-39，可读 patch 169 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py`；技术摘要: 覆盖「[SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding」；主要实现面是 `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
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

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
