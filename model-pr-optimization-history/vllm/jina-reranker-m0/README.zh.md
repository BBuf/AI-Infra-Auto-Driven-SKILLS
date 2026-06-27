# vllm Jina Reranker M0 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/pooling/embed/embed_jina_embeddings_v3_offline.py` | 无直接 PR 号提交 |
| `examples/pooling/score/template/qwen3_vl_reranker.jinja` | [#31890](https://github.com/vllm-project/vllm/pull/31890), [#42412](https://github.com/vllm-project/vllm/pull/42412) |
| `examples/pooling/score/vision_reranker_offline.py` | 无直接 PR 号提交 |
| `examples/pooling/token_embed/jina_embeddings_v4_offline.py` | 无直接 PR 号提交 |
| `examples/pooling/token_embed/jina_reranker_v3_offline.py` | [#38800](https://github.com/vllm-project/vllm/pull/38800) |
| `tests/models/language/pooling/test_jina_reranker_v3.py` | [#38800](https://github.com/vllm-project/vllm/pull/38800) |
| `tests/models/language/pooling_mteb_test/test_bge_reranker_v2_gemma.py` | 无直接 PR 号提交 |
| `tests/models/language/pooling_mteb_test/test_jina.py` | [#26687](https://github.com/vllm-project/vllm/pull/26687), [#38633](https://github.com/vllm-project/vllm/pull/38633), [#39575](https://github.com/vllm-project/vllm/pull/39575) |
| `tests/models/multimodal/pooling/test_jinavl_reranker.py` | [#20260](https://github.com/vllm-project/vllm/pull/20260), [#20907](https://github.com/vllm-project/vllm/pull/20907), [#31445](https://github.com/vllm-project/vllm/pull/31445) |
| `vllm/model_executor/models/jina.py` | [#38633](https://github.com/vllm-project/vllm/pull/38633), [#38800](https://github.com/vllm-project/vllm/pull/38800), [#39575](https://github.com/vllm-project/vllm/pull/39575) |
| `vllm/model_executor/models/jina_vl.py` | [#20260](https://github.com/vllm-project/vllm/pull/20260) |

## PR 覆盖总览

- git 追溯 PR 数: 8
- 原文档显式引用补充 PR 数: 35
- 当前文档总 PR 数: 43
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-10 | [#20260](https://github.com/vllm-project/vllm/pull/20260) | merged | [Model][VLM] Support JinaVL Reranker | `tests/models/multimodal/pooling/test_jinavl_reranker.py`, `vllm/model_executor/models/jina_vl.py` |
| 2025-07-14 | [#20907](https://github.com/vllm-project/vllm/pull/20907) | merged | [CI/Build] Fix OOM issue in Jina-VL test | `tests/models/multimodal/pooling/test_jinavl_reranker.py` |
| 2025-07-17 | [#21058](https://github.com/vllm-project/vllm/pull/21058) | merged | [Model] Update pooling model interface | `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/bert.py` |
| 2025-07-21 | [#21227](https://github.com/vllm-project/vllm/pull/21227) | merged | [Model][1/N] Support multiple poolers at model level | `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/models/bert.py`, `vllm/model_executor/models/adapters.py` |
| 2025-07-21 | [#20996](https://github.com/vllm-project/vllm/pull/20996) | merged | [Misc] unify variable for LLM instance | `tests/detokenizer/test_stop_strings.py`, `tests/models/language/pooling/mteb_utils.py`, `tests/models/language/generation/test_mistral.py` |
| 2025-07-28 | [#21470](https://github.com/vllm-project/vllm/pull/21470) | merged | [Deprecation][2/N] Replace `--task` with `--runner` and `--convert` | `vllm/model_executor/models/registry.py`, `vllm/model_executor/model_loader/utils.py`, `docs/models/supported_models.md` |
| 2025-08-05 | [#20538](https://github.com/vllm-project/vllm/pull/20538) | merged | [Model] Pooling model activation supports per request control by PoolingParams | `vllm/model_executor/layers/pooler.py`, `tests/models/language/pooling/test_override_pooler_config.py`, `tests/entrypoints/llm/test_score.py` |
| 2025-09-02 | [#24031](https://github.com/vllm-project/vllm/pull/24031) | merged | [Model] Classification models support logit_bias / sigmoid_normalize | `vllm/model_executor/models/jina_vl.py`, `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/models/config.py` |
| 2025-09-09 | [#23810](https://github.com/vllm-project/vllm/pull/23810) | merged | [Model] Systematic support for fp32 head, pooling models part | `vllm/model_executor/layers/pooler.py`, `tests/models/language/pooling/mteb_utils.py`, `vllm/model_executor/models/internlm2.py` |
| 2025-10-05 | [#26247](https://github.com/vllm-project/vllm/pull/26247) | merged | Convert formatting to use `ruff` instead of `yapf` + `isort` | `tests/entrypoints/test_chat_utils.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py` |
| 2025-10-12 | [#26633](https://github.com/vllm-project/vllm/pull/26633) | merged | Update `Optional[x]` -> `x \| None` and `Union[x, y]` to `x \| y` | `vllm/entrypoints/openai/protocol.py`, `vllm/entrypoints/llm.py`, `vllm/model_executor/layers/fused_moe/layer.py` |
| 2025-10-13 | [#26687](https://github.com/vllm-project/vllm/pull/26687) | merged | [Bugfix] Fix out of bound index issue for Jina-embedding-v3 RoPE with cuda graph | `tests/models/language/pooling_mteb_test/test_jina.py` |
| 2025-10-15 | [#25370](https://github.com/vllm-project/vllm/pull/25370) | merged | [Model][2/N] Improve all pooling task \| Support multi-vector retrieval | `vllm/model_executor/layers/pooler.py`, `tests/models/language/pooling/test_pooler_config_init_behaviour.py`, `tests/models/language/pooling/test_multi_vector_retrieval.py` |
| 2025-12-02 | [#29802](https://github.com/vllm-project/vllm/pull/29802) | merged | Fix some Transformers nightly tests | `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/jina_vl.py`, `vllm/model_executor/models/qwen2.py` |
| 2025-12-29 | [#31445](https://github.com/vllm-project/vllm/pull/31445) | merged | [Bugfix][Frontend] Fix Jina reranker multimodal input compatibility | `tests/models/multimodal/pooling/test_jinavl_reranker.py` |
| 2026-01-05 | [#31669](https://github.com/vllm-project/vllm/pull/31669) | merged | [Misc][Model][Refactor] Pass the prefix into Linear layers | `vllm/model_executor/models/molmo.py`, `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/qwen_vl.py` |
| 2026-01-08 | [#31890](https://github.com/vllm-project/vllm/pull/31890) | merged | [Models] Allow converting Qwen3-VL into Reranker model | `examples/pooling/score/template/qwen3_vl_reranker.jinja` |
| 2026-01-09 | [#31973](https://github.com/vllm-project/vllm/pull/31973) | merged | [Model] Reorganize pooling layers | `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/layers/pooler/activations.py`, `vllm/model_executor/layers/pooler/seqwise/heads.py` |
| 2026-01-12 | [#32085](https://github.com/vllm-project/vllm/pull/32085) | merged | [Model] Improve multimodal pooling examples | `examples/pooling/score/openai_cross_encoder_score_for_multimodal.py`, `examples/pooling/embed/vision_embedding_online.py`, `examples/pooling/embed/vision_embedding_offline.py` |
| 2026-01-16 | [#32395](https://github.com/vllm-project/vllm/pull/32395) | merged | [Frontend][1/n] Make pooling entrypoints request schema consensus \| CompletionRequest | `tests/entrypoints/pooling/embed/test_online.py`, `tests/entrypoints/pooling/classify/test_online.py`, `tests/entrypoints/pooling/pooling/test_online.py` |
| 2026-01-19 | [#32577](https://github.com/vllm-project/vllm/pull/32577) | merged | [Frontend] Score entrypoint support data_1 & data_2 and queries & documents as inputs | `tests/entrypoints/pooling/score/test_online_score.py`, `vllm/entrypoints/pooling/score/protocol.py`, `vllm/entrypoints/pooling/score/serving.py` |
| 2026-01-22 | [#32287](https://github.com/vllm-project/vllm/pull/32287) | merged | Upgrade transformers-4.57.5 | `tests/models/multimodal/pooling/test_jinavl_reranker.py`, `requirements/nightly_torch_test.txt`, `requirements/test.in` |
| 2026-01-26 | [#33063](https://github.com/vllm-project/vllm/pull/33063) | merged | [Chore] Update type annotation of `input_ids` in model forward | `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py` |
| 2026-01-29 | [#33298](https://github.com/vllm-project/vllm/pull/33298) | merged | [Bugfix] Fix Qwen3-VL-Reranker load. | `tests/entrypoints/pooling/score/test_online_score_vision.py`, `vllm/model_executor/models/adapters.py`, `tests/entrypoints/test_utils.py` |
| 2026-02-04 | [#33060](https://github.com/vllm-project/vllm/pull/33060) | merged | [Frontend][4/n] Make pooling entrypoints request schema consensus \| ScoreRequest | `vllm/entrypoints/pooling/score/serving.py`, `tests/entrypoints/pooling/score/test_online_score_vision.py`, `vllm/entrypoints/llm.py` |
| 2026-02-05 | [#33837](https://github.com/vllm-project/vllm/pull/33837) | merged | [Bugfix] Fix ScoreMultiModalParam multi-document scoring returning single result | `tests/models/multimodal/pooling/test_jinavl_reranker.py` |
| 2026-02-09 | [#31127](https://github.com/vllm-project/vllm/pull/31127) | merged | [Frontend][last/5] Make pooling entrypoints request schema consensus. | `tests/entrypoints/pooling/embed/test_online_vision.py`, `tests/entrypoints/pooling/classify/test_offline.py`, `vllm/entrypoints/pooling/score/protocol.py` |
| 2026-03-19 | [#35592](https://github.com/vllm-project/vllm/pull/35592) | merged | [Docs] Reorganize pooling docs. | `docs/models/pooling_models.md`, `docs/models/pooling_models/embed.md`, `docs/models/pooling_models/scoring.md` |
| 2026-03-20 | [#37537](https://github.com/vllm-project/vllm/pull/37537) | merged | [Model] Deprecate the score task (this will not affect users). | `docs/models/pooling_models/README.md`, `vllm/model_executor/layers/pooler/seqwise/heads.py`, `vllm/entrypoints/pooling/__init__.py` |
| 2026-03-25 | [#37902](https://github.com/vllm-project/vllm/pull/37902) | merged | [Mypy] Better fixes for the `mypy` issues in `vllm/config` | `tests/models/multimodal/generation/test_keye.py`, `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `examples/offline_inference/encoder_decoder_multimodal.py` |
| 2026-03-31 | [#28631](https://github.com/vllm-project/vllm/pull/28631) | merged | [Frontend][3/n] Improve pooling entrypoints \| scoring. | `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/pooling/scoring/io_processor.py`, `tests/entrypoints/pooling/scoring/test_utils.py` |
| 2026-04-10 | [#38800](https://github.com/vllm-project/vllm/pull/38800) | merged | [New Model]: jinaai/jina-reranker-v3 | `tests/models/language/pooling/test_jina_reranker_v3.py`, `vllm/model_executor/models/jina.py`, `examples/pooling/token_embed/jina_reranker_v3_offline.py` |
| 2026-04-15 | [#30566](https://github.com/vllm-project/vllm/pull/30566) | merged | Update to transformers v5 | `tests/models/registry.py`, `vllm/model_executor/models/gemma4_mm.py`, `tests/models/multimodal/generation/test_common.py` |
| 2026-04-16 | [#39575](https://github.com/vllm-project/vllm/pull/39575) | merged | Add Jina Embeddings v5 model support (fixes #38633) | `vllm/model_executor/models/jina.py`, `tests/models/language/pooling_mteb_test/test_jina.py` |
| 2026-04-16 | [#39675](https://github.com/vllm-project/vllm/pull/39675) | merged | [Frontend][last/5] Improve pooling entrypoints \| clean up. | `vllm/entrypoints/pooling/factories.py`, `vllm/entrypoints/pooling/__init__.py`, `vllm/entrypoints/sagemaker/api_router.py` |
| 2026-05-06 | [#41832](https://github.com/vllm-project/vllm/pull/41832) | merged | [Doc] Add ModernBertForSequenceClassification to scoring.md cross-en… | `docs/models/pooling_models/scoring.md` |
| 2026-05-14 | [#42412](https://github.com/vllm-project/vllm/pull/42412) | merged | [Feature] Add instruction support for score/rerank chat templates | `examples/pooling/score/template/qwen3_vl_reranker.jinja` |
| 2026-05-15 | [#42267](https://github.com/vllm-project/vllm/pull/42267) | merged | [Entrypoints] Split the pooling offline API into PoolingOfflineMixin. | `vllm/entrypoints/pooling/offline.py`, `vllm/entrypoints/llm.py`, `docs/models/pooling_models/README.md` |
| 2026-05-19 | [#42626](https://github.com/vllm-project/vllm/pull/42626) | merged | [Docs] Add SVG images for pooling models. | `docs/assets/models/pooling_models/score_types.svg`, `docs/assets/models/pooling_models/cheat_sheet.svg`, `docs/assets/models/pooling_models/pooling_types.svg` |
| 2026-05-19 | [#41907](https://github.com/vllm-project/vllm/pull/41907) | merged | [Docs] Reorganize online serving docs. | `docs/assets/models/pooling_models/cheat_sheet.svg`, `docs/models/pooling_models/README.md`, `docs/models/supported_models.md` |
| 2026-05-22 | [#43393](https://github.com/vllm-project/vllm/pull/43393) | merged | [Docs] Note image preprocessing difference between qwen_vl_utils and vllm. | `docs/models/supported_models.md`, `docs/models/pooling_models/embed.md`, `docs/models/pooling_models/scoring.md` |
| 2026-06-15 | [#45676](https://github.com/vllm-project/vllm/pull/45676) | merged | [Docs] Update the online serving docs. | `docs/models/pooling_models/scoring.md`, `docs/models/pooling_models/README.md`, `docs/serving/online_serving/README.md` |
| 2026-06-23 | [#46398](https://github.com/vllm-project/vllm/pull/46398) | merged | [Doc] Fix typos, grammar, and broken commands across docs | `docs/models/pooling_models/README.md`, `docs/models/pooling_models/scoring.md`, `docs/benchmarking/cli.md` |

## 逐 PR diff 审计卡

### PR #20260 - [Model][VLM] Support JinaVL Reranker

- 链接: https://github.com/vllm-project/vllm/pull/20260
- 状态/时间: merged / 2025-07-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/pooling/test_jinavl_reranker.py`, `vllm/model_executor/models/jina_vl.py`；关联提交 `4bed167768bd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+993/-133，可读 patch 1479 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][VLM] Support JinaVL Reranker」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `tests/models/multimodal/pooling/test_jinavl_reranker.py`, `vllm/model_executor/models/jina_vl.py`；技术摘要: 覆盖「[Model][VLM] Support JinaVL Reranker」；主要实现面是 `tests/models/multimodal/pooling/test_jinavl_reranker.py`, `vllm/model_executor/models/jina_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/pooling/test_jinavl_reranker.py` added +160/-0 (160 lines); hunks: -0,0 +1,160; symbols: vllm_reranker, create_image_param, hf_reranker, test_model_text_image，涉及 `vllm_reranker, create_image_param, hf_reranker`；`vllm/model_executor/models/jina_vl.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: JinaVLScorer, __init__, forward, JinaVLMultiModalProcessor，涉及 `JinaVLScorer, __init__, forward`。
- 代码 diff 细节:
  - `tests/models/multimodal/pooling/test_jinavl_reranker.py` added +160/-0 (160 lines); hunks: -0,0 +1,160; symbols: vllm_reranker, create_image_param, hf_reranker, test_model_text_image
  - `vllm/model_executor/models/jina_vl.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: JinaVLScorer, __init__, forward, JinaVLMultiModalProcessor
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/pooling/test_jinavl_reranker.py
@@ -0,0 +1,160 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from transformers import AutoModel
+model_name = "jinaai/jina-reranker-m0"
+mm_processor_kwargs = {
diff -- vllm/model_executor/models/jina_vl.py
@@ -0,0 +1,150 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable, Mapping
+from typing import Optional
+import torch
+import torch.nn as nn
```

- 已读文件:
  - tests: `tests/models/multimodal/pooling/test_jinavl_reranker.py` added +160/-0
  - runtime: `vllm/model_executor/models/jina_vl.py` added +150/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/pooling/test_jinavl_reranker.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20907 - [CI/Build] Fix OOM issue in Jina-VL test

- 链接: https://github.com/vllm-project/vllm/pull/20907
- 状态/时间: merged / 2025-07-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/pooling/test_jinavl_reranker.py`；关联提交 `dcf2a5e2088d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+85/-58，可读 patch 225 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI/Build] Fix OOM issue in Jina-VL test」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `tests/models/multimodal/pooling/test_jinavl_reranker.py`；技术摘要: 覆盖「[CI/Build] Fix OOM issue in Jina-VL test」；主要实现面是 `tests/models/multimodal/pooling/test_jinavl_reranker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +85/-58 (143 lines); hunks: -1,9 +1,15; -14,82 +20,99; symbols: vllm_reranker, create_image_param, hf_reranker，涉及 `vllm_reranker, create_image_param, hf_reranker`。
- 代码 diff 细节:
  - `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +85/-58 (143 lines); hunks: -1,9 +1,15; -14,82 +20,99; symbols: vllm_reranker, create_image_param, hf_reranker
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/pooling/test_jinavl_reranker.py
@@ -1,9 +1,15 @@
+from typing import Union
+from vllm.entrypoints.chat_utils import ChatCompletionContentPartImageParam
+from vllm.entrypoints.score_utils import ScoreMultiModalParam
+from ....conftest import HfRunner, VllmRunner
@@ -14,82 +20,99 @@
-def vllm_reranker(model_name,
```

- 已读文件:
  - tests: `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +85/-58
- 验证与风险: diff 自带测试面 `tests/models/multimodal/pooling/test_jinavl_reranker.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21058 - [Model] Update pooling model interface

- 链接: https://github.com/vllm-project/vllm/pull/21058
- 状态/时间: merged / 2025-07-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+247/-345，可读 patch 1411 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Update pooling model interface」；模型线: Jina Reranker M0；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/bert.py`；技术摘要: 覆盖「[Model] Update pooling model interface」；主要实现面是 `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/bert.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/pooler.py` modified +112/-64 (176 lines); hunks: -3,22 +3,25; -64,6 +67,48 @@ def from_config_with_defaults(; symbols: PoolingType, from_config_with_defaults, Pooler, get_pooling_params，涉及 `PoolingType, from_config_with_defaults, Pooler`；`vllm/model_executor/models/interfaces.py` modified +12/-74 (86 lines); hunks: -119,13 +119,6 @@ def get_input_embeddings(; -140,10 +133,7 @@ def supports_multimodal(model: object) -> TypeIs[SupportsMu...; symbols: get_input_embeddings, as, _SupportsMultiModalType, supports_multimodal，涉及 `get_input_embeddings, as, _SupportsMultiModalType`；`vllm/model_executor/models/bert.py` modified +18/-19 (37 lines); hunks: -18,12 +18,14; -80,7 +82,7 @@ def forward(; symbols: forward, BertPooler, __init__，涉及 `forward, BertPooler, __init__`；`vllm/entrypoints/openai/protocol.py` modified +5/-29 (34 lines); hunks: -1237,10 +1237,6 @@ class EmbeddingCompletionRequest(OpenAIBaseModel):; -1259,8 +1255,7 @@ class EmbeddingCompletionRequest(OpenAIBaseModel):; symbols: EmbeddingCompletionRequest, to_pooling_params, EmbeddingChatRequest，涉及 `EmbeddingCompletionRequest, to_pooling_params, EmbeddingChatRequest`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/pooler.py` modified +112/-64 (176 lines); hunks: -3,22 +3,25; -64,6 +67,48 @@ def from_config_with_defaults(; symbols: PoolingType, from_config_with_defaults, Pooler, get_pooling_params
  - `vllm/model_executor/models/interfaces.py` modified +12/-74 (86 lines); hunks: -119,13 +119,6 @@ def get_input_embeddings(; -140,10 +133,7 @@ def supports_multimodal(model: object) -> TypeIs[SupportsMu...; symbols: get_input_embeddings, as, _SupportsMultiModalType, supports_multimodal
  - `vllm/model_executor/models/bert.py` modified +18/-19 (37 lines); hunks: -18,12 +18,14; -80,7 +82,7 @@ def forward(; symbols: forward, BertPooler, __init__
  - `vllm/entrypoints/openai/protocol.py` modified +5/-29 (34 lines); hunks: -1237,10 +1237,6 @@ class EmbeddingCompletionRequest(OpenAIBaseModel):; -1259,8 +1255,7 @@ class EmbeddingCompletionRequest(OpenAIBaseModel):; symbols: EmbeddingCompletionRequest, to_pooling_params, EmbeddingChatRequest
  - `vllm/model_executor/models/interfaces_base.py` modified +16/-17 (33 lines); hunks: -1,8 +1,7; -13,8 +12,7; symbols: is_text_generation_model, VllmModelForPooling, pooler, is
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/pooler.py
@@ -3,22 +3,25 @@
-from typing import Callable, Optional, TypeVar, Union
+from typing import Callable, Literal, Optional, TypeVar, Union
+from typing_extensions import assert_never
+from vllm.pooling_params import PoolingParams
+PoolingTask = Literal["encode", "embed", "classify", "score"]
@@ -64,6 +67,48 @@ def from_config_with_defaults(
diff -- vllm/model_executor/models/interfaces.py
@@ -119,13 +119,6 @@ def get_input_embeddings(
-# We can't use runtime_checkable with ClassVar for issubclass checks
-# so we need to treat the class as an instance and use isinstance instead
-@runtime_checkable
-class _SupportsMultiModalType(Protocol):
-    supports_multimodal: Literal[True]
@@ -140,10 +133,7 @@ def supports_multimodal(model: object) -> TypeIs[SupportsMultiModal]:
diff -- vllm/model_executor/models/bert.py
@@ -18,12 +18,14 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/pooler.py` modified +112/-64; `vllm/model_executor/models/interfaces.py` modified +12/-74; `vllm/model_executor/models/bert.py` modified +18/-19; `vllm/entrypoints/openai/protocol.py` modified +5/-29; `vllm/model_executor/models/interfaces_base.py` modified +16/-17; `vllm/model_executor/models/adapters.py` modified +8/-23
- 验证与风险: diff 自带测试面 `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21227 - [Model][1/N] Support multiple poolers at model level

- 链接: https://github.com/vllm-project/vllm/pull/21227
- 状态/时间: merged / 2025-07-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 22 个文件，+550/-414，可读 patch 1581 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][1/N] Support multiple poolers at model level」；模型线: Jina Reranker M0；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/models/bert.py`, `vllm/model_executor/models/adapters.py`；技术摘要: 覆盖「[Model][1/N] Support multiple poolers at model level」；主要实现面是 `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/models/bert.py`, `vllm/model_executor/models/adapters.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/pooler.py` modified +175/-171 (346 lines); hunks: -1,15 +1,16; -21,6 +22,10; symbols: PoolingType, Pooler, from_config_with_defaults, for_encode，涉及 `PoolingType, Pooler, from_config_with_defaults`；`vllm/model_executor/models/bert.py` modified +99/-33 (132 lines); hunks: -1,7 +1,7; -17,7 +17,8; symbols: __init__, get_pooling_updates, get_supported_tasks, _head，涉及 `__init__, get_pooling_updates, get_supported_tasks`；`vllm/model_executor/models/adapters.py` modified +51/-57 (108 lines); hunks: -13,7 +13,6; -34,16 +33,8 @@ def _get_pooling_model_name(orig_model_name: str, pooling_suf...; symbols: _get_pooling_model_name, _create_pooling_model_cls, ModelForPooling, __init__，涉及 `_get_pooling_model_name, _create_pooling_model_cls, ModelForPooling`；`docs/models/pooling_models.md` modified +39/-14 (53 lines); hunks: -11,26 +11,51 @@ before returning them.。
- 代码 diff 细节:
  - `vllm/model_executor/layers/pooler.py` modified +175/-171 (346 lines); hunks: -1,15 +1,16; -21,6 +22,10; symbols: PoolingType, Pooler, from_config_with_defaults, for_encode
  - `vllm/model_executor/models/bert.py` modified +99/-33 (132 lines); hunks: -1,7 +1,7; -17,7 +17,8; symbols: __init__, get_pooling_updates, get_supported_tasks, _head
  - `vllm/model_executor/models/adapters.py` modified +51/-57 (108 lines); hunks: -13,7 +13,6; -34,16 +33,8 @@ def _get_pooling_model_name(orig_model_name: str, pooling_suf...; symbols: _get_pooling_model_name, _create_pooling_model_cls, ModelForPooling, __init__
  - `docs/models/pooling_models.md` modified +39/-14 (53 lines); hunks: -11,26 +11,51 @@ before returning them.
  - `vllm/model_executor/models/modernbert.py` modified +38/-12 (50 lines); hunks: -1,6 +1,6; -13,7 +13,8; symbols: __init__, get_pooling_updates, get_supported_tasks, _head
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/pooler.py
@@ -1,15 +1,16 @@
+from collections.abc import Mapping, Set
+from itertools import groupby
-from typing_extensions import assert_never
@@ -21,6 +22,10 @@
+PoolingFn = Callable[
+    [Union[torch.Tensor, list[torch.Tensor]], PoolingMetadata],
diff -- vllm/model_executor/models/bert.py
@@ -1,7 +1,7 @@
-from collections.abc import Iterable
+from collections.abc import Iterable, Set
@@ -17,7 +17,8 @@
-from vllm.model_executor.layers.pooler import (ClassifierPooler, Pooler,
+from vllm.model_executor.layers.pooler import (ClassifierPooler,
+                                               DispatchPooler, Pooler,
diff -- vllm/model_executor/models/adapters.py
@@ -13,7 +13,6 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/pooler.py` modified +175/-171; `vllm/model_executor/models/bert.py` modified +99/-33; `vllm/model_executor/models/adapters.py` modified +51/-57; `vllm/model_executor/models/modernbert.py` modified +38/-12; `vllm/model_executor/models/roberta.py` modified +27/-17; `vllm/model_executor/models/gritlm.py` modified +19/-20
  - docs: `docs/models/pooling_models.md` modified +39/-14
- 验证与风险: diff 自带测试面 `tests/models/test_transformers.py`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20996 - [Misc] unify variable for LLM instance

- 链接: https://github.com/vllm-project/vllm/pull/20996
- 状态/时间: merged / 2025-07-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 53 个文件，+237/-236，可读 patch 1417 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] unify variable for LLM instance」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `tests/detokenizer/test_stop_strings.py`, `tests/models/language/pooling/mteb_utils.py`, `tests/models/language/generation/test_mistral.py`；技术摘要: 覆盖「[Misc] unify variable for LLM instance」；主要实现面是 `tests/detokenizer/test_stop_strings.py`, `tests/models/language/pooling/mteb_utils.py`, `tests/models/language/generation/test_mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/detokenizer/test_stop_strings.py` modified +21/-21 (42 lines); hunks: -101,42 +101,42 @@ def _stop_token_id(llm):; symbols: _stop_token_id, test_stop_strings，涉及 `_stop_token_id, test_stop_strings`；`tests/models/language/pooling/mteb_utils.py` modified +9/-9 (18 lines); hunks: -30,7 +30,7 @@ class VllmMtebEncoder(mteb.Encoder):; -43,7 +43,7 @@ def encode(; symbols: VllmMtebEncoder, __init__, encode, predict，涉及 `VllmMtebEncoder, __init__, encode`；`tests/models/language/generation/test_mistral.py` modified +7/-7 (14 lines); hunks: -238,8 +238,8 @@ def test_mistral_symbolic_languages(vllm_runner, model: str,; -253,11 +253,11 @@ def test_mistral_function_calling(vllm_runner, model: str,...; symbols: test_mistral_symbolic_languages, test_mistral_function_calling, test_mistral_guided_decoding，涉及 `test_mistral_symbolic_languages, test_mistral_function_calling, test_mistral_guided_decoding`；`docs/models/pooling_models.md` modified +5/-5 (10 lines); hunks: -149,11 +149,11 @@ You can change the output dimensions of embedding models t...。
- 代码 diff 细节:
  - `tests/detokenizer/test_stop_strings.py` modified +21/-21 (42 lines); hunks: -101,42 +101,42 @@ def _stop_token_id(llm):; symbols: _stop_token_id, test_stop_strings
  - `tests/models/language/pooling/mteb_utils.py` modified +9/-9 (18 lines); hunks: -30,7 +30,7 @@ class VllmMtebEncoder(mteb.Encoder):; -43,7 +43,7 @@ def encode(; symbols: VllmMtebEncoder, __init__, encode, predict
  - `tests/models/language/generation/test_mistral.py` modified +7/-7 (14 lines); hunks: -238,8 +238,8 @@ def test_mistral_symbolic_languages(vllm_runner, model: str,; -253,11 +253,11 @@ def test_mistral_function_calling(vllm_runner, model: str,...; symbols: test_mistral_symbolic_languages, test_mistral_function_calling, test_mistral_guided_decoding
  - `docs/models/pooling_models.md` modified +5/-5 (10 lines); hunks: -149,11 +149,11 @@ You can change the output dimensions of embedding models t...
  - `tests/model_executor/test_model_load_with_params.py` modified +5/-5 (10 lines); hunks: -32,8 +32,8 @@ def test_model_loading_with_params(vllm_runner):; -70,8 +70,8 @@ def test_roberta_model_loading_with_params(vllm_runner):; symbols: test_model_loading_with_params, test_roberta_model_loading_with_params, test_facebook_roberta_model_loading_with_params, check_model
- 关键代码摘录:

```diff
diff -- tests/detokenizer/test_stop_strings.py
@@ -101,42 +101,42 @@ def _stop_token_id(llm):
-    vllm_model = LLM(MODEL, enforce_eager=envs.VLLM_USE_V1)
+    llm = LLM(MODEL, enforce_eager=envs.VLLM_USE_V1)
-        _stop_basic(vllm_model)
+        _stop_basic(llm)
-        _set_async_mode(vllm_model, True)
-        _stop_basic(vllm_model)
diff -- tests/models/language/pooling/mteb_utils.py
@@ -30,7 +30,7 @@ class VllmMtebEncoder(mteb.Encoder):
-        self.model = vllm_model
+        self.llm = vllm_model
@@ -43,7 +43,7 @@ def encode(
-        outputs = self.model.embed(sentences, use_tqdm=False)
+        outputs = self.llm.embed(sentences, use_tqdm=False)
@@ -61,10 +61,10 @@ def predict(
diff -- tests/models/language/generation/test_mistral.py
@@ -238,8 +238,8 @@ def test_mistral_symbolic_languages(vllm_runner, model: str,
```

- 已读文件:
  - tests: `tests/detokenizer/test_stop_strings.py` modified +21/-21; `tests/models/language/pooling/mteb_utils.py` modified +9/-9; `tests/models/language/generation/test_mistral.py` modified +7/-7; `tests/model_executor/test_model_load_with_params.py` modified +5/-5; `tests/models/language/pooling/test_nomic_max_model_len.py` modified +3/-3; `tests/models/language/pooling/test_truncation_control.py` modified +3/-3
  - docs: `docs/models/pooling_models.md` modified +5/-5
- 验证与风险: diff 自带测试面 `tests/basic_correctness/test_basic_correctness.py`, `tests/basic_correctness/test_preemption.py`, `tests/conftest.py`, `tests/core/test_num_computed_tokens_update.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21470 - [Deprecation][2/N] Replace `--task` with `--runner` and `--convert`

- 链接: https://github.com/vllm-project/vllm/pull/21470
- 状态/时间: merged / 2025-07-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 94 个文件，+1111/-1077，可读 patch 4435 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Deprecation][2/N] Replace `--task` with `--runner` and `--convert`」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/registry.py`, `vllm/model_executor/model_loader/utils.py`, `docs/models/supported_models.md`；技术摘要: 覆盖「[Deprecation][2/N] Replace `--task` with `--runner` and `--convert`」；主要实现面是 `vllm/model_executor/models/registry.py`, `vllm/model_executor/model_loader/utils.py`, `docs/models/supported_models.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/registry.py` modified +187/-62 (249 lines); hunks: -12,19 +12,24; -311,7 +316,7 @@ def from_model_cls(model: type[nn.Module]) -> "_ModelInfo":; symbols: from_model_cls, _raise_for_unsupported, _try_load_model_cls, _try_inspect_model_cls，涉及 `from_model_cls, _raise_for_unsupported, _try_load_model_cls`；`vllm/model_executor/model_loader/utils.py` modified +27/-104 (131 lines); hunks: -9,9 +9,8; -20,13 +19,10; symbols: device_loading_context, resolve_transformers_arch, is, won，涉及 `device_loading_context, resolve_transformers_arch, is`；`docs/models/supported_models.md` modified +51/-50 (101 lines); hunks: -1,7 +1,6; -24,7 +23,7 @@ To check if the modeling backend is Transformers, you can simp...; symbols: probabilities，涉及 `probabilities`；`vllm/entrypoints/llm.py` modified +37/-56 (93 lines); hunks: -20,8 +20,8; -170,7 +170,8 @@ def __init__(; symbols: __init__, generate, encode，涉及 `__init__, generate, encode`。
- 代码 diff 细节:
  - `vllm/model_executor/models/registry.py` modified +187/-62 (249 lines); hunks: -12,19 +12,24; -311,7 +316,7 @@ def from_model_cls(model: type[nn.Module]) -> "_ModelInfo":; symbols: from_model_cls, _raise_for_unsupported, _try_load_model_cls, _try_inspect_model_cls
  - `vllm/model_executor/model_loader/utils.py` modified +27/-104 (131 lines); hunks: -9,9 +9,8; -20,13 +19,10; symbols: device_loading_context, resolve_transformers_arch, is, won
  - `docs/models/supported_models.md` modified +51/-50 (101 lines); hunks: -1,7 +1,6; -24,7 +23,7 @@ To check if the modeling backend is Transformers, you can simp...; symbols: probabilities
  - `vllm/entrypoints/llm.py` modified +37/-56 (93 lines); hunks: -20,8 +20,8; -170,7 +170,8 @@ def __init__(; symbols: __init__, generate, encode
  - `docs/models/pooling_models.md` modified +49/-28 (77 lines); hunks: -1,28 +1,49; -31,32 +52,32 @@ In vLLM, we define the following pooling tasks and correspon...; symbols: provides
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/registry.py
@@ -12,19 +12,24 @@
-from dataclasses import asdict, dataclass, field
+from dataclasses import dataclass, field
+import transformers
+from vllm.config import (ModelConfig, ModelImpl, iter_architecture_defaults,
+                         try_match_architecture_defaults)
+from vllm.transformers_utils.dynamic_module import (
diff -- vllm/model_executor/model_loader/utils.py
@@ -9,9 +9,8 @@
-import transformers
-from transformers.dynamic_module_utils import get_class_from_dynamic_module
+from typing_extensions import assert_never
@@ -20,13 +19,10 @@
-from vllm.model_executor.models import ModelRegistry
-from vllm.model_executor.models.registry import (_PREVIOUSLY_SUPPORTED_MODELS,
diff -- docs/models/supported_models.md
@@ -1,7 +1,6 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/registry.py` modified +187/-62; `vllm/model_executor/model_loader/utils.py` modified +27/-104; `vllm/entrypoints/llm.py` modified +37/-56
  - docs: `docs/models/supported_models.md` modified +51/-50; `docs/models/pooling_models.md` modified +49/-28
  - tests: `tests/entrypoints/test_chat_utils.py` modified +9/-33; `tests/multimodal/test_processing.py` modified +1/-24; `tests/models/test_registry.py` modified +12/-9
- 验证与风险: diff 自带测试面 `tests/compile/test_async_tp.py`, `tests/compile/test_basic_correctness.py`, `tests/compile/test_fusion_all_reduce.py`, `tests/compile/test_sequence_parallelism.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20538 - [Model] Pooling model activation supports per request control by PoolingParams

- 链接: https://github.com/vllm-project/vllm/pull/20538
- 状态/时间: merged / 2025-08-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 21 个文件，+948/-173，可读 patch 1566 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Pooling model activation supports per request control by PoolingParams」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/layers/pooler.py`, `tests/models/language/pooling/test_override_pooler_config.py`, `tests/entrypoints/llm/test_score.py`；技术摘要: 覆盖「[Model] Pooling model activation supports per request control by PoolingParams」；主要实现面是 `vllm/model_executor/layers/pooler.py`, `tests/models/language/pooling/test_override_pooler_config.py`, `tests/entrypoints/llm/test_score.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/pooler.py` modified +106/-116 (222 lines); hunks: -41,35 +41,18 @@ class PoolingType(IntEnum):; -89,22 +72,15 @@ def for_encode(; symbols: PoolingType, ResolvedPoolingConfig, from_config_with_defaults, for_encode，涉及 `PoolingType, ResolvedPoolingConfig, from_config_with_defaults`；`tests/models/language/pooling/test_override_pooler_config.py` added +127/-0 (127 lines); hunks: -0,0 +1,127; symbols: test_classify_models_using_activation, test_embed_models_using_normalize, test_reward_models_using_softmax，涉及 `test_classify_models_using_activation, test_embed_models_using_normalize, test_reward_models_using_softmax`；`tests/entrypoints/llm/test_score.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: v1, llm, test_pooling_params, get_outputs，涉及 `v1, llm, test_pooling_params`；`tests/entrypoints/llm/test_classify.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: v1, llm, test_pooling_params, get_outputs，涉及 `v1, llm, test_pooling_params`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/pooler.py` modified +106/-116 (222 lines); hunks: -41,35 +41,18 @@ class PoolingType(IntEnum):; -89,22 +72,15 @@ def for_encode(; symbols: PoolingType, ResolvedPoolingConfig, from_config_with_defaults, for_encode
  - `tests/models/language/pooling/test_override_pooler_config.py` added +127/-0 (127 lines); hunks: -0,0 +1,127; symbols: test_classify_models_using_activation, test_embed_models_using_normalize, test_reward_models_using_softmax
  - `tests/entrypoints/llm/test_score.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: v1, llm, test_pooling_params, get_outputs
  - `tests/entrypoints/llm/test_classify.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: v1, llm, test_pooling_params, get_outputs
  - `tests/entrypoints/llm/test_reward.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: v1, llm, test_pooling_params, get_outputs
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/pooler.py
@@ -41,35 +41,18 @@ class PoolingType(IntEnum):
-    normalize: bool
-    softmax: bool
-    step_tag_id: Optional[int]
-    returned_token_ids: Optional[list[int]]
+    task: PoolingTask
+        task: PoolingTask,
diff -- tests/models/language/pooling/test_override_pooler_config.py
@@ -0,0 +1,127 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+import torch
+import torch.nn.functional as F
+from tests.models.utils import softmax
diff -- tests/entrypoints/llm/test_score.py
@@ -0,0 +1,69 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/pooler.py` modified +106/-116
  - tests: `tests/models/language/pooling/test_override_pooler_config.py` added +127/-0; `tests/entrypoints/llm/test_score.py` added +69/-0; `tests/entrypoints/llm/test_classify.py` added +67/-0; `tests/entrypoints/llm/test_reward.py` added +66/-0; `tests/entrypoints/llm/test_embedding.py` added +56/-0; `tests/entrypoints/openai/test_score.py` modified +41/-0
- 验证与风险: diff 自带测试面 `tests/entrypoints/llm/test_classify.py`, `tests/entrypoints/llm/test_embedding.py`, `tests/entrypoints/llm/test_reward.py`, `tests/entrypoints/llm/test_score.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24031 - [Model] Classification models support logit_bias / sigmoid_normalize

- 链接: https://github.com/vllm-project/vllm/pull/24031
- 状态/时间: merged / 2025-09-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+38/-30，可读 patch 143 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Classification models support logit_bias / sigmoid_normalize」；模型线: Jina Reranker M0；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/jina_vl.py`, `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/models/config.py`；技术摘要: 覆盖「[Model] Classification models support logit_bias / sigmoid_normalize」；主要实现面是 `vllm/model_executor/models/jina_vl.py`, `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/models/config.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/jina_vl.py` modified +3/-8 (11 lines); hunks: -92,17 +92,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -137,9 +134,7 @@ def forward(; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`；`vllm/model_executor/layers/pooler.py` modified +8/-0 (8 lines); hunks: -633,9 +633,14 @@ def __init__(; -654,6 +659,9 @@ def forward(; symbols: __init__, get_supported_tasks, forward，涉及 `__init__, get_supported_tasks, forward`；`vllm/model_executor/models/config.py` modified +3/-1 (4 lines); hunks: -210,8 +210,10 @@ class JinaVLForSequenceClassificationConfig(VerifyAndUpdate...; symbols: JinaVLForSequenceClassificationConfig, verify_and_update_config, SnowflakeGteNewModelConfig，涉及 `JinaVLForSequenceClassificationConfig, verify_and_update_config, SnowflakeGteNewModelConfig`；`vllm/config/__init__.py` modified +24/-21 (45 lines); hunks: -2651,24 +2651,46 @@ class PoolerConfig:; -2683,25 +2705,6 @@ class PoolerConfig:; symbols: PoolerConfig, compute_hash，涉及 `PoolerConfig, compute_hash`。
- 代码 diff 细节:
  - `vllm/model_executor/models/jina_vl.py` modified +3/-8 (11 lines); hunks: -92,17 +92,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -137,9 +134,7 @@ def forward(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/layers/pooler.py` modified +8/-0 (8 lines); hunks: -633,9 +633,14 @@ def __init__(; -654,6 +659,9 @@ def forward(; symbols: __init__, get_supported_tasks, forward
  - `vllm/model_executor/models/config.py` modified +3/-1 (4 lines); hunks: -210,8 +210,10 @@ class JinaVLForSequenceClassificationConfig(VerifyAndUpdate...; symbols: JinaVLForSequenceClassificationConfig, verify_and_update_config, SnowflakeGteNewModelConfig
  - `vllm/config/__init__.py` modified +24/-21 (45 lines); hunks: -2651,24 +2651,46 @@ class PoolerConfig:; -2683,25 +2705,6 @@ class PoolerConfig:; symbols: PoolerConfig, compute_hash
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/jina_vl.py
@@ -92,17 +92,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        # logit bias for sigmoid normalization
-        self.LOGIT_BIAS = 2.65
-            Pooler.for_classify(pooler_config, classifier=None),
+            Pooler.for_classify(pooler_config, classifier=self.score),
-            Pooler.for_classify(pooler_config, classifier=None),
+            Pooler.for_classify(pooler_config, classifier=self.score),
diff -- vllm/model_executor/layers/pooler.py
@@ -633,9 +633,14 @@ def __init__(
+        from vllm.config import get_current_vllm_config
+        vllm_config = get_current_vllm_config()
+        self.logit_bias: Optional[
+            float] = vllm_config.model_config.pooler_config.logit_bias
@@ -654,6 +659,9 @@ def forward(
+        if self.logit_bias is not None:
diff -- vllm/model_executor/models/config.py
@@ -210,8 +210,10 @@ class JinaVLForSequenceClassificationConfig(VerifyAndUpdateConfig):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/jina_vl.py` modified +3/-8; `vllm/model_executor/layers/pooler.py` modified +8/-0; `vllm/model_executor/models/config.py` modified +3/-1; `vllm/config/__init__.py` modified +24/-21
- 验证与风险: runtime 路径改动集中在 `vllm/config/__init__.py`, `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/models/config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23810 - [Model] Systematic support for fp32 head, pooling models part

- 链接: https://github.com/vllm-project/vllm/pull/23810
- 状态/时间: merged / 2025-09-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+166/-61，可读 patch 557 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Systematic support for fp32 head, pooling models part」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/layers/pooler.py`, `tests/models/language/pooling/mteb_utils.py`, `vllm/model_executor/models/internlm2.py`；技术摘要: 覆盖「[Model] Systematic support for fp32 head, pooling models part」；主要实现面是 `vllm/model_executor/layers/pooler.py`, `tests/models/language/pooling/mteb_utils.py`, `vllm/model_executor/models/internlm2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/pooler.py` modified +23/-15 (38 lines); hunks: -5,7 +5,7; -362,14 +362,13 @@ def forward_chunk(self, pooled_data: torch.Tensor) -> torc...; symbols: forward_chunk, PoolerNormalize, PoolerMultiLabelClassify，涉及 `forward_chunk, PoolerNormalize, PoolerMultiLabelClassify`；`tests/models/language/pooling/mteb_utils.py` modified +31/-6 (37 lines); hunks: -9,6 +9,7; -165,16 +166,19 @@ def mteb_test_embed_models(hf_runner,; symbols: mteb_test_embed_models, mteb_test_rerank_models，涉及 `mteb_test_embed_models, mteb_test_rerank_models`；`vllm/model_executor/models/internlm2.py` modified +11/-8 (19 lines); hunks: -423,13 +423,15 @@ def __init__(; -446,5 +448,6 @@ def forward(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/bert_with_rope.py` modified +8/-8 (16 lines); hunks: -637,14 +637,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/pooler.py` modified +23/-15 (38 lines); hunks: -5,7 +5,7; -362,14 +362,13 @@ def forward_chunk(self, pooled_data: torch.Tensor) -> torc...; symbols: forward_chunk, PoolerNormalize, PoolerMultiLabelClassify
  - `tests/models/language/pooling/mteb_utils.py` modified +31/-6 (37 lines); hunks: -9,6 +9,7; -165,16 +166,19 @@ def mteb_test_embed_models(hf_runner,; symbols: mteb_test_embed_models, mteb_test_rerank_models
  - `vllm/model_executor/models/internlm2.py` modified +11/-8 (19 lines); hunks: -423,13 +423,15 @@ def __init__(; -446,5 +448,6 @@ def forward(; symbols: __init__, forward
  - `vllm/model_executor/models/bert_with_rope.py` modified +8/-8 (16 lines); hunks: -637,14 +637,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/roberta.py` modified +11/-5 (16 lines); hunks: -8,7 +8,7; -73,10 +73,16 @@ def forward(; symbols: forward, RobertaClassificationHead, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/pooler.py
@@ -5,7 +5,7 @@
-from typing import Callable, Optional, TypeVar, Union, cast
+from typing import Callable, Optional, TypeVar, Union
@@ -362,14 +362,13 @@ def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
-        x = F.normalize(pooled_data.float(), p=2, dim=-1)
-        return x.to(pooled_data.dtype)
+        return F.normalize(pooled_data, p=2, dim=-1)
diff -- tests/models/language/pooling/mteb_utils.py
@@ -9,6 +9,7 @@
+import torch
@@ -165,16 +166,19 @@ def mteb_test_embed_models(hf_runner,
+    # A model family has many models with the same architecture,
+    # and we don't need to test each one.
-        # A model family has many models with the same architecture,
-        # and we don't need to test each one.
diff -- vllm/model_executor/models/internlm2.py
@@ -423,13 +423,15 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/pooler.py` modified +23/-15; `vllm/model_executor/models/internlm2.py` modified +11/-8; `vllm/model_executor/models/bert_with_rope.py` modified +8/-8; `vllm/model_executor/models/roberta.py` modified +11/-5; `vllm/model_executor/models/jina_vl.py` modified +8/-5; `vllm/model_executor/models/adapters.py` modified +4/-6
  - tests: `tests/models/language/pooling/mteb_utils.py` modified +31/-6
- 验证与风险: diff 自带测试面 `tests/models/language/pooling/mteb_utils.py`, `tests/models/language/pooling/test_bge_reranker_v2_gemma.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26247 - Convert formatting to use `ruff` instead of `yapf` + `isort`

- 链接: https://github.com/vllm-project/vllm/pull/26247
- 状态/时间: merged / 2025-10-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1508 个文件，+83935/-68959，可读 patch 272044 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Convert formatting to use `ruff` instead of `yapf` + `isort`」；模型线: Jina Reranker M0；类别: 性能/后端优化；主要 diff: `tests/entrypoints/test_chat_utils.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`；技术摘要: 覆盖「Convert formatting to use `ruff` instead of `yapf` + `isort`」；主要实现面是 `tests/entrypoints/test_chat_utils.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #26633 - Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y`

- 链接: https://github.com/vllm-project/vllm/pull/26633
- 状态/时间: merged / 2025-10-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 944 个文件，+9491/-10122，可读 patch 61484 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y`」；模型线: Jina Reranker M0；类别: 性能/后端优化；主要 diff: `vllm/entrypoints/openai/protocol.py`, `vllm/entrypoints/llm.py`, `vllm/model_executor/layers/fused_moe/layer.py`；技术摘要: 覆盖「Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y`」；主要实现面是 `vllm/entrypoints/openai/protocol.py`, `vllm/entrypoints/llm.py`, `vllm/model_executor/layers/fused_moe/layer.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #26687 - [Bugfix] Fix out of bound index issue for Jina-embedding-v3 RoPE with cuda graph

- 链接: https://github.com/vllm-project/vllm/pull/26687
- 状态/时间: merged / 2025-10-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/language/pooling_mteb_test/test_jina.py`；关联提交 `8e67b2557aae`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+13/-7，可读 patch 49 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix out of bound index issue for Jina-embedding-v3 RoPE with cuda graph」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `tests/models/language/pooling_mteb_test/test_jina.py`；技术摘要: 覆盖「[Bugfix] Fix out of bound index issue for Jina-embedding-v3 RoPE with cuda graph」；主要实现面是 `tests/models/language/pooling_mteb_test/test_jina.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/language/pooling_mteb_test/test_jina.py` modified +0/-4 (4 lines); hunks: -25,10 +25,6。
- 代码 diff 细节:
  - `tests/models/language/pooling_mteb_test/test_jina.py` modified +0/-4 (4 lines); hunks: -25,10 +25,6
- 关键代码摘录:

```diff
diff -- tests/models/language/pooling_mteb_test/test_jina.py
@@ -25,10 +25,6 @@
-        # The default max length of the model is 8194, which will crash
-        # CUDAGraph due to odd length for Gemm. We set it to 8192 to avoid
-        # avoid this issue.
-        max_model_len=8192,
```

- 已读文件:
  - tests: `tests/models/language/pooling_mteb_test/test_jina.py` modified +0/-4
- 验证与风险: diff 自带测试面 `tests/models/language/pooling_mteb_test/test_jina.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25370 - [Model][2/N] Improve all pooling task | Support multi-vector retrieval

- 链接: https://github.com/vllm-project/vllm/pull/25370
- 状态/时间: merged / 2025-10-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 41 个文件，+786/-399，可读 patch 1862 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][2/N] Improve all pooling task | Support multi-vector retrieval」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/layers/pooler.py`, `tests/models/language/pooling/test_pooler_config_init_behaviour.py`, `tests/models/language/pooling/test_multi_vector_retrieval.py`；技术摘要: 覆盖「[Model][2/N] Improve all pooling task | Support multi-vector retrieval」；主要实现面是 `vllm/model_executor/layers/pooler.py`, `tests/models/language/pooling/test_pooler_config_init_behaviour.py`, `tests/models/language/pooling/test_multi_vector_retrieval.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/pooler.py` modified +251/-171 (422 lines); hunks: -64,66 +64,6 @@ def apply(self, params: PoolingParams) -> None:; -237,7 +177,7 @@ def forward(; symbols: apply, Pooler, for_encode, for_embed，涉及 `apply, Pooler, for_encode`；`tests/models/language/pooling/test_pooler_config_init_behaviour.py` modified +50/-8 (58 lines); hunks: -93,7 +93,7 @@ def test_embed_models_using_normalize(; -104,22 +104,64 @@ def test_reward_models_using_softmax(; symbols: test_embed_models_using_normalize, test_reward_models_using_softmax, test_reward_models_using_activation, test_multi_vector_retrieval_models_using_normalize，涉及 `test_embed_models_using_normalize, test_reward_models_using_softmax, test_reward_models_using_activation`；`tests/models/language/pooling/test_multi_vector_retrieval.py` added +45/-0 (45 lines); hunks: -0,0 +1,45; symbols: test_embed_models，涉及 `test_embed_models`；`vllm/model_executor/models/adapters.py` modified +15/-27 (42 lines); hunks: -250,7 +250,7 @@ def _init_pooler(self, vllm_config: "VllmConfig", prefix: st...; -279,11 +279,8 @@ def as_seq_cls_model(cls: _T) -> _T:; symbols: _init_pooler, as_seq_cls_model, _classifier, forward，涉及 `_init_pooler, as_seq_cls_model, _classifier`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/pooler.py` modified +251/-171 (422 lines); hunks: -64,66 +64,6 @@ def apply(self, params: PoolingParams) -> None:; -237,7 +177,7 @@ def forward(; symbols: apply, Pooler, for_encode, for_embed
  - `tests/models/language/pooling/test_pooler_config_init_behaviour.py` modified +50/-8 (58 lines); hunks: -93,7 +93,7 @@ def test_embed_models_using_normalize(; -104,22 +104,64 @@ def test_reward_models_using_softmax(; symbols: test_embed_models_using_normalize, test_reward_models_using_softmax, test_reward_models_using_activation, test_multi_vector_retrieval_models_using_normalize
  - `tests/models/language/pooling/test_multi_vector_retrieval.py` added +45/-0 (45 lines); hunks: -0,0 +1,45; symbols: test_embed_models
  - `vllm/model_executor/models/adapters.py` modified +15/-27 (42 lines); hunks: -250,7 +250,7 @@ def _init_pooler(self, vllm_config: "VllmConfig", prefix: st...; -279,11 +279,8 @@ def as_seq_cls_model(cls: _T) -> _T:; symbols: _init_pooler, as_seq_cls_model, _classifier, forward
  - `vllm/entrypoints/llm.py` modified +18/-19 (37 lines); hunks: -951,7 +951,7 @@ def encode(; -986,25 +986,24 @@ def encode(; symbols: encode, reward, _embedding_score
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/pooler.py
@@ -64,66 +64,6 @@ def apply(self, params: PoolingParams) -> None:
-class Pooler(nn.Module, ABC):
-    """The interface required for all poolers used in pooling models in vLLM."""
-    @staticmethod
-    def for_encode(pooler_config: PoolerConfig):
-        if pooler_config.pooling_type == "STEP":
-            return StepPooler()
diff -- tests/models/language/pooling/test_pooler_config_init_behaviour.py
@@ -93,7 +93,7 @@ def test_embed_models_using_normalize(
-def test_reward_models_using_softmax(
+def test_reward_models_using_activation(
@@ -104,22 +104,64 @@ def test_reward_models_using_softmax(
-        pooler_config=PoolerConfig(softmax=False),
+        pooler_config=PoolerConfig(activation=False),
-        wo_softmax = vllm_model.encode(example_prompts)
diff -- tests/models/language/pooling/test_multi_vector_retrieval.py
@@ -0,0 +1,45 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/pooler.py` modified +251/-171; `vllm/model_executor/models/adapters.py` modified +15/-27; `vllm/entrypoints/llm.py` modified +18/-19; `vllm/model_executor/models/roberta.py` modified +6/-20; `vllm/model_executor/models/bert.py` modified +10/-12
  - tests: `tests/models/language/pooling/test_pooler_config_init_behaviour.py` modified +50/-8; `tests/models/language/pooling/test_multi_vector_retrieval.py` added +45/-0; `tests/entrypoints/pooling/llm/test_reward.py` modified +12/-11
- 验证与风险: diff 自带测试面 `tests/conftest.py`, `tests/entrypoints/pooling/llm/test_classify.py`, `tests/entrypoints/pooling/llm/test_embedding.py`, `tests/entrypoints/pooling/llm/test_encode.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29802 - Fix some Transformers nightly tests

- 链接: https://github.com/vllm-project/vllm/pull/29802
- 状态/时间: merged / 2025-12-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+29/-28，可读 patch 93 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix some Transformers nightly tests」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/jina_vl.py`, `vllm/model_executor/models/qwen2.py`；技术摘要: 覆盖「Fix some Transformers nightly tests」；主要实现面是 `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/jina_vl.py`, `vllm/model_executor/models/qwen2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/modernbert.py` modified +27/-26 (53 lines); hunks: -20,7 +20,7; -62,19 +62,6 @@ def forward(; symbols: forward, ModernBertRotaryEmbedding, __init__, ModernBertAttention，涉及 `forward, ModernBertRotaryEmbedding, __init__`；`vllm/model_executor/models/jina_vl.py` modified +1/-1 (2 lines); hunks: -29,7 +29,7; symbols: JinaVLScorer, __init__，涉及 `JinaVLScorer, __init__`；`vllm/model_executor/models/qwen2.py` modified +1/-1 (2 lines); hunks: -503,7 +503,7 @@ class Qwen2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP,...; symbols: Qwen2ForCausalLM, __init__，涉及 `Qwen2ForCausalLM, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/modernbert.py` modified +27/-26 (53 lines); hunks: -20,7 +20,7; -62,19 +62,6 @@ def forward(; symbols: forward, ModernBertRotaryEmbedding, __init__, ModernBertAttention
  - `vllm/model_executor/models/jina_vl.py` modified +1/-1 (2 lines); hunks: -29,7 +29,7; symbols: JinaVLScorer, __init__
  - `vllm/model_executor/models/qwen2.py` modified +1/-1 (2 lines); hunks: -503,7 +503,7 @@ class Qwen2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP,...; symbols: Qwen2ForCausalLM, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/modernbert.py
@@ -20,7 +20,7 @@
-from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
+from vllm.model_executor.layers.rotary_embedding import get_rope
@@ -62,19 +62,6 @@ def forward(
-class ModernBertRotaryEmbedding(RotaryEmbedding):
-    def __init__(self, config: ModernBertConfig, head_size: int, dim: int, base: float):
-        super().__init__(
diff -- vllm/model_executor/models/jina_vl.py
@@ -29,7 +29,7 @@
-        config = model_config.hf_config
+        config = model_config.hf_config.get_text_config()
diff -- vllm/model_executor/models/qwen2.py
@@ -503,7 +503,7 @@ class Qwen2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
-        config = vllm_config.model_config.hf_config
+        config = vllm_config.model_config.hf_config.get_text_config()
```

- 已读文件:
  - runtime: `vllm/model_executor/models/modernbert.py` modified +27/-26; `vllm/model_executor/models/jina_vl.py` modified +1/-1; `vllm/model_executor/models/qwen2.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/jina_vl.py`, `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/qwen2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31445 - [Bugfix][Frontend] Fix Jina reranker multimodal input compatibility

- 链接: https://github.com/vllm-project/vllm/pull/31445
- 状态/时间: merged / 2025-12-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/pooling/test_jinavl_reranker.py`；关联提交 `bf73a3e4d7e1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+316/-138，可读 patch 519 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Frontend] Fix Jina reranker multimodal input compatibility」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `tests/models/multimodal/pooling/test_jinavl_reranker.py`；技术摘要: 覆盖「[Bugfix][Frontend] Fix Jina reranker multimodal input compatibility」；主要实现面是 `tests/models/multimodal/pooling/test_jinavl_reranker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +313/-137 (450 lines); hunks: -1,194 +1,370; symbols: vllm_reranker, _normalize_image, create_score_multimodal_param, _run_vllm，涉及 `vllm_reranker, _normalize_image, create_score_multimodal_param`。
- 代码 diff 细节:
  - `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +313/-137 (450 lines); hunks: -1,194 +1,370; symbols: vllm_reranker, _normalize_image, create_score_multimodal_param, _run_vllm
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/pooling/test_jinavl_reranker.py
@@ -1,194 +1,370 @@
+from typing import cast
-from vllm.entrypoints.chat_utils import ChatCompletionContentPartImageParam
+from vllm.entrypoints.chat_utils import (
+    ChatCompletionContentPartImageEmbedsParam,
+    ChatCompletionContentPartImageParam,
+    ChatCompletionContentPartTextParam,
```

- 已读文件:
  - tests: `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +313/-137
- 验证与风险: diff 自带测试面 `tests/models/multimodal/pooling/test_jinavl_reranker.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31669 - [Misc][Model][Refactor] Pass the prefix into Linear layers

- 链接: https://github.com/vllm-project/vllm/pull/31669
- 状态/时间: merged / 2026-01-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+181/-40，可读 patch 753 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc][Model][Refactor] Pass the prefix into Linear layers」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/molmo.py`, `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/qwen_vl.py`；技术摘要: 覆盖「[Misc][Model][Refactor] Pass the prefix into Linear layers」；主要实现面是 `vllm/model_executor/models/molmo.py`, `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/qwen_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/molmo.py` modified +50/-9 (59 lines); hunks: -142,13 +142,15 @@ def __init__(; -158,6 +160,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/modernbert.py` modified +25/-8 (33 lines); hunks: -63,7 +63,9 @@ def forward(; -80,6 +82,7 @@ def __init__(self, config: ModernBertConfig, layer_id: int | N...; symbols: forward, ModernBertAttention, __init__，涉及 `forward, ModernBertAttention, __init__`；`vllm/model_executor/models/qwen_vl.py` modified +26/-6 (32 lines); hunks: -109,6 +109,7 @@ def __init__(; -128,8 +129,12 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/aria.py` modified +15/-5 (20 lines); hunks: -127,11 +127,16 @@ def __init__(; -154,7 +159,7 @@ class AriaProjector(nn.Module):; symbols: __init__, forward, AriaProjector，涉及 `__init__, forward, AriaProjector`。
- 代码 diff 细节:
  - `vllm/model_executor/models/molmo.py` modified +50/-9 (59 lines); hunks: -142,13 +142,15 @@ def __init__(; -158,6 +160,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/modernbert.py` modified +25/-8 (33 lines); hunks: -63,7 +63,9 @@ def forward(; -80,6 +82,7 @@ def __init__(self, config: ModernBertConfig, layer_id: int | N...; symbols: forward, ModernBertAttention, __init__
  - `vllm/model_executor/models/qwen_vl.py` modified +26/-6 (32 lines); hunks: -109,6 +109,7 @@ def __init__(; -128,8 +129,12 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/aria.py` modified +15/-5 (20 lines); hunks: -127,11 +127,16 @@ def __init__(; -154,7 +159,7 @@ class AriaProjector(nn.Module):; symbols: __init__, forward, AriaProjector
  - `vllm/model_executor/models/jina_vl.py` modified +14/-4 (18 lines); hunks: -27,15 +27,23; -94,7 +102,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: JinaVLScorer, __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/molmo.py
@@ -142,13 +142,15 @@ def __init__(
+        prefix: str = "",
+            prefix=f"{prefix}.w1",
@@ -158,6 +160,7 @@ def __init__(
+            prefix=f"{prefix}.w2",
@@ -176,6 +179,7 @@ def __init__(
+        prefix: str = "",
diff -- vllm/model_executor/models/modernbert.py
@@ -63,7 +63,9 @@ def forward(
-    def __init__(self, config: ModernBertConfig, layer_id: int | None = None):
+    def __init__(
+        self, config: ModernBertConfig, layer_id: int | None = None, prefix: str = ""
+    ):
@@ -80,6 +82,7 @@ def __init__(self, config: ModernBertConfig, layer_id: int | None = None):
+            prefix=f"{prefix}.Wqkv",
diff -- vllm/model_executor/models/qwen_vl.py
@@ -109,6 +109,7 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/molmo.py` modified +50/-9; `vllm/model_executor/models/modernbert.py` modified +25/-8; `vllm/model_executor/models/qwen_vl.py` modified +26/-6; `vllm/model_executor/models/aria.py` modified +15/-5; `vllm/model_executor/models/jina_vl.py` modified +14/-4; `vllm/model_executor/models/qwen.py` modified +15/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mamba/mamba_mixer.py`, `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/gpt_neox.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31890 - [Models] Allow converting Qwen3-VL into Reranker model

- 链接: https://github.com/vllm-project/vllm/pull/31890
- 状态/时间: merged / 2026-01-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/pooling/score/template/qwen3_vl_reranker.jinja`；关联提交 `eac3b96ec04d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+287/-13，可读 patch 415 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models] Allow converting Qwen3-VL into Reranker model」；模型线: Jina Reranker M0；类别: 模型实现调整；主要 diff: `examples/pooling/score/template/qwen3_vl_reranker.jinja`；技术摘要: 覆盖「[Models] Allow converting Qwen3-VL into Reranker model」；主要实现面是 `examples/pooling/score/template/qwen3_vl_reranker.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/pooling/score/template/qwen3_vl_reranker.jinja` added +23/-0 (23 lines); hunks: -0,0 +1,23。
- 代码 diff 细节:
  - `examples/pooling/score/template/qwen3_vl_reranker.jinja` added +23/-0 (23 lines); hunks: -0,0 +1,23
- 关键代码摘录:

```diff
diff -- examples/pooling/score/template/qwen3_vl_reranker.jinja
@@ -0,0 +1,23 @@
+<|im_start|>system
+Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
+<|im_start|>user
+<Instruct>: {{
+    messages
+    | selectattr("role", "eq", "system")
```

- 已读文件:
  - docs: `examples/pooling/score/template/qwen3_vl_reranker.jinja` added +23/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31973 - [Model] Reorganize pooling layers

- 链接: https://github.com/vllm-project/vllm/pull/31973
- 状态/时间: merged / 2026-01-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 34 个文件，+1290/-1143，可读 patch 2875 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Reorganize pooling layers」；模型线: Jina Reranker M0；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/layers/pooler/activations.py`, `vllm/model_executor/layers/pooler/seqwise/heads.py`；技术摘要: 覆盖「[Model] Reorganize pooling layers」；主要实现面是 `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/layers/pooler/activations.py`, `vllm/model_executor/layers/pooler/seqwise/heads.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/pooler.py` removed +0/-845 (845 lines); hunks: -1,845 +0,0; symbols: ResolvedPoolingConfig, from_config, PoolingParamsUpdate, apply，涉及 `ResolvedPoolingConfig, from_config, PoolingParamsUpdate`；`vllm/model_executor/layers/pooler/activations.py` added +162/-0 (162 lines); hunks: -0,0 +1,162; symbols: get_classification_act_fn, get_cross_encoder_act_fn, resolve_classifier_act_fn, PoolerActivation，涉及 `get_classification_act_fn, get_cross_encoder_act_fn, resolve_classifier_act_fn`；`vllm/model_executor/layers/pooler/seqwise/heads.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: SequencePoolerHead, get_supported_tasks, forward, EmbeddingPoolerHead，涉及 `SequencePoolerHead, get_supported_tasks, forward`；`vllm/model_executor/layers/pooler/tokwise/heads.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: TokenPoolerHead, get_supported_tasks, forward_chunk, forward，涉及 `TokenPoolerHead, get_supported_tasks, forward_chunk`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/pooler.py` removed +0/-845 (845 lines); hunks: -1,845 +0,0; symbols: ResolvedPoolingConfig, from_config, PoolingParamsUpdate, apply
  - `vllm/model_executor/layers/pooler/activations.py` added +162/-0 (162 lines); hunks: -0,0 +1,162; symbols: get_classification_act_fn, get_cross_encoder_act_fn, resolve_classifier_act_fn, PoolerActivation
  - `vllm/model_executor/layers/pooler/seqwise/heads.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: SequencePoolerHead, get_supported_tasks, forward, EmbeddingPoolerHead
  - `vllm/model_executor/layers/pooler/tokwise/heads.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: TokenPoolerHead, get_supported_tasks, forward_chunk, forward
  - `vllm/model_executor/layers/pooler/special.py` added +128/-0 (128 lines); hunks: -0,0 +1,128; symbols: DispatchPooler, for_embedding, for_seq_cls, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/pooler.py
@@ -1,845 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-from abc import ABC, abstractmethod
-from collections.abc import Callable, Mapping, Set
-from dataclasses import dataclass
-from itertools import groupby
diff -- vllm/model_executor/layers/pooler/activations.py
@@ -0,0 +1,162 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from abc import ABC, abstractmethod
+from collections.abc import Callable
+from typing import TypeVar
+import torch
diff -- vllm/model_executor/layers/pooler/seqwise/heads.py
@@ -0,0 +1,157 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/pooler.py` removed +0/-845; `vllm/model_executor/layers/pooler/activations.py` added +162/-0; `vllm/model_executor/layers/pooler/seqwise/heads.py` added +157/-0; `vllm/model_executor/layers/pooler/tokwise/heads.py` added +142/-0; `vllm/model_executor/layers/pooler/special.py` added +128/-0; `vllm/model_executor/layers/pooler/tokwise/methods.py` added +124/-0
- 验证与风险: diff 自带测试面 `tests/model_executor/test_model_load_with_params.py`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32085 - [Model] Improve multimodal pooling examples

- 链接: https://github.com/vllm-project/vllm/pull/32085
- 状态/时间: merged / 2026-01-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+381/-69，可读 patch 538 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Improve multimodal pooling examples」；模型线: Jina Reranker M0；类别: 模型实现调整；主要 diff: `examples/pooling/score/openai_cross_encoder_score_for_multimodal.py`, `examples/pooling/embed/vision_embedding_online.py`, `examples/pooling/embed/vision_embedding_offline.py`；技术摘要: 覆盖「[Model] Improve multimodal pooling examples」；主要实现面是 `examples/pooling/score/openai_cross_encoder_score_for_multimodal.py`, `examples/pooling/embed/vision_embedding_online.py`, `examples/pooling/embed/vision_embedding_offline.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/pooling/score/openai_cross_encoder_score_for_multimodal.py` removed +0/-60 (60 lines); hunks: -1,60 +0,0; symbols: post_http_request, parse_args, main，涉及 `post_http_request, parse_args, main`；`examples/pooling/embed/vision_embedding_online.py` renamed +130/-5 (135 lines); hunks: -21,7 +21,8; -30,6 +31,8 @@ def create_chat_embeddings(; symbols: create_chat_embeddings, print_embeddings, run_clip，涉及 `create_chat_embeddings, print_embeddings, run_clip`；`examples/pooling/embed/vision_embedding_offline.py` added +93/-0 (93 lines); hunks: -0,0 +1,93; symbols: print_embeddings, run_qwen3_vl, parse_args, main，涉及 `print_embeddings, run_qwen3_vl, parse_args`；`examples/pooling/score/vision_rerank_api_online.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: parse_args, main，涉及 `parse_args, main`。
- 代码 diff 细节:
  - `examples/pooling/score/openai_cross_encoder_score_for_multimodal.py` removed +0/-60 (60 lines); hunks: -1,60 +0,0; symbols: post_http_request, parse_args, main
  - `examples/pooling/embed/vision_embedding_online.py` renamed +130/-5 (135 lines); hunks: -21,7 +21,8; -30,6 +31,8 @@ def create_chat_embeddings(; symbols: create_chat_embeddings, print_embeddings, run_clip
  - `examples/pooling/embed/vision_embedding_offline.py` added +93/-0 (93 lines); hunks: -0,0 +1,93; symbols: print_embeddings, run_qwen3_vl, parse_args, main
  - `examples/pooling/score/vision_rerank_api_online.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: parse_args, main
  - `examples/pooling/score/vision_score_api_online.py` added +71/-0 (71 lines); hunks: -0,0 +1,71; symbols: parse_args, main
- 关键代码摘录:

```diff
diff -- examples/pooling/score/openai_cross_encoder_score_for_multimodal.py
@@ -1,60 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-"""
-Example online usage of Score API.
-Run `vllm serve <model> --runner pooling` to start up the server in vLLM.
-"""
diff -- examples/pooling/embed/vision_embedding_online.py
@@ -21,7 +21,8 @@
-image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
+image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
+text = "A cat standing in the snow."
@@ -30,6 +31,8 @@ def create_chat_embeddings(
+    continue_final_message: bool = False,
+    add_special_tokens: bool = False,
diff -- examples/pooling/embed/vision_embedding_offline.py
@@ -0,0 +1,93 @@
```

- 已读文件:
  - docs: `examples/pooling/score/openai_cross_encoder_score_for_multimodal.py` removed +0/-60; `examples/pooling/embed/vision_embedding_online.py` renamed +130/-5; `examples/pooling/embed/vision_embedding_offline.py` added +93/-0; `examples/pooling/score/vision_rerank_api_online.py` added +80/-0; `examples/pooling/score/vision_score_api_online.py` added +71/-0; `docs/serving/openai_compatible_server.md` modified +7/-4
- 验证与风险: 该 PR 主要落在文档/示例 `docs/serving/openai_compatible_server.md`, `examples/pooling/embed/vision_embedding_offline.py`, `examples/pooling/embed/vision_embedding_online.py`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #32395 - [Frontend][1/n] Make pooling entrypoints request schema consensus | CompletionRequest

- 链接: https://github.com/vllm-project/vllm/pull/32395
- 状态/时间: merged / 2026-01-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 22 个文件，+629/-594，可读 patch 1838 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend][1/n] Make pooling entrypoints request schema consensus | CompletionRequest」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `tests/entrypoints/pooling/embed/test_online.py`, `tests/entrypoints/pooling/classify/test_online.py`, `tests/entrypoints/pooling/pooling/test_online.py`；技术摘要: 覆盖「[Frontend][1/n] Make pooling entrypoints request schema consensus | CompletionRequest」；主要实现面是 `tests/entrypoints/pooling/embed/test_online.py`, `tests/entrypoints/pooling/classify/test_online.py`, `tests/entrypoints/pooling/pooling/test_online.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/entrypoints/pooling/embed/test_online.py` modified +233/-221 (454 lines); hunks: -31,7 +31,26; -79,15 +98,36 @@ def hf_model(hf_runner):; symbols: hf_model, test_single_embedding, test_basic, test_completion_request，涉及 `hf_model, test_single_embedding, test_basic`；`tests/entrypoints/pooling/classify/test_online.py` modified +86/-49 (135 lines); hunks: -12,6 +12,8; -29,9 +31,23 @@ def server():; symbols: server, test_single_input_classification, test_basic, test_completion_request，涉及 `server, test_single_input_classification, test_basic`；`tests/entrypoints/pooling/pooling/test_online.py` modified +49/-57 (106 lines); hunks: -24,6 +24,8; -46,30 +48,40 @@ def server():; symbols: server, test_single_pooling, test_basic, test_completion_request，涉及 `server, test_single_pooling, test_basic`；`vllm/entrypoints/pooling/__init__.py` modified +88/-0 (88 lines); hunks: -1,8 +1,17; -14,3 +23,82 @@ def register_pooling_api_routers(app: FastAPI):; symbols: register_pooling_api_routers, init_pooling_state，涉及 `register_pooling_api_routers, init_pooling_state`。
- 代码 diff 细节:
  - `tests/entrypoints/pooling/embed/test_online.py` modified +233/-221 (454 lines); hunks: -31,7 +31,26; -79,15 +98,36 @@ def hf_model(hf_runner):; symbols: hf_model, test_single_embedding, test_basic, test_completion_request
  - `tests/entrypoints/pooling/classify/test_online.py` modified +86/-49 (135 lines); hunks: -12,6 +12,8; -29,9 +31,23 @@ def server():; symbols: server, test_single_input_classification, test_basic, test_completion_request
  - `tests/entrypoints/pooling/pooling/test_online.py` modified +49/-57 (106 lines); hunks: -24,6 +24,8; -46,30 +48,40 @@ def server():; symbols: server, test_single_pooling, test_basic, test_completion_request
  - `vllm/entrypoints/pooling/__init__.py` modified +88/-0 (88 lines); hunks: -1,8 +1,17; -14,3 +23,82 @@ def register_pooling_api_routers(app: FastAPI):; symbols: register_pooling_api_routers, init_pooling_state
  - `vllm/entrypoints/openai/api_server.py` modified +4/-58 (62 lines); hunks: -54,10 +54,6; -73,7 +69,6; symbols: init_app_state
- 关键代码摘录:

```diff
diff -- tests/entrypoints/pooling/embed/test_online.py
@@ -31,7 +31,26 @@
+input_text = "The best thing about vLLM is that it supports many different models"
+input_tokens = [
+    0,
+    581,
+    2965,
+    13580,
diff -- tests/entrypoints/pooling/classify/test_online.py
@@ -12,6 +12,8 @@
+input_text = "This product was excellent and exceeded my expectations"
+input_tokens = [1986, 1985, 572, 9073, 323, 33808, 847, 16665]
@@ -29,9 +31,23 @@ def server():
-def test_single_input_classification(server: RemoteOpenAIServer, model_name: str):
-    input_text = "This product was excellent and exceeded my expectations"
+def test_basic(server: RemoteOpenAIServer, model_name: str):
diff -- tests/entrypoints/pooling/pooling/test_online.py
@@ -24,6 +24,8 @@
```

- 已读文件:
  - tests: `tests/entrypoints/pooling/embed/test_online.py` modified +233/-221; `tests/entrypoints/pooling/classify/test_online.py` modified +86/-49; `tests/entrypoints/pooling/pooling/test_online.py` modified +49/-57
  - runtime: `vllm/entrypoints/pooling/__init__.py` modified +88/-0; `vllm/entrypoints/openai/api_server.py` modified +4/-58; `vllm/entrypoints/pooling/embed/protocol.py` modified +9/-50; `vllm/entrypoints/pooling/classify/protocol.py` modified +7/-51; `vllm/entrypoints/pooling/base/protocol.py` added +46/-0
- 验证与风险: diff 自带测试面 `tests/entrypoints/pooling/classify/test_online.py`, `tests/entrypoints/pooling/embed/test_online.py`, `tests/entrypoints/pooling/pooling/test_online.py`, `tests/entrypoints/pooling/score/test_online_rerank.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32577 - [Frontend] Score entrypoint support data_1 & data_2 and queries & documents as inputs

- 链接: https://github.com/vllm-project/vllm/pull/32577
- 状态/时间: merged / 2026-01-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+253/-113，可读 patch 749 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend] Score entrypoint support data_1 & data_2 and queries & documents as inputs」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `tests/entrypoints/pooling/score/test_online_score.py`, `vllm/entrypoints/pooling/score/protocol.py`, `vllm/entrypoints/pooling/score/serving.py`；技术摘要: 覆盖「[Frontend] Score entrypoint support data_1 & data_2 and queries & documents as inputs」；主要实现面是 `tests/entrypoints/pooling/score/test_online_score.py`, `vllm/entrypoints/pooling/score/protocol.py`, `vllm/entrypoints/pooling/score/serving.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/entrypoints/pooling/score/test_online_score.py` modified +95/-37 (132 lines); hunks: -61,14 +61,40 @@ def runner(model: dict[str, Any], hf_runner):; -83,24 +109,50 @@ def test_text_1_str_text_2_list(; symbols: runner, TestModel, test_text_1_str_text_2_list, test_queries_str_documents_str，涉及 `runner, TestModel, test_text_1_str_text_2_list`；`vllm/entrypoints/pooling/score/protocol.py` modified +38/-5 (43 lines); hunks: -1,7 +1,7; -19,10 +19,7; symbols: ScoreRequest, ScoreRequestMixin, to_pooling_params, ScoreDataRequest，涉及 `ScoreRequest, ScoreRequestMixin, to_pooling_params`；`vllm/entrypoints/pooling/score/serving.py` modified +16/-16 (32 lines); hunks: -66,15 +66,15 @@ def __init__(; -135,22 +135,22 @@ async def _embedding_score(; symbols: __init__, _embedding_score, _run_scoring, create_score，涉及 `__init__, _embedding_score, _run_scoring`；`vllm/entrypoints/openai/engine/serving.py` modified +9/-2 (11 lines); hunks: -84,8 +84,11; -1032,7 +1035,9 @@ def _validate_input(; symbols: _validate_input，涉及 `_validate_input`。
- 代码 diff 细节:
  - `tests/entrypoints/pooling/score/test_online_score.py` modified +95/-37 (132 lines); hunks: -61,14 +61,40 @@ def runner(model: dict[str, Any], hf_runner):; -83,24 +109,50 @@ def test_text_1_str_text_2_list(; symbols: runner, TestModel, test_text_1_str_text_2_list, test_queries_str_documents_str
  - `vllm/entrypoints/pooling/score/protocol.py` modified +38/-5 (43 lines); hunks: -1,7 +1,7; -19,10 +19,7; symbols: ScoreRequest, ScoreRequestMixin, to_pooling_params, ScoreDataRequest
  - `vllm/entrypoints/pooling/score/serving.py` modified +16/-16 (32 lines); hunks: -66,15 +66,15 @@ def __init__(; -135,22 +135,22 @@ async def _embedding_score(; symbols: __init__, _embedding_score, _run_scoring, create_score
  - `vllm/entrypoints/openai/engine/serving.py` modified +9/-2 (11 lines); hunks: -84,8 +84,11; -1032,7 +1035,9 @@ def _validate_input(; symbols: _validate_input
  - `tests/entrypoints/pooling/score/test_offline.py` modified +4/-4 (8 lines); hunks: -43,12 +43,12 @@ def llm():; symbols: llm, test_pooling_params, get_outputs
- 关键代码摘录:

```diff
diff -- tests/entrypoints/pooling/score/test_online_score.py
@@ -61,14 +61,40 @@ def runner(model: dict[str, Any], hf_runner):
-    def test_text_1_str_text_2_list(
+    def test_queries_str_documents_str(
+        self, server: RemoteOpenAIServer, model: dict[str, Any], runner
+    ):
+        queries = "What is the capital of France?"
+        documents = "The capital of France is Paris."
diff -- vllm/entrypoints/pooling/score/protocol.py
@@ -1,7 +1,7 @@
-from typing import Any
+from typing import Any, TypeAlias
@@ -19,10 +19,7 @@
-class ScoreRequest(PoolingBasicRequestMixin):
-    text_1: list[str] | str | ScoreMultiModalParam
-    text_2: list[str] | str | ScoreMultiModalParam
diff -- vllm/entrypoints/pooling/score/serving.py
@@ -66,15 +66,15 @@ def __init__(
```

- 已读文件:
  - tests: `tests/entrypoints/pooling/score/test_online_score.py` modified +95/-37; `tests/entrypoints/pooling/score/test_offline.py` modified +4/-4; `tests/entrypoints/openai/test_run_batch.py` modified +2/-2; `tests/entrypoints/pooling/classify/test_online.py` modified +2/-2; `tests/models/language/pooling_mteb_test/mteb_score_utils.py` modified +2/-2
  - runtime: `vllm/entrypoints/pooling/score/protocol.py` modified +38/-5; `vllm/entrypoints/pooling/score/serving.py` modified +16/-16; `vllm/entrypoints/openai/engine/serving.py` modified +9/-2
- 验证与风险: diff 自带测试面 `tests/entrypoints/openai/test_run_batch.py`, `tests/entrypoints/pooling/classify/test_online.py`, `tests/entrypoints/pooling/score/test_offline.py`, `tests/entrypoints/pooling/score/test_online_score.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32287 - Upgrade transformers-4.57.5

- 链接: https://github.com/vllm-project/vllm/pull/32287
- 状态/时间: merged / 2026-01-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+25/-3，可读 patch 91 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Upgrade transformers-4.57.5」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `tests/models/multimodal/pooling/test_jinavl_reranker.py`, `requirements/nightly_torch_test.txt`, `requirements/test.in`；技术摘要: 覆盖「Upgrade transformers-4.57.5」；主要实现面是 `tests/models/multimodal/pooling/test_jinavl_reranker.py`, `requirements/nightly_torch_test.txt`, `requirements/test.in`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +22/-0 (22 lines); hunks: -3,6 +3,8; -277,6 +279,10 @@ def _run_test(; symbols: _run_test, test_model_text_image, test_model_text_text，涉及 `_run_test, test_model_text_image, test_model_text_text`；`requirements/nightly_torch_test.txt` modified +1/-1 (2 lines); hunks: -29,7 +29,7 @@ opencv-python-headless >= 4.11.0 # required for video test；`requirements/test.in` modified +1/-1 (2 lines); hunks: -37,7 +37,7 @@ opencv-python-headless >= 4.11.0 # required for video test；`requirements/test.txt` modified +1/-1 (2 lines); hunks: -1214,7 +1214,7 @@ tqdm==4.66.6。
- 代码 diff 细节:
  - `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +22/-0 (22 lines); hunks: -3,6 +3,8; -277,6 +279,10 @@ def _run_test(; symbols: _run_test, test_model_text_image, test_model_text_text
  - `requirements/nightly_torch_test.txt` modified +1/-1 (2 lines); hunks: -29,7 +29,7 @@ opencv-python-headless >= 4.11.0 # required for video test
  - `requirements/test.in` modified +1/-1 (2 lines); hunks: -37,7 +37,7 @@ opencv-python-headless >= 4.11.0 # required for video test
  - `requirements/test.txt` modified +1/-1 (2 lines); hunks: -1214,7 +1214,7 @@ tqdm==4.66.6
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/pooling/test_jinavl_reranker.py
@@ -3,6 +3,8 @@
+import transformers
+from packaging import version
@@ -277,6 +279,10 @@ def _run_test(
+@pytest.mark.skipif(
+    version.parse(transformers.__version__) == version.parse("4.57.5"),
+    reason="Skipped for transformers==4.57.5, https://github.com/huggingface/transformers/issues/43295",
diff -- requirements/nightly_torch_test.txt
@@ -29,7 +29,7 @@ opencv-python-headless >= 4.11.0 # required for video test
-transformers==4.57.3
+transformers==4.57.5
diff -- requirements/test.in
@@ -37,7 +37,7 @@ opencv-python-headless >= 4.11.0 # required for video test
-transformers==4.57.3
+transformers==4.57.5
diff -- requirements/test.txt
@@ -1214,7 +1214,7 @@ tqdm==4.66.6
```

- 已读文件:
  - tests: `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +22/-0
  - other: `requirements/nightly_torch_test.txt` modified +1/-1; `requirements/test.in` modified +1/-1; `requirements/test.txt` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/multimodal/pooling/test_jinavl_reranker.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33063 - [Chore] Update type annotation of `input_ids` in model forward

- 链接: https://github.com/vllm-project/vllm/pull/33063
- 状态/时间: merged / 2026-01-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 164 个文件，+243/-241，可读 patch 2158 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Chore] Update type annotation of `input_ids` in model forward」；模型线: Jina Reranker M0；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py`；技术摘要: 覆盖「[Chore] Update type annotation of `input_ids` in model forward」；主要实现面是 `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #33298 - [Bugfix] Fix Qwen3-VL-Reranker load.

- 链接: https://github.com/vllm-project/vllm/pull/33298
- 状态/时间: merged / 2026-01-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+234/-112，可读 patch 457 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen3-VL-Reranker load.」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `tests/entrypoints/pooling/score/test_online_score_vision.py`, `vllm/model_executor/models/adapters.py`, `tests/entrypoints/test_utils.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen3-VL-Reranker load.」；主要实现面是 `tests/entrypoints/pooling/score/test_online_score_vision.py`, `vllm/model_executor/models/adapters.py`, `tests/entrypoints/test_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/entrypoints/pooling/score/test_online_score_vision.py` added +122/-0 (122 lines); hunks: -0,0 +1,122; symbols: server, test_score_api_queries_str_documents_str, test_score_api_queries_str_documents_text_content, test_score_api_queries_str_documents_image_url_content，涉及 `server, test_score_api_queries_str_documents_str, test_score_api_queries_str_documents_text_content`；`vllm/model_executor/models/adapters.py` modified +10/-4 (14 lines); hunks: -466,6 +466,7 @@ def load_weights_using_from_2_way_softmax(; -506,14 +507,16 @@ def load_weights_using_from_2_way_softmax(; symbols: load_weights_using_from_2_way_softmax, load_weights_no_post_processing，涉及 `load_weights_using_from_2_way_softmax, load_weights_no_post_processing`；`tests/entrypoints/test_utils.py` modified +0/-12 (12 lines); hunks: -1,9 +1,5; -12,11 +8,3 @@ def test_sanitize_message():; symbols: test_sanitize_message, encode_base64_content_from_url，涉及 `test_sanitize_message, encode_base64_content_from_url`；`tests/entrypoints/pooling/classify/test_online_vision.py` modified +2/-2 (4 lines); hunks: -5,9 +5,9; -19,7 +19,7。
- 代码 diff 细节:
  - `tests/entrypoints/pooling/score/test_online_score_vision.py` added +122/-0 (122 lines); hunks: -0,0 +1,122; symbols: server, test_score_api_queries_str_documents_str, test_score_api_queries_str_documents_text_content, test_score_api_queries_str_documents_image_url_content
  - `vllm/model_executor/models/adapters.py` modified +10/-4 (14 lines); hunks: -466,6 +466,7 @@ def load_weights_using_from_2_way_softmax(; -506,14 +507,16 @@ def load_weights_using_from_2_way_softmax(; symbols: load_weights_using_from_2_way_softmax, load_weights_no_post_processing
  - `tests/entrypoints/test_utils.py` modified +0/-12 (12 lines); hunks: -1,9 +1,5; -12,11 +8,3 @@ def test_sanitize_message():; symbols: test_sanitize_message, encode_base64_content_from_url
  - `tests/entrypoints/pooling/classify/test_online_vision.py` modified +2/-2 (4 lines); hunks: -5,9 +5,9; -19,7 +19,7
  - `examples/pooling/score/vision_score_api_online.py` modified +54/-45 (99 lines); hunks: -17,48 +17,32; -73,15 +57,40 @@ def main(args):; symbols: encode_base64_content_from_url, parse_args, main
- 关键代码摘录:

```diff
diff -- tests/entrypoints/pooling/score/test_online_score_vision.py
@@ -0,0 +1,122 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+import requests
+from tests.utils import VLLM_PATH, RemoteOpenAIServer
+from vllm.entrypoints.pooling.score.protocol import ScoreResponse
diff -- vllm/model_executor/models/adapters.py
@@ -466,6 +466,7 @@ def load_weights_using_from_2_way_softmax(
+    using_vlm_head = is_vlm and hasattr(language_model, "score")
@@ -506,14 +507,16 @@ def load_weights_using_from_2_way_softmax(
-    score_layer = language_model.score if is_vlm else model.score
+    score_layer = language_model.score if using_vlm_head else model.score
-    score_weight_name = "language_model.score.weight" if is_vlm else "score.weight"
+    score_weight_name = (
diff -- tests/entrypoints/test_utils.py
@@ -1,9 +1,5 @@
```

- 已读文件:
  - tests: `tests/entrypoints/pooling/score/test_online_score_vision.py` added +122/-0; `tests/entrypoints/test_utils.py` modified +0/-12; `tests/entrypoints/pooling/classify/test_online_vision.py` modified +2/-2
  - runtime: `vllm/model_executor/models/adapters.py` modified +10/-4
  - docs: `examples/pooling/score/vision_score_api_online.py` modified +54/-45; `examples/pooling/score/vision_rerank_api_online.py` modified +46/-49
- 验证与风险: diff 自带测试面 `tests/entrypoints/pooling/classify/test_online_vision.py`, `tests/entrypoints/pooling/score/test_online_score_vision.py`, `tests/entrypoints/test_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33060 - [Frontend][4/n] Make pooling entrypoints request schema consensus | ScoreRequest

- 链接: https://github.com/vllm-project/vllm/pull/33060
- 状态/时间: merged / 2026-02-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+432/-205，可读 patch 1008 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend][4/n] Make pooling entrypoints request schema consensus | ScoreRequest」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `vllm/entrypoints/pooling/score/serving.py`, `tests/entrypoints/pooling/score/test_online_score_vision.py`, `vllm/entrypoints/llm.py`；技术摘要: 覆盖「[Frontend][4/n] Make pooling entrypoints request schema consensus | ScoreRequest」；主要实现面是 `vllm/entrypoints/pooling/score/serving.py`, `tests/entrypoints/pooling/score/test_online_score_vision.py`, `vllm/entrypoints/llm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/entrypoints/pooling/score/serving.py` modified +84/-94 (178 lines); hunks: -27,12 +27,12; -65,15 +65,32 @@ def __init__(; symbols: __init__, _embedding_score, _preprocess_score，涉及 `__init__, _embedding_score, _preprocess_score`；`tests/entrypoints/pooling/score/test_online_score_vision.py` modified +124/-7 (131 lines); hunks: -5,7 +5,7; -16,11 +16,12; symbols: server, test_score_api_queries_str_documents_str, test_score_api_queries_str_documents_text_content，涉及 `server, test_score_api_queries_str_documents_str, test_score_api_queries_str_documents_text_content`；`vllm/entrypoints/llm.py` modified +37/-73 (110 lines); hunks: -40,12 +40,12; -1326,8 +1326,8 @@ def reward(; symbols: reward, _embedding_score, _cross_encoding_score，涉及 `reward, _embedding_score, _cross_encoding_score`；`vllm/entrypoints/pooling/score/utils.py` modified +69/-20 (89 lines); hunks: -1,5 +1,6; -10,12 +11,13; symbols: ScoreMultiModalParam, _cosine_similarity, _validate_score_input_lens，涉及 `ScoreMultiModalParam, _cosine_similarity, _validate_score_input_lens`。
- 代码 diff 细节:
  - `vllm/entrypoints/pooling/score/serving.py` modified +84/-94 (178 lines); hunks: -27,12 +27,12; -65,15 +65,32 @@ def __init__(; symbols: __init__, _embedding_score, _preprocess_score
  - `tests/entrypoints/pooling/score/test_online_score_vision.py` modified +124/-7 (131 lines); hunks: -5,7 +5,7; -16,11 +16,12; symbols: server, test_score_api_queries_str_documents_str, test_score_api_queries_str_documents_text_content
  - `vllm/entrypoints/llm.py` modified +37/-73 (110 lines); hunks: -40,12 +40,12; -1326,8 +1326,8 @@ def reward(; symbols: reward, _embedding_score, _cross_encoding_score
  - `vllm/entrypoints/pooling/score/utils.py` modified +69/-20 (89 lines); hunks: -1,5 +1,6; -10,12 +11,13; symbols: ScoreMultiModalParam, _cosine_similarity, _validate_score_input_lens
  - `vllm/entrypoints/pooling/score/protocol.py` modified +28/-11 (39 lines); hunks: -14,7 +14,8; -47,13 +48,13 @@ def to_pooling_params(self):; symbols: to_pooling_params, ScoreDataRequest, ScoreQueriesDocumentsRequest, data_1
- 关键代码摘录:

```diff
diff -- vllm/entrypoints/pooling/score/serving.py
@@ -27,12 +27,12 @@
-    ScoreContentPartParam,
-    ScoreMultiModalParam,
+    ScoreData,
+    ScoreInputs,
-    _validate_score_input_lens,
+    validate_score_input,
diff -- tests/entrypoints/pooling/score/test_online_score_vision.py
@@ -5,7 +5,7 @@
-from vllm.entrypoints.pooling.score.protocol import ScoreResponse
+from vllm.entrypoints.pooling.score.protocol import RerankResponse, ScoreResponse
@@ -16,11 +16,12 @@
+document = "This product was excellent and exceeded my expectations."
-        "text": query,
+        "text": document,
diff -- vllm/entrypoints/llm.py
@@ -40,12 +40,12 @@
```

- 已读文件:
  - runtime: `vllm/entrypoints/pooling/score/serving.py` modified +84/-94; `vllm/entrypoints/llm.py` modified +37/-73; `vllm/entrypoints/pooling/score/utils.py` modified +69/-20; `vllm/entrypoints/pooling/score/protocol.py` modified +28/-11
  - tests: `tests/entrypoints/pooling/score/test_online_score_vision.py` modified +124/-7; `tests/entrypoints/pooling/score/test_online_score.py` modified +29/-0
  - docs: `examples/pooling/score/vision_score_api_online.py` modified +38/-0; `examples/pooling/score/vision_rerank_api_online.py` modified +23/-0
- 验证与风险: diff 自带测试面 `tests/entrypoints/pooling/score/test_online_score.py`, `tests/entrypoints/pooling/score/test_online_score_vision.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33837 - [Bugfix] Fix ScoreMultiModalParam multi-document scoring returning single result

- 链接: https://github.com/vllm-project/vllm/pull/33837
- 状态/时间: merged / 2026-02-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+21/-44，可读 patch 99 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix ScoreMultiModalParam multi-document scoring returning single result」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `tests/models/multimodal/pooling/test_jinavl_reranker.py`；技术摘要: 覆盖「[Bugfix] Fix ScoreMultiModalParam multi-document scoring returning single result」；主要实现面是 `tests/models/multimodal/pooling/test_jinavl_reranker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +21/-44 (65 lines); hunks: -1,6 +1,5; -117,7 +116,7 @@ def _normalize_image(image_val: str) -> str:; symbols: _normalize_image, create_score_multimodal_param, _run_vllm, _run_hf，涉及 `_normalize_image, create_score_multimodal_param, _run_vllm`。
- 代码 diff 细节:
  - `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +21/-44 (65 lines); hunks: -1,6 +1,5; -117,7 +116,7 @@ def _normalize_image(image_val: str) -> str:; symbols: _normalize_image, create_score_multimodal_param, _run_vllm, _run_hf
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/pooling/test_jinavl_reranker.py
@@ -1,6 +1,5 @@
-from typing import cast
@@ -117,7 +116,7 @@ def _normalize_image(image_val: str) -> str:
-) -> ScoreMultiModalParam:
+) -> list[ScoreMultiModalParam]:
@@ -152,7 +151,7 @@ def create_score_multimodal_param(
-    return ScoreMultiModalParam(content=formatted_content)
```

- 已读文件:
  - tests: `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +21/-44
- 验证与风险: diff 自带测试面 `tests/models/multimodal/pooling/test_jinavl_reranker.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31127 - [Frontend][last/5] Make pooling entrypoints request schema consensus.

- 链接: https://github.com/vllm-project/vllm/pull/31127
- 状态/时间: merged / 2026-02-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 24 个文件，+658/-612，可读 patch 1726 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend][last/5] Make pooling entrypoints request schema consensus.」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `tests/entrypoints/pooling/embed/test_online_vision.py`, `tests/entrypoints/pooling/classify/test_offline.py`, `vllm/entrypoints/pooling/score/protocol.py`；技术摘要: 覆盖「[Frontend][last/5] Make pooling entrypoints request schema consensus.」；主要实现面是 `tests/entrypoints/pooling/embed/test_online_vision.py`, `tests/entrypoints/pooling/classify/test_offline.py`, `vllm/entrypoints/pooling/score/protocol.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/entrypoints/pooling/embed/test_online_vision.py` modified +81/-2 (83 lines); hunks: -10,12 +10,12; -26,6 +26,10; symbols: server, test_chat_text_request, test_chat_image_url_request, test_chat_image_base64_request，涉及 `server, test_chat_text_request, test_chat_image_url_request`；`tests/entrypoints/pooling/classify/test_offline.py` modified +50/-7 (57 lines); hunks: -7,12 +7,15; -35,11 +38,48 @@ def llm():; symbols: llm, test_str_prompts, test_token_ids_prompts, test_list_prompts，涉及 `llm, test_str_prompts, test_token_ids_prompts`；`vllm/entrypoints/pooling/score/protocol.py` modified +1/-15 (16 lines); hunks: -1,7 +1,7; -23,13 +23,6; symbols: ScoreRequestMixin, build_tok_params, RerankRequest，涉及 `ScoreRequestMixin, build_tok_params, RerankRequest`；`vllm/entrypoints/pooling/base/protocol.py` modified +15/-0 (15 lines); hunks: -40,6 +40,21 @@ class PoolingBasicRequestMixin(OpenAIBaseModel):; symbols: PoolingBasicRequestMixin，涉及 `PoolingBasicRequestMixin`。
- 代码 diff 细节:
  - `tests/entrypoints/pooling/embed/test_online_vision.py` modified +81/-2 (83 lines); hunks: -10,12 +10,12; -26,6 +26,10; symbols: server, test_chat_text_request, test_chat_image_url_request, test_chat_image_base64_request
  - `tests/entrypoints/pooling/classify/test_offline.py` modified +50/-7 (57 lines); hunks: -7,12 +7,15; -35,11 +38,48 @@ def llm():; symbols: llm, test_str_prompts, test_token_ids_prompts, test_list_prompts
  - `vllm/entrypoints/pooling/score/protocol.py` modified +1/-15 (16 lines); hunks: -1,7 +1,7; -23,13 +23,6; symbols: ScoreRequestMixin, build_tok_params, RerankRequest
  - `vllm/entrypoints/pooling/base/protocol.py` modified +15/-0 (15 lines); hunks: -40,6 +40,21 @@ class PoolingBasicRequestMixin(OpenAIBaseModel):; symbols: PoolingBasicRequestMixin
  - `vllm/entrypoints/pooling/classify/protocol.py` modified +1/-7 (8 lines); hunks: -2,7 +2,7; -48,12 +48,6 @@ def to_pooling_params(self):; symbols: to_pooling_params, ClassificationChatRequest, build_tok_params
- 关键代码摘录:

```diff
diff -- tests/entrypoints/pooling/embed/test_online_vision.py
@@ -10,12 +10,12 @@
-from vllm.multimodal.utils import fetch_image
+from vllm.multimodal.utils import encode_image_url, fetch_image
-vlm2vec_jinja_path = VLLM_PATH / "examples/template_vlm2vec_phi3v.jinja"
+vlm2vec_jinja_path = VLLM_PATH / "examples/pooling/embed/template/vlm2vec_phi3v.jinja"
@@ -26,6 +26,10 @@
+input_text = "The best thing about vLLM is that it supports many different models"
diff -- tests/entrypoints/pooling/classify/test_offline.py
@@ -7,12 +7,15 @@
-from vllm import LLM, PoolingParams
+from vllm import LLM, ClassificationRequestOutput, PoolingParams, PoolingRequestOutput
+from vllm.tasks import PoolingTask
-prompts = ["The chef prepared a delicious meal."]
+prompt = "The chef prepared a delicious meal."
+prompt_token_ids = [785, 29706, 10030, 264, 17923, 15145, 13]
diff -- vllm/entrypoints/pooling/score/protocol.py
@@ -1,7 +1,7 @@
```

- 已读文件:
  - tests: `tests/entrypoints/pooling/embed/test_online_vision.py` modified +81/-2; `tests/entrypoints/pooling/classify/test_offline.py` modified +50/-7
  - runtime: `vllm/entrypoints/pooling/score/protocol.py` modified +1/-15; `vllm/entrypoints/pooling/base/protocol.py` modified +15/-0; `vllm/entrypoints/pooling/classify/protocol.py` modified +1/-7; `vllm/entrypoints/pooling/embed/protocol.py` modified +1/-6; `vllm/entrypoints/pooling/pooling/protocol.py` modified +1/-6
  - docs: `docs/features/multimodal_inputs.md` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/entrypoints/pooling/classify/test_offline.py`, `tests/entrypoints/pooling/embed/test_online_vision.py`, `tests/renderers/test_hf.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35592 - [Docs] Reorganize pooling docs.

- 链接: https://github.com/vllm-project/vllm/pull/35592
- 状态/时间: merged / 2026-03-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+2393/-1736，可读 patch 4283 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Reorganize pooling docs.」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/models/pooling_models.md`, `docs/models/pooling_models/embed.md`, `docs/models/pooling_models/scoring.md`；技术摘要: 覆盖「[Docs] Reorganize pooling docs.」；主要实现面是 `docs/models/pooling_models.md`, `docs/models/pooling_models/embed.md`, `docs/models/pooling_models/scoring.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/models/pooling_models.md` removed +0/-716 (716 lines); hunks: -1,716 +0,0; symbols: provides，涉及 `provides`；`docs/models/pooling_models/embed.md` added +546/-0 (546 lines); hunks: -0,0 +1,546; symbols: of, create_chat_embeddings，涉及 `of, create_chat_embeddings`；`docs/models/pooling_models/scoring.md` added +448/-0 (448 lines); hunks: -0,0 +1,448；`docs/models/pooling_models/specific_models.md` added +395/-0 (395 lines); hunks: -0,0 +1,395。
- 代码 diff 细节:
  - `docs/models/pooling_models.md` removed +0/-716 (716 lines); hunks: -1,716 +0,0; symbols: provides
  - `docs/models/pooling_models/embed.md` added +546/-0 (546 lines); hunks: -0,0 +1,546; symbols: of, create_chat_embeddings
  - `docs/models/pooling_models/scoring.md` added +448/-0 (448 lines); hunks: -0,0 +1,448
  - `docs/models/pooling_models/specific_models.md` added +395/-0 (395 lines); hunks: -0,0 +1,395
  - `docs/models/pooling_models/classify.md` added +276/-0 (276 lines); hunks: -0,0 +1,276; symbols: probabilities
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models.md
@@ -1,716 +0,0 @@
-# Pooling Models
-vLLM also supports pooling models, such as embedding, classification, and reward models.
-In vLLM, pooling models implement the [VllmModelForPooling][vllm.model_executor.models.VllmModelForPooling] interface.
-These models use a [Pooler][vllm.model_executor.layers.pooler.Pooler] to extract the final hidden states of the input
-before returning them.
-!!! note
diff -- docs/models/pooling_models/embed.md
@@ -0,0 +1,546 @@
+# Embedding Usages
+Embedding models are a class of machine learning models designed to transform unstructured data—such as text, images, or audio—into a structured numerical representation known as
+## Summary
+- Model Usage: (sequence) embedding
+- Pooling Task: `embed`
+- Offline APIs:
diff -- docs/models/pooling_models/scoring.md
@@ -0,0 +1,448 @@
```

- 已读文件:
  - docs: `docs/models/pooling_models.md` removed +0/-716; `docs/models/pooling_models/embed.md` added +546/-0; `docs/models/pooling_models/scoring.md` added +448/-0; `docs/models/pooling_models/specific_models.md` added +395/-0; `docs/models/pooling_models/classify.md` added +276/-0; `docs/models/pooling_models/README.md` added +253/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/.nav.yml`, `docs/contributing/model/tests.md`, `docs/features/README.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #37537 - [Model] Deprecate the score task (this will not affect users).

- 链接: https://github.com/vllm-project/vllm/pull/37537
- 状态/时间: merged / 2026-03-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 22 个文件，+184/-163，可读 patch 808 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Deprecate the score task (this will not affect users).」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/models/pooling_models/README.md`, `vllm/model_executor/layers/pooler/seqwise/heads.py`, `vllm/entrypoints/pooling/__init__.py`；技术摘要: 覆盖「[Model] Deprecate the score task (this will not affect users).」；主要实现面是 `docs/models/pooling_models/README.md`, `vllm/model_executor/layers/pooler/seqwise/heads.py`, `vllm/entrypoints/pooling/__init__.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/models/pooling_models/README.md` modified +35/-28 (63 lines); hunks: -31,28 +31,29 @@ Of course, we also have "plugin" tasks that allow users to c...; -85,14 +86,16 @@ enabling the corresponding APIs.；`vllm/model_executor/layers/pooler/seqwise/heads.py` modified +26/-22 (48 lines); hunks: -56,45 +56,47 @@ def forward(; -113,7 +115,7 @@ def __init__(; symbols: forward, ClassifierPoolerHead, __init__, get_supported_tasks，涉及 `forward, ClassifierPoolerHead, __init__`；`vllm/entrypoints/pooling/__init__.py` modified +29/-11 (40 lines); hunks: -5,6 +5,9; -17,9 +20,30; symbols: enable_scoring_api, register_pooling_api_routers, init_pooling_state，涉及 `enable_scoring_api, register_pooling_api_routers, init_pooling_state`；`vllm/model_executor/layers/pooler/activations.py` modified +10/-22 (32 lines); hunks: -16,25 +16,22; -55,24 +52,16 @@ def get_cross_encoder_act_fn(; symbols: get_classification_act_fn, get_act_fn, get_cross_encoder_act_fn, resolve_classifier_act_fn，涉及 `get_classification_act_fn, get_act_fn, get_cross_encoder_act_fn`。
- 代码 diff 细节:
  - `docs/models/pooling_models/README.md` modified +35/-28 (63 lines); hunks: -31,28 +31,29 @@ Of course, we also have "plugin" tasks that allow users to c...; -85,14 +86,16 @@ enabling the corresponding APIs.
  - `vllm/model_executor/layers/pooler/seqwise/heads.py` modified +26/-22 (48 lines); hunks: -56,45 +56,47 @@ def forward(; -113,7 +115,7 @@ def __init__(; symbols: forward, ClassifierPoolerHead, __init__, get_supported_tasks
  - `vllm/entrypoints/pooling/__init__.py` modified +29/-11 (40 lines); hunks: -5,6 +5,9; -17,9 +20,30; symbols: enable_scoring_api, register_pooling_api_routers, init_pooling_state
  - `vllm/model_executor/layers/pooler/activations.py` modified +10/-22 (32 lines); hunks: -16,25 +16,22; -55,24 +52,16 @@ def get_cross_encoder_act_fn(; symbols: get_classification_act_fn, get_act_fn, get_cross_encoder_act_fn, resolve_classifier_act_fn
  - `vllm/model_executor/layers/pooler/tokwise/heads.py` modified +16/-14 (30 lines); hunks: -68,22 +68,24 @@ def forward_chunk(; -118,16 +120,16 @@ def forward_chunk(; symbols: forward_chunk, TokenClassifierPoolerHead
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models/README.md
@@ -31,28 +31,29 @@ Of course, we also have "plugin" tasks that allow users to customize input and o
-| Pooling Tasks      | Granularity   | Outputs                                         |
-|--------------------|---------------|-------------------------------------------------|
-| `classify`         | Sequence-wise | probability vector of classes for each sequence |
-| `score` (see note) | Sequence-wise | reranker score for each sequence                |
-| `embed`            | Sequence-wise | vector representations for each sequence        |
-| `token_classify`   | Token-wise    | probability vector of classes for each token    |
diff -- vllm/model_executor/layers/pooler/seqwise/heads.py
@@ -56,45 +56,47 @@ def forward(
-        # pooled_data shape: [batchsize, hidden_dimension]
+        # pooled_data shape: [batchsize, hidden_size]
-            pooled_data = self.projector(pooled_data)
-        # pooled_data shape: [batchsize, embedding_dimension]
+            embeddings = self.projector(pooled_data)
+        else:
diff -- vllm/entrypoints/pooling/__init__.py
@@ -5,6 +5,9 @@
```

- 已读文件:
  - docs: `docs/models/pooling_models/README.md` modified +35/-28; `docs/models/pooling_models/scoring.md` modified +10/-7
  - runtime: `vllm/model_executor/layers/pooler/seqwise/heads.py` modified +26/-22; `vllm/entrypoints/pooling/__init__.py` modified +29/-11; `vllm/model_executor/layers/pooler/activations.py` modified +10/-22; `vllm/model_executor/layers/pooler/tokwise/heads.py` modified +16/-14; `vllm/entrypoints/sagemaker/api_router.py` modified +13/-5; `vllm/entrypoints/openai/api_server.py` modified +9/-5
- 验证与风险: diff 自带测试面 `tests/test_pooling_params.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37902 - [Mypy] Better fixes for the `mypy` issues in `vllm/config`

- 链接: https://github.com/vllm-project/vllm/pull/37902
- 状态/时间: merged / 2026-03-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+153/-182，可读 patch 1078 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Mypy] Better fixes for the `mypy` issues in `vllm/config`」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `tests/models/multimodal/generation/test_keye.py`, `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `examples/offline_inference/encoder_decoder_multimodal.py`；技术摘要: 覆盖「[Mypy] Better fixes for the `mypy` issues in `vllm/config`」；主要实现面是 `tests/models/multimodal/generation/test_keye.py`, `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `examples/offline_inference/encoder_decoder_multimodal.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/generation/test_keye.py` modified +8/-11 (19 lines); hunks: -1,6 +1,5; -29,14 +28,6 @@ def test_keye_vl(image_assets, question: str):; symbols: test_keye_vl，涉及 `test_keye_vl`；`tests/models/multimodal/generation/test_vit_backend_functionality.py` modified +5/-10 (15 lines); hunks: -7,13 +7,12; -274,7 +273,7 @@ def run_llm_generate_test(config, mm_encoder_attn_backend, i...; symbols: run_llm_generate_test, run_llm_chat_test，涉及 `run_llm_generate_test, run_llm_chat_test`；`examples/offline_inference/encoder_decoder_multimodal.py` modified +5/-7 (12 lines); hunks: -8,7 +8,6; -91,13 +90,12 @@ def main(args):; symbols: main，涉及 `main`；`tests/entrypoints/offline_mode/test_offline_mode.py` modified +1/-4 (5 lines); hunks: -2,7 +2,6; -11,7 +10,6; symbols: disable_connect，涉及 `disable_connect`。
- 代码 diff 细节:
  - `tests/models/multimodal/generation/test_keye.py` modified +8/-11 (19 lines); hunks: -1,6 +1,5; -29,14 +28,6 @@ def test_keye_vl(image_assets, question: str):; symbols: test_keye_vl
  - `tests/models/multimodal/generation/test_vit_backend_functionality.py` modified +5/-10 (15 lines); hunks: -7,13 +7,12; -274,7 +273,7 @@ def run_llm_generate_test(config, mm_encoder_attn_backend, i...; symbols: run_llm_generate_test, run_llm_chat_test
  - `examples/offline_inference/encoder_decoder_multimodal.py` modified +5/-7 (12 lines); hunks: -8,7 +8,6; -91,13 +90,12 @@ def main(args):; symbols: main
  - `tests/entrypoints/offline_mode/test_offline_mode.py` modified +1/-4 (5 lines); hunks: -2,7 +2,6; -11,7 +10,6; symbols: disable_connect
  - `vllm/entrypoints/llm.py` modified +5/-0 (5 lines); hunks: -409,6 +409,11 @@ def _make_config(value: Any, cls: type[_R]) -> _R:; symbols: _make_config, from_engine_args, get_tokenizer
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/generation/test_keye.py
@@ -1,6 +1,5 @@
-from dataclasses import asdict
@@ -29,14 +28,6 @@ def test_keye_vl(image_assets, question: str):
-    engine_args = EngineArgs(
-        model=MODEL_NAME,
-        trust_remote_code=True,
-        max_model_len=8192,
diff -- tests/models/multimodal/generation/test_vit_backend_functionality.py
@@ -7,13 +7,12 @@
-from dataclasses import asdict
-from vllm import LLM, EngineArgs, SamplingParams
+from vllm import LLM, SamplingParams
@@ -274,7 +273,7 @@ def run_llm_generate_test(config, mm_encoder_attn_backend, image_assets):
-    engine_args = EngineArgs(
+    llm = LLM(
diff -- examples/offline_inference/encoder_decoder_multimodal.py
@@ -8,7 +8,6 @@
```

- 已读文件:
  - tests: `tests/models/multimodal/generation/test_keye.py` modified +8/-11; `tests/models/multimodal/generation/test_vit_backend_functionality.py` modified +5/-10; `tests/entrypoints/offline_mode/test_offline_mode.py` modified +1/-4; `tests/models/multimodal/generation/test_voxtral_realtime.py` modified +1/-2; `tests/v1/kv_connector/unit/test_example_connector.py` modified +15/-22
  - docs: `examples/offline_inference/encoder_decoder_multimodal.py` modified +5/-7; `examples/pooling/embed/vision_embedding_offline.py` modified +19/-25
  - runtime: `vllm/entrypoints/llm.py` modified +5/-0
- 验证与风险: diff 自带测试面 `tests/compile/test_config.py`, `tests/entrypoints/offline_mode/test_offline_mode.py`, `tests/models/multimodal/generation/test_keye.py`, `tests/models/multimodal/generation/test_vit_backend_functionality.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28631 - [Frontend][3/n] Improve pooling entrypoints | scoring.

- 链接: https://github.com/vllm-project/vllm/pull/28631
- 状态/时间: merged / 2026-03-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 37 个文件，+1257/-1780，可读 patch 3713 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend][3/n] Improve pooling entrypoints | scoring.」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/pooling/scoring/io_processor.py`, `tests/entrypoints/pooling/scoring/test_utils.py`；技术摘要: 覆盖「[Frontend][3/n] Improve pooling entrypoints | scoring.」；主要实现面是 `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/pooling/scoring/io_processor.py`, `tests/entrypoints/pooling/scoring/test_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/entrypoints/pooling/score/serving.py` removed +0/-667 (667 lines); hunks: -1,667 +0,0; symbols: ServingScores, __init__, _embedding_score, _preprocess_late_interaction_item，涉及 `ServingScores, __init__, _embedding_score`；`vllm/entrypoints/pooling/scoring/io_processor.py` added +419/-0 (419 lines); hunks: -0,0 +1,419; symbols: ScoringIOProcessor, __init__, create_pooling_params, valid_inputs，涉及 `ScoringIOProcessor, __init__, create_pooling_params`；`tests/entrypoints/pooling/scoring/test_utils.py` removed +0/-353 (353 lines); hunks: -1,353 +0,0; symbols: assert_prompt_tokenization_consistent, cross_encoder_model_config, cross_encoder_tokenizer, llm_reranker_model_config，涉及 `assert_prompt_tokenization_consistent, cross_encoder_model_config, cross_encoder_tokenizer`；`vllm/entrypoints/pooling/scoring/utils.py` renamed +71/-246 (317 lines); hunks: -1,42 +1,27; -57,72 +42,6 @@ def compute_maxsim_score(q_emb: torch.Tensor, d_emb: torch.Te...; symbols: compute_maxsim_score, ScoreMultiModalParam, _cosine_similarity, _validate_score_input_lens，涉及 `compute_maxsim_score, ScoreMultiModalParam, _cosine_similarity`。
- 代码 diff 细节:
  - `vllm/entrypoints/pooling/score/serving.py` removed +0/-667 (667 lines); hunks: -1,667 +0,0; symbols: ServingScores, __init__, _embedding_score, _preprocess_late_interaction_item
  - `vllm/entrypoints/pooling/scoring/io_processor.py` added +419/-0 (419 lines); hunks: -0,0 +1,419; symbols: ScoringIOProcessor, __init__, create_pooling_params, valid_inputs
  - `tests/entrypoints/pooling/scoring/test_utils.py` removed +0/-353 (353 lines); hunks: -1,353 +0,0; symbols: assert_prompt_tokenization_consistent, cross_encoder_model_config, cross_encoder_tokenizer, llm_reranker_model_config
  - `vllm/entrypoints/pooling/scoring/utils.py` renamed +71/-246 (317 lines); hunks: -1,42 +1,27; -57,72 +42,6 @@ def compute_maxsim_score(q_emb: torch.Tensor, d_emb: torch.Te...; symbols: compute_maxsim_score, ScoreMultiModalParam, _cosine_similarity, _validate_score_input_lens
  - `vllm/entrypoints/llm.py` modified +60/-252 (312 lines); hunks: -46,22 +46,16; -1161,7 +1155,9 @@ def encode(; symbols: encode, reward, _embedding_score, _late_interaction_score
- 关键代码摘录:

```diff
diff -- vllm/entrypoints/pooling/score/serving.py
@@ -1,667 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-import asyncio
-import time
-from collections.abc import AsyncGenerator, Mapping
-from concurrent.futures import ThreadPoolExecutor
diff -- vllm/entrypoints/pooling/scoring/io_processor.py
@@ -0,0 +1,419 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import time
+from collections.abc import Sequence
+from typing import Any, TypeAlias, cast
+import torch.nn.functional as F
diff -- tests/entrypoints/pooling/scoring/test_utils.py
@@ -1,353 +0,0 @@
```

- 已读文件:
  - runtime: `vllm/entrypoints/pooling/score/serving.py` removed +0/-667; `vllm/entrypoints/pooling/scoring/io_processor.py` added +419/-0; `vllm/entrypoints/pooling/scoring/utils.py` renamed +71/-246; `vllm/entrypoints/llm.py` modified +60/-252; `vllm/entrypoints/pooling/scoring/serving.py` added +160/-0; `vllm/entrypoints/openai/engine/serving.py` modified +4/-104
  - tests: `tests/entrypoints/pooling/scoring/test_utils.py` removed +0/-353; `tests/entrypoints/pooling/scoring/test_late_interaction_online_vision.py` added +193/-0
- 验证与风险: diff 自带测试面 `tests/entrypoints/openai/utils.py`, `tests/entrypoints/pooling/classify/test_offline.py`, `tests/entrypoints/pooling/classify/test_online.py`, `tests/entrypoints/pooling/scoring/test_bi_encoder_online.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38800 - [New Model]: jinaai/jina-reranker-v3

- 链接: https://github.com/vllm-project/vllm/pull/38800
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/pooling/token_embed/jina_reranker_v3_offline.py`, `tests/models/language/pooling/test_jina_reranker_v3.py`, `vllm/model_executor/models/jina.py`；关联提交 `cb5f7501cbc8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+660/-19，可读 patch 795 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[New Model]: jinaai/jina-reranker-v3」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `tests/models/language/pooling/test_jina_reranker_v3.py`, `vllm/model_executor/models/jina.py`, `examples/pooling/token_embed/jina_reranker_v3_offline.py`；技术摘要: 覆盖「[New Model]: jinaai/jina-reranker-v3」；主要实现面是 `tests/models/language/pooling/test_jina_reranker_v3.py`, `vllm/model_executor/models/jina.py`, `examples/pooling/token_embed/jina_reranker_v3_offline.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/language/pooling/test_jina_reranker_v3.py` added +275/-0 (275 lines); hunks: -0,0 +1,275; symbols: test_offline, test_online, _test_offline_1_v_1, _test_offline_1_v_n，涉及 `test_offline, test_online, _test_offline_1_v_1`；`vllm/model_executor/models/jina.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: JinaForRanking, __init__, embed_input_ids, forward，涉及 `JinaForRanking, __init__, embed_input_ids`；`examples/pooling/token_embed/jina_reranker_v3_offline.py` added +56/-0 (56 lines); hunks: -0,0 +1,56; symbols: main，涉及 `main`。
- 代码 diff 细节:
  - `tests/models/language/pooling/test_jina_reranker_v3.py` added +275/-0 (275 lines); hunks: -0,0 +1,275; symbols: test_offline, test_online, _test_offline_1_v_1, _test_offline_1_v_n
  - `vllm/model_executor/models/jina.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: JinaForRanking, __init__, embed_input_ids, forward
  - `examples/pooling/token_embed/jina_reranker_v3_offline.py` added +56/-0 (56 lines); hunks: -0,0 +1,56; symbols: main
- 关键代码摘录:

```diff
diff -- tests/models/language/pooling/test_jina_reranker_v3.py
@@ -0,0 +1,275 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+import pytest
+import requests
+import torch
diff -- vllm/model_executor/models/jina.py
@@ -0,0 +1,110 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from https://huggingface.co/jinaai/jina-reranker-v3/blob/main/modeling.py
+from collections.abc import Iterable
+import torch
+from torch import nn
diff -- examples/pooling/token_embed/jina_reranker_v3_offline.py
@@ -0,0 +1,56 @@
```

- 已读文件:
  - tests: `tests/models/language/pooling/test_jina_reranker_v3.py` added +275/-0
  - runtime: `vllm/model_executor/models/jina.py` added +110/-0
  - docs: `examples/pooling/token_embed/jina_reranker_v3_offline.py` added +56/-0
- 验证与风险: diff 自带测试面 `tests/models/language/pooling/test_jina_reranker_v3.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30566 - Update to transformers v5

- 链接: https://github.com/vllm-project/vllm/pull/30566
- 状态/时间: merged / 2026-04-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 41 个文件，+445/-115，可读 patch 1409 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update to transformers v5」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `tests/models/registry.py`, `vllm/model_executor/models/gemma4_mm.py`, `tests/models/multimodal/generation/test_common.py`；技术摘要: 覆盖「Update to transformers v5」；主要实现面是 `tests/models/registry.py`, `vllm/model_executor/models/gemma4_mm.py`, `tests/models/multimodal/generation/test_common.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/registry.py` modified +130/-9 (139 lines); hunks: -335,7 +335,15 @@ def check_available_online(; -475,6 +483,13 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`vllm/model_executor/models/gemma4_mm.py` modified +36/-15 (51 lines); hunks: -125,8 +125,12 @@ class Gemma4AudioInputs(TensorSchema):; -505,6 +509,8 @@ def _call_hf_processor(; symbols: Gemma4AudioInputs, _call_hf_processor，涉及 `Gemma4AudioInputs, _call_hf_processor`；`tests/models/multimodal/generation/test_common.py` modified +38/-6 (44 lines); hunks: -186,7 +186,14; -397,14 +404,14；`vllm/tokenizers/registry.py` modified +34/-1 (35 lines); hunks: -1,5 +1,6; -10,6 +11,7; symbols: get_tokenizer，涉及 `get_tokenizer`。
- 代码 diff 细节:
  - `tests/models/registry.py` modified +130/-9 (139 lines); hunks: -335,7 +335,15 @@ def check_available_online(; -475,6 +483,13 @@ def check_available_online(; symbols: check_available_online
  - `vllm/model_executor/models/gemma4_mm.py` modified +36/-15 (51 lines); hunks: -125,8 +125,12 @@ class Gemma4AudioInputs(TensorSchema):; -505,6 +509,8 @@ def _call_hf_processor(; symbols: Gemma4AudioInputs, _call_hf_processor
  - `tests/models/multimodal/generation/test_common.py` modified +38/-6 (44 lines); hunks: -186,7 +186,14; -397,14 +404,14
  - `vllm/tokenizers/registry.py` modified +34/-1 (35 lines); hunks: -1,5 +1,6; -10,6 +11,7; symbols: get_tokenizer
  - `tests/model_executor/test_weight_utils.py` modified +0/-18 (18 lines); hunks: -1,7 +1,6; -10,26 +9,10; symbols: test_hf_transfer_auto_activation, test_download_weights_from_hf, test_missing_target_returns_none
- 关键代码摘录:

```diff
diff -- tests/models/registry.py
@@ -335,7 +335,15 @@ def check_available_online(
-        "OpenGVLab/Mono-InternVL-2B", trust_remote_code=True
+        "OpenGVLab/Mono-InternVL-2B",
+        trust_remote_code=True,
+        max_transformers_version="4.57",
+        transformers_version_reason={
+            "vllm": (
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -125,8 +125,12 @@ class Gemma4AudioInputs(TensorSchema):
-    input_features_padded: Annotated[torch.Tensor, TensorShape("bn", "s", "f")]
-    input_features_mask: Annotated[torch.Tensor, TensorShape("bn", "s")]
+    input_features_padded: Annotated[
+        torch.Tensor, TensorShape("bn", "s", "f", dynamic_dims={"s"})
+    ]
+    input_features_mask: Annotated[
diff -- tests/models/multimodal/generation/test_common.py
@@ -186,7 +186,14 @@
```

- 已读文件:
  - tests: `tests/models/registry.py` modified +130/-9; `tests/models/multimodal/generation/test_common.py` modified +38/-6; `tests/model_executor/test_weight_utils.py` modified +0/-18; `tests/models/multimodal/generation/test_phi4siglip.py` modified +11/-0; `tests/models/utils.py` modified +10/-1
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +36/-15; `vllm/tokenizers/registry.py` modified +34/-1; `vllm/model_executor/model_loader/gguf_loader.py` modified +12/-0
- 验证与风险: diff 自带测试面 `requirements/test/cuda.in`, `requirements/test/cuda.txt`, `requirements/test/nightly-torch.txt`, `requirements/test/rocm.in`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39575 - Add Jina Embeddings v5 model support (fixes #38633)

- 链接: https://github.com/vllm-project/vllm/pull/39575
- 状态/时间: merged / 2026-04-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/language/pooling_mteb_test/test_jina.py`, `vllm/model_executor/models/jina.py`；关联提交 `2cdf86044d7e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+218/-10，可读 patch 401 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Jina Embeddings v5 model support (fixes #38633)」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/jina.py`, `tests/models/language/pooling_mteb_test/test_jina.py`；技术摘要: 覆盖「Add Jina Embeddings v5 model support (fixes #38633)」；主要实现面是 `vllm/model_executor/models/jina.py`, `tests/models/language/pooling_mteb_test/test_jina.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/jina.py` modified +149/-1 (150 lines); hunks: -1,14 +1,19; -18,9 +23,12; symbols: JinaForRanking, forward, _load_adapter, _build_lora_pairs，涉及 `JinaForRanking, forward, _load_adapter`；`tests/models/language/pooling_mteb_test/test_jina.py` modified +25/-5 (30 lines); hunks: -28,7 +28,16; -46,20 +55,29; symbols: test_embed_models_mteb, hf_model_callback, test_embed_models_correctness, test_matryoshka，涉及 `test_embed_models_mteb, hf_model_callback, test_embed_models_correctness`。
- 代码 diff 细节:
  - `vllm/model_executor/models/jina.py` modified +149/-1 (150 lines); hunks: -1,14 +1,19; -18,9 +23,12; symbols: JinaForRanking, forward, _load_adapter, _build_lora_pairs
  - `tests/models/language/pooling_mteb_test/test_jina.py` modified +25/-5 (30 lines); hunks: -28,7 +28,16; -46,20 +55,29; symbols: test_embed_models_mteb, hf_model_callback, test_embed_models_correctness, test_matryoshka
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/jina.py
@@ -1,14 +1,19 @@
+import json
+import logging
+from collections import defaultdict
+from safetensors.torch import load as safetensors_load
+from vllm.transformers_utils.repo_utils import get_hf_file_bytes
@@ -18,9 +23,12 @@
diff -- tests/models/language/pooling_mteb_test/test_jina.py
@@ -28,7 +28,16 @@
-    )
+    ),
+    EmbedModelInfo(
+        "jinaai/jina-embeddings-v5-text-small",
+        mteb_score=0.794535707854956,
+        architecture="JinaEmbeddingsV5Model",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/jina.py` modified +149/-1
  - tests: `tests/models/language/pooling_mteb_test/test_jina.py` modified +25/-5
- 验证与风险: diff 自带测试面 `tests/conftest.py`, `tests/models/language/pooling_mteb_test/mteb_embed_utils.py`, `tests/models/language/pooling_mteb_test/test_jina.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39675 - [Frontend][last/5] Improve pooling entrypoints | clean up.

- 链接: https://github.com/vllm-project/vllm/pull/39675
- 状态/时间: merged / 2026-04-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 29 个文件，+465/-427，可读 patch 1334 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend][last/5] Improve pooling entrypoints | clean up.」；模型线: Jina Reranker M0；类别: 模型实现调整；主要 diff: `vllm/entrypoints/pooling/factories.py`, `vllm/entrypoints/pooling/__init__.py`, `vllm/entrypoints/sagemaker/api_router.py`；技术摘要: 覆盖「[Frontend][last/5] Improve pooling entrypoints | clean up.」；主要实现面是 `vllm/entrypoints/pooling/factories.py`, `vllm/entrypoints/pooling/__init__.py`, `vllm/entrypoints/sagemaker/api_router.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/entrypoints/pooling/factories.py` added +256/-0 (256 lines); hunks: -0,0 +1,256; symbols: init_pooling_io_processors, register_pooling_api_routers, init_pooling_state, get_pooling_invocation_types，涉及 `init_pooling_io_processors, register_pooling_api_routers, init_pooling_state`；`vllm/entrypoints/pooling/__init__.py` modified +0/-130 (130 lines); hunks: -1,130 +0,0; symbols: register_pooling_api_routers, init_pooling_state，涉及 `register_pooling_api_routers, init_pooling_state`；`vllm/entrypoints/sagemaker/api_router.py` modified +7/-77 (84 lines); hunks: -13,12 +13,13; -27,80 +28,6; symbols: get_invocation_types, attach_router，涉及 `get_invocation_types, attach_router`；`vllm/entrypoints/pooling/io_processor_factories.py` removed +0/-76 (76 lines); hunks: -1,76 +0,0; symbols: init_pooling_io_processors，涉及 `init_pooling_io_processors`。
- 代码 diff 细节:
  - `vllm/entrypoints/pooling/factories.py` added +256/-0 (256 lines); hunks: -0,0 +1,256; symbols: init_pooling_io_processors, register_pooling_api_routers, init_pooling_state, get_pooling_invocation_types
  - `vllm/entrypoints/pooling/__init__.py` modified +0/-130 (130 lines); hunks: -1,130 +0,0; symbols: register_pooling_api_routers, init_pooling_state
  - `vllm/entrypoints/sagemaker/api_router.py` modified +7/-77 (84 lines); hunks: -13,12 +13,13; -27,80 +28,6; symbols: get_invocation_types, attach_router
  - `vllm/entrypoints/pooling/io_processor_factories.py` removed +0/-76 (76 lines); hunks: -1,76 +0,0; symbols: init_pooling_io_processors
  - `vllm/entrypoints/openai/generate/factories.py` added +42/-0 (42 lines); hunks: -0,0 +1,42; symbols: get_generate_invocation_types
- 关键代码摘录:

```diff
diff -- vllm/entrypoints/pooling/factories.py
@@ -0,0 +1,256 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from typing import TYPE_CHECKING
+from fastapi import FastAPI
+from vllm.config import ModelConfig, VllmConfig
+from vllm.entrypoints.chat_utils import ChatTemplateConfig
diff -- vllm/entrypoints/pooling/__init__.py
@@ -1,130 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-from typing import TYPE_CHECKING
-from fastapi import FastAPI
-from vllm.config import ModelConfig
-from vllm.entrypoints.pooling.utils import enable_scoring_api
diff -- vllm/entrypoints/sagemaker/api_router.py
@@ -13,12 +13,13 @@
```

- 已读文件:
  - runtime: `vllm/entrypoints/pooling/factories.py` added +256/-0; `vllm/entrypoints/pooling/__init__.py` modified +0/-130; `vllm/entrypoints/sagemaker/api_router.py` modified +7/-77; `vllm/entrypoints/pooling/io_processor_factories.py` removed +0/-76; `vllm/entrypoints/openai/generate/factories.py` added +42/-0; `vllm/entrypoints/pooling/embed/io_processor.py` modified +20/-19
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/llm.py`, `vllm/entrypoints/openai/api_server.py`, `vllm/entrypoints/openai/generate/factories.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41832 - [Doc] Add ModernBertForSequenceClassification to scoring.md cross-en…

- 链接: https://github.com/vllm-project/vllm/pull/41832
- 状态/时间: merged / 2026-05-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc] Add ModernBertForSequenceClassification to scoring.md cross-en…」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/models/pooling_models/scoring.md`；技术摘要: 覆盖「[Doc] Add ModernBertForSequenceClassification to scoring.md cross-en…」；主要实现面是 `docs/models/pooling_models/scoring.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/models/pooling_models/scoring.md` modified +1/-0 (1 lines); hunks: -41,6 +41,7 @@ The score models is designed to compute similarity scores betw...。
- 代码 diff 细节:
  - `docs/models/pooling_models/scoring.md` modified +1/-0 (1 lines); hunks: -41,6 +41,7 @@ The score models is designed to compute similarity scores betw...
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models/scoring.md
@@ -41,6 +41,7 @@ The score models is designed to compute similarity scores between two input prom
+| `ModernBertForSequenceClassification` | ModernBERT-based | `Alibaba-NLP/gte-reranker-modernbert-base`, etc. | N/A | | |
```

- 已读文件:
  - docs: `docs/models/pooling_models/scoring.md` modified +1/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/models/pooling_models/scoring.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #42412 - [Feature] Add instruction support for score/rerank chat templates

- 链接: https://github.com/vllm-project/vllm/pull/42412
- 状态/时间: merged / 2026-05-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/pooling/score/template/qwen3_vl_reranker.jinja`；关联提交 `70c00163ffa8`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+182/-12，可读 patch 285 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Add instruction support for score/rerank chat templates」；模型线: Jina Reranker M0；类别: 模型支持/运行时入口；主要 diff: `examples/pooling/score/template/qwen3_vl_reranker.jinja`；技术摘要: 覆盖「[Feature] Add instruction support for score/rerank chat templates」；主要实现面是 `examples/pooling/score/template/qwen3_vl_reranker.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/pooling/score/template/qwen3_vl_reranker.jinja` modified +1/-7 (8 lines); hunks: -1,13 +1,7。
- 代码 diff 细节:
  - `examples/pooling/score/template/qwen3_vl_reranker.jinja` modified +1/-7 (8 lines); hunks: -1,13 +1,7
- 关键代码摘录:

```diff
diff -- examples/pooling/score/template/qwen3_vl_reranker.jinja
@@ -1,13 +1,7 @@
-<Instruct>: {{
-    messages
-    | selectattr("role", "eq", "system")
-    | map(attribute="content")
-    | first
-    | default("Given a search query, retrieve relevant candidates that answer the query.")
```

- 已读文件:
  - docs: `examples/pooling/score/template/qwen3_vl_reranker.jinja` modified +1/-7
- 验证与风险: diff 自带测试面 `tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42267 - [Entrypoints] Split the pooling offline API into PoolingOfflineMixin.

- 链接: https://github.com/vllm-project/vllm/pull/42267
- 状态/时间: merged / 2026-05-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+531/-439，可读 patch 1121 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Entrypoints] Split the pooling offline API into PoolingOfflineMixin.」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `vllm/entrypoints/pooling/offline.py`, `vllm/entrypoints/llm.py`, `docs/models/pooling_models/README.md`；技术摘要: 覆盖「[Entrypoints] Split the pooling offline API into PoolingOfflineMixin.」；主要实现面是 `vllm/entrypoints/pooling/offline.py`, `vllm/entrypoints/llm.py`, `docs/models/pooling_models/README.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/entrypoints/pooling/offline.py` added +510/-0 (510 lines); hunks: -0,0 +1,510; symbols: PoolingOfflineMixin, __init__, encode, automatically，涉及 `PoolingOfflineMixin, __init__, encode`；`vllm/entrypoints/llm.py` modified +7/-425 (432 lines); hunks: -34,27 +34,20; -63,13 +56,7; symbols: LLM, includes, _make_config, chat，涉及 `LLM, includes, _make_config`；`docs/models/pooling_models/README.md` modified +4/-4 (8 lines); hunks: -131,24 +131,24 @@ enabling the corresponding APIs.；`docs/models/pooling_models/embed.md` modified +3/-3 (6 lines); hunks: -120,7 +120,7 @@ The following [pooling parameters][vllm.PoolingParams] are s...; -136,7 +136,7 @@ A code example can be found here: [examples/basic/offline_in...。
- 代码 diff 细节:
  - `vllm/entrypoints/pooling/offline.py` added +510/-0 (510 lines); hunks: -0,0 +1,510; symbols: PoolingOfflineMixin, __init__, encode, automatically
  - `vllm/entrypoints/llm.py` modified +7/-425 (432 lines); hunks: -34,27 +34,20; -63,13 +56,7; symbols: LLM, includes, _make_config, chat
  - `docs/models/pooling_models/README.md` modified +4/-4 (8 lines); hunks: -131,24 +131,24 @@ enabling the corresponding APIs.
  - `docs/models/pooling_models/embed.md` modified +3/-3 (6 lines); hunks: -120,7 +120,7 @@ The following [pooling parameters][vllm.PoolingParams] are s...; -136,7 +136,7 @@ A code example can be found here: [examples/basic/offline_in...
  - `docs/models/pooling_models/classify.md` modified +2/-2 (4 lines); hunks: -77,7 +77,7 @@ The following [pooling parameters][vllm.PoolingParams] are sup...; -93,7 +93,7 @@ A code example can be found here: [examples/basic/offline_infe...
- 关键代码摘录:

```diff
diff -- vllm/entrypoints/pooling/offline.py
@@ -0,0 +1,510 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from abc import ABC, abstractmethod
+from collections.abc import Callable, Iterable, Sequence
+from typing import Any
+from tqdm.auto import tqdm
diff -- vllm/entrypoints/llm.py
@@ -34,27 +34,20 @@
-from vllm.config.quantization import (
-    QuantizationConfigArgs,
-)
+from vllm.config.quantization import QuantizationConfigArgs
-    ChatTemplateConfig,
-from vllm.entrypoints.pooling.factories import init_pooling_io_processors
diff -- docs/models/pooling_models/README.md
@@ -131,24 +131,24 @@ enabling the corresponding APIs.
```

- 已读文件:
  - runtime: `vllm/entrypoints/pooling/offline.py` added +510/-0; `vllm/entrypoints/llm.py` modified +7/-425
  - docs: `docs/models/pooling_models/README.md` modified +4/-4; `docs/models/pooling_models/embed.md` modified +3/-3; `docs/models/pooling_models/classify.md` modified +2/-2; `docs/models/pooling_models/token_embed.md` modified +2/-2; `docs/models/pooling_models/reward.md` modified +1/-1; `docs/models/pooling_models/scoring.md` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/llm.py`, `vllm/entrypoints/pooling/offline.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42626 - [Docs] Add SVG images for pooling models.

- 链接: https://github.com/vllm-project/vllm/pull/42626
- 状态/时间: merged / 2026-05-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+2336/-0，可读 patch 44 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add SVG images for pooling models.」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/assets/models/pooling_models/score_types.svg`, `docs/assets/models/pooling_models/cheat_sheet.svg`, `docs/assets/models/pooling_models/pooling_types.svg`；技术摘要: 覆盖「[Docs] Add SVG images for pooling models.」；主要实现面是 `docs/assets/models/pooling_models/score_types.svg`, `docs/assets/models/pooling_models/cheat_sheet.svg`, `docs/assets/models/pooling_models/pooling_types.svg`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/assets/models/pooling_models/score_types.svg` added +902/-0 (902 lines)；`docs/assets/models/pooling_models/cheat_sheet.svg` added +785/-0 (785 lines)；`docs/assets/models/pooling_models/pooling_types.svg` added +633/-0 (633 lines)；`docs/models/pooling_models/README.md` modified +10/-0 (10 lines); hunks: -33,6 +33,12 @@ from large language models, allowing them to benefit from the...; -61,6 +67,8 @@ are a subset of classification models that accept two prompts...。
- 代码 diff 细节:
  - `docs/assets/models/pooling_models/score_types.svg` added +902/-0 (902 lines)
  - `docs/assets/models/pooling_models/cheat_sheet.svg` added +785/-0 (785 lines)
  - `docs/assets/models/pooling_models/pooling_types.svg` added +633/-0 (633 lines)
  - `docs/models/pooling_models/README.md` modified +10/-0 (10 lines); hunks: -33,6 +33,12 @@ from large language models, allowing them to benefit from the...; -61,6 +67,8 @@ are a subset of classification models that accept two prompts...
  - `docs/models/pooling_models/scoring.md` modified +6/-0 (6 lines); hunks: -25,6 +25,12 @@ The score models is designed to compute similarity scores bet...
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models/README.md
@@ -33,6 +33,12 @@ from large language models, allowing them to benefit from the continuous improve
+### Cheat Sheet
+As illustrated in the figure below, we have summarized the relationships among the key elements of pooling models as a takeaway.
+![Cheat Sheet](../../assets/models/pooling_models/cheat_sheet.svg)
@@ -61,6 +67,8 @@ are a subset of classification models that accept two prompts as input and outpu
+![Pooling Types](../../assets/models/pooling_models/pooling_types.svg)
@@ -71,6 +79,8 @@ are a subset of classification models that accept two prompts as input and outpu
diff -- docs/models/pooling_models/scoring.md
@@ -25,6 +25,12 @@ The score models is designed to compute similarity scores between two input prom
+### Score Types
+The three supported scoring functions are as illustrated in the figure below.
+![Score Types](../../assets/models/pooling_models/score_types.svg)
```

- 已读文件:
  - docs: `docs/assets/models/pooling_models/score_types.svg` added +902/-0; `docs/assets/models/pooling_models/cheat_sheet.svg` added +785/-0; `docs/assets/models/pooling_models/pooling_types.svg` added +633/-0; `docs/models/pooling_models/README.md` modified +10/-0; `docs/models/pooling_models/scoring.md` modified +6/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/assets/models/pooling_models/cheat_sheet.svg`, `docs/assets/models/pooling_models/pooling_types.svg`, `docs/assets/models/pooling_models/score_types.svg`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #41907 - [Docs] Reorganize online serving docs.

- 链接: https://github.com/vllm-project/vllm/pull/41907
- 状态/时间: merged / 2026-05-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 24 个文件，+1348/-1241，可读 patch 1469 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Reorganize online serving docs.」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/assets/models/pooling_models/cheat_sheet.svg`, `docs/models/pooling_models/README.md`, `docs/models/supported_models.md`；技术摘要: 覆盖「[Docs] Reorganize online serving docs.」；主要实现面是 `docs/assets/models/pooling_models/cheat_sheet.svg`, `docs/models/pooling_models/README.md`, `docs/models/supported_models.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/assets/models/pooling_models/cheat_sheet.svg` modified +671/-660 (1331 lines)；`docs/models/pooling_models/README.md` modified +6/-6 (12 lines); hunks: -141,24 +141,24 @@ enabling the corresponding APIs.; -180,12 +180,12 @@ Our online Server provides endpoints that correspond to th...；`docs/models/supported_models.md` modified +4/-4 (8 lines); hunks: -44,7 +44,7 @@ llm.apply_model(lambda model: print(type(model))); -63,8 +63,8 @@ For a model to be compatible with the Transformers modeling ba...；`docs/models/generative_models.md` modified +3/-3 (6 lines); hunks: -138,7 +138,7 @@ outputs = llm.chat(conversation, chat_template=custom_template)。
- 代码 diff 细节:
  - `docs/assets/models/pooling_models/cheat_sheet.svg` modified +671/-660 (1331 lines)
  - `docs/models/pooling_models/README.md` modified +6/-6 (12 lines); hunks: -141,24 +141,24 @@ enabling the corresponding APIs.; -180,12 +180,12 @@ Our online Server provides endpoints that correspond to th...
  - `docs/models/supported_models.md` modified +4/-4 (8 lines); hunks: -44,7 +44,7 @@ llm.apply_model(lambda model: print(type(model))); -63,8 +63,8 @@ For a model to be compatible with the Transformers modeling ba...
  - `docs/models/generative_models.md` modified +3/-3 (6 lines); hunks: -138,7 +138,7 @@ outputs = llm.chat(conversation, chat_template=custom_template)
  - `docs/models/pooling_models/scoring.md` modified +3/-3 (6 lines); hunks: -20,7 +20,7 @@ The score models is designed to compute similarity scores betw...; -363,7 +363,7 @@ Full example:
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models/README.md
@@ -141,24 +141,24 @@ enabling the corresponding APIs.
-The [classify][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.classify] method outputs a probability vector for each prompt.
+The [classify][vllm.LLM.classify] method outputs a probability vector for each prompt.
-The [embed][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.embed] method outputs an embedding vector for each prompt.
+The [embed][vllm.LLM.embed] method outputs an embedding vector for each prompt.
-The [score][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.score] method outputs similarity scores between sentence pairs.
+The [score][vllm.LLM.score] method outputs similarity scores between sentence pairs.
diff -- docs/models/supported_models.md
@@ -44,7 +44,7 @@ llm.apply_model(lambda model: print(type(model)))
-If a model has a vLLM implementation but you would prefer to use the Transformers implementation via the Transformers modeling backend, set `model_impl="transformers"` for [offlin
+If a model has a vLLM implementation but you would prefer to use the Transformers implementation via the Transformers modeling backend, set `model_impl="transformers"` for [offlin
@@ -63,8 +63,8 @@ For a model to be compatible with the Transformers modeling backend for vLLM it
-- on the Hugging Face Model Hub, simply set `trust_remote_code=True` for [offline-inference](../serving/offline_inference.md) or `--trust-remote-code` for the [openai-compatible-s
-- in a local directory, simply pass directory path to `model=<MODEL_DIR>` for [offline-inference](../serving/offline_inference.md) or `vllm serve <MODEL_DIR>` for the [openai-comp
+- on the Hugging Face Model Hub, simply set `trust_remote_code=True` for [offline-inference](../serving/offline_inference.md) or `--trust-remote-code` for the [online serving](../
diff -- docs/models/generative_models.md
@@ -138,7 +138,7 @@ outputs = llm.chat(conversation, chat_template=custom_template)
```

- 已读文件:
  - docs: `docs/assets/models/pooling_models/cheat_sheet.svg` modified +671/-660; `docs/models/pooling_models/README.md` modified +6/-6; `docs/models/supported_models.md` modified +4/-4; `docs/models/generative_models.md` modified +3/-3; `docs/models/pooling_models/scoring.md` modified +3/-3; `docs/models/pooling_models/embed.md` modified +2/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs/.nav.yml`, `docs/assets/models/pooling_models/cheat_sheet.svg`, `docs/configuration/README.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #43393 - [Docs] Note image preprocessing difference between qwen_vl_utils and vllm.

- 链接: https://github.com/vllm-project/vllm/pull/43393
- 状态/时间: merged / 2026-05-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+10/-6，可读 patch 54 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Note image preprocessing difference between qwen_vl_utils and vllm.」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/models/supported_models.md`, `docs/models/pooling_models/embed.md`, `docs/models/pooling_models/scoring.md`；技术摘要: 覆盖「[Docs] Note image preprocessing difference between qwen_vl_utils and vllm.」；主要实现面是 `docs/models/supported_models.md`, `docs/models/pooling_models/embed.md`, `docs/models/pooling_models/scoring.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/models/supported_models.md` modified +5/-4 (9 lines); hunks: -619,13 +619,13 @@ These models primarily accept the [`LLM.generate`](./gener...; -647,6 +647,7 @@ Some models are supported only via the [Transformers modelin...；`docs/models/pooling_models/embed.md` modified +4/-1 (5 lines); hunks: -91,7 +91,7 @@ You can compute pairwise similarity scores to build a similari...; -102,6 +102,9 @@ If your model is not in the above list, we will try to autom...；`docs/models/pooling_models/scoring.md` modified +1/-1 (2 lines); hunks: -103,7 +103,7 @@ The three supported scoring functions are as illustrated in...。
- 代码 diff 细节:
  - `docs/models/supported_models.md` modified +5/-4 (9 lines); hunks: -619,13 +619,13 @@ These models primarily accept the [`LLM.generate`](./gener...; -647,6 +647,7 @@ Some models are supported only via the [Transformers modelin...
  - `docs/models/pooling_models/embed.md` modified +4/-1 (5 lines); hunks: -91,7 +91,7 @@ You can compute pairwise similarity scores to build a similari...; -102,6 +102,9 @@ If your model is not in the above list, we will try to autom...
  - `docs/models/pooling_models/scoring.md` modified +1/-1 (2 lines); hunks: -103,7 +103,7 @@ The three supported scoring functions are as illustrated in...
- 关键代码摘录:

```diff
diff -- docs/models/supported_models.md
@@ -619,13 +619,13 @@ These models primarily accept the [`LLM.generate`](./generative_models.md#llmgen
-| `Qwen2VLForConditionalGeneration` | QVQ, Qwen2-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/QVQ-72B-Preview`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2-VL-72B-Instruct`, etc.
-| `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`, etc. | ✅︎ | ✅︎ |
+| `Qwen2VLForConditionalGeneration` <sup>Q</sup> | QVQ, Qwen2-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/QVQ-72B-Preview`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2-VL-72B-In
+| `Qwen2_5_VLForConditionalGeneration` <sup>Q</sup> | Qwen2.5-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`, etc. | ✅︎ |
-| `Qwen3VLForConditionalGeneration` | Qwen3-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-4B-Instruct`, etc. | ✅︎ | ✅︎ |
-| `Qwen3VLMoeForConditionalGeneration` | Qwen3-VL-MOE | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-30B-A3B-Instruct`, etc. | ✅︎ | ✅︎ |
diff -- docs/models/pooling_models/embed.md
@@ -91,7 +91,7 @@ You can compute pairwise similarity scores to build a similarity matrix using th
-| `Qwen3VLForConditionalGeneration`<sup>C</sup> | Qwen3-VL | T + I + V | `Qwen/Qwen3-VL-Embedding-2B`, etc. | ✅︎ | ✅︎ |
+| `Qwen3VLForConditionalGeneration`<sup>C</sup> (see note) | Qwen3-VL | T + I + V | `Qwen/Qwen3-VL-Embedding-2B`, etc. | ✅︎ | ✅︎ |
@@ -102,6 +102,9 @@ If your model is not in the above list, we will try to automatically convert the
+!!! note
+    `Qwen3-VL-Embedding` officially uses `qwen_vl_utils` for image preprocessing, while vLLM uses `transformers`' `video_processing_qwen3_vl`, which leads to slightly different re
diff -- docs/models/pooling_models/scoring.md
@@ -103,7 +103,7 @@ The three supported scoring functions are as illustrated in the figure below.
-    Similar to Qwen3-Reranker, you need to use the following `--hf_overrides` to load the official original `Qwen3-VL-Reranker`.
```

- 已读文件:
  - docs: `docs/models/supported_models.md` modified +5/-4; `docs/models/pooling_models/embed.md` modified +4/-1; `docs/models/pooling_models/scoring.md` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs/models/pooling_models/embed.md`, `docs/models/pooling_models/scoring.md`, `docs/models/supported_models.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #45676 - [Docs] Update the online serving docs.

- 链接: https://github.com/vllm-project/vllm/pull/45676
- 状态/时间: merged / 2026-06-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+83/-37，可读 patch 216 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Update the online serving docs.」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/models/pooling_models/scoring.md`, `docs/models/pooling_models/README.md`, `docs/serving/online_serving/README.md`；技术摘要: 覆盖「[Docs] Update the online serving docs.」；主要实现面是 `docs/models/pooling_models/scoring.md`, `docs/models/pooling_models/README.md`, `docs/serving/online_serving/README.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/models/pooling_models/scoring.md` modified +2/-2 (4 lines); hunks: -19,7 +19,7 @@ The score models is designed to compute similarity scores betw...; -157,7 +157,7 @@ A code example can be found here: [examples/basic/offline_in...；`docs/models/pooling_models/README.md` modified +1/-1 (2 lines); hunks: -184,7 +184,7 @@ Our online Server provides endpoints that correspond to the...；`docs/serving/online_serving/README.md` modified +77/-32 (109 lines); hunks: -9,12 +9,13 @@ We currently support the following OpenAI APIs:; -24,7 +25,7 @@ We currently support the following OpenAI APIs:；`docs/serving/online_serving/openai_compatible_server.md` modified +3/-2 (5 lines); hunks: -9,12 +9,13 @@ We currently support the following OpenAI APIs:。
- 代码 diff 细节:
  - `docs/models/pooling_models/scoring.md` modified +2/-2 (4 lines); hunks: -19,7 +19,7 @@ The score models is designed to compute similarity scores betw...; -157,7 +157,7 @@ A code example can be found here: [examples/basic/offline_in...
  - `docs/models/pooling_models/README.md` modified +1/-1 (2 lines); hunks: -184,7 +184,7 @@ Our online Server provides endpoints that correspond to the...
  - `docs/serving/online_serving/README.md` modified +77/-32 (109 lines); hunks: -9,12 +9,13 @@ We currently support the following OpenAI APIs:; -24,7 +25,7 @@ We currently support the following OpenAI APIs:
  - `docs/serving/online_serving/openai_compatible_server.md` modified +3/-2 (5 lines); hunks: -9,12 +9,13 @@ We currently support the following OpenAI APIs:
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models/scoring.md
@@ -19,7 +19,7 @@ The score models is designed to compute similarity scores between two input prom
-    - [Score API](scoring.md#score-api) (`/score`)
+    - [Score API](scoring.md#score-api) (`/score`, `/v1/score`)
@@ -157,7 +157,7 @@ A code example can be found here: [examples/basic/offline_inference/score.py](..
-Our Score API (`/score`) is similar to `LLM.score`, compute similarity scores between two input prompts.
+Our Score API (`/score`, `/v1/score`) is similar to `LLM.score`, compute similarity scores between two input prompts.
diff -- docs/models/pooling_models/README.md
@@ -184,7 +184,7 @@ Our online Server provides endpoints that correspond to the offline APIs:
-    - [Score API](scoring.md#score-api)(`/score`)
+    - [Score API](scoring.md#score-api) (`/score`, `/v1/score`)
diff -- docs/serving/online_serving/README.md
@@ -9,12 +9,13 @@ We currently support the following OpenAI APIs:
-- [Responses API](./openai_compatible_server.md#responses-api) (`/v1/responses`)
-    - Only applicable to [text generation models](../../models/generative_models.md).
+- [Chat Completions batch API](./openai_compatible_server.md#chat-api) (`/v1/chat/completions/batch`)
+- [Responses API](./openai_compatible_server.md#responses-api) (`/v1/responses`, `/v1/responses/{response_id}`, `/v1/responses/{response_id}/cancel`)
+    - Only applicable to [text generation models](../../models/generative_models.md).
```

- 已读文件:
  - docs: `docs/models/pooling_models/scoring.md` modified +2/-2; `docs/models/pooling_models/README.md` modified +1/-1; `docs/serving/online_serving/README.md` modified +77/-32; `docs/serving/online_serving/openai_compatible_server.md` modified +3/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs/models/pooling_models/README.md`, `docs/models/pooling_models/scoring.md`, `docs/serving/online_serving/README.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #46398 - [Doc] Fix typos, grammar, and broken commands across docs

- 链接: https://github.com/vllm-project/vllm/pull/46398
- 状态/时间: merged / 2026-06-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+16/-18，可读 patch 149 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc] Fix typos, grammar, and broken commands across docs」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `docs/models/pooling_models/README.md`, `docs/models/pooling_models/scoring.md`, `docs/benchmarking/cli.md`；技术摘要: 覆盖「[Doc] Fix typos, grammar, and broken commands across docs」；主要实现面是 `docs/models/pooling_models/README.md`, `docs/models/pooling_models/scoring.md`, `docs/benchmarking/cli.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/models/pooling_models/README.md` modified +2/-2 (4 lines); hunks: -143,7 +143,7 @@ enabling the corresponding APIs.; -302,7 +302,7 @@ Pooling models now support token-wise task.；`docs/models/pooling_models/scoring.md` modified +1/-1 (2 lines); hunks: -440,7 +440,7 @@ More examples can be found here: [examples/pooling/score](.....；`docs/benchmarking/cli.md` modified +4/-4 (8 lines); hunks: -338,7 +338,7 @@ vllm bench serve \; -352,7 +352,7 @@ vllm bench serve \；`docs/design/cuda_graphs.md` modified +2/-2 (4 lines); hunks: -161,11 +161,11 @@ class AttentionCGSupport(enum.Enum):; symbols: AttentionCGSupport，涉及 `AttentionCGSupport`。
- 代码 diff 细节:
  - `docs/models/pooling_models/README.md` modified +2/-2 (4 lines); hunks: -143,7 +143,7 @@ enabling the corresponding APIs.; -302,7 +302,7 @@ Pooling models now support token-wise task.
  - `docs/models/pooling_models/scoring.md` modified +1/-1 (2 lines); hunks: -440,7 +440,7 @@ More examples can be found here: [examples/pooling/score](.....
  - `docs/benchmarking/cli.md` modified +4/-4 (8 lines); hunks: -338,7 +338,7 @@ vllm bench serve \; -352,7 +352,7 @@ vllm bench serve \
  - `docs/design/cuda_graphs.md` modified +2/-2 (4 lines); hunks: -161,11 +161,11 @@ class AttentionCGSupport(enum.Enum):; symbols: AttentionCGSupport
  - `docs/design/prefix_caching.md` modified +2/-2 (4 lines); hunks: -27,7 +27,7 @@ In the example above, the KV cache in the first block can be u...; -197,7 +197,7 @@ As can be seen, block 3 is a new full block and is cached. H...
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models/README.md
@@ -143,7 +143,7 @@ enabling the corresponding APIs.
-For more information about `LLM.embed`, see [this page](classify.md#offline-inference).
+For more information about `LLM.classify`, see [this page](classify.md#offline-inference).
@@ -302,7 +302,7 @@ Pooling models now support token-wise task.
-`score` task have has been removed in v0.21, use `classify` instead. Only when a classification model outputs num_labels
+`score` task has been removed in v0.21, use `classify` instead. Only when a classification model outputs num_labels
diff -- docs/models/pooling_models/scoring.md
@@ -440,7 +440,7 @@ More examples can be found here: [examples/pooling/score](../../../examples/pool
-AS cross-encoder models are a subset of classification models that accept two prompts as input and output num_labels equal to 1, cross-encoder features should be consistent with (
+As cross-encoder models are a subset of classification models that accept two prompts as input and output num_labels equal to 1, cross-encoder features should be consistent with (
diff -- docs/benchmarking/cli.md
@@ -338,7 +338,7 @@ vllm bench serve \
-    --num-prompts -1
+    --num-prompts -1 \
@@ -352,7 +352,7 @@ vllm bench serve \
-curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \| python3 -
+curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py | python3 -
```

- 已读文件:
  - docs: `docs/models/pooling_models/README.md` modified +2/-2; `docs/models/pooling_models/scoring.md` modified +1/-1; `docs/benchmarking/cli.md` modified +4/-4; `docs/design/cuda_graphs.md` modified +2/-2; `docs/design/prefix_caching.md` modified +2/-2; `docs/configuration/optimization.md` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs/benchmarking/cli.md`, `docs/configuration/optimization.md`, `docs/design/cuda_graphs.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
