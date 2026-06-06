# vllm Jina Reranker M0 模型 PR 优化历史

## 2026-06-05 PR 补漏复核

已于 2026-06-05 按 vllm 上游 `origin/main@c66b19800` 复核；自上次时效基准（2026-05-19）以来，共有 1 个带 PR 编号的合并改动到所跟踪的实现文件，这些 PR 尚未并入下方时间线 / 逐 PR diff 审计卡，应在下次完整重生成时补齐。

| 合并日期 | PR | 标题 | 改动到的跟踪文件 |
| --- | --- | --- | --- |
| 2026-05-22 | [#43393](https://github.com/vllm-project/vllm/pull/43393) | [Docs] Note image preprocessing difference between qwen_vl_utils and vllm. | `scoring.md` |


## 2026-05-19 新增覆盖

按 vllm 上游 `origin/main@ef54a4d604`、模型相关文件的 `git log --name-only -- <model-files>` 以及 GitHub Pull Request files API 生成。本页用于补齐 sgl-cookbook 中 `Jina Reranker M0` 缺失的历史 PR 优化文档。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `vllm/model_executor/models/jina_vl.py` | [#33063](https://github.com/vllm-project/vllm/pull/33063), [#31973](https://github.com/vllm-project/vllm/pull/31973), [#31669](https://github.com/vllm-project/vllm/pull/31669), [#29802](https://github.com/vllm-project/vllm/pull/29802), [#25370](https://github.com/vllm-project/vllm/pull/25370), [#26633](https://github.com/vllm-project/vllm/pull/26633), [#26247](https://github.com/vllm-project/vllm/pull/26247), [#23810](https://github.com/vllm-project/vllm/pull/23810), [#24031](https://github.com/vllm-project/vllm/pull/24031), [#20538](https://github.com/vllm-project/vllm/pull/20538), [#21227](https://github.com/vllm-project/vllm/pull/21227), [#21058](https://github.com/vllm-project/vllm/pull/21058), ... (13 total) |
| `vllm/model_executor/models/jina.py` | [#39575](https://github.com/vllm-project/vllm/pull/39575), [#38800](https://github.com/vllm-project/vllm/pull/38800) |
| `examples/pooling/score/vision_reranker_offline.py` | [#37902](https://github.com/vllm-project/vllm/pull/37902), [#31127](https://github.com/vllm-project/vllm/pull/31127), [#32395](https://github.com/vllm-project/vllm/pull/32395), [#32085](https://github.com/vllm-project/vllm/pull/32085) |
| `examples/pooling/score/vision_rerank_api_online.py` | [#31127](https://github.com/vllm-project/vllm/pull/31127), [#33060](https://github.com/vllm-project/vllm/pull/33060), [#33298](https://github.com/vllm-project/vllm/pull/33298), [#32577](https://github.com/vllm-project/vllm/pull/32577), [#32085](https://github.com/vllm-project/vllm/pull/32085) |
| `examples/pooling/score/vision_score_api_online.py` | [#31127](https://github.com/vllm-project/vllm/pull/31127), [#33060](https://github.com/vllm-project/vllm/pull/33060), [#33298](https://github.com/vllm-project/vllm/pull/33298), [#32577](https://github.com/vllm-project/vllm/pull/32577), [#32085](https://github.com/vllm-project/vllm/pull/32085) |
| `tests/models/multimodal/pooling/test_jinavl_reranker.py` | [#30566](https://github.com/vllm-project/vllm/pull/30566), [#28631](https://github.com/vllm-project/vllm/pull/28631), [#33837](https://github.com/vllm-project/vllm/pull/33837), [#32287](https://github.com/vllm-project/vllm/pull/32287), [#32395](https://github.com/vllm-project/vllm/pull/32395), [#31445](https://github.com/vllm-project/vllm/pull/31445), [#26633](https://github.com/vllm-project/vllm/pull/26633), [#26247](https://github.com/vllm-project/vllm/pull/26247), [#21470](https://github.com/vllm-project/vllm/pull/21470), [#20996](https://github.com/vllm-project/vllm/pull/20996), [#20907](https://github.com/vllm-project/vllm/pull/20907), [#20260](https://github.com/vllm-project/vllm/pull/20260) |
| `docs/models/pooling_models/scoring.md` | [#41907](https://github.com/vllm-project/vllm/pull/41907), [#42626](https://github.com/vllm-project/vllm/pull/42626), [#42267](https://github.com/vllm-project/vllm/pull/42267), [#41832](https://github.com/vllm-project/vllm/pull/41832), [#39675](https://github.com/vllm-project/vllm/pull/39675), [#37537](https://github.com/vllm-project/vllm/pull/37537), [#35592](https://github.com/vllm-project/vllm/pull/35592) |

## PR 覆盖总览

- git 追溯 PR 数: 37
- 关键词/补充 PR 数: 0
- 当前文档总 PR 数: 37
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-10 | [#20260](https://github.com/vllm-project/vllm/pull/20260) | merged | [Model][VLM] Support JinaVL Reranker | `.buildkite/test-pipeline.yaml`, `docs/models/supported_models.md`, `docs/serving/openai_compatible_server.md` |
| 2025-07-14 | [#20907](https://github.com/vllm-project/vllm/pull/20907) | merged | [CI/Build] Fix OOM issue in Jina-VL test | `tests/models/multimodal/pooling/test_jinavl_reranker.py` |
| 2025-07-17 | [#21058](https://github.com/vllm-project/vllm/pull/21058) | merged | [Model] Update pooling model interface | `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`, `vllm/entrypoints/openai/protocol.py`, `vllm/model_executor/layers/pooler.py` |
| 2025-07-21 | [#20996](https://github.com/vllm-project/vllm/pull/20996) | merged | [Misc] unify variable for LLM instance | `docs/configuration/model_resolution.md`, `docs/features/lora.md`, `docs/features/quantization/fp8.md` |
| 2025-07-21 | [#21227](https://github.com/vllm-project/vllm/pull/21227) | merged | [Model][1/N] Support multiple poolers at model level | `docs/models/pooling_models.md`, `tests/models/test_transformers.py`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` |
| 2025-07-28 | [#21470](https://github.com/vllm-project/vllm/pull/21470) | merged | [Deprecation][2/N] Replace `--task` with `--runner` and `--convert` | `docs/features/multimodal_inputs.md`, `docs/features/prompt_embeds.md`, `docs/models/generative_models.md` |
| 2025-08-05 | [#20538](https://github.com/vllm-project/vllm/pull/20538) | merged | [Model] Pooling model activation supports per request control by PoolingParams | `tests/entrypoints/llm/test_classify.py`, `tests/entrypoints/llm/test_embedding.py`, `tests/entrypoints/llm/test_reward.py` |
| 2025-09-02 | [#24031](https://github.com/vllm-project/vllm/pull/24031) | merged | [Model] Classification models support logit_bias / sigmoid_normalize | `vllm/config/__init__.py`, `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/models/config.py` |
| 2025-09-09 | [#23810](https://github.com/vllm-project/vllm/pull/23810) | merged | [Model] Systematic support for fp32 head, pooling models part | `tests/models/language/pooling/mteb_utils.py`, `tests/models/language/pooling/test_bge_reranker_v2_gemma.py`, `vllm/config/__init__.py` |
| 2025-10-05 | [#26247](https://github.com/vllm-project/vllm/pull/26247) | merged | Convert formatting to use `ruff` instead of `yapf` + `isort` | `.buildkite/pyproject.toml`, `.pre-commit-config.yaml`, `benchmarks/benchmark_block_pool.py` |
| 2025-10-12 | [#26633](https://github.com/vllm-project/vllm/pull/26633) | merged | Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y` | `benchmarks/backend_request_func.py`, `benchmarks/benchmark_prefix_caching.py`, `benchmarks/benchmark_prioritization.py` |
| 2025-10-15 | [#25370](https://github.com/vllm-project/vllm/pull/25370) | merged | [Model][2/N] Improve all pooling task | Support multi-vector retrieval | `examples/offline_inference/pooling/README.md`, `examples/offline_inference/pooling/multi_vector_retrieval.py`, `examples/offline_inference/prithvi_geospatial_mae_io_processor.py` |
| 2025-12-02 | [#29802](https://github.com/vllm-project/vllm/pull/29802) | merged | Fix some Transformers nightly tests | `vllm/model_executor/models/jina_vl.py`, `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/qwen2.py` |
| 2025-12-29 | [#31445](https://github.com/vllm-project/vllm/pull/31445) | merged | [Bugfix][Frontend] Fix Jina reranker multimodal input compatibility | `tests/models/multimodal/pooling/test_jinavl_reranker.py`, `vllm/entrypoints/score_utils.py` |
| 2026-01-05 | [#31669](https://github.com/vllm-project/vllm/pull/31669) | merged | [Misc][Model][Refactor] Pass the prefix into Linear layers | `vllm/model_executor/layers/mamba/mamba_mixer.py`, `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/gpt_neox.py` |
| 2026-01-09 | [#31973](https://github.com/vllm-project/vllm/pull/31973) | merged | [Model] Reorganize pooling layers | `.github/CODEOWNERS`, `tests/model_executor/test_model_load_with_params.py`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` |
| 2026-01-12 | [#32085](https://github.com/vllm-project/vllm/pull/32085) | merged | [Model] Improve multimodal pooling examples | `docs/serving/openai_compatible_server.md`, `examples/pooling/embed/vision_embedding_offline.py`, `examples/pooling/embed/vision_embedding_online.py` |
| 2026-01-16 | [#32395](https://github.com/vllm-project/vllm/pull/32395) | merged | [Frontend][1/n] Make pooling entrypoints request schema consensus | CompletionRequest  | `docs/serving/openai_compatible_server.md`, `examples/pooling/classify/classification_online.py`, `examples/pooling/classify/openai_classification_client.py` |
| 2026-01-19 | [#32577](https://github.com/vllm-project/vllm/pull/32577) | merged | [Frontend] Score entrypoint support data_1 & data_2 and queries & documents as inputs | `docs/serving/openai_compatible_server.md`, `examples/offline_inference/basic/score.py`, `examples/offline_inference/openai_batch/README.md` |
| 2026-01-22 | [#32287](https://github.com/vllm-project/vllm/pull/32287) | merged | Upgrade transformers-4.57.5 | `requirements/nightly_torch_test.txt`, `requirements/test.in`, `requirements/test.txt` |
| 2026-01-26 | [#33063](https://github.com/vllm-project/vllm/pull/33063) | merged | [Chore] Update type annotation of `input_ids` in model forward | `docs/contributing/model/basic.md`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`, `vllm/model_executor/models/afmoe.py` |
| 2026-01-29 | [#33298](https://github.com/vllm-project/vllm/pull/33298) | merged | [Bugfix] Fix Qwen3-VL-Reranker load. | `examples/pooling/score/vision_rerank_api_online.py`, `examples/pooling/score/vision_score_api_online.py`, `tests/entrypoints/pooling/classify/test_online_vision.py` |
| 2026-02-04 | [#33060](https://github.com/vllm-project/vllm/pull/33060) | merged | [Frontend][4/n] Make pooling entrypoints request schema consensus | ScoreRequest | `examples/pooling/score/vision_rerank_api_online.py`, `examples/pooling/score/vision_score_api_online.py`, `tests/entrypoints/pooling/score/test_online_score.py` |
| 2026-02-05 | [#33837](https://github.com/vllm-project/vllm/pull/33837) | merged | [Bugfix] Fix ScoreMultiModalParam multi-document scoring returning single result | `tests/models/multimodal/pooling/test_jinavl_reranker.py` |
| 2026-02-09 | [#31127](https://github.com/vllm-project/vllm/pull/31127) | merged | [Frontend][last/5] Make pooling entrypoints request schema consensus.  | `.buildkite/test-amd.yaml`, `.buildkite/test-pipeline.yaml`, `.buildkite/test_areas/misc.yaml` |
| 2026-03-19 | [#35592](https://github.com/vllm-project/vllm/pull/35592) | merged | [Docs] Reorganize pooling docs. | `.github/CODEOWNERS`, `docs/.nav.yml`, `docs/contributing/model/tests.md` |
| 2026-03-20 | [#37537](https://github.com/vllm-project/vllm/pull/37537) | merged | [Model] Deprecate the score task (this will not affect users).  | `docs/models/pooling_models/README.md`, `docs/models/pooling_models/classify.md`, `docs/models/pooling_models/scoring.md` |
| 2026-03-25 | [#37902](https://github.com/vllm-project/vllm/pull/37902) | merged | [Mypy] Better fixes for the `mypy` issues in `vllm/config` | `benchmarks/benchmark_long_document_qa_throughput.py`, `benchmarks/benchmark_prefix_caching.py`, `benchmarks/benchmark_prioritization.py` |
| 2026-03-31 | [#28631](https://github.com/vllm-project/vllm/pull/28631) | merged | [Frontend][3/n] Improve pooling entrypoints | scoring. | `tests/entrypoints/openai/utils.py`, `tests/entrypoints/pooling/classify/test_offline.py`, `tests/entrypoints/pooling/classify/test_online.py` |
| 2026-04-10 | [#38800](https://github.com/vllm-project/vllm/pull/38800) | merged | [New Model]: jinaai/jina-reranker-v3 | `docs/models/pooling_models/token_embed.md`, `examples/pooling/token_embed/jina_reranker_v3_offline.py`, `pyproject.toml` |
| 2026-04-15 | [#30566](https://github.com/vllm-project/vllm/pull/30566) | merged | Update to transformers v5 | `.buildkite/scripts/hardware_ci/run-cpu-test.sh`, `.buildkite/test_areas/models_basic.yaml`, `docker/Dockerfile` |
| 2026-04-16 | [#39575](https://github.com/vllm-project/vllm/pull/39575) | merged | Add Jina Embeddings v5 model support (fixes #38633) | `docs/models/pooling_models/embed.md`, `tests/conftest.py`, `tests/models/language/pooling_mteb_test/mteb_embed_utils.py` |
| 2026-04-16 | [#39675](https://github.com/vllm-project/vllm/pull/39675) | merged | [Frontend][last/5] Improve pooling entrypoints | clean up. | `docs/models/pooling_models/README.md`, `docs/models/pooling_models/scoring.md`, `docs/models/pooling_models/token_classify.md` |
| 2026-05-06 | [#41832](https://github.com/vllm-project/vllm/pull/41832) | merged | [Doc] Add ModernBertForSequenceClassification to scoring.md cross-en… | `docs/models/pooling_models/scoring.md` |
| 2026-05-15 | [#42267](https://github.com/vllm-project/vllm/pull/42267) | merged | [Entrypoints] Split the pooling offline API into PoolingOfflineMixin. | `docs/models/pooling_models/README.md`, `docs/models/pooling_models/classify.md`, `docs/models/pooling_models/embed.md` |
| 2026-05-19 | [#41907](https://github.com/vllm-project/vllm/pull/41907) | merged | [Docs] Reorganize online serving docs. | `docs/.nav.yml`, `docs/assets/models/pooling_models/cheat_sheet.svg`, `docs/configuration/README.md` |
| 2026-05-19 | [#42626](https://github.com/vllm-project/vllm/pull/42626) | merged | [Docs] Add SVG images for pooling models. | `docs/assets/models/pooling_models/cheat_sheet.svg`, `docs/assets/models/pooling_models/pooling_types.svg`, `docs/assets/models/pooling_models/score_types.svg` |

## 逐 PR diff 审计卡

### PR #20260 - [Model][VLM] Support JinaVL Reranker

- 链接: https://github.com/vllm-project/vllm/pull/20260
- 状态/时间: merged / 2025-07-10
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+993/-133，可读 patch 1479 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][VLM] Support JinaVL Reranker」；模型线: Jina Reranker M0；类别: 模型支持/运行时入口；主要 diff: `.buildkite/test-pipeline.yaml`, `docs/models/supported_models.md`, `docs/serving/openai_compatible_server.md`。
- 实现要点:
  - `.buildkite/test-pipeline.yaml` modified +1/-1
  - `docs/models/supported_models.md` modified +8/-0
  - `docs/serving/openai_compatible_server.md` modified +51/-3
  - `examples/offline_inference/vision_language_pooling.py` renamed +89/-7；symbols: TextImagesQuery, run_jinavl_reranker, run_score
- 代码 diff 细节:
  - `.buildkite/test-pipeline.yaml` modified +1/-1
  - `docs/models/supported_models.md` modified +8/-0
  - `docs/serving/openai_compatible_server.md` modified +51/-3
  - `examples/offline_inference/vision_language_pooling.py` renamed +89/-7
- 关键代码摘录:

```diff
diff -- .buildkite/test-pipeline.yaml
@@ -282,7 +282,7 @@ steps:
     - python3 offline_inference/llm_engine_example.py
     - python3 offline_inference/audio_language.py --seed 0
     - python3 offline_inference/vision_language.py --seed 0
-    - python3 offline_inference/vision_language_embedding.py --seed 0
+    - python3 offline_inference/vision_language_pooling.py --seed 0
     - python3 offline_inference/vision_language_multi_image.py --seed 0
     - VLLM_USE_V1=0 python3 others/tensorize_vllm_model.py --model facebook/opt-125m serialize --serialized-directory /tmp/ --suffix v1 && python3 others/tensorize_vllm_model.py --model facebook/opt-125m deserialize --path-to-tensors /tmp/vllm/facebook/opt-125m/v1/model.tensors
     - python3 offline_inference/encoder_decoder.py
diff -- docs/models/supported_models.md
@@ -712,6 +712,14 @@ The following table lists those that are tested in vLLM.

 ---

+#### Scoring
+
+Specified using `--task score`.
+
+| Architecture                        | Models             | Inputs   | Example HF Models        | [LoRA][lora-adapter]   | [PP][distributed-serving]   | [V1](gh-issue:8779)   |
+|-------------------------------------|--------------------|----------|--------------------------|------------------------|-----------------------------|-----------------------|
+| `JinaVLForSequenceClassification` | JinaVL-based | T + I<sup>E+</sup> | `jinaai/jina-reranker-m0`, etc. | | | ✅︎ |
+
 ## Model Support Policy

```
- 已读文件:
  - runtime: `vllm/entrypoints/llm.py` modified +122/-61; `vllm/entrypoints/openai/protocol.py` modified +19/-5; `vllm/entrypoints/openai/serving_score.py` modified +110/-48; `vllm/entrypoints/score_utils.py` modified +152/-8; `vllm/model_executor/models/config.py` modified +10/-0; `vllm/model_executor/models/interfaces.py` modified +57/-0; `vllm/model_executor/models/jina_vl.py` added +150/-0; `vllm/model_executor/models/registry.py` modified +1/-0
  - tests: `tests/models/multimodal/pooling/test_jinavl_reranker.py` added +160/-0; `tests/models/registry.py` modified +3/-0
  - docs/bench: `.buildkite/test-pipeline.yaml` modified +1/-1; `docs/models/supported_models.md` modified +8/-0; `docs/serving/openai_compatible_server.md` modified +51/-3; `examples/offline_inference/vision_language_pooling.py` renamed +89/-7; `examples/online_serving/openai_cross_encoder_score_for_multimodal.py` added +60/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #20907 - [CI/Build] Fix OOM issue in Jina-VL test

- 链接: https://github.com/vllm-project/vllm/pull/20907
- 状态/时间: merged / 2025-07-14
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+85/-58，可读 patch 225 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI/Build] Fix OOM issue in Jina-VL test」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `tests/models/multimodal/pooling/test_jinavl_reranker.py`。
- 实现要点:
  - `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +85/-58；symbols: vllm_reranker, create_image_param, hf_reranker, test_model_text_image
- 代码 diff 细节:
  - `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +85/-58
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/pooling/test_jinavl_reranker.py
@@ -1,9 +1,15 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from typing import Union

 import pytest
 from transformers import AutoModel

+from vllm.entrypoints.chat_utils import ChatCompletionContentPartImageParam
+from vllm.entrypoints.score_utils import ScoreMultiModalParam
+
+from ....conftest import HfRunner, VllmRunner
+
 model_name = "jinaai/jina-reranker-m0"
```
- 已读文件:
  - tests: `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +85/-58
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #21058 - [Model] Update pooling model interface

- 链接: https://github.com/vllm-project/vllm/pull/21058
- 状态/时间: merged / 2025-07-17
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+247/-345，可读 patch 1411 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Update pooling model interface」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`, `vllm/entrypoints/openai/protocol.py`, `vllm/model_executor/layers/pooler.py`。
- 实现要点:
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +5/-10
  - `vllm/entrypoints/openai/protocol.py` modified +5/-29
  - `vllm/model_executor/layers/pooler.py` modified +112/-64；symbols: Pooler, from_config_with_defaults, get_pooling_params, forward
  - `vllm/model_executor/models/adapters.py` modified +8/-23
- 代码 diff 细节:
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +5/-10
  - `vllm/entrypoints/openai/protocol.py` modified +5/-29
  - `vllm/model_executor/layers/pooler.py` modified +112/-64
  - `vllm/model_executor/models/adapters.py` modified +8/-23
- 关键代码摘录:

```diff
diff -- tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py
@@ -11,11 +11,13 @@
 from vllm.model_executor.layers.pooler import Pooler, PoolingType
 from vllm.model_executor.models.gemma2 import Gemma2Model
 from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
-from vllm.model_executor.pooling_metadata import PoolingMetadata
-from vllm.sequence import IntermediateTensors, PoolerOutput
+from vllm.sequence import IntermediateTensors


 class MyGemma2Embedding(nn.Module):
+
+    is_pooling_model = True
+
     hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})
diff -- vllm/entrypoints/openai/protocol.py
@@ -1237,10 +1237,6 @@ class EmbeddingCompletionRequest(OpenAIBaseModel):
     user: Optional[str] = None
     truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None

-    # --8<-- [start:embedding-pooling-params]
-    additional_data: Optional[Any] = None
-    # --8<-- [end:embedding-pooling-params]
-
     # --8<-- [start:embedding-extra-params]
     add_special_tokens: bool = Field(
         default=True,
@@ -1259,8 +1255,7 @@ class EmbeddingCompletionRequest(OpenAIBaseModel):
     # --8<-- [end:embedding-extra-params]

```
- 已读文件:
  - runtime: `vllm/entrypoints/openai/protocol.py` modified +5/-29; `vllm/model_executor/layers/pooler.py` modified +112/-64; `vllm/model_executor/models/adapters.py` modified +8/-23; `vllm/model_executor/models/bert.py` modified +18/-19; `vllm/model_executor/models/gpt2.py` modified +4/-10; `vllm/model_executor/models/gritlm.py` modified +3/-9; `vllm/model_executor/models/interfaces.py` modified +12/-74; `vllm/model_executor/models/interfaces_base.py` modified +16/-17
  - tests: `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +5/-10
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #20996 - [Misc] unify variable for LLM instance

- 链接: https://github.com/vllm-project/vllm/pull/20996
- 状态/时间: merged / 2025-07-21
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 53 个文件，+237/-236，可读 patch 1417 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] unify variable for LLM instance」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/configuration/model_resolution.md`, `docs/features/lora.md`, `docs/features/quantization/fp8.md`。
- 实现要点:
  - `docs/configuration/model_resolution.md` modified +1/-1
  - `docs/features/lora.md` modified +2/-2
  - `docs/features/quantization/fp8.md` modified +6/-4
  - `docs/features/quantization/int4.md` modified +2/-1
- 代码 diff 细节:
  - `docs/configuration/model_resolution.md` modified +1/-1
  - `docs/features/lora.md` modified +2/-2
  - `docs/features/quantization/fp8.md` modified +6/-4
  - `docs/features/quantization/int4.md` modified +2/-1
- 关键代码摘录:

```diff
diff -- docs/configuration/model_resolution.md
@@ -14,7 +14,7 @@ For example:
 ```python
 from vllm import LLM

-model = LLM(
+llm = LLM(
     model="cerebras/Cerebras-GPT-1.3B",
     hf_overrides={"architectures": ["GPT2LMHeadModel"]},  # GPT-2
 )
diff -- docs/features/lora.md
@@ -302,7 +302,7 @@ To this end, we allow registration of default multimodal LoRAs to handle this au
         return tokenizer.apply_chat_template(chat, tokenize=False)


-    model = LLM(
+    llm = LLM(
         model=model_id,
         enable_lora=True,
         max_lora_rank=64,
@@ -329,7 +329,7 @@ To this end, we allow registration of default multimodal LoRAs to handle this au
     }


-    outputs = model.generate(
```
- 已读文件:
  - tests: `tests/basic_correctness/test_basic_correctness.py` modified +2/-2; `tests/basic_correctness/test_preemption.py` modified +5/-5; `tests/conftest.py` modified +16/-16; `tests/core/test_num_computed_tokens_update.py` modified +1/-1; `tests/detokenizer/test_stop_reason.py` modified +1/-1; `tests/detokenizer/test_stop_strings.py` modified +21/-21; `tests/lora/test_llama_tp.py` modified +10/-10; `tests/metrics/test_metrics.py` modified +7/-7
  - docs/bench: `docs/configuration/model_resolution.md` modified +1/-1; `docs/features/lora.md` modified +2/-2; `docs/features/quantization/fp8.md` modified +6/-4; `docs/features/quantization/int4.md` modified +2/-1; `docs/features/quantization/int8.md` modified +2/-1; `docs/models/pooling_models.md` modified +5/-5; `examples/offline_inference/basic/classify.py` modified +2/-2; `examples/offline_inference/basic/embed.py` modified +2/-2
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #21227 - [Model][1/N] Support multiple poolers at model level

- 链接: https://github.com/vllm-project/vllm/pull/21227
- 状态/时间: merged / 2025-07-21
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 22 个文件，+550/-414，可读 patch 1581 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][1/N] Support multiple poolers at model level」；模型线: Jina Reranker M0；类别: 模型支持/运行时入口；主要 diff: `docs/models/pooling_models.md`, `tests/models/test_transformers.py`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`。
- 实现要点:
  - `docs/models/pooling_models.md` modified +39/-14
  - `tests/models/test_transformers.py` modified +1/-1
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +8/-7
  - `vllm/config.py` modified +4/-4
- 代码 diff 细节:
  - `docs/models/pooling_models.md` modified +39/-14
  - `tests/models/test_transformers.py` modified +1/-1
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +8/-7
  - `vllm/config.py` modified +4/-4
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models.md
@@ -11,26 +11,51 @@ before returning them.
     As shown in the [Compatibility Matrix](../features/compatibility_matrix.md), most vLLM features are not applicable to
     pooling models as they only work on the generation or decode stage, so performance may not improve as much.

-For pooling models, we support the following `--task` options.
-The selected option sets the default pooler used to extract the final hidden states:
+If the model doesn't implement this interface, you can set `--task` which tells vLLM
+to convert the model into a pooling model.

-| Task                            | Pooling Type   | Normalization   | Softmax   |
-|---------------------------------|----------------|-----------------|-----------|
-| Embedding (`embed`)             | `LAST`         | ✅︎              | ❌         |
-| Classification (`classify`)     | `LAST`         | ❌               | ✅︎        |
-| Sentence Pair Scoring (`score`) | \*             | \*              | \*        |
diff -- tests/models/test_transformers.py
@@ -144,7 +144,7 @@ def test_quantization(
     "model",
     ["jason9693/Qwen2.5-1.5B-apeach"],
 )
-@pytest.mark.parametrize("dtype", ["half"])
+@pytest.mark.parametrize("dtype", ["float"])
 def test_classify(
     hf_runner,
     vllm_runner,
```
- 已读文件:
  - runtime: `vllm/config.py` modified +4/-4; `vllm/entrypoints/openai/api_server.py` modified +1/-1; `vllm/model_executor/layers/pooler.py` modified +175/-171; `vllm/model_executor/models/adapters.py` modified +51/-57; `vllm/model_executor/models/bert.py` modified +99/-33; `vllm/model_executor/models/gpt2.py` modified +10/-6; `vllm/model_executor/models/gritlm.py` modified +19/-20; `vllm/model_executor/models/internlm2.py` modified +5/-7
  - tests: `tests/models/test_transformers.py` modified +1/-1; `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +8/-7
  - docs/bench: `docs/models/pooling_models.md` modified +39/-14
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #21470 - [Deprecation][2/N] Replace `--task` with `--runner` and `--convert`

- 链接: https://github.com/vllm-project/vllm/pull/21470
- 状态/时间: merged / 2025-07-28
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 94 个文件，+1111/-1077，可读 patch 4435 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Deprecation][2/N] Replace `--task` with `--runner` and `--convert`」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/features/multimodal_inputs.md`, `docs/features/prompt_embeds.md`, `docs/models/generative_models.md`。
- 实现要点:
  - `docs/features/multimodal_inputs.md` modified +2/-2
  - `docs/features/prompt_embeds.md` modified +1/-1
  - `docs/models/generative_models.md` modified +10/-3
  - `docs/models/pooling_models.md` modified +49/-28
- 代码 diff 细节:
  - `docs/features/multimodal_inputs.md` modified +2/-2
  - `docs/features/prompt_embeds.md` modified +1/-1
  - `docs/models/generative_models.md` modified +10/-3
  - `docs/models/pooling_models.md` modified +49/-28
- 关键代码摘录:

```diff
diff -- docs/features/multimodal_inputs.md
@@ -343,7 +343,7 @@ Here is a simple example using Phi-3.5-Vision.
 First, launch the OpenAI-compatible server:

 ```bash
-vllm serve microsoft/Phi-3.5-vision-instruct --task generate \
+vllm serve microsoft/Phi-3.5-vision-instruct --runner generate \
   --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt '{"image":2}'
 ```

@@ -422,7 +422,7 @@ Instead of `image_url`, you can pass a video file via `video_url`. Here is a sim
 First, launch the OpenAI-compatible server:

 ```bash
-vllm serve llava-hf/llava-onevision-qwen2-0.5b-ov-hf --task generate --max-model-len 8192
diff -- docs/features/prompt_embeds.md
@@ -34,7 +34,7 @@ Prompt embeddings are passed in as base64 encoded torch tensors.
 First, launch the OpenAI-compatible server:

 ```bash
-vllm serve meta-llama/Llama-3.2-1B-Instruct --task generate \
+vllm serve meta-llama/Llama-3.2-1B-Instruct --runner generate \
   --max-model-len 4096 --enable-prompt-embeds
 ```

```
- 已读文件:
  - tests: `tests/compile/test_async_tp.py` modified +0/-3; `tests/compile/test_basic_correctness.py` modified +3/-3; `tests/compile/test_fusion_all_reduce.py` modified +0/-3; `tests/compile/test_sequence_parallelism.py` modified +0/-3; `tests/conftest.py` modified +5/-3; `tests/distributed/test_expert_parallel.py` modified +13/-13; `tests/distributed/test_pipeline_parallel.py` modified +22/-22; `tests/distributed/test_sequence_parallel.py` modified +15/-15
  - docs/bench: `docs/features/multimodal_inputs.md` modified +2/-2; `docs/features/prompt_embeds.md` modified +1/-1; `docs/models/generative_models.md` modified +10/-3; `docs/models/pooling_models.md` modified +49/-28; `docs/models/supported_models.md` modified +51/-50; `docs/serving/openai_compatible_server.md` modified +12/-12; `examples/offline_inference/basic/classify.py` modified +4/-2; `examples/offline_inference/basic/embed.py` modified +2/-2
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #20538 - [Model] Pooling model activation supports per request control by PoolingParams

- 链接: https://github.com/vllm-project/vllm/pull/20538
- 状态/时间: merged / 2025-08-05
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 21 个文件，+948/-173，可读 patch 1566 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Pooling model activation supports per request control by PoolingParams」；模型线: Jina Reranker M0；类别: 模型支持/运行时入口；主要 diff: `tests/entrypoints/llm/test_classify.py`, `tests/entrypoints/llm/test_embedding.py`, `tests/entrypoints/llm/test_reward.py`。
- 实现要点:
  - `tests/entrypoints/llm/test_classify.py` added +67/-0；symbols: v1, llm, test_pooling_params, get_outputs
  - `tests/entrypoints/llm/test_embedding.py` added +56/-0；symbols: llm, test_pooling_params, get_outputs
  - `tests/entrypoints/llm/test_reward.py` added +66/-0；symbols: v1, llm, test_pooling_params, get_outputs
  - `tests/entrypoints/llm/test_score.py` added +69/-0；symbols: v1, llm, test_pooling_params, get_outputs
- 代码 diff 细节:
  - `tests/entrypoints/llm/test_classify.py` added +67/-0
  - `tests/entrypoints/llm/test_embedding.py` added +56/-0
  - `tests/entrypoints/llm/test_reward.py` added +66/-0
  - `tests/entrypoints/llm/test_score.py` added +69/-0
- 关键代码摘录:

```diff
diff -- tests/entrypoints/llm/test_classify.py
@@ -0,0 +1,67 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+
+import weakref
+
+import pytest
+import torch
+
+from vllm import LLM, PoolingParams
+from vllm.distributed import cleanup_dist_env_and_memory
+
+from ...models.utils import softmax
+
diff -- tests/entrypoints/llm/test_embedding.py
@@ -0,0 +1,56 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+
+import weakref
+
+import pytest
+import torch
+import torch.nn.functional as F
+
+from vllm import LLM, PoolingParams
+from vllm.distributed import cleanup_dist_env_and_memory
+
+MODEL_NAME = "intfloat/multilingual-e5-small"
```
- 已读文件:
  - runtime: `vllm/config.py` modified +15/-15; `vllm/entrypoints/llm.py` modified +19/-3; `vllm/entrypoints/openai/protocol.py` modified +15/-5; `vllm/model_executor/layers/pooler.py` modified +106/-116; `vllm/model_executor/models/config.py` modified +32/-0; `vllm/model_executor/models/jamba.py` modified +0/-2; `vllm/model_executor/models/jina_vl.py` modified +1/-4; `vllm/model_executor/models/qwen2_rm.py` modified +0/-3
  - tests: `tests/entrypoints/llm/test_classify.py` added +67/-0; `tests/entrypoints/llm/test_embedding.py` added +56/-0; `tests/entrypoints/llm/test_reward.py` added +66/-0; `tests/entrypoints/llm/test_score.py` added +69/-0; `tests/entrypoints/openai/test_classification.py` modified +31/-0; `tests/entrypoints/openai/test_embedding.py` modified +34/-0; `tests/entrypoints/openai/test_rerank.py` modified +38/-0; `tests/entrypoints/openai/test_score.py` modified +41/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #24031 - [Model] Classification models support logit_bias / sigmoid_normalize

- 链接: https://github.com/vllm-project/vllm/pull/24031
- 状态/时间: merged / 2025-09-02
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+38/-30，可读 patch 143 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Classification models support logit_bias / sigmoid_normalize」；模型线: Jina Reranker M0；类别: 模型支持/运行时入口；主要 diff: `vllm/config/__init__.py`, `vllm/model_executor/layers/pooler.py`, `vllm/model_executor/models/config.py`。
- 实现要点:
  - `vllm/config/__init__.py` modified +24/-21
  - `vllm/model_executor/layers/pooler.py` modified +8/-0
  - `vllm/model_executor/models/config.py` modified +3/-1
  - `vllm/model_executor/models/jina_vl.py` modified +3/-8
- 代码 diff 细节:
  - `vllm/config/__init__.py` modified +24/-21
  - `vllm/model_executor/layers/pooler.py` modified +8/-0
  - `vllm/model_executor/models/config.py` modified +3/-1
  - `vllm/model_executor/models/jina_vl.py` modified +3/-8
- 关键代码摘录:

```diff
diff -- vllm/config/__init__.py
@@ -2651,24 +2651,46 @@ class PoolerConfig:
     ## for embeddings models
     normalize: Optional[bool] = None
     """
-    Whether to normalize the embeddings outputs.
+    Whether to normalize the embeddings outputs. Defaults to True.
     """
     dimensions: Optional[int] = None
     """
     Reduce the dimensions of embeddings if model
-    support matryoshka representation.
+    support matryoshka representation. Defaults to None.
+    """
+    enable_chunked_processing: Optional[bool] = None
diff -- vllm/model_executor/layers/pooler.py
@@ -633,9 +633,14 @@ def __init__(
     ) -> None:
         super().__init__()

+        from vllm.config import get_current_vllm_config
+        vllm_config = get_current_vllm_config()
+
         self.pooling = pooling
         self.classifier = classifier
         self.act_fn = act_fn or PoolerClassify()
+        self.logit_bias: Optional[
+            float] = vllm_config.model_config.pooler_config.logit_bias

     def get_supported_tasks(self) -> Set[PoolingTask]:
```
- 已读文件:
  - runtime: `vllm/config/__init__.py` modified +24/-21; `vllm/model_executor/layers/pooler.py` modified +8/-0; `vllm/model_executor/models/config.py` modified +3/-1; `vllm/model_executor/models/jina_vl.py` modified +3/-8
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #23810 - [Model] Systematic support for fp32 head, pooling models part

- 链接: https://github.com/vllm-project/vllm/pull/23810
- 状态/时间: merged / 2025-09-09
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+166/-61，可读 patch 557 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Systematic support for fp32 head, pooling models part」；模型线: Jina Reranker M0；类别: 模型支持/运行时入口；主要 diff: `tests/models/language/pooling/mteb_utils.py`, `tests/models/language/pooling/test_bge_reranker_v2_gemma.py`, `vllm/config/__init__.py`。
- 实现要点:
  - `tests/models/language/pooling/mteb_utils.py` modified +31/-6
  - `tests/models/language/pooling/test_bge_reranker_v2_gemma.py` modified +1/-0
  - `vllm/config/__init__.py` modified +52/-1；symbols: head_dtype, _get_head_dtype
  - `vllm/model_executor/layers/pooler.py` modified +23/-15
- 代码 diff 细节:
  - `tests/models/language/pooling/mteb_utils.py` modified +31/-6
  - `tests/models/language/pooling/test_bge_reranker_v2_gemma.py` modified +1/-0
  - `vllm/config/__init__.py` modified +52/-1
  - `vllm/model_executor/layers/pooler.py` modified +23/-15
- 关键代码摘录:

```diff
diff -- tests/models/language/pooling/mteb_utils.py
@@ -9,6 +9,7 @@
 import numpy as np
 import pytest
 import requests
+import torch

 from tests.models.utils import (EmbedModelInfo, RerankModelInfo,
                                 check_embeddings_close)
@@ -165,16 +166,19 @@ def mteb_test_embed_models(hf_runner,
                            vllm_extra_kwargs=None,
                            hf_model_callback=None,
                            atol=MTEB_EMBED_TOL):
+    # A model family has many models with the same architecture,
+    # and we don't need to test each one.
diff -- tests/models/language/pooling/test_bge_reranker_v2_gemma.py
@@ -14,6 +14,7 @@
 RERANK_MODELS = [
     LASTPoolingRerankModelInfo("BAAI/bge-reranker-v2-gemma",
                                architecture="GemmaForSequenceClassification",
+                               mteb_score=0.33757,
                                hf_overrides={
                                    "architectures":
                                    ["GemmaForSequenceClassification"],
```
- 已读文件:
  - runtime: `vllm/config/__init__.py` modified +52/-1; `vllm/model_executor/layers/pooler.py` modified +23/-15; `vllm/model_executor/models/adapters.py` modified +4/-6; `vllm/model_executor/models/bert.py` modified +3/-1; `vllm/model_executor/models/bert_with_rope.py` modified +8/-8; `vllm/model_executor/models/gpt2.py` modified +6/-4; `vllm/model_executor/models/internlm2.py` modified +11/-8; `vllm/model_executor/models/jamba.py` modified +1/-1
  - tests: `tests/models/language/pooling/mteb_utils.py` modified +31/-6; `tests/models/language/pooling/test_bge_reranker_v2_gemma.py` modified +1/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #26247 - Convert formatting to use `ruff` instead of `yapf` + `isort`

- 链接: https://github.com/vllm-project/vllm/pull/26247
- 状态/时间: merged / 2025-10-05
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+4320/-3882，可读 patch 14041 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Convert formatting to use `ruff` instead of `yapf` + `isort`」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `.buildkite/pyproject.toml`, `.pre-commit-config.yaml`, `benchmarks/benchmark_block_pool.py`。
- 实现要点:
  - `.buildkite/pyproject.toml` removed +0/-46
  - `.pre-commit-config.yaml` modified +0/-12
  - `benchmarks/benchmark_block_pool.py` modified +1/-1
  - `benchmarks/benchmark_ngram_proposer.py` modified +1/-1
- 代码 diff 细节:
  - `.buildkite/pyproject.toml` removed +0/-46
  - `.pre-commit-config.yaml` modified +0/-12
  - `benchmarks/benchmark_block_pool.py` modified +1/-1
  - `benchmarks/benchmark_ngram_proposer.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- .buildkite/pyproject.toml
@@ -1,46 +0,0 @@
-# This local pyproject file is part of the migration from yapf to ruff format.
-# It uses the same core rules as the main pyproject.toml file, but with the
-# following differences:
-# - ruff line length is overridden to 88
-# - deprecated typing ignores (UP006, UP035) have been removed
-
-[tool.ruff]
-line-length = 88
-
-[tool.ruff.lint.per-file-ignores]
-"vllm/third_party/**" = ["ALL"]
-"vllm/version.py" = ["F401"]
-"vllm/_version.py" = ["ALL"]
diff -- .pre-commit-config.yaml
@@ -6,28 +6,16 @@ default_stages:
   - manual # Run in CI
 exclude: 'vllm/third_party/.*'
 repos:
-- repo: https://github.com/google/yapf
-  rev: v0.43.0
-  hooks:
-  - id: yapf
-    args: [--in-place, --verbose]
-    # Keep the same list from yapfignore here to avoid yapf failing without any inputs
-    exclude: '(.buildkite|benchmarks|build|examples)/.*'
 - repo: https://github.com/astral-sh/ruff-pre-commit
   rev: v0.11.7
   hooks:
```
- 已读文件:
  - runtime: `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` modified +13/-15; `csrc/moe/marlin_moe_wna16/generate_kernels.py` modified +24/-18; `csrc/quantization/gptq_marlin/generate_kernels.py` modified +26/-22; `csrc/quantization/machete/generate.py` modified +100/-66
  - tests: `tests/basic_correctness/test_basic_correctness.py` modified +60/-62; `tests/basic_correctness/test_cpu_offload.py` modified +3/-2; `tests/basic_correctness/test_cumem.py` modified +12/-11; `tests/benchmarks/test_latency_cli.py` modified +12/-2; `tests/benchmarks/test_random_dataset.py` modified +69/-52; `tests/benchmarks/test_serve_cli.py` modified +2/-3; `tests/benchmarks/test_throughput_cli.py` modified +12/-2; `tests/compile/backend.py` modified +7/-11
  - docs/bench: `.buildkite/pyproject.toml` removed +0/-46; `benchmarks/benchmark_block_pool.py` modified +1/-1; `benchmarks/benchmark_ngram_proposer.py` modified +1/-1; `benchmarks/benchmark_serving_structured_output.py` modified +2/-3; `benchmarks/pyproject.toml` removed +0/-49; `docs/mkdocs/hooks/generate_argparse.py` modified +27/-36; `docs/mkdocs/hooks/generate_examples.py` modified +18/-16; `docs/mkdocs/hooks/remove_announcement.py` modified +1/-1
  - other: `.pre-commit-config.yaml` modified +0/-12; `cmake/hipify.py` modified +24/-19; `pyproject.toml` modified +100/-27; `setup.py` modified +151/-104
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #26633 - Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y`

- 链接: https://github.com/vllm-project/vllm/pull/26633
- 状态/时间: merged / 2025-10-12
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+481/-581，可读 patch 3511 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y`」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `benchmarks/backend_request_func.py`, `benchmarks/benchmark_prefix_caching.py`, `benchmarks/benchmark_prioritization.py`。
- 实现要点:
  - `benchmarks/backend_request_func.py` modified +13/-14
  - `benchmarks/benchmark_prefix_caching.py` modified +2/-3
  - `benchmarks/benchmark_prioritization.py` modified +1/-2
  - `benchmarks/benchmark_serving_structured_output.py` modified +3/-4
- 代码 diff 细节:
  - `benchmarks/backend_request_func.py` modified +13/-14
  - `benchmarks/benchmark_prefix_caching.py` modified +2/-3
  - `benchmarks/benchmark_prioritization.py` modified +1/-2
  - `benchmarks/benchmark_serving_structured_output.py` modified +3/-4
- 关键代码摘录:

```diff
diff -- benchmarks/backend_request_func.py
@@ -8,7 +8,6 @@
 import time
 import traceback
 from dataclasses import dataclass, field
-from typing import Optional, Union

 import aiohttp
 import huggingface_hub.constants
@@ -28,13 +27,13 @@ class RequestFuncInput:
     prompt_len: int
     output_len: int
     model: str
-    model_name: Optional[str] = None
-    logprobs: Optional[int] = None
diff -- benchmarks/benchmark_prefix_caching.py
@@ -32,7 +32,6 @@
 import json
 import random
 import time
-from typing import Optional

 from transformers import PreTrainedTokenizerBase

@@ -80,7 +79,7 @@ def sample_requests_from_dataset(
     num_requests: int,
     tokenizer: PreTrainedTokenizerBase,
     input_length_range: tuple[int, int],
-    fixed_output_len: Optional[int],
+    fixed_output_len: int | None,
```
- 已读文件:
  - runtime: `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` modified +6/-9; `csrc/quantization/machete/generate.py` modified +2/-3
  - docs/bench: `benchmarks/backend_request_func.py` modified +13/-14; `benchmarks/benchmark_prefix_caching.py` modified +2/-3; `benchmarks/benchmark_prioritization.py` modified +1/-2; `benchmarks/benchmark_serving_structured_output.py` modified +3/-4; `benchmarks/benchmark_utils.py` modified +8/-8; `benchmarks/cutlass_benchmarks/sparse_benchmarks.py` modified +1/-2; `benchmarks/cutlass_benchmarks/w8a8_benchmarks.py` modified +5/-6; `benchmarks/fused_kernels/layernorm_rms_benchmarks.py` modified +4/-5
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #25370 - [Model][2/N] Improve all pooling task | Support multi-vector retrieval

- 链接: https://github.com/vllm-project/vllm/pull/25370
- 状态/时间: merged / 2025-10-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 41 个文件，+786/-399，可读 patch 1862 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][2/N] Improve all pooling task | Support multi-vector retrieval」；模型线: Jina Reranker M0；类别: 模型支持/运行时入口；主要 diff: `examples/offline_inference/pooling/README.md`, `examples/offline_inference/pooling/multi_vector_retrieval.py`, `examples/offline_inference/prithvi_geospatial_mae_io_processor.py`。
- 实现要点:
  - `examples/offline_inference/pooling/README.md` modified +6/-0
  - `examples/offline_inference/pooling/multi_vector_retrieval.py` added +56/-0；symbols: parse_args, main
  - `examples/offline_inference/prithvi_geospatial_mae_io_processor.py` modified +1/-1
  - `examples/online_serving/pooling/README.md` modified +6/-0
- 代码 diff 细节:
  - `examples/offline_inference/pooling/README.md` modified +6/-0
  - `examples/offline_inference/pooling/multi_vector_retrieval.py` added +56/-0
  - `examples/offline_inference/prithvi_geospatial_mae_io_processor.py` modified +1/-1
  - `examples/online_serving/pooling/README.md` modified +6/-0
- 关键代码摘录:

```diff
diff -- examples/offline_inference/pooling/README.md
@@ -26,6 +26,12 @@ python examples/offline_inference/pooling/embed_jina_embeddings_v3.py
 python examples/offline_inference/pooling/embed_matryoshka_fy.py
 ```

+## Multi vector retrieval usage
+
+```bash
+python examples/offline_inference/pooling/multi_vector_retrieval.py
+```
+
 ## Named Entity Recognition (NER) usage

 ```bash
diff -- examples/offline_inference/pooling/multi_vector_retrieval.py
@@ -0,0 +1,56 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+
+from argparse import Namespace
+
+from vllm import LLM, EngineArgs
+from vllm.utils import FlexibleArgumentParser
+
+
+def parse_args():
+    parser = FlexibleArgumentParser()
+    parser = EngineArgs.add_cli_args(parser)
+    # Set example specific arguments
```
- 已读文件:
  - runtime: `vllm/entrypoints/llm.py` modified +18/-19; `vllm/entrypoints/openai/api_server.py` modified +12/-9; `vllm/entrypoints/openai/protocol.py` modified +2/-2; `vllm/entrypoints/openai/serving_pooling.py` modified +13/-1; `vllm/model_executor/layers/pooler.py` modified +251/-171; `vllm/model_executor/models/adapters.py` modified +15/-27; `vllm/model_executor/models/bert.py` modified +10/-12; `vllm/model_executor/models/bert_with_rope.py` modified +5/-9
  - tests: `tests/conftest.py` modified +6/-2; `tests/entrypoints/pooling/llm/test_classify.py` modified +1/-1; `tests/entrypoints/pooling/llm/test_embedding.py` modified +7/-0; `tests/entrypoints/pooling/llm/test_encode.py` modified +8/-4; `tests/entrypoints/pooling/llm/test_reward.py` modified +12/-11; `tests/entrypoints/pooling/openai/test_embedding.py` modified +18/-0; `tests/entrypoints/pooling/openai/test_rerank.py` modified +18/-1; `tests/models/language/pooling/test_multi_vector_retrieval.py` added +45/-0
  - docs/bench: `examples/offline_inference/pooling/README.md` modified +6/-0; `examples/offline_inference/pooling/multi_vector_retrieval.py` added +56/-0; `examples/offline_inference/prithvi_geospatial_mae_io_processor.py` modified +1/-1; `examples/online_serving/pooling/README.md` modified +6/-0; `examples/online_serving/pooling/multi_vector_retrieval_client.py` added +54/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #29802 - Fix some Transformers nightly tests

- 链接: https://github.com/vllm-project/vllm/pull/29802
- 状态/时间: merged / 2025-12-02
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+29/-28，可读 patch 93 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix some Transformers nightly tests」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/jina_vl.py`, `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/qwen2.py`。
- 实现要点:
  - `vllm/model_executor/models/jina_vl.py` modified +1/-1
  - `vllm/model_executor/models/modernbert.py` modified +27/-26
  - `vllm/model_executor/models/qwen2.py` modified +1/-1
- 代码 diff 细节:
  - `vllm/model_executor/models/jina_vl.py` modified +1/-1
  - `vllm/model_executor/models/modernbert.py` modified +27/-26
  - `vllm/model_executor/models/qwen2.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/jina_vl.py
@@ -29,7 +29,7 @@
 class JinaVLScorer(nn.Module):
     def __init__(self, model_config: "ModelConfig"):
         super().__init__()
-        config = model_config.hf_config
+        config = model_config.hf_config.get_text_config()
         head_dtype = model_config.head_dtype
         self.dense = ColumnParallelLinear(
             config.hidden_size, config.hidden_size, params_dtype=head_dtype, bias=True
diff -- vllm/model_executor/models/modernbert.py
@@ -20,7 +20,7 @@
     PoolingParamsUpdate,
     PoolingType,
 )
-from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
+from vllm.model_executor.layers.rotary_embedding import get_rope
 from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
 from vllm.model_executor.model_loader.weight_utils import default_weight_loader
 from vllm.sequence import IntermediateTensors
@@ -62,19 +62,6 @@ def forward(
             return embeddings


-class ModernBertRotaryEmbedding(RotaryEmbedding):
```
- 已读文件:
  - runtime: `vllm/model_executor/models/jina_vl.py` modified +1/-1; `vllm/model_executor/models/modernbert.py` modified +27/-26; `vllm/model_executor/models/qwen2.py` modified +1/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #31445 - [Bugfix][Frontend] Fix Jina reranker multimodal input compatibility

- 链接: https://github.com/vllm-project/vllm/pull/31445
- 状态/时间: merged / 2025-12-29
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+316/-138，可读 patch 519 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Frontend] Fix Jina reranker multimodal input compatibility」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `tests/models/multimodal/pooling/test_jinavl_reranker.py`, `vllm/entrypoints/score_utils.py`。
- 实现要点:
  - `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +313/-137；symbols: _normalize_image, create_score_multimodal_param, _run_vllm, _run_hf
  - `vllm/entrypoints/score_utils.py` modified +3/-1
- 代码 diff 细节:
  - `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +313/-137
  - `vllm/entrypoints/score_utils.py` modified +3/-1
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/pooling/test_jinavl_reranker.py
@@ -1,194 +1,370 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from typing import cast

 import pytest
 from transformers import AutoModel

-from vllm.entrypoints.chat_utils import ChatCompletionContentPartImageParam
+from vllm.entrypoints.chat_utils import (
+    ChatCompletionContentPartImageEmbedsParam,
+    ChatCompletionContentPartImageParam,
+    ChatCompletionContentPartTextParam,
+)
diff -- vllm/entrypoints/score_utils.py
@@ -24,7 +24,9 @@
 from vllm.tokenizers import TokenizerLike

 ScoreContentPartParam: TypeAlias = (
-    ChatCompletionContentPartImageParam | ChatCompletionContentPartImageEmbedsParam
+    ChatCompletionContentPartImageParam
+    | ChatCompletionContentPartImageEmbedsParam
+    | ChatCompletionContentPartTextParam
 )


```
- 已读文件:
  - runtime: `vllm/entrypoints/score_utils.py` modified +3/-1
  - tests: `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +313/-137
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #31669 - [Misc][Model][Refactor] Pass the prefix into Linear layers

- 链接: https://github.com/vllm-project/vllm/pull/31669
- 状态/时间: merged / 2026-01-05
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+181/-40，可读 patch 753 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc][Model][Refactor] Pass the prefix into Linear layers」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/mamba/mamba_mixer.py`, `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/gpt_neox.py`。
- 实现要点:
  - `vllm/model_executor/layers/mamba/mamba_mixer.py` modified +12/-2
  - `vllm/model_executor/models/aria.py` modified +15/-5；symbols: __init__
  - `vllm/model_executor/models/gpt_neox.py` modified +1/-1
  - `vllm/model_executor/models/hunyuan_v1.py` modified +1/-0
- 代码 diff 细节:
  - `vllm/model_executor/layers/mamba/mamba_mixer.py` modified +12/-2
  - `vllm/model_executor/models/aria.py` modified +15/-5
  - `vllm/model_executor/models/gpt_neox.py` modified +1/-1
  - `vllm/model_executor/models/hunyuan_v1.py` modified +1/-0
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/mamba/mamba_mixer.py
@@ -82,6 +82,7 @@ def __init__(
             input_size=conv_kernel_size,
             output_size=intermediate_size,
             bias=use_conv_bias,
+            prefix=f"{prefix}.conv1d",
         )
         # unsqueeze to fit conv1d weights shape into the linear weights shape.
         # Can't do this in `weight_loader` since it already exists in
@@ -90,20 +91,28 @@ def __init__(
         self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

         self.in_proj = MergedColumnParallelLinear(
-            hidden_size, [intermediate_size] * 2, bias=use_bias
+            hidden_size,
diff -- vllm/model_executor/models/aria.py
@@ -127,11 +127,16 @@ def __init__(
         in_features: int,
         hidden_features: int,
         output_dim: int,
+        prefix: str = "",
     ) -> None:
         super().__init__()

-        self.linear_in = ColumnParallelLinear(in_features, hidden_features, bias=False)
-        self.linear_out = RowParallelLinear(hidden_features, output_dim, bias=False)
+        self.linear_in = ColumnParallelLinear(
+            in_features, hidden_features, bias=False, prefix=f"{prefix}.linear_in"
+        )
+        self.linear_out = RowParallelLinear(
```
- 已读文件:
  - runtime: `vllm/model_executor/layers/mamba/mamba_mixer.py` modified +12/-2; `vllm/model_executor/models/aria.py` modified +15/-5; `vllm/model_executor/models/gpt_neox.py` modified +1/-1; `vllm/model_executor/models/hunyuan_v1.py` modified +1/-0; `vllm/model_executor/models/jamba.py` modified +1/-0; `vllm/model_executor/models/jina_vl.py` modified +14/-4; `vllm/model_executor/models/minicpm.py` modified +4/-0; `vllm/model_executor/models/minicpm_eagle.py` modified +1/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #31973 - [Model] Reorganize pooling layers

- 链接: https://github.com/vllm-project/vllm/pull/31973
- 状态/时间: merged / 2026-01-09
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 34 个文件，+1290/-1143，可读 patch 2875 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Reorganize pooling layers」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `.github/CODEOWNERS`, `tests/model_executor/test_model_load_with_params.py`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`。
- 实现要点:
  - `.github/CODEOWNERS` modified +1/-1
  - `tests/model_executor/test_model_load_with_params.py` modified +2/-1
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +2/-7
  - `vllm/config/pooler.py` modified +4/-0；symbols: get_pooling_type
- 代码 diff 细节:
  - `.github/CODEOWNERS` modified +1/-1
  - `tests/model_executor/test_model_load_with_params.py` modified +2/-1
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +2/-7
  - `vllm/config/pooler.py` modified +4/-0
- 关键代码摘录:

```diff
diff -- .github/CODEOWNERS
@@ -153,7 +153,7 @@ mkdocs.yaml @hmellor
 /vllm/entrypoints/pooling @noooop
 /vllm/config/pooler.py @noooop
 /vllm/pooling_params.py @noooop
-/vllm/model_executor/layers/pooler.py @noooop
+/vllm/model_executor/layers/pooler @noooop

 # Security guide and policies
 /docs/usage/security.md @russellb
diff -- tests/model_executor/test_model_load_with_params.py
@@ -5,7 +5,8 @@

 import pytest

-from vllm.model_executor.layers.pooler import CLSPool, DispatchPooler, MeanPool
+from vllm.model_executor.layers.pooler import DispatchPooler
+from vllm.model_executor.layers.pooler.seqwise import CLSPool, MeanPool
 from vllm.model_executor.models.bert import BertEmbeddingModel
 from vllm.model_executor.models.roberta import RobertaEmbeddingModel
 from vllm.platforms import current_platform
```
- 已读文件:
  - runtime: `vllm/config/pooler.py` modified +4/-0; `vllm/model_executor/layers/pooler.py` removed +0/-845; `vllm/model_executor/layers/pooler/__init__.py` added +5/-0; `vllm/model_executor/layers/pooler/abstract.py` added +39/-0; `vllm/model_executor/layers/pooler/activations.py` added +162/-0; `vllm/model_executor/layers/pooler/common.py` added +27/-0; `vllm/model_executor/layers/pooler/seqwise/__init__.py` added +45/-0; `vllm/model_executor/layers/pooler/seqwise/heads.py` added +157/-0
  - tests: `tests/model_executor/test_model_load_with_params.py` modified +2/-1; `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +2/-7
  - other: `.github/CODEOWNERS` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #32085 - [Model] Improve multimodal pooling examples

- 链接: https://github.com/vllm-project/vllm/pull/32085
- 状态/时间: merged / 2026-01-12
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+381/-69，可读 patch 538 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Improve multimodal pooling examples」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/serving/openai_compatible_server.md`, `examples/pooling/embed/vision_embedding_offline.py`, `examples/pooling/embed/vision_embedding_online.py`。
- 实现要点:
  - `docs/serving/openai_compatible_server.md` modified +7/-4
  - `examples/pooling/embed/vision_embedding_offline.py` added +93/-0；symbols: print_embeddings, run_qwen3_vl, parse_args, main
  - `examples/pooling/embed/vision_embedding_online.py` renamed +130/-5；symbols: print_embeddings, run_qwen3_vl
  - `examples/pooling/score/cohere_rerank_online.py` renamed +0/-0
- 代码 diff 细节:
  - `docs/serving/openai_compatible_server.md` modified +7/-4
  - `examples/pooling/embed/vision_embedding_offline.py` added +93/-0
  - `examples/pooling/embed/vision_embedding_online.py` renamed +130/-5
  - `examples/pooling/score/cohere_rerank_online.py` renamed +0/-0
- 关键代码摘录:

```diff
diff -- docs/serving/openai_compatible_server.md
@@ -362,7 +362,7 @@ and passing a list of `messages` in the request. Refer to the examples below for
         `MrLight/dse-qwen2-2b-mrl-v1` requires a placeholder image of the minimum image size for text query embeddings. See the full code
         example below for details.

-Full example: [examples/pooling/embed/openai_chat_embedding_client_for_multimodal.py](../../examples/pooling/embed/openai_chat_embedding_client_for_multimodal.py)
+Full example: [examples/pooling/embed/vision_embedding_online.py](../../examples/pooling/embed/vision_embedding_online.py)

 #### Extra parameters

@@ -667,7 +667,7 @@ Usually, the score for a sentence pair refers to the similarity between two sent

 You can find the documentation for cross encoder models at [sbert.net](https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html).

-Code example: [examples/pooling/score/openai_cross_encoder_score.py](../../examples/pooling/score/openai_cross_encoder_score.py)
diff -- examples/pooling/embed/vision_embedding_offline.py
@@ -0,0 +1,93 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+"""
+This example shows how to use vLLM for running offline inference with
+the correct prompt format on vision language models for multimodal embedding.
+
+For most models, the prompt format should follow corresponding examples
+on HuggingFace model repository.
+"""
+
+import argparse
+from dataclasses import asdict
```
- 已读文件:
  - docs/bench: `docs/serving/openai_compatible_server.md` modified +7/-4; `examples/pooling/embed/vision_embedding_offline.py` added +93/-0; `examples/pooling/embed/vision_embedding_online.py` renamed +130/-5; `examples/pooling/score/cohere_rerank_online.py` renamed +0/-0; `examples/pooling/score/openai_cross_encoder_score_for_multimodal.py` removed +0/-60; `examples/pooling/score/rerank_api_online.py` renamed +0/-0; `examples/pooling/score/score_api_online.py` renamed +0/-0; `examples/pooling/score/vision_rerank_api_online.py` added +80/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #32395 - [Frontend][1/n] Make pooling entrypoints request schema consensus | CompletionRequest

- 链接: https://github.com/vllm-project/vllm/pull/32395
- 状态/时间: merged / 2026-01-16
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 22 个文件，+629/-594，可读 patch 1838 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend][1/n] Make pooling entrypoints request schema consensus | CompletionRequest 」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/serving/openai_compatible_server.md`, `examples/pooling/classify/classification_online.py`, `examples/pooling/classify/openai_classification_client.py`。
- 实现要点:
  - `docs/serving/openai_compatible_server.md` modified +1/-1
  - `examples/pooling/classify/classification_online.py` added +67/-0；symbols: parse_args, main
  - `examples/pooling/classify/openai_classification_client.py` removed +0/-53
  - `examples/pooling/pooling/vision_language_pooling.py` modified +1/-1
- 代码 diff 细节:
  - `docs/serving/openai_compatible_server.md` modified +1/-1
  - `examples/pooling/classify/classification_online.py` added +67/-0
  - `examples/pooling/classify/openai_classification_client.py` removed +0/-53
  - `examples/pooling/pooling/vision_language_pooling.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- docs/serving/openai_compatible_server.md
@@ -559,7 +559,7 @@ Our Classification API directly supports Hugging Face sequence-classification mo

 We automatically wrap any other transformer via `as_seq_cls_model()`, which pools on the last token, attaches a `RowParallelLinear` head, and applies a softmax to produce per-class probabilities.

-Code example: [examples/pooling/classify/openai_classification_client.py](../../examples/pooling/classify/openai_classification_client.py)
+Code example: [examples/pooling/classify/classification_online.py](../../examples/pooling/classify/classification_online.py)

 #### Example Requests

diff -- examples/pooling/classify/classification_online.py
@@ -0,0 +1,67 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Example Python client for classification API using vLLM API server
+NOTE:
+    start a supported classification model server with `vllm serve`, e.g.
+    vllm serve jason9693/Qwen2.5-1.5B-apeach
+"""
+
+import argparse
+import pprint
+
+import requests
+
```
- 已读文件:
  - runtime: `vllm/entrypoints/llm.py` modified +1/-1; `vllm/entrypoints/openai/api_server.py` modified +4/-58; `vllm/entrypoints/pooling/__init__.py` modified +88/-0; `vllm/entrypoints/pooling/base/__init__.py` added +0/-0; `vllm/entrypoints/pooling/base/protocol.py` added +46/-0; `vllm/entrypoints/pooling/classify/protocol.py` modified +7/-51; `vllm/entrypoints/pooling/embed/protocol.py` modified +9/-50; `vllm/entrypoints/pooling/pooling/protocol.py` modified +2/-10
  - tests: `tests/entrypoints/pooling/classify/test_online.py` modified +86/-49; `tests/entrypoints/pooling/embed/test_online.py` modified +233/-221; `tests/entrypoints/pooling/pooling/test_online.py` modified +49/-57; `tests/entrypoints/pooling/score/test_online_rerank.py` modified +19/-5; `tests/entrypoints/pooling/score/test_utils.py` modified +5/-5; `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +1/-1
  - docs/bench: `docs/serving/openai_compatible_server.md` modified +1/-1; `examples/pooling/classify/classification_online.py` added +67/-0; `examples/pooling/classify/openai_classification_client.py` removed +0/-53; `examples/pooling/pooling/vision_language_pooling.py` modified +1/-1; `examples/pooling/score/vision_reranker_offline.py` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #32577 - [Frontend] Score entrypoint support data_1 & data_2 and queries & documents as inputs

- 链接: https://github.com/vllm-project/vllm/pull/32577
- 状态/时间: merged / 2026-01-19
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+253/-113，可读 patch 749 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend] Score entrypoint support data_1 & data_2 and queries & documents as inputs」；模型线: Jina Reranker M0；类别: 模型支持/运行时入口；主要 diff: `docs/serving/openai_compatible_server.md`, `examples/offline_inference/basic/score.py`, `examples/offline_inference/openai_batch/README.md`。
- 实现要点:
  - `docs/serving/openai_compatible_server.md` modified +15/-15
  - `examples/offline_inference/basic/score.py` modified +5/-5
  - `examples/offline_inference/openai_batch/README.md` modified +2/-2
  - `examples/pooling/score/cohere_rerank_client.py` renamed +0/-0
- 代码 diff 细节:
  - `docs/serving/openai_compatible_server.md` modified +15/-15
  - `examples/offline_inference/basic/score.py` modified +5/-5
  - `examples/offline_inference/openai_batch/README.md` modified +2/-2
  - `examples/pooling/score/cohere_rerank_client.py` renamed +0/-0
- 关键代码摘录:

```diff
diff -- docs/serving/openai_compatible_server.md
@@ -694,7 +694,7 @@ Example template file: [examples/pooling/score/template/nemotron-rerank.jinja](.

 #### Single inference

-You can pass a string to both `text_1` and `text_2`, forming a single sentence pair.
+You can pass a string to both `queries` and `documents`, forming a single sentence pair.

 ```bash
 curl -X 'POST' \
@@ -704,8 +704,8 @@ curl -X 'POST' \
   -d '{
   "model": "BAAI/bge-reranker-v2-m3",
   "encoding_format": "float",
-  "text_1": "What is the capital of France?",
diff -- examples/offline_inference/basic/score.py
@@ -21,8 +21,8 @@ def parse_args():

 def main(args: Namespace):
     # Sample prompts.
-    text_1 = "What is the capital of France?"
-    texts_2 = [
+    query = "What is the capital of France?"
+    documents = [
         "The capital of Brazil is Brasilia.",
         "The capital of France is Paris.",
     ]
@@ -32,13 +32,13 @@ def main(args: Namespace):
     llm = LLM(**vars(args))

```
- 已读文件:
  - runtime: `vllm/entrypoints/openai/engine/serving.py` modified +9/-2; `vllm/entrypoints/openai/run_batch.py` modified +1/-1; `vllm/entrypoints/pooling/score/protocol.py` modified +38/-5; `vllm/entrypoints/pooling/score/serving.py` modified +16/-16
  - tests: `tests/entrypoints/openai/test_run_batch.py` modified +2/-2; `tests/entrypoints/pooling/classify/test_online.py` modified +2/-2; `tests/entrypoints/pooling/score/test_offline.py` modified +4/-4; `tests/entrypoints/pooling/score/test_online_score.py` modified +95/-37; `tests/models/language/pooling_mteb_test/mteb_score_utils.py` modified +2/-2
  - docs/bench: `docs/serving/openai_compatible_server.md` modified +15/-15; `examples/offline_inference/basic/score.py` modified +5/-5; `examples/offline_inference/openai_batch/README.md` modified +2/-2; `examples/pooling/score/cohere_rerank_client.py` renamed +0/-0; `examples/pooling/score/qwen3_reranker_online.py` modified +2/-2; `examples/pooling/score/score_api_online.py` modified +18/-12; `examples/pooling/score/vision_rerank_api_online.py` modified +20/-2; `examples/pooling/score/vision_score_api_online.py` modified +22/-4
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #32287 - Upgrade transformers-4.57.5

- 链接: https://github.com/vllm-project/vllm/pull/32287
- 状态/时间: merged / 2026-01-22
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+25/-3，可读 patch 91 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Upgrade transformers-4.57.5」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `requirements/nightly_torch_test.txt`, `requirements/test.in`, `requirements/test.txt`。
- 实现要点:
  - `requirements/nightly_torch_test.txt` modified +1/-1
  - `requirements/test.in` modified +1/-1
  - `requirements/test.txt` modified +1/-1
  - `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +22/-0
- 代码 diff 细节:
  - `requirements/nightly_torch_test.txt` modified +1/-1
  - `requirements/test.in` modified +1/-1
  - `requirements/test.txt` modified +1/-1
  - `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +22/-0
- 关键代码摘录:

```diff
diff -- requirements/nightly_torch_test.txt
@@ -29,7 +29,7 @@ opencv-python-headless >= 4.11.0 # required for video test
 datamodel_code_generator # required for minicpm3 test
 lm-eval[api]>=0.4.9.2 # required for model evaluation test
 mteb>=1.38.11, <2 # required for mteb test
-transformers==4.57.3
+transformers==4.57.5
 tokenizers==0.22.0
 schemathesis>=3.39.15 # Required for openai schema test.
 # quantization
diff -- requirements/test.in
@@ -37,7 +37,7 @@ opencv-python-headless >= 4.11.0 # required for video test
 datamodel_code_generator # required for minicpm3 test
 lm-eval[api]>=0.4.9.2 # required for model evaluation test
 mteb[bm25s]>=2, <3 # required for mteb test
-transformers==4.57.3
+transformers==4.57.5
 tokenizers==0.22.0
 schemathesis>=3.39.15 # Required for openai schema test.
 # quantization
```
- 已读文件:
  - tests: `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +22/-0
  - other: `requirements/nightly_torch_test.txt` modified +1/-1; `requirements/test.in` modified +1/-1; `requirements/test.txt` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #33063 - [Chore] Update type annotation of `input_ids` in model forward

- 链接: https://github.com/vllm-project/vllm/pull/33063
- 状态/时间: merged / 2026-01-26
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+146/-143，可读 patch 1304 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Chore] Update type annotation of `input_ids` in model forward」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/contributing/model/basic.md`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`, `vllm/model_executor/models/afmoe.py`。
- 实现要点:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +1/-1
  - `vllm/model_executor/models/afmoe.py` modified +2/-2
  - `vllm/model_executor/models/apertus.py` modified +1/-1
- 代码 diff 细节:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +1/-1
  - `vllm/model_executor/models/afmoe.py` modified +2/-2
  - `vllm/model_executor/models/apertus.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- docs/contributing/model/basic.md
@@ -71,7 +71,7 @@ class MyModel(nn.Module):
 ```python
 def forward(
     self,
-    input_ids: torch.Tensor,
+    input_ids: torch.Tensor | None,
     positions: torch.Tensor,
     intermediate_tensors: IntermediateTensors | None = None,
     inputs_embeds: torch.Tensor | None = None,
diff -- tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py
@@ -36,7 +36,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):

     def forward(
         self,
-        input_ids: torch.Tensor,
+        input_ids: torch.Tensor | None,
         positions: torch.Tensor,
         intermediate_tensors: IntermediateTensors | None = None,
         inputs_embeds: torch.Tensor | None = None,
```
- 已读文件:
  - runtime: `vllm/model_executor/models/afmoe.py` modified +2/-2; `vllm/model_executor/models/apertus.py` modified +1/-1; `vllm/model_executor/models/arcee.py` modified +1/-1; `vllm/model_executor/models/arctic.py` modified +2/-2; `vllm/model_executor/models/aria.py` modified +1/-1; `vllm/model_executor/models/audioflamingo3.py` modified +1/-1; `vllm/model_executor/models/aya_vision.py` modified +1/-1; `vllm/model_executor/models/bagel.py` modified +1/-1
  - tests: `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +1/-1
  - docs/bench: `docs/contributing/model/basic.md` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #33298 - [Bugfix] Fix Qwen3-VL-Reranker load.

- 链接: https://github.com/vllm-project/vllm/pull/33298
- 状态/时间: merged / 2026-01-29
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+234/-112，可读 patch 457 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen3-VL-Reranker load.」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `examples/pooling/score/vision_rerank_api_online.py`, `examples/pooling/score/vision_score_api_online.py`, `tests/entrypoints/pooling/classify/test_online_vision.py`。
- 实现要点:
  - `examples/pooling/score/vision_rerank_api_online.py` modified +46/-49
  - `examples/pooling/score/vision_score_api_online.py` modified +54/-45
  - `tests/entrypoints/pooling/classify/test_online_vision.py` modified +2/-2
  - `tests/entrypoints/pooling/score/test_online_score_vision.py` added +122/-0；symbols: server, test_score_api_queries_str_documents_str, test_score_api_queries_str_documents_text_content, test_score_api_queries_str_documents_image_url_content
- 代码 diff 细节:
  - `examples/pooling/score/vision_rerank_api_online.py` modified +46/-49
  - `examples/pooling/score/vision_score_api_online.py` modified +54/-45
  - `tests/entrypoints/pooling/classify/test_online_vision.py` modified +2/-2
  - `tests/entrypoints/pooling/score/test_online_score_vision.py` added +122/-0
- 关键代码摘录:

```diff
diff -- examples/pooling/score/vision_rerank_api_online.py
@@ -18,48 +18,32 @@
 """

 import argparse
-import base64
-import json
+import pprint

 import requests

-
-def encode_base64_content_from_url(content_url: str) -> dict[str, str]:
-    """Encode a content retrieved from a remote url to base64 format."""
-
diff -- examples/pooling/score/vision_score_api_online.py
@@ -17,48 +17,32 @@
 """

 import argparse
-import base64
-import json
 import pprint

 import requests

-
-def encode_base64_content_from_url(content_url: str) -> dict[str, str]:
-    """Encode a content retrieved from a remote url to base64 format."""
-
```
- 已读文件:
  - runtime: `vllm/model_executor/models/adapters.py` modified +10/-4
  - tests: `tests/entrypoints/pooling/classify/test_online_vision.py` modified +2/-2; `tests/entrypoints/pooling/score/test_online_score_vision.py` added +122/-0; `tests/entrypoints/test_utils.py` modified +0/-12
  - docs/bench: `examples/pooling/score/vision_rerank_api_online.py` modified +46/-49; `examples/pooling/score/vision_score_api_online.py` modified +54/-45
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #33060 - [Frontend][4/n] Make pooling entrypoints request schema consensus | ScoreRequest

- 链接: https://github.com/vllm-project/vllm/pull/33060
- 状态/时间: merged / 2026-02-04
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+432/-205，可读 patch 1008 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend][4/n] Make pooling entrypoints request schema consensus | ScoreRequest」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `examples/pooling/score/vision_rerank_api_online.py`, `examples/pooling/score/vision_score_api_online.py`, `tests/entrypoints/pooling/score/test_online_score.py`。
- 实现要点:
  - `examples/pooling/score/vision_rerank_api_online.py` modified +23/-0
  - `examples/pooling/score/vision_score_api_online.py` modified +38/-0
  - `tests/entrypoints/pooling/score/test_online_score.py` modified +29/-0；symbols: test_queries_str_items_str
  - `tests/entrypoints/pooling/score/test_online_score_vision.py` modified +124/-7；symbols: test_score_api_queries_str_documents_image_url_plus_text_content, test_score_api_queries_str_documents_list, test_rerank_api_queries_str_documents_list, test_score_api_queries_list_documents_list
- 代码 diff 细节:
  - `examples/pooling/score/vision_rerank_api_online.py` modified +23/-0
  - `examples/pooling/score/vision_score_api_online.py` modified +38/-0
  - `tests/entrypoints/pooling/score/test_online_score.py` modified +29/-0
  - `tests/entrypoints/pooling/score/test_online_score_vision.py` modified +124/-7
- 关键代码摘录:

```diff
diff -- examples/pooling/score/vision_rerank_api_online.py
@@ -89,6 +89,29 @@ def main(args):
     response = requests.post(rerank_url, json=prompt)
     pprint.pprint(response.json())

+    print("Query: string & Document: text + image url")
+    prompt = {
+        "model": model,
+        "query": query,
+        "documents": {"content": [documents[0], documents[1]]},
+    }
+    response = requests.post(rerank_url, json=prompt)
+    pprint.pprint(response.json())
+
+    print("Query: string & Document: list")
diff -- examples/pooling/score/vision_score_api_online.py
@@ -92,6 +92,44 @@ def main(args):
     response = requests.post(score_url, json=prompt)
     pprint.pprint(response.json())

+    print("Query: string & Document: text + image url")
+    prompt = {
+        "model": model,
+        "queries": query,
+        "documents": {"content": [documents[0], documents[1]]},
+    }
+    response = requests.post(score_url, json=prompt)
+    pprint.pprint(response.json())
+
+    print("Query: string & Document: list")
```
- 已读文件:
  - runtime: `vllm/entrypoints/llm.py` modified +37/-73; `vllm/entrypoints/pooling/score/protocol.py` modified +28/-11; `vllm/entrypoints/pooling/score/serving.py` modified +84/-94; `vllm/entrypoints/pooling/score/utils.py` modified +69/-20
  - tests: `tests/entrypoints/pooling/score/test_online_score.py` modified +29/-0; `tests/entrypoints/pooling/score/test_online_score_vision.py` modified +124/-7
  - docs/bench: `examples/pooling/score/vision_rerank_api_online.py` modified +23/-0; `examples/pooling/score/vision_score_api_online.py` modified +38/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #33837 - [Bugfix] Fix ScoreMultiModalParam multi-document scoring returning single result

- 链接: https://github.com/vllm-project/vllm/pull/33837
- 状态/时间: merged / 2026-02-05
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+21/-44，可读 patch 99 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix ScoreMultiModalParam multi-document scoring returning single result」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `tests/models/multimodal/pooling/test_jinavl_reranker.py`。
- 实现要点:
  - `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +21/-44
- 代码 diff 细节:
  - `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +21/-44
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/pooling/test_jinavl_reranker.py
@@ -1,6 +1,5 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-from typing import cast

 import pytest
 import transformers
@@ -117,7 +116,7 @@ def _normalize_image(image_val: str) -> str:

 def create_score_multimodal_param(
     content_parts: list[dict],
-) -> ScoreMultiModalParam:
+) -> list[ScoreMultiModalParam]:
     """
```
- 已读文件:
  - tests: `tests/models/multimodal/pooling/test_jinavl_reranker.py` modified +21/-44
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #31127 - [Frontend][last/5] Make pooling entrypoints request schema consensus.

- 链接: https://github.com/vllm-project/vllm/pull/31127
- 状态/时间: merged / 2026-02-09
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 24 个文件，+658/-612，可读 patch 1726 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend][last/5] Make pooling entrypoints request schema consensus. 」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `.buildkite/test-amd.yaml`, `.buildkite/test-pipeline.yaml`, `.buildkite/test_areas/misc.yaml`。
- 实现要点:
  - `.buildkite/test-amd.yaml` modified +1/-1
  - `.buildkite/test-pipeline.yaml` modified +1/-1
  - `.buildkite/test_areas/misc.yaml` modified +1/-1
  - `docs/features/multimodal_inputs.md` modified +1/-1
- 代码 diff 细节:
  - `.buildkite/test-amd.yaml` modified +1/-1
  - `.buildkite/test-pipeline.yaml` modified +1/-1
  - `.buildkite/test_areas/misc.yaml` modified +1/-1
  - `docs/features/multimodal_inputs.md` modified +1/-1
- 关键代码摘录:

```diff
diff -- .buildkite/test-amd.yaml
@@ -514,7 +514,7 @@ steps:
     - python3 offline_inference/vision_language_multi_image.py --seed 0
     - python3 offline_inference/encoder_decoder_multimodal.py --model-type whisper --seed 0
     # for pooling models
-    - python3 pooling/pooling/vision_language_pooling.py --seed 0
+    - python3 pooling/embed/vision_embedding_offline.py --seed 0
     # for features demo
     - python3 offline_inference/prefix_caching.py
     - python3 offline_inference/llm_engine_example.py
diff -- .buildkite/test-pipeline.yaml
@@ -453,7 +453,7 @@ steps:
     - python3 offline_inference/vision_language_multi_image.py --seed 0
     - python3 offline_inference/encoder_decoder_multimodal.py --model-type whisper --seed 0
     # for pooling models
-    - python3 pooling/pooling/vision_language_pooling.py --seed 0
+    - python3 pooling/embed/vision_embedding_offline.py --seed 0
     # for features demo
     - python3 offline_inference/prefix_caching.py
     - python3 offline_inference/llm_engine_example.py
```
- 已读文件:
  - runtime: `vllm/entrypoints/pooling/base/protocol.py` modified +15/-0; `vllm/entrypoints/pooling/classify/protocol.py` modified +1/-7; `vllm/entrypoints/pooling/embed/protocol.py` modified +1/-6; `vllm/entrypoints/pooling/pooling/protocol.py` modified +1/-6; `vllm/entrypoints/pooling/score/protocol.py` modified +1/-15; `vllm/utils/print_utils.py` added +7/-0
  - tests: `tests/entrypoints/pooling/classify/test_offline.py` modified +50/-7; `tests/entrypoints/pooling/embed/test_online_vision.py` modified +81/-2; `tests/renderers/test_hf.py` modified +3/-3
  - docs/bench: `.buildkite/test-amd.yaml` modified +1/-1; `.buildkite/test-pipeline.yaml` modified +1/-1; `.buildkite/test_areas/misc.yaml` modified +1/-1; `docs/features/multimodal_inputs.md` modified +1/-1; `docs/serving/openai_compatible_server.md` modified +31/-29; `examples/pooling/classify/vision_classification_online.py` added +110/-0; `examples/pooling/embed/template/dse_qwen2_vl.jinja` renamed +0/-0; `examples/pooling/embed/template/vlm2vec_phi3v.jinja` renamed +0/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #35592 - [Docs] Reorganize pooling docs.

- 链接: https://github.com/vllm-project/vllm/pull/35592
- 状态/时间: merged / 2026-03-19
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+2393/-1736，可读 patch 4283 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Reorganize pooling docs.」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `.github/CODEOWNERS`, `docs/.nav.yml`, `docs/contributing/model/tests.md`。
- 实现要点:
  - `.github/CODEOWNERS` modified +1/-0
  - `docs/.nav.yml` modified +1/-1
  - `docs/contributing/model/tests.md` modified +1/-1
  - `docs/features/README.md` modified +3/-3
- 代码 diff 细节:
  - `.github/CODEOWNERS` modified +1/-0
  - `docs/.nav.yml` modified +1/-1
  - `docs/contributing/model/tests.md` modified +1/-1
  - `docs/features/README.md` modified +3/-3
- 关键代码摘录:

```diff
diff -- .github/CODEOWNERS
@@ -171,6 +171,7 @@ mkdocs.yaml @hmellor

 # Pooling models
 /examples/pooling @noooop
+/docs/models/pooling_models @noooop
 /tests/models/*/pooling* @noooop
 /tests/entrypoints/pooling @noooop
 /vllm/config/pooler.py @noooop
diff -- docs/.nav.yml
@@ -25,7 +25,7 @@ nav:
     - Models:
       - models/supported_models.md
       - models/generative_models.md
-      - models/pooling_models.md
+      - Pooling Models: models/pooling_models
       - models/extensions
       - Hardware Supported Models:
         - models/hardware_supported_models/*
```
- 已读文件:
  - docs/bench: `docs/.nav.yml` modified +1/-1; `docs/contributing/model/tests.md` modified +1/-1; `docs/features/README.md` modified +3/-3; `docs/mkdocs/hooks/url_schemes.py` modified +91/-43; `docs/models/pooling_models.md` removed +0/-716; `docs/models/pooling_models/README.md` added +253/-0; `docs/models/pooling_models/classify.md` added +276/-0; `docs/models/pooling_models/embed.md` added +546/-0
  - other: `.github/CODEOWNERS` modified +1/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #37537 - [Model] Deprecate the score task (this will not affect users).

- 链接: https://github.com/vllm-project/vllm/pull/37537
- 状态/时间: merged / 2026-03-20
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 22 个文件，+184/-163，可读 patch 808 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Deprecate the score task (this will not affect users). 」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/models/pooling_models/README.md`, `docs/models/pooling_models/classify.md`, `docs/models/pooling_models/scoring.md`。
- 实现要点:
  - `docs/models/pooling_models/README.md` modified +35/-28
  - `docs/models/pooling_models/classify.md` modified +3/-1
  - `docs/models/pooling_models/scoring.md` modified +10/-7
  - `tests/test_pooling_params.py` modified +1/-1
- 代码 diff 细节:
  - `docs/models/pooling_models/README.md` modified +35/-28
  - `docs/models/pooling_models/classify.md` modified +3/-1
  - `docs/models/pooling_models/scoring.md` modified +10/-7
  - `tests/test_pooling_params.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models/README.md
@@ -31,28 +31,29 @@ Of course, we also have "plugin" tasks that allow users to customize input and o

 ### Pooling Tasks

-| Pooling Tasks      | Granularity   | Outputs                                         |
-|--------------------|---------------|-------------------------------------------------|
-| `classify`         | Sequence-wise | probability vector of classes for each sequence |
-| `score` (see note) | Sequence-wise | reranker score for each sequence                |
-| `embed`            | Sequence-wise | vector representations for each sequence        |
-| `token_classify`   | Token-wise    | probability vector of classes for each token    |
-| `token_embed`      | Token-wise    | vector representations for each token           |
+| Pooling Tasks         | Granularity   | Outputs                                         |
+|-----------------------|---------------|-------------------------------------------------|
+| `classify` (see note) | Sequence-wise | probability vector of classes for each sequence |
diff -- docs/models/pooling_models/classify.md
@@ -17,6 +17,8 @@ The key distinction between (sequence) classification and token classification l

 Many classification models support both (sequence) classification and token classification. For further details on token classification, please refer to [this page](token_classify.md).

+Only when a classification model outputs num_labels equal to 1 can it be used as a scoring model and have its scoring API enabled, please refer to [this page](scoring.md).
+
 ## Typical Use Cases

 ### Classification
@@ -54,7 +56,7 @@ If your model is not in the above list, we will try to automatically convert the

 Cross-encoder (aka reranker) models are a subset of classification models that accept two prompts as input and output num_labels equal to 1. Most classification models can also be used as [cross-encoder models](scoring.md#cross-encoder-models). For more information on cross-encoder models, please refer to [this page](scoring.md).

---8<-- "docs/models/pooling_models/scoring.md:supported-score-models"
```
- 已读文件:
  - runtime: `vllm/config/model.py` modified +4/-4; `vllm/entrypoints/llm.py` modified +2/-2; `vllm/entrypoints/openai/api_server.py` modified +9/-5; `vllm/entrypoints/pooling/__init__.py` modified +29/-11; `vllm/entrypoints/pooling/score/protocol.py` modified +2/-2; `vllm/entrypoints/pooling/score/serving.py` modified +1/-1; `vllm/entrypoints/sagemaker/api_router.py` modified +13/-5; `vllm/model_executor/layers/pooler/activations.py` modified +10/-22
  - tests: `tests/test_pooling_params.py` modified +1/-1
  - docs/bench: `docs/models/pooling_models/README.md` modified +35/-28; `docs/models/pooling_models/classify.md` modified +3/-1; `docs/models/pooling_models/scoring.md` modified +10/-7
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #37902 - [Mypy] Better fixes for the `mypy` issues in `vllm/config`

- 链接: https://github.com/vllm-project/vllm/pull/37902
- 状态/时间: merged / 2026-03-25
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+153/-182，可读 patch 1078 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Mypy] Better fixes for the `mypy` issues in `vllm/config`」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `benchmarks/benchmark_long_document_qa_throughput.py`, `benchmarks/benchmark_prefix_caching.py`, `benchmarks/benchmark_prioritization.py`。
- 实现要点:
  - `benchmarks/benchmark_long_document_qa_throughput.py` modified +1/-2
  - `benchmarks/benchmark_prefix_caching.py` modified +1/-2
  - `benchmarks/benchmark_prioritization.py` modified +1/-2
  - `examples/offline_inference/audio_language.py` modified +1/-2
- 代码 diff 细节:
  - `benchmarks/benchmark_long_document_qa_throughput.py` modified +1/-2
  - `benchmarks/benchmark_prefix_caching.py` modified +1/-2
  - `benchmarks/benchmark_prioritization.py` modified +1/-2
  - `examples/offline_inference/audio_language.py` modified +1/-2
- 关键代码摘录:

```diff
diff -- benchmarks/benchmark_long_document_qa_throughput.py
@@ -42,7 +42,6 @@

 import random
 import time
-from dataclasses import fields

 from vllm import LLM, SamplingParams
 from vllm.engine.arg_utils import EngineArgs
@@ -124,7 +123,7 @@ def main(args):

     # Create the LLM engine
     engine_args = EngineArgs.from_cli_args(args)
-    llm = LLM(**{f.name: getattr(engine_args, f.name) for f in fields(engine_args)})
+    llm = LLM.from_engine_args(engine_args)
diff -- benchmarks/benchmark_prefix_caching.py
@@ -32,7 +32,6 @@
 import json
 import random
 import time
-from dataclasses import fields

 from transformers import PreTrainedTokenizerBase

@@ -197,7 +196,7 @@ def main(args):

     engine_args = EngineArgs.from_cli_args(args)

-    llm = LLM(**{f.name: getattr(engine_args, f.name) for f in fields(engine_args)})
+    llm = LLM.from_engine_args(engine_args)
```
- 已读文件:
  - runtime: `vllm/benchmarks/latency.py` modified +1/-2; `vllm/benchmarks/mm_processor.py` modified +1/-2; `vllm/benchmarks/startup.py` modified +1/-2; `vllm/benchmarks/throughput.py` modified +2/-3; `vllm/config/compilation.py` modified +15/-20; `vllm/config/device.py` modified +2/-2; `vllm/config/kernel.py` modified +2/-2; `vllm/config/kv_events.py` modified +1/-3
  - tests: `tests/compile/test_config.py` modified +6/-2; `tests/entrypoints/offline_mode/test_offline_mode.py` modified +1/-4; `tests/models/multimodal/generation/test_keye.py` modified +8/-11; `tests/models/multimodal/generation/test_vit_backend_functionality.py` modified +5/-10; `tests/models/multimodal/generation/test_voxtral_realtime.py` modified +1/-2; `tests/v1/kv_connector/unit/test_example_connector.py` modified +15/-22
  - docs/bench: `benchmarks/benchmark_long_document_qa_throughput.py` modified +1/-2; `benchmarks/benchmark_prefix_caching.py` modified +1/-2; `benchmarks/benchmark_prioritization.py` modified +1/-2; `examples/offline_inference/audio_language.py` modified +1/-2; `examples/offline_inference/encoder_decoder_multimodal.py` modified +5/-7; `examples/offline_inference/load_sharded_state.py` modified +1/-3; `examples/offline_inference/save_sharded_state.py` modified +1/-2; `examples/offline_inference/vision_language.py` modified +6/-7
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #28631 - [Frontend][3/n] Improve pooling entrypoints | scoring.

- 链接: https://github.com/vllm-project/vllm/pull/28631
- 状态/时间: merged / 2026-03-31
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 37 个文件，+1257/-1780，可读 patch 3713 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend][3/n] Improve pooling entrypoints | scoring.」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `tests/entrypoints/openai/utils.py`, `tests/entrypoints/pooling/classify/test_offline.py`, `tests/entrypoints/pooling/classify/test_online.py`。
- 实现要点:
  - `tests/entrypoints/openai/utils.py` modified +1/-3
  - `tests/entrypoints/pooling/classify/test_offline.py` modified +1/-1
  - `tests/entrypoints/pooling/classify/test_online.py` modified +2/-2
  - `tests/entrypoints/pooling/scoring/test_bi_encoder_online.py` modified +1/-1
- 代码 diff 细节:
  - `tests/entrypoints/openai/utils.py` modified +1/-3
  - `tests/entrypoints/pooling/classify/test_offline.py` modified +1/-1
  - `tests/entrypoints/pooling/classify/test_online.py` modified +2/-2
  - `tests/entrypoints/pooling/scoring/test_bi_encoder_online.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- tests/entrypoints/openai/utils.py
@@ -10,9 +10,7 @@
     ChatCompletionStreamResponse,
     ChatMessage,
 )
-from vllm.entrypoints.openai.engine.protocol import (
-    UsageInfo,
-)
+from vllm.entrypoints.openai.engine.protocol import UsageInfo


 async def accumulate_streaming_response(
diff -- tests/entrypoints/pooling/classify/test_offline.py
@@ -105,7 +105,7 @@ def get_outputs(use_activation):

 @pytest.mark.skip_global_cleanup
 def test_score_api(llm: LLM):
-    err_msg = "Score API is only enabled for num_labels == 1."
+    err_msg = "Scoring API is only enabled for num_labels == 1."
     with pytest.raises(ValueError, match=err_msg):
         llm.score("ping", "pong", use_tqdm=False)

```
- 已读文件:
  - runtime: `vllm/entrypoints/llm.py` modified +60/-252; `vllm/entrypoints/openai/engine/serving.py` modified +4/-104; `vllm/entrypoints/openai/run_batch.py` modified +27/-27; `vllm/entrypoints/pooling/__init__.py` modified +6/-21; `vllm/entrypoints/pooling/base/io_processor.py` modified +21/-25; `vllm/entrypoints/pooling/base/serving.py` modified +18/-4; `vllm/entrypoints/pooling/io_processor_factories.py` modified +7/-0; `vllm/entrypoints/pooling/score/serving.py` removed +0/-667
  - tests: `tests/entrypoints/openai/utils.py` modified +1/-3; `tests/entrypoints/pooling/classify/test_offline.py` modified +1/-1; `tests/entrypoints/pooling/classify/test_online.py` modified +2/-2; `tests/entrypoints/pooling/scoring/test_bi_encoder_online.py` modified +1/-1; `tests/entrypoints/pooling/scoring/test_cross_encoder_online.py` modified +1/-1; `tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py` modified +1/-1; `tests/entrypoints/pooling/scoring/test_late_interaction_offline_vision.py` added +93/-0; `tests/entrypoints/pooling/scoring/test_late_interaction_online.py` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #38800 - [New Model]: jinaai/jina-reranker-v3

- 链接: https://github.com/vllm-project/vllm/pull/38800
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+660/-19，可读 patch 795 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[New Model]: jinaai/jina-reranker-v3」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/models/pooling_models/token_embed.md`, `examples/pooling/token_embed/jina_reranker_v3_offline.py`, `pyproject.toml`。
- 实现要点:
  - `docs/models/pooling_models/token_embed.md` modified +8/-0
  - `examples/pooling/token_embed/jina_reranker_v3_offline.py` added +56/-0；symbols: main
  - `pyproject.toml` modified +2/-1
  - `tests/models/language/pooling/test_jina_reranker_v3.py` added +275/-0；symbols: test_offline, test_online, _test_offline_1_v_1, _test_offline_1_v_n
- 代码 diff 细节:
  - `docs/models/pooling_models/token_embed.md` modified +8/-0
  - `examples/pooling/token_embed/jina_reranker_v3_offline.py` added +56/-0
  - `pyproject.toml` modified +2/-1
  - `tests/models/language/pooling/test_jina_reranker_v3.py` added +275/-0
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models/token_embed.md
@@ -71,6 +71,14 @@ Models of any architecture can be converted into embedding models using `--conve

 If your model is not in the above list, we will try to automatically convert the model using [as_embedding_model][vllm.model_executor.models.adapters.as_embedding_model].

+### Special models
+
+| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
+| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
+| `JinaForRanking` | Qwen3-based | `jinaai/jina-reranker-v3` | | |
+
+jina-reranker-v3 is a listwise document reranker model with a novel `last but not late interaction` architecture. More information can be found at: [examples/pooling/token_embed/jina_reranker_v3_offline.py](../../../examples/pooling/token_embed/jina_reranker_v3_offline.py)
+
 --8<-- [end:supported-token-embed-models]

diff -- examples/pooling/token_embed/jina_reranker_v3_offline.py
@@ -0,0 +1,56 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+
+import torch.nn.functional as F
+
+from vllm import LLM
+
+query = "What are the health benefits of green tea?"
+documents = [
+    "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
+    "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
+    "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
```
- 已读文件:
  - runtime: `vllm/entrypoints/pooling/embed/io_processor.py` modified +52/-1; `vllm/entrypoints/pooling/io_processor_factories.py` modified +7/-0; `vllm/entrypoints/pooling/scoring/io_processor.py` modified +126/-0; `vllm/entrypoints/pooling/scoring/serving.py` modified +11/-14; `vllm/model_executor/layers/pooler/tokwise/methods.py` modified +1/-1; `vllm/model_executor/layers/pooler/tokwise/poolers.py` modified +3/-2; `vllm/model_executor/models/config.py` modified +7/-0; `vllm/model_executor/models/jina.py` added +110/-0
  - tests: `tests/models/language/pooling/test_jina_reranker_v3.py` added +275/-0; `tests/models/registry.py` modified +1/-0
  - docs/bench: `docs/models/pooling_models/token_embed.md` modified +8/-0; `examples/pooling/token_embed/jina_reranker_v3_offline.py` added +56/-0
  - other: `pyproject.toml` modified +2/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #30566 - Update to transformers v5

- 链接: https://github.com/vllm-project/vllm/pull/30566
- 状态/时间: merged / 2026-04-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 41 个文件，+445/-115，可读 patch 1409 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update to transformers v5」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `.buildkite/scripts/hardware_ci/run-cpu-test.sh`, `.buildkite/test_areas/models_basic.yaml`, `docker/Dockerfile`。
- 实现要点:
  - `.buildkite/scripts/hardware_ci/run-cpu-test.sh` modified +1/-1
  - `.buildkite/test_areas/models_basic.yaml` modified +15/-1
  - `docker/Dockerfile` modified +5/-4
  - `docker/Dockerfile.cpu` modified +6/-0
- 代码 diff 细节:
  - `.buildkite/scripts/hardware_ci/run-cpu-test.sh` modified +1/-1
  - `.buildkite/test_areas/models_basic.yaml` modified +15/-1
  - `docker/Dockerfile` modified +5/-4
  - `docker/Dockerfile.cpu` modified +6/-0
- 关键代码摘录:

```diff
diff -- .buildkite/scripts/hardware_ci/run-cpu-test.sh
@@ -16,5 +16,5 @@ echo "--- :docker: Building Docker image"
 docker build --progress plain --tag "$IMAGE_NAME" --target vllm-test -f docker/Dockerfile.cpu .

 # Run the image, setting --shm-size=4g for tensor parallel.
-docker run --rm --cpuset-cpus="$CORE_RANGE" --cpuset-mems="$NUMA_NODE" -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true -e HF_TOKEN -e VLLM_CPU_KVCACHE_SPACE=16 -e VLLM_CPU_CI_ENV=1 -e VLLM_CPU_SIM_MULTI_NUMA=1 --shm-size=4g "$IMAGE_NAME" \
+docker run --rm --cpuset-cpus="$CORE_RANGE" --cpuset-mems="$NUMA_NODE" -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true -e HF_TOKEN -e VLLM_CPU_KVCACHE_SPACE=16 -e VLLM_CPU_CI_ENV=1 -e VLLM_CPU_SIM_MULTI_NUMA=1 -e VLLM_CPU_ATTN_SPLIT_KV=0 --shm-size=4g "$IMAGE_NAME" \
         timeout "$TIMEOUT_VAL" bash -c "set -euox pipefail; echo \"--- Print packages\"; pip list; echo \"--- Running tests\"; ${TEST_COMMAND}"
diff -- .buildkite/test_areas/models_basic.yaml
@@ -4,7 +4,6 @@ depends_on:
 steps:
 - label: Basic Models Tests (Initialization)
   timeout_in_minutes: 45
-  device: h200_18gb
   torch_nightly: true
   source_file_dependencies:
   - vllm/
@@ -73,3 +72,18 @@ steps:
     - python3 examples/offline_inference/vision_language.py --model-type qwen2_5_vl
     # Whisper needs spawn method to avoid deadlock
     - VLLM_WORKER_MULTIPROC_METHOD=spawn python3 examples/offline_inference/audio_language.py --model-type whisper
+
+- label: Transformers Backward Compatibility Models Test
```
- 已读文件:
  - tests: `tests/conftest.py` modified +9/-0; `tests/lora/conftest.py` modified +6/-0; `tests/lora/test_minicpmv_tp.py` modified +11/-0; `tests/model_executor/test_weight_utils.py` modified +0/-18; `tests/models/language/generation/test_common.py` modified +5/-0; `tests/models/language/pooling_mteb_test/test_baai.py` modified +4/-1; `tests/models/language/pooling_mteb_test/test_gte.py` modified +2/-1; `tests/models/language/pooling_mteb_test/test_jina.py` modified +4/-0
  - docs/bench: `.buildkite/scripts/hardware_ci/run-cpu-test.sh` modified +1/-1; `.buildkite/test_areas/models_basic.yaml` modified +15/-1; `docs/getting_started/installation/gpu.rocm.inc.md` modified +1/-1
  - other: `docker/Dockerfile` modified +5/-4; `docker/Dockerfile.cpu` modified +6/-0; `docker/Dockerfile.nightly_torch` modified +4/-3; `docker/Dockerfile.rocm` modified +4/-3; `requirements/common.txt` modified +2/-2; `requirements/test/cuda.in` modified +3/-3; `requirements/test/cuda.txt` modified +10/-10; `requirements/test/nightly-torch.txt` modified +2/-2
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #39575 - Add Jina Embeddings v5 model support (fixes #38633)

- 链接: https://github.com/vllm-project/vllm/pull/39575
- 状态/时间: merged / 2026-04-16
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+218/-10，可读 patch 401 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Jina Embeddings v5 model support (fixes #38633)」；模型线: Jina Reranker M0；类别: 缺陷修复；主要 diff: `docs/models/pooling_models/embed.md`, `tests/conftest.py`, `tests/models/language/pooling_mteb_test/mteb_embed_utils.py`。
- 实现要点:
  - `docs/models/pooling_models/embed.md` modified +7/-0
  - `tests/conftest.py` modified +6/-0
  - `tests/models/language/pooling_mteb_test/mteb_embed_utils.py` modified +26/-4；symbols: HfMtebEncoder, __init__, encode
  - `tests/models/language/pooling_mteb_test/test_jina.py` modified +25/-5
- 代码 diff 细节:
  - `docs/models/pooling_models/embed.md` modified +7/-0
  - `tests/conftest.py` modified +6/-0
  - `tests/models/language/pooling_mteb_test/mteb_embed_utils.py` modified +26/-4
  - `tests/models/language/pooling_mteb_test/test_jina.py` modified +25/-5
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models/embed.md
@@ -45,6 +45,7 @@ You can compute pairwise similarity scores to build a similarity matrix using th
 | `GritLM` | GritLM | `parasail-ai/GritLM-7B-vllm`. | ✅︎ | ✅︎ |
 | `GteModel` | Arctic-Embed-2.0-M | `Snowflake/snowflake-arctic-embed-m-v2.0`. | | |
 | `GteNewModel` | mGTE-TRM (see note) | `Alibaba-NLP/gte-multilingual-base`, etc. | | |
+| `JinaEmbeddingsV5Model`<sup>C</sup> | Qwen3-based with task-specific LoRA adapters | `jinaai/jina-embeddings-v5-text-small` (see note) | ✅︎ | ✅︎ |
 | `LlamaBidirectionalModel`<sup>C</sup> | Llama-based with bidirectional attention | `nvidia/llama-nemotron-embed-1b-v2`, etc. | ✅︎ | ✅︎ |
 | `LlamaModel`<sup>C</sup>, `LlamaForCausalLM`<sup>C</sup>, `MistralModel`<sup>C</sup>, etc. | Llama-based | `intfloat/e5-mistral-7b-instruct`, etc. | ✅︎ | ✅︎ |
 | `ModernBertModel` | ModernBERT-based | `Alibaba-NLP/gte-modernbert-base`, etc. | | |
@@ -73,6 +74,12 @@ You can compute pairwise similarity scores to build a similarity matrix using th
 !!! note
     `jinaai/jina-embeddings-v3` supports multiple tasks through LoRA, while vllm temporarily only supports text-matching tasks by merging LoRA weights.

+!!! note
+    `jinaai/jina-embeddings-v5-text-small` ships with four task-specific LoRA adapters
diff -- tests/conftest.py
@@ -364,6 +364,7 @@ def __init__(
         model_name: str,
         dtype: str = "auto",
         *,
+        revision: str | None = None,
         model_kwargs: dict[str, Any] | None = None,
         trust_remote_code: bool = True,
         is_sentence_transformer: bool = False,
@@ -383,6 +384,7 @@ def __init__(
             self._init(
                 model_name=model_name,
                 dtype=dtype,
+                revision=revision,
                 model_kwargs=model_kwargs,
```
- 已读文件:
  - runtime: `vllm/model_executor/models/jina.py` modified +149/-1; `vllm/model_executor/models/registry.py` modified +1/-0
  - tests: `tests/conftest.py` modified +6/-0; `tests/models/language/pooling_mteb_test/mteb_embed_utils.py` modified +26/-4; `tests/models/language/pooling_mteb_test/test_jina.py` modified +25/-5; `tests/models/registry.py` modified +4/-0
  - docs/bench: `docs/models/pooling_models/embed.md` modified +7/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #39675 - [Frontend][last/5] Improve pooling entrypoints | clean up.

- 链接: https://github.com/vllm-project/vllm/pull/39675
- 状态/时间: merged / 2026-04-16
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 29 个文件，+465/-427，可读 patch 1334 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend][last/5] Improve pooling entrypoints | clean up.」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/models/pooling_models/README.md`, `docs/models/pooling_models/scoring.md`, `docs/models/pooling_models/token_classify.md`。
- 实现要点:
  - `docs/models/pooling_models/README.md` modified +10/-0
  - `docs/models/pooling_models/scoring.md` modified +4/-0
  - `docs/models/pooling_models/token_classify.md` modified +1/-1
  - `docs/serving/openai_compatible_server.md` modified +1/-18
- 代码 diff 细节:
  - `docs/models/pooling_models/README.md` modified +10/-0
  - `docs/models/pooling_models/scoring.md` modified +4/-0
  - `docs/models/pooling_models/token_classify.md` modified +1/-1
  - `docs/serving/openai_compatible_server.md` modified +1/-18
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models/README.md
@@ -59,6 +59,16 @@ please refer to [IO Processor Plugins](../../design/io_processor_plugins.md).
     Within classification tasks, there is a specialized subcategory: Cross-encoder (aka reranker) models. These models
 are a subset of classification models that accept two prompts as input and output num_labels equal to 1.

+### Pooling Types
+
+| Pooling Tasks  | Granularity   | Description                                                                                                                                                                                       |
+|----------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
+| `CLS` pooling  | Sequence-wise | For BERT‑like (bidirectional self‑attention) models, CLS pooling is used by default. This means the last_hidden_states corresponding to the first token (the [CLS] token) is taken as the output. |
+| `LAST` pooling | Sequence-wise | For GPT‑like (causal self‑attention) models, LAST pooling is used by default. This means the last_hidden_states corresponding to the last token is taken as the output.                           |
+| `MEAN` pooling | Sequence-wise | Many studies have shown that averaging the last_hidden_states over all input tokens performs better on certain downstream tasks. Therefore, more and more models are using MEAN pooling.          |
+| `ALL` pooling  | Token-wise    | Outputs the last_hidden_states for all input tokens.                                                                                                                                              |
+| `STEP` pooling | Token-wise    | Filters and outputs the last_hidden_states corresponding to the token IDs returned by returned_token_ids.                                                                                         |
+
diff -- docs/models/pooling_models/scoring.md
@@ -160,6 +160,8 @@ The following Score API parameters are supported:
 --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
 --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
 --8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
+--8<-- "vllm/entrypoints/pooling/scoring/protocol.py:scoring-common-params"
+--8<-- "vllm/entrypoints/pooling/scoring/protocol.py:score-request-params"
 ```

 #### Examples
@@ -370,6 +372,8 @@ The following rerank api parameters are supported:
 --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
 --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
 --8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
+--8<-- "vllm/entrypoints/pooling/scoring/protocol.py:scoring-common-params"
```
- 已读文件:
  - runtime: `vllm/entrypoints/llm.py` modified +1/-1; `vllm/entrypoints/openai/api_server.py` modified +14/-16; `vllm/entrypoints/openai/generate/factories.py` added +42/-0; `vllm/entrypoints/pooling/__init__.py` modified +0/-130; `vllm/entrypoints/pooling/base/io_processor.py` modified +15/-8; `vllm/entrypoints/pooling/base/serving.py` modified +1/-1; `vllm/entrypoints/pooling/classify/api_router.py` modified +3/-2; `vllm/entrypoints/pooling/classify/io_processor.py` modified +1/-1
  - docs/bench: `docs/models/pooling_models/README.md` modified +10/-0; `docs/models/pooling_models/scoring.md` modified +4/-0; `docs/models/pooling_models/token_classify.md` modified +1/-1; `docs/serving/openai_compatible_server.md` modified +1/-18
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #41832 - [Doc] Add ModernBertForSequenceClassification to scoring.md cross-en…

- 链接: https://github.com/vllm-project/vllm/pull/41832
- 状态/时间: merged / 2026-05-06
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc] Add ModernBertForSequenceClassification to scoring.md cross-en…」；模型线: Jina Reranker M0；类别: 模型支持/运行时入口；主要 diff: `docs/models/pooling_models/scoring.md`。
- 实现要点:
  - `docs/models/pooling_models/scoring.md` modified +1/-0
- 代码 diff 细节:
  - `docs/models/pooling_models/scoring.md` modified +1/-0
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models/scoring.md
@@ -41,6 +41,7 @@ The score models is designed to compute similarity scores between two input prom
 | `GemmaForSequenceClassification` | Gemma-based | `BAAI/bge-reranker-v2-gemma`(see note), etc. | [bge-reranker-v2-gemma.jinja](../../../examples/pooling/score/template/bge-reranker-v2-gemma.jinja) | ✅︎ | ✅︎ |
 | `GteNewForSequenceClassification` | mGTE-TRM (see note) | `Alibaba-NLP/gte-multilingual-reranker-base`, etc. | N/A | | |
 | `LlamaBidirectionalForSequenceClassification`<sup>C</sup> | Llama-based with bidirectional attention | `nvidia/llama-nemotron-rerank-1b-v2`, etc. | [nemotron-rerank.jinja](../../../examples/pooling/score/template/nemotron-rerank.jinja) | ✅︎ | ✅︎ |
+| `ModernBertForSequenceClassification` | ModernBERT-based | `Alibaba-NLP/gte-reranker-modernbert-base`, etc. | N/A | | |
 | `Qwen2ForSequenceClassification`<sup>C</sup> | Qwen2-based | `mixedbread-ai/mxbai-rerank-base-v2`(see note), etc. | [mxbai_rerank_v2.jinja](../../../examples/pooling/score/template/mxbai_rerank_v2.jinja) | ✅︎ | ✅︎ |
 | `Qwen3ForSequenceClassification`<sup>C</sup> | Qwen3-based | `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`, `Qwen/Qwen3-Reranker-0.6B`(see note), etc. | [qwen3_reranker.jinja](../../../examples/pooling/score/template/qwen3_reranker.jinja) | ✅︎ | ✅︎ |
 | `RobertaForSequenceClassification` | RoBERTa-based | `cross-encoder/quora-roberta-base`, etc. | N/A | | |
```
- 已读文件:
  - docs/bench: `docs/models/pooling_models/scoring.md` modified +1/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #42267 - [Entrypoints] Split the pooling offline API into PoolingOfflineMixin.

- 链接: https://github.com/vllm-project/vllm/pull/42267
- 状态/时间: merged / 2026-05-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+531/-439，可读 patch 1121 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Entrypoints] Split the pooling offline API into PoolingOfflineMixin.」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/models/pooling_models/README.md`, `docs/models/pooling_models/classify.md`, `docs/models/pooling_models/embed.md`。
- 实现要点:
  - `docs/models/pooling_models/README.md` modified +4/-4
  - `docs/models/pooling_models/classify.md` modified +2/-2
  - `docs/models/pooling_models/embed.md` modified +3/-3
  - `docs/models/pooling_models/reward.md` modified +1/-1
- 代码 diff 细节:
  - `docs/models/pooling_models/README.md` modified +4/-4
  - `docs/models/pooling_models/classify.md` modified +2/-2
  - `docs/models/pooling_models/embed.md` modified +3/-3
  - `docs/models/pooling_models/reward.md` modified +1/-1
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models/README.md
@@ -131,24 +131,24 @@ enabling the corresponding APIs.

 ### `LLM.classify`

-The [classify][vllm.LLM.classify] method outputs a probability vector for each prompt.
+The [classify][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.classify] method outputs a probability vector for each prompt.
 It is primarily designed for [classification models](classify.md).
 For more information about `LLM.embed`, see [this page](classify.md#offline-inference).

 ### `LLM.embed`

-The [embed][vllm.LLM.embed] method outputs an embedding vector for each prompt.
+The [embed][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.embed] method outputs an embedding vector for each prompt.
 It is primarily designed for [embedding models](embed.md).
diff -- docs/models/pooling_models/classify.md
@@ -77,7 +77,7 @@ The following [pooling parameters][vllm.PoolingParams] are supported.

 ### `LLM.classify`

-The [classify][vllm.LLM.classify] method outputs a probability vector for each prompt.
+The [classify][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.classify] method outputs a probability vector for each prompt.

 ```python
 from vllm import LLM
@@ -93,7 +93,7 @@ A code example can be found here: [examples/basic/offline_inference/classify.py]

 ### `LLM.encode`

-The [encode][vllm.LLM.encode] method is available to all pooling models in vLLM.
```
- 已读文件:
  - runtime: `vllm/entrypoints/llm.py` modified +7/-425; `vllm/entrypoints/pooling/offline.py` added +510/-0
  - docs/bench: `docs/models/pooling_models/README.md` modified +4/-4; `docs/models/pooling_models/classify.md` modified +2/-2; `docs/models/pooling_models/embed.md` modified +3/-3; `docs/models/pooling_models/reward.md` modified +1/-1; `docs/models/pooling_models/scoring.md` modified +1/-1; `docs/models/pooling_models/token_classify.md` modified +1/-1; `docs/models/pooling_models/token_embed.md` modified +2/-2
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #41907 - [Docs] Reorganize online serving docs.

- 链接: https://github.com/vllm-project/vllm/pull/41907
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 24 个文件，+1348/-1241，可读 patch 1469 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Reorganize online serving docs.」；模型线: Jina Reranker M0；类别: 文档/测试/CI；主要 diff: `docs/.nav.yml`, `docs/assets/models/pooling_models/cheat_sheet.svg`, `docs/configuration/README.md`。
- 实现要点:
  - `docs/.nav.yml` modified +1/-1
  - `docs/assets/models/pooling_models/cheat_sheet.svg` modified +671/-660
  - `docs/configuration/README.md` modified +1/-1
  - `docs/configuration/engine_args.md` modified +1/-1
- 代码 diff 细节:
  - `docs/.nav.yml` modified +1/-1
  - `docs/assets/models/pooling_models/cheat_sheet.svg` modified +671/-660
  - `docs/configuration/README.md` modified +1/-1
  - `docs/configuration/engine_args.md` modified +1/-1
- 关键代码摘录:

```diff
diff -- docs/.nav.yml
@@ -11,7 +11,7 @@ nav:
       - usage/*
     - Inference and Serving:
       - serving/offline_inference.md
-      - serving/openai_compatible_server.md
+      - Online Serving: serving/online_serving
       - serving/*
       - serving/integrations
     - Deployment:
diff -- docs/configuration/README.md
@@ -4,6 +4,6 @@ This section lists the most common options for running vLLM.

 There are three main levels of configuration, from highest priority to lowest priority:

-- [Request parameters](../serving/openai_compatible_server.md#completions-api) and [input arguments](../api/README.md#inference-parameters)
+- [Request parameters](../serving/online_serving/openai_compatible_server.md#completions-api) and [input arguments](../api/README.md#inference-parameters)
 - [Engine arguments](./engine_args.md)
 - [Environment variables](./env_vars.md)
```
- 已读文件:
  - docs/bench: `docs/.nav.yml` modified +1/-1; `docs/assets/models/pooling_models/cheat_sheet.svg` modified +671/-660; `docs/configuration/README.md` modified +1/-1; `docs/configuration/engine_args.md` modified +1/-1; `docs/design/arch_overview.md` modified +3/-3; `docs/features/structured_outputs.md` modified +1/-1; `docs/getting_started/installation/gpu.apple.inc.md` modified +1/-1; `docs/getting_started/quickstart.md` modified +3/-3
  - other: `mkdocs.yaml` modified +1/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #42626 - [Docs] Add SVG images for pooling models.

- 链接: https://github.com/vllm-project/vllm/pull/42626
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+2336/-0，可读 patch 44 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add SVG images for pooling models.」；模型线: Jina Reranker M0；类别: 模型支持/运行时入口；主要 diff: `docs/assets/models/pooling_models/cheat_sheet.svg`, `docs/assets/models/pooling_models/pooling_types.svg`, `docs/assets/models/pooling_models/score_types.svg`。
- 实现要点:
  - `docs/assets/models/pooling_models/cheat_sheet.svg` added +785/-0
  - `docs/assets/models/pooling_models/pooling_types.svg` added +633/-0
  - `docs/assets/models/pooling_models/score_types.svg` added +902/-0
  - `docs/models/pooling_models/README.md` modified +10/-0
- 代码 diff 细节:
  - `docs/assets/models/pooling_models/cheat_sheet.svg` added +785/-0
  - `docs/assets/models/pooling_models/pooling_types.svg` added +633/-0
  - `docs/assets/models/pooling_models/score_types.svg` added +902/-0
  - `docs/models/pooling_models/README.md` modified +10/-0
- 关键代码摘录:

```diff
diff -- docs/models/pooling_models/README.md
@@ -33,6 +33,12 @@ from large language models, allowing them to benefit from the continuous improve
 similarity enables them to reuse much of vLLM’s infrastructure. If compatible, we would be happy to help them leverage
 the latest features of vLLM as well.

+### Cheat Sheet
+
+As illustrated in the figure below, we have summarized the relationships among the key elements of pooling models as a takeaway.
+
+![Cheat Sheet](../../assets/models/pooling_models/cheat_sheet.svg)
+
 ### Sequence-wise Task and Token-wise Task

 The key distinction between sequence-wise task and token-wise task lies in their output granularity: sequence-wise task
@@ -61,6 +67,8 @@ are a subset of classification models that accept two prompts as input and outpu
diff -- docs/models/pooling_models/scoring.md
@@ -25,6 +25,12 @@ The score models is designed to compute similarity scores between two input prom
 !!! note
     Only when a classification model outputs num_labels equal to 1 can it be used as a scoring model and have its scoring API enabled.

+### Score Types
+
+The three supported scoring functions are as illustrated in the figure below.
+
+![Score Types](../../assets/models/pooling_models/score_types.svg)
+
 ## Supported Models

 ### Cross-encoder models
```
- 已读文件:
  - docs/bench: `docs/assets/models/pooling_models/cheat_sheet.svg` added +785/-0; `docs/assets/models/pooling_models/pooling_types.svg` added +633/-0; `docs/assets/models/pooling_models/score_types.svg` added +902/-0; `docs/models/pooling_models/README.md` modified +10/-0; `docs/models/pooling_models/scoring.md` modified +6/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。
