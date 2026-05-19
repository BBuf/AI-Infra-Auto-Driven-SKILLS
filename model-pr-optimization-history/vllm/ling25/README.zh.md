# vllm Ling 2.5 1T 模型 PR 优化历史

## 2026-05-19 新增覆盖

按 vllm 上游 `origin/main@ef54a4d604`、模型相关文件的 `git log --name-only -- <model-files>` 以及 GitHub Pull Request files API 生成。本页用于补齐 sgl-cookbook 中 `Ling 2.5 1T` 缺失的历史 PR 优化文档。

> vLLM 的 Ling/Ring 支持走 Bailing MoE 实现；`docs/models/supported_models.md` 只作为能力表，不纳入逐 PR 卡片以避免引入全仓 supported-models 噪声。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `vllm/model_executor/models/bailing_moe.py` | [#40671](https://github.com/vllm-project/vllm/pull/40671), [#35782](https://github.com/vllm-project/vllm/pull/35782), [#35949](https://github.com/vllm-project/vllm/pull/35949), [#33737](https://github.com/vllm-project/vllm/pull/33737), [#32064](https://github.com/vllm-project/vllm/pull/32064), [#33063](https://github.com/vllm-project/vllm/pull/33063), [#31104](https://github.com/vllm-project/vllm/pull/31104), [#30389](https://github.com/vllm-project/vllm/pull/30389), [#29966](https://github.com/vllm-project/vllm/pull/29966), [#29342](https://github.com/vllm-project/vllm/pull/29342), [#28542](https://github.com/vllm-project/vllm/pull/28542), [#28777](https://github.com/vllm-project/vllm/pull/28777), ... (28 total) |
| `vllm/model_executor/models/bailing_moe_linear.py` | [#41188](https://github.com/vllm-project/vllm/pull/41188), [#41185](https://github.com/vllm-project/vllm/pull/41185), [#40859](https://github.com/vllm-project/vllm/pull/40859), [#40671](https://github.com/vllm-project/vllm/pull/40671), [#35782](https://github.com/vllm-project/vllm/pull/35782), [#35949](https://github.com/vllm-project/vllm/pull/35949), [#37487](https://github.com/vllm-project/vllm/pull/37487), [#37195](https://github.com/vllm-project/vllm/pull/37195), [#35102](https://github.com/vllm-project/vllm/pull/35102) |

## PR 覆盖总览

- git 追溯 PR 数: 34
- 关键词/补充 PR 数: 0
- 当前文档总 PR 数: 34
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-14 | [#20680](https://github.com/vllm-project/vllm/pull/20680) | merged | [Model] Add Ling implementation | `docs/models/supported_models.md`, `tests/models/registry.py`, `vllm/model_executor/models/bailing_moe.py` |
| 2025-07-16 | [#21059](https://github.com/vllm-project/vllm/pull/21059) | merged | [Model] Remove model sampler | `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/granite_speech.py`, `vllm/model_executor/models/hunyuan_v1_moe.py` |
| 2025-07-19 | [#21100](https://github.com/vllm-project/vllm/pull/21100) | merged | [Quantization] Enable BNB support for more MoE models | `docs/models/supported_models.md`, `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/ernie45_moe.py` |
| 2025-07-26 | [#21664](https://github.com/vllm-project/vllm/pull/21664) | merged | support `torch.compile` for bailing moe | `vllm/model_executor/models/bailing_moe.py` |
| 2025-08-29 | [#19497](https://github.com/vllm-project/vllm/pull/19497) | merged | [Models] Improve iteration over layers | `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/arctic.py`, `vllm/model_executor/models/baichuan.py` |
| 2025-09-15 | [#24627](https://github.com/vllm-project/vllm/pull/24627) | merged | [Model]: support Ling2.0 | `docs/models/supported_models.md`, `tests/models/registry.py`, `vllm/model_executor/models/bailing_moe.py` |
| 2025-09-21 | [#25345](https://github.com/vllm-project/vllm/pull/25345) | merged | [V0 Deprecation] Remove V0 sampling metadata | `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_opt.py`, `vllm/model_executor/__init__.py` |
| 2025-09-30 | [#25271](https://github.com/vllm-project/vllm/pull/25271) | merged | Move`VllmConfig` from `config/__init__.py` to `config/vllm.py` | `vllm/attention/layer.py`, `vllm/attention/layers/chunked_local_attention.py`, `vllm/config/__init__.py` |
| 2025-10-05 | [#26247](https://github.com/vllm-project/vllm/pull/26247) | merged | Convert formatting to use `ruff` instead of `yapf` + `isort` | `.buildkite/pyproject.toml`, `.pre-commit-config.yaml`, `benchmarks/benchmark_block_pool.py` |
| 2025-10-06 | [#26262](https://github.com/vllm-project/vllm/pull/26262) | merged | Fix per file ruff ignores related to line length | `benchmarks/benchmark_ngram_proposer.py`, `benchmarks/benchmark_serving_structured_output.py`, `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` |
| 2025-10-09 | [#26145](https://github.com/vllm-project/vllm/pull/26145) | merged | [Model] Apply shared experts overlap optimization to all models with shared experts | `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/layers/quantization/fp8.py` |
| 2025-10-12 | [#26633](https://github.com/vllm-project/vllm/pull/26633) | merged | Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y` | `benchmarks/backend_request_func.py`, `benchmarks/benchmark_prefix_caching.py`, `benchmarks/benchmark_prioritization.py` |
| 2025-10-15 | [#26876](https://github.com/vllm-project/vllm/pull/26876) | merged | [Fix] Remove divisibility requirement between num_kv_heads and tp_size in bailing_moe | `vllm/model_executor/models/bailing_moe.py` |
| 2025-11-11 | [#28382](https://github.com/vllm-project/vllm/pull/28382) | merged | [LoRA][1/N]Remove LoRA extra vocab | `vllm/model_executor/models/apertus.py`, `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/arctic.py` |
| 2025-11-13 | [#27583](https://github.com/vllm-project/vllm/pull/27583) | merged | Rename clashing method names for vLLM model protocol | `docs/contributing/model/basic.md`, `docs/contributing/model/multimodal.md`, `vllm/model_executor/models/apertus.py` |
| 2025-11-14 | [#28277](https://github.com/vllm-project/vllm/pull/28277) | merged | [Model] Fix bailing_moe accuracy problem | `vllm/model_executor/models/bailing_moe.py` |
| 2025-11-15 | [#28777](https://github.com/vllm-project/vllm/pull/28777) | merged | [Model] Fix lmhead init bug of bailing_moe | `vllm/model_executor/models/bailing_moe.py` |
| 2025-11-19 | [#28542](https://github.com/vllm-project/vllm/pull/28542) | merged | Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5 | `.buildkite/test-pipeline.yaml`, `benchmarks/kernels/benchmark_mrope.py`, `examples/offline_inference/context_extension.py` |
| 2025-11-26 | [#29342](https://github.com/vllm-project/vllm/pull/29342) | merged | [Attention] Remove imports from `vllm/attention/__init__.py` | `docs/contributing/model/basic.md`, `tests/compile/test_fusion_attn.py`, `tests/compile/test_qk_norm_rope_fusion.py` |
| 2025-12-04 | [#29966](https://github.com/vllm-project/vllm/pull/29966) | merged | Access `partial_rotary_factor` from `rope_parameters` | `tests/kernels/core/test_mrope.py`, `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/model_executor/models/apertus.py` |
| 2025-12-11 | [#30389](https://github.com/vllm-project/vllm/pull/30389) | merged | Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim` | `benchmarks/kernels/benchmark_mrope.py`, `benchmarks/kernels/benchmark_rope.py`, `tests/compile/test_functionalization.py` |
| 2026-01-07 | [#31104](https://github.com/vllm-project/vllm/pull/31104) | merged | [BugFix] LoRA: Support loading base_layer of experts | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/bailing_moe.py` |
| 2026-01-26 | [#33063](https://github.com/vllm-project/vllm/pull/33063) | merged | [Chore] Update type annotation of `input_ids` in model forward | `docs/contributing/model/basic.md`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`, `vllm/model_executor/models/afmoe.py` |
| 2026-01-27 | [#32064](https://github.com/vllm-project/vllm/pull/32064) | merged | [5/N][Attention] Finish eliminating `vllm/attention` folder | `.buildkite/test-amd.yaml`, `.buildkite/test-pipeline.yaml`, `.buildkite/test_areas/kernels.yaml` |
| 2026-02-04 | [#33737](https://github.com/vllm-project/vllm/pull/33737) | merged | [Bugfix] Define router_logits_dtype for remaining MoE models | `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/flex_olmo.py` |
| 2026-02-26 | [#35102](https://github.com/vllm-project/vllm/pull/35102) | merged | [Model] Ring 2.5 | `docs/models/supported_models.md`, `tests/models/registry.py`, `vllm/model_executor/layers/fla/ops/layernorm_guard.py` |
| 2026-03-18 | [#37195](https://github.com/vllm-project/vllm/pull/37195) | merged | [V0 Deprecation] Deprecate virtual engine | `tests/compile/passes/test_rope_kvcache_fusion.py`, `tests/v1/kv_connector/unit/test_decode_bench_connector.py`, `tests/v1/kv_connector/unit/test_lmcache_integration.py` |
| 2026-03-24 | [#37487](https://github.com/vllm-project/vllm/pull/37487) | merged | [V0 Deprecation] Refactor kv cache from list to element | `tests/compile/passes/test_fusion_attn.py`, `tests/compile/passes/test_rope_kvcache_fusion.py`, `tests/v1/e2e/general/test_mamba_prefix_cache.py` |
| 2026-04-20 | [#35949](https://github.com/vllm-project/vllm/pull/35949) | merged | [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase | `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py`, `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py` |
| 2026-04-21 | [#35782](https://github.com/vllm-project/vllm/pull/35782) | merged | [MoE Refactor] Remove SharedFusedMoE class | `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`, `vllm/distributed/device_communicators/base_device_communicator.py` |
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | merged | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py` |
| 2026-04-28 | [#40859](https://github.com/vllm-project/vllm/pull/40859) | merged | [Bugfix ] fix bailing_moe_linear | `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/models/bailing_moe_linear.py` |
| 2026-04-29 | [#41185](https://github.com/vllm-project/vllm/pull/41185) | merged | [Bugfix] BailingMoeV2.5: rotate full qk_rope_head_dim in MLA RoPE | `vllm/model_executor/models/bailing_moe_linear.py` |
| 2026-05-11 | [#41188](https://github.com/vllm-project/vllm/pull/41188) | merged | [Misc] Replace mamba_type string literals with MambaAttentionBackendEnum | `docs/contributing/model/basic.md`, `tests/kernels/mamba/test_ssu_dispatch.py`, `tests/v1/attention/test_attention_backends_selection.py` |

## 逐 PR diff 审计卡

### PR #20680 - [Model] Add Ling implementation

- 链接: https://github.com/vllm-project/vllm/pull/20680
- 状态/时间: merged / 2025-07-14
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+534/-0，可读 patch 556 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add Ling implementation」；模型线: Ling 2.5 1T；类别: 模型支持/运行时入口；主要 diff: `docs/models/supported_models.md`, `tests/models/registry.py`, `vllm/model_executor/models/bailing_moe.py`。
- 实现要点:
  - `docs/models/supported_models.md` modified +1/-0
  - `tests/models/registry.py` modified +2/-0
  - `vllm/model_executor/models/bailing_moe.py` added +530/-0；symbols: BailingAttention, __init__, forward, BailingMLP
  - `vllm/model_executor/models/registry.py` modified +1/-0
- 代码 diff 细节:
  - `docs/models/supported_models.md` modified +1/-0
  - `tests/models/registry.py` modified +2/-0
  - `vllm/model_executor/models/bailing_moe.py` added +530/-0
  - `vllm/model_executor/models/registry.py` modified +1/-0
- 关键代码摘录:

```diff
diff -- docs/models/supported_models.md
@@ -316,6 +316,7 @@ Specified using `--task generate`.
 | `AquilaForCausalLM` | Aquila, Aquila2 | `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc. | ✅︎ | ✅︎ | ✅︎ |
 | `ArcticForCausalLM` | Arctic | `Snowflake/snowflake-arctic-base`, `Snowflake/snowflake-arctic-instruct`, etc. | | ✅︎ | ✅︎ |
 | `BaiChuanForCausalLM` | Baichuan2, Baichuan | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc. | ✅︎ | ✅︎ | ✅︎ |
+| `BailingMoeForCausalLM` | Ling | `inclusionAI/Ling-lite-1.5`, `inclusionAI/Ling-plus`, etc. | | ✅︎ | ✅︎ |
 | `BambaForCausalLM` | Bamba | `ibm-ai-platform/Bamba-9B-fp8`, `ibm-ai-platform/Bamba-9B` | ✅︎ | ✅︎ | ✅︎ |
 | `BloomForCausalLM` | BLOOM, BLOOMZ, BLOOMChat | `bigscience/bloom`, `bigscience/bloomz`, etc. | | ✅︎ | |
 | `BartForConditionalGeneration` | BART | `facebook/bart-base`, `facebook/bart-large-cnn`, etc. | | | |
diff -- tests/models/registry.py
@@ -141,6 +141,8 @@ def check_available_online(
                                          trust_remote_code=True),
     "BaichuanForCausalLM": _HfExamplesInfo("baichuan-inc/Baichuan2-7B-chat",
                                          trust_remote_code=True),
+    "BailingMoeForCausalLM": _HfExamplesInfo("inclusionAI/Ling-lite-1.5",
+                                         trust_remote_code=True),
     "BambaForCausalLM": _HfExamplesInfo("ibm-ai-platform/Bamba-9B",
                                         extras={"tiny": "hmellor/tiny-random-BambaForCausalLM"}),  # noqa: E501
     "BloomForCausalLM": _HfExamplesInfo("bigscience/bloom-560m",
```
- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` added +530/-0; `vllm/model_executor/models/registry.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +2/-0
  - docs/bench: `docs/models/supported_models.md` modified +1/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #21059 - [Model] Remove model sampler

- 链接: https://github.com/vllm-project/vllm/pull/21059
- 状态/时间: merged / 2025-07-16
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+0/-45，可读 patch 157 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Remove model sampler」；模型线: Ling 2.5 1T；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/granite_speech.py`, `vllm/model_executor/models/hunyuan_v1_moe.py`。
- 实现要点:
  - `vllm/model_executor/models/bailing_moe.py` modified +0/-10
  - `vllm/model_executor/models/granite_speech.py` modified +0/-2
  - `vllm/model_executor/models/hunyuan_v1_moe.py` modified +0/-10
  - `vllm/model_executor/models/mimo.py` modified +0/-2
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe.py` modified +0/-10
  - `vllm/model_executor/models/granite_speech.py` modified +0/-2
  - `vllm/model_executor/models/hunyuan_v1_moe.py` modified +0/-10
  - `vllm/model_executor/models/mimo.py` modified +0/-2
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe.py
@@ -47,7 +47,6 @@
 from vllm.model_executor.layers.quantization.base_config import (
     QuantizationConfig)
 from vllm.model_executor.layers.rotary_embedding import get_rope
-from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
 from vllm.model_executor.layers.vocab_parallel_embedding import (
     ParallelLMHead, VocabParallelEmbedding)
 from vllm.model_executor.model_loader.weight_utils import default_weight_loader
@@ -485,7 +484,6 @@ def __init__(
         else:
             self.lm_head = PPMissingLayer()

-        self.sampler = get_sampler()
         self.make_empty_intermediate_tensors = (
diff -- vllm/model_executor/models/granite_speech.py
@@ -36,7 +36,6 @@
 from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                                RowParallelLinear)
 from vllm.model_executor.layers.quantization import QuantizationConfig
-from vllm.model_executor.layers.sampler import get_sampler
 from vllm.model_executor.models.module_mapping import MultiModelKeys
 from vllm.model_executor.sampling_metadata import SamplingMetadata
 from vllm.multimodal import MULTIMODAL_REGISTRY
@@ -549,7 +548,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str):
         self.config = config
         self.quant_config = quant_config
         self.cache_config = cache_config
-        self.sampler = get_sampler()

```
- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +0/-10; `vllm/model_executor/models/granite_speech.py` modified +0/-2; `vllm/model_executor/models/hunyuan_v1_moe.py` modified +0/-10; `vllm/model_executor/models/mimo.py` modified +0/-2; `vllm/model_executor/models/mimo_mtp.py` modified +0/-11; `vllm/model_executor/models/phi4flash.py` modified +0/-10
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #21100 - [Quantization] Enable BNB support for more MoE models

- 链接: https://github.com/vllm-project/vllm/pull/21100
- 状态/时间: merged / 2025-07-19
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+223/-181，可读 patch 548 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] Enable BNB support for more MoE models」；模型线: Ling 2.5 1T；类别: 模型支持/运行时入口；主要 diff: `docs/models/supported_models.md`, `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/ernie45_moe.py`。
- 实现要点:
  - `docs/models/supported_models.md` modified +4/-4
  - `vllm/model_executor/models/bailing_moe.py` modified +14/-7；symbols: get_expert_mapping, BailingMoeForCausalLM
  - `vllm/model_executor/models/ernie45_moe.py` modified +84/-69；symbols: get_expert_mapping, Ernie4_5_MoeForCausalLM, __init__, get_input_embeddings
  - `vllm/model_executor/models/grok1.py` modified +14/-10；symbols: get_expert_mapping
- 代码 diff 细节:
  - `docs/models/supported_models.md` modified +4/-4
  - `vllm/model_executor/models/bailing_moe.py` modified +14/-7
  - `vllm/model_executor/models/ernie45_moe.py` modified +84/-69
  - `vllm/model_executor/models/grok1.py` modified +14/-10
- 关键代码摘录:

```diff
diff -- docs/models/supported_models.md
@@ -316,7 +316,7 @@ Specified using `--task generate`.
 | `AquilaForCausalLM` | Aquila, Aquila2 | `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc. | ✅︎ | ✅︎ | ✅︎ |
 | `ArcticForCausalLM` | Arctic | `Snowflake/snowflake-arctic-base`, `Snowflake/snowflake-arctic-instruct`, etc. | | ✅︎ | ✅︎ |
 | `BaiChuanForCausalLM` | Baichuan2, Baichuan | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc. | ✅︎ | ✅︎ | ✅︎ |
-| `BailingMoeForCausalLM` | Ling | `inclusionAI/Ling-lite-1.5`, `inclusionAI/Ling-plus`, etc. | | ✅︎ | ✅︎ |
+| `BailingMoeForCausalLM` | Ling | `inclusionAI/Ling-lite-1.5`, `inclusionAI/Ling-plus`, etc. | ✅︎ | ✅︎ | ✅︎ |
 | `BambaForCausalLM` | Bamba | `ibm-ai-platform/Bamba-9B-fp8`, `ibm-ai-platform/Bamba-9B` | ✅︎ | ✅︎ | ✅︎ |
 | `BloomForCausalLM` | BLOOM, BLOOMZ, BLOOMChat | `bigscience/bloom`, `bigscience/bloomz`, etc. | | ✅︎ | |
 | `BartForConditionalGeneration` | BART | `facebook/bart-base`, `facebook/bart-large-cnn`, etc. | | | |
@@ -328,8 +328,8 @@ Specified using `--task generate`.
 | `DeepseekV2ForCausalLM` | DeepSeek-V2 | `deepseek-ai/DeepSeek-V2`, `deepseek-ai/DeepSeek-V2-Chat`, etc. | | ✅︎ | ✅︎ |
 | `DeepseekV3ForCausalLM` | DeepSeek-V3 | `deepseek-ai/DeepSeek-V3-Base`, `deepseek-ai/DeepSeek-V3`, etc. | | ✅︎ | ✅︎ |
 | `Dots1ForCausalLM` | dots.llm1 | `rednote-hilab/dots.llm1.base`, `rednote-hilab/dots.llm1.inst`, etc. | | ✅︎ | ✅︎ |
-| `Ernie4_5_ForCausalLM` | Ernie4.5 | `baidu/ERNIE-4.5-0.3B-PT`, etc. | | ✅︎ | ✅︎ |
diff -- vllm/model_executor/models/bailing_moe.py
@@ -53,7 +53,7 @@
 from vllm.model_executor.sampling_metadata import SamplingMetadata
 from vllm.sequence import IntermediateTensors

-from .interfaces import SupportsPP
+from .interfaces import SupportsLoRA, SupportsPP
 from .utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                     make_empty_intermediate_tensors_factory, make_layers,
                     maybe_prefix)
@@ -374,21 +374,25 @@ def forward(
         hidden_states, _ = self.norm(hidden_states, residual)
         return hidden_states

+    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
```
- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +14/-7; `vllm/model_executor/models/ernie45_moe.py` modified +84/-69; `vllm/model_executor/models/grok1.py` modified +14/-10; `vllm/model_executor/models/hunyuan_v1_moe.py` modified +107/-91
  - docs/bench: `docs/models/supported_models.md` modified +4/-4
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #21664 - support `torch.compile` for bailing moe

- 链接: https://github.com/vllm-project/vllm/pull/21664
- 状态/时间: merged / 2025-07-26
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support `torch.compile` for bailing moe」；模型线: Ling 2.5 1T；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/bailing_moe.py`。
- 实现要点:
  - `vllm/model_executor/models/bailing_moe.py` modified +2/-0
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe.py` modified +2/-0
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe.py
@@ -32,6 +32,7 @@
 from transformers.configuration_utils import PretrainedConfig

 from vllm.attention import Attention
+from vllm.compilation.decorators import support_torch_compile
 from vllm.config import CacheConfig, VllmConfig
 from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                               get_tensor_model_parallel_world_size,
@@ -291,6 +292,7 @@ def forward(
         return hidden_states, residual


+@support_torch_compile
 class BailingMoeModel(nn.Module):
```
- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +2/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #19497 - [Models] Improve iteration over layers

- 链接: https://github.com/vllm-project/vllm/pull/19497
- 状态/时间: merged / 2025-08-29
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 65 个文件，+129/-83，可读 patch 1109 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models] Improve iteration over layers」；模型线: Ling 2.5 1T；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/arctic.py`, `vllm/model_executor/models/baichuan.py`。
- 实现要点:
  - `vllm/model_executor/models/arcee.py` modified +2/-1
  - `vllm/model_executor/models/arctic.py` modified +2/-1
  - `vllm/model_executor/models/baichuan.py` modified +2/-1
  - `vllm/model_executor/models/bailing_moe.py` modified +2/-2
- 代码 diff 细节:
  - `vllm/model_executor/models/arcee.py` modified +2/-1
  - `vllm/model_executor/models/arctic.py` modified +2/-1
  - `vllm/model_executor/models/baichuan.py` modified +2/-1
  - `vllm/model_executor/models/bailing_moe.py` modified +2/-2
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/arcee.py
@@ -9,6 +9,7 @@
 # activation.

 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union

 import torch
@@ -243,7 +244,7 @@ def forward(

         aux_hidden_states: list[torch.Tensor] = []
         for idx, layer in enumerate(
-                self.layers[self.start_layer:self.end_layer]):
+                islice(self.layers, self.start_layer, self.end_layer)):
diff -- vllm/model_executor/models/arctic.py
@@ -2,6 +2,7 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 """Inference-only Snowflake Arctic model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union

 import torch
@@ -403,7 +404,7 @@ def forward(
         else:
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
```
- 已读文件:
  - runtime: `vllm/model_executor/models/arcee.py` modified +2/-1; `vllm/model_executor/models/arctic.py` modified +2/-1; `vllm/model_executor/models/baichuan.py` modified +2/-1; `vllm/model_executor/models/bailing_moe.py` modified +2/-2; `vllm/model_executor/models/bamba.py` modified +1/-2; `vllm/model_executor/models/bloom.py` modified +2/-1; `vllm/model_executor/models/chameleon.py` modified +2/-1; `vllm/model_executor/models/chatglm.py` modified +2/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #24627 - [Model]: support Ling2.0

- 链接: https://github.com/vllm-project/vllm/pull/24627
- 状态/时间: merged / 2025-09-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+170/-50，可读 patch 388 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model]: support Ling2.0」；模型线: Ling 2.5 1T；类别: 模型支持/运行时入口；主要 diff: `docs/models/supported_models.md`, `tests/models/registry.py`, `vllm/model_executor/models/bailing_moe.py`。
- 实现要点:
  - `docs/models/supported_models.md` modified +1/-0
  - `tests/models/registry.py` modified +2/-0
  - `vllm/model_executor/models/bailing_moe.py` modified +166/-50；symbols: BailingMoeV2ForCausalLM
  - `vllm/model_executor/models/registry.py` modified +1/-0
- 代码 diff 细节:
  - `docs/models/supported_models.md` modified +1/-0
  - `tests/models/registry.py` modified +2/-0
  - `vllm/model_executor/models/bailing_moe.py` modified +166/-50
  - `vllm/model_executor/models/registry.py` modified +1/-0
- 关键代码摘录:

```diff
diff -- docs/models/supported_models.md
@@ -328,6 +328,7 @@ th {
 | `ArcticForCausalLM` | Arctic | `Snowflake/snowflake-arctic-base`, `Snowflake/snowflake-arctic-instruct`, etc. | | ✅︎ | ✅︎ |
 | `BaiChuanForCausalLM` | Baichuan2, Baichuan | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc. | ✅︎ | ✅︎ | ✅︎ |
 | `BailingMoeForCausalLM` | Ling | `inclusionAI/Ling-lite-1.5`, `inclusionAI/Ling-plus`, etc. | ✅︎ | ✅︎ | ✅︎ |
+| `BailingMoeV2ForCausalLM` | Ling | `inclusionAI/Ling-mini-2.0`, etc. | ✅︎ | ✅︎ | ✅︎ |
 | `BambaForCausalLM` | Bamba | `ibm-ai-platform/Bamba-9B-fp8`, `ibm-ai-platform/Bamba-9B` | ✅︎ | ✅︎ | ✅︎ |
 | `BloomForCausalLM` | BLOOM, BLOOMZ, BLOOMChat | `bigscience/bloom`, `bigscience/bloomz`, etc. | | ✅︎ | ✅︎ |
 | `BartForConditionalGeneration` | BART | `facebook/bart-base`, `facebook/bart-large-cnn`, etc. | | | |
diff -- tests/models/registry.py
@@ -180,6 +180,8 @@ def check_available_online(
                                          trust_remote_code=True),
     "BailingMoeForCausalLM": _HfExamplesInfo("inclusionAI/Ling-lite-1.5",
                                          trust_remote_code=True),
+    "BailingMoeV2ForCausalLM": _HfExamplesInfo("inclusionAI/Ling-mini-2.0",
+                                         trust_remote_code=True),
     "BambaForCausalLM": _HfExamplesInfo("ibm-ai-platform/Bamba-9B-v1",
                                         min_transformers_version="4.55.3",
                                         extras={"tiny": "hmellor/tiny-random-BambaForCausalLM"}),  # noqa: E501
```
- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +166/-50; `vllm/model_executor/models/registry.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +2/-0
  - docs/bench: `docs/models/supported_models.md` modified +1/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #25345 - [V0 Deprecation] Remove V0 sampling metadata

- 链接: https://github.com/vllm-project/vllm/pull/25345
- 状态/时间: merged / 2025-09-21
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+123/-417，可读 patch 2063 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V0 Deprecation] Remove V0 sampling metadata」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_opt.py`, `vllm/model_executor/__init__.py`。
- 实现要点:
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py` modified +3/-5；symbols: compute_logits
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_opt.py` modified +3/-5；symbols: compute_logits
  - `vllm/model_executor/__init__.py` modified +0/-2
  - `vllm/model_executor/layers/logits_processor.py` modified +0/-2
- 代码 diff 细节:
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py` modified +3/-5
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_opt.py` modified +3/-5
  - `vllm/model_executor/__init__.py` modified +0/-2
  - `vllm/model_executor/layers/logits_processor.py` modified +0/-2
- 关键代码摘录:

```diff
diff -- tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py
@@ -9,7 +9,6 @@
                                               LlavaForConditionalGeneration,
                                               LlavaMultiModalProcessor,
                                               LlavaProcessingInfo)
-from vllm.model_executor.sampling_metadata import SamplingMetadata
 from vllm.multimodal import MULTIMODAL_REGISTRY


@@ -18,11 +17,10 @@
                                         dummy_inputs=LlavaDummyInputsBuilder)
 class MyLlava(LlavaForConditionalGeneration):

-    def compute_logits(
-            self, hidden_states: torch.Tensor,
diff -- tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_opt.py
@@ -6,16 +6,14 @@
 import torch

 from vllm.model_executor.models.opt import OPTForCausalLM
-from vllm.model_executor.sampling_metadata import SamplingMetadata


 class MyOPTForCausalLM(OPTForCausalLM):

-    def compute_logits(
-            self, hidden_states: torch.Tensor,
-            sampling_metadata: SamplingMetadata) -> Optional[torch.Tensor]:
+    def compute_logits(self,
+                       hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
```
- 已读文件:
  - runtime: `vllm/model_executor/__init__.py` modified +0/-2; `vllm/model_executor/layers/logits_processor.py` modified +0/-2; `vllm/model_executor/models/apertus.py` modified +1/-4; `vllm/model_executor/models/arcee.py` modified +3/-4; `vllm/model_executor/models/arctic.py` modified +1/-4; `vllm/model_executor/models/aria.py` modified +2/-5; `vllm/model_executor/models/aya_vision.py` modified +1/-4; `vllm/model_executor/models/baichuan.py` modified +1/-4
  - tests: `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py` modified +3/-5; `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_opt.py` modified +3/-5
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #25271 - Move`VllmConfig` from `config/__init__.py` to `config/vllm.py`

- 链接: https://github.com/vllm-project/vllm/pull/25271
- 状态/时间: merged / 2025-09-30
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 36 个文件，+964/-905，可读 patch 2200 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Move`VllmConfig` from `config/__init__.py` to `config/vllm.py`」；模型线: Ling 2.5 1T；类别: 模型实现调整；主要 diff: `vllm/attention/layer.py`, `vllm/attention/layers/chunked_local_attention.py`, `vllm/config/__init__.py`。
- 实现要点:
  - `vllm/attention/layer.py` modified +1/-2
  - `vllm/attention/layers/chunked_local_attention.py` modified +2/-1
  - `vllm/config/__init__.py` modified +79/-826
  - `vllm/config/utils.py` modified +36/-6；symbols: SupportsHash, compute_hash, SupportsMetricsInfo, metrics_info
- 代码 diff 细节:
  - `vllm/attention/layer.py` modified +1/-2
  - `vllm/attention/layers/chunked_local_attention.py` modified +2/-1
  - `vllm/config/__init__.py` modified +79/-826
  - `vllm/config/utils.py` modified +36/-6
- 关键代码摘录:

```diff
diff -- vllm/attention/layer.py
@@ -20,8 +20,7 @@
 from vllm.logger import init_logger
 from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
 from vllm.model_executor.layers.linear import UnquantizedLinearMethod
-from vllm.model_executor.layers.quantization.base_config import (
-    QuantizationConfig)
+from vllm.model_executor.layers.quantization import QuantizationConfig
 from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
 from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
 from vllm.model_executor.layers.quantization.utils.quant_utils import (
diff -- vllm/attention/layers/chunked_local_attention.py
@@ -9,7 +9,8 @@
 from vllm.attention.backends.abstract import (AttentionBackend,
                                               AttentionMetadata)
 from vllm.attention.selector import get_attn_backend
-from vllm.config import CacheConfig, QuantizationConfig
+from vllm.config import CacheConfig
+from vllm.model_executor.layers.quantization import QuantizationConfig
 from vllm.v1.attention.backends.utils import (
     CommonAttentionMetadata, make_local_attention_virtual_batches,
     subclass_attention_backend)
```
- 已读文件:
  - runtime: `vllm/attention/layer.py` modified +1/-2; `vllm/attention/layers/chunked_local_attention.py` modified +2/-1; `vllm/config/__init__.py` modified +79/-826; `vllm/config/utils.py` modified +36/-6; `vllm/config/vllm.py` added +789/-0; `vllm/model_executor/layers/mamba/linear_attn.py` modified +1/-2; `vllm/model_executor/layers/quantization/auto_round.py` modified +2/-3; `vllm/model_executor/layers/quantization/bitblas.py` modified +2/-3
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #26247 - Convert formatting to use `ruff` instead of `yapf` + `isort`

- 链接: https://github.com/vllm-project/vllm/pull/26247
- 状态/时间: merged / 2025-10-05
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+4320/-3882，可读 patch 14041 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Convert formatting to use `ruff` instead of `yapf` + `isort`」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `.buildkite/pyproject.toml`, `.pre-commit-config.yaml`, `benchmarks/benchmark_block_pool.py`。
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

### PR #26262 - Fix per file ruff ignores related to line length

- 链接: https://github.com/vllm-project/vllm/pull/26262
- 状态/时间: merged / 2025-10-06
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 65 个文件，+301/-291，可读 patch 1525 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix per file ruff ignores related to line length」；模型线: Ling 2.5 1T；类别: 缺陷修复；主要 diff: `benchmarks/benchmark_ngram_proposer.py`, `benchmarks/benchmark_serving_structured_output.py`, `csrc/cutlass_extensions/vllm_cutlass_library_extension.py`。
- 实现要点:
  - `benchmarks/benchmark_ngram_proposer.py` modified +1/-1
  - `benchmarks/benchmark_serving_structured_output.py` modified +2/-2
  - `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` modified +3/-3
  - `examples/offline_inference/vision_language_pooling.py` modified +1/-1
- 代码 diff 细节:
  - `benchmarks/benchmark_ngram_proposer.py` modified +1/-1
  - `benchmarks/benchmark_serving_structured_output.py` modified +2/-2
  - `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` modified +3/-3
  - `examples/offline_inference/vision_language_pooling.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- benchmarks/benchmark_ngram_proposer.py
@@ -164,7 +164,7 @@ def invoke_main() -> None:
     )
     parser.add_argument(
         "--batched", action="store_true", help="consider time to prepare batch"
-    )  # noqa: E501
+    )
     parser.add_argument(
         "--num-iteration",
         type=int,
diff -- benchmarks/benchmark_serving_structured_output.py
@@ -909,13 +909,13 @@ def create_argument_parser():
     parser.add_argument(
         "--tokenizer",
         type=str,
-        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
+        help="Name or path of the tokenizer, if not using the default tokenizer.",
     )
     parser.add_argument(
         "--tokenizer-mode",
         type=str,
         default="auto",
-        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
+        help="Name or path of the tokenizer, if not using the default tokenizer.",
     )
```
- 已读文件:
  - runtime: `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` modified +3/-3
  - tests: `tests/compile/piecewise/test_simple.py` modified +17/-11; `tests/compile/piecewise/test_toy_llama.py` modified +9/-5; `tests/compile/test_functionalization.py` modified +1/-1; `tests/compile/test_fusion_attn.py` modified +1/-1; `tests/compile/test_sequence_parallelism.py` modified +2/-2; `tests/distributed/test_pipeline_parallel.py` modified +4/-4; `tests/entrypoints/conftest.py` modified +2/-1; `tests/entrypoints/openai/test_audio.py` modified +1/-1
  - docs/bench: `benchmarks/benchmark_ngram_proposer.py` modified +1/-1; `benchmarks/benchmark_serving_structured_output.py` modified +2/-2; `examples/offline_inference/vision_language_pooling.py` modified +1/-1; `examples/online_serving/disaggregated_serving/disagg_proxy_demo.py` modified +2/-2
  - other: `pyproject.toml` modified +0/-46
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #26145 - [Model] Apply shared experts overlap optimization to all models with shared experts

- 链接: https://github.com/vllm-project/vllm/pull/26145
- 状态/时间: merged / 2025-10-09
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+271/-283，可读 patch 1118 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Apply shared experts overlap optimization to all models with shared experts」；模型线: Ling 2.5 1T；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/layers/quantization/fp8.py`。
- 实现要点:
  - `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0
  - `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` renamed +23/-12
  - `vllm/model_executor/layers/quantization/fp8.py` modified +2/-0
  - `vllm/model_executor/layers/shared_fused_moe/__init__.py` removed +0/-5
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0
  - `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` renamed +23/-12
  - `vllm/model_executor/layers/quantization/fp8.py` modified +2/-0
  - `vllm/model_executor/layers/shared_fused_moe/__init__.py` removed +0/-5
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/__init__.py
@@ -15,6 +15,7 @@
     FusedMoEPermuteExpertsUnpermute,
     FusedMoEPrepareAndFinalize,
 )
+from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
 from vllm.model_executor.layers.fused_moe.utils import activation_without_mul
 from vllm.triton_utils import HAS_TRITON

@@ -42,6 +43,7 @@ def get_config() -> Optional[dict[str, Any]]:
     "FusedMoEPermuteExpertsUnpermute",
     "FusedMoEActivationFormat",
     "FusedMoEPrepareAndFinalize",
+    "SharedFusedMoE",
     "activation_without_mul",
diff -- vllm/model_executor/layers/fused_moe/shared_fused_moe.py
@@ -18,13 +18,21 @@ class SharedFusedMoE(FusedMoE):

     def __init__(
         self,
-        shared_experts: torch.nn.Module,
+        shared_experts: Optional[torch.nn.Module],
         use_overlapped: bool = True,
         **kwargs,
     ):
         super().__init__(**kwargs)
         self._shared_experts = shared_experts
-        self.use_overlapped = use_overlapped
+        # Disable shared expert overlap if EP is disabled or we are not using
+        # flashinfer + DP since there is nothing to be gained in this case.
```
- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0; `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` renamed +23/-12; `vllm/model_executor/layers/quantization/fp8.py` modified +2/-0; `vllm/model_executor/layers/shared_fused_moe/__init__.py` removed +0/-5; `vllm/model_executor/models/aria.py` modified +15/-13; `vllm/model_executor/models/bailing_moe.py` modified +26/-21; `vllm/model_executor/models/deepseek_v2.py` modified +24/-45; `vllm/model_executor/models/dots1.py` modified +22/-18
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #26633 - Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y`

- 链接: https://github.com/vllm-project/vllm/pull/26633
- 状态/时间: merged / 2025-10-12
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+481/-581，可读 patch 3511 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y`」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `benchmarks/backend_request_func.py`, `benchmarks/benchmark_prefix_caching.py`, `benchmarks/benchmark_prioritization.py`。
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

### PR #26876 - [Fix] Remove divisibility requirement between num_kv_heads and tp_size in bailing_moe

- 链接: https://github.com/vllm-project/vllm/pull/26876
- 状态/时间: merged / 2025-10-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 15 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Remove divisibility requirement between num_kv_heads and tp_size in bailing_moe」；模型线: Ling 2.5 1T；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/bailing_moe.py`。
- 实现要点:
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-2
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-2
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe.py
@@ -86,13 +86,12 @@ def __init__(
         tp_size = get_tensor_model_parallel_world_size()

         assert self.total_num_heads % tp_size == 0
-        assert self.total_kv_heads % tp_size == 0
         assert self.total_num_heads >= self.total_kv_heads

         self.num_heads = self.total_num_heads // tp_size
         self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
         self.q_size_per_rank = self.head_dim * self.num_heads
-        self.num_kv_heads = self.total_kv_heads // tp_size
+        self.num_kv_heads = max(1, self.total_kv_heads // tp_size)
         self.kv_size_per_rank = self.num_kv_heads * self.head_dim
         self.scale = self.head_dim**-0.5
```
- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +1/-2
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #28382 - [LoRA][1/N]Remove LoRA extra vocab

- 链接: https://github.com/vllm-project/vllm/pull/28382
- 状态/时间: merged / 2025-11-11
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 65 个文件，+197/-754，可读 patch 2645 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[LoRA][1/N]Remove LoRA extra vocab」；模型线: Ling 2.5 1T；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/apertus.py`, `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/arctic.py`。
- 实现要点:
  - `vllm/model_executor/models/apertus.py` modified +5/-25
  - `vllm/model_executor/models/arcee.py` modified +2/-8
  - `vllm/model_executor/models/arctic.py` modified +2/-4
  - `vllm/model_executor/models/aria.py` modified +2/-6
- 代码 diff 细节:
  - `vllm/model_executor/models/apertus.py` modified +5/-25
  - `vllm/model_executor/models/arcee.py` modified +2/-8
  - `vllm/model_executor/models/arctic.py` modified +2/-4
  - `vllm/model_executor/models/aria.py` modified +2/-6
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/apertus.py
@@ -49,7 +49,6 @@
 from vllm.model_executor.layers.quantization import QuantizationConfig
 from vllm.model_executor.layers.rotary_embedding import get_rope
 from vllm.model_executor.layers.vocab_parallel_embedding import (
-    DEFAULT_VOCAB_PADDING_SIZE,
     ParallelLMHead,
     VocabParallelEmbedding,
 )
@@ -346,24 +345,18 @@ def __init__(
         config = vllm_config.model_config.hf_config
         cache_config = vllm_config.cache_config
         quant_config = vllm_config.quant_config
-        lora_config = vllm_config.lora_config

diff -- vllm/model_executor/models/arcee.py
@@ -23,7 +23,6 @@
 from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
 from vllm.model_executor.layers.logits_processor import LogitsProcessor
 from vllm.model_executor.layers.vocab_parallel_embedding import (
-    DEFAULT_VOCAB_PADDING_SIZE,
     ParallelLMHead,
     VocabParallelEmbedding,
 )
@@ -200,7 +199,6 @@ def __init__(
         self.quant_config = quant_config
         self.config = config
         self.vocab_size = config.vocab_size
-        self.org_vocab_size = config.vocab_size

```
- 已读文件:
  - runtime: `vllm/model_executor/models/apertus.py` modified +5/-25; `vllm/model_executor/models/arcee.py` modified +2/-8; `vllm/model_executor/models/arctic.py` modified +2/-4; `vllm/model_executor/models/aria.py` modified +2/-6; `vllm/model_executor/models/baichuan.py` modified +2/-2; `vllm/model_executor/models/bailing_moe.py` modified +0/-2; `vllm/model_executor/models/bamba.py` modified +6/-24; `vllm/model_executor/models/chameleon.py` modified +3/-5
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #27583 - Rename clashing method names for vLLM model protocol

- 链接: https://github.com/vllm-project/vllm/pull/27583
- 状态/时间: merged / 2025-11-13
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+368/-367，可读 patch 2559 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Rename clashing method names for vLLM model protocol」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `docs/contributing/model/basic.md`, `docs/contributing/model/multimodal.md`, `vllm/model_executor/models/apertus.py`。
- 实现要点:
  - `docs/contributing/model/basic.md` modified +2/-2；symbols: embed_input_ids
  - `docs/contributing/model/multimodal.md` modified +3/-3；symbols: embed_multimodal
  - `vllm/model_executor/models/apertus.py` modified +4/-4；symbols: embed_input_ids
  - `vllm/model_executor/models/arcee.py` modified +4/-4；symbols: embed_input_ids
- 代码 diff 细节:
  - `docs/contributing/model/basic.md` modified +2/-2
  - `docs/contributing/model/multimodal.md` modified +3/-3
  - `vllm/model_executor/models/apertus.py` modified +4/-4
  - `vllm/model_executor/models/arcee.py` modified +4/-4
- 关键代码摘录:

```diff
diff -- docs/contributing/model/basic.md
@@ -56,13 +56,13 @@ The initialization code should look like this:

 ### Computation Code

-- Add a `get_input_embeddings` method inside `MyModel` module that returns the text embeddings given `input_ids`. This is equivalent to directly calling the text embedding layer, but provides a unified interface in case `MyModel` is used within a composite multimodal model.
+- Add a `embed_input_ids` method inside `MyModel` module that returns the text embeddings given `input_ids`. This is equivalent to directly calling the text embedding layer, but provides a unified interface in case `MyModel` is used within a composite multimodal model.

 ```python
 class MyModel(nn.Module):
         ...

-    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
+    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
         ...
diff -- docs/contributing/model/multimodal.md
@@ -36,7 +36,7 @@ Further update the model as follows:

   More conveniently, you can simply pass `**kwargs` to the [forward][torch.nn.Module.forward] method and retrieve the keyword parameters for multimodal inputs from it.

-- Implement [get_multimodal_embeddings][vllm.model_executor.models.interfaces.SupportsMultiModal.get_multimodal_embeddings] that returns the embeddings from running the multimodal inputs through the multimodal tokenizer of the model. Below we provide a boilerplate of a typical implementation pattern, but feel free to adjust it to your own needs.
+- Implement [embed_multimodal][vllm.model_executor.models.interfaces.SupportsMultiModal.embed_multimodal] that returns the embeddings from running the multimodal inputs through the multimodal tokenizer of the model. Below we provide a boilerplate of a typical implementation pattern, but feel free to adjust it to your own needs.

     ??? code

@@ -49,7 +49,7 @@ Further update the model as follows:
                 image_features = self.vision_encoder(image_input)
                 return self.multi_modal_projector(image_features)

-            def get_multimodal_embeddings(
```
- 已读文件:
  - runtime: `vllm/model_executor/models/apertus.py` modified +4/-4; `vllm/model_executor/models/arcee.py` modified +4/-4; `vllm/model_executor/models/arctic.py` modified +4/-4; `vllm/model_executor/models/aria.py` modified +3/-3; `vllm/model_executor/models/aya_vision.py` modified +1/-1; `vllm/model_executor/models/baichuan.py` modified +4/-4; `vllm/model_executor/models/bailing_moe.py` modified +4/-4; `vllm/model_executor/models/bamba.py` modified +4/-4
  - docs/bench: `docs/contributing/model/basic.md` modified +2/-2; `docs/contributing/model/multimodal.md` modified +3/-3
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #28277 - [Model] Fix bailing_moe accuracy problem

- 链接: https://github.com/vllm-project/vllm/pull/28277
- 状态/时间: merged / 2025-11-14
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-2，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Fix bailing_moe accuracy problem」；模型线: Ling 2.5 1T；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/bailing_moe.py`。
- 实现要点:
  - `vllm/model_executor/models/bailing_moe.py` modified +3/-2
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe.py` modified +3/-2
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe.py
@@ -39,7 +39,6 @@
     get_pp_group,
     get_tensor_model_parallel_rank,
     get_tensor_model_parallel_world_size,
-    tensor_model_parallel_all_reduce,
 )
 from vllm.model_executor.layers.activation import SiluAndMul
 from vllm.model_executor.layers.fused_moe import SharedFusedMoE
@@ -330,7 +329,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
             final_hidden_states = final_hidden_states + shared_output

         if self.tp_size > 1:
-            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
+            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
```
- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +3/-2
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #28777 - [Model] Fix lmhead init bug of bailing_moe

- 链接: https://github.com/vllm-project/vllm/pull/28777
- 状态/时间: merged / 2025-11-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Fix lmhead init bug of bailing_moe」；模型线: Ling 2.5 1T；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/bailing_moe.py`。
- 实现要点:
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-1
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe.py
@@ -599,7 +599,7 @@ def __init__(
                     config.vocab_size,
                     config.hidden_size,
                     quant_config=quant_config,
-                    prefix=f"{prefix}.lm_head",
+                    prefix=maybe_prefix(prefix, "lm_head"),
                 )
             self.logits_processor = LogitsProcessor(config.vocab_size)
         else:
```
- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +1/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #28542 - Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5

- 链接: https://github.com/vllm-project/vllm/pull/28542
- 状态/时间: merged / 2025-11-19
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+487/-868，可读 patch 4411 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `.buildkite/test-pipeline.yaml`, `benchmarks/kernels/benchmark_mrope.py`, `examples/offline_inference/context_extension.py`。
- 实现要点:
  - `.buildkite/test-pipeline.yaml` modified +3/-3
  - `benchmarks/kernels/benchmark_mrope.py` modified +7/-12
  - `examples/offline_inference/context_extension.py` modified +3/-3
  - `tests/compile/test_functionalization.py` modified +2/-2
- 代码 diff 细节:
  - `.buildkite/test-pipeline.yaml` modified +3/-3
  - `benchmarks/kernels/benchmark_mrope.py` modified +7/-12
  - `examples/offline_inference/context_extension.py` modified +3/-3
  - `tests/compile/test_functionalization.py` modified +2/-2
- 关键代码摘录:

```diff
diff -- .buildkite/test-pipeline.yaml
@@ -876,12 +876,12 @@ steps:
   optional: true
   commands:
     - pip install --upgrade git+https://github.com/huggingface/transformers
-    - pytest -v -s tests/models/test_initialization.py -k 'not (Gemma3 or ModernBert or Qwen2_5_VL or Qwen2_5vl or Qwen2VL or TransformersMultiModalEmbeddingModel or TransformersMultiModalForSequenceClassification or Ultravox or Phi4Multimodal or LlavaNextVideo or MiniCPMO or Lfm2Moe or PaliGemma or RobertaForSequenceClassification or Ovis2_5 or Fuyu or DeepseekOCR or KimiVL)'
+    - pytest -v -s tests/models/test_initialization.py -k 'not (Ultravox or Phi4Multimodal or MiniCPMO or Lfm2Moe or RobertaForSequenceClassification or Ovis2_5 or DeepseekOCR or KimiVL)'
     - pytest -v -s tests/models/test_transformers.py
     # - pytest -v -s tests/models/multimodal/processing/
-    - pytest -v -s tests/models/multimodal/test_mapping.py -k 'not (Gemma3 or Qwen2VL or Qwen2_5_VL)'
+    - pytest -v -s tests/models/multimodal/test_mapping.py
     - python3 examples/offline_inference/basic/chat.py
-    # - python3 examples/offline_inference/vision_language.py --model-type qwen2_5_vl
+    - python3 examples/offline_inference/vision_language.py --model-type qwen2_5_vl
     # Whisper needs spawn method to avoid deadlock
diff -- benchmarks/kernels/benchmark_mrope.py
@@ -6,7 +6,7 @@
 #
 # The CSV file (named with current date/time) contains these columns:
 # model_name, tp_size, num_tokens, num_heads, num_kv_heads, head_dim, max_position,
-# rope_theta, is_neox_style, rope_scaling, dtype, torch_mean, torch_median, torch_p99,
+# is_neox_style, rope_parameters, dtype, torch_mean, torch_median, torch_p99,
 # torch_min, torch_max, triton_mean, triton_median, triton_p99, triton_min, triton_max,
 # speedup
 #
@@ -86,9 +86,8 @@ def benchmark_mrope(
     num_heads: int,
     num_kv_heads: int,
     max_position: int = 8192,
-    rope_theta: float = 10000,
```
- 已读文件:
  - runtime: `vllm/config/model.py` modified +29/-34; `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38; `vllm/model_executor/models/afmoe.py` modified +1/-16; `vllm/model_executor/models/apertus.py` modified +2/-20; `vllm/model_executor/models/arcee.py` modified +0/-11; `vllm/model_executor/models/arctic.py` modified +1/-2; `vllm/model_executor/models/baichuan.py` modified +3/-5; `vllm/model_executor/models/bailing_moe.py` modified +1/-2
  - tests: `tests/compile/test_functionalization.py` modified +2/-2; `tests/kernels/core/test_mrope.py` modified +5/-11; `tests/kernels/core/test_pos_encoding.py` modified +20/-19; `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +1/-1; `tests/models/language/pooling/test_nomic_max_model_len.py` modified +9/-7; `tests/test_config.py` modified +20/-17
  - docs/bench: `.buildkite/test-pipeline.yaml` modified +3/-3; `benchmarks/kernels/benchmark_mrope.py` modified +7/-12; `examples/offline_inference/context_extension.py` modified +3/-3
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #29342 - [Attention] Remove imports from `vllm/attention/__init__.py`

- 链接: https://github.com/vllm-project/vllm/pull/29342
- 状态/时间: merged / 2025-11-26
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 96 个文件，+120/-121，可读 patch 923 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Attention] Remove imports from `vllm/attention/__init__.py`」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `docs/contributing/model/basic.md`, `tests/compile/test_fusion_attn.py`, `tests/compile/test_qk_norm_rope_fusion.py`。
- 实现要点:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/compile/test_fusion_attn.py` modified +2/-1
  - `tests/compile/test_qk_norm_rope_fusion.py` modified +2/-1
  - `tests/kernels/utils.py` modified +1/-1
- 代码 diff 细节:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/compile/test_fusion_attn.py` modified +2/-1
  - `tests/compile/test_qk_norm_rope_fusion.py` modified +2/-1
  - `tests/kernels/utils.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- docs/contributing/model/basic.md
@@ -29,7 +29,7 @@ The initialization code should look like this:
     ```python
     from torch import nn
     from vllm.config import VllmConfig
-    from vllm.attention import Attention
+    from vllm.attention.layer import Attention

     class MyAttention(nn.Module):
         def __init__(self, vllm_config: VllmConfig, prefix: str):
diff -- tests/compile/test_fusion_attn.py
@@ -9,8 +9,9 @@
 from tests.utils import flat_product
 from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
 from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
-from vllm.attention import Attention, AttentionMetadata
+from vllm.attention.backends.abstract import AttentionMetadata
 from vllm.attention.backends.registry import AttentionBackendEnum
+from vllm.attention.layer import Attention
 from vllm.attention.selector import global_force_attn_backend_context_manager
 from vllm.compilation.fusion_attn import ATTN_OP, AttnFusionPass
 from vllm.compilation.fx_utils import find_op_nodes
```
- 已读文件:
  - runtime: `vllm/attention/__init__.py` modified +0/-19; `vllm/attention/backends/abstract.py` modified +1/-1; `vllm/attention/layer.py` modified +5/-2; `vllm/compilation/fusion_attn.py` modified +1/-1; `vllm/compilation/qk_norm_rope_fusion.py` modified +1/-1; `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` modified +1/-1; `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` modified +2/-1; `vllm/model_executor/layers/mamba/linear_attn.py` modified +1/-1
  - tests: `tests/compile/test_fusion_attn.py` modified +2/-1; `tests/compile/test_qk_norm_rope_fusion.py` modified +2/-1; `tests/kernels/utils.py` modified +1/-1; `tests/v1/worker/test_gpu_model_runner.py` modified +1/-1; `tests/v1/worker/test_utils.py` modified +2/-2
  - docs/bench: `docs/contributing/model/basic.md` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #29966 - Access `partial_rotary_factor` from `rope_parameters`

- 链接: https://github.com/vllm-project/vllm/pull/29966
- 状态/时间: merged / 2025-12-04
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 21 个文件，+43/-62，可读 patch 396 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Access `partial_rotary_factor` from `rope_parameters`」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `tests/kernels/core/test_mrope.py`, `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/model_executor/models/apertus.py`。
- 实现要点:
  - `tests/kernels/core/test_mrope.py` modified +2/-6
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +4/-1
  - `vllm/model_executor/models/apertus.py` modified +1/-4
  - `vllm/model_executor/models/bailing_moe.py` modified +0/-3
- 代码 diff 细节:
  - `tests/kernels/core/test_mrope.py` modified +2/-6
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +4/-1
  - `vllm/model_executor/models/apertus.py` modified +1/-4
  - `vllm/model_executor/models/bailing_moe.py` modified +0/-3
- 关键代码摘录:

```diff
diff -- tests/kernels/core/test_mrope.py
@@ -113,12 +113,10 @@ def test_mrope(
     is_neox_style = True

     max_position = config.max_position_embeddings
-    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
-    rotary_dim = int(head_dim * partial_rotary_factor)

     mrope_helper_class = get_rope(
         head_size=head_dim,
-        rotary_dim=rotary_dim,
+        rotary_dim=head_dim,
         max_position=max_position,
         is_neox_style=is_neox_style,
         rope_parameters=config.rope_parameters,
diff -- vllm/model_executor/layers/rotary_embedding/__init__.py
@@ -30,7 +30,6 @@ def get_rope(
     is_neox_style: bool = True,
     rope_parameters: dict[str, Any] | None = None,
     dtype: torch.dtype | None = None,
-    partial_rotary_factor: float = 1.0,
     dual_chunk_attention_config: dict[str, Any] | None = None,
 ) -> RotaryEmbedding:
     if dtype is None:
@@ -55,6 +54,10 @@ def get_rope(
     else:
         dual_chunk_attention_args = None

+    partial_rotary_factor = 1.0
+    if rope_parameters is not None:
```
- 已读文件:
  - runtime: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +4/-1; `vllm/model_executor/models/apertus.py` modified +1/-4; `vllm/model_executor/models/bailing_moe.py` modified +0/-3; `vllm/model_executor/models/bamba.py` modified +1/-3; `vllm/model_executor/models/config.py` modified +0/-5; `vllm/model_executor/models/falcon_h1.py` modified +1/-3; `vllm/model_executor/models/glm.py` modified +2/-1; `vllm/model_executor/models/glm4.py` modified +1/-2
  - tests: `tests/kernels/core/test_mrope.py` modified +2/-6
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #30389 - Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim`

- 链接: https://github.com/vllm-project/vllm/pull/30389
- 状态/时间: merged / 2025-12-11
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 83 个文件，+238/-292，可读 patch 1379 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim`」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `benchmarks/kernels/benchmark_mrope.py`, `benchmarks/kernels/benchmark_rope.py`, `tests/compile/test_functionalization.py`。
- 实现要点:
  - `benchmarks/kernels/benchmark_mrope.py` modified +0/-1
  - `benchmarks/kernels/benchmark_rope.py` modified +2/-2
  - `tests/compile/test_functionalization.py` modified +1/-4；symbols: __init__
  - `tests/kernels/core/test_mrope.py` modified +0/-2
- 代码 diff 细节:
  - `benchmarks/kernels/benchmark_mrope.py` modified +0/-1
  - `benchmarks/kernels/benchmark_rope.py` modified +2/-2
  - `tests/compile/test_functionalization.py` modified +1/-4
  - `tests/kernels/core/test_mrope.py` modified +0/-2
- 关键代码摘录:

```diff
diff -- benchmarks/kernels/benchmark_mrope.py
@@ -99,7 +99,6 @@ def benchmark_mrope(
     # the parameters to compute the q k v size based on tp_size
     mrope_helper_class = get_rope(
         head_size=head_dim,
-        rotary_dim=head_dim,
         max_position=max_position,
         is_neox_style=is_neox_style,
         rope_parameters=rope_parameters,
diff -- benchmarks/kernels/benchmark_rope.py
@@ -32,8 +32,8 @@ def get_benchmark(head_size, rotary_dim, is_neox_style, device):
     def benchmark(batch_size, seq_len, num_heads, provider):
         dtype = torch.bfloat16
         max_position = 8192
-        base = 10000
-        rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style)
+        rope_parameters = {"partial_rotary_factor": rotary_dim / head_size}
+        rope = get_rope(head_size, max_position, is_neox_style, rope_parameters)
         rope = rope.to(dtype=dtype, device=device)
         cos_sin_cache = rope.cos_sin_cache.to(dtype=torch.float, device=device)

```
- 已读文件:
  - runtime: `vllm/config/utils.py` modified +16/-2; `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +160/-166; `vllm/model_executor/models/afmoe.py` modified +0/-1; `vllm/model_executor/models/apertus.py` modified +0/-1; `vllm/model_executor/models/arctic.py` modified +0/-1; `vllm/model_executor/models/baichuan.py` modified +0/-1; `vllm/model_executor/models/bailing_moe.py` modified +2/-2; `vllm/model_executor/models/bamba.py` modified +2/-5
  - tests: `tests/compile/test_functionalization.py` modified +1/-4; `tests/kernels/core/test_mrope.py` modified +0/-2; `tests/kernels/core/test_pos_encoding.py` modified +8/-4
  - docs/bench: `benchmarks/kernels/benchmark_mrope.py` modified +0/-1; `benchmarks/kernels/benchmark_rope.py` modified +2/-2
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #31104 - [BugFix] LoRA: Support loading base_layer of experts

- 链接: https://github.com/vllm-project/vllm/pull/31104
- 状态/时间: merged / 2026-01-07
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+46/-3，可读 patch 319 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] LoRA: Support loading base_layer of experts」；模型线: Ling 2.5 1T；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/bailing_moe.py`。
- 实现要点:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3
  - `vllm/model_executor/models/afmoe.py` modified +1/-0
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-0
  - `vllm/model_executor/models/deepseek_eagle.py` modified +1/-0
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3
  - `vllm/model_executor/models/afmoe.py` modified +1/-0
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-0
  - `vllm/model_executor/models/deepseek_eagle.py` modified +1/-0
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -2007,6 +2007,7 @@ def combine_output(states: torch.Tensor) -> torch.Tensor:
     @classmethod
     def make_expert_params_mapping(
         cls,
+        model: torch.nn.Module,
         ckpt_gate_proj_name: str,
         ckpt_down_proj_name: str,
         ckpt_up_proj_name: str,
@@ -2025,13 +2026,19 @@ def make_expert_params_mapping(
             )
         )

+        base_layer = (
+            "base_layer."
diff -- vllm/model_executor/models/afmoe.py
@@ -475,6 +475,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
         # Params for weights, fp8 weight scales, fp8 activation scales
         # (param_name, weight_name, expert_id, shard_id)
         return SharedFusedMoE.make_expert_params_mapping(
+            self,
             ckpt_gate_proj_name="gate_proj",
             ckpt_down_proj_name="down_proj",
             ckpt_up_proj_name="up_proj",
```
- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3; `vllm/model_executor/models/afmoe.py` modified +1/-0; `vllm/model_executor/models/bailing_moe.py` modified +1/-0; `vllm/model_executor/models/deepseek_eagle.py` modified +1/-0; `vllm/model_executor/models/deepseek_mtp.py` modified +1/-0; `vllm/model_executor/models/deepseek_v2.py` modified +2/-0; `vllm/model_executor/models/dots1.py` modified +1/-0; `vllm/model_executor/models/ernie45_moe.py` modified +1/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #33063 - [Chore] Update type annotation of `input_ids` in model forward

- 链接: https://github.com/vllm-project/vllm/pull/33063
- 状态/时间: merged / 2026-01-26
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+146/-143，可读 patch 1304 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Chore] Update type annotation of `input_ids` in model forward」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `docs/contributing/model/basic.md`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`, `vllm/model_executor/models/afmoe.py`。
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

### PR #32064 - [5/N][Attention] Finish eliminating `vllm/attention` folder

- 链接: https://github.com/vllm-project/vllm/pull/32064
- 状态/时间: merged / 2026-01-27
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+524/-468，可读 patch 2150 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[5/N][Attention] Finish eliminating `vllm/attention` folder」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `.buildkite/test-amd.yaml`, `.buildkite/test-pipeline.yaml`, `.buildkite/test_areas/kernels.yaml`。
- 实现要点:
  - `.buildkite/test-amd.yaml` modified +2/-1
  - `.buildkite/test-pipeline.yaml` modified +2/-1
  - `.buildkite/test_areas/kernels.yaml` modified +2/-1
  - `.github/CODEOWNERS` modified +1/-1
- 代码 diff 细节:
  - `.buildkite/test-amd.yaml` modified +2/-1
  - `.buildkite/test-pipeline.yaml` modified +2/-1
  - `.buildkite/test_areas/kernels.yaml` modified +2/-1
  - `.github/CODEOWNERS` modified +1/-1
- 关键代码摘录:

```diff
diff -- .buildkite/test-amd.yaml
@@ -640,8 +640,9 @@ steps:
   # grade: Blocking
   source_file_dependencies:
   - csrc/attention/
-  - vllm/attention
   - vllm/v1/attention
+    # TODO: remove this dependency (https://github.com/vllm-project/vllm/issues/32267)
+  - vllm/model_executor/layers/attention
   - tests/kernels/attention
   commands:
     - pytest -v -s kernels/attention --shard-id=$$BUILDKITE_PARALLEL_JOB --num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT
diff -- .buildkite/test-pipeline.yaml
@@ -568,8 +568,9 @@ steps:
   mirror_hardwares: [amdexperimental]
   source_file_dependencies:
   - csrc/attention/
-  - vllm/attention
   - vllm/v1/attention
+    # TODO: remove this dependency (https://github.com/vllm-project/vllm/issues/32267)
+  - vllm/model_executor/layers/attention
   - tests/kernels/attention
   commands:
     - pytest -v -s kernels/attention --shard-id=$$BUILDKITE_PARALLEL_JOB --num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT
```
- 已读文件:
  - runtime: `vllm/attention/__init__.py` removed +0/-0; `vllm/attention/utils/__init__.py` removed +0/-0; `vllm/attention/utils/kv_sharing_utils.py` removed +0/-33; `vllm/compilation/fusion_attn.py` modified +1/-1; `vllm/compilation/qk_norm_rope_fusion.py` modified +1/-1; `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` modified +1/-1; `vllm/model_executor/layers/attention/__init__.py` modified +26/-0; `vllm/model_executor/layers/attention/attention.py` renamed +42/-315
  - tests: `tests/compile/test_fusion_attn.py` modified +1/-1; `tests/compile/test_qk_norm_rope_fusion.py` modified +1/-1; `tests/kernels/attention/test_attention.py` modified +1/-2; `tests/kernels/attention/test_mha_attn.py` modified +1/-1; `tests/v1/worker/test_gpu_model_runner.py` modified +1/-1; `tests/v1/worker/test_utils.py` modified +3/-3
  - docs/bench: `.buildkite/test-amd.yaml` modified +2/-1; `.buildkite/test-pipeline.yaml` modified +2/-1; `.buildkite/test_areas/kernels.yaml` modified +2/-1; `docs/contributing/model/basic.md` modified +1/-1; `docs/design/custom_op.md` modified +1/-1
  - other: `.github/CODEOWNERS` modified +1/-1; `tools/pre_commit/mypy.py` modified +0/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #33737 - [Bugfix] Define router_logits_dtype for remaining MoE models

- 链接: https://github.com/vllm-project/vllm/pull/33737
- 状态/时间: merged / 2026-02-04
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+9/-4，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Define router_logits_dtype for remaining MoE models」；模型线: Ling 2.5 1T；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/flex_olmo.py`。
- 实现要点:
  - `vllm/model_executor/models/afmoe.py` modified +1/-0
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-0
  - `vllm/model_executor/models/flex_olmo.py` modified +1/-1
  - `vllm/model_executor/models/longcat_flash.py` modified +4/-3
- 代码 diff 细节:
  - `vllm/model_executor/models/afmoe.py` modified +1/-0
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-0
  - `vllm/model_executor/models/flex_olmo.py` modified +1/-1
  - `vllm/model_executor/models/longcat_flash.py` modified +4/-3
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/afmoe.py
@@ -142,6 +142,7 @@ def __init__(
             e_score_correction_bias=self.expert_bias,
             enable_eplb=self.enable_eplb,
             num_redundant_experts=self.n_redundant_experts,
+            router_logits_dtype=torch.float32,
         )

     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
diff -- vllm/model_executor/models/bailing_moe.py
@@ -300,6 +300,7 @@ def __init__(
             num_expert_group=self.n_group,
             topk_group=self.topk_group,
             use_grouped_topk=self.use_grouped_topk,
+            router_logits_dtype=self.router_dtype,
         )

     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
```
- 已读文件:
  - runtime: `vllm/model_executor/models/afmoe.py` modified +1/-0; `vllm/model_executor/models/bailing_moe.py` modified +1/-0; `vllm/model_executor/models/flex_olmo.py` modified +1/-1; `vllm/model_executor/models/longcat_flash.py` modified +4/-3; `vllm/model_executor/models/mimo_v2_flash.py` modified +1/-0; `vllm/model_executor/models/step3p5.py` modified +1/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #35102 - [Model] Ring 2.5

- 链接: https://github.com/vllm-project/vllm/pull/35102
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+1407/-70，可读 patch 1650 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Ring 2.5」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `docs/models/supported_models.md`, `tests/models/registry.py`, `vllm/model_executor/layers/fla/ops/layernorm_guard.py`。
- 实现要点:
  - `docs/models/supported_models.md` modified +1/-0
  - `tests/models/registry.py` modified +3/-0
  - `vllm/model_executor/layers/fla/ops/layernorm_guard.py` modified +30/-5
  - `vllm/model_executor/layers/layernorm.py` modified +1/-0
- 代码 diff 细节:
  - `docs/models/supported_models.md` modified +1/-0
  - `tests/models/registry.py` modified +3/-0
  - `vllm/model_executor/layers/fla/ops/layernorm_guard.py` modified +30/-5
  - `vllm/model_executor/layers/layernorm.py` modified +1/-0
- 关键代码摘录:

```diff
diff -- docs/models/supported_models.md
@@ -372,6 +372,7 @@ th {
 | `BaiChuanForCausalLM` | Baichuan2, Baichuan | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc. | ✅︎ | ✅︎ |
 | `BailingMoeForCausalLM` | Ling | `inclusionAI/Ling-lite-1.5`, `inclusionAI/Ling-plus`, etc. | ✅︎ | ✅︎ |
 | `BailingMoeV2ForCausalLM` | Ling | `inclusionAI/Ling-mini-2.0`, etc. | ✅︎ | ✅︎ |
+| `BailingMoeV2_5ForCausalLM` | Ling | `inclusionAI/Ling-2.5-1T`, `inclusionAI/Ring-2.5-1T` | | ✅︎ |
 | `BambaForCausalLM` | Bamba | `ibm-ai-platform/Bamba-9B-fp8`, `ibm-ai-platform/Bamba-9B` | ✅︎ | ✅︎ |
 | `BloomForCausalLM` | BLOOM, BLOOMZ, BLOOMChat | `bigscience/bloom`, `bigscience/bloomz`, etc. | | ✅︎ |
 | `ChatGLMModel`, `ChatGLMForConditionalGeneration` | ChatGLM | `zai-org/chatglm2-6b`, `zai-org/chatglm3-6b`, `thu-coai/ShieldLM-6B-chatglm3`, etc. | ✅︎ | ✅︎ |
diff -- tests/models/registry.py
@@ -206,6 +206,9 @@ def check_available_online(
     "BailingMoeV2ForCausalLM": _HfExamplesInfo(
         "inclusionAI/Ling-mini-2.0", trust_remote_code=True
     ),
+    "BailingMoeV2_5ForCausalLM": _HfExamplesInfo(
+        "inclusionAI/Ring-2.5-1T", trust_remote_code=True
+    ),
     "BambaForCausalLM": _HfExamplesInfo(
         "ibm-ai-platform/Bamba-9B-v1",
         extras={"tiny": "hmellor/tiny-random-BambaForCausalLM"},
```
- 已读文件:
  - runtime: `vllm/model_executor/layers/fla/ops/layernorm_guard.py` modified +30/-5; `vllm/model_executor/layers/layernorm.py` modified +1/-0; `vllm/model_executor/layers/mamba/linear_attn.py` modified +124/-65; `vllm/model_executor/models/bailing_moe_linear.py` added +1246/-0; `vllm/model_executor/models/registry.py` modified +1/-0; `vllm/transformers_utils/model_arch_config_convertor.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +3/-0
  - docs/bench: `docs/models/supported_models.md` modified +1/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #37195 - [V0 Deprecation] Deprecate virtual engine

- 链接: https://github.com/vllm-project/vllm/pull/37195
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 23 个文件，+23/-45，可读 patch 353 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V0 Deprecation] Deprecate virtual engine」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `tests/compile/passes/test_rope_kvcache_fusion.py`, `tests/v1/kv_connector/unit/test_decode_bench_connector.py`, `tests/v1/kv_connector/unit/test_lmcache_integration.py`。
- 实现要点:
  - `tests/compile/passes/test_rope_kvcache_fusion.py` modified +2/-2
  - `tests/v1/kv_connector/unit/test_decode_bench_connector.py` modified +1/-1
  - `tests/v1/kv_connector/unit/test_lmcache_integration.py` modified +0/-1
  - `tests/v1/kv_connector/unit/test_nixl_connector.py` modified +0/-8
- 代码 diff 细节:
  - `tests/compile/passes/test_rope_kvcache_fusion.py` modified +2/-2
  - `tests/v1/kv_connector/unit/test_decode_bench_connector.py` modified +1/-1
  - `tests/v1/kv_connector/unit/test_lmcache_integration.py` modified +0/-1
  - `tests/v1/kv_connector/unit/test_nixl_connector.py` modified +0/-8
- 关键代码摘录:

```diff
diff -- tests/compile/passes/test_rope_kvcache_fusion.py
@@ -295,7 +295,7 @@ def test_rope_kvcache_fusion(
             }
             q_unfused, k_unfused, v_unfused, dummy = model(qkv_unfused, pos_unfused)
             attn_layer = forward_context.no_compile_layers[model.layer_name]
-            kv_cache_unfused = attn_layer.kv_cache[forward_context.virtual_engine]
+            kv_cache_unfused = attn_layer.kv_cache[0]
         del dummy

         torch._dynamo.mark_dynamic(qkv, 0)
@@ -309,7 +309,7 @@ def test_rope_kvcache_fusion(
             }
             q_fused, k_fused, v_fused, dummy = model_fused(qkv, pos)
             attn_layer = forward_context.no_compile_layers[model.layer_name]
-            kv_cache_fused = attn_layer.kv_cache[forward_context.virtual_engine]
diff -- tests/v1/kv_connector/unit/test_decode_bench_connector.py
@@ -86,7 +86,7 @@ def __init__(self, block_size: int, num_gpu_blocks: int):
         self._block_hasher = get_request_block_hasher(block_size, sha256)

         self._dummy_ctx: ForwardContext = ForwardContext(
-            no_compile_layers={}, attn_metadata={}, virtual_engine=0, slot_mapping={}
+            no_compile_layers={}, attn_metadata={}, slot_mapping={}
         )

     def new_request(self, token_ids: list[int]) -> Request:
```
- 已读文件:
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py` modified +1/-1; `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py` modified +1/-3; `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` modified +1/-1; `vllm/forward_context.py` modified +0/-7; `vllm/model_executor/layers/attention/attention.py` modified +2/-2; `vllm/model_executor/layers/attention/mla_attention.py` modified +2/-2; `vllm/model_executor/layers/attention/static_sink_attention.py` modified +1/-2; `vllm/model_executor/layers/kda.py` modified +1/-1
  - tests: `tests/compile/passes/test_rope_kvcache_fusion.py` modified +2/-2; `tests/v1/kv_connector/unit/test_decode_bench_connector.py` modified +1/-1; `tests/v1/kv_connector/unit/test_lmcache_integration.py` modified +0/-1; `tests/v1/kv_connector/unit/test_nixl_connector.py` modified +0/-8; `tests/v1/kv_connector/unit/test_offloading_connector.py` modified +0/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #37487 - [V0 Deprecation] Refactor kv cache from list to element

- 链接: https://github.com/vllm-project/vllm/pull/37487
- 状态/时间: merged / 2026-03-24
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+70/-85，可读 patch 478 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V0 Deprecation] Refactor kv cache from list to element」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `tests/compile/passes/test_fusion_attn.py`, `tests/compile/passes/test_rope_kvcache_fusion.py`, `tests/v1/e2e/general/test_mamba_prefix_cache.py`。
- 实现要点:
  - `tests/compile/passes/test_fusion_attn.py` modified +1/-1
  - `tests/compile/passes/test_rope_kvcache_fusion.py` modified +3/-3
  - `tests/v1/e2e/general/test_mamba_prefix_cache.py` modified +3/-3
  - `tests/v1/worker/test_gpu_model_runner.py` modified +17/-17
- 代码 diff 细节:
  - `tests/compile/passes/test_fusion_attn.py` modified +1/-1
  - `tests/compile/passes/test_rope_kvcache_fusion.py` modified +3/-3
  - `tests/v1/e2e/general/test_mamba_prefix_cache.py` modified +3/-3
  - `tests/v1/worker/test_gpu_model_runner.py` modified +17/-17
- 关键代码摘录:

```diff
diff -- tests/compile/passes/test_fusion_attn.py
@@ -127,7 +127,7 @@ def build_attn_metadata(self, batch_size: int) -> AttentionMetadata:
         raw_tensor = raw_tensor.view(kv_cache_shape)
         kv_cache = raw_tensor.permute(*inv_order)

-        self.attn.kv_cache = [kv_cache]
+        self.attn.kv_cache = kv_cache

         # Build attn metadata
         self.attn_metadata = self.builder.build(
diff -- tests/compile/passes/test_rope_kvcache_fusion.py
@@ -148,7 +148,7 @@ def build_attn_metadata(self, batch_size: int) -> CommonAttentionMetadata:
         raw_tensor = raw_tensor.view(kv_cache_shape)
         kv_cache = raw_tensor.permute(*inv_order)

-        self.attn.kv_cache = [kv_cache]
+        self.attn.kv_cache = kv_cache

         # Build attn metadata
         attn_metadata = self.builder.build(
@@ -295,7 +295,7 @@ def test_rope_kvcache_fusion(
             }
             q_unfused, k_unfused, v_unfused, dummy = model(qkv_unfused, pos_unfused)
             attn_layer = forward_context.no_compile_layers[model.layer_name]
-            kv_cache_unfused = attn_layer.kv_cache[0]
```
- 已读文件:
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py` modified +2/-4; `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py` modified +1/-1; `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` modified +1/-1; `vllm/model_executor/layers/attention/attention.py` modified +2/-5; `vllm/model_executor/layers/attention/mla_attention.py` modified +3/-8; `vllm/model_executor/layers/attention/static_sink_attention.py` modified +1/-1; `vllm/model_executor/layers/kda.py` modified +1/-1; `vllm/model_executor/layers/mamba/linear_attn.py` modified +1/-1
  - tests: `tests/compile/passes/test_fusion_attn.py` modified +1/-1; `tests/compile/passes/test_rope_kvcache_fusion.py` modified +3/-3; `tests/v1/e2e/general/test_mamba_prefix_cache.py` modified +3/-3; `tests/v1/worker/test_gpu_model_runner.py` modified +17/-17; `tests/v1/worker/test_utils.py` modified +10/-10
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #35949 - [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase

- 链接: https://github.com/vllm-project/vllm/pull/35949
- 状态/时间: merged / 2026-04-20
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 53 个文件，+325/-702，可读 patch 2430 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase」；模型线: Ling 2.5 1T；类别: 性能/后端优化；主要 diff: `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py`, `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`。
- 实现要点:
  - `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py` modified +3/-5
  - `tests/kernels/moe/test_moe_layer.py` modified +17/-76
  - `tests/kernels/moe/test_shared_fused_moe_routed_transform.py` modified +10/-26
  - `vllm/lora/layers/fused_moe.py` modified +0/-3
- 代码 diff 细节:
  - `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py` modified +3/-5
  - `tests/kernels/moe/test_moe_layer.py` modified +17/-76
  - `tests/kernels/moe/test_shared_fused_moe_routed_transform.py` modified +10/-26
  - `vllm/lora/layers/fused_moe.py` modified +0/-3
- 关键代码摘录:

```diff
diff -- tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py
@@ -6,14 +6,13 @@

 import vllm.config
 from tests.compile.backend import TestBackend
-from vllm.platforms import current_platform
 from vllm.compilation.passes.vllm_inductor_pass import (
     VllmFusionPatternMatcherPass,
     VllmPatternMatcherPass,
     VllmPatternReplacement,
 )
 from vllm.config import CompilationConfig, CompilationMode, VllmConfig
-
+from vllm.platforms import current_platform

diff -- tests/kernels/moe/test_moe_layer.py
@@ -236,7 +236,6 @@ class MoETestConfig:
     use_gate: bool
     use_routed_input_transform: bool
     enable_eplb: bool = False
-    reduce_results: bool = False
     backend: str | None = None
     ep_size: int = 1
     dp_size: int = 1
@@ -295,7 +294,6 @@ def generate_valid_test_configs(
         use_shared_experts,
         use_gate,
         use_routed_input_transform,
-        reduce_results,
     ) in product(
```
- 已读文件:
  - runtime: `vllm/lora/layers/fused_moe.py` modified +0/-3; `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` modified +1/-1; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32; `vllm/model_executor/layers/fused_moe/modular_kernel.py` modified +1/-1; `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` modified +0/-4; `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +1/-12; `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86; `vllm/model_executor/layers/fused_moe/runner/moe_runner_factory.py` modified +4/-2
  - tests: `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py` modified +3/-5; `tests/kernels/moe/test_moe_layer.py` modified +17/-76; `tests/kernels/moe/test_shared_fused_moe_routed_transform.py` modified +10/-26
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #35782 - [MoE Refactor] Remove SharedFusedMoE class

- 链接: https://github.com/vllm-project/vllm/pull/35782
- 状态/时间: merged / 2026-04-21
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 33 个文件，+112/-141，可读 patch 926 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Remove SharedFusedMoE class」；模型线: Ling 2.5 1T；类别: 性能/后端优化；主要 diff: `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`, `vllm/distributed/device_communicators/base_device_communicator.py`。
- 实现要点:
  - `tests/kernels/moe/test_moe_layer.py` modified +3/-7
  - `tests/kernels/moe/test_shared_fused_moe_routed_transform.py` modified +10/-10
  - `vllm/distributed/device_communicators/base_device_communicator.py` modified +3/-10
  - `vllm/distributed/elastic_ep/elastic_execute.py` modified +2/-4
- 代码 diff 细节:
  - `tests/kernels/moe/test_moe_layer.py` modified +3/-7
  - `tests/kernels/moe/test_shared_fused_moe_routed_transform.py` modified +10/-10
  - `vllm/distributed/device_communicators/base_device_communicator.py` modified +3/-10
  - `vllm/distributed/elastic_ep/elastic_execute.py` modified +2/-4
- 关键代码摘录:

```diff
diff -- tests/kernels/moe/test_moe_layer.py
@@ -37,7 +37,7 @@
     get_eplb_group,
 )
 from vllm.forward_context import set_forward_context
-from vllm.model_executor.layers.fused_moe import FusedMoE, SharedFusedMoE, fused_experts
+from vllm.model_executor.layers.fused_moe import FusedMoE, fused_experts
 from vllm.model_executor.layers.fused_moe.activation import MoEActivation
 from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
 from vllm.model_executor.layers.fused_moe.router.router_factory import (
@@ -858,11 +858,7 @@ def make_fused_moe_layer(
     quant_config, qw = make_quant_config(quantization, w1, w2, global_num_experts)

     kwargs = dict()
-    if shared_experts is None:
diff -- tests/kernels/moe/test_shared_fused_moe_routed_transform.py
@@ -1,9 +1,9 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 """
-Tests for SharedFusedMoE with routed_input_transform.
+Tests for FusedMoE with routed_input_transform.

-Verifies that applying routed_input_transform inside SharedFusedMoE
+Verifies that applying routed_input_transform inside FusedMoE
 produces the same results as applying the transform manually outside.
 """

@@ -13,7 +13,7 @@

```
- 已读文件:
  - runtime: `vllm/distributed/device_communicators/base_device_communicator.py` modified +3/-10; `vllm/distributed/elastic_ep/elastic_execute.py` modified +2/-4; `vllm/lora/layers/fused_moe.py` modified +2/-2; `vllm/model_executor/layers/fused_moe/__init__.py` modified +0/-2; `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25; `vllm/model_executor/models/AXK1.py` modified +4/-4; `vllm/model_executor/models/afmoe.py` modified +5/-5; `vllm/model_executor/models/aria.py` modified +2/-2
  - tests: `tests/kernels/moe/test_moe_layer.py` modified +3/-7; `tests/kernels/moe/test_shared_fused_moe_routed_transform.py` modified +10/-10
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- 链接: https://github.com/vllm-project/vllm/pull/40671
- 状态/时间: merged / 2026-04-23
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 53 个文件，+254/-98，可读 patch 1073 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；模型线: Ling 2.5 1T；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py`。
- 实现要点:
  - `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0；symbols: fused_moe_make_expert_params_mapping
  - `vllm/model_executor/models/AXK1.py` modified +6/-3
  - `vllm/model_executor/models/afmoe.py` modified +5/-2
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0
  - `vllm/model_executor/models/AXK1.py` modified +6/-3
  - `vllm/model_executor/models/afmoe.py` modified +5/-2
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/__init__.py
@@ -19,6 +19,7 @@
 from vllm.model_executor.layers.fused_moe.layer import (
     FusedMoE,
     FusedMoeWeightScaleSupported,
+    fused_moe_make_expert_params_mapping,
 )
 from vllm.model_executor.layers.fused_moe.modular_kernel import (
     FusedMoEActivationFormat,
@@ -65,6 +66,7 @@ def get_config() -> dict[str, Any] | None:
     "RoutingMethodType",
     "activation_without_mul",
     "apply_moe_activation",
+    "fused_moe_make_expert_params_mapping",
     "override_config",
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -1618,6 +1618,25 @@ def extra_repr(self) -> str:
         return s


+# This is a temporary forwarding method which will be removed/modified layer.
+def fused_moe_make_expert_params_mapping(
+    model: torch.nn.Module,
+    ckpt_gate_proj_name: str,
+    ckpt_down_proj_name: str,
+    ckpt_up_proj_name: str,
+    num_experts: int,
+    num_redundant_experts: int = 0,
+) -> list[tuple[str, str, int, str]]:
+    return FusedMoE.make_expert_params_mapping(
```
- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0; `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0; `vllm/model_executor/models/AXK1.py` modified +6/-3; `vllm/model_executor/models/afmoe.py` modified +5/-2; `vllm/model_executor/models/arctic.py` modified +4/-1; `vllm/model_executor/models/aria.py` modified +3/-1; `vllm/model_executor/models/bailing_moe.py` modified +5/-2; `vllm/model_executor/models/bailing_moe_linear.py` modified +5/-2
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #40859 - [Bugfix ] fix bailing_moe_linear

- 链接: https://github.com/vllm-project/vllm/pull/40859
- 状态/时间: merged / 2026-04-28
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+15/-16，可读 patch 90 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix ] fix bailing_moe_linear」；模型线: Ling 2.5 1T；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/models/bailing_moe_linear.py`。
- 实现要点:
  - `vllm/model_executor/layers/mamba/mamba_utils.py` modified +0/-3
  - `vllm/model_executor/models/bailing_moe_linear.py` modified +15/-13；symbols: BailingMoELinearAttention
- 代码 diff 细节:
  - `vllm/model_executor/layers/mamba/mamba_utils.py` modified +0/-3
  - `vllm/model_executor/models/bailing_moe_linear.py` modified +15/-13
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/mamba/mamba_utils.py
@@ -55,9 +55,6 @@ def linear_attention_state_dtype(
         model_dtype: ModelDType | torch.dtype,
         mamba_cache_dtype: MambaDType,
     ) -> tuple[torch.dtype, ...]:
-        # TODO (tdoublep) requires testing
-        if mamba_cache_dtype == "float32":
-            raise ValueError("fp32 state for minimax is not yet supported")
         state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
         return (state_dtype,)

diff -- vllm/model_executor/models/bailing_moe_linear.py
@@ -17,6 +17,7 @@
 )
 from vllm.forward_context import get_forward_context
 from vllm.logger import init_logger
+from vllm.model_executor.custom_op import PluggableLayer
 from vllm.model_executor.layers.fla.ops.layernorm_guard import (
     RMSNormGated,
     layernorm_fn,
@@ -211,7 +212,6 @@ def __init__(
             max_position=max_position,
             is_neox_style=False,
             rope_parameters=rope_parameters or None,
-            dtype=torch.float32,
         )
```
- 已读文件:
  - runtime: `vllm/model_executor/layers/mamba/mamba_utils.py` modified +0/-3; `vllm/model_executor/models/bailing_moe_linear.py` modified +15/-13
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #41185 - [Bugfix] BailingMoeV2.5: rotate full qk_rope_head_dim in MLA RoPE

- 链接: https://github.com/vllm-project/vllm/pull/41185
- 状态/时间: merged / 2026-04-29
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-2，可读 patch 22 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] BailingMoeV2.5: rotate full qk_rope_head_dim in MLA RoPE」；模型线: Ling 2.5 1T；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/bailing_moe_linear.py`。
- 实现要点:
  - `vllm/model_executor/models/bailing_moe_linear.py` modified +8/-2
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe_linear.py` modified +8/-2
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe_linear.py
@@ -205,13 +205,19 @@ def __init__(
             self.q_a_layernorm = None
             self.q_b_proj = None

-        rope_parameters = _build_rope_parameters(config)
+        rope_parameters = _build_rope_parameters(config) or {}
+        # MLA rotates the full qk_rope_head_dim,
+        # partial_rotary_factor is for the linear-attn head only.
+        rope_parameters = {
+            k: v for k, v in rope_parameters.items() if k != "partial_rotary_factor"
+        }
+        rope_parameters["rope_dim"] = self.qk_rope_head_dim
         max_position = getattr(config, "max_position_embeddings", 8192)
         self.rotary_emb = get_rope(
```
- 已读文件:
  - runtime: `vllm/model_executor/models/bailing_moe_linear.py` modified +8/-2
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #41188 - [Misc] Replace mamba_type string literals with MambaAttentionBackendEnum

- 链接: https://github.com/vllm-project/vllm/pull/41188
- 状态/时间: merged / 2026-05-11
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+64/-58，可读 patch 404 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Replace mamba_type string literals with MambaAttentionBackendEnum」；模型线: Ling 2.5 1T；类别: 文档/测试/CI；主要 diff: `docs/contributing/model/basic.md`, `tests/kernels/mamba/test_ssu_dispatch.py`, `tests/v1/attention/test_attention_backends_selection.py`。
- 实现要点:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/kernels/mamba/test_ssu_dispatch.py` modified +10/-2；symbols: _kv_cache_config_with_ssu
  - `tests/v1/attention/test_attention_backends_selection.py` modified +13/-8
  - `vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py` modified +2/-1
- 代码 diff 细节:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/kernels/mamba/test_ssu_dispatch.py` modified +10/-2
  - `tests/v1/attention/test_attention_backends_selection.py` modified +13/-8
  - `vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py` modified +2/-1
- 关键代码摘录:

```diff
diff -- docs/contributing/model/basic.md
@@ -142,7 +142,7 @@ We use "mamba-like" to refer to layers that possess a state that is updated in-p
 For implementing new custom mamba-like layers, one should inherit from `MambaBase` and implement the methods `get_state_dtype`, `get_state_shape` to calculate the data types and state shapes at runtime, as well as `mamba_type` and `get_attn_backend`.
 It is also necessary to implement the "attention meta-data" class which handles the meta-data that is common across all layers.
 Please see [`LinearAttentionMetadata`](../../../vllm/v1/attention/backends/linear_attn.py) or [`ShortConvAttentionMetadata`](../../../vllm/v1/attention/backends/short_conv_attn.py) for examples of this.
-It is also worth noting that we should update `MAMBA_TYPE_TO_BACKEND_MAP` and `MambaAttentionBackendEnum` in [`registry.py`](../../../vllm/v1/attention/backends/registry.py) when adding a new mamba backend.
+It is also worth noting that we should update `MambaAttentionBackendEnum` in [`registry.py`](../../../vllm/v1/attention/backends/registry.py) when adding a new mamba backend.
 Finally, if one wants to support torch compile and CUDA graphs, it necessary to wrap the call to the mamba-like layer inside a custom op and register it.
 Please see the calls to `direct_register_custom_op` in [vllm/model_executor/models/minimax_text_01.py](../../../vllm/model_executor/models/minimax_text_01.py) or [vllm/model_executor/layers/mamba/short_conv.py](../../../vllm/model_executor/layers/mamba/short_conv.py) for examples of this.
 The new custom op should then be added to the list `_attention_ops` in [vllm/config/compilation.py](../../../vllm/config/compilation.py) to ensure that piecewise CUDA graphs works as intended.
diff -- tests/kernels/mamba/test_ssu_dispatch.py
@@ -13,6 +13,7 @@
     selective_state_update,
 )
 from vllm.utils.torch_utils import set_random_seed
+from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
 from vllm.v1.kv_cache_interface import (
     KVCacheConfig,
     KVCacheGroupSpec,
@@ -27,7 +28,9 @@
     HAS_FLASHINFER = False


-def _kv_cache_config_with_ssu(mamba_type: str = "mamba2") -> KVCacheConfig:
+def _kv_cache_config_with_ssu(
```
- 已读文件:
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py` modified +2/-1; `vllm/model_executor/layers/kda.py` modified +3/-2; `vllm/model_executor/layers/mamba/abstract.py` modified +2/-1; `vllm/model_executor/layers/mamba/gdn_linear_attn.py` modified +3/-2; `vllm/model_executor/layers/mamba/linear_attn.py` modified +3/-2; `vllm/model_executor/layers/mamba/mamba_mixer.py` modified +3/-2; `vllm/model_executor/layers/mamba/mamba_mixer2.py` modified +3/-2; `vllm/model_executor/layers/mamba/ops/ssu_dispatch.py` modified +3/-1
  - tests: `tests/kernels/mamba/test_ssu_dispatch.py` modified +10/-2; `tests/v1/attention/test_attention_backends_selection.py` modified +13/-8
  - docs/bench: `docs/contributing/model/basic.md` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。
