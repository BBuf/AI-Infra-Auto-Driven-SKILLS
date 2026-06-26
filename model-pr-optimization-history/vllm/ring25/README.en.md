# vllm Ring 2.5 1T Model PR Optimization History

## 2026-06-26 Latest Source Scan

Rechecked vLLM upstream `vllm-project/vllm@abc71548ef029132c3316b902207f254a246d593` against the tracked files listed below.
The file-level match used a GitHub mirror `git log --name-only`; PR titles, links, and merge times were batch-verified through the GitHub GraphQL Pull Request API. Previous freshness anchor: `2026-06-05`.

Result: 1 additional PR-numbered merge(s) touched tracked files and are not yet promoted into full per-PR diff audit cards below. Treat this section as a freshness index; promote any row into a full card only after manual diff review.

| Merged | PR | Title | Tracked files touched |
| --- | --- | --- | --- |
| 2026-06-08 | [#41184](https://github.com/vllm-project/vllm/pull/41184) | [MoE Refactor] FusedMoE/MoERunner inversion refactor | `bailing_moe_linear.py` |

## 2026-06-05 PR Backfill Audit

Rechecked vllm upstream `origin/main@c66b19800` on 2026-06-05; 3 additional PR-numbered merge(s) touched the tracked implementation files after the previous freshness cutoff (2026-05-19). These are not yet reflected in the timeline / diff-audit cards below and should be folded in on the next full regeneration.

| Merged | PR | Title | Tracked files touched |
| --- | --- | --- | --- |
| 2026-06-04 | [#43556](https://github.com/vllm-project/vllm/pull/43556) | [Attention] Mamba attention module refactor - LINEAR | `bailing_moe_linear.py` |
| 2026-06-01 | [#43770](https://github.com/vllm-project/vllm/pull/43770) | [Bugfix] fix wrong partial_rotary_factor calculation for bailing_moe model. | `bailing_moe.py` |
| 2026-05-26 | [#43410](https://github.com/vllm-project/vllm/pull/43410) | [Kernel] Porting  fuse_minimax_qk_norm  to manual fusion | `bailing_moe_linear.py` |


## 2026-05-19 Coverage Addition

Generated from vllm upstream `origin/main@ef54a4d604`, `git log --name-only -- <model-files>` over model-related paths, and the GitHub Pull Request files API. This page fills the missing `Ring 2.5 1T` history entry found from sgl-cookbook coverage.

> vLLM serves Ling/Ring through the Bailing MoE implementation. `docs/models/supported_models.md` is treated as a capability table and is not used for PR-card generation to avoid broad supported-models noise.

## Implementation File Coverage

| File | PRs traced by git |
| --- | --- |
| `vllm/model_executor/models/bailing_moe.py` | [#40671](https://github.com/vllm-project/vllm/pull/40671), [#35782](https://github.com/vllm-project/vllm/pull/35782), [#35949](https://github.com/vllm-project/vllm/pull/35949), [#33737](https://github.com/vllm-project/vllm/pull/33737), [#32064](https://github.com/vllm-project/vllm/pull/32064), [#33063](https://github.com/vllm-project/vllm/pull/33063), [#31104](https://github.com/vllm-project/vllm/pull/31104), [#30389](https://github.com/vllm-project/vllm/pull/30389), [#29966](https://github.com/vllm-project/vllm/pull/29966), [#29342](https://github.com/vllm-project/vllm/pull/29342), [#28542](https://github.com/vllm-project/vllm/pull/28542), [#28777](https://github.com/vllm-project/vllm/pull/28777), ... (28 total) |
| `vllm/model_executor/models/bailing_moe_linear.py` | [#41188](https://github.com/vllm-project/vllm/pull/41188), [#41185](https://github.com/vllm-project/vllm/pull/41185), [#40859](https://github.com/vllm-project/vllm/pull/40859), [#40671](https://github.com/vllm-project/vllm/pull/40671), [#35782](https://github.com/vllm-project/vllm/pull/35782), [#35949](https://github.com/vllm-project/vllm/pull/35949), [#37487](https://github.com/vllm-project/vllm/pull/37487), [#37195](https://github.com/vllm-project/vllm/pull/37195), [#35102](https://github.com/vllm-project/vllm/pull/35102) |

## PR Coverage Summary

- git-traced PR count: 34
- keyword/supplemental PR count: 0
- total PR count in this document: 34
- file trace command: `git log --name-only -- <model-files>`
- diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | Status | Title | Main files |
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

## Per-PR Diff Audit Cards

### PR #20680 - [Model] Add Ling implementation

- Link: https://github.com/vllm-project/vllm/pull/20680
- Status/date: merged / 2025-07-14
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 4 files, +534/-0, with 556 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add Ling implementation"; model line: Ring 2.5 1T; category: model support/runtime entry; main diff: `docs/models/supported_models.md`, `tests/models/registry.py`, `vllm/model_executor/models/bailing_moe.py`.
- Key implementation:
  - `docs/models/supported_models.md` modified +1/-0
  - `tests/models/registry.py` modified +2/-0
  - `vllm/model_executor/models/bailing_moe.py` added +530/-0; symbols: BailingAttention, __init__, forward, BailingMLP
  - `vllm/model_executor/models/registry.py` modified +1/-0
- Code diff details:
  - `docs/models/supported_models.md` modified +1/-0
  - `tests/models/registry.py` modified +2/-0
  - `vllm/model_executor/models/bailing_moe.py` added +530/-0
  - `vllm/model_executor/models/registry.py` modified +1/-0
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/bailing_moe.py` added +530/-0; `vllm/model_executor/models/registry.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +2/-0
  - docs/bench: `docs/models/supported_models.md` modified +1/-0
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #21059 - [Model] Remove model sampler

- Link: https://github.com/vllm-project/vllm/pull/21059
- Status/date: merged / 2025-07-16
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 6 files, +0/-45, with 157 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Remove model sampler"; model line: Ring 2.5 1T; category: model implementation change; main diff: `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/granite_speech.py`, `vllm/model_executor/models/hunyuan_v1_moe.py`.
- Key implementation:
  - `vllm/model_executor/models/bailing_moe.py` modified +0/-10
  - `vllm/model_executor/models/granite_speech.py` modified +0/-2
  - `vllm/model_executor/models/hunyuan_v1_moe.py` modified +0/-10
  - `vllm/model_executor/models/mimo.py` modified +0/-2
- Code diff details:
  - `vllm/model_executor/models/bailing_moe.py` modified +0/-10
  - `vllm/model_executor/models/granite_speech.py` modified +0/-2
  - `vllm/model_executor/models/hunyuan_v1_moe.py` modified +0/-10
  - `vllm/model_executor/models/mimo.py` modified +0/-2
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +0/-10; `vllm/model_executor/models/granite_speech.py` modified +0/-2; `vllm/model_executor/models/hunyuan_v1_moe.py` modified +0/-10; `vllm/model_executor/models/mimo.py` modified +0/-2; `vllm/model_executor/models/mimo_mtp.py` modified +0/-11; `vllm/model_executor/models/phi4flash.py` modified +0/-10
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #21100 - [Quantization] Enable BNB support for more MoE models

- Link: https://github.com/vllm-project/vllm/pull/21100
- Status/date: merged / 2025-07-19
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 5 files, +223/-181, with 548 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] Enable BNB support for more MoE models"; model line: Ring 2.5 1T; category: model support/runtime entry; main diff: `docs/models/supported_models.md`, `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/ernie45_moe.py`.
- Key implementation:
  - `docs/models/supported_models.md` modified +4/-4
  - `vllm/model_executor/models/bailing_moe.py` modified +14/-7; symbols: get_expert_mapping, BailingMoeForCausalLM
  - `vllm/model_executor/models/ernie45_moe.py` modified +84/-69; symbols: get_expert_mapping, Ernie4_5_MoeForCausalLM, __init__, get_input_embeddings
  - `vllm/model_executor/models/grok1.py` modified +14/-10; symbols: get_expert_mapping
- Code diff details:
  - `docs/models/supported_models.md` modified +4/-4
  - `vllm/model_executor/models/bailing_moe.py` modified +14/-7
  - `vllm/model_executor/models/ernie45_moe.py` modified +84/-69
  - `vllm/model_executor/models/grok1.py` modified +14/-10
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +14/-7; `vllm/model_executor/models/ernie45_moe.py` modified +84/-69; `vllm/model_executor/models/grok1.py` modified +14/-10; `vllm/model_executor/models/hunyuan_v1_moe.py` modified +107/-91
  - docs/bench: `docs/models/supported_models.md` modified +4/-4
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #21664 - support `torch.compile` for bailing moe

- Link: https://github.com/vllm-project/vllm/pull/21664
- Status/date: merged / 2025-07-26
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, with 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support `torch.compile` for bailing moe"; model line: Ring 2.5 1T; category: model support/runtime entry; main diff: `vllm/model_executor/models/bailing_moe.py`.
- Key implementation:
  - `vllm/model_executor/models/bailing_moe.py` modified +2/-0
- Code diff details:
  - `vllm/model_executor/models/bailing_moe.py` modified +2/-0
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +2/-0
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #19497 - [Models] Improve iteration over layers

- Link: https://github.com/vllm-project/vllm/pull/19497
- Status/date: merged / 2025-08-29
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 65 files, +129/-83, with 1109 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Improve iteration over layers"; model line: Ring 2.5 1T; category: model implementation change; main diff: `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/arctic.py`, `vllm/model_executor/models/baichuan.py`.
- Key implementation:
  - `vllm/model_executor/models/arcee.py` modified +2/-1
  - `vllm/model_executor/models/arctic.py` modified +2/-1
  - `vllm/model_executor/models/baichuan.py` modified +2/-1
  - `vllm/model_executor/models/bailing_moe.py` modified +2/-2
- Code diff details:
  - `vllm/model_executor/models/arcee.py` modified +2/-1
  - `vllm/model_executor/models/arctic.py` modified +2/-1
  - `vllm/model_executor/models/baichuan.py` modified +2/-1
  - `vllm/model_executor/models/bailing_moe.py` modified +2/-2
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/arcee.py` modified +2/-1; `vllm/model_executor/models/arctic.py` modified +2/-1; `vllm/model_executor/models/baichuan.py` modified +2/-1; `vllm/model_executor/models/bailing_moe.py` modified +2/-2; `vllm/model_executor/models/bamba.py` modified +1/-2; `vllm/model_executor/models/bloom.py` modified +2/-1; `vllm/model_executor/models/chameleon.py` modified +2/-1; `vllm/model_executor/models/chatglm.py` modified +2/-1
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #24627 - [Model]: support Ling2.0

- Link: https://github.com/vllm-project/vllm/pull/24627
- Status/date: merged / 2025-09-15
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 4 files, +170/-50, with 388 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model]: support Ling2.0"; model line: Ring 2.5 1T; category: model support/runtime entry; main diff: `docs/models/supported_models.md`, `tests/models/registry.py`, `vllm/model_executor/models/bailing_moe.py`.
- Key implementation:
  - `docs/models/supported_models.md` modified +1/-0
  - `tests/models/registry.py` modified +2/-0
  - `vllm/model_executor/models/bailing_moe.py` modified +166/-50; symbols: BailingMoeV2ForCausalLM
  - `vllm/model_executor/models/registry.py` modified +1/-0
- Code diff details:
  - `docs/models/supported_models.md` modified +1/-0
  - `tests/models/registry.py` modified +2/-0
  - `vllm/model_executor/models/bailing_moe.py` modified +166/-50
  - `vllm/model_executor/models/registry.py` modified +1/-0
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +166/-50; `vllm/model_executor/models/registry.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +2/-0
  - docs/bench: `docs/models/supported_models.md` modified +1/-0
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #25345 - [V0 Deprecation] Remove V0 sampling metadata

- Link: https://github.com/vllm-project/vllm/pull/25345
- Status/date: merged / 2025-09-21
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +123/-417, with 2063 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[V0 Deprecation] Remove V0 sampling metadata"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_opt.py`, `vllm/model_executor/__init__.py`.
- Key implementation:
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py` modified +3/-5; symbols: compute_logits
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_opt.py` modified +3/-5; symbols: compute_logits
  - `vllm/model_executor/__init__.py` modified +0/-2
  - `vllm/model_executor/layers/logits_processor.py` modified +0/-2
- Code diff details:
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py` modified +3/-5
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_opt.py` modified +3/-5
  - `vllm/model_executor/__init__.py` modified +0/-2
  - `vllm/model_executor/layers/logits_processor.py` modified +0/-2
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/__init__.py` modified +0/-2; `vllm/model_executor/layers/logits_processor.py` modified +0/-2; `vllm/model_executor/models/apertus.py` modified +1/-4; `vllm/model_executor/models/arcee.py` modified +3/-4; `vllm/model_executor/models/arctic.py` modified +1/-4; `vllm/model_executor/models/aria.py` modified +2/-5; `vllm/model_executor/models/aya_vision.py` modified +1/-4; `vllm/model_executor/models/baichuan.py` modified +1/-4
  - tests: `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py` modified +3/-5; `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_opt.py` modified +3/-5
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #25271 - Move`VllmConfig` from `config/__init__.py` to `config/vllm.py`

- Link: https://github.com/vllm-project/vllm/pull/25271
- Status/date: merged / 2025-09-30
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 36 files, +964/-905, with 2200 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Move`VllmConfig` from `config/__init__.py` to `config/vllm.py`"; model line: Ring 2.5 1T; category: model implementation change; main diff: `vllm/attention/layer.py`, `vllm/attention/layers/chunked_local_attention.py`, `vllm/config/__init__.py`.
- Key implementation:
  - `vllm/attention/layer.py` modified +1/-2
  - `vllm/attention/layers/chunked_local_attention.py` modified +2/-1
  - `vllm/config/__init__.py` modified +79/-826
  - `vllm/config/utils.py` modified +36/-6; symbols: SupportsHash, compute_hash, SupportsMetricsInfo, metrics_info
- Code diff details:
  - `vllm/attention/layer.py` modified +1/-2
  - `vllm/attention/layers/chunked_local_attention.py` modified +2/-1
  - `vllm/config/__init__.py` modified +79/-826
  - `vllm/config/utils.py` modified +36/-6
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/attention/layer.py` modified +1/-2; `vllm/attention/layers/chunked_local_attention.py` modified +2/-1; `vllm/config/__init__.py` modified +79/-826; `vllm/config/utils.py` modified +36/-6; `vllm/config/vllm.py` added +789/-0; `vllm/model_executor/layers/mamba/linear_attn.py` modified +1/-2; `vllm/model_executor/layers/quantization/auto_round.py` modified +2/-3; `vllm/model_executor/layers/quantization/bitblas.py` modified +2/-3
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #26247 - Convert formatting to use `ruff` instead of `yapf` + `isort`

- Link: https://github.com/vllm-project/vllm/pull/26247
- Status/date: merged / 2025-10-05
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +4320/-3882, with 14041 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Convert formatting to use `ruff` instead of `yapf` + `isort`"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `.buildkite/pyproject.toml`, `.pre-commit-config.yaml`, `benchmarks/benchmark_block_pool.py`.
- Key implementation:
  - `.buildkite/pyproject.toml` removed +0/-46
  - `.pre-commit-config.yaml` modified +0/-12
  - `benchmarks/benchmark_block_pool.py` modified +1/-1
  - `benchmarks/benchmark_ngram_proposer.py` modified +1/-1
- Code diff details:
  - `.buildkite/pyproject.toml` removed +0/-46
  - `.pre-commit-config.yaml` modified +0/-12
  - `benchmarks/benchmark_block_pool.py` modified +1/-1
  - `benchmarks/benchmark_ngram_proposer.py` modified +1/-1
- Key code excerpts:

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
- Reviewed files:
  - runtime: `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` modified +13/-15; `csrc/moe/marlin_moe_wna16/generate_kernels.py` modified +24/-18; `csrc/quantization/gptq_marlin/generate_kernels.py` modified +26/-22; `csrc/quantization/machete/generate.py` modified +100/-66
  - tests: `tests/basic_correctness/test_basic_correctness.py` modified +60/-62; `tests/basic_correctness/test_cpu_offload.py` modified +3/-2; `tests/basic_correctness/test_cumem.py` modified +12/-11; `tests/benchmarks/test_latency_cli.py` modified +12/-2; `tests/benchmarks/test_random_dataset.py` modified +69/-52; `tests/benchmarks/test_serve_cli.py` modified +2/-3; `tests/benchmarks/test_throughput_cli.py` modified +12/-2; `tests/compile/backend.py` modified +7/-11
  - docs/bench: `.buildkite/pyproject.toml` removed +0/-46; `benchmarks/benchmark_block_pool.py` modified +1/-1; `benchmarks/benchmark_ngram_proposer.py` modified +1/-1; `benchmarks/benchmark_serving_structured_output.py` modified +2/-3; `benchmarks/pyproject.toml` removed +0/-49; `docs/mkdocs/hooks/generate_argparse.py` modified +27/-36; `docs/mkdocs/hooks/generate_examples.py` modified +18/-16; `docs/mkdocs/hooks/remove_announcement.py` modified +1/-1
  - other: `.pre-commit-config.yaml` modified +0/-12; `cmake/hipify.py` modified +24/-19; `pyproject.toml` modified +100/-27; `setup.py` modified +151/-104
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #26262 - Fix per file ruff ignores related to line length

- Link: https://github.com/vllm-project/vllm/pull/26262
- Status/date: merged / 2025-10-06
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 65 files, +301/-291, with 1525 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix per file ruff ignores related to line length"; model line: Ring 2.5 1T; category: bug fix; main diff: `benchmarks/benchmark_ngram_proposer.py`, `benchmarks/benchmark_serving_structured_output.py`, `csrc/cutlass_extensions/vllm_cutlass_library_extension.py`.
- Key implementation:
  - `benchmarks/benchmark_ngram_proposer.py` modified +1/-1
  - `benchmarks/benchmark_serving_structured_output.py` modified +2/-2
  - `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` modified +3/-3
  - `examples/offline_inference/vision_language_pooling.py` modified +1/-1
- Code diff details:
  - `benchmarks/benchmark_ngram_proposer.py` modified +1/-1
  - `benchmarks/benchmark_serving_structured_output.py` modified +2/-2
  - `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` modified +3/-3
  - `examples/offline_inference/vision_language_pooling.py` modified +1/-1
- Key code excerpts:

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
- Reviewed files:
  - runtime: `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` modified +3/-3
  - tests: `tests/compile/piecewise/test_simple.py` modified +17/-11; `tests/compile/piecewise/test_toy_llama.py` modified +9/-5; `tests/compile/test_functionalization.py` modified +1/-1; `tests/compile/test_fusion_attn.py` modified +1/-1; `tests/compile/test_sequence_parallelism.py` modified +2/-2; `tests/distributed/test_pipeline_parallel.py` modified +4/-4; `tests/entrypoints/conftest.py` modified +2/-1; `tests/entrypoints/openai/test_audio.py` modified +1/-1
  - docs/bench: `benchmarks/benchmark_ngram_proposer.py` modified +1/-1; `benchmarks/benchmark_serving_structured_output.py` modified +2/-2; `examples/offline_inference/vision_language_pooling.py` modified +1/-1; `examples/online_serving/disaggregated_serving/disagg_proxy_demo.py` modified +2/-2
  - other: `pyproject.toml` modified +0/-46
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #26145 - [Model] Apply shared experts overlap optimization to all models with shared experts

- Link: https://github.com/vllm-project/vllm/pull/26145
- Status/date: merged / 2025-10-09
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 15 files, +271/-283, with 1118 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Apply shared experts overlap optimization to all models with shared experts"; model line: Ring 2.5 1T; category: model implementation change; main diff: `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/layers/quantization/fp8.py`.
- Key implementation:
  - `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0
  - `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` renamed +23/-12
  - `vllm/model_executor/layers/quantization/fp8.py` modified +2/-0
  - `vllm/model_executor/layers/shared_fused_moe/__init__.py` removed +0/-5
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0
  - `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` renamed +23/-12
  - `vllm/model_executor/layers/quantization/fp8.py` modified +2/-0
  - `vllm/model_executor/layers/shared_fused_moe/__init__.py` removed +0/-5
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0; `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` renamed +23/-12; `vllm/model_executor/layers/quantization/fp8.py` modified +2/-0; `vllm/model_executor/layers/shared_fused_moe/__init__.py` removed +0/-5; `vllm/model_executor/models/aria.py` modified +15/-13; `vllm/model_executor/models/bailing_moe.py` modified +26/-21; `vllm/model_executor/models/deepseek_v2.py` modified +24/-45; `vllm/model_executor/models/dots1.py` modified +22/-18
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #26633 - Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y`

- Link: https://github.com/vllm-project/vllm/pull/26633
- Status/date: merged / 2025-10-12
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +481/-581, with 3511 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y`"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `benchmarks/backend_request_func.py`, `benchmarks/benchmark_prefix_caching.py`, `benchmarks/benchmark_prioritization.py`.
- Key implementation:
  - `benchmarks/backend_request_func.py` modified +13/-14
  - `benchmarks/benchmark_prefix_caching.py` modified +2/-3
  - `benchmarks/benchmark_prioritization.py` modified +1/-2
  - `benchmarks/benchmark_serving_structured_output.py` modified +3/-4
- Code diff details:
  - `benchmarks/backend_request_func.py` modified +13/-14
  - `benchmarks/benchmark_prefix_caching.py` modified +2/-3
  - `benchmarks/benchmark_prioritization.py` modified +1/-2
  - `benchmarks/benchmark_serving_structured_output.py` modified +3/-4
- Key code excerpts:

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
- Reviewed files:
  - runtime: `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` modified +6/-9; `csrc/quantization/machete/generate.py` modified +2/-3
  - docs/bench: `benchmarks/backend_request_func.py` modified +13/-14; `benchmarks/benchmark_prefix_caching.py` modified +2/-3; `benchmarks/benchmark_prioritization.py` modified +1/-2; `benchmarks/benchmark_serving_structured_output.py` modified +3/-4; `benchmarks/benchmark_utils.py` modified +8/-8; `benchmarks/cutlass_benchmarks/sparse_benchmarks.py` modified +1/-2; `benchmarks/cutlass_benchmarks/w8a8_benchmarks.py` modified +5/-6; `benchmarks/fused_kernels/layernorm_rms_benchmarks.py` modified +4/-5
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #26876 - [Fix] Remove divisibility requirement between num_kv_heads and tp_size in bailing_moe

- Link: https://github.com/vllm-project/vllm/pull/26876
- Status/date: merged / 2025-10-15
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, with 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Remove divisibility requirement between num_kv_heads and tp_size in bailing_moe"; model line: Ring 2.5 1T; category: bug fix; main diff: `vllm/model_executor/models/bailing_moe.py`.
- Key implementation:
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-2
- Code diff details:
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-2
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +1/-2
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #28382 - [LoRA][1/N]Remove LoRA extra vocab

- Link: https://github.com/vllm-project/vllm/pull/28382
- Status/date: merged / 2025-11-11
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 65 files, +197/-754, with 2645 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[LoRA][1/N]Remove LoRA extra vocab"; model line: Ring 2.5 1T; category: model implementation change; main diff: `vllm/model_executor/models/apertus.py`, `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/arctic.py`.
- Key implementation:
  - `vllm/model_executor/models/apertus.py` modified +5/-25
  - `vllm/model_executor/models/arcee.py` modified +2/-8
  - `vllm/model_executor/models/arctic.py` modified +2/-4
  - `vllm/model_executor/models/aria.py` modified +2/-6
- Code diff details:
  - `vllm/model_executor/models/apertus.py` modified +5/-25
  - `vllm/model_executor/models/arcee.py` modified +2/-8
  - `vllm/model_executor/models/arctic.py` modified +2/-4
  - `vllm/model_executor/models/aria.py` modified +2/-6
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/apertus.py` modified +5/-25; `vllm/model_executor/models/arcee.py` modified +2/-8; `vllm/model_executor/models/arctic.py` modified +2/-4; `vllm/model_executor/models/aria.py` modified +2/-6; `vllm/model_executor/models/baichuan.py` modified +2/-2; `vllm/model_executor/models/bailing_moe.py` modified +0/-2; `vllm/model_executor/models/bamba.py` modified +6/-24; `vllm/model_executor/models/chameleon.py` modified +3/-5
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #27583 - Rename clashing method names for vLLM model protocol

- Link: https://github.com/vllm-project/vllm/pull/27583
- Status/date: merged / 2025-11-13
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +368/-367, with 2559 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Rename clashing method names for vLLM model protocol"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `docs/contributing/model/basic.md`, `docs/contributing/model/multimodal.md`, `vllm/model_executor/models/apertus.py`.
- Key implementation:
  - `docs/contributing/model/basic.md` modified +2/-2; symbols: embed_input_ids
  - `docs/contributing/model/multimodal.md` modified +3/-3; symbols: embed_multimodal
  - `vllm/model_executor/models/apertus.py` modified +4/-4; symbols: embed_input_ids
  - `vllm/model_executor/models/arcee.py` modified +4/-4; symbols: embed_input_ids
- Code diff details:
  - `docs/contributing/model/basic.md` modified +2/-2
  - `docs/contributing/model/multimodal.md` modified +3/-3
  - `vllm/model_executor/models/apertus.py` modified +4/-4
  - `vllm/model_executor/models/arcee.py` modified +4/-4
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/apertus.py` modified +4/-4; `vllm/model_executor/models/arcee.py` modified +4/-4; `vllm/model_executor/models/arctic.py` modified +4/-4; `vllm/model_executor/models/aria.py` modified +3/-3; `vllm/model_executor/models/aya_vision.py` modified +1/-1; `vllm/model_executor/models/baichuan.py` modified +4/-4; `vllm/model_executor/models/bailing_moe.py` modified +4/-4; `vllm/model_executor/models/bamba.py` modified +4/-4
  - docs/bench: `docs/contributing/model/basic.md` modified +2/-2; `docs/contributing/model/multimodal.md` modified +3/-3
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #28277 - [Model] Fix bailing_moe accuracy problem

- Link: https://github.com/vllm-project/vllm/pull/28277
- Status/date: merged / 2025-11-14
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-2, with 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Fix bailing_moe accuracy problem"; model line: Ring 2.5 1T; category: bug fix; main diff: `vllm/model_executor/models/bailing_moe.py`.
- Key implementation:
  - `vllm/model_executor/models/bailing_moe.py` modified +3/-2
- Code diff details:
  - `vllm/model_executor/models/bailing_moe.py` modified +3/-2
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +3/-2
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #28777 - [Model] Fix lmhead init bug of bailing_moe

- Link: https://github.com/vllm-project/vllm/pull/28777
- Status/date: merged / 2025-11-15
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, with 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Fix lmhead init bug of bailing_moe"; model line: Ring 2.5 1T; category: bug fix; main diff: `vllm/model_executor/models/bailing_moe.py`.
- Key implementation:
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-1
- Code diff details:
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-1
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/bailing_moe.py` modified +1/-1
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #28542 - Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5

- Link: https://github.com/vllm-project/vllm/pull/28542
- Status/date: merged / 2025-11-19
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +487/-868, with 4411 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `.buildkite/test-pipeline.yaml`, `benchmarks/kernels/benchmark_mrope.py`, `examples/offline_inference/context_extension.py`.
- Key implementation:
  - `.buildkite/test-pipeline.yaml` modified +3/-3
  - `benchmarks/kernels/benchmark_mrope.py` modified +7/-12
  - `examples/offline_inference/context_extension.py` modified +3/-3
  - `tests/compile/test_functionalization.py` modified +2/-2
- Code diff details:
  - `.buildkite/test-pipeline.yaml` modified +3/-3
  - `benchmarks/kernels/benchmark_mrope.py` modified +7/-12
  - `examples/offline_inference/context_extension.py` modified +3/-3
  - `tests/compile/test_functionalization.py` modified +2/-2
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/config/model.py` modified +29/-34; `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38; `vllm/model_executor/models/afmoe.py` modified +1/-16; `vllm/model_executor/models/apertus.py` modified +2/-20; `vllm/model_executor/models/arcee.py` modified +0/-11; `vllm/model_executor/models/arctic.py` modified +1/-2; `vllm/model_executor/models/baichuan.py` modified +3/-5; `vllm/model_executor/models/bailing_moe.py` modified +1/-2
  - tests: `tests/compile/test_functionalization.py` modified +2/-2; `tests/kernels/core/test_mrope.py` modified +5/-11; `tests/kernels/core/test_pos_encoding.py` modified +20/-19; `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +1/-1; `tests/models/language/pooling/test_nomic_max_model_len.py` modified +9/-7; `tests/test_config.py` modified +20/-17
  - docs/bench: `.buildkite/test-pipeline.yaml` modified +3/-3; `benchmarks/kernels/benchmark_mrope.py` modified +7/-12; `examples/offline_inference/context_extension.py` modified +3/-3
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #29342 - [Attention] Remove imports from `vllm/attention/__init__.py`

- Link: https://github.com/vllm-project/vllm/pull/29342
- Status/date: merged / 2025-11-26
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 96 files, +120/-121, with 923 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Attention] Remove imports from `vllm/attention/__init__.py`"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `docs/contributing/model/basic.md`, `tests/compile/test_fusion_attn.py`, `tests/compile/test_qk_norm_rope_fusion.py`.
- Key implementation:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/compile/test_fusion_attn.py` modified +2/-1
  - `tests/compile/test_qk_norm_rope_fusion.py` modified +2/-1
  - `tests/kernels/utils.py` modified +1/-1
- Code diff details:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/compile/test_fusion_attn.py` modified +2/-1
  - `tests/compile/test_qk_norm_rope_fusion.py` modified +2/-1
  - `tests/kernels/utils.py` modified +1/-1
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/attention/__init__.py` modified +0/-19; `vllm/attention/backends/abstract.py` modified +1/-1; `vllm/attention/layer.py` modified +5/-2; `vllm/compilation/fusion_attn.py` modified +1/-1; `vllm/compilation/qk_norm_rope_fusion.py` modified +1/-1; `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` modified +1/-1; `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` modified +2/-1; `vllm/model_executor/layers/mamba/linear_attn.py` modified +1/-1
  - tests: `tests/compile/test_fusion_attn.py` modified +2/-1; `tests/compile/test_qk_norm_rope_fusion.py` modified +2/-1; `tests/kernels/utils.py` modified +1/-1; `tests/v1/worker/test_gpu_model_runner.py` modified +1/-1; `tests/v1/worker/test_utils.py` modified +2/-2
  - docs/bench: `docs/contributing/model/basic.md` modified +1/-1
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #29966 - Access `partial_rotary_factor` from `rope_parameters`

- Link: https://github.com/vllm-project/vllm/pull/29966
- Status/date: merged / 2025-12-04
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 21 files, +43/-62, with 396 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Access `partial_rotary_factor` from `rope_parameters`"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `tests/kernels/core/test_mrope.py`, `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/model_executor/models/apertus.py`.
- Key implementation:
  - `tests/kernels/core/test_mrope.py` modified +2/-6
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +4/-1
  - `vllm/model_executor/models/apertus.py` modified +1/-4
  - `vllm/model_executor/models/bailing_moe.py` modified +0/-3
- Code diff details:
  - `tests/kernels/core/test_mrope.py` modified +2/-6
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +4/-1
  - `vllm/model_executor/models/apertus.py` modified +1/-4
  - `vllm/model_executor/models/bailing_moe.py` modified +0/-3
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +4/-1; `vllm/model_executor/models/apertus.py` modified +1/-4; `vllm/model_executor/models/bailing_moe.py` modified +0/-3; `vllm/model_executor/models/bamba.py` modified +1/-3; `vllm/model_executor/models/config.py` modified +0/-5; `vllm/model_executor/models/falcon_h1.py` modified +1/-3; `vllm/model_executor/models/glm.py` modified +2/-1; `vllm/model_executor/models/glm4.py` modified +1/-2
  - tests: `tests/kernels/core/test_mrope.py` modified +2/-6
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #30389 - Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim`

- Link: https://github.com/vllm-project/vllm/pull/30389
- Status/date: merged / 2025-12-11
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 83 files, +238/-292, with 1379 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim`"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `benchmarks/kernels/benchmark_mrope.py`, `benchmarks/kernels/benchmark_rope.py`, `tests/compile/test_functionalization.py`.
- Key implementation:
  - `benchmarks/kernels/benchmark_mrope.py` modified +0/-1
  - `benchmarks/kernels/benchmark_rope.py` modified +2/-2
  - `tests/compile/test_functionalization.py` modified +1/-4; symbols: __init__
  - `tests/kernels/core/test_mrope.py` modified +0/-2
- Code diff details:
  - `benchmarks/kernels/benchmark_mrope.py` modified +0/-1
  - `benchmarks/kernels/benchmark_rope.py` modified +2/-2
  - `tests/compile/test_functionalization.py` modified +1/-4
  - `tests/kernels/core/test_mrope.py` modified +0/-2
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/config/utils.py` modified +16/-2; `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +160/-166; `vllm/model_executor/models/afmoe.py` modified +0/-1; `vllm/model_executor/models/apertus.py` modified +0/-1; `vllm/model_executor/models/arctic.py` modified +0/-1; `vllm/model_executor/models/baichuan.py` modified +0/-1; `vllm/model_executor/models/bailing_moe.py` modified +2/-2; `vllm/model_executor/models/bamba.py` modified +2/-5
  - tests: `tests/compile/test_functionalization.py` modified +1/-4; `tests/kernels/core/test_mrope.py` modified +0/-2; `tests/kernels/core/test_pos_encoding.py` modified +8/-4
  - docs/bench: `benchmarks/kernels/benchmark_mrope.py` modified +0/-1; `benchmarks/kernels/benchmark_rope.py` modified +2/-2
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #31104 - [BugFix] LoRA: Support loading base_layer of experts

- Link: https://github.com/vllm-project/vllm/pull/31104
- Status/date: merged / 2026-01-07
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 35 files, +46/-3, with 319 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] LoRA: Support loading base_layer of experts"; model line: Ring 2.5 1T; category: bug fix; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/bailing_moe.py`.
- Key implementation:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3
  - `vllm/model_executor/models/afmoe.py` modified +1/-0
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-0
  - `vllm/model_executor/models/deepseek_eagle.py` modified +1/-0
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3
  - `vllm/model_executor/models/afmoe.py` modified +1/-0
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-0
  - `vllm/model_executor/models/deepseek_eagle.py` modified +1/-0
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3; `vllm/model_executor/models/afmoe.py` modified +1/-0; `vllm/model_executor/models/bailing_moe.py` modified +1/-0; `vllm/model_executor/models/deepseek_eagle.py` modified +1/-0; `vllm/model_executor/models/deepseek_mtp.py` modified +1/-0; `vllm/model_executor/models/deepseek_v2.py` modified +2/-0; `vllm/model_executor/models/dots1.py` modified +1/-0; `vllm/model_executor/models/ernie45_moe.py` modified +1/-0
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #33063 - [Chore] Update type annotation of `input_ids` in model forward

- Link: https://github.com/vllm-project/vllm/pull/33063
- Status/date: merged / 2026-01-26
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +146/-143, with 1304 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Chore] Update type annotation of `input_ids` in model forward"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `docs/contributing/model/basic.md`, `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`, `vllm/model_executor/models/afmoe.py`.
- Key implementation:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +1/-1
  - `vllm/model_executor/models/afmoe.py` modified +2/-2
  - `vllm/model_executor/models/apertus.py` modified +1/-1
- Code diff details:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +1/-1
  - `vllm/model_executor/models/afmoe.py` modified +2/-2
  - `vllm/model_executor/models/apertus.py` modified +1/-1
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/afmoe.py` modified +2/-2; `vllm/model_executor/models/apertus.py` modified +1/-1; `vllm/model_executor/models/arcee.py` modified +1/-1; `vllm/model_executor/models/arctic.py` modified +2/-2; `vllm/model_executor/models/aria.py` modified +1/-1; `vllm/model_executor/models/audioflamingo3.py` modified +1/-1; `vllm/model_executor/models/aya_vision.py` modified +1/-1; `vllm/model_executor/models/bagel.py` modified +1/-1
  - tests: `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py` modified +1/-1
  - docs/bench: `docs/contributing/model/basic.md` modified +1/-1
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #32064 - [5/N][Attention] Finish eliminating `vllm/attention` folder

- Link: https://github.com/vllm-project/vllm/pull/32064
- Status/date: merged / 2026-01-27
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +524/-468, with 2150 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[5/N][Attention] Finish eliminating `vllm/attention` folder"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `.buildkite/test-amd.yaml`, `.buildkite/test-pipeline.yaml`, `.buildkite/test_areas/kernels.yaml`.
- Key implementation:
  - `.buildkite/test-amd.yaml` modified +2/-1
  - `.buildkite/test-pipeline.yaml` modified +2/-1
  - `.buildkite/test_areas/kernels.yaml` modified +2/-1
  - `.github/CODEOWNERS` modified +1/-1
- Code diff details:
  - `.buildkite/test-amd.yaml` modified +2/-1
  - `.buildkite/test-pipeline.yaml` modified +2/-1
  - `.buildkite/test_areas/kernels.yaml` modified +2/-1
  - `.github/CODEOWNERS` modified +1/-1
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/attention/__init__.py` removed +0/-0; `vllm/attention/utils/__init__.py` removed +0/-0; `vllm/attention/utils/kv_sharing_utils.py` removed +0/-33; `vllm/compilation/fusion_attn.py` modified +1/-1; `vllm/compilation/qk_norm_rope_fusion.py` modified +1/-1; `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` modified +1/-1; `vllm/model_executor/layers/attention/__init__.py` modified +26/-0; `vllm/model_executor/layers/attention/attention.py` renamed +42/-315
  - tests: `tests/compile/test_fusion_attn.py` modified +1/-1; `tests/compile/test_qk_norm_rope_fusion.py` modified +1/-1; `tests/kernels/attention/test_attention.py` modified +1/-2; `tests/kernels/attention/test_mha_attn.py` modified +1/-1; `tests/v1/worker/test_gpu_model_runner.py` modified +1/-1; `tests/v1/worker/test_utils.py` modified +3/-3
  - docs/bench: `.buildkite/test-amd.yaml` modified +2/-1; `.buildkite/test-pipeline.yaml` modified +2/-1; `.buildkite/test_areas/kernels.yaml` modified +2/-1; `docs/contributing/model/basic.md` modified +1/-1; `docs/design/custom_op.md` modified +1/-1
  - other: `.github/CODEOWNERS` modified +1/-1; `tools/pre_commit/mypy.py` modified +0/-1
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #33737 - [Bugfix] Define router_logits_dtype for remaining MoE models

- Link: https://github.com/vllm-project/vllm/pull/33737
- Status/date: merged / 2026-02-04
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 6 files, +9/-4, with 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Define router_logits_dtype for remaining MoE models"; model line: Ring 2.5 1T; category: bug fix; main diff: `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/bailing_moe.py`, `vllm/model_executor/models/flex_olmo.py`.
- Key implementation:
  - `vllm/model_executor/models/afmoe.py` modified +1/-0
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-0
  - `vllm/model_executor/models/flex_olmo.py` modified +1/-1
  - `vllm/model_executor/models/longcat_flash.py` modified +4/-3
- Code diff details:
  - `vllm/model_executor/models/afmoe.py` modified +1/-0
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-0
  - `vllm/model_executor/models/flex_olmo.py` modified +1/-1
  - `vllm/model_executor/models/longcat_flash.py` modified +4/-3
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/afmoe.py` modified +1/-0; `vllm/model_executor/models/bailing_moe.py` modified +1/-0; `vllm/model_executor/models/flex_olmo.py` modified +1/-1; `vllm/model_executor/models/longcat_flash.py` modified +4/-3; `vllm/model_executor/models/mimo_v2_flash.py` modified +1/-0; `vllm/model_executor/models/step3p5.py` modified +1/-0
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #35102 - [Model] Ring 2.5

- Link: https://github.com/vllm-project/vllm/pull/35102
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 8 files, +1407/-70, with 1650 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Ring 2.5"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `docs/models/supported_models.md`, `tests/models/registry.py`, `vllm/model_executor/layers/fla/ops/layernorm_guard.py`.
- Key implementation:
  - `docs/models/supported_models.md` modified +1/-0
  - `tests/models/registry.py` modified +3/-0
  - `vllm/model_executor/layers/fla/ops/layernorm_guard.py` modified +30/-5
  - `vllm/model_executor/layers/layernorm.py` modified +1/-0
- Code diff details:
  - `docs/models/supported_models.md` modified +1/-0
  - `tests/models/registry.py` modified +3/-0
  - `vllm/model_executor/layers/fla/ops/layernorm_guard.py` modified +30/-5
  - `vllm/model_executor/layers/layernorm.py` modified +1/-0
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/layers/fla/ops/layernorm_guard.py` modified +30/-5; `vllm/model_executor/layers/layernorm.py` modified +1/-0; `vllm/model_executor/layers/mamba/linear_attn.py` modified +124/-65; `vllm/model_executor/models/bailing_moe_linear.py` added +1246/-0; `vllm/model_executor/models/registry.py` modified +1/-0; `vllm/transformers_utils/model_arch_config_convertor.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +3/-0
  - docs/bench: `docs/models/supported_models.md` modified +1/-0
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #37195 - [V0 Deprecation] Deprecate virtual engine

- Link: https://github.com/vllm-project/vllm/pull/37195
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 23 files, +23/-45, with 353 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[V0 Deprecation] Deprecate virtual engine"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `tests/compile/passes/test_rope_kvcache_fusion.py`, `tests/v1/kv_connector/unit/test_decode_bench_connector.py`, `tests/v1/kv_connector/unit/test_lmcache_integration.py`.
- Key implementation:
  - `tests/compile/passes/test_rope_kvcache_fusion.py` modified +2/-2
  - `tests/v1/kv_connector/unit/test_decode_bench_connector.py` modified +1/-1
  - `tests/v1/kv_connector/unit/test_lmcache_integration.py` modified +0/-1
  - `tests/v1/kv_connector/unit/test_nixl_connector.py` modified +0/-8
- Code diff details:
  - `tests/compile/passes/test_rope_kvcache_fusion.py` modified +2/-2
  - `tests/v1/kv_connector/unit/test_decode_bench_connector.py` modified +1/-1
  - `tests/v1/kv_connector/unit/test_lmcache_integration.py` modified +0/-1
  - `tests/v1/kv_connector/unit/test_nixl_connector.py` modified +0/-8
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py` modified +1/-1; `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py` modified +1/-3; `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` modified +1/-1; `vllm/forward_context.py` modified +0/-7; `vllm/model_executor/layers/attention/attention.py` modified +2/-2; `vllm/model_executor/layers/attention/mla_attention.py` modified +2/-2; `vllm/model_executor/layers/attention/static_sink_attention.py` modified +1/-2; `vllm/model_executor/layers/kda.py` modified +1/-1
  - tests: `tests/compile/passes/test_rope_kvcache_fusion.py` modified +2/-2; `tests/v1/kv_connector/unit/test_decode_bench_connector.py` modified +1/-1; `tests/v1/kv_connector/unit/test_lmcache_integration.py` modified +0/-1; `tests/v1/kv_connector/unit/test_nixl_connector.py` modified +0/-8; `tests/v1/kv_connector/unit/test_offloading_connector.py` modified +0/-1
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #37487 - [V0 Deprecation] Refactor kv cache from list to element

- Link: https://github.com/vllm-project/vllm/pull/37487
- Status/date: merged / 2026-03-24
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 27 files, +70/-85, with 478 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[V0 Deprecation] Refactor kv cache from list to element"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `tests/compile/passes/test_fusion_attn.py`, `tests/compile/passes/test_rope_kvcache_fusion.py`, `tests/v1/e2e/general/test_mamba_prefix_cache.py`.
- Key implementation:
  - `tests/compile/passes/test_fusion_attn.py` modified +1/-1
  - `tests/compile/passes/test_rope_kvcache_fusion.py` modified +3/-3
  - `tests/v1/e2e/general/test_mamba_prefix_cache.py` modified +3/-3
  - `tests/v1/worker/test_gpu_model_runner.py` modified +17/-17
- Code diff details:
  - `tests/compile/passes/test_fusion_attn.py` modified +1/-1
  - `tests/compile/passes/test_rope_kvcache_fusion.py` modified +3/-3
  - `tests/v1/e2e/general/test_mamba_prefix_cache.py` modified +3/-3
  - `tests/v1/worker/test_gpu_model_runner.py` modified +17/-17
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py` modified +2/-4; `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py` modified +1/-1; `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` modified +1/-1; `vllm/model_executor/layers/attention/attention.py` modified +2/-5; `vllm/model_executor/layers/attention/mla_attention.py` modified +3/-8; `vllm/model_executor/layers/attention/static_sink_attention.py` modified +1/-1; `vllm/model_executor/layers/kda.py` modified +1/-1; `vllm/model_executor/layers/mamba/linear_attn.py` modified +1/-1
  - tests: `tests/compile/passes/test_fusion_attn.py` modified +1/-1; `tests/compile/passes/test_rope_kvcache_fusion.py` modified +3/-3; `tests/v1/e2e/general/test_mamba_prefix_cache.py` modified +3/-3; `tests/v1/worker/test_gpu_model_runner.py` modified +17/-17; `tests/v1/worker/test_utils.py` modified +10/-10
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #35949 - [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase

- Link: https://github.com/vllm-project/vllm/pull/35949
- Status/date: merged / 2026-04-20
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 53 files, +325/-702, with 2430 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase"; model line: Ring 2.5 1T; category: performance/backend optimization; main diff: `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py`, `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`.
- Key implementation:
  - `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py` modified +3/-5
  - `tests/kernels/moe/test_moe_layer.py` modified +17/-76
  - `tests/kernels/moe/test_shared_fused_moe_routed_transform.py` modified +10/-26
  - `vllm/lora/layers/fused_moe.py` modified +0/-3
- Code diff details:
  - `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py` modified +3/-5
  - `tests/kernels/moe/test_moe_layer.py` modified +17/-76
  - `tests/kernels/moe/test_shared_fused_moe_routed_transform.py` modified +10/-26
  - `vllm/lora/layers/fused_moe.py` modified +0/-3
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/lora/layers/fused_moe.py` modified +0/-3; `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` modified +1/-1; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32; `vllm/model_executor/layers/fused_moe/modular_kernel.py` modified +1/-1; `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` modified +0/-4; `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +1/-12; `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86; `vllm/model_executor/layers/fused_moe/runner/moe_runner_factory.py` modified +4/-2
  - tests: `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py` modified +3/-5; `tests/kernels/moe/test_moe_layer.py` modified +17/-76; `tests/kernels/moe/test_shared_fused_moe_routed_transform.py` modified +10/-26
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #35782 - [MoE Refactor] Remove SharedFusedMoE class

- Link: https://github.com/vllm-project/vllm/pull/35782
- Status/date: merged / 2026-04-21
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 33 files, +112/-141, with 926 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Remove SharedFusedMoE class"; model line: Ring 2.5 1T; category: performance/backend optimization; main diff: `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`, `vllm/distributed/device_communicators/base_device_communicator.py`.
- Key implementation:
  - `tests/kernels/moe/test_moe_layer.py` modified +3/-7
  - `tests/kernels/moe/test_shared_fused_moe_routed_transform.py` modified +10/-10
  - `vllm/distributed/device_communicators/base_device_communicator.py` modified +3/-10
  - `vllm/distributed/elastic_ep/elastic_execute.py` modified +2/-4
- Code diff details:
  - `tests/kernels/moe/test_moe_layer.py` modified +3/-7
  - `tests/kernels/moe/test_shared_fused_moe_routed_transform.py` modified +10/-10
  - `vllm/distributed/device_communicators/base_device_communicator.py` modified +3/-10
  - `vllm/distributed/elastic_ep/elastic_execute.py` modified +2/-4
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/distributed/device_communicators/base_device_communicator.py` modified +3/-10; `vllm/distributed/elastic_ep/elastic_execute.py` modified +2/-4; `vllm/lora/layers/fused_moe.py` modified +2/-2; `vllm/model_executor/layers/fused_moe/__init__.py` modified +0/-2; `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25; `vllm/model_executor/models/AXK1.py` modified +4/-4; `vllm/model_executor/models/afmoe.py` modified +5/-5; `vllm/model_executor/models/aria.py` modified +2/-2
  - tests: `tests/kernels/moe/test_moe_layer.py` modified +3/-7; `tests/kernels/moe/test_shared_fused_moe_routed_transform.py` modified +10/-10
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- Link: https://github.com/vllm-project/vllm/pull/40671
- Status/date: merged / 2026-04-23
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 53 files, +254/-98, with 1073 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; model line: Ring 2.5 1T; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py`.
- Key implementation:
  - `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0; symbols: fused_moe_make_expert_params_mapping
  - `vllm/model_executor/models/AXK1.py` modified +6/-3
  - `vllm/model_executor/models/afmoe.py` modified +5/-2
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0
  - `vllm/model_executor/models/AXK1.py` modified +6/-3
  - `vllm/model_executor/models/afmoe.py` modified +5/-2
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0; `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0; `vllm/model_executor/models/AXK1.py` modified +6/-3; `vllm/model_executor/models/afmoe.py` modified +5/-2; `vllm/model_executor/models/arctic.py` modified +4/-1; `vllm/model_executor/models/aria.py` modified +3/-1; `vllm/model_executor/models/bailing_moe.py` modified +5/-2; `vllm/model_executor/models/bailing_moe_linear.py` modified +5/-2
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #40859 - [Bugfix ] fix bailing_moe_linear

- Link: https://github.com/vllm-project/vllm/pull/40859
- Status/date: merged / 2026-04-28
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 2 files, +15/-16, with 90 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix ] fix bailing_moe_linear"; model line: Ring 2.5 1T; category: bug fix; main diff: `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/models/bailing_moe_linear.py`.
- Key implementation:
  - `vllm/model_executor/layers/mamba/mamba_utils.py` modified +0/-3
  - `vllm/model_executor/models/bailing_moe_linear.py` modified +15/-13; symbols: BailingMoELinearAttention
- Code diff details:
  - `vllm/model_executor/layers/mamba/mamba_utils.py` modified +0/-3
  - `vllm/model_executor/models/bailing_moe_linear.py` modified +15/-13
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/layers/mamba/mamba_utils.py` modified +0/-3; `vllm/model_executor/models/bailing_moe_linear.py` modified +15/-13
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #41185 - [Bugfix] BailingMoeV2.5: rotate full qk_rope_head_dim in MLA RoPE

- Link: https://github.com/vllm-project/vllm/pull/41185
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-2, with 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] BailingMoeV2.5: rotate full qk_rope_head_dim in MLA RoPE"; model line: Ring 2.5 1T; category: bug fix; main diff: `vllm/model_executor/models/bailing_moe_linear.py`.
- Key implementation:
  - `vllm/model_executor/models/bailing_moe_linear.py` modified +8/-2
- Code diff details:
  - `vllm/model_executor/models/bailing_moe_linear.py` modified +8/-2
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/model_executor/models/bailing_moe_linear.py` modified +8/-2
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #41188 - [Misc] Replace mamba_type string literals with MambaAttentionBackendEnum

- Link: https://github.com/vllm-project/vllm/pull/41188
- Status/date: merged / 2026-05-11
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 18 files, +64/-58, with 404 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Replace mamba_type string literals with MambaAttentionBackendEnum"; model line: Ring 2.5 1T; category: docs/tests/CI; main diff: `docs/contributing/model/basic.md`, `tests/kernels/mamba/test_ssu_dispatch.py`, `tests/v1/attention/test_attention_backends_selection.py`.
- Key implementation:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/kernels/mamba/test_ssu_dispatch.py` modified +10/-2; symbols: _kv_cache_config_with_ssu
  - `tests/v1/attention/test_attention_backends_selection.py` modified +13/-8
  - `vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py` modified +2/-1
- Code diff details:
  - `docs/contributing/model/basic.md` modified +1/-1
  - `tests/kernels/mamba/test_ssu_dispatch.py` modified +10/-2
  - `tests/v1/attention/test_attention_backends_selection.py` modified +13/-8
  - `vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py` modified +2/-1
- Key code excerpts:

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
- Reviewed files:
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py` modified +2/-1; `vllm/model_executor/layers/kda.py` modified +3/-2; `vllm/model_executor/layers/mamba/abstract.py` modified +2/-1; `vllm/model_executor/layers/mamba/gdn_linear_attn.py` modified +3/-2; `vllm/model_executor/layers/mamba/linear_attn.py` modified +3/-2; `vllm/model_executor/layers/mamba/mamba_mixer.py` modified +3/-2; `vllm/model_executor/layers/mamba/mamba_mixer2.py` modified +3/-2; `vllm/model_executor/layers/mamba/ops/ssu_dispatch.py` modified +3/-1
  - tests: `tests/kernels/mamba/test_ssu_dispatch.py` modified +10/-2; `tests/v1/attention/test_attention_backends_selection.py` modified +13/-8
  - docs/bench: `docs/contributing/model/basic.md` modified +1/-1
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.
