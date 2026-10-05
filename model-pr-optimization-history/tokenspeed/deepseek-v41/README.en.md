# TokenSpeed DeepSeek V4.1 Model PR Optimization History


## 2026-10-05 Manual Diff Review

[Four reviewed launch/JIT optimizations](reviewed-2026-10-05.en.md) cover
#1663, #1685, #1794 and #1814 with complete diff scope, concrete rationale,
code excerpts and validation implications. Other generated entries below
remain discovery inventories pending manual review.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/tokenspeed/runtime/configs/deepseek_v41_config.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567) |
| `python/tokenspeed/runtime/configs/deepseek_v4_config.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/execution/drafter/deepseek_v41_dspark.py` | [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) |
| `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` | [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) |
| `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` | [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594) |
| `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1623](https://github.com/lightseekorg/tokenspeed/pull/1623), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660), [#1663](https://github.com/lightseekorg/tokenspeed/pull/1663), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) |
| `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636) |
| `python/tokenspeed/runtime/layers/attention/deepseek_v4/draft_rounds.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/layers/attention/deepseek_v4/graph_buffers.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/layers/attention/deepseek_v4/slot_mappings.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636) |
| `python/tokenspeed/runtime/layers/attention/deepseek_v4_geometry.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664) |
| `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1611](https://github.com/lightseekorg/tokenspeed/pull/1611), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636) |
| `python/tokenspeed/runtime/models/deepseek_v4.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664) |
| `python/tokenspeed/runtime/models/deepseek_v41.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863), [#1971](https://github.com/lightseekorg/tokenspeed/pull/1971) |
| `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686) |
| `python/tokenspeed/runtime/models/deepseek_v41_engram.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1971](https://github.com/lightseekorg/tokenspeed/pull/1971) |
| `python/tokenspeed/runtime/models/deepseek_v41_vision.py` | [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567), [#1844](https://github.com/lightseekorg/tokenspeed/pull/1844) |
| `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` | [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686) |
| `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/__init__.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/attention.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py` | [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686) |
| `python/tokenspeed/runtime/models/deepseek_v4_next.py` | no direct PR-number commit |
| `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh` | no direct PR-number commit |
| `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/build_swe_smith_dataset.patch` | no direct PR-number commit |
| `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/collect_outputs.py` | no direct PR-number commit |
| `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh` | no direct PR-number commit |
| `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_tp4.sh` | no direct PR-number commit |
| `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/deepseek_v4_tokenizer.py` | no direct PR-number commit |
| `test/ci/eval/deepseek-v4-flash-dcp4-mtp-evalscope-gsm8k.yaml` | no direct PR-number commit |
| `test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml` | no direct PR-number commit |
| `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k-amd.yaml` | no direct PR-number commit |
| `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml` | no direct PR-number commit |
| `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml` | [#1748](https://github.com/lightseekorg/tokenspeed/pull/1748) |
| `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml` | [#1748](https://github.com/lightseekorg/tokenspeed/pull/1748) |
| `test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml` | [#1603](https://github.com/lightseekorg/tokenspeed/pull/1603) |
| `test/ci/ut/deepseek-v4.1-flash-pd-1p1d.yaml` | [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595) |
| `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh` | [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1603](https://github.com/lightseekorg/tokenspeed/pull/1603) |
| `test/ci_system/test_deepseek_v41_pd_launcher.py` | [#1603](https://github.com/lightseekorg/tokenspeed/pull/1603) |
| `test/cli/test_serve_smg_deepseek_v41.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567) |
| `test/runtime/distributed/test_deepseek_v41_pd_1p1d.py` | [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595) |
| `test/runtime/run_deepseek_v41_eval.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552) |
| `test/runtime/test_deepseek_v41_cache.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552), [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1611](https://github.com/lightseekorg/tokenspeed/pull/1611), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660), [#1663](https://github.com/lightseekorg/tokenspeed/pull/1663), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) |
| `test/runtime/test_deepseek_v41_config.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552), [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660) |
| `test/runtime/test_deepseek_v41_dspark.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552), [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) |
| `test/runtime/test_deepseek_v41_engram.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1971](https://github.com/lightseekorg/tokenspeed/pull/1971) |
| `test/runtime/test_deepseek_v41_eval.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552) |
| `test/runtime/test_deepseek_v41_inputs.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) |
| `test/runtime/test_deepseek_v41_model.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552), [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863), [#1971](https://github.com/lightseekorg/tokenspeed/pull/1971) |
| `test/runtime/test_deepseek_v41_tokenizer.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549) |
| `test/runtime/test_deepseek_v41_vision.py` | [#1844](https://github.com/lightseekorg/tokenspeed/pull/1844) |
| `test/runtime/test_deepseek_v4_config.py` | [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686) |
| `test/runtime/test_deepseek_v4_hopper_fp8.py` | no direct PR-number commit |
| `test/runtime/test_deepseek_v4_mega_moe.py` | no direct PR-number commit |
| `test/runtime/test_deepseek_v4_mtp_prefix_cache.py` | no direct PR-number commit |
| `test/runtime/test_deepseek_v4_slot_mappings.py` | no direct PR-number commit |
| `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/__init__.py` | [#1619](https://github.com/lightseekorg/tokenspeed/pull/1619), [#1623](https://github.com/lightseekorg/tokenspeed/pull/1623) |
| `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/indexer.py` | [#1623](https://github.com/lightseekorg/tokenspeed/pull/1623) |
| `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/selected.py` | [#1619](https://github.com/lightseekorg/tokenspeed/pull/1619) |
| `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/__init__.py` | [#1619](https://github.com/lightseekorg/tokenspeed/pull/1619), [#1623](https://github.com/lightseekorg/tokenspeed/pull/1623) |
| `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/indexer.py` | [#1623](https://github.com/lightseekorg/tokenspeed/pull/1623) |
| `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/selected.py` | [#1619](https://github.com/lightseekorg/tokenspeed/pull/1619) |
| `tokenspeed-kernel/benchmarks/amd/gfx950/dsv41_flash/gemm.json` | [#1742](https://github.com/lightseekorg/tokenspeed/pull/1742) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/__init__.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1619](https://github.com/lightseekorg/tokenspeed/pull/1619), [#1623](https://github.com/lightseekorg/tokenspeed/pull/1623), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_cute_dsl/__init__.py` | [#1689](https://github.com/lightseekorg/tokenspeed/pull/1689) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_cute_dsl/sparse_index_scores.py` | [#1689](https://github.com/lightseekorg/tokenspeed/pull/1689) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_gluon/__init__.py` | [#1623](https://github.com/lightseekorg/tokenspeed/pull/1623) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_gluon/indexer.py` | [#1623](https://github.com/lightseekorg/tokenspeed/pull/1623) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cute_dsl.py` | [#1689](https://github.com/lightseekorg/tokenspeed/pull/1689), [#1719](https://github.com/lightseekorg/tokenspeed/pull/1719) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1683](https://github.com/lightseekorg/tokenspeed/pull/1683), [#1685](https://github.com/lightseekorg/tokenspeed/pull/1685), [#1689](https://github.com/lightseekorg/tokenspeed/pull/1689) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_select.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/flash_mla.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/gluon.py` | [#1619](https://github.com/lightseekorg/tokenspeed/pull/1619), [#1623](https://github.com/lightseekorg/tokenspeed/pull/1623) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1580](https://github.com/lightseekorg/tokenspeed/pull/1580), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1683](https://github.com/lightseekorg/tokenspeed/pull/1683), [#1685](https://github.com/lightseekorg/tokenspeed/pull/1685), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1794](https://github.com/lightseekorg/tokenspeed/pull/1794), [#1814](https://github.com/lightseekorg/tokenspeed/pull/1814), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863), [#1868](https://github.com/lightseekorg/tokenspeed/pull/1868) |
| `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_index_topk.py` | [#1623](https://github.com/lightseekorg/tokenspeed/pull/1623) |
| ... | 5 more files omitted from table; all were used for git tracing. |

## PR Coverage Summary

- Git-traced PRs: 28
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 28
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-09-15 | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549) | merged | feat(model): support basic deepseek v4.1 flash | `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_engram.py` |
| 2026-09-15 | [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552) | merged | feat(model): support basic deepseek v4.1 flash for amd | `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `test/runtime/run_deepseek_v41_eval.py`, `test/runtime/test_deepseek_v41_eval.py` |
| 2026-09-15 | [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567) | merged | feat(dsv41): support vision inputs | `python/tokenspeed/runtime/models/deepseek_v41_vision.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/configs/deepseek_v41_config.py` |
| 2026-09-15 | [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565) | merged | feat: enable pcg for DeepSeek V4.1 Flash | `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `test/runtime/test_deepseek_v41_model.py` |
| 2026-09-16 | [#1580](https://github.com/lightseekorg/tokenspeed/pull/1580) | merged | ci(dsv41): add deepseek v4.1 flash eval | `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py`, `tokenspeed-kernel/test/ops/test_attention_dsv41.py` |
| 2026-09-16 | [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594) | merged | feat(v41): SWA bounded replay and CED decoder narrowing | `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` |
| 2026-09-16 | [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595) | merged | feat(dsv41): support basic prefill/decode disaggregation with DSpark for DeepSeek V4.1 Flash | `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` |
| 2026-09-17 | [#1603](https://github.com/lightseekorg/tokenspeed/pull/1603) | merged | ci: cover DeepSeek V4.1 Flash PD on GB300 | `test/ci_system/test_deepseek_v41_pd_launcher.py`, `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh`, `test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml` |
| 2026-09-17 | [#1611](https://github.com/lightseekorg/tokenspeed/pull/1611) | merged | refactor(dsv41): retain exactly the attention window and the compressor pair | `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py`, `test/runtime/test_deepseek_v41_cache.py` |
| 2026-09-17 | [#1619](https://github.com/lightseekorg/tokenspeed/pull/1619) | merged | feat(amd): add gluon DSV4.1 selected attention and dense prefill | `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/selected.py`, `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/selected.py`, `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_selected.py` |
| 2026-09-18 | [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636) | merged | perf(dsv41): Optimize DeepSeek V4.1-flash on H20 | `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` |
| 2026-09-19 | [#1623](https://github.com/lightseekorg/tokenspeed/pull/1623) | merged | feat(amd): add gluon DSV4.1 CSA2 indexer | `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/indexer.py`, `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_index_topk.py` |
| 2026-09-20 | [#1663](https://github.com/lightseekorg/tokenspeed/pull/1663) | merged | perf(dsv41): share the native decode schedule across layers in a forward | `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `test/runtime/test_deepseek_v41_cache.py` |
| 2026-09-20 | [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660) | merged | perf(dsv41): run the DSpark draft window attention on selected_attention | `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` |
| 2026-09-21 | [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664) | merged | perf(moe): FlashInfer CUTLASS MXFP4 experts on Hopper (W4A16 auto, W4A8 opt-in) + V4.1 log fixes | `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_engram.py`, `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-09-21 | [#1683](https://github.com/lightseekorg/tokenspeed/pull/1683) | merged | perf(dsv41): select Reindex rows from the compacted candidate pool | `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py`, `tokenspeed-kernel/test/ops/test_attention_dsv41.py` |
| 2026-09-21 | [#1685](https://github.com/lightseekorg/tokenspeed/pull/1685) | merged | perf(dsv41): fuse the FP8 index-query quantization into one kernel | `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py`, `tokenspeed-kernel/test/ops/test_attention_dsv41.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py` |
| 2026-09-21 | [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686) | merged | perf(dsv41): fuse the DSpark decode prologue, draft tail and sampler sync into a handful of launches | `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py`, `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` |
| 2026-09-22 | [#1689](https://github.com/lightseekorg/tokenspeed/pull/1689) | merged | perf(dsv41): score only the candidate pool on Hopper with a CuTe DSL sparse indexer | `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_cute_dsl/sparse_index_scores.py`, `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cute_dsl.py` |
| 2026-09-22 | [#1719](https://github.com/lightseekorg/tokenspeed/pull/1719) | merged | fix(dsv41): stop recompiling the sparse index scorer per prefill chunk | `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cute_dsl.py` |
| 2026-09-23 | [#1742](https://github.com/lightseekorg/tokenspeed/pull/1742) | merged | ci(amd-kernel): Group DeepSeek V4.1 MXFP8 GEMM benchmarks by model | `tokenspeed-kernel/benchmarks/amd/gfx950/dsv41_flash/gemm.json` |
| 2026-09-25 | [#1748](https://github.com/lightseekorg/tokenspeed/pull/1748) | merged | ci: turn on dspark for dsv4.1 ci | `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml`, `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml` |
| 2026-09-25 | [#1794](https://github.com/lightseekorg/tokenspeed/pull/1794) | merged | perf(dsv41): stop recompiling compressor_metadata on every prefill chunk | `tokenspeed-kernel/test/ops/test_attention_dsv41.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` |
| 2026-09-27 | [#1814](https://github.com/lightseekorg/tokenspeed/pull/1814) | merged | perf(dsv41): stop recompiling compressor_pool for every prefill length | `tokenspeed-kernel/test/ops/test_attention_dsv41.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` |
| 2026-09-28 | [#1844](https://github.com/lightseekorg/tokenspeed/pull/1844) | merged | feat(dsv41): support vit batching | `python/tokenspeed/runtime/models/deepseek_v41_vision.py`, `test/runtime/test_deepseek_v41_vision.py` |
| 2026-09-29 | [#1868](https://github.com/lightseekorg/tokenspeed/pull/1868) | merged | fix(kernel): stop dsv41 address kernels recompiling on page-table geometry | `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` |
| 2026-09-30 | [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) | merged | perf(dsv41): skip decoder work for incomplete prefill chunks | `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `test/runtime/test_deepseek_v41_model.py` |
| 2026-10-04 | [#1971](https://github.com/lightseekorg/tokenspeed/pull/1971) | merged | refactor(linear): select V4.1 FP8 methods during construction | `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_engram.py`, `test/runtime/test_deepseek_v41_model.py` |

## Per-PR Diff Audit Cards

### PR #1549 - feat(model): support basic deepseek v4.1 flash

- Link: https://github.com/lightseekorg/tokenspeed/pull/1549
- Status/date: merged / 2026-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/configs/deepseek_v41_config.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` and 29 files; associated commits `eaa58e30be06`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 87 files, +18837/-99, 19915 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/deepseek_v41.py` added +1645/-0 (1645 lines); hunks: -0,0 +1,1645; symbols: v41_quantize_fp8, v41_mxfp8_config, _ReferenceFp8LinearMethod, apply, touching `v41_quantize_fp8, v41_mxfp8_config, _ReferenceFp8LinearMethod`; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` added +1413/-0 (1413 lines); hunks: -0,0 +1,1413; symbols: V41CompressorPlan, allocate, window, V41Metadata, touching `V41CompressorPlan, allocate, window`; `python/tokenspeed/runtime/models/deepseek_v41_engram.py` added +892/-0 (892 lines); hunks: -0,0 +1,892; symbols: is_engram_embed_checkpoint_name, build_compressed_token_map, compute_hash_multipliers, EngramLayout, touching `is_engram_embed_checkpoint_name, build_compressed_token_map, compute_hash_multipliers`; `test/runtime/test_deepseek_v41_tokenizer.py` added +468/-0 (468 lines); hunks: -0,0 +1,468; symbols: base_tokenizer, test_effort_is_forwarded_without_v4_filtering, test_tokenization_has_one_bos_and_preserves_backend, test_media_is_rejected_before_encoding, touching `base_tokenizer, test_effort_is_forwarded_without_v4_filtering, test_tokenization_has_one_bos_and_preserves_backend`.
- Code diff details:
  - `python/tokenspeed/runtime/models/deepseek_v41.py` added +1645/-0 (1645 lines); hunks: -0,0 +1,1645; symbols: v41_quantize_fp8, v41_mxfp8_config, _ReferenceFp8LinearMethod, apply
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` added +1413/-0 (1413 lines); hunks: -0,0 +1,1413; symbols: V41CompressorPlan, allocate, window, V41Metadata
  - `python/tokenspeed/runtime/models/deepseek_v41_engram.py` added +892/-0 (892 lines); hunks: -0,0 +1,892; symbols: is_engram_embed_checkpoint_name, build_compressed_token_map, compute_hash_multipliers, EngramLayout
  - `test/runtime/test_deepseek_v41_tokenizer.py` added +468/-0 (468 lines); hunks: -0,0 +1,468; symbols: base_tokenizer, test_effort_is_forwarded_without_v4_filtering, test_tokenization_has_one_bos_and_preserves_backend, test_media_is_rejected_before_encoding
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` added +353/-0 (353 lines); hunks: -0,0 +1,353; symbols: _quantized_kv, _WindowAttention, __init__, query_metadata
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v41.py
@@ -0,0 +1,1645 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py
@@ -0,0 +1,1413 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/models/deepseek_v41_engram.py
@@ -0,0 +1,892 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41.py` added +1645/-0; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` added +1413/-0; `python/tokenspeed/runtime/models/deepseek_v41_engram.py` added +892/-0; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` added +353/-0; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` added +280/-0; `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` added +126/-0
  - tests: `test/runtime/test_deepseek_v41_tokenizer.py` added +468/-0
- Risk and verification: The diff ships test coverage in `test/cli/test_logprefix.py`, `test/cli/test_serve_smg_deepseek_v41.py`, `test/runtime/run_deepseek_v41_eval.py`, `test/runtime/test_cache_pd_shutdown.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1552 - feat(model): support basic deepseek v4.1 flash for amd

- Link: https://github.com/lightseekorg/tokenspeed/pull/1552
- Status/date: merged / 2026-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `test/runtime/run_deepseek_v41_eval.py`, `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_config.py`, `test/runtime/test_deepseek_v41_dspark.py` and 7 files; associated commits `df3fe8aefef7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +61/-21, 228 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +6/-1 (7 lines); hunks: -34,6 +34,7; -95,11 +96,14 @@ def forward_v41(; symbols: forward_v41, _main_input, _main_kv, touching `forward_v41, _main_input, _main_kv`; `test/runtime/run_deepseek_v41_eval.py` modified +8/-3 (11 lines); hunks: -21,8 +21,9; -131,6 +132,8 @@ def main():; symbols: main, touching `main`; `test/runtime/test_deepseek_v41_eval.py` modified +8/-0 (8 lines); hunks: -258,6 +258,10 @@ def main_env(tmp_path, report, monkeypatch):; -347,6 +351,10 @@ def test_full_eval_writes_result_only_after_completion(; symbols: main_env, test_full_eval_writes_result_only_after_completion, touching `main_env, test_full_eval_writes_result_only_after_completion`; `test/runtime/test_deepseek_v41_config.py` modified +3/-2 (5 lines); hunks: -213,6 +213,7 @@ def test_registry_and_wrapper_roundtrip(config_dir, tmp_path):; -343,7 +344,7 @@ def test_config_selects_flash_recipe_and_checks_geometry(run...; symbols: test_registry_and_wrapper_roundtrip, test_config_selects_flash_recipe_and_checks_geometry, test_real_server_args_prepare_cache_pool_and_backend, touching `test_registry_and_wrapper_roundtrip, test_config_selects_flash_recipe_and_checks_geometry, test_real_server_args_prepare_cache_pool_and_backend`.
- Code diff details:
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +6/-1 (7 lines); hunks: -34,6 +34,7; -95,11 +96,14 @@ def forward_v41(; symbols: forward_v41, _main_input, _main_kv
  - `test/runtime/run_deepseek_v41_eval.py` modified +8/-3 (11 lines); hunks: -21,8 +21,9; -131,6 +132,8 @@ def main():; symbols: main
  - `test/runtime/test_deepseek_v41_eval.py` modified +8/-0 (8 lines); hunks: -258,6 +258,10 @@ def main_env(tmp_path, report, monkeypatch):; -347,6 +351,10 @@ def test_full_eval_writes_result_only_after_completion(; symbols: main_env, test_full_eval_writes_result_only_after_completion
  - `test/runtime/test_deepseek_v41_config.py` modified +3/-2 (5 lines); hunks: -213,6 +213,7 @@ def test_registry_and_wrapper_roundtrip(config_dir, tmp_path):; -343,7 +344,7 @@ def test_config_selects_flash_recipe_and_checks_geometry(run...; symbols: test_registry_and_wrapper_roundtrip, test_config_selects_flash_recipe_and_checks_geometry, test_real_server_args_prepare_cache_pool_and_backend
  - `test/runtime/test_deepseek_v41_cache.py` modified +3/-1 (4 lines); hunks: -892,6 +892,7 @@ def test_recipe_exact_geometry_capacity_and_dispatch():; -901,7 +902,8 @@ def test_recipe_exact_geometry_capacity_and_dispatch():; symbols: test_recipe_exact_geometry_capacity_and_dispatch, test_owner_topology_and_reject_invalid_recipes
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v41_dspark.py
@@ -34,6 +34,7 @@
+from tokenspeed_kernel.ops.attention.dsv41 import rope_inplace
@@ -95,11 +96,14 @@ def forward_v41(
+        swa_rope_cache,
+        if swa_rope_cache is not None:
+            swa = rope_inplace(swa.clone(), positions, swa_rope_cache, None)
@@ -221,7 +225,8 @@ def _main_input(self, captured):
diff -- test/runtime/run_deepseek_v41_eval.py
@@ -21,8 +21,9 @@
-All paths/ports/time limits, --execution-mode eager|graph, --batch-size,
+All paths/ports/time limits, --execution-mode eager|graph, --moe-backend,
+--reasoning-parser, --batch-size, --max-total-tokens, --max-model-len and
+--chunked-prefill-size are explicit;
@@ -131,6 +132,8 @@ def main():
+    parser.add_argument("--moe-backend", required=True)
diff -- test/runtime/test_deepseek_v41_eval.py
@@ -258,6 +258,10 @@ def main_env(tmp_path, report, monkeypatch):
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +6/-1
  - tests: `test/runtime/run_deepseek_v41_eval.py` modified +8/-3; `test/runtime/test_deepseek_v41_eval.py` modified +8/-0; `test/runtime/test_deepseek_v41_config.py` modified +3/-2; `test/runtime/test_deepseek_v41_cache.py` modified +3/-1; `test/runtime/test_deepseek_v41_dspark.py` modified +1/-0; `test/runtime/test_deepseek_v41_model.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/runtime/run_deepseek_v41_eval.py`, `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_config.py`, `test/runtime/test_deepseek_v41_dspark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1567 - feat(dsv41): support vision inputs

- Link: https://github.com/lightseekorg/tokenspeed/pull/1567
- Status/date: merged / 2026-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/configs/deepseek_v41_config.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/models/deepseek_v41_vision.py`, `test/cli/test_serve_smg_deepseek_v41.py` and 8 files; associated commits `cb05b7cf2574`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 12 files, +811/-276, 1658 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/deepseek_v41_vision.py` added +300/-0 (300 lines); hunks: -0,0 +1,300; symbols: get_vision_cos_sin, apply_vision_rotary, DeepseekV41VisionPatchEmbed, __init__, touching `get_vision_cos_sin, apply_vision_rotary, DeepseekV41VisionPatchEmbed`; `python/tokenspeed/runtime/models/deepseek_v41.py` modified +172/-26 (198 lines); hunks: -18,7 +18,7; -64,6 +64,7; symbols: __init__, _select_experts, touching `__init__, _select_experts`; `python/tokenspeed/runtime/configs/deepseek_v41_config.py` modified +53/-14 (67 lines); hunks: -18,7 +18,7; -64,13 +64,55 @@ def ngram_context_len(self):; symbols: ngram_context_len, DeepseekV41VisionConfig, __init__, DeepseekV41Config, touching `ngram_context_len, DeepseekV41VisionConfig, __init__`; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +16/-7 (23 lines); hunks: -278,8 +278,8 @@ def forward_backbone(; -289,20 +289,29 @@ def forward_backbone(; symbols: forward_backbone, DeepseekV41ForCausalLMDSpark, __init__, resolve_model, touching `forward_backbone, DeepseekV41ForCausalLMDSpark, __init__`.
- Code diff details:
  - `python/tokenspeed/runtime/models/deepseek_v41_vision.py` added +300/-0 (300 lines); hunks: -0,0 +1,300; symbols: get_vision_cos_sin, apply_vision_rotary, DeepseekV41VisionPatchEmbed, __init__
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +172/-26 (198 lines); hunks: -18,7 +18,7; -64,6 +64,7; symbols: __init__, _select_experts
  - `python/tokenspeed/runtime/configs/deepseek_v41_config.py` modified +53/-14 (67 lines); hunks: -18,7 +18,7; -64,13 +64,55 @@ def ngram_context_len(self):; symbols: ngram_context_len, DeepseekV41VisionConfig, __init__, DeepseekV41Config
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +16/-7 (23 lines); hunks: -278,8 +278,8 @@ def forward_backbone(; -289,20 +289,29 @@ def forward_backbone(; symbols: forward_backbone, DeepseekV41ForCausalLMDSpark, __init__, resolve_model
  - `test/cli/test_serve_smg_deepseek_v41.py` modified +88/-82 (170 lines); hunks: -77,8 +77,9 @@ def GetModelInfo(self, request, context):; -91,6 +92,7 @@ def GetLoads(self, request, context):; symbols: GetModelInfo, HealthCheck, GetLoads, Generate
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v41_vision.py
@@ -0,0 +1,300 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/models/deepseek_v41.py
@@ -18,7 +18,7 @@
-"""V4.1 full-prompt backbone: FlatKV attention and single-pass hyperconnections.
+"""V4.1 multimodal model with FlatKV attention and single-pass hyperconnections.
@@ -64,6 +64,7 @@
+from tokenspeed.runtime.configs.deepseek_v41_config import DeepseekV41Config
@@ -87,6 +88,7 @@
+from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype
diff -- python/tokenspeed/runtime/configs/deepseek_v41_config.py
@@ -18,7 +18,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41_vision.py` added +300/-0; `python/tokenspeed/runtime/models/deepseek_v41.py` modified +172/-26; `python/tokenspeed/runtime/configs/deepseek_v41_config.py` modified +53/-14; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +16/-7
  - tests: `test/cli/test_serve_smg_deepseek_v41.py` modified +88/-82; `test/runtime/test_deepseek_v41_model.py` modified +112/-8; `test/runtime/test_deepseek_v41_config.py` modified +49/-4; `test/runtime/test_deepseek_v41_dspark.py` modified +5/-2
- Risk and verification: The diff ships test coverage in `test/cli/test_serve_smg_deepseek_v41.py`, `test/runtime/test_deepseek_v41_config.py`, `test/runtime/test_deepseek_v41_dspark.py`, `test/runtime/test_deepseek_v41_model.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1565 - feat: enable pcg for DeepSeek V4.1 Flash

- Link: https://github.com/lightseekorg/tokenspeed/pull/1565
- Status/date: merged / 2026-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_config.py` and 7 files; associated commits `fcbfd6467d89`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +328/-27, 531 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/deepseek_v41.py` modified +12/-0 (12 lines); hunks: -68,6 +68,11; -707,13 +712,20 @@ def _kernel_attn_sink(self):; symbols: _kernel_attn_sink, forward, touching `_kernel_attn_sink, forward`; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +2/-2 (4 lines); hunks: -149,8 +149,8 @@ class V41SWAQueryPlan:; symbols: V41SWAQueryPlan, DeepseekV41AttentionBackend, __init__, touching `V41SWAQueryPlan, DeepseekV41AttentionBackend, __init__`; `test/runtime/test_deepseek_v41_model.py` modified +119/-3 (122 lines); hunks: -774,8 +774,8 @@ def test_cuda_exact_fp8_linear_and_engram_method(monkeypatch):; -840,14 +840,130 @@ def test_cuda_40_layer_real_flatkv_and_moe(monkeypatch, t...; symbols: test_cuda_exact_fp8_linear_and_engram_method, test_cuda_40_layer_real_flatkv_and_moe, _assert_prefill_graph_matches_eager, touching `test_cuda_exact_fp8_linear_and_engram_method, test_cuda_40_layer_real_flatkv_and_moe, _assert_prefill_graph_matches_eager`; `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` modified +80/-15 (95 lines); hunks: -34,6 +34,7; -44,6 +45,9; symbols: _dspark_decode_position_plan, __init__, _bonus_tokens_from_output, capture_prefill_graph, touching `_dspark_decode_position_plan, __init__, _bonus_tokens_from_output`.
- Code diff details:
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +12/-0 (12 lines); hunks: -68,6 +68,11; -707,13 +712,20 @@ def _kernel_attn_sink(self):; symbols: _kernel_attn_sink, forward
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +2/-2 (4 lines); hunks: -149,8 +149,8 @@ class V41SWAQueryPlan:; symbols: V41SWAQueryPlan, DeepseekV41AttentionBackend, __init__
  - `test/runtime/test_deepseek_v41_model.py` modified +119/-3 (122 lines); hunks: -774,8 +774,8 @@ def test_cuda_exact_fp8_linear_and_engram_method(monkeypatch):; -840,14 +840,130 @@ def test_cuda_40_layer_real_flatkv_and_moe(monkeypatch, t...; symbols: test_cuda_exact_fp8_linear_and_engram_method, test_cuda_40_layer_real_flatkv_and_moe, _assert_prefill_graph_matches_eager
  - `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` modified +80/-15 (95 lines); hunks: -34,6 +34,7; -44,6 +45,9; symbols: _dspark_decode_position_plan, __init__, _bonus_tokens_from_output, capture_prefill_graph
  - `test/runtime/test_deepseek_v41_dspark.py` modified +68/-0 (68 lines); hunks: -260,6 +260,7 @@ def test_window_attention_matches_dense_reference():; -304,6 +305,7 @@ def test_draft_forward_graph_and_context_seeding(monkeypatch):; symbols: test_window_attention_matches_dense_reference, test_draft_forward_graph_and_context_seeding, forward, _assert_prefill_graph_matches_eager
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v41.py
@@ -68,6 +68,11 @@
+from tokenspeed.runtime.execution.breakable_cuda_graph import (
+    break_point,
+    current_forward_ctx,
+    slice_to_real_tokens,
+)
@@ -707,13 +712,20 @@ def _kernel_attn_sink(self):
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py
@@ -149,8 +149,8 @@ class V41SWAQueryPlan:
-    # Arbitrary-chunk/mixed prefill still needs host-side dependency checks.
-    cuda_graph_support = CudaGraphSupport(decode_graph=True, prefill_graph=False)
+    # Breakable prefill keeps attention and its host-side checks eager.
+    cuda_graph_support = CudaGraphSupport(decode_graph=True, prefill_graph=True)
diff -- test/runtime/test_deepseek_v41_model.py
@@ -774,8 +774,8 @@ def test_cuda_exact_fp8_linear_and_engram_method(monkeypatch):
-@pytest.mark.parametrize("capture_decode", [False, True])
-def test_cuda_40_layer_real_flatkv_and_moe(monkeypatch, tmp_path, capture_decode):
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41.py` modified +12/-0; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +2/-2; `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` modified +80/-15
  - tests: `test/runtime/test_deepseek_v41_model.py` modified +119/-3; `test/runtime/test_deepseek_v41_dspark.py` modified +68/-0; `test/runtime/test_deepseek_v41_cache.py` modified +1/-1; `test/runtime/test_deepseek_v41_config.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_config.py`, `test/runtime/test_deepseek_v41_dspark.py`, `test/runtime/test_deepseek_v41_model.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1580 - ci(dsv41): add deepseek v4.1 flash eval

- Link: https://github.com/lightseekorg/tokenspeed/pull/1580
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py`, `tokenspeed-kernel/test/ops/test_attention_dsv41.py`; associated commits `688bbb04f5b4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +189/-56, 359 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +2/-9 (11 lines); hunks: -1075,15 +1075,8 @@ def _compressor_pool(; symbols: _compressor_pool, touching `_compressor_pool`; `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +8/-0 (8 lines); hunks: -730,6 +730,14 @@ def test_compressor_fused_norm_preserves_pooled_bf16_bounda...; symbols: test_compressor_fused_norm_preserves_pooled_bf16_boundary, touching `test_compressor_fused_norm_preserves_pooled_bf16_boundary`.
- Code diff details:
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +2/-9 (11 lines); hunks: -1075,15 +1075,8 @@ def _compressor_pool(; symbols: _compressor_pool
  - `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +8/-0 (8 lines); hunks: -730,6 +730,14 @@ def test_compressor_fused_norm_preserves_pooled_bf16_bounda...; symbols: test_compressor_fused_norm_preserves_pooled_bf16_boundary
- Key code excerpts:

```diff
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py
@@ -1075,15 +1075,8 @@ def _compressor_pool(
-    # Preserve separate FP32 products/addition before the caller's BF16 cast.
-    pooled = tl.inline_asm_elementwise(
-        "{ .reg .f32 a, b; mul.rn.f32 a, $1, $2; mul.rn.f32 b, $3, $4; add.rn.f32 $0, a, b; }",
-        constraints="=f,f,f,f,f",
-        args=[old_content, old_weight, current, new_weight],
-        dtype=tl.float32,
diff -- tokenspeed-kernel/test/ops/test_attention_dsv41.py
@@ -730,6 +730,14 @@ def test_compressor_fused_norm_preserves_pooled_bf16_boundary(device):
+    maximum = torch.maximum(scores[previous], scores)
+    old_exp = torch.exp(scores[previous] - maximum)
+    new_exp = torch.exp(scores - maximum)
+    expected_raw = (content[previous] * old_exp + content * new_exp) / (
+        old_exp + new_exp
+    )
```

- Extracted files (not manually reviewed):
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +2/-9
  - tests: `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +8/-0
- Risk and verification: The diff ships test coverage in `test/ci/eval/deepseek-v4.1-flash-evalscope-gsm8k-amd.yaml`, `test/ci/eval/deepseek-v4.1-flash-evalscope-gsm8k.yaml`, `test/ci_system/test_eval_configs.py`, `tokenspeed-kernel/test/ops/test_attention_dsv41.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1594 - feat(v41): SWA bounded replay and CED decoder narrowing

- Link: https://github.com/lightseekorg/tokenspeed/pull/1594
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41.py` and 11 files; associated commits `4986d149784c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 43 files, +1508/-290, 3246 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +296/-57 (353 lines); hunks: -29,10 +29,17; -108,6 +115,52 @@ class V41Metadata:; symbols: V41Metadata, V41PrefillSpan, V41RowPlan, V41DecoderView, touching `V41Metadata, V41PrefillSpan, V41RowPlan`; `python/tokenspeed/runtime/models/deepseek_v41.py` modified +125/-19 (144 lines); hunks: -37,9 +37,14; -51,6 +56,7; symbols: _kernel_attn_sink, forward, __init__, touching `_kernel_attn_sink, forward, __init__`; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +22/-2 (24 lines); hunks: -22,8 +22,12; -168,6 +172,21 @@ def add(; symbols: add, touching `add`; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` modified +8/-1 (9 lines); hunks: -47,7 +47,10; -910,13 +913,17 @@ def init_forward_metadata(; symbols: init_forward_metadata, touching `init_forward_metadata`.
- Code diff details:
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +296/-57 (353 lines); hunks: -29,10 +29,17; -108,6 +115,52 @@ class V41Metadata:; symbols: V41Metadata, V41PrefillSpan, V41RowPlan, V41DecoderView
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +125/-19 (144 lines); hunks: -37,9 +37,14; -51,6 +56,7; symbols: _kernel_attn_sink, forward, __init__
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +22/-2 (24 lines); hunks: -22,8 +22,12; -168,6 +172,21 @@ def add(; symbols: add
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` modified +8/-1 (9 lines); hunks: -47,7 +47,10; -910,13 +913,17 @@ def init_forward_metadata(; symbols: init_forward_metadata
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +4/-0 (4 lines); hunks: -37,6 +37,9; -280,6 +283,7 @@ def forward_backbone(; symbols: forward_backbone
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py
@@ -29,10 +29,17 @@
-ordinary full-query inputs from query_metadata(mode); CED may explicitly subset
-those positions. A source must select every query needed by its Reuse consumers.
-Compressor projections cover the full canonical query window: request spans have
-consecutive positions, so pair metadata is prepared once by direct row lookup.
+ordinary full-query inputs from query_metadata(mode); the CED decoder layers use
+decoder_view() instead, the per-request tail of the same rows. A source must
diff -- python/tokenspeed/runtime/models/deepseek_v41.py
@@ -37,9 +37,14 @@
-This baseline executes every scheduled token through every backbone layer. CED
-shortening, PP and CP are not implemented. DSpark captures layer inputs. Attention
-and MoE TPxEP widths must match to keep the HC stream replicated on attention TP.
+Every scheduled token runs through the encoder layers; the CED decoder (from
+the candidate source on) runs on the rows the backend's ``decoder_view()``
+keeps -- each prompt's last window (one row for a chunk that leaves its prompt
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py
@@ -22,8 +22,12 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +296/-57; `python/tokenspeed/runtime/models/deepseek_v41.py` modified +125/-19; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +22/-2; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` modified +8/-1; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +4/-0
  - tests: `test/runtime/test_deepseek_v41_model.py` modified +213/-111; `test/runtime/test_deepseek_v41_cache.py` modified +306/-17; `test/runtime/test_deepseek_v41_inputs.py` modified +53/-23
- Risk and verification: The diff ships test coverage in `test/runtime/test_cache_group_router.py`, `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_dspark.py`, `test/runtime/test_deepseek_v41_inputs.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1595 - feat(dsv41): support basic prefill/decode disaggregation with DSpark for DeepSeek V4.1 Flash

- Link: https://github.com/lightseekorg/tokenspeed/pull/1595
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/execution/drafter/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` and 13 files; associated commits `2dfc2c4dc468`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 24 files, +1318/-240, 2005 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +73/-27 (100 lines); hunks: -37,6 +37,8; -45,10 +47,12; symbols: DeepseekV41Recipe, layer_types, max_padding_fraction, dspark_stages, touching `DeepseekV41Recipe, layer_types, max_padding_fraction`; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +55/-32 (87 lines); hunks: -67,19 +67,45 @@ def _quantized_kv(x: torch.Tensor) -> torch.Tensor:; -103,19 +129,16 @@ def forward_v41(; symbols: _quantized_kv, _window_rows, _write_window_rows, _WindowAttention, touching `_quantized_kv, _window_rows, _write_window_rows`; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +17/-4 (21 lines); hunks: -447,15 +447,28 @@ def _prepare_compressor(self, metadata: V41Metadata) -> None:; -1534,7 +1547,7 @@ def forward_v41(; symbols: _prepare_compressor, _decode_window, window_slots, forward_v41, touching `_prepare_compressor, _decode_window, window_slots`; `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` modified +18/-0 (18 lines); hunks: -59,6 +59,24; symbols: v41_dspark_field_name, v41_layer_mapping, touching `v41_dspark_field_name, v41_layer_mapping`.
- Code diff details:
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +73/-27 (100 lines); hunks: -37,6 +37,8; -45,10 +47,12; symbols: DeepseekV41Recipe, layer_types, max_padding_fraction, dspark_stages
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +55/-32 (87 lines); hunks: -67,19 +67,45 @@ def _quantized_kv(x: torch.Tensor) -> torch.Tensor:; -103,19 +129,16 @@ def forward_v41(; symbols: _quantized_kv, _window_rows, _write_window_rows, _WindowAttention
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +17/-4 (21 lines); hunks: -447,15 +447,28 @@ def _prepare_compressor(self, metadata: V41Metadata) -> None:; -1534,7 +1547,7 @@ def forward_v41(; symbols: _prepare_compressor, _decode_window, window_slots, forward_v41
  - `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` modified +18/-0 (18 lines); hunks: -59,6 +59,24; symbols: v41_dspark_field_name, v41_layer_mapping
  - `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` modified +15/-0 (15 lines); hunks: -23,6 +23,9; -95,6 +98,18 @@ def compressor_tail(self, owner: int) -> torch.Tensor:; symbols: compressor_tail, zero_new_blocks, dspark_kv, get_key_buffer
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py
@@ -37,6 +37,8 @@
+    V41_DSPARK_GROUP_PACKING,
+    V41_DSPARK_LCM_BLOCK_BYTES,
@@ -45,10 +47,12 @@
+    V41_LCM_BLOCK_BYTES,
+    v41_dspark_field_name,
@@ -60,14 +64,18 @@
diff -- python/tokenspeed/runtime/models/deepseek_v41_dspark.py
@@ -67,19 +67,45 @@ def _quantized_kv(x: torch.Tensor) -> torch.Tensor:
+def _window_rows(window: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
+    """Gather ``[..., head_dim]`` rows of a paged window field by SWA slots.
+    Negative slots resolve to the null page's first row; callers mask them.
+    """
+    rows_per_page = window.shape[1]
+    slots = slots.clamp_min(0).long()
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py
@@ -447,15 +447,28 @@ def _prepare_compressor(self, metadata: V41Metadata) -> None:
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +73/-27; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +55/-32; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +17/-4; `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` modified +18/-0; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` modified +15/-0; `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py` modified +10/-0
  - tests: `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh` added +242/-0
- Risk and verification: The diff ships test coverage in `test/ci/ut/deepseek-v4.1-flash-pd-1p1d.yaml`, `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh`, `test/runtime/distributed/test_deepseek_v41_pd_1p1d.py`, `test/runtime/test_deepseek_v41_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1603 - ci: cover DeepSeek V4.1 Flash PD on GB300

- Link: https://github.com/lightseekorg/tokenspeed/pull/1603
- Status/date: merged / 2026-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml`, `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh`, `test/ci_system/test_deepseek_v41_pd_launcher.py`; associated commits `4438b65134fc`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +468/-19, 650 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci_system/test_deepseek_v41_pd_launcher.py` added +235/-0 (235 lines); hunks: -0,0 +1,235; symbols: launcher, start, calls, wait_for, touching `launcher, start, calls`; `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh` modified +129/-17 (146 lines); hunks: -1,7 +1,8; -24,6 +25,10 @@ MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}; symbols: main, touching `main`; `test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml` added +57/-0 (57 lines); hunks: -0,0 +1,57.
- Code diff details:
  - `test/ci_system/test_deepseek_v41_pd_launcher.py` added +235/-0 (235 lines); hunks: -0,0 +1,235; symbols: launcher, start, calls, wait_for
  - `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh` modified +129/-17 (146 lines); hunks: -1,7 +1,8; -24,6 +25,10 @@ MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}; symbols: main
  - `test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml` added +57/-0 (57 lines); hunks: -0,0 +1,57
- Key code excerpts:

```diff
diff -- test/ci_system/test_deepseek_v41_pd_launcher.py
@@ -0,0 +1,235 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh
@@ -1,7 +1,8 @@
-# DeepSeek V4.1 Flash PD (prefill-decode) 1P-1D topology on a single node, with
+# DeepSeek V4.1 Flash PD (prefill-decode) 1P-1D topology, with
+# PD_SLURM=1 places prefill + gateway on Slurm node 0 and decode on node 1.
@@ -24,6 +25,10 @@ MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
+MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-16}
+REASONING_PARSER=${REASONING_PARSER:-passthrough}
diff -- test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml
@@ -0,0 +1,57 @@
```

- Extracted files (not manually reviewed):
  - tests: `test/ci_system/test_deepseek_v41_pd_launcher.py` added +235/-0; `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh` modified +129/-17; `test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml` added +57/-0
- Risk and verification: The diff ships test coverage in `test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml`, `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh`, `test/ci_system/test_deepseek_v41_pd_launcher.py`, `test/ci_system/test_dispatch_workflows.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1611 - refactor(dsv41): retain exactly the attention window and the compressor pair

- Link: https://github.com/lightseekorg/tokenspeed/pull/1611
- Status/date: merged / 2026-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py`, `test/runtime/test_deepseek_v41_cache.py`; associated commits `fff486b06521`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +26/-27, 93 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +15/-17 (32 lines); hunks: -193,28 +193,26 @@ def add(; symbols: add, touching `add`; `test/runtime/test_deepseek_v41_cache.py` modified +8/-8 (16 lines); hunks: -1041,10 +1041,10 @@ def test_packed_config_and_recipe_capacity(verify_width,...; -1248,8 +1248,8 @@ def test_recipe_exact_geometry_capacity_and_dispatch():; symbols: test_packed_config_and_recipe_capacity, test_recipe_exact_geometry_capacity_and_dispatch, test_recipe_declares_replay_windows_for_the_private_groups, touching `test_packed_config_and_recipe_capacity, test_recipe_exact_geometry_capacity_and_dispatch, test_recipe_declares_replay_windows_for_the_private_groups`.
- Code diff details:
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +15/-17 (32 lines); hunks: -193,28 +193,26 @@ def add(; symbols: add
  - `test/runtime/test_deepseek_v41_cache.py` modified +8/-8 (16 lines); hunks: -1041,10 +1041,10 @@ def test_packed_config_and_recipe_capacity(verify_width,...; -1248,8 +1248,8 @@ def test_recipe_exact_geometry_capacity_and_dispatch():; symbols: test_packed_config_and_recipe_capacity, test_recipe_exact_geometry_capacity_and_dispatch, test_recipe_declares_replay_windows_for_the_private_groups
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py
@@ -193,28 +193,26 @@ def add(
-        # Admission can be ahead of the completed forward. Retain its input
-        # window as well as the unfinished pair; the allocator also reserves
-        # in-flight pages through the shared scheduler_limits demand formula.
-        # The window is part of the PD peer contract, so size it for the
-        # deepest schedule (depth 1) on every role: a prefill node runs without
-        # overlap, and its pages must land in a decode node's retention.
diff -- test/runtime/test_deepseek_v41_cache.py
@@ -1041,10 +1041,10 @@ def test_packed_config_and_recipe_capacity(verify_width, overlap_depth):
-    # Retention is sized for the deepest schedule on every role so that a
-    # prefill node (no overlap) and a decode node agree on the PD contract.
-    assert specs[SWA].sliding_window_tokens == 128 + 2 * verify_width
-    assert specs[TAIL].sliding_window_tokens == 2 + 2 * verify_width
+    # Retention is the attention window (or the pair) whatever the verify
+    # width or schedule depth: the scheduled rows are reserved, not retained.
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +15/-17
  - tests: `test/runtime/test_deepseek_v41_cache.py` modified +8/-8
- Risk and verification: The diff ships test coverage in `test/runtime/test_deepseek_v41_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1619 - feat(amd): add gluon DSV4.1 selected attention and dense prefill

- Link: https://github.com/lightseekorg/tokenspeed/pull/1619
- Status/date: merged / 2026-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/__init__.py`, `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/selected.py`, `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/__init__.py`, `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/selected.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/__init__.py` and 7 files; associated commits `fbb3265cdd7e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 14 files, +2021/-85, 2161 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/selected.py` added +594/-0 (594 lines); hunks: -0,0 +1,594; symbols: _e2m1_decode, _planar_offset, _load_segment_slot, _load_v41_tile, touching `_e2m1_decode, _planar_offset, _load_segment_slot`; `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/selected.py` added +571/-0 (571 lines); hunks: -0,0 +1,571; symbols: _e2m1_decode, _planar_offset, _load_segment_slot, _load_v41_tile, touching `_e2m1_decode, _planar_offset, _load_segment_slot`; `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_selected.py` added +131/-0 (131 lines); hunks: -0,0 +1,131; symbols: _selected_name, test_fused_gluon_is_selected_on_supported_amd, _make_cache, test_fused_selected_attention_matches_gather_softmax, touching `_selected_name, test_fused_gluon_is_selected_on_supported_amd, _make_cache`; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/gluon.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: gluon_dsv41_selected_attention_gfx950, gluon_dsv41_selected_attention_gfx1250, touching `gluon_dsv41_selected_attention_gfx950, gluon_dsv41_selected_attention_gfx1250`.
- Code diff details:
  - `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/selected.py` added +594/-0 (594 lines); hunks: -0,0 +1,594; symbols: _e2m1_decode, _planar_offset, _load_segment_slot, _load_v41_tile
  - `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/selected.py` added +571/-0 (571 lines); hunks: -0,0 +1,571; symbols: _e2m1_decode, _planar_offset, _load_segment_slot, _load_v41_tile
  - `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_selected.py` added +131/-0 (131 lines); hunks: -0,0 +1,131; symbols: _selected_name, test_fused_gluon_is_selected_on_supported_amd, _make_cache, test_fused_selected_attention_matches_gather_softmax
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/gluon.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: gluon_dsv41_selected_attention_gfx950, gluon_dsv41_selected_attention_gfx1250
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv4/gluon.py` modified +35/-1 (36 lines); hunks: -47,6 +47,9; -177,10 +180,41 @@ def gluon_dsv4_decode_split_gfx950(*args, **kwargs):; symbols: gluon_dsv4_decode_split_gfx950, gluon_dsv4_prefill_gfx950, gluon_dsv4_prefill_gfx1250
- Key code excerpts:

```diff
diff -- tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/selected.py
@@ -0,0 +1,594 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/selected.py
@@ -0,0 +1,571 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_selected.py
@@ -0,0 +1,131 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/selected.py` added +594/-0; `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/selected.py` added +571/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/gluon.py` added +76/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv4/gluon.py` modified +35/-1; `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv4/__init__.py` added +27/-0; `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/__init__.py` added +27/-0
  - tests: `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_selected.py` added +131/-0
- Risk and verification: The diff ships test coverage in `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_selected.py`, `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv4_prefill.py`, `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv4_prefill_static.py`, `tokenspeed-kernel/test/test_kernel_api_selection.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1636 - perf(dsv41): Optimize DeepSeek V4.1-flash on H20

- Link: https://github.com/lightseekorg/tokenspeed/pull/1636
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` and 14 files; associated commits `7c70d5805e68`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 15 files, +1258/-220, 2297 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` modified +97/-21 (118 lines); hunks: -18,14 +18,21; -37,9 +44,6; symbols: V41CacheFormat, v41_dspark_field_name, touching `V41CacheFormat, v41_dspark_field_name`; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +26/-20 (46 lines); hunks: -18,7 +18,7; -37,20 +37,13; symbols: DeepseekV41Recipe, _row_layout, layer_types, add, touching `DeepseekV41Recipe, _row_layout, layer_types`; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +34/-9 (43 lines); hunks: -231,6 +231,12 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Con...; -1034,8 +1040,12 @@ def write_global(; symbols: __init__, write_global, _selection_rows, forward_v41, touching `__init__, write_global, _selection_rows`; `python/tokenspeed/runtime/models/deepseek_v41.py` modified +41/-2 (43 lines); hunks: -23,6 +23,8; -68,6 +70,7; symbols: v41_mxfp8_config, _ReferenceFp8LinearMethod, __init__, process_weights_after_loading, touching `v41_mxfp8_config, _ReferenceFp8LinearMethod, __init__`.
- Code diff details:
  - `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` modified +97/-21 (118 lines); hunks: -18,14 +18,21; -37,9 +44,6; symbols: V41CacheFormat, v41_dspark_field_name
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +26/-20 (46 lines); hunks: -18,7 +18,7; -37,20 +37,13; symbols: DeepseekV41Recipe, _row_layout, layer_types, add
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +34/-9 (43 lines); hunks: -231,6 +231,12 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Con...; -1034,8 +1040,12 @@ def write_global(; symbols: __init__, write_global, _selection_rows, forward_v41
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +41/-2 (43 lines); hunks: -23,6 +23,8; -68,6 +70,7; symbols: v41_mxfp8_config, _ReferenceFp8LinearMethod, __init__, process_weights_after_loading
  - `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py` modified +22/-2 (24 lines); hunks: -21,14 +21,18; -50,6 +54,7 @@ class DeepseekV41Config(SoftmaxAttnConfig):; symbols: is_deepseek_v41_config, DeepseekV41Config, generate, row_layout
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py
@@ -18,14 +18,21 @@
-"""DeepSeek V4.1 FlatKV geometry; independent of V4's FP8/RoPE layout.
+"""DeepSeek V4.1 FlatKV geometry.
-by scales (SWA: E4M3/E8M0, global: E2M1/E4M3, index: E2M1/E8M0).
+by scales.
+Every row width is a property of the cache format the caller selects through
+``V41_CACHE_FORMATS``; this module knows nothing about which hardware wants
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py
@@ -18,7 +18,7 @@
-"""V4.1 Flash: four history groups sharing one 1,382,400-byte LCM plane.
+"""V4.1 Flash: four history groups sharing one LCM plane.
@@ -37,20 +37,13 @@
-    V41_DSPARK_GROUP_PACKING,
-    V41_DSPARK_LCM_BLOCK_BYTES,
-    V41_GLOBAL_ROW_BYTES,
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py
@@ -231,6 +231,12 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Config) -> None:
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` modified +97/-21; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +26/-20; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +34/-9; `python/tokenspeed/runtime/models/deepseek_v41.py` modified +41/-2; `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py` modified +22/-2; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_config.py`, `test/runtime/test_deepseek_v41_model.py`, `tokenspeed-kernel/test/ops/test_attention_dsv41.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1623 - feat(amd): add gluon DSV4.1 CSA2 indexer

- Link: https://github.com/lightseekorg/tokenspeed/pull/1623
- Status/date: merged / 2026-09-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/__init__.py`, `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/indexer.py`, `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/__init__.py`, `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/indexer.py` and 13 files; associated commits `547b317ef089`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 16 files, +2001/-33, 2359 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +1/-0 (1 lines); hunks: -1247,6 +1247,7 @@ def select_global(; symbols: select_global, touching `select_global`; `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/indexer.py` added +438/-0 (438 lines); hunks: -0,0 +1,438; symbols: _csa2_page_rows, _score_csa2_group, _index_launch_metadata, _dsv41_mxfp4_logits_kernel, touching `_csa2_page_rows, _score_csa2_group, _index_launch_metadata`; `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_index_topk.py` added +424/-0 (424 lines); hunks: -0,0 +1,424; symbols: _index_name, test_gluon_index_topk_is_selected_on_supported_amd, test_gluon_index_topk_query_tile_bounds_logits, device, touching `_index_name, test_gluon_index_topk_is_selected_on_supported_amd, test_gluon_index_topk_query_tile_bounds_logits`; `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/indexer.py` added +358/-0 (358 lines); hunks: -0,0 +1,358; symbols: _e2m1_decode, _wmma_layout, _csa2_page_rows, _load_query, touching `_e2m1_decode, _wmma_layout, _csa2_page_rows`.
- Code diff details:
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +1/-0 (1 lines); hunks: -1247,6 +1247,7 @@ def select_global(; symbols: select_global
  - `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/indexer.py` added +438/-0 (438 lines); hunks: -0,0 +1,438; symbols: _csa2_page_rows, _score_csa2_group, _index_launch_metadata, _dsv41_mxfp4_logits_kernel
  - `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_index_topk.py` added +424/-0 (424 lines); hunks: -0,0 +1,424; symbols: _index_name, test_gluon_index_topk_is_selected_on_supported_amd, test_gluon_index_topk_query_tile_bounds_logits, device
  - `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/indexer.py` added +358/-0 (358 lines); hunks: -0,0 +1,358; symbols: _e2m1_decode, _wmma_layout, _csa2_page_rows, _load_query
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_gluon/indexer.py` added +291/-0 (291 lines); hunks: -0,0 +1,291; symbols: _score_query_tile, _pack_index_q, _pad_query, _select_sorted
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py
@@ -1247,6 +1247,7 @@ def select_global(
+                None,
diff -- tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/indexer.py
@@ -0,0 +1,438 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_index_topk.py
@@ -0,0 +1,424 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +1/-0; `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv41/indexer.py` added +438/-0; `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsv41/indexer.py` added +358/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_gluon/indexer.py` added +291/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/gluon.py` modified +52/-1
  - tests: `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_index_topk.py` added +424/-0; `tokenspeed-kernel/test/ops/test_attention_dsv41_index_scan.py` modified +152/-14; `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +40/-10
- Risk and verification: The diff ships test coverage in `test/ci/ut/ut-tokenspeed-kernel-mi450-sim.yaml`, `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_index_topk.py`, `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py`, `tokenspeed-kernel/test/ops/test_attention_dsv41.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1663 - perf(dsv41): share the native decode schedule across layers in a forward

- Link: https://github.com/lightseekorg/tokenspeed/pull/1663
- Status/date: merged / 2026-09-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `test/runtime/test_deepseek_v41_cache.py`; associated commits `3cfd377de881`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +69/-6, 141 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +23/-6 (29 lines); hunks: -249,6 +249,11 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Con...; -282,6 +287,7 @@ def _publish_cache_pool(self, cache_pool: CachePool) -> None:; symbols: __init__, validate_cache_pool, _publish_cache_pool, init_cuda_graph_state, touching `__init__, validate_cache_pool, _publish_cache_pool`; `test/runtime/test_deepseek_v41_cache.py` modified +46/-0 (46 lines); hunks: -208,6 +208,7 @@ def test_rebinding_cache_pool_drops_pool_derived_state():; -224,6 +225,7 @@ def test_rebinding_cache_pool_drops_pool_derived_state():; symbols: test_rebinding_cache_pool_drops_pool_derived_state, selected, test_native_decode_schedule_shared_per_length_pair_within_a_forward, refresh, touching `test_rebinding_cache_pool_drops_pool_derived_state, selected, test_native_decode_schedule_shared_per_length_pair_within_a_forward`.
- Code diff details:
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +23/-6 (29 lines); hunks: -249,6 +249,11 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Con...; -282,6 +287,7 @@ def _publish_cache_pool(self, cache_pool: CachePool) -> None:; symbols: __init__, validate_cache_pool, _publish_cache_pool, init_cuda_graph_state
  - `test/runtime/test_deepseek_v41_cache.py` modified +46/-0 (46 lines); hunks: -208,6 +208,7 @@ def test_rebinding_cache_pool_drops_pool_derived_state():; -224,6 +225,7 @@ def test_rebinding_cache_pool_drops_pool_derived_state():; symbols: test_rebinding_cache_pool_drops_pool_derived_state, selected, test_native_decode_schedule_shared_per_length_pair_within_a_forward, refresh
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py
@@ -249,6 +249,11 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Config) -> None:
+        # One native decode schedule per distinct (SWA lengths, global lengths)
+        # pair within a forward; layers reusing a selection share it.
+        self._decode_schedules: dict[
+            tuple[int, int], tuple[torch.Tensor, torch.Tensor | None, object]
+        ] = {}
@@ -282,6 +287,7 @@ def _publish_cache_pool(self, cache_pool: CachePool) -> None:
diff -- test/runtime/test_deepseek_v41_cache.py
@@ -208,6 +208,7 @@ def test_rebinding_cache_pool_drops_pool_derived_state():
+    backend._decode_schedules[(0, 0)] = (torch.empty(1), None, object())
@@ -224,6 +225,7 @@ def test_rebinding_cache_pool_drops_pool_derived_state():
+    assert not backend._decode_schedules
@@ -2045,6 +2047,50 @@ def selected(*args):
+def test_native_decode_schedule_shared_per_length_pair_within_a_forward(
+    monkeypatch,
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +23/-6
  - tests: `test/runtime/test_deepseek_v41_cache.py` modified +46/-0
- Risk and verification: The diff ships test coverage in `test/runtime/test_deepseek_v41_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1660 - perf(dsv41): run the DSpark draft window attention on selected_attention

- Link: https://github.com/lightseekorg/tokenspeed/pull/1660
- Status/date: merged / 2026-09-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_config.py` and 7 files; associated commits `13657d1d6762`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 14 files, +1644/-138, 2410 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/deepseek_v41.py` modified +302/-54 (356 lines); hunks: -47,6 +47,17; -58,8 +69,8; symbols: key, _row_plan, DeepseekV41Attention, __init__, touching `key, _row_plan, DeepseekV41Attention`; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +91/-18 (109 lines); hunks: -23,22 +23,26; -68,14 +72,35 @@ def _quantized_kv(x: torch.Tensor) -> torch.Tensor:; symbols: _quantized_kv, _window_rows, _WindowSelection, build, touching `_quantized_kv, _window_rows, _WindowSelection`; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +5/-4 (9 lines); hunks: -202,10 +202,11 @@ class V41SWAQueryPlan:; symbols: V41SWAQueryPlan, DeepseekV41AttentionBackend, __init__, touching `V41SWAQueryPlan, DeepseekV41AttentionBackend, __init__`; `test/runtime/test_deepseek_v41_model.py` modified +417/-17 (434 lines); hunks: -32,6 +32,7; -57,7 +58,6; symbols: _config, query_metadata, decoder_view, rows, touching `_config, query_metadata, decoder_view`.
- Code diff details:
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +302/-54 (356 lines); hunks: -47,6 +47,17; -58,8 +69,8; symbols: key, _row_plan, DeepseekV41Attention, __init__
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +91/-18 (109 lines); hunks: -23,22 +23,26; -68,14 +72,35 @@ def _quantized_kv(x: torch.Tensor) -> torch.Tensor:; symbols: _quantized_kv, _window_rows, _WindowSelection, build
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +5/-4 (9 lines); hunks: -202,10 +202,11 @@ class V41SWAQueryPlan:; symbols: V41SWAQueryPlan, DeepseekV41AttentionBackend, __init__
  - `test/runtime/test_deepseek_v41_model.py` modified +417/-17 (434 lines); hunks: -32,6 +32,7; -57,7 +58,6; symbols: _config, query_metadata, decoder_view, rows
  - `test/runtime/test_deepseek_v41_dspark.py` modified +121/-2 (123 lines); hunks: -58,6 +58,7; -66,6 +67,7; symbols: __init__, forward, test_window_attention_matches_dense_reference
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v41.py
@@ -47,6 +47,17 @@
+The forward is four stages on one path -- ``encoder_forward`` (embedding and
+the layers below the candidate source, one row per token), ``narrowing_forward``
+(the candidate source layer: all rows in, the decoder view's rows out),
+``decoder_forward`` (the remaining layers and the final norm on the narrowed
+rows) and ``finish_forward`` (the sampled-row gather and the DSpark row
+report). ``forward`` composes them; the prefill graph replays the first and
diff -- python/tokenspeed/runtime/models/deepseek_v41_dspark.py
@@ -23,22 +23,26 @@
-within its small context window plus the entire non-causal proposal block.
+within its small context window plus the entire non-causal proposal block, and
+runs through the same ``selected_attention`` workspace kernel as the target's
+prefill (FlashMLA on sm90+, Triton elsewhere); the fp32 arithmetic stays as the
+CPU reference.
-from dataclasses import replace
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py
@@ -202,10 +202,11 @@ class V41SWAQueryPlan:
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41.py` modified +302/-54; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +91/-18; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +5/-4
  - tests: `test/runtime/test_deepseek_v41_model.py` modified +417/-17; `test/runtime/test_deepseek_v41_dspark.py` modified +121/-2; `test/runtime/test_deepseek_v41_cache.py` modified +3/-2; `test/runtime/test_deepseek_v41_config.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `test/ci/eval/deepseek-v4.1-flash-evalscope-gsm8k.yaml`, `test/ci_system/test_eval_configs.py`, `test/runtime/execution/test_kda_prefill_graph_cache.py`, `test/runtime/test_deepseek_v41_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1664 - perf(moe): FlashInfer CUTLASS MXFP4 experts on Hopper (W4A16 auto, W4A8 opt-in) + V4.1 log fixes

- Link: https://github.com/lightseekorg/tokenspeed/pull/1664
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v41_engram.py` and 11 files; associated commits `f93145609230`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 55 files, +2411/-307, 4046 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +38/-80 (118 lines); hunks: -68,6 +68,7; -244,7 +245,6 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Conf...; symbols: V41CompressorPlan, __init__, _publish_cache_pool, init_cuda_graph_state, touching `V41CompressorPlan, __init__, _publish_cache_pool`; `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +43/-3 (46 lines); hunks: -66,13 +66,16; -721,12 +724,20 @@ def forward(self, indices: torch.Tensor) -> torch.Tensor:; symbols: forward, _load_projection_scale, engram_reduce_lane_width, DeepseekV41Engram, touching `forward, _load_projection_scale, engram_reduce_lane_width`; `python/tokenspeed/runtime/models/deepseek_v4.py` modified +5/-0 (5 lines); hunks: -1963,6 +1963,11 @@ def __init__(; symbols: __init__, touching `__init__`; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` modified +4/-0 (4 lines); hunks: -98,6 +98,10 @@ def compressor_tail(self, owner: int) -> torch.Tensor:; symbols: compressor_tail, get_kv_size_bytes, zero_new_blocks, touching `compressor_tail, get_kv_size_bytes, zero_new_blocks`.
- Code diff details:
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +38/-80 (118 lines); hunks: -68,6 +68,7; -244,7 +245,6 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Conf...; symbols: V41CompressorPlan, __init__, _publish_cache_pool, init_cuda_graph_state
  - `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +43/-3 (46 lines); hunks: -66,13 +66,16; -721,12 +724,20 @@ def forward(self, indices: torch.Tensor) -> torch.Tensor:; symbols: forward, _load_projection_scale, engram_reduce_lane_width, DeepseekV41Engram
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +5/-0 (5 lines); hunks: -1963,6 +1963,11 @@ def __init__(; symbols: __init__
  - `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` modified +4/-0 (4 lines); hunks: -98,6 +98,10 @@ def compressor_tail(self, owner: int) -> torch.Tensor:; symbols: compressor_tail, get_kv_size_bytes, zero_new_blocks
  - `test/runtime/test_deepseek_v41_cache.py` modified +9/-104 (113 lines); hunks: -220,7 +220,6 @@ def test_rebinding_cache_pool_drops_pool_derived_state():; -290,29 +289,6 @@ def _final(lengths, prefixes):; symbols: test_rebinding_cache_pool_drops_pool_derived_state, _final, test_refresh_rejects_missing_history_before_execution, test_packed_refresh_positions_padding_and_token_count
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py
@@ -68,6 +68,7 @@
+from tokenspeed.runtime.utils.host_sync import allow_host_sync
@@ -244,7 +245,6 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Config) -> None:
-        self._decode_history_status: torch.Tensor | None = None
@@ -282,7 +282,6 @@ def _publish_cache_pool(self, cache_pool: CachePool) -> None:
-        self._decode_history_status = None
@@ -315,9 +314,6 @@ def init_cuda_graph_state(
diff -- python/tokenspeed/runtime/models/deepseek_v41_engram.py
@@ -66,13 +66,16 @@
-from tokenspeed.runtime.distributed.comm_ops import all_reduce
+from tokenspeed.runtime.distributed.comm_ops import all_reduce, prepare_all_reduce_lane
+from tokenspeed.runtime.utils import get_colorful_logger
+logger = get_colorful_logger(__name__)
@@ -721,12 +724,20 @@ def forward(self, indices: torch.Tensor) -> torch.Tensor:
+            # The workspace all-reduce sizes itself on rows x trailing width.
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -1963,6 +1963,11 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +38/-80; `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +43/-3; `python/tokenspeed/runtime/models/deepseek_v4.py` modified +5/-0; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` modified +4/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +3/-35; `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/__init__.py` modified +38/-0
  - tests: `test/runtime/test_deepseek_v41_cache.py` modified +9/-104; `test/runtime/test_deepseek_v41_inputs.py` modified +58/-0
- Risk and verification: The diff ships test coverage in `test/runtime/layers/test_moe_layer_activation_flag.py`, `test/runtime/layers/test_moe_layer_plan_traits.py`, `test/runtime/test_batch_log.py`, `test/runtime/test_deepseek_v41_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1683 - perf(dsv41): select Reindex rows from the compacted candidate pool

- Link: https://github.com/lightseekorg/tokenspeed/pull/1683
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py`, `tokenspeed-kernel/test/ops/test_attention_dsv41.py`; associated commits `97c0066e6c63`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +198/-36, 304 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +126/-7 (133 lines); hunks: -1019,13 +1019,76 @@ def _index_finish_parts(scores, ids, k, output, lengths):; -1877,6 +1940,62 @@ def dense_ranges(lengths, capacity):; symbols: _index_finish_parts, _finish_topk, _finish_topk_kernel, _index_topk_outputs, touching `_index_finish_parts, _finish_topk, _finish_topk_kernel`; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py` modified +23/-29 (52 lines); hunks: -35,6 +35,7; -443,26 +444,6 @@ def _hopper_paged_scores(queries, cache, weights, block_tab...; symbols: _hopper_paged_scores, _restrict_to_candidates, _block_maxima, _select, touching `_hopper_paged_scores, _restrict_to_candidates, _block_maxima`; `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +49/-0 (49 lines); hunks: -633,6 +633,55 @@ def test_index_topk_selects_top_scores_on_every_index_forma...; symbols: test_index_topk_selects_top_scores_on_every_index_format, test_reindex_maps_every_candidate_block_order_back_to_row_ids, test_candidate_block_max_not_sum, touching `test_index_topk_selects_top_scores_on_every_index_format, test_reindex_maps_every_candidate_block_order_back_to_row_ids, test_candidate_block_max_not_sum`.
- Code diff details:
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +126/-7 (133 lines); hunks: -1019,13 +1019,76 @@ def _index_finish_parts(scores, ids, k, output, lengths):; -1877,6 +1940,62 @@ def dense_ranges(lengths, capacity):; symbols: _index_finish_parts, _finish_topk, _finish_topk_kernel, _index_topk_outputs
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py` modified +23/-29 (52 lines); hunks: -35,6 +35,7; -443,26 +444,6 @@ def _hopper_paged_scores(queries, cache, weights, block_tab...; symbols: _hopper_paged_scores, _restrict_to_candidates, _block_maxima, _select
  - `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +49/-0 (49 lines); hunks: -633,6 +633,55 @@ def test_index_topk_selects_top_scores_on_every_index_forma...; symbols: test_index_topk_selects_top_scores_on_every_index_format, test_reindex_maps_every_candidate_block_order_back_to_row_ids, test_candidate_block_max_not_sum
- Key code excerpts:

```diff
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py
@@ -1019,13 +1019,76 @@ def _index_finish_parts(scores, ids, k, output, lengths):
-def _finish_topk(scores, ids, output, lengths):
-    valid = scores > -torch.inf
-    ids = ids.masked_fill(~valid, torch.iinfo(torch.int64).max).sort(dim=1).values
-    ids = ids.masked_fill(ids == torch.iinfo(torch.int64).max, -1)
-    output.fill_(-1)
-    output[:, : ids.shape[1]].copy_(ids)
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py
@@ -35,6 +35,7 @@
+    candidate_scores,
@@ -443,26 +444,6 @@ def _hopper_paged_scores(queries, cache, weights, block_table, valid_lengths, ca
-def _restrict_to_candidates(logits, candidates):
-    """Mask every row outside the candidate blocks, keeping the row addressing.
-    Null blocks are routed to a sentinel column instead of being scattered as
-    False: a scatter writes duplicate indices in an undefined order, so a null
diff -- tokenspeed-kernel/test/ops/test_attention_dsv41.py
@@ -633,6 +633,55 @@ def test_index_topk_selects_top_scores_on_every_index_format(device, fmt, shared
```

- Extracted files (not manually reviewed):
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +126/-7; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py` modified +23/-29
  - tests: `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +49/-0
- Risk and verification: The diff ships test coverage in `tokenspeed-kernel/test/ops/test_attention_dsv41.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1685 - perf(dsv41): fuse the FP8 index-query quantization into one kernel

- Link: https://github.com/lightseekorg/tokenspeed/pull/1685
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py`, `tokenspeed-kernel/test/ops/test_attention_dsv41.py`; associated commits `927bf1a844ad`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +92/-14, 142 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +65/-0 (65 lines); hunks: -1741,6 +1741,71 @@ def _pack_index_queries(X, V, S, X0, X1):; symbols: _pack_index_queries, _quantize_index_queries_kernel, quantize_index_queries, pack_index_queries, touching `_pack_index_queries, _quantize_index_queries_kernel, quantize_index_queries`; `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +25/-0 (25 lines); hunks: -633,6 +633,31 @@ def test_index_topk_selects_top_scores_on_every_index_forma...; symbols: test_index_topk_selects_top_scores_on_every_index_format, test_index_query_quantization_matches_the_reference_codec, test_reindex_maps_every_candidate_block_order_back_to_row_ids, touching `test_index_topk_selects_top_scores_on_every_index_format, test_index_query_quantization_matches_the_reference_codec, test_reindex_maps_every_candidate_block_order_back_to_row_ids`; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py` modified +2/-14 (16 lines); hunks: -40,6 +40,7; -343,18 +344,6 @@ def _hopper_api(queries):; symbols: _hopper_api, _quantize_index_queries, _index_pages, hopper_index_topk, touching `_hopper_api, _quantize_index_queries, _index_pages`.
- Code diff details:
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +65/-0 (65 lines); hunks: -1741,6 +1741,71 @@ def _pack_index_queries(X, V, S, X0, X1):; symbols: _pack_index_queries, _quantize_index_queries_kernel, quantize_index_queries, pack_index_queries
  - `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +25/-0 (25 lines); hunks: -633,6 +633,31 @@ def test_index_topk_selects_top_scores_on_every_index_forma...; symbols: test_index_topk_selects_top_scores_on_every_index_format, test_index_query_quantization_matches_the_reference_codec, test_reindex_maps_every_candidate_block_order_back_to_row_ids
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py` modified +2/-14 (16 lines); hunks: -40,6 +40,7; -343,18 +344,6 @@ def _hopper_api(queries):; symbols: _hopper_api, _quantize_index_queries, _index_pages, hopper_index_topk
- Key code excerpts:

```diff
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py
@@ -1741,6 +1741,71 @@ def _pack_index_queries(X, V, S, X0, X1):
+@triton.jit
+def _quantize_index_queries_kernel(
+    Q,
+    W,
+    QOut,
+    WOut,
diff -- tokenspeed-kernel/test/ops/test_attention_dsv41.py
@@ -633,6 +633,31 @@ def test_index_topk_selects_top_scores_on_every_index_format(device, fmt, shared
+@pytest.mark.parametrize("contiguous", [True, False])
+def test_index_query_quantization_matches_the_reference_codec(device, contiguous):
+    # The fused quantizer replaces a chain of eager ops, so it has to reproduce
+    # them bit for bit: the E4M3 payload decides which rows a pass scores, and
+    # the folded weight carries the query scale the logits kernels never apply.
+    # A head-major view is the shape a projection hands over before any copy.
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py
@@ -40,6 +40,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +65/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py` modified +2/-14
  - tests: `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +25/-0
- Risk and verification: The diff ships test coverage in `tokenspeed-kernel/test/ops/test_attention_dsv41.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1686 - perf(dsv41): fuse the DSpark decode prologue, draft tail and sampler sync into a handful of launches

- Link: https://github.com/lightseekorg/tokenspeed/pull/1686
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/execution/drafter/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` and 15 files; associated commits `e87f7267c0b3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 43 files, +3759/-607, 5117 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py` modified +120/-115 (235 lines); hunks: -2,122 +2,45; -144,27 +67,109 @@ def forward(; symbols: _local_vocab_argmax, DSparkVanillaMarkov, __init__, local_bias, touching `_local_vocab_argmax, DSparkVanillaMarkov, __init__`; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +96/-62 (158 lines); hunks: -54,6 +54,7; -81,27 +82,14 @@ class _WindowSelection:; symbols: _WindowSelection, build, _write_window_rows, _WindowAttention, touching `_WindowSelection, build, _write_window_rows`; `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +17/-64 (81 lines); hunks: -336,17 +336,18 @@ def __init__(; -364,9 +365,6 @@ def __init__(; symbols: __init__, forward_backbone, local_base_logits, refresh_local_base_logits_head, touching `__init__, forward_backbone, local_base_logits`; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +20/-10 (30 lines); hunks: -468,17 +468,27 @@ def window_slots(; symbols: window_slots, _refresh_decode_window, touching `window_slots, _refresh_decode_window`.
- Code diff details:
  - `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py` modified +120/-115 (235 lines); hunks: -2,122 +2,45; -144,27 +67,109 @@ def forward(; symbols: _local_vocab_argmax, DSparkVanillaMarkov, __init__, local_bias
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +96/-62 (158 lines); hunks: -54,6 +54,7; -81,27 +82,14 @@ class _WindowSelection:; symbols: _WindowSelection, build, _write_window_rows, _WindowAttention
  - `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +17/-64 (81 lines); hunks: -336,17 +336,18 @@ def __init__(; -364,9 +365,6 @@ def __init__(; symbols: __init__, forward_backbone, local_base_logits, refresh_local_base_logits_head
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +20/-10 (30 lines); hunks: -468,17 +468,27 @@ def window_slots(; symbols: window_slots, _refresh_decode_window
  - `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +15/-0 (15 lines); hunks: -290,6 +290,21 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py
@@ -2,122 +2,45 @@
-"""Tensor-parallel DSpark Markov and confidence heads."""
+"""Tensor-parallel DSpark Markov and confidence heads.
+The Markov head keeps the checkpoint's BF16 dtypes: its bigram table is
+replicated on every rank so a lookup is a plain gather, and its rank-``R``
+projection is sharded over the vocabulary exactly like the LM head, so the
+bias of a shard column lands on the rank that owns that column's base logit.
diff -- python/tokenspeed/runtime/models/deepseek_v41_dspark.py
@@ -54,6 +54,7 @@
+    _merged,
@@ -81,27 +82,14 @@ class _WindowSelection:
-    rows are always visible. All three are shared by every stage.
+    rows are always visible. All three come from ``dsv41.dspark_block`` and
+    are shared by every stage.
-    @classmethod
diff -- python/tokenspeed/runtime/models/deepseek_v4_dspark.py
@@ -336,17 +336,18 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py` modified +120/-115; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +96/-62; `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +17/-64; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +20/-10; `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +15/-0; `python/tokenspeed/runtime/models/deepseek_v41.py` modified +5/-1
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +95/-142
- Risk and verification: The diff ships test coverage in `test/runtime/execution/test_draft_target_wiring.py`, `test/runtime/layers/test_vocab_parallel_embedding.py`, `test/runtime/sampling/test_verify_output_pack.py`, `test/runtime/test_deepseek_v41_dspark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1689 - perf(dsv41): score only the candidate pool on Hopper with a CuTe DSL sparse indexer

- Link: https://github.com/lightseekorg/tokenspeed/pull/1689
- Status/date: merged / 2026-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_cute_dsl/__init__.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_cute_dsl/sparse_index_scores.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cute_dsl.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py`, `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py`; associated commits `9263c4a34630`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +1093/-3, 1154 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_cute_dsl/sparse_index_scores.py` added +469/-0 (469 lines); hunks: -0,0 +1,469; symbols: SparseIndexScoreKernel, __init__, __call__, kernel, touching `SparseIndexScoreKernel, __init__, __call__`; `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py` modified +298/-0 (298 lines); hunks: -18,6 +18,8; -86,3 +88,299 @@ def run():; symbols: run, _hopper_index_case, test_sparse_index_scores_match_the_dense_scorer, dense, touching `run, _hopper_index_case, test_sparse_index_scores_match_the_dense_scorer`; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cute_dsl.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: _GraphSafeDLPack, __init__, __dlpack__, __dlpack_device__, touching `_GraphSafeDLPack, __init__, __dlpack__`; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py` modified +49/-2 (51 lines); hunks: -27,6 +27,10; -477,13 +481,20 @@ def _select(scores, k, graph_safe, destination, lengths, c...; symbols: _select, _flashinfer_select, hopper_index_topk, touching `_select, _flashinfer_select, hopper_index_topk`.
- Code diff details:
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_cute_dsl/sparse_index_scores.py` added +469/-0 (469 lines); hunks: -0,0 +1,469; symbols: SparseIndexScoreKernel, __init__, __call__, kernel
  - `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py` modified +298/-0 (298 lines); hunks: -18,6 +18,8; -86,3 +88,299 @@ def run():; symbols: run, _hopper_index_case, test_sparse_index_scores_match_the_dense_scorer, dense
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cute_dsl.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: _GraphSafeDLPack, __init__, __dlpack__, __dlpack_device__
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py` modified +49/-2 (51 lines); hunks: -27,6 +27,10; -477,13 +481,20 @@ def _select(scores, k, graph_safe, destination, lengths, c...; symbols: _select, _flashinfer_select, hopper_index_topk
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/_cute_dsl/__init__.py` added +19/-0 (19 lines); hunks: -0,0 +1,19
- Key code excerpts:

```diff
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_cute_dsl/sparse_index_scores.py
@@ -0,0 +1,469 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py
@@ -18,6 +18,8 @@
+from unittest.mock import patch
@@ -86,3 +88,299 @@ def run():
+def _hopper_index_case(device, tokens, heads, blocks, pages, table_width, seed):
+    """An FP8 index cache, quantized queries and a candidate pool for the Hopper scorers.
+    The page table is a permutation with a null and an out-of-range page, so a
+    scorer that addressed pages by block id instead of through the table, or
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cute_dsl.py
@@ -0,0 +1,238 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_cute_dsl/sparse_index_scores.py` added +469/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cute_dsl.py` added +238/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py` modified +49/-2; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/_cute_dsl/__init__.py` added +19/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_cute_dsl/__init__.py` added +19/-0
  - tests: `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py` modified +298/-0
- Risk and verification: The diff ships test coverage in `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1719 - fix(dsv41): stop recompiling the sparse index scorer per prefill chunk

- Link: https://github.com/lightseekorg/tokenspeed/pull/1719
- Status/date: merged / 2026-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cute_dsl.py`, `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py`; associated commits `c2c0730670b8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +153/-32, 252 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py` modified +102/-3 (105 lines); hunks: -155,12 +155,13 @@ def test_sparse_index_scores_match_the_dense_scorer(blocks...; -218,6 +219,104 @@ def sparse(field):; symbols: test_sparse_index_scores_match_the_dense_scorer, sparse, test_sparse_index_scores_compile_once_across_table_and_pool_widths, one_row, touching `test_sparse_index_scores_match_the_dense_scorer, sparse, test_sparse_index_scores_compile_once_across_table_and_pool_widths`; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cute_dsl.py` modified +51/-29 (80 lines); hunks: -49,26 +49,52 @@ def __dlpack_device__(self):; -87,11 +113,11 @@ def sparse_index_scores_supported(queries, weights, table,...; symbols: __dlpack_device__, _to_cute, _specialised, _kernel, touching `__dlpack_device__, _to_cute, _specialised`.
- Code diff details:
  - `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py` modified +102/-3 (105 lines); hunks: -155,12 +155,13 @@ def test_sparse_index_scores_match_the_dense_scorer(blocks...; -218,6 +219,104 @@ def sparse(field):; symbols: test_sparse_index_scores_match_the_dense_scorer, sparse, test_sparse_index_scores_compile_once_across_table_and_pool_widths, one_row
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cute_dsl.py` modified +51/-29 (80 lines); hunks: -49,26 +49,52 @@ def __dlpack_device__(self):; -87,11 +113,11 @@ def sparse_index_scores_supported(queries, weights, table,...; symbols: __dlpack_device__, _to_cute, _specialised, _kernel
- Key code excerpts:

```diff
diff -- tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py
@@ -155,12 +155,13 @@ def test_sparse_index_scores_match_the_dense_scorer(blocks, pages, table_width):
-    # The op contract admits noncontiguous table and candidate views; the
-    # scorer compiles compact layouts, so those stay on the dense path.
+    # The op contract admits any table and candidate view. The scorer's
+    # dynamic layouts express a row stride but not a column one, so a view
+    # sliced to fewer columns is served and one striding along the row is not.
-    assert not cute_dsl.sparse_index_scores_supported(
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cute_dsl.py
@@ -49,26 +49,52 @@ def __dlpack_device__(self):
-def _to_cute(tensor, dynamic_rows, align):
-    """Wrap a torch tensor for CuTe, leaving only the row count dynamic.
+def _to_cute(tensor, dynamic, align):
+    """Wrap a torch tensor for CuTe with the requested specialisation.
-    Every other extent stays static so the candidate width and table width
-    become compile-time tile counts, and every stride is compiled in as well;
```

- Extracted files (not manually reviewed):
  - tests: `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py` modified +102/-3
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/cute_dsl.py` modified +51/-29
- Risk and verification: The diff ships test coverage in `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1742 - ci(amd-kernel): Group DeepSeek V4.1 MXFP8 GEMM benchmarks by model

- Link: https://github.com/lightseekorg/tokenspeed/pull/1742
- Status/date: merged / 2026-09-23
- Trace source: `git log --name-only -- <model-files>` found it through `tokenspeed-kernel/benchmarks/amd/gfx950/dsv41_flash/gemm.json`; associated commits `e60df80f69bd`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +297/-224, 543 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tokenspeed-kernel/benchmarks/amd/gfx950/dsv41_flash/gemm.json` added +295/-0 (295 lines); hunks: -0,0 +1,295.
- Code diff details:
  - `tokenspeed-kernel/benchmarks/amd/gfx950/dsv41_flash/gemm.json` added +295/-0 (295 lines); hunks: -0,0 +1,295
- Key code excerpts:

```diff
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/dsv41_flash/gemm.json
@@ -0,0 +1,295 @@
+{
+  "schema_version": 1,
+  "cases": [
+    {
+      "id": "dsv41_flash.gemm.mm/gluon_mm_mxfp8_gfx950/wq_a_wkv-m1024-n1792-k5120-mxfp8-bfloat16",
+      "comparison_epoch": 1,
```

- Extracted files (not manually reviewed):
  - runtime: `tokenspeed-kernel/benchmarks/amd/gfx950/dsv41_flash/gemm.json` added +295/-0
- Risk and verification: The diff ships test coverage in `tokenspeed-kernel/test/test_benchmark_ci.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1748 - ci: turn on dspark for dsv4.1 ci

- Link: https://github.com/lightseekorg/tokenspeed/pull/1748
- Status/date: merged / 2026-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml`, `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml`; associated commits `162de66c7166`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +15/-8, 76 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml` renamed +2/-1 (3 lines); hunks: -1,5 +1,5; -16,6 +16,7 @@ server:; `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml` renamed +2/-1 (3 lines); hunks: -1,5 +1,5; -16,6 +16,7 @@ server:.
- Code diff details:
  - `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml` renamed +2/-1 (3 lines); hunks: -1,5 +1,5; -16,6 +16,7 @@ server:
  - `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml` renamed +2/-1 (3 lines); hunks: -1,5 +1,5; -16,6 +16,7 @@ server:
- Key code excerpts:

```diff
diff -- test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml
@@ -1,5 +1,5 @@
-name: eval-deepseek-v4.1-flash-gsm8k-amd
+name: eval-deepseek-v4.1-flash-dspark-gsm8k-amd
@@ -16,6 +16,7 @@ server:
+    --speculative-algorithm DSPARK
diff -- test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml
@@ -1,5 +1,5 @@
-name: eval-deepseek-v4.1-flash-gsm8k
+name: eval-deepseek-v4.1-flash-dspark-gsm8k
@@ -16,6 +16,7 @@ server:
+    --speculative-algorithm DSPARK
```

- Extracted files (not manually reviewed):
  - tests: `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml` renamed +2/-1; `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml` renamed +2/-1
- Risk and verification: The diff ships test coverage in `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml`, `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml`, `test/ci_system/test_eval_configs.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1794 - perf(dsv41): stop recompiling compressor_metadata on every prefill chunk

- Link: https://github.com/lightseekorg/tokenspeed/pull/1794
- Status/date: merged / 2026-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py`, `tokenspeed-kernel/test/ops/test_attention_dsv41.py`; associated commits `6cb384bef40e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +48/-4, 66 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +42/-0 (42 lines); hunks: -1250,6 +1250,48 @@ def prepare():; symbols: prepare, test_compressor_metadata_table_width_does_not_recompile, run, touching `prepare, test_compressor_metadata_table_width_does_not_recompile, run`; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +6/-4 (10 lines); hunks: -3046,10 +3046,12 @@ def _compressor_metadata(; symbols: _compressor_metadata, touching `_compressor_metadata`.
- Code diff details:
  - `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +42/-0 (42 lines); hunks: -1250,6 +1250,48 @@ def prepare():; symbols: prepare, test_compressor_metadata_table_width_does_not_recompile, run
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +6/-4 (10 lines); hunks: -3046,10 +3046,12 @@ def _compressor_metadata(; symbols: _compressor_metadata
- Key code excerpts:

```diff
diff -- tokenspeed-kernel/test/ops/test_attention_dsv41.py
@@ -1250,6 +1250,48 @@ def prepare():
+def test_compressor_metadata_table_width_does_not_recompile():
+    """Chunked prefill widens the tail table every chunk; the kernel must not
+    key its compile cache on the table geometry."""
+    if not torch.cuda.is_available():
+        pytest.skip("requires CUDA/ROCm")
+    def run(rows, width):
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py
@@ -3046,10 +3046,12 @@ def _compressor_metadata(
-    TR: tl.constexpr,
-    TC: tl.constexpr,
-    TS0: tl.constexpr,
-    TS1: tl.constexpr,
+    # Table geometry grows with every prefill chunk; a constexpr here would
+    # recompile the kernel once per chunk.
```

- Extracted files (not manually reviewed):
  - tests: `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +42/-0
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +6/-4
- Risk and verification: The diff ships test coverage in `tokenspeed-kernel/test/ops/test_attention_dsv41.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1814 - perf(dsv41): stop recompiling compressor_pool for every prefill length

- Link: https://github.com/lightseekorg/tokenspeed/pull/1814
- Status/date: merged / 2026-09-27
- Trace source: `git log --name-only -- <model-files>` found it through `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py`, `tokenspeed-kernel/test/ops/test_attention_dsv41.py`; associated commits `66d5f3b6e833`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +75/-8, 141 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +36/-6 (42 lines); hunks: -28,6 +28,7; -1000,6 +1001,40 @@ def test_compressor_fused_norm_preserves_pooled_bf16_boun...; symbols: test_compressor_fused_norm_preserves_pooled_bf16_boundary, test_compressor_pool_token_count_does_not_recompile, run, _native_available, touching `test_compressor_fused_norm_preserves_pooled_bf16_boundary, test_compressor_pool_token_count_does_not_recompile, run`; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +3/-1 (4 lines); hunks: -1285,7 +1285,9 @@ def _compressor_pool(; symbols: _compressor_pool, touching `_compressor_pool`.
- Code diff details:
  - `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +36/-6 (42 lines); hunks: -28,6 +28,7; -1000,6 +1001,40 @@ def test_compressor_fused_norm_preserves_pooled_bf16_boun...; symbols: test_compressor_fused_norm_preserves_pooled_bf16_boundary, test_compressor_pool_token_count_does_not_recompile, run, _native_available
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +3/-1 (4 lines); hunks: -1285,7 +1285,9 @@ def _compressor_pool(; symbols: _compressor_pool
- Key code excerpts:

```diff
diff -- tokenspeed-kernel/test/ops/test_attention_dsv41.py
@@ -28,6 +28,7 @@
+from utils import assert_no_triton_compile
@@ -1000,6 +1001,40 @@ def test_compressor_fused_norm_preserves_pooled_bf16_boundary(device):
+def test_compressor_pool_token_count_does_not_recompile(device):
+    """Every prefill brings a new token count; the kernel must not key its
+    compile cache on the row count."""
+    tail = torch.randn(5, 2, 2, 512, device=device)
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py
@@ -1285,7 +1285,9 @@ def _compressor_pool(
-    N: tl.constexpr,
+    # The row count is the forward's token count; a constexpr here would
+    # recompile the kernel for every new prefill length.
+    N,
```

- Extracted files (not manually reviewed):
  - tests: `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +36/-6
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `tokenspeed-kernel/test/ops/test_attention_dsv41.py`, `tokenspeed-kernel/test/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1844 - feat(dsv41): support vit batching

- Link: https://github.com/lightseekorg/tokenspeed/pull/1844
- Status/date: merged / 2026-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/deepseek_v41_vision.py`, `test/runtime/test_deepseek_v41_vision.py`; associated commits `b9544177b196`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +557/-62, 728 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/deepseek_v41_vision.py` modified +108/-19 (127 lines); hunks: -23,6 +23,7; -34,7 +35,9; symbols: __init__, forward, touching `__init__, forward`; `test/runtime/test_deepseek_v41_vision.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: _model, _item, test_batch_limits_and_order, encode, touching `_model, _item, test_batch_limits_and_order`.
- Code diff details:
  - `python/tokenspeed/runtime/models/deepseek_v41_vision.py` modified +108/-19 (127 lines); hunks: -23,6 +23,7; -34,7 +35,9; symbols: __init__, forward
  - `test/runtime/test_deepseek_v41_vision.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: _model, _item, test_batch_limits_and_order, encode
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v41_vision.py
@@ -23,6 +23,7 @@
+from itertools import accumulate
@@ -34,7 +35,9 @@
+from tokenspeed.runtime.multimodal.encoder_batching import pack_encoder_batches
+from tokenspeed.runtime.utils.env import envs
@@ -112,7 +115,11 @@ def __init__(
-            dim, eps=1e-6, elementwise_affine=True, device=None, dtype=torch.float32
diff -- test/runtime/test_deepseek_v41_vision.py
@@ -0,0 +1,224 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41_vision.py` modified +108/-19
  - tests: `test/runtime/test_deepseek_v41_vision.py` added +224/-0
- Risk and verification: The diff ships test coverage in `test/runtime/test_deepseek_v41_vision.py`, `test/runtime/test_encoder_batching.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1868 - fix(kernel): stop dsv41 address kernels recompiling on page-table geometry

- Link: https://github.com/lightseekorg/tokenspeed/pull/1868
- Status/date: merged / 2026-09-29
- Trace source: `git log --name-only -- <model-files>` found it through `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py`; associated commits `f69da88adb3e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +79/-5, 126 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +4/-4 (8 lines); hunks: -2837,7 +2837,7 @@ def decode_rows(; -2913,7 +2913,7 @@ def decode_window(; symbols: decode_rows, _decode_window_kernel, decode_window, _global_slots_kernel, touching `decode_rows, _decode_window_kernel, decode_window`.
- Code diff details:
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +4/-4 (8 lines); hunks: -2837,7 +2837,7 @@ def decode_rows(; -2913,7 +2913,7 @@ def decode_window(; symbols: decode_rows, _decode_window_kernel, decode_window, _global_slots_kernel
- Key code excerpts:

```diff
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py
@@ -2837,7 +2837,7 @@ def decode_rows(
-@triton.jit
+@triton.jit(do_not_specialize=["SW", "SS", "SC", "TABLE_ROWS"])
@@ -2913,7 +2913,7 @@ def decode_window(
-@triton.jit(do_not_specialize=["N"])
+@triton.jit(do_not_specialize=["N", "TR", "TC", "TS0", "TS1"])
@@ -2981,7 +2981,7 @@ def global_slots(rows, positions, requests, table, ratio, pages):
```

- Extracted files (not manually reviewed):
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +4/-4
- Risk and verification: The diff ships test coverage in `tokenspeed-kernel/test/ops/test_batch_shape_recompiles.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1863 - perf(dsv41): skip decoder work for incomplete prefill chunks

- Link: https://github.com/lightseekorg/tokenspeed/pull/1863
- Status/date: merged / 2026-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/execution/drafter/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `test/runtime/test_deepseek_v41_cache.py` and 11 files; associated commits `cb127fde197a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 39 files, +1842/-173, 3184 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/deepseek_v41.py` modified +44/-21 (65 lines); hunks: -809,6 +809,18 @@ def _kernel_attn_sink(self):; -834,6 +846,9 @@ def forward(; symbols: _kernel_attn_sink, _write_global_kv, forward, touching `_kernel_attn_sink, _write_global_kv, forward`; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +19/-9 (28 lines); hunks: -210,6 +210,8 @@ class DeepseekV41AttentionBackend(AttentionBackend):; -671,20 +673,27 @@ def _build_decoder_view(; symbols: DeepseekV41AttentionBackend, __init__, _build_decoder_view, touching `DeepseekV41AttentionBackend, __init__, _build_decoder_view`; `test/runtime/test_deepseek_v41_model.py` modified +54/-15 (69 lines); hunks: -57,6 +57,7; -189,6 +190,12 @@ def _ctx(backend, tokens, mode):; symbols: _ctx, test_decoder_narrowing_projects_global_from_all_rows_then_runs_the_tail, observe_mixes, observe_qkv, touching `_ctx, test_decoder_narrowing_projects_global_from_all_rows_then_runs_the_tail, observe_mixes`; `test/runtime/test_deepseek_v41_cache.py` modified +50/-15 (65 lines); hunks: -570,14 +570,20 @@ def test_write_global_masks_rows_below_the_replay_floor(mo...; -608,10 +614,29 @@ def test_decoder_view_is_the_identity_for_decode_and_compl...; symbols: test_write_global_masks_rows_below_the_replay_floor, test_decoder_view_is_the_identity_for_decode_and_complete_short_chunks, test_decoder_view_keeps_one_row_per_open_chunk_and_the_final_window, test_decoder_view_preserves_compute_rows_without_open_chunk_logits, touching `test_write_global_masks_rows_below_the_replay_floor, test_decoder_view_is_the_identity_for_decode_and_complete_short_chunks, test_decoder_view_keeps_one_row_per_open_chunk_and_the_final_window`.
- Code diff details:
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +44/-21 (65 lines); hunks: -809,6 +809,18 @@ def _kernel_attn_sink(self):; -834,6 +846,9 @@ def forward(; symbols: _kernel_attn_sink, _write_global_kv, forward
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +19/-9 (28 lines); hunks: -210,6 +210,8 @@ class DeepseekV41AttentionBackend(AttentionBackend):; -671,20 +673,27 @@ def _build_decoder_view(; symbols: DeepseekV41AttentionBackend, __init__, _build_decoder_view
  - `test/runtime/test_deepseek_v41_model.py` modified +54/-15 (69 lines); hunks: -57,6 +57,7; -189,6 +190,12 @@ def _ctx(backend, tokens, mode):; symbols: _ctx, test_decoder_narrowing_projects_global_from_all_rows_then_runs_the_tail, observe_mixes, observe_qkv
  - `test/runtime/test_deepseek_v41_cache.py` modified +50/-15 (65 lines); hunks: -570,14 +570,20 @@ def test_write_global_masks_rows_below_the_replay_floor(mo...; -608,10 +614,29 @@ def test_decoder_view_is_the_identity_for_decode_and_compl...; symbols: test_write_global_masks_rows_below_the_replay_floor, test_decoder_view_is_the_identity_for_decode_and_complete_short_chunks, test_decoder_view_keeps_one_row_per_open_chunk_and_the_final_window, test_decoder_view_preserves_compute_rows_without_open_chunk_logits
  - `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +59/-3 (62 lines); hunks: -1563,22 +1563,45 @@ def test_dspark_anchors_pick_last_accepted_verify_rows(d...; -1626,3 +1649,36 @@ def test_dspark_block_expands_anchors_and_window_addressi...; symbols: test_dspark_anchors_pick_last_accepted_verify_rows, test_dspark_block_expands_anchors_and_window_addressing, test_dspark_sparse_outputs_preserve_inactive_rows_without_recompile, run
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v41.py
@@ -809,6 +809,18 @@ def _kernel_attn_sink(self):
+    def _write_global_kv(self, hidden_states, positions, requests, backend, mode):
+        if self.compressor is None:
+            return
+        latent, row_positions, row_requests = self.compressor(
+            hidden_states, self.layer_id, positions, requests, backend, mode
+        )
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py
@@ -210,6 +210,8 @@ class DeepseekV41AttentionBackend(AttentionBackend):
+    skips_incomplete_prefill_outputs = True
@@ -671,20 +673,27 @@ def _build_decoder_view(
-        its prompt keeps its last ``window`` rows and any other chunk keeps one
-        row (its logits are discarded, and one row per request keeps the
-        sampler's row contract). Decode rows are all kept. The scheduler never
-        leaves a final chunk shorter than the window, so a kept tail is the
diff -- test/runtime/test_deepseek_v41_model.py
@@ -57,6 +57,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41.py` modified +44/-21; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +19/-9; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py` modified +19/-5; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/__init__.py` modified +15/-8; `python/tokenspeed/runtime/execution/drafter/deepseek_v41_dspark.py` modified +14/-4
  - tests: `test/runtime/test_deepseek_v41_model.py` modified +54/-15; `test/runtime/test_deepseek_v41_cache.py` modified +50/-15; `tokenspeed-kernel/test/ops/test_attention_dsv41.py` modified +59/-3
- Risk and verification: The diff ships test coverage in `test/runtime/distributed/test_draft_moe_capture_global_bs.py`, `test/runtime/execution/test_draft_target_wiring.py`, `test/runtime/models/test_qwen3_moe_models.py`, `test/runtime/sampling/test_mixed_batch_sampling.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1971 - refactor(linear): select V4.1 FP8 methods during construction

- Link: https://github.com/lightseekorg/tokenspeed/pull/1971
- Status/date: merged / 2026-10-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_engram.py`, `test/runtime/test_deepseek_v41_engram.py`, `test/runtime/test_deepseek_v41_model.py`; associated commits `408f926466b0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +445/-119, 812 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/deepseek_v41.py` modified +68/-60 (128 lines); hunks: -193,7 +193,7 @@ def v41_quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor,...; -205,7 +205,7 @@ def v41_mxfp8_config(quant_config: QuantizationConfig | None...; symbols: v41_quantize_fp8, v41_mxfp8_config, _V41Fp8Config, touching `v41_quantize_fp8, v41_mxfp8_config, _V41Fp8Config`; `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +12/-30 (42 lines); hunks: -38,9 +38,9; -67,9 +67,10; symbols: forward, _load_projection_scale, engram_reduce_lane_width, __init__, touching `forward, _load_projection_scale, engram_reduce_lane_width`; `test/runtime/test_deepseek_v41_model.py` modified +106/-17 (123 lines); hunks: -77,7 +77,6; -432,23 +431,31 @@ def test_scale_expansion_then_tp_and_merged_sharding():; symbols: test_scale_expansion_then_tp_and_merged_sharding, test_reference_linear_construction_preserves_checkpoint_loading, _mix_reference, touching `test_scale_expansion_then_tp_and_merged_sharding, test_reference_linear_construction_preserves_checkpoint_loading, _mix_reference`; `test/runtime/test_deepseek_v41_engram.py` modified +13/-4 (17 lines); hunks: -46,7 +46,9; -340,7 +342,7 @@ def _model(device, quant_config):; symbols: _model, test_gate_matches_reference_and_mask_is_identity, test_quantized_projection_loader_aliases_and_real_table_metadata, touching `_model, test_gate_matches_reference_and_mask_is_identity, test_quantized_projection_loader_aliases_and_real_table_metadata`.
- Code diff details:
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +68/-60 (128 lines); hunks: -193,7 +193,7 @@ def v41_quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor,...; -205,7 +205,7 @@ def v41_mxfp8_config(quant_config: QuantizationConfig | None...; symbols: v41_quantize_fp8, v41_mxfp8_config, _V41Fp8Config
  - `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +12/-30 (42 lines); hunks: -38,9 +38,9; -67,9 +67,10; symbols: forward, _load_projection_scale, engram_reduce_lane_width, __init__
  - `test/runtime/test_deepseek_v41_model.py` modified +106/-17 (123 lines); hunks: -77,7 +77,6; -432,23 +431,31 @@ def test_scale_expansion_then_tp_and_merged_sharding():; symbols: test_scale_expansion_then_tp_and_merged_sharding, test_reference_linear_construction_preserves_checkpoint_loading, _mix_reference
  - `test/runtime/test_deepseek_v41_engram.py` modified +13/-4 (17 lines); hunks: -46,7 +46,9; -340,7 +342,7 @@ def _model(device, quant_config):; symbols: _model, test_gate_matches_reference_and_mask_is_identity, test_quantized_projection_loader_aliases_and_real_table_metadata
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v41.py
@@ -193,7 +193,7 @@ def v41_quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
-    """Return a lossless per-row scale configuration from checkpoint 32x32 FP8."""
+    """Select V4.1 dense loading/execution with lossless 1x32 runtime scales."""
@@ -205,7 +205,7 @@ def v41_mxfp8_config(quant_config: QuantizationConfig | None) -> Mxfp8Config | N
-    return Mxfp8Config(
+    return _V41Fp8Config(
@@ -214,6 +214,15 @@ def v41_mxfp8_config(quant_config: QuantizationConfig | None) -> Mxfp8Config | N
diff -- python/tokenspeed/runtime/models/deepseek_v41_engram.py
@@ -38,9 +38,9 @@
-* wkv uses the existing quantized ReplicatedLinear and its normal post-load
-  processing. Its 32x32 scales are repeated across output rows as 1x32 MXFP8
-  scales, losslessly; no weight values are expanded or requantized.
+* wkv uses the caller-selected V4.1 Linear method for checkpoint scale
+  expansion, hardware storage and post-load preparation. Engram owns the
+  embedding loader and projection aliases, not a second projection loader.
diff -- test/runtime/test_deepseek_v41_model.py
@@ -77,7 +77,6 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41.py` modified +68/-60; `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +12/-30
  - tests: `test/runtime/test_deepseek_v41_model.py` modified +106/-17; `test/runtime/test_deepseek_v41_engram.py` modified +13/-4
- Risk and verification: The diff ships test coverage in `test/runtime/test_deepseek_v41_engram.py`, `test/runtime/test_deepseek_v41_model.py`, `test/runtime/test_linear_method_construction.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
