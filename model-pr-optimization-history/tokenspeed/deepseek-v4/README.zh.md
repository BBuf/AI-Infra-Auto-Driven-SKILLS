# TokenSpeed DeepSeek V4 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/tokenspeed/runtime/configs/deepseek_v41_config.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567) |
| `python/tokenspeed/runtime/configs/deepseek_v4_config.py` | 无直接 PR 号提交 |
| `python/tokenspeed/runtime/execution/drafter/deepseek_v41_dspark.py` | [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) |
| `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` | [#940](https://github.com/lightseekorg/tokenspeed/pull/940), [#1116](https://github.com/lightseekorg/tokenspeed/pull/1116), [#1404](https://github.com/lightseekorg/tokenspeed/pull/1404), [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) |
| `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` | [#1404](https://github.com/lightseekorg/tokenspeed/pull/1404), [#1498](https://github.com/lightseekorg/tokenspeed/pull/1498), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594) |
| `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1623](https://github.com/lightseekorg/tokenspeed/pull/1623), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660), [#1663](https://github.com/lightseekorg/tokenspeed/pull/1663), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) |
| `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636) |
| `python/tokenspeed/runtime/layers/attention/deepseek_v4/draft_rounds.py` | 无直接 PR 号提交 |
| `python/tokenspeed/runtime/layers/attention/deepseek_v4/graph_buffers.py` | [#1498](https://github.com/lightseekorg/tokenspeed/pull/1498) |
| `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py` | [#207](https://github.com/lightseekorg/tokenspeed/pull/207), [#242](https://github.com/lightseekorg/tokenspeed/pull/242), [#1201](https://github.com/lightseekorg/tokenspeed/pull/1201), [#1404](https://github.com/lightseekorg/tokenspeed/pull/1404), [#1498](https://github.com/lightseekorg/tokenspeed/pull/1498) |
| `python/tokenspeed/runtime/layers/attention/deepseek_v4/slot_mappings.py` | [#1404](https://github.com/lightseekorg/tokenspeed/pull/1404) |
| `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636) |
| `python/tokenspeed/runtime/layers/attention/deepseek_v4_geometry.py` | [#1498](https://github.com/lightseekorg/tokenspeed/pull/1498), [#1612](https://github.com/lightseekorg/tokenspeed/pull/1612) |
| `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` | [#122](https://github.com/lightseekorg/tokenspeed/pull/122), [#203](https://github.com/lightseekorg/tokenspeed/pull/203), [#242](https://github.com/lightseekorg/tokenspeed/pull/242), [#339](https://github.com/lightseekorg/tokenspeed/pull/339), [#620](https://github.com/lightseekorg/tokenspeed/pull/620), [#1048](https://github.com/lightseekorg/tokenspeed/pull/1048), [#1201](https://github.com/lightseekorg/tokenspeed/pull/1201), [#1404](https://github.com/lightseekorg/tokenspeed/pull/1404), [#1498](https://github.com/lightseekorg/tokenspeed/pull/1498) |
| `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664) |
| `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py` | [#1048](https://github.com/lightseekorg/tokenspeed/pull/1048), [#1201](https://github.com/lightseekorg/tokenspeed/pull/1201), [#1404](https://github.com/lightseekorg/tokenspeed/pull/1404), [#1498](https://github.com/lightseekorg/tokenspeed/pull/1498) |
| `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` | [#940](https://github.com/lightseekorg/tokenspeed/pull/940), [#1048](https://github.com/lightseekorg/tokenspeed/pull/1048), [#1201](https://github.com/lightseekorg/tokenspeed/pull/1201), [#1498](https://github.com/lightseekorg/tokenspeed/pull/1498), [#1612](https://github.com/lightseekorg/tokenspeed/pull/1612) |
| `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1611](https://github.com/lightseekorg/tokenspeed/pull/1611), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636) |
| `python/tokenspeed/runtime/models/deepseek_v4.py` | [#122](https://github.com/lightseekorg/tokenspeed/pull/122), [#157](https://github.com/lightseekorg/tokenspeed/pull/157), [#172](https://github.com/lightseekorg/tokenspeed/pull/172), [#192](https://github.com/lightseekorg/tokenspeed/pull/192), [#203](https://github.com/lightseekorg/tokenspeed/pull/203), [#207](https://github.com/lightseekorg/tokenspeed/pull/207), [#213](https://github.com/lightseekorg/tokenspeed/pull/213), [#242](https://github.com/lightseekorg/tokenspeed/pull/242), [#254](https://github.com/lightseekorg/tokenspeed/pull/254), [#288](https://github.com/lightseekorg/tokenspeed/pull/288), [#329](https://github.com/lightseekorg/tokenspeed/pull/329), [#339](https://github.com/lightseekorg/tokenspeed/pull/339), ... (35 total) |
| `python/tokenspeed/runtime/models/deepseek_v41.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1701](https://github.com/lightseekorg/tokenspeed/pull/1701), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863), [#1971](https://github.com/lightseekorg/tokenspeed/pull/1971) |
| `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686) |
| `python/tokenspeed/runtime/models/deepseek_v41_engram.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1971](https://github.com/lightseekorg/tokenspeed/pull/1971) |
| `python/tokenspeed/runtime/models/deepseek_v41_vision.py` | [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567), [#1844](https://github.com/lightseekorg/tokenspeed/pull/1844) |
| `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` | [#940](https://github.com/lightseekorg/tokenspeed/pull/940), [#1048](https://github.com/lightseekorg/tokenspeed/pull/1048), [#1095](https://github.com/lightseekorg/tokenspeed/pull/1095), [#1201](https://github.com/lightseekorg/tokenspeed/pull/1201), [#1404](https://github.com/lightseekorg/tokenspeed/pull/1404), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1701](https://github.com/lightseekorg/tokenspeed/pull/1701) |
| `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/__init__.py` | [#940](https://github.com/lightseekorg/tokenspeed/pull/940) |
| `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/attention.py` | [#940](https://github.com/lightseekorg/tokenspeed/pull/940), [#1404](https://github.com/lightseekorg/tokenspeed/pull/1404) |
| `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py` | [#940](https://github.com/lightseekorg/tokenspeed/pull/940), [#1404](https://github.com/lightseekorg/tokenspeed/pull/1404), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686) |
| `python/tokenspeed/runtime/models/deepseek_v4_next.py` | [#1364](https://github.com/lightseekorg/tokenspeed/pull/1364), [#1701](https://github.com/lightseekorg/tokenspeed/pull/1701) |
| `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh` | [#1428](https://github.com/lightseekorg/tokenspeed/pull/1428), [#1481](https://github.com/lightseekorg/tokenspeed/pull/1481) |
| `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/build_swe_smith_dataset.patch` | [#1428](https://github.com/lightseekorg/tokenspeed/pull/1428) |
| `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/collect_outputs.py` | [#1428](https://github.com/lightseekorg/tokenspeed/pull/1428) |
| `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh` | [#1428](https://github.com/lightseekorg/tokenspeed/pull/1428), [#1481](https://github.com/lightseekorg/tokenspeed/pull/1481) |
| `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_tp4.sh` | [#1481](https://github.com/lightseekorg/tokenspeed/pull/1481) |
| `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/deepseek_v4_tokenizer.py` | [#1428](https://github.com/lightseekorg/tokenspeed/pull/1428) |
| `test/ci/eval/deepseek-v4-flash-dcp4-mtp-evalscope-gsm8k.yaml` | [#1498](https://github.com/lightseekorg/tokenspeed/pull/1498) |
| `test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml` | [#344](https://github.com/lightseekorg/tokenspeed/pull/344), [#1632](https://github.com/lightseekorg/tokenspeed/pull/1632) |
| `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k-amd.yaml` | [#1201](https://github.com/lightseekorg/tokenspeed/pull/1201) |
| `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml` | [#344](https://github.com/lightseekorg/tokenspeed/pull/344), [#1632](https://github.com/lightseekorg/tokenspeed/pull/1632) |
| `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml` | [#1748](https://github.com/lightseekorg/tokenspeed/pull/1748) |
| `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml` | [#1748](https://github.com/lightseekorg/tokenspeed/pull/1748) |
| `test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml` | [#1603](https://github.com/lightseekorg/tokenspeed/pull/1603) |
| `test/ci/ut/deepseek-v4.1-flash-pd-1p1d.yaml` | [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595) |
| `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh` | [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1603](https://github.com/lightseekorg/tokenspeed/pull/1603) |
| `test/ci_system/test_deepseek_v41_pd_launcher.py` | [#1603](https://github.com/lightseekorg/tokenspeed/pull/1603) |
| `test/cli/test_serve_smg_deepseek_v41.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567) |
| `test/runtime/distributed/test_deepseek_v41_pd_1p1d.py` | [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595) |
| `test/runtime/run_deepseek_v41_eval.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552) |
| `test/runtime/test_deepseek_v41_cache.py` | [#1498](https://github.com/lightseekorg/tokenspeed/pull/1498), [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552), [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1611](https://github.com/lightseekorg/tokenspeed/pull/1611), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660), [#1663](https://github.com/lightseekorg/tokenspeed/pull/1663), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) |
| `test/runtime/test_deepseek_v41_config.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552), [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660) |
| `test/runtime/test_deepseek_v41_dspark.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552), [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) |
| `test/runtime/test_deepseek_v41_engram.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686), [#1971](https://github.com/lightseekorg/tokenspeed/pull/1971) |
| `test/runtime/test_deepseek_v41_eval.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552) |
| `test/runtime/test_deepseek_v41_inputs.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) |
| `test/runtime/test_deepseek_v41_model.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549), [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552), [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565), [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567), [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594), [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636), [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660), [#1701](https://github.com/lightseekorg/tokenspeed/pull/1701), [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863), [#1971](https://github.com/lightseekorg/tokenspeed/pull/1971) |
| `test/runtime/test_deepseek_v41_tokenizer.py` | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549) |
| `test/runtime/test_deepseek_v41_vision.py` | [#1844](https://github.com/lightseekorg/tokenspeed/pull/1844) |
| `test/runtime/test_deepseek_v4_config.py` | [#122](https://github.com/lightseekorg/tokenspeed/pull/122), [#192](https://github.com/lightseekorg/tokenspeed/pull/192), [#203](https://github.com/lightseekorg/tokenspeed/pull/203), [#207](https://github.com/lightseekorg/tokenspeed/pull/207), [#213](https://github.com/lightseekorg/tokenspeed/pull/213), [#224](https://github.com/lightseekorg/tokenspeed/pull/224), [#242](https://github.com/lightseekorg/tokenspeed/pull/242), [#288](https://github.com/lightseekorg/tokenspeed/pull/288), [#329](https://github.com/lightseekorg/tokenspeed/pull/329), [#339](https://github.com/lightseekorg/tokenspeed/pull/339), [#503](https://github.com/lightseekorg/tokenspeed/pull/503), [#583](https://github.com/lightseekorg/tokenspeed/pull/583), ... (26 total) |
| `test/runtime/test_deepseek_v4_hopper_fp8.py` | [#1048](https://github.com/lightseekorg/tokenspeed/pull/1048) |
| `test/runtime/test_deepseek_v4_mega_moe.py` | [#339](https://github.com/lightseekorg/tokenspeed/pull/339), [#1201](https://github.com/lightseekorg/tokenspeed/pull/1201), [#1701](https://github.com/lightseekorg/tokenspeed/pull/1701) |
| `test/runtime/test_deepseek_v4_mtp_prefix_cache.py` | [#361](https://github.com/lightseekorg/tokenspeed/pull/361), [#503](https://github.com/lightseekorg/tokenspeed/pull/503), [#614](https://github.com/lightseekorg/tokenspeed/pull/614) |
| `test/runtime/test_deepseek_v4_slot_mappings.py` | [#1404](https://github.com/lightseekorg/tokenspeed/pull/1404) |

## PR 覆盖总览

- git 追溯 PR 数: 60
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 60
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-05-14 | [#122](https://github.com/lightseekorg/tokenspeed/pull/122) | merged | feat(deepseek-v4): support mixed prefill/decode batches | `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` |
| 2026-05-15 | [#157](https://github.com/lightseekorg/tokenspeed/pull/157) | merged | chore: use StreamFork in DSv4 | `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-05-17 | [#172](https://github.com/lightseekorg/tokenspeed/pull/172) | merged | feat(deepseek-v4): add persistent topk path | `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-05-20 | [#192](https://github.com/lightseekorg/tokenspeed/pull/192) | merged | feat(deepseek-v4): overlap routed and shared MoE experts | `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py` |
| 2026-05-21 | [#203](https://github.com/lightseekorg/tokenspeed/pull/203) | merged | refactor(deepseek-v4): clean up helper and kernel paths | `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` |
| 2026-05-22 | [#213](https://github.com/lightseekorg/tokenspeed/pull/213) | merged | fix(deepseek-v4): refine cache sizing and shared expert comm | `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py` |
| 2026-05-23 | [#224](https://github.com/lightseekorg/tokenspeed/pull/224) | merged | fix(deepseek-v4): enable DeepSeek-V4 unit tests on CI | `test/runtime/test_deepseek_v4_config.py` |
| 2026-05-24 | [#242](https://github.com/lightseekorg/tokenspeed/pull/242) | merged | refactor(deepseek-v4): clean up attention metadata and cache helpers | `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py` |
| 2026-05-26 | [#254](https://github.com/lightseekorg/tokenspeed/pull/254) | merged | fix(deepseek-v4): shard attn_sink for tensor-parallel > 1 | `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-05-28 | [#288](https://github.com/lightseekorg/tokenspeed/pull/288) | merged | refactor(deepseek-v4): native deep_gemm FP8 GEMM + snapshot fix | `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py` |
| 2026-06-01 | [#207](https://github.com/lightseekorg/tokenspeed/pull/207) | merged | fix(deepseek-v4): close MTP acceptance gap | `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` |
| 2026-06-02 | [#339](https://github.com/lightseekorg/tokenspeed/pull/339) | merged | perf(deepseek-v4): decode attention optimizations | `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py`, `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` |
| 2026-06-03 | [#329](https://github.com/lightseekorg/tokenspeed/pull/329) | merged | feat: reduce DeepSeek V4 prefix state snapshots with replay reuse | `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` |
| 2026-06-05 | [#356](https://github.com/lightseekorg/tokenspeed/pull/356) | merged | fix(deepseek-v4): defer mega-MoE warmup and fix MoE TP weight loading | `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-06-07 | [#375](https://github.com/lightseekorg/tokenspeed/pull/375) | merged | perf(deepseek-v4): decode kernel fusion and routing optimization | `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-06-08 | [#361](https://github.com/lightseekorg/tokenspeed/pull/361) | merged | feat(deepseek-v4): support MTP prefix cache reuse | `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_mtp_prefix_cache.py` |
| 2026-06-08 | [#344](https://github.com/lightseekorg/tokenspeed/pull/344) | merged | ci(eval): add DeepSeek V4-Flash eval CI tasks | `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml`, `test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml` |
| 2026-06-11 | [#398](https://github.com/lightseekorg/tokenspeed/pull/398) | merged | perf(deepseek-v4): pre-compile deep_gemm JIT kernels at startup | `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-06-12 | [#427](https://github.com/lightseekorg/tokenspeed/pull/427) | merged | perf(deepseek-v4): dense deep_gemm warmup M-sweep + fp8_einsum coverage | `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-06-24 | [#503](https://github.com/lightseekorg/tokenspeed/pull/503) | merged | fix(spec): remove V4 MTP special forward modes | `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py` |
| 2026-06-26 | [#529](https://github.com/lightseekorg/tokenspeed/pull/529) | merged | perf(deepseek-v4): deferred-state MHC forward for cross-layer fusion | `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-07-05 | [#583](https://github.com/lightseekorg/tokenspeed/pull/583) | merged | perf(deepseek-v4): enable MTP overlap scheduling with paged cache | `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-07-09 | [#614](https://github.com/lightseekorg/tokenspeed/pull/614) | merged | perf(deepseek-v4): sanitize SWA slot mapping once per step | `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py`, `test/runtime/test_deepseek_v4_mtp_prefix_cache.py` |
| 2026-07-31 | [#620](https://github.com/lightseekorg/tokenspeed/pull/620) | merged | feat: add DeepSeek V4 L2 KV cache offload and perf optimize. | `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` |
| 2026-08-06 | [#940](https://github.com/lightseekorg/tokenspeed/pull/940) | merged | Add DeepSeek V4 DSpark decoding and complete Flat KV replay support | `python/tokenspeed/runtime/models/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/attention.py` |
| 2026-08-11 | [#1048](https://github.com/lightseekorg/tokenspeed/pull/1048) | merged | feat(deepseek-v4): run DeepSeek-V4-Flash on Hopper (SM90 FP8 indexer + DSpark) | `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py`, `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` |
| 2026-08-14 | [#1095](https://github.com/lightseekorg/tokenspeed/pull/1095) | merged | fix(deepseek-v4): honor routed expert checkpoint dtype | `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4_dspark.py`, `test/runtime/test_deepseek_v4_config.py` |
| 2026-08-16 | [#1115](https://github.com/lightseekorg/tokenspeed/pull/1115) | merged | perf(v4): avoid prefill host synchronizations | `test/runtime/test_deepseek_v4_config.py`, `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` |
| 2026-08-17 | [#1116](https://github.com/lightseekorg/tokenspeed/pull/1116) | merged | perf(v4): avoid DSpark prefill chunk synchronizations | `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` |
| 2026-08-24 | [#1201](https://github.com/lightseekorg/tokenspeed/pull/1201) | merged | feat(deepseek-v4): add AMD MI350 support | `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` |
| 2026-08-27 | [#1251](https://github.com/lightseekorg/tokenspeed/pull/1251) | merged | feat(deepseek-v4): support fp4 indexer and ep on amd | `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-09-02 | [#1364](https://github.com/lightseekorg/tokenspeed/pull/1364) | merged | Rename DeepSeek V4 NextN model module | `python/tokenspeed/runtime/models/deepseek_v4_next.py`, `test/runtime/test_deepseek_v4_config.py` |
| 2026-09-10 | [#1428](https://github.com/lightseekorg/tokenspeed/pull/1428) | merged | test: add DeepSeek V4 Flash agentic benchmark | `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/deepseek_v4_tokenizer.py`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh` |
| 2026-09-10 | [#1404](https://github.com/lightseekorg/tokenspeed/pull/1404) | merged | perf(v4): reduce decode path overhead | `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` |
| 2026-09-10 | [#1481](https://github.com/lightseekorg/tokenspeed/pull/1481) | merged | test: update deepseek-v4-flash agentic bench | `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_tp4.sh`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh` |
| 2026-09-11 | [#1504](https://github.com/lightseekorg/tokenspeed/pull/1504) | merged | fix(deepseek v4): use BF16 LM head for DP attention | `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-09-15 | [#1549](https://github.com/lightseekorg/tokenspeed/pull/1549) | merged | feat(model): support basic deepseek v4.1 flash | `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_engram.py` |
| 2026-09-15 | [#1552](https://github.com/lightseekorg/tokenspeed/pull/1552) | merged | feat(model): support basic deepseek v4.1 flash for amd | `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `test/runtime/run_deepseek_v41_eval.py`, `test/runtime/test_deepseek_v41_eval.py` |
| 2026-09-15 | [#1567](https://github.com/lightseekorg/tokenspeed/pull/1567) | merged | feat(dsv41): support vision inputs | `python/tokenspeed/runtime/models/deepseek_v41_vision.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/configs/deepseek_v41_config.py` |
| 2026-09-15 | [#1565](https://github.com/lightseekorg/tokenspeed/pull/1565) | merged | feat: enable pcg for DeepSeek V4.1 Flash | `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `test/runtime/test_deepseek_v41_model.py` |
| 2026-09-15 | [#1498](https://github.com/lightseekorg/tokenspeed/pull/1498) | merged | feat(dcp): Add DeepSeek V4 decode context parallelism | `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` |
| 2026-09-16 | [#1594](https://github.com/lightseekorg/tokenspeed/pull/1594) | merged | feat(v41): SWA bounded replay and CED decoder narrowing | `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` |
| 2026-09-16 | [#1595](https://github.com/lightseekorg/tokenspeed/pull/1595) | merged | feat(dsv41): support basic prefill/decode disaggregation with DSpark for DeepSeek V4.1 Flash | `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` |
| 2026-09-17 | [#1603](https://github.com/lightseekorg/tokenspeed/pull/1603) | merged | ci: cover DeepSeek V4.1 Flash PD on GB300 | `test/ci_system/test_deepseek_v41_pd_launcher.py`, `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh`, `test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml` |
| 2026-09-17 | [#1611](https://github.com/lightseekorg/tokenspeed/pull/1611) | merged | refactor(dsv41): retain exactly the attention window and the compressor pair | `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py`, `test/runtime/test_deepseek_v41_cache.py` |
| 2026-09-17 | [#1612](https://github.com/lightseekorg/tokenspeed/pull/1612) | merged | refactor(dsv4): retain the compressor state window the kernel reads | `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4_geometry.py`, `test/runtime/test_deepseek_v4_config.py` |
| 2026-09-18 | [#1632](https://github.com/lightseekorg/tokenspeed/pull/1632) | merged | ci: keep non-DCP DeepSeek V4 Flash evaluations manual | `test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml`, `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml` |
| 2026-09-18 | [#1636](https://github.com/lightseekorg/tokenspeed/pull/1636) | merged | perf(dsv41): Optimize DeepSeek V4.1-flash on H20 | `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` |
| 2026-09-19 | [#1623](https://github.com/lightseekorg/tokenspeed/pull/1623) | merged | feat(amd): add gluon DSV4.1 CSA2 indexer | `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` |
| 2026-09-20 | [#1663](https://github.com/lightseekorg/tokenspeed/pull/1663) | merged | perf(dsv41): share the native decode schedule across layers in a forward | `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `test/runtime/test_deepseek_v41_cache.py` |
| 2026-09-20 | [#1660](https://github.com/lightseekorg/tokenspeed/pull/1660) | merged | perf(dsv41): run the DSpark draft window attention on selected_attention | `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` |
| 2026-09-21 | [#1664](https://github.com/lightseekorg/tokenspeed/pull/1664) | merged | perf(moe): FlashInfer CUTLASS MXFP4 experts on Hopper (W4A16 auto, W4A8 opt-in) + V4.1 log fixes | `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_engram.py`, `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-09-21 | [#1686](https://github.com/lightseekorg/tokenspeed/pull/1686) | merged | perf(dsv41): fuse the DSpark decode prologue, draft tail and sampler sync into a handful of launches | `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py`, `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` |
| 2026-09-22 | [#1701](https://github.com/lightseekorg/tokenspeed/pull/1701) | merged | refactor(kernel): clean up dsv4 mega moe apis | `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` |
| 2026-09-23 | [#1709](https://github.com/lightseekorg/tokenspeed/pull/1709) | merged | refactor(kernel): remove dsv4 pack topk router | `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py` |
| 2026-09-25 | [#1748](https://github.com/lightseekorg/tokenspeed/pull/1748) | merged | ci: turn on dspark for dsv4.1 ci | `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml`, `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml` |
| 2026-09-28 | [#1844](https://github.com/lightseekorg/tokenspeed/pull/1844) | merged | feat(dsv41): support vit batching | `python/tokenspeed/runtime/models/deepseek_v41_vision.py`, `test/runtime/test_deepseek_v41_vision.py` |
| 2026-09-29 | [#1841](https://github.com/lightseekorg/tokenspeed/pull/1841) | merged | fix(dcp): support FP8 query gathers and DeepGEMM V4 indexing | `python/tokenspeed/runtime/models/deepseek_v4.py` |
| 2026-09-30 | [#1863](https://github.com/lightseekorg/tokenspeed/pull/1863) | merged | perf(dsv41): skip decoder work for incomplete prefill chunks | `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `test/runtime/test_deepseek_v41_model.py` |
| 2026-10-04 | [#1971](https://github.com/lightseekorg/tokenspeed/pull/1971) | merged | refactor(linear): select V4.1 FP8 methods during construction | `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_engram.py`, `test/runtime/test_deepseek_v41_model.py` |

## 逐 PR diff 审计卡

### PR #122 - feat(deepseek-v4): support mixed prefill/decode batches

- 链接: https://github.com/lightseekorg/tokenspeed/pull/122
- 状态/时间: merged / 2026-05-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py`；关联提交 `4d3b7dc8eb5e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 32 个文件，+5315/-432，可读 patch 3800 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +2438/-335 (2773 lines)；`python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +444/-13 (457 lines); hunks: -14,6 +14,9; -56,10 +59,167 @@ def _cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:; symbols: _cu_seqlens, _decode_positions_from_metadata, _refresh_decode_indexer_plan_cache, _refresh_decode_indexer_schedule_metadata，涉及 `_cu_seqlens, _decode_positions_from_metadata, _refresh_decode_indexer_plan_cache`；`python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +48/-1 (49 lines); hunks: -2203,11 +2203,18 @@ def _deepseek_v4_compute_global_topk_indices_and_lens_ke...; -2245,6 +2252,7 @@ def deepseek_v4_compute_global_topk_indices_and_lens(; symbols: _deepseek_v4_compute_global_topk_indices_and_lens_kernel, deepseek_v4_compute_global_topk_indices_and_lens，涉及 `_deepseek_v4_compute_global_topk_indices_and_lens_kernel, deepseek_v4_compute_global_topk_indices_and_lens`；`python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +35/-0 (35 lines); hunks: -234,6 +234,15 @@ class DeepseekV4ForwardMetadata:; -257,9 +266,35 @@ class DeepseekV4ForwardMetadata:; symbols: DeepseekV4ForwardMetadata, decode_req_count, decode_token_count, _use_decode_compressed_slot_cache，涉及 `DeepseekV4ForwardMetadata, decode_req_count, decode_token_count`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +2438/-335 (2773 lines)
  - `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +444/-13 (457 lines); hunks: -14,6 +14,9; -56,10 +59,167 @@ def _cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:; symbols: _cu_seqlens, _decode_positions_from_metadata, _refresh_decode_indexer_plan_cache, _refresh_decode_indexer_schedule_metadata
  - `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +48/-1 (49 lines); hunks: -2203,11 +2203,18 @@ def _deepseek_v4_compute_global_topk_indices_and_lens_ke...; -2245,6 +2252,7 @@ def deepseek_v4_compute_global_topk_indices_and_lens(; symbols: _deepseek_v4_compute_global_topk_indices_and_lens_kernel, deepseek_v4_compute_global_topk_indices_and_lens
  - `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +35/-0 (35 lines); hunks: -234,6 +234,15 @@ class DeepseekV4ForwardMetadata:; -257,9 +266,35 @@ class DeepseekV4ForwardMetadata:; symbols: DeepseekV4ForwardMetadata, decode_req_count, decode_token_count, _use_decode_compressed_slot_cache
  - `test/runtime/test_deepseek_v4_config.py` modified +1253/-42 (1295 lines); hunks: -12,12 +12,17; -33,21 +38,26; symbols: test_config_registry, test_forward_mode_mixed_predicate, test_cuda_graph_group_table_padding_uses_dummy_page_rows, test_cuda_graph_replay_keeps_idle_actual_bs_with_padded_group_tables
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py
@@ -14,6 +14,9 @@
+from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
+    deepseek_v4_indexer_decode_metadata_compute,
+)
@@ -56,10 +59,167 @@ def _cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
+def _decode_positions_from_metadata(
+    metadata: DeepseekV4ForwardMetadata,
diff -- python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py
@@ -2203,11 +2203,18 @@ def _deepseek_v4_compute_global_topk_indices_and_lens_kernel(
+    is_valid_token_ptr,
+    has_valid_token: tl.constexpr,
+    if has_valid_token:
+        is_valid_token = tl.load(is_valid_token_ptr + token_idx)
+        if not is_valid_token:
+            tl.store(topk_lens_ptr + token_idx, 0)
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py
@@ -234,6 +234,15 @@ class DeepseekV4ForwardMetadata:
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +2438/-335; `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +444/-13; `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +48/-1; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +35/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/deepseek_v4.py` added +115/-0
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +1253/-42
- 验证与风险: diff 自带测试面 `test/runtime/kernels/test_trtllm_wrapper.py`, `test/runtime/test_cli_config_compat.py`, `test/runtime/test_deepseek_v4_attention_ops.py`, `test/runtime/test_deepseek_v4_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #157 - chore: use StreamFork in DSv4

- 链接: https://github.com/lightseekorg/tokenspeed/pull/157
- 状态/时间: merged / 2026-05-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`；关联提交 `be3efc5efaec`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+18/-47，可读 patch 108 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +18/-47 (65 lines); hunks: -34,7 +34,7; -126,6 +126,7; symbols: _deepseek_v4_indexer_decode_metadata, _deepseek_v4_sparse_attn_indexer, _deepseek_v4_maybe_execute_in_parallel, __init__，涉及 `_deepseek_v4_indexer_decode_metadata, _deepseek_v4_sparse_attn_indexer, _deepseek_v4_maybe_execute_in_parallel`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +18/-47 (65 lines); hunks: -34,7 +34,7; -126,6 +126,7; symbols: _deepseek_v4_indexer_decode_metadata, _deepseek_v4_sparse_attn_indexer, _deepseek_v4_maybe_execute_in_parallel, __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -34,7 +34,7 @@
-from typing import Any, Callable, Iterable, Optional, Tuple
+from typing import Any, Iterable, Optional, Tuple
@@ -126,6 +126,7 @@
+from tokenspeed.runtime.utils.cuda_stream import StreamFork
@@ -1943,10 +1944,9 @@ def _deepseek_v4_indexer_decode_metadata(
-    # here, the per-layer
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +18/-47
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #172 - feat(deepseek-v4): add persistent topk path

- 链接: https://github.com/lightseekorg/tokenspeed/pull/172
- 状态/时间: merged / 2026-05-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`；关联提交 `a71bddc3a66a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+1775/-1，可读 patch 1994 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +89/-0 (89 lines); hunks: -635,6 +635,7 @@ def _deepseek_v4_indexer_topk_from_cache_batched(; -702,6 +703,15 @@ def _deepseek_v4_indexer_topk_from_cache_batched(; symbols: _deepseek_v4_indexer_topk_from_cache_batched, _deepseek_v4_indexer_topk_from_logits, _deepseek_v4_indexer_topk_from_cache_deepgemm_decode，涉及 `_deepseek_v4_indexer_topk_from_cache_batched, _deepseek_v4_indexer_topk_from_logits, _deepseek_v4_indexer_topk_from_cache_deepgemm_decode`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +89/-0 (89 lines); hunks: -635,6 +635,7 @@ def _deepseek_v4_indexer_topk_from_cache_batched(; -702,6 +703,15 @@ def _deepseek_v4_indexer_topk_from_cache_batched(; symbols: _deepseek_v4_indexer_topk_from_cache_batched, _deepseek_v4_indexer_topk_from_logits, _deepseek_v4_indexer_topk_from_cache_deepgemm_decode
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -635,6 +635,7 @@ def _deepseek_v4_indexer_topk_from_cache_batched(
+    persistent_topk_workspace: torch.Tensor | None = None,
@@ -702,6 +703,15 @@ def _deepseek_v4_indexer_topk_from_cache_batched(
+        if _deepseek_v4_try_persistent_topk(
+            logits,
+            compressed_lens,
+            topk,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +89/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_attention_ops.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #192 - feat(deepseek-v4): overlap routed and shared MoE experts

- 链接: https://github.com/lightseekorg/tokenspeed/pull/192
- 状态/时间: merged / 2026-05-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py`；关联提交 `4e51bed84091`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+393/-88，可读 patch 649 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +136/-87 (223 lines); hunks: -77,6 +77,7; -3205,21 +3206,20 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`test/runtime/test_deepseek_v4_config.py` modified +257/-1 (258 lines); hunks: -1,6 +1,6; -12,6 +12,7; symbols: test_forward_mode_mixed_predicate, _bind_deepseek_v4_moe_methods, _make_fake_deepseek_v4_moe, select_experts，涉及 `test_forward_mode_mixed_predicate, _bind_deepseek_v4_moe_methods, _make_fake_deepseek_v4_moe`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +136/-87 (223 lines); hunks: -77,6 +77,7; -3205,21 +3206,20 @@ def __init__(; symbols: __init__, forward
  - `test/runtime/test_deepseek_v4_config.py` modified +257/-1 (258 lines); hunks: -1,6 +1,6; -12,6 +12,7; symbols: test_forward_mode_mixed_predicate, _bind_deepseek_v4_moe_methods, _make_fake_deepseek_v4_moe, select_experts
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -77,6 +77,7 @@
+from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
@@ -3205,21 +3206,20 @@ def __init__(
-        is_shared_expert: bool = False,
-        tp = mapping.moe if is_shared_expert else mapping.dense
+        tp = mapping.dense
-            tp_rank=tp.tp_ep_rank if is_shared_expert else tp.tp_rank,
diff -- test/runtime/test_deepseek_v4_config.py
@@ -1,6 +1,6 @@
-from types import SimpleNamespace
+from types import MethodType, SimpleNamespace
@@ -12,6 +12,7 @@
+from tokenspeed.runtime.distributed import Mapping
@@ -44,6 +45,7 @@
+    DeepseekV4MoE,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +136/-87
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +257/-1
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #203 - refactor(deepseek-v4): clean up helper and kernel paths

- 链接: https://github.com/lightseekorg/tokenspeed/pull/203
- 状态/时间: merged / 2026-05-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py`；关联提交 `63f64936377e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 16 个文件，+2857/-3486，可读 patch 5414 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +323/-2524 (2847 lines)；`python/tokenspeed/runtime/models/deepseek_v4.py` modified +206/-538 (744 lines); hunks: -27,12 +27,7; -47,25 +42,44; symbols: is_blackwell, is_sm90_supported, _deepseek_v4_router_gemm, _deepseek_v4_bf16_linear_fp32，涉及 `is_blackwell, is_sm90_supported, _deepseek_v4_router_gemm`；`python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +114/-121 (235 lines); hunks: -14,51 +14,47; -158,9 +154,7 @@ def _refresh_decode_indexer_schedule_metadata(; symbols: _swa_block_table, _cu_seqlens, _decode_positions_from_metadata, _refresh_decode_indexer_schedule_metadata，涉及 `_swa_block_table, _cu_seqlens, _decode_positions_from_metadata`；`python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +57/-53 (110 lines); hunks: -20,10 +20,15; -32,37 +37,40; symbols: DeepseekV4CacheLayout, swa_token_stride, swa_scale_dim, swa_row_bytes，涉及 `DeepseekV4CacheLayout, swa_token_stride, swa_scale_dim`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +323/-2524 (2847 lines)
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +206/-538 (744 lines); hunks: -27,12 +27,7; -47,25 +42,44; symbols: is_blackwell, is_sm90_supported, _deepseek_v4_router_gemm, _deepseek_v4_bf16_linear_fp32
  - `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +114/-121 (235 lines); hunks: -14,51 +14,47; -158,9 +154,7 @@ def _refresh_decode_indexer_schedule_metadata(; symbols: _swa_block_table, _cu_seqlens, _decode_positions_from_metadata, _refresh_decode_indexer_schedule_metadata
  - `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +57/-53 (110 lines); hunks: -20,10 +20,15; -32,37 +37,40; symbols: DeepseekV4CacheLayout, swa_token_stride, swa_scale_dim, swa_row_bytes
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/deepseek_v4.py` modified +1662/-1 (1663 lines); hunks: -23,7 +23,1668; symbols: _deepseek_v4_mxfp4_e2m1_nibble, _deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4_kernel, deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4, _deepseek_v4_fused_sparse_compress_cache_kernel
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -27,12 +27,7 @@
-import glob
-import importlib
-import os
-import site
-import sys
@@ -47,25 +42,44 @@
diff -- python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py
@@ -14,51 +14,47 @@
+from tokenspeed_kernel.ops.attention.flash_mla import (
+    flash_mla_sparse_fwd,
+    flash_mla_with_kvcache,
+    get_mla_metadata,
+)
+from tokenspeed_kernel.registry import error_fn
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py
@@ -20,10 +20,15 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +323/-2524; `python/tokenspeed/runtime/models/deepseek_v4.py` modified +206/-538; `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +114/-121; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +57/-53; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/deepseek_v4.py` modified +1662/-1; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/cuda/deepseek_v4.py` added +57/-0
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +109/-85
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_attention_ops.py`, `test/runtime/test_deepseek_v4_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #213 - fix(deepseek-v4): refine cache sizing and shared expert comm

- 链接: https://github.com/lightseekorg/tokenspeed/pull/213
- 状态/时间: merged / 2026-05-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py`；关联提交 `e0754f0b571d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+281/-181，可读 patch 676 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +136/-65 (201 lines); hunks: -14,7 +14,8; -34,13 +35,15; symbols: cache_cell_size, _estimate_deepseek_v4_cache_bytes, _deepseek_v4_cache_group_page_bytes, profile_deepseek_v4_max_num_pages，涉及 `cache_cell_size, _estimate_deepseek_v4_cache_bytes, _deepseek_v4_cache_group_page_bytes`；`python/tokenspeed/runtime/models/deepseek_v4.py` modified +28/-62 (90 lines); hunks: -82,11 +82,7; -3388,22 +3384,29 @@ def _make_topk_output(; symbols: _make_topk_output, _forward_shared_experts, forward_mega_moe，涉及 `_make_topk_output, _forward_shared_experts, forward_mega_moe`；`test/runtime/test_deepseek_v4_config.py` modified +45/-51 (96 lines); hunks: -324,34 +324,36 @@ def __call__(self, states):; -360,48 +362,40 @@ class SharedExperts:; symbols: __call__, FakeCommManager, pre_dense_comm, post_dense_comm，涉及 `__call__, FakeCommManager, pre_dense_comm`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +136/-65 (201 lines); hunks: -14,7 +14,8; -34,13 +35,15; symbols: cache_cell_size, _estimate_deepseek_v4_cache_bytes, _deepseek_v4_cache_group_page_bytes, profile_deepseek_v4_max_num_pages
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +28/-62 (90 lines); hunks: -82,11 +82,7; -3388,22 +3384,29 @@ def _make_topk_output(; symbols: _make_topk_output, _forward_shared_experts, forward_mega_moe
  - `test/runtime/test_deepseek_v4_config.py` modified +45/-51 (96 lines); hunks: -324,34 +324,36 @@ def __call__(self, states):; -360,48 +362,40 @@ class SharedExperts:; symbols: __call__, FakeCommManager, pre_dense_comm, post_dense_comm
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py
@@ -14,7 +14,8 @@
-from typing import Any
+from fractions import Fraction
+from typing import Any, Sequence
@@ -34,13 +35,15 @@
+    PagedCacheGroupSpec,
+from tokenspeed.runtime.utils.common import ceil_div
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -82,11 +82,7 @@
-from tokenspeed.runtime.distributed.comm_ops import (
-    all_reduce,
-    token_all_gather,
-    token_reduce_scatter,
-)
+from tokenspeed.runtime.distributed.comm_ops import all_reduce
diff -- test/runtime/test_deepseek_v4_config.py
@@ -324,34 +324,36 @@ def __call__(self, states):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +136/-65; `python/tokenspeed/runtime/models/deepseek_v4.py` modified +28/-62
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +45/-51
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`, `test/runtime/test_v4_sliding_window_groups_smoke.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #224 - fix(deepseek-v4): enable DeepSeek-V4 unit tests on CI

- 链接: https://github.com/lightseekorg/tokenspeed/pull/224
- 状态/时间: merged / 2026-05-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/runtime/test_deepseek_v4_config.py`；关联提交 `2cce4a3b7963`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+24/-0，可读 patch 61 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/test_deepseek_v4_config.py` modified +24/-0 (24 lines); hunks: -1,8 +1,16; -14,6 +22,7; symbols: test_fp8_quantization_config, test_deepseek_v4_fp8_linear_deep_gemm_pads_partial_n_block, test_deepseek_v4_fused_hash_topk_matches_reference, test_deepseek_v4_fp8_activation_quant_matches_reference，涉及 `test_fp8_quantization_config, test_deepseek_v4_fp8_linear_deep_gemm_pads_partial_n_block, test_deepseek_v4_fused_hash_topk_matches_reference`。
- 代码 diff 细节:
  - `test/runtime/test_deepseek_v4_config.py` modified +24/-0 (24 lines); hunks: -1,8 +1,16; -14,6 +22,7; symbols: test_fp8_quantization_config, test_deepseek_v4_fp8_linear_deep_gemm_pads_partial_n_block, test_deepseek_v4_fused_hash_topk_matches_reference, test_deepseek_v4_fp8_activation_quant_matches_reference
- 关键代码摘录:

```diff
diff -- test/runtime/test_deepseek_v4_config.py
@@ -1,8 +1,16 @@
+import os
+import sys
+# CI Registration (parsed via AST, runtime no-op)
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+from ci_system.ci_register import register_cuda_ci
+register_cuda_ci(est_time=30, suite="runtime-1gpu")
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +24/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #242 - refactor(deepseek-v4): clean up attention metadata and cache helpers

- 链接: https://github.com/lightseekorg/tokenspeed/pull/242
- 状态/时间: merged / 2026-05-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py`；关联提交 `c5cc8c4ec74c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+1598/-3483，可读 patch 2547 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +847/-2622 (3469 lines)；`python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +148/-127 (275 lines); hunks: -38,6 +38,9; -47,7 +50,7; symbols: _refresh_decode_indexer_plan_cache, _refresh_decode_indexer_schedule_metadata，涉及 `_refresh_decode_indexer_plan_cache, _refresh_decode_indexer_schedule_metadata`；`python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: DeepseekV4IndexerPrefillChunkPlan, DeepseekV4IndexerPrefillMetadata, max_gather_rows, empty，涉及 `DeepseekV4IndexerPrefillChunkPlan, DeepseekV4IndexerPrefillMetadata, max_gather_rows`；`python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +3/-128 (131 lines); hunks: -11,8 +11,8; -49,9 +49,6; symbols: _fp8_e4m3_pow2_bytes, _fp8_e4m3_pow2_dequant_rows, _e2m1_nibbles, _e2m1_values，涉及 `_fp8_e4m3_pow2_bytes, _fp8_e4m3_pow2_dequant_rows, _e2m1_nibbles`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +847/-2622 (3469 lines)
  - `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +148/-127 (275 lines); hunks: -38,6 +38,9; -47,7 +50,7; symbols: _refresh_decode_indexer_plan_cache, _refresh_decode_indexer_schedule_metadata
  - `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: DeepseekV4IndexerPrefillChunkPlan, DeepseekV4IndexerPrefillMetadata, max_gather_rows, empty
  - `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +3/-128 (131 lines); hunks: -11,8 +11,8; -49,9 +49,6; symbols: _fp8_e4m3_pow2_bytes, _fp8_e4m3_pow2_dequant_rows, _e2m1_nibbles, _e2m1_values
  - `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +37/-73 (110 lines); hunks: -437,28 +437,9 @@ def _group_slot_mapping_from_raw(; -477,35 +458,6 @@ class DeepseekV4ForwardMetadata:; symbols: _group_slot_mapping_from_raw, DeepseekV4ForwardMetadata, DeepseekV4CacheMetadata, decode_req_count
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py
@@ -38,6 +38,9 @@
+from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
+    DeepseekV4ForwardMetadata,
+)
@@ -47,7 +50,7 @@
-    DeepseekV4ForwardMetadata,
+    DeepseekV4CacheMetadata,
diff -- python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py
@@ -0,0 +1,157 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py
@@ -11,8 +11,8 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +847/-2622; `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +148/-127; `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py` added +157/-0; `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +3/-128; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +37/-73
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +336/-528
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_attention_ops.py`, `test/runtime/test_deepseek_v4_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #254 - fix(deepseek-v4): shard attn_sink for tensor-parallel > 1

- 链接: https://github.com/lightseekorg/tokenspeed/pull/254
- 状态/时间: merged / 2026-05-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`；关联提交 `4ddb95723feb`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+26/-13，可读 patch 81 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +26/-13 (39 lines); hunks: -3159,12 +3159,15 @@ def __init__(; -3177,11 +3180,21 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +26/-13 (39 lines); hunks: -3159,12 +3159,15 @@ def __init__(; -3177,11 +3180,21 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -3159,12 +3159,15 @@ def __init__(
-        if self.num_heads % mapping.attn.tp_size != 0:
+        tp_rank = mapping.attn.tp_rank
+        tp_size = mapping.attn.tp_size
+        tp_group = mapping.attn.tp_group
+        if self.num_heads % tp_size != 0:
-                f"by attn_tp_size={mapping.attn.tp_size}"
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +26/-13
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #288 - refactor(deepseek-v4): native deep_gemm FP8 GEMM + snapshot fix

- 链接: https://github.com/lightseekorg/tokenspeed/pull/288
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py`；关联提交 `1e634e34fc32`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 13 个文件，+333/-227，可读 patch 879 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +6/-95 (101 lines); hunks: -63,14 +63,10; -154,12 +150,6 @@ def _dequant_fp8_weight(layer: nn.Module, shape: tuple[int,...; symbols: _dequant_fp8_weight, _fp8_act_quant_dequant, _fp8_linear，涉及 `_dequant_fp8_weight, _fp8_act_quant_dequant, _fp8_linear`；`test/runtime/test_deepseek_v4_config.py` modified +0/-24 (24 lines); hunks: -85,7 +85,6; -2969,29 +2968,6 @@ def test_deepseek_v4_fused_hash_topk_matches_reference(se...; symbols: test_deepseek_v4_fused_hash_topk_matches_reference, test_deepseek_v4_fp8_activation_quant_matches_reference, test_packed_topk_router_logits_recover_weights_after_softmax，涉及 `test_deepseek_v4_fused_hash_topk_matches_reference, test_deepseek_v4_fp8_activation_quant_matches_reference, test_packed_topk_router_logits_recover_weights_after_softmax`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +6/-95 (101 lines); hunks: -63,14 +63,10; -154,12 +150,6 @@ def _dequant_fp8_weight(layer: nn.Module, shape: tuple[int,...; symbols: _dequant_fp8_weight, _fp8_act_quant_dequant, _fp8_linear
  - `test/runtime/test_deepseek_v4_config.py` modified +0/-24 (24 lines); hunks: -85,7 +85,6; -2969,29 +2968,6 @@ def test_deepseek_v4_fused_hash_topk_matches_reference(se...; symbols: test_deepseek_v4_fused_hash_topk_matches_reference, test_deepseek_v4_fp8_activation_quant_matches_reference, test_packed_topk_router_logits_recover_weights_after_softmax
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -63,14 +63,10 @@
-from tokenspeed_kernel.thirdparty.trtllm import (
-    per_token_group_quant_8bit as trtllm_fp8_quantize_1x128,
-)
-    DEEPSEEK_V4_FP8_BLOCK_SIZE,
@@ -154,12 +150,6 @@ def _dequant_fp8_weight(layer: nn.Module, shape: tuple[int, ...]) -> torch.Tenso
-    cache = getattr(layer, "_deepseek_v4_dequant_cache", None)
diff -- test/runtime/test_deepseek_v4_config.py
@@ -85,7 +85,6 @@
-    _fp8_act_quant_dequant,
@@ -2969,29 +2968,6 @@ def test_deepseek_v4_fused_hash_topk_matches_reference(self):
-    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
-    @unittest.skipUnless(
-        current_platform().is_blackwell_plus,
-        "trtllm fp8_quantize_1x128 kernel layout requires sm100+",
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +6/-95
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +0/-24
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`, `test/runtime/test_mxfp4_weights.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #207 - fix(deepseek-v4): close MTP acceptance gap

- 链接: https://github.com/lightseekorg/tokenspeed/pull/207
- 状态/时间: merged / 2026-06-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py`；关联提交 `a8a2d0ead438`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 26 个文件，+3120/-193，可读 patch 4718 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +494/-46 (540 lines); hunks: -34,6 +34,7; -60,6 +61,15; symbols: _compressed_block_table_base_offsets, _decode_positions_from_metadata, _refresh_decode_indexer_plan_cache, __init__，涉及 `_compressed_block_table_base_offsets, _decode_positions_from_metadata, _refresh_decode_indexer_plan_cache`；`python/tokenspeed/runtime/models/deepseek_v4.py` modified +148/-22 (170 lines); hunks: -72,6 +72,7; -81,6 +82,7; symbols: _deepseek_v4_metadata_matches_tokens, _deepseek_v4_indexer_token_split, _deepseek_v4_forward_metadata, _dequant_fp8_weight，涉及 `_deepseek_v4_metadata_matches_tokens, _deepseek_v4_indexer_token_split, _deepseek_v4_forward_metadata`；`python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +129/-12 (141 lines); hunks: -15,7 +15,7; -401,6 +401,21 @@ def _safe_page_ids(; symbols: _safe_page_ids, _expand_group_values_for_tokens, _group_slot_mapping_from_raw，涉及 `_safe_page_ids, _expand_group_values_for_tokens, _group_slot_mapping_from_raw`；`python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py` modified +2/-0 (2 lines); hunks: -17,6 +17,7; -137,6 +138,7 @@ class DeepseekV4ForwardMetadata:; symbols: DeepseekV4ForwardMetadata，涉及 `DeepseekV4ForwardMetadata`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +494/-46 (540 lines); hunks: -34,6 +34,7; -60,6 +61,15; symbols: _compressed_block_table_base_offsets, _decode_positions_from_metadata, _refresh_decode_indexer_plan_cache, __init__
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +148/-22 (170 lines); hunks: -72,6 +72,7; -81,6 +82,7; symbols: _deepseek_v4_metadata_matches_tokens, _deepseek_v4_indexer_token_split, _deepseek_v4_forward_metadata, _dequant_fp8_weight
  - `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +129/-12 (141 lines); hunks: -15,7 +15,7; -401,6 +401,21 @@ def _safe_page_ids(; symbols: _safe_page_ids, _expand_group_values_for_tokens, _group_slot_mapping_from_raw
  - `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py` modified +2/-0 (2 lines); hunks: -17,6 +17,7; -137,6 +138,7 @@ class DeepseekV4ForwardMetadata:; symbols: DeepseekV4ForwardMetadata
  - `test/runtime/test_deepseek_v4_config.py` modified +1186/-6 (1192 lines); hunks: -35,12 +35,22; -62,6 +72,9; symbols: test_forward_mode_mixed_predicate, test_model_runner_forwards_supported_spec_step_idx, ModelWithSpecStep, __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py
@@ -34,6 +34,7 @@
+    v4_compressed_kv_group_id,
@@ -60,6 +61,15 @@
+def _compressed_block_table_base_offsets(
+    metadata: DeepseekV4ForwardMetadata,
+    compress_ratio: int,
+) -> torch.Tensor | None:
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -72,6 +72,7 @@
+    v4_compressed_kv_group_id,
@@ -81,6 +82,7 @@
+from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
@@ -100,6 +102,7 @@
+    _mask_invalid_graph_tokens,
@@ -144,6 +147,53 @@
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py
@@ -15,7 +15,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +494/-46; `python/tokenspeed/runtime/models/deepseek_v4.py` modified +148/-22; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +129/-12; `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py` modified +2/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/deepseek_v4.py` modified +16/-1
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +1186/-6
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_attention_ops.py`, `test/runtime/test_deepseek_v4_config.py`, `test/runtime/test_logits_processor.py`, `test/runtime/test_resolve_architecture.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #339 - perf(deepseek-v4): decode attention optimizations

- 链接: https://github.com/lightseekorg/tokenspeed/pull/339
- 状态/时间: merged / 2026-06-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py`, `test/runtime/test_deepseek_v4_mega_moe.py`；关联提交 `b80268f3edf9`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+584/-145，可读 patch 1038 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +337/-86 (423 lines); hunks: -27,6 +27,7; -94,8 +95,8; symbols: __init__, forward, warmup_jit_variants, DeepseekV4MoE，涉及 `__init__, forward, warmup_jit_variants`；`python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +3/-51 (54 lines); hunks: -46,6 +46,9; -165,27 +168,6 @@ def _apply_gptj_rope_tail_rows(; symbols: _apply_gptj_rope_tail_rows, _apply_inverse_gptj_rope_tail, _fp8_e4m3_pow2_bytes, _deepseek_v4_hadamard_rotate，涉及 `_apply_gptj_rope_tail_rows, _apply_inverse_gptj_rope_tail, _fp8_e4m3_pow2_bytes`；`python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +8/-1 (9 lines); hunks: -1864,6 +1864,12 @@ def init_forward_metadata_capture_cuda_graph(; -1875,8 +1881,9 @@ def init_forward_metadata_capture_cuda_graph(; symbols: init_forward_metadata_capture_cuda_graph，涉及 `init_forward_metadata_capture_cuda_graph`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/deepseek_v4.py` modified +179/-0 (179 lines); hunks: -1789,3 +1789,182 @@ def deepseek_v4_indexer_decode_metadata_compute(; symbols: deepseek_v4_indexer_decode_metadata_compute, _deepseek_v4_fused_inv_rope_fp8_quant_per_head, deepseek_v4_fused_inv_rope_fp8_quant，涉及 `deepseek_v4_indexer_decode_metadata_compute, _deepseek_v4_fused_inv_rope_fp8_quant_per_head, deepseek_v4_fused_inv_rope_fp8_quant`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +337/-86 (423 lines); hunks: -27,6 +27,7; -94,8 +95,8; symbols: __init__, forward, warmup_jit_variants, DeepseekV4MoE
  - `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +3/-51 (54 lines); hunks: -46,6 +46,9; -165,27 +168,6 @@ def _apply_gptj_rope_tail_rows(; symbols: _apply_gptj_rope_tail_rows, _apply_inverse_gptj_rope_tail, _fp8_e4m3_pow2_bytes, _deepseek_v4_hadamard_rotate
  - `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +8/-1 (9 lines); hunks: -1864,6 +1864,12 @@ def init_forward_metadata_capture_cuda_graph(; -1875,8 +1881,9 @@ def init_forward_metadata_capture_cuda_graph(; symbols: init_forward_metadata_capture_cuda_graph
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/deepseek_v4.py` modified +179/-0 (179 lines); hunks: -1789,3 +1789,182 @@ def deepseek_v4_indexer_decode_metadata_compute(; symbols: deepseek_v4_indexer_decode_metadata_compute, _deepseek_v4_fused_inv_rope_fp8_quant_per_head, deepseek_v4_fused_inv_rope_fp8_quant
  - `test/runtime/test_deepseek_v4_mega_moe.py` modified +17/-0 (17 lines); hunks: -28,6 +28,7 @@ def test_weight_loader_places_expert_shards(self):; -51,6 +52,22 @@ def test_weight_loader_places_expert_shards(self):; symbols: test_weight_loader_places_expert_shards, test_init_stores_swiglu_limit
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -27,6 +27,7 @@
+import os
@@ -94,8 +95,8 @@
+    deepseek_v4_fused_inv_rope_fp8_quant,
-    deepseek_v4_inv_rope_grouped,
@@ -2098,6 +2099,7 @@ def __init__(
+        swiglu_limit: float | None,
diff -- python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py
@@ -46,6 +46,9 @@
+from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
+    deepseek_v4_fused_inv_rope_fp8_quant,
+)
@@ -165,27 +168,6 @@ def _apply_gptj_rope_tail_rows(
-def _apply_inverse_gptj_rope_tail(
-    x: torch.Tensor,
diff -- python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py
@@ -1864,6 +1864,12 @@ def init_forward_metadata_capture_cuda_graph(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +337/-86; `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +3/-51; `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +8/-1; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/deepseek_v4.py` modified +179/-0
  - tests: `test/runtime/test_deepseek_v4_mega_moe.py` modified +17/-0; `test/runtime/test_deepseek_v4_config.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`, `test/runtime/test_deepseek_v4_mega_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #329 - feat: reduce DeepSeek V4 prefix state snapshots with replay reuse

- 链接: https://github.com/lightseekorg/tokenspeed/pull/329
- 状态/时间: merged / 2026-06-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py`；关联提交 `11767316d0b7`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 19 个文件，+1126/-204，可读 patch 1995 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +46/-7 (53 lines); hunks: -13,6 +13,7; -511,10 +512,17 @@ def compressed_block_table(; symbols: compressed_block_table, safe_page_ids, DeepseekV4TokenToKVPool, __init__，涉及 `compressed_block_table, safe_page_ids, DeepseekV4TokenToKVPool`；`python/tokenspeed/runtime/models/deepseek_v4.py` modified +23/-25 (48 lines); hunks: -2784,17 +2784,18 @@ def forward(; -3210,22 +3211,19 @@ def forward(; symbols: forward，涉及 `forward`；`python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +7/-10 (17 lines); hunks: -757,15 +757,14 @@ def _update_decode_swa_metadata(; -1192,11 +1191,9 @@ def _prefill_workspace(; symbols: _update_decode_swa_metadata, _prefill_workspace，涉及 `_update_decode_swa_metadata, _prefill_workspace`；`test/runtime/test_deepseek_v4_config.py` modified +102/-23 (125 lines); hunks: -2,6 +2,8; -70,6 +72,8; symbols: _make_deepseek_v4_forward_metadata, _v4_compressed_kv_tables, _mhc_sinkhorn_reference，涉及 `_make_deepseek_v4_forward_metadata, _v4_compressed_kv_tables, _mhc_sinkhorn_reference`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +46/-7 (53 lines); hunks: -13,6 +13,7; -511,10 +512,17 @@ def compressed_block_table(; symbols: compressed_block_table, safe_page_ids, DeepseekV4TokenToKVPool, __init__
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +23/-25 (48 lines); hunks: -2784,17 +2784,18 @@ def forward(; -3210,22 +3211,19 @@ def forward(; symbols: forward
  - `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +7/-10 (17 lines); hunks: -757,15 +757,14 @@ def _update_decode_swa_metadata(; -1192,11 +1191,9 @@ def _prefill_workspace(; symbols: _update_decode_swa_metadata, _prefill_workspace
  - `test/runtime/test_deepseek_v4_config.py` modified +102/-23 (125 lines); hunks: -2,6 +2,8; -70,6 +72,8; symbols: _make_deepseek_v4_forward_metadata, _v4_compressed_kv_tables, _mhc_sinkhorn_reference
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py
@@ -13,6 +13,7 @@
+import logging
@@ -511,10 +512,17 @@ def compressed_block_table(
+        if compress_ratio <= 1:
+            return self.block_table
-        return table if table is not None else self.block_table
+        if table is None:
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -2784,17 +2784,18 @@ def forward(
-                if state_block_table is not None:
-                    state_slot_mapping = _group_slot_mapping_from_raw(
-                        positions,
-                        metadata.token_to_req_indices[: positions.numel()],
-                        state_block_table,
-                        state_block_size,
diff -- python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py
@@ -757,15 +757,14 @@ def _update_decode_swa_metadata(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +46/-7; `python/tokenspeed/runtime/models/deepseek_v4.py` modified +23/-25; `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +7/-10
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +102/-23
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`, `test/runtime/test_v4_sliding_window_groups_smoke.py`, `tokenspeed-scheduler/tests/cpp/test_paged_cache_replay.cpp`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #356 - fix(deepseek-v4): defer mega-MoE warmup and fix MoE TP weight loading

- 链接: https://github.com/lightseekorg/tokenspeed/pull/356
- 状态/时间: merged / 2026-06-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`；关联提交 `c186ce616b9f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+59/-31，可读 patch 171 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +32/-21 (53 lines); hunks: -2243,17 +2243,19 @@ def finalize_weights(self) -> None:; -4117,7 +4119,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: finalize_weights, get_symm_buffer, load_weights, post_load_weights，涉及 `finalize_weights, get_symm_buffer, load_weights`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +32/-21 (53 lines); hunks: -2243,17 +2243,19 @@ def finalize_weights(self) -> None:; -4117,7 +4119,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: finalize_weights, get_symm_buffer, load_weights, post_load_weights
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -2243,17 +2243,19 @@ def finalize_weights(self) -> None:
-        self._transformed_l1_weights, self._transformed_l2_weights = (
-            deep_gemm.transform_weights_for_mega_moe(
-                (self.w13_weight.data.view(torch.int8).contiguous(), w13_scale),
-                (self.w2_weight.data.view(torch.int8).contiguous(), w2_scale),
-            )
+        l1, l2 = deep_gemm.transform_weights_for_mega_moe(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +32/-21
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/layers/moe/backends/base.py`, `python/tokenspeed/runtime/layers/moe/backends/weight_loaders.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #375 - perf(deepseek-v4): decode kernel fusion and routing optimization

- 链接: https://github.com/lightseekorg/tokenspeed/pull/375
- 状态/时间: merged / 2026-06-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`；关联提交 `d57cf9f1619d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+392/-159，可读 patch 703 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +42/-10 (52 lines); hunks: -292,13 +292,14 @@ def _deepseek_v4_fused_select_experts(; -312,12 +313,17 @@ def _deepseek_v4_fused_select_experts(; symbols: _deepseek_v4_fused_select_experts, deepseek_v4_select_experts，涉及 `_deepseek_v4_fused_select_experts, deepseek_v4_select_experts`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +42/-10 (52 lines); hunks: -292,13 +292,14 @@ def _deepseek_v4_fused_select_experts(; -312,12 +313,17 @@ def _deepseek_v4_fused_select_experts(; symbols: _deepseek_v4_fused_select_experts, deepseek_v4_select_experts
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -292,13 +292,14 @@ def _deepseek_v4_fused_select_experts(
-        or router_logits.shape[1] != 256
+    num_experts = router_logits.shape[1]
@@ -312,12 +313,17 @@ def _deepseek_v4_fused_select_experts(
+    if num_experts not in (256, 384) or top_k != 6 or not renormalize:
+        return None
+    logits_f32 = router_logits.float().contiguous()
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +42/-10
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/models/deepseek_v4.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/activation/triton.py`, `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/routing_flash.cu`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #361 - feat(deepseek-v4): support MTP prefix cache reuse

- 链接: https://github.com/lightseekorg/tokenspeed/pull/361
- 状态/时间: merged / 2026-06-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_mtp_prefix_cache.py`；关联提交 `fd6b592449d0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 14 个文件，+486/-60，可读 patch 761 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +38/-27 (65 lines); hunks: -1954,6 +1954,34 @@ def _deepseek_v4_sanitize_swa_slot_mapping(; -3597,14 +3625,12 @@ def forward(; symbols: _deepseek_v4_sanitize_swa_slot_mapping, _deepseek_v4_swa_slot_mapping, _attention_use_fp4_indexer_cache, forward，涉及 `_deepseek_v4_sanitize_swa_slot_mapping, _deepseek_v4_swa_slot_mapping, _attention_use_fp4_indexer_cache`；`test/runtime/test_deepseek_v4_mtp_prefix_cache.py` added +136/-0 (136 lines); hunks: -0,0 +1,136; symbols: test_deepseek_v4_swa_slot_mapping_expands_mtp_decode_requests, test_deepseek_v4_swa_slot_mapping_prefers_draft_prefill_metadata, test_deepseek_v4_swa_slot_mapping_falls_back_for_incompatible_draft_metadata, test_draft_idle_global_num_tokens_match_multi_step_decode_shape，涉及 `test_deepseek_v4_swa_slot_mapping_expands_mtp_decode_requests, test_deepseek_v4_swa_slot_mapping_prefers_draft_prefill_metadata, test_deepseek_v4_swa_slot_mapping_falls_back_for_incompatible_draft_metadata`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +38/-27 (65 lines); hunks: -1954,6 +1954,34 @@ def _deepseek_v4_sanitize_swa_slot_mapping(; -3597,14 +3625,12 @@ def forward(; symbols: _deepseek_v4_sanitize_swa_slot_mapping, _deepseek_v4_swa_slot_mapping, _attention_use_fp4_indexer_cache, forward
  - `test/runtime/test_deepseek_v4_mtp_prefix_cache.py` added +136/-0 (136 lines); hunks: -0,0 +1,136; symbols: test_deepseek_v4_swa_slot_mapping_expands_mtp_decode_requests, test_deepseek_v4_swa_slot_mapping_prefers_draft_prefill_metadata, test_deepseek_v4_swa_slot_mapping_falls_back_for_incompatible_draft_metadata, test_draft_idle_global_num_tokens_match_multi_step_decode_shape
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -1954,6 +1954,34 @@ def _deepseek_v4_sanitize_swa_slot_mapping(
+def _deepseek_v4_swa_slot_mapping(
+    ctx: ForwardContext,
+    positions: torch.Tensor,
+    out_cache_loc: torch.Tensor,
+) -> torch.Tensor:
+    if positions.numel() == 0:
diff -- test/runtime/test_deepseek_v4_mtp_prefix_cache.py
@@ -0,0 +1,136 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +38/-27
  - tests: `test/runtime/test_deepseek_v4_mtp_prefix_cache.py` added +136/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_mtp_prefix_cache.py`, `tokenspeed-scheduler/tests/cpp/test_paged_cache_replay.cpp`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #344 - ci(eval): add DeepSeek V4-Flash eval CI tasks

- 链接: https://github.com/lightseekorg/tokenspeed/pull/344
- 状态/时间: merged / 2026-06-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml`, `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml`；关联提交 `462ab6fe721c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+102/-0，可读 patch 104 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml` added +53/-0 (53 lines); hunks: -0,0 +1,53；`test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml` added +49/-0 (49 lines); hunks: -0,0 +1,49。
- 代码 diff 细节:
  - `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml` added +53/-0 (53 lines); hunks: -0,0 +1,53
  - `test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml` added +49/-0 (49 lines); hunks: -0,0 +1,49
- 关键代码摘录:

```diff
diff -- test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml
@@ -0,0 +1,53 @@
+api_version: ci.tokenspeed.io/v1
+name: eval-deepseek-v4-flash-mtp-gsm8k
+type: eval
+triggers:
+  - per-commit
+  - manual
diff -- test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml
@@ -0,0 +1,49 @@
+api_version: ci.tokenspeed.io/v1
+name: eval-deepseek-v4-flash-gsm8k
+type: eval
+triggers:
+  - per-commit
+  - manual
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml` added +53/-0; `test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml` added +49/-0
- 验证与风险: diff 自带测试面 `test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml`, `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #398 - perf(deepseek-v4): pre-compile deep_gemm JIT kernels at startup

- 链接: https://github.com/lightseekorg/tokenspeed/pull/398
- 状态/时间: merged / 2026-06-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`；关联提交 `f13b16b25987`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+669/-73，可读 patch 822 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +78/-73 (151 lines); hunks: -69,6 +69,7; -2387,71 +2388,6 @@ def forward(; symbols: forward, warmup_jit_variants, DeepseekV4MoE, __init__，涉及 `forward, warmup_jit_variants, DeepseekV4MoE`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +78/-73 (151 lines); hunks: -69,6 +69,7; -2387,71 +2388,6 @@ def forward(; symbols: forward, warmup_jit_variants, DeepseekV4MoE, __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -69,6 +69,7 @@
+    V4_KERNEL_BLOCK_ROWS,
@@ -2387,71 +2388,6 @@ def forward(
-    def warmup_jit_variants(self) -> None:
-        """Pre-compile every DeepGEMM mega-MoE tile this layer can hit at runtime.
-        DeepGEMM JITs ``fp8_fp4_mega_moe`` on first use and selects the tile
-        (``block_m``) from the per-rank token count -- roughly
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +78/-73
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/model_loader/loader.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/utils/env.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #427 - perf(deepseek-v4): dense deep_gemm warmup M-sweep + fp8_einsum coverage

- 链接: https://github.com/lightseekorg/tokenspeed/pull/427
- 状态/时间: merged / 2026-06-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`；关联提交 `4d0d32cf810c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+189/-164，可读 patch 447 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +7/-2 (9 lines); hunks: -4189,14 +4189,19 @@ def _warmup_prefill_jit(self) -> None:; symbols: _warmup_prefill_jit, post_quant_warmup, get_model_config_for_expert_location，涉及 `_warmup_prefill_jit, post_quant_warmup, get_model_config_for_expert_location`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +7/-2 (9 lines); hunks: -4189,14 +4189,19 @@ def _warmup_prefill_jit(self) -> None:; symbols: _warmup_prefill_jit, post_quant_warmup, get_model_config_for_expert_location
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -4189,14 +4189,19 @@ def _warmup_prefill_jit(self) -> None:
-            max_tokens=min(getattr(config, "max_position_embeddings", 8192), 8192),
+            # Prefill GEMM/prenorm M is capped per forward by chunked_prefill_size
+            # (continuous batching), the same ceiling mega_moe warms to. Hardcoding
+            # 8192 would leave M in (8192, chunked_prefill_size] to JIT inline.
+            max_tokens=_deepseek_v4_mega_moe_max_num_tokens(),
-            deep_gemm.warmup_fp8_gemm_nt_from_model(self)
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +7/-2
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/models/deepseek_v4.py`, `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/deep_gemm/warmup.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #503 - fix(spec): remove V4 MTP special forward modes

- 链接: https://github.com/lightseekorg/tokenspeed/pull/503
- 状态/时间: merged / 2026-06-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py`, `test/runtime/test_deepseek_v4_mtp_prefix_cache.py`；关联提交 `ed3f06314b3d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 14 个文件，+189/-312，可读 patch 1174 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +45/-63 (108 lines); hunks: -331,6 +331,22 @@ def _query_lens(; -356,21 +372,6 @@ def _query_lens(; symbols: _query_lens, init_forward_metadata，涉及 `_query_lens, init_forward_metadata`；`python/tokenspeed/runtime/models/deepseek_v4.py` modified +16/-24 (40 lines); hunks: -164,35 +164,31 @@ def _deepseek_v4_indexer_token_split(; -2966,11 +2962,7 @@ def prepare_decode_metadata(; symbols: _deepseek_v4_indexer_token_split, _deepseek_v4_forward_metadata, prepare_decode_metadata, run_compressor，涉及 `_deepseek_v4_indexer_token_split, _deepseek_v4_forward_metadata, prepare_decode_metadata`；`test/runtime/test_deepseek_v4_config.py` modified +83/-89 (172 lines); hunks: -44,7 +44,6; -277,42 +276,13 @@ def test_forward_mode_mixed_predicate(self):; symbols: test_forward_mode_mixed_predicate, test_model_runner_forwards_supported_spec_step_idx, ModelWithSpecStep, test_deepseek_v4_indexer_token_split_treats_spec_modes_as_decode，涉及 `test_forward_mode_mixed_predicate, test_model_runner_forwards_supported_spec_step_idx, ModelWithSpecStep`；`test/runtime/test_deepseek_v4_mtp_prefix_cache.py` modified +2/-2 (4 lines); hunks: -70,7 +70,7 @@ def test_deepseek_v4_swa_slot_mapping_prefers_draft_prefill_me...; -101,7 +101,7 @@ def test_deepseek_v4_swa_slot_mapping_falls_back_for_incompa...; symbols: test_deepseek_v4_swa_slot_mapping_prefers_draft_prefill_metadata, test_deepseek_v4_swa_slot_mapping_falls_back_for_incompatible_draft_metadata，涉及 `test_deepseek_v4_swa_slot_mapping_prefers_draft_prefill_metadata, test_deepseek_v4_swa_slot_mapping_falls_back_for_incompatible_draft_metadata`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +45/-63 (108 lines); hunks: -331,6 +331,22 @@ def _query_lens(; -356,21 +372,6 @@ def _query_lens(; symbols: _query_lens, init_forward_metadata
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +16/-24 (40 lines); hunks: -164,35 +164,31 @@ def _deepseek_v4_indexer_token_split(; -2966,11 +2962,7 @@ def prepare_decode_metadata(; symbols: _deepseek_v4_indexer_token_split, _deepseek_v4_forward_metadata, prepare_decode_metadata, run_compressor
  - `test/runtime/test_deepseek_v4_config.py` modified +83/-89 (172 lines); hunks: -44,7 +44,6; -277,42 +276,13 @@ def test_forward_mode_mixed_predicate(self):; symbols: test_forward_mode_mixed_predicate, test_model_runner_forwards_supported_spec_step_idx, ModelWithSpecStep, test_deepseek_v4_indexer_token_split_treats_spec_modes_as_decode
  - `test/runtime/test_deepseek_v4_mtp_prefix_cache.py` modified +2/-2 (4 lines); hunks: -70,7 +70,7 @@ def test_deepseek_v4_swa_slot_mapping_prefers_draft_prefill_me...; -101,7 +101,7 @@ def test_deepseek_v4_swa_slot_mapping_falls_back_for_incompa...; symbols: test_deepseek_v4_swa_slot_mapping_prefers_draft_prefill_metadata, test_deepseek_v4_swa_slot_mapping_falls_back_for_incompatible_draft_metadata
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py
@@ -331,6 +331,22 @@ def _query_lens(
+            if forward_mode.is_decode() and num_tokens != bs:
+                if bs == 0:
+                    return torch.zeros(0, dtype=torch.int32, device=seq_lens.device)
+                if num_tokens % bs != 0:
+                    raise RuntimeError(
+                        "DeepSeek V4 packed decode metadata expects uniformly "
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -164,35 +164,31 @@ def _deepseek_v4_indexer_token_split(
-    if forward_mode is not None and (
-        forward_mode.is_decode()
-        or forward_mode.is_target_verify()
-        or forward_mode.is_draft_extend()
-    ):
+    if forward_mode is not None and forward_mode.is_decode():
diff -- test/runtime/test_deepseek_v4_config.py
@@ -44,7 +44,6 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +45/-63; `python/tokenspeed/runtime/models/deepseek_v4.py` modified +16/-24
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +83/-89; `test/runtime/test_deepseek_v4_mtp_prefix_cache.py` modified +2/-2
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`, `test/runtime/test_deepseek_v4_mtp_prefix_cache.py`, `test/runtime/test_dp_sampling_routing_metadata.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #529 - perf(deepseek-v4): deferred-state MHC forward for cross-layer fusion

- 链接: https://github.com/lightseekorg/tokenspeed/pull/529
- 状态/时间: merged / 2026-06-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`；关联提交 `520abd049ed2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+109/-22，可读 patch 220 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +77/-21 (98 lines); hunks: -106,6 +106,7; -391,6 +392,32 @@ def mhc_post(; symbols: mhc_post, mhc_fused_hc, hc_head, forward，涉及 `mhc_post, mhc_fused_hc, hc_head`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +77/-21 (98 lines); hunks: -106,6 +106,7; -391,6 +392,32 @@ def mhc_post(; symbols: mhc_post, mhc_fused_hc, hc_head, forward
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -106,6 +106,7 @@
+from tokenspeed.runtime.layers.deepseek_v4_mhc import mhc_fused_hc as fast_mhc_fused_hc
@@ -391,6 +392,32 @@ def mhc_post(
+def mhc_fused_hc(
+    x_prev: torch.Tensor,
+    residual_prev: torch.Tensor,
+    post_prev: torch.Tensor,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +77/-21
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/layers/deepseek_v4_mhc.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #583 - perf(deepseek-v4): enable MTP overlap scheduling with paged cache

- 链接: https://github.com/lightseekorg/tokenspeed/pull/583
- 状态/时间: merged / 2026-07-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py`；关联提交 `754c34056c0f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 25 个文件，+1492/-166，可读 patch 2470 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +46/-19 (65 lines); hunks: -35,6 +35,9; -347,7 +350,13 @@ def _query_lens(; symbols: _query_lens, _query_lens_cpu, init_forward_metadata, forward_deepseek_v4_mixed，涉及 `_query_lens, _query_lens_cpu, init_forward_metadata`；`python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +10/-0 (10 lines); hunks: -238,6 +238,8 @@ def profile_deepseek_v4_max_num_pages(; -265,6 +267,8 @@ def _bytes_for_pages(num_pages: int) -> int:; symbols: profile_deepseek_v4_max_num_pages, _bytes_for_pages, __init__，涉及 `profile_deepseek_v4_max_num_pages, _bytes_for_pages, __init__`；`python/tokenspeed/runtime/models/deepseek_v4.py` modified +1/-1 (2 lines); hunks: -172,7 +172,7 @@ def _deepseek_v4_indexer_token_split(; symbols: _deepseek_v4_indexer_token_split, _deepseek_v4_forward_metadata，涉及 `_deepseek_v4_indexer_token_split, _deepseek_v4_forward_metadata`；`test/runtime/test_deepseek_v4_config.py` modified +152/-2 (154 lines); hunks: -1640,6 +1640,155 @@ def test_deepseek_v4_mixed_metadata_keeps_decode_rows_si...; -2256,7 +2405,7 @@ def fake_decode(**kwargs):; symbols: test_deepseek_v4_mixed_metadata_keeps_decode_rows_single_token, test_deepseek_v4_mixed_metadata_uses_runtime_verify_width, test_deepseek_v4_mixed_metadata_rejects_packed_token_mismatch, test_deepseek_v4_draft_keeps_mixed_step0_and_decode_step_metadata，涉及 `test_deepseek_v4_mixed_metadata_keeps_decode_rows_single_token, test_deepseek_v4_mixed_metadata_uses_runtime_verify_width, test_deepseek_v4_mixed_metadata_rejects_packed_token_mismatch`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +46/-19 (65 lines); hunks: -35,6 +35,9; -347,7 +350,13 @@ def _query_lens(; symbols: _query_lens, _query_lens_cpu, init_forward_metadata, forward_deepseek_v4_mixed
  - `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +10/-0 (10 lines); hunks: -238,6 +238,8 @@ def profile_deepseek_v4_max_num_pages(; -265,6 +267,8 @@ def _bytes_for_pages(num_pages: int) -> int:; symbols: profile_deepseek_v4_max_num_pages, _bytes_for_pages, __init__
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +1/-1 (2 lines); hunks: -172,7 +172,7 @@ def _deepseek_v4_indexer_token_split(; symbols: _deepseek_v4_indexer_token_split, _deepseek_v4_forward_metadata
  - `test/runtime/test_deepseek_v4_config.py` modified +152/-2 (154 lines); hunks: -1640,6 +1640,155 @@ def test_deepseek_v4_mixed_metadata_keeps_decode_rows_si...; -2256,7 +2405,7 @@ def fake_decode(**kwargs):; symbols: test_deepseek_v4_mixed_metadata_keeps_decode_rows_single_token, test_deepseek_v4_mixed_metadata_uses_runtime_verify_width, test_deepseek_v4_mixed_metadata_rejects_packed_token_mismatch, test_deepseek_v4_draft_keeps_mixed_step0_and_decode_step_metadata
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py
@@ -35,6 +35,9 @@
+from tokenspeed.runtime.configs.paged_cache_spec import (
+    compute_max_logical_pages_for_capture,
+)
@@ -347,7 +350,13 @@ def _query_lens(
-            lens = torch.ones(bs, dtype=torch.int32, device=seq_lens.device)
+            verify_width = max(1, int(self.speculative_num_draft_tokens))
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py
@@ -238,6 +238,8 @@ def profile_deepseek_v4_max_num_pages(
+    decode_input_tokens: int = 1,
+    overlap_schedule_depth: int = 0,
@@ -265,6 +267,8 @@ def _bytes_for_pages(num_pages: int) -> int:
+            decode_input_tokens=decode_input_tokens,
+            overlap_schedule_depth=overlap_schedule_depth,
@@ -291,6 +295,8 @@ def _bytes_for_pages(num_pages: int) -> int:
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -172,7 +172,7 @@ def _deepseek_v4_indexer_token_split(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +46/-19; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +10/-0; `python/tokenspeed/runtime/models/deepseek_v4.py` modified +1/-1
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +152/-2
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`, `test/runtime/test_model_executor_retraction.py`, `test/runtime/test_v4_prefix_cache_metadata.py`, `test/runtime/test_v4_sliding_window_groups_smoke.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #614 - perf(deepseek-v4): sanitize SWA slot mapping once per step

- 链接: https://github.com/lightseekorg/tokenspeed/pull/614
- 状态/时间: merged / 2026-07-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_mtp_prefix_cache.py`；关联提交 `b9df166b0fe9`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+261/-34，可读 patch 434 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +38/-26 (64 lines); hunks: -1981,12 +1981,14 @@ def _deepseek_v4_padded_heads(num_local_heads: int) -> int:; -2000,26 +2002,46 @@ def _deepseek_v4_swa_slot_mapping(; symbols: _deepseek_v4_padded_heads, _deepseek_v4_sanitize_swa_slot_mapping, _deepseek_v4_swa_slot_mapping, _insert_swa_cache，涉及 `_deepseek_v4_padded_heads, _deepseek_v4_sanitize_swa_slot_mapping, _deepseek_v4_swa_slot_mapping`；`python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +13/-0 (13 lines); hunks: -1018,6 +1018,19 @@ def _require(; symbols: _require, get_swa_kv_buffer, swa_capacity_slots, get_compressed_kv_buffer_2d，涉及 `_require, get_swa_kv_buffer, swa_capacity_slots`；`test/runtime/test_deepseek_v4_mtp_prefix_cache.py` modified +84/-3 (87 lines); hunks: -13,10 +13,19; -40,7 +49,7 @@ def test_deepseek_v4_swa_slot_mapping_expands_mtp_decode_reque...; symbols: test_deepseek_v4_swa_slot_mapping_expands_mtp_decode_requests, test_deepseek_v4_swa_slot_mapping_prefers_draft_prefill_metadata, test_deepseek_v4_swa_slot_mapping_masks_invalid_and_overflow_slots, test_deepseek_v4_swa_slot_mapping_fails_closed_without_capacity，涉及 `test_deepseek_v4_swa_slot_mapping_expands_mtp_decode_requests, test_deepseek_v4_swa_slot_mapping_prefers_draft_prefill_metadata, test_deepseek_v4_swa_slot_mapping_masks_invalid_and_overflow_slots`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +38/-26 (64 lines); hunks: -1981,12 +1981,14 @@ def _deepseek_v4_padded_heads(num_local_heads: int) -> int:; -2000,26 +2002,46 @@ def _deepseek_v4_swa_slot_mapping(; symbols: _deepseek_v4_padded_heads, _deepseek_v4_sanitize_swa_slot_mapping, _deepseek_v4_swa_slot_mapping, _insert_swa_cache
  - `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +13/-0 (13 lines); hunks: -1018,6 +1018,19 @@ def _require(; symbols: _require, get_swa_kv_buffer, swa_capacity_slots, get_compressed_kv_buffer_2d
  - `test/runtime/test_deepseek_v4_mtp_prefix_cache.py` modified +84/-3 (87 lines); hunks: -13,10 +13,19; -40,7 +49,7 @@ def test_deepseek_v4_swa_slot_mapping_expands_mtp_decode_reque...; symbols: test_deepseek_v4_swa_slot_mapping_expands_mtp_decode_requests, test_deepseek_v4_swa_slot_mapping_prefers_draft_prefill_metadata, test_deepseek_v4_swa_slot_mapping_masks_invalid_and_overflow_slots, test_deepseek_v4_swa_slot_mapping_fails_closed_without_capacity
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -1981,12 +1981,14 @@ def _deepseek_v4_padded_heads(num_local_heads: int) -> int:
-    swa_kv_cache: torch.Tensor,
-    block_size: int,
+    capacity: int,
+    # Fail closed: with no writable SWA capacity every slot is masked, so an
+    # unchecked mapping can never reach the fused cache-insert kernels.
+    if capacity <= 0:
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py
@@ -1018,6 +1018,19 @@ def _require(
+    @property
+    def swa_capacity_slots(self) -> int:
+        """Writable SWA cache capacity shared by every layer, in token slots.
+        Every layer's SWA buffer is allocated with the same page count, so a
+        single capacity (pages * tokens per block) bounds the write-slot
+        mapping shared across layers. Returns 0 when no SWA buffers exist;
diff -- test/runtime/test_deepseek_v4_mtp_prefix_cache.py
@@ -13,10 +13,19 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +38/-26; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +13/-0
  - tests: `test/runtime/test_deepseek_v4_mtp_prefix_cache.py` modified +84/-3
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_attention_ops.py`, `test/runtime/test_deepseek_v4_mtp_prefix_cache.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #620 - feat: add DeepSeek V4 L2 KV cache offload and perf optimize.

- 链接: https://github.com/lightseekorg/tokenspeed/pull/620
- 状态/时间: merged / 2026-07-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`；关联提交 `c8385dcd7801`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 74 个文件，+6146/-1190，可读 patch 10178 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +167/-24 (191 lines); hunks: -102,6 +102,7; -197,6 +198,77 @@ def _deepseek_v4_forward_metadata(ctx: ForwardContext):; symbols: _deepseek_v4_forward_metadata, _deepseek_v4_hca_active_token_offsets, _deepseek_v4_hca_active_token_indices, _dequant_fp8_weight，涉及 `_deepseek_v4_forward_metadata, _deepseek_v4_hca_active_token_offsets, _deepseek_v4_hca_active_token_indices`；`python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +116/-0 (116 lines); hunks: -43,6 +43,9; -87,6 +90,7; symbols: deepseek_v4_hca_compress_kv_cache_insert, deepseek_v4_hca_direct_compress_kv_cache_insert, deepseek_v4_csa_compress_kv_cache_insert，涉及 `deepseek_v4_hca_compress_kv_cache_insert, deepseek_v4_hca_direct_compress_kv_cache_insert, deepseek_v4_csa_compress_kv_cache_insert`；`python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +12/-33 (45 lines); hunks: -13,7 +13,6; -763,7 +762,7 @@ class DeepseekV4TokenToKVPool(BaseTokenToKVPool):; symbols: DeepseekV4TokenToKVPool, __init__, prefix_cache_required_group_ids, bind_paged_cache_scheduler，涉及 `DeepseekV4TokenToKVPool, __init__, prefix_cache_required_group_ids`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/deepseek_v4.py` modified +349/-5 (354 lines); hunks: -41,7 +41,6; -51,6 +50,7; symbols: deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4, _deepseek_v4_fused_sparse_compress_cache_kernel, deepseek_v4_fused_sparse_compress_cache_insert，涉及 `deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4, _deepseek_v4_fused_sparse_compress_cache_kernel, deepseek_v4_fused_sparse_compress_cache_insert`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +167/-24 (191 lines); hunks: -102,6 +102,7; -197,6 +198,77 @@ def _deepseek_v4_forward_metadata(ctx: ForwardContext):; symbols: _deepseek_v4_forward_metadata, _deepseek_v4_hca_active_token_offsets, _deepseek_v4_hca_active_token_indices, _dequant_fp8_weight
  - `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +116/-0 (116 lines); hunks: -43,6 +43,9; -87,6 +90,7; symbols: deepseek_v4_hca_compress_kv_cache_insert, deepseek_v4_hca_direct_compress_kv_cache_insert, deepseek_v4_csa_compress_kv_cache_insert
  - `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +12/-33 (45 lines); hunks: -13,7 +13,6; -763,7 +762,7 @@ class DeepseekV4TokenToKVPool(BaseTokenToKVPool):; symbols: DeepseekV4TokenToKVPool, __init__, prefix_cache_required_group_ids, bind_paged_cache_scheduler
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/deepseek_v4.py` modified +349/-5 (354 lines); hunks: -41,7 +41,6; -51,6 +50,7; symbols: deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4, _deepseek_v4_fused_sparse_compress_cache_kernel, deepseek_v4_fused_sparse_compress_cache_insert
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -102,6 +102,7 @@
+    deepseek_v4_hca_direct_compress_kv_cache_insert,
@@ -197,6 +198,77 @@ def _deepseek_v4_forward_metadata(ctx: ForwardContext):
+def _deepseek_v4_hca_active_token_offsets(
+    ctx: ForwardContext,
+    metadata: DeepseekV4ForwardMetadata,
+    positions: torch.Tensor,
diff -- python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py
@@ -43,6 +43,9 @@
+from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
+    deepseek_v4_fused_hca_direct_compress_cache_insert as _triton_fused_hca_direct_compress_cache_insert,
+)
@@ -87,6 +90,7 @@
+    "deepseek_v4_hca_direct_compress_kv_cache_insert",
@@ -910,6 +914,118 @@ def deepseek_v4_hca_compress_kv_cache_insert(
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py
@@ -13,7 +13,6 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +167/-24; `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +116/-0; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` modified +12/-33; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/deepseek_v4.py` modified +349/-5
- 验证与风险: diff 自带测试面 `test/runtime/cache/test_deepseek_v4_l2_offload.py`, `test/runtime/cache/test_transfer_host_executor.py`, `test/runtime/test_deepseek_v4_attention_ops.py`, `test/runtime/test_flatkv_pd_shutdown.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #940 - Add DeepSeek V4 DSpark decoding and complete Flat KV replay support

- 链接: https://github.com/lightseekorg/tokenspeed/pull/940
- 状态/时间: merged / 2026-08-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/__init__.py` 等 8 个文件；关联提交 `a424bd389108`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 25 个文件，+3793/-105，可读 patch 4577 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` added +821/-0 (821 lines); hunks: -0,0 +1,821; symbols: _is_zero_initialized_expert_bias, count_dspark_stages, _block_dequant, _apply_dspark_hc_head，涉及 `_is_zero_initialized_expert_bias, count_dspark_stages, _block_dequant`；`python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +305/-22 (327 lines); hunks: -13,6 +13,8; -242,6 +244,8 @@ class DeepseekV4AttentionBackend(AttentionBackend):; symbols: DeepseekV4AttentionBackend, __init__, _configure_cache_group_contract, configure_runtime，涉及 `DeepseekV4AttentionBackend, __init__, _configure_cache_group_contract`；`python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/attention.py` added +296/-0 (296 lines); hunks: -0,0 +1,296; symbols: dspark_fp8_quant_dequant, _dspark_fp8_linear, _quantize_dspark_non_rope, _dspark_output_projection，涉及 `dspark_fp8_quant_dequant, _dspark_fp8_linear, _quantize_dspark_non_rope`；`python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: _local_vocab_argmax, DSparkVanillaMarkov, __init__, local_bias，涉及 `_local_vocab_argmax, DSparkVanillaMarkov, __init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` added +821/-0 (821 lines); hunks: -0,0 +1,821; symbols: _is_zero_initialized_expert_bias, count_dspark_stages, _block_dequant, _apply_dspark_hc_head
  - `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +305/-22 (327 lines); hunks: -13,6 +13,8; -242,6 +244,8 @@ class DeepseekV4AttentionBackend(AttentionBackend):; symbols: DeepseekV4AttentionBackend, __init__, _configure_cache_group_contract, configure_runtime
  - `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/attention.py` added +296/-0 (296 lines); hunks: -0,0 +1,296; symbols: dspark_fp8_quant_dequant, _dspark_fp8_linear, _quantize_dspark_non_rope, _dspark_output_projection
  - `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: _local_vocab_argmax, DSparkVanillaMarkov, __init__, local_bias
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +48/-2 (50 lines); hunks: -4053,6 +4053,29 @@ def __init__(; -4075,6 +4098,8 @@ def forward(; symbols: __init__, set_dspark_layers_to_capture, forward
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4_dspark.py
@@ -0,0 +1,821 @@
+# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
+# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
+# SPDX-License-Identifier: MIT AND Apache-2.0
+"""Inference-only DeepSeek V4 DSpark draft model."""
+from __future__ import annotations
+import json
diff -- python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py
@@ -13,6 +13,8 @@
+from collections.abc import Mapping
@@ -242,6 +244,8 @@ class DeepseekV4AttentionBackend(AttentionBackend):
+    cache_group_tables_replace_draft_page_table = True
+    cache_active_pages_must_be_real = True
@@ -276,6 +280,9 @@ def __init__(self, config) -> None:
+        self._expected_cache_group_ids: tuple[str, ...] | None = None
diff -- python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/attention.py
@@ -0,0 +1,296 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` added +821/-0; `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +305/-22; `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/attention.py` added +296/-0; `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py` added +170/-0; `python/tokenspeed/runtime/models/deepseek_v4.py` modified +48/-2; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` modified +1/-6
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +1064/-11
- 验证与风险: diff 自带测试面 `test/runtime/test_block_tables_bridge.py`, `test/runtime/test_cli_config_compat.py`, `test/runtime/test_cudagraph_per_group.py`, `test/runtime/test_deepseek_v4_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1048 - feat(deepseek-v4): run DeepSeek-V4-Flash on Hopper (SM90 FP8 indexer + DSpark)

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1048
- 状态/时间: merged / 2026-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` 等 7 个文件；关联提交 `6a08766a33f3`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 50 个文件，+1826/-855，可读 patch 4555 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +197/-67 (264 lines); hunks: -93,8 +93,10; -131,7 +133,7; symbols: _deepseek_v4_indexer_topk_prefill_deepgemm, _deepseek_v4_indexer_topk_from_cache_deepgemm_decode, _deepseek_v4_sparse_attn_indexer_native，涉及 `_deepseek_v4_indexer_topk_prefill_deepgemm, _deepseek_v4_indexer_topk_from_cache_deepgemm_decode, _deepseek_v4_sparse_attn_indexer_native`；`python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +192/-41 (233 lines); hunks: -87,8 +87,10; -239,23 +241,25 @@ def dequantize_deepseek_v4_fp8_ds_mla_cache(; symbols: dequantize_deepseek_v4_fp8_ds_mla_cache, deepseek_v4_prepare_indexer_q_mxfp4, deepseek_v4_prepare_indexer_q_fp8, gather_paged_indexer_fp8_cache，涉及 `dequantize_deepseek_v4_fp8_ds_mla_cache, deepseek_v4_prepare_indexer_q_mxfp4, deepseek_v4_prepare_indexer_q_fp8`；`python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +31/-2 (33 lines); hunks: -25,6 +25,7; -34,6 +35,7; symbols: _map_checkpoint_name, load_weights, dequant，涉及 `_map_checkpoint_name, load_weights, dequant`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` modified +11/-3 (14 lines); hunks: -74,7 +74,7 @@ def build_deepseek_v4_cache_fields(; -220,8 +220,16 @@ def use_fp4_indexer(hf_config) -> bool:; symbols: build_deepseek_v4_cache_fields, solve_deepseek_v4_memory_layout, use_fp4_indexer, layout_and_fields，涉及 `build_deepseek_v4_cache_fields, solve_deepseek_v4_memory_layout, use_fp4_indexer`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +197/-67 (264 lines); hunks: -93,8 +93,10; -131,7 +133,7; symbols: _deepseek_v4_indexer_topk_prefill_deepgemm, _deepseek_v4_indexer_topk_from_cache_deepgemm_decode, _deepseek_v4_sparse_attn_indexer_native
  - `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +192/-41 (233 lines); hunks: -87,8 +87,10; -239,23 +241,25 @@ def dequantize_deepseek_v4_fp8_ds_mla_cache(; symbols: dequantize_deepseek_v4_fp8_ds_mla_cache, deepseek_v4_prepare_indexer_q_mxfp4, deepseek_v4_prepare_indexer_q_fp8, gather_paged_indexer_fp8_cache
  - `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +31/-2 (33 lines); hunks: -25,6 +25,7; -34,6 +35,7; symbols: _map_checkpoint_name, load_weights, dequant
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` modified +11/-3 (14 lines); hunks: -74,7 +74,7 @@ def build_deepseek_v4_cache_fields(; -220,8 +220,16 @@ def use_fp4_indexer(hf_config) -> bool:; symbols: build_deepseek_v4_cache_fields, solve_deepseek_v4_memory_layout, use_fp4_indexer, layout_and_fields
  - `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +1/-5 (6 lines); hunks: -1939,13 +1939,9 @@ def init_cuda_graph_state(; -2004,7 +2000,7 @@ def init_cuda_graph_state(; symbols: init_cuda_graph_state
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -93,8 +93,10 @@
+    deepseek_v4_prepare_indexer_q_fp8,
+    gather_paged_indexer_fp8_cache,
@@ -131,7 +133,7 @@
-from tokenspeed.runtime.layers.quantization import Mxfp4Config
+from tokenspeed.runtime.layers.quantization import Fp8Config, Mxfp4Config
@@ -1377,12 +1379,15 @@ def _deepseek_v4_indexer_topk_prefill_deepgemm(
diff -- python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py
@@ -87,8 +87,10 @@
+    "deepseek_v4_prepare_indexer_q_fp8",
+    "gather_paged_indexer_fp8_cache",
@@ -239,23 +241,25 @@ def dequantize_deepseek_v4_fp8_ds_mla_cache(
-    flat_cache = cache_2d.reshape(-1)
-    page_base = pages * cache_2d.stride(0)
-    value_base = page_base + pos * token_stride
diff -- python/tokenspeed/runtime/models/deepseek_v4_dspark.py
@@ -25,6 +25,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +197/-67; `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +192/-41; `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +31/-2; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` modified +11/-3; `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +1/-5; `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py` modified +2/-0
  - tests: `test/runtime/test_deepseek_v4_hopper_fp8.py` added +282/-0; `test/runtime/test_deepseek_v4_config.py` modified +33/-4
- 验证与风险: diff 自带测试面 `test/runtime/distributed/test_cache_pd_executors.py`, `test/runtime/distributed/test_mla_dsa_pd_contract.py`, `test/runtime/test_cache_metadata_kernel_table.py`, `test/runtime/test_cache_pool.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1095 - fix(deepseek-v4): honor routed expert checkpoint dtype

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1095
- 状态/时间: merged / 2026-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4_dspark.py`, `test/runtime/test_deepseek_v4_config.py`；关联提交 `e4fc38b9043f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+141/-37，可读 patch 260 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +40/-23 (63 lines); hunks: -2504,6 +2504,37 @@ def forward(; -2581,20 +2612,9 @@ def __init__(; symbols: forward, _deepseek_v4_routed_expert_quant_config, _deepseek_v4_expert_scale_parameter_name, DeepseekV4MoE，涉及 `forward, _deepseek_v4_routed_expert_quant_config, _deepseek_v4_expert_scale_parameter_name`；`python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +7/-9 (16 lines); hunks: -37,6 +37,7; -668,15 +669,12 @@ def _map_checkpoint_name(self, raw_name: str) -> str | None:; symbols: _map_checkpoint_name，涉及 `_map_checkpoint_name`；`test/runtime/test_deepseek_v4_config.py` modified +94/-5 (99 lines); hunks: -95,14 +95,20; -115,6 +121,7; symbols: test_config_registry, test_fp4_expert_contract_overrides_model_wide_fp8_config, test_fp8_expert_contract_keeps_model_wide_fp8_config, test_expert_scale_name_follows_expert_format_and_backend，涉及 `test_config_registry, test_fp4_expert_contract_overrides_model_wide_fp8_config, test_fp8_expert_contract_keeps_model_wide_fp8_config`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +40/-23 (63 lines); hunks: -2504,6 +2504,37 @@ def forward(; -2581,20 +2612,9 @@ def __init__(; symbols: forward, _deepseek_v4_routed_expert_quant_config, _deepseek_v4_expert_scale_parameter_name, DeepseekV4MoE
  - `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +7/-9 (16 lines); hunks: -37,6 +37,7; -668,15 +669,12 @@ def _map_checkpoint_name(self, raw_name: str) -> str | None:; symbols: _map_checkpoint_name
  - `test/runtime/test_deepseek_v4_config.py` modified +94/-5 (99 lines); hunks: -95,14 +95,20; -115,6 +121,7; symbols: test_config_registry, test_fp4_expert_contract_overrides_model_wide_fp8_config, test_fp8_expert_contract_keeps_model_wide_fp8_config, test_expert_scale_name_follows_expert_format_and_backend
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -2504,6 +2504,37 @@ def forward(
+def _deepseek_v4_routed_expert_quant_config(
+    config: PretrainedConfig,
+    quant_config: QuantizationConfig,
+) -> tuple[QuantizationConfig, bool]:
+    # ``quant_method=fp8`` describes the model-wide quantization, but DeepSeek
+    # V4 can still serialize routed experts as packed MXFP4. Prefer the explicit
diff -- python/tokenspeed/runtime/models/deepseek_v4_dspark.py
@@ -37,6 +37,7 @@
+    _deepseek_v4_expert_scale_parameter_name,
@@ -668,15 +669,12 @@ def _map_checkpoint_name(self, raw_name: str) -> str | None:
-            # MegaMoE experts register block scales as ``w{13,2}_weight_scale``;
-            # the generic block-FP8 MoELayer (non-mega, e.g. flashinfer_cutlass
-            # on Hopper) registers ``..._weight_scale_inv``. Mirror
-            # DeepseekV4ForCausalLM._map_weight_name.
diff -- test/runtime/test_deepseek_v4_config.py
@@ -95,14 +95,20 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +40/-23; `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +7/-9
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +94/-5
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1115 - perf(v4): avoid prefill host synchronizations

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1115
- 状态/时间: merged / 2026-08-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/runtime/test_deepseek_v4_config.py`；关联提交 `18e5fd4bd49c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+417/-35，可读 patch 521 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/test_deepseek_v4_config.py` modified +268/-0 (268 lines); hunks: -3098,6 +3098,272 @@ def test_deepseek_v4_mixed_metadata_rejects_packed_token...; -4923,6 +5189,8 @@ def test_deepseek_v4_eager_draft_decode_refreshes_stale_gr...; symbols: test_deepseek_v4_mixed_metadata_rejects_packed_token_mismatch, test_deepseek_v4_prefill_metadata_uses_complete_cpu_mirrors, test_deepseek_v4_prefill_metadata_requires_complete_cpu_mirrors, test_deepseek_v4_prefill_workspace_bounds_use_cpu_mirrors，涉及 `test_deepseek_v4_mixed_metadata_rejects_packed_token_mismatch, test_deepseek_v4_prefill_metadata_uses_complete_cpu_mirrors, test_deepseek_v4_prefill_metadata_requires_complete_cpu_mirrors`；`python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +149/-35 (184 lines); hunks: -960,7 +960,6 @@ def init_forward_metadata(; -969,6 +968,16 @@ def init_forward_metadata(; symbols: init_forward_metadata, _prefill_workspace，涉及 `init_forward_metadata, _prefill_workspace`。
- 代码 diff 细节:
  - `test/runtime/test_deepseek_v4_config.py` modified +268/-0 (268 lines); hunks: -3098,6 +3098,272 @@ def test_deepseek_v4_mixed_metadata_rejects_packed_token...; -4923,6 +5189,8 @@ def test_deepseek_v4_eager_draft_decode_refreshes_stale_gr...; symbols: test_deepseek_v4_mixed_metadata_rejects_packed_token_mismatch, test_deepseek_v4_prefill_metadata_uses_complete_cpu_mirrors, test_deepseek_v4_prefill_metadata_requires_complete_cpu_mirrors, test_deepseek_v4_prefill_workspace_bounds_use_cpu_mirrors
  - `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +149/-35 (184 lines); hunks: -960,7 +960,6 @@ def init_forward_metadata(; -969,6 +968,16 @@ def init_forward_metadata(; symbols: init_forward_metadata, _prefill_workspace
- 关键代码摘录:

```diff
diff -- test/runtime/test_deepseek_v4_config.py
@@ -3098,6 +3098,272 @@ def test_deepseek_v4_mixed_metadata_rejects_packed_token_mismatch(self):
+    def test_deepseek_v4_prefill_metadata_uses_complete_cpu_mirrors(self):
+        backend = DeepseekV4AttentionBackend(
+            SimpleNamespace(
+                prefix_granularity=64,
+                kernel_page_size=64,
+                device="cpu",
diff -- python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py
@@ -960,7 +960,6 @@ def init_forward_metadata(
-            seq_lens_cpu = seq_lens[:bs].to(dtype=torch.int32, device="cpu")
@@ -969,6 +968,16 @@ def init_forward_metadata(
+            if prefix_count == bs:
+                seq_lens_cpu = (
+                    extend_prefix_lens_cpu[:bs].to(dtype=torch.int32, device="cpu")
+                    + query_lens_cpu[:bs]
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +268/-0
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +149/-35
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1116 - perf(v4): avoid DSpark prefill chunk synchronizations

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1116
- 状态/时间: merged / 2026-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py`；关联提交 `ceaf86ae17f5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+154/-2，可读 patch 190 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` modified +31/-2 (33 lines); hunks: -294,9 +294,38 @@ def _seed_prefill_windows(; symbols: _seed_prefill_windows，涉及 `_seed_prefill_windows`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` modified +31/-2 (33 lines); hunks: -294,9 +294,38 @@ def _seed_prefill_windows(; symbols: _seed_prefill_windows
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py
@@ -294,9 +294,38 @@ def _seed_prefill_windows(
+        if num_extends < 0:
+            raise ValueError(f"DSPARK num_extends must be non-negative: {num_extends}.")
+        if num_extends == 0:
+            return 0
+        # fill_input_buffers derives this host mirror from the same scheduler
+        # lengths as input_lengths_buf. Reading the CUDA buffer row-by-row here
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` modified +31/-2
- 验证与风险: diff 自带测试面 `test/runtime/test_dspark_proposal.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1201 - feat(deepseek-v4): add AMD MI350 support

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1201
- 状态/时间: merged / 2026-08-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v4.py` 等 9 个文件；关联提交 `7cd7ca0b3028`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 89 个文件，+9794/-2363，可读 patch 15992 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +440/-1080 (1520 lines); hunks: -27,42 +27,43; -81,6 +82,7; symbols: _dequant_fp8_weight, _deepseek_v4_router_gemm, _deepseek_v4_bf16_linear_fp32, _deepseek_v4_fused_select_experts，涉及 `_dequant_fp8_weight, _deepseek_v4_router_gemm, _deepseek_v4_bf16_linear_fp32`；`python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +114/-166 (280 lines); hunks: -16,20 +16,19; -40,18 +39,9; symbols: _refresh_decode_indexer_plan_cache, _refresh_decode_indexer_schedule_metadata, __init__，涉及 `_refresh_decode_indexer_plan_cache, _refresh_decode_indexer_schedule_metadata, __init__`；`python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +64/-93 (157 lines); hunks: -25,38 +25,15; -76,17 +53,11; symbols: fused_qnorm_rope_kv_insert, deepseek_v4_prepare_indexer_q_mxfp4, deepseek_v4_prepare_indexer_q，涉及 `fused_qnorm_rope_kv_insert, deepseek_v4_prepare_indexer_q_mxfp4, deepseek_v4_prepare_indexer_q`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` modified +4/-11 (15 lines); hunks: -33,6 +33,7; -404,8 +405,7 @@ def num_lcm_blocks(self, layout: CacheLayout) -> int:; symbols: num_lcm_blocks, pool_options, _use_fp4_indexer，涉及 `num_lcm_blocks, pool_options, _use_fp4_indexer`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +440/-1080 (1520 lines); hunks: -27,42 +27,43; -81,6 +82,7; symbols: _dequant_fp8_weight, _deepseek_v4_router_gemm, _deepseek_v4_bf16_linear_fp32, _deepseek_v4_fused_select_experts
  - `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +114/-166 (280 lines); hunks: -16,20 +16,19; -40,18 +39,9; symbols: _refresh_decode_indexer_plan_cache, _refresh_decode_indexer_schedule_metadata, __init__
  - `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +64/-93 (157 lines); hunks: -25,38 +25,15; -76,17 +53,11; symbols: fused_qnorm_rope_kv_insert, deepseek_v4_prepare_indexer_q_mxfp4, deepseek_v4_prepare_indexer_q
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` modified +4/-11 (15 lines); hunks: -33,6 +33,7; -404,8 +405,7 @@ def num_lcm_blocks(self, layout: CacheLayout) -> int:; symbols: num_lcm_blocks, pool_options, _use_fp4_indexer
  - `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py` modified +2/-4 (6 lines); hunks: -18,6 +18,7; -28,9 +29,6; symbols: _update_decode_compressed_slot_mapping
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -27,42 +27,43 @@
-import os
-try:
-    # Optional dependency; the module-level wrapper imports the external
-    # `deep_gemm` package unguarded, which is not installed in baseline V4
-    # builds. Callsites guard usage with `deep_gemm is not None`.
-    from tokenspeed_kernel.thirdparty import deep_gemm
diff -- python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py
@@ -16,20 +16,19 @@
-from tokenspeed_kernel.ops.attention.flash_mla import (
-    flash_mla_sparse_fwd,
-    flash_mla_with_kvcache,
-    get_mla_metadata,
+from tokenspeed_kernel import (
+    dsv4_build_dense_prefill_local_compressed_indices,
diff -- python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py
@@ -25,38 +25,15 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +440/-1080; `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py` modified +114/-166; `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` modified +64/-93; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` modified +4/-11; `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py` modified +2/-4; `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py` modified +2/-2
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +28/-142
- 验证与风险: diff 自带测试面 `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k-amd.yaml`, `test/ci/eval/deepseek-v4-pro-evalscope-gsm8k-amd.yaml`, `test/runtime/test_deepseek_v4_config.py`, `test/runtime/test_deepseek_v4_mega_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1251 - feat(deepseek-v4): support fp4 indexer and ep on amd

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1251
- 状态/时间: merged / 2026-08-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`；关联提交 `0a47a2a6cf95`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+1333/-135，可读 patch 1977 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +119/-43 (162 lines); hunks: -505,8 +505,6 @@ def _deepseek_v4_indexer_prefill_request_gather_plan(; -522,21 +520,25 @@ def _deepseek_v4_indexer_prefill_request_gather_plan(; symbols: _deepseek_v4_indexer_prefill_request_gather_plan，涉及 `_deepseek_v4_indexer_prefill_request_gather_plan`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +119/-43 (162 lines); hunks: -505,8 +505,6 @@ def _deepseek_v4_indexer_prefill_request_gather_plan(; -522,21 +520,25 @@ def _deepseek_v4_indexer_prefill_request_gather_plan(; symbols: _deepseek_v4_indexer_prefill_request_gather_plan
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -505,8 +505,6 @@ def _deepseek_v4_indexer_prefill_request_gather_plan(
-    total_k = sum(compressed_lens_list)
@@ -522,21 +520,25 @@ def _deepseek_v4_indexer_prefill_request_gather_plan(
+    total_k = sum(compressed_lens_list)
-        compressed_lens_list,
-        dtype=torch.int64,
-        device=device,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +119/-43
- 验证与风险: diff 自带测试面 `test/ci/eval/deepseek-v4-pro-evalscope-gsm8k-amd.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1364 - Rename DeepSeek V4 NextN model module

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1364
- 状态/时间: merged / 2026-09-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4_next.py`, `test/runtime/test_deepseek_v4_config.py`；关联提交 `0b1061eb9fe1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+1/-1，可读 patch 9 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4_next.py` renamed +0/-0 (0 lines)；`test/runtime/test_deepseek_v4_config.py` modified +1/-1 (2 lines); hunks: -144,7 +144,7。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4_next.py` renamed +0/-0 (0 lines)
  - `test/runtime/test_deepseek_v4_config.py` modified +1/-1 (2 lines); hunks: -144,7 +144,7
- 关键代码摘录:

```diff
diff -- test/runtime/test_deepseek_v4_config.py
@@ -144,7 +144,7 @@
-from tokenspeed.runtime.models.deepseek_v4_mtp import DeepseekV4ForCausalLMNextN
+from tokenspeed.runtime.models.deepseek_v4_next import DeepseekV4ForCausalLMNextN
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4_next.py` renamed +0/-0
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1428 - test: add DeepSeek V4 Flash agentic benchmark

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1428
- 状态/时间: merged / 2026-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/build_swe_smith_dataset.patch`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/collect_outputs.py`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/deepseek_v4_tokenizer.py`；关联提交 `ba1565df077b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+442/-0，可读 patch 447 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/deepseek_v4_tokenizer.py` added +93/-0 (93 lines); hunks: -0,0 +1,93; symbols: load_tokenizer, apply_chat_template，涉及 `load_tokenizer, apply_chat_template`；`test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh` added +34/-0 (34 lines); hunks: -0,0 +1,34；`test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh` added +150/-0 (150 lines); hunks: -0,0 +1,150；`test/agentic_benchmark/deepseek_v4_flash/tokenspeed/collect_outputs.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: num_gpus_from_config, decoded_tok_per_iter, collect, main，涉及 `num_gpus_from_config, decoded_tok_per_iter, collect`。
- 代码 diff 细节:
  - `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/deepseek_v4_tokenizer.py` added +93/-0 (93 lines); hunks: -0,0 +1,93; symbols: load_tokenizer, apply_chat_template
  - `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh` added +34/-0 (34 lines); hunks: -0,0 +1,34
  - `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh` added +150/-0 (150 lines); hunks: -0,0 +1,150
  - `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/collect_outputs.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: num_gpus_from_config, decoded_tok_per_iter, collect, main
  - `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/build_swe_smith_dataset.patch` added +55/-0 (55 lines); hunks: -0,0 +1,55
- 关键代码摘录:

```diff
diff -- test/agentic_benchmark/deepseek_v4_flash/tokenspeed/deepseek_v4_tokenizer.py
@@ -0,0 +1,93 @@
+# MIT License
+#
+# Copyright (c) 2026 TokenSpeed contributors
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
diff -- test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh
@@ -0,0 +1,34 @@
+#!/usr/bin/bash
+set -euo pipefail
+# Target verification and the DSpark draft use their validated MoE backends.
+exec ts serve \
+    --model deepseek-ai/DeepSeek-V4-Flash-0731 \
+    --trust-remote-code \
diff -- test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh
@@ -0,0 +1,150 @@
```

- 提取文件（未人工审阅）:
  - tests: `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/deepseek_v4_tokenizer.py` added +93/-0; `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh` added +34/-0; `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh` added +150/-0; `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/collect_outputs.py` added +110/-0; `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/build_swe_smith_dataset.patch` added +55/-0
- 验证与风险: diff 自带测试面 `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/build_swe_smith_dataset.patch`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/collect_outputs.py`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1404 - perf(v4): reduce decode path overhead

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1404
- 状态/时间: merged / 2026-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4/slot_mappings.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` 等 12 个文件；关联提交 `ab7deb681e46`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 50 个文件，+4162/-164，可读 patch 5586 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` modified +132/-39 (171 lines); hunks: -18,6 +18,7; -27,9 +28,11; symbols: _refresh_decode_indexer_schedule_metadata, DeepseekV4AttentionBackend, __init__, _assert_active_cache_pages，涉及 `_refresh_decode_indexer_schedule_metadata, DeepseekV4AttentionBackend, __init__`；`python/tokenspeed/runtime/models/deepseek_v4.py` modified +103/-23 (126 lines); hunks: -61,7 +61,11; -249,6 +253,8 @@ def mhc_pre(; symbols: mhc_pre, mhc_fused_hc, pack_topk_as_router_logits，涉及 `mhc_pre, mhc_fused_hc, pack_topk_as_router_logits`；`python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +81/-12 (93 lines); hunks: -310,6 +310,11 @@ def __init__(; -328,29 +333,23 @@ def __init__(; symbols: __init__, _forward_stage，涉及 `__init__, _forward_stage`；`python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py` modified +15/-33 (48 lines); hunks: -17,7 +17,10; -135,39 +138,18 @@ def _update_decode_compressed_slot_mapping(; symbols: _update_decode_compressed_slot_mapping，涉及 `_update_decode_compressed_slot_mapping`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` modified +132/-39 (171 lines); hunks: -18,6 +18,7; -27,9 +28,11; symbols: _refresh_decode_indexer_schedule_metadata, DeepseekV4AttentionBackend, __init__, _assert_active_cache_pages
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +103/-23 (126 lines); hunks: -61,7 +61,11; -249,6 +253,8 @@ def mhc_pre(; symbols: mhc_pre, mhc_fused_hc, pack_topk_as_router_logits
  - `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +81/-12 (93 lines); hunks: -310,6 +310,11 @@ def __init__(; -328,29 +333,23 @@ def __init__(; symbols: __init__, _forward_stage
  - `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py` modified +15/-33 (48 lines); hunks: -17,7 +17,10; -135,39 +138,18 @@ def _update_decode_compressed_slot_mapping(; symbols: _update_decode_compressed_slot_mapping
  - `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/attention.py` modified +34/-2 (36 lines); hunks: -17,6 +17,9; -30,6 +33,14 @@ def dspark_fp8_quant_dequant(; symbols: dspark_fp8_quant_dequant, dspark_sparse_attn, _rmsnorm
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py
@@ -18,6 +18,7 @@
+    dsv4_padded_heads,
@@ -27,9 +28,11 @@
+    dsv4_decode_dense_compressed_indices_and_lens,
+    dsv4_validate_active_cache_pages,
@@ -62,7 +65,6 @@
-from tokenspeed.runtime.layers.attention.page_table import safe_page_ids
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -61,7 +61,11 @@
+from tokenspeed_kernel import (
+    pack_topk_router_logits,
+)
+    dsv4_group_slot_mapping,
@@ -249,6 +253,8 @@ def mhc_pre(
+    norm_weight: torch.Tensor | None,
diff -- python/tokenspeed/runtime/models/deepseek_v4_dspark.py
@@ -310,6 +310,11 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` modified +132/-39; `python/tokenspeed/runtime/models/deepseek_v4.py` modified +103/-23; `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +81/-12; `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py` modified +15/-33; `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/attention.py` modified +34/-2; `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py` modified +15/-15
- 验证与风险: diff 自带测试面 `test/runtime/execution/test_draft_target_wiring.py`, `test/runtime/test_cache_pool.py`, `test/runtime/test_deepseek_v4_config.py`, `test/runtime/test_deepseek_v4_slot_mappings.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1481 - test: update deepseek-v4-flash agentic bench

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1481
- 状态/时间: merged / 2026-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_tp4.sh`；关联提交 `2895cac9acea`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+40/-17，可读 patch 82 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh` modified +11/-17 (28 lines); hunks: -2,33 +2,27；`test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_tp4.sh` added +28/-0 (28 lines); hunks: -0,0 +1,28；`test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh` modified +1/-0 (1 lines); hunks: -35,6 +35,7 @@ fi。
- 代码 diff 细节:
  - `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh` modified +11/-17 (28 lines); hunks: -2,33 +2,27
  - `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_tp4.sh` added +28/-0 (28 lines); hunks: -0,0 +1,28
  - `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh` modified +1/-0 (1 lines); hunks: -35,6 +35,7 @@ fi
- 关键代码摘录:

```diff
diff -- test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh
@@ -2,33 +2,27 @@
-# Target verification and the DSpark draft use their validated MoE backends.
-    --trust-remote-code \
-    --world-size 4 \
-    --moe-tp-size 1 \
-    --moe-backend flashinfer_trtllm \
-    --draft-moe-backend mega_moe \
diff -- test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_tp4.sh
@@ -0,0 +1,28 @@
+#!/usr/bin/bash
+set -euo pipefail
+exec ts serve \
+    --model deepseek-ai/DeepSeek-V4-Flash-0731 \
+    --attn-tp-size 4 \
+    --dense-tp-size 4 \
diff -- test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh
@@ -35,6 +35,7 @@ fi
```

- 提取文件（未人工审阅）:
  - tests: `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh` modified +11/-17; `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_tp4.sh` added +28/-0; `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh` modified +1/-0
- 验证与风险: diff 自带测试面 `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/deepseek_v4_flash/tokenspeed/configs/attn_tp4_moe_tp4.sh`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1504 - fix(deepseek v4): use BF16 LM head for DP attention

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1504
- 状态/时间: merged / 2026-09-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`；关联提交 `efc2c294b127`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+12/-0，可读 patch 19 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +12/-0 (12 lines); hunks: -3733,6 +3733,18 @@ def forward(; symbols: forward, DeepseekV4ForCausalLM, resolve_lm_head, set_dspark_layers_to_capture，涉及 `forward, DeepseekV4ForCausalLM, resolve_lm_head`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +12/-0 (12 lines); hunks: -3733,6 +3733,18 @@ def forward(; symbols: forward, DeepseekV4ForCausalLM, resolve_lm_head, set_dspark_layers_to_capture
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -3733,6 +3733,18 @@ def forward(
+    def resolve_lm_head(
+        self,
+        config: PretrainedConfig,
+        quant_config: QuantizationConfig | None,
+        prefix: str,
+    ) -> nn.Module:
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +12/-0
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1549 - feat(model): support basic deepseek v4.1 flash

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1549
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/deepseek_v41_config.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` 等 20 个文件；关联提交 `eaa58e30be06`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 87 个文件，+18837/-99，可读 patch 19915 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v41.py` added +1645/-0 (1645 lines); hunks: -0,0 +1,1645; symbols: v41_quantize_fp8, v41_mxfp8_config, _ReferenceFp8LinearMethod, apply，涉及 `v41_quantize_fp8, v41_mxfp8_config, _ReferenceFp8LinearMethod`；`python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` added +1413/-0 (1413 lines); hunks: -0,0 +1,1413; symbols: V41CompressorPlan, allocate, window, V41Metadata，涉及 `V41CompressorPlan, allocate, window`；`python/tokenspeed/runtime/models/deepseek_v41_engram.py` added +892/-0 (892 lines); hunks: -0,0 +1,892; symbols: is_engram_embed_checkpoint_name, build_compressed_token_map, compute_hash_multipliers, EngramLayout，涉及 `is_engram_embed_checkpoint_name, build_compressed_token_map, compute_hash_multipliers`；`test/runtime/test_deepseek_v41_tokenizer.py` added +468/-0 (468 lines); hunks: -0,0 +1,468; symbols: base_tokenizer, test_effort_is_forwarded_without_v4_filtering, test_tokenization_has_one_bos_and_preserves_backend, test_media_is_rejected_before_encoding，涉及 `base_tokenizer, test_effort_is_forwarded_without_v4_filtering, test_tokenization_has_one_bos_and_preserves_backend`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v41.py` added +1645/-0 (1645 lines); hunks: -0,0 +1,1645; symbols: v41_quantize_fp8, v41_mxfp8_config, _ReferenceFp8LinearMethod, apply
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` added +1413/-0 (1413 lines); hunks: -0,0 +1,1413; symbols: V41CompressorPlan, allocate, window, V41Metadata
  - `python/tokenspeed/runtime/models/deepseek_v41_engram.py` added +892/-0 (892 lines); hunks: -0,0 +1,892; symbols: is_engram_embed_checkpoint_name, build_compressed_token_map, compute_hash_multipliers, EngramLayout
  - `test/runtime/test_deepseek_v41_tokenizer.py` added +468/-0 (468 lines); hunks: -0,0 +1,468; symbols: base_tokenizer, test_effort_is_forwarded_without_v4_filtering, test_tokenization_has_one_bos_and_preserves_backend, test_media_is_rejected_before_encoding
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` added +353/-0 (353 lines); hunks: -0,0 +1,353; symbols: _quantized_kv, _WindowAttention, __init__, query_metadata
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41.py` added +1645/-0; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` added +1413/-0; `python/tokenspeed/runtime/models/deepseek_v41_engram.py` added +892/-0; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` added +353/-0; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` added +280/-0; `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` added +126/-0
  - tests: `test/runtime/test_deepseek_v41_tokenizer.py` added +468/-0
- 验证与风险: diff 自带测试面 `test/cli/test_logprefix.py`, `test/cli/test_serve_smg_deepseek_v41.py`, `test/runtime/run_deepseek_v41_eval.py`, `test/runtime/test_cache_pd_shutdown.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1552 - feat(model): support basic deepseek v4.1 flash for amd

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1552
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `test/runtime/run_deepseek_v41_eval.py`, `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_config.py`, `test/runtime/test_deepseek_v41_dspark.py` 等 7 个文件；关联提交 `df3fe8aefef7`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+61/-21，可读 patch 228 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +6/-1 (7 lines); hunks: -34,6 +34,7; -95,11 +96,14 @@ def forward_v41(; symbols: forward_v41, _main_input, _main_kv，涉及 `forward_v41, _main_input, _main_kv`；`test/runtime/run_deepseek_v41_eval.py` modified +8/-3 (11 lines); hunks: -21,8 +21,9; -131,6 +132,8 @@ def main():; symbols: main，涉及 `main`；`test/runtime/test_deepseek_v41_eval.py` modified +8/-0 (8 lines); hunks: -258,6 +258,10 @@ def main_env(tmp_path, report, monkeypatch):; -347,6 +351,10 @@ def test_full_eval_writes_result_only_after_completion(; symbols: main_env, test_full_eval_writes_result_only_after_completion，涉及 `main_env, test_full_eval_writes_result_only_after_completion`；`test/runtime/test_deepseek_v41_config.py` modified +3/-2 (5 lines); hunks: -213,6 +213,7 @@ def test_registry_and_wrapper_roundtrip(config_dir, tmp_path):; -343,7 +344,7 @@ def test_config_selects_flash_recipe_and_checks_geometry(run...; symbols: test_registry_and_wrapper_roundtrip, test_config_selects_flash_recipe_and_checks_geometry, test_real_server_args_prepare_cache_pool_and_backend，涉及 `test_registry_and_wrapper_roundtrip, test_config_selects_flash_recipe_and_checks_geometry, test_real_server_args_prepare_cache_pool_and_backend`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +6/-1 (7 lines); hunks: -34,6 +34,7; -95,11 +96,14 @@ def forward_v41(; symbols: forward_v41, _main_input, _main_kv
  - `test/runtime/run_deepseek_v41_eval.py` modified +8/-3 (11 lines); hunks: -21,8 +21,9; -131,6 +132,8 @@ def main():; symbols: main
  - `test/runtime/test_deepseek_v41_eval.py` modified +8/-0 (8 lines); hunks: -258,6 +258,10 @@ def main_env(tmp_path, report, monkeypatch):; -347,6 +351,10 @@ def test_full_eval_writes_result_only_after_completion(; symbols: main_env, test_full_eval_writes_result_only_after_completion
  - `test/runtime/test_deepseek_v41_config.py` modified +3/-2 (5 lines); hunks: -213,6 +213,7 @@ def test_registry_and_wrapper_roundtrip(config_dir, tmp_path):; -343,7 +344,7 @@ def test_config_selects_flash_recipe_and_checks_geometry(run...; symbols: test_registry_and_wrapper_roundtrip, test_config_selects_flash_recipe_and_checks_geometry, test_real_server_args_prepare_cache_pool_and_backend
  - `test/runtime/test_deepseek_v41_cache.py` modified +3/-1 (4 lines); hunks: -892,6 +892,7 @@ def test_recipe_exact_geometry_capacity_and_dispatch():; -901,7 +902,8 @@ def test_recipe_exact_geometry_capacity_and_dispatch():; symbols: test_recipe_exact_geometry_capacity_and_dispatch, test_owner_topology_and_reject_invalid_recipes
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +6/-1
  - tests: `test/runtime/run_deepseek_v41_eval.py` modified +8/-3; `test/runtime/test_deepseek_v41_eval.py` modified +8/-0; `test/runtime/test_deepseek_v41_config.py` modified +3/-2; `test/runtime/test_deepseek_v41_cache.py` modified +3/-1; `test/runtime/test_deepseek_v41_dspark.py` modified +1/-0; `test/runtime/test_deepseek_v41_model.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/runtime/run_deepseek_v41_eval.py`, `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_config.py`, `test/runtime/test_deepseek_v41_dspark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1567 - feat(dsv41): support vision inputs

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1567
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/deepseek_v41_config.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/models/deepseek_v41_vision.py`, `test/cli/test_serve_smg_deepseek_v41.py` 等 8 个文件；关联提交 `cb05b7cf2574`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 12 个文件，+811/-276，可读 patch 1658 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v41_vision.py` added +300/-0 (300 lines); hunks: -0,0 +1,300; symbols: get_vision_cos_sin, apply_vision_rotary, DeepseekV41VisionPatchEmbed, __init__，涉及 `get_vision_cos_sin, apply_vision_rotary, DeepseekV41VisionPatchEmbed`；`python/tokenspeed/runtime/models/deepseek_v41.py` modified +172/-26 (198 lines); hunks: -18,7 +18,7; -64,6 +64,7; symbols: __init__, _select_experts，涉及 `__init__, _select_experts`；`python/tokenspeed/runtime/configs/deepseek_v41_config.py` modified +53/-14 (67 lines); hunks: -18,7 +18,7; -64,13 +64,55 @@ def ngram_context_len(self):; symbols: ngram_context_len, DeepseekV41VisionConfig, __init__, DeepseekV41Config，涉及 `ngram_context_len, DeepseekV41VisionConfig, __init__`；`python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +16/-7 (23 lines); hunks: -278,8 +278,8 @@ def forward_backbone(; -289,20 +289,29 @@ def forward_backbone(; symbols: forward_backbone, DeepseekV41ForCausalLMDSpark, __init__, resolve_model，涉及 `forward_backbone, DeepseekV41ForCausalLMDSpark, __init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v41_vision.py` added +300/-0 (300 lines); hunks: -0,0 +1,300; symbols: get_vision_cos_sin, apply_vision_rotary, DeepseekV41VisionPatchEmbed, __init__
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +172/-26 (198 lines); hunks: -18,7 +18,7; -64,6 +64,7; symbols: __init__, _select_experts
  - `python/tokenspeed/runtime/configs/deepseek_v41_config.py` modified +53/-14 (67 lines); hunks: -18,7 +18,7; -64,13 +64,55 @@ def ngram_context_len(self):; symbols: ngram_context_len, DeepseekV41VisionConfig, __init__, DeepseekV41Config
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +16/-7 (23 lines); hunks: -278,8 +278,8 @@ def forward_backbone(; -289,20 +289,29 @@ def forward_backbone(; symbols: forward_backbone, DeepseekV41ForCausalLMDSpark, __init__, resolve_model
  - `test/cli/test_serve_smg_deepseek_v41.py` modified +88/-82 (170 lines); hunks: -77,8 +77,9 @@ def GetModelInfo(self, request, context):; -91,6 +92,7 @@ def GetLoads(self, request, context):; symbols: GetModelInfo, HealthCheck, GetLoads, Generate
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41_vision.py` added +300/-0; `python/tokenspeed/runtime/models/deepseek_v41.py` modified +172/-26; `python/tokenspeed/runtime/configs/deepseek_v41_config.py` modified +53/-14; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +16/-7
  - tests: `test/cli/test_serve_smg_deepseek_v41.py` modified +88/-82; `test/runtime/test_deepseek_v41_model.py` modified +112/-8; `test/runtime/test_deepseek_v41_config.py` modified +49/-4; `test/runtime/test_deepseek_v41_dspark.py` modified +5/-2
- 验证与风险: diff 自带测试面 `test/cli/test_serve_smg_deepseek_v41.py`, `test/runtime/test_deepseek_v41_config.py`, `test/runtime/test_deepseek_v41_dspark.py`, `test/runtime/test_deepseek_v41_model.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1565 - feat: enable pcg for DeepSeek V4.1 Flash

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1565
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_config.py` 等 7 个文件；关联提交 `fcbfd6467d89`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+328/-27，可读 patch 531 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v41.py` modified +12/-0 (12 lines); hunks: -68,6 +68,11; -707,13 +712,20 @@ def _kernel_attn_sink(self):; symbols: _kernel_attn_sink, forward，涉及 `_kernel_attn_sink, forward`；`python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +2/-2 (4 lines); hunks: -149,8 +149,8 @@ class V41SWAQueryPlan:; symbols: V41SWAQueryPlan, DeepseekV41AttentionBackend, __init__，涉及 `V41SWAQueryPlan, DeepseekV41AttentionBackend, __init__`；`test/runtime/test_deepseek_v41_model.py` modified +119/-3 (122 lines); hunks: -774,8 +774,8 @@ def test_cuda_exact_fp8_linear_and_engram_method(monkeypatch):; -840,14 +840,130 @@ def test_cuda_40_layer_real_flatkv_and_moe(monkeypatch, t...; symbols: test_cuda_exact_fp8_linear_and_engram_method, test_cuda_40_layer_real_flatkv_and_moe, _assert_prefill_graph_matches_eager，涉及 `test_cuda_exact_fp8_linear_and_engram_method, test_cuda_40_layer_real_flatkv_and_moe, _assert_prefill_graph_matches_eager`；`python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` modified +80/-15 (95 lines); hunks: -34,6 +34,7; -44,6 +45,9; symbols: _dspark_decode_position_plan, __init__, _bonus_tokens_from_output, capture_prefill_graph，涉及 `_dspark_decode_position_plan, __init__, _bonus_tokens_from_output`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +12/-0 (12 lines); hunks: -68,6 +68,11; -707,13 +712,20 @@ def _kernel_attn_sink(self):; symbols: _kernel_attn_sink, forward
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +2/-2 (4 lines); hunks: -149,8 +149,8 @@ class V41SWAQueryPlan:; symbols: V41SWAQueryPlan, DeepseekV41AttentionBackend, __init__
  - `test/runtime/test_deepseek_v41_model.py` modified +119/-3 (122 lines); hunks: -774,8 +774,8 @@ def test_cuda_exact_fp8_linear_and_engram_method(monkeypatch):; -840,14 +840,130 @@ def test_cuda_40_layer_real_flatkv_and_moe(monkeypatch, t...; symbols: test_cuda_exact_fp8_linear_and_engram_method, test_cuda_40_layer_real_flatkv_and_moe, _assert_prefill_graph_matches_eager
  - `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` modified +80/-15 (95 lines); hunks: -34,6 +34,7; -44,6 +45,9; symbols: _dspark_decode_position_plan, __init__, _bonus_tokens_from_output, capture_prefill_graph
  - `test/runtime/test_deepseek_v41_dspark.py` modified +68/-0 (68 lines); hunks: -260,6 +260,7 @@ def test_window_attention_matches_dense_reference():; -304,6 +305,7 @@ def test_draft_forward_graph_and_context_seeding(monkeypatch):; symbols: test_window_attention_matches_dense_reference, test_draft_forward_graph_and_context_seeding, forward, _assert_prefill_graph_matches_eager
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41.py` modified +12/-0; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +2/-2; `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` modified +80/-15
  - tests: `test/runtime/test_deepseek_v41_model.py` modified +119/-3; `test/runtime/test_deepseek_v41_dspark.py` modified +68/-0; `test/runtime/test_deepseek_v41_cache.py` modified +1/-1; `test/runtime/test_deepseek_v41_config.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_config.py`, `test/runtime/test_deepseek_v41_dspark.py`, `test/runtime/test_deepseek_v41_model.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1498 - feat(dcp): Add DeepSeek V4 decode context parallelism

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1498
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4/graph_buffers.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4_geometry.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py` 等 11 个文件；关联提交 `9fb644168384`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 61 个文件，+4232/-575，可读 patch 7317 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` modified +376/-104 (480 lines); hunks: -7,9 +7,16; -24,6 +31,7; symbols: _refresh_decode_indexer_plan_cache, __init__, _configure_cache_group_contract, _init_cache_group_latches，涉及 `_refresh_decode_indexer_plan_cache, __init__, _configure_cache_group_contract`；`python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py` modified +180/-19 (199 lines); hunks: -7,9 +7,16; -21,19 +28,26; symbols: _compressed_boundary_mask, DeepseekV4CacheMetadata, from_group_tables，涉及 `_compressed_boundary_mask, DeepseekV4CacheMetadata, from_group_tables`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` modified +32/-8 (40 lines); hunks: -40,6 +40,7; -112,7 +113,7 @@ def v4_compressor_state_spec(ratio: int, *, c4_state_window:...; symbols: v4_compressor_state_spec, v4_compressed_kv_spec, v4_indexer_kv_spec, v4_indexer_state_spec，涉及 `v4_compressor_state_spec, v4_compressed_kv_spec, v4_indexer_kv_spec`；`python/tokenspeed/runtime/models/deepseek_v4.py` modified +21/-14 (35 lines); hunks: -1265,7 +1265,7 @@ def _deepseek_v4_sparse_attn_indexer(; -1304,14 +1304,14 @@ def _deepseek_v4_sparse_attn_indexer(; symbols: _deepseek_v4_sparse_attn_indexer, resolve_state_slots, resolve_compressed_slots，涉及 `_deepseek_v4_sparse_attn_indexer, resolve_state_slots, resolve_compressed_slots`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` modified +376/-104 (480 lines); hunks: -7,9 +7,16; -24,6 +31,7; symbols: _refresh_decode_indexer_plan_cache, __init__, _configure_cache_group_contract, _init_cache_group_latches
  - `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py` modified +180/-19 (199 lines); hunks: -7,9 +7,16; -21,19 +28,26; symbols: _compressed_boundary_mask, DeepseekV4CacheMetadata, from_group_tables
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` modified +32/-8 (40 lines); hunks: -40,6 +40,7; -112,7 +113,7 @@ def v4_compressor_state_spec(ratio: int, *, c4_state_window:...; symbols: v4_compressor_state_spec, v4_compressed_kv_spec, v4_indexer_kv_spec, v4_indexer_state_spec
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +21/-14 (35 lines); hunks: -1265,7 +1265,7 @@ def _deepseek_v4_sparse_attn_indexer(; -1304,14 +1304,14 @@ def _deepseek_v4_sparse_attn_indexer(; symbols: _deepseek_v4_sparse_attn_indexer, resolve_state_slots, resolve_compressed_slots
  - `python/tokenspeed/runtime/layers/attention/backends/cache_metadata.py` modified +26/-6 (32 lines); hunks: -28,7 +28,7; -46,6 +46,7 @@ class CacheBatchMetadata:; symbols: CacheBatchMetadata, __init__, from_forward_op
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py
@@ -7,9 +7,16 @@
+# The above copyright notice and this permission notice shall be included in
+# all copies or substantial portions of the Software.
+#
-# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
+# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
+# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py
@@ -7,9 +7,16 @@
+# The above copyright notice and this permission notice shall be included in
+# all copies or substantial portions of the Software.
+#
-# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
+# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
+# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py
@@ -40,6 +40,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` modified +376/-104; `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py` modified +180/-19; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` modified +32/-8; `python/tokenspeed/runtime/models/deepseek_v4.py` modified +21/-14; `python/tokenspeed/runtime/layers/attention/backends/cache_metadata.py` modified +26/-6; `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py` modified +27/-2
- 验证与风险: diff 自带测试面 `test/ci/eval/deepseek-v4-flash-dcp4-mtp-evalscope-gsm8k.yaml`, `test/ci_system/test_eval_configs.py`, `test/runtime/test_dcp_cache_contract.py`, `test/runtime/test_deepseek_v41_cache.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1594 - feat(v41): SWA bounded replay and CED decoder narrowing

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1594
- 状态/时间: merged / 2026-09-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41.py` 等 11 个文件；关联提交 `4986d149784c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 43 个文件，+1508/-290，可读 patch 3246 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +296/-57 (353 lines); hunks: -29,10 +29,17; -108,6 +115,52 @@ class V41Metadata:; symbols: V41Metadata, V41PrefillSpan, V41RowPlan, V41DecoderView，涉及 `V41Metadata, V41PrefillSpan, V41RowPlan`；`python/tokenspeed/runtime/models/deepseek_v41.py` modified +125/-19 (144 lines); hunks: -37,9 +37,14; -51,6 +56,7; symbols: _kernel_attn_sink, forward, __init__，涉及 `_kernel_attn_sink, forward, __init__`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +22/-2 (24 lines); hunks: -22,8 +22,12; -168,6 +172,21 @@ def add(; symbols: add，涉及 `add`；`python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` modified +8/-1 (9 lines); hunks: -47,7 +47,10; -910,13 +913,17 @@ def init_forward_metadata(; symbols: init_forward_metadata，涉及 `init_forward_metadata`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +296/-57 (353 lines); hunks: -29,10 +29,17; -108,6 +115,52 @@ class V41Metadata:; symbols: V41Metadata, V41PrefillSpan, V41RowPlan, V41DecoderView
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +125/-19 (144 lines); hunks: -37,9 +37,14; -51,6 +56,7; symbols: _kernel_attn_sink, forward, __init__
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +22/-2 (24 lines); hunks: -22,8 +22,12; -168,6 +172,21 @@ def add(; symbols: add
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` modified +8/-1 (9 lines); hunks: -47,7 +47,10; -910,13 +913,17 @@ def init_forward_metadata(; symbols: init_forward_metadata
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +4/-0 (4 lines); hunks: -37,6 +37,9; -280,6 +283,7 @@ def forward_backbone(; symbols: forward_backbone
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +296/-57; `python/tokenspeed/runtime/models/deepseek_v41.py` modified +125/-19; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +22/-2; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v4.py` modified +8/-1; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +4/-0
  - tests: `test/runtime/test_deepseek_v41_model.py` modified +213/-111; `test/runtime/test_deepseek_v41_cache.py` modified +306/-17; `test/runtime/test_deepseek_v41_inputs.py` modified +53/-23
- 验证与风险: diff 自带测试面 `test/runtime/test_cache_group_router.py`, `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_dspark.py`, `test/runtime/test_deepseek_v41_inputs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1595 - feat(dsv41): support basic prefill/decode disaggregation with DSpark for DeepSeek V4.1 Flash

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1595
- 状态/时间: merged / 2026-09-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/execution/drafter/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` 等 13 个文件；关联提交 `2dfc2c4dc468`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 24 个文件，+1318/-240，可读 patch 2005 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +73/-27 (100 lines); hunks: -37,6 +37,8; -45,10 +47,12; symbols: DeepseekV41Recipe, layer_types, max_padding_fraction, dspark_stages，涉及 `DeepseekV41Recipe, layer_types, max_padding_fraction`；`python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +55/-32 (87 lines); hunks: -67,19 +67,45 @@ def _quantized_kv(x: torch.Tensor) -> torch.Tensor:; -103,19 +129,16 @@ def forward_v41(; symbols: _quantized_kv, _window_rows, _write_window_rows, _WindowAttention，涉及 `_quantized_kv, _window_rows, _write_window_rows`；`python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +17/-4 (21 lines); hunks: -447,15 +447,28 @@ def _prepare_compressor(self, metadata: V41Metadata) -> None:; -1534,7 +1547,7 @@ def forward_v41(; symbols: _prepare_compressor, _decode_window, window_slots, forward_v41，涉及 `_prepare_compressor, _decode_window, window_slots`；`python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` modified +18/-0 (18 lines); hunks: -59,6 +59,24; symbols: v41_dspark_field_name, v41_layer_mapping，涉及 `v41_dspark_field_name, v41_layer_mapping`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +73/-27 (100 lines); hunks: -37,6 +37,8; -45,10 +47,12; symbols: DeepseekV41Recipe, layer_types, max_padding_fraction, dspark_stages
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +55/-32 (87 lines); hunks: -67,19 +67,45 @@ def _quantized_kv(x: torch.Tensor) -> torch.Tensor:; -103,19 +129,16 @@ def forward_v41(; symbols: _quantized_kv, _window_rows, _write_window_rows, _WindowAttention
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +17/-4 (21 lines); hunks: -447,15 +447,28 @@ def _prepare_compressor(self, metadata: V41Metadata) -> None:; -1534,7 +1547,7 @@ def forward_v41(; symbols: _prepare_compressor, _decode_window, window_slots, forward_v41
  - `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` modified +18/-0 (18 lines); hunks: -59,6 +59,24; symbols: v41_dspark_field_name, v41_layer_mapping
  - `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` modified +15/-0 (15 lines); hunks: -23,6 +23,9; -95,6 +98,18 @@ def compressor_tail(self, owner: int) -> torch.Tensor:; symbols: compressor_tail, zero_new_blocks, dspark_kv, get_key_buffer
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +73/-27; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +55/-32; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +17/-4; `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` modified +18/-0; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` modified +15/-0; `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py` modified +10/-0
  - tests: `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh` added +242/-0
- 验证与风险: diff 自带测试面 `test/ci/ut/deepseek-v4.1-flash-pd-1p1d.yaml`, `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh`, `test/runtime/distributed/test_deepseek_v41_pd_1p1d.py`, `test/runtime/test_deepseek_v41_cache.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1603 - ci: cover DeepSeek V4.1 Flash PD on GB300

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1603
- 状态/时间: merged / 2026-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml`, `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh`, `test/ci_system/test_deepseek_v41_pd_launcher.py`；关联提交 `4438b65134fc`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+468/-19，可读 patch 650 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci_system/test_deepseek_v41_pd_launcher.py` added +235/-0 (235 lines); hunks: -0,0 +1,235; symbols: launcher, start, calls, wait_for，涉及 `launcher, start, calls`；`test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh` modified +129/-17 (146 lines); hunks: -1,7 +1,8; -24,6 +25,10 @@ MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}; symbols: main，涉及 `main`；`test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml` added +57/-0 (57 lines); hunks: -0,0 +1,57。
- 代码 diff 细节:
  - `test/ci_system/test_deepseek_v41_pd_launcher.py` added +235/-0 (235 lines); hunks: -0,0 +1,235; symbols: launcher, start, calls, wait_for
  - `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh` modified +129/-17 (146 lines); hunks: -1,7 +1,8; -24,6 +25,10 @@ MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}; symbols: main
  - `test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml` added +57/-0 (57 lines); hunks: -0,0 +1,57
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - tests: `test/ci_system/test_deepseek_v41_pd_launcher.py` added +235/-0; `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh` modified +129/-17; `test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml` added +57/-0
- 验证与风险: diff 自带测试面 `test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml`, `test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh`, `test/ci_system/test_deepseek_v41_pd_launcher.py`, `test/ci_system/test_dispatch_workflows.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1611 - refactor(dsv41): retain exactly the attention window and the compressor pair

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1611
- 状态/时间: merged / 2026-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py`, `test/runtime/test_deepseek_v41_cache.py`；关联提交 `fff486b06521`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+26/-27，可读 patch 93 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +15/-17 (32 lines); hunks: -193,28 +193,26 @@ def add(; symbols: add，涉及 `add`；`test/runtime/test_deepseek_v41_cache.py` modified +8/-8 (16 lines); hunks: -1041,10 +1041,10 @@ def test_packed_config_and_recipe_capacity(verify_width,...; -1248,8 +1248,8 @@ def test_recipe_exact_geometry_capacity_and_dispatch():; symbols: test_packed_config_and_recipe_capacity, test_recipe_exact_geometry_capacity_and_dispatch, test_recipe_declares_replay_windows_for_the_private_groups，涉及 `test_packed_config_and_recipe_capacity, test_recipe_exact_geometry_capacity_and_dispatch, test_recipe_declares_replay_windows_for_the_private_groups`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +15/-17 (32 lines); hunks: -193,28 +193,26 @@ def add(; symbols: add
  - `test/runtime/test_deepseek_v41_cache.py` modified +8/-8 (16 lines); hunks: -1041,10 +1041,10 @@ def test_packed_config_and_recipe_capacity(verify_width,...; -1248,8 +1248,8 @@ def test_recipe_exact_geometry_capacity_and_dispatch():; symbols: test_packed_config_and_recipe_capacity, test_recipe_exact_geometry_capacity_and_dispatch, test_recipe_declares_replay_windows_for_the_private_groups
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +15/-17
  - tests: `test/runtime/test_deepseek_v41_cache.py` modified +8/-8
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v41_cache.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1612 - refactor(dsv4): retain the compressor state window the kernel reads

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1612
- 状态/时间: merged / 2026-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/deepseek_v4_geometry.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py`；关联提交 `a98c3fc06ed1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+37/-60，可读 patch 231 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` modified +14/-27 (41 lines); hunks: -68,22 +68,6; -96,18 +80,22 @@ def v4_swa_kv_spec(hf_config) -> CacheGroupSpec:; symbols: v4_c4_state_window, v4_swa_kv_spec, v4_compressor_state_spec，涉及 `v4_c4_state_window, v4_swa_kv_spec, v4_compressor_state_spec`；`python/tokenspeed/runtime/layers/attention/deepseek_v4_geometry.py` modified +5/-3 (8 lines); hunks: -48,9 +48,11；`test/runtime/test_deepseek_v4_config.py` modified +5/-8 (13 lines); hunks: -96,7 +96,6; -253,17 +252,16 @@ def _extend_kwargs(; symbols: _extend_kwargs, _v4_spec_set, _v4_cache_group_spec, test_deepseek_v4_spec_geometry_is_prefix_granularity_free，涉及 `_extend_kwargs, _v4_spec_set, _v4_cache_group_spec`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` modified +14/-27 (41 lines); hunks: -68,22 +68,6; -96,18 +80,22 @@ def v4_swa_kv_spec(hf_config) -> CacheGroupSpec:; symbols: v4_c4_state_window, v4_swa_kv_spec, v4_compressor_state_spec
  - `python/tokenspeed/runtime/layers/attention/deepseek_v4_geometry.py` modified +5/-3 (8 lines); hunks: -48,9 +48,11
  - `test/runtime/test_deepseek_v4_config.py` modified +5/-8 (13 lines); hunks: -96,7 +96,6; -253,17 +252,16 @@ def _extend_kwargs(; symbols: _extend_kwargs, _v4_spec_set, _v4_cache_group_spec, test_deepseek_v4_spec_geometry_is_prefix_granularity_free
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py
@@ -68,22 +68,6 @@
-def v4_c4_state_window(decode_input_tokens: int) -> int:
-    """Tokens the ratio-4 compressor state must retain.
-    c4 compression consumes the prior four-token state plus every token in the
-    target verify block. Preserve the historical eight-token window for verify
-    widths <= 4 and grow it for wider block-speculative decoders.
-    """
diff -- python/tokenspeed/runtime/layers/attention/deepseek_v4_geometry.py
@@ -48,9 +48,11 @@
-# Per compression ratio: how long the compressor tail must be retained, and how
-# many rows of it share one cache block. Both the kernel cache layout and the
-# recipe's group specs read these, so the tables live here once.
+# Per compression ratio: how long the compressor tail must be retained -- the
+# positions the compress kernel reads when a group completes (two groups for
+# the overlapping ratio-4 compressor, one for ratio 128) -- and how many rows
diff -- test/runtime/test_deepseek_v4_config.py
@@ -96,7 +96,6 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py` modified +14/-27; `python/tokenspeed/runtime/layers/attention/deepseek_v4_geometry.py` modified +5/-3
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +5/-8
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`, `test/runtime/test_v4_sliding_window_groups_smoke.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1632 - ci: keep non-DCP DeepSeek V4 Flash evaluations manual

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1632
- 状态/时间: merged / 2026-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml`, `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml`；关联提交 `a12a4a4034a6`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+0/-2，可读 patch 16 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml` modified +0/-1 (1 lines); hunks: -3,7 +3,6 @@ name: eval-deepseek-v4-flash-gsm8k；`test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml` modified +0/-1 (1 lines); hunks: -3,7 +3,6 @@ name: eval-deepseek-v4-flash-mtp-gsm8k。
- 代码 diff 细节:
  - `test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml` modified +0/-1 (1 lines); hunks: -3,7 +3,6 @@ name: eval-deepseek-v4-flash-gsm8k
  - `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml` modified +0/-1 (1 lines); hunks: -3,7 +3,6 @@ name: eval-deepseek-v4-flash-mtp-gsm8k
- 关键代码摘录:

```diff
diff -- test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml
@@ -3,7 +3,6 @@ name: eval-deepseek-v4-flash-gsm8k
-  - per-commit
diff -- test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml
@@ -3,7 +3,6 @@ name: eval-deepseek-v4-flash-mtp-gsm8k
-  - per-commit
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml` modified +0/-1; `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml` modified +0/-1
- 验证与风险: diff 自带测试面 `test/ci/eval/deepseek-v4-flash-evalscope-gsm8k.yaml`, `test/ci/eval/deepseek-v4-flash-mtp-evalscope-gsm8k.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1636 - perf(dsv41): Optimize DeepSeek V4.1-flash on H20

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1636
- 状态/时间: merged / 2026-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` 等 9 个文件；关联提交 `7c70d5805e68`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 15 个文件，+1258/-220，可读 patch 2297 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` modified +97/-21 (118 lines); hunks: -18,14 +18,21; -37,9 +44,6; symbols: V41CacheFormat, v41_dspark_field_name，涉及 `V41CacheFormat, v41_dspark_field_name`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +26/-20 (46 lines); hunks: -18,7 +18,7; -37,20 +37,13; symbols: DeepseekV41Recipe, _row_layout, layer_types, add，涉及 `DeepseekV41Recipe, _row_layout, layer_types`；`python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +34/-9 (43 lines); hunks: -231,6 +231,12 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Con...; -1034,8 +1040,12 @@ def write_global(; symbols: __init__, write_global, _selection_rows, forward_v41，涉及 `__init__, write_global, _selection_rows`；`python/tokenspeed/runtime/models/deepseek_v41.py` modified +41/-2 (43 lines); hunks: -23,6 +23,8; -68,6 +70,7; symbols: v41_mxfp8_config, _ReferenceFp8LinearMethod, __init__, process_weights_after_loading，涉及 `v41_mxfp8_config, _ReferenceFp8LinearMethod, __init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` modified +97/-21 (118 lines); hunks: -18,14 +18,21; -37,9 +44,6; symbols: V41CacheFormat, v41_dspark_field_name
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +26/-20 (46 lines); hunks: -18,7 +18,7; -37,20 +37,13; symbols: DeepseekV41Recipe, _row_layout, layer_types, add
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +34/-9 (43 lines); hunks: -231,6 +231,12 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Con...; -1034,8 +1040,12 @@ def write_global(; symbols: __init__, write_global, _selection_rows, forward_v41
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +41/-2 (43 lines); hunks: -23,6 +23,8; -68,6 +70,7; symbols: v41_mxfp8_config, _ReferenceFp8LinearMethod, __init__, process_weights_after_loading
  - `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py` modified +22/-2 (24 lines); hunks: -21,14 +21,18; -50,6 +54,7 @@ class DeepseekV41Config(SoftmaxAttnConfig):; symbols: is_deepseek_v41_config, DeepseekV41Config, generate, row_layout
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/deepseek_v41_geometry.py` modified +97/-21; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` modified +26/-20; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +34/-9; `python/tokenspeed/runtime/models/deepseek_v41.py` modified +41/-2; `python/tokenspeed/runtime/layers/attention/configs/deepseek_v41.py` modified +22/-2; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` modified +2/-2
  - tests: `test/runtime/test_deepseek_v41_cache.py` modified +75/-15; `test/runtime/test_deepseek_v41_model.py` modified +38/-16
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_config.py`, `test/runtime/test_deepseek_v41_model.py`, `tokenspeed-kernel/test/ops/test_attention_dsv41.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1623 - feat(amd): add gluon DSV4.1 CSA2 indexer

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1623
- 状态/时间: merged / 2026-09-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`；关联提交 `547b317ef089`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 16 个文件，+2001/-33，可读 patch 2359 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +1/-0 (1 lines); hunks: -1247,6 +1247,7 @@ def select_global(; symbols: select_global，涉及 `select_global`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +1/-0 (1 lines); hunks: -1247,6 +1247,7 @@ def select_global(; symbols: select_global
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py
@@ -1247,6 +1247,7 @@ def select_global(
+                None,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/ci/ut/ut-tokenspeed-kernel-mi450-sim.yaml`, `tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_index_topk.py`, `tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_graph.py`, `tokenspeed-kernel/test/ops/test_attention_dsv41.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1663 - perf(dsv41): share the native decode schedule across layers in a forward

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1663
- 状态/时间: merged / 2026-09-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `test/runtime/test_deepseek_v41_cache.py`；关联提交 `3cfd377de881`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+69/-6，可读 patch 141 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +23/-6 (29 lines); hunks: -249,6 +249,11 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Con...; -282,6 +287,7 @@ def _publish_cache_pool(self, cache_pool: CachePool) -> None:; symbols: __init__, validate_cache_pool, _publish_cache_pool, init_cuda_graph_state，涉及 `__init__, validate_cache_pool, _publish_cache_pool`；`test/runtime/test_deepseek_v41_cache.py` modified +46/-0 (46 lines); hunks: -208,6 +208,7 @@ def test_rebinding_cache_pool_drops_pool_derived_state():; -224,6 +225,7 @@ def test_rebinding_cache_pool_drops_pool_derived_state():; symbols: test_rebinding_cache_pool_drops_pool_derived_state, selected, test_native_decode_schedule_shared_per_length_pair_within_a_forward, refresh，涉及 `test_rebinding_cache_pool_drops_pool_derived_state, selected, test_native_decode_schedule_shared_per_length_pair_within_a_forward`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +23/-6 (29 lines); hunks: -249,6 +249,11 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Con...; -282,6 +287,7 @@ def _publish_cache_pool(self, cache_pool: CachePool) -> None:; symbols: __init__, validate_cache_pool, _publish_cache_pool, init_cuda_graph_state
  - `test/runtime/test_deepseek_v41_cache.py` modified +46/-0 (46 lines); hunks: -208,6 +208,7 @@ def test_rebinding_cache_pool_drops_pool_derived_state():; -224,6 +225,7 @@ def test_rebinding_cache_pool_drops_pool_derived_state():; symbols: test_rebinding_cache_pool_drops_pool_derived_state, selected, test_native_decode_schedule_shared_per_length_pair_within_a_forward, refresh
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +23/-6
  - tests: `test/runtime/test_deepseek_v41_cache.py` modified +46/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v41_cache.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1660 - perf(dsv41): run the DSpark draft window attention on selected_attention

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1660
- 状态/时间: merged / 2026-09-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_dspark.py`, `test/runtime/test_deepseek_v41_cache.py`, `test/runtime/test_deepseek_v41_config.py` 等 7 个文件；关联提交 `13657d1d6762`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 14 个文件，+1644/-138，可读 patch 2410 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v41.py` modified +302/-54 (356 lines); hunks: -47,6 +47,17; -58,8 +69,8; symbols: key, _row_plan, DeepseekV41Attention, __init__，涉及 `key, _row_plan, DeepseekV41Attention`；`python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +91/-18 (109 lines); hunks: -23,22 +23,26; -68,14 +72,35 @@ def _quantized_kv(x: torch.Tensor) -> torch.Tensor:; symbols: _quantized_kv, _window_rows, _WindowSelection, build，涉及 `_quantized_kv, _window_rows, _WindowSelection`；`python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +5/-4 (9 lines); hunks: -202,10 +202,11 @@ class V41SWAQueryPlan:; symbols: V41SWAQueryPlan, DeepseekV41AttentionBackend, __init__，涉及 `V41SWAQueryPlan, DeepseekV41AttentionBackend, __init__`；`test/runtime/test_deepseek_v41_model.py` modified +417/-17 (434 lines); hunks: -32,6 +32,7; -57,7 +58,6; symbols: _config, query_metadata, decoder_view, rows，涉及 `_config, query_metadata, decoder_view`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +302/-54 (356 lines); hunks: -47,6 +47,17; -58,8 +69,8; symbols: key, _row_plan, DeepseekV41Attention, __init__
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +91/-18 (109 lines); hunks: -23,22 +23,26; -68,14 +72,35 @@ def _quantized_kv(x: torch.Tensor) -> torch.Tensor:; symbols: _quantized_kv, _window_rows, _WindowSelection, build
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +5/-4 (9 lines); hunks: -202,10 +202,11 @@ class V41SWAQueryPlan:; symbols: V41SWAQueryPlan, DeepseekV41AttentionBackend, __init__
  - `test/runtime/test_deepseek_v41_model.py` modified +417/-17 (434 lines); hunks: -32,6 +32,7; -57,7 +58,6; symbols: _config, query_metadata, decoder_view, rows
  - `test/runtime/test_deepseek_v41_dspark.py` modified +121/-2 (123 lines); hunks: -58,6 +58,7; -66,6 +67,7; symbols: __init__, forward, test_window_attention_matches_dense_reference
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41.py` modified +302/-54; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +91/-18; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +5/-4
  - tests: `test/runtime/test_deepseek_v41_model.py` modified +417/-17; `test/runtime/test_deepseek_v41_dspark.py` modified +121/-2; `test/runtime/test_deepseek_v41_cache.py` modified +3/-2; `test/runtime/test_deepseek_v41_config.py` modified +3/-1
- 验证与风险: diff 自带测试面 `test/ci/eval/deepseek-v4.1-flash-evalscope-gsm8k.yaml`, `test/ci_system/test_eval_configs.py`, `test/runtime/execution/test_kda_prefill_graph_cache.py`, `test/runtime/test_deepseek_v41_cache.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1664 - perf(moe): FlashInfer CUTLASS MXFP4 experts on Hopper (W4A16 auto, W4A8 opt-in) + V4.1 log fixes

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1664
- 状态/时间: merged / 2026-09-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v41_engram.py` 等 8 个文件；关联提交 `f93145609230`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 55 个文件，+2411/-307，可读 patch 4046 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +38/-80 (118 lines); hunks: -68,6 +68,7; -244,7 +245,6 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Conf...; symbols: V41CompressorPlan, __init__, _publish_cache_pool, init_cuda_graph_state，涉及 `V41CompressorPlan, __init__, _publish_cache_pool`；`python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +43/-3 (46 lines); hunks: -66,13 +66,16; -721,12 +724,20 @@ def forward(self, indices: torch.Tensor) -> torch.Tensor:; symbols: forward, _load_projection_scale, engram_reduce_lane_width, DeepseekV41Engram，涉及 `forward, _load_projection_scale, engram_reduce_lane_width`；`python/tokenspeed/runtime/models/deepseek_v4.py` modified +5/-0 (5 lines); hunks: -1963,6 +1963,11 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` modified +4/-0 (4 lines); hunks: -98,6 +98,10 @@ def compressor_tail(self, owner: int) -> torch.Tensor:; symbols: compressor_tail, get_kv_size_bytes, zero_new_blocks，涉及 `compressor_tail, get_kv_size_bytes, zero_new_blocks`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +38/-80 (118 lines); hunks: -68,6 +68,7; -244,7 +245,6 @@ def __init__(self, config: AttnConfig, spec: DeepseekV41Conf...; symbols: V41CompressorPlan, __init__, _publish_cache_pool, init_cuda_graph_state
  - `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +43/-3 (46 lines); hunks: -66,13 +66,16; -721,12 +724,20 @@ def forward(self, indices: torch.Tensor) -> torch.Tensor:; symbols: forward, _load_projection_scale, engram_reduce_lane_width, DeepseekV41Engram
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +5/-0 (5 lines); hunks: -1963,6 +1963,11 @@ def __init__(; symbols: __init__
  - `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` modified +4/-0 (4 lines); hunks: -98,6 +98,10 @@ def compressor_tail(self, owner: int) -> torch.Tensor:; symbols: compressor_tail, get_kv_size_bytes, zero_new_blocks
  - `test/runtime/test_deepseek_v41_cache.py` modified +9/-104 (113 lines); hunks: -220,7 +220,6 @@ def test_rebinding_cache_pool_drops_pool_derived_state():; -290,29 +289,6 @@ def _final(lengths, prefixes):; symbols: test_rebinding_cache_pool_drops_pool_derived_state, _final, test_refresh_rejects_missing_history_before_execution, test_packed_refresh_positions_padding_and_token_count
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +38/-80; `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +43/-3; `python/tokenspeed/runtime/models/deepseek_v4.py` modified +5/-0; `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v41.py` modified +4/-0; `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` modified +5/-1
  - tests: `test/runtime/test_deepseek_v41_cache.py` modified +9/-104; `test/runtime/test_deepseek_v41_inputs.py` modified +58/-0; `test/runtime/test_deepseek_v41_engram.py` modified +14/-0
- 验证与风险: diff 自带测试面 `test/runtime/layers/test_moe_layer_activation_flag.py`, `test/runtime/layers/test_moe_layer_plan_traits.py`, `test/runtime/test_batch_log.py`, `test/runtime/test_deepseek_v41_cache.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1686 - perf(dsv41): fuse the DSpark decode prologue, draft tail and sampler sync into a handful of launches

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1686
- 状态/时间: merged / 2026-09-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/execution/drafter/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` 等 11 个文件；关联提交 `e87f7267c0b3`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 43 个文件，+3759/-607，可读 patch 5117 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py` modified +120/-115 (235 lines); hunks: -2,122 +2,45; -144,27 +67,109 @@ def forward(; symbols: _local_vocab_argmax, DSparkVanillaMarkov, __init__, local_bias，涉及 `_local_vocab_argmax, DSparkVanillaMarkov, __init__`；`python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +96/-62 (158 lines); hunks: -54,6 +54,7; -81,27 +82,14 @@ class _WindowSelection:; symbols: _WindowSelection, build, _write_window_rows, _WindowAttention，涉及 `_WindowSelection, build, _write_window_rows`；`python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +17/-64 (81 lines); hunks: -336,17 +336,18 @@ def __init__(; -364,9 +365,6 @@ def __init__(; symbols: __init__, forward_backbone, local_base_logits, refresh_local_base_logits_head，涉及 `__init__, forward_backbone, local_base_logits`；`python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +20/-10 (30 lines); hunks: -468,17 +468,27 @@ def window_slots(; symbols: window_slots, _refresh_decode_window，涉及 `window_slots, _refresh_decode_window`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py` modified +120/-115 (235 lines); hunks: -2,122 +2,45; -144,27 +67,109 @@ def forward(; symbols: _local_vocab_argmax, DSparkVanillaMarkov, __init__, local_bias
  - `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +96/-62 (158 lines); hunks: -54,6 +54,7; -81,27 +82,14 @@ class _WindowSelection:; symbols: _WindowSelection, build, _write_window_rows, _WindowAttention
  - `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +17/-64 (81 lines); hunks: -336,17 +336,18 @@ def __init__(; -364,9 +365,6 @@ def __init__(; symbols: __init__, forward_backbone, local_base_logits, refresh_local_base_logits_head
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +20/-10 (30 lines); hunks: -468,17 +468,27 @@ def window_slots(; symbols: window_slots, _refresh_decode_window
  - `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +15/-0 (15 lines); hunks: -290,6 +290,21 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4_dspark_ops/heads.py` modified +120/-115; `python/tokenspeed/runtime/models/deepseek_v41_dspark.py` modified +96/-62; `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +17/-64; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +20/-10; `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +15/-0; `python/tokenspeed/runtime/models/deepseek_v41.py` modified +5/-1
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +95/-142; `test/runtime/test_deepseek_v41_dspark.py` modified +57/-10
- 验证与风险: diff 自带测试面 `test/runtime/execution/test_draft_target_wiring.py`, `test/runtime/layers/test_vocab_parallel_embedding.py`, `test/runtime/sampling/test_verify_output_pack.py`, `test/runtime/test_deepseek_v41_dspark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1701 - refactor(kernel): clean up dsv4 mega moe apis

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1701
- 状态/时间: merged / 2026-09-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/models/deepseek_v4_next.py`, `test/runtime/test_deepseek_v41_model.py` 等 7 个文件；关联提交 `828b34272a20`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 40 个文件，+952/-991，可读 patch 3603 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +175/-357 (532 lines); hunks: -35,23 +35,16; -133,8 +126,11; symbols: hc_head, pack_topk_as_router_logits, _DeepseekV4IndexerPrefillChunk, forward，涉及 `hc_head, pack_topk_as_router_logits, _DeepseekV4IndexerPrefillChunk`；`python/tokenspeed/runtime/models/deepseek_v41.py` modified +12/-16 (28 lines); hunks: -30,8 +30,8; -112,6 +112,7; symbols: __init__, _select_experts, _renormalize_routing_weights, _routing_inputs，涉及 `__init__, _select_experts, _renormalize_routing_weights`；`python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +0/-3 (3 lines); hunks: -37,7 +37,6; -867,8 +866,6 @@ def post_load_weights(self) -> None:; symbols: post_load_weights，涉及 `post_load_weights`；`python/tokenspeed/runtime/models/deepseek_v4_next.py` modified +0/-3 (3 lines); hunks: -50,7 +50,6; -582,8 +581,6 @@ def post_load_weights(self):; symbols: post_load_weights，涉及 `post_load_weights`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +175/-357 (532 lines); hunks: -35,23 +35,16; -133,8 +126,11; symbols: hc_head, pack_topk_as_router_logits, _DeepseekV4IndexerPrefillChunk, forward
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +12/-16 (28 lines); hunks: -30,8 +30,8; -112,6 +112,7; symbols: __init__, _select_experts, _renormalize_routing_weights, _routing_inputs
  - `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +0/-3 (3 lines); hunks: -37,7 +37,6; -867,8 +866,6 @@ def post_load_weights(self) -> None:; symbols: post_load_weights
  - `python/tokenspeed/runtime/models/deepseek_v4_next.py` modified +0/-3 (3 lines); hunks: -50,7 +50,6; -582,8 +581,6 @@ def post_load_weights(self):; symbols: post_load_weights
  - `test/runtime/test_deepseek_v4_config.py` modified +126/-73 (199 lines); hunks: -29,6 +29,7; -112,6 +113,7; symbols: advance_draft_forward_metadata, _bind_deepseek_v4_moe_methods, _make_fake_deepseek_v4_moe, select_experts
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -35,23 +35,16 @@
-    NoKernelFoundError,
+    dsv4_linear_fp32,
-from tokenspeed_kernel import dsv4_linear_fp32 as _kernel_dsv4_linear_fp32
-from tokenspeed_kernel import (
-    dsv4_mega_moe_apply,
-    dsv4_mega_moe_plan,
diff -- python/tokenspeed/runtime/models/deepseek_v41.py
@@ -30,8 +30,8 @@
-The generic model loader still owns
-dense/MoELayer postprocessing, while post_load_weights finalizes MegaMoE.
+The generic model loader owns dense and MoELayer postprocessing, including
+MegaMoE weight preparation.
@@ -112,6 +112,7 @@
+from tokenspeed.runtime.layers.moe.expert import MoELayer
diff -- python/tokenspeed/runtime/models/deepseek_v4_dspark.py
@@ -37,7 +37,6 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +175/-357; `python/tokenspeed/runtime/models/deepseek_v41.py` modified +12/-16; `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` modified +0/-3; `python/tokenspeed/runtime/models/deepseek_v4_next.py` modified +0/-3
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +126/-73; `test/runtime/test_deepseek_v4_mega_moe.py` modified +34/-39; `test/runtime/test_deepseek_v41_model.py` modified +38/-10
- 验证与风险: diff 自带测试面 `test/ci/ut/ut-tokenspeed-kernel-mi450-sim.yaml`, `test/runtime/layers/test_moe_expert.py`, `test/runtime/test_deepseek_v41_model.py`, `test/runtime/test_deepseek_v4_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1709 - refactor(kernel): remove dsv4 pack topk router

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1709
- 状态/时间: merged / 2026-09-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`, `test/runtime/test_deepseek_v4_config.py`；关联提交 `4c9e8bcf6134`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+8/-342，可读 patch 483 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +2/-36 (38 lines); hunks: -45,7 +45,6; -124,14 +123,7; symbols: __init__, forward，涉及 `__init__, forward`；`test/runtime/test_deepseek_v4_config.py` modified +6/-24 (30 lines); hunks: -113,7 +113,6; -778,7 +777,7 @@ def shared_experts(states):; symbols: shared_experts, test_deepseek_v4_topk_builds_standard_and_bypassed_outputs, test_deepseek_v4_topk_builds_precomputed_output, fake_moe_topk，涉及 `shared_experts, test_deepseek_v4_topk_builds_standard_and_bypassed_outputs, test_deepseek_v4_topk_builds_precomputed_output`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +2/-36 (38 lines); hunks: -45,7 +45,6; -124,14 +123,7; symbols: __init__, forward
  - `test/runtime/test_deepseek_v4_config.py` modified +6/-24 (30 lines); hunks: -113,7 +113,6; -778,7 +777,7 @@ def shared_experts(states):; symbols: shared_experts, test_deepseek_v4_topk_builds_standard_and_bypassed_outputs, test_deepseek_v4_topk_builds_precomputed_output, fake_moe_topk
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -45,7 +45,6 @@
-    pack_topk_router_logits,
@@ -124,14 +123,7 @@
-from tokenspeed.runtime.layers.moe.topk import (
-    BypassedTopKOutput,
-    ExpertLocationDispatchInfo,
-    StandardTopKOutput,
diff -- test/runtime/test_deepseek_v4_config.py
@@ -113,7 +113,6 @@
-from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
@@ -778,7 +777,7 @@ def shared_experts(states):
-    def test_deepseek_v4_topk_builds_standard_and_bypassed_outputs(self):
+    def test_deepseek_v4_topk_builds_precomputed_output(self):
@@ -791,36 +790,18 @@ def fake_moe_topk(*args, **kwargs):
-            standard = DeepseekV4TopK(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +2/-36
  - tests: `test/runtime/test_deepseek_v4_config.py` modified +6/-24
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`, `tokenspeed-kernel/test/ops/test_pack_topk_router_logits.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1748 - ci: turn on dspark for dsv4.1 ci

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1748
- 状态/时间: merged / 2026-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml`, `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml`；关联提交 `162de66c7166`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+15/-8，可读 patch 76 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml` renamed +2/-1 (3 lines); hunks: -1,5 +1,5; -16,6 +16,7 @@ server:；`test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml` renamed +2/-1 (3 lines); hunks: -1,5 +1,5; -16,6 +16,7 @@ server:。
- 代码 diff 细节:
  - `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml` renamed +2/-1 (3 lines); hunks: -1,5 +1,5; -16,6 +16,7 @@ server:
  - `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml` renamed +2/-1 (3 lines); hunks: -1,5 +1,5; -16,6 +16,7 @@ server:
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml` renamed +2/-1; `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml` renamed +2/-1
- 验证与风险: diff 自带测试面 `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml`, `test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml`, `test/ci_system/test_eval_configs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1844 - feat(dsv41): support vit batching

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1844
- 状态/时间: merged / 2026-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v41_vision.py`, `test/runtime/test_deepseek_v41_vision.py`；关联提交 `b9544177b196`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+557/-62，可读 patch 728 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v41_vision.py` modified +108/-19 (127 lines); hunks: -23,6 +23,7; -34,7 +35,9; symbols: __init__, forward，涉及 `__init__, forward`；`test/runtime/test_deepseek_v41_vision.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: _model, _item, test_batch_limits_and_order, encode，涉及 `_model, _item, test_batch_limits_and_order`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v41_vision.py` modified +108/-19 (127 lines); hunks: -23,6 +23,7; -34,7 +35,9; symbols: __init__, forward
  - `test/runtime/test_deepseek_v41_vision.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: _model, _item, test_batch_limits_and_order, encode
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41_vision.py` modified +108/-19
  - tests: `test/runtime/test_deepseek_v41_vision.py` added +224/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v41_vision.py`, `test/runtime/test_encoder_batching.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1841 - fix(dcp): support FP8 query gathers and DeepGEMM V4 indexing

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1841
- 状态/时间: merged / 2026-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v4.py`；关联提交 `e0c2f513744d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+383/-77，可读 patch 595 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +14/-30 (44 lines); hunks: -47,9 +47,9; -59,7 +59,6; symbols: _forward_sharded_indexer，涉及 `_forward_sharded_indexer`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v4.py` modified +14/-30 (44 lines); hunks: -47,9 +47,9; -59,7 +59,6; symbols: _forward_sharded_indexer
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v4.py
@@ -47,9 +47,9 @@
-from tokenspeed_kernel.ops.attention.dsa.triton import triton_dsa_index_candidates
+    dsv4_index_candidates,
@@ -59,7 +59,6 @@
-    triton_dsv4_index_candidates,
@@ -2412,34 +2411,19 @@ def _forward_sharded_indexer(
-            if self.use_fp4_cache:
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v4.py` modified +14/-30
- 验证与风险: diff 自带测试面 `test/runtime/test_flashmla_dcp.py`, `tokenspeed-kernel/test/ops/test_attention_dsv4_dcp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1863 - perf(dsv41): skip decoder work for incomplete prefill chunks

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1863
- 状态/时间: merged / 2026-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/execution/drafter/deepseek_v41_dspark.py`, `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41.py`, `test/runtime/test_deepseek_v41_cache.py` 等 8 个文件；关联提交 `cb127fde197a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 39 个文件，+1842/-173，可读 patch 3184 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v41.py` modified +44/-21 (65 lines); hunks: -809,6 +809,18 @@ def _kernel_attn_sink(self):; -834,6 +846,9 @@ def forward(; symbols: _kernel_attn_sink, _write_global_kv, forward，涉及 `_kernel_attn_sink, _write_global_kv, forward`；`python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +19/-9 (28 lines); hunks: -210,6 +210,8 @@ class DeepseekV41AttentionBackend(AttentionBackend):; -671,20 +673,27 @@ def _build_decoder_view(; symbols: DeepseekV41AttentionBackend, __init__, _build_decoder_view，涉及 `DeepseekV41AttentionBackend, __init__, _build_decoder_view`；`test/runtime/test_deepseek_v41_model.py` modified +54/-15 (69 lines); hunks: -57,6 +57,7; -189,6 +190,12 @@ def _ctx(backend, tokens, mode):; symbols: _ctx, test_decoder_narrowing_projects_global_from_all_rows_then_runs_the_tail, observe_mixes, observe_qkv，涉及 `_ctx, test_decoder_narrowing_projects_global_from_all_rows_then_runs_the_tail, observe_mixes`；`test/runtime/test_deepseek_v41_cache.py` modified +50/-15 (65 lines); hunks: -570,14 +570,20 @@ def test_write_global_masks_rows_below_the_replay_floor(mo...; -608,10 +614,29 @@ def test_decoder_view_is_the_identity_for_decode_and_compl...; symbols: test_write_global_masks_rows_below_the_replay_floor, test_decoder_view_is_the_identity_for_decode_and_complete_short_chunks, test_decoder_view_keeps_one_row_per_open_chunk_and_the_final_window, test_decoder_view_preserves_compute_rows_without_open_chunk_logits，涉及 `test_write_global_masks_rows_below_the_replay_floor, test_decoder_view_is_the_identity_for_decode_and_complete_short_chunks, test_decoder_view_keeps_one_row_per_open_chunk_and_the_final_window`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +44/-21 (65 lines); hunks: -809,6 +809,18 @@ def _kernel_attn_sink(self):; -834,6 +846,9 @@ def forward(; symbols: _kernel_attn_sink, _write_global_kv, forward
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +19/-9 (28 lines); hunks: -210,6 +210,8 @@ class DeepseekV41AttentionBackend(AttentionBackend):; -671,20 +673,27 @@ def _build_decoder_view(; symbols: DeepseekV41AttentionBackend, __init__, _build_decoder_view
  - `test/runtime/test_deepseek_v41_model.py` modified +54/-15 (69 lines); hunks: -57,6 +57,7; -189,6 +190,12 @@ def _ctx(backend, tokens, mode):; symbols: _ctx, test_decoder_narrowing_projects_global_from_all_rows_then_runs_the_tail, observe_mixes, observe_qkv
  - `test/runtime/test_deepseek_v41_cache.py` modified +50/-15 (65 lines); hunks: -570,14 +570,20 @@ def test_write_global_masks_rows_below_the_replay_floor(mo...; -608,10 +614,29 @@ def test_decoder_view_is_the_identity_for_decode_and_compl...; symbols: test_write_global_masks_rows_below_the_replay_floor, test_decoder_view_is_the_identity_for_decode_and_complete_short_chunks, test_decoder_view_keeps_one_row_per_open_chunk_and_the_final_window, test_decoder_view_preserves_compute_rows_without_open_chunk_logits
  - `python/tokenspeed/runtime/execution/drafter/deepseek_v41_dspark.py` modified +14/-4 (18 lines); hunks: -46,6 +46,7; -164,6 +165,7 @@ def _draft_decode_rows(; symbols: _draft_decode_rows, run, draft
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41.py` modified +44/-21; `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py` modified +19/-9; `python/tokenspeed/runtime/execution/drafter/deepseek_v41_dspark.py` modified +14/-4; `python/tokenspeed/runtime/execution/drafter/deepseek_v4_dspark.py` modified +2/-0
  - tests: `test/runtime/test_deepseek_v41_model.py` modified +54/-15; `test/runtime/test_deepseek_v41_cache.py` modified +50/-15; `test/runtime/test_deepseek_v41_inputs.py` modified +5/-0; `test/runtime/test_deepseek_v41_dspark.py` modified +4/-0
- 验证与风险: diff 自带测试面 `test/runtime/distributed/test_draft_moe_capture_global_bs.py`, `test/runtime/execution/test_draft_target_wiring.py`, `test/runtime/models/test_qwen3_moe_models.py`, `test/runtime/sampling/test_mixed_batch_sampling.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1971 - refactor(linear): select V4.1 FP8 methods during construction

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1971
- 状态/时间: merged / 2026-10-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v41.py`, `python/tokenspeed/runtime/models/deepseek_v41_engram.py`, `test/runtime/test_deepseek_v41_engram.py`, `test/runtime/test_deepseek_v41_model.py`；关联提交 `408f926466b0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+445/-119，可读 patch 812 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v41.py` modified +68/-60 (128 lines); hunks: -193,7 +193,7 @@ def v41_quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor,...; -205,7 +205,7 @@ def v41_mxfp8_config(quant_config: QuantizationConfig | None...; symbols: v41_quantize_fp8, v41_mxfp8_config, _V41Fp8Config，涉及 `v41_quantize_fp8, v41_mxfp8_config, _V41Fp8Config`；`python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +12/-30 (42 lines); hunks: -38,9 +38,9; -67,9 +67,10; symbols: forward, _load_projection_scale, engram_reduce_lane_width, __init__，涉及 `forward, _load_projection_scale, engram_reduce_lane_width`；`test/runtime/test_deepseek_v41_model.py` modified +106/-17 (123 lines); hunks: -77,7 +77,6; -432,23 +431,31 @@ def test_scale_expansion_then_tp_and_merged_sharding():; symbols: test_scale_expansion_then_tp_and_merged_sharding, test_reference_linear_construction_preserves_checkpoint_loading, _mix_reference，涉及 `test_scale_expansion_then_tp_and_merged_sharding, test_reference_linear_construction_preserves_checkpoint_loading, _mix_reference`；`test/runtime/test_deepseek_v41_engram.py` modified +13/-4 (17 lines); hunks: -46,7 +46,9; -340,7 +342,7 @@ def _model(device, quant_config):; symbols: _model, test_gate_matches_reference_and_mask_is_identity, test_quantized_projection_loader_aliases_and_real_table_metadata，涉及 `_model, test_gate_matches_reference_and_mask_is_identity, test_quantized_projection_loader_aliases_and_real_table_metadata`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v41.py` modified +68/-60 (128 lines); hunks: -193,7 +193,7 @@ def v41_quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor,...; -205,7 +205,7 @@ def v41_mxfp8_config(quant_config: QuantizationConfig | None...; symbols: v41_quantize_fp8, v41_mxfp8_config, _V41Fp8Config
  - `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +12/-30 (42 lines); hunks: -38,9 +38,9; -67,9 +67,10; symbols: forward, _load_projection_scale, engram_reduce_lane_width, __init__
  - `test/runtime/test_deepseek_v41_model.py` modified +106/-17 (123 lines); hunks: -77,7 +77,6; -432,23 +431,31 @@ def test_scale_expansion_then_tp_and_merged_sharding():; symbols: test_scale_expansion_then_tp_and_merged_sharding, test_reference_linear_construction_preserves_checkpoint_loading, _mix_reference
  - `test/runtime/test_deepseek_v41_engram.py` modified +13/-4 (17 lines); hunks: -46,7 +46,9; -340,7 +342,7 @@ def _model(device, quant_config):; symbols: _model, test_gate_matches_reference_and_mask_is_identity, test_quantized_projection_loader_aliases_and_real_table_metadata
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v41.py` modified +68/-60; `python/tokenspeed/runtime/models/deepseek_v41_engram.py` modified +12/-30
  - tests: `test/runtime/test_deepseek_v41_model.py` modified +106/-17; `test/runtime/test_deepseek_v41_engram.py` modified +13/-4
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v41_engram.py`, `test/runtime/test_deepseek_v41_model.py`, `test/runtime/test_linear_method_construction.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
