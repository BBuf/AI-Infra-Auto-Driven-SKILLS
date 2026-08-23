# vllm DeepSeek V4 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-confidence-TP4.yaml` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml` | [#47972](https://github.com/vllm-project/vllm/pull/47972) |
| `tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml` | [#47972](https://github.com/vllm-project/vllm/pull/47972) |
| `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml` | [#42111](https://github.com/vllm-project/vllm/pull/42111) |
| `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#42353](https://github.com/vllm-project/vllm/pull/42353), [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#45681](https://github.com/vllm-project/vllm/pull/45681), [#49236](https://github.com/vllm-project/vllm/pull/49236), [#52836](https://github.com/vllm-project/vllm/pull/52836) |
| `tests/models/test_deepseek_v4_mega_moe.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#43004](https://github.com/vllm-project/vllm/pull/43004), [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43632](https://github.com/vllm-project/vllm/pull/43632), [#51368](https://github.com/vllm-project/vllm/pull/51368), [#53040](https://github.com/vllm-project/vllm/pull/53040) |
| `tests/parser/engine/test_deepseek_v4.py` | [#45877](https://github.com/vllm-project/vllm/pull/45877), [#51296](https://github.com/vllm-project/vllm/pull/51296) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_1.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_2.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_3.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_4.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_1.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_2.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_3.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_4.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/test_deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#40982](https://github.com/vllm-project/vllm/pull/40982), [#50580](https://github.com/vllm-project/vllm/pull/50580) |
| `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py` | [#43477](https://github.com/vllm-project/vllm/pull/43477) |
| `vllm/models/deepseek_v4/__init__.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953), [#43004](https://github.com/vllm-project/vllm/pull/43004), [#43077](https://github.com/vllm-project/vllm/pull/43077), [#47419](https://github.com/vllm-project/vllm/pull/47419), [#47677](https://github.com/vllm-project/vllm/pull/47677) |
| `vllm/models/deepseek_v4/amd/__init__.py` | [#43004](https://github.com/vllm-project/vllm/pull/43004) |
| `vllm/models/deepseek_v4/amd/dspark.py` | [#47419](https://github.com/vllm-project/vllm/pull/47419), [#51145](https://github.com/vllm-project/vllm/pull/51145), [#52737](https://github.com/vllm-project/vllm/pull/52737) |
| `vllm/models/deepseek_v4/amd/model.py` | [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43385](https://github.com/vllm-project/vllm/pull/43385), [#43629](https://github.com/vllm-project/vllm/pull/43629), [#43679](https://github.com/vllm-project/vllm/pull/43679), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#43950](https://github.com/vllm-project/vllm/pull/43950), [#44246](https://github.com/vllm-project/vllm/pull/44246), [#44262](https://github.com/vllm-project/vllm/pull/44262), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#45931](https://github.com/vllm-project/vllm/pull/45931), [#46720](https://github.com/vllm-project/vllm/pull/46720), ... (18 total) |
| `vllm/models/deepseek_v4/amd/mtp.py` | [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43385](https://github.com/vllm-project/vllm/pull/43385), [#43629](https://github.com/vllm-project/vllm/pull/43629), [#43679](https://github.com/vllm-project/vllm/pull/43679), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#43950](https://github.com/vllm-project/vllm/pull/43950), [#44821](https://github.com/vllm-project/vllm/pull/44821), [#45931](https://github.com/vllm-project/vllm/pull/45931), [#48044](https://github.com/vllm-project/vllm/pull/48044) |
| `vllm/models/deepseek_v4/amd/rocm.py` | [#43149](https://github.com/vllm-project/vllm/pull/43149), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43385](https://github.com/vllm-project/vllm/pull/43385), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#44699](https://github.com/vllm-project/vllm/pull/44699), [#45681](https://github.com/vllm-project/vllm/pull/45681), [#46720](https://github.com/vllm-project/vllm/pull/46720), [#47419](https://github.com/vllm-project/vllm/pull/47419), [#51538](https://github.com/vllm-project/vllm/pull/51538), [#52212](https://github.com/vllm-project/vllm/pull/52212) |
| `vllm/models/deepseek_v4/attention.py` | [#43039](https://github.com/vllm-project/vllm/pull/43039), [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43149](https://github.com/vllm-project/vllm/pull/43149), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43477](https://github.com/vllm-project/vllm/pull/43477), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#43891](https://github.com/vllm-project/vllm/pull/43891), [#44246](https://github.com/vllm-project/vllm/pull/44246), [#44561](https://github.com/vllm-project/vllm/pull/44561), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#45309](https://github.com/vllm-project/vllm/pull/45309), [#45972](https://github.com/vllm-project/vllm/pull/45972), ... (24 total) |
| `vllm/models/deepseek_v4/common/__init__.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073) |
| `vllm/models/deepseek_v4/common/ops/__init__.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#43827](https://github.com/vllm-project/vllm/pull/43827) |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#45681](https://github.com/vllm-project/vllm/pull/45681), [#47493](https://github.com/vllm-project/vllm/pull/47493), [#49236](https://github.com/vllm-project/vllm/pull/49236), [#50298](https://github.com/vllm-project/vllm/pull/50298), [#51538](https://github.com/vllm-project/vllm/pull/51538), [#51967](https://github.com/vllm-project/vllm/pull/51967), [#52084](https://github.com/vllm-project/vllm/pull/52084), [#52836](https://github.com/vllm-project/vllm/pull/52836) |
| `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#47718](https://github.com/vllm-project/vllm/pull/47718), [#52212](https://github.com/vllm-project/vllm/pull/52212) |
| `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#45991](https://github.com/vllm-project/vllm/pull/45991), [#46730](https://github.com/vllm-project/vllm/pull/46730), [#49236](https://github.com/vllm-project/vllm/pull/49236), [#50803](https://github.com/vllm-project/vllm/pull/50803), [#52836](https://github.com/vllm-project/vllm/pull/52836) |
| `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` | [#42950](https://github.com/vllm-project/vllm/pull/42950), [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43477](https://github.com/vllm-project/vllm/pull/43477) |
| `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` | [#43746](https://github.com/vllm-project/vllm/pull/43746) |
| `vllm/models/deepseek_v4/common/ops/save_partial_states.py` | [#43710](https://github.com/vllm-project/vllm/pull/43710) |
| `vllm/models/deepseek_v4/common/rope.py` | [#44262](https://github.com/vllm-project/vllm/pull/44262) |
| `vllm/models/deepseek_v4/compressor.py` | [#42950](https://github.com/vllm-project/vllm/pull/42950), [#42953](https://github.com/vllm-project/vllm/pull/42953), [#43039](https://github.com/vllm-project/vllm/pull/43039), [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43477](https://github.com/vllm-project/vllm/pull/43477), [#43690](https://github.com/vllm-project/vllm/pull/43690), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#47474](https://github.com/vllm-project/vllm/pull/47474), [#47493](https://github.com/vllm-project/vllm/pull/47493), [#47718](https://github.com/vllm-project/vllm/pull/47718), [#48957](https://github.com/vllm-project/vllm/pull/48957), ... (14 total) |
| `vllm/models/deepseek_v4/nvidia/__init__.py` | [#43004](https://github.com/vllm-project/vllm/pull/43004) |
| `vllm/models/deepseek_v4/nvidia/dspark.py` | [#46789](https://github.com/vllm-project/vllm/pull/46789), [#47429](https://github.com/vllm-project/vllm/pull/47429), [#49415](https://github.com/vllm-project/vllm/pull/49415), [#51368](https://github.com/vllm-project/vllm/pull/51368) |
| `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` | [#43477](https://github.com/vllm-project/vllm/pull/43477), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#44699](https://github.com/vllm-project/vllm/pull/44699), [#44892](https://github.com/vllm-project/vllm/pull/44892), [#45863](https://github.com/vllm-project/vllm/pull/45863), [#47493](https://github.com/vllm-project/vllm/pull/47493), [#48047](https://github.com/vllm-project/vllm/pull/48047), [#49236](https://github.com/vllm-project/vllm/pull/49236), [#51538](https://github.com/vllm-project/vllm/pull/51538), [#52836](https://github.com/vllm-project/vllm/pull/52836) |
| `vllm/models/deepseek_v4/nvidia/flashmla.py` | [#43149](https://github.com/vllm-project/vllm/pull/43149), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#44699](https://github.com/vllm-project/vllm/pull/44699), [#45061](https://github.com/vllm-project/vllm/pull/45061), [#49236](https://github.com/vllm-project/vllm/pull/49236), [#50298](https://github.com/vllm-project/vllm/pull/50298), [#52836](https://github.com/vllm-project/vllm/pull/52836) |
| `vllm/models/deepseek_v4/nvidia/model.py` | [#42925](https://github.com/vllm-project/vllm/pull/42925), [#42950](https://github.com/vllm-project/vllm/pull/42950), [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43149](https://github.com/vllm-project/vllm/pull/43149), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43339](https://github.com/vllm-project/vllm/pull/43339), [#43477](https://github.com/vllm-project/vllm/pull/43477), [#43632](https://github.com/vllm-project/vllm/pull/43632), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#43829](https://github.com/vllm-project/vllm/pull/43829), [#43891](https://github.com/vllm-project/vllm/pull/43891), ... (29 total) |
| `vllm/models/deepseek_v4/nvidia/mtp.py` | [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#43829](https://github.com/vllm-project/vllm/pull/43829), [#43905](https://github.com/vllm-project/vllm/pull/43905), [#44821](https://github.com/vllm-project/vllm/pull/44821), [#46789](https://github.com/vllm-project/vllm/pull/46789), [#49415](https://github.com/vllm-project/vllm/pull/49415), [#51368](https://github.com/vllm-project/vllm/pull/51368) |
| `vllm/models/deepseek_v4/nvidia/ops/__init__.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710) |
| `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073) |
| `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073) |
| `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` | [#44569](https://github.com/vllm-project/vllm/pull/44569), [#45681](https://github.com/vllm-project/vllm/pull/45681) |
| `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` | [#43632](https://github.com/vllm-project/vllm/pull/43632), [#53040](https://github.com/vllm-project/vllm/pull/53040) |
| `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` | [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#44161](https://github.com/vllm-project/vllm/pull/44161), [#44236](https://github.com/vllm-project/vllm/pull/44236), [#49236](https://github.com/vllm-project/vllm/pull/49236), [#52836](https://github.com/vllm-project/vllm/pull/52836) |
| `vllm/models/deepseek_v4/quant_config.py` | [#42209](https://github.com/vllm-project/vllm/pull/42209), [#43004](https://github.com/vllm-project/vllm/pull/43004), [#44914](https://github.com/vllm-project/vllm/pull/44914), [#48044](https://github.com/vllm-project/vllm/pull/48044), [#49634](https://github.com/vllm-project/vllm/pull/49634) |
| `vllm/models/deepseek_v4/sparse_mla.py` | [#43477](https://github.com/vllm-project/vllm/pull/43477), [#44699](https://github.com/vllm-project/vllm/pull/44699), [#44892](https://github.com/vllm-project/vllm/pull/44892), [#47474](https://github.com/vllm-project/vllm/pull/47474), [#50004](https://github.com/vllm-project/vllm/pull/50004), [#51318](https://github.com/vllm-project/vllm/pull/51318), [#52823](https://github.com/vllm-project/vllm/pull/52823) |
| `vllm/models/deepseek_v4/xpu/__init__.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953) |
| `vllm/models/deepseek_v4/xpu/dspark.py` | [#47677](https://github.com/vllm-project/vllm/pull/47677) |
| `vllm/models/deepseek_v4/xpu/model.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953), [#44144](https://github.com/vllm-project/vllm/pull/44144), [#47677](https://github.com/vllm-project/vllm/pull/47677), [#50312](https://github.com/vllm-project/vllm/pull/50312) |
| `vllm/models/deepseek_v4/xpu/mtp.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#44821](https://github.com/vllm-project/vllm/pull/44821), [#45240](https://github.com/vllm-project/vllm/pull/45240) |
| `vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953) |
| `vllm/models/deepseek_v4/xpu/xpu_sparse.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953) |
| `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953) |
| `vllm/parser/deepseek_v4.py` | [#45877](https://github.com/vllm-project/vllm/pull/45877), [#51296](https://github.com/vllm-project/vllm/pull/51296) |
| `vllm/reasoning/deepseek_v4_engine_reasoning_parser.py` | [#45877](https://github.com/vllm-project/vllm/pull/45877) |
| `vllm/renderers/deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/tokenizers/deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#40982](https://github.com/vllm-project/vllm/pull/40982), [#50580](https://github.com/vllm-project/vllm/pull/50580), [#51727](https://github.com/vllm-project/vllm/pull/51727) |
| `vllm/tokenizers/deepseek_v4_encoding.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#50580](https://github.com/vllm-project/vllm/pull/50580) |
| `vllm/transformers_utils/configs/deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |

## PR Coverage Summary

- Git-traced PRs: 95
- Extra PRs preserved from existing docs: 39
- Total PRs in this document: 134
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-26 | [#40806](https://github.com/vllm-project/vllm/pull/40806) | merged | [Bugfix] Fix the DSML token leakage in DSV4/3.2 | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-04-27 | [#40860](https://github.com/vllm-project/vllm/pull/40860) | merged | [Feat] DeepSeek V4 Rebased | `vllm/model_executor/models/deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`, `tests/tokenizers_/test_deepseek_v4.py` |
| 2026-04-27 | [#40760](https://github.com/vllm-project/vllm/pull/40760) | closed | [New Model] Support DeepseekV4 | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py` |
| 2026-04-27 | [#40950](https://github.com/vllm-project/vllm/pull/40950) | merged | [DSV4] Add silu clamp limit to shared expert | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/activation.py`, `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` |
| 2026-04-28 | [#41006](https://github.com/vllm-project/vllm/pull/41006) | merged | [Model][DSV4] Support base model | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-04-28 | [#41061](https://github.com/vllm-project/vllm/pull/41061) | merged | [DSV4] Enable Multi-stream for Pre-Attn GEMM | `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_compressor.py` |
| 2026-04-29 | [#41171](https://github.com/vllm-project/vllm/pull/41171) | merged | [DSV4] Align aux stream API with DeepseekV4DecoderLayer | `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-04-29 | [#41090](https://github.com/vllm-project/vllm/pull/41090) | merged | [Bugfix] Fix Deepseek V4 import error due to AOT compile cache loading | `vllm/model_executor/models/deepseek_v4.py` |
| 2026-04-29 | [#41135](https://github.com/vllm-project/vllm/pull/41135) | merged | [Bugfix] fix inductor error for dpsk v4 | `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` |
| 2026-04-29 | [#40982](https://github.com/vllm-project/vllm/pull/40982) | merged | [DSV4] Support `max` reasoning effort | `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py` |
| 2026-04-29 | [#41148](https://github.com/vllm-project/vllm/pull/41148) | merged | [Bugfix] Fix repeated DSv4 RoPE cache initialization | `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/models/deepseek_v4.py` |
| 2026-04-29 | [#41015](https://github.com/vllm-project/vllm/pull/41015) | merged | [DSv4] Use `cvt` PTX for FP32->FP4 conversion | `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` |
| 2026-04-30 | [#41374](https://github.com/vllm-project/vllm/pull/41374) | merged | [DSV4] Avoid redundant dtype conversion. | `vllm/model_executor/models/deepseek_v4.py` |
| 2026-05-01 | [#41255](https://github.com/vllm-project/vllm/pull/41255) | merged | [Perf] Intergrate Tile Kernels `head_compute_mix_kernel` for Deepseek-V4 | `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py` |
| 2026-05-01 | [#41443](https://github.com/vllm-project/vllm/pull/41443) | merged | [DSV4] Add knob to enable pre-attn gemm | `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/envs.py`, `vllm/utils/multi_stream_utils.py` |
| 2026-05-02 | [#41522](https://github.com/vllm-project/vllm/pull/41522) | merged | [DSV4] Guard megamoe flag with Pure TP | `vllm/model_executor/models/deepseek_v4.py` |
| 2026-05-05 | [#40871](https://github.com/vllm-project/vllm/pull/40871) | merged | [New Model][ROCm] Add AMD support for DeepSeek V4 | `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` |
| 2026-05-06 | [#41801](https://github.com/vllm-project/vllm/pull/41801) | merged | [Bugfix] DeepSeekV32/v4: respect string='true\|false' attribute andunwrap arguments/input wrapper | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py` |
| 2026-05-09 | [#41428](https://github.com/vllm-project/vllm/pull/41428) | merged | [DSv4] Improved fused Indexer Q quant kernel | `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `vllm/utils/import_utils.py` |
| 2026-05-09 | [#41957](https://github.com/vllm-project/vllm/pull/41957) | merged | [Bugfix][PD] Fix DSv4 Disaggregated | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py`, `tests/v1/kv_connector/unit/test_tp_mapping.py` |
| 2026-05-10 | [#42169](https://github.com/vllm-project/vllm/pull/42169) | merged | [Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len | `csrc/topk.cu` |
| 2026-05-10 | [#41694](https://github.com/vllm-project/vllm/pull/41694) | merged | [DSV4] Add PP support for deepseek-v4 | `vllm/model_executor/models/deepseek_v4.py`, `docs/models/supported_models.md` |
| 2026-05-11 | [#41536](https://github.com/vllm-project/vllm/pull/41536) | merged | add fused mhc_post_pre kernel | `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py`, `tests/kernels/test_mhc_kernels.py` |
| 2026-05-11 | [#40392](https://github.com/vllm-project/vllm/pull/40392) | merged | [Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA | `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` |
| 2026-05-11 | [#42236](https://github.com/vllm-project/vllm/pull/42236) | merged | [DSv4] Improved dequant gather K cache kernel | `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py`, `tests/kernels/test_compressor_kv_cache.py`, `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` |
| 2026-05-11 | [#41812](https://github.com/vllm-project/vllm/pull/41812) | merged | [ROCm][DSv4] implement flash sparse mla with triton kernels | `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` |
| 2026-05-13 | [#41946](https://github.com/vllm-project/vllm/pull/41946) | merged | [Bugfix] [ROCm] [DSV4] [Perf] Add aiter mhc support | `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/kernels/mhc/tilelang.py`, `vllm/model_executor/kernels/mhc/triton.py` |
| 2026-05-13 | [#42320](https://github.com/vllm-project/vllm/pull/42320) | merged | [Bugfix] Fix DeepSeek V4 MTP HC state handling | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-05-14 | [#41778](https://github.com/vllm-project/vllm/pull/41778) | merged | [MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell | `benchmarks/attention_benchmarks/configs/mla_prefill.yaml`, `benchmarks/attention_benchmarks/configs/mla_decode.yaml`, `vllm/model_executor/layers/attention/mla_attention.py` |
| 2026-05-14 | [#42342](https://github.com/vllm-project/vllm/pull/42342) | merged | [Bug] Fix DeepSeek V4 `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'LoadCacheMode'` | `requirements/cuda.txt` |
| 2026-05-14 | [#41263](https://github.com/vllm-project/vllm/pull/41263) | merged | [DSV4] Fuse norm and router for low latency scenario | `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-05-14 | [#42112](https://github.com/vllm-project/vllm/pull/42112) | merged | [Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup | `vllm/v1/attention/backends/mla/prefill/flashinfer.py`, `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` |
| 2026-05-15 | [#42604](https://github.com/vllm-project/vllm/pull/42604) | merged | DeepSeekV4-Pro enable cuda graph full and piecewise mode | `vllm/model_executor/layers/mhc.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` |
| 2026-05-17 | [#42810](https://github.com/vllm-project/vllm/pull/42810) | merged | [ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy | `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v4.py` |
| 2026-05-18 | [#41710](https://github.com/vllm-project/vllm/pull/41710) | merged | fix: remove unused norm for dpskv4 | `vllm/model_executor/layers/deepseek_v4_attention.py` |
| 2026-05-18 | [#42930](https://github.com/vllm-project/vllm/pull/42930) | merged | [Bugfix] Fix DSV4 MTP after ROCm mHC integration | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-05-18 | [#42541](https://github.com/vllm-project/vllm/pull/42541) | merged | [Bugfix] fix swiglu limit issue for humming backend + deepseek v4 | `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`, `vllm/model_executor/layers/quantization/utils/humming_utils.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` |
| 2026-05-19 | [#43004](https://github.com/vllm-project/vllm/pull/43004) | merged | [Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N] | `vllm/models/deepseek_v4/quant_config.py`, `vllm/models/deepseek_v4/__init__.py`, `tests/models/test_deepseek_v4_mega_moe.py` |
| 2026-05-19 | [#43039](https://github.com/vllm-project/vllm/pull/43039) | merged | [Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N] | `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/compressor.py` |
| 2026-05-19 | [#42899](https://github.com/vllm-project/vllm/pull/42899) | merged | add cutedsl dsv4 indexer fp8 kernel | `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py` |
| 2026-05-19 | [#43073](https://github.com/vllm-project/vllm/pull/43073) | merged | [Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N] | `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/ops/__init__.py`, `vllm/models/deepseek_v4/attention.py` |
| 2026-05-19 | [#42828](https://github.com/vllm-project/vllm/pull/42828) | merged | [KVConnector][DSV4] HMA support for Mooncake store connector | `tests/v1/kv_connector/unit/test_mooncake_store_worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`, `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py` |
| 2026-05-19 | [#43077](https://github.com/vllm-project/vllm/pull/43077) | merged | [Model Refactoring] Rename deepseek_v4.py to model.py [4/N] | `vllm/models/deepseek_v4/__init__.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-05-20 | [#42111](https://github.com/vllm-project/vllm/pull/42111) | merged | [CI] Add DSV4-Flash to gsm8k moe-refactor/config-b200.txt | `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml` |
| 2026-05-22 | [#42209](https://github.com/vllm-project/vllm/pull/42209) | merged | Add NVFP4 MOE support for Deepseek V4. | `vllm/models/deepseek_v4/quant_config.py` |
| 2026-05-22 | [#43149](https://github.com/vllm-project/vllm/pull/43149) | merged | [Refactor] Extract DeepSeek V4 sparse MLA impl into model folder | `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/ops/attention.py`, `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-05-22 | [#42353](https://github.com/vllm-project/vllm/pull/42353) | merged | DSv4 fused Q-norm kernel grid refactor | `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` |
| 2026-05-22 | [#42950](https://github.com/vllm-project/vllm/pull/42950) | merged | [XPU]fix: add XPU platform guards to DeepSeek-V4 ops | `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/compressor.py` |
| 2026-05-23 | [#42925](https://github.com/vllm-project/vllm/pull/42925) | merged | [DSV4] More multi-stream enablement for c4a | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-05-24 | [#43385](https://github.com/vllm-project/vllm/pull/43385) | merged | [ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-05-26 | [#43632](https://github.com/vllm-project/vllm/pull/43632) | merged | [DeepSeek V4] Move MegaMoE input prep kernel to nvidia/ops | `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `tests/models/test_deepseek_v4_mega_moe.py` |
| 2026-05-26 | [#43162](https://github.com/vllm-project/vllm/pull/43162) | merged | [Feat][DSV4] Fuse q pad into deepseek v4 fused kernel | `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-05-26 | [#43629](https://github.com/vllm-project/vllm/pull/43629) | merged | [ROCm] Remove MegaMoE integration in deepseek v4 | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py` |
| 2026-05-26 | [#43690](https://github.com/vllm-project/vllm/pull/43690) | merged | [DSv4] Drop _get_compressed_kv_buffer in DeepseekCompressor | `vllm/models/deepseek_v4/compressor.py` |
| 2026-05-27 | [#43710](https://github.com/vllm-project/vllm/pull/43710) | merged | [DSv4] Refactor compressor & Fix ROCm compatibility | `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/common/ops/save_partial_states.py` |
| 2026-05-28 | [#43679](https://github.com/vllm-project/vllm/pull/43679) | merged | [ROCm][DSV4] Enable Tilelang MHC replacing torch/triton mhc | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py` |
| 2026-05-28 | [#43829](https://github.com/vllm-project/vllm/pull/43829) | merged | [DSV4] Remove AMD/XPU path in deepseek_v4/nvidia | `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-05-28 | [#43746](https://github.com/vllm-project/vllm/pull/43746) | merged | [Model Refactoring] Remove torch compile dependency in DSv4 | `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-05-28 | [#43891](https://github.com/vllm-project/vllm/pull/43891) | merged | [Model Refactoring] Remove unncessary torch op registration for DSv4 | `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-05-29 | [#43905](https://github.com/vllm-project/vllm/pull/43905) | merged | [DSv4] Move mHC tilelang kernels & Don't use CustomOP in dsv4/nvidia | `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-06-01 | [#44161](https://github.com/vllm-project/vllm/pull/44161) | merged | [Kernel][DSv4] Optimize sparse FP8 compressor kernels | `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` |
| 2026-06-01 | [#44246](https://github.com/vllm-project/vllm/pull/44246) | merged | [DSV4] Remove unncessary classes & functions | `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-02 | [#44262](https://github.com/vllm-project/vllm/pull/44262) | merged | [DSV4] Refactor RoPE initialization | `vllm/models/deepseek_v4/common/rope.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-02 | [#43339](https://github.com/vllm-project/vllm/pull/43339) | merged | [Feature] Support EPLB for DeepSeek v4 Mega Moe | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-03 | [#44367](https://github.com/vllm-project/vllm/pull/44367) | merged | [DSV4] Minor cleanup for DeepseekV4MegaMoEExperts | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-03 | [#44356](https://github.com/vllm-project/vllm/pull/44356) | merged | [Bugfix] Fix Deepseek v4 non-mega-moe model init error | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-03 | [#44236](https://github.com/vllm-project/vllm/pull/44236) | merged | fix: resolve CUTLASS fmin compatibility for DeepSeek-V4 init | `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` |
| 2026-06-04 | [#43827](https://github.com/vllm-project/vllm/pull/43827) | merged | [DSv4] Adding TRTLLM gen attention kernel | `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py` |
| 2026-06-05 | [#44569](https://github.com/vllm-project/vllm/pull/44569) | merged | [DSV4] Refactor DeepseekV4Attention | `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-05 | [#44561](https://github.com/vllm-project/vllm/pull/44561) | merged | [DSV4] Move more ops out of eager breakpoint | `vllm/models/deepseek_v4/attention.py` |
| 2026-06-07 | [#44699](https://github.com/vllm-project/vllm/pull/44699) | merged | [DSV4] Decouple DS V4 Sparse MLA Metadata from DS V3.2 | `vllm/models/deepseek_v4/sparse_mla.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` |
| 2026-06-08 | [#42953](https://github.com/vllm-project/vllm/pull/42953) | merged | feat: add DeepSeek-V4 XPU attention decode path | `vllm/models/deepseek_v4/xpu/model.py`, `vllm/models/deepseek_v4/xpu/mtp.py`, `vllm/models/deepseek_v4/xpu/xpu_sparse.py` |
| 2026-06-09 | [#44144](https://github.com/vllm-project/vllm/pull/44144) | merged | [DSV4][XPU] Add MHC fused_post_pre support | `vllm/models/deepseek_v4/xpu/model.py` |
| 2026-06-09 | [#44914](https://github.com/vllm-project/vllm/pull/44914) | merged | [Bug] Fix deepseek v4 OOM issue | `vllm/models/deepseek_v4/quant_config.py` |
| 2026-06-10 | [#44821](https://github.com/vllm-project/vllm/pull/44821) | merged | fix: prefix DeepSeek V4 MTP projections | `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-06-12 | [#45240](https://github.com/vllm-project/vllm/pull/45240) | merged | [XPU][DeepSeek-V4] Fix MTP: sync with upstream fixes #44821 and #43746 | `vllm/models/deepseek_v4/xpu/mtp.py` |
| 2026-06-15 | [#45061](https://github.com/vllm-project/vllm/pull/45061) | merged | [Perf] Optimize DSv4 prefill chunk planning, 4.0% E2E Throughput Improvement | `vllm/models/deepseek_v4/nvidia/flashmla.py` |
| 2026-06-16 | [#44892](https://github.com/vllm-project/vllm/pull/44892) | merged | [DSV4][Minor] Fix supported KV cache dtypes | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/sparse_mla.py` |
| 2026-06-17 | [#45863](https://github.com/vllm-project/vllm/pull/45863) | merged | [DSv4 Perf] DSv4 flashinfer sparse index cache for metadata, 2%~4% TTFT improvement | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` |
| 2026-06-17 | [#45309](https://github.com/vllm-project/vllm/pull/45309) | merged | [DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`, 26.8% ~ 27.9% E2E TTFT improvement | `vllm/models/deepseek_v4/attention.py` |
| 2026-06-18 | [#45972](https://github.com/vllm-project/vllm/pull/45972) | merged | Revert "[DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`" (#45309) | `vllm/models/deepseek_v4/attention.py` |
| 2026-06-18 | [#45681](https://github.com/vllm-project/vllm/pull/45681) | merged | [ROCm][DSv4] Functional fixes for DeepSeek V4 on MI300X/MI325X | `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` |
| 2026-06-19 | [#46001](https://github.com/vllm-project/vllm/pull/46001) | merged | [DeepSeek-V4] Support TEP=16 for the block-FP8 shared expert | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-22 | [#45931](https://github.com/vllm-project/vllm/pull/45931) | merged | [ROCm][DSV4] Disable TileLang MHC dispatch on gfx942 | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py` |
| 2026-06-22 | [#43477](https://github.com/vllm-project/vllm/pull/43477) | merged | Enable DeepSeek V4 and GLM-5.1 on SM120 | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py`, `vllm/models/deepseek_v4/attention.py` |
| 2026-06-23 | [#46428](https://github.com/vllm-project/vllm/pull/46428) | merged | [Optimization] Skip DP padding tokens in MoE | `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/model_executor/layers/fused_moe/modular_kernel.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` |
| 2026-06-25 | [#40811](https://github.com/vllm-project/vllm/pull/40811) | closed | [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4 | `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `csrc/persistent_topk.cuh` |
| 2026-07-01 | [#43950](https://github.com/vllm-project/vllm/pull/43950) | merged | [ROCm][DSV4] Use aiter mHC pre/post as the default ROCm path | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py` |
| 2026-07-01 | [#46730](https://github.com/vllm-project/vllm/pull/46730) | merged | [ROCm][Perf][Bugfix] DSv4 indexer: use platform FP8 dtype (fnuz) for Q-quant on gfx942 | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` |
| 2026-07-04 | [#45877](https://github.com/vllm-project/vllm/pull/45877) | merged | [Frontend] [Parser] Port DeepSeek V4 to streaming parser engine framework | `vllm/reasoning/deepseek_v4_engine_reasoning_parser.py`, `tests/parser/engine/test_deepseek_v4.py`, `vllm/parser/deepseek_v4.py` |
| 2026-07-06 | [#47429](https://github.com/vllm-project/vllm/pull/47429) | merged | [Bugfix][Spec Decode] Add missing draft_id_to_target_id to DSparkDeepseekV4ForCausalLM | `vllm/models/deepseek_v4/nvidia/dspark.py` |
| 2026-07-06 | [#47474](https://github.com/vllm-project/vllm/pull/47474) | merged | [Perf] Cache `token_to_req_indices` for dsv4, 5x~6x kernel performance improvement | `vllm/models/deepseek_v4/sparse_mla.py`, `vllm/models/deepseek_v4/compressor.py` |
| 2026-07-06 | [#47716](https://github.com/vllm-project/vllm/pull/47716) | merged | [Bugfix]Fix DeepSeek-V4 fp8_ds_mla KV cache reshape | `vllm/models/deepseek_v4/attention.py` |
| 2026-07-08 | [#47493](https://github.com/vllm-project/vllm/pull/47493) | merged | [Bugfix] DSV4 TP16 garbage output | `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/attention.py` |
| 2026-07-10 | [#47419](https://github.com/vllm-project/vllm/pull/47419) | merged | [ROCm] Enable DeepSeek-V4 DSpark speculative decoding on AMD (MI350X / MI355X, gfx950) | `vllm/models/deepseek_v4/amd/dspark.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-07-15 | [#48137](https://github.com/vllm-project/vllm/pull/48137) | merged | [Perf] Remove redundant repeat and copy for dsv4, 1.8% E2E TPOT improvement. | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-07-15 | [#47718](https://github.com/vllm-project/vllm/pull/47718) | merged | [ROCm][Perf] DSv4 two-stage compressor kernel for HCA prefill | `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/compressor.py` |
| 2026-07-16 | [#47677](https://github.com/vllm-project/vllm/pull/47677) | merged | [XPU] Add DSpark speculative decoding support for DeepSeek-V4 | `vllm/models/deepseek_v4/xpu/dspark.py`, `vllm/models/deepseek_v4/xpu/model.py`, `vllm/models/deepseek_v4/__init__.py` |
| 2026-07-21 | [#45991](https://github.com/vllm-project/vllm/pull/45991) | merged | [XPU][DeepSeekV4]Add DeepSeek-V4 fuse_index_q SYCL kernel path | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` |
| 2026-07-22 | [#48957](https://github.com/vllm-project/vllm/pull/48957) | merged | [DSv4 Perf] Skip empty c128 kernel launch, around 2x kernel performance improvement. | `vllm/models/deepseek_v4/compressor.py` |
| 2026-07-22 | [#48993](https://github.com/vllm-project/vllm/pull/48993) | merged | [Core][DSV4] Compact MXFP4 indexer KV cache and packed group overlays | `vllm/models/deepseek_v4/attention.py` |
| 2026-07-23 | [#48044](https://github.com/vllm-project/vllm/pull/48044) | merged | [ROCm] Fused Shared Expert Support for AMD Quark DeepSeek-V4 Model Checkpoints | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/quant_config.py`, `vllm/models/deepseek_v4/amd/mtp.py` |
| 2026-07-23 | [#49415](https://github.com/vllm-project/vllm/pull/49415) | merged | [Bugfix] Fix DeepSeek-V4 DSpark draft shared-expert padding for TP > 8 | `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-07-23 | [#49486](https://github.com/vllm-project/vllm/pull/49486) | merged | [DSv4 Perf] Skip topk and router when not needed, 3.4% E2E TTFT improvement for Decode case | `vllm/models/deepseek_v4/attention.py` |
| 2026-07-27 | [#50004](https://github.com/vllm-project/vllm/pull/50004) | merged | [DSv4 Perf] Adaptive topk width, 1.0% E2E throughput improvement | `vllm/models/deepseek_v4/sparse_mla.py` |
| 2026-07-28 | [#49634](https://github.com/vllm-project/vllm/pull/49634) | merged | [Bugfix] Fix DeepseekV4FP8 Quark MXFP4 crash on list-valued weight | `vllm/models/deepseek_v4/quant_config.py` |
| 2026-07-30 | [#50312](https://github.com/vllm-project/vllm/pull/50312) | merged | [DSv4 Perf] Fix redundant memory allocation and copy for dsv4 pp buffer, 448 MiB GPU memory saved | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/xpu/model.py` |
| 2026-07-30 | [#50298](https://github.com/vllm-project/vllm/pull/50298) | merged | [DSv4 Perf] Remove redundant full kernel for dsv4, 1.88x kernel performance improvement | `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py` |
| 2026-07-30 | [#46720](https://github.com/vllm-project/vllm/pull/46720) | merged | [ROCm][DSV4] B-preshuffle the attention fp8 projections | `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/attention.py` |
| 2026-07-31 | [#48047](https://github.com/vllm-project/vllm/pull/48047) | merged | [DSv4] Remove sparse-MLA q-head padding for FlashInfer >=0.6.14 | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` |
| 2026-07-31 | [#49236](https://github.com/vllm-project/vllm/pull/49236) | merged | [DSv4 Perf] Optimize workspace reuse for eager break | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-08-01 | [#46789](https://github.com/vllm-project/vllm/pull/46789) | merged | [DSV4] Implement Sequence Parallelism | `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-08-04 | [#50580](https://github.com/vllm-project/vllm/pull/50580) | merged | [Frontend] DeepSeek V4 0731 reasoning effort prompts & mappings | `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`, `vllm/tokenizers/deepseek_v4.py` |
| 2026-08-07 | [#47972](https://github.com/vllm-project/vllm/pull/47972) | merged | Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel | `vllm/models/deepseek_v4/amd/model.py`, `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml` |
| 2026-08-10 | [#51430](https://github.com/vllm-project/vllm/pull/51430) | merged | [Perf] Narrow DeepSeek V4 eager CUDA graph region | `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-08-10 | [#51296](https://github.com/vllm-project/vllm/pull/51296) | merged | [Bugfix] Align deepseek v4 parser thinking default with tokenizer | `tests/parser/engine/test_deepseek_v4.py`, `vllm/parser/deepseek_v4.py` |
| 2026-08-10 | [#51727](https://github.com/vllm-project/vllm/pull/51727) | merged | [Bugfix] Fix DeepSeek V4/3.2 tokenizer vocab size overcount crashing guided decoding | `vllm/tokenizers/deepseek_v4.py` |
| 2026-08-11 | [#51145](https://github.com/vllm-project/vllm/pull/51145) | merged | [Bugfix][ROCm] Fix DeepSeek V4 DSpark probabilistic startup | `vllm/models/deepseek_v4/amd/dspark.py` |
| 2026-08-13 | [#51821](https://github.com/vllm-project/vllm/pull/51821) | merged | [Bugfix][ROCm][CI] Restore the DeepSeek-V4 input GEMM override point | `vllm/models/deepseek_v4/attention.py` |
| 2026-08-15 | [#51538](https://github.com/vllm-project/vllm/pull/51538) | merged | [Bugfix] Make DSV4 sparse MLA work end-to-end for plain decode, MTP, and DSpark | `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-08-16 | [#51318](https://github.com/vllm-project/vllm/pull/51318) | merged | [Bugfix][DSv4] Revert adaptive C128A metadata packing | `vllm/models/deepseek_v4/sparse_mla.py` |
| 2026-08-16 | [#52401](https://github.com/vllm-project/vllm/pull/52401) | merged | [Bugfix] Pick the DeepSeek V4 eager cudagraph region per model runner | `vllm/models/deepseek_v4/attention.py` |
| 2026-08-16 | [#52084](https://github.com/vllm-project/vllm/pull/52084) | merged | [Perf][DSV4] Optimize sparse top-k metadata kernels for higher prefill throughput | `vllm/models/deepseek_v4/common/ops/cache_utils.py` |
| 2026-08-16 | [#51967](https://github.com/vllm-project/vllm/pull/51967) | merged | [Perf][DSV4] Optimize global top-k index kernel with compile-time constants | `vllm/models/deepseek_v4/common/ops/cache_utils.py` |
| 2026-08-16 | [#52212](https://github.com/vllm-project/vllm/pull/52212) | merged | [ROCm][DSV4][Perf] Optimize Triton sparse-MLA decode on gfx950 | `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` |
| 2026-08-17 | [#52492](https://github.com/vllm-project/vllm/pull/52492) | merged | [Bugfix][DSv4] Keep indexer scoring in breakable graphs | `vllm/models/deepseek_v4/attention.py` |
| 2026-08-18 | [#52626](https://github.com/vllm-project/vllm/pull/52626) | merged | [Bugfix] Fix DeepSeek V4 mHC broadcast buffer for weight sync | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-08-19 | [#51368](https://github.com/vllm-project/vllm/pull/51368) | merged | [Bugfix] Fix DeepSeek V4 mHC broadcast buffer for dummy load | `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/dspark.py` |
| 2026-08-19 | [#52836](https://github.com/vllm-project/vllm/pull/52836) | merged | Revert DSv4 eager workspace reuse | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-08-20 | [#52737](https://github.com/vllm-project/vllm/pull/52737) | merged | [ROCm][Perf] Fuse DeepSeek-V4 mHC post/pre and RMSNorm with AITER | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/dspark.py` |
| 2026-08-20 | [#50803](https://github.com/vllm-project/vllm/pull/50803) | merged | [ROCm] Fix DeepSeek V4 indexer numerics and coverage | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` |
| 2026-08-20 | [#53040](https://github.com/vllm-project/vllm/pull/53040) | merged | [DSV4][Kernel] Fuse shared experts into MegaMoE | `vllm/models/deepseek_v4/nvidia/model.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` |
| 2026-08-21 | [#52882](https://github.com/vllm-project/vllm/pull/52882) | merged | [ROCm][Perf] Optimize DeepSeek V4 C4A top-k with AITER | `vllm/models/deepseek_v4/attention.py` |
| 2026-08-21 | [#52823](https://github.com/vllm-project/vllm/pull/52823) | merged | [DSv4 Perf] Adaptive topk width for dsv4, making #50004 back | `vllm/models/deepseek_v4/sparse_mla.py` |

## Per-PR Diff Audit Cards

### PR #40806 - [Bugfix] Fix the DSML token leakage in DSV4/3.2

- Link: https://github.com/vllm-project/vllm/pull/40806
- Status/date: merged / 2026-04-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +76/-23, 144 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix the DSML token leakage in DSV4/3.2"; model line: DeepSeek V4; category: bug fix; main diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; technical summary: Covers "[Bugfix] Fix the DSML token leakage in DSV4/3.2"; the main implementation surface is `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0 (52 lines); hunks: -484,6 +484,58 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked, test_no_marker_leak_char_by_char, touching `test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked`; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23 (47 lines); hunks: -26,6 +26,7; -54,8 +55,8 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, extract_tool_calls, _reset_streaming_state, _extract_delta_tool_calls, touching `__init__, extract_tool_calls, _reset_streaming_state`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0 (52 lines); hunks: -484,6 +484,58 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked, test_no_marker_leak_char_by_char
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23 (47 lines); hunks: -26,6 +26,7; -54,8 +55,8 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, extract_tool_calls, _reset_streaming_state, _extract_delta_tool_calls
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -484,6 +484,58 @@ def test_no_emission_while_incomplete(self, parser):
+    def test_no_marker_leak_chunked(self, parser):
+        """Chunked streaming must NOT leak DSML start-marker fragments
+        as content (GitHub #40801)."""
+        full_text = build_tool_call("fn", {"k": "v"})
+        deltas = self._stream_chunked(parser, full_text, chunk_size=5)
+        content = "".join(d.content for d in deltas if d.content is not None)
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -26,6 +26,7 @@
+from vllm.tool_parsers.utils import partial_tag_overlap
@@ -54,8 +55,8 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
-        self.is_tool_call_started: bool = False
+        self._sent_content_idx: int = 0
@@ -219,7 +220,7 @@ def extract_tool_calls(
-        self.is_tool_call_started = False
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40860 - [Feat] DeepSeek V4 Rebased

- Link: https://github.com/vllm-project/vllm/pull/40860
- Status/date: merged / 2026-04-27
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `tests/tokenizers_/fixtures/deepseek_v4/test_input_1.json`, `tests/tokenizers_/fixtures/deepseek_v4/test_input_2.json`, `tests/tokenizers_/fixtures/deepseek_v4/test_input_3.json` and 16 files; associated commits `4d51588e2381`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 150 files, +16313/-717, 20516 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feat] DeepSeek V4 Rebased"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`, `tests/tokenizers_/test_deepseek_v4.py`; technical summary: Covers "[Feat] DeepSeek V4 Rebased"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`, `tests/tokenizers_/test_deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method, touching `DeepseekV4FP8Config, __init__, get_name`; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`; `tests/tokenizers_/test_deepseek_v4.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: FakeHfTokenizer, get_added_vocab, encode, _tokenizer, touching `FakeHfTokenizer, get_added_vocab, encode`; `tests/models/test_deepseek_v4_mega_moe.py` added +184/-0 (184 lines); hunks: -0,0 +1,184; symbols: test_deepseek_v4_mega_moe_expert_mapping, test_deepseek_v4_mega_moe_ue8m0_uint8_to_float, test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, touching `test_deepseek_v4_mega_moe_expert_mapping, test_deepseek_v4_mega_moe_ue8m0_uint8_to_float, test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method
  - `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `tests/tokenizers_/test_deepseek_v4.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: FakeHfTokenizer, get_added_vocab, encode, _tokenizer
  - `tests/models/test_deepseek_v4_mega_moe.py` added +184/-0 (184 lines); hunks: -0,0 +1,184; symbols: test_deepseek_v4_mega_moe_expert_mapping, test_deepseek_v4_mega_moe_ue8m0_uint8_to_float, test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact
  - `tests/tokenizers_/fixtures/deepseek_v4/test_input_3.json` added +159/-0 (159 lines); hunks: -0,0 +1,159
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -0,0 +1,1437 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/tokenizers/deepseek_v4_encoding.py
@@ -0,0 +1,757 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa
+# fmt: off
+"""
+DeepSeek-V4 Encoding
diff -- tests/tokenizers_/test_deepseek_v4.py
@@ -0,0 +1,224 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0; `vllm/tokenizers/deepseek_v4.py` added +90/-0
  - tests: `tests/tokenizers_/test_deepseek_v4.py` added +224/-0; `tests/models/test_deepseek_v4_mega_moe.py` added +184/-0; `tests/tokenizers_/fixtures/deepseek_v4/test_input_3.json` added +159/-0; `tests/tokenizers_/fixtures/deepseek_v4/test_input_1.json` added +81/-0; `tests/tokenizers_/fixtures/deepseek_v4/test_output_3.txt` added +38/-0
- Risk and verification: The diff ships test coverage in `tests/compile/fusions_e2e/conftest.py`, `tests/kernels/attention/test_deepgemm_attention.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/moe/test_deepgemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40760 - [New Model] Support DeepseekV4

- Link: https://github.com/vllm-project/vllm/pull/40760
- Status/date: closed / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 158 files, +16968/-760, 21398 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[New Model] Support DeepseekV4"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py`; technical summary: Covers "[New Model] Support DeepseekV4"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method, touching `DeepseekV4FP8Config, __init__, get_name`; `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0 (1062 lines); hunks: -0,0 +1,1062; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does, touching `DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes`; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`; `vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor, touching `DeepSeekV4MultiTokenPredictorLayer, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method
  - `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0 (1062 lines); hunks: -0,0 +1,1062; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does
  - `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor
  - `vllm/model_executor/layers/deepseek_compressor.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: CompressorBackend, __init__, get_name, get_supported_kernel_block_sizes
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -0,0 +1,1437 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -0,0 +1,1062 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+DeepseekV4 MLA Attention Layer
+"""
+from dataclasses import dataclass
diff -- vllm/tokenizers/deepseek_v4_encoding.py
@@ -0,0 +1,757 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0; `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0; `vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0; `vllm/model_executor/layers/deepseek_compressor.py` added +436/-0; `vllm/model_executor/layers/mhc.py` added +436/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_use_trtllm_attention.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/moe/test_deepgemm.py`, `tests/kernels/moe/test_ocp_mx_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40950 - [DSV4] Add silu clamp limit to shared expert

- Link: https://github.com/vllm-project/vllm/pull/40950
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +269/-29, 466 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Add silu clamp limit to shared expert"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/activation.py`, `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`; technical summary: Covers "[DSV4] Add silu clamp limit to shared expert"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/activation.py`, `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +58/-3 (61 lines); hunks: -17,6 +17,7; -34,7 +35,10; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4FP8Config, touching `DeepseekV4MLP, __init__, forward`; `vllm/model_executor/layers/activation.py` modified +40/-0 (40 lines); hunks: -151,6 +151,46 @@ def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward_xpu, SiluAndMulWithClamp, __init__, forward_native, touching `forward_xpu, SiluAndMulWithClamp, __init__`; `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` modified +1/-1 (2 lines); hunks: -45,7 +45,7 @@ def _gelu_and_mul(; symbols: _gelu_and_mul, touching `_gelu_and_mul`; `csrc/activation_kernels.cu` modified +82/-25 (107 lines); hunks: -11,29 +11,74; -58,8 +103,9 @@ __global__ void act_and_mul_kernel(.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +58/-3 (61 lines); hunks: -17,6 +17,7; -34,7 +35,10; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4FP8Config
  - `vllm/model_executor/layers/activation.py` modified +40/-0 (40 lines); hunks: -151,6 +151,46 @@ def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward_xpu, SiluAndMulWithClamp, __init__, forward_native
  - `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` modified +1/-1 (2 lines); hunks: -45,7 +45,7 @@ def _gelu_and_mul(; symbols: _gelu_and_mul
  - `csrc/activation_kernels.cu` modified +82/-25 (107 lines); hunks: -11,29 +11,74; -58,8 +103,9 @@ __global__ void act_and_mul_kernel(
  - `tests/kernels/core/test_activation.py` modified +80/-0 (80 lines); hunks: -16,6 +16,7; -116,6 +117,85 @@ def _get_rtol(output) -> float:; symbols: _get_rtol, test_silu_and_mul_with_clamp
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -17,6 +17,7 @@
+from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
@@ -34,7 +35,10 @@
-from vllm.model_executor.layers.quantization import QuantizationMethods
+from vllm.model_executor.layers.quantization import (
+    QuantizationConfig,
+    QuantizationMethods,
diff -- vllm/model_executor/layers/activation.py
@@ -151,6 +151,46 @@ def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
+@CustomOp.register("silu_and_mul_with_clamp")
+class SiluAndMulWithClamp(CustomOp):
+    """SwiGLU activation with input clamping (used by some MoE shared experts).
+    Computes:
+        gate = clamp(x[..., :d], max=swiglu_limit)
+        up   = clamp(x[..., d:], min=-swiglu_limit, max=swiglu_limit)
diff -- vllm/model_executor/layers/fused_moe/cpu_fused_moe.py
@@ -45,7 +45,7 @@ def _gelu_and_mul(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +58/-3; `vllm/model_executor/layers/activation.py` modified +40/-0; `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` modified +1/-1
  - other: `csrc/activation_kernels.cu` modified +82/-25; `csrc/torch_bindings.cpp` modified +6/-0; `csrc/ops.h` modified +2/-0
  - tests: `tests/kernels/core/test_activation.py` modified +80/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_activation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41006 - [Model][DSV4] Support base model

- Link: https://github.com/vllm-project/vllm/pull/41006
- Status/date: merged / 2026-04-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +111/-23, 223 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][DSV4] Support base model"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; technical summary: Covers "[Model][DSV4] Support base model"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +93/-19 (112 lines); hunks: -10,7 +10,7; -65,6 +65,8; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4FP8Config, touching `DeepseekV4MLP, __init__, forward`; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +18/-4 (22 lines); hunks: -48,9 +48,14; -326,6 +331,15 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx, touching `_find_mtp_layer_idx`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +93/-19 (112 lines); hunks: -10,7 +10,7; -65,6 +65,8; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4FP8Config
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +18/-4 (22 lines); hunks: -48,9 +48,14; -326,6 +331,15 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -10,7 +10,7 @@
-from vllm.config import VllmConfig
+from vllm.config import VllmConfig, get_current_vllm_config
@@ -65,6 +65,8 @@
+_DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")
@@ -118,16 +120,59 @@ def forward(self, x):
-    """FP8 config that routes MoE layers to MXFP4 quantization.
diff -- vllm/model_executor/models/deepseek_v4_mtp.py
@@ -48,9 +48,14 @@
-# MoE expert scales are fused into per-layer w13/w2 tensors; other FP8 linear
-# scales use `.weight_scale_inv`. Mirrors the regex in
-# DeepseekV4ForCausalLM.hf_to_vllm_mapper.
+# MoE expert scales are fused into per-layer w13/w2 tensors. The exact
+# parameter suffix depends on which FusedMoE method handles the experts:
+# - fp4 experts (Mxfp4MoEMethod) register ``w{1,2,3}_weight_scale``;
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +93/-19; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +18/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41061 - [DSV4] Enable Multi-stream for Pre-Attn GEMM

- Link: https://github.com/vllm-project/vllm/pull/41061
- Status/date: merged / 2026-04-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +187/-57, 439 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Enable Multi-stream for Pre-Attn GEMM"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_compressor.py`; technical summary: Covers "[DSV4] Enable Multi-stream for Pre-Attn GEMM"; the main implementation surface is `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_compressor.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +111/-38 (149 lines); hunks: -4,8 +4,9; -16,6 +17,7; symbols: DeepseekV4MLAModules, __init__, forward, touching `DeepseekV4MLAModules, __init__, forward`; `vllm/model_executor/models/deepseek_v4.py` modified +10/-12 (22 lines); hunks: -54,7 +54,6; -872,7 +871,7 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/model_executor/layers/deepseek_compressor.py` modified +2/-7 (9 lines); hunks: -14,7 +14,6; -271,16 +270,12 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/utils/multi_stream_utils.py` modified +64/-0 (64 lines); hunks: -56,3 +56,67 @@ def maybe_execute_in_parallel(; symbols: maybe_execute_in_parallel, execute_in_parallel, touching `maybe_execute_in_parallel, execute_in_parallel`.
- Code diff details:
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +111/-38 (149 lines); hunks: -4,8 +4,9; -16,6 +17,7; symbols: DeepseekV4MLAModules, __init__, forward
  - `vllm/model_executor/models/deepseek_v4.py` modified +10/-12 (22 lines); hunks: -54,7 +54,6; -872,7 +871,7 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/layers/deepseek_compressor.py` modified +2/-7 (9 lines); hunks: -14,7 +14,6; -271,16 +270,12 @@ def __init__(; symbols: __init__, forward
  - `vllm/utils/multi_stream_utils.py` modified +64/-0 (64 lines); hunks: -56,3 +56,67 @@ def maybe_execute_in_parallel(; symbols: maybe_execute_in_parallel, execute_in_parallel
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -4,8 +4,9 @@
+from collections.abc import Callable
-from typing import TYPE_CHECKING, cast
+from typing import TYPE_CHECKING, Any, cast
@@ -16,6 +17,7 @@
+from vllm.model_executor.layers.utils import cublas_gemm_bf16_bf16_fp32
@@ -51,7 +53,10 @@
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -54,7 +54,6 @@
-from vllm.utils.multi_stream_utils import AuxStreamType
@@ -872,7 +871,7 @@ def __init__(
-        aux_stream: torch.cuda.Stream | None = None,
+        aux_stream_list: list[torch.cuda.Stream] | None = None,
@@ -1005,7 +1004,7 @@ def __init__(
-            aux_stream=aux_stream,
diff -- vllm/model_executor/layers/deepseek_compressor.py
@@ -14,7 +14,6 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +111/-38; `vllm/model_executor/models/deepseek_v4.py` modified +10/-12; `vllm/model_executor/layers/deepseek_compressor.py` modified +2/-7; `vllm/utils/multi_stream_utils.py` modified +64/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/deepseek_compressor.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41171 - [DSV4] Align aux stream API with DeepseekV4DecoderLayer

- Link: https://github.com/vllm-project/vllm/pull/41171
- Status/date: merged / 2026-04-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-5, 51 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Align aux stream API with DeepseekV4DecoderLayer"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_v4_mtp.py`; technical summary: Covers "[DSV4] Align aux stream API with DeepseekV4DecoderLayer"; the main implementation surface is `vllm/model_executor/models/deepseek_v4_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4_mtp.py` modified +7/-5 (12 lines); hunks: -35,7 +35,6; -65,6 +64,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +7/-5 (12 lines); hunks: -35,7 +35,6; -65,6 +64,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4_mtp.py
@@ -35,7 +35,6 @@
-from vllm.utils.multi_stream_utils import AuxStreamType
@@ -65,6 +64,7 @@ def __init__(
+        aux_stream_list: list[torch.cuda.Stream] | None = None,
@@ -112,14 +112,11 @@ def __init__(
-        self.aux_stream_dict = {
-            AuxStreamType.Attention: torch.cuda.Stream(),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4_mtp.py` modified +7/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41090 - [Bugfix] Fix Deepseek V4 import error due to AOT compile cache loading

- Link: https://github.com/vllm-project/vllm/pull/41090
- Status/date: merged / 2026-04-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-5, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Deepseek V4 import error due to AOT compile cache loading"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v4.py`; technical summary: Covers "[Bugfix] Fix Deepseek V4 import error due to AOT compile cache loading"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +5/-5 (10 lines); hunks: -1098,6 +1098,11 @@ def __init__(; -1170,11 +1175,6 @@ def hc_pre(; symbols: __init__, hc_pre, touching `__init__, hc_pre`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +5/-5 (10 lines); hunks: -1098,6 +1098,11 @@ def __init__(; -1170,11 +1175,6 @@ def hc_pre(; symbols: __init__, hc_pre
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -1098,6 +1098,11 @@ def __init__(
+        # Lazy import to avoid top-level tilelang dependency.
+        # Registers both torch.ops.vllm.mhc_pre and mhc_post
+        import vllm.model_executor.layers.mhc  # noqa: F401
@@ -1170,11 +1175,6 @@ def hc_pre(
-        # Lazy import to avoid top-level tilelang dependency.
-        # Registers both torch.ops.vllm.mhc_pre and mhc_post,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +5/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41135 - [Bugfix] fix inductor error for dpsk v4

- Link: https://github.com/vllm-project/vllm/pull/41135
- Status/date: merged / 2026-04-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +106/-36, 172 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix inductor error for dpsk v4"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`; technical summary: Covers "[Bugfix] fix inductor error for dpsk v4"; the main implementation surface is `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` modified +106/-36 (142 lines); hunks: -10,6 +10,7; -180,34 +181,74 @@ def fused_inv_rope_fp8_quant(; symbols: fused_inv_rope_fp8_quant, _fused_inv_rope_fp8_quant_kernel_impl, _fused_inv_rope_fp8_quant_kernel_fake, touching `fused_inv_rope_fp8_quant, _fused_inv_rope_fp8_quant_kernel_impl, _fused_inv_rope_fp8_quant_kernel_fake`.
- Code diff details:
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` modified +106/-36 (142 lines); hunks: -10,6 +10,7; -180,34 +181,74 @@ def fused_inv_rope_fp8_quant(; symbols: fused_inv_rope_fp8_quant, _fused_inv_rope_fp8_quant_kernel_impl, _fused_inv_rope_fp8_quant_kernel_fake
- Key code excerpts:

```diff
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py
@@ -10,6 +10,7 @@
+from vllm.utils.torch_utils import direct_register_custom_op
@@ -180,34 +181,74 @@ def fused_inv_rope_fp8_quant(
-    fp8_buf = torch.empty(
-        (n_groups, num_tokens, d),
-        dtype=fp8_dtype,
-        device=o.device,
```

- Reviewed files:
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` modified +106/-36
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40982 - [DSV4] Support `max` reasoning effort

- Link: https://github.com/vllm-project/vllm/pull/40982
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py`; associated commits `33f36d42605a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +126/-6, 204 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Support `max` reasoning effort"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py`; technical summary: Covers "[DSV4] Support `max` reasoning effort"; the main implementation surface is `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tokenizers_/test_deepseek_v4.py` modified +66/-1 (67 lines); hunks: -182,7 +182,7 @@ def test_deepseek_v4_renders_parsed_history_tool_arguments():; -195,6 +195,58 @@ def test_deepseek_v4_accepts_openai_reasoning_effort_values...; symbols: test_deepseek_v4_renders_parsed_history_tool_arguments, test_deepseek_v4_accepts_openai_reasoning_effort_values, test_deepseek_v4_none_reasoning_effort_disables_thinking, test_deepseek_v4_maps_compatible_thinking_reasoning_effort_values, touching `test_deepseek_v4_renders_parsed_history_tool_arguments, test_deepseek_v4_accepts_openai_reasoning_effort_values, test_deepseek_v4_none_reasoning_effort_disables_thinking`; `vllm/tokenizers/deepseek_v4.py` modified +8/-2 (10 lines); hunks: -40,10 +40,16 @@ def apply_chat_template(; symbols: apply_chat_template, touching `apply_chat_template`.
- Code diff details:
  - `tests/tokenizers_/test_deepseek_v4.py` modified +66/-1 (67 lines); hunks: -182,7 +182,7 @@ def test_deepseek_v4_renders_parsed_history_tool_arguments():; -195,6 +195,58 @@ def test_deepseek_v4_accepts_openai_reasoning_effort_values...; symbols: test_deepseek_v4_renders_parsed_history_tool_arguments, test_deepseek_v4_accepts_openai_reasoning_effort_values, test_deepseek_v4_none_reasoning_effort_disables_thinking, test_deepseek_v4_maps_compatible_thinking_reasoning_effort_values
  - `vllm/tokenizers/deepseek_v4.py` modified +8/-2 (10 lines); hunks: -40,10 +40,16 @@ def apply_chat_template(; symbols: apply_chat_template
- Key code excerpts:

```diff
diff -- tests/tokenizers_/test_deepseek_v4.py
@@ -182,7 +182,7 @@ def test_deepseek_v4_renders_parsed_history_tool_arguments():
-@pytest.mark.parametrize("reasoning_effort", ["none", "low", "medium", "high"])
+@pytest.mark.parametrize("reasoning_effort", ["minimal", "low", "medium", "high"])
@@ -195,6 +195,58 @@ def test_deepseek_v4_accepts_openai_reasoning_effort_values(reasoning_effort):
+def test_deepseek_v4_none_reasoning_effort_disables_thinking():
+    prompt = _tokenizer().apply_chat_template(
+        [{"role": "user", "content": "Hello"}],
diff -- vllm/tokenizers/deepseek_v4.py
@@ -40,10 +40,16 @@ def apply_chat_template(
-            # The V4 reference currently accepts only "max", "high", or None.
-            if reasoning_effort not in ("max", "high"):
+            if not isinstance(reasoning_effort, str):
+            elif reasoning_effort == "none":
+                thinking_mode = "chat"
+                reasoning_effort = None
```

- Reviewed files:
  - tests: `tests/tokenizers_/test_deepseek_v4.py` modified +66/-1
  - runtime: `vllm/tokenizers/deepseek_v4.py` modified +8/-2
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/chat_completion/test_chat.py`, `tests/entrypoints/openai/parser/test_harmony_utils.py`, `tests/tokenizers_/test_deepseek_v4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41148 - [Bugfix] Fix repeated DSv4 RoPE cache initialization

- Link: https://github.com/vllm-project/vllm/pull/41148
- Status/date: merged / 2026-04-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-3, 42 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix repeated DSv4 RoPE cache initialization"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/models/deepseek_v4.py`; technical summary: Covers "[Bugfix] Fix repeated DSv4 RoPE cache initialization"; the main implementation surface is `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +11/-2 (13 lines); hunks: -45,6 +45,7 @@ def __init__(; -65,7 +66,13 @@ def __init__(; symbols: __init__, _compute_inv_freq, DeepseekV4ScalingRotaryEmbedding, touching `__init__, _compute_inv_freq, DeepseekV4ScalingRotaryEmbedding`; `vllm/model_executor/models/deepseek_v4.py` modified +0/-1 (1 lines); hunks: -1027,7 +1027,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +11/-2 (13 lines); hunks: -45,6 +45,7 @@ def __init__(; -65,7 +66,13 @@ def __init__(; symbols: __init__, _compute_inv_freq, DeepseekV4ScalingRotaryEmbedding
  - `vllm/model_executor/models/deepseek_v4.py` modified +0/-1 (1 lines); hunks: -1027,7 +1027,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py
@@ -45,6 +45,7 @@ def __init__(
+        init_cache: bool = True,
@@ -65,7 +66,13 @@ def __init__(
-            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
+            head_size,
+            rotary_dim,
+            max_position_embeddings,
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -1027,7 +1027,6 @@ def __init__(
-            dtype=config.torch_dtype,
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +11/-2; `vllm/model_executor/models/deepseek_v4.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41015 - [DSv4] Use `cvt` PTX for FP32->FP4 conversion

- Link: https://github.com/vllm-project/vllm/pull/41015
- Status/date: merged / 2026-04-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +344/-62, 509 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4] Use `cvt` PTX for FP32->FP4 conversion"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`; technical summary: Covers "[DSv4] Use `cvt` PTX for FP32->FP4 conversion"; the main implementation surface is `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/kernels/test_compressor_kv_cache.py` modified +228/-4 (232 lines); hunks: -3,12 +3,11; -21,6 +20,12; symbols: _ue8m0_reference, test_deepseek_v4_quant_magnitude_range, _reference_kv_compress_norm_rope, test_fused_kv_insert_indexer, touching `_ue8m0_reference, test_deepseek_v4_quant_magnitude_range, _reference_kv_compress_norm_rope`; `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +90/-17 (107 lines); hunks: -30,13 +30,64; -49,22 +100,33 @@ def _reference(; symbols: quantize_to_mxfp4, _reference, test_fused_indexer_q_rope_quant_matches_unfused, touching `quantize_to_mxfp4, _reference, test_fused_indexer_q_rope_quant_matches_unfused`; `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +20/-35 (55 lines); hunks: -24,36 +24,22 @@ def _get_cos_sin(; -65,17 +51,16 @@ def _quantize_mxfp4_pair(x_lo, x_hi):; symbols: _get_cos_sin, _e2m1_nibble, _fp32x2_to_fp4x2, _quantize_mxfp4_pair, touching `_get_cos_sin, _e2m1_nibble, _fp32x2_to_fp4x2`; `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` modified +6/-6 (12 lines); hunks: -21,7 +21,7; -566,18 +566,18 @@ def _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(; symbols: _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn, touching `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn`.
- Code diff details:
  - `tests/kernels/test_compressor_kv_cache.py` modified +228/-4 (232 lines); hunks: -3,12 +3,11; -21,6 +20,12; symbols: _ue8m0_reference, test_deepseek_v4_quant_magnitude_range, _reference_kv_compress_norm_rope, test_fused_kv_insert_indexer
  - `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +90/-17 (107 lines); hunks: -30,13 +30,64; -49,22 +100,33 @@ def _reference(; symbols: quantize_to_mxfp4, _reference, test_fused_indexer_q_rope_quant_matches_unfused
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +20/-35 (55 lines); hunks: -24,36 +24,22 @@ def _get_cos_sin(; -65,17 +51,16 @@ def _quantize_mxfp4_pair(x_lo, x_hi):; symbols: _get_cos_sin, _e2m1_nibble, _fp32x2_to_fp4x2, _quantize_mxfp4_pair
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` modified +6/-6 (12 lines); hunks: -21,7 +21,7; -566,18 +566,18 @@ def _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(; symbols: _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
- Key code excerpts:

```diff
diff -- tests/kernels/test_compressor_kv_cache.py
@@ -3,12 +3,11 @@
-Two paths tested:
+Four test functions cover five paths:
-These serve as golden references for validating the future fused
-compressor+quant+cache kernel.
+  C) DeepseekV4 Attention magnitude range: correctness across small/large values
+  D) Indexer fused Triton kernel: compress+norm+rope+quant+insert
diff -- tests/kernels/test_fused_indexer_q_rope_quant.py
@@ -30,13 +30,64 @@
+def quantize_to_mxfp4(
+    x: torch.Tensor,
+) -> tuple[torch.Tensor, torch.Tensor]:
+    """Reference MXFP4 quantization.
+    Args:
+        x: [..., head_dim] where head_dim is divisible by 32
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py
@@ -24,36 +24,22 @@ def _get_cos_sin(
```

- Reviewed files:
  - tests: `tests/kernels/test_compressor_kv_cache.py` modified +228/-4; `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +90/-17
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +20/-35; `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` modified +6/-6
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41374 - [DSV4] Avoid redundant dtype conversion.

- Link: https://github.com/vllm-project/vllm/pull/41374
- Status/date: merged / 2026-04-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-6, 38 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Avoid redundant dtype conversion."; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_v4.py`; technical summary: Covers "[DSV4] Avoid redundant dtype conversion."; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +11/-6 (17 lines); hunks: -854,10 +854,9 @@ def _init_fused_moe_experts(; -1225,7 +1224,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: st...; symbols: _init_fused_moe_experts, forward, __init__, touching `_init_fused_moe_experts, forward, __init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +11/-6 (17 lines); hunks: -854,10 +854,9 @@ def _init_fused_moe_experts(; -1225,7 +1224,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: st...; symbols: _init_fused_moe_experts, forward, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -854,10 +854,9 @@ def _init_fused_moe_experts(
-        if self.gate.tid2eid is not None:
-            if input_ids is None:
-                raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
-            input_ids = input_ids.to(dtype=self.hash_indices_dtype)
+        if self.gate.tid2eid is not None and input_ids is None:
+            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +11/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41255 - [Perf] Intergrate Tile Kernels `head_compute_mix_kernel` for Deepseek-V4

- Link: https://github.com/vllm-project/vllm/pull/41255
- Status/date: merged / 2026-05-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +153/-9, 180 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Intergrate Tile Kernels `head_compute_mix_kernel` for Deepseek-V4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py`; technical summary: Covers "[Perf] Intergrate Tile Kernels `head_compute_mix_kernel` for Deepseek-V4"; the main implementation surface is `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/mhc.py` modified +134/-0 (134 lines); hunks: -448,3 +448,137 @@ def _mhc_post_fake(; symbols: _mhc_post_fake, hc_head_fuse_tilelang, _hc_head_fused_kernel, touching `_mhc_post_fake, hc_head_fuse_tilelang, _hc_head_fused_kernel`; `vllm/model_executor/models/deepseek_v4.py` modified +19/-9 (28 lines); hunks: -7,7 +7,6; -1456,14 +1455,25 @@ def hc_head(; symbols: hc_head, _make_deepseek_v4_weights_mapper, touching `hc_head, _make_deepseek_v4_weights_mapper`.
- Code diff details:
  - `vllm/model_executor/layers/mhc.py` modified +134/-0 (134 lines); hunks: -448,3 +448,137 @@ def _mhc_post_fake(; symbols: _mhc_post_fake, hc_head_fuse_tilelang, _hc_head_fused_kernel
  - `vllm/model_executor/models/deepseek_v4.py` modified +19/-9 (28 lines); hunks: -7,7 +7,6; -1456,14 +1455,25 @@ def hc_head(; symbols: hc_head, _make_deepseek_v4_weights_mapper
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/mhc.py
@@ -448,3 +448,137 @@ def _mhc_post_fake(
+@tilelang.jit(
+    pass_configs={
+        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
+        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
+        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
+    },
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -7,7 +7,6 @@
-import torch.nn.functional as F
@@ -1456,14 +1455,25 @@ def hc_head(
-    x = hidden_states
-    shape, dtype = x.size(), x.dtype
-    x = x.flatten(1).float()
-    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/mhc.py` modified +134/-0; `vllm/model_executor/models/deepseek_v4.py` modified +19/-9
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41443 - [DSV4] Add knob to enable pre-attn gemm

- Link: https://github.com/vllm-project/vllm/pull/41443
- Status/date: merged / 2026-05-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +24/-3, 82 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Add knob to enable pre-attn gemm"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/envs.py`, `vllm/utils/multi_stream_utils.py`; technical summary: Covers "[DSV4] Add knob to enable pre-attn gemm"; the main implementation surface is `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/envs.py`, `vllm/utils/multi_stream_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +3/-0 (3 lines); hunks: -13,6 +13,7; -385,6 +386,8 @@ def fused_wqa_wkv() -> torch.Tensor:; symbols: fused_wqa_wkv, touching `fused_wqa_wkv`; `vllm/envs.py` modified +12/-0 (12 lines); hunks: -245,6 +245,7; -1669,6 +1670,17 @@ def _get_or_set_default() -> str:; symbols: _get_or_set_default, touching `_get_or_set_default`; `vllm/utils/multi_stream_utils.py` modified +9/-3 (12 lines); hunks: -64,6 +64,7 @@ def execute_in_parallel(; -74,8 +75,9 @@ def execute_in_parallel(; symbols: execute_in_parallel, touching `execute_in_parallel`.
- Code diff details:
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +3/-0 (3 lines); hunks: -13,6 +13,7; -385,6 +386,8 @@ def fused_wqa_wkv() -> torch.Tensor:; symbols: fused_wqa_wkv
  - `vllm/envs.py` modified +12/-0 (12 lines); hunks: -245,6 +245,7; -1669,6 +1670,17 @@ def _get_or_set_default() -> str:; symbols: _get_or_set_default
  - `vllm/utils/multi_stream_utils.py` modified +9/-3 (12 lines); hunks: -64,6 +64,7 @@ def execute_in_parallel(; -74,8 +75,9 @@ def execute_in_parallel(; symbols: execute_in_parallel
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -13,6 +13,7 @@
+import vllm.envs as envs
@@ -385,6 +386,8 @@ def fused_wqa_wkv() -> torch.Tensor:
+            enable=hidden_states.shape[0]
+            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
diff -- vllm/envs.py
@@ -245,6 +245,7 @@
+    VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD: int = 4096
@@ -1669,6 +1670,17 @@ def _get_or_set_default() -> str:
+    # Token-count cutoff for multi-stream overlap of the attention input
+    # GEMM with auxiliary GEMMs (e.g. fused_wqa_wkv overlapped with indexer
+    # weights / kv-score projections in DeepSeek-V4). At or below this many
+    # tokens the FP8 main GEMM has idle SMs to share with the bf16 aux GEMMs
diff -- vllm/utils/multi_stream_utils.py
@@ -64,6 +64,7 @@ def execute_in_parallel(
+    enable: bool = False,
@@ -74,8 +75,9 @@ def execute_in_parallel(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +3/-0; `vllm/envs.py` modified +12/-0; `vllm/utils/multi_stream_utils.py` modified +9/-3
- Risk and verification: Runtime changes concentrate in `vllm/envs.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/utils/multi_stream_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41522 - [DSV4] Guard megamoe flag with Pure TP

- Link: https://github.com/vllm-project/vllm/pull/41522
- Status/date: merged / 2026-05-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-10, 42 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Guard megamoe flag with Pure TP"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_v4.py`; technical summary: Covers "[DSV4] Guard megamoe flag with Pure TP"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +16/-10 (26 lines); hunks: -715,12 +715,15 @@ def __init__(; -1223,12 +1226,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +16/-10 (26 lines); hunks: -715,12 +715,15 @@ def __init__(; -1223,12 +1226,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -715,12 +715,15 @@ def __init__(
-        if vllm_config.parallel_config.enable_expert_parallel:
-            self.use_mega_moe = (
-                vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
+        self.use_mega_moe = (
+            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
+        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +16/-10
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40871 - [New Model][ROCm] Add AMD support for DeepSeek V4

- Link: https://github.com/vllm-project/vllm/pull/40871
- Status/date: merged / 2026-05-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 22 files, +939/-134, 1657 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[New Model][ROCm] Add AMD support for DeepSeek V4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`; technical summary: Covers "[New Model][ROCm] Add AMD support for DeepSeek V4"; the main implementation surface is `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/mhc.py` modified +105/-2 (107 lines); hunks: -234,6 +234,39 @@ def mhc_pre(; -414,6 +447,14 @@ def mhc_post(; symbols: mhc_pre, mhc_post, hc_head_fuse_tilelang, _hc_head_fused_reference, touching `mhc_pre, mhc_post, hc_head_fuse_tilelang`; `vllm/model_executor/layers/deepseek_v4_attention.py` modified +73/-19 (92 lines); hunks: -28,6 +28,11; -53,6 +58,7; symbols: __init__, forward, attn_gemm_parallel_execute, touching `__init__, forward, attn_gemm_parallel_execute`; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +79/-2 (81 lines); hunks: -18,6 +18,7; -64,6 +65,8 @@ class Mxfp4MoeBackend(Enum):; symbols: Mxfp4MoeBackend, _get_priority_backends, _return_or_raise, convert_weight_to_mxfp4_moe_kernel_format, touching `Mxfp4MoeBackend, _get_priority_backends, _return_or_raise`; `vllm/model_executor/layers/sparse_attn_indexer.py` modified +22/-8 (30 lines); hunks: -499,13 +499,31 @@ def forward_hip(; -522,8 +540,4 @@ def forward_hip(; symbols: forward_hip, touching `forward_hip`.
- Code diff details:
  - `vllm/model_executor/layers/mhc.py` modified +105/-2 (107 lines); hunks: -234,6 +234,39 @@ def mhc_pre(; -414,6 +447,14 @@ def mhc_post(; symbols: mhc_pre, mhc_post, hc_head_fuse_tilelang, _hc_head_fused_reference
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +73/-19 (92 lines); hunks: -28,6 +28,11; -53,6 +58,7; symbols: __init__, forward, attn_gemm_parallel_execute
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +79/-2 (81 lines); hunks: -18,6 +18,7; -64,6 +65,8 @@ class Mxfp4MoeBackend(Enum):; symbols: Mxfp4MoeBackend, _get_priority_backends, _return_or_raise, convert_weight_to_mxfp4_moe_kernel_format
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +22/-8 (30 lines); hunks: -499,13 +499,31 @@ def forward_hip(; -522,8 +540,4 @@ def forward_hip(; symbols: forward_hip
  - `vllm/model_executor/kernels/linear/scaled_mm/aiter.py` modified +15/-0 (15 lines); hunks: -312,6 +312,21 @@ def apply_block_scaled_mm(; symbols: apply_block_scaled_mm
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/mhc.py
@@ -234,6 +234,39 @@ def mhc_pre(
+    if current_platform.is_rocm():
+        x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
+        mixes = torch.matmul(x, fn_flat.t())
+        sqrsum = x.square().sum(dim=-1, keepdim=True)
+        mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
+        pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -28,6 +28,11 @@
+from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
+    rocm_forward_decode_fallback,
+    rocm_inv_rope_einsum,
+    rocm_sparse_attn_prefill,
+)
@@ -53,6 +58,7 @@
diff -- vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
@@ -18,6 +18,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/mhc.py` modified +105/-2; `vllm/model_executor/layers/deepseek_v4_attention.py` modified +73/-19; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +79/-2; `vllm/model_executor/layers/sparse_attn_indexer.py` modified +22/-8; `vllm/model_executor/kernels/linear/scaled_mm/aiter.py` modified +15/-0; `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +9/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_topk_softplus_sqrt.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41801 - [Bugfix] DeepSeekV32/v4: respect string='true|false' attribute andunwrap arguments/input wrapper

- Link: https://github.com/vllm-project/vllm/pull/41801
- Status/date: merged / 2026-05-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +224/-10, 298 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] DeepSeekV32/v4: respect string='true|false' attribute andunwrap arguments/input wrapper"; model line: DeepSeek V4; category: bug fix; main diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`; technical summary: Covers "[Bugfix] DeepSeekV32/v4: respect string='true|false' attribute andunwrap arguments/input wrapper"; the main implementation surface is `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +155/-2 (157 lines); hunks: -203,7 +203,14 @@ def test_type_conversion_in_non_streaming(self):; -212,6 +219,118 @@ def test_type_conversion_in_non_streaming(self):; symbols: test_type_conversion_in_non_streaming, test_string_attr_true_preserves_literal_despite_schema, test_string_attr_false_allows_schema_conversion, test_arguments_wrapper_repaired, touching `test_type_conversion_in_non_streaming, test_string_attr_true_preserves_literal_despite_schema, test_string_attr_false_allows_schema_conversion`; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +36/-8 (44 lines); hunks: -69,7 +69,7 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; -101,10 +101,12 @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, _generate_tool_call_id, _parse_invoke_params, _convert_param_value_checked, touching `__init__, _generate_tool_call_id, _parse_invoke_params`; `tests/tool_parsers/test_deepseekv4_tool_parser.py` modified +33/-0 (33 lines); hunks: -203,3 +203,36 @@ def test_get_vllm_registry_structural_tag_returns_structura...; symbols: test_get_vllm_registry_structural_tag_returns_structural_tag, test_extract_tool_calls_arguments_wrapper, touching `test_get_vllm_registry_structural_tag_returns_structural_tag, test_extract_tool_calls_arguments_wrapper`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +155/-2 (157 lines); hunks: -203,7 +203,14 @@ def test_type_conversion_in_non_streaming(self):; -212,6 +219,118 @@ def test_type_conversion_in_non_streaming(self):; symbols: test_type_conversion_in_non_streaming, test_string_attr_true_preserves_literal_despite_schema, test_string_attr_false_allows_schema_conversion, test_arguments_wrapper_repaired
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +36/-8 (44 lines); hunks: -69,7 +69,7 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; -101,10 +101,12 @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, _generate_tool_call_id, _parse_invoke_params, _convert_param_value_checked
  - `tests/tool_parsers/test_deepseekv4_tool_parser.py` modified +33/-0 (33 lines); hunks: -203,3 +203,36 @@ def test_get_vllm_registry_structural_tag_returns_structura...; symbols: test_get_vllm_registry_structural_tag_returns_structural_tag, test_extract_tool_calls_arguments_wrapper
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -203,7 +203,14 @@ def test_type_conversion_in_non_streaming(self):
-        model_output = build_tool_call("toggle", {"enabled": "true", "count": "42"})
+        model_output = (
+            f"{FC_START}\n"
+            f'{INV_START}toggle">\n'
+            f'{PARAM_START}enabled" string="false">true{PARAM_END}\n'
+            f'{PARAM_START}count" string="false">42{PARAM_END}\n'
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -69,7 +69,7 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
-            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(?:true|false)"\s*>(.*?)</｜DSML｜parameter>',
+            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>',
@@ -101,10 +101,12 @@ def _generate_tool_call_id(self) -> str:
-    def _parse_invoke_params(self, invoke_str: str) -> dict:
-        param_dict = dict()
-        for param_name, param_val in self.parameter_complete_regex.findall(invoke_str):
diff -- tests/tool_parsers/test_deepseekv4_tool_parser.py
@@ -203,3 +203,36 @@ def test_get_vllm_registry_structural_tag_returns_structural_tag(
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +155/-2; `tests/tool_parsers/test_deepseekv4_tool_parser.py` modified +33/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +36/-8
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41428 - [DSv4] Improved fused Indexer Q quant kernel

- Link: https://github.com/vllm-project/vllm/pull/41428
- Status/date: merged / 2026-05-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +474/-25, 527 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4] Improved fused Indexer Q quant kernel"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `vllm/utils/import_utils.py`; technical summary: Covers "[DSv4] Improved fused Indexer Q quant kernel"; the main implementation surface is `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `vllm/utils/import_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, _recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32, touching `fused_indexer_q_rope_quant_mxfp4_cutedsl, _recast_val, _fp32x2_to_bf16x2`; `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +45/-24 (69 lines); hunks: -1,8 +1,10; -342,30 +344,49 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant, touching `fused_indexer_q_rope_quant`; `vllm/utils/import_utils.py` modified +5/-0 (5 lines); hunks: -469,3 +469,8 @@ def has_mori() -> bool:; symbols: has_mori, has_fbgemm_gpu, has_cutedsl, touching `has_mori, has_fbgemm_gpu, has_cutedsl`; `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7 @@ def _reference(; symbols: _reference, touching `_reference`.
- Code diff details:
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, _recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +45/-24 (69 lines); hunks: -1,8 +1,10; -342,30 +344,49 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant
  - `vllm/utils/import_utils.py` modified +5/-0 (5 lines); hunks: -469,3 +469,8 @@ def has_mori() -> bool:; symbols: has_mori, has_fbgemm_gpu, has_cutedsl
  - `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7 @@ def _reference(; symbols: _reference
- Key code excerpts:

```diff
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py
@@ -0,0 +1,423 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# once we have more CuteDSL kernels in vLLM, we can refactor small helper functions
+# to a separate file
+from functools import cache
+import cutlass
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py
@@ -1,8 +1,10 @@
+from vllm.utils.import_utils import has_cutedsl
@@ -342,30 +344,49 @@ def fused_indexer_q_rope_quant(
-        _fused_indexer_q_rope_mxfp4_kernel[(num_tokens, num_index_q_heads)](
-            positions,
-            index_q,
-            index_q.stride(0),
diff -- vllm/utils/import_utils.py
@@ -469,3 +469,8 @@ def has_mori() -> bool:
```

- Reviewed files:
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` added +423/-0; `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +45/-24; `vllm/utils/import_utils.py` modified +5/-0
  - tests: `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_indexer_q_rope_quant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41957 - [Bugfix][PD] Fix DSv4 Disaggregated

- Link: https://github.com/vllm-project/vllm/pull/41957
- Status/date: merged / 2026-05-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +49/-35, 213 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][PD] Fix DSv4 Disaggregated"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py`, `tests/v1/kv_connector/unit/test_tp_mapping.py`; technical summary: Covers "[Bugfix][PD] Fix DSv4 Disaggregated"; the main implementation surface is `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py`, `tests/v1/kv_connector/unit/test_tp_mapping.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py` modified +23/-23 (46 lines); hunks: -53,6 +53,7; -100,24 +101,24 @@ def _compute_desc_ids(; symbols: _compute_desc_ids, __init__, add_remote_agent, _validate_remote_agent_handshake, touching `_compute_desc_ids, __init__, add_remote_agent`; `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py` modified +9/-9 (18 lines); hunks: -10,6 +10,7; -62,25 +63,24 @@ class TPMapping:; symbols: TPMapping, compute_tp_mapping, touching `TPMapping, compute_tp_mapping`; `tests/v1/kv_connector/unit/test_tp_mapping.py` modified +7/-2 (9 lines); hunks: -9,6 +9,8; -33,12 +35,15 @@ def _compute_mapping(; symbols: _compute_mapping, touching `_compute_mapping`; `vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py` modified +9/-0 (9 lines); hunks: -10,6 +10,7; -46,3 +47,11 @@ def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Sock...; symbols: zmq_ctx, get_representative_spec_type, touching `zmq_ctx, get_representative_spec_type`.
- Code diff details:
  - `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py` modified +23/-23 (46 lines); hunks: -53,6 +53,7; -100,24 +101,24 @@ def _compute_desc_ids(; symbols: _compute_desc_ids, __init__, add_remote_agent, _validate_remote_agent_handshake
  - `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py` modified +9/-9 (18 lines); hunks: -10,6 +10,7; -62,25 +63,24 @@ class TPMapping:; symbols: TPMapping, compute_tp_mapping
  - `tests/v1/kv_connector/unit/test_tp_mapping.py` modified +7/-2 (9 lines); hunks: -9,6 +9,8; -33,12 +35,15 @@ def _compute_mapping(; symbols: _compute_mapping
  - `vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py` modified +9/-0 (9 lines); hunks: -10,6 +10,7; -46,3 +47,11 @@ def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Sock...; symbols: zmq_ctx, get_representative_spec_type
  - `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1 (2 lines); hunks: -593,7 +593,7 @@ def describe(self, remote_engine_id: EngineId) -> str:; symbols: describe
- Key code excerpts:

```diff
diff -- vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py
@@ -53,6 +53,7 @@
+    get_representative_spec_type,
@@ -100,24 +101,24 @@ def _compute_desc_ids(
-        ratio = physical_blocks_per_logical
-        logical_blocks = num_blocks // ratio
+            # NOTE (NickLucche) With HMA, every kv group has the same number of layers
+            # and layers from different groups share the same kv tensor.
diff -- vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py
@@ -10,6 +10,7 @@
+    TransferTopology,
@@ -62,25 +63,24 @@ class TPMapping:
-    tp_rank: int,
-    tp_size: int,
+    transfer_topology: TransferTopology,
-    is_mla: bool,
diff -- tests/v1/kv_connector/unit/test_tp_mapping.py
@@ -9,6 +9,8 @@
```

- Reviewed files:
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py` modified +23/-23; `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py` modified +9/-9; `vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py` modified +9/-0; `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1
  - tests: `tests/v1/kv_connector/unit/test_tp_mapping.py` modified +7/-2
- Risk and verification: The diff ships test coverage in `tests/v1/kv_connector/unit/test_tp_mapping.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42169 - [Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len

- Link: https://github.com/vllm-project/vllm/pull/42169
- Status/date: merged / 2026-05-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len"; model line: DeepSeek V4; category: bug fix; main diff: `csrc/topk.cu`; technical summary: Covers "[Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len"; the main implementation surface is `csrc/topk.cu`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `csrc/topk.cu` modified +2/-2 (4 lines); hunks: -20,7 +20,7 @@ void launch_persistent_topk(const torch::Tensor& logits,; -243,7 +243,7 @@ void persistent_topk(const torch::Tensor& logits, const torc....
- Code diff details:
  - `csrc/topk.cu` modified +2/-2 (4 lines); hunks: -20,7 +20,7 @@ void launch_persistent_topk(const torch::Tensor& logits,; -243,7 +243,7 @@ void persistent_topk(const torch::Tensor& logits, const torc...
- Key code excerpts:

```diff
diff -- csrc/topk.cu
@@ -20,7 +20,7 @@ void launch_persistent_topk(const torch::Tensor& logits,
-  const int64_t stride = logits.size(1);
+  const int64_t stride = logits.stride(0);
@@ -243,7 +243,7 @@ void persistent_topk(const torch::Tensor& logits, const torch::Tensor& lengths,
-  const int64_t stride = logits.size(1);
+  const int64_t stride = logits.stride(0);
```

- Reviewed files:
  - other: `csrc/topk.cu` modified +2/-2
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #41694 - [DSV4] Add PP support for deepseek-v4

- Link: https://github.com/vllm-project/vllm/pull/41694
- Status/date: merged / 2026-05-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +83/-22, 216 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Add PP support for deepseek-v4"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `vllm/model_executor/models/deepseek_v4.py`, `docs/models/supported_models.md`; technical summary: Covers "[DSV4] Add PP support for deepseek-v4"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`, `docs/models/supported_models.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +82/-21 (103 lines); hunks: -12,6 +12,7; -49,6 +50,7; symbols: __init__, embed_input_ids, make_empty_intermediate_tensors, touching `__init__, embed_input_ids, make_empty_intermediate_tensors`; `docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: -385,7 +385,7 @@ th {.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +82/-21 (103 lines); hunks: -12,6 +12,7; -49,6 +50,7; symbols: __init__, embed_input_ids, make_empty_intermediate_tensors
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: -385,7 +385,7 @@ th {
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -12,6 +12,7 @@
+    get_pp_group,
@@ -49,6 +50,7 @@
+from vllm.model_executor.models.interfaces import SupportsPP
@@ -57,8 +59,10 @@
+    PPMissingLayer,
+    is_pp_missing_parameter,
diff -- docs/models/supported_models.md
@@ -385,7 +385,7 @@ th {
-| `DeepseekV4ForCausalLM` | DeepSeek-V4 | `deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V4-Pro`, etc. | | |
+| `DeepseekV4ForCausalLM` | DeepSeek-V4 | `deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V4-Pro`, etc. | | ✅︎ |
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +82/-21
  - docs: `docs/models/supported_models.md` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41536 - add fused mhc_post_pre kernel

- Link: https://github.com/vllm-project/vllm/pull/41536
- Status/date: merged / 2026-05-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +533/-11, 592 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "add fused mhc_post_pre kernel"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py`, `tests/kernels/test_mhc_kernels.py`; technical summary: Covers "add fused mhc_post_pre kernel"; the main implementation surface is `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py`, `tests/kernels/test_mhc_kernels.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/mhc.py` modified +343/-0 (343 lines); hunks: -408,6 +408,131 @@ def mhc_post_tilelang(; -427,6 +552,218 @@ def mhc_post(; symbols: mhc_post_tilelang, mhc_fused_tilelang, mhc_post, mhc_fused_post_pre, touching `mhc_post_tilelang, mhc_fused_tilelang, mhc_post`; `vllm/model_executor/models/deepseek_v4.py` modified +48/-11 (59 lines); hunks: -1199,23 +1199,53 @@ def forward(; -1320,12 +1350,19 @@ def forward(; symbols: forward, touching `forward`; `tests/kernels/test_mhc_kernels.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: sinkhorn_normalize_ref, mhc_pre_ref, mhc_post_ref, test_mhc_fused_post_pre, touching `sinkhorn_normalize_ref, mhc_pre_ref, mhc_post_ref`.
- Code diff details:
  - `vllm/model_executor/layers/mhc.py` modified +343/-0 (343 lines); hunks: -408,6 +408,131 @@ def mhc_post_tilelang(; -427,6 +552,218 @@ def mhc_post(; symbols: mhc_post_tilelang, mhc_fused_tilelang, mhc_post, mhc_fused_post_pre
  - `vllm/model_executor/models/deepseek_v4.py` modified +48/-11 (59 lines); hunks: -1199,23 +1199,53 @@ def forward(; -1320,12 +1350,19 @@ def forward(; symbols: forward
  - `tests/kernels/test_mhc_kernels.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: sinkhorn_normalize_ref, mhc_pre_ref, mhc_post_ref, test_mhc_fused_post_pre
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/mhc.py
@@ -408,6 +408,131 @@ def mhc_post_tilelang(
+@tilelang.jit(
+    pass_configs={
+        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
+        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
+        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
+    },
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -1199,23 +1199,53 @@ def forward(
+        post_mix: torch.Tensor | None,
+        res_mix: torch.Tensor | None,
+        residual: torch.Tensor | None,
-        residual = x
-        x, post, comb = self.hc_pre(
-            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
diff -- tests/kernels/test_mhc_kernels.py
@@ -0,0 +1,142 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/mhc.py` modified +343/-0; `vllm/model_executor/models/deepseek_v4.py` modified +48/-11
  - tests: `tests/kernels/test_mhc_kernels.py` added +142/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40392 - [Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA

- Link: https://github.com/vllm-project/vllm/pull/40392
- Status/date: merged / 2026-05-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +966/-109, 1331 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py`; technical summary: Covers "[Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA"; the main implementation surface is `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/attention/mla_attention.py` modified +19/-24 (43 lines); hunks: -345,6 +345,7 @@ def __init__(; -374,14 +375,21 @@ def __init__(; symbols: __init__, unified_mla_kv_cache_update, unified_mla_attention_with_output_fake, touching `__init__, unified_mla_kv_cache_update, unified_mla_attention_with_output_fake`; `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +32/-9 (41 lines); hunks: -127,29 +127,52 @@ def forward_native(; symbols: forward_native, forward_static, touching `forward_native, forward_static`; `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` modified +2/-4 (6 lines); hunks: -195,10 +195,8 @@ def forward_cuda(; symbols: forward_cuda, _apply_rotary_embedding, touching `forward_cuda, _apply_rotary_embedding`; `tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py` added +413/-0 (413 lines); hunks: -0,0 +1,413; symbols: MLARoPEKVCacheCatTestModel, __init__, build_attn_metadata, forward, touching `MLARoPEKVCacheCatTestModel, __init__, build_attn_metadata`.
- Code diff details:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +19/-24 (43 lines); hunks: -345,6 +345,7 @@ def __init__(; -374,14 +375,21 @@ def __init__(; symbols: __init__, unified_mla_kv_cache_update, unified_mla_attention_with_output_fake
  - `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +32/-9 (41 lines); hunks: -127,29 +127,52 @@ def forward_native(; symbols: forward_native, forward_static
  - `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` modified +2/-4 (6 lines); hunks: -195,10 +195,8 @@ def forward_cuda(; symbols: forward_cuda, _apply_rotary_embedding
  - `tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py` added +413/-0 (413 lines); hunks: -0,0 +1,413; symbols: MLARoPEKVCacheCatTestModel, __init__, build_attn_metadata, forward
  - `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py` added +271/-0 (271 lines); hunks: -0,0 +1,271; symbols: fused_rope_unified_mla_kv_cache_update_impl, fused_rope_unified_mla_kv_cache_update_fake, MLARoPEKVCacheCatPattern, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -345,6 +345,7 @@ def __init__(
+        attn_backend: type[AttentionBackend] | None = None,
@@ -374,14 +375,21 @@ def __init__(
-        self.attn_backend = get_attn_backend(
-            self.head_size,
-            dtype,
-            kv_cache_dtype,
diff -- vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py
@@ -127,29 +127,52 @@ def forward_native(
-        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
-        query_rot = query[..., : self.rotary_dim]
-        key_rot = key[..., : self.rotary_dim]
-        if self.rotary_dim < self.head_size:
-            query_pass = query[..., self.rotary_dim :]
-            key_pass = key[..., self.rotary_dim :]
diff -- vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py
@@ -195,10 +195,8 @@ def forward_cuda(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/attention/mla_attention.py` modified +19/-24; `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +32/-9; `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` modified +2/-4; `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py` added +271/-0; `vllm/compilation/passes/fusion/matcher_utils.py` modified +84/-0; `vllm/compilation/passes/utility/fix_functionalization.py` modified +39/-0
  - tests: `tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py` added +413/-0
  - other: `csrc/cache_kernels_fused.cu` modified +75/-60
- Risk and verification: The diff ships test coverage in `tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py`, `tests/compile/passes/test_rope_kvcache_fusion.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42236 - [DSv4] Improved dequant gather K cache kernel

- Link: https://github.com/vllm-project/vllm/pull/42236
- Status/date: merged / 2026-05-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +658/-100, 832 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4] Improved dequant gather K cache kernel"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py`, `tests/kernels/test_compressor_kv_cache.py`, `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py`; technical summary: Covers "[DSv4] Improved dequant gather K cache kernel"; the main implementation surface is `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py`, `tests/kernels/test_compressor_kv_cache.py`, `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py` added +334/-0 (334 lines); hunks: -0,0 +1,334; symbols: dequantize_and_gather_k_cache_cutedsl, DequantGatherKCacheKernel, __init__, __call__, touching `dequantize_and_gather_k_cache_cutedsl, DequantGatherKCacheKernel, __init__`; `tests/kernels/test_compressor_kv_cache.py` modified +141/-7 (148 lines); hunks: -3,11 +3,12; -134,7 +135,140 @@ def test_deepseek_v4_attention_quant_cache_roundtrip(num_t...; symbols: test_deepseek_v4_attention_quant_cache_roundtrip, _dequantize_and_gather_k_cache_reference, test_dequantize_and_gather_k_cache, test_indexer_gather_accepts_upper_bound_output, touching `test_deepseek_v4_attention_quant_cache_roundtrip, _dequantize_and_gather_k_cache_reference, test_dequantize_and_gather_k_cache`; `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` added +145/-0 (145 lines); hunks: -0,0 +1,145; symbols: _recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32, _bf16x2_abs, touching `_recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32`; `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +8/-92 (100 lines); hunks: -1,18 +1,22; -61,94 +65,6 @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, _recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32, touching `fused_indexer_q_rope_quant_mxfp4_cutedsl, _recast_val, _fp32x2_to_bf16x2`.
- Code diff details:
  - `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py` added +334/-0 (334 lines); hunks: -0,0 +1,334; symbols: dequantize_and_gather_k_cache_cutedsl, DequantGatherKCacheKernel, __init__, __call__
  - `tests/kernels/test_compressor_kv_cache.py` modified +141/-7 (148 lines); hunks: -3,11 +3,12; -134,7 +135,140 @@ def test_deepseek_v4_attention_quant_cache_roundtrip(num_t...; symbols: test_deepseek_v4_attention_quant_cache_roundtrip, _dequantize_and_gather_k_cache_reference, test_dequantize_and_gather_k_cache, test_indexer_gather_accepts_upper_bound_output
  - `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` added +145/-0 (145 lines); hunks: -0,0 +1,145; symbols: _recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32, _bf16x2_abs
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +8/-92 (100 lines); hunks: -1,18 +1,22; -61,94 +65,6 @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, _recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32
  - `vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py` modified +30/-1 (31 lines); hunks: -17,6 +17,7; -303,7 +304,7 @@ def _dequantize_and_gather_k_kernel(; symbols: _dequantize_and_gather_k_kernel, dequantize_and_gather_k_cache, dequantize_and_gather_k_cache_triton
- Key code excerpts:

```diff
diff -- vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py
@@ -0,0 +1,334 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from functools import cache
+import cutlass
+import cutlass.cute as cute
+import torch
diff -- tests/kernels/test_compressor_kv_cache.py
@@ -3,11 +3,12 @@
-Four test functions cover five paths:
+These tests cover:
-  B) Indexer:       head_dim=128 (all FP8), quant_block=128
-  C) DeepseekV4 Attention magnitude range: correctness across small/large values
-  D) Indexer fused Triton kernel: compress+norm+rope+quant+insert
+  B) Fused dequant+gather K cache
diff -- vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py
@@ -0,0 +1,145 @@
```

- Reviewed files:
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py` added +334/-0; `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` added +145/-0; `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +8/-92; `vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py` modified +30/-1
  - tests: `tests/kernels/test_compressor_kv_cache.py` modified +141/-7
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41812 - [ROCm][DSv4] implement flash sparse mla with triton kernels

- Link: https://github.com/vllm-project/vllm/pull/41812
- Status/date: merged / 2026-05-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +1849/-212, 2180 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][DSv4] implement flash sparse mla with triton kernels"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`; technical summary: Covers "[ROCm][DSv4] implement flash sparse mla with triton kernels"; the main implementation surface is `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +24/-46 (70 lines); hunks: -28,11 +28,7; -725,6 +721,12 @@ def __init__(; symbols: __init__, get_attn_backend, get_kv_cache_spec, forward, touching `__init__, get_attn_backend, get_kv_cache_spec`; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +758/-164 (922 lines); hunks: -905,185 +905,757 @@ def rocm_inv_rope_einsum(; -1092,38 +1664,60 @@ def rocm_forward_decode_fallback(; symbols: rocm_inv_rope_einsum, rocm_ref_sparse_attn_prefill, _validate_dsv4_sparse_dims, _pack_dense_prefix_to_ragged_kernel, touching `rocm_inv_rope_einsum, rocm_ref_sparse_attn_prefill, _validate_dsv4_sparse_dims`; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` added +682/-0 (682 lines); hunks: -0,0 +1,682; symbols: _build_indptr_from_lengths, _compute_topk_lens_kernel, _pack_global_topk_ragged_kernel, compute_global_topk_ragged_indices_and_indptr, touching `_build_indptr_from_lengths, _compute_topk_lens_kernel, _pack_global_topk_ragged_kernel`; `tests/kernels/attention/test_rocm_triton_attn_dsv4.py` added +377/-0 (377 lines); hunks: -0,0 +1,377; symbols: _ref_global_topk_ragged, _ref_sparse_prefill_ragged, _pack_fp8_ds_mla_cache, _read_fp8_ds_mla_cache, touching `_ref_global_topk_ragged, _ref_sparse_prefill_ragged, _pack_fp8_ds_mla_cache`.
- Code diff details:
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +24/-46 (70 lines); hunks: -28,11 +28,7; -725,6 +721,12 @@ def __init__(; symbols: __init__, get_attn_backend, get_kv_cache_spec, forward
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +758/-164 (922 lines); hunks: -905,185 +905,757 @@ def rocm_inv_rope_einsum(; -1092,38 +1664,60 @@ def rocm_forward_decode_fallback(; symbols: rocm_inv_rope_einsum, rocm_ref_sparse_attn_prefill, _validate_dsv4_sparse_dims, _pack_dense_prefix_to_ragged_kernel
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` added +682/-0 (682 lines); hunks: -0,0 +1,682; symbols: _build_indptr_from_lengths, _compute_topk_lens_kernel, _pack_global_topk_ragged_kernel, compute_global_topk_ragged_indices_and_indptr
  - `tests/kernels/attention/test_rocm_triton_attn_dsv4.py` added +377/-0 (377 lines); hunks: -0,0 +1,377; symbols: _ref_global_topk_ragged, _ref_sparse_prefill_ragged, _pack_fp8_ds_mla_cache, _read_fp8_ds_mla_cache
  - `vllm/v1/attention/backends/mla/sparse_swa.py` modified +6/-0 (6 lines); hunks: -112,6 +112,12 @@ def get_supported_head_sizes(cls) -> list[int]:; symbols: get_supported_head_sizes, get_builder_cls
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -28,11 +28,7 @@
-from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
-    rocm_forward_decode_fallback,
-    rocm_inv_rope_einsum,
-    rocm_sparse_attn_prefill,
-)
+from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inv_rope_einsum
diff -- vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
@@ -905,185 +905,757 @@ def rocm_inv_rope_einsum(
-def rocm_ref_sparse_attn_prefill(
+_DSV4_SPARSE_NOPE_DIM = 448
+_DSV4_SPARSE_ROPE_DIM = 64
+def _validate_dsv4_sparse_dims(
+    head_dim: int,
+    nope_head_dim: int,
diff -- vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py
@@ -0,0 +1,682 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +24/-46; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +758/-164; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` added +682/-0; `vllm/v1/attention/backends/mla/sparse_swa.py` modified +6/-0; `vllm/v1/attention/backends/mla/flashmla_sparse.py` modified +2/-2
  - tests: `tests/kernels/attention/test_rocm_triton_attn_dsv4.py` added +377/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41946 - [Bugfix] [ROCm] [DSV4] [Perf] Add aiter mhc support

- Link: https://github.com/vllm-project/vllm/pull/41946
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +1920/-1033, 3143 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] [ROCm] [DSV4] [Perf] Add aiter mhc support"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/kernels/mhc/tilelang.py`, `vllm/model_executor/kernels/mhc/triton.py`; technical summary: Covers "[Bugfix] [ROCm] [DSV4] [Perf] Add aiter mhc support"; the main implementation surface is `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/kernels/mhc/tilelang.py`, `vllm/model_executor/kernels/mhc/triton.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/mhc.py` modified +244/-988 (1232 lines); hunks: -1,1030 +1,286; symbols: compute_num_split, mhc_pre_big_fuse_tilelang, MHCPreOp, enabled, touching `compute_num_split, mhc_pre_big_fuse_tilelang, MHCPreOp`; `vllm/model_executor/kernels/mhc/tilelang.py` added +468/-0 (468 lines); hunks: -0,0 +1,468; symbols: mhc_pre_tilelang, _mhc_pre_tilelang_fake, mhc_post_tilelang, mhc_fused_post_pre_tilelang, touching `mhc_pre_tilelang, _mhc_pre_tilelang_fake, mhc_post_tilelang`; `vllm/model_executor/kernels/mhc/triton.py` added +174/-0 (174 lines); hunks: -0,0 +1,174; symbols: _rmsnorm_nw_kernel, rmsnorm_nw, _hc_head_reduce_store_kernel, hc_head_reduce_triton_kernel, touching `_rmsnorm_nw_kernel, rmsnorm_nw, _hc_head_reduce_store_kernel`; `vllm/model_executor/kernels/mhc/aiter.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: mhc_pre_aiter, _mhc_pre_aiter_fake, mhc_post_aiter, _mhc_post_aiter_fake, touching `mhc_pre_aiter, _mhc_pre_aiter_fake, mhc_post_aiter`.
- Code diff details:
  - `vllm/model_executor/layers/mhc.py` modified +244/-988 (1232 lines); hunks: -1,1030 +1,286; symbols: compute_num_split, mhc_pre_big_fuse_tilelang, MHCPreOp, enabled
  - `vllm/model_executor/kernels/mhc/tilelang.py` added +468/-0 (468 lines); hunks: -0,0 +1,468; symbols: mhc_pre_tilelang, _mhc_pre_tilelang_fake, mhc_post_tilelang, mhc_fused_post_pre_tilelang
  - `vllm/model_executor/kernels/mhc/triton.py` added +174/-0 (174 lines); hunks: -0,0 +1,174; symbols: _rmsnorm_nw_kernel, rmsnorm_nw, _hc_head_reduce_store_kernel, hc_head_reduce_triton_kernel
  - `vllm/model_executor/kernels/mhc/aiter.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: mhc_pre_aiter, _mhc_pre_aiter_fake, mhc_post_aiter, _mhc_post_aiter_fake
  - `vllm/model_executor/kernels/mhc/torch.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: mhc_pre_torch, mhc_post_torch
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/mhc.py
@@ -1,1030 +1,286 @@
-import math
-from functools import cache
-from typing import TYPE_CHECKING
+# this import will also register the custom ops
+import vllm.model_executor.kernels.mhc as mhc_kernels
+from vllm.model_executor.custom_op import CustomOp
diff -- vllm/model_executor/kernels/mhc/tilelang.py
@@ -0,0 +1,468 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+from vllm.utils.torch_utils import direct_register_custom_op
+def mhc_pre_tilelang(
+    residual: torch.Tensor,
diff -- vllm/model_executor/kernels/mhc/triton.py
@@ -0,0 +1,174 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/mhc.py` modified +244/-988; `vllm/model_executor/kernels/mhc/tilelang.py` added +468/-0; `vllm/model_executor/kernels/mhc/triton.py` added +174/-0; `vllm/model_executor/kernels/mhc/aiter.py` added +138/-0; `vllm/model_executor/kernels/mhc/torch.py` added +106/-0; `vllm/model_executor/models/deepseek_v4.py` modified +59/-38
- Risk and verification: The diff ships test coverage in `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42320 - [Bugfix] Fix DeepSeek V4 MTP HC state handling

- Link: https://github.com/vllm-project/vllm/pull/42320
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +8/-5, 29 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DeepSeek V4 MTP HC state handling"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; technical summary: Covers "[Bugfix] Fix DeepSeek V4 MTP HC state handling"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -1203,10 +1203,10 @@ def forward(; symbols: forward, touching `forward`; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +4/-1 (5 lines); hunks: -141,9 +141,12 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -1203,10 +1203,10 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +4/-1 (5 lines); hunks: -141,9 +141,12 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -1203,10 +1203,10 @@ def forward(
-        post_mix: torch.Tensor | None,
-        res_mix: torch.Tensor | None,
-        residual: torch.Tensor | None,
-    ) -> torch.Tensor:
+        post_mix: torch.Tensor | None = None,
+        res_mix: torch.Tensor | None = None,
diff -- vllm/model_executor/models/deepseek_v4_mtp.py
@@ -141,9 +141,12 @@ def forward(
-        hidden_states = self.mtp_block(
+        hidden_states, residual, post_mix, res_mix = self.mtp_block(
+        hidden_states = self.mtp_block.hc_post(
+            hidden_states, residual, post_mix, res_mix
+        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +4/-4; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41778 - [MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell

- Link: https://github.com/vllm-project/vllm/pull/41778
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +640/-89, 975 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `benchmarks/attention_benchmarks/configs/mla_prefill.yaml`, `benchmarks/attention_benchmarks/configs/mla_decode.yaml`, `vllm/model_executor/layers/attention/mla_attention.py`; technical summary: Covers "[MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell"; the main implementation surface is `benchmarks/attention_benchmarks/configs/mla_prefill.yaml`, `benchmarks/attention_benchmarks/configs/mla_decode.yaml`, `vllm/model_executor/layers/attention/mla_attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `benchmarks/attention_benchmarks/configs/mla_prefill.yaml` modified +2/-0 (2 lines); hunks: -3,6 +3,7; -120,6 +121,7 @@ prefill_backends:; `benchmarks/attention_benchmarks/configs/mla_decode.yaml` modified +1/-0 (1 lines); hunks: -53,6 +53,7 @@ backends:; `vllm/model_executor/layers/attention/mla_attention.py` modified +1/-0 (1 lines); hunks: -1362,6 +1362,7 @@ def backend_supports_prefill_query_quantization() -> bool:; symbols: backend_supports_prefill_query_quantization, touching `backend_supports_prefill_query_quantization`; `vllm/v1/attention/backends/mla/tokenspeed_mla.py` added +277/-0 (277 lines); hunks: -0,0 +1,277; symbols: _get_workspace, TokenspeedMLAMetadataBuilder, TokenspeedMLABackend, get_supported_kernel_block_sizes, touching `_get_workspace, TokenspeedMLAMetadataBuilder, TokenspeedMLABackend`.
- Code diff details:
  - `benchmarks/attention_benchmarks/configs/mla_prefill.yaml` modified +2/-0 (2 lines); hunks: -3,6 +3,7; -120,6 +121,7 @@ prefill_backends:
  - `benchmarks/attention_benchmarks/configs/mla_decode.yaml` modified +1/-0 (1 lines); hunks: -53,6 +53,7 @@ backends:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +1/-0 (1 lines); hunks: -1362,6 +1362,7 @@ def backend_supports_prefill_query_quantization() -> bool:; symbols: backend_supports_prefill_query_quantization
  - `vllm/v1/attention/backends/mla/tokenspeed_mla.py` added +277/-0 (277 lines); hunks: -0,0 +1,277; symbols: _get_workspace, TokenspeedMLAMetadataBuilder, TokenspeedMLABackend, get_supported_kernel_block_sizes
  - `vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py` added +180/-0 (180 lines); hunks: -0,0 +1,180; symbols: TokenspeedMLAPrefillBackend, get_name, supports_compute_capability, is_available
- Key code excerpts:

```diff
diff -- benchmarks/attention_benchmarks/configs/mla_prefill.yaml
@@ -3,6 +3,7 @@
+#   CuTe DSL:     tokenspeed (Blackwell + R1 dims, requires tokenspeed_mla)
@@ -120,6 +121,7 @@ prefill_backends:
+  - tokenspeed
diff -- benchmarks/attention_benchmarks/configs/mla_decode.yaml
@@ -53,6 +53,7 @@ backends:
+  - TOKENSPEED_MLA  # Blackwell + R1 dims + FP8 KV (use --kv-cache-dtype fp8)
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -1362,6 +1362,7 @@ def backend_supports_prefill_query_quantization() -> bool:
+        "TOKENSPEED_MLA",
diff -- vllm/v1/attention/backends/mla/tokenspeed_mla.py
@@ -0,0 +1,277 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""TokenSpeed CuTe DSL MLA decode backend (Blackwell, FP8 KV cache only)."""
+from typing import ClassVar
+import torch
```

- Reviewed files:
  - runtime: `benchmarks/attention_benchmarks/configs/mla_prefill.yaml` modified +2/-0; `benchmarks/attention_benchmarks/configs/mla_decode.yaml` modified +1/-0; `vllm/model_executor/layers/attention/mla_attention.py` modified +1/-0; `vllm/v1/attention/backends/mla/tokenspeed_mla.py` added +277/-0; `vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py` added +180/-0
  - other: `benchmarks/attention_benchmarks/mla_runner.py` modified +67/-63
  - tests: `tests/v1/attention/test_mla_backends.py` modified +66/-7; `tests/conftest.py` modified +22/-13
- Risk and verification: The diff ships test coverage in `tests/conftest.py`, `tests/v1/attention/test_mla_backends.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42342 - [Bug] Fix DeepSeek V4 `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'LoadCacheMode'`

- Link: https://github.com/vllm-project/vllm/pull/42342
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 7 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix DeepSeek V4 `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'LoadCacheMode'`"; model line: DeepSeek V4; category: bug fix; main diff: `requirements/cuda.txt`; technical summary: Covers "[Bug] Fix DeepSeek V4 `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'LoadCacheMode'`"; the main implementation surface is `requirements/cuda.txt`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `requirements/cuda.txt` modified +1/-1 (2 lines); hunks: -21,5 +21,5 @@ nvidia-cudnn-frontend>=1.13.0,<1.19.0.
- Code diff details:
  - `requirements/cuda.txt` modified +1/-1 (2 lines); hunks: -21,5 +21,5 @@ nvidia-cudnn-frontend>=1.13.0,<1.19.0
- Key code excerpts:

```diff
diff -- requirements/cuda.txt
@@ -21,5 +21,5 @@ nvidia-cudnn-frontend>=1.13.0,<1.19.0
-nvidia-cutlass-dsl[cu13]>=4.4.2
+nvidia-cutlass-dsl[cu13]==4.5.0
```

- Reviewed files:
  - other: `requirements/cuda.txt` modified +1/-1
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #41263 - [DSV4] Fuse norm and router for low latency scenario

- Link: https://github.com/vllm-project/vllm/pull/41263
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +815/-43, 1013 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Fuse norm and router for low latency scenario"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; technical summary: Covers "[DSV4] Fuse norm and router for low latency scenario"; the main implementation surface is `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: _dsv4_pro_norm_gate, _dsv4_pro_norm_gate_fake, NormGateLinear, __init__, touching `_dsv4_pro_norm_gate, _dsv4_pro_norm_gate_fake, NormGateLinear`; `vllm/model_executor/models/deepseek_v4.py` modified +44/-42 (86 lines); hunks: -23,11 +23,14; -755,23 +758,23 @@ def __init__(; symbols: __init__, _init_fused_moe_experts, forward, touching `__init__, _init_fused_moe_experts, forward`; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +11/-1 (12 lines); hunks: -290,6 +290,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -437,7 +442,12 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: load_weights, _remap_weight_name, _find_mtp_layer_idx, touching `load_weights, _remap_weight_name, _find_mtp_layer_idx`; `csrc/moe/dsv4_norm_router_gemm_kernel.cu` added +249/-0 (249 lines); hunks: -0,0 +1,249.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: _dsv4_pro_norm_gate, _dsv4_pro_norm_gate_fake, NormGateLinear, __init__
  - `vllm/model_executor/models/deepseek_v4.py` modified +44/-42 (86 lines); hunks: -23,11 +23,14; -755,23 +758,23 @@ def __init__(; symbols: __init__, _init_fused_moe_experts, forward
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +11/-1 (12 lines); hunks: -290,6 +290,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -437,7 +442,12 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: load_weights, _remap_weight_name, _find_mtp_layer_idx
  - `csrc/moe/dsv4_norm_router_gemm_kernel.cu` added +249/-0 (249 lines); hunks: -0,0 +1,249
  - `benchmarks/kernels/benchmark_norm_router_gemm.py` added +183/-0 (183 lines); hunks: -0,0 +1,183; symbols: unfused_norm_router_gemm, fused_norm_router_gemm, _make_inputs, calculate_diff
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py
@@ -0,0 +1,114 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Fused RMSNorm + GateLinear for DeepSeek V4 MoE routing."""
+import torch
+from torch import nn
+import vllm._custom_ops as ops
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -23,11 +23,14 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
+from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe.router.norm_gate_linear import (
+    NormGateLinear,
+)
@@ -755,23 +758,23 @@ def __init__(
diff -- vllm/model_executor/models/deepseek_v4_mtp.py
@@ -290,6 +290,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py` added +114/-0; `vllm/model_executor/models/deepseek_v4.py` modified +44/-42; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +11/-1; `vllm/_custom_ops.py` modified +30/-0
  - other: `csrc/moe/dsv4_norm_router_gemm_kernel.cu` added +249/-0; `benchmarks/kernels/benchmark_norm_router_gemm.py` added +183/-0; `csrc/moe/dsv4_norm_router_gemm_entry.cu` added +130/-0; `csrc/moe/dsv4_norm_router_gemm.h` added +30/-0
- Risk and verification: Runtime changes concentrate in `vllm/_custom_ops.py`, `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py`, `vllm/model_executor/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42112 - [Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup

- Link: https://github.com/vllm-project/vllm/pull/42112
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +9/-15, 86 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/v1/attention/backends/mla/prefill/flashinfer.py`, `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py`; technical summary: Covers "[Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup"; the main implementation surface is `vllm/v1/attention/backends/mla/prefill/flashinfer.py`, `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/v1/attention/backends/mla/prefill/flashinfer.py` modified +6/-7 (13 lines); hunks: -77,6 +77,9 @@ def __init__(; -123,21 +126,17 @@ def prepare_metadata(; symbols: __init__, _ensure_chunks, prepare_metadata, touching `__init__, _ensure_chunks, prepare_metadata`; `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` modified +3/-8 (11 lines); hunks: -61,15 +61,12 @@ def __init__(; -89,7 +86,6 @@ def run_prefill_new_tokens(; symbols: __init__, _get_workspace_buffer, prepare_metadata, run_prefill_new_tokens, touching `__init__, _get_workspace_buffer, prepare_metadata`.
- Code diff details:
  - `vllm/v1/attention/backends/mla/prefill/flashinfer.py` modified +6/-7 (13 lines); hunks: -77,6 +77,9 @@ def __init__(; -123,21 +126,17 @@ def prepare_metadata(; symbols: __init__, _ensure_chunks, prepare_metadata
  - `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` modified +3/-8 (11 lines); hunks: -61,15 +61,12 @@ def __init__(; -89,7 +86,6 @@ def run_prefill_new_tokens(; symbols: __init__, _get_workspace_buffer, prepare_metadata, run_prefill_new_tokens
- Key code excerpts:

```diff
diff -- vllm/v1/attention/backends/mla/prefill/flashinfer.py
@@ -77,6 +77,9 @@ def __init__(
+        (self._workspace_buffer,) = current_workspace_manager().get_simultaneous(
+            ((envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,), torch.uint8),
+        )
@@ -123,21 +126,17 @@ def prepare_metadata(
-        (workspace_buffer,) = current_workspace_manager().get_simultaneous(
-            ((envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,), torch.uint8),
diff -- vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py
@@ -61,15 +61,12 @@ def __init__(
-    def _get_workspace_buffer(self) -> torch.Tensor:
-        (workspace_buffer,) = current_workspace_manager().get_simultaneous(
+        (self._workspace_buffer,) = current_workspace_manager().get_simultaneous(
-        return workspace_buffer
@@ -89,7 +86,6 @@ def run_prefill_new_tokens(
-        workspace_buffer = self._get_workspace_buffer()
```

- Reviewed files:
  - runtime: `vllm/v1/attention/backends/mla/prefill/flashinfer.py` modified +6/-7; `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` modified +3/-8
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/backends/mla/prefill/flashinfer.py`, `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42604 - DeepSeekV4-Pro enable cuda graph full and piecewise mode

- Link: https://github.com/vllm-project/vllm/pull/42604
- Status/date: merged / 2026-05-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +73/-3, 125 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DeepSeekV4-Pro enable cuda graph full and piecewise mode"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/mhc.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`; technical summary: Covers "DeepSeekV4-Pro enable cuda graph full and piecewise mode"; the main implementation surface is `vllm/model_executor/layers/mhc.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/mhc.py` modified +0/-3 (3 lines); hunks: -5,7 +5,6; -190,8 +189,6 @@ def forward_cuda(; symbols: forward_cuda, forward_hip, touching `forward_cuda, forward_hip`; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` modified +73/-0 (73 lines); hunks: -302,6 +302,30 @@ def combine_topk_swa_indices_ragged(; -317,6 +341,23 @@ class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSW...; symbols: combine_topk_swa_indices_ragged, _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata, DeepseekV4ROCMAiterSparseSWAMetadata, touching `combine_topk_swa_indices_ragged, _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata`.
- Code diff details:
  - `vllm/model_executor/layers/mhc.py` modified +0/-3 (3 lines); hunks: -5,7 +5,6; -190,8 +189,6 @@ def forward_cuda(; symbols: forward_cuda, forward_hip
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` modified +73/-0 (73 lines); hunks: -302,6 +302,30 @@ def combine_topk_swa_indices_ragged(; -317,6 +341,23 @@ class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSW...; symbols: combine_topk_swa_indices_ragged, _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata, DeepseekV4ROCMAiterSparseSWAMetadata
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/mhc.py
@@ -5,7 +5,6 @@
-from vllm.platforms import current_platform
@@ -190,8 +189,6 @@ def forward_cuda(
-    # This @torch.compile is necessary for accuracy as well as performance.
-    @torch.compile(backend=current_platform.simple_compile_backend)
diff -- vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py
@@ -302,6 +302,30 @@ def combine_topk_swa_indices_ragged(
+def _copy_ragged_to_graph_buffers(
+    ragged_indices: torch.Tensor,
+    ragged_indptr: torch.Tensor,
+    ragged_indices_buffer: torch.Tensor,
+    ragged_indptr_buffer: torch.Tensor,
+    num_rows: int,
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/mhc.py` modified +0/-3; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` modified +73/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mhc.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42810 - [ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy

- Link: https://github.com/vllm-project/vllm/pull/42810
- Status/date: merged / 2026-05-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +88/-177, 364 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v4.py`; technical summary: Covers "[ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy"; the main implementation surface is `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/mhc.py` modified +48/-40 (88 lines); hunks: -61,31 +61,35 @@ def forward_hip(; -124,21 +128,25 @@ def forward_hip(; symbols: forward_hip, forward_native, touching `forward_hip, forward_native`; `vllm/model_executor/layers/sparse_attn_indexer.py` modified +5/-22 (27 lines); hunks: -505,27 +505,6 @@ def forward_hip(; -541,5 +520,9 @@ def forward_hip(; symbols: forward_hip, touching `forward_hip`; `vllm/model_executor/models/deepseek_v4.py` modified +2/-1 (3 lines); hunks: -1277,7 +1277,8 @@ def _forward_rocm(; symbols: _forward_rocm, touching `_forward_rocm`; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +33/-114 (147 lines); hunks: -542,7 +542,11 @@ def rocm_fp8_mqa_logits(; -551,6 +555,12 @@ def _topk_indices_torch(logits: torch.Tensor, topk_tokens:...; symbols: rocm_fp8_mqa_logits, _topk_indices_torch, touching `rocm_fp8_mqa_logits, _topk_indices_torch`.
- Code diff details:
  - `vllm/model_executor/layers/mhc.py` modified +48/-40 (88 lines); hunks: -61,31 +61,35 @@ def forward_hip(; -124,21 +128,25 @@ def forward_hip(; symbols: forward_hip, forward_native
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +5/-22 (27 lines); hunks: -505,27 +505,6 @@ def forward_hip(; -541,5 +520,9 @@ def forward_hip(; symbols: forward_hip
  - `vllm/model_executor/models/deepseek_v4.py` modified +2/-1 (3 lines); hunks: -1277,7 +1277,8 @@ def _forward_rocm(; symbols: _forward_rocm
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +33/-114 (147 lines); hunks: -542,7 +542,11 @@ def rocm_fp8_mqa_logits(; -551,6 +555,12 @@ def _topk_indices_torch(logits: torch.Tensor, topk_tokens:...; symbols: rocm_fp8_mqa_logits, _topk_indices_torch
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/mhc.py
@@ -61,31 +61,35 @@ def forward_hip(
-        hidden_size = residual.shape[-1]
-        if hidden_size % 256 == 0:
-            return torch.ops.vllm.mhc_pre_aiter(
-                residual,
-                fn,
-                hc_scale,
diff -- vllm/model_executor/layers/sparse_attn_indexer.py
@@ -505,27 +505,6 @@ def forward_hip(
-        if self.skip_k_cache_insert or not rocm_aiter_ops.is_enabled():
-            from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
-                rocm_aiter_sparse_attn_indexer_native,
-            )
-            return rocm_aiter_sparse_attn_indexer_native(
-                hidden_states,
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -1277,7 +1277,8 @@ def _forward_rocm(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/mhc.py` modified +48/-40; `vllm/model_executor/layers/sparse_attn_indexer.py` modified +5/-22; `vllm/model_executor/models/deepseek_v4.py` modified +2/-1; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +33/-114
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41710 - fix: remove unused norm for dpskv4

- Link: https://github.com/vllm-project/vllm/pull/41710
- Status/date: merged / 2026-05-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: remove unused norm for dpskv4"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/layers/deepseek_v4_attention.py`; technical summary: Covers "fix: remove unused norm for dpskv4"; the main implementation surface is `vllm/model_executor/layers/deepseek_v4_attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-2 (3 lines); hunks: -47,7 +47,7; -1111,7 +1111,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-2 (3 lines); hunks: -47,7 +47,7; -1111,7 +1111,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -47,7 +47,7 @@
-from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
+from vllm.model_executor.layers.layernorm import RMSNorm
@@ -1111,7 +1111,6 @@ def __init__(
-        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/deepseek_v4_attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42930 - [Bugfix] Fix DSV4 MTP after ROCm mHC integration

- Link: https://github.com/vllm-project/vllm/pull/42930
- Status/date: merged / 2026-05-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +17/-12, 57 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DSV4 MTP after ROCm mHC integration"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; technical summary: Covers "[Bugfix] Fix DSV4 MTP after ROCm mHC integration"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +12/-8 (20 lines); hunks: -1261,10 +1261,12 @@ def _forward_rocm(; -1288,10 +1290,12 @@ def forward(; symbols: _forward_rocm, forward, touching `_forward_rocm, forward`; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +5/-4 (9 lines); hunks: -146,9 +146,10 @@ def forward(; -235,7 +236,7 @@ def compute_logits(; symbols: forward, compute_logits, touching `forward, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +12/-8 (20 lines); hunks: -1261,10 +1261,12 @@ def _forward_rocm(; -1288,10 +1290,12 @@ def forward(; symbols: _forward_rocm, forward
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +5/-4 (9 lines); hunks: -146,9 +146,10 @@ def forward(; -235,7 +236,7 @@ def compute_logits(; symbols: forward, compute_logits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -1261,10 +1261,12 @@ def _forward_rocm(
-        post_mix: torch.Tensor | None,
-        res_mix: torch.Tensor | None,
-        residual: torch.Tensor | None,
-    ) -> torch.Tensor:
+        post_mix: torch.Tensor | None = None,
+        res_mix: torch.Tensor | None = None,
diff -- vllm/model_executor/models/deepseek_v4_mtp.py
@@ -146,9 +146,10 @@ def forward(
-        hidden_states = self.mtp_block.hc_post(
-            hidden_states, residual, post_mix, res_mix
-        )
+        if current_platform.is_cuda():
+            hidden_states = self.mtp_block.hc_post(
+                hidden_states, residual, post_mix, res_mix
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +12/-8; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +5/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42541 - [Bugfix] fix swiglu limit issue for humming backend + deepseek v4

- Link: https://github.com/vllm-project/vllm/pull/42541
- Status/date: merged / 2026-05-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +34/-6, 94 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix swiglu limit issue for humming backend + deepseek v4"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`, `vllm/model_executor/layers/quantization/utils/humming_utils.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`; technical summary: Covers "[Bugfix] fix swiglu limit issue for humming backend + deepseek v4"; the main implementation surface is `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`, `vllm/model_executor/layers/quantization/utils/humming_utils.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py` modified +19/-4 (23 lines); hunks: -33,7 +33,10; -422,6 +425,18 @@ def is_supported_config(; symbols: is_supported_config, apply_activation, HummingIndexedExperts, finalize_weight_and_reduce_impl, touching `is_supported_config, apply_activation, HummingIndexedExperts`; `vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +9/-1 (10 lines); hunks: -164,7 +164,12 @@ def prepare_humming_moe_layer(layer: RoutedExperts, quant_c...; -211,4 +216,7 @@ def get_humming_moe_quant_config(layer: RoutedExperts):; symbols: prepare_humming_moe_layer, get_humming_moe_quant_config, touching `prepare_humming_moe_layer, get_humming_moe_quant_config`; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +6/-1 (7 lines); hunks: -1567,7 +1567,12 @@ def make_mxfp4_moe_quant_config(; symbols: make_mxfp4_moe_quant_config, touching `make_mxfp4_moe_quant_config`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py` modified +19/-4 (23 lines); hunks: -33,7 +33,10; -422,6 +425,18 @@ def is_supported_config(; symbols: is_supported_config, apply_activation, HummingIndexedExperts, finalize_weight_and_reduce_impl
  - `vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +9/-1 (10 lines); hunks: -164,7 +164,12 @@ def prepare_humming_moe_layer(layer: RoutedExperts, quant_c...; -211,4 +216,7 @@ def get_humming_moe_quant_config(layer: RoutedExperts):; symbols: prepare_humming_moe_layer, get_humming_moe_quant_config
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +6/-1 (7 lines); hunks: -1567,7 +1567,12 @@ def make_mxfp4_moe_quant_config(; symbols: make_mxfp4_moe_quant_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py
@@ -33,7 +33,10 @@
-from vllm.model_executor.layers.fused_moe.utils import _resize_cache
+from vllm.model_executor.layers.fused_moe.utils import (
+    _resize_cache,
+    swiglu_limit_func,
+)
@@ -422,6 +425,18 @@ def is_supported_config(
diff -- vllm/model_executor/layers/quantization/utils/humming_utils.py
@@ -164,7 +164,12 @@ def prepare_humming_moe_layer(layer: RoutedExperts, quant_config: dict):
-def get_humming_moe_quant_config(layer: RoutedExperts):
+def get_humming_moe_quant_config(
+    layer: RoutedExperts,
+    gemm1_alpha: float | None = None,
+    gemm1_beta: float | None = None,
+    gemm1_clamp_limit: float | None = None,
diff -- vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
@@ -1567,7 +1567,12 @@ def make_mxfp4_moe_quant_config(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py` modified +19/-4; `vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +9/-1; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/quantization/utils/humming_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43004 - [Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]

- Link: https://github.com/vllm-project/vllm/pull/43004
- Status/date: merged / 2026-05-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/amd/__init__.py`, `vllm/models/deepseek_v4/nvidia/__init__.py`, `vllm/models/deepseek_v4/quant_config.py`; associated commits `287471b99442`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +189/-126, 476 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `vllm/models/deepseek_v4/quant_config.py`, `vllm/models/deepseek_v4/__init__.py`, `tests/models/test_deepseek_v4_mega_moe.py`; technical summary: Covers "[Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]"; the main implementation surface is `vllm/models/deepseek_v4/quant_config.py`, `vllm/models/deepseek_v4/__init__.py`, `tests/models/test_deepseek_v4_mega_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/quant_config.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: DeepseekV4FP8Config, __init__, expert_dtype, is_scale_e8m0, touching `DeepseekV4FP8Config, __init__, expert_dtype`; `vllm/models/deepseek_v4/__init__.py` added +30/-0 (30 lines); hunks: -0,0 +1,30; `tests/models/test_deepseek_v4_mega_moe.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7; `vllm/model_executor/layers/quantization/__init__.py` modified +1/-1 (2 lines); hunks: -113,7 +113,7 @@ def get_quantization_config(quantization: str) -> type[Quant...; symbols: get_quantization_config, touching `get_quantization_config`.
- Code diff details:
  - `vllm/models/deepseek_v4/quant_config.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: DeepseekV4FP8Config, __init__, expert_dtype, is_scale_e8m0
  - `vllm/models/deepseek_v4/__init__.py` added +30/-0 (30 lines); hunks: -0,0 +1,30
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7
  - `vllm/model_executor/layers/quantization/__init__.py` modified +1/-1 (2 lines); hunks: -113,7 +113,7 @@ def get_quantization_config(quantization: str) -> type[Quant...; symbols: get_quantization_config
  - `vllm/models/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/quant_config.py
@@ -0,0 +1,106 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Quantization config for DeepSeek V4."""
+from vllm.config import get_current_vllm_config
+from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
diff -- vllm/models/deepseek_v4/__init__.py
@@ -0,0 +1,30 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DeepSeek V4 model — hardware-isolated entry point.
+The actual implementation lives under ``nvidia/`` and ``amd/``; this module
+picks the right one for the current platform and re-exports the public
+classes used by the model registry and quantization config lookup.
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -6,7 +6,7 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/quant_config.py` added +106/-0; `vllm/models/deepseek_v4/__init__.py` added +30/-0; `vllm/model_executor/layers/quantization/__init__.py` modified +1/-1; `vllm/models/__init__.py` added +2/-0; `vllm/models/deepseek_v4/amd/__init__.py` added +2/-0; `vllm/models/deepseek_v4/nvidia/__init__.py` added +2/-0
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_mega_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43039 - [Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N]

- Link: https://github.com/vllm-project/vllm/pull/43039
- Status/date: merged / 2026-05-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/compressor.py`; associated commits `87b08c5f6460`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +8/-11, 62 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N]"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/compressor.py`; technical summary: Covers "[Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N]"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/compressor.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` renamed +1/-1 (2 lines); hunks: -46,7 +46,6; -55,6 +54,7; `vllm/models/deepseek_v4/compressor.py` renamed +0/-0 (0 lines).
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` renamed +1/-1 (2 lines); hunks: -46,7 +46,6; -55,6 +54,7
  - `vllm/models/deepseek_v4/compressor.py` renamed +0/-0 (0 lines)
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -46,7 +46,6 @@
-from vllm.model_executor.layers.deepseek_compressor import DeepseekCompressor
@@ -55,6 +54,7 @@
+from vllm.models.deepseek_v4.compressor import DeepseekCompressor
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` renamed +1/-1; `vllm/models/deepseek_v4/compressor.py` renamed +0/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42899 - add cutedsl dsv4 indexer fp8 kernel

- Link: https://github.com/vllm-project/vllm/pull/42899
- Status/date: merged / 2026-05-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +411/-60, 562 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "add cutedsl dsv4 indexer fp8 kernel"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`; technical summary: Covers "add cutedsl dsv4 indexer fp8 kernel"; the main implementation surface is `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +311/-37 (348 lines); hunks: -14,6 +14,7; -65,8 +66,48 @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, IndexerQMxFp4Kernel, fused_indexer_q_rope_quant_fp8_cutedsl, IndexerQRopeQuantKernel, touching `fused_indexer_q_rope_quant_mxfp4_cutedsl, IndexerQMxFp4Kernel, fused_indexer_q_rope_quant_fp8_cutedsl`; `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +37/-20 (57 lines); hunks: -398,24 +398,41 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant, touching `fused_indexer_q_rope_quant`; `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +30/-3 (33 lines); hunks: -13,13 +13,17; -125,8 +129,14 @@ def _reference(; symbols: _reference, test_fused_indexer_q_rope_quant_matches_unfused, touching `_reference, test_fused_indexer_q_rope_quant_matches_unfused`; `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` modified +33/-0 (33 lines); hunks: -117,6 +117,39 @@ def _fp8x4_to_bf16x4(x: Uint32, *, loc=None, ip=None) -> cu...; symbols: _fp8x4_to_bf16x4, _fp32x4_to_fp8x4, _fp32x8_to_fp4x8, touching `_fp8x4_to_bf16x4, _fp32x4_to_fp8x4, _fp32x8_to_fp4x8`.
- Code diff details:
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +311/-37 (348 lines); hunks: -14,6 +14,7; -65,8 +66,48 @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, IndexerQMxFp4Kernel, fused_indexer_q_rope_quant_fp8_cutedsl, IndexerQRopeQuantKernel
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +37/-20 (57 lines); hunks: -398,24 +398,41 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant
  - `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +30/-3 (33 lines); hunks: -13,13 +13,17; -125,8 +129,14 @@ def _reference(; symbols: _reference, test_fused_indexer_q_rope_quant_matches_unfused
  - `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` modified +33/-0 (33 lines); hunks: -117,6 +117,39 @@ def _fp8x4_to_bf16x4(x: Uint32, *, loc=None, ip=None) -> cu...; symbols: _fp8x4_to_bf16x4, _fp32x4_to_fp8x4, _fp32x8_to_fp4x8
- Key code excerpts:

```diff
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py
@@ -14,6 +14,7 @@
+    _fp32x4_to_fp8x4,
@@ -65,8 +66,48 @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(
-class IndexerQMxFp4Kernel:
-    """Eight-thread subwarps process one ``(token, head)`` row."""
+def fused_indexer_q_rope_quant_fp8_cutedsl(
+    positions: torch.Tensor,
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py
@@ -398,24 +398,41 @@ def fused_indexer_q_rope_quant(
-    _fused_indexer_q_rope_quant_kernel[(num_tokens, num_index_q_heads)](
-        positions,
-        index_q,
-        index_q.stride(0),
-        index_q.stride(1),
-        index_q_cos_sin_cache,
diff -- tests/kernels/test_fused_indexer_q_rope_quant.py
@@ -13,13 +13,17 @@
```

- Reviewed files:
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +311/-37; `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +37/-20; `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` modified +33/-0
  - tests: `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +30/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_indexer_q_rope_quant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43073 - [Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N]

- Link: https://github.com/vllm-project/vllm/pull/43073
- Status/date: merged / 2026-05-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/__init__.py`, `vllm/models/deepseek_v4/common/ops/__init__.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py` and 12 files; associated commits `b14be81c1f63`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +34/-29, 197 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N]"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/ops/__init__.py`, `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N]"; the main implementation surface is `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/ops/__init__.py`, `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/compressor.py` modified +6/-10 (16 lines); hunks: -11,9 +11,13; -23,14 +27,6; `vllm/models/deepseek_v4/nvidia/ops/__init__.py` added +8/-0 (8 lines); hunks: -0,0 +1,8; `vllm/models/deepseek_v4/attention.py` modified +3/-3 (6 lines); hunks: -19,16 +19,16; `vllm/models/deepseek_v4/common/ops/cache_utils.py` renamed +3/-1 (4 lines); hunks: -366,7 +366,9 @@ def dequantize_and_gather_k_cache(; symbols: dequantize_and_gather_k_cache, touching `dequantize_and_gather_k_cache`.
- Code diff details:
  - `vllm/models/deepseek_v4/compressor.py` modified +6/-10 (16 lines); hunks: -11,9 +11,13; -23,14 +27,6
  - `vllm/models/deepseek_v4/nvidia/ops/__init__.py` added +8/-0 (8 lines); hunks: -0,0 +1,8
  - `vllm/models/deepseek_v4/attention.py` modified +3/-3 (6 lines); hunks: -19,16 +19,16
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` renamed +3/-1 (4 lines); hunks: -366,7 +366,9 @@ def dequantize_and_gather_k_cache(; symbols: dequantize_and_gather_k_cache
  - `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` renamed +2/-2 (4 lines); hunks: -346,7 +346,7 @@ def fused_indexer_q_rope_quant(; -400,7 +400,7 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/compressor.py
@@ -11,9 +11,13 @@
-from vllm.model_executor.layers.linear import (
-    MergedColumnParallelLinear,
+from vllm.model_executor.layers.linear import MergedColumnParallelLinear
+from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
+    _fused_kv_compress_norm_rope_insert_indexer_attn,
+    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
diff -- vllm/models/deepseek_v4/nvidia/ops/__init__.py
@@ -0,0 +1,8 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""NVIDIA-only (cutedsl/cutlass) kernels for DeepSeek V4.
+These modules import ``cutlass``/``cutedsl`` at module top level, so they must
+not be imported on non-CUDA platforms. Callers should gate on
+``vllm.utils.import_utils.has_cutedsl()`` before importing from here.
diff -- vllm/models/deepseek_v4/attention.py
@@ -19,16 +19,16 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/compressor.py` modified +6/-10; `vllm/models/deepseek_v4/nvidia/ops/__init__.py` added +8/-0; `vllm/models/deepseek_v4/attention.py` modified +3/-3; `vllm/models/deepseek_v4/common/ops/cache_utils.py` renamed +3/-1; `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` renamed +2/-2; `vllm/models/deepseek_v4/common/__init__.py` added +2/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42828 - [KVConnector][DSV4] HMA support for Mooncake store connector

- Link: https://github.com/vllm-project/vllm/pull/42828
- Status/date: merged / 2026-05-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +1835/-446, 3088 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[KVConnector][DSV4] HMA support for Mooncake store connector"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `tests/v1/kv_connector/unit/test_mooncake_store_worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`, `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py`; technical summary: Covers "[KVConnector][DSV4] HMA support for Mooncake store connector"; the main implementation surface is `tests/v1/kv_connector/unit/test_mooncake_store_worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`, `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/v1/kv_connector/unit/test_mooncake_store_worker.py` modified +357/-117 (474 lines); hunks: -19,28 +19,48; -55,18 +75,27 @@ def _make_store_recving_thread(; symbols: _default_send_coord, _make_store_sending_thread, _make_store_recving_thread, _make_load_req, touching `_default_send_coord, _make_store_sending_thread, _make_store_recving_thread`; `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` modified +237/-180 (417 lines); hunks: -10,6 +10,7; -36,15 +37,25; symbols: KVTransferThread, __init__, KVCacheStoreSendingThread, touching `KVTransferThread, __init__, KVCacheStoreSendingThread`; `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py` added +342/-0 (342 lines); hunks: -0,0 +1,342; symbols: _DictStore, __init__, setup, register_buffer, touching `_DictStore, __init__, setup`; `tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py` added +302/-0 (302 lines); hunks: -0,0 +1,302; symbols: _make_coord, test_external_cached_block_pool_tautological_returns_present_for_any_hash, test_external_cached_block_pool_hit_all_groups, test_external_cached_block_pool_miss_one_group, touching `_make_coord, test_external_cached_block_pool_tautological_returns_present_for_any_hash, test_external_cached_block_pool_hit_all_groups`.
- Code diff details:
  - `tests/v1/kv_connector/unit/test_mooncake_store_worker.py` modified +357/-117 (474 lines); hunks: -19,28 +19,48; -55,18 +75,27 @@ def _make_store_recving_thread(; symbols: _default_send_coord, _make_store_sending_thread, _make_store_recving_thread, _make_load_req
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` modified +237/-180 (417 lines); hunks: -10,6 +10,7; -36,15 +37,25; symbols: KVTransferThread, __init__, KVCacheStoreSendingThread
  - `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py` added +342/-0 (342 lines); hunks: -0,0 +1,342; symbols: _DictStore, __init__, setup, register_buffer
  - `tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py` added +302/-0 (302 lines); hunks: -0,0 +1,302; symbols: _make_coord, test_external_cached_block_pool_tautological_returns_present_for_any_hash, test_external_cached_block_pool_hit_all_groups, test_external_cached_block_pool_miss_one_group
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py` added +290/-0 (290 lines); hunks: -0,0 +1,290; symbols: ExternalCachedBlockPool, __init__, get_cached_block, MooncakeStoreCoordinator
- Key code excerpts:

```diff
diff -- tests/v1/kv_connector/unit/test_mooncake_store_worker.py
@@ -19,28 +19,48 @@
-from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
+from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
+    worker as mooncake_store_worker,
+)
+from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
+def _default_send_coord() -> mooncake_store_worker.MooncakeStoreCoordinator:
diff -- vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py
@@ -10,6 +10,7 @@
+import dataclasses
@@ -36,15 +37,25 @@
-from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
+from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
+    ExternalCachedBlockPool,
+    MooncakeStoreCoordinator,
diff -- tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py
@@ -0,0 +1,342 @@
```

- Reviewed files:
  - tests: `tests/v1/kv_connector/unit/test_mooncake_store_worker.py` modified +357/-117; `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py` added +342/-0; `tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py` added +302/-0; `tests/v1/kv_connector/unit/test_mooncake_store_connector.py` modified +72/-94; `tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py` added +111/-0
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` modified +237/-180; `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py` added +290/-0; `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py` modified +47/-33
- Risk and verification: The diff ships test coverage in `tests/v1/kv_connector/unit/test_mooncake_store_connector.py`, `tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py`, `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py`, `tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43077 - [Model Refactoring] Rename deepseek_v4.py to model.py [4/N]

- Link: https://github.com/vllm-project/vllm/pull/43077
- Status/date: merged / 2026-05-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/model.py` and 6 files; associated commits `07beaed8422d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +8/-8, 46 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Refactoring] Rename deepseek_v4.py to model.py [4/N]"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `vllm/models/deepseek_v4/__init__.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; technical summary: Covers "[Model Refactoring] Rename deepseek_v4.py to model.py [4/N]"; the main implementation surface is `vllm/models/deepseek_v4/__init__.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/__init__.py` modified +4/-4 (8 lines); hunks: -17,11 +17,11; `tests/models/test_deepseek_v4_mega_moe.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7; `vllm/models/deepseek_v4/nvidia/mtp.py` renamed +1/-1 (2 lines); hunks: -40,7 +40,7; `vllm/models/deepseek_v4/amd/deepseek_v4_mtp.py` removed +0/-1 (1 lines); hunks: -1 +0,0.
- Code diff details:
  - `vllm/models/deepseek_v4/__init__.py` modified +4/-4 (8 lines); hunks: -17,11 +17,11
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7
  - `vllm/models/deepseek_v4/nvidia/mtp.py` renamed +1/-1 (2 lines); hunks: -40,7 +40,7
  - `vllm/models/deepseek_v4/amd/deepseek_v4_mtp.py` removed +0/-1 (1 lines); hunks: -1 +0,0
  - `vllm/models/deepseek_v4/amd/model.py` added +1/-0 (1 lines); hunks: -0,0 +1
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/__init__.py
@@ -17,11 +17,11 @@
-    from .nvidia.deepseek_v4 import DeepseekV4ForCausalLM
-    from .nvidia.deepseek_v4_mtp import DeepSeekV4MTP
+    from .nvidia.model import DeepseekV4ForCausalLM
+    from .nvidia.mtp import DeepSeekV4MTP
-    from .amd.deepseek_v4 import DeepseekV4ForCausalLM  # type: ignore[assignment]
-    from .amd.deepseek_v4_mtp import DeepSeekV4MTP  # type: ignore[assignment]
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -6,7 +6,7 @@
-from vllm.models.deepseek_v4.nvidia.deepseek_v4 import (
+from vllm.models.deepseek_v4.nvidia.model import (
diff -- vllm/models/deepseek_v4/nvidia/mtp.py
@@ -40,7 +40,7 @@
-from .deepseek_v4 import (
+from .model import (
diff -- vllm/models/deepseek_v4/amd/deepseek_v4_mtp.py
@@ -1 +0,0 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/__init__.py` modified +4/-4; `vllm/models/deepseek_v4/nvidia/mtp.py` renamed +1/-1; `vllm/models/deepseek_v4/amd/deepseek_v4_mtp.py` removed +0/-1; `vllm/models/deepseek_v4/amd/model.py` added +1/-0; `vllm/models/deepseek_v4/amd/mtp.py` added +1/-0; `vllm/models/deepseek_v4/nvidia/model.py` renamed +0/-0
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_mega_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42111 - [CI] Add DSV4-Flash to gsm8k moe-refactor/config-b200.txt

- Link: https://github.com/vllm-project/vllm/pull/42111
- Status/date: merged / 2026-05-20
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml`; associated commits `cd0ff26e7acf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +12/-1, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Add DSV4-Flash to gsm8k moe-refactor/config-b200.txt"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml`; technical summary: Covers "[CI] Add DSV4-Flash to gsm8k moe-refactor/config-b200.txt"; the main implementation surface is `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml` added +5/-0 (5 lines); hunks: -0,0 +1,5.
- Code diff details:
  - `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml` added +5/-0 (5 lines); hunks: -0,0 +1,5
- Key code excerpts:

```diff
diff -- tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml
@@ -0,0 +1,5 @@
+model_name: "deepseek-ai/DeepSeek-V4-Flash"
+accuracy_threshold: 0.95
+num_questions: 1319
+num_fewshot: 5
+server_args: "--trust-remote-code --kv-cache-dtype fp8 --block-size 256 --enable-expert-parallel --tensor-parallel-size 2 --attention_config.use_fp4_indexer_cache=True --moe-backe
```

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml` added +5/-0
- Risk and verification: The diff ships test coverage in `requirements/test/cuda.txt`, `requirements/test/rocm.txt`, `requirements/test/xpu.txt`, `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42209 - Add NVFP4 MOE support for Deepseek V4.

- Link: https://github.com/vllm-project/vllm/pull/42209
- Status/date: merged / 2026-05-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/quant_config.py`; associated commits `fb21d8b4f902`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +217/-17, 488 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add NVFP4 MOE support for Deepseek V4."; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/quant_config.py`; technical summary: Covers "Add NVFP4 MOE support for Deepseek V4."; the main implementation surface is `vllm/models/deepseek_v4/quant_config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/quant_config.py` modified +53/-1 (54 lines); hunks: -2,6 +2,10; -14,6 +18,11; symbols: DeepseekV4FP8Config, __init__, is_scale_e8m0, _resolve_moe_overrides, touching `DeepseekV4FP8Config, __init__, is_scale_e8m0`.
- Code diff details:
  - `vllm/models/deepseek_v4/quant_config.py` modified +53/-1 (54 lines); hunks: -2,6 +2,10; -14,6 +18,11; symbols: DeepseekV4FP8Config, __init__, is_scale_e8m0, _resolve_moe_overrides
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/quant_config.py
@@ -2,6 +2,10 @@
+from __future__ import annotations
+from typing import TYPE_CHECKING
@@ -14,6 +18,11 @@
+if TYPE_CHECKING:
+    from vllm.model_executor.layers.quantization.modelopt import (
+        ModelOptNvFp4Config,
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/quant_config.py` modified +53/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_trtllm_nvfp4_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43149 - [Refactor] Extract DeepSeek V4 sparse MLA impl into model folder

- Link: https://github.com/vllm-project/vllm/pull/43149
- Status/date: merged / 2026-05-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `843715739b7b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +485/-402, 1059 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Extract DeepSeek V4 sparse MLA impl into model folder"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/ops/attention.py`, `vllm/models/deepseek_v4/amd/rocm.py`; technical summary: Covers "[Refactor] Extract DeepSeek V4 sparse MLA impl into model folder"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/ops/attention.py`, `vllm/models/deepseek_v4/amd/rocm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/flashmla.py` added +402/-0 (402 lines); hunks: -0,0 +1,402; symbols: DeepseekV4SparseMLAAttentionImpl, forward_mqa, DeepseekV4FlashMLASparseBackend, get_supported_kernel_block_sizes, touching `DeepseekV4SparseMLAAttentionImpl, forward_mqa, DeepseekV4FlashMLASparseBackend`; `vllm/models/deepseek_v4/nvidia/ops/attention.py` renamed +23/-309 (332 lines); hunks: -20,9 +20,6; -62,28 +59,36; symbols: _select_v4_sparse_impl, wq_b_kv_insert, __init__, get_attn_backend, touching `_select_v4_sparse_impl, wq_b_kv_insert, __init__`; `vllm/models/deepseek_v4/amd/rocm.py` renamed +47/-59 (106 lines); hunks: -8,14 +8,15; -31,7 +32,9; symbols: _build_indptr_from_lengths, build, DeepseekV4ROCMAiterMLASparseImpl, DeepseekV4ROCMAiterMLASparseBackend, touching `_build_indptr_from_lengths, build, DeepseekV4ROCMAiterMLASparseImpl`; `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-1 (2 lines); hunks: -56,7 +56,7.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/flashmla.py` added +402/-0 (402 lines); hunks: -0,0 +1,402; symbols: DeepseekV4SparseMLAAttentionImpl, forward_mqa, DeepseekV4FlashMLASparseBackend, get_supported_kernel_block_sizes
  - `vllm/models/deepseek_v4/nvidia/ops/attention.py` renamed +23/-309 (332 lines); hunks: -20,9 +20,6; -62,28 +59,36; symbols: _select_v4_sparse_impl, wq_b_kv_insert, __init__, get_attn_backend
  - `vllm/models/deepseek_v4/amd/rocm.py` renamed +47/-59 (106 lines); hunks: -8,14 +8,15; -31,7 +32,9; symbols: _build_indptr_from_lengths, build, DeepseekV4ROCMAiterMLASparseImpl, DeepseekV4ROCMAiterMLASparseBackend
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-1 (2 lines); hunks: -56,7 +56,7
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/flashmla.py
@@ -0,0 +1,402 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from abc import abstractmethod
+from typing import TYPE_CHECKING, ClassVar, cast
+import torch
+from vllm.forward_context import get_forward_context
diff -- vllm/models/deepseek_v4/nvidia/ops/attention.py
@@ -20,9 +20,6 @@
-    combine_topk_swa_indices,
-    compute_global_topk_indices_and_lens,
-    dequantize_and_gather_k_cache,
@@ -62,28 +59,36 @@
-    DeepseekV4FlashMLASparseBackend,
-    FlashMLASparseMetadata,
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -8,14 +8,15 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/flashmla.py` added +402/-0; `vllm/models/deepseek_v4/nvidia/ops/attention.py` renamed +23/-309; `vllm/models/deepseek_v4/amd/rocm.py` renamed +47/-59; `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`, `tests/kernels/test_fused_inv_rope_fp8_quant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42353 - DSv4 fused Q-norm kernel grid refactor

- Link: https://github.com/vllm-project/vllm/pull/42353
- Status/date: merged / 2026-05-22
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`; associated commits `f743254143f2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +330/-216, 670 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DSv4 fused Q-norm kernel grid refactor"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`; technical summary: Covers "DSv4 fused Q-norm kernel grid refactor"; the main implementation surface is `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +27/-24 (51 lines); hunks: -67,29 +67,26 @@ def apply_rope_gptj_last_k(; -99,11 +96,15 @@ def apply_rope_gptj_last_k(; symbols: apply_rope_gptj_last_k, rmsnorm_no_weight, _call_fused, test_q_path_matches_reference, touching `apply_rope_gptj_last_k, rmsnorm_no_weight, _call_fused`.
- Code diff details:
  - `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +27/-24 (51 lines); hunks: -67,29 +67,26 @@ def apply_rope_gptj_last_k(; -99,11 +96,15 @@ def apply_rope_gptj_last_k(; symbols: apply_rope_gptj_last_k, rmsnorm_no_weight, _call_fused, test_q_path_matches_reference
- Key code excerpts:

```diff
diff -- tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py
@@ -67,29 +67,26 @@ def apply_rope_gptj_last_k(
-    # Gather cos/sin for each token position: [num_tokens, rope_dim]
-    cs = cos_sin_cache[positions].to(torch.float32)  # [N, rope_dim]
-    cos = cs[..., :half]  # [N, half]
-    sin = cs[..., half:]  # [N, half]
-    # Reshape leading dims so we can broadcast: x shape [..., head_dim].
-    # Bring token dim to front; assume x is [num_tokens, ..., head_dim].
```

- Reviewed files:
  - tests: `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +27/-24
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42950 - [XPU]fix: add XPU platform guards to DeepSeek-V4 ops

- Link: https://github.com/vllm-project/vllm/pull/42950
- Status/date: merged / 2026-05-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `8de5cabeb70d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +31/-18, 133 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU]fix: add XPU platform guards to DeepSeek-V4 ops"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/compressor.py`; technical summary: Covers "[XPU]fix: add XPU platform guards to DeepSeek-V4 ops"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/compressor.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-5 (10 lines); hunks: -1153,7 +1153,7 @@ def _forward_cuda(; -1193,8 +1193,8 @@ def forward(; symbols: _forward_cuda, _forward_rocm, _forward_native, forward, touching `_forward_cuda, _forward_rocm, _forward_native`; `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +5/-1 (6 lines); hunks: -243,7 +243,11 @@ def _fused_inv_rope_fp8_quant_kernel_impl(; symbols: _fused_inv_rope_fp8_quant_kernel_impl, touching `_fused_inv_rope_fp8_quant_kernel_impl`; `vllm/models/deepseek_v4/compressor.py` modified +5/-1 (6 lines); hunks: -296,7 +296,11 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-5 (10 lines); hunks: -1153,7 +1153,7 @@ def _forward_cuda(; -1193,8 +1193,8 @@ def forward(; symbols: _forward_cuda, _forward_rocm, _forward_native, forward
  - `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +5/-1 (6 lines); hunks: -243,7 +243,11 @@ def _fused_inv_rope_fp8_quant_kernel_impl(; symbols: _fused_inv_rope_fp8_quant_kernel_impl
  - `vllm/models/deepseek_v4/compressor.py` modified +5/-1 (6 lines); hunks: -296,7 +296,11 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -1153,7 +1153,7 @@ def _forward_cuda(
-    def _forward_rocm(
+    def _forward_native(
@@ -1193,8 +1193,8 @@ def forward(
-        if current_platform.is_rocm():
-            return self._forward_rocm(
+        if current_platform.is_rocm() or current_platform.is_xpu():
diff -- vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py
@@ -243,7 +243,11 @@ def _fused_inv_rope_fp8_quant_kernel_impl(
-    pdl_kwargs = {} if current_platform.is_rocm() else {"launch_pdl": False}
+    pdl_kwargs = (
+        {}
+        if current_platform.is_rocm() or current_platform.is_xpu()
+        else {"launch_pdl": False}
+    )
diff -- vllm/models/deepseek_v4/compressor.py
@@ -296,7 +296,11 @@ def forward(
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-5; `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +5/-1; `vllm/models/deepseek_v4/compressor.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/activation.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/compressor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42925 - [DSV4] More multi-stream enablement for c4a

- Link: https://github.com/vllm-project/vllm/pull/42925
- Status/date: merged / 2026-05-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `367cb81966f9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +61/-29, 141 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] More multi-stream enablement for c4a"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[DSV4] More multi-stream enablement for c4a"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/model.py` modified +7/-0 (7 lines); hunks: -935,6 +935,12 @@ def __init__(; -945,6 +951,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +7/-0 (7 lines); hunks: -935,6 +935,12 @@ def __init__(; -945,6 +951,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -935,6 +935,12 @@ def __init__(
+            # aux_stream_list[0] runs indexer.forward() in the wrapper; [2] is
+            # free here (outer GEMMs joined) for the inner overlap of
+            # wq_b+fused_indexer_q_rope_quant vs compressor.
+            indexer_aux_stream = (
+                aux_stream_list[2] if aux_stream_list is not None else None
+            )
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/ops/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43385 - [ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP

- Link: https://github.com/vllm-project/vllm/pull/43385
- Status/date: merged / 2026-05-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/amd/rocm.py`; associated commits `1806d1adfc9b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +2340/-52, 2496 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/amd/rocm.py`; technical summary: Covers "[ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP"; the main implementation surface is `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/amd/rocm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/model.py` added +1612/-0 (1612 lines); hunks: -0,0 +1,1612; symbols: DeepseekV4MLP, __init__, forward, _deepseek_v4_stage_mega_moe_inputs_kernel, touching `DeepseekV4MLP, __init__, forward`; `vllm/models/deepseek_v4/amd/mtp.py` added +520/-0 (520 lines); hunks: -0,0 +1,520; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor, touching `DeepSeekV4MultiTokenPredictorLayer, __init__, forward`; `vllm/models/deepseek_v4/amd/rocm.py` modified +134/-23 (157 lines); hunks: -44,6 +44,127 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> tor...; -704,38 +825,28 @@ def _forward_prefill(; symbols: _build_indptr_from_lengths, _combine_topk_swa_indices_kernel, combine_topk_swa_indices, _compute_topk_lens_kernel, touching `_build_indptr_from_lengths, _combine_topk_swa_indices_kernel, combine_topk_swa_indices`; `vllm/models/deepseek_v4/amd/model.py` removed +0/-1 (1 lines); hunks: -1 +0,0.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/model.py` added +1612/-0 (1612 lines); hunks: -0,0 +1,1612; symbols: DeepseekV4MLP, __init__, forward, _deepseek_v4_stage_mega_moe_inputs_kernel
  - `vllm/models/deepseek_v4/amd/mtp.py` added +520/-0 (520 lines); hunks: -0,0 +1,520; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +134/-23 (157 lines); hunks: -44,6 +44,127 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> tor...; -704,38 +825,28 @@ def _forward_prefill(; symbols: _build_indptr_from_lengths, _combine_topk_swa_indices_kernel, combine_topk_swa_indices, _compute_topk_lens_kernel
  - `vllm/models/deepseek_v4/amd/model.py` removed +0/-1 (1 lines); hunks: -1 +0,0
  - `vllm/models/deepseek_v4/amd/mtp.py` removed +0/-1 (1 lines); hunks: -1 +0,0
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -0,0 +1,1612 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/models/deepseek_v4/amd/mtp.py
@@ -0,0 +1,520 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""MTP draft model for DeepSeek V4 (internal codename: DeepseekV4).
+Split from ``deepseek_mtp.py`` because the V4 architecture introduces several
+pieces that have no analogue in V3/V32:
+  * separate ``e_proj`` / ``h_proj`` with fp8 linear quantization (instead of
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -44,6 +44,127 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` added +1612/-0; `vllm/models/deepseek_v4/amd/mtp.py` added +520/-0; `vllm/models/deepseek_v4/amd/rocm.py` modified +134/-23; `vllm/models/deepseek_v4/amd/model.py` removed +0/-1; `vllm/models/deepseek_v4/amd/mtp.py` removed +0/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43632 - [DeepSeek V4] Move MegaMoE input prep kernel to nvidia/ops

- Link: https://github.com/vllm-project/vllm/pull/43632
- Status/date: merged / 2026-05-26
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`; associated commits `aa2b56ffb0c1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +177/-165, 382 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek V4] Move MegaMoE input prep kernel to nvidia/ops"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `tests/models/test_deepseek_v4_mega_moe.py`; technical summary: Covers "[DeepSeek V4] Move MegaMoE input prep kernel to nvidia/ops"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `tests/models/test_deepseek_v4_mega_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` added +173/-0 (173 lines); hunks: -0,0 +1,173; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs, touching `_prepare_megamoe_inputs_kernel, prepare_megamoe_inputs`; `vllm/models/deepseek_v4/nvidia/model.py` modified +2/-163 (165 lines); hunks: -59,9 +59,9; -116,167 +116,6 @@ def forward(self, x):; symbols: forward, _deepseek_v4_stage_mega_moe_inputs_kernel, _stage_deepseek_v4_mega_moe_inputs, make_deepseek_v4_expert_params_mapping, touching `forward, _deepseek_v4_stage_mega_moe_inputs_kernel, _stage_deepseek_v4_mega_moe_inputs`; `tests/models/test_deepseek_v4_mega_moe.py` modified +2/-2 (4 lines); hunks: -8,9 +8,9; -164,7 +164,7 @@ def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise...; symbols: test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, touching `test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` added +173/-0 (173 lines); hunks: -0,0 +1,173; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +2/-163 (165 lines); hunks: -59,9 +59,9; -116,167 +116,6 @@ def forward(self, x):; symbols: forward, _deepseek_v4_stage_mega_moe_inputs_kernel, _stage_deepseek_v4_mega_moe_inputs, make_deepseek_v4_expert_params_mapping
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +2/-2 (4 lines); hunks: -8,9 +8,9; -164,7 +164,7 @@ def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise...; symbols: test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py
@@ -0,0 +1,173 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Triton input-staging kernel for DeepSeek V4 MegaMoE.
+Quantizes hidden states to fp8 with E8M0 group scales and repacks the
+routing top-k tensors into the int64/float32 layout that the DeepGEMM
+MegaMoE kernels consume.
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -59,9 +59,9 @@
+from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
-from vllm.triton_utils import tl, triton
@@ -116,167 +116,6 @@ def forward(self, x):
-@triton.jit
-def _deepseek_v4_stage_mega_moe_inputs_kernel(
-    hidden_states,
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -8,9 +8,9 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` added +173/-0; `vllm/models/deepseek_v4/nvidia/model.py` modified +2/-163
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_mega_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43162 - [Feat][DSV4] Fuse q pad into deepseek v4 fused kernel

- Link: https://github.com/vllm-project/vllm/pull/43162
- Status/date: merged / 2026-05-26
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py` and 6 files; associated commits `6ab6ffb428be`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +339/-151, 888 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feat][DSV4] Fuse q pad into deepseek v4 fused kernel"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/amd/rocm.py`; technical summary: Covers "[Feat][DSV4] Fuse q pad into deepseek v4 fused kernel"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/amd/rocm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` renamed +23/-40 (63 lines); hunks: -156,18 +156,6 @@ def __init__(; -263,6 +251,9 @@ def __init__(; symbols: __init__, attention_impl, wq_b_kv_insert, touching `__init__, attention_impl, wq_b_kv_insert`; `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +23/-1 (24 lines); hunks: -28,7 +28,7; -63,6 +63,18 @@ def forward_mqa( # type: ignore[override]; symbols: forward_mqa, get_padded_num_q_heads, DeepseekV4FlashMLASparseBackend, DeepseekV4FlashMLASparseImpl, touching `forward_mqa, get_padded_num_q_heads, DeepseekV4FlashMLASparseBackend`; `vllm/models/deepseek_v4/amd/rocm.py` modified +5/-1 (6 lines); hunks: -32,7 +32,7; -592,6 +592,10 @@ class DeepseekV4ROCMAiterMLASparseImpl(DeepseekV4SparseMLAA...; symbols: DeepseekV4ROCMAiterMLASparseImpl, get_padded_num_q_heads, forward_mqa, touching `DeepseekV4ROCMAiterMLASparseImpl, get_padded_num_q_heads, forward_mqa`; `vllm/models/deepseek_v4/amd/model.py` modified +1/-1 (2 lines); hunks: -53,7 +53,7.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` renamed +23/-40 (63 lines); hunks: -156,18 +156,6 @@ def __init__(; -263,6 +251,9 @@ def __init__(; symbols: __init__, attention_impl, wq_b_kv_insert
  - `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +23/-1 (24 lines); hunks: -28,7 +28,7; -63,6 +63,18 @@ def forward_mqa( # type: ignore[override]; symbols: forward_mqa, get_padded_num_q_heads, DeepseekV4FlashMLASparseBackend, DeepseekV4FlashMLASparseImpl
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +5/-1 (6 lines); hunks: -32,7 +32,7; -592,6 +592,10 @@ class DeepseekV4ROCMAiterMLASparseImpl(DeepseekV4SparseMLAA...; symbols: DeepseekV4ROCMAiterMLASparseImpl, get_padded_num_q_heads, forward_mqa
  - `vllm/models/deepseek_v4/amd/model.py` modified +1/-1 (2 lines); hunks: -53,7 +53,7
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-1 (2 lines); hunks: -54,7 +54,7
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -156,18 +156,6 @@ def __init__(
-        # FlashMLA sparse kernel only supports 64 or 128 heads; pad up to the
-        # next supported size. Must match DeepseekV4MLAAttention.padded_heads.
-        if num_heads <= 64:
-            self.padded_heads = 64
-        elif num_heads <= 128:
-            self.padded_heads = 128
diff -- vllm/models/deepseek_v4/nvidia/flashmla.py
@@ -28,7 +28,7 @@
-    from vllm.models.deepseek_v4.nvidia.ops.attention import (
+    from vllm.models.deepseek_v4.attention import (
@@ -63,6 +63,18 @@ def forward_mqa(  # type: ignore[override]
+    @classmethod
+    @abstractmethod
+    def get_padded_num_q_heads(cls, num_heads: int) -> int:
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -32,7 +32,7 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` renamed +23/-40; `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +23/-1; `vllm/models/deepseek_v4/amd/rocm.py` modified +5/-1; `vllm/models/deepseek_v4/amd/model.py` modified +1/-1; `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-1
  - tests: `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +72/-17
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `tests/kernels/test_fused_inv_rope_fp8_quant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43629 - [ROCm] Remove MegaMoE integration in deepseek v4

- Link: https://github.com/vllm-project/vllm/pull/43629
- Status/date: merged / 2026-05-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`; associated commits `c8414a82712b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-645, 793 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Remove MegaMoE integration in deepseek v4"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`; technical summary: Covers "[ROCm] Remove MegaMoE integration in deepseek v4"; the main implementation surface is `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/model.py` modified +2/-623 (625 lines); hunks: -11,17 +11,12; -52,16 +47,13; symbols: DeepseekV4MLP, forward, _deepseek_v4_stage_mega_moe_inputs_kernel, _stage_deepseek_v4_mega_moe_inputs, touching `DeepseekV4MLP, forward, _deepseek_v4_stage_mega_moe_inputs_kernel`; `vllm/models/deepseek_v4/amd/mtp.py` modified +8/-22 (30 lines); hunks: -40,10 +40,7; -330,19 +327,13 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx, finalize_mega_moe_weights, _rewrite_spec_layer_name, touching `_find_mtp_layer_idx, finalize_mega_moe_weights, _rewrite_spec_layer_name`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/model.py` modified +2/-623 (625 lines); hunks: -11,17 +11,12; -52,16 +47,13; symbols: DeepseekV4MLP, forward, _deepseek_v4_stage_mega_moe_inputs_kernel, _stage_deepseek_v4_mega_moe_inputs
  - `vllm/models/deepseek_v4/amd/mtp.py` modified +8/-22 (30 lines); hunks: -40,10 +40,7; -330,19 +327,13 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx, finalize_mega_moe_weights, _rewrite_spec_layer_name
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -11,17 +11,12 @@
-    get_ep_group,
-from vllm.forward_context import get_forward_context
-from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
-    fused_topk_bias,
-)
@@ -52,16 +47,13 @@
diff -- vllm/models/deepseek_v4/amd/mtp.py
@@ -40,10 +40,7 @@
-from .model import (
-    DeepseekV4DecoderLayer,
-    make_deepseek_v4_expert_params_mapping,
-)
+from .model import DeepseekV4DecoderLayer
@@ -330,19 +327,13 @@ def _find_mtp_layer_idx(name: str) -> int:
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` modified +2/-623; `vllm/models/deepseek_v4/amd/mtp.py` modified +8/-22
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43690 - [DSv4] Drop _get_compressed_kv_buffer in DeepseekCompressor

- Link: https://github.com/vllm-project/vllm/pull/43690
- Status/date: merged / 2026-05-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/compressor.py`; associated commits `193ce8812eb4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-33, 59 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4] Drop _get_compressed_kv_buffer in DeepseekCompressor"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/models/deepseek_v4/compressor.py`; technical summary: Covers "[DSv4] Drop _get_compressed_kv_buffer in DeepseekCompressor"; the main implementation surface is `vllm/models/deepseek_v4/compressor.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/compressor.py` modified +5/-33 (38 lines); hunks: -173,33 +173,6 @@ def get_attn_backend(self) -> type[AttentionBackend]:; -276,11 +249,6 @@ def __init__(; symbols: get_attn_backend, DeepseekCompressor, _get_compressed_kv_buffer, __init__, touching `get_attn_backend, DeepseekCompressor, _get_compressed_kv_buffer`.
- Code diff details:
  - `vllm/models/deepseek_v4/compressor.py` modified +5/-33 (38 lines); hunks: -173,33 +173,6 @@ def get_attn_backend(self) -> type[AttentionBackend]:; -276,11 +249,6 @@ def __init__(; symbols: get_attn_backend, DeepseekCompressor, _get_compressed_kv_buffer, __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/compressor.py
@@ -173,33 +173,6 @@ def get_attn_backend(self) -> type[AttentionBackend]:
-    _compressed_kv_buffers: ClassVar[dict[tuple[str, int, int], torch.Tensor]] = {}
-    @classmethod
-    def _get_compressed_kv_buffer(
-        cls,
-        device: str,
-        max_num_tokens: int,
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/compressor.py` modified +5/-33
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/compressor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43710 - [DSv4] Refactor compressor & Fix ROCm compatibility

- Link: https://github.com/vllm-project/vllm/pull/43710
- Status/date: merged / 2026-05-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/ops/__init__.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/common/ops/save_partial_states.py` and 9 files; associated commits `adaa5e455ad8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +364/-239, 753 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4] Refactor compressor & Fix ROCm compatibility"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/common/ops/save_partial_states.py`; technical summary: Covers "[DSv4] Refactor compressor & Fix ROCm compatibility"; the main implementation surface is `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/common/ops/save_partial_states.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/compressor.py` modified +68/-198 (266 lines); hunks: -13,15 +13,13; -173,6 +171,16 @@ def get_attn_backend(self) -> type[AttentionBackend]:; symbols: get_attn_backend, DeepseekCompressor, __init__, forward, touching `get_attn_backend, DeepseekCompressor, __init__`; `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +78/-34 (112 lines); hunks: -11,12 +11,6; -25,43 +19,93; symbols: _get_sparse_attn_cutedsl_impls, compress_norm_rope_store_triton, _compress_kv_sparse_attn_cutedsl, _norm_rope_insert_sparse_attn_cutedsl, touching `_get_sparse_attn_cutedsl_impls, compress_norm_rope_store_triton, _compress_kv_sparse_attn_cutedsl`; `vllm/models/deepseek_v4/common/ops/save_partial_states.py` added +101/-0 (101 lines); hunks: -0,0 +1,101; symbols: save_partial_states, _save_partial_states_kernel, touching `save_partial_states, _save_partial_states_kernel`; `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` renamed +95/-3 (98 lines); hunks: -8,6 +8,7; -1086,7 +1087,7 @@ def compile(; symbols: compile, _compress_kv_sparse_attn_cutedsl, compress_kv_sparse_attn_cutedsl, _norm_rope_insert_sparse_attn_cutedsl, touching `compile, _compress_kv_sparse_attn_cutedsl, compress_kv_sparse_attn_cutedsl`.
- Code diff details:
  - `vllm/models/deepseek_v4/compressor.py` modified +68/-198 (266 lines); hunks: -13,15 +13,13; -173,6 +171,16 @@ def get_attn_backend(self) -> type[AttentionBackend]:; symbols: get_attn_backend, DeepseekCompressor, __init__, forward
  - `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +78/-34 (112 lines); hunks: -11,12 +11,6; -25,43 +19,93; symbols: _get_sparse_attn_cutedsl_impls, compress_norm_rope_store_triton, _compress_kv_sparse_attn_cutedsl, _norm_rope_insert_sparse_attn_cutedsl
  - `vllm/models/deepseek_v4/common/ops/save_partial_states.py` added +101/-0 (101 lines); hunks: -0,0 +1,101; symbols: save_partial_states, _save_partial_states_kernel
  - `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` renamed +95/-3 (98 lines); hunks: -8,6 +8,7; -1086,7 +1087,7 @@ def compile(; symbols: compile, _compress_kv_sparse_attn_cutedsl, compress_kv_sparse_attn_cutedsl, _norm_rope_insert_sparse_attn_cutedsl
  - `vllm/models/deepseek_v4/nvidia/ops/__init__.py` modified +16/-0 (16 lines); hunks: -6,3 +6,19
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/compressor.py
@@ -13,15 +13,13 @@
-    _compress_kv_sparse_attn_cutedsl,
-    _fused_kv_compress_norm_rope_insert_indexer_attn,
-    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
-    _fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl,
-    _norm_rope_insert_sparse_attn_cutedsl,
+    compress_norm_rope_store_triton,
diff -- vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
@@ -11,12 +11,6 @@
-Additional cutedsl kernels:
-  - _compress_kv_sparse_attn_cutedsl / _norm_rope_insert_sparse_attn_cutedsl:
-        CuTe DSL split kernels for C128
-  - _fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl:
-        CuTe DSL fused kernels for C4
@@ -25,43 +19,93 @@
diff -- vllm/models/deepseek_v4/common/ops/save_partial_states.py
@@ -0,0 +1,101 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/compressor.py` modified +68/-198; `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +78/-34; `vllm/models/deepseek_v4/common/ops/save_partial_states.py` added +101/-0; `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` renamed +95/-3; `vllm/models/deepseek_v4/nvidia/ops/__init__.py` modified +16/-0; `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/common/ops/__init__.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43679 - [ROCm][DSV4] Enable Tilelang MHC replacing torch/triton mhc

- Link: https://github.com/vllm-project/vllm/pull/43679
- Status/date: merged / 2026-05-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`; associated commits `0ba46d4b11d2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +716/-99, 1234 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][DSV4] Enable Tilelang MHC replacing torch/triton mhc"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`; technical summary: Covers "[ROCm][DSV4] Enable Tilelang MHC replacing torch/triton mhc"; the main implementation surface is `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/model.py` modified +11/-7 (18 lines); hunks: -54,6 +54,7; -473,6 +474,7 @@ def __init__(; symbols: DeepseekV4MLP, __init__, hc_pre, hc_post, touching `DeepseekV4MLP, __init__, hc_pre`; `vllm/models/deepseek_v4/amd/mtp.py` modified +3/-1 (4 lines); hunks: -39,6 +39,7; -118,6 +119,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/model.py` modified +11/-7 (18 lines); hunks: -54,6 +54,7; -473,6 +474,7 @@ def __init__(; symbols: DeepseekV4MLP, __init__, hc_pre, hc_post
  - `vllm/models/deepseek_v4/amd/mtp.py` modified +3/-1 (4 lines); hunks: -39,6 +39,7; -118,6 +119,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -54,6 +54,7 @@
+from vllm.utils.import_utils import has_tilelang
@@ -473,6 +474,7 @@ def __init__(
+        self.has_tilelang = has_tilelang()
@@ -503,7 +505,7 @@ def hc_post(
-    def _forward_cuda(
+    def _forward_fused_post_pre(
diff -- vllm/models/deepseek_v4/amd/mtp.py
@@ -39,6 +39,7 @@
+from vllm.utils.import_utils import has_tilelang
@@ -118,6 +119,7 @@ def __init__(
+        self.has_tilelang = has_tilelang()
@@ -144,7 +146,7 @@ def forward(
-        if current_platform.is_cuda():
+        if self.has_tilelang:
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` modified +11/-7; `vllm/models/deepseek_v4/amd/mtp.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `requirements/test/rocm.in`, `requirements/test/rocm.txt`, `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43829 - [DSV4] Remove AMD/XPU path in deepseek_v4/nvidia

- Link: https://github.com/vllm-project/vllm/pull/43829
- Status/date: merged / 2026-05-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; associated commits `a04afd76aa91`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +8/-78, 171 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Remove AMD/XPU path in deepseek_v4/nvidia"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; technical summary: Covers "[DSV4] Remove AMD/XPU path in deepseek_v4/nvidia"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/model.py` modified +3/-64 (67 lines); hunks: -60,7 +60,6; -262,13 +261,7 @@ def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:; symbols: _ue8m0_uint8_to_float, _check_runtime_supported, hc_post, _forward_cuda, touching `_ue8m0_uint8_to_float, _check_runtime_supported, hc_post`; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +5/-14 (19 lines); hunks: -37,7 +37,6; -147,10 +146,9 @@ def forward(; symbols: forward, __init__, touching `forward, __init__`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +3/-64 (67 lines); hunks: -60,7 +60,6; -262,13 +261,7 @@ def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:; symbols: _ue8m0_uint8_to_float, _check_runtime_supported, hc_post, _forward_cuda
  - `vllm/models/deepseek_v4/nvidia/mtp.py` modified +5/-14 (19 lines); hunks: -37,7 +37,6; -147,10 +146,9 @@ def forward(; symbols: forward, __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -60,7 +60,6 @@
-from vllm.platforms import current_platform
@@ -262,13 +261,7 @@ def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
-        if not torch.cuda.is_available():
-            raise NotImplementedError("DeepSeek V4 MegaMoE requires CUDA.")
-        if device.type != "cuda":
-            raise NotImplementedError(
diff -- vllm/models/deepseek_v4/nvidia/mtp.py
@@ -37,7 +37,6 @@
-from vllm.platforms import current_platform
@@ -147,10 +146,9 @@ def forward(
-        if current_platform.is_cuda():
-            hidden_states = self.mtp_block.hc_post(
-                hidden_states, residual, post_mix, res_mix
-            )
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +3/-64; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +5/-14
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43746 - [Model Refactoring] Remove torch compile dependency in DSv4

- Link: https://github.com/vllm-project/vllm/pull/43746
- Status/date: merged / 2026-05-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/common/ops/__init__.py`, `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py`, `vllm/models/deepseek_v4/nvidia/model.py` and 7 files; associated commits `04cec9e4d846`, `9957e4d240aa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +270/-24, 424 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Refactoring] Remove torch compile dependency in DSv4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; technical summary: Covers "[Model Refactoring] Remove torch compile dependency in DSv4"; the main implementation surface is `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: _rmsnorm_row, _fused_mtp_input_rmsnorm_kernel, _mtp_shared_head_rmsnorm_kernel, mtp_shared_head_rmsnorm, touching `_rmsnorm_row, _fused_mtp_input_rmsnorm_kernel, _mtp_shared_head_rmsnorm_kernel`; `vllm/models/deepseek_v4/amd/mtp.py` modified +21/-10 (31 lines); hunks: -18,7 +18,6; -37,6 +36,10; symbols: forward, compute_logits, DeepSeekV4MTP, __init__, touching `forward, compute_logits, DeepSeekV4MTP`; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +21/-10 (31 lines); hunks: -18,7 +18,6; -37,6 +36,10; symbols: forward, compute_logits, DeepSeekV4MTP, __init__, touching `forward, compute_logits, DeepSeekV4MTP`; `vllm/models/deepseek_v4/common/ops/__init__.py` modified +3/-0 (3 lines); hunks: -9,6 +9,7; -19,7 +20,9.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: _rmsnorm_row, _fused_mtp_input_rmsnorm_kernel, _mtp_shared_head_rmsnorm_kernel, mtp_shared_head_rmsnorm
  - `vllm/models/deepseek_v4/amd/mtp.py` modified +21/-10 (31 lines); hunks: -18,7 +18,6; -37,6 +36,10; symbols: forward, compute_logits, DeepSeekV4MTP, __init__
  - `vllm/models/deepseek_v4/nvidia/mtp.py` modified +21/-10 (31 lines); hunks: -18,7 +18,6; -37,6 +36,10; symbols: forward, compute_logits, DeepSeekV4MTP, __init__
  - `vllm/models/deepseek_v4/common/ops/__init__.py` modified +3/-0 (3 lines); hunks: -9,6 +9,7; -19,7 +20,9
  - `vllm/models/deepseek_v4/amd/model.py` modified +0/-2 (2 lines); hunks: -8,7 +8,6; -605,7 +604,6 @@ def forward(; symbols: forward, DeepseekV4Model, __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py
@@ -0,0 +1,203 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Fused MTP-input RMSNorm: enorm (with mask-zero at position 0) + hnorm.
+Replaces the eager sequence at the top of the MTP draft forward:
+    inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
+    inputs_embeds = self.enorm(inputs_embeds)
diff -- vllm/models/deepseek_v4/amd/mtp.py
@@ -18,7 +18,6 @@
-from vllm.compilation.decorators import support_torch_compile
@@ -37,6 +36,10 @@
+from vllm.models.deepseek_v4.common.ops import (
+    fused_mtp_input_rmsnorm,
+    mtp_shared_head_rmsnorm,
+)
diff -- vllm/models/deepseek_v4/nvidia/mtp.py
@@ -18,7 +18,6 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` added +203/-0; `vllm/models/deepseek_v4/amd/mtp.py` modified +21/-10; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +21/-10; `vllm/models/deepseek_v4/common/ops/__init__.py` modified +3/-0; `vllm/models/deepseek_v4/amd/model.py` modified +0/-2; `vllm/models/deepseek_v4/nvidia/model.py` modified +0/-2
- Risk and verification: Runtime changes concentrate in `vllm/config/vllm.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43891 - [Model Refactoring] Remove unncessary torch op registration for DSv4

- Link: https://github.com/vllm-project/vllm/pull/43891
- Status/date: merged / 2026-05-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `69b8956dcd5a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-110, 188 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Refactoring] Remove unncessary torch op registration for DSv4"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[Model Refactoring] Remove unncessary torch op registration for DSv4"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +9/-59 (68 lines); hunks: -25,7 +25,6; -292,8 +291,10 @@ def forward(; symbols: forward, deepseek_v4_attention, deepseek_v4_attention_fake, deepseek_v4_fp8_einsum, touching `forward, deepseek_v4_attention, deepseek_v4_attention_fake`; `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-51 (52 lines); hunks: -15,7 +15,6; -60,7 +59,6; symbols: DeepseekV4MLP, __init__, _map_global_expert_id, forward, touching `DeepseekV4MLP, __init__, _map_global_expert_id`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +9/-59 (68 lines); hunks: -25,7 +25,6; -292,8 +291,10 @@ def forward(; symbols: forward, deepseek_v4_attention, deepseek_v4_attention_fake, deepseek_v4_fp8_einsum
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-51 (52 lines); hunks: -15,7 +15,6; -60,7 +59,6; symbols: DeepseekV4MLP, __init__, _map_global_expert_id, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -25,7 +25,6 @@
-from vllm.utils.torch_utils import direct_register_custom_op
@@ -292,8 +291,10 @@ def forward(
-        # Attention (inside custom op for torch.compile boundary)
-        torch.ops.vllm.deepseek_v4_attention(
+        # @eager_break_during_capture: this is where the breakable
+        # cudagraph capture breaks (the attention op runs eagerly between
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -15,7 +15,6 @@
-from vllm.forward_context import get_forward_context
@@ -60,7 +59,6 @@
-from vllm.utils.torch_utils import direct_register_custom_op
@@ -209,13 +207,6 @@ def __init__(
-        # Register in the static forward context so the custom-op wrapper
-        # can look up this module by name from within a torch.compile graph.
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +9/-59; `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-51
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43905 - [DSv4] Move mHC tilelang kernels & Don't use CustomOP in dsv4/nvidia

- Link: https://github.com/vllm-project/vllm/pull/43905
- Status/date: merged / 2026-05-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; associated commits `7bd45da5857d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +72/-102, 380 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4] Move mHC tilelang kernels & Don't use CustomOP in dsv4/nvidia"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; technical summary: Covers "[DSv4] Move mHC tilelang kernels & Don't use CustomOP in dsv4/nvidia"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/model.py` modified +20/-54 (74 lines); hunks: -15,6 +15,12; -28,12 +34,6; symbols: __init__, hc_pre, hc_post, forward, touching `__init__, hc_pre, hc_post`; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +6/-7 (13 lines); hunks: -24,11 +24,14; -122,8 +125,6 @@ def __init__(; symbols: __init__, forward, compute_logits, touching `__init__, forward, compute_logits`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +20/-54 (74 lines); hunks: -15,6 +15,12; -28,12 +34,6; symbols: __init__, hc_pre, hc_post, forward
  - `vllm/models/deepseek_v4/nvidia/mtp.py` modified +6/-7 (13 lines); hunks: -24,11 +24,14; -122,8 +125,6 @@ def __init__(; symbols: __init__, forward, compute_logits
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -15,6 +15,12 @@
+from vllm.model_executor.kernels.mhc.tilelang import (
+    hc_head_fused_kernel_tilelang,
+    mhc_fused_post_pre_tilelang,
+    mhc_post_tilelang,
+    mhc_pre_tilelang,
+)
diff -- vllm/models/deepseek_v4/nvidia/mtp.py
@@ -24,11 +24,14 @@
+from vllm.model_executor.kernels.mhc.tilelang import (
+    hc_head_fused_kernel_tilelang,
+    mhc_post_tilelang,
+)
-from vllm.model_executor.layers.mhc import HCHeadOp
@@ -122,8 +125,6 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +20/-54; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +6/-7
- Risk and verification: The diff ships test coverage in `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #44161 - [Kernel][DSv4] Optimize sparse FP8 compressor kernels

- Link: https://github.com/vllm-project/vllm/pull/44161
- Status/date: merged / 2026-06-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`; associated commits `035733515f25`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +139/-91, 312 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel][DSv4] Optimize sparse FP8 compressor kernels"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`; technical summary: Covers "[Kernel][DSv4] Optimize sparse FP8 compressor kernels"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +139/-91 (230 lines); hunks: -96,9 +96,16 @@ def __init__(; -156,8 +163,9 @@ def kernel(; symbols: __init__, kernel, touching `__init__, kernel`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +139/-91 (230 lines); hunks: -96,9 +96,16 @@ def __init__(; -156,8 +163,9 @@ def kernel(; symbols: __init__, kernel
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py
@@ -96,9 +96,16 @@ def __init__(
-        self.num_warps = head_size // quant_block
+        self.elems_per_lane = 8
+        self.copy_elems = 4
+        self.copy_chunks = self.elems_per_lane // self.copy_elems
+        self.lanes_per_group = quant_block // self.elems_per_lane
+        self.groups_per_warp = 32 // self.lanes_per_group
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +139/-91
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44246 - [DSV4] Remove unncessary classes & functions

- Link: https://github.com/vllm-project/vllm/pull/44246
- Status/date: merged / 2026-06-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `8c3cc98cffd3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +62/-124, 362 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Remove unncessary classes & functions"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[DSV4] Remove unncessary classes & functions"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +34/-88 (122 lines); hunks: -5,7 +5,6; -38,9 +37,8; symbols: _select_v4_sparse_impl, DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, touching `_select_v4_sparse_impl, DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper`; `vllm/models/deepseek_v4/amd/model.py` modified +14/-18 (32 lines); hunks: -48,8 +48,7; -314,7 +313,7 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/models/deepseek_v4/nvidia/model.py` modified +14/-18 (32 lines); hunks: -54,8 +54,7; -697,7 +696,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +34/-88 (122 lines); hunks: -5,7 +5,6; -38,9 +37,8; symbols: _select_v4_sparse_impl, DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes
  - `vllm/models/deepseek_v4/amd/model.py` modified +14/-18 (32 lines); hunks: -48,8 +48,7; -314,7 +313,7 @@ def __init__(; symbols: __init__
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +14/-18 (32 lines); hunks: -54,8 +54,7; -697,7 +696,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -5,7 +5,6 @@
-from dataclasses import dataclass
@@ -38,9 +37,8 @@
-from vllm.forward_context import ForwardContext, get_forward_context
+from vllm.forward_context import get_forward_context
-from vllm.model_executor.custom_op import PluggableLayer
@@ -90,46 +88,7 @@ def _select_v4_sparse_impl() -> "type[DeepseekV4SparseMLAAttentionImpl]":
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -48,8 +48,7 @@
-    DeepseekV4MLAModules,
-    DeepseekV4MultiHeadLatentAttentionWrapper,
+    DeepseekV4MLA,
@@ -314,7 +313,7 @@ def __init__(
-        # Initialize rotary embedding BEFORE DeepseekV4MLAModules (which needs it)
+        # Initialize rotary embedding BEFORE DeepseekV4MLA (which needs it)
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -54,8 +54,7 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +34/-88; `vllm/models/deepseek_v4/amd/model.py` modified +14/-18; `vllm/models/deepseek_v4/nvidia/model.py` modified +14/-18
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44262 - [DSV4] Refactor RoPE initialization

- Link: https://github.com/vllm-project/vllm/pull/44262
- Status/date: merged / 2026-06-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/common/rope.py`, `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `517e74a9644f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +50/-40, 133 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Refactor RoPE initialization"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/models/deepseek_v4/common/rope.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[DSV4] Refactor RoPE initialization"; the main implementation surface is `vllm/models/deepseek_v4/common/rope.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/rope.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: build_deepseek_v4_rope, touching `build_deepseek_v4_rope`; `vllm/models/deepseek_v4/amd/model.py` modified +7/-20 (27 lines); hunks: -30,7 +30,6; -50,6 +49,7; symbols: __init__, touching `__init__`; `vllm/models/deepseek_v4/nvidia/model.py` modified +7/-20 (27 lines); hunks: -35,7 +35,6; -56,6 +55,7; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/rope.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: build_deepseek_v4_rope
  - `vllm/models/deepseek_v4/amd/model.py` modified +7/-20 (27 lines); hunks: -30,7 +30,6; -50,6 +49,7; symbols: __init__
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +7/-20 (27 lines); hunks: -35,7 +35,6; -56,6 +55,7; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/rope.py
@@ -0,0 +1,36 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DeepseekV4 rotary embedding initialization."""
+from vllm.model_executor.layers.rotary_embedding import get_rope
+from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
+def build_deepseek_v4_rope(
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -30,7 +30,6 @@
-from vllm.model_executor.layers.rotary_embedding import get_rope
@@ -50,6 +49,7 @@
+from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
@@ -314,25 +314,12 @@ def __init__(
-        rope_parameters = config.rope_parameters
-        rope_parameters["rope_theta"] = (
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -35,7 +35,6 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/rope.py` added +36/-0; `vllm/models/deepseek_v4/amd/model.py` modified +7/-20; `vllm/models/deepseek_v4/nvidia/model.py` modified +7/-20
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/common/rope.py`, `vllm/models/deepseek_v4/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43339 - [Feature] Support EPLB for DeepSeek v4 Mega Moe

- Link: https://github.com/vllm-project/vllm/pull/43339
- Status/date: merged / 2026-06-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `242709415287`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +232/-46, 449 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Support EPLB for DeepSeek v4 Mega Moe"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[Feature] Support EPLB for DeepSeek v4 Mega Moe"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/model.py` modified +211/-38 (249 lines); hunks: -1,7 +1,7; -15,6 +15,7; symbols: __init__, _map_global_expert_id, touching `__init__, _map_global_expert_id`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +211/-38 (249 lines); hunks: -1,7 +1,7; -15,6 +15,7; symbols: __init__, _map_global_expert_id
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -1,7 +1,7 @@
-from collections.abc import Callable, Iterable
+from collections.abc import Callable, Iterable, MutableSequence, Sequence
@@ -15,6 +15,7 @@
+from vllm.distributed.eplb.eplb_state import EplbLayerState
@@ -23,6 +24,9 @@
+from vllm.model_executor.layers.fused_moe.router.base_router import (
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +211/-38
- Risk and verification: Runtime changes concentrate in `vllm/distributed/eplb/eplb_utils.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/utils/deep_gemm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44367 - [DSV4] Minor cleanup for DeepseekV4MegaMoEExperts

- Link: https://github.com/vllm-project/vllm/pull/44367
- Status/date: merged / 2026-06-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `b254e0456c98`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-18, 34 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Minor cleanup for DeepseekV4MegaMoEExperts"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[DSV4] Minor cleanup for DeepseekV4MegaMoEExperts"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-18 (19 lines); hunks: -420,25 +420,7 @@ def forward(; -484,6 +466,7 @@ def _run_mega_moe(; symbols: forward, _run_mega_moe, touching `forward, _run_mega_moe`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-18 (19 lines); hunks: -420,25 +420,7 @@ def forward(; -484,6 +466,7 @@ def _run_mega_moe(; symbols: forward, _run_mega_moe
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -420,25 +420,7 @@ def forward(
-        self._run_mega_moe(
-            hidden_states,
-            topk_weights,
-            topk_ids,
-            y,
-            activation_clamp,
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-18
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44356 - [Bugfix] Fix Deepseek v4 non-mega-moe model init error

- Link: https://github.com/vllm-project/vllm/pull/44356
- Status/date: merged / 2026-06-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `969aec4bc845`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-0, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Deepseek v4 non-mega-moe model init error"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[Bugfix] Fix Deepseek v4 non-mega-moe model init error"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/model.py` modified +8/-0 (8 lines); hunks: -637,6 +637,14 @@ def _init_fused_moe_experts(; symbols: _init_fused_moe_experts, touching `_init_fused_moe_experts`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +8/-0 (8 lines); hunks: -637,6 +637,14 @@ def _init_fused_moe_experts(; symbols: _init_fused_moe_experts
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -637,6 +637,14 @@ def _init_fused_moe_experts(
+        self.n_redundant_experts = 0
+        self.n_shared_experts = config.n_shared_experts or 0
+        self.n_logical_experts = self.n_routed_experts
+        self.n_physical_experts = self.n_logical_experts
+        self.n_local_physical_experts = self.n_local_experts
+        self.physical_expert_start = self.experts_start_idx
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44236 - fix: resolve CUTLASS fmin compatibility for DeepSeek-V4 init

- Link: https://github.com/vllm-project/vllm/pull/44236
- Status/date: merged / 2026-06-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`; associated commits `597bc1593635`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-4, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: resolve CUTLASS fmin compatibility for DeepSeek-V4 init"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`; technical summary: Covers "fix: resolve CUTLASS fmin compatibility for DeepSeek-V4 init"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +4/-4 (8 lines); hunks: -370,11 +370,11 @@ def kernel(; -1026,11 +1026,11 @@ def kernel(; symbols: kernel, touching `kernel`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +4/-4 (8 lines); hunks: -370,11 +370,11 @@ def kernel(; -1026,11 +1026,11 @@ def kernel(; symbols: kernel
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py
@@ -370,11 +370,11 @@ def kernel(
-                    y0 = cute.arch.fmin(
+                    y0 = cutlass.min(
-                    y1 = cute.arch.fmin(
+                    y1 = cutlass.min(
@@ -1026,11 +1026,11 @@ def kernel(
-                y0 = cute.arch.fmin(
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +4/-4
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43827 - [DSv4] Adding TRTLLM gen attention kernel

- Link: https://github.com/vllm-project/vllm/pull/43827
- Status/date: merged / 2026-06-04
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/ops/__init__.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py` and 9 files; associated commits `b5235fca2eb7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +2971/-398, 4003 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4] Adding TRTLLM gen attention kernel"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`; technical summary: Covers "[DSv4] Adding TRTLLM gen attention kernel"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +1102/-323 (1425 lines); hunks: -508,28 +508,34 @@ def compile(; -539,17 +545,31 @@ def __call__(; symbols: compile, SparseAttnCompressC128Block8Kernel, SparseAttnCompressNormRopeStoreFullC4Kernel, __init__, touching `compile, SparseAttnCompressC128Block8Kernel, SparseAttnCompressNormRopeStoreFullC4Kernel`; `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` added +407/-0 (407 lines); hunks: -0,0 +1,407; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_name, get_impl_cls, touching `_get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_name`; `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +305/-0 (305 lines); hunks: -592,3 +592,308 @@ def _combine_topk_swa_indices_kernel(; symbols: _combine_topk_swa_indices_kernel, build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel, touching `_combine_topk_swa_indices_kernel, build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel`; `vllm/models/deepseek_v4/attention.py` modified +141/-42 (183 lines); hunks: -55,9 +55,6; -73,21 +70,82; symbols: _select_v4_sparse_impl, _resolve_dsv4_backend, _resolve_dsv4_kv_cache_dtype, DeepseekV4MLA, touching `_select_v4_sparse_impl, _resolve_dsv4_backend, _resolve_dsv4_kv_cache_dtype`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +1102/-323 (1425 lines); hunks: -508,28 +508,34 @@ def compile(; -539,17 +545,31 @@ def __call__(; symbols: compile, SparseAttnCompressC128Block8Kernel, SparseAttnCompressNormRopeStoreFullC4Kernel, __init__
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` added +407/-0 (407 lines); hunks: -0,0 +1,407; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_name, get_impl_cls
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +305/-0 (305 lines); hunks: -592,3 +592,308 @@ def _combine_topk_swa_indices_kernel(; symbols: _combine_topk_swa_indices_kernel, build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel
  - `vllm/models/deepseek_v4/attention.py` modified +141/-42 (183 lines); hunks: -55,9 +55,6; -73,21 +70,82; symbols: _select_v4_sparse_impl, _resolve_dsv4_backend, _resolve_dsv4_kv_cache_dtype, DeepseekV4MLA
  - `vllm/models/deepseek_v4/compressor.py` modified +38/-19 (57 lines); hunks: -155,13 +155,17 @@ def __init__(; -333,26 +337,40 @@ def forward(; symbols: __init__, get_kv_cache_spec, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py
@@ -508,28 +508,34 @@ def compile(
-class SparseAttnCompressC128Block8Kernel:
-    head_tile = 64
-    rows_per_warp = 16
-    elems_per_lane = 2
-    lanes_per_row = head_tile // elems_per_lane
-    num_warps = 8
diff -- vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
@@ -0,0 +1,407 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DeepSeek V4 FlashInfer TRTLLM-gen sparse MLA backend.
+Uses FlashInfer's public ``trtllm_batch_decode_sparse_mla_dsv4`` launcher with a
+contiguous bf16 / per-tensor FP8 KV cache. Shares the V4 sparse-index pipeline
+(SWA cache + compressor + indexer, 256-token blocks, head_size 512) with the
diff -- vllm/models/deepseek_v4/common/ops/cache_utils.py
@@ -592,3 +592,308 @@ def _combine_topk_swa_indices_kernel(
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +1102/-323; `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` added +407/-0; `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +305/-0; `vllm/models/deepseek_v4/attention.py` modified +141/-42; `vllm/models/deepseek_v4/compressor.py` modified +38/-19; `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +10/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #44569 - [DSV4] Refactor DeepseekV4Attention

- Link: https://github.com/vllm-project/vllm/pull/44569
- Status/date: merged / 2026-06-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py` and 7 files; associated commits `4efd6ffde094`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +521/-918, 2210 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Refactor DeepseekV4Attention"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[DSV4] Refactor DeepseekV4Attention"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +224/-345 (569 lines); hunks: -4,8 +4,9; -15,16 +16,16; symbols: _resolve_dsv4_backend, _select_v4_sparse_impl, _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention, touching `_resolve_dsv4_backend, _select_v4_sparse_impl, _resolve_dsv4_kv_cache_dtype`; `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +72/-118 (190 lines); hunks: -1,22 +1,22; -28,63 +28,9; symbols: DeepseekV4SparseMLAAttentionImpl, forward_mqa, get_padded_num_q_heads, init_layer_buffers, touching `DeepseekV4SparseMLAAttentionImpl, forward_mqa, get_padded_num_q_heads`; `vllm/models/deepseek_v4/nvidia/model.py` modified +17/-163 (180 lines); hunks: -33,7 +33,6; -55,13 +54,14; symbols: DeepseekV4MLP, finalize_mega_moe_weights, DeepseekV4Attention, __init__, touching `DeepseekV4MLP, finalize_mega_moe_weights, DeepseekV4Attention`; `vllm/models/deepseek_v4/amd/model.py` modified +3/-161 (164 lines); hunks: -18,7 +18,6; -45,11 +44,7; symbols: forward, DeepseekV4Attention, __init__, DeepseekV4DecoderLayer, touching `forward, DeepseekV4Attention, __init__`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +224/-345 (569 lines); hunks: -4,8 +4,9; -15,16 +16,16; symbols: _resolve_dsv4_backend, _select_v4_sparse_impl, _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention
  - `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +72/-118 (190 lines); hunks: -1,22 +1,22; -28,63 +28,9; symbols: DeepseekV4SparseMLAAttentionImpl, forward_mqa, get_padded_num_q_heads, init_layer_buffers
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +17/-163 (180 lines); hunks: -33,7 +33,6; -55,13 +54,14; symbols: DeepseekV4MLP, finalize_mega_moe_weights, DeepseekV4Attention, __init__
  - `vllm/models/deepseek_v4/amd/model.py` modified +3/-161 (164 lines); hunks: -18,7 +18,6; -45,11 +44,7; symbols: forward, DeepseekV4Attention, __init__, DeepseekV4DecoderLayer
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +65/-68 (133 lines); hunks: -2,15 +2,15; -26,16 +26,12; symbols: _build_indptr_from_lengths, get_name, get_builder_cls, get_impl_cls
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -4,8 +4,9 @@
+from abc import ABC, abstractmethod
-from typing import TYPE_CHECKING, Any, cast
+from typing import TYPE_CHECKING, Any, ClassVar, cast
@@ -15,16 +16,16 @@
+    ColumnParallelLinear,
+    MergedColumnParallelLinear,
diff -- vllm/models/deepseek_v4/nvidia/flashmla.py
@@ -1,22 +1,22 @@
-from abc import abstractmethod
-from typing import TYPE_CHECKING, ClassVar, cast
+from typing import TYPE_CHECKING, cast
+from vllm.models.deepseek_v4.attention import DeepseekV4Attention
-from vllm.v1.attention.backend import (
-    AttentionBackend,
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -33,7 +33,6 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +224/-345; `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +72/-118; `vllm/models/deepseek_v4/nvidia/model.py` modified +17/-163; `vllm/models/deepseek_v4/amd/model.py` modified +3/-161; `vllm/models/deepseek_v4/amd/rocm.py` modified +65/-68; `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +71/-62
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44561 - [DSV4] Move more ops out of eager breakpoint

- Link: https://github.com/vllm-project/vllm/pull/44561
- Status/date: merged / 2026-06-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`; associated commits `02d2da0748a1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +30/-14, 67 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Move more ops out of eager breakpoint"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[DSV4] Move more ops out of eager breakpoint"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +30/-14 (44 lines); hunks: -330,10 +330,34 @@ def forward(; -403,25 +427,17 @@ def fused_wqa_wkv() -> torch.Tensor:; symbols: forward, fused_wqa_wkv, attention_impl, touching `forward, fused_wqa_wkv, attention_impl`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +30/-14 (44 lines); hunks: -330,10 +330,34 @@ def forward(; -403,25 +427,17 @@ def fused_wqa_wkv() -> torch.Tensor:; symbols: forward, fused_wqa_wkv, attention_impl
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -330,10 +330,34 @@ def forward(
+        # Metadata-independent input GEMMs + RMSNorm stay in the captured
+        # graph; the metadata-dependent rest (q up-proj + kv-insert, indexer,
+        # compressor, MLA attention) runs in the eager break.
+        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
+            self.attn_gemm_parallel_execute(hidden_states)
+        )
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +30/-14
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44699 - [DSV4] Decouple DS V4 Sparse MLA Metadata from DS V3.2

- Link: https://github.com/vllm-project/vllm/pull/44699
- Status/date: merged / 2026-06-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/sparse_mla.py`; associated commits `2a983c79acdb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +449/-333, 984 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Decouple DS V4 Sparse MLA Metadata from DS V3.2"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/sparse_mla.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`; technical summary: Covers "[DSV4] Decouple DS V4 Sparse MLA Metadata from DS V3.2"; the main implementation surface is `vllm/models/deepseek_v4/sparse_mla.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/sparse_mla.py` added +416/-0 (416 lines); hunks: -0,0 +1,416; symbols: DeepseekV4FlashMLABackend, get_supported_kernel_block_sizes, get_name, get_builder_cls, touching `DeepseekV4FlashMLABackend, get_supported_kernel_block_sizes, get_name`; `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +7/-39 (46 lines); hunks: -16,10 +16,9; -31,41 +30,10; symbols: DeepseekV4FlashMLASparseBackend, get_supported_kernel_block_sizes, get_name, get_supported_head_sizes, touching `DeepseekV4FlashMLASparseBackend, get_supported_kernel_block_sizes, get_name`; `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +13/-10 (23 lines); hunks: -18,13 +18,15; -47,13 +49,14 @@ def _get_flashinfer_dsv4_workspace(device: torch.device) ->...; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, forward_mqa, _build_sparse_index_metadata, touching `_get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, forward_mqa`; `vllm/models/deepseek_v4/amd/rocm.py` modified +8/-10 (18 lines); hunks: -9,17 +9,15; -445,7 +443,7 @@ def _copy_ragged_to_graph_buffers(; symbols: _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata, DeepseekV4ROCMAiterSparseSWAMetadata, DeepseekV4ROCMAiterMLASparseMetadataBuilder, touching `_copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata, DeepseekV4ROCMAiterSparseSWAMetadata`.
- Code diff details:
  - `vllm/models/deepseek_v4/sparse_mla.py` added +416/-0 (416 lines); hunks: -0,0 +1,416; symbols: DeepseekV4FlashMLABackend, get_supported_kernel_block_sizes, get_name, get_builder_cls
  - `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +7/-39 (46 lines); hunks: -16,10 +16,9; -31,41 +30,10; symbols: DeepseekV4FlashMLASparseBackend, get_supported_kernel_block_sizes, get_name, get_supported_head_sizes
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +13/-10 (23 lines); hunks: -18,13 +18,15; -47,13 +49,14 @@ def _get_flashinfer_dsv4_workspace(device: torch.device) ->...; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, forward_mqa, _build_sparse_index_metadata
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +8/-10 (18 lines); hunks: -9,17 +9,15; -445,7 +443,7 @@ def _copy_ragged_to_graph_buffers(; symbols: _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata, DeepseekV4ROCMAiterSparseSWAMetadata, DeepseekV4ROCMAiterMLASparseMetadataBuilder
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/sparse_mla.py
@@ -0,0 +1,416 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DeepSeek-V4 FlashMLA sparse backend, metadata, and metadata builder."""
+from dataclasses import dataclass
+from typing import Any, ClassVar
+import numpy as np
diff -- vllm/models/deepseek_v4/nvidia/flashmla.py
@@ -16,10 +16,9 @@
-from vllm.v1.attention.backend import MultipleOf
-from vllm.v1.attention.backends.mla.flashmla_sparse import (
-    FlashMLASparseBackend,
-    FlashMLASparseMetadata,
+from vllm.models.deepseek_v4.sparse_mla import (
+    DeepseekV4FlashMLABackend,
diff -- vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
@@ -18,13 +18,15 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/sparse_mla.py` added +416/-0; `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +7/-39; `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +13/-10; `vllm/models/deepseek_v4/amd/rocm.py` modified +8/-10
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42953 - feat: add DeepSeek-V4 XPU attention decode path

- Link: https://github.com/vllm-project/vllm/pull/42953
- Status/date: merged / 2026-06-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/xpu/__init__.py`, `vllm/models/deepseek_v4/xpu/model.py`, `vllm/models/deepseek_v4/xpu/mtp.py` and 8 files; associated commits `eebce65756f0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +2759/-11, 2844 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: add DeepSeek-V4 XPU attention decode path"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/models/deepseek_v4/xpu/model.py`, `vllm/models/deepseek_v4/xpu/mtp.py`, `vllm/models/deepseek_v4/xpu/xpu_sparse.py`; technical summary: Covers "feat: add DeepSeek-V4 XPU attention decode path"; the main implementation surface is `vllm/models/deepseek_v4/xpu/model.py`, `vllm/models/deepseek_v4/xpu/mtp.py`, `vllm/models/deepseek_v4/xpu/xpu_sparse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/xpu/model.py` added +1340/-0 (1340 lines); hunks: -0,0 +1,1340; symbols: DeepseekV4MLP, __init__, forward, _deepseek_v4_stage_mega_moe_inputs_kernel, touching `DeepseekV4MLP, __init__, forward`; `vllm/models/deepseek_v4/xpu/mtp.py` added +511/-0 (511 lines); hunks: -0,0 +1,511; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor, touching `DeepSeekV4MultiTokenPredictorLayer, __init__, forward`; `vllm/models/deepseek_v4/xpu/xpu_sparse.py` added +350/-0 (350 lines); hunks: -0,0 +1,350; symbols: DeepseekV4XPUSparseBackend, get_name, DeepseekV4XPUAttention, __init__, touching `DeepseekV4XPUSparseBackend, get_name, DeepseekV4XPUAttention`; `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` added +290/-0 (290 lines); hunks: -0,0 +1,290; symbols: _dequant_gather_slots_kernel, dequant_gather_slots, xpu_sparse_decode_fp8, touching `_dequant_gather_slots_kernel, dequant_gather_slots, xpu_sparse_decode_fp8`.
- Code diff details:
  - `vllm/models/deepseek_v4/xpu/model.py` added +1340/-0 (1340 lines); hunks: -0,0 +1,1340; symbols: DeepseekV4MLP, __init__, forward, _deepseek_v4_stage_mega_moe_inputs_kernel
  - `vllm/models/deepseek_v4/xpu/mtp.py` added +511/-0 (511 lines); hunks: -0,0 +1,511; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor
  - `vllm/models/deepseek_v4/xpu/xpu_sparse.py` added +350/-0 (350 lines); hunks: -0,0 +1,350; symbols: DeepseekV4XPUSparseBackend, get_name, DeepseekV4XPUAttention, __init__
  - `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` added +290/-0 (290 lines); hunks: -0,0 +1,290; symbols: _dequant_gather_slots_kernel, dequant_gather_slots, xpu_sparse_decode_fp8
  - `vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py` added +159/-0 (159 lines); hunks: -0,0 +1,159; symbols: _xpu_qnorm_rope_kernel, xpu_qnorm_rope_kv_fp8_insert
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/xpu/model.py
@@ -0,0 +1,1340 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/models/deepseek_v4/xpu/mtp.py
@@ -0,0 +1,511 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""MTP draft model for DeepSeek V4 (internal codename: DeepseekV4).
+Split from ``deepseek_mtp.py`` because the V4 architecture introduces several
+pieces that have no analogue in V3/V32:
+  * separate ``e_proj`` / ``h_proj`` with fp8 linear quantization (instead of
diff -- vllm/models/deepseek_v4/xpu/xpu_sparse.py
@@ -0,0 +1,350 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/xpu/model.py` added +1340/-0; `vllm/models/deepseek_v4/xpu/mtp.py` added +511/-0; `vllm/models/deepseek_v4/xpu/xpu_sparse.py` added +350/-0; `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` added +290/-0; `vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py` added +159/-0; `vllm/models/deepseek_v4/__init__.py` modified +9/-8
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/kernels/linear/scaled_mm/xpu.py`, `vllm/model_executor/layers/mhc.py`, `vllm/models/deepseek_v4/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44144 - [DSV4][XPU] Add MHC fused_post_pre support

- Link: https://github.com/vllm-project/vllm/pull/44144
- Status/date: merged / 2026-06-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/xpu/model.py`; associated commits `70db1488c5d5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +112/-17, 168 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4][XPU] Add MHC fused_post_pre support"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/xpu/model.py`; technical summary: Covers "[DSV4][XPU] Add MHC fused_post_pre support"; the main implementation surface is `vllm/models/deepseek_v4/xpu/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/xpu/model.py` modified +40/-14 (54 lines); hunks: -930,22 +930,48 @@ def forward(; -1096,6 +1122,10 @@ def forward(; symbols: forward, load_weights, touching `forward, load_weights`.
- Code diff details:
  - `vllm/models/deepseek_v4/xpu/model.py` modified +40/-14 (54 lines); hunks: -930,22 +930,48 @@ def forward(; -1096,6 +1122,10 @@ def forward(; symbols: forward, load_weights
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/xpu/model.py
@@ -930,22 +930,48 @@ def forward(
-        residual = x
-        x, post, comb = self.hc_pre(
-            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
-        )
+        if residual is None:
+            # First layer: run standalone hc_pre
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/xpu/model.py` modified +40/-14
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mhc.py`, `vllm/models/deepseek_v4/xpu/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44914 - [Bug] Fix deepseek v4 OOM issue

- Link: https://github.com/vllm-project/vllm/pull/44914
- Status/date: merged / 2026-06-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/quant_config.py`; associated commits `d7607ad2730f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-3, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix deepseek v4 OOM issue"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/quant_config.py`; technical summary: Covers "[Bug] Fix deepseek v4 OOM issue"; the main implementation surface is `vllm/models/deepseek_v4/quant_config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/quant_config.py` modified +10/-3 (13 lines); hunks: -7,7 +7,11; -129,7 +133,7 @@ def override_quantization_method(; symbols: override_quantization_method, get_quant_method, is_mxfp4_quant, touching `override_quantization_method, get_quant_method, is_mxfp4_quant`.
- Code diff details:
  - `vllm/models/deepseek_v4/quant_config.py` modified +10/-3 (13 lines); hunks: -7,7 +7,11; -129,7 +133,7 @@ def override_quantization_method(; symbols: override_quantization_method, get_quant_method, is_mxfp4_quant
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/quant_config.py
@@ -7,7 +7,11 @@
-from vllm.model_executor.layers.fused_moe import MoERunner, UnquantizedFusedMoEMethod
+from vllm.model_executor.layers.fused_moe import (
+    MoERunner,
+    RoutedExperts,
+    UnquantizedFusedMoEMethod,
+)
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/quant_config.py` modified +10/-3
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/quant_config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44821 - fix: prefix DeepSeek V4 MTP projections

- Link: https://github.com/vllm-project/vllm/pull/44821
- Status/date: merged / 2026-06-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`, `vllm/models/deepseek_v4/xpu/mtp.py`; associated commits `04cec9e4d846`, `4673ca1d7869`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-0, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: prefix DeepSeek V4 MTP projections"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; technical summary: Covers "fix: prefix DeepSeek V4 MTP projections"; the main implementation surface is `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-0 (2 lines); hunks: -86,13 +86,15 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +2/-0 (2 lines); hunks: -92,13 +92,15 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-0 (2 lines); hunks: -86,13 +86,15 @@ def __init__(; symbols: __init__
  - `vllm/models/deepseek_v4/nvidia/mtp.py` modified +2/-0 (2 lines); hunks: -92,13 +92,15 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/mtp.py
@@ -86,13 +86,15 @@ def __init__(
+            prefix=f"{prefix}.e_proj",
+            prefix=f"{prefix}.h_proj",
diff -- vllm/models/deepseek_v4/nvidia/mtp.py
@@ -92,13 +92,15 @@ def __init__(
+            prefix=f"{prefix}.e_proj",
+            prefix=f"{prefix}.h_proj",
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-0; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45240 - [XPU][DeepSeek-V4] Fix MTP: sync with upstream fixes #44821 and #43746

- Link: https://github.com/vllm-project/vllm/pull/45240
- Status/date: merged / 2026-06-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/xpu/mtp.py`; associated commits `04cec9e4d846`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +29/-18, 119 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU][DeepSeek-V4] Fix MTP: sync with upstream fixes #44821 and #43746"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/xpu/mtp.py`; technical summary: Covers "[XPU][DeepSeek-V4] Fix MTP: sync with upstream fixes #44821 and #43746"; the main implementation surface is `vllm/models/deepseek_v4/xpu/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/xpu/mtp.py` modified +29/-18 (47 lines); hunks: -18,7 +18,6; -39,6 +38,10; symbols: __init__, forward, compute_logits, DeepSeekV4MTP, touching `__init__, forward, compute_logits`.
- Code diff details:
  - `vllm/models/deepseek_v4/xpu/mtp.py` modified +29/-18 (47 lines); hunks: -18,7 +18,6; -39,6 +38,10; symbols: __init__, forward, compute_logits, DeepSeekV4MTP
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/xpu/mtp.py
@@ -18,7 +18,6 @@
-from vllm.compilation.decorators import support_torch_compile
@@ -39,6 +38,10 @@
+from vllm.models.deepseek_v4.common.ops import (
+    fused_mtp_input_rmsnorm,
+    mtp_shared_head_rmsnorm,
+)
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/xpu/mtp.py` modified +29/-18
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/xpu/mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45061 - [Perf] Optimize DSv4 prefill chunk planning, 4.0% E2E Throughput Improvement

- Link: https://github.com/vllm-project/vllm/pull/45061
- Status/date: merged / 2026-06-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/flashmla.py`; associated commits `e18fe932ca61`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +133/-23, 272 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Optimize DSv4 prefill chunk planning, 4.0% E2E Throughput Improvement"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/nvidia/flashmla.py`; technical summary: Covers "[Perf] Optimize DSv4 prefill chunk planning, 4.0% E2E Throughput Improvement"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/flashmla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +12/-20 (32 lines); hunks: -246,7 +246,6 @@ def _forward_prefill(; -274,29 +273,22 @@ def _forward_prefill(; symbols: _forward_prefill, touching `_forward_prefill`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +12/-20 (32 lines); hunks: -246,7 +246,6 @@ def _forward_prefill(; -274,29 +273,22 @@ def _forward_prefill(; symbols: _forward_prefill
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/flashmla.py
@@ -246,7 +246,6 @@ def _forward_prefill(
-        num_prefills = swa_metadata.num_prefills
@@ -274,29 +273,22 @@ def _forward_prefill(
-            # Compressed region must fit the full compressed pool (seq_len //
-            # compress_ratio), not just top_k. top_k bounds how many indices
-            # the indexer selects, not the pool size it indexes into.
-            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +12/-20
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_flashmla_sparse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #44892 - [DSV4][Minor] Fix supported KV cache dtypes

- Link: https://github.com/vllm-project/vllm/pull/44892
- Status/date: merged / 2026-06-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/sparse_mla.py`; associated commits `f4359a70f9e0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +8/-8, 44 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4][Minor] Fix supported KV cache dtypes"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/sparse_mla.py`; technical summary: Covers "[DSV4][Minor] Fix supported KV cache dtypes"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/sparse_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +6/-5 (11 lines); hunks: -13,6 +13,7; -52,13 +53,13 @@ def _get_flashinfer_dsv4_workspace(device: torch.device) ->...; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_name, touching `_get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_name`; `vllm/models/deepseek_v4/sparse_mla.py` modified +0/-1 (1 lines); hunks: -46,7 +46,6 @@ class DeepseekV4FlashMLABackend(AttentionBackend):; symbols: DeepseekV4FlashMLABackend, touching `DeepseekV4FlashMLABackend`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +6/-5 (11 lines); hunks: -13,6 +13,7; -52,13 +53,13 @@ def _get_flashinfer_dsv4_workspace(device: torch.device) ->...; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_name
  - `vllm/models/deepseek_v4/sparse_mla.py` modified +0/-1 (1 lines); hunks: -46,7 +46,6 @@ class DeepseekV4FlashMLABackend(AttentionBackend):; symbols: DeepseekV4FlashMLABackend
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
@@ -13,6 +13,7 @@
+from vllm.config.cache import CacheDType
@@ -52,13 +53,13 @@ def _get_flashinfer_dsv4_workspace(device: torch.device) -> torch.Tensor:
-    Inheriting from the FlashMLA V4 backend reuses its
-    ``DeepseekV4FlashMLAMetadata`` builder (which the V4 sparse-index
-    pipeline needs — the V3.2 FlashInfer builder lacks the ``c128a_*`` fields),
-    256-token blocks, head_size 512, and the (num_blocks, block_size, 512) cache
diff -- vllm/models/deepseek_v4/sparse_mla.py
@@ -46,7 +46,6 @@ class DeepseekV4FlashMLABackend(AttentionBackend):
-        "bfloat16",
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +6/-5; `vllm/models/deepseek_v4/sparse_mla.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/sparse_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45863 - [DSv4 Perf] DSv4 flashinfer sparse index cache for metadata, 2%~4% TTFT improvement

- Link: https://github.com/vllm-project/vllm/pull/45863
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`; associated commits `0a7bacdcacc5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +184/-18, 227 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4 Perf] DSv4 flashinfer sparse index cache for metadata, 2%~4% TTFT improvement"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`; technical summary: Covers "[DSv4 Perf] DSv4 flashinfer sparse index cache for metadata, 2%~4% TTFT improvement"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +33/-17 (50 lines); hunks: -288,24 +288,40 @@ def _build_sparse_index_metadata(; symbols: _build_sparse_index_metadata, _forward, touching `_build_sparse_index_metadata, _forward`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +33/-17 (50 lines); hunks: -288,24 +288,40 @@ def _build_sparse_index_metadata(; symbols: _build_sparse_index_metadata, _forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
@@ -288,24 +288,40 @@ def _build_sparse_index_metadata(
-        sparse_indices, sparse_topk_lens = build_flashinfer_mixed_sparse_indices(
-            decode_swa_indices,
-            decode_compressed_indices,
-            decode_compressed_topk_lens,
-            prefill_topk_indices[:num_prefill_tokens],
-            query_start_loc,
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +33/-17
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_flashmla_sparse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45309 - [DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`, 26.8% ~ 27.9% E2E TTFT improvement

- Link: https://github.com/vllm-project/vllm/pull/45309
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`; associated commits `1797576237cc`, `2a47a9ff0f4f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +37/-30, 104 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`, 26.8% ~ 27.9% E2E TTFT improvement"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`, 26.8% ~ 27.9% E2E TTFT improvement"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +37/-30 (67 lines); hunks: -14,7 +14,7; -331,8 +331,8 @@ def forward(; symbols: forward, fused_wqa_wkv, attention_impl, touching `forward, fused_wqa_wkv, attention_impl`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +37/-30 (67 lines); hunks: -14,7 +14,7; -331,8 +331,8 @@ def forward(; symbols: forward, fused_wqa_wkv, attention_impl
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -14,7 +14,7 @@
-from vllm.compilation.breakable_cudagraph import eager_break_during_capture
+from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
@@ -331,8 +331,8 @@ def forward(
-        # graph; the metadata-dependent rest (q up-proj + kv-insert, indexer,
-        # compressor, MLA attention) runs in the eager break.
+        # graph. For C4A layers, the inner sparse_attn_indexer custom op
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +37/-30
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45972 - Revert "[DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`" (#45309)

- Link: https://github.com/vllm-project/vllm/pull/45972
- Status/date: merged / 2026-06-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`; associated commits `1797576237cc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +30/-37, 104 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`" (#45309)"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "Revert "[DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`" (#45309)"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +30/-37 (67 lines); hunks: -14,7 +14,7; -331,8 +331,8 @@ def forward(; symbols: forward, fused_wqa_wkv, attention_impl, touching `forward, fused_wqa_wkv, attention_impl`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +30/-37 (67 lines); hunks: -14,7 +14,7; -331,8 +331,8 @@ def forward(; symbols: forward, fused_wqa_wkv, attention_impl
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -14,7 +14,7 @@
-from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
+from vllm.compilation.breakable_cudagraph import eager_break_during_capture
@@ -331,8 +331,8 @@ def forward(
-        # graph. For C4A layers, the inner sparse_attn_indexer custom op
-        # runs in the eager break.
+        # graph; the metadata-dependent rest (q up-proj + kv-insert, indexer,
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +30/-37
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45681 - [ROCm][DSv4] Functional fixes for DeepSeek V4 on MI300X/MI325X

- Link: https://github.com/vllm-project/vllm/pull/45681
- Status/date: merged / 2026-06-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`; associated commits `afdcbd5d39ea`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +545/-52, 953 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][DSv4] Functional fixes for DeepSeek V4 on MI300X/MI325X"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`; technical summary: Covers "[ROCm][DSv4] Functional fixes for DeepSeek V4 on MI300X/MI325X"; the main implementation surface is `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +49/-6 (55 lines); hunks: -16,6 +16,10; -39,6 +43,7 @@ def quantize_and_insert_k_kernel(; symbols: quantize_and_insert_k_kernel, quantize_and_insert_k_cache, touching `quantize_and_insert_k_kernel, quantize_and_insert_k_cache`; `vllm/models/deepseek_v4/amd/rocm.py` modified +4/-0 (4 lines); hunks: -14,6 +14,7; -796,6 +797,7 @@ def _forward_prefill(; symbols: _forward_prefill, touching `_forward_prefill`; `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +3/-1 (4 lines); hunks: -3,7 +3,9; `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +147/-24 (171 lines); hunks: -19,17 +19,28; -81,10 +92,11 @@ def apply_rope_gptj_last_k(; symbols: apply_rope_gptj_last_k, _call_fused, _bf16_ulp_distance, key, touching `apply_rope_gptj_last_k, _call_fused, _bf16_ulp_distance`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +49/-6 (55 lines); hunks: -16,6 +16,10; -39,6 +43,7 @@ def quantize_and_insert_k_kernel(; symbols: quantize_and_insert_k_kernel, quantize_and_insert_k_cache
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +4/-0 (4 lines); hunks: -14,6 +14,7; -796,6 +797,7 @@ def _forward_prefill(; symbols: _forward_prefill
  - `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +3/-1 (4 lines); hunks: -3,7 +3,9
  - `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +147/-24 (171 lines); hunks: -19,17 +19,28; -81,10 +92,11 @@ def apply_rope_gptj_last_k(; symbols: apply_rope_gptj_last_k, _call_fused, _bf16_ulp_distance, key
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/cache_utils.py
@@ -16,6 +16,10 @@
+from vllm.model_executor.layers.quantization.utils.quant_utils import (
+    get_fp8_min_max,
+)
+from vllm.platforms import current_platform
@@ -39,6 +43,7 @@ def quantize_and_insert_k_kernel(
+    use_fnuz: tl.constexpr = False,
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -14,6 +14,7 @@
+from vllm.platforms import current_platform
@@ -796,6 +797,7 @@ def _forward_prefill(
+                # compressed_k_cache is OCP on every platform (Triton encoder).
@@ -804,6 +806,7 @@ def _forward_prefill(
+                    use_fnuz=False,
@@ -815,6 +818,7 @@ def _forward_prefill(
diff -- vllm/models/deepseek_v4/nvidia/ops/o_proj.py
@@ -3,7 +3,9 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +49/-6; `vllm/models/deepseek_v4/amd/rocm.py` modified +4/-0; `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +3/-1
  - tests: `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +147/-24
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #46001 - [DeepSeek-V4] Support TEP=16 for the block-FP8 shared expert

- Link: https://github.com/vllm-project/vllm/pull/46001
- Status/date: merged / 2026-06-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `2a6c6b94293e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +44/-0, 80 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek-V4] Support TEP=16 for the block-FP8 shared expert"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[DeepSeek-V4] Support TEP=16 for the block-FP8 shared expert"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/model.py` modified +44/-0 (44 lines); hunks: -64,6 +64,7; -85,6 +86,15 @@ def __init__(; symbols: __init__, load_weights, _pad_shared_expert_weight, touching `__init__, load_weights, _pad_shared_expert_weight`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +44/-0 (44 lines); hunks: -64,6 +64,7; -85,6 +86,15 @@ def __init__(; symbols: __init__, load_weights, _pad_shared_expert_weight
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -64,6 +64,7 @@
+from vllm.utils.math_utils import cdiv
@@ -85,6 +86,15 @@ def __init__(
+        #
+        # Block-FP8 shards in whole 128-blocks; cdiv rounds the per-rank block
+        # count up so the linear's even TP split stays block-aligned, with the
+        # trailing ranks zero-filled by load_weights.
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +44/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45931 - [ROCm][DSV4] Disable TileLang MHC dispatch on gfx942

- Link: https://github.com/vllm-project/vllm/pull/45931
- Status/date: merged / 2026-06-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`; associated commits `89accad2cc96`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +56/-22, 207 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][DSV4] Disable TileLang MHC dispatch on gfx942"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`; technical summary: Covers "[ROCm][DSV4] Disable TileLang MHC dispatch on gfx942"; the main implementation surface is `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/model.py` modified +3/-3 (6 lines); hunks: -27,6 +27,7; -51,7 +52,6; symbols: DeepseekV4MLP, __init__, hc_pre, touching `DeepseekV4MLP, __init__, hc_pre`; `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-3 (5 lines); hunks: -28,7 +28,7; -42,7 +42,6; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/model.py` modified +3/-3 (6 lines); hunks: -27,6 +27,7; -51,7 +52,6; symbols: DeepseekV4MLP, __init__, hc_pre
  - `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-3 (5 lines); hunks: -28,7 +28,7; -42,7 +42,6; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -27,6 +27,7 @@
+    HAS_TILELANG_MHC,
@@ -51,7 +52,6 @@
-from vllm.utils.import_utils import has_tilelang
@@ -303,7 +303,7 @@ def __init__(
-        self.has_tilelang = has_tilelang()
+        self.has_tilelang = HAS_TILELANG_MHC
diff -- vllm/models/deepseek_v4/amd/mtp.py
@@ -28,7 +28,7 @@
-from vllm.model_executor.layers.mhc import HCHeadOp
+from vllm.model_executor.layers.mhc import HAS_TILELANG_MHC, HCHeadOp
@@ -42,7 +42,6 @@
-from vllm.utils.import_utils import has_tilelang
@@ -124,7 +123,7 @@ def __init__(
-        self.has_tilelang = has_tilelang()
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` modified +3/-3; `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43477 - Enable DeepSeek V4 and GLM-5.1 on SM120

- Link: https://github.com/vllm-project/vllm/pull/43477
- Status/date: merged / 2026-06-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` and 7 files; associated commits `44d95069e9d6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 37 files, +2340/-469, 3895 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable DeepSeek V4 and GLM-5.1 on SM120"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py`, `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "Enable DeepSeek V4 and GLM-5.1 on SM120"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py`, `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +473/-27 (500 lines); hunks: -1,13 +1,6; -18,6 +11,7; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_supported_kernel_block_sizes, get_name, touching `_get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_supported_kernel_block_sizes`; `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py` added +226/-0 (226 lines); hunks: -0,0 +1,226; symbols: _compute_mhc_pre_num_split, _normalize_token_sizes, _select_mhc_warmup_token_sizes, _find_first_mhc_layer, touching `_compute_mhc_pre_num_split, _normalize_token_sizes, _select_mhc_warmup_token_sizes`; `vllm/models/deepseek_v4/attention.py` modified +34/-32 (66 lines); hunks: -62,23 +62,22; -100,18 +99,20 @@ class DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC):; symbols: _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention, dispatches, _o_proj, touching `_resolve_dsv4_kv_cache_dtype, DeepseekV4Attention, dispatches`; `vllm/model_executor/layers/attention/mla_attention.py` modified +29/-14 (43 lines); hunks: -208,6 +208,7; -319,6 +320,22 @@ def _detect_output_quant_key(; symbols: _detect_output_quant_key, _canonicalize_sparse_mla_kv_cache_dtype, MLAAttention, __init__, touching `_detect_output_quant_key, _canonicalize_sparse_mla_kv_cache_dtype, MLAAttention`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +473/-27 (500 lines); hunks: -1,13 +1,6; -18,6 +11,7; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_supported_kernel_block_sizes, get_name
  - `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py` added +226/-0 (226 lines); hunks: -0,0 +1,226; symbols: _compute_mhc_pre_num_split, _normalize_token_sizes, _select_mhc_warmup_token_sizes, _find_first_mhc_layer
  - `vllm/models/deepseek_v4/attention.py` modified +34/-32 (66 lines); hunks: -62,23 +62,22; -100,18 +99,20 @@ class DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC):; symbols: _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention, dispatches, _o_proj
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +29/-14 (43 lines); hunks: -208,6 +208,7; -319,6 +320,22 @@ def _detect_output_quant_key(; symbols: _detect_output_quant_key, _canonicalize_sparse_mla_kv_cache_dtype, MLAAttention, __init__
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +27/-5 (32 lines); hunks: -60,9 +60,11; -736,14 +738,34 @@ def finalize_mega_moe_weights(self) -> None:; symbols: finalize_mega_moe_weights, _select_dsv4_attn_cls, for
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
@@ -1,13 +1,6 @@
-"""DeepSeek V4 FlashInfer TRTLLM-gen sparse MLA backend.
-Uses FlashInfer's public ``trtllm_batch_decode_sparse_mla_dsv4`` launcher with a
-plain bf16 / per-tensor FP8 KV row (vs FlashMLA's packed ``fp8_ds_mla`` block
-format). Shares the V4 sparse-index pipeline (SWA cache + compressor + indexer,
-256-token blocks, head_size 512) with the FlashMLA V4 backend; only the
-attention forward differs.
diff -- vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py
@@ -0,0 +1,226 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.
+Ported from lucifer1004/vllm-jasl with the two env-var knobs removed
+(`VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP`, `VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES`).
+Gating is intrinsic: non-DSv4 models and layers without hc_* attributes
diff -- vllm/models/deepseek_v4/attention.py
@@ -62,23 +62,22 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +473/-27; `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py` added +226/-0; `vllm/models/deepseek_v4/attention.py` modified +34/-32; `vllm/model_executor/layers/attention/mla_attention.py` modified +29/-14; `vllm/models/deepseek_v4/nvidia/model.py` modified +27/-5; `vllm/models/deepseek_v4/compressor.py` modified +9/-9
- Risk and verification: The diff ships test coverage in `tests/model_executor/test_flashinfer_autotune_cache.py`, `tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py`, `tests/v1/attention/test_sparse_mla_backends.py`, `tests/v1/spec_decode/test_acceptance_length.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #46428 - [Optimization] Skip DP padding tokens in MoE

- Link: https://github.com/vllm-project/vllm/pull/46428
- Status/date: merged / 2026-06-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +154/-1, 395 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Optimization] Skip DP padding tokens in MoE"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/model_executor/layers/fused_moe/modular_kernel.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`; technical summary: Covers "[Optimization] Skip DP padding tokens in MoE"; the main implementation surface is `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/model_executor/layers/fused_moe/modular_kernel.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/test_deepseek_v4_mega_moe.py` modified +80/-1 (81 lines); hunks: -46,7 +46,8 @@ def test_deepseek_v4_mega_moe_ue8m0_uint8_to_float():; -182,3 +183,81 @@ def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwis...; symbols: test_deepseek_v4_mega_moe_ue8m0_uint8_to_float, test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, test_deepseek_v4_mega_moe_fused_input_staging_masks_padding, touching `test_deepseek_v4_mega_moe_ue8m0_uint8_to_float, test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact`; `vllm/model_executor/layers/fused_moe/modular_kernel.py` modified +17/-0 (17 lines); hunks: -10,6 +10,7; -1133,6 +1134,22 @@ def _prepare(; symbols: _prepare, touching `_prepare`; `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +11/-0 (11 lines); hunks: -19,6 +19,7 @@ def _prepare_megamoe_inputs_kernel(; -31,6 +32,7 @@ def _prepare_megamoe_inputs_kernel(; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs, touching `_prepare_megamoe_inputs_kernel, prepare_megamoe_inputs`; `vllm/models/deepseek_v4/nvidia/model.py` modified +10/-0 (10 lines); hunks: -8,6 +8,7; -16,6 +17,7; symbols: forward, touching `forward`.
- Code diff details:
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +80/-1 (81 lines); hunks: -46,7 +46,8 @@ def test_deepseek_v4_mega_moe_ue8m0_uint8_to_float():; -182,3 +183,81 @@ def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwis...; symbols: test_deepseek_v4_mega_moe_ue8m0_uint8_to_float, test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, test_deepseek_v4_mega_moe_fused_input_staging_masks_padding
  - `vllm/model_executor/layers/fused_moe/modular_kernel.py` modified +17/-0 (17 lines); hunks: -10,6 +10,7; -1133,6 +1134,22 @@ def _prepare(; symbols: _prepare
  - `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +11/-0 (11 lines); hunks: -19,6 +19,7 @@ def _prepare_megamoe_inputs_kernel(; -31,6 +32,7 @@ def _prepare_megamoe_inputs_kernel(; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +10/-0 (10 lines); hunks: -8,6 +8,7; -16,6 +17,7; symbols: forward
  - `vllm/v1/worker/gpu/model_runner.py` modified +10/-0 (10 lines); hunks: -27,6 +27,7; -847,6 +848,13 @@ def prepare_inputs(; symbols: prepare_inputs, execute_model
- Key code excerpts:

```diff
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -46,7 +46,8 @@ def test_deepseek_v4_mega_moe_ue8m0_uint8_to_float():
-        scheduler_config=SimpleNamespace(max_num_batched_tokens=4)
+        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
+        compilation_config=SimpleNamespace(static_forward_context={}),
@@ -182,3 +183,81 @@ def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact():
+@pytest.mark.skipif(
+    not torch.cuda.is_available(),
diff -- vllm/model_executor/layers/fused_moe/modular_kernel.py
@@ -10,6 +10,7 @@
+from vllm.forward_context import get_forward_context, is_forward_context_available
@@ -1133,6 +1134,22 @@ def _prepare(
+        # Skip cudagraph/DP padding tokens uniformly across all a2a backends:
+        # forcing padded rows' expert ids to -1 makes every prepare_finalize drop
+        # them (not dispatched / not computed by the experts). The V2 model runner
+        # marks them in forward_context.is_padding; it is None for runners that do
diff -- vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py
@@ -19,6 +19,7 @@ def _prepare_megamoe_inputs_kernel(
```

- Reviewed files:
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +80/-1
  - runtime: `vllm/model_executor/layers/fused_moe/modular_kernel.py` modified +17/-0; `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +11/-0; `vllm/models/deepseek_v4/nvidia/model.py` modified +10/-0; `vllm/v1/worker/gpu/model_runner.py` modified +10/-0; `vllm/forward_context.py` modified +9/-0; `vllm/v1/worker/gpu/input_batch.py` modified +7/-0
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_mega_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40811 - [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4

- Link: https://github.com/vllm-project/vllm/pull/40811
- Status/date: closed / 2026-06-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +777/-347, 1666 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `csrc/persistent_topk.cuh`; technical summary: Covers "[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4"; the main implementation surface is `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `csrc/persistent_topk.cuh`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/sparse_attn_indexer.py` modified +6/-0 (6 lines); hunks: -98,6 +98,7 @@ def sparse_attn_indexer(; -227,6 +228,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, __init__, forward_cuda, touching `sparse_attn_indexer, __init__, forward_cuda`; `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-0 (1 lines); hunks: -1051,6 +1051,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `csrc/persistent_topk.cuh` modified +623/-232 (855 lines); hunks: -6,10 +6,12; -58,6 +60,76 @@ __device__ __forceinline__ auto convert_to_uint8(float x) ->...; `csrc/topk.cu` modified +143/-115 (258 lines); hunks: -1,5 +1,4; -13,131 +12,158.
- Code diff details:
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +6/-0 (6 lines); hunks: -98,6 +98,7 @@ def sparse_attn_indexer(; -227,6 +228,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, __init__, forward_cuda
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-0 (1 lines); hunks: -1051,6 +1051,7 @@ def __init__(; symbols: __init__, forward
  - `csrc/persistent_topk.cuh` modified +623/-232 (855 lines); hunks: -6,10 +6,12; -58,6 +60,76 @@ __device__ __forceinline__ auto convert_to_uint8(float x) ->...
  - `csrc/topk.cu` modified +143/-115 (258 lines); hunks: -1,5 +1,4; -13,131 +12,158
  - `vllm/utils/deep_gemm.py` modified +4/-0 (4 lines); hunks: -345,6 +345,7 @@ def fp8_fp4_mqa_logits(; -380,6 +381,7 @@ def fp8_fp4_mqa_logits(; symbols: fp8_fp4_mqa_logits, fp8_fp4_paged_mqa_logits
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/sparse_attn_indexer.py
@@ -98,6 +98,7 @@ def sparse_attn_indexer(
+    use_bf16_scores: bool = False,
@@ -227,6 +228,7 @@ def sparse_attn_indexer(
+                logits_dtype=torch.float32,
@@ -316,6 +318,7 @@ def sparse_attn_indexer(
+            logits_dtype=torch.bfloat16 if use_bf16_scores else torch.float32,
@@ -426,8 +429,10 @@ def __init__(
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -1051,6 +1051,7 @@ def __init__(
+            use_bf16_scores=True,
diff -- csrc/persistent_topk.cuh
@@ -6,10 +6,12 @@
+#include <cuda_bf16.h>
+#include <type_traits>
@@ -58,6 +60,76 @@ __device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
+__device__ __forceinline__ auto convert_to_uint16_bf16(__nv_bfloat16 x)
+    -> uint16_t {
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/sparse_attn_indexer.py` modified +6/-0; `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-0; `vllm/utils/deep_gemm.py` modified +4/-0
  - other: `csrc/persistent_topk.cuh` modified +623/-232; `csrc/topk.cu` modified +143/-115
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/utils/deep_gemm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43950 - [ROCm][DSV4] Use aiter mHC pre/post as the default ROCm path

- Link: https://github.com/vllm-project/vllm/pull/43950
- Status/date: merged / 2026-07-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`; associated commits `ed41aa270a9e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +32/-39, 155 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][DSV4] Use aiter mHC pre/post as the default ROCm path"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`; technical summary: Covers "[ROCm][DSV4] Use aiter mHC pre/post as the default ROCm path"; the main implementation surface is `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/model.py` modified +6/-4 (10 lines); hunks: -27,6 +27,7; -303,7 +304,9 @@ def __init__(; symbols: __init__, hc_pre, forward, touching `__init__, hc_pre, forward`; `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-3 (5 lines); hunks: -28,7 +28,7; -123,7 +123,6 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/model.py` modified +6/-4 (10 lines); hunks: -27,6 +27,7; -303,7 +304,9 @@ def __init__(; symbols: __init__, hc_pre, forward
  - `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-3 (5 lines); hunks: -28,7 +28,7; -123,7 +123,6 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -27,6 +27,7 @@
+    HAS_AITER_MHC,
@@ -303,7 +304,9 @@ def __init__(
-        self.has_tilelang = HAS_TILELANG_MHC
+        self.use_fused_mhc = HAS_TILELANG_MHC and not (
+            HAS_AITER_MHC and self.hidden_size % 256 == 0
+        )
diff -- vllm/models/deepseek_v4/amd/mtp.py
@@ -28,7 +28,7 @@
-from vllm.model_executor.layers.mhc import HAS_TILELANG_MHC, HCHeadOp
+from vllm.model_executor.layers.mhc import HCHeadOp
@@ -123,7 +123,6 @@ def __init__(
-        self.has_tilelang = HAS_TILELANG_MHC
@@ -156,7 +155,7 @@ def forward(
-        if self.has_tilelang:
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` modified +6/-4; `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mhc.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46730 - [ROCm][Perf][Bugfix] DSv4 indexer: use platform FP8 dtype (fnuz) for Q-quant on gfx942

- Link: https://github.com/vllm-project/vllm/pull/46730
- Status/date: merged / 2026-07-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`; associated commits `aa8bb5562ebe`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +19/-8, 84 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Perf][Bugfix] DSv4 indexer: use platform FP8 dtype (fnuz) for Q-quant on gfx942"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`; technical summary: Covers "[ROCm][Perf][Bugfix] DSv4 indexer: use platform FP8 dtype (fnuz) for Q-quant on gfx942"; the main implementation surface is `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +19/-8 (27 lines); hunks: -3,6 +3,7; -88,6 +89,8 @@ def _fused_indexer_q_rope_quant_kernel(; symbols: _fused_indexer_q_rope_quant_kernel, fused_indexer_q_rope_quant, touching `_fused_indexer_q_rope_quant_kernel, fused_indexer_q_rope_quant`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +19/-8 (27 lines); hunks: -3,6 +3,7; -88,6 +89,8 @@ def _fused_indexer_q_rope_quant_kernel(; symbols: _fused_indexer_q_rope_quant_kernel, fused_indexer_q_rope_quant
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
@@ -3,6 +3,7 @@
+from vllm.platforms import current_platform
@@ -88,6 +89,8 @@ def _fused_indexer_q_rope_quant_kernel(
+    FP8_MAX: tl.constexpr = 448.0,
+    USE_FNUZ: tl.constexpr = False,
@@ -128,26 +131,28 @@ def _fused_indexer_q_rope_quant_kernel(
-    index_q_scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +19/-8
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45877 - [Frontend] [Parser] Port DeepSeek V4 to streaming parser engine framework

- Link: https://github.com/vllm-project/vllm/pull/45877
- Status/date: merged / 2026-07-04
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_deepseek_v4.py`, `vllm/parser/deepseek_v4.py`, `vllm/reasoning/deepseek_v4_engine_reasoning_parser.py`; associated commits `fb5291b35b0b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 22 files, +1900/-671, 2959 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] [Parser] Port DeepSeek V4 to streaming parser engine framework"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `vllm/reasoning/deepseek_v4_engine_reasoning_parser.py`, `tests/parser/engine/test_deepseek_v4.py`, `vllm/parser/deepseek_v4.py`; technical summary: Covers "[Frontend] [Parser] Port DeepSeek V4 to streaming parser engine framework"; the main implementation surface is `vllm/reasoning/deepseek_v4_engine_reasoning_parser.py`, `tests/parser/engine/test_deepseek_v4.py`, `vllm/parser/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/reasoning/deepseek_v4_engine_reasoning_parser.py` added +6/-0 (6 lines); hunks: -0,0 +1,6; `tests/parser/engine/test_deepseek_v4.py` added +922/-0 (922 lines); hunks: -0,0 +1,922; symbols: _param, mock_tokenizer, TestArgConverter, _raw, touching `_param, mock_tokenizer, TestArgConverter`; `vllm/parser/deepseek_v4.py` added +237/-0 (237 lines); hunks: -0,0 +1,237; symbols: _dsml_arg_converter, _unwrap_wrapper_args, deepseek_v4_config, DeepSeekV4Parser, touching `_dsml_arg_converter, _unwrap_wrapper_args, deepseek_v4_config`.
- Code diff details:
  - `vllm/reasoning/deepseek_v4_engine_reasoning_parser.py` added +6/-0 (6 lines); hunks: -0,0 +1,6
  - `tests/parser/engine/test_deepseek_v4.py` added +922/-0 (922 lines); hunks: -0,0 +1,922; symbols: _param, mock_tokenizer, TestArgConverter, _raw
  - `vllm/parser/deepseek_v4.py` added +237/-0 (237 lines); hunks: -0,0 +1,237; symbols: _dsml_arg_converter, _unwrap_wrapper_args, deepseek_v4_config, DeepSeekV4Parser
- Key code excerpts:

```diff
diff -- vllm/reasoning/deepseek_v4_engine_reasoning_parser.py
@@ -0,0 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from vllm.parser.engine.registered_adapters import DeepSeekV4ParserReasoningAdapter
+__all__ = ["DeepSeekV4ParserReasoningAdapter"]
diff -- tests/parser/engine/test_deepseek_v4.py
@@ -0,0 +1,922 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tests for DeepSeek V4-specific parser engine semantics."""
+import json
+import pytest
+from tests.parser.engine.conftest import make_mock_tokenizer
diff -- vllm/parser/deepseek_v4.py
@@ -0,0 +1,237 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

- Reviewed files:
  - runtime: `vllm/reasoning/deepseek_v4_engine_reasoning_parser.py` added +6/-0; `vllm/parser/deepseek_v4.py` added +237/-0
  - tests: `tests/parser/engine/test_deepseek_v4.py` added +922/-0
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_deepseek_v32.py`, `tests/parser/engine/test_deepseek_v4.py`, `tests/parser/engine/test_replay.py`, `tests/parser/engine/trace_builder.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #47429 - [Bugfix][Spec Decode] Add missing draft_id_to_target_id to DSparkDeepseekV4ForCausalLM

- Link: https://github.com/vllm-project/vllm/pull/47429
- Status/date: merged / 2026-07-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/dspark.py`; associated commits `8d8ec383619d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Spec Decode] Add missing draft_id_to_target_id to DSparkDeepseekV4ForCausalLM"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/nvidia/dspark.py`; technical summary: Covers "[Bugfix][Spec Decode] Add missing draft_id_to_target_id to DSparkDeepseekV4ForCausalLM"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/dspark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/dspark.py` modified +2/-0 (2 lines); hunks: -269,6 +269,8 @@ class DSparkDeepseekV4ForCausalLM(nn.Module):; symbols: DSparkDeepseekV4ForCausalLM, __init__, touching `DSparkDeepseekV4ForCausalLM, __init__`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/dspark.py` modified +2/-0 (2 lines); hunks: -269,6 +269,8 @@ class DSparkDeepseekV4ForCausalLM(nn.Module):; symbols: DSparkDeepseekV4ForCausalLM, __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/dspark.py
@@ -269,6 +269,8 @@ class DSparkDeepseekV4ForCausalLM(nn.Module):
+    # Full-vocab draft: draft ids are target ids, no remapping needed.
+    draft_id_to_target_id = None
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/dspark.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/dspark.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #47474 - [Perf] Cache `token_to_req_indices` for dsv4, 5x~6x kernel performance improvement

- Link: https://github.com/vllm-project/vllm/pull/47474
- Status/date: merged / 2026-07-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/sparse_mla.py`; associated commits `f70caef48b92`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +34/-25, 122 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Cache `token_to_req_indices` for dsv4, 5x~6x kernel performance improvement"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/sparse_mla.py`, `vllm/models/deepseek_v4/compressor.py`; technical summary: Covers "[Perf] Cache `token_to_req_indices` for dsv4, 5x~6x kernel performance improvement"; the main implementation surface is `vllm/models/deepseek_v4/sparse_mla.py`, `vllm/models/deepseek_v4/compressor.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/sparse_mla.py` modified +1/-14 (15 lines); hunks: -5,15 +5,13; -203,18 +201,7 @@ def build(; symbols: build, touching `build`; `vllm/models/deepseek_v4/compressor.py` modified +3/-6 (9 lines); hunks: -104,12 +104,9 @@ def build(; symbols: build, touching `build`.
- Code diff details:
  - `vllm/models/deepseek_v4/sparse_mla.py` modified +1/-14 (15 lines); hunks: -5,15 +5,13; -203,18 +201,7 @@ def build(; symbols: build
  - `vllm/models/deepseek_v4/compressor.py` modified +3/-6 (9 lines); hunks: -104,12 +104,9 @@ def build(; symbols: build
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/sparse_mla.py
@@ -5,15 +5,13 @@
-import numpy as np
-from vllm.utils.torch_utils import np_to_pinned_tensor
@@ -203,18 +201,7 @@ def build(
-        num_tokens = cm.num_actual_tokens
-        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
-        seg_lengths = np.diff(starts)
diff -- vllm/models/deepseek_v4/compressor.py
@@ -104,12 +104,9 @@ def build(
-        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
-        num_reqs = common_attn_metadata.num_reqs
-        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
-        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
-        token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
-        token_to_req_indices.copy_(x, non_blocking=True)
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/sparse_mla.py` modified +1/-14; `vllm/models/deepseek_v4/compressor.py` modified +3/-6
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/sparse_mla.py`, `vllm/v1/attention/backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #47716 - [Bugfix]Fix DeepSeek-V4 fp8_ds_mla KV cache reshape

- Link: https://github.com/vllm-project/vllm/pull/47716
- Status/date: merged / 2026-07-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`; associated commits `04adc8843bbe`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +14/-2, 51 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix]Fix DeepSeek-V4 fp8_ds_mla KV cache reshape"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[Bugfix]Fix DeepSeek-V4 fp8_ds_mla KV cache reshape"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +6/-1 (7 lines); hunks: -56,7 +56,11; -616,6 +620,7 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCa...; symbols: get_kv_cache_spec, touching `get_kv_cache_spec`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +6/-1 (7 lines); hunks: -56,7 +56,11; -616,6 +620,7 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCa...; symbols: get_kv_cache_spec
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -56,7 +56,11 @@
-from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
+from vllm.v1.kv_cache_interface import (
+    KVCacheSpec,
+    MLAAttentionSpec,
+    get_kv_quant_mode,
+)
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/attention.py`, `vllm/v1/attention/backends/mla/sparse_swa.py`, `vllm/v1/worker/gpu_model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #47493 - [Bugfix] DSV4 TP16 garbage output

- Link: https://github.com/vllm-project/vllm/pull/47493
- Status/date: merged / 2026-07-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`; associated commits `80eb01e93dcd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +61/-6, 200 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] DSV4 TP16 garbage output"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[Bugfix] DSV4 TP16 garbage output"; the main implementation surface is `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +35/-0 (35 lines); hunks: -654,6 +654,8 @@ def build_flashinfer_mixed_sparse_indices(; -730,6 +732,13 @@ def build_flashinfer_mixed_sparse_indices(; symbols: build_flashinfer_mixed_sparse_indices, _remap_flashinfer_index, touching `build_flashinfer_mixed_sparse_indices, _remap_flashinfer_index`; `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +19/-0 (19 lines); hunks: -45,6 +45,21 @@ def _get_flashinfer_dsv4_workspace(device: torch.device) -> t...; -368,6 +383,8 @@ def _build_sparse_index_metadata(; symbols: _get_flashinfer_dsv4_workspace, _packed_block_span, DeepseekV4FlashInferMLASparseBackend, _build_sparse_index_metadata, touching `_get_flashinfer_dsv4_workspace, _packed_block_span, DeepseekV4FlashInferMLASparseBackend`; `vllm/models/deepseek_v4/attention.py` modified +4/-4 (8 lines); hunks: -618,7 +618,7 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCa...; -648,15 +648,15 @@ def __init__(; symbols: get_kv_cache_spec, __init__, forward, touching `get_kv_cache_spec, __init__, forward`; `vllm/models/deepseek_v4/compressor.py` modified +1/-1 (2 lines); hunks: -162,7 +162,7 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCa...; symbols: get_kv_cache_spec, forward, touching `get_kv_cache_spec, forward`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +35/-0 (35 lines); hunks: -654,6 +654,8 @@ def build_flashinfer_mixed_sparse_indices(; -730,6 +732,13 @@ def build_flashinfer_mixed_sparse_indices(; symbols: build_flashinfer_mixed_sparse_indices, _remap_flashinfer_index
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +19/-0 (19 lines); hunks: -45,6 +45,21 @@ def _get_flashinfer_dsv4_workspace(device: torch.device) -> t...; -368,6 +383,8 @@ def _build_sparse_index_metadata(; symbols: _get_flashinfer_dsv4_workspace, _packed_block_span, DeepseekV4FlashInferMLASparseBackend, _build_sparse_index_metadata
  - `vllm/models/deepseek_v4/attention.py` modified +4/-4 (8 lines); hunks: -618,7 +618,7 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCa...; -648,15 +648,15 @@ def __init__(; symbols: get_kv_cache_spec, __init__, forward
  - `vllm/models/deepseek_v4/compressor.py` modified +1/-1 (2 lines); hunks: -162,7 +162,7 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCa...; symbols: get_kv_cache_spec, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/cache_utils.py
@@ -654,6 +654,8 @@ def build_flashinfer_mixed_sparse_indices(
+    swa_block_span: int | None = None,
+    compressed_block_span: int | None = None,
@@ -730,6 +732,13 @@ def build_flashinfer_mixed_sparse_indices(
+    # block_span = page_stride / token_stride; == block_size (no-op) for unpacked KV.
+    swa_span = swa_block_size if swa_block_span is None else swa_block_span
+    compressed_span = (
diff -- vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
@@ -45,6 +45,21 @@ def _get_flashinfer_dsv4_workspace(device: torch.device) -> torch.Tensor:
+def _packed_block_span(pool: torch.Tensor) -> int:
+    """Per-block stride of ``pool`` in tokens (``stride(0)//stride(-2)``): ==
+    block_size for unpacked KV, larger when packed (#44577). Raises if not
+    token-aligned."""
+    block_stride = pool.stride(0)
+    token_stride = pool.stride(-2)
diff -- vllm/models/deepseek_v4/attention.py
@@ -618,7 +618,7 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +35/-0; `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +19/-0; `vllm/models/deepseek_v4/attention.py` modified +4/-4; `vllm/models/deepseek_v4/compressor.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/compressor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #47419 - [ROCm] Enable DeepSeek-V4 DSpark speculative decoding on AMD (MI350X / MI355X, gfx950)

- Link: https://github.com/vllm-project/vllm/pull/47419
- Status/date: merged / 2026-07-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/amd/dspark.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`; associated commits `c227aaa3f8ed`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +586/-14, 686 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Enable DeepSeek-V4 DSpark speculative decoding on AMD (MI350X / MI355X, gfx950)"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/models/deepseek_v4/amd/dspark.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`; technical summary: Covers "[ROCm] Enable DeepSeek-V4 DSpark speculative decoding on AMD (MI350X / MI355X, gfx950)"; the main implementation surface is `vllm/models/deepseek_v4/amd/dspark.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/dspark.py` added +499/-0 (499 lines); hunks: -0,0 +1,499; symbols: DSparkDeepseekV4Model, __init__, embed_input_ids, combine_hidden_states, touching `DSparkDeepseekV4Model, __init__, embed_input_ids`; `vllm/models/deepseek_v4/amd/model.py` modified +45/-5 (50 lines); hunks: -40,7 +40,11; -437,7 +441,7 @@ def forward(; symbols: forward, DeepseekV4Model, __init__, touching `forward, DeepseekV4Model, __init__`; `vllm/models/deepseek_v4/amd/rocm.py` modified +8/-2 (10 lines); hunks: -518,8 +518,12 @@ class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekS...; -558,7 +562,9 @@ def build(; symbols: DeepseekV4ROCMAiterSparseSWAMetadataBuilder, __init__, build, touching `DeepseekV4ROCMAiterSparseSWAMetadataBuilder, __init__, build`; `vllm/models/deepseek_v4/__init__.py` modified +4/-4 (8 lines); hunks: -15,16 +15,16.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/dspark.py` added +499/-0 (499 lines); hunks: -0,0 +1,499; symbols: DSparkDeepseekV4Model, __init__, embed_input_ids, combine_hidden_states
  - `vllm/models/deepseek_v4/amd/model.py` modified +45/-5 (50 lines); hunks: -40,7 +40,11; -437,7 +441,7 @@ def forward(; symbols: forward, DeepseekV4Model, __init__
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +8/-2 (10 lines); hunks: -518,8 +518,12 @@ class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekS...; -558,7 +562,9 @@ def build(; symbols: DeepseekV4ROCMAiterSparseSWAMetadataBuilder, __init__, build
  - `vllm/models/deepseek_v4/__init__.py` modified +4/-4 (8 lines); hunks: -15,16 +15,16
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/dspark.py
@@ -0,0 +1,499 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DSpark draft model for DeepSeek-V4 on ROCm/AMD (gfx950).
+ROCm port of ``nvidia/dspark.py``. Follows the same nvidia->amd recipe used for
+``amd/mtp.py``:
+  * import ``DeepseekV4DecoderLayer`` from the AMD ``.model`` (aiter/triton
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -40,7 +40,11 @@
-from vllm.model_executor.models.interfaces import SupportsPP
+from vllm.model_executor.models.interfaces import (
+    EagleModelMixin,
+    SupportsEagle3,
+    SupportsPP,
+)
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -518,8 +518,12 @@ class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuild
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/dspark.py` added +499/-0; `vllm/models/deepseek_v4/amd/model.py` modified +45/-5; `vllm/models/deepseek_v4/amd/rocm.py` modified +8/-2; `vllm/models/deepseek_v4/__init__.py` modified +4/-4
- Risk and verification: The diff ships test coverage in `tests/models/test_registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #48137 - [Perf] Remove redundant repeat and copy for dsv4, 1.8% E2E TPOT improvement.

- Link: https://github.com/vllm-project/vllm/pull/48137
- Status/date: merged / 2026-07-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `442c421e7943`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +316/-15, 387 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Remove redundant repeat and copy for dsv4, 1.8% E2E TPOT improvement."; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[Perf] Remove redundant repeat and copy for dsv4, 1.8% E2E TPOT improvement."; the main implementation surface is `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/model.py` modified +45/-15 (60 lines); hunks: -22,6 +22,7; -827,6 +828,7 @@ def __init__(; symbols: __init__, forward, finalize_mega_moe_weights, finalize_mhc_broadcast_weights, touching `__init__, forward, finalize_mega_moe_weights`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +45/-15 (60 lines); hunks: -22,6 +22,7; -827,6 +828,7 @@ def __init__(; symbols: __init__, forward, finalize_mega_moe_weights, finalize_mhc_broadcast_weights
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -22,6 +22,7 @@
+    mhc_pre_broadcast_tilelang,
@@ -827,6 +828,7 @@ def __init__(
+        self.hc_attn_fn_broadcast: torch.Tensor | None = None
@@ -876,20 +878,37 @@ def forward(
-            residual = x
-            post_mix, res_mix, x = mhc_pre_tilelang(
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +45/-15
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/kernels/mhc/tilelang.py`, `vllm/model_executor/kernels/mhc/tilelang_kernels.py`, `vllm/models/deepseek_v4/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #47718 - [ROCm][Perf] DSv4 two-stage compressor kernel for HCA prefill

- Link: https://github.com/vllm-project/vllm/pull/47718
- Status/date: merged / 2026-07-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/compressor.py`; associated commits `eb33ff34dd65`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +528/-0, 615 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Perf] DSv4 two-stage compressor kernel for HCA prefill"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/compressor.py`; technical summary: Covers "[ROCm][Perf] DSv4 two-stage compressor kernel for HCA prefill"; the main implementation surface is `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/compressor.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +355/-0 (355 lines); hunks: -19,6 +19,7; -296,6 +297,360 @@ def _fused_kv_compress_norm_rope_insert_sparse_attn(; symbols: _fused_kv_compress_norm_rope_insert_sparse_attn, _n_cu, _pick_compress_num_splits, _compress_gather_split_sparse_attn, touching `_fused_kv_compress_norm_rope_insert_sparse_attn, _n_cu, _pick_compress_num_splits`; `vllm/models/deepseek_v4/compressor.py` modified +43/-0 (43 lines); hunks: -14,6 +14,7; -27,13 +28,20; symbols: _prefer_two_stage_compressor, CompressorBackend, __init__, CompressorMetadata, touching `_prefer_two_stage_compressor, CompressorBackend, __init__`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +355/-0 (355 lines); hunks: -19,6 +19,7; -296,6 +297,360 @@ def _fused_kv_compress_norm_rope_insert_sparse_attn(; symbols: _fused_kv_compress_norm_rope_insert_sparse_attn, _n_cu, _pick_compress_num_splits, _compress_gather_split_sparse_attn
  - `vllm/models/deepseek_v4/compressor.py` modified +43/-0 (43 lines); hunks: -14,6 +14,7; -27,13 +28,20; symbols: _prefer_two_stage_compressor, CompressorBackend, __init__, CompressorMetadata
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
@@ -19,6 +19,7 @@
+from functools import lru_cache
@@ -296,6 +297,360 @@ def _fused_kv_compress_norm_rope_insert_sparse_attn(
+# =============================================================================
+# Split kernels variant of the head=512 compressor (deep cr=128 gather).
+#  - compress gather: instead of launching one program per token, split along
+#    the head dimension to maximize CU occupancy. The head dimension split
diff -- vllm/models/deepseek_v4/compressor.py
@@ -14,6 +14,7 @@
+    compress_norm_rope_store_two_stage_triton,
@@ -27,13 +28,20 @@
+from vllm.v1.attention.backends.utils import split_decodes_and_prefills
+def _prefer_two_stage_compressor() -> bool:
+    # Platforms that favor the triton variant of two-stage compressor split.
+    # Currently only tested on ROCm
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +355/-0; `vllm/models/deepseek_v4/compressor.py` modified +43/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #47677 - [XPU] Add DSpark speculative decoding support for DeepSeek-V4

- Link: https://github.com/vllm-project/vllm/pull/47677
- Status/date: merged / 2026-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/xpu/dspark.py`, `vllm/models/deepseek_v4/xpu/model.py`; associated commits `9d1c695be587`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +454/-6, 512 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU] Add DSpark speculative decoding support for DeepSeek-V4"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/models/deepseek_v4/xpu/dspark.py`, `vllm/models/deepseek_v4/xpu/model.py`, `vllm/models/deepseek_v4/__init__.py`; technical summary: Covers "[XPU] Add DSpark speculative decoding support for DeepSeek-V4"; the main implementation surface is `vllm/models/deepseek_v4/xpu/dspark.py`, `vllm/models/deepseek_v4/xpu/model.py`, `vllm/models/deepseek_v4/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/xpu/dspark.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: DSparkDeepseekV4Model, __init__, embed_input_ids, combine_hidden_states, touching `DSparkDeepseekV4Model, __init__, embed_input_ids`; `vllm/models/deepseek_v4/xpu/model.py` modified +17/-4 (21 lines); hunks: -44,7 +44,11; -975,7 +979,7 @@ def forward(; symbols: forward, DeepseekV4Model, __init__, touching `forward, DeepseekV4Model, __init__`; `vllm/models/deepseek_v4/__init__.py` modified +1/-2 (3 lines); hunks: -21,10 +21,9.
- Code diff details:
  - `vllm/models/deepseek_v4/xpu/dspark.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: DSparkDeepseekV4Model, __init__, embed_input_ids, combine_hidden_states
  - `vllm/models/deepseek_v4/xpu/model.py` modified +17/-4 (21 lines); hunks: -44,7 +44,11; -975,7 +979,7 @@ def forward(; symbols: forward, DeepseekV4Model, __init__
  - `vllm/models/deepseek_v4/__init__.py` modified +1/-2 (3 lines); hunks: -21,10 +21,9
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/xpu/dspark.py
@@ -0,0 +1,436 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DSpark draft model for DeepSeek-V4 on Intel XPU.
+Minimal XPU port of nvidia/dspark.py. Replaces tilelang MHC kernels with
+the platform-agnostic custom ops (HCHeadOp, MHCPostOp) already used by the
+XPU MTP path, and uses the XPU Triton-based qnorm_rope_kv_fp8_insert for
diff -- vllm/models/deepseek_v4/xpu/model.py
@@ -44,7 +44,11 @@
-from vllm.model_executor.models.interfaces import SupportsPP
+from vllm.model_executor.models.interfaces import (
+    EagleModelMixin,
+    SupportsEagle3,
+    SupportsPP,
+)
diff -- vllm/models/deepseek_v4/__init__.py
@@ -21,10 +21,9 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/xpu/dspark.py` added +436/-0; `vllm/models/deepseek_v4/xpu/model.py` modified +17/-4; `vllm/models/deepseek_v4/__init__.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/xpu/dspark.py`, `vllm/models/deepseek_v4/xpu/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45991 - [XPU][DeepSeekV4]Add DeepSeek-V4 fuse_index_q SYCL kernel path

- Link: https://github.com/vllm-project/vllm/pull/45991
- Status/date: merged / 2026-07-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`; associated commits `c67650f04bc9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +150/-0, 178 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU][DeepSeekV4]Add DeepSeek-V4 fuse_index_q SYCL kernel path"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`; technical summary: Covers "[XPU][DeepSeekV4]Add DeepSeek-V4 fuse_index_q SYCL kernel path"; the main implementation surface is `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +23/-0 (23 lines); hunks: -367,6 +367,18 @@ def fused_indexer_q_rope_quant(; -423,6 +435,17 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant, touching `fused_indexer_q_rope_quant`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +23/-0 (23 lines); hunks: -367,6 +367,18 @@ def fused_indexer_q_rope_quant(; -423,6 +435,17 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
@@ -367,6 +367,18 @@ def fused_indexer_q_rope_quant(
+        elif current_platform.is_xpu():
+            torch.ops.vllm.xpu_deepseek_fused_indexer_q_rope_mxfp4(
+                index_q,
+                positions,
+                index_q_cos_sin_cache,
+                index_weights,
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +23/-0
- Risk and verification: Runtime changes concentrate in `vllm/_xpu_ops.py`, `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #48957 - [DSv4 Perf] Skip empty c128 kernel launch, around 2x kernel performance improvement.

- Link: https://github.com/vllm-project/vllm/pull/48957
- Status/date: merged / 2026-07-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/compressor.py`; associated commits `37e370fe936f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +53/-2, 118 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4 Perf] Skip empty c128 kernel launch, around 2x kernel performance improvement."; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/compressor.py`; technical summary: Covers "[DSv4 Perf] Skip empty c128 kernel launch, around 2x kernel performance improvement."; the main implementation surface is `vllm/models/deepseek_v4/compressor.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/compressor.py` modified +32/-2 (34 lines); hunks: -7,7 +7,7; -42,6 +42,19 @@ def _prefer_two_stage_compressor() -> bool:; symbols: _prefer_two_stage_compressor, _get_c128_boundary, CompressorBackend, __init__, touching `_prefer_two_stage_compressor, _get_c128_boundary, CompressorBackend`.
- Code diff details:
  - `vllm/models/deepseek_v4/compressor.py` modified +32/-2 (34 lines); hunks: -7,7 +7,7; -42,6 +42,19 @@ def _prefer_two_stage_compressor() -> bool:; symbols: _prefer_two_stage_compressor, _get_c128_boundary, CompressorBackend, __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/compressor.py
@@ -7,7 +7,7 @@
-from vllm.config import VllmConfig, get_current_vllm_config
+from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
@@ -42,6 +42,19 @@ def _prefer_two_stage_compressor() -> bool:
+def _get_c128_boundary(metadata: CommonAttentionMetadata) -> bool | None:
+    starts = metadata._num_computed_tokens_cpu
+    if starts is None:
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/compressor.py` modified +32/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #48993 - [Core][DSV4] Compact MXFP4 indexer KV cache and packed group overlays

- Link: https://github.com/vllm-project/vllm/pull/48993
- Status/date: merged / 2026-07-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`; associated commits `f3a920a07640`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +216/-84, 401 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Core][DSV4] Compact MXFP4 indexer KV cache and packed group overlays"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[Core][DSV4] Compact MXFP4 indexer KV cache and packed group overlays"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +11/-5 (16 lines); hunks: -26,6 +26,7; -727,11 +728,16 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +11/-5 (16 lines); hunks: -26,6 +26,7; -727,11 +728,16 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -26,6 +26,7 @@
+from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
@@ -727,11 +728,16 @@ def __init__(
-        # NOTE(yifan): FP8 indxer cache use the same layout as V3.2:
-        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
-        # For FP4 indexer cache, we still allocate the same amount of memory as FP8,
-        # but only use the first half of the memory.
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +11/-5
- Risk and verification: The diff ships test coverage in `tests/v1/core/test_contiguous_kv_packing.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #48044 - [ROCm] Fused Shared Expert Support for AMD Quark DeepSeek-V4 Model Checkpoints

- Link: https://github.com/vllm-project/vllm/pull/48044
- Status/date: merged / 2026-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/quant_config.py`; associated commits `27ffbfde8dec`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +175/-13, 314 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Fused Shared Expert Support for AMD Quark DeepSeek-V4 Model Checkpoints"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/quant_config.py`, `vllm/models/deepseek_v4/amd/mtp.py`; technical summary: Covers "[ROCm] Fused Shared Expert Support for AMD Quark DeepSeek-V4 Model Checkpoints"; the main implementation surface is `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/quant_config.py`, `vllm/models/deepseek_v4/amd/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/model.py` modified +121/-11 (132 lines); hunks: -8,7 +8,8; -110,6 +111,51 @@ def forward(self, x):; symbols: forward, _shared_experts_are_fp4, _fuse_shared_experts_enabled, DeepseekV4MoE, touching `forward, _shared_experts_are_fp4, _fuse_shared_experts_enabled`; `vllm/models/deepseek_v4/quant_config.py` modified +37/-2 (39 lines); hunks: -4,7 +4,7; -117,20 +117,55 @@ def _get_nvfp4_config(self) -> ModelOptNvFp4Config:; symbols: _get_nvfp4_config, get_name, _is_quark_mxfp4_ocp, override_quantization_method, touching `_get_nvfp4_config, get_name, _is_quark_mxfp4_ocp`; `vllm/models/deepseek_v4/amd/mtp.py` modified +17/-0 (17 lines); hunks: -334,6 +334,21 @@ def _find_mtp_layer_idx(name: str) -> int:; -393,6 +408,7 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx, _resolve_scale_name, touching `_find_mtp_layer_idx, _resolve_scale_name`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/model.py` modified +121/-11 (132 lines); hunks: -8,7 +8,8; -110,6 +111,51 @@ def forward(self, x):; symbols: forward, _shared_experts_are_fp4, _fuse_shared_experts_enabled, DeepseekV4MoE
  - `vllm/models/deepseek_v4/quant_config.py` modified +37/-2 (39 lines); hunks: -4,7 +4,7; -117,20 +117,55 @@ def _get_nvfp4_config(self) -> ModelOptNvFp4Config:; symbols: _get_nvfp4_config, get_name, _is_quark_mxfp4_ocp, override_quantization_method
  - `vllm/models/deepseek_v4/amd/mtp.py` modified +17/-0 (17 lines); hunks: -334,6 +334,21 @@ def _find_mtp_layer_idx(name: str) -> int:; -393,6 +408,7 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx, _resolve_scale_name
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -8,7 +8,8 @@
-from vllm.config import VllmConfig
+import vllm.envs as envs
+from vllm.config import VllmConfig, get_current_vllm_config
@@ -110,6 +111,51 @@ def forward(self, x):
+def _shared_experts_are_fp4(config, layer_idx: int | None = None) -> bool:
+    """Whether the shared experts are MXFP4 and thus fusable.
diff -- vllm/models/deepseek_v4/quant_config.py
@@ -4,7 +4,7 @@
-from typing import TYPE_CHECKING
+from typing import TYPE_CHECKING, cast
@@ -117,20 +117,55 @@ def _get_nvfp4_config(self) -> ModelOptNvFp4Config:
+    @staticmethod
+    def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:
+        """True for AMD-Quark exports whose global scheme is MXFP4."""
diff -- vllm/models/deepseek_v4/amd/mtp.py
@@ -334,6 +334,21 @@ def _find_mtp_layer_idx(name: str) -> int:
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` modified +121/-11; `vllm/models/deepseek_v4/quant_config.py` modified +37/-2; `vllm/models/deepseek_v4/amd/mtp.py` modified +17/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/quant_config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #49415 - [Bugfix] Fix DeepSeek-V4 DSpark draft shared-expert padding for TP > 8

- Link: https://github.com/vllm-project/vllm/pull/49415
- Status/date: merged / 2026-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; associated commits `76bf55240cf8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +31/-11, 119 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DeepSeek-V4 DSpark draft shared-expert padding for TP > 8"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`, `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[Bugfix] Fix DeepSeek-V4 DSpark draft shared-expert padding for TP > 8"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`, `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/dspark.py` modified +12/-4 (16 lines); hunks: -43,6 +43,7; -277,6 +278,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, load_weights, touching `__init__, load_weights`; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +11/-4 (15 lines); hunks: -52,6 +52,7; -265,6 +266,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, _find_mtp_layer_idx, touching `__init__, _find_mtp_layer_idx`; `vllm/models/deepseek_v4/nvidia/model.py` modified +8/-3 (11 lines); hunks: -1181,7 +1181,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; -1256,15 +1258,18 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; symbols: load_weights, _pad_shared_expert_weight, touching `load_weights, _pad_shared_expert_weight`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/dspark.py` modified +12/-4 (16 lines); hunks: -43,6 +43,7; -277,6 +278,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, load_weights
  - `vllm/models/deepseek_v4/nvidia/mtp.py` modified +11/-4 (15 lines); hunks: -52,6 +52,7; -265,6 +266,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, _find_mtp_layer_idx
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +8/-3 (11 lines); hunks: -1181,7 +1181,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; -1256,15 +1258,18 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; symbols: load_weights, _pad_shared_expert_weight
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/dspark.py
@@ -43,6 +43,7 @@
+    DeepseekV4Model,
@@ -277,6 +278,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
+        self.quant_config = vllm_config.quant_config
+        self.pad_shared_expert = (
+            getattr(self.quant_config, "weight_block_size", None) is not None
+            and not vllm_config.parallel_config.use_sequence_parallel_moe
diff -- vllm/models/deepseek_v4/nvidia/mtp.py
@@ -52,6 +52,7 @@
+    DeepseekV4Model,
@@ -265,6 +266,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        self.pad_shared_expert = (
+            getattr(self.quant_config, "weight_block_size", None) is not None
+            and not vllm_config.parallel_config.use_sequence_parallel_moe
+        )
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -1181,7 +1181,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/dspark.py` modified +12/-4; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +11/-4; `vllm/models/deepseek_v4/nvidia/model.py` modified +8/-3
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #49486 - [DSv4 Perf] Skip topk and router when not needed, 3.4% E2E TTFT improvement for Decode case

- Link: https://github.com/vllm-project/vllm/pull/49486
- Status/date: merged / 2026-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`; associated commits `b0cb1da1bde6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +43/-0, 64 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4 Perf] Skip topk and router when not needed, 3.4% E2E TTFT improvement for Decode case"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[DSv4 Perf] Skip topk and router when not needed, 3.4% E2E TTFT improvement for Decode case"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +43/-0 (43 lines); hunks: -46,6 +46,7; -65,6 +66,25; symbols: _fill_short_context_topk_indices, _resolve_dsv4_kv_cache_dtype, forward, wq_b_and_q_quant, touching `_fill_short_context_topk_indices, _resolve_dsv4_kv_cache_dtype, forward`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +43/-0 (43 lines); hunks: -46,6 +46,7; -65,6 +66,25; symbols: _fill_short_context_topk_indices, _resolve_dsv4_kv_cache_dtype, forward, wq_b_and_q_quant
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -46,6 +46,7 @@
+from vllm.triton_utils import tl, triton
@@ -65,6 +66,25 @@
+@triton.jit
+def _fill_short_context_topk_indices(
+    output,
+    positions,
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +43/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50004 - [DSv4 Perf] Adaptive topk width, 1.0% E2E throughput improvement

- Link: https://github.com/vllm-project/vllm/pull/50004
- Status/date: merged / 2026-07-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/sparse_mla.py`; associated commits `b2f9e4caa494`, `e6f35d3c69b2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +56/-7, 104 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4 Perf] Adaptive topk width, 1.0% E2E throughput improvement"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/sparse_mla.py`; technical summary: Covers "[DSv4 Perf] Adaptive topk width, 1.0% E2E throughput improvement"; the main implementation surface is `vllm/models/deepseek_v4/sparse_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/sparse_mla.py` modified +19/-7 (26 lines); hunks: -261,6 +261,13 @@ def _build_c128a_metadata(; -273,7 +280,7 @@ def _build_c128a_metadata(; symbols: _build_c128a_metadata, build_c128a_topk_metadata, touching `_build_c128a_metadata, build_c128a_topk_metadata`.
- Code diff details:
  - `vllm/models/deepseek_v4/sparse_mla.py` modified +19/-7 (26 lines); hunks: -261,6 +261,13 @@ def _build_c128a_metadata(; -273,7 +280,7 @@ def _build_c128a_metadata(; symbols: _build_c128a_metadata, build_c128a_topk_metadata
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/sparse_mla.py
@@ -261,6 +261,13 @@ def _build_c128a_metadata(
+        active_topk_width = min(
+            max(
+                triton.next_power_of_2(max(cm.max_seq_len // self.compress_ratio, 1)),
+                _C128A_TOPK_ALIGNMENT,
+            ),
+            self.c128a_max_compressed,
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/sparse_mla.py` modified +19/-7
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_flashmla_sparse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #49634 - [Bugfix] Fix DeepseekV4FP8 Quark MXFP4 crash on list-valued weight

- Link: https://github.com/vllm-project/vllm/pull/49634
- Status/date: merged / 2026-07-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/quant_config.py`; associated commits `fbb1ef680309`
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DeepseekV4FP8 Quark MXFP4 crash on list-valued weight"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/quant_config.py`; technical summary: Covers "[Bugfix] Fix DeepseekV4FP8 Quark MXFP4 crash on list-valued weight"; the main implementation surface is `vllm/models/deepseek_v4/quant_config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/quant_config.py` modified +5/-1 (6 lines); hunks: -120,7 +120,11 @@ def get_name(cls) -> QuantizationMethods:; symbols: get_name, _is_quark_mxfp4_ocp, touching `get_name, _is_quark_mxfp4_ocp`.
- Code diff details:
  - `vllm/models/deepseek_v4/quant_config.py` modified +5/-1 (6 lines); hunks: -120,7 +120,11 @@ def get_name(cls) -> QuantizationMethods:; symbols: get_name, _is_quark_mxfp4_ocp
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/quant_config.py
@@ -120,7 +120,11 @@ def get_name(cls) -> QuantizationMethods:
-        weight = (hf_quant_cfg.get("global_quant_config") or {}).get("weight") or {}
+        weight = (hf_quant_cfg.get("global_quant_config") or {}).get("weight")
+        # A non-dict weight (e.g. a list of multiple specs) means not an OCP
+        # MXFP4 scheme (e.g. NVFP4 with 2-level scale).
+        if not isinstance(weight, dict):
+            return False
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/quant_config.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/quant_config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50312 - [DSv4 Perf] Fix redundant memory allocation and copy for dsv4 pp buffer, 448 MiB GPU memory saved

- Link: https://github.com/vllm-project/vllm/pull/50312
- Status/date: merged / 2026-07-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/xpu/model.py`; associated commits `904fae8be12f`
- Diff scope read: GitHub Pull Request files API returned 3 files, +24/-28, 94 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4 Perf] Fix redundant memory allocation and copy for dsv4 pp buffer, 448 MiB GPU memory saved"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/xpu/model.py`; technical summary: Covers "[DSv4 Perf] Fix redundant memory allocation and copy for dsv4 pp buffer, 448 MiB GPU memory saved"; the main implementation surface is `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/xpu/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/model.py` modified +8/-10 (18 lines); hunks: -573,13 +573,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -679,9 +677,9 @@ def forward(; symbols: __init__, forward, touching `__init__, forward`; `vllm/models/deepseek_v4/nvidia/model.py` modified +8/-10 (18 lines); hunks: -1039,13 +1039,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; -1130,9 +1128,9 @@ def forward(; symbols: __init__, forward, touching `__init__, forward`; `vllm/models/deepseek_v4/xpu/model.py` modified +8/-8 (16 lines); hunks: -1057,11 +1057,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; -1139,9 +1139,9 @@ def forward(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/model.py` modified +8/-10 (18 lines); hunks: -573,13 +573,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -679,9 +677,9 @@ def forward(; symbols: __init__, forward
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +8/-10 (18 lines); hunks: -1039,13 +1039,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; -1130,9 +1128,9 @@ def forward(; symbols: __init__, forward
  - `vllm/models/deepseek_v4/xpu/model.py` modified +8/-8 (16 lines); hunks: -1057,11 +1057,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; -1139,9 +1139,9 @@ def forward(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -573,13 +573,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        # Pre-hc_head residual stream buffer for the MTP draft. Stable
-        # address (outside the cudagraph pool) so the copy_ in forward()
-        # refreshes it correctly across captured shapes.
-        # refreshes it correctly across captured shapes. Only allocated on
-        # the last PP rank — that's where MTP target hidden states are
-        # produced.
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -1039,13 +1039,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        # Pre-hc_head residual stream buffer for the MTP draft. Stable
-        # address (outside the cudagraph pool) so the copy_ in forward()
-        # refreshes it correctly across captured shapes.
-        # refreshes it correctly across captured shapes. Only allocated on
-        # the last PP rank — that's where MTP target hidden states are
-        # produced.
diff -- vllm/models/deepseek_v4/xpu/model.py
@@ -1057,11 +1057,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` modified +8/-10; `vllm/models/deepseek_v4/nvidia/model.py` modified +8/-10; `vllm/models/deepseek_v4/xpu/model.py` modified +8/-8
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/xpu/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50298 - [DSv4 Perf] Remove redundant full kernel for dsv4, 1.88x kernel performance improvement

- Link: https://github.com/vllm-project/vllm/pull/50298
- Status/date: merged / 2026-07-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`; associated commits `837eae64580c`
- Diff scope read: GitHub Pull Request files API returned 3 files, +44/-23, 140 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4 Perf] Remove redundant full kernel for dsv4, 1.88x kernel performance improvement"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`; technical summary: Covers "[DSv4 Perf] Remove redundant full kernel for dsv4, 1.88x kernel performance improvement"; the main implementation surface is `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +13/-9 (22 lines); hunks: -535,22 +535,26 @@ def combine_topk_swa_indices(; symbols: combine_topk_swa_indices, touching `combine_topk_swa_indices`; `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +15/-2 (17 lines); hunks: -20,6 +20,7; -95,8 +96,13 @@ def forward_mqa(; symbols: forward_mqa, _forward_prefill, touching `forward_mqa, _forward_prefill`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +13/-9 (22 lines); hunks: -535,22 +535,26 @@ def combine_topk_swa_indices(; symbols: combine_topk_swa_indices
  - `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +15/-2 (17 lines); hunks: -20,6 +20,7; -95,8 +96,13 @@ def forward_mqa(; symbols: forward_mqa, _forward_prefill
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/cache_utils.py
@@ -535,22 +535,26 @@ def combine_topk_swa_indices(
+    out: tuple[torch.Tensor, torch.Tensor] | None = None,
-    combined_indices = torch.full(
-        (num_tokens, combined_topk),
-        fill_value=-1,
-        dtype=torch.int32,
-        device=topk_indices.device,
diff -- vllm/models/deepseek_v4/nvidia/flashmla.py
@@ -20,6 +20,7 @@
+from vllm.utils.math_utils import round_up
@@ -95,8 +96,13 @@ def forward_mqa(
+            assert self.topk_indices_buffer is not None
+            top_k = 0 if swa_only else self.topk_indices_buffer.shape[-1]
+            combined_topk = round_up(top_k + self.window_size, 128)
+                ((self.max_num_batched_tokens, combined_topk), torch.int32),
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +13/-9; `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +15/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_flashmla_sparse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #46720 - [ROCm][DSV4] B-preshuffle the attention fp8 projections

- Link: https://github.com/vllm-project/vllm/pull/46720
- Status/date: merged / 2026-07-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`; associated commits `12a34a6bc794`
- Diff scope read: GitHub Pull Request files API returned 5 files, +187/-9, 311 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][DSV4] B-preshuffle the attention fp8 projections"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[ROCm][DSV4] B-preshuffle the attention fp8 projections"; the main implementation surface is `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/rocm.py` modified +71/-1 (72 lines); hunks: -6,6 +6,10; -442,10 +446,73 @@ class DeepseekV4ROCMAiterMLAAttention(DeepseekV4Attention):; symbols: DeepseekV4ROCMAiterMLAAttention, __init__, get_padded_num_q_heads, prepare_attn_preshuffle, touching `DeepseekV4ROCMAiterMLAAttention, __init__, get_padded_num_q_heads`; `vllm/models/deepseek_v4/amd/model.py` modified +53/-3 (56 lines); hunks: -9,6 +9,7; -104,8 +105,50 @@ def __init__(; symbols: __init__, prepare_gateup_preshuffle, forward, get_mtp_target_hidden_states, touching `__init__, prepare_gateup_preshuffle, forward`; `vllm/models/deepseek_v4/attention.py` modified +6/-3 (9 lines); hunks: -390,6 +390,11 @@ def forward(; -434,9 +439,7 @@ def indexer_compressor_kv_score() -> torch.Tensor:; symbols: forward, _fused_wqa_wkv_gemm, attn_gemm_parallel_execute, indexer_compressor_kv_score, touching `forward, _fused_wqa_wkv_gemm, attn_gemm_parallel_execute`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +71/-1 (72 lines); hunks: -6,6 +6,10; -442,10 +446,73 @@ class DeepseekV4ROCMAiterMLAAttention(DeepseekV4Attention):; symbols: DeepseekV4ROCMAiterMLAAttention, __init__, get_padded_num_q_heads, prepare_attn_preshuffle
  - `vllm/models/deepseek_v4/amd/model.py` modified +53/-3 (56 lines); hunks: -9,6 +9,7; -104,8 +105,50 @@ def __init__(; symbols: __init__, prepare_gateup_preshuffle, forward, get_mtp_target_hidden_states
  - `vllm/models/deepseek_v4/attention.py` modified +6/-3 (9 lines); hunks: -390,6 +390,11 @@ def forward(; -434,9 +439,7 @@ def indexer_compressor_kv_score() -> torch.Tensor:; symbols: forward, _fused_wqa_wkv_gemm, attn_gemm_parallel_execute, indexer_compressor_kv_score
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -6,6 +6,10 @@
+from vllm.distributed import (
+    get_tensor_model_parallel_world_size,
+    tensor_model_parallel_all_reduce,
+)
@@ -442,10 +446,73 @@ class DeepseekV4ROCMAiterMLAAttention(DeepseekV4Attention):
+    def __init__(self, *args, **kwargs):
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -9,6 +9,7 @@
+from vllm._aiter_ops import rocm_aiter_ops
@@ -104,8 +105,50 @@ def __init__(
+        # gate_up_proj B-preshuffle (ColumnParallel -> no all-reduce); set at load.
+        self._gateup = rocm_aiter_ops.is_enabled()
+        # Block scale for the preshuffled gate_up weight; None = not preshuffled.
+        self._gateup_scale: torch.Tensor | None = None
diff -- vllm/models/deepseek_v4/attention.py
@@ -390,6 +390,11 @@ def forward(
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/rocm.py` modified +71/-1; `vllm/models/deepseek_v4/amd/model.py` modified +53/-3; `vllm/models/deepseek_v4/attention.py` modified +6/-3
- Risk and verification: Runtime changes concentrate in `vllm/_aiter_ops.py`, `vllm/model_executor/model_loader/utils.py`, `vllm/models/deepseek_v4/amd/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #48047 - [DSv4] Remove sparse-MLA q-head padding for FlashInfer >=0.6.14

- Link: https://github.com/vllm-project/vllm/pull/48047
- Status/date: merged / 2026-07-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`; associated commits `b49eaf205a75`
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-19, 56 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4] Remove sparse-MLA q-head padding for FlashInfer >=0.6.14"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`; technical summary: Covers "[DSv4] Remove sparse-MLA q-head padding for FlashInfer >=0.6.14"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +16/-19 (35 lines); hunks: -60,6 +60,20 @@ def _packed_block_span(pool: torch.Tensor) -> int:; -164,13 +178,7 @@ class DeepseekV4FlashInferMLAAttention(DeepseekV4Attention):; symbols: _packed_block_span, _pad_to_supported_q_heads, DeepseekV4FlashInferMLASparseBackend, DeepseekV4FlashInferMLAAttention, touching `_packed_block_span, _pad_to_supported_q_heads, DeepseekV4FlashInferMLASparseBackend`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +16/-19 (35 lines); hunks: -60,6 +60,20 @@ def _packed_block_span(pool: torch.Tensor) -> int:; -164,13 +178,7 @@ class DeepseekV4FlashInferMLAAttention(DeepseekV4Attention):; symbols: _packed_block_span, _pad_to_supported_q_heads, DeepseekV4FlashInferMLASparseBackend, DeepseekV4FlashInferMLAAttention
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
@@ -60,6 +60,20 @@ def _packed_block_span(pool: torch.Tensor) -> int:
+# Sparse MLA h_q counts accepted natively (flashinfer>=0.6.14, #3545).
+_SPARSE_MLA_SUPPORTED_Q_HEADS = (8, 16, 32, 64, 128)
+def _pad_to_supported_q_heads(num_heads: int) -> int:
+    for supported in _SPARSE_MLA_SUPPORTED_Q_HEADS:
+        if num_heads <= supported:
+            return supported
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +16/-19
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #49236 - [DSv4 Perf] Optimize workspace reuse for eager break

- Link: https://github.com/vllm-project/vllm/pull/49236
- Status/date: merged / 2026-07-31
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/compressor.py` and 9 files; associated commits `df71917cf17c`
- Diff scope read: GitHub Pull Request files API returned 15 files, +354/-30, 708 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4 Perf] Optimize workspace reuse for eager break"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[DSv4 Perf] Optimize workspace reuse for eager break"; the main implementation surface is `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +30/-12 (42 lines); hunks: -295,6 +295,7 @@ def fused_indexer_q_rope_quant(; -332,24 +333,37 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant, touching `fused_indexer_q_rope_quant`; `vllm/models/deepseek_v4/attention.py` modified +35/-2 (37 lines); hunks: -29,6 +29,7; -181,6 +182,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/models/deepseek_v4/nvidia/model.py` modified +20/-0 (20 lines); hunks: -66,6 +66,7; -798,6 +799,7 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +10/-5 (15 lines); hunks: -2097,6 +2097,7 @@ def compress_norm_rope_store_cutedsl(; -2129,11 +2130,15 @@ def compress_norm_rope_store_cutedsl(; symbols: compress_norm_rope_store_cutedsl, touching `compress_norm_rope_store_cutedsl`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +30/-12 (42 lines); hunks: -295,6 +295,7 @@ def fused_indexer_q_rope_quant(; -332,24 +333,37 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant
  - `vllm/models/deepseek_v4/attention.py` modified +35/-2 (37 lines); hunks: -29,6 +29,7; -181,6 +182,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +20/-0 (20 lines); hunks: -66,6 +66,7; -798,6 +799,7 @@ def __init__(; symbols: __init__
  - `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +10/-5 (15 lines); hunks: -2097,6 +2097,7 @@ def compress_norm_rope_store_cutedsl(; -2129,11 +2130,15 @@ def compress_norm_rope_store_cutedsl(; symbols: compress_norm_rope_store_cutedsl
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +10/-2 (12 lines); hunks: -438,6 +438,7 @@ def compute_global_topk_indices_and_lens(; -447,8 +448,15 @@ def compute_global_topk_indices_and_lens(; symbols: compute_global_topk_indices_and_lens
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
@@ -295,6 +295,7 @@ def fused_indexer_q_rope_quant(
+    output_buffers: tuple[torch.Tensor, ...] | None = None,
@@ -332,24 +333,37 @@ def fused_indexer_q_rope_quant(
-    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)
+    if output_buffers is None:
+        index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)
+    else:
diff -- vllm/models/deepseek_v4/attention.py
@@ -29,6 +29,7 @@
+    from vllm.models.deepseek_v4.eager_scratch import DeepseekV4EagerScratchPool
@@ -181,6 +182,7 @@ def __init__(
+        eager_scratch_pool: "DeepseekV4EagerScratchPool | None" = None,
@@ -269,6 +271,7 @@ def __init__(
+        self.eager_scratch_pool = eager_scratch_pool
@@ -290,6 +293,7 @@ def __init__(
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -66,6 +66,7 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +30/-12; `vllm/models/deepseek_v4/attention.py` modified +35/-2; `vllm/models/deepseek_v4/nvidia/model.py` modified +20/-0; `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +10/-5; `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +10/-2; `vllm/models/deepseek_v4/compressor.py` modified +10/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #46789 - [DSV4] Implement Sequence Parallelism

- Link: https://github.com/vllm-project/vllm/pull/46789
- Status/date: merged / 2026-08-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; associated commits `38a466e7b6e0`
- Diff scope read: GitHub Pull Request files API returned 7 files, +138/-29, 450 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Implement Sequence Parallelism"; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; technical summary: Covers "[DSV4] Implement Sequence Parallelism"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/model.py` modified +53/-4 (57 lines); hunks: -65,6 +65,12; -515,13 +521,15 @@ def __init__(; symbols: __init__, _init_fused_moe_experts, forward, _select_dsv4_attn_cls, touching `__init__, _init_fused_moe_experts, forward`; `vllm/models/deepseek_v4/nvidia/dspark.py` modified +30/-4 (34 lines); hunks: -15,11 +15,13; -40,10 +42,16; symbols: __init__, forward, touching `__init__, forward`; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +21/-4 (25 lines); hunks: -18,11 +18,13; -44,6 +46,11; symbols: forward, __init__, touching `forward, __init__`; `vllm/models/kimi_k3/nvidia/model.py` modified +6/-6 (12 lines); hunks: -88,6 +88,12; -96,12 +102,6.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +53/-4 (57 lines); hunks: -65,6 +65,12; -515,13 +521,15 @@ def __init__(; symbols: __init__, _init_fused_moe_experts, forward, _select_dsv4_attn_cls
  - `vllm/models/deepseek_v4/nvidia/dspark.py` modified +30/-4 (34 lines); hunks: -15,11 +15,13; -40,10 +42,16; symbols: __init__, forward
  - `vllm/models/deepseek_v4/nvidia/mtp.py` modified +21/-4 (25 lines); hunks: -18,11 +18,13; -44,6 +46,11; symbols: forward, __init__
  - `vllm/models/kimi_k3/nvidia/model.py` modified +6/-6 (12 lines); hunks: -88,6 +88,12; -96,12 +102,6
  - `vllm/models/kimi_k3/nvidia/mtp.py` modified +5/-1 (6 lines); hunks: -27,6 +27,11; -38,7 +43,6
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -65,6 +65,12 @@
+from vllm.models.common.ops.sequence_parallel import (
+    sp_all_gather,
+    sp_padding_mask,
+    sp_reduce_scatter,
+    sp_shard,
+)
diff -- vllm/models/deepseek_v4/nvidia/dspark.py
@@ -15,11 +15,13 @@
+import vllm.envs as envs
+from vllm.forward_context import get_forward_context, is_forward_context_available
@@ -40,10 +42,16 @@
+from vllm.models.common.ops.sequence_parallel import (
+    sp_all_gather,
+    sp_padding_mask,
diff -- vllm/models/deepseek_v4/nvidia/mtp.py
@@ -18,11 +18,13 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +53/-4; `vllm/models/deepseek_v4/nvidia/dspark.py` modified +30/-4; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +21/-4; `vllm/models/kimi_k3/nvidia/model.py` modified +6/-6; `vllm/models/kimi_k3/nvidia/mtp.py` modified +5/-1
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_sequence_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50580 - [Frontend] DeepSeek V4 0731 reasoning effort prompts & mappings

- Link: https://github.com/vllm-project/vllm/pull/50580
- Status/date: merged / 2026-08-04
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`; associated commits `77434861904a`
- Diff scope read: GitHub Pull Request files API returned 5 files, +205/-51, 473 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] DeepSeek V4 0731 reasoning effort prompts & mappings"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`, `vllm/tokenizers/deepseek_v4.py`; technical summary: Covers "[Frontend] DeepSeek V4 0731 reasoning effort prompts & mappings"; the main implementation surface is `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`, `vllm/tokenizers/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tokenizers_/test_deepseek_v4.py` modified +47/-20 (67 lines); hunks: -76,13 +76,16 @@ def test_deepseek_v4_tokenizer_registered():; -93,7 +96,21 @@ def test_deepseek_v4_enables_thinking_with_compatible_kwargs(...; symbols: test_deepseek_v4_tokenizer_registered, test_deepseek_v4_defaults_to_chat_mode, test_deepseek_v4_defaults_to_thinking_with_high_effort, test_deepseek_v4_enables_thinking_with_compatible_kwargs, touching `test_deepseek_v4_tokenizer_registered, test_deepseek_v4_defaults_to_chat_mode, test_deepseek_v4_defaults_to_thinking_with_high_effort`; `vllm/tokenizers/deepseek_v4_encoding.py` modified +25/-11 (36 lines); hunks: -65,11 +65,20; -202,7 +211,8 @@ def render_message(index: int, messages: List[Dict[str, Any]...; symbols: render_message, encode_messages, touching `render_message, encode_messages`; `vllm/tokenizers/deepseek_v4.py` modified +10/-6 (16 lines); hunks: -29,10 +29,12 @@ def apply_chat_template(; -42,12 +44,14 @@ def apply_chat_template(; symbols: apply_chat_template, touching `apply_chat_template`.
- Code diff details:
  - `tests/tokenizers_/test_deepseek_v4.py` modified +47/-20 (67 lines); hunks: -76,13 +76,16 @@ def test_deepseek_v4_tokenizer_registered():; -93,7 +96,21 @@ def test_deepseek_v4_enables_thinking_with_compatible_kwargs(...; symbols: test_deepseek_v4_tokenizer_registered, test_deepseek_v4_defaults_to_chat_mode, test_deepseek_v4_defaults_to_thinking_with_high_effort, test_deepseek_v4_enables_thinking_with_compatible_kwargs
  - `vllm/tokenizers/deepseek_v4_encoding.py` modified +25/-11 (36 lines); hunks: -65,11 +65,20; -202,7 +211,8 @@ def render_message(index: int, messages: List[Dict[str, Any]...; symbols: render_message, encode_messages
  - `vllm/tokenizers/deepseek_v4.py` modified +10/-6 (16 lines); hunks: -29,10 +29,12 @@ def apply_chat_template(; -42,12 +44,14 @@ def apply_chat_template(; symbols: apply_chat_template
- Key code excerpts:

```diff
diff -- tests/tokenizers_/test_deepseek_v4.py
@@ -76,13 +76,16 @@ def test_deepseek_v4_tokenizer_registered():
-def test_deepseek_v4_defaults_to_chat_mode():
+def test_deepseek_v4_defaults_to_thinking_with_high_effort():
-    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>")
+    assert prompt.startswith(
+        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"
+    )
diff -- vllm/tokenizers/deepseek_v4_encoding.py
@@ -65,11 +65,20 @@
-REASONING_EFFORT_MAX = (
-    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
-    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential pat
-    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption
-)
+REASONING_EFFORT_PROMPTS: Dict[str, str] = {
diff -- vllm/tokenizers/deepseek_v4.py
@@ -29,10 +29,12 @@ def apply_chat_template(
```

- Reviewed files:
  - tests: `tests/tokenizers_/test_deepseek_v4.py` modified +47/-20
  - runtime: `vllm/tokenizers/deepseek_v4_encoding.py` modified +25/-11; `vllm/tokenizers/deepseek_v4.py` modified +10/-6
- Risk and verification: The diff ships test coverage in `tests/tokenizers_/test_deepseek_v4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #47972 - Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel

- Link: https://github.com/vllm-project/vllm/pull/47972
- Status/date: merged / 2026-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml`, `vllm/models/deepseek_v4/amd/model.py`; associated commits `da788334bc06`
- Diff scope read: GitHub Pull Request files API returned 11 files, +332/-17, 511 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/amd/model.py`, `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml`; technical summary: Covers "Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel"; the main implementation surface is `vllm/models/deepseek_v4/amd/model.py`, `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/model.py` modified +17/-4 (21 lines); hunks: -750,6 +750,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -803,10 +809,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights, _resolve_param_name, _make_deepseek_v4_weights_mapper, touching `load_weights, _resolve_param_name, _make_deepseek_v4_weights_mapper`; `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml` added +15/-0 (15 lines); hunks: -0,0 +1,15; `tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml` added +15/-0 (15 lines); hunks: -0,0 +1,15.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/model.py` modified +17/-4 (21 lines); hunks: -750,6 +750,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -803,10 +809,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights, _resolve_param_name, _make_deepseek_v4_weights_mapper
  - `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml` added +15/-0 (15 lines); hunks: -0,0 +1,15
  - `tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml` added +15/-0 (15 lines); hunks: -0,0 +1,15
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -750,6 +750,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        def _resolve_param_name(name: str) -> str:
+            inv_name = f"{name}_inv"
+            if name not in params_dict and inv_name in params_dict:
+                return inv_name
+            return name
@@ -803,10 +809,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
diff -- tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml
@@ -0,0 +1,15 @@
+model_name: "amd/DeepSeek-V4-Flash-NVFP4"
+accuracy_threshold: 0.92
+num_questions: 1319
+num_fewshot: 8
+startup_max_wait_seconds: 1800
+server_args: >-
diff -- tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml
@@ -0,0 +1,15 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` modified +17/-4
  - tests: `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml` added +15/-0; `tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml` added +15/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml`, `tests/evals/gsm8k/configs/models-gfx950-large.txt`, `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51430 - [Perf] Narrow DeepSeek V4 eager CUDA graph region

- Link: https://github.com/vllm-project/vllm/pull/51430
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `79c865b838e3`
- Diff scope read: GitHub Pull Request files API returned 3 files, +90/-97, 270 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Narrow DeepSeek V4 eager CUDA graph region"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[Perf] Narrow DeepSeek V4 eager CUDA graph region"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +88/-95 (183 lines); hunks: -362,11 +362,10 @@ def forward(; -377,16 +376,62 @@ def forward(; symbols: forward, project_query_and_cache_kv, _fused_wqa_wkv_gemm, touching `forward, project_query_and_cache_kv, _fused_wqa_wkv_gemm`; `vllm/models/deepseek_v4/amd/model.py` modified +1/-1 (2 lines); hunks: -551,7 +551,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, touching `__init__`; `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-1 (2 lines); hunks: -1018,7 +1018,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +88/-95 (183 lines); hunks: -362,11 +362,10 @@ def forward(; -377,16 +376,62 @@ def forward(; symbols: forward, project_query_and_cache_kv, _fused_wqa_wkv_gemm
  - `vllm/models/deepseek_v4/amd/model.py` modified +1/-1 (2 lines); hunks: -551,7 +551,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-1 (2 lines); hunks: -1018,7 +1018,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -362,11 +362,10 @@ def forward(
-        # Metadata-independent input GEMMs + RMSNorm stay in the captured
-        # graph; the metadata-dependent rest (q up-proj + kv-insert, indexer,
-        # compressor, MLA attention) runs in the eager break.
+        # Keep the attention input preparation in the captured graph. Only the
+        # sparse indexer and MLA attention run in the eager break below.
-            self.attn_gemm_parallel_execute(hidden_states)
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -551,7 +551,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        # DeepseekV4Attention.attn_gemm_parallel_execute
+        # DeepseekV4Attention._run_parallel_input_projections
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -1018,7 +1018,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        # DeepseekV4Attention.attn_gemm_parallel_execute
+        # DeepseekV4Attention._run_parallel_input_projections
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +88/-95; `vllm/models/deepseek_v4/amd/model.py` modified +1/-1; `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51296 - [Bugfix] Align deepseek v4 parser thinking default with tokenizer

- Link: https://github.com/vllm-project/vllm/pull/51296
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_deepseek_v4.py`, `vllm/parser/deepseek_v4.py`; associated commits `3f142bd85e96`
- Diff scope read: GitHub Pull Request files API returned 3 files, +57/-12, 99 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Align deepseek v4 parser thinking default with tokenizer"; model line: DeepSeek V4; category: bug fix; main diff: `tests/parser/engine/test_deepseek_v4.py`, `vllm/parser/deepseek_v4.py`; technical summary: Covers "[Bugfix] Align deepseek v4 parser thinking default with tokenizer"; the main implementation surface is `tests/parser/engine/test_deepseek_v4.py`, `vllm/parser/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/parser/engine/test_deepseek_v4.py` modified +51/-8 (59 lines); hunks: -243,15 +243,29 @@ def test_thinking_false_starts_in_content(self):; -838,6 +852,35 @@ def test_tool_calls_extracted_at_all_chunk_sizes(; symbols: test_thinking_false_starts_in_content, test_enable_thinking_kwarg, test_parser_thinking_mode_matches_tokenizer_default, test_no_thinking_kwarg_defaults_to_content, touching `test_thinking_false_starts_in_content, test_enable_thinking_kwarg, test_parser_thinking_mode_matches_tokenizer_default`; `vllm/parser/deepseek_v4.py` modified +5/-3 (8 lines); hunks: -217,10 +217,12 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tests/parser/engine/test_deepseek_v4.py` modified +51/-8 (59 lines); hunks: -243,15 +243,29 @@ def test_thinking_false_starts_in_content(self):; -838,6 +852,35 @@ def test_tool_calls_extracted_at_all_chunk_sizes(; symbols: test_thinking_false_starts_in_content, test_enable_thinking_kwarg, test_parser_thinking_mode_matches_tokenizer_default, test_no_thinking_kwarg_defaults_to_content
  - `vllm/parser/deepseek_v4.py` modified +5/-3 (8 lines); hunks: -217,10 +217,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tests/parser/engine/test_deepseek_v4.py
@@ -243,15 +243,29 @@ def test_thinking_false_starts_in_content(self):
-    def test_enable_thinking_kwarg(self, mock_tokenizer):
-        p = DeepSeekV4Parser(
-            mock_tokenizer, chat_template_kwargs={"enable_thinking": True}
+    @pytest.mark.parametrize(
+        ("chat_template_kwargs", "expected_state"),
+        [
diff -- vllm/parser/deepseek_v4.py
@@ -217,10 +217,12 @@ def __init__(
-        thinking = (
-            bool(chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking"))
-            and chat_kwargs.get("reasoning_effort") != "none"
+        thinking = bool(
+            chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking")
+        if "thinking" not in chat_kwargs and "enable_thinking" not in chat_kwargs:
```

- Reviewed files:
  - tests: `tests/parser/engine/test_deepseek_v4.py` modified +51/-8
  - runtime: `vllm/parser/deepseek_v4.py` modified +5/-3
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_deepseek_v4.py`, `tests/parser/engine/trace_builder.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51727 - [Bugfix] Fix DeepSeek V4/3.2 tokenizer vocab size overcount crashing guided decoding

- Link: https://github.com/vllm-project/vllm/pull/51727
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/deepseek_v4.py`; associated commits `8bcc916a98a9`
- Diff scope read: GitHub Pull Request files API returned 2 files, +0/-11, 39 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DeepSeek V4/3.2 tokenizer vocab size overcount crashing guided decoding"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/tokenizers/deepseek_v4.py`; technical summary: Covers "[Bugfix] Fix DeepSeek V4/3.2 tokenizer vocab size overcount crashing guided decoding"; the main implementation surface is `vllm/tokenizers/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tokenizers/deepseek_v4.py` modified +0/-5 (5 lines); hunks: -19,8 +19,6 @@ def get_deepseek_v4_tokenizer(tokenizer: HfTokenizer) -> HfTok...; -78,9 +76,6 @@ def apply_chat_template(; symbols: get_deepseek_v4_tokenizer, _DeepseekV4Tokenizer, apply_chat_template, num_special_tokens_to_add, touching `get_deepseek_v4_tokenizer, _DeepseekV4Tokenizer, apply_chat_template`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v4.py` modified +0/-5 (5 lines); hunks: -19,8 +19,6 @@ def get_deepseek_v4_tokenizer(tokenizer: HfTokenizer) -> HfTok...; -78,9 +76,6 @@ def apply_chat_template(; symbols: get_deepseek_v4_tokenizer, _DeepseekV4Tokenizer, apply_chat_template, num_special_tokens_to_add
- Key code excerpts:

```diff
diff -- vllm/tokenizers/deepseek_v4.py
@@ -19,8 +19,6 @@ def get_deepseek_v4_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
-    added_vocab_size = len(added_vocab)
-    tokenizer_vocab_size = tokenizer.vocab_size
@@ -78,9 +76,6 @@ def apply_chat_template(
-        def __len__(self) -> int:
-            return tokenizer_vocab_size + added_vocab_size
```

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v4.py` modified +0/-5
- Risk and verification: Runtime changes concentrate in `vllm/tokenizers/deepseek_v32.py`, `vllm/tokenizers/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51145 - [Bugfix][ROCm] Fix DeepSeek V4 DSpark probabilistic startup

- Link: https://github.com/vllm-project/vllm/pull/51145
- Status/date: merged / 2026-08-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/dspark.py`; associated commits `12bea3eedcdf`
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][ROCm] Fix DeepSeek V4 DSpark probabilistic startup"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/amd/dspark.py`; technical summary: Covers "[Bugfix][ROCm] Fix DeepSeek V4 DSpark probabilistic startup"; the main implementation surface is `vllm/models/deepseek_v4/amd/dspark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/dspark.py` modified +2/-0 (2 lines); hunks: -290,6 +290,8 @@ class DSparkDeepseekV4ForCausalLM(nn.Module):; symbols: DSparkDeepseekV4ForCausalLM, __init__, touching `DSparkDeepseekV4ForCausalLM, __init__`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/dspark.py` modified +2/-0 (2 lines); hunks: -290,6 +290,8 @@ class DSparkDeepseekV4ForCausalLM(nn.Module):; symbols: DSparkDeepseekV4ForCausalLM, __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/dspark.py
@@ -290,6 +290,8 @@ class DSparkDeepseekV4ForCausalLM(nn.Module):
+    # Full-vocab draft: draft ids are target ids, no remapping needed.
+    draft_id_to_target_id = None
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/dspark.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/dspark.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51821 - [Bugfix][ROCm][CI] Restore the DeepSeek-V4 input GEMM override point

- Link: https://github.com/vllm-project/vllm/pull/51821
- Status/date: merged / 2026-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`; associated commits `b369f10d5c5d`
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-1, 23 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][ROCm][CI] Restore the DeepSeek-V4 input GEMM override point"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[Bugfix][ROCm][CI] Restore the DeepSeek-V4 input GEMM override point"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +8/-1 (9 lines); hunks: -443,6 +443,13 @@ def project_query_and_cache_kv() -> torch.Tensor:; -494,7 +501,7 @@ def indexer_compressor_kv_score() -> torch.Tensor:; symbols: project_query_and_cache_kv, _fused_wqa_wkv_gemm, _run_parallel_input_projections, indexer_compressor_kv_score, touching `project_query_and_cache_kv, _fused_wqa_wkv_gemm, _run_parallel_input_projections`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +8/-1 (9 lines); hunks: -443,6 +443,13 @@ def project_query_and_cache_kv() -> torch.Tensor:; -494,7 +501,7 @@ def indexer_compressor_kv_score() -> torch.Tensor:; symbols: project_query_and_cache_kv, _fused_wqa_wkv_gemm, _run_parallel_input_projections, indexer_compressor_kv_score
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -443,6 +443,13 @@ def project_query_and_cache_kv() -> torch.Tensor:
+    def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
+        # Override point: the ROCm layer preshuffles this weight in place, so
+        # it cannot go through fused_wqa_wkv directly.
+        # MergedColumnParallelLinear returns (output, bias); bias is None.
+        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
+        return qr_kv
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +8/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51538 - [Bugfix] Make DSV4 sparse MLA work end-to-end for plain decode, MTP, and DSpark

- Link: https://github.com/vllm-project/vllm/pull/51538
- Status/date: merged / 2026-08-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`; associated commits `97388c44f9c6`
- Diff scope read: GitHub Pull Request files API returned 20 files, +797/-120, 1367 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Make DSV4 sparse MLA work end-to-end for plain decode, MTP, and DSpark"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/amd/rocm.py`; technical summary: Covers "[Bugfix] Make DSV4 sparse MLA work end-to-end for plain decode, MTP, and DSpark"; the main implementation surface is `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/amd/rocm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +20/-17 (37 lines); hunks: -839,16 +839,17 @@ def build_flashinfer_mixed_sparse_indices(; -897,7 +898,7 @@ def build_flashinfer_mixed_sparse_indices(; symbols: build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel, touching `build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel`; `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +28/-7 (35 lines); hunks: -6,6 +6,7; -25,6 +26,9; symbols: _pad_to_supported_q_heads, _required_sm120_sparse_topk, DeepseekV4FlashInferMLASparseBackend, _build_sparse_index_metadata, touching `_pad_to_supported_q_heads, _required_sm120_sparse_topk, DeepseekV4FlashInferMLASparseBackend`; `vllm/models/deepseek_v4/amd/rocm.py` modified +4/-4 (8 lines); hunks: -414,7 +414,9 @@ def build(; -423,9 +425,7 @@ def build(; symbols: build, touching `build`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +20/-17 (37 lines); hunks: -839,16 +839,17 @@ def build_flashinfer_mixed_sparse_indices(; -897,7 +898,7 @@ def build_flashinfer_mixed_sparse_indices(; symbols: build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +28/-7 (35 lines); hunks: -6,6 +6,7; -25,6 +26,9; symbols: _pad_to_supported_q_heads, _required_sm120_sparse_topk, DeepseekV4FlashInferMLASparseBackend, _build_sparse_index_metadata
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +4/-4 (8 lines); hunks: -414,7 +414,9 @@ def build(; -423,9 +425,7 @@ def build(; symbols: build
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/cache_utils.py
@@ -839,16 +839,17 @@ def build_flashinfer_mixed_sparse_indices(
-    Produces ``sparse_indices`` of shape ``[num_tokens, window_size +
-    padded_topk]`` (the first ``window_size`` columns are SWA slot ids, the rest
-    are compressed/top-k slot ids) and ``sparse_topk_lens`` (active length per
-    token). Decode tokens read precomputed SWA/compressed indices; prefill tokens
-    derive their SWA window from the position and translate local compressed
-    indices to global slots via the block tables.
diff -- vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
@@ -6,6 +6,7 @@
+from vllm.config import VllmConfig
@@ -25,6 +26,9 @@
+from vllm.v1.attention.backends.mla.compressor_utils import (
+    get_dspark_swa_index_width,
+)
@@ -74,6 +78,19 @@ def _pad_to_supported_q_heads(num_heads: int) -> int:
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -414,7 +414,9 @@ def build(
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +20/-17; `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +28/-7; `vllm/models/deepseek_v4/amd/rocm.py` modified +4/-4
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_flashmla_sparse.py`, `tests/kernels/moe/test_ocp_mx_moe.py`, `tests/kernels/test_compressor_kv_cache.py`, `tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51318 - [Bugfix][DSv4] Revert adaptive C128A metadata packing

- Link: https://github.com/vllm-project/vllm/pull/51318
- Status/date: merged / 2026-08-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/sparse_mla.py`; associated commits `edd4c8176cfd`
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-56, 104 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][DSv4] Revert adaptive C128A metadata packing"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/sparse_mla.py`; technical summary: Covers "[Bugfix][DSv4] Revert adaptive C128A metadata packing"; the main implementation surface is `vllm/models/deepseek_v4/sparse_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/sparse_mla.py` modified +7/-19 (26 lines); hunks: -257,13 +257,6 @@ def _build_c128a_metadata(; -276,7 +269,7 @@ def _build_c128a_metadata(; symbols: _build_c128a_metadata, build_c128a_topk_metadata, touching `_build_c128a_metadata, build_c128a_topk_metadata`.
- Code diff details:
  - `vllm/models/deepseek_v4/sparse_mla.py` modified +7/-19 (26 lines); hunks: -257,13 +257,6 @@ def _build_c128a_metadata(; -276,7 +269,7 @@ def _build_c128a_metadata(; symbols: _build_c128a_metadata, build_c128a_topk_metadata
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/sparse_mla.py
@@ -257,13 +257,6 @@ def _build_c128a_metadata(
-        active_topk_width = min(
-            max(
-                triton.next_power_of_2(max(cm.max_seq_len // self.compress_ratio, 1)),
-                _C128A_TOPK_ALIGNMENT,
-            ),
-            self.c128a_max_compressed,
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/sparse_mla.py` modified +7/-19
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_flashmla_sparse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52401 - [Bugfix] Pick the DeepSeek V4 eager cudagraph region per model runner

- Link: https://github.com/vllm-project/vllm/pull/52401
- Status/date: merged / 2026-08-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`; associated commits `8efa13b700f1`
- Diff scope read: GitHub Pull Request files API returned 3 files, +96/-66, 227 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Pick the DeepSeek V4 eager cudagraph region per model runner"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[Bugfix] Pick the DeepSeek V4 eager cudagraph region per model runner"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +65/-4 (69 lines); hunks: -298,6 +298,13 @@ def __init__(; -379,6 +386,64 @@ def forward(; symbols: __init__, forward, _prepare_and_attn_eager, _prepare_and_attn, touching `__init__, forward, _prepare_and_attn_eager`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +65/-4 (69 lines); hunks: -298,6 +298,13 @@ def __init__(; -379,6 +386,64 @@ def forward(; symbols: __init__, forward, _prepare_and_attn_eager, _prepare_and_attn
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -298,6 +298,13 @@ def __init__(
+        self._prepare_and_attn_fn = self._prepare_and_attn
+        if not vllm_config.use_v2_model_runner:
+            # MRV1's piecewise capture only tolerates the wide eager region: with
+            # the narrow one the attention input preparation stays in the captured
+            # graph and MRV1 produces garbage (#51430).
+            self._prepare_and_attn_fn = self._prepare_and_attn_eager
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +65/-4
- Risk and verification: The diff ships test coverage in `tests/test_config.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52084 - [Perf][DSV4] Optimize sparse top-k metadata kernels for higher prefill throughput

- Link: https://github.com/vllm-project/vllm/pull/52084
- Status/date: merged / 2026-08-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/ops/cache_utils.py`; associated commits `836aac92ffda`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf][DSV4] Optimize sparse top-k metadata kernels for higher prefill throughput"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/common/ops/cache_utils.py`; technical summary: Covers "[Perf][DSV4] Optimize sparse top-k metadata kernels for higher prefill throughput"; the main implementation surface is `vllm/models/deepseek_v4/common/ops/cache_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +1/-1 (2 lines); hunks: -580,7 +580,7 @@ def combine_topk_swa_indices(; symbols: combine_topk_swa_indices, touching `combine_topk_swa_indices`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +1/-1 (2 lines); hunks: -580,7 +580,7 @@ def combine_topk_swa_indices(; symbols: combine_topk_swa_indices
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/cache_utils.py
@@ -580,7 +580,7 @@ def combine_topk_swa_indices(
-_COMBINE_TOPK_SWA_NUM_WORKERS = 128
+_COMBINE_TOPK_SWA_NUM_WORKERS = 256
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/common/ops/cache_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51967 - [Perf][DSV4] Optimize global top-k index kernel with compile-time constants

- Link: https://github.com/vllm-project/vllm/pull/51967
- Status/date: merged / 2026-08-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/ops/cache_utils.py`; associated commits `83f591d7f694`
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-5, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf][DSV4] Optimize global top-k index kernel with compile-time constants"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/common/ops/cache_utils.py`; technical summary: Covers "[Perf][DSV4] Optimize global top-k index kernel with compile-time constants"; the main implementation surface is `vllm/models/deepseek_v4/common/ops/cache_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +5/-5 (10 lines); hunks: -477,15 +477,15 @@ def compute_global_topk_indices_and_lens(; symbols: compute_global_topk_indices_and_lens, _compute_global_topk_indices_and_lens_kernel, touching `compute_global_topk_indices_and_lens, _compute_global_topk_indices_and_lens_kernel`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +5/-5 (10 lines); hunks: -477,15 +477,15 @@ def compute_global_topk_indices_and_lens(; symbols: compute_global_topk_indices_and_lens, _compute_global_topk_indices_and_lens_kernel
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/cache_utils.py
@@ -477,15 +477,15 @@ def compute_global_topk_indices_and_lens(
-    global_topk_indices_stride,
+    global_topk_indices_stride: tl.constexpr,
-    topk_indices_stride,
-    topk,
+    topk_indices_stride: tl.constexpr,
+    topk: tl.constexpr,
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +5/-5
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/common/ops/cache_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #52212 - [ROCm][DSV4][Perf] Optimize Triton sparse-MLA decode on gfx950

- Link: https://github.com/vllm-project/vllm/pull/52212
- Status/date: merged / 2026-08-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`; associated commits `ef43e3101b8f`
- Diff scope read: GitHub Pull Request files API returned 5 files, +1550/-94, 2081 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][DSV4][Perf] Optimize Triton sparse-MLA decode on gfx950"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`; technical summary: Covers "[ROCm][DSV4][Perf] Optimize Triton sparse-MLA decode on gfx950"; the main implementation surface is `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/rocm.py` modified +48/-3 (51 lines); hunks: -19,6 +19,7; -36,6 +37,19; symbols: _trust_dsv4_extra_cache_nan_free, _build_indptr_from_lengths, _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata, touching `_trust_dsv4_extra_cache_nan_free, _build_indptr_from_lengths, _copy_ragged_to_graph_buffers`; `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +21/-2 (23 lines); hunks: -24,8 +24,14; -61,12 +67,15 @@ def compress_norm_rope_store_triton(; symbols: compress_norm_rope_store_triton, _fused_kv_compress_norm_rope_insert_sparse_attn, touching `compress_norm_rope_store_triton, _fused_kv_compress_norm_rope_insert_sparse_attn`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +48/-3 (51 lines); hunks: -19,6 +19,7; -36,6 +37,19; symbols: _trust_dsv4_extra_cache_nan_free, _build_indptr_from_lengths, _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata
  - `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +21/-2 (23 lines); hunks: -24,8 +24,14; -61,12 +67,15 @@ def compress_norm_rope_store_triton(; symbols: compress_norm_rope_store_triton, _fused_kv_compress_norm_rope_insert_sparse_attn
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -19,6 +19,7 @@
+from vllm.platforms.rocm import _ON_GFX950
@@ -36,6 +37,19 @@
+def _trust_dsv4_extra_cache_nan_free(
+    kv_cache_dtype: str,
+    has_kv_transfer: bool,
+    has_extra_cache: bool,
diff -- vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
@@ -24,8 +24,14 @@
+from vllm.platforms import current_platform
+if current_platform.is_rocm():
+    from vllm.platforms.rocm import _ON_GFX950
+else:
+    _ON_GFX950 = False
@@ -61,12 +67,15 @@ def compress_norm_rope_store_triton(
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/rocm.py` modified +48/-3; `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +21/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`, `tests/kernels/test_compressor_kv_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52492 - [Bugfix][DSv4] Keep indexer scoring in breakable graphs

- Link: https://github.com/vllm-project/vllm/pull/52492
- Status/date: merged / 2026-08-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`; associated commits `292187dd8ca1`
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-1, 12 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][DSv4] Keep indexer scoring in breakable graphs"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[Bugfix][DSv4] Keep indexer scoring in breakable graphs"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +4/-1 (5 lines); hunks: -905,7 +905,10 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +4/-1 (5 lines); hunks: -905,7 +905,10 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -905,7 +905,10 @@ def forward(
-            if indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens:
+            if (
+                indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens
+                and not torch.cuda.is_current_stream_capturing()
+            ):
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #52626 - [Bugfix] Fix DeepSeek V4 mHC broadcast buffer for weight sync

- Link: https://github.com/vllm-project/vllm/pull/52626
- Status/date: merged / 2026-08-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `d5f5de7a7db4`
- Diff scope read: GitHub Pull Request files API returned 2 files, +66/-1, 97 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DeepSeek V4 mHC broadcast buffer for weight sync"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "[Bugfix] Fix DeepSeek V4 mHC broadcast buffer for weight sync"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-1 (6 lines); hunks: -1372,11 +1372,15 @@ def finalize_mhc_broadcast_weights(self) -> None:; symbols: finalize_mhc_broadcast_weights, _make_deepseek_v4_weights_mapper, touching `finalize_mhc_broadcast_weights, _make_deepseek_v4_weights_mapper`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-1 (6 lines); hunks: -1372,11 +1372,15 @@ def finalize_mhc_broadcast_weights(self) -> None:; symbols: finalize_mhc_broadcast_weights, _make_deepseek_v4_weights_mapper
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -1372,11 +1372,15 @@ def finalize_mhc_broadcast_weights(self) -> None:
-            layer.hc_attn_fn_broadcast = (
+            broadcast = (
+            if layer.hc_attn_fn_broadcast is None:
+                layer.hc_attn_fn_broadcast = broadcast
+            else:
+                layer.hc_attn_fn_broadcast.copy_(broadcast)
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51368 - [Bugfix] Fix DeepSeek V4 mHC broadcast buffer for dummy load

- Link: https://github.com/vllm-project/vllm/pull/51368
- Status/date: merged / 2026-08-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`; associated commits `a9f4afb66f77`
- Diff scope read: GitHub Pull Request files API returned 4 files, +46/-7, 117 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DeepSeek V4 mHC broadcast buffer for dummy load"; model line: DeepSeek V4; category: bug fix; main diff: `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/dspark.py`; technical summary: Covers "[Bugfix] Fix DeepSeek V4 mHC broadcast buffer for dummy load"; the main implementation surface is `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/dspark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/test_deepseek_v4_mega_moe.py` modified +34/-0 (34 lines); hunks: -9,10 +9,13; -243,6 +246,37 @@ def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwis...; symbols: test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, test_deepseek_v4_pwal_hook_finalizes_mega_moe_and_mhc_broadcast, test_deepseek_v4_drafter_pwal_hooks_finalize_mega_moe, touching `test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, test_deepseek_v4_pwal_hook_finalizes_mega_moe_and_mhc_broadcast, test_deepseek_v4_drafter_pwal_hooks_finalize_mega_moe`; `vllm/models/deepseek_v4/nvidia/model.py` modified +4/-5 (9 lines); hunks: -504,10 +504,6 @@ def forward(; -1543,9 +1539,12 @@ def get_mtp_target_hidden_states(self) -> torch.Tensor |...; symbols: forward, get_mtp_target_hidden_states, load_weights, process_weights_after_loading, touching `forward, get_mtp_target_hidden_states, load_weights`; `vllm/models/deepseek_v4/nvidia/dspark.py` modified +4/-1 (5 lines); hunks: -504,16 +504,19 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights, _finalize_moe, process_weights_after_loading, _remap_dspark_name, touching `load_weights, _finalize_moe, process_weights_after_loading`; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +4/-1 (5 lines); hunks: -502,14 +502,17 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx, finalize_mega_moe_weights, process_weights_after_loading, _rewrite_spec_layer_name, touching `_find_mtp_layer_idx, finalize_mega_moe_weights, process_weights_after_loading`.
- Code diff details:
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +34/-0 (34 lines); hunks: -9,10 +9,13; -243,6 +246,37 @@ def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwis...; symbols: test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, test_deepseek_v4_pwal_hook_finalizes_mega_moe_and_mhc_broadcast, test_deepseek_v4_drafter_pwal_hooks_finalize_mega_moe
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +4/-5 (9 lines); hunks: -504,10 +504,6 @@ def forward(; -1543,9 +1539,12 @@ def get_mtp_target_hidden_states(self) -> torch.Tensor |...; symbols: forward, get_mtp_target_hidden_states, load_weights, process_weights_after_loading
  - `vllm/models/deepseek_v4/nvidia/dspark.py` modified +4/-1 (5 lines); hunks: -504,16 +504,19 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights, _finalize_moe, process_weights_after_loading, _remap_dspark_name
  - `vllm/models/deepseek_v4/nvidia/mtp.py` modified +4/-1 (5 lines); hunks: -502,14 +502,17 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx, finalize_mega_moe_weights, process_weights_after_loading, _rewrite_spec_layer_name
- Key code excerpts:

```diff
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -9,10 +9,13 @@
+from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4ForCausalLM
+    DeepseekV4ForCausalLM,
+from vllm.models.deepseek_v4.nvidia.mtp import DeepSeekV4MTP
@@ -243,6 +246,37 @@ def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact():
+def test_deepseek_v4_pwal_hook_finalizes_mega_moe_and_mhc_broadcast():
+    """The loader invokes the model-level PWAL hook for every load format,
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -504,10 +504,6 @@ def forward(
-        # This method must have been already called during the weight loading phase.
-        # We call it again here to cover the dummy weight loading case.
-        self.finalize_weights()
@@ -1543,9 +1539,12 @@ def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
+        self.process_weights_after_loading()
+        return loaded_params
diff -- vllm/models/deepseek_v4/nvidia/dspark.py
@@ -504,16 +504,19 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- Reviewed files:
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +34/-0
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +4/-5; `vllm/models/deepseek_v4/nvidia/dspark.py` modified +4/-1; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +4/-1
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_mega_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52836 - Revert DSv4 eager workspace reuse

- Link: https://github.com/vllm-project/vllm/pull/52836
- Status/date: merged / 2026-08-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/compressor.py` and 9 files; associated commits `f1178f3a06fa`
- Diff scope read: GitHub Pull Request files API returned 15 files, +30/-354, 708 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert DSv4 eager workspace reuse"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`; technical summary: Covers "Revert DSv4 eager workspace reuse"; the main implementation surface is `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +12/-30 (42 lines); hunks: -295,7 +295,6 @@ def fused_indexer_q_rope_quant(; -333,37 +332,24 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant, touching `fused_indexer_q_rope_quant`; `vllm/models/deepseek_v4/attention.py` modified +2/-35 (37 lines); hunks: -29,7 +29,6; -185,7 +184,6 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/models/deepseek_v4/nvidia/model.py` modified +0/-20 (20 lines); hunks: -72,7 +72,6; -819,7 +818,6 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +5/-10 (15 lines); hunks: -2097,7 +2097,6 @@ def compress_norm_rope_store_cutedsl(; -2130,15 +2129,11 @@ def compress_norm_rope_store_cutedsl(; symbols: compress_norm_rope_store_cutedsl, touching `compress_norm_rope_store_cutedsl`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +12/-30 (42 lines); hunks: -295,7 +295,6 @@ def fused_indexer_q_rope_quant(; -333,37 +332,24 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant
  - `vllm/models/deepseek_v4/attention.py` modified +2/-35 (37 lines); hunks: -29,7 +29,6; -185,7 +184,6 @@ def __init__(; symbols: __init__, forward
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +0/-20 (20 lines); hunks: -72,7 +72,6; -819,7 +818,6 @@ def __init__(; symbols: __init__
  - `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +5/-10 (15 lines); hunks: -2097,7 +2097,6 @@ def compress_norm_rope_store_cutedsl(; -2130,15 +2129,11 @@ def compress_norm_rope_store_cutedsl(; symbols: compress_norm_rope_store_cutedsl
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +2/-10 (12 lines); hunks: -438,7 +438,6 @@ def compute_global_topk_indices_and_lens(; -448,15 +447,8 @@ def compute_global_topk_indices_and_lens(; symbols: compute_global_topk_indices_and_lens
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
@@ -295,7 +295,6 @@ def fused_indexer_q_rope_quant(
-    output_buffers: tuple[torch.Tensor, ...] | None = None,
@@ -333,37 +332,24 @@ def fused_indexer_q_rope_quant(
-    if output_buffers is None:
-        index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)
-    else:
-        expected_num_buffers = 3 if use_fp4 else 2
diff -- vllm/models/deepseek_v4/attention.py
@@ -29,7 +29,6 @@
-    from vllm.models.deepseek_v4.eager_scratch import DeepseekV4EagerScratchPool
@@ -185,7 +184,6 @@ def __init__(
-        eager_scratch_pool: "DeepseekV4EagerScratchPool | None" = None,
@@ -274,7 +272,6 @@ def __init__(
-        self.eager_scratch_pool = eager_scratch_pool
@@ -296,7 +293,6 @@ def __init__(
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -72,7 +72,6 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +12/-30; `vllm/models/deepseek_v4/attention.py` modified +2/-35; `vllm/models/deepseek_v4/nvidia/model.py` modified +0/-20; `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +5/-10; `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +2/-10; `vllm/models/deepseek_v4/compressor.py` modified +1/-10
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52737 - [ROCm][Perf] Fuse DeepSeek-V4 mHC post/pre and RMSNorm with AITER

- Link: https://github.com/vllm-project/vllm/pull/52737
- Status/date: merged / 2026-08-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/dspark.py`, `vllm/models/deepseek_v4/amd/model.py`; associated commits `d626108b1841`
- Diff scope read: GitHub Pull Request files API returned 5 files, +272/-10, 450 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Perf] Fuse DeepSeek-V4 mHC post/pre and RMSNorm with AITER"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/dspark.py`; technical summary: Covers "[ROCm][Perf] Fuse DeepSeek-V4 mHC post/pre and RMSNorm with AITER"; the main implementation surface is `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/dspark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/model.py` modified +44/-6 (50 lines); hunks: -305,6 +305,10 @@ def forward(; -384,17 +388,32 @@ def __init__(; symbols: forward, DeepseekV4DecoderLayer, __init__, hc_pre, touching `forward, DeepseekV4DecoderLayer, __init__`; `vllm/models/deepseek_v4/amd/dspark.py` modified +2/-2 (4 lines); hunks: -9,8 +9,8.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/model.py` modified +44/-6 (50 lines); hunks: -305,6 +305,10 @@ def forward(; -384,17 +388,32 @@ def __init__(; symbols: forward, DeepseekV4DecoderLayer, __init__, hc_pre
  - `vllm/models/deepseek_v4/amd/dspark.py` modified +2/-2 (4 lines); hunks: -9,8 +9,8
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -305,6 +305,10 @@ def forward(
+# Hidden sizes supported by AITER mhc_pre_big_fuse_rmsnorm.
+_AITER_MHC_FUSED_RMSNORM_SIZES = frozenset({1280, 2560, 4096, 7168})
@@ -384,17 +388,32 @@ def __init__(
-        self.use_fused_mhc = HAS_TILELANG_MHC and not (
-            HAS_AITER_MHC and self.hidden_size % 256 == 0
+        # AITER mhc kernels (pre/post/fused) require hc_mult == 4.
diff -- vllm/models/deepseek_v4/amd/dspark.py
@@ -9,8 +9,8 @@
-    and gate the trailing ``mhc_post`` on ``use_fused_mhc`` (False on the aiter
-    path, where the decoder layer already applies hc_post in-layer);
+    and gate the trailing ``mhc_post`` on ``use_fused_mhc`` (True when AITER
+    or TileLang fused MHC is available; False only on the torch fallback);
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` modified +44/-6; `vllm/models/deepseek_v4/amd/dspark.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/_aiter_ops.py`, `vllm/model_executor/kernels/mhc/aiter.py`, `vllm/model_executor/layers/mhc.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50803 - [ROCm] Fix DeepSeek V4 indexer numerics and coverage

- Link: https://github.com/vllm-project/vllm/pull/50803
- Status/date: merged / 2026-08-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`; associated commits `6b68db441e76`
- Diff scope read: GitHub Pull Request files API returned 3 files, +42/-4, 88 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Fix DeepSeek V4 indexer numerics and coverage"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`; technical summary: Covers "[ROCm] Fix DeepSeek V4 indexer numerics and coverage"; the main implementation surface is `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +9/-2 (11 lines); hunks: -91,6 +91,7 @@ def _fused_indexer_q_rope_quant_kernel(; -118,8 +119,13 @@ def _fused_indexer_q_rope_quant_kernel(; symbols: _fused_indexer_q_rope_quant_kernel, fused_indexer_q_rope_quant, touching `_fused_indexer_q_rope_quant_kernel, fused_indexer_q_rope_quant`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +9/-2 (11 lines); hunks: -91,6 +91,7 @@ def _fused_indexer_q_rope_quant_kernel(; -118,8 +119,13 @@ def _fused_indexer_q_rope_quant_kernel(; symbols: _fused_indexer_q_rope_quant_kernel, fused_indexer_q_rope_quant
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
@@ -91,6 +91,7 @@ def _fused_indexer_q_rope_quant_kernel(
+    USE_EXPLICIT_FMA: tl.constexpr = False,
@@ -118,8 +119,13 @@ def _fused_indexer_q_rope_quant_kernel(
-    r_even = x_even * cos - x_odd * sin
-    r_odd = x_odd * cos + x_even * sin
+    if USE_EXPLICIT_FMA:
+        # Match HIP rotary_embedding contraction before bf16 materialization.
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +9/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53040 - [DSV4][Kernel] Fuse shared experts into MegaMoE

- Link: https://github.com/vllm-project/vllm/pull/53040
- Status/date: merged / 2026-08-20
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`; associated commits `4f6885fffc93`
- Diff scope read: GitHub Pull Request files API returned 5 files, +569/-44, 848 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4][Kernel] Fuse shared experts into MegaMoE"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/nvidia/model.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`; technical summary: Covers "[DSV4][Kernel] Fuse shared experts into MegaMoE"; the main implementation surface is `vllm/models/deepseek_v4/nvidia/model.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/model.py` modified +274/-42 (316 lines); hunks: -2,6 +2,7; -18,6 +19,7; symbols: DeepseekV4MLP, __init__, make_deepseek_v4_expert_params_mapping, DeepseekV4MegaMoEExperts, touching `DeepseekV4MLP, __init__, make_deepseek_v4_expert_params_mapping`; `tests/models/test_deepseek_v4_mega_moe.py` modified +232/-0 (232 lines); hunks: -13,6 +13,7; -168,6 +169,149 @@ def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert...; symbols: test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights, FakeDeepGemm, get_symm_buffer_for_mega_moe, touching `test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights, FakeDeepGemm`; `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +51/-0 (51 lines); hunks: -17,6 +17,7 @@ def _prepare_megamoe_inputs_kernel(; -28,6 +29,8 @@ def _prepare_megamoe_inputs_kernel(; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs, touching `_prepare_megamoe_inputs_kernel, prepare_megamoe_inputs`; `vllm/models/kimi_k3/nvidia/model.py` modified +5/-2 (7 lines); hunks: -93,7 +93,10; -354,7 +357,7 @@ def synchronize_first_launch(self) -> None:; symbols: synchronize_first_launch, finalize_weights, touching `synchronize_first_launch, finalize_weights`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +274/-42 (316 lines); hunks: -2,6 +2,7; -18,6 +19,7; symbols: DeepseekV4MLP, __init__, make_deepseek_v4_expert_params_mapping, DeepseekV4MegaMoEExperts
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +232/-0 (232 lines); hunks: -13,6 +13,7; -168,6 +169,149 @@ def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert...; symbols: test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights, FakeDeepGemm, get_symm_buffer_for_mega_moe
  - `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +51/-0 (51 lines); hunks: -17,6 +17,7 @@ def _prepare_megamoe_inputs_kernel(; -28,6 +29,8 @@ def _prepare_megamoe_inputs_kernel(; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs
  - `vllm/models/kimi_k3/nvidia/model.py` modified +5/-2 (7 lines); hunks: -93,7 +93,10; -354,7 +357,7 @@ def synchronize_first_launch(self) -> None:; symbols: synchronize_first_launch, finalize_weights
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -2,6 +2,7 @@
+from inspect import signature
@@ -18,6 +19,7 @@
+from vllm.logger import init_logger
@@ -84,6 +86,8 @@
+logger = init_logger(__name__)
@@ -165,7 +169,7 @@ def make_deepseek_v4_expert_params_mapping(
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -13,6 +13,7 @@
+    DeepseekV4MoE,
@@ -168,6 +169,149 @@ def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership():
+def test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights(monkeypatch):
+    class FakeDeepGemm:
+        transformed_dims: list[tuple[int, int]] = []
+        scale_inputs: list[tuple[int, ...]] = []
diff -- vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py
@@ -17,6 +17,7 @@ def _prepare_megamoe_inputs_kernel(
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +274/-42; `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +51/-0; `vllm/models/kimi_k3/nvidia/model.py` modified +5/-2
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +232/-0
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_mega_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52882 - [ROCm][Perf] Optimize DeepSeek V4 C4A top-k with AITER

- Link: https://github.com/vllm-project/vllm/pull/52882
- Status/date: merged / 2026-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`; associated commits `fe76112ff298`
- Diff scope read: GitHub Pull Request files API returned 5 files, +612/-31, 781 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Perf] Optimize DeepSeek V4 C4A top-k with AITER"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[ROCm][Perf] Optimize DeepSeek V4 C4A top-k with AITER"; the main implementation surface is `vllm/models/deepseek_v4/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/attention.py` modified +1/-0 (1 lines); hunks: -854,6 +854,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +1/-0 (1 lines); hunks: -854,6 +854,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -854,6 +854,7 @@ def __init__(
+            compress_ratio=self.compress_ratio,
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52823 - [DSv4 Perf] Adaptive topk width for dsv4, making #50004 back

- Link: https://github.com/vllm-project/vllm/pull/52823
- Status/date: merged / 2026-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/sparse_mla.py`; associated commits `e6f35d3c69b2`
- Diff scope read: GitHub Pull Request files API returned 2 files, +78/-3, 111 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4 Perf] Adaptive topk width for dsv4, making #50004 back"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/sparse_mla.py`; technical summary: Covers "[DSv4 Perf] Adaptive topk width for dsv4, making #50004 back"; the main implementation surface is `vllm/models/deepseek_v4/sparse_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/sparse_mla.py` modified +21/-3 (24 lines); hunks: -257,6 +257,15 @@ def _build_c128a_metadata(; -269,7 +278,7 @@ def _build_c128a_metadata(; symbols: _build_c128a_metadata, build_c128a_topk_metadata, touching `_build_c128a_metadata, build_c128a_topk_metadata`.
- Code diff details:
  - `vllm/models/deepseek_v4/sparse_mla.py` modified +21/-3 (24 lines); hunks: -257,6 +257,15 @@ def _build_c128a_metadata(; -269,7 +278,7 @@ def _build_c128a_metadata(; symbols: _build_c128a_metadata, build_c128a_topk_metadata
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/sparse_mla.py
@@ -257,6 +257,15 @@ def _build_c128a_metadata(
+        active_topk_width = min(
+            max(
+                triton.next_power_of_2(max(cm.max_seq_len // self.compress_ratio, 1)),
+                _C128A_TOPK_ALIGNMENT,
+            ),
+            self.c128a_max_compressed,
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/sparse_mla.py` modified +21/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_flashmla_sparse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
