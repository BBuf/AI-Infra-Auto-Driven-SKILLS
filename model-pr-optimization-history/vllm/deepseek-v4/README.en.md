# vLLM DeepSeek V4 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-AITER-TEP4.yaml` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-FP8-TP4-ROCm.yaml` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-confidence-TEP4.yaml` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml` | [#47972](https://github.com/vllm-project/vllm/pull/47972) |
| `tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml` | [#47972](https://github.com/vllm-project/vllm/pull/47972) |
| `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml` | [#42111](https://github.com/vllm-project/vllm/pull/42111) |
| `tests/kernels/test_deepseek_v4_cpu_kernels.py` | [#55355](https://github.com/vllm-project/vllm/pull/55355) |
| `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#42353](https://github.com/vllm-project/vllm/pull/42353), [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#45681](https://github.com/vllm-project/vllm/pull/45681), [#49236](https://github.com/vllm-project/vllm/pull/49236), [#52836](https://github.com/vllm-project/vllm/pull/52836), [#56215](https://github.com/vllm-project/vllm/pull/56215), [#56893](https://github.com/vllm-project/vllm/pull/56893), [#56935](https://github.com/vllm-project/vllm/pull/56935) |
| `tests/models/multimodal/processing/test_deepseek_v4_vl.py` | [#59271](https://github.com/vllm-project/vllm/pull/59271), [#59373](https://github.com/vllm-project/vllm/pull/59373) |
| `tests/models/test_deepseek_v41_decoder_replay_layers.py` | [#58132](https://github.com/vllm-project/vllm/pull/58132) |
| `tests/models/test_deepseek_v41_replay_batch.py` | [#58132](https://github.com/vllm-project/vllm/pull/58132) |
| `tests/models/test_deepseek_v41_replay_start.py` | [#56227](https://github.com/vllm-project/vllm/pull/56227) |
| `tests/models/test_deepseek_v4_dspark_rocm.py` | [#52362](https://github.com/vllm-project/vllm/pull/52362) |
| `tests/models/test_deepseek_v4_fi_moe_ep.py` | [#49636](https://github.com/vllm-project/vllm/pull/49636) |
| `tests/models/test_deepseek_v4_mega_moe.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#43004](https://github.com/vllm-project/vllm/pull/43004), [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43632](https://github.com/vllm-project/vllm/pull/43632), [#51368](https://github.com/vllm-project/vllm/pull/51368), [#53040](https://github.com/vllm-project/vllm/pull/53040), [#56214](https://github.com/vllm-project/vllm/pull/56214), [#56228](https://github.com/vllm-project/vllm/pull/56228), [#56266](https://github.com/vllm-project/vllm/pull/56266), [#56568](https://github.com/vllm-project/vllm/pull/56568), [#56599](https://github.com/vllm-project/vllm/pull/56599), [#56741](https://github.com/vllm-project/vllm/pull/56741), ... (14 total) |
| `tests/models/test_deepseek_v4_rocm_compressor_gemm_fusion.py` | [#53838](https://github.com/vllm-project/vllm/pull/53838) |
| `tests/models/test_deepseek_v4_rocm_wo_a.py` | [#54894](https://github.com/vllm-project/vllm/pull/54894) |
| `tests/models/test_deepseek_v4_vl_rocm.py` | [#51692](https://github.com/vllm-project/vllm/pull/51692), [#55107](https://github.com/vllm-project/vllm/pull/55107), [#56433](https://github.com/vllm-project/vllm/pull/56433), [#57132](https://github.com/vllm-project/vllm/pull/57132), [#57919](https://github.com/vllm-project/vllm/pull/57919), [#58740](https://github.com/vllm-project/vllm/pull/58740) |
| `tests/parser/engine/test_deepseek_v4.py` | [#45877](https://github.com/vllm-project/vllm/pull/45877), [#51296](https://github.com/vllm-project/vllm/pull/51296), [#56271](https://github.com/vllm-project/vllm/pull/56271) |
| `tests/parser/engine/test_deepseek_v41.py` | [#56208](https://github.com/vllm-project/vllm/pull/56208), [#56271](https://github.com/vllm-project/vllm/pull/56271) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_1.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_2.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_3.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_4.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_5.json` | [#56882](https://github.com/vllm-project/vllm/pull/56882) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_1.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#51856](https://github.com/vllm-project/vllm/pull/51856) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_2.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_3.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_4.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_5.txt` | [#56882](https://github.com/vllm-project/vllm/pull/56882) |
| `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` | [#56208](https://github.com/vllm-project/vllm/pull/56208), [#58316](https://github.com/vllm-project/vllm/pull/58316) |
| `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt` | [#56208](https://github.com/vllm-project/vllm/pull/56208), [#58316](https://github.com/vllm-project/vllm/pull/58316) |
| `tests/tokenizers_/test_deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#40982](https://github.com/vllm-project/vllm/pull/40982), [#50580](https://github.com/vllm-project/vllm/pull/50580), [#51262](https://github.com/vllm-project/vllm/pull/51262), [#51856](https://github.com/vllm-project/vllm/pull/51856), [#53747](https://github.com/vllm-project/vllm/pull/53747), [#54566](https://github.com/vllm-project/vllm/pull/54566), [#56882](https://github.com/vllm-project/vllm/pull/56882) |
| `tests/tokenizers_/test_deepseek_v41.py` | [#56208](https://github.com/vllm-project/vllm/pull/56208), [#56299](https://github.com/vllm-project/vllm/pull/56299), [#58316](https://github.com/vllm-project/vllm/pull/58316) |
| `tests/v1/attention/test_deepseek_v4_rocm_adaptive.py` | [#52362](https://github.com/vllm-project/vllm/pull/52362) |
| `tests/v1/attention/test_deepseek_v4_swa_visible.py` | [#54566](https://github.com/vllm-project/vllm/pull/54566), [#56227](https://github.com/vllm-project/vllm/pull/56227), [#57152](https://github.com/vllm-project/vllm/pull/57152) |
| `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#50175](https://github.com/vllm-project/vllm/pull/50175), [#56214](https://github.com/vllm-project/vllm/pull/56214), [#56227](https://github.com/vllm-project/vllm/pull/56227), [#56562](https://github.com/vllm-project/vllm/pull/56562), [#56741](https://github.com/vllm-project/vllm/pull/56741) |
| `vllm/models/deepseek_v4/__init__.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953), [#43004](https://github.com/vllm-project/vllm/pull/43004), [#43077](https://github.com/vllm-project/vllm/pull/43077), [#47419](https://github.com/vllm-project/vllm/pull/47419), [#47677](https://github.com/vllm-project/vllm/pull/47677), [#54566](https://github.com/vllm-project/vllm/pull/54566), [#55107](https://github.com/vllm-project/vllm/pull/55107), [#55355](https://github.com/vllm-project/vllm/pull/55355) |
| `vllm/models/deepseek_v4/amd/__init__.py` | [#43004](https://github.com/vllm-project/vllm/pull/43004) |
| `vllm/models/deepseek_v4/amd/dspark.py` | [#47419](https://github.com/vllm-project/vllm/pull/47419), [#51145](https://github.com/vllm-project/vllm/pull/51145), [#52362](https://github.com/vllm-project/vllm/pull/52362), [#52737](https://github.com/vllm-project/vllm/pull/52737) |
| `vllm/models/deepseek_v4/amd/model.py` | [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43385](https://github.com/vllm-project/vllm/pull/43385), [#43629](https://github.com/vllm-project/vllm/pull/43629), [#43679](https://github.com/vllm-project/vllm/pull/43679), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#43950](https://github.com/vllm-project/vllm/pull/43950), [#44246](https://github.com/vllm-project/vllm/pull/44246), [#44262](https://github.com/vllm-project/vllm/pull/44262), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#45931](https://github.com/vllm-project/vllm/pull/45931), [#46720](https://github.com/vllm-project/vllm/pull/46720), ... (26 total) |
| `vllm/models/deepseek_v4/amd/mtp.py` | [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43385](https://github.com/vllm-project/vllm/pull/43385), [#43629](https://github.com/vllm-project/vllm/pull/43629), [#43679](https://github.com/vllm-project/vllm/pull/43679), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#43950](https://github.com/vllm-project/vllm/pull/43950), [#44821](https://github.com/vllm-project/vllm/pull/44821), [#45931](https://github.com/vllm-project/vllm/pull/45931), [#48044](https://github.com/vllm-project/vllm/pull/48044), [#50176](https://github.com/vllm-project/vllm/pull/50176), [#55107](https://github.com/vllm-project/vllm/pull/55107), [#57568](https://github.com/vllm-project/vllm/pull/57568) |
| `vllm/models/deepseek_v4/amd/rocm.py` | [#43149](https://github.com/vllm-project/vllm/pull/43149), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43385](https://github.com/vllm-project/vllm/pull/43385), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#44699](https://github.com/vllm-project/vllm/pull/44699), [#45681](https://github.com/vllm-project/vllm/pull/45681), [#46720](https://github.com/vllm-project/vllm/pull/46720), [#47419](https://github.com/vllm-project/vllm/pull/47419), [#51538](https://github.com/vllm-project/vllm/pull/51538), [#51692](https://github.com/vllm-project/vllm/pull/51692), [#51794](https://github.com/vllm-project/vllm/pull/51794), ... (25 total) |
| `vllm/models/deepseek_v4/attention.py` | [#43039](https://github.com/vllm-project/vllm/pull/43039), [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43149](https://github.com/vllm-project/vllm/pull/43149), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43477](https://github.com/vllm-project/vllm/pull/43477), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#43891](https://github.com/vllm-project/vllm/pull/43891), [#44246](https://github.com/vllm-project/vllm/pull/44246), [#44561](https://github.com/vllm-project/vllm/pull/44561), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#45091](https://github.com/vllm-project/vllm/pull/45091), [#45309](https://github.com/vllm-project/vllm/pull/45309), ... (33 total) |
| `vllm/models/deepseek_v4/common/__init__.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073) |
| `vllm/models/deepseek_v4/common/mm_preprocess.py` | [#54566](https://github.com/vllm-project/vllm/pull/54566), [#59271](https://github.com/vllm-project/vllm/pull/59271), [#59373](https://github.com/vllm-project/vllm/pull/59373) |
| `vllm/models/deepseek_v4/common/ops/__init__.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#50176](https://github.com/vllm-project/vllm/pull/50176) |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#45681](https://github.com/vllm-project/vllm/pull/45681), [#47493](https://github.com/vllm-project/vllm/pull/47493), [#49236](https://github.com/vllm-project/vllm/pull/49236), [#50176](https://github.com/vllm-project/vllm/pull/50176), [#50298](https://github.com/vllm-project/vllm/pull/50298), [#51538](https://github.com/vllm-project/vllm/pull/51538), [#51967](https://github.com/vllm-project/vllm/pull/51967), [#52084](https://github.com/vllm-project/vllm/pull/52084), [#52836](https://github.com/vllm-project/vllm/pull/52836), ... (16 total) |
| `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#47718](https://github.com/vllm-project/vllm/pull/47718), [#50176](https://github.com/vllm-project/vllm/pull/50176), [#52212](https://github.com/vllm-project/vllm/pull/52212), [#53566](https://github.com/vllm-project/vllm/pull/53566) |
| `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#45991](https://github.com/vllm-project/vllm/pull/45991), [#46730](https://github.com/vllm-project/vllm/pull/46730), [#49236](https://github.com/vllm-project/vllm/pull/49236), [#50176](https://github.com/vllm-project/vllm/pull/50176), [#50803](https://github.com/vllm-project/vllm/pull/50803), [#52836](https://github.com/vllm-project/vllm/pull/52836), [#53566](https://github.com/vllm-project/vllm/pull/53566), [#55355](https://github.com/vllm-project/vllm/pull/55355), [#56254](https://github.com/vllm-project/vllm/pull/56254) |
| `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` | [#42950](https://github.com/vllm-project/vllm/pull/42950), [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43477](https://github.com/vllm-project/vllm/pull/43477), [#50176](https://github.com/vllm-project/vllm/pull/50176), [#53566](https://github.com/vllm-project/vllm/pull/53566), [#56035](https://github.com/vllm-project/vllm/pull/56035), [#56228](https://github.com/vllm-project/vllm/pull/56228) |
| `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` | [#43746](https://github.com/vllm-project/vllm/pull/43746), [#50176](https://github.com/vllm-project/vllm/pull/50176), [#53566](https://github.com/vllm-project/vllm/pull/53566) |
| `vllm/models/deepseek_v4/common/ops/save_partial_states.py` | [#43710](https://github.com/vllm-project/vllm/pull/43710), [#50176](https://github.com/vllm-project/vllm/pull/50176), [#53566](https://github.com/vllm-project/vllm/pull/53566) |
| `vllm/models/deepseek_v4/common/rope.py` | [#44262](https://github.com/vllm-project/vllm/pull/44262), [#54815](https://github.com/vllm-project/vllm/pull/54815) |
| `vllm/models/deepseek_v4/common/vision.py` | [#54566](https://github.com/vllm-project/vllm/pull/54566), [#56228](https://github.com/vllm-project/vllm/pull/56228), [#56625](https://github.com/vllm-project/vllm/pull/56625), [#58499](https://github.com/vllm-project/vllm/pull/58499) |
| `vllm/models/deepseek_v4/common/vl_model.py` | [#55107](https://github.com/vllm-project/vllm/pull/55107), [#55897](https://github.com/vllm-project/vllm/pull/55897) |
| `vllm/models/deepseek_v4/compressor.py` | [#42950](https://github.com/vllm-project/vllm/pull/42950), [#42953](https://github.com/vllm-project/vllm/pull/42953), [#43039](https://github.com/vllm-project/vllm/pull/43039), [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43477](https://github.com/vllm-project/vllm/pull/43477), [#43690](https://github.com/vllm-project/vllm/pull/43690), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#47474](https://github.com/vllm-project/vllm/pull/47474), [#47493](https://github.com/vllm-project/vllm/pull/47493), [#47718](https://github.com/vllm-project/vllm/pull/47718), [#48957](https://github.com/vllm-project/vllm/pull/48957), ... (16 total) |
| `vllm/models/deepseek_v4/cpu/__init__.py` | [#55355](https://github.com/vllm-project/vllm/pull/55355) |
| `vllm/models/deepseek_v4/cpu/cpu_compressor.py` | [#55355](https://github.com/vllm-project/vllm/pull/55355) |
| `vllm/models/deepseek_v4/cpu/cpu_mla.py` | [#55355](https://github.com/vllm-project/vllm/pull/55355) |
| `vllm/models/deepseek_v4/cpu/cpu_sparse.py` | [#51794](https://github.com/vllm-project/vllm/pull/51794), [#55355](https://github.com/vllm-project/vllm/pull/55355) |
| `vllm/models/deepseek_v4/cpu/cpu_utils.py` | [#55355](https://github.com/vllm-project/vllm/pull/55355) |
| `vllm/models/deepseek_v4/cpu/dspark.py` | [#55355](https://github.com/vllm-project/vllm/pull/55355) |
| `vllm/models/deepseek_v4/cpu/model.py` | [#55355](https://github.com/vllm-project/vllm/pull/55355) |
| `vllm/models/deepseek_v4/cpu/mtp.py` | [#55355](https://github.com/vllm-project/vllm/pull/55355) |
| `vllm/models/deepseek_v4/nvidia/__init__.py` | [#43004](https://github.com/vllm-project/vllm/pull/43004) |
| `vllm/models/deepseek_v4/nvidia/dspark.py` | [#46789](https://github.com/vllm-project/vllm/pull/46789), [#47429](https://github.com/vllm-project/vllm/pull/47429), [#49415](https://github.com/vllm-project/vllm/pull/49415), [#51368](https://github.com/vllm-project/vllm/pull/51368), [#54674](https://github.com/vllm-project/vllm/pull/54674), [#56266](https://github.com/vllm-project/vllm/pull/56266) |
| `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` | [#43477](https://github.com/vllm-project/vllm/pull/43477), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#44699](https://github.com/vllm-project/vllm/pull/44699), [#44892](https://github.com/vllm-project/vllm/pull/44892), [#45863](https://github.com/vllm-project/vllm/pull/45863), [#47493](https://github.com/vllm-project/vllm/pull/47493), [#48047](https://github.com/vllm-project/vllm/pull/48047), [#49236](https://github.com/vllm-project/vllm/pull/49236), [#51538](https://github.com/vllm-project/vllm/pull/51538), [#52724](https://github.com/vllm-project/vllm/pull/52724), [#52836](https://github.com/vllm-project/vllm/pull/52836), ... (14 total) |
| `vllm/models/deepseek_v4/nvidia/flashmla.py` | [#43149](https://github.com/vllm-project/vllm/pull/43149), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#44699](https://github.com/vllm-project/vllm/pull/44699), [#45061](https://github.com/vllm-project/vllm/pull/45061), [#49236](https://github.com/vllm-project/vllm/pull/49236), [#50298](https://github.com/vllm-project/vllm/pull/50298), [#52836](https://github.com/vllm-project/vllm/pull/52836), [#54566](https://github.com/vllm-project/vllm/pull/54566) |
| `vllm/models/deepseek_v4/nvidia/model.py` | [#42925](https://github.com/vllm-project/vllm/pull/42925), [#42950](https://github.com/vllm-project/vllm/pull/42950), [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43149](https://github.com/vllm-project/vllm/pull/43149), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43339](https://github.com/vllm-project/vllm/pull/43339), [#43477](https://github.com/vllm-project/vllm/pull/43477), [#43632](https://github.com/vllm-project/vllm/pull/43632), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#43829](https://github.com/vllm-project/vllm/pull/43829), [#43891](https://github.com/vllm-project/vllm/pull/43891), ... (40 total) |
| `vllm/models/deepseek_v4/nvidia/mtp.py` | [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#43829](https://github.com/vllm-project/vllm/pull/43829), [#43905](https://github.com/vllm-project/vllm/pull/43905), [#44821](https://github.com/vllm-project/vllm/pull/44821), [#46789](https://github.com/vllm-project/vllm/pull/46789), [#49415](https://github.com/vllm-project/vllm/pull/49415), [#50176](https://github.com/vllm-project/vllm/pull/50176), [#51368](https://github.com/vllm-project/vllm/pull/51368), [#54566](https://github.com/vllm-project/vllm/pull/54566), [#56266](https://github.com/vllm-project/vllm/pull/56266) |
| `vllm/models/deepseek_v4/nvidia/ops/__init__.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710) |
| `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#53566](https://github.com/vllm-project/vllm/pull/53566), [#55061](https://github.com/vllm-project/vllm/pull/55061), [#56893](https://github.com/vllm-project/vllm/pull/56893) |
| `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#53566](https://github.com/vllm-project/vllm/pull/53566), [#56228](https://github.com/vllm-project/vllm/pull/56228), [#56254](https://github.com/vllm-project/vllm/pull/56254) |
| `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` | [#44569](https://github.com/vllm-project/vllm/pull/44569), [#45681](https://github.com/vllm-project/vllm/pull/45681), [#56228](https://github.com/vllm-project/vllm/pull/56228), [#57428](https://github.com/vllm-project/vllm/pull/57428), [#58621](https://github.com/vllm-project/vllm/pull/58621) |
| `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` | [#43632](https://github.com/vllm-project/vllm/pull/43632), [#53040](https://github.com/vllm-project/vllm/pull/53040), [#57604](https://github.com/vllm-project/vllm/pull/57604) |
| `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` | [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#44161](https://github.com/vllm-project/vllm/pull/44161), [#44236](https://github.com/vllm-project/vllm/pull/44236), [#49236](https://github.com/vllm-project/vllm/pull/49236), [#52836](https://github.com/vllm-project/vllm/pull/52836), [#53566](https://github.com/vllm-project/vllm/pull/53566) |
| `vllm/models/deepseek_v4/nvidia/vl_model.py` | [#54566](https://github.com/vllm-project/vllm/pull/54566), [#55107](https://github.com/vllm-project/vllm/pull/55107) |
| `vllm/models/deepseek_v4/quant_config.py` | [#42209](https://github.com/vllm-project/vllm/pull/42209), [#43004](https://github.com/vllm-project/vllm/pull/43004), [#44914](https://github.com/vllm-project/vllm/pull/44914), [#48044](https://github.com/vllm-project/vllm/pull/48044), [#49634](https://github.com/vllm-project/vllm/pull/49634), [#57071](https://github.com/vllm-project/vllm/pull/57071) |
| `vllm/models/deepseek_v4/sparse_mla.py` | [#43477](https://github.com/vllm-project/vllm/pull/43477), [#44699](https://github.com/vllm-project/vllm/pull/44699), [#44892](https://github.com/vllm-project/vllm/pull/44892), [#47474](https://github.com/vllm-project/vllm/pull/47474), [#50004](https://github.com/vllm-project/vllm/pull/50004), [#50176](https://github.com/vllm-project/vllm/pull/50176), [#51318](https://github.com/vllm-project/vllm/pull/51318), [#52823](https://github.com/vllm-project/vllm/pull/52823), [#53566](https://github.com/vllm-project/vllm/pull/53566), [#53574](https://github.com/vllm-project/vllm/pull/53574), [#56227](https://github.com/vllm-project/vllm/pull/56227) |
| ... | 57 more files omitted from table; all were used for git tracing. |

## PR Coverage Summary

- Git-traced PRs: 192
- Extra PRs preserved from existing docs: 39
- Total PRs in this document: 231
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-26 | [#40806](https://github.com/vllm-project/vllm/pull/40806) | merged | [Bugfix] Fix the DSML token leakage in DSV4/3.2 | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-04-27 | [#40760](https://github.com/vllm-project/vllm/pull/40760) | closed | [New Model] Support DeepseekV4 | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py` |
| 2026-04-27 | [#40860](https://github.com/vllm-project/vllm/pull/40860) | merged | [Feat] DeepSeek V4 Rebased | `vllm/model_executor/models/deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`, `tests/tokenizers_/test_deepseek_v4.py` |
| 2026-04-27 | [#40950](https://github.com/vllm-project/vllm/pull/40950) | merged | [DSV4] Add silu clamp limit to shared expert | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/activation.py`, `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` |
| 2026-04-28 | [#41006](https://github.com/vllm-project/vllm/pull/41006) | merged | [Model][DSV4] Support base model | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-04-28 | [#41061](https://github.com/vllm-project/vllm/pull/41061) | merged | [DSV4] Enable Multi-stream for Pre-Attn GEMM | `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_compressor.py` |
| 2026-04-29 | [#40982](https://github.com/vllm-project/vllm/pull/40982) | merged | [DSV4] Support `max` reasoning effort | `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py` |
| 2026-04-29 | [#41015](https://github.com/vllm-project/vllm/pull/41015) | merged | [DSv4] Use `cvt` PTX for FP32->FP4 conversion | `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` |
| 2026-04-29 | [#41090](https://github.com/vllm-project/vllm/pull/41090) | merged | [Bugfix] Fix Deepseek V4 import error due to AOT compile cache loading | `vllm/model_executor/models/deepseek_v4.py` |
| 2026-04-29 | [#41135](https://github.com/vllm-project/vllm/pull/41135) | merged | [Bugfix] fix inductor error for dpsk v4 | `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` |
| 2026-04-29 | [#41148](https://github.com/vllm-project/vllm/pull/41148) | merged | [Bugfix] Fix repeated DSv4 RoPE cache initialization | `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/models/deepseek_v4.py` |
| 2026-04-29 | [#41171](https://github.com/vllm-project/vllm/pull/41171) | merged | [DSV4] Align aux stream API with DeepseekV4DecoderLayer | `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-04-30 | [#41374](https://github.com/vllm-project/vllm/pull/41374) | merged | [DSV4] Avoid redundant dtype conversion. | `vllm/model_executor/models/deepseek_v4.py` |
| 2026-05-01 | [#41255](https://github.com/vllm-project/vllm/pull/41255) | merged | [Perf] Intergrate Tile Kernels `head_compute_mix_kernel` for Deepseek-V4 | `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py` |
| 2026-05-01 | [#41443](https://github.com/vllm-project/vllm/pull/41443) | merged | [DSV4] Add knob to enable pre-attn gemm | `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/envs.py`, `vllm/utils/multi_stream_utils.py` |
| 2026-05-02 | [#41522](https://github.com/vllm-project/vllm/pull/41522) | merged | [DSV4] Guard megamoe flag with Pure TP | `vllm/model_executor/models/deepseek_v4.py` |
| 2026-05-05 | [#40871](https://github.com/vllm-project/vllm/pull/40871) | merged | [New Model][ROCm] Add AMD support for DeepSeek V4 | `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` |
| 2026-05-06 | [#41801](https://github.com/vllm-project/vllm/pull/41801) | merged | [Bugfix] DeepSeekV32/v4: respect string='true\|false' attribute andunwrap arguments/input wrapper | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py` |
| 2026-05-09 | [#41428](https://github.com/vllm-project/vllm/pull/41428) | merged | [DSv4] Improved fused Indexer Q quant kernel | `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `vllm/utils/import_utils.py` |
| 2026-05-09 | [#41957](https://github.com/vllm-project/vllm/pull/41957) | merged | [Bugfix][PD] Fix DSv4 Disaggregated | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py`, `tests/v1/kv_connector/unit/test_tp_mapping.py` |
| 2026-05-10 | [#41694](https://github.com/vllm-project/vllm/pull/41694) | merged | [DSV4] Add PP support for deepseek-v4 | `vllm/model_executor/models/deepseek_v4.py`, `docs/models/supported_models.md` |
| 2026-05-10 | [#42169](https://github.com/vllm-project/vllm/pull/42169) | merged | [Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len | `csrc/topk.cu` |
| 2026-05-11 | [#40392](https://github.com/vllm-project/vllm/pull/40392) | merged | [Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA | `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` |
| 2026-05-11 | [#41536](https://github.com/vllm-project/vllm/pull/41536) | merged | add fused mhc_post_pre kernel | `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py`, `tests/kernels/test_mhc_kernels.py` |
| 2026-05-11 | [#41812](https://github.com/vllm-project/vllm/pull/41812) | merged | [ROCm][DSv4] implement flash sparse mla with triton kernels | `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` |
| 2026-05-11 | [#42236](https://github.com/vllm-project/vllm/pull/42236) | merged | [DSv4] Improved dequant gather K cache kernel | `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py`, `tests/kernels/test_compressor_kv_cache.py`, `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` |
| 2026-05-13 | [#41946](https://github.com/vllm-project/vllm/pull/41946) | merged | [Bugfix] [ROCm] [DSV4] [Perf] Add aiter mhc support | `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/kernels/mhc/tilelang.py`, `vllm/model_executor/kernels/mhc/triton.py` |
| 2026-05-13 | [#42320](https://github.com/vllm-project/vllm/pull/42320) | merged | [Bugfix] Fix DeepSeek V4 MTP HC state handling | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-05-14 | [#41263](https://github.com/vllm-project/vllm/pull/41263) | merged | [DSV4] Fuse norm and router for low latency scenario | `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-05-14 | [#41778](https://github.com/vllm-project/vllm/pull/41778) | merged | [MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell | `benchmarks/attention_benchmarks/configs/mla_prefill.yaml`, `benchmarks/attention_benchmarks/configs/mla_decode.yaml`, `vllm/model_executor/layers/attention/mla_attention.py` |
| 2026-05-14 | [#42112](https://github.com/vllm-project/vllm/pull/42112) | merged | [Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup | `vllm/v1/attention/backends/mla/prefill/flashinfer.py`, `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` |
| 2026-05-14 | [#42342](https://github.com/vllm-project/vllm/pull/42342) | merged | [Bug] Fix DeepSeek V4 `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'LoadCacheMode'` | `requirements/cuda.txt` |
| 2026-05-15 | [#42604](https://github.com/vllm-project/vllm/pull/42604) | merged | DeepSeekV4-Pro enable cuda graph full and piecewise mode | `vllm/model_executor/layers/mhc.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` |
| 2026-05-17 | [#42810](https://github.com/vllm-project/vllm/pull/42810) | merged | [ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy | `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v4.py` |
| 2026-05-18 | [#41710](https://github.com/vllm-project/vllm/pull/41710) | merged | fix: remove unused norm for dpskv4 | `vllm/model_executor/layers/deepseek_v4_attention.py` |
| 2026-05-18 | [#42541](https://github.com/vllm-project/vllm/pull/42541) | merged | [Bugfix] fix swiglu limit issue for humming backend + deepseek v4 | `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`, `vllm/model_executor/layers/quantization/utils/humming_utils.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` |
| 2026-05-18 | [#42930](https://github.com/vllm-project/vllm/pull/42930) | merged | [Bugfix] Fix DSV4 MTP after ROCm mHC integration | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-05-19 | [#42828](https://github.com/vllm-project/vllm/pull/42828) | merged | [KVConnector][DSV4] HMA support for Mooncake store connector | `tests/v1/kv_connector/unit/test_mooncake_store_worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`, `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py` |
| 2026-05-19 | [#42899](https://github.com/vllm-project/vllm/pull/42899) | merged | add cutedsl dsv4 indexer fp8 kernel | `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py` |
| 2026-05-19 | [#43004](https://github.com/vllm-project/vllm/pull/43004) | merged | [Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N] | `vllm/models/deepseek_v4/quant_config.py`, `vllm/models/deepseek_v4/__init__.py`, `tests/models/test_deepseek_v4_mega_moe.py` |
| 2026-05-19 | [#43039](https://github.com/vllm-project/vllm/pull/43039) | merged | [Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N] | `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/compressor.py` |
| 2026-05-19 | [#43073](https://github.com/vllm-project/vllm/pull/43073) | merged | [Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N] | `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/ops/__init__.py`, `vllm/models/deepseek_v4/attention.py` |
| 2026-05-19 | [#43077](https://github.com/vllm-project/vllm/pull/43077) | merged | [Model Refactoring] Rename deepseek_v4.py to model.py [4/N] | `vllm/models/deepseek_v4/__init__.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-05-20 | [#42111](https://github.com/vllm-project/vllm/pull/42111) | merged | [CI] Add DSV4-Flash to gsm8k moe-refactor/config-b200.txt | `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml` |
| 2026-05-22 | [#42209](https://github.com/vllm-project/vllm/pull/42209) | merged | Add NVFP4 MOE support for Deepseek V4. | `vllm/models/deepseek_v4/quant_config.py` |
| 2026-05-22 | [#42353](https://github.com/vllm-project/vllm/pull/42353) | merged | DSv4 fused Q-norm kernel grid refactor | `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` |
| 2026-05-22 | [#42950](https://github.com/vllm-project/vllm/pull/42950) | merged | [XPU]fix: add XPU platform guards to DeepSeek-V4 ops | `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/compressor.py` |
| 2026-05-22 | [#43149](https://github.com/vllm-project/vllm/pull/43149) | merged | [Refactor] Extract DeepSeek V4 sparse MLA impl into model folder | `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/ops/attention.py`, `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-05-23 | [#42925](https://github.com/vllm-project/vllm/pull/42925) | merged | [DSV4] More multi-stream enablement for c4a | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-05-24 | [#43385](https://github.com/vllm-project/vllm/pull/43385) | merged | [ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-05-26 | [#43162](https://github.com/vllm-project/vllm/pull/43162) | merged | [Feat][DSV4] Fuse q pad into deepseek v4 fused kernel | `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-05-26 | [#43629](https://github.com/vllm-project/vllm/pull/43629) | merged | [ROCm] Remove MegaMoE integration in deepseek v4 | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py` |
| 2026-05-26 | [#43632](https://github.com/vllm-project/vllm/pull/43632) | merged | [DeepSeek V4] Move MegaMoE input prep kernel to nvidia/ops | `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `tests/models/test_deepseek_v4_mega_moe.py` |
| 2026-05-26 | [#43690](https://github.com/vllm-project/vllm/pull/43690) | merged | [DSv4] Drop _get_compressed_kv_buffer in DeepseekCompressor | `vllm/models/deepseek_v4/compressor.py` |
| 2026-05-27 | [#43710](https://github.com/vllm-project/vllm/pull/43710) | merged | [DSv4] Refactor compressor & Fix ROCm compatibility | `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/common/ops/save_partial_states.py` |
| 2026-05-28 | [#43679](https://github.com/vllm-project/vllm/pull/43679) | merged | [ROCm][DSV4] Enable Tilelang MHC replacing torch/triton mhc | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py` |
| 2026-05-28 | [#43746](https://github.com/vllm-project/vllm/pull/43746) | merged | [Model Refactoring] Remove torch compile dependency in DSv4 | `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-05-28 | [#43829](https://github.com/vllm-project/vllm/pull/43829) | merged | [DSV4] Remove AMD/XPU path in deepseek_v4/nvidia | `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-05-28 | [#43891](https://github.com/vllm-project/vllm/pull/43891) | merged | [Model Refactoring] Remove unncessary torch op registration for DSv4 | `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-05-29 | [#43905](https://github.com/vllm-project/vllm/pull/43905) | merged | [DSv4] Move mHC tilelang kernels & Don't use CustomOP in dsv4/nvidia | `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-06-01 | [#44161](https://github.com/vllm-project/vllm/pull/44161) | merged | [Kernel][DSv4] Optimize sparse FP8 compressor kernels | `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` |
| 2026-06-01 | [#44246](https://github.com/vllm-project/vllm/pull/44246) | merged | [DSV4] Remove unncessary classes & functions | `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-02 | [#43339](https://github.com/vllm-project/vllm/pull/43339) | merged | [Feature] Support EPLB for DeepSeek v4 Mega Moe | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-02 | [#44262](https://github.com/vllm-project/vllm/pull/44262) | merged | [DSV4] Refactor RoPE initialization | `vllm/models/deepseek_v4/common/rope.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-03 | [#44236](https://github.com/vllm-project/vllm/pull/44236) | merged | fix: resolve CUTLASS fmin compatibility for DeepSeek-V4 init | `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` |
| 2026-06-03 | [#44356](https://github.com/vllm-project/vllm/pull/44356) | merged | [Bugfix] Fix Deepseek v4 non-mega-moe model init error | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-03 | [#44367](https://github.com/vllm-project/vllm/pull/44367) | merged | [DSV4] Minor cleanup for DeepseekV4MegaMoEExperts | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-04 | [#43827](https://github.com/vllm-project/vllm/pull/43827) | merged | [DSv4] Adding TRTLLM gen attention kernel | `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py` |
| 2026-06-05 | [#44561](https://github.com/vllm-project/vllm/pull/44561) | merged | [DSV4] Move more ops out of eager breakpoint | `vllm/models/deepseek_v4/attention.py` |
| 2026-06-05 | [#44569](https://github.com/vllm-project/vllm/pull/44569) | merged | [DSV4] Refactor DeepseekV4Attention | `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-07 | [#44699](https://github.com/vllm-project/vllm/pull/44699) | merged | [DSV4] Decouple DS V4 Sparse MLA Metadata from DS V3.2 | `vllm/models/deepseek_v4/sparse_mla.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` |
| 2026-06-08 | [#42953](https://github.com/vllm-project/vllm/pull/42953) | merged | feat: add DeepSeek-V4 XPU attention decode path | `vllm/models/deepseek_v4/xpu/model.py`, `vllm/models/deepseek_v4/xpu/mtp.py`, `vllm/models/deepseek_v4/xpu/xpu_sparse.py` |
| 2026-06-09 | [#44144](https://github.com/vllm-project/vllm/pull/44144) | merged | [DSV4][XPU] Add MHC fused_post_pre support | `vllm/models/deepseek_v4/xpu/model.py` |
| 2026-06-09 | [#44914](https://github.com/vllm-project/vllm/pull/44914) | merged | [Bug] Fix deepseek v4 OOM issue | `vllm/models/deepseek_v4/quant_config.py` |
| 2026-06-10 | [#44821](https://github.com/vllm-project/vllm/pull/44821) | merged | fix: prefix DeepSeek V4 MTP projections | `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-06-12 | [#45240](https://github.com/vllm-project/vllm/pull/45240) | merged | [XPU][DeepSeek-V4] Fix MTP: sync with upstream fixes #44821 and #43746 | `vllm/models/deepseek_v4/xpu/mtp.py` |
| 2026-06-15 | [#45061](https://github.com/vllm-project/vllm/pull/45061) | merged | [Perf] Optimize DSv4 prefill chunk planning, 4.0% E2E Throughput Improvement | `vllm/models/deepseek_v4/nvidia/flashmla.py` |
| 2026-06-16 | [#44892](https://github.com/vllm-project/vllm/pull/44892) | merged | [DSV4][Minor] Fix supported KV cache dtypes | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/sparse_mla.py` |
| 2026-06-17 | [#45309](https://github.com/vllm-project/vllm/pull/45309) | merged | [DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`, 26.8% ~ 27.9% E2E TTFT improvement | `vllm/models/deepseek_v4/attention.py` |
| 2026-06-17 | [#45863](https://github.com/vllm-project/vllm/pull/45863) | merged | [DSv4 Perf] DSv4 flashinfer sparse index cache for metadata, 2%~4% TTFT improvement | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` |
| 2026-06-18 | [#45681](https://github.com/vllm-project/vllm/pull/45681) | merged | [ROCm][DSv4] Functional fixes for DeepSeek V4 on MI300X/MI325X | `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` |
| 2026-06-18 | [#45972](https://github.com/vllm-project/vllm/pull/45972) | merged | Revert "[DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`" (#45309) | `vllm/models/deepseek_v4/attention.py` |
| 2026-06-19 | [#46001](https://github.com/vllm-project/vllm/pull/46001) | merged | [DeepSeek-V4] Support TEP=16 for the block-FP8 shared expert | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-06-22 | [#43477](https://github.com/vllm-project/vllm/pull/43477) | merged | Enable DeepSeek V4 and GLM-5.1 on SM120 | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py`, `vllm/models/deepseek_v4/attention.py` |
| 2026-06-22 | [#45931](https://github.com/vllm-project/vllm/pull/45931) | merged | [ROCm][DSV4] Disable TileLang MHC dispatch on gfx942 | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py` |
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
| 2026-07-15 | [#47718](https://github.com/vllm-project/vllm/pull/47718) | merged | [ROCm][Perf] DSv4 two-stage compressor kernel for HCA prefill | `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/compressor.py` |
| 2026-07-15 | [#48137](https://github.com/vllm-project/vllm/pull/48137) | merged | [Perf] Remove redundant repeat and copy for dsv4, 1.8% E2E TPOT improvement. | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-07-16 | [#47677](https://github.com/vllm-project/vllm/pull/47677) | merged | [XPU] Add DSpark speculative decoding support for DeepSeek-V4 | `vllm/models/deepseek_v4/xpu/dspark.py`, `vllm/models/deepseek_v4/xpu/model.py`, `vllm/models/deepseek_v4/__init__.py` |
| 2026-07-21 | [#45991](https://github.com/vllm-project/vllm/pull/45991) | merged | [XPU][DeepSeekV4]Add DeepSeek-V4 fuse_index_q SYCL kernel path | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` |
| 2026-07-22 | [#48957](https://github.com/vllm-project/vllm/pull/48957) | merged | [DSv4 Perf] Skip empty c128 kernel launch, around 2x kernel performance improvement. | `vllm/models/deepseek_v4/compressor.py` |
| 2026-07-22 | [#48993](https://github.com/vllm-project/vllm/pull/48993) | merged | [Core][DSV4] Compact MXFP4 indexer KV cache and packed group overlays | `vllm/models/deepseek_v4/attention.py` |
| 2026-07-23 | [#48044](https://github.com/vllm-project/vllm/pull/48044) | merged | [ROCm] Fused Shared Expert Support for AMD Quark DeepSeek-V4 Model Checkpoints | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/quant_config.py`, `vllm/models/deepseek_v4/amd/mtp.py` |
| 2026-07-23 | [#49415](https://github.com/vllm-project/vllm/pull/49415) | merged | [Bugfix] Fix DeepSeek-V4 DSpark draft shared-expert padding for TP > 8 | `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-07-23 | [#49486](https://github.com/vllm-project/vllm/pull/49486) | merged | [DSv4 Perf] Skip topk and router when not needed, 3.4% E2E TTFT improvement for Decode case | `vllm/models/deepseek_v4/attention.py` |
| 2026-07-27 | [#50004](https://github.com/vllm-project/vllm/pull/50004) | merged | [DSv4 Perf] Adaptive topk width, 1.0% E2E throughput improvement | `vllm/models/deepseek_v4/sparse_mla.py` |
| 2026-07-28 | [#49634](https://github.com/vllm-project/vllm/pull/49634) | merged | [Bugfix] Fix DeepseekV4FP8 Quark MXFP4 crash on list-valued weight | `vllm/models/deepseek_v4/quant_config.py` |
| 2026-07-30 | [#46720](https://github.com/vllm-project/vllm/pull/46720) | merged | [ROCm][DSV4] B-preshuffle the attention fp8 projections | `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/attention.py` |
| 2026-07-30 | [#50298](https://github.com/vllm-project/vllm/pull/50298) | merged | [DSv4 Perf] Remove redundant full kernel for dsv4, 1.88x kernel performance improvement | `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py` |
| 2026-07-30 | [#50312](https://github.com/vllm-project/vllm/pull/50312) | merged | [DSv4 Perf] Fix redundant memory allocation and copy for dsv4 pp buffer, 448 MiB GPU memory saved | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/xpu/model.py` |
| 2026-07-31 | [#48047](https://github.com/vllm-project/vllm/pull/48047) | merged | [DSv4] Remove sparse-MLA q-head padding for FlashInfer >=0.6.14 | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` |
| 2026-07-31 | [#49236](https://github.com/vllm-project/vllm/pull/49236) | merged | [DSv4 Perf] Optimize workspace reuse for eager break | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-08-01 | [#46789](https://github.com/vllm-project/vllm/pull/46789) | merged | [DSV4] Implement Sequence Parallelism | `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-08-04 | [#50580](https://github.com/vllm-project/vllm/pull/50580) | merged | [Frontend] DeepSeek V4 0731 reasoning effort prompts & mappings | `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`, `vllm/tokenizers/deepseek_v4.py` |
| 2026-08-07 | [#47972](https://github.com/vllm-project/vllm/pull/47972) | merged | Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel | `vllm/models/deepseek_v4/amd/model.py`, `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml` |
| 2026-08-10 | [#51296](https://github.com/vllm-project/vllm/pull/51296) | merged | [Bugfix] Align deepseek v4 parser thinking default with tokenizer | `tests/parser/engine/test_deepseek_v4.py`, `vllm/parser/deepseek_v4.py` |
| 2026-08-10 | [#51430](https://github.com/vllm-project/vllm/pull/51430) | merged | [Perf] Narrow DeepSeek V4 eager CUDA graph region | `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-08-10 | [#51727](https://github.com/vllm-project/vllm/pull/51727) | merged | [Bugfix] Fix DeepSeek V4/3.2 tokenizer vocab size overcount crashing guided decoding | `vllm/tokenizers/deepseek_v4.py` |
| 2026-08-11 | [#51145](https://github.com/vllm-project/vllm/pull/51145) | merged | [Bugfix][ROCm] Fix DeepSeek V4 DSpark probabilistic startup | `vllm/models/deepseek_v4/amd/dspark.py` |
| 2026-08-13 | [#51821](https://github.com/vllm-project/vllm/pull/51821) | merged | [Bugfix][ROCm][CI] Restore the DeepSeek-V4 input GEMM override point | `vllm/models/deepseek_v4/attention.py` |
| 2026-08-15 | [#51538](https://github.com/vllm-project/vllm/pull/51538) | merged | [Bugfix] Make DSV4 sparse MLA work end-to-end for plain decode, MTP, and DSpark | `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-08-16 | [#51318](https://github.com/vllm-project/vllm/pull/51318) | merged | [Bugfix][DSv4] Revert adaptive C128A metadata packing | `vllm/models/deepseek_v4/sparse_mla.py` |
| 2026-08-16 | [#51967](https://github.com/vllm-project/vllm/pull/51967) | merged | [Perf][DSV4] Optimize global top-k index kernel with compile-time constants | `vllm/models/deepseek_v4/common/ops/cache_utils.py` |
| 2026-08-16 | [#52084](https://github.com/vllm-project/vllm/pull/52084) | merged | [Perf][DSV4] Optimize sparse top-k metadata kernels for higher prefill throughput | `vllm/models/deepseek_v4/common/ops/cache_utils.py` |
| 2026-08-16 | [#52212](https://github.com/vllm-project/vllm/pull/52212) | merged | [ROCm][DSV4][Perf] Optimize Triton sparse-MLA decode on gfx950 | `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` |
| 2026-08-16 | [#52401](https://github.com/vllm-project/vllm/pull/52401) | merged | [Bugfix] Pick the DeepSeek V4 eager cudagraph region per model runner | `vllm/models/deepseek_v4/attention.py` |
| 2026-08-17 | [#52492](https://github.com/vllm-project/vllm/pull/52492) | merged | [Bugfix][DSv4] Keep indexer scoring in breakable graphs | `vllm/models/deepseek_v4/attention.py` |
| 2026-08-18 | [#52626](https://github.com/vllm-project/vllm/pull/52626) | merged | [Bugfix] Fix DeepSeek V4 mHC broadcast buffer for weight sync | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-08-19 | [#51368](https://github.com/vllm-project/vllm/pull/51368) | merged | [Bugfix] Fix DeepSeek V4 mHC broadcast buffer for dummy load | `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/dspark.py` |
| 2026-08-19 | [#52836](https://github.com/vllm-project/vllm/pull/52836) | merged | Revert DSv4 eager workspace reuse | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-08-20 | [#50803](https://github.com/vllm-project/vllm/pull/50803) | merged | [ROCm] Fix DeepSeek V4 indexer numerics and coverage | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` |
| 2026-08-20 | [#52737](https://github.com/vllm-project/vllm/pull/52737) | merged | [ROCm][Perf] Fuse DeepSeek-V4 mHC post/pre and RMSNorm with AITER | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/dspark.py` |
| 2026-08-20 | [#53040](https://github.com/vllm-project/vllm/pull/53040) | merged | [DSV4][Kernel] Fuse shared experts into MegaMoE | `vllm/models/deepseek_v4/nvidia/model.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` |
| 2026-08-21 | [#52823](https://github.com/vllm-project/vllm/pull/52823) | merged | [DSv4 Perf] Adaptive topk width for dsv4, making #50004 back | `vllm/models/deepseek_v4/sparse_mla.py` |
| 2026-08-21 | [#52882](https://github.com/vllm-project/vllm/pull/52882) | merged | [ROCm][Perf] Optimize DeepSeek V4 C4A top-k with AITER | `vllm/models/deepseek_v4/attention.py` |
| 2026-08-24 | [#53361](https://github.com/vllm-project/vllm/pull/53361) | merged | [LoRA] feat: Support LoRA for DeepSeek V4 | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-08-25 | [#49636](https://github.com/vllm-project/vllm/pull/49636) | merged | [Model][MoE] DeepSeek-V4: add opt-in FlashInfer moe_ep expert backend | `tests/models/test_deepseek_v4_fi_moe_ep.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-08-25 | [#51262](https://github.com/vllm-project/vllm/pull/51262) | merged | [Bugfix][DeepSeek V4] Handle trailing system messages in prompt rendering | `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py` |
| 2026-08-25 | [#53747](https://github.com/vllm-project/vllm/pull/53747) | merged | [Bugfix][Tokenizer] Replace bare asserts in the DeepSeek V4 encoder | `vllm/tokenizers/deepseek_v4_encoding.py`, `tests/tokenizers_/test_deepseek_v4.py` |
| 2026-08-26 | [#53838](https://github.com/vllm-project/vllm/pull/53838) | merged | [ROCm][DSV4][Perf] Fuse DeepSeek V4 C4 compressor GEMMs | `tests/models/test_deepseek_v4_rocm_compressor_gemm_fusion.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/amd/model.py` |
| 2026-08-26 | [#53697](https://github.com/vllm-project/vllm/pull/53697) | merged | [Model] Remove unused DeepSeek V4 top-k buffer helper | `vllm/models/deepseek_v4/attention.py` |
| 2026-08-27 | [#53540](https://github.com/vllm-project/vllm/pull/53540) | merged | [ROCm][Perf] Fuse SWA q/kv RMSNorm and q FP8 group quant for DeepSeek-V4 | `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py` |
| 2026-08-31 | [#53574](https://github.com/vllm-project/vllm/pull/53574) | merged | [Bugfix][SM120] DSv4: pass contiguous C128A decode topk indices on SM120 | `vllm/models/deepseek_v4/sparse_mla.py` |
| 2026-09-01 | [#50175](https://github.com/vllm-project/vllm/pull/50175) | merged | [1/N][warmup][DSv4] Migrate generic MLA metadata and indexing kernels | `vllm/models/deepseek_v4/attention.py`, `vllm/model_executor/layers/attention/mla_attention.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` |
| 2026-09-01 | [#52724](https://github.com/vllm-project/vllm/pull/52724) | merged | [Attention] Enable adaptive verification for FLASHINFER_MLA_SPARSE_DSV4 | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` |
| 2026-09-02 | [#54815](https://github.com/vllm-project/vllm/pull/54815) | merged | [Bugfix] Fix RoPE construction for deepseek-v4 sparse SWA layers | `vllm/models/deepseek_v4/common/rope.py` |
| 2026-09-02 | [#54566](https://github.com/vllm-project/vllm/pull/54566) | merged | [New model][Multimodal] Add DeepSeek-V4-Flash-Vision-Exp support | `vllm/models/deepseek_v4/common/mm_preprocess.py`, `vllm/models/deepseek_v4/nvidia/vl_model.py`, `vllm/models/deepseek_v4/common/vision.py` |
| 2026-09-03 | [#45091](https://github.com/vllm-project/vllm/pull/45091) | merged | Fix DeepSeek V4 FlashMLA auto KV cache dtype | `vllm/models/deepseek_v4/attention.py` |
| 2026-09-04 | [#55061](https://github.com/vllm-project/vllm/pull/55061) | merged | [Performance][DSv4] Size dequant gather launch grid by rows | `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` |
| 2026-09-05 | [#55299](https://github.com/vllm-project/vllm/pull/55299) | merged | [Bugfix][DSv4] Seed the -1 sentinel in the prefill sparse index workspace | `vllm/models/deepseek_v4/common/ops/cache_utils.py` |
| 2026-09-07 | [#53161](https://github.com/vllm-project/vllm/pull/53161) | merged | [ROCm][Perf][DeepSeek V4] Fuse native FP8 shared expert with MXFP4 routed experts | `vllm/models/deepseek_v4/amd/model.py` |
| 2026-09-07 | [#53689](https://github.com/vllm-project/vllm/pull/53689) | merged | [XPU][LoRA] Support LoRA for DeepSeek V4 on XPU | `vllm/models/deepseek_v4/xpu/model.py` |
| 2026-09-08 | [#50176](https://github.com/vllm-project/vllm/pull/50176) | merged | [4/N][warmup][DSv4] Migrate common attention kernels | `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` |
| 2026-09-09 | [#56035](https://github.com/vllm-project/vllm/pull/56035) | merged | [Bugfix][ROCm][DSv4] Skip launch_pdl=True JIT warmup when PDL is unsupported | `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` |
| 2026-09-10 | [#55355](https://github.com/vllm-project/vllm/pull/55355) | merged | [Model] Add DeepSeek-V4 CPU backend | `vllm/models/deepseek_v4/cpu/model.py`, `vllm/models/deepseek_v4/cpu/cpu_sparse.py`, `vllm/models/deepseek_v4/cpu/cpu_mla.py` |
| 2026-09-10 | [#56215](https://github.com/vllm-project/vllm/pull/56215) | merged | [Kernel] Optional Q-norm in fused DSv4 MLA epilogue; group_size=32 for packed FP8 quant | `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` |
| 2026-09-10 | [#56208](https://github.com/vllm-project/vllm/pull/56208) | merged | [Model][Frontend] Support DeepSeek-V4.1-Flash in Rust and Python frontends | `vllm/tokenizers/deepseek_v41_encoding.py`, `tests/tokenizers_/test_deepseek_v41.py`, `vllm/tokenizers/deepseek_v41.py` |
| 2026-09-10 | [#56228](https://github.com/vllm-project/vllm/pull/56228) | merged | [Model] DeepSeek-V4.1-Flash Model Definitions | `vllm/models/deepseek_v4_1/nvidia/model.py`, `vllm/models/deepseek_v4_1/amd/model.py`, `vllm/models/deepseek_v4_1/amd/vl_model.py` |
| 2026-09-10 | [#51692](https://github.com/vllm-project/vllm/pull/51692) | merged | [ROCm][Perf] Add bpreshuffled blockscaled fp8 GEMM | `vllm/model_executor/kernels/linear/scaled_mm/aiter.py`, `vllm/model_executor/kernels/linear/__init__.py`, `vllm/_aiter_ops.py` |
| 2026-09-11 | [#56214](https://github.com/vllm-project/vllm/pull/56214) | merged | [Model] Support DeepSeek-V4.1-Flash | `tests/models/test_deepseek_v4_mega_moe.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py` |
| 2026-09-11 | [#56433](https://github.com/vllm-project/vllm/pull/56433) | merged | [ROCm][Bugfix] Fix AITER preshuffled FP8 block-scale kernel | `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/amd/model.py` |
| 2026-09-11 | [#55107](https://github.com/vllm-project/vllm/pull/55107) | merged | [Model][ROCm] Enable DeepSeek V4 Vision | `tests/models/test_deepseek_v4_vl_rocm.py`, `vllm/models/deepseek_v4/nvidia/vl_model.py`, `vllm/models/deepseek_v4/common/vl_model.py` |
| 2026-09-12 | [#53566](https://github.com/vllm-project/vllm/pull/53566) | merged | [5/N][warmup][DSv4] Migrate NVIDIA CuTeDSL attention kernels | `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`, `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py`, `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` |
| 2026-09-12 | [#56554](https://github.com/vllm-project/vllm/pull/56554) | merged | [DSV4.1] Remove compressor-aware image sentinel token padding | `vllm/transformers_utils/configs/deepseek_v41.py` |
| 2026-09-12 | [#56562](https://github.com/vllm-project/vllm/pull/56562) | merged | [Perf] Fuse DSV4.1 input metadata preparation with Triton | `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`, `vllm/v1/attention/backends/mla/indexer.py`, `vllm/v1/attention/ops/metadata.py` |
| 2026-09-12 | [#56599](https://github.com/vllm-project/vllm/pull/56599) | merged | [CI] Update DeepSeek V4.1 MegaMoE routing test | `tests/models/test_deepseek_v4_mega_moe.py` |
| 2026-09-13 | [#50178](https://github.com/vllm-project/vllm/pull/50178) | merged | [9/N][warmup][DSv4] Migrate MHC TileLang kernels | `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/glm5next/nvidia/model.py` |
| 2026-09-13 | [#55897](https://github.com/vllm-project/vllm/pull/55897) | merged | [LoRA] Add LoRA support for DeepSeek-V4 Flash Vision | `vllm/models/deepseek_v4/common/vl_model.py` |
| 2026-09-14 | [#56299](https://github.com/vllm-project/vllm/pull/56299) | merged | [Bugfix][Frontend] Support Responses text types in DeepSeek V4.1 | `tests/tokenizers_/test_deepseek_v41.py`, `vllm/tokenizers/deepseek_v41.py` |
| 2026-09-14 | [#56741](https://github.com/vllm-project/vllm/pull/56741) | merged | [Refactor] Normalize DeepSeek V4.1 model package naming | `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/models/deepseek_v41/nvidia/engram.py` |
| 2026-09-14 | [#51794](https://github.com/vllm-project/vllm/pull/51794) | merged | [ROCm][Perf] Enable CSA multi-stream overlap for DeepSeek-V4 | `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/amd/model.py` |
| 2026-09-14 | [#56633](https://github.com/vllm-project/vllm/pull/56633) | merged | [Perf][DSv4.1] Fold the mHC post block into the delayed pre projection | `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/dspark.py` |
| 2026-09-14 | [#56513](https://github.com/vllm-project/vllm/pull/56513) | merged | [ROCm][DSV4.1][Perf] Fold the mHC post step into the delayed pre projection | `vllm/models/deepseek_v41/amd/model.py` |
| 2026-09-15 | [#56255](https://github.com/vllm-project/vllm/pull/56255) | closed | [DSv4.1] Integrate Mega-mHC from DeepGEMM | `vllm/models/deepseek_v4_1/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v4_1/nvidia/model.py`, `vllm/models/deepseek_v4_1/nvidia/ops/__init__.py` |
| 2026-09-15 | [#56893](https://github.com/vllm-project/vllm/pull/56893) | merged | [Model][DSv4.1] Store the whole KV in MXFP8 (FlashMLA V4.1 record) | `vllm/models/deepseek_v41/common/ops/cache_utils.py`, `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` |
| 2026-09-15 | [#56903](https://github.com/vllm-project/vllm/pull/56903) | merged | [Perf][DSpark] Collapse DeepSeek-V4.1 draft states before SP all-gather | `vllm/models/deepseek_v41/nvidia/dspark.py` |
| 2026-09-15 | [#56962](https://github.com/vllm-project/vllm/pull/56962) | merged | [Perf][Kernel] Integrate Mega-mHC from DeepGEMM for DeepSeek V4.1 (reopen of #56255) | `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/__init__.py` |
| 2026-09-15 | [#56254](https://github.com/vllm-project/vllm/pull/56254) | merged | [DSA] Wire DeepGEMM sparse MQA logits into the DeepSeek V4.1 indexer | `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` |
| 2026-09-16 | [#56441](https://github.com/vllm-project/vllm/pull/56441) | merged | [Perf][DSpark] Add KV-only context insertion across V4.1 cache formats | `vllm/models/deepseek_v41/nvidia/dspark.py` |
| 2026-09-16 | [#56568](https://github.com/vllm-project/vllm/pull/56568) | merged | [Perf][DSV4.1] Pad shared experts for native MegaMoE fusion | `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-09-16 | [#56935](https://github.com/vllm-project/vllm/pull/56935) | merged | [Model][DSv4.1] FlashMLA mega attention and the NVFP4 compressed KV cache | `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py`, `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/common/ops/cache_utils.py` |
| 2026-09-16 | [#57152](https://github.com/vllm-project/vllm/pull/57152) | merged | [Bugfix][Model] Restore causal image SWA for DeepSeek V4.1 | `vllm/models/deepseek_v41/common/ops/cache_utils.py`, `vllm/models/deepseek_v41/nvidia/flashmla.py`, `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` |
| 2026-09-16 | [#57132](https://github.com/vllm-project/vllm/pull/57132) | merged | [ROCm][Bugfix] Revert #56433 + #51692 to fix accuracy breakdown for DeepSeek-V4 | `tests/models/test_deepseek_v4_vl_rocm.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/amd/model.py` |
| 2026-09-16 | [#57204](https://github.com/vllm-project/vllm/pull/57204) | merged | [Perf][DSV4.1] Remove MegaMoE padding and shared padding workaround | `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-09-17 | [#56853](https://github.com/vllm-project/vllm/pull/56853) | merged | [ROCm][Perf] Enable HCA dual-stream overlap for DeepSeek-V4 | `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-09-17 | [#54674](https://github.com/vllm-project/vllm/pull/54674) | merged | [Perf][DSpark] Stack DeepSeek V4 context WKV projections | `vllm/models/deepseek_v4/nvidia/dspark.py` |
| 2026-09-17 | [#56266](https://github.com/vllm-project/vllm/pull/56266) | merged | [DSv4.1] Integrate Mega-Gate from DeepGEMM | `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-09-17 | [#57432](https://github.com/vllm-project/vllm/pull/57432) | merged | [Bugfix][DSv4.1] Fix FlashInfer DSpark non-causal attention | `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` |
| 2026-09-18 | [#51856](https://github.com/vllm-project/vllm/pull/51856) | merged | [Bugfix] Attach request-level tools to existing system message in DeepSeek V4 Python renderer | `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py`, `tests/tokenizers_/fixtures/deepseek_v4/test_output_1.txt` |
| 2026-09-18 | [#56271](https://github.com/vllm-project/vllm/pull/56271) | merged | [Frontend] Fix the parsing of missing `string=` in DeepSeek V4 | `tests/parser/engine/test_deepseek_v4.py`, `vllm/parser/deepseek_v4.py`, `vllm/parser/deepseek_v41.py` |
| 2026-09-18 | [#56882](https://github.com/vllm-project/vllm/pull/56882) | merged | [Bugfix][Multimodal] Preserve DeepSeek V4 image block spacing | `tests/tokenizers_/fixtures/deepseek_v4/test_input_5.json`, `tests/tokenizers_/test_deepseek_v4.py`, `tests/tokenizers_/fixtures/deepseek_v4/test_output_5.txt` |
| 2026-09-18 | [#57465](https://github.com/vllm-project/vllm/pull/57465) | merged | [DeepSeek V4] Fix fused MoE expert distribution | `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-09-18 | [#57604](https://github.com/vllm-project/vllm/pull/57604) | merged | [Perf][DSV4.1] Optimize MegaMoE staging and NVFP4 cache gathers | `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v41/common/ops/cache_utils.py` |
| 2026-09-18 | [#56227](https://github.com/vllm-project/vllm/pull/56227) | merged | [Feat][Model] Support encoder-side SWA-bounded replay for DeepSeek-V4.1-Flash | `vllm/models/deepseek_v41/nvidia/model_state.py`, `tests/models/test_deepseek_v41_replay_start.py`, `vllm/models/deepseek_v41/attention.py` |
| 2026-09-20 | [#57434](https://github.com/vllm-project/vllm/pull/57434) | merged | [ROCm][DSv4.1][Perf] Reuse the decode topk ragged metadata across layers | `vllm/models/deepseek_v41/amd/rocm.py` |
| 2026-09-20 | [#54894](https://github.com/vllm-project/vllm/pull/54894) | merged | [ROCm][DSV4][Perf] Use FP8 WO_A output projection | `vllm/models/deepseek_v4/amd/rocm.py`, `tests/models/test_deepseek_v4_rocm_wo_a.py` |
| 2026-09-20 | [#56625](https://github.com/vllm-project/vllm/pull/56625) | merged | [DSV4.1] Add encoder cuda graph support for deepseek-v4.1-flash | `vllm/models/deepseek_v41/common/vl_cudagraph.py`, `vllm/models/deepseek_v4/common/vision.py`, `vllm/models/deepseek_v41/amd/vl_model.py` |
| 2026-09-21 | [#57603](https://github.com/vllm-project/vllm/pull/57603) | merged | [Perf][DSV4.1] Overlap mHC coefficients for small TP batches | `vllm/models/deepseek_v41/nvidia/ops/mhc.py`, `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` |
| 2026-09-21 | [#57491](https://github.com/vllm-project/vllm/pull/57491) | merged | [ROCm][DSv4.1] Keep the Engram tables in host memory on ROCm | `vllm/models/deepseek_v41/amd/model.py` |
| 2026-09-21 | [#57643](https://github.com/vllm-project/vllm/pull/57643) | merged | [Perf][DSV4.1] Fuse TP all-reduce with mHC input preparation | `vllm/models/deepseek_v41/nvidia/ops/mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v41/nvidia/model.py` |
| 2026-09-21 | [#57874](https://github.com/vllm-project/vllm/pull/57874) | merged | [Bugfix][DSV4.1] Restrict mHC overlap to full CUDA graphs | `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/mhc.py`, `vllm/models/deepseek_v41/nvidia/model.py` |
| 2026-09-21 | [#57906](https://github.com/vllm-project/vllm/pull/57906) | merged | [Bugfix][ROCm][DSv4.1] Disable SWA bounded replay on ROCm | `vllm/models/deepseek_v41/attention.py` |
| 2026-09-21 | [#52362](https://github.com/vllm-project/vllm/pull/52362) | merged | [ROCm][DSv4] Enable DSpark adaptive verification | `tests/models/test_deepseek_v4_dspark_rocm.py`, `vllm/models/deepseek_v4/amd/dspark.py`, `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-09-22 | [#57919](https://github.com/vllm-project/vllm/pull/57919) | merged | [ROCm][Bugfix] Explicitly reject FSE=1 with DPA+ETP deployment for DeepSeek-V4 | `vllm/models/deepseek_v4/amd/model.py`, `tests/models/test_deepseek_v4_vl_rocm.py` |
| 2026-09-22 | [#57428](https://github.com/vllm-project/vllm/pull/57428) | merged | [Kernel][DSV4.1] Fuse MXFP8 wo_b GEMM with sequence-parallel reduce-scatter | `vllm/models/kimi_k3/nvidia/model.py`, `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/nvidia/model.py` |
| 2026-09-22 | [#57435](https://github.com/vllm-project/vllm/pull/57435) | merged | [ROCm][DSv4.1][Perf] Fuse the inverse RoPE into the sparse decode reduce | `vllm/models/deepseek_v41/amd/rocm.py` |
| 2026-09-23 | [#57451](https://github.com/vllm-project/vllm/pull/57451) | merged | [ROCm][DSv4][Perf] Fuse the inverse RoPE into the sparse decode reduce | `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-09-23 | [#44229](https://github.com/vllm-project/vllm/pull/44229) | merged | [Feature][Frontend] Add DeepSeek-V4 FIM completion rendering | `vllm/renderers/deepseek_v4.py` |
| 2026-09-24 | [#58456](https://github.com/vllm-project/vllm/pull/58456) | merged | [ROCm][DSv4.1][Perf] Emit MXFP8 from the sparse decode reduce and run wo_a as a grouped FP8 GEMM | `vllm/models/deepseek_v41/amd/rocm.py` |
| 2026-09-25 | [#57679](https://github.com/vllm-project/vllm/pull/57679) | merged | [Perf][DSv4.1] Restore the fused query RMSNorm + MXFP8 quantization path | `vllm/models/deepseek_v41/common/ops/query_quant.py` |
| 2026-09-25 | [#58621](https://github.com/vllm-project/vllm/pull/58621) | merged | [Perf][DSv4] Fuse inverse RoPE + FP8 quant into FlashInfer sparse MLA | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py` |
| 2026-09-25 | [#58740](https://github.com/vllm-project/vllm/pull/58740) | merged | [ROCm][CI] Test AMD DeepSeek V4 MoE routing against a PyTorch reference | `tests/models/test_deepseek_v4_vl_rocm.py` |
| 2026-09-26 | [#58678](https://github.com/vllm-project/vllm/pull/58678) | merged | [Perf][DSv4.1] Shard the Engram wkv projection across TP ranks | `vllm/models/deepseek_v41/common/engram.py`, `tests/kernels/test_engram.py` |
| 2026-09-26 | [#57071](https://github.com/vllm-project/vllm/pull/57071) | merged | [Bugfix][ROCm] AMD-Quark mixed-precision DeepSeek-V4.1 support | `vllm/models/deepseek_v41/amd/vl_model.py`, `vllm/models/deepseek_v41/quant_config.py`, `vllm/models/deepseek_v4/quant_config.py` |
| 2026-09-26 | [#58316](https://github.com/vllm-project/vllm/pull/58316) | merged | [Bugfix][Frontend][Rust Frontend] Update DeepSeek V4.1 Flash reasoning effort mappings | `tests/tokenizers_/test_deepseek_v41.py`, `vllm/tokenizers/deepseek_v41_encoding.py`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` |
| 2026-09-26 | [#58499](https://github.com/vllm-project/vllm/pull/58499) | merged | [Bugfix][DSV4.1] Avoid host sync in ViT CUDA graph replay metadata | `vllm/models/deepseek_v41/common/vl_cudagraph.py`, `vllm/models/deepseek_v4/common/vision.py` |
| 2026-09-26 | [#58586](https://github.com/vllm-project/vllm/pull/58586) | merged | [Kernel][DSV4.1] Fuse MoE finalize into the TP all-reduce + mHC boundary | `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py`, `vllm/models/deepseek_v41/nvidia/ops/mhc.py` |
| 2026-09-27 | [#58634](https://github.com/vllm-project/vllm/pull/58634) | merged | [Perf][DSv4.1] Fuse small-batch WO-A with inverse RoPE and MXFP8 quant on SM100/SM103 | `vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py`, `vllm/models/deepseek_v41/nvidia/ops/o_proj.py`, `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` |
| 2026-09-27 | [#57407](https://github.com/vllm-project/vllm/pull/57407) | merged | [ROCm][Perf] Enable layer-aware CSA2 multi-stream overlap for DeepSeek-V4.1-Flash | `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/models/deepseek_v41/attention.py` |
| 2026-09-28 | [#57568](https://github.com/vllm-project/vllm/pull/57568) | merged | [Bugfix][Spec Decode] Implement get_top_tokens() on the ROCm DeepSeek V4 MTP drafter | `vllm/models/deepseek_v4/amd/mtp.py` |
| 2026-09-28 | [#58983](https://github.com/vllm-project/vllm/pull/58983) | merged | [ROCm][Refactor] Move DeepSeek-V4/V4.1 multi-stream overlap gate to ROCm platform | `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/platforms/rocm.py` |
| 2026-09-29 | [#58655](https://github.com/vllm-project/vllm/pull/58655) | merged | [ROCm][DSv4.1][Perf] Run the delayed mHC seams through aiter's fused Triton kernel | `vllm/models/deepseek_v41/amd/model.py` |
| 2026-09-29 | [#58671](https://github.com/vllm-project/vllm/pull/58671) | merged | [ROCm][DSv4.1] Paged MXFP4 sparse indexer on aiter's MQA-logits kernel | `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/models/deepseek_v41/attention.py`, `vllm/config/attention.py` |
| 2026-09-29 | [#59119](https://github.com/vllm-project/vllm/pull/59119) | merged | [DSv4.1] Avoid runtime recompiles of _ring_slot_mapping_kernel | `vllm/models/deepseek_v41/compressor.py` |
| 2026-09-29 | [#58132](https://github.com/vllm-project/vllm/pull/58132) | merged | [Model] Decoder-side SWA bounded replay for DeepSeek-V4.1 | `vllm/models/deepseek_v41/nvidia/model_state.py`, `vllm/models/deepseek_v41/nvidia/model.py`, `tests/models/test_deepseek_v41_replay_batch.py` |
| 2026-09-29 | [#57898](https://github.com/vllm-project/vllm/pull/57898) | merged | [Bugfix] Profile maximum DeepSeek V4.1 vision features | `vllm/models/deepseek_v41/common/mm_preprocess.py` |
| 2026-09-30 | [#58405](https://github.com/vllm-project/vllm/pull/58405) | merged | [ROCm][DSv4][Perf] Use the shared prefill chunk plan in the ROCm sparse prefill | `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-09-30 | [#58539](https://github.com/vllm-project/vllm/pull/58539) | merged | [ROCm][DSv4.1][Perf] Use the shared prefill chunk plan in the ROCm sparse prefill | `vllm/models/deepseek_v41/amd/rocm.py` |
| 2026-09-30 | [#59373](https://github.com/vllm-project/vllm/pull/59373) | merged | [BugFix][Multimodal] Pick worst-case DeepSeek-V4 VL dummy image size (#59271) | `tests/models/multimodal/processing/test_deepseek_v4_vl.py`, `vllm/models/deepseek_v4/common/mm_preprocess.py` |
| 2026-10-01 | [#59327](https://github.com/vllm-project/vllm/pull/59327) | merged | [Perf][DSv4.1] Faster Engram host lookups: sorted rows, inline big lookups | `vllm/models/deepseek_v41/common/engram.py`, `vllm/models/deepseek_v41/nvidia/engram.py`, `vllm/models/deepseek_v41/nvidia/model.py` |
| 2026-10-04 | [#59159](https://github.com/vllm-project/vllm/pull/59159) | merged | [Bugfix][XPU] Make DeepSeek V4 FP8 sparse decode graph-capturable | `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` |
| 2026-10-05 | [#58560](https://github.com/vllm-project/vllm/pull/58560) | merged | [Bugfix][DSv4.1] Keep the compressor ring out of the null block | `vllm/models/deepseek_v41/compressor.py` |

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

### PR #53361 - [LoRA] feat: Support LoRA for DeepSeek V4

- Link: https://github.com/vllm-project/vllm/pull/53361
- Status/date: merged / 2026-08-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `702e1d718646`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +70/-5, 166 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/nvidia/model.py` modified +35/-1 (36 lines); hunks: -55,6 +55,7; -1468,6 +1469,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, _make_deepseek_v4_weights_mapper, update_physical_experts_metadata, touching `load_weights, _make_deepseek_v4_weights_mapper, update_physical_experts_metadata`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +35/-1 (36 lines); hunks: -55,6 +55,7; -1468,6 +1469,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, _make_deepseek_v4_weights_mapper, update_physical_experts_metadata
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -55,6 +55,7 @@
+    SupportsLoRA,
@@ -1468,6 +1469,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+                if name not in params_dict:
+                    head, _, leaf = name.rpartition(".")
+                    suffixed = f"{head}.base_layer.{leaf}"
+                    if suffixed in params_dict:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +35/-1
- Risk and verification: Runtime changes concentrate in `vllm/lora/layers/base.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/models/deepseek_v4/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #49636 - [Model][MoE] DeepSeek-V4: add opt-in FlashInfer moe_ep expert backend

- Link: https://github.com/vllm-project/vllm/pull/49636
- Status/date: merged / 2026-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_fi_moe_ep.py`, `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `8fe9317f2e40`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +1083/-12, 1195 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/test_deepseek_v4_fi_moe_ep.py` added +259/-0 (259 lines); hunks: -0,0 +1,259; symbols: _FakeBootstrapConfig, _FakeDeepGemmMegaMoeConfig, _FakeNvfp4CutedslMegaMoeConfig, _FakeMegaConfig, touching `_FakeBootstrapConfig, _FakeDeepGemmMegaMoeConfig, _FakeNvfp4CutedslMegaMoeConfig`; `vllm/models/deepseek_v4/nvidia/model.py` modified +32/-11 (43 lines); hunks: -11,6 +11,7; -83,6 +84,10; symbols: __init__, _init_mega_moe_experts, _init_fused_moe_experts, touching `__init__, _init_mega_moe_experts, _init_fused_moe_experts`.
- Code diff details:
  - `tests/models/test_deepseek_v4_fi_moe_ep.py` added +259/-0 (259 lines); hunks: -0,0 +1,259; symbols: _FakeBootstrapConfig, _FakeDeepGemmMegaMoeConfig, _FakeNvfp4CutedslMegaMoeConfig, _FakeMegaConfig
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +32/-11 (43 lines); hunks: -11,6 +11,7; -83,6 +84,10; symbols: __init__, _init_mega_moe_experts, _init_fused_moe_experts
- Key code excerpts:

```diff
diff -- tests/models/test_deepseek_v4_fi_moe_ep.py
@@ -0,0 +1,259 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Unit tests for the flashinfer moe_ep backend plumbing.
+Everything here runs without a GPU or a flashinfer install: the flashinfer
+modules the helpers import lazily are replaced with capture fakes.
+"""
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -11,6 +11,7 @@
+from vllm.config.kernel import MEGA_MOE_BACKENDS
@@ -83,6 +84,10 @@
+from vllm.utils.flashinfer_moe_ep import (
+    is_fi_moe_ep_backend,
+    validate_fi_moe_ep_config,
+)
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/test_deepseek_v4_fi_moe_ep.py` added +259/-0
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +32/-11
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_fi_moe_ep.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51262 - [Bugfix][DeepSeek V4] Handle trailing system messages in prompt rendering

- Link: https://github.com/vllm-project/vllm/pull/51262
- Status/date: merged / 2026-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`; associated commits `59d7fc92ea03`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +150/-3, 188 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/tokenizers_/test_deepseek_v4.py` modified +77/-0 (77 lines); hunks: -113,6 +113,83 @@ def test_deepseek_v4_explicitly_disables_thinking(kwargs):; symbols: test_deepseek_v4_explicitly_disables_thinking, test_deepseek_v4_appends_assistant_transition_after_trailing_system, test_deepseek_v4_does_not_transition_after_mid_conversation_system, test_deepseek_v4_transitions_from_system_to_assistant, touching `test_deepseek_v4_explicitly_disables_thinking, test_deepseek_v4_appends_assistant_transition_after_trailing_system, test_deepseek_v4_does_not_transition_after_mid_conversation_system`; `vllm/tokenizers/deepseek_v4_encoding.py` modified +9/-1 (10 lines); hunks: -374,7 +374,15 @@ def render_message(; symbols: render_message, touching `render_message`.
- Code diff details:
  - `tests/tokenizers_/test_deepseek_v4.py` modified +77/-0 (77 lines); hunks: -113,6 +113,83 @@ def test_deepseek_v4_explicitly_disables_thinking(kwargs):; symbols: test_deepseek_v4_explicitly_disables_thinking, test_deepseek_v4_appends_assistant_transition_after_trailing_system, test_deepseek_v4_does_not_transition_after_mid_conversation_system, test_deepseek_v4_transitions_from_system_to_assistant
  - `vllm/tokenizers/deepseek_v4_encoding.py` modified +9/-1 (10 lines); hunks: -374,7 +374,15 @@ def render_message(; symbols: render_message
- Key code excerpts:

```diff
diff -- tests/tokenizers_/test_deepseek_v4.py
@@ -113,6 +113,83 @@ def test_deepseek_v4_explicitly_disables_thinking(kwargs):
+@pytest.mark.parametrize(
+    ("enable_thinking", "thinking_token"),
+    [(False, "</think>"), (True, "<think>")],
+)
+def test_deepseek_v4_appends_assistant_transition_after_trailing_system(
+    enable_thinking, thinking_token
diff -- vllm/tokenizers/deepseek_v4_encoding.py
@@ -374,7 +374,15 @@ def render_message(
-    elif messages[index].get("role") in ["user", "developer"]:
+    # A trailing system message opens generation, while a system message
+    # followed by assistant opens that assistant history turn.
+    elif role in ["user", "developer"] or (
+        role == "system"
+        and (
```

- Extracted files (not manually reviewed):
  - tests: `tests/tokenizers_/test_deepseek_v4.py` modified +77/-0
  - runtime: `vllm/tokenizers/deepseek_v4_encoding.py` modified +9/-1
- Risk and verification: The diff ships test coverage in `tests/tokenizers_/test_deepseek_v4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53747 - [Bugfix][Tokenizer] Replace bare asserts in the DeepSeek V4 encoder

- Link: https://github.com/vllm-project/vllm/pull/53747
- Status/date: merged / 2026-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`; associated commits `19406fae2873`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +41/-8, 81 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/tokenizers/deepseek_v4_encoding.py` modified +17/-8 (25 lines); hunks: -225,8 +225,12 @@ def render_message(; -248,10 +252,11 @@ def render_message(; symbols: render_message, touching `render_message`; `tests/tokenizers_/test_deepseek_v4.py` modified +24/-0 (24 lines); hunks: -403,3 +403,27 @@ def test_deepseek_v4_matches_reference_golden_fixtures(case...; symbols: test_deepseek_v4_matches_reference_golden_fixtures, test_deepseek_v4_rejects_empty_developer_content, test_deepseek_v4_encode_messages_rejects_invalid_arguments, touching `test_deepseek_v4_matches_reference_golden_fixtures, test_deepseek_v4_rejects_empty_developer_content, test_deepseek_v4_encode_messages_rejects_invalid_arguments`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v4_encoding.py` modified +17/-8 (25 lines); hunks: -225,8 +225,12 @@ def render_message(; -248,10 +252,11 @@ def render_message(; symbols: render_message
  - `tests/tokenizers_/test_deepseek_v4.py` modified +24/-0 (24 lines); hunks: -403,3 +403,27 @@ def test_deepseek_v4_matches_reference_golden_fixtures(case...; symbols: test_deepseek_v4_matches_reference_golden_fixtures, test_deepseek_v4_rejects_empty_developer_content, test_deepseek_v4_encode_messages_rejects_invalid_arguments
- Key code excerpts:

```diff
diff -- vllm/tokenizers/deepseek_v4_encoding.py
@@ -225,8 +225,12 @@ def render_message(
-    assert 0 <= index < len(messages)
-    assert thinking_mode in ["chat", "thinking"], f"Invalid thinking_mode `{thinking_mode}`"
+    if not (0 <= index < len(messages)):
+        raise ValueError(
+            f"Index {index} out of range for messages list of length {len(messages)}"
+        )
diff -- tests/tokenizers_/test_deepseek_v4.py
@@ -403,3 +403,27 @@ def test_deepseek_v4_matches_reference_golden_fixtures(case_id, kwargs):
+def test_deepseek_v4_rejects_empty_developer_content():
+    with pytest.raises(ValueError):
+        _tokenizer().apply_chat_template(
+            [{"role": "developer", "content": ""}],
+            tokenize=False,
+            enable_thinking=True,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/tokenizers/deepseek_v4_encoding.py` modified +17/-8
  - tests: `tests/tokenizers_/test_deepseek_v4.py` modified +24/-0
- Risk and verification: The diff ships test coverage in `tests/tokenizers_/test_deepseek_v4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53838 - [ROCm][DSV4][Perf] Fuse DeepSeek V4 C4 compressor GEMMs

- Link: https://github.com/vllm-project/vllm/pull/53838
- Status/date: merged / 2026-08-26
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_rocm_compressor_gemm_fusion.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`; associated commits `657f9b9ce241`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +221/-0, 269 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/test_deepseek_v4_rocm_compressor_gemm_fusion.py` added +141/-0 (141 lines); hunks: -0,0 +1,141; symbols: _WeightOnlyLinear, __init__, _Compressor, _IndexerWeightsProjection, touching `_WeightOnlyLinear, __init__, _Compressor`; `vllm/models/deepseek_v4/amd/rocm.py` modified +73/-0 (73 lines); hunks: -11,6 +11,7; -36,6 +37,8; symbols: _trust_dsv4_extra_cache_nan_free, __init__, get_padded_num_q_heads, _prep, touching `_trust_dsv4_extra_cache_nan_free, __init__, get_padded_num_q_heads`; `vllm/models/deepseek_v4/amd/model.py` modified +7/-0 (7 lines); hunks: -1044,11 +1044,18 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; symbols: load_weights, process_weights_after_loading, get_expert_mapping, touching `load_weights, process_weights_after_loading, get_expert_mapping`.
- Code diff details:
  - `tests/models/test_deepseek_v4_rocm_compressor_gemm_fusion.py` added +141/-0 (141 lines); hunks: -0,0 +1,141; symbols: _WeightOnlyLinear, __init__, _Compressor, _IndexerWeightsProjection
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +73/-0 (73 lines); hunks: -11,6 +11,7; -36,6 +37,8; symbols: _trust_dsv4_extra_cache_nan_free, __init__, get_padded_num_q_heads, _prep
  - `vllm/models/deepseek_v4/amd/model.py` modified +7/-0 (7 lines); hunks: -1044,11 +1044,18 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; symbols: load_weights, process_weights_after_loading, get_expert_mapping
- Key code excerpts:

```diff
diff -- tests/models/test_deepseek_v4_rocm_compressor_gemm_fusion.py
@@ -0,0 +1,141 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+import vllm.model_executor.offloader as offloader
+from vllm.models.deepseek_v4.amd.rocm import (
+    DeepseekV4ROCMAiterMLAAttention,
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -11,6 +11,7 @@
+from vllm.logger import init_logger
@@ -36,6 +37,8 @@
+logger = init_logger(__name__)
@@ -486,6 +489,9 @@ def __init__(self, *args, **kwargs):
+        self._fused_compressor_weight: torch.Tensor | None
+        self.register_buffer("_fused_compressor_weight", None, persistent=False)
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -1044,11 +1044,18 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/test_deepseek_v4_rocm_compressor_gemm_fusion.py` added +141/-0
  - runtime: `vllm/models/deepseek_v4/amd/rocm.py` modified +73/-0; `vllm/models/deepseek_v4/amd/model.py` modified +7/-0
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_rocm_compressor_gemm_fusion.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53697 - [Model] Remove unused DeepSeek V4 top-k buffer helper

- Link: https://github.com/vllm-project/vllm/pull/53697
- Status/date: merged / 2026-08-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`; associated commits `0a5ad6f0d42c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +0/-7, 14 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/attention.py` modified +0/-7 (7 lines); hunks: -685,13 +685,6 @@ def _fused_qnorm_rope_kv_insert(; symbols: _fused_qnorm_rope_kv_insert, _global_topk_output_buffers, bind_kv_cache, touching `_fused_qnorm_rope_kv_insert, _global_topk_output_buffers, bind_kv_cache`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +0/-7 (7 lines); hunks: -685,13 +685,6 @@ def _fused_qnorm_rope_kv_insert(; symbols: _fused_qnorm_rope_kv_insert, _global_topk_output_buffers, bind_kv_cache
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -685,13 +685,6 @@ def _fused_qnorm_rope_kv_insert(
-    def _global_topk_output_buffers(
-        self, topk_indices: torch.Tensor
-    ) -> tuple[torch.Tensor, torch.Tensor] | None:
-        if self.compress_ratio != 4 or self.eager_scratch_pool is None:
-            return None
-        return self.eager_scratch_pool.global_topk_outputs(topk_indices)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +0/-7
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #53540 - [ROCm][Perf] Fuse SWA q/kv RMSNorm and q FP8 group quant for DeepSeek-V4

- Link: https://github.com/vllm-project/vllm/pull/53540
- Status/date: merged / 2026-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`; associated commits `32ad1400d7fa`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +333/-12, 479 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/amd/rocm.py` modified +96/-0 (96 lines); hunks: -1,6 +1,7; -60,6 +61,36 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> torc...; symbols: _build_indptr_from_lengths, apply_pre_quantized_block_scaled_mm, _run_parallel_input_projections, _wq_b_uses_aiter_block_scaled, touching `_build_indptr_from_lengths, apply_pre_quantized_block_scaled_mm, _run_parallel_input_projections`; `vllm/models/deepseek_v4/attention.py` modified +67/-11 (78 lines); hunks: -373,19 +373,13 @@ def forward(; -397,12 +391,32 @@ def forward(; symbols: forward, _split_qkv_and_norm, _prepare_and_attn_eager, touching `forward, _split_qkv_and_norm, _prepare_and_attn_eager`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +96/-0 (96 lines); hunks: -1,6 +1,7; -60,6 +61,36 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> torc...; symbols: _build_indptr_from_lengths, apply_pre_quantized_block_scaled_mm, _run_parallel_input_projections, _wq_b_uses_aiter_block_scaled
  - `vllm/models/deepseek_v4/attention.py` modified +67/-11 (78 lines); hunks: -373,19 +373,13 @@ def forward(; -397,12 +391,32 @@ def forward(; symbols: forward, _split_qkv_and_norm, _prepare_and_attn_eager
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -1,6 +1,7 @@
+import functools
@@ -60,6 +61,36 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
+def apply_pre_quantized_block_scaled_mm(
+    linear: torch.nn.Module,
+    x_fp8: torch.Tensor,
+    x_scale: torch.Tensor,
diff -- vllm/models/deepseek_v4/attention.py
@@ -373,19 +373,13 @@ def forward(
-        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
-        qr, kv = fused_q_kv_rmsnorm(
-            qr,
-            kv,
-            self.q_norm.weight.data,
-            self.kv_norm.weight.data,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/amd/rocm.py` modified +96/-0; `vllm/models/deepseek_v4/attention.py` modified +67/-11
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_rocm_aiter_ops.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53574 - [Bugfix][SM120] DSv4: pass contiguous C128A decode topk indices on SM120

- Link: https://github.com/vllm-project/vllm/pull/53574
- Status/date: merged / 2026-08-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/sparse_mla.py`; associated commits `699e180df48d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +31/-7, 84 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/sparse_mla.py` modified +9/-2 (11 lines); hunks: -9,6 +9,7; -310,7 +311,7 @@ def build_c128a_topk_metadata(; symbols: build_c128a_topk_metadata, touching `build_c128a_topk_metadata`.
- Code diff details:
  - `vllm/models/deepseek_v4/sparse_mla.py` modified +9/-2 (11 lines); hunks: -9,6 +9,7; -310,7 +311,7 @@ def build_c128a_topk_metadata(; symbols: build_c128a_topk_metadata
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/sparse_mla.py
@@ -9,6 +9,7 @@
+from vllm.platforms import current_platform
@@ -310,7 +311,7 @@ def build_c128a_topk_metadata(
-    Returns slices of the buffers.
+    Returns views of the buffers.
@@ -322,7 +323,13 @@ def build_c128a_topk_metadata(
-    global_decode = global_decode_buffer[:num_decode_tokens, :max_compressed_tokens]
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/sparse_mla.py` modified +9/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_flashmla_sparse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50175 - [1/N][warmup][DSv4] Migrate generic MLA metadata and indexing kernels

- Link: https://github.com/vllm-project/vllm/pull/50175
- Status/date: merged / 2026-09-01
- Trace source: `git log --name-only -- <model-files>` found it through `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`, `vllm/models/deepseek_v4/attention.py`; associated commits `e16b5e518db8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 17 files, +1269/-625, 2446 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/attention.py` modified +51/-0 (51 lines); hunks: -353,6 +353,57 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/layers/attention/mla_attention.py` modified +47/-0 (47 lines); hunks: -604,6 +604,33 @@ def __init__(; -652,6 +679,26 @@ def __init__(; symbols: __init__, bind_kv_cache, chunked_prefill_workspace_size, touching `__init__, bind_kv_cache, chunked_prefill_workspace_size`; `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +43/-2 (45 lines); hunks: -8,10 +8,16; -20,7 +26,7 @@ def test_indexer_warmup_normalizes_zero_compress_ratios():; symbols: test_indexer_warmup_normalizes_zero_compress_ratios, test_compressed_slot_mapping_warmup_includes_index_kpool, test_index_conversion_warmup_uses_physical_block_stride, touching `test_indexer_warmup_normalizes_zero_compress_ratios, test_compressed_slot_mapping_warmup_includes_index_kpool, test_index_conversion_warmup_uses_physical_block_stride`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +51/-0 (51 lines); hunks: -353,6 +353,57 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +47/-0 (47 lines); hunks: -604,6 +604,33 @@ def __init__(; -652,6 +679,26 @@ def __init__(; symbols: __init__, bind_kv_cache, chunked_prefill_workspace_size
  - `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +43/-2 (45 lines); hunks: -8,10 +8,16; -20,7 +26,7 @@ def test_indexer_warmup_normalizes_zero_compress_ratios():; symbols: test_indexer_warmup_normalizes_zero_compress_ratios, test_compressed_slot_mapping_warmup_includes_index_kpool, test_index_conversion_warmup_uses_physical_block_stride
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -353,6 +353,57 @@ def __init__(
+        if vllm_config.kernel_config.enable_jit_warmup:
+            from vllm.v1.attention.backends.mla.sparse_swa import (
+                _COMPUTE_PREFILL_METADATA_KERNEL,
+                _COMPUTE_SWA_INDICES_AND_LENS_KERNEL,
+            )
+            _COMPUTE_PREFILL_METADATA_KERNEL.register_warmup()
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -604,6 +604,33 @@ def __init__(
+        if vllm_config.kernel_config.enable_jit_warmup:
+            backend_name = self.attn_backend.get_name()
+            if backend_name in (
+                "FLASHMLA_SPARSE",
+                "FLASHINFER_MLA_SPARSE",
+                "FLASHINFER_MLA_SPARSE_SM120",
diff -- tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py
@@ -8,10 +8,16 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +51/-0; `vllm/model_executor/layers/attention/mla_attention.py` modified +47/-0
  - tests: `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +43/-2
- Risk and verification: The diff ships test coverage in `tests/model_executor/layers/test_fused_shared_expert.py`, `tests/model_executor/test_jit_warmup.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`, `tests/v1/worker/test_gpu_model_runner_v2_eplb.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52724 - [Attention] Enable adaptive verification for FLASHINFER_MLA_SPARSE_DSV4

- Link: https://github.com/vllm-project/vllm/pull/52724
- Status/date: merged / 2026-09-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`; associated commits `25efcfa7887c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +30/-1, 63 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +30/-1 (31 lines); hunks: -21,13 +21,18; -164,11 +169,34 @@ def supports_combination(; symbols: supports_combination, get_builder_cls, DeepseekV4FlashInferSparseMLAMetadataBuilder, DeepseekSparseSWAFlashInferMetadataBuilder, touching `supports_combination, get_builder_cls, DeepseekV4FlashInferSparseMLAMetadataBuilder`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +30/-1 (31 lines); hunks: -21,13 +21,18; -164,11 +169,34 @@ def supports_combination(; symbols: supports_combination, get_builder_cls, DeepseekV4FlashInferSparseMLAMetadataBuilder, DeepseekSparseSWAFlashInferMetadataBuilder
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
@@ -21,13 +21,18 @@
+    DeepseekV4SparseMLAMetadataBuilder,
-from vllm.v1.attention.backend import MultipleOf
+from vllm.v1.attention.backend import AttentionCGSupport, MultipleOf
+from vllm.v1.attention.backends.mla.sparse_swa import (
+    DeepseekSparseSWABackend,
+    DeepseekSparseSWAMetadataBuilder,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +30/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #54815 - [Bugfix] Fix RoPE construction for deepseek-v4 sparse SWA layers

- Link: https://github.com/vllm-project/vllm/pull/54815
- Status/date: merged / 2026-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/rope.py`; associated commits `1d8d7a396527`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +17/-1, 33 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/common/rope.py` modified +17/-1 (18 lines); hunks: -15,15 +15,31 @@ def build_deepseek_v4_rope(; symbols: build_deepseek_v4_rope, touching `build_deepseek_v4_rope`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/rope.py` modified +17/-1 (18 lines); hunks: -15,15 +15,31 @@ def build_deepseek_v4_rope(; symbols: build_deepseek_v4_rope
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/rope.py
@@ -15,15 +15,31 @@ def build_deepseek_v4_rope(
+    # Newer checkpoints nest per-layer-type rope dicts ({"main", "compress"});
+    # older ones ship a single flat dict shared by all layer types.
+    if isinstance(rope_parameters.get("main"), dict) and isinstance(
+        rope_parameters.get("compress"), dict
+    ):
+        key = "compress" if compress_ratio > 1 else "main"
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/common/rope.py` modified +17/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/common/rope.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #54566 - [New model][Multimodal] Add DeepSeek-V4-Flash-Vision-Exp support

- Link: https://github.com/vllm-project/vllm/pull/54566
- Status/date: merged / 2026-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tokenizers_/test_deepseek_v4.py`, `tests/v1/attention/test_deepseek_v4_swa_visible.py`, `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/mm_preprocess.py` and 15 files; associated commits `1356635d837c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 37 files, +2918/-152, 4087 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/common/mm_preprocess.py` added +512/-0 (512 lines); hunks: -0,0 +1,512; symbols: image_sentinel_mask, validate_image_sentinel_ids, grid_tokens, solve_resize_ratio, touching `image_sentinel_mask, validate_image_sentinel_ids, grid_tokens`; `vllm/models/deepseek_v4/nvidia/vl_model.py` added +333/-0 (333 lines); hunks: -0,0 +1,333; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV4ForConditionalGeneration, get_placeholder_str, __init__, touching `_make_deepseek_v4_vl_weights_mapper, DeepseekV4ForConditionalGeneration, get_placeholder_str`; `vllm/models/deepseek_v4/common/vision.py` added +232/-0 (232 lines); hunks: -0,0 +1,232; symbols: get_vision_cos_sin, apply_rotary, DeepseekV4RMSNorm, __init__, touching `get_vision_cos_sin, apply_rotary, DeepseekV4RMSNorm`; `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +107/-27 (134 lines); hunks: -536,10 +536,16 @@ def combine_topk_swa_indices(; -568,6 +574,9 @@ def combine_topk_swa_indices(; symbols: combine_topk_swa_indices, CompileKey, kernel, touching `combine_topk_swa_indices, CompileKey, kernel`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/mm_preprocess.py` added +512/-0 (512 lines); hunks: -0,0 +1,512; symbols: image_sentinel_mask, validate_image_sentinel_ids, grid_tokens, solve_resize_ratio
  - `vllm/models/deepseek_v4/nvidia/vl_model.py` added +333/-0 (333 lines); hunks: -0,0 +1,333; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV4ForConditionalGeneration, get_placeholder_str, __init__
  - `vllm/models/deepseek_v4/common/vision.py` added +232/-0 (232 lines); hunks: -0,0 +1,232; symbols: get_vision_cos_sin, apply_rotary, DeepseekV4RMSNorm, __init__
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +107/-27 (134 lines); hunks: -536,10 +536,16 @@ def combine_topk_swa_indices(; -568,6 +574,9 @@ def combine_topk_swa_indices(; symbols: combine_topk_swa_indices, CompileKey, kernel
  - `tests/tokenizers_/test_deepseek_v4.py` modified +58/-0 (58 lines); hunks: -6,6 +6,7; -427,3 +428,60 @@ def test_deepseek_v4_encode_messages_rejects_invalid_argume...; symbols: test_deepseek_v4_encode_messages_rejects_invalid_arguments, test_deepseek_v4_image_blocks_become_placeholders, test_deepseek_v4_image_sentinel_ids_match_tokenizer, FakeTokenizer
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/mm_preprocess.py
@@ -0,0 +1,512 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Multimodal preprocessing for the DeepSeek-V4 vision variants
+(DeepSeek-V4-Flash-Vision-Exp).
+The image transform and sentinel-block construction are ported from the
+official repository's ``image_processor.py`` so that token counts bit-match
diff -- vllm/models/deepseek_v4/nvidia/vl_model.py
@@ -0,0 +1,333 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DeepSeek-V4 vision variant (e.g. DeepSeek-V4-Flash-Vision-Exp).
+Thin multimodal wrapper around the text-only ``DeepseekV4ForCausalLM``:
+- ``vision`` ViT + ``aligner`` produce per-image embeddings for the IMAGE
+  sentinel positions; four learned vectors (``image_start`` / ``image_pad`` /
diff -- vllm/models/deepseek_v4/common/vision.py
@@ -0,0 +1,232 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/common/mm_preprocess.py` added +512/-0; `vllm/models/deepseek_v4/nvidia/vl_model.py` added +333/-0; `vllm/models/deepseek_v4/common/vision.py` added +232/-0; `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +107/-27; `vllm/tokenizers/deepseek_v4_encoding.py` modified +35/-3; `vllm/models/deepseek_v4/nvidia/model.py` modified +30/-1
  - tests: `tests/tokenizers_/test_deepseek_v4.py` modified +58/-0
- Risk and verification: The diff ships test coverage in `tests/config/test_model_arch_config.py`, `tests/kernels/moe/test_topk_softplus_sqrt.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45091 - Fix DeepSeek V4 FlashMLA auto KV cache dtype

- Link: https://github.com/vllm-project/vllm/pull/45091
- Status/date: merged / 2026-09-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`; associated commits `fc8f10792c59`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +9/-4, 20 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/attention.py` modified +9/-4 (13 lines); hunks: -102,10 +102,15 @@ def _resolve_dsv4_kv_cache_dtype(; symbols: _resolve_dsv4_kv_cache_dtype, touching `_resolve_dsv4_kv_cache_dtype`.
- Code diff details:
  - `vllm/models/deepseek_v4/attention.py` modified +9/-4 (13 lines); hunks: -102,10 +102,15 @@ def _resolve_dsv4_kv_cache_dtype(; symbols: _resolve_dsv4_kv_cache_dtype
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -102,10 +102,15 @@ def _resolve_dsv4_kv_cache_dtype(
-        assert kv_cache_dtype.startswith("fp8"), (
-            f"DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, "
-            f"got {kv_cache_dtype}"
-        )
+        if kv_cache_dtype == "auto":
+            kv_cache_dtype = "fp8"
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +9/-4
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #55061 - [Performance][DSv4] Size dequant gather launch grid by rows

- Link: https://github.com/vllm-project/vllm/pull/55061
- Status/date: merged / 2026-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py`; associated commits `69cf05593606`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +6/-1, 14 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` modified +6/-1 (7 lines); hunks: -73,7 +73,12 @@ def __call__(; symbols: __call__, touching `__call__`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` modified +6/-1 (7 lines); hunks: -73,7 +73,12 @@ def __call__(; symbols: __call__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py
@@ -73,7 +73,12 @@ def __call__(
-        grid = (out.shape[0], 1024, 1)
+        num_reqs = out.shape[0]
+        max_rows = out.shape[1] - offset
+        num_workers = cutlass.max(
+            1, cutlass.min(cute.ceil_div(max_rows, 4), cute.ceil_div(8192, num_reqs))
+        )
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #55299 - [Bugfix][DSv4] Seed the -1 sentinel in the prefill sparse index workspace

- Link: https://github.com/vllm-project/vllm/pull/55299
- Status/date: merged / 2026-09-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/ops/cache_utils.py`; associated commits `7fbd44cbe0a9`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +1/-0 (1 lines); hunks: -561,6 +561,7 @@ def combine_topk_swa_indices(; symbols: combine_topk_swa_indices, touching `combine_topk_swa_indices`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +1/-0 (1 lines); hunks: -561,6 +561,7 @@ def combine_topk_swa_indices(; symbols: combine_topk_swa_indices
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/cache_utils.py
@@ -561,6 +561,7 @@ def combine_topk_swa_indices(
+        combined_indices.fill_(-1)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/common/ops/cache_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #53161 - [ROCm][Perf][DeepSeek V4] Fuse native FP8 shared expert with MXFP4 routed experts

- Link: https://github.com/vllm-project/vllm/pull/53161
- Status/date: merged / 2026-09-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`; associated commits `de69e821b7c8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +839/-6, 1031 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/amd/model.py` modified +377/-5 (382 lines); hunks: -1,7 +1,9; -20,9 +22,14; symbols: forward, _use_heterogeneous_fhmoe, _validate_heterogeneous_routes, _heterogeneous_shared_expert_enabled, touching `forward, _use_heterogeneous_fhmoe, _validate_heterogeneous_routes`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/model.py` modified +377/-5 (382 lines); hunks: -1,7 +1,9; -20,9 +22,14; symbols: forward, _use_heterogeneous_fhmoe, _validate_heterogeneous_routes, _heterogeneous_shared_expert_enabled
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -1,7 +1,9 @@
+import weakref
+from dataclasses import replace
@@ -20,9 +22,14 @@
+    FusedMoEQuantConfig,
+    RoutedExperts,
+from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/amd/model.py` modified +377/-5
- Risk and verification: The diff ships test coverage in `tests/model_executor/layers/test_fused_shared_expert.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53689 - [XPU][LoRA] Support LoRA for DeepSeek V4 on XPU

- Link: https://github.com/vllm-project/vllm/pull/53689
- Status/date: merged / 2026-09-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/xpu/model.py`; associated commits `86484465153f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +31/-1, 80 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/xpu/model.py` modified +31/-1 (32 lines); hunks: -46,6 +46,7; -1183,6 +1184,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, _make_deepseek_v4_weights_mapper, touching `load_weights, _make_deepseek_v4_weights_mapper`.
- Code diff details:
  - `vllm/models/deepseek_v4/xpu/model.py` modified +31/-1 (32 lines); hunks: -46,6 +46,7; -1183,6 +1184,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, _make_deepseek_v4_weights_mapper
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/xpu/model.py
@@ -46,6 +46,7 @@
+    SupportsLoRA,
@@ -1183,6 +1184,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+                if name not in params_dict:
+                    head, _, leaf = name.rpartition(".")
+                    suffixed = f"{head}.base_layer.{leaf}"
+                    if suffixed in params_dict:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/xpu/model.py` modified +31/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/xpu/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50176 - [4/N][warmup][DSv4] Migrate common attention kernels

- Link: https://github.com/vllm-project/vllm/pull/50176
- Status/date: merged / 2026-09-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/ops/__init__.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` and 13 files; associated commits `6ddbab03defe`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 19 files, +2978/-1272, 4923 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +903/-454 (1357 lines); hunks: -22,10 +22,17; -225,166 +232,268 @@ def quantize_and_insert_k_cache(; symbols: quantize_and_insert_k_cache, _dequantize_and_gather_k_kernel, DequantizeAndGatherKCacheKernel, CompileKey, touching `quantize_and_insert_k_cache, _dequantize_and_gather_k_kernel, DequantizeAndGatherKCacheKernel`; `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +464/-245 (709 lines); hunks: -1,8 +1,17; -67,229 +76,459 @@ def _quantize_mxfp4_pair(x_lo, x_hi):; symbols: _quantize_mxfp4_pair, _fused_indexer_q_rope_quant_kernel, FusedIndexerQRopeQuantTritonKernel, CompileKey, touching `_quantize_mxfp4_pair, _fused_indexer_q_rope_quant_kernel, FusedIndexerQRopeQuantTritonKernel`; `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` modified +286/-149 (435 lines); hunks: -17,9 +17,19; -40,164 +50,291 @@ def _rmsnorm_row(; symbols: _rmsnorm_row, _fused_mtp_input_rmsnorm_kernel, FusedMTPInputRMSNormKernel, CompileKey, touching `_rmsnorm_row, _fused_mtp_input_rmsnorm_kernel, FusedMTPInputRMSNormKernel`; `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +304/-129 (433 lines); hunks: -7,145 +7,327; -256,33 +438,23 @@ def _fused_inv_rope_fp8_quant_kernel_impl(; symbols: _fused_inv_rope_fp8_quant_per_head, FusedInvRopeFP8QuantKernel, CompileKey, varies, touching `_fused_inv_rope_fp8_quant_per_head, FusedInvRopeFP8QuantKernel, CompileKey`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +903/-454 (1357 lines); hunks: -22,10 +22,17; -225,166 +232,268 @@ def quantize_and_insert_k_cache(; symbols: quantize_and_insert_k_cache, _dequantize_and_gather_k_kernel, DequantizeAndGatherKCacheKernel, CompileKey
  - `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +464/-245 (709 lines); hunks: -1,8 +1,17; -67,229 +76,459 @@ def _quantize_mxfp4_pair(x_lo, x_hi):; symbols: _quantize_mxfp4_pair, _fused_indexer_q_rope_quant_kernel, FusedIndexerQRopeQuantTritonKernel, CompileKey
  - `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` modified +286/-149 (435 lines); hunks: -17,9 +17,19; -40,164 +50,291 @@ def _rmsnorm_row(; symbols: _rmsnorm_row, _fused_mtp_input_rmsnorm_kernel, FusedMTPInputRMSNormKernel, CompileKey
  - `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +304/-129 (433 lines); hunks: -7,145 +7,327; -256,33 +438,23 @@ def _fused_inv_rope_fp8_quant_kernel_impl(; symbols: _fused_inv_rope_fp8_quant_per_head, FusedInvRopeFP8QuantKernel, CompileKey, varies
  - `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +281/-9 (290 lines); hunks: -19,13 +19,22; -68,14 +77,32 @@ def compress_norm_rope_store_triton(; symbols: compress_norm_rope_store_triton, compress_norm_rope_store_two_stage_triton, _fused_kv_compress_norm_rope_insert_indexer_attn, _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/cache_utils.py
@@ -22,10 +22,17 @@
-from vllm.model_executor.warmup.jit_warmup import VllmJitKernel, zip_inputs
+from vllm.model_executor.warmup.jit_warmup import (
+    WarmupIntRange,
+    zip_inputs,
+)
+    LaunchSpec,
diff -- vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
@@ -1,8 +1,17 @@
+from dataclasses import dataclass
+from typing import Any
+from vllm.model_executor.warmup.jit_warmup_triton_helper import (
+    LaunchSpec,
+    TritonWarmupTensor,
+    VllmTritonJitKernel,
diff -- vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py
@@ -17,9 +17,19 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +903/-454; `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +464/-245; `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` modified +286/-149; `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +304/-129; `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +281/-9; `vllm/models/deepseek_v4/common/ops/save_partial_states.py` modified +184/-92
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/model_executor/test_jit_warmup.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56035 - [Bugfix][ROCm][DSv4] Skip launch_pdl=True JIT warmup when PDL is unsupported

- Link: https://github.com/vllm-project/vllm/pull/56035
- Status/date: merged / 2026-09-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`; associated commits `62f3bf58a504`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +2/-2, 18 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +1/-1 (2 lines); hunks: -249,7 +249,7 @@ def get_warmup_keys(self, vllm_config: Any) -> list[CompileK...; symbols: get_warmup_keys, warmup_inputs, touching `get_warmup_keys, warmup_inputs`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +1/-1 (2 lines); hunks: -249,7 +249,7 @@ def get_warmup_keys(self, vllm_config: Any) -> list[CompileK...; symbols: get_warmup_keys, warmup_inputs
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py
@@ -249,7 +249,7 @@ def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
-            launch_pdl=(False, True),
+            launch_pdl=current_platform.is_arch_support_pdl(),
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/common/ops/fused_qk_rmsnorm.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #55355 - [Model] Add DeepSeek-V4 CPU backend

- Link: https://github.com/vllm-project/vllm/pull/55355
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_deepseek_v4_cpu_kernels.py`, `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/cpu/__init__.py`, `vllm/models/deepseek_v4/cpu/cpu_compressor.py` and 11 files; associated commits `c3ccc0e957fb`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 47 files, +9564/-627, 11219 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/cpu/model.py` added +922/-0 (922 lines); hunks: -0,0 +1,922; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4MoE, touching `DeepseekV4MLP, __init__, forward`; `vllm/models/deepseek_v4/cpu/cpu_sparse.py` added +648/-0 (648 lines); hunks: -0,0 +1,648; symbols: _deepseek_v4_cpu_prepare_and_attn, _deepseek_v4_cpu_prepare_and_attn_fake, _dequant_linear_weight, _fused_indexer_q_rope_quant_cpu, touching `_deepseek_v4_cpu_prepare_and_attn, _deepseek_v4_cpu_prepare_and_attn_fake, _dequant_linear_weight`; `vllm/models/deepseek_v4/cpu/cpu_mla.py` added +126/-0 (126 lines); hunks: -0,0 +1,126; symbols: DeepseekV4CPUFlashMLAMetadataBuilder, _build_c128a_metadata, DeepseekV4CPUSparseBackend, get_name, touching `DeepseekV4CPUFlashMLAMetadataBuilder, _build_c128a_metadata, DeepseekV4CPUSparseBackend`; `vllm/models/deepseek_v4/cpu/cpu_compressor.py` added +112/-0 (112 lines); hunks: -0,0 +1,112; symbols: DeepseekV4CPUCompressor, cache_norm_weight_fp32, forward, touching `DeepseekV4CPUCompressor, cache_norm_weight_fp32, forward`.
- Code diff details:
  - `vllm/models/deepseek_v4/cpu/model.py` added +922/-0 (922 lines); hunks: -0,0 +1,922; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4MoE
  - `vllm/models/deepseek_v4/cpu/cpu_sparse.py` added +648/-0 (648 lines); hunks: -0,0 +1,648; symbols: _deepseek_v4_cpu_prepare_and_attn, _deepseek_v4_cpu_prepare_and_attn_fake, _dequant_linear_weight, _fused_indexer_q_rope_quant_cpu
  - `vllm/models/deepseek_v4/cpu/cpu_mla.py` added +126/-0 (126 lines); hunks: -0,0 +1,126; symbols: DeepseekV4CPUFlashMLAMetadataBuilder, _build_c128a_metadata, DeepseekV4CPUSparseBackend, get_name
  - `vllm/models/deepseek_v4/cpu/cpu_compressor.py` added +112/-0 (112 lines); hunks: -0,0 +1,112; symbols: DeepseekV4CPUCompressor, cache_norm_weight_fp32, forward
  - `vllm/models/deepseek_v4/cpu/cpu_utils.py` added +34/-0 (34 lines); hunks: -0,0 +1,34; symbols: map_local_to_global_slots_cpu
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/cpu/model.py
@@ -0,0 +1,922 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/models/deepseek_v4/cpu/cpu_sparse.py
@@ -0,0 +1,648 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""CPU DeepSeek-V4 attention subclass.
+``forward_mqa`` resolves SWA/compressed top-k indices to paged-cache slot
+ids and calls the fused ``flash_mla_with_kvcache_cpu`` kernel. ``_o_proj``
+stays eager -- not on the attention hot path.
diff -- vllm/models/deepseek_v4/cpu/cpu_mla.py
@@ -0,0 +1,126 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/cpu/model.py` added +922/-0; `vllm/models/deepseek_v4/cpu/cpu_sparse.py` added +648/-0; `vllm/models/deepseek_v4/cpu/cpu_mla.py` added +126/-0; `vllm/models/deepseek_v4/cpu/cpu_compressor.py` added +112/-0; `vllm/models/deepseek_v4/cpu/cpu_utils.py` added +34/-0; `vllm/models/deepseek_v4/cpu/mtp.py` added +19/-0
- Risk and verification: The diff ships test coverage in `.buildkite/hardware_tests/cpu.yaml`, `tests/kernels/moe/test_cpu_fused_moe.py`, `tests/kernels/moe/test_cpu_quant_fused_moe.py`, `tests/kernels/moe/test_zen_cpu_fused_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56215 - [Kernel] Optional Q-norm in fused DSv4 MLA epilogue; group_size=32 for packed FP8 quant

- Link: https://github.com/vllm-project/vllm/pull/56215
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`; associated commits `be1cb9834b3d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +267/-148, 857 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +99/-13 (112 lines); hunks: -4,11 +4,11; -146,9 +146,18 @@ def _full_cache_bf16_op_available() -> bool:; symbols: _full_cache_bf16_op_available, _call_fused, _as_stored_fp8, test_q_path_matches_reference, touching `_full_cache_bf16_op_available, _call_fused, _as_stored_fp8`.
- Code diff details:
  - `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +99/-13 (112 lines); hunks: -4,11 +4,11; -146,9 +146,18 @@ def _full_cache_bf16_op_available() -> bool:; symbols: _full_cache_bf16_op_available, _call_fused, _as_stored_fp8, test_q_path_matches_reference
- Key code excerpts:

```diff
diff -- tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py
@@ -4,11 +4,11 @@
-    - Q side:  per-head RMSNorm (no weight) + GPT-J RoPE on last 64 dims
+    - Q side:  optional per-head RMSNorm + GPT-J RoPE on last 64 dims
-  - PyTorch reference for RMSNorm + GPT-J RoPE on Q
+  - PyTorch references for RoPE with and without RMSNorm on Q
@@ -146,9 +146,18 @@ def _full_cache_bf16_op_available() -> bool:
-    q_in, q_head_padded, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, bs
```

- Extracted files (not manually reviewed):
  - tests: `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +99/-13
- Risk and verification: The diff ships test coverage in `tests/kernels/quantization/test_per_token_group_quant.py`, `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56208 - [Model][Frontend] Support DeepSeek-V4.1-Flash in Rust and Python frontends

- Link: https://github.com/vllm-project/vllm/pull/56208
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_deepseek_v41.py`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt`, `tests/tokenizers_/test_deepseek_v41.py`, `vllm/parser/deepseek_v4.py` and 10 files; associated commits `b47b01cf3383`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 52 files, +2986/-113, 3929 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/tokenizers/deepseek_v41_encoding.py` added +577/-0 (577 lines); hunks: -0,0 +1,577; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, merge_tool_messages, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`; `tests/tokenizers_/test_deepseek_v41.py` added +169/-0 (169 lines); hunks: -0,0 +1,169; symbols: render, test_reference_encoder_fixtures, test_numeric_reasoning_effort, test_invalid_effort_is_a_request_error, touching `render, test_reference_encoder_fixtures, test_numeric_reasoning_effort`; `vllm/tokenizers/deepseek_v41.py` added +121/-0 (121 lines); hunks: -0,0 +1,121; symbols: _normalize_messages, get_deepseek_v41_tokenizer, _DeepseekV41Tokenizer, apply_chat_template, touching `_normalize_messages, get_deepseek_v41_tokenizer, _DeepseekV41Tokenizer`; `vllm/transformers_utils/configs/deepseek_v41.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: DeepseekV41Config, __init__, touching `DeepseekV41Config, __init__`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v41_encoding.py` added +577/-0 (577 lines); hunks: -0,0 +1,577; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, merge_tool_messages
  - `tests/tokenizers_/test_deepseek_v41.py` added +169/-0 (169 lines); hunks: -0,0 +1,169; symbols: render, test_reference_encoder_fixtures, test_numeric_reasoning_effort, test_invalid_effort_is_a_request_error
  - `vllm/tokenizers/deepseek_v41.py` added +121/-0 (121 lines); hunks: -0,0 +1,121; symbols: _normalize_messages, get_deepseek_v41_tokenizer, _DeepseekV41Tokenizer, apply_chat_template
  - `vllm/transformers_utils/configs/deepseek_v41.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: DeepseekV41Config, __init__
  - `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` added +38/-0 (38 lines); hunks: -0,0 +1,38
- Key code excerpts:

```diff
diff -- vllm/tokenizers/deepseek_v41_encoding.py
@@ -0,0 +1,577 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa
+"""Text encoding adapted from the DeepSeek V4.1 reference encoder.
+Message normalization and serving controls live in deepseek_v41.py.
+"""
diff -- tests/tokenizers_/test_deepseek_v41.py
@@ -0,0 +1,169 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import copy
+import json
+from pathlib import Path
+import pytest
diff -- vllm/tokenizers/deepseek_v41.py
@@ -0,0 +1,121 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/tokenizers/deepseek_v41_encoding.py` added +577/-0; `vllm/tokenizers/deepseek_v41.py` added +121/-0; `vllm/transformers_utils/configs/deepseek_v41.py` added +76/-0; `vllm/reasoning/deepseek_v41_engine_reasoning_parser.py` added +6/-0
  - tests: `tests/tokenizers_/test_deepseek_v41.py` added +169/-0; `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` added +38/-0; `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt` added +3/-0; `tests/parser/engine/test_deepseek_v41.py` added +146/-0
- Risk and verification: The diff ships test coverage in `rust/src/chat/tests/roundtrip.rs`, `tests/parser/engine/test_deepseek_v41.py`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56228 - [Model] DeepSeek-V4.1-Flash Model Definitions

- Link: https://github.com/vllm-project/vllm/pull/56228
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/common/vision.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` and 6 files; associated commits `9b959b86577c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 47 files, +12902/-196, 14210 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4_1/nvidia/model.py` added +1111/-0 (1111 lines); hunks: -0,0 +1,1111; symbols: DeepseekV4MoE, __init__, _select_dsv4_attn_cls, for, touching `DeepseekV4MoE, __init__, _select_dsv4_attn_cls`; `vllm/models/deepseek_v4_1/amd/model.py` added +1086/-0 (1086 lines); hunks: -0,0 +1,1086; symbols: DeepseekV4MoE, __init__, _select_dsv4_attn_cls, _use_sequence_parallel, touching `DeepseekV4MoE, __init__, _select_dsv4_attn_cls`; `vllm/models/deepseek_v4_1/amd/vl_model.py` added +357/-0 (357 lines); hunks: -0,0 +1,357; symbols: DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, get_placeholder_str, touching `DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM`; `vllm/models/deepseek_v4_1/nvidia/vl_model.py` added +357/-0 (357 lines); hunks: -0,0 +1,357; symbols: DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, get_placeholder_str, touching `DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM`.
- Code diff details:
  - `vllm/models/deepseek_v4_1/nvidia/model.py` added +1111/-0 (1111 lines); hunks: -0,0 +1,1111; symbols: DeepseekV4MoE, __init__, _select_dsv4_attn_cls, for
  - `vllm/models/deepseek_v4_1/amd/model.py` added +1086/-0 (1086 lines); hunks: -0,0 +1,1086; symbols: DeepseekV4MoE, __init__, _select_dsv4_attn_cls, _use_sequence_parallel
  - `vllm/models/deepseek_v4_1/amd/vl_model.py` added +357/-0 (357 lines); hunks: -0,0 +1,357; symbols: DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, get_placeholder_str
  - `vllm/models/deepseek_v4_1/nvidia/vl_model.py` added +357/-0 (357 lines); hunks: -0,0 +1,357; symbols: DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, get_placeholder_str
  - `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +105/-70 (175 lines); hunks: -32,37 +32,39 @@ class CompileKey:; -74,8 +76,8 @@ def kernel(; symbols: CompileKey, varies, kernel
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4_1/nvidia/model.py
@@ -0,0 +1,1111 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/models/deepseek_v4_1/amd/model.py
@@ -0,0 +1,1086 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/models/deepseek_v4_1/amd/vl_model.py
@@ -0,0 +1,357 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4_1/nvidia/model.py` added +1111/-0; `vllm/models/deepseek_v4_1/amd/model.py` added +1086/-0; `vllm/models/deepseek_v4_1/amd/vl_model.py` added +357/-0; `vllm/models/deepseek_v4_1/nvidia/vl_model.py` added +357/-0; `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +105/-70; `vllm/models/deepseek_v4/common/vision.py` modified +93/-2
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +47/-11
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_mega_moe.py`, `tests/parser/engine/trace_builder.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51692 - [ROCm][Perf] Add bpreshuffled blockscaled fp8 GEMM

- Link: https://github.com/vllm-project/vllm/pull/51692
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_vl_rocm.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`; associated commits `bc0f47cd03d6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +159/-5, 217 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/kernels/linear/scaled_mm/aiter.py` modified +128/-0 (128 lines); hunks: -457,3 +457,131 @@ def apply_block_scaled_mm(; symbols: apply_block_scaled_mm, AiterPreshuffledFp8BlockScaledMMKernel, is_supported, can_implement, touching `apply_block_scaled_mm, AiterPreshuffledFp8BlockScaledMMKernel, is_supported`; `vllm/model_executor/kernels/linear/__init__.py` modified +4/-0 (4 lines); hunks: -178,6 +178,7; -316,6 +317,7 @@ def _get_linear_backend() -> str:; symbols: _get_linear_backend, _resolve_backend_kernels, register_linear_kernel, touching `_get_linear_backend, _resolve_backend_kernels, register_linear_kernel`; `vllm/_aiter_ops.py` modified +27/-5 (32 lines); hunks: -175,18 +175,26 @@ def is_aiter_found_and_supported_on_rdna4() -> bool:; -3200,6 +3208,20 @@ def is_per_token_w8a8_gemm_tuned(N: int, K: int, q_dtype_...; symbols: is_aiter_found_and_supported_on_rdna4, _load_gemm_tuned_configs, _check_kernel_tuned, is_per_token_w8a8_gemm_tuned, touching `is_aiter_found_and_supported_on_rdna4, _load_gemm_tuned_configs, _check_kernel_tuned`.
- Code diff details:
  - `vllm/model_executor/kernels/linear/scaled_mm/aiter.py` modified +128/-0 (128 lines); hunks: -457,3 +457,131 @@ def apply_block_scaled_mm(; symbols: apply_block_scaled_mm, AiterPreshuffledFp8BlockScaledMMKernel, is_supported, can_implement
  - `vllm/model_executor/kernels/linear/__init__.py` modified +4/-0 (4 lines); hunks: -178,6 +178,7; -316,6 +317,7 @@ def _get_linear_backend() -> str:; symbols: _get_linear_backend, _resolve_backend_kernels, register_linear_kernel
  - `vllm/_aiter_ops.py` modified +27/-5 (32 lines); hunks: -175,18 +175,26 @@ def is_aiter_found_and_supported_on_rdna4() -> bool:; -3200,6 +3208,20 @@ def is_per_token_w8a8_gemm_tuned(N: int, K: int, q_dtype_...; symbols: is_aiter_found_and_supported_on_rdna4, _load_gemm_tuned_configs, _check_kernel_tuned, is_per_token_w8a8_gemm_tuned
- Key code excerpts:

```diff
diff -- vllm/model_executor/kernels/linear/scaled_mm/aiter.py
@@ -457,3 +457,131 @@ def apply_block_scaled_mm(
+class AiterPreshuffledFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
+    """Aiter FP8 block-scaled GEMM using a pre-shuffled (bpreshuffle) weight."""
+    @classmethod
+    def is_supported(
+        cls, compute_capability: int | None = None
+    ) -> tuple[bool, str | None]:
diff -- vllm/model_executor/kernels/linear/__init__.py
@@ -178,6 +178,7 @@
+    AiterPreshuffledFp8BlockScaledMMKernel,
@@ -316,6 +317,7 @@ def _get_linear_backend() -> str:
+        AiterPreshuffledFp8BlockScaledMMKernel,
@@ -456,6 +458,7 @@ def _resolve_backend_kernels(
+        AiterPreshuffledFp8BlockScaledMMKernel,
@@ -1210,6 +1213,7 @@ def register_linear_kernel(
diff -- vllm/_aiter_ops.py
@@ -175,18 +175,26 @@ def is_aiter_found_and_supported_on_rdna4() -> bool:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/kernels/linear/scaled_mm/aiter.py` modified +128/-0; `vllm/model_executor/kernels/linear/__init__.py` modified +4/-0; `vllm/_aiter_ops.py` modified +27/-5
- Risk and verification: Runtime changes concentrate in `vllm/_aiter_ops.py`, `vllm/model_executor/kernels/linear/__init__.py`, `vllm/model_executor/kernels/linear/scaled_mm/aiter.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #56214 - [Model] Support DeepSeek-V4.1-Flash

- Link: https://github.com/vllm-project/vllm/pull/56214
- Status/date: merged / 2026-09-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_mega_moe.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`; associated commits `e77daef89e18`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 49 files, +3996/-138, 4994 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/test_deepseek_v4_mega_moe.py` modified +211/-4 (215 lines); hunks: -18,14 +18,147; -169,15 +302,79 @@ def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert...; symbols: v41_moe_config, test_deepseek_v41_moe_routes_without_hash_table, check_routing, test_deepseek_v41_fused_moe_uses_draft_counts_or_main_defaults, touching `v41_moe_config, test_deepseek_v41_moe_routes_without_hash_table, check_routing`; `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +43/-1 (44 lines); hunks: -7,19 +7,61; symbols: test_indexer_shares_uncompressed_block_size_with_deepseek_v4_mla, test_indexer_preserves_deepseek_v41_mla_block_size, test_indexer_warmup_normalizes_zero_compress_ratios, touching `test_indexer_shares_uncompressed_block_size_with_deepseek_v4_mla, test_indexer_preserves_deepseek_v41_mla_block_size, test_indexer_warmup_normalizes_zero_compress_ratios`; `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +9/-17 (26 lines); hunks: -937,14 +937,13 @@ def w8a8_triton_block_scaled_mm(; -1089,8 +1088,6 @@ def deepgemm_post_process_weight_scale_block(; symbols: w8a8_triton_block_scaled_mm, deepgemm_post_process_weight_scale_block, deepgemm_post_process_fp8_weight_block, touching `w8a8_triton_block_scaled_mm, deepgemm_post_process_weight_scale_block, deepgemm_post_process_fp8_weight_block`; `vllm/model_executor/layers/quantization/__init__.py` modified +15/-2 (17 lines); hunks: -115,7 +115,20 @@ def get_quantization_config(quantization: str) -> type[Quan...; -161,7 +174,7 @@ def get_quantization_config(quantization: str) -> type[Quant...; symbols: get_quantization_config, is, touching `get_quantization_config, is`.
- Code diff details:
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +211/-4 (215 lines); hunks: -18,14 +18,147; -169,15 +302,79 @@ def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert...; symbols: v41_moe_config, test_deepseek_v41_moe_routes_without_hash_table, check_routing, test_deepseek_v41_fused_moe_uses_draft_counts_or_main_defaults
  - `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +43/-1 (44 lines); hunks: -7,19 +7,61; symbols: test_indexer_shares_uncompressed_block_size_with_deepseek_v4_mla, test_indexer_preserves_deepseek_v41_mla_block_size, test_indexer_warmup_normalizes_zero_compress_ratios
  - `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +9/-17 (26 lines); hunks: -937,14 +937,13 @@ def w8a8_triton_block_scaled_mm(; -1089,8 +1088,6 @@ def deepgemm_post_process_weight_scale_block(; symbols: w8a8_triton_block_scaled_mm, deepgemm_post_process_weight_scale_block, deepgemm_post_process_fp8_weight_block
  - `vllm/model_executor/layers/quantization/__init__.py` modified +15/-2 (17 lines); hunks: -115,7 +115,20 @@ def get_quantization_config(quantization: str) -> type[Quan...; -161,7 +174,7 @@ def get_quantization_config(quantization: str) -> type[Quant...; symbols: get_quantization_config, is
  - `vllm/model_executor/layers/fusion/quant_activation.py` modified +11/-1 (12 lines); hunks: -8,7 +8,7; -34,6 +34,16 @@ class QuantizedActivation:; symbols: QuantizedActivation, weak_ref, expose_input_quant_key
- Key code excerpts:

```diff
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -18,14 +18,147 @@
+from vllm.models.deepseek_v4_1.common.mm_preprocess import (
+    IMAGE_PAD_ID,
+    IMAGE_SENTINEL_BASE_ID,
+)
+from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4MoE as DeepseekV41MoE
+from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config
diff -- tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py
@@ -7,19 +7,61 @@
+from vllm.models.deepseek_v4.sparse_mla import DeepseekV4SparseMLABackend
+from vllm.models.deepseek_v4_1.sparse_mla import (
+    DeepseekV4SparseMLABackend as DeepseekV41SparseMLABackend,
+)
+    DeepseekV4IndexerBackend,
+    DeepseekV41IndexerBackend,
diff -- vllm/model_executor/layers/quantization/utils/fp8_utils.py
@@ -937,14 +937,13 @@ def w8a8_triton_block_scaled_mm(
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +211/-4; `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +43/-1
  - runtime: `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +9/-17; `vllm/model_executor/layers/quantization/__init__.py` modified +15/-2; `vllm/model_executor/layers/fusion/quant_activation.py` modified +11/-1; `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py` modified +8/-2; `vllm/model_executor/models/registry.py` modified +8/-0; `vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py` modified +6/-1
- Risk and verification: The diff ships test coverage in `tests/fusion/test_quant_activation_contract.py`, `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/quantization/test_rocm_mxfp8_linear.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56433 - [ROCm][Bugfix] Fix AITER preshuffled FP8 block-scale kernel

- Link: https://github.com/vllm-project/vllm/pull/56433
- Status/date: merged / 2026-09-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_vl_rocm.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`; associated commits `b4da4d17ae0c`, `bc0f47cd03d6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +120/-36, 279 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/amd/rocm.py` modified +54/-16 (70 lines); hunks: -61,6 +61,16 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> torc...; -551,12 +561,13 @@ def _prep(linear) -> torch.Tensor | None:; symbols: _build_indptr_from_lengths, weight_already_preshuffled, apply_pre_quantized_block_scaled_mm, _prep, touching `_build_indptr_from_lengths, weight_already_preshuffled, apply_pre_quantized_block_scaled_mm`; `vllm/models/deepseek_v4/amd/model.py` modified +11/-6 (17 lines); hunks: -71,7 +71,10; -147,11 +150,13 @@ def prepare_gateup_preshuffle(self) -> None:; symbols: prepare_gateup_preshuffle, forward, touching `prepare_gateup_preshuffle, forward`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +54/-16 (70 lines); hunks: -61,6 +61,16 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> torc...; -551,12 +561,13 @@ def _prep(linear) -> torch.Tensor | None:; symbols: _build_indptr_from_lengths, weight_already_preshuffled, apply_pre_quantized_block_scaled_mm, _prep
  - `vllm/models/deepseek_v4/amd/model.py` modified +11/-6 (17 lines); hunks: -71,7 +71,10; -147,11 +150,13 @@ def prepare_gateup_preshuffle(self) -> None:; symbols: prepare_gateup_preshuffle, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -61,6 +61,16 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
+def weight_already_preshuffled(linear: torch.nn.Module) -> bool:
+    """True when the linear's kernel already B-preshuffled ``weight``.
+    The hand-shuffles below (fused_wqa_wkv, wo_b, gate_up_proj) must be skipped
+    for those, since shuffle_weight is a permutation rather than an involution.
+    """
+    kernel = getattr(getattr(linear, "quant_method", None), "fp8_linear", None)
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -71,7 +71,10 @@
-from vllm.models.deepseek_v4.amd.rocm import DeepseekV4ROCMAiterMLAAttention
+from vllm.models.deepseek_v4.amd.rocm import (
+    DeepseekV4ROCMAiterMLAAttention,
+    weight_already_preshuffled,
+)
@@ -147,11 +150,13 @@ def prepare_gateup_preshuffle(self) -> None:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/amd/rocm.py` modified +54/-16; `vllm/models/deepseek_v4/amd/model.py` modified +11/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/kernels/linear/scaled_mm/aiter.py`, `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #55107 - [Model][ROCm] Enable DeepSeek V4 Vision

- Link: https://github.com/vllm-project/vllm/pull/55107
- Status/date: merged / 2026-09-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_vl_rocm.py`, `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/amd/rocm.py` and 9 files; associated commits `9dd969da096e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 17 files, +1055/-347, 1656 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/test_deepseek_v4_vl_rocm.py` added +458/-0 (458 lines); hunks: -0,0 +1,458; symbols: _preshuffled_fp8_linear, test_rocm_wo_a_keeps_row_major_weights, test_rocm_gateup_shuffles_once_and_down_proj_keeps_preshuffled_backend, test_rocm_fused_qkv_quant_matches_preshuffled_gemm_scale_layout, touching `_preshuffled_fp8_linear, test_rocm_wo_a_keeps_row_major_weights, test_rocm_gateup_shuffles_once_and_down_proj_keeps_preshuffled_backend`; `vllm/models/deepseek_v4/nvidia/vl_model.py` modified +10/-322 (332 lines); hunks: -1,333 +1,21; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV4ForConditionalGeneration, get_placeholder_str, __init__, touching `_make_deepseek_v4_vl_weights_mapper, DeepseekV4ForConditionalGeneration, get_placeholder_str`; `vllm/models/deepseek_v4/common/vl_model.py` added +331/-0 (331 lines); hunks: -0,0 +1,331; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV4ForConditionalGeneration, get_placeholder_str, __init__, touching `_make_deepseek_v4_vl_weights_mapper, DeepseekV4ForConditionalGeneration, get_placeholder_str`; `vllm/models/deepseek_v4/amd/rocm.py` modified +61/-7 (68 lines); hunks: -67,8 +67,13 @@ def weight_already_preshuffled(linear: torch.nn.Module) -> bool:; -116,13 +121,18 @@ def _combine_topk_swa_indices_kernel(; symbols: weight_already_preshuffled, apply_pre_quantized_block_scaled_mm, _combine_topk_swa_indices_kernel, touching `weight_already_preshuffled, apply_pre_quantized_block_scaled_mm, _combine_topk_swa_indices_kernel`.
- Code diff details:
  - `tests/models/test_deepseek_v4_vl_rocm.py` added +458/-0 (458 lines); hunks: -0,0 +1,458; symbols: _preshuffled_fp8_linear, test_rocm_wo_a_keeps_row_major_weights, test_rocm_gateup_shuffles_once_and_down_proj_keeps_preshuffled_backend, test_rocm_fused_qkv_quant_matches_preshuffled_gemm_scale_layout
  - `vllm/models/deepseek_v4/nvidia/vl_model.py` modified +10/-322 (332 lines); hunks: -1,333 +1,21; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV4ForConditionalGeneration, get_placeholder_str, __init__
  - `vllm/models/deepseek_v4/common/vl_model.py` added +331/-0 (331 lines); hunks: -0,0 +1,331; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV4ForConditionalGeneration, get_placeholder_str, __init__
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +61/-7 (68 lines); hunks: -67,8 +67,13 @@ def weight_already_preshuffled(linear: torch.nn.Module) -> bool:; -116,13 +121,18 @@ def _combine_topk_swa_indices_kernel(; symbols: weight_already_preshuffled, apply_pre_quantized_block_scaled_mm, _combine_topk_swa_indices_kernel
  - `vllm/models/deepseek_v4/amd/model.py` modified +23/-1 (24 lines); hunks: -79,6 +79,8; -549,6 +551,10 @@ def __init__(; symbols: __init__, forward, _make_deepseek_v4_weights_mapper
- Key code excerpts:

```diff
diff -- tests/models/test_deepseek_v4_vl_rocm.py
@@ -0,0 +1,458 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from types import SimpleNamespace
+import pytest
+import torch
+from torch import nn
diff -- vllm/models/deepseek_v4/nvidia/vl_model.py
@@ -1,333 +1,21 @@
-"""DeepSeek-V4 vision variant (e.g. DeepSeek-V4-Flash-Vision-Exp).
+"""Compatibility imports for the platform-neutral DeepSeek-V4 vision model."""
-Thin multimodal wrapper around the text-only ``DeepseekV4ForCausalLM``:
-- ``vision`` ViT + ``aligner`` produce per-image embeddings for the IMAGE
-  sentinel positions; four learned vectors (``image_start`` / ``image_pad`` /
-  ``image_newline`` / ``image_end``) fill the remaining sentinel positions.
diff -- vllm/models/deepseek_v4/common/vl_model.py
@@ -0,0 +1,331 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/test_deepseek_v4_vl_rocm.py` added +458/-0
  - runtime: `vllm/models/deepseek_v4/nvidia/vl_model.py` modified +10/-322; `vllm/models/deepseek_v4/common/vl_model.py` added +331/-0; `vllm/models/deepseek_v4/amd/rocm.py` modified +61/-7; `vllm/models/deepseek_v4/amd/model.py` modified +23/-1; `vllm/models/deepseek_v4/__init__.py` modified +4/-4; `vllm/models/deepseek_v4/vl_stub.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/test_deepseek_v4_vl_rocm.py`, `tests/models/test_initialization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53566 - [5/N][warmup][DSv4] Migrate NVIDIA CuTeDSL attention kernels

- Link: https://github.com/vllm-project/vllm/pull/53566
- Status/date: merged / 2026-09-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` and 12 files; associated commits `120ec4ebd280`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 28 files, +3108/-2535, 2685 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +1945/-1606 (3551 lines); `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` modified +591/-510 (1101 lines); hunks: -1,6 +1,9; -14,601 +17,679; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, fused_indexer_q_rope_quant_fp8_cutedsl, _load_q_and_rope, IndexerQRopeQuantKernel, touching `fused_indexer_q_rope_quant_mxfp4_cutedsl, fused_indexer_q_rope_quant_fp8_cutedsl, _load_q_and_rope`; `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` modified +346/-279 (625 lines); hunks: -1,7 +1,8; -12,302 +13,340; symbols: dequantize_and_gather_k_cache_cutedsl, DequantGatherKCacheKernel, __init__, __call__, touching `dequantize_and_gather_k_cache_cutedsl, DequantGatherKCacheKernel, __init__`; `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +22/-22 (44 lines); hunks: -6,11 +6,11; -598,19 +598,19 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant, touching `fused_indexer_q_rope_quant`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +1945/-1606 (3551 lines)
  - `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` modified +591/-510 (1101 lines); hunks: -1,6 +1,9; -14,601 +17,679; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, fused_indexer_q_rope_quant_fp8_cutedsl, _load_q_and_rope, IndexerQRopeQuantKernel
  - `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` modified +346/-279 (625 lines); hunks: -1,7 +1,8; -12,302 +13,340; symbols: dequantize_and_gather_k_cache_cutedsl, DequantGatherKCacheKernel, __init__, __call__
  - `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +22/-22 (44 lines); hunks: -6,11 +6,11; -598,19 +598,19 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant
  - `vllm/models/deepseek_v4/compressor.py` modified +31/-5 (36 lines); hunks: -197,8 +197,8 @@ class DeepseekCompressor(nn.Module):; -311,7 +311,33 @@ def __init__(; symbols: DeepseekCompressor, __init__, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py
@@ -1,6 +1,9 @@
-from functools import cache
+# ruff: noqa: E501
+from dataclasses import dataclass
+from typing import Any
@@ -14,601 +17,679 @@
+    torch_to_cute_dtype,
diff -- vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py
@@ -1,7 +1,8 @@
-from functools import cache
+from dataclasses import dataclass
+from typing import Any
@@ -12,302 +13,340 @@
+from vllm.model_executor.warmup.jit_warmup import kernel_launcher, zip_inputs
+from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
diff -- vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
@@ -6,11 +6,11 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +1945/-1606; `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` modified +591/-510; `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` modified +346/-279; `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +22/-22; `vllm/models/deepseek_v4/compressor.py` modified +31/-5; `vllm/models/deepseek_v4/attention.py` modified +23/-3
- Risk and verification: The diff ships test coverage in `tests/model_executor/test_jit_warmup_cutedsl_launcher.py`, `tests/model_executor/test_jit_warmup_triton_launcher.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56554 - [DSV4.1] Remove compressor-aware image sentinel token padding

- Link: https://github.com/vllm-project/vllm/pull/56554
- Status/date: merged / 2026-09-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/transformers_utils/configs/deepseek_v41.py`; associated commits `30118ba27d1d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +10/-132, 245 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/transformers_utils/configs/deepseek_v41.py` modified +0/-6 (6 lines); hunks: -68,9 +68,3 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/transformers_utils/configs/deepseek_v41.py` modified +0/-6 (6 lines); hunks: -68,9 +68,3 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/deepseek_v41.py
@@ -68,9 +68,3 @@ def __init__(
-        # The visibility span covers the image block [IMAGE_START,
-        # IMAGE_END]; the mm placeholder additionally carries a leading
-        # compressor-alignment pad of ``COMPRESS_PAD_TO - 1 - offset %
-        # COMPRESS_PAD_TO`` tokens (see common/mm_preprocess.py), with
-        # COMPRESS_PAD_TO = 2 for v4.1's ratio-2 compressors.
-        self.mm_prefix_span_leading_pad_modulus = 2 if vision_n_layers > 0 else 0
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/transformers_utils/configs/deepseek_v41.py` modified +0/-6
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4_1/amd/vl_model.py`, `vllm/models/deepseek_v4_1/common/mm_preprocess.py`, `vllm/models/deepseek_v4_1/nvidia/vl_model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #56562 - [Perf] Fuse DSV4.1 input metadata preparation with Triton

- Link: https://github.com/vllm-project/vllm/pull/56562
- Status/date: merged / 2026-09-12
- Trace source: `git log --name-only -- <model-files>` found it through `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`; associated commits `13e221f83084`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +274/-63, 383 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +105/-0 (105 lines); hunks: -32,6 +32,111; symbols: test_fused_indexer_decode_metadata, test_device_token_request_mapping, test_indexer_shares_uncompressed_block_size_with_deepseek_v4_mla, touching `test_fused_indexer_decode_metadata, test_device_token_request_mapping, test_indexer_shares_uncompressed_block_size_with_deepseek_v4_mla`; `vllm/v1/attention/backends/mla/indexer.py` modified +69/-52 (121 lines); hunks: -960,28 +960,6 @@ def _prepare_global_decode_seq_lens(; -1156,12 +1134,15 @@ def build(; symbols: _prepare_global_decode_seq_lens, _build_varlen_decode_indices, _split_indexer_prefill_chunks, build, touching `_prepare_global_decode_seq_lens, _build_varlen_decode_indices, _split_indexer_prefill_chunks`; `vllm/v1/attention/ops/metadata.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: _token_request, _token_request_mapping_kernel, _indexer_decode_metadata_kernel, touching `_token_request, _token_request_mapping_kernel, _indexer_decode_metadata_kernel`; `vllm/v1/attention/backend.py` modified +13/-11 (24 lines); hunks: -483,19 +483,21 @@ def token_to_req_indices(self, buffer: torch.Tensor) -> to...; symbols: token_to_req_indices, touching `token_to_req_indices`.
- Code diff details:
  - `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +105/-0 (105 lines); hunks: -32,6 +32,111; symbols: test_fused_indexer_decode_metadata, test_device_token_request_mapping, test_indexer_shares_uncompressed_block_size_with_deepseek_v4_mla
  - `vllm/v1/attention/backends/mla/indexer.py` modified +69/-52 (121 lines); hunks: -960,28 +960,6 @@ def _prepare_global_decode_seq_lens(; -1156,12 +1134,15 @@ def build(; symbols: _prepare_global_decode_seq_lens, _build_varlen_decode_indices, _split_indexer_prefill_chunks, build
  - `vllm/v1/attention/ops/metadata.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: _token_request, _token_request_mapping_kernel, _indexer_decode_metadata_kernel
  - `vllm/v1/attention/backend.py` modified +13/-11 (24 lines); hunks: -483,19 +483,21 @@ def token_to_req_indices(self, buffer: torch.Tensor) -> to...; symbols: token_to_req_indices
- Key code excerpts:

```diff
diff -- tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py
@@ -32,6 +32,111 @@
+@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
+@pytest.mark.parametrize("query_lens", [[1], [6] * 16, [0, 2, 6, 0, 4], [0, 0, 0]])
+@pytest.mark.parametrize("padding", [0, 17])
+def test_fused_indexer_decode_metadata(query_lens, padding):
+    """Flatten device query boundaries and clear graph padding on every replay."""
+    from vllm.v1.attention.ops.metadata import _indexer_decode_metadata_kernel
diff -- vllm/v1/attention/backends/mla/indexer.py
@@ -960,28 +960,6 @@ def _prepare_global_decode_seq_lens(
-    def _build_varlen_decode_indices(
-        self,
-        decode_lens: torch.Tensor,
-        decode_lens_cpu: torch.Tensor,
-        num_decode_tokens: int,
-    ) -> torch.Tensor:
diff -- vllm/v1/attention/ops/metadata.py
@@ -0,0 +1,87 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +105/-0
  - runtime: `vllm/v1/attention/backends/mla/indexer.py` modified +69/-52; `vllm/v1/attention/ops/metadata.py` added +87/-0; `vllm/v1/attention/backend.py` modified +13/-11
- Risk and verification: The diff ships test coverage in `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56599 - [CI] Update DeepSeek V4.1 MegaMoE routing test

- Link: https://github.com/vllm-project/vllm/pull/56599
- Status/date: merged / 2026-09-12
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_mega_moe.py`; associated commits `1ee4be4dbc57`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +3/-6, 30 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/test_deepseek_v4_mega_moe.py` modified +3/-6 (9 lines); hunks: -18,10 +18,7; -78,7 +75,7 @@ def test_deepseek_v41_moe_routes_without_hash_table(; symbols: test_deepseek_v41_moe_routes_without_hash_table, touching `test_deepseek_v41_moe_routes_without_hash_table`.
- Code diff details:
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +3/-6 (9 lines); hunks: -18,10 +18,7; -78,7 +75,7 @@ def test_deepseek_v41_moe_routes_without_hash_table(; symbols: test_deepseek_v41_moe_routes_without_hash_table
- Key code excerpts:

```diff
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -18,10 +18,7 @@
-from vllm.models.deepseek_v4_1.common.mm_preprocess import (
-    IMAGE_PAD_ID,
-    IMAGE_SENTINEL_BASE_ID,
-)
+from vllm.models.deepseek_v4_1.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID
@@ -78,7 +75,7 @@ def test_deepseek_v41_moe_routes_without_hash_table(
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +3/-6
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_mega_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50178 - [9/N][warmup][DSv4] Migrate MHC TileLang kernels

- Link: https://github.com/vllm-project/vllm/pull/50178
- Status/date: merged / 2026-09-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `e52be1a62d38`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +1454/-489, 2222 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/nvidia/model.py` modified +59/-0 (59 lines); hunks: -1210,6 +1210,53 @@ def __init__(; -1402,6 +1449,18 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: st...; symbols: __init__, forward, embed_input_ids, touching `__init__, forward, embed_input_ids`; `vllm/models/glm5next/nvidia/model.py` modified +44/-0 (44 lines); hunks: -403,6 +403,50 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +59/-0 (59 lines); hunks: -1210,6 +1210,53 @@ def __init__(; -1402,6 +1449,18 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: st...; symbols: __init__, forward, embed_input_ids
  - `vllm/models/glm5next/nvidia/model.py` modified +44/-0 (44 lines); hunks: -403,6 +403,50 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -1210,6 +1210,53 @@ def __init__(
+        if vllm_config.kernel_config.enable_jit_warmup:
+            from vllm.model_executor.kernels.mhc.tilelang_kernels import (
+                _HC_PRENORM_GEMM_TILELANG_KERNEL,
+                _MHC_FUSED_TILELANG_KERNEL,
+                _MHC_POST_TILELANG_KERNEL,
+                _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
diff -- vllm/models/glm5next/nvidia/model.py
@@ -403,6 +403,50 @@ def __init__(
+            if vllm_config.kernel_config.enable_jit_warmup:
+                from vllm.model_executor.kernels.mhc.tilelang_kernels import (
+                    _HC_PRENORM_GEMM_TILELANG_KERNEL,
+                    _MHC_FUSED_TILELANG_KERNEL,
+                    _MHC_POST_TILELANG_KERNEL,
+                    _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +59/-0; `vllm/models/glm5next/nvidia/model.py` modified +44/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_mhc_jit_warmup.py`, `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #55897 - [LoRA] Add LoRA support for DeepSeek-V4 Flash Vision

- Link: https://github.com/vllm-project/vllm/pull/55897
- Status/date: merged / 2026-09-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/vl_model.py`; associated commits `dd4c8410707b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +42/-3, 97 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/common/vl_model.py` modified +40/-1 (41 lines); hunks: -25,16 +25,19; -81,17 +84,27 @@ def _make_deepseek_v4_vl_weights_mapper(; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV4ForConditionalGeneration, get_placeholder_str, process_weights_after_loading, touching `_make_deepseek_v4_vl_weights_mapper, DeepseekV4ForConditionalGeneration, get_placeholder_str`.
- Code diff details:
  - `vllm/models/deepseek_v4/common/vl_model.py` modified +40/-1 (41 lines); hunks: -25,16 +25,19; -81,17 +84,27 @@ def _make_deepseek_v4_vl_weights_mapper(; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV4ForConditionalGeneration, get_placeholder_str, process_weights_after_loading
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/common/vl_model.py
@@ -25,16 +25,19 @@
+    SupportsLoRA,
+from vllm.model_executor.models.module_mapping import MultiModelKeys
+from vllm.multimodal.inputs import MultiModalKwargsItem
@@ -81,17 +84,27 @@ def _make_deepseek_v4_vl_weights_mapper(
-    nn.Module, SupportsMultiModal, SupportsPP, SupportsEagle3
+    nn.Module, SupportsMultiModal, SupportsPP, SupportsEagle3, SupportsLoRA
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/common/vl_model.py` modified +40/-1
- Risk and verification: Runtime changes concentrate in `vllm/lora/ops/triton_ops/lora_shrink_op.py`, `vllm/models/deepseek_v4/common/vl_model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #56299 - [Bugfix][Frontend] Support Responses text types in DeepSeek V4.1

- Link: https://github.com/vllm-project/vllm/pull/56299
- Status/date: merged / 2026-09-14
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tokenizers_/test_deepseek_v41.py`, `vllm/tokenizers/deepseek_v41.py`; associated commits `eb42686a30cd`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +19/-1, 34 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/tokenizers_/test_deepseek_v41.py` modified +18/-0 (18 lines); hunks: -99,6 +99,24 @@ def test_raw_text_parts_preserve_reference_separator():; symbols: test_raw_text_parts_preserve_reference_separator, test_responses_text_parts_match_chat_text_parts, test_mid_system_gets_its_own_marker_and_generation_header, touching `test_raw_text_parts_preserve_reference_separator, test_responses_text_parts_match_chat_text_parts, test_mid_system_gets_its_own_marker_and_generation_header`; `vllm/tokenizers/deepseek_v41.py` modified +1/-1 (2 lines); hunks: -32,7 +32,7 @@ def _normalize_messages(; symbols: _normalize_messages, touching `_normalize_messages`.
- Code diff details:
  - `tests/tokenizers_/test_deepseek_v41.py` modified +18/-0 (18 lines); hunks: -99,6 +99,24 @@ def test_raw_text_parts_preserve_reference_separator():; symbols: test_raw_text_parts_preserve_reference_separator, test_responses_text_parts_match_chat_text_parts, test_mid_system_gets_its_own_marker_and_generation_header
  - `vllm/tokenizers/deepseek_v41.py` modified +1/-1 (2 lines); hunks: -32,7 +32,7 @@ def _normalize_messages(; symbols: _normalize_messages
- Key code excerpts:

```diff
diff -- tests/tokenizers_/test_deepseek_v41.py
@@ -99,6 +99,24 @@ def test_raw_text_parts_preserve_reference_separator():
+@pytest.mark.parametrize(
+    ("role", "responses_type"),
+    [("user", "input_text"), ("assistant", "output_text")],
+)
+def test_responses_text_parts_match_chat_text_parts(role, responses_type):
+    responses_message = {
diff -- vllm/tokenizers/deepseek_v41.py
@@ -32,7 +32,7 @@ def _normalize_messages(
-                if part_type == "text":
+                if part_type in ("text", "input_text", "output_text"):
```

- Extracted files (not manually reviewed):
  - tests: `tests/tokenizers_/test_deepseek_v41.py` modified +18/-0
  - runtime: `vllm/tokenizers/deepseek_v41.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/tokenizers_/test_deepseek_v41.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56741 - [Refactor] Normalize DeepSeek V4.1 model package naming

- Link: https://github.com/vllm-project/vllm/pull/56741
- Status/date: merged / 2026-09-14
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_mega_moe.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`, `vllm/models/deepseek_v41/__init__.py`, `vllm/models/deepseek_v41/amd/__init__.py`, `vllm/models/deepseek_v41/amd/dspark.py` and 29 files; associated commits `cf1584f373d0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 44 files, +71/-72, 476 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/attention.py` renamed +7/-7 (14 lines); hunks: -26,7 +26,7; -49,8 +49,8; symbols: DeepseekV4Attention, __init__, forward, _can_fuse_query_quant, touching `DeepseekV4Attention, __init__, forward`; `vllm/models/deepseek_v41/amd/rocm.py` renamed +3/-3 (6 lines); hunks: -13,9 +13,9; `vllm/models/deepseek_v41/nvidia/engram.py` renamed +3/-3 (6 lines); hunks: -23,16 +23,16; `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` renamed +3/-3 (6 lines); hunks: -13,12 +13,12.
- Code diff details:
  - `vllm/models/deepseek_v41/attention.py` renamed +7/-7 (14 lines); hunks: -26,7 +26,7; -49,8 +49,8; symbols: DeepseekV4Attention, __init__, forward, _can_fuse_query_quant
  - `vllm/models/deepseek_v41/amd/rocm.py` renamed +3/-3 (6 lines); hunks: -13,9 +13,9
  - `vllm/models/deepseek_v41/nvidia/engram.py` renamed +3/-3 (6 lines); hunks: -23,16 +23,16
  - `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` renamed +3/-3 (6 lines); hunks: -13,12 +13,12
  - `vllm/models/deepseek_v41/nvidia/flashmla.py` renamed +3/-3 (6 lines); hunks: -10,13 +10,13
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/attention.py
@@ -26,7 +26,7 @@
-from vllm.models.deepseek_v4_1.common.ops import (
+from vllm.models.deepseek_v41.common.ops import (
@@ -49,8 +49,8 @@
-from vllm.models.deepseek_v4_1.common.rope import build_deepseek_v4_rope
-from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
+from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -13,9 +13,9 @@
-from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
-from vllm.models.deepseek_v4_1.common.ops import dequantize_and_gather_k_cache
-from vllm.models.deepseek_v4_1.sparse_mla import (
+from vllm.models.deepseek_v41.attention import DeepseekV4Attention
+from vllm.models.deepseek_v41.common.ops import dequantize_and_gather_k_cache
+from vllm.models.deepseek_v41.sparse_mla import (
diff -- vllm/models/deepseek_v41/nvidia/engram.py
@@ -23,16 +23,16 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/attention.py` renamed +7/-7; `vllm/models/deepseek_v41/amd/rocm.py` renamed +3/-3; `vllm/models/deepseek_v41/nvidia/engram.py` renamed +3/-3; `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` renamed +3/-3; `vllm/models/deepseek_v41/nvidia/flashmla.py` renamed +3/-3; `vllm/models/deepseek_v41/nvidia/model.py` renamed +3/-3
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/distributed/test_engram_dp_shard.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_engram.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51794 - [ROCm][Perf] Enable CSA multi-stream overlap for DeepSeek-V4

- Link: https://github.com/vllm-project/vllm/pull/51794
- Status/date: merged / 2026-09-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/cpu/cpu_sparse.py`; associated commits `a6c5d6d0fcd7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +214/-25, 317 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/amd/rocm.py` modified +194/-0 (194 lines); hunks: -23,6 +23,7; -571,6 +572,199 @@ def __init__(self, *args, **kwargs):; symbols: __init__, _enable_csa_multi_stream, _run_sequential_pipeline, forward, touching `__init__, _enable_csa_multi_stream, _run_sequential_pipeline`; `vllm/models/deepseek_v4/attention.py` modified +17/-14 (31 lines); hunks: -318,7 +318,6 @@ def __init__(; -584,7 +583,7 @@ def project_query_and_cache_kv() -> torch.Tensor:; symbols: __init__, project_query_and_cache_kv, _run_parallel_input_projections, forward, touching `__init__, project_query_and_cache_kv, _run_parallel_input_projections`; `vllm/models/deepseek_v4/amd/model.py` modified +1/-10 (11 lines); hunks: -957,16 +957,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`; `vllm/models/deepseek_v4/cpu/cpu_sparse.py` modified +2/-1 (3 lines); hunks: -198,11 +198,12 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +194/-0 (194 lines); hunks: -23,6 +23,7; -571,6 +572,199 @@ def __init__(self, *args, **kwargs):; symbols: __init__, _enable_csa_multi_stream, _run_sequential_pipeline, forward
  - `vllm/models/deepseek_v4/attention.py` modified +17/-14 (31 lines); hunks: -318,7 +318,6 @@ def __init__(; -584,7 +583,7 @@ def project_query_and_cache_kv() -> torch.Tensor:; symbols: __init__, project_query_and_cache_kv, _run_parallel_input_projections, forward
  - `vllm/models/deepseek_v4/amd/model.py` modified +1/-10 (11 lines); hunks: -957,16 +957,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/models/deepseek_v4/cpu/cpu_sparse.py` modified +2/-1 (3 lines); hunks: -198,11 +198,12 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -23,6 +23,7 @@
+from vllm.utils.multi_stream_utils import execute_in_parallel
@@ -571,6 +572,199 @@ def __init__(self, *args, **kwargs):
+        if self.indexer is None:
+            # Only enable multi-stream overlap for CSA layer now.
+            self.aux_stream_list = None
+        else:
diff -- vllm/models/deepseek_v4/attention.py
@@ -318,7 +318,6 @@ def __init__(
-        # Will be None on ROCm for now.
@@ -584,7 +583,7 @@ def project_query_and_cache_kv() -> torch.Tensor:
-        # indexer. ROCm runs the same work sequentially without aux streams.
+        # indexer.
@@ -671,7 +670,6 @@ def _run_parallel_input_projections(
-        # On ROCm, aux_streams is None and execute_in_parallel runs serially.
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -957,16 +957,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/amd/rocm.py` modified +194/-0; `vllm/models/deepseek_v4/attention.py` modified +17/-14; `vllm/models/deepseek_v4/amd/model.py` modified +1/-10; `vllm/models/deepseek_v4/cpu/cpu_sparse.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #56633 - [Perf][DSv4.1] Fold the mHC post block into the delayed pre projection

- Link: https://github.com/vllm-project/vllm/pull/56633
- Status/date: merged / 2026-09-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/nvidia/dspark.py`, `vllm/models/deepseek_v41/nvidia/model.py`; associated commits `5372e72a9884`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +837/-92, 1316 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/nvidia/model.py` modified +126/-52 (178 lines); hunks: -20,6 +20,7; -214,17 +215,45 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/models/deepseek_v41/nvidia/dspark.py` modified +1/-1 (2 lines); hunks: -207,7 +207,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +126/-52 (178 lines); hunks: -20,6 +20,7; -214,17 +215,45 @@ def __init__(; symbols: __init__, forward
  - `vllm/models/deepseek_v41/nvidia/dspark.py` modified +1/-1 (2 lines); hunks: -207,7 +207,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -20,6 +20,7 @@
+    mhc_fused_post_pre_delayed_tilelang,
@@ -214,17 +215,45 @@ def __init__(
+            from vllm.model_executor.kernels.mhc.tilelang_kernels import (
+                mhc_fused_post_pre_splits,
+            )
-            for input_size, use_pre_mix in (
diff -- vllm/models/deepseek_v41/nvidia/dspark.py
@@ -207,7 +207,7 @@ def forward(
-            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
+            hidden_states, residual, post_mix, res_mix, pre_mix, _ = layer(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/nvidia/model.py` modified +126/-52; `vllm/models/deepseek_v41/nvidia/dspark.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/distributed/test_engram_dp_shard.py`, `tests/kernels/test_mhc_jit_warmup.py`, `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56513 - [ROCm][DSV4.1][Perf] Fold the mHC post step into the delayed pre projection

- Link: https://github.com/vllm-project/vllm/pull/56513
- Status/date: merged / 2026-09-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/amd/model.py`; associated commits `00972dfd7298`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +346/-64, 633 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/amd/model.py` modified +37/-17 (54 lines); hunks: -274,7 +274,7 @@ def forward(; -288,7 +288,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/models/deepseek_v41/amd/model.py` modified +37/-17 (54 lines); hunks: -274,7 +274,7 @@ def forward(; -288,7 +288,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/amd/model.py
@@ -274,7 +274,7 @@ def forward(
-                post_mix, res_mix, x, attn_pre = self.mhc_pre_delayed(
+                residual, post_mix, res_mix, x, attn_pre = self.mhc_pre_delayed(
@@ -288,7 +288,7 @@ def forward(
-                post_mix, res_mix, x, attn_pre = self.mhc_pre_delayed(
+                residual, post_mix, res_mix, x, attn_pre = self.mhc_pre_delayed(
@@ -301,18 +301,7 @@ def forward(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/amd/model.py` modified +37/-17
- Risk and verification: The diff ships test coverage in `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56255 - [DSv4.1] Integrate Mega-mHC from DeepGEMM

- Link: https://github.com/vllm-project/vllm/pull/56255
- Status/date: closed / 2026-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/__init__.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`; associated commits `bb5507741155`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +315/-43, 566 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4_1/nvidia/ops/mega_mhc.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre, touching `is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre`; `vllm/models/deepseek_v4_1/nvidia/model.py` modified +41/-21 (62 lines); hunks: -76,6 +76,7; -334,30 +335,48 @@ def forward(; symbols: forward, touching `forward`; `vllm/models/deepseek_v4_1/nvidia/ops/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2.
- Code diff details:
  - `vllm/models/deepseek_v4_1/nvidia/ops/mega_mhc.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre
  - `vllm/models/deepseek_v4_1/nvidia/model.py` modified +41/-21 (62 lines); hunks: -76,6 +76,7; -334,30 +335,48 @@ def forward(; symbols: forward
  - `vllm/models/deepseek_v4_1/nvidia/ops/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4_1/nvidia/ops/mega_mhc.py
@@ -0,0 +1,144 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import functools
+import torch
+from vllm.model_executor.kernels.mhc.tilelang import (
+    mhc_post_tilelang,
diff -- vllm/models/deepseek_v4_1/nvidia/model.py
@@ -76,6 +76,7 @@
+from .ops.mega_mhc import mhc_shifted_post_pre
@@ -334,30 +335,48 @@ def forward(
-            residual = mhc_post_tilelang(x, residual, post_mix, res_mix)
+                residual = mhc_post_tilelang(x, residual, post_mix, res_mix)
-            post_mix, res_mix, x, attn_pre = mhc_pre_delayed_tilelang(
-                residual,
diff -- vllm/models/deepseek_v4_1/nvidia/ops/__init__.py
@@ -0,0 +1,2 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4_1/nvidia/ops/mega_mhc.py` added +144/-0; `vllm/models/deepseek_v4_1/nvidia/model.py` modified +41/-21; `vllm/models/deepseek_v4_1/nvidia/ops/__init__.py` added +2/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/quantization/test_block_fp8.py`, `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56893 - [Model][DSv4.1] Store the whole KV in MXFP8 (FlashMLA V4.1 record)

- Link: https://github.com/vllm-project/vllm/pull/56893
- Status/date: merged / 2026-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py`, `vllm/models/deepseek_v41/amd/dspark.py`, `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/common/ops/cache_utils.py` and 7 files; associated commits `e6eb0d120c48`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 14 files, +709/-110, 1334 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +186/-8 (194 lines); hunks: -36,6 +36,18; -163,23 +175,81 @@ def quantize_and_insert_k_kernel(; symbols: quantize_and_insert_k_kernel, _quantize_and_insert_k_mxfp8_kernel, quantize_and_insert_k_cache, touching `quantize_and_insert_k_kernel, _quantize_and_insert_k_mxfp8_kernel, quantize_and_insert_k_cache`; `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` modified +70/-6 (76 lines); hunks: -232,10 +232,13 @@ def rope_quant_insert(; -246,8 +249,15 @@ def rope_quant_insert(; symbols: rope_quant_insert, _rope_quant_insert_kernel, _rope_quant_insert_mxfp8_kernel, _rope_plain_insert_kernel, touching `rope_quant_insert, _rope_quant_insert_kernel, _rope_quant_insert_mxfp8_kernel`; `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` modified +39/-23 (62 lines); hunks: -23,10 +23,11; -35,17 +36,20 @@ class DequantGatherKCacheKernel(; symbols: DequantGatherKCacheKernel, CompileKey, kernel, load_g2s, touching `DequantGatherKCacheKernel, CompileKey, kernel`; `vllm/models/deepseek_v41/attention.py` modified +34/-9 (43 lines); hunks: -51,6 +51,7; -109,6 +110,16 @@ def _fill_short_context_topk_indices(; symbols: _fill_short_context_topk_indices, _use_v41_mxfp8_kv_record, _resolve_dsv4_kv_cache_dtype, __init__, touching `_fill_short_context_topk_indices, _use_v41_mxfp8_kv_record, _resolve_dsv4_kv_cache_dtype`.
- Code diff details:
  - `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +186/-8 (194 lines); hunks: -36,6 +36,18; -163,23 +175,81 @@ def quantize_and_insert_k_kernel(; symbols: quantize_and_insert_k_kernel, _quantize_and_insert_k_mxfp8_kernel, quantize_and_insert_k_cache
  - `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` modified +70/-6 (76 lines); hunks: -232,10 +232,13 @@ def rope_quant_insert(; -246,8 +249,15 @@ def rope_quant_insert(; symbols: rope_quant_insert, _rope_quant_insert_kernel, _rope_quant_insert_mxfp8_kernel, _rope_plain_insert_kernel
  - `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` modified +39/-23 (62 lines); hunks: -23,10 +23,11; -35,17 +36,20 @@ class DequantGatherKCacheKernel(; symbols: DequantGatherKCacheKernel, CompileKey, kernel, load_g2s
  - `vllm/models/deepseek_v41/attention.py` modified +34/-9 (43 lines); hunks: -51,6 +51,7; -109,6 +110,16 @@ def _fill_short_context_topk_indices(; symbols: _fill_short_context_topk_indices, _use_v41_mxfp8_kv_record, _resolve_dsv4_kv_cache_dtype, __init__
  - `vllm/models/deepseek_v41/amd/dspark.py` modified +2/-0 (2 lines); hunks: -259,6 +259,8 @@ def _insert_context_kv(; symbols: _insert_context_kv
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/common/ops/cache_utils.py
@@ -36,6 +36,18 @@
+# Per-token byte width of the two paged fp8 records. Both put a page's whole
+# data region ahead of its whole scale region, so ``k_cache.shape[-1]`` -- the
+# width the KV-cache spec publishes -- identifies the record:
+#   V4   (584 B): 448 fp8 e4m3 NoPE + 64 bf16 RoPE, then 7 UE8M0 scales of 64
+#                 dims each plus a pad byte.
+#   V4.1 (528 B): all 512 dims as fp8 e4m3 (RoPE included), then 16 UE8M0
diff -- vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py
@@ -232,10 +232,13 @@ def rope_quant_insert(
-    layout: ``uint8`` is the fp8_ds_mla paged layout (576 value bytes and eight
-    segregated UE8M0 scale bytes per token, including one zero padding scale);
-    ``bfloat16`` and ``float8_e4m3fn`` are the plain [448 NoPE | 64 RoPE] rows
-    read by FlashInfer, the latter scaled by the per-tensor ``fp8_scale``.
+    layout: ``uint8`` is the fp8_ds_mla paged layout, whose record the
+    per-token byte width names -- 584 B for V4 (576 value bytes and eight
diff -- vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py
@@ -23,10 +23,11 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +186/-8; `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` modified +70/-6; `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` modified +39/-23; `vllm/models/deepseek_v41/attention.py` modified +34/-9; `vllm/models/deepseek_v41/amd/dspark.py` modified +2/-0; `vllm/models/deepseek_v41/nvidia/dspark.py` modified +2/-0
  - tests: `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +98/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56903 - [Perf][DSpark] Collapse DeepSeek-V4.1 draft states before SP all-gather

- Link: https://github.com/vllm-project/vllm/pull/56903
- Status/date: merged / 2026-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/nvidia/dspark.py`; associated commits `073f883b56a5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +23/-3, 46 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/nvidia/dspark.py` modified +2/-3 (5 lines); hunks: -217,15 +217,14 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/models/deepseek_v41/nvidia/dspark.py` modified +2/-3 (5 lines); hunks: -217,15 +217,14 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/nvidia/dspark.py
@@ -217,15 +217,14 @@ def forward(
-        if self.use_sequence_parallel:
-            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
-            pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]
+        if self.use_sequence_parallel:
+            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/nvidia/dspark.py` modified +2/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56962 - [Perf][Kernel] Integrate Mega-mHC from DeepGEMM for DeepSeek V4.1 (reopen of #56255)

- Link: https://github.com/vllm-project/vllm/pull/56962
- Status/date: merged / 2026-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/__init__.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`; associated commits `bb5507741155`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +316/-41, 471 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre, touching `is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre`; `vllm/models/deepseek_v41/nvidia/model.py` modified +34/-38 (72 lines); hunks: -20,7 +20,6; -79,6 +78,7; symbols: forward, touching `forward`; `vllm/models/deepseek_v41/nvidia/ops/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2.
- Code diff details:
  - `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +34/-38 (72 lines); hunks: -20,7 +20,6; -79,6 +78,7; symbols: forward
  - `vllm/models/deepseek_v41/nvidia/ops/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py
@@ -0,0 +1,156 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import functools
+import torch
+from vllm.model_executor.kernels.mhc.tilelang import (
+    mhc_fused_post_pre_delayed_tilelang,
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -20,7 +20,6 @@
-    mhc_fused_post_pre_delayed_tilelang,
@@ -79,6 +78,7 @@
+from .ops.mega_mhc import mhc_shifted_post_pre
@@ -403,25 +403,23 @@ def forward(
-            residual, post_mix, res_mix, x, attn_pre, aux = (
-                mhc_fused_post_pre_delayed_tilelang(
diff -- vllm/models/deepseek_v41/nvidia/ops/__init__.py
@@ -0,0 +1,2 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` added +156/-0; `vllm/models/deepseek_v41/nvidia/model.py` modified +34/-38; `vllm/models/deepseek_v41/nvidia/ops/__init__.py` added +2/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/quantization/test_block_fp8.py`, `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56254 - [DSA] Wire DeepGEMM sparse MQA logits into the DeepSeek V4.1 indexer

- Link: https://github.com/vllm-project/vllm/pull/56254
- Status/date: merged / 2026-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py`, `vllm/models/deepseek_v41/attention.py`; associated commits `12575123059c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 13 files, +1830/-26, 2149 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/attention.py` modified +67/-20 (87 lines); hunks: -4,6 +4,7; -25,6 +26,7; symbols: __init__, get_kv_cache_spec, forward, get_attn_backend, touching `__init__, get_kv_cache_spec, forward`; `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +22/-2 (24 lines); hunks: -309,12 +309,22 @@ def __call__(; -434,12 +444,14 @@ def dispatch( # type: ignore[override]; symbols: __call__, _indexer_weights_out_dtypes, FusedIndexerQRopeMxFp4TritonKernel, CompileKey, touching `__call__, _indexer_weights_out_dtypes, FusedIndexerQRopeMxFp4TritonKernel`; `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` modified +15/-2 (17 lines); hunks: -165,6 +165,7 @@ class CompileKey:; -177,6 +178,7 @@ def kernel(compile_key: CompileKey) -> Any:; symbols: CompileKey, kernel, device_kernel, host_entrypoint, touching `CompileKey, kernel, device_kernel`; `vllm/config/attention.py` modified +9/-0 (9 lines); hunks: -86,6 +86,15 @@ class AttentionConfig:; symbols: AttentionConfig, GPU, touching `AttentionConfig, GPU`.
- Code diff details:
  - `vllm/models/deepseek_v41/attention.py` modified +67/-20 (87 lines); hunks: -4,6 +4,7; -25,6 +26,7; symbols: __init__, get_kv_cache_spec, forward, get_attn_backend
  - `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +22/-2 (24 lines); hunks: -309,12 +309,22 @@ def __call__(; -434,12 +444,14 @@ def dispatch( # type: ignore[override]; symbols: __call__, _indexer_weights_out_dtypes, FusedIndexerQRopeMxFp4TritonKernel, CompileKey
  - `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` modified +15/-2 (17 lines); hunks: -165,6 +165,7 @@ class CompileKey:; -177,6 +178,7 @@ def kernel(compile_key: CompileKey) -> Any:; symbols: CompileKey, kernel, device_kernel, host_entrypoint
  - `vllm/config/attention.py` modified +9/-0 (9 lines); hunks: -86,6 +86,15 @@ class AttentionConfig:; symbols: AttentionConfig, GPU
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/attention.py
@@ -4,6 +4,7 @@
+import math
@@ -25,6 +26,7 @@
+from vllm.model_executor.layers.sparse_mqa_indexer import SparseMQAIndexer
@@ -63,6 +65,9 @@
+from vllm.v1.attention.backends.mla.sparse_indexer import (
+    DeepseekV41SparseIndexerBackend,
diff -- vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
@@ -309,12 +309,22 @@ def __call__(
+def _indexer_weights_out_dtypes(vllm_config: Any) -> tuple[torch.dtype, ...]:
+    """Weights dtypes the model's indexer layers ask for: fp32 for the dense
+    scoring kernels, plus bf16 when the DeepSeek V4.1 sparse-logits indexer
+    (`SparseMQAIndexer`) is enabled."""
+    if vllm_config.attention_config.indexer_sparse_logits:
+        return (torch.float32, torch.bfloat16)
diff -- vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py
@@ -165,6 +165,7 @@ class CompileKey:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/attention.py` modified +67/-20; `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +22/-2; `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` modified +15/-2; `vllm/config/attention.py` modified +9/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_indexer_q_rope_quant.py`, `tests/model_executor/layers/test_mla_short_prefill_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56441 - [Perf][DSpark] Add KV-only context insertion across V4.1 cache formats

- Link: https://github.com/vllm-project/vllm/pull/56441
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/nvidia/dspark.py`; associated commits `6ecd97f1b7fb`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +234/-61, 345 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/nvidia/dspark.py` modified +17/-60 (77 lines); hunks: -234,67 +234,24 @@ def _insert_context_kv(; symbols: _insert_context_kv, DSparkDeepseekV4ForCausalLM, touching `_insert_context_kv, DSparkDeepseekV4ForCausalLM`.
- Code diff details:
  - `vllm/models/deepseek_v41/nvidia/dspark.py` modified +17/-60 (77 lines); hunks: -234,67 +234,24 @@ def _insert_context_kv(; symbols: _insert_context_kv, DSparkDeepseekV4ForCausalLM
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/nvidia/dspark.py
@@ -234,67 +234,24 @@ def _insert_context_kv(
-    """RoPE + quant + paged-cache insert of (already kv_norm'd) context KV.
-    Reuses the DSV4 fused insert ops (which also process a query; we pass a dummy
-    query and discard it, since context tokens have no query). Mirrors
-    ``DeepseekV4Attention._fused_qnorm_rope_kv_insert``.
-    """
-    swa_cache = attn.swa_cache_layer.kv_cache
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/nvidia/dspark.py` modified +17/-60
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56568 - [Perf][DSV4.1] Pad shared experts for native MegaMoE fusion

- Link: https://github.com/vllm-project/vllm/pull/56568
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `03f67b3ad1e6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +128/-23, 259 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/test_deepseek_v4_mega_moe.py` modified +80/-18 (98 lines); hunks: -367,17 +367,21 @@ def test_deepseek_v4_mega_moe_padding_preserves_weights(mo...; -402,6 +406,7 @@ def fp8_fp4_mega_moe(; symbols: test_deepseek_v4_mega_moe_padding_preserves_weights, test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights, FakeDeepGemm, get_symm_buffer_for_mega_moe, touching `test_deepseek_v4_mega_moe_padding_preserves_weights, test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights, FakeDeepGemm`; `vllm/models/deepseek_v4/nvidia/model.py` modified +48/-5 (53 lines); hunks: -204,6 +204,7 @@ def __init__(; -386,6 +387,17 @@ def _finalize_shared_expert_weights(; symbols: __init__, _finalize_shared_expert_weights, _prepare_shared_expert_scale, touching `__init__, _finalize_shared_expert_weights, _prepare_shared_expert_scale`.
- Code diff details:
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +80/-18 (98 lines); hunks: -367,17 +367,21 @@ def test_deepseek_v4_mega_moe_padding_preserves_weights(mo...; -402,6 +406,7 @@ def fp8_fp4_mega_moe(; symbols: test_deepseek_v4_mega_moe_padding_preserves_weights, test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights, FakeDeepGemm, get_symm_buffer_for_mega_moe
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +48/-5 (53 lines); hunks: -204,6 +204,7 @@ def __init__(; -386,6 +387,17 @@ def _finalize_shared_expert_weights(; symbols: __init__, _finalize_shared_expert_weights, _prepare_shared_expert_scale
- Key code excerpts:

```diff
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -367,17 +367,21 @@ def test_deepseek_v4_mega_moe_padding_preserves_weights(monkeypatch, intermediat
-        pytest.param(5120, 2304, 32, False, False, id="deepseek-v41-flash"),
+        pytest.param(5120, 2304, 32, True, False, id="deepseek-v41-flash"),
+        pytest.param(128, 2304, 128, True, False, id="padded-block128"),
+        pytest.param(5120, 2304, 32, True, True, id="deepseek-v41-flash-mxfp8"),
+        pytest.param(128, 2304, 128, False, False, id="padded-packed-scale-fallback"),
-    """V4.1-Flash's routed padding must preserve the serial shared-expert fallback."""
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -204,6 +204,7 @@ def __init__(
+        self.unpadded_intermediate_size = intermediate_size
@@ -386,6 +387,17 @@ def _finalize_shared_expert_weights(
+        unpadded_size = self.unpadded_intermediate_size * self.num_shared_experts
+        padding = self.intermediate_size * self.num_shared_experts - unpadded_size
+        pad_weights = (
+            padding > 0
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +80/-18
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +48/-5
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_mega_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56935 - [Model][DSv4.1] FlashMLA mega attention and the NVFP4 compressed KV cache

- Link: https://github.com/vllm-project/vllm/pull/56935
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/common/ops/cache_utils.py`, `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` and 12 files; associated commits `d6a1677d5504`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 20 files, +1811/-156, 2550 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` added +497/-0 (497 lines); hunks: -0,0 +1,497; symbols: is_flashmla_mega_attn_supported, alloc_mega_attn_output, _token_slice, DeepseekV4MegaAttnAttention, touching `is_flashmla_mega_attn_supported, alloc_mega_attn_output, _token_slice`; `vllm/models/deepseek_v41/attention.py` modified +150/-54 (204 lines); hunks: -44,6 +44,7; -127,34 +128,40 @@ def _use_v41_mxfp8_kv_record() -> bool:; symbols: _use_v41_mxfp8_kv_record, _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention, accepts_unnormed_unroped_query, touching `_use_v41_mxfp8_kv_record, _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention`; `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +96/-2 (98 lines); hunks: -43,10 +43,16; -496,6 +502,73 @@ def _dequantize_and_gather_k_mxfp8_kernel(; symbols: _dequantize_and_gather_k_mxfp8_kernel, _dequantize_and_gather_k_nvfp4_kernel, dequantize_and_gather_k_cache_triton, dequantize_and_gather_k_cache, touching `_dequantize_and_gather_k_mxfp8_kernel, _dequantize_and_gather_k_nvfp4_kernel, dequantize_and_gather_k_cache_triton`; `vllm/models/deepseek_v41/common/ops/fused_layout.py` added +86/-0 (86 lines); hunks: -0,0 +1,86; symbols: q_fused_permutation, o_fused_permutation, o_fused_chunk_permutation, permute_q_to_fused, touching `q_fused_permutation, o_fused_permutation, o_fused_chunk_permutation`.
- Code diff details:
  - `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` added +497/-0 (497 lines); hunks: -0,0 +1,497; symbols: is_flashmla_mega_attn_supported, alloc_mega_attn_output, _token_slice, DeepseekV4MegaAttnAttention
  - `vllm/models/deepseek_v41/attention.py` modified +150/-54 (204 lines); hunks: -44,6 +44,7; -127,34 +128,40 @@ def _use_v41_mxfp8_kv_record() -> bool:; symbols: _use_v41_mxfp8_kv_record, _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention, accepts_unnormed_unroped_query
  - `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +96/-2 (98 lines); hunks: -43,10 +43,16; -496,6 +502,73 @@ def _dequantize_and_gather_k_mxfp8_kernel(; symbols: _dequantize_and_gather_k_mxfp8_kernel, _dequantize_and_gather_k_nvfp4_kernel, dequantize_and_gather_k_cache_triton, dequantize_and_gather_k_cache
  - `vllm/models/deepseek_v41/common/ops/fused_layout.py` added +86/-0 (86 lines); hunks: -0,0 +1,86; symbols: q_fused_permutation, o_fused_permutation, o_fused_chunk_permutation, permute_q_to_fused
  - `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` modified +71/-13 (84 lines); hunks: -4,6 +4,7; -232,13 +233,14 @@ def rope_quant_insert(; symbols: rope_quant_insert, _rope_quant_insert_mxfp8_kernel, _rope_quant_insert_nvfp4_kernel, _rope_plain_insert_kernel
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py
@@ -0,0 +1,497 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DeepSeek V4.1 attention on FlashMLA's mega-attention kernel (SM100).
+One kernel does Q RoPE, sparse attention, the inverse RoPE of the output and
+its FP8 cast, and writes straight into the buffer ``wo_a`` consumes. So the
+layer declares ``accepts_unnormed_unroped_query`` -- the kernel RoPEs Q itself,
diff -- vllm/models/deepseek_v41/attention.py
@@ -44,6 +44,7 @@
+from vllm.config.cache import CacheDType
@@ -127,34 +128,40 @@ def _use_v41_mxfp8_kv_record() -> bool:
-    kv_cache_dtype: str,
+    kv_cache_dtype: CacheDType,
-) -> tuple[str, torch.dtype]:
+    packed_kv_cache_dtype: CacheDType = "fp8_ds_mla",
diff -- vllm/models/deepseek_v41/common/ops/cache_utils.py
@@ -43,10 +43,16 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` added +497/-0; `vllm/models/deepseek_v41/attention.py` modified +150/-54; `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +96/-2; `vllm/models/deepseek_v41/common/ops/fused_layout.py` added +86/-0; `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` modified +71/-13; `vllm/models/deepseek_v41/sparse_mla.py` modified +51/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_dsv41_mega_attn_layouts.py`, `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57152 - [Bugfix][Model] Restore causal image SWA for DeepSeek V4.1

- Link: https://github.com/vllm-project/vllm/pull/57152
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/v1/attention/test_deepseek_v4_swa_visible.py`, `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/common/ops/cache_utils.py`, `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py`, `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` and 7 files; associated commits `9f9e1dac26ff`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +122/-174, 606 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +31/-108 (139 lines); hunks: -818,16 +818,12 @@ def combine_topk_swa_indices(; -856,9 +852,6 @@ def combine_topk_swa_indices(; symbols: combine_topk_swa_indices, CompileKey, kernel, touching `combine_topk_swa_indices, CompileKey, kernel`; `vllm/models/deepseek_v41/nvidia/flashmla.py` modified +2/-19 (21 lines); hunks: -120,9 +120,7 @@ def forward_mqa(; -303,7 +301,7 @@ def _forward_prefill(; symbols: forward_mqa, _forward_prefill, touching `forward_mqa, _forward_prefill`; `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` modified +2/-17 (19 lines); hunks: -311,7 +311,7 @@ def _reserve_prefill_workspace(self, q: torch.Tensor) -> None:; -411,7 +411,7 @@ def _forward_prefill_mega(; symbols: _reserve_prefill_workspace, _forward_prefill_mega, touching `_reserve_prefill_workspace, _forward_prefill_mega`; `vllm/models/deepseek_v41/attention.py` modified +0/-8 (8 lines); hunks: -290,13 +290,6 @@ def __init__(; -578,7 +571,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +31/-108 (139 lines); hunks: -818,16 +818,12 @@ def combine_topk_swa_indices(; -856,9 +852,6 @@ def combine_topk_swa_indices(; symbols: combine_topk_swa_indices, CompileKey, kernel
  - `vllm/models/deepseek_v41/nvidia/flashmla.py` modified +2/-19 (21 lines); hunks: -120,9 +120,7 @@ def forward_mqa(; -303,7 +301,7 @@ def _forward_prefill(; symbols: forward_mqa, _forward_prefill
  - `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` modified +2/-17 (19 lines); hunks: -311,7 +311,7 @@ def _reserve_prefill_workspace(self, q: torch.Tensor) -> None:; -411,7 +411,7 @@ def _forward_prefill_mega(; symbols: _reserve_prefill_workspace, _forward_prefill_mega
  - `vllm/models/deepseek_v41/attention.py` modified +0/-8 (8 lines); hunks: -290,13 +290,6 @@ def __init__(; -578,7 +571,6 @@ def __init__(; symbols: __init__
  - `vllm/transformers_utils/configs/deepseek_v41.py` modified +0/-8 (8 lines); hunks: -60,11 +60,3 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/common/ops/cache_utils.py
@@ -818,16 +818,12 @@ def combine_topk_swa_indices(
-    left_visible: torch.Tensor | None = None,
-    right_visible: torch.Tensor | None = None,
-    max_image_tokens: int = 0,
-    # max_image_tokens widens the SWA column region for in-image
-    # bidirectional visibility; the width is fixed per model so the
-    # caller-provided workspace stays valid on image-free batches.
diff -- vllm/models/deepseek_v41/nvidia/flashmla.py
@@ -120,9 +120,7 @@ def forward_mqa(
-            combined_topk = round_up(
-                top_k + self.window_size + self.max_image_tokens, 128
-            )
+            combined_topk = round_up(top_k + self.window_size, 128)
@@ -303,7 +301,7 @@ def _forward_prefill(
-        combined_topk = round_up(top_k + self.window_size + self.max_image_tokens, 128)
diff -- vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py
@@ -311,7 +311,7 @@ def _reserve_prefill_workspace(self, q: torch.Tensor) -> None:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +31/-108; `vllm/models/deepseek_v41/nvidia/flashmla.py` modified +2/-19; `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` modified +2/-17; `vllm/models/deepseek_v41/attention.py` modified +0/-8; `vllm/transformers_utils/configs/deepseek_v41.py` modified +0/-8; `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +0/-4
  - tests: `tests/v1/attention/test_deepseek_v4_swa_visible.py` modified +81/-6
- Risk and verification: The diff ships test coverage in `tests/v1/attention/test_deepseek_v4_swa_visible.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57132 - [ROCm][Bugfix] Revert #56433 + #51692 to fix accuracy breakdown for DeepSeek-V4

- Link: https://github.com/vllm-project/vllm/pull/57132
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_vl_rocm.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`; associated commits `bc0f47cd03d6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +34/-402, 589 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/test_deepseek_v4_vl_rocm.py` modified +0/-125 (125 lines); hunks: -15,131 +15,6; symbols: _preshuffled_fp8_linear, test_rocm_wo_a_keeps_row_major_weights, test_rocm_gateup_shuffles_once_and_down_proj_keeps_preshuffled_backend, test_rocm_fused_qkv_quant_matches_preshuffled_gemm_scale_layout, touching `_preshuffled_fp8_linear, test_rocm_wo_a_keeps_row_major_weights, test_rocm_gateup_shuffles_once_and_down_proj_keeps_preshuffled_backend`; `vllm/models/deepseek_v4/amd/rocm.py` modified +16/-59 (75 lines); hunks: -62,21 +62,6 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> torc...; -792,13 +777,12 @@ def _prep(linear) -> torch.Tensor | None:; symbols: _build_indptr_from_lengths, weight_already_preshuffled, apply_pre_quantized_block_scaled_mm, _prep, touching `_build_indptr_from_lengths, weight_already_preshuffled, apply_pre_quantized_block_scaled_mm`; `vllm/models/deepseek_v4/amd/model.py` modified +6/-11 (17 lines); hunks: -71,10 +71,7; -152,13 +149,11 @@ def prepare_gateup_preshuffle(self) -> None:; symbols: prepare_gateup_preshuffle, forward, touching `prepare_gateup_preshuffle, forward`.
- Code diff details:
  - `tests/models/test_deepseek_v4_vl_rocm.py` modified +0/-125 (125 lines); hunks: -15,131 +15,6; symbols: _preshuffled_fp8_linear, test_rocm_wo_a_keeps_row_major_weights, test_rocm_gateup_shuffles_once_and_down_proj_keeps_preshuffled_backend, test_rocm_fused_qkv_quant_matches_preshuffled_gemm_scale_layout
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +16/-59 (75 lines); hunks: -62,21 +62,6 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> torc...; -792,13 +777,12 @@ def _prep(linear) -> torch.Tensor | None:; symbols: _build_indptr_from_lengths, weight_already_preshuffled, apply_pre_quantized_block_scaled_mm, _prep
  - `vllm/models/deepseek_v4/amd/model.py` modified +6/-11 (17 lines); hunks: -71,10 +71,7; -152,13 +149,11 @@ def prepare_gateup_preshuffle(self) -> None:; symbols: prepare_gateup_preshuffle, forward
- Key code excerpts:

```diff
diff -- tests/models/test_deepseek_v4_vl_rocm.py
@@ -15,131 +15,6 @@
-def _preshuffled_fp8_linear(
-    holder: str = "quant_method", weight_shape: tuple[int, int] = (256, 128)
-) -> nn.Module:
-    pytest.importorskip("aiter")
-    from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
-        AiterPreshuffledFp8BlockScaledMMKernel,
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -62,21 +62,6 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
-def weight_already_preshuffled(linear: torch.nn.Module) -> bool:
-    """True when the linear's kernel already B-preshuffled ``weight``.
-    The hand-shuffles below (fused_wqa_wkv, wo_b, gate_up_proj) must be skipped
-    for those, since shuffle_weight is a permutation rather than an involution.
-    """
-    return any(
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -71,10 +71,7 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/test_deepseek_v4_vl_rocm.py` modified +0/-125
  - runtime: `vllm/models/deepseek_v4/amd/rocm.py` modified +16/-59; `vllm/models/deepseek_v4/amd/model.py` modified +6/-11
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_vl_rocm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57204 - [Perf][DSV4.1] Remove MegaMoE padding and shared padding workaround

- Link: https://github.com/vllm-project/vllm/pull/57204
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `30b847c1fa54`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +42/-157, 313 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/test_deepseek_v4_mega_moe.py` modified +37/-92 (129 lines); hunks: -300,8 +300,10 @@ def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_...; -333,50 +335,35 @@ def test_deepseek_v4_mega_moe_padding_preserves_weights(mo...; symbols: test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_padding_preserves_weights, test_deepseek_v4_mega_moe_preserves_checkpoint_dimensions, test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights, touching `test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_padding_preserves_weights, test_deepseek_v4_mega_moe_preserves_checkpoint_dimensions`; `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-65 (70 lines); hunks: -204,7 +204,6 @@ def __init__(; -387,17 +386,6 @@ def _finalize_shared_expert_weights(; symbols: __init__, _finalize_shared_expert_weights, _prepare_shared_expert_scale, touching `__init__, _finalize_shared_expert_weights, _prepare_shared_expert_scale`.
- Code diff details:
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +37/-92 (129 lines); hunks: -300,8 +300,10 @@ def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_...; -333,50 +335,35 @@ def test_deepseek_v4_mega_moe_padding_preserves_weights(mo...; symbols: test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_padding_preserves_weights, test_deepseek_v4_mega_moe_preserves_checkpoint_dimensions, test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-65 (70 lines); hunks: -204,7 +204,6 @@ def __init__(; -387,17 +386,6 @@ def _finalize_shared_expert_weights(; symbols: __init__, _finalize_shared_expert_weights, _prepare_shared_expert_scale
- Key code excerpts:

```diff
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -300,8 +300,10 @@ def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership():
-def test_deepseek_v4_mega_moe_padding_preserves_weights(monkeypatch, intermediate_size):
-    """Pad gate/up separately and zero every added down-projection column."""
+def test_deepseek_v4_mega_moe_preserves_checkpoint_dimensions(
+    monkeypatch, intermediate_size
+):
+    """Keep native widths so V4.1 routed and shared experts can fuse."""
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -204,7 +204,6 @@ def __init__(
-        self.unpadded_intermediate_size = intermediate_size
@@ -387,17 +386,6 @@ def _finalize_shared_expert_weights(
-        unpadded_size = self.unpadded_intermediate_size * self.num_shared_experts
-        padding = self.intermediate_size * self.num_shared_experts - unpadded_size
-        pad_weights = (
-            padding > 0
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +37/-92
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-65
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_mega_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56853 - [ROCm][Perf] Enable HCA dual-stream overlap for DeepSeek-V4

- Link: https://github.com/vllm-project/vllm/pull/56853
- Status/date: merged / 2026-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/rocm.py`; associated commits `fdcd42350bd2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +35/-9, 102 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/amd/rocm.py` modified +35/-9 (44 lines); hunks: -558,22 +558,25 @@ def __init__(self, *args, **kwargs):; -624,7 +627,7 @@ def forward(; symbols: __init__, _enable_csa_multi_stream, _enable_multi_stream_overlap, forward, touching `__init__, _enable_csa_multi_stream, _enable_multi_stream_overlap`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +35/-9 (44 lines); hunks: -558,22 +558,25 @@ def __init__(self, *args, **kwargs):; -624,7 +627,7 @@ def forward(; symbols: __init__, _enable_csa_multi_stream, _enable_multi_stream_overlap, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -558,22 +558,25 @@ def __init__(self, *args, **kwargs):
-            # Only enable multi-stream overlap for CSA layer now.
-            self.aux_stream_list = None
+            # Dense layers have no compressor work to overlap; HCA layers
+            # (compressor, no indexer) keep the streams for the dual-stream
+            # fork below.
+            if self.compressor is None:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/amd/rocm.py` modified +35/-9
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/rocm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #54674 - [Perf][DSpark] Stack DeepSeek V4 context WKV projections

- Link: https://github.com/vllm-project/vllm/pull/54674
- Status/date: merged / 2026-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/dspark.py`; associated commits `b3079e6e46a2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +119/-6, 189 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/nvidia/dspark.py` modified +46/-6 (52 lines); hunks: -31,7 +31,10; -61,6 +64,24; symbols: _duplicate_context_wkv_weights, DSparkDeepseekV4Model, __init__, precompute_and_store_context_kv, touching `_duplicate_context_wkv_weights, DSparkDeepseekV4Model, __init__`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/dspark.py` modified +46/-6 (52 lines); hunks: -31,7 +31,10; -61,6 +64,24; symbols: _duplicate_context_wkv_weights, DSparkDeepseekV4Model, __init__, precompute_and_store_context_kv
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/dspark.py
@@ -31,7 +31,10 @@
-from vllm.model_executor.layers.linear import ReplicatedLinear
+from vllm.model_executor.layers.linear import (
+    MergedColumnParallelLinear,
+    ReplicatedLinear,
+)
@@ -61,6 +64,24 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/nvidia/dspark.py` modified +46/-6
- Risk and verification: The diff ships test coverage in `tests/models/test_dspark_mla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56266 - [DSv4.1] Integrate Mega-Gate from DeepGEMM

- Link: https://github.com/vllm-project/vllm/pull/56266
- Status/date: merged / 2026-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`, `vllm/models/deepseek_v41/nvidia/dspark.py` and 6 files; associated commits `41f9104fa649`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +369/-35, 782 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/test_deepseek_v4_mega_moe.py` modified +138/-12 (150 lines); hunks: -15,13 +15,15; -33,6 +35,7; symbols: v41_moe_config, test_deepseek_v41_moe_routes_without_hash_table, touching `v41_moe_config, test_deepseek_v41_moe_routes_without_hash_table`; `vllm/models/deepseek_v4/nvidia/model.py` modified +110/-19 (129 lines); hunks: -98,6 +98,37; -803,6 +834,10 @@ def __init__(; symbols: MegaGateRoutingMetadata, prepare_mega_gate_routing_metadata, DeepseekV4MLP, __init__, touching `MegaGateRoutingMetadata, prepare_mega_gate_routing_metadata, DeepseekV4MLP`; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +20/-1 (21 lines); hunks: -20,6 +20,7; -51,6 +52,7; symbols: __init__, forward, touching `__init__, forward`; `vllm/models/deepseek_v41/nvidia/model.py` modified +15/-1 (16 lines); hunks: -61,7 +61,9; -339,6 +341,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +138/-12 (150 lines); hunks: -15,13 +15,15; -33,6 +35,7; symbols: v41_moe_config, test_deepseek_v41_moe_routes_without_hash_table
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +110/-19 (129 lines); hunks: -98,6 +98,37; -803,6 +834,10 @@ def __init__(; symbols: MegaGateRoutingMetadata, prepare_mega_gate_routing_metadata, DeepseekV4MLP, __init__
  - `vllm/models/deepseek_v4/nvidia/mtp.py` modified +20/-1 (21 lines); hunks: -20,6 +20,7; -51,6 +52,7; symbols: __init__, forward
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +15/-1 (16 lines); hunks: -61,7 +61,9; -339,6 +341,7 @@ def forward(; symbols: forward
  - `vllm/models/deepseek_v4/nvidia/dspark.py` modified +14/-0 (14 lines); hunks: -17,6 +17,7; -51,12 +52,14; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -15,13 +15,15 @@
+    prepare_mega_gate_routing_metadata,
+from vllm.utils.torch_utils import set_default_torch_dtype
@@ -33,6 +35,7 @@
+            dtype=torch.bfloat16,
@@ -63,28 +66,52 @@ def v41_moe_config(dist_init):
+@pytest.mark.parametrize("use_cudagraph", [False, True])
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -98,6 +98,37 @@
+class MegaGateRoutingMetadata(typing.NamedTuple):
+    safe_hash_input_ids: torch.Tensor | None
+    image_token_mask: torch.Tensor | None
+    hash_token_mask: torch.Tensor | None
+def prepare_mega_gate_routing_metadata(
+    input_ids: torch.Tensor,
diff -- vllm/models/deepseek_v4/nvidia/mtp.py
@@ -20,6 +20,7 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +138/-12
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +110/-19; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +20/-1; `vllm/models/deepseek_v41/nvidia/model.py` modified +15/-1; `vllm/models/deepseek_v4/nvidia/dspark.py` modified +14/-0; `vllm/models/deepseek_v41/nvidia/dspark.py` modified +14/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_mhc_kernels.py`, `tests/models/test_deepseek_v4_mega_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57432 - [Bugfix][DSv4.1] Fix FlashInfer DSpark non-causal attention

- Link: https://github.com/vllm-project/vllm/pull/57432
- Status/date: merged / 2026-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py`; associated commits `80447d276559`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +214/-6, 262 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +62/-6 (68 lines); hunks: -26,7 +26,11; -183,6 +187,39 @@ class DeepseekSparseSWAFlashInferMetadataBuilder(DeepseekV4...; symbols: DeepseekSparseSWAFlashInferMetadataBuilder, __init__, build, DeepseekSparseSWAFlashInferBackend, touching `DeepseekSparseSWAFlashInferMetadataBuilder, __init__, build`.
- Code diff details:
  - `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +62/-6 (68 lines); hunks: -26,7 +26,11; -183,6 +187,39 @@ class DeepseekSparseSWAFlashInferMetadataBuilder(DeepseekV4...; symbols: DeepseekSparseSWAFlashInferMetadataBuilder, __init__, build, DeepseekSparseSWAFlashInferBackend
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py
@@ -26,7 +26,11 @@
-from vllm.v1.attention.backend import AttentionCGSupport, MultipleOf
+from vllm.v1.attention.backend import (
+    AttentionCGSupport,
+    CommonAttentionMetadata,
+    MultipleOf,
+)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +62/-6
- Risk and verification: The diff ships test coverage in `tests/v1/attention/test_dspark_noncausal_sparse_mla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51856 - [Bugfix] Attach request-level tools to existing system message in DeepSeek V4 Python renderer

- Link: https://github.com/vllm-project/vllm/pull/51856
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tokenizers_/fixtures/deepseek_v4/test_output_1.txt`, `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py`; associated commits `2909ad8fa472`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +157/-4, 183 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/tokenizers_/test_deepseek_v4.py` modified +143/-0 (143 lines); hunks: -485,3 +485,146 @@ def convert_tokens_to_ids(self, token: str) -> int:; symbols: convert_tokens_to_ids, _request_tools, _encode_reference, test_deepseek_v4_attaches_request_tools_to_existing_system_message, touching `convert_tokens_to_ids, _request_tools, _encode_reference`; `vllm/tokenizers/deepseek_v4.py` modified +12/-2 (14 lines); hunks: -35,8 +35,18 @@ def apply_chat_template(; symbols: apply_chat_template, touching `apply_chat_template`; `tests/tokenizers_/fixtures/deepseek_v4/test_output_1.txt` modified +2/-2 (4 lines); hunks: -1,4 +1,4; -26,7 +26,7 @@ Otherwise, output directly after with tool calls or final resp....
- Code diff details:
  - `tests/tokenizers_/test_deepseek_v4.py` modified +143/-0 (143 lines); hunks: -485,3 +485,146 @@ def convert_tokens_to_ids(self, token: str) -> int:; symbols: convert_tokens_to_ids, _request_tools, _encode_reference, test_deepseek_v4_attaches_request_tools_to_existing_system_message
  - `vllm/tokenizers/deepseek_v4.py` modified +12/-2 (14 lines); hunks: -35,8 +35,18 @@ def apply_chat_template(; symbols: apply_chat_template
  - `tests/tokenizers_/fixtures/deepseek_v4/test_output_1.txt` modified +2/-2 (4 lines); hunks: -1,4 +1,4; -26,7 +26,7 @@ Otherwise, output directly after with tool calls or final resp...
- Key code excerpts:

```diff
diff -- tests/tokenizers_/test_deepseek_v4.py
@@ -485,3 +485,146 @@ def convert_tokens_to_ids(self, token: str) -> int:
+def _request_tools():
+    return [
+        {
+            "type": "function",
+            "function": {
+                "name": "get_weather",
diff -- vllm/tokenizers/deepseek_v4.py
@@ -35,8 +35,18 @@ def apply_chat_template(
-                messages.insert(0, {"role": "system"})
-                messages[0]["tools"] = tools  # type: ignore[typeddict-unknown-key]
+                # Match the Rust renderer: request tools attach to the first
+                # system message; synthesize one only when none exists.
+                system_idx = next(
+                    (i for i, m in enumerate(messages) if m.get("role") == "system"),
diff -- tests/tokenizers_/fixtures/deepseek_v4/test_output_1.txt
@@ -1,4 +1,4 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/tokenizers_/test_deepseek_v4.py` modified +143/-0; `tests/tokenizers_/fixtures/deepseek_v4/test_output_1.txt` modified +2/-2
  - runtime: `vllm/tokenizers/deepseek_v4.py` modified +12/-2
- Risk and verification: The diff ships test coverage in `tests/tokenizers_/fixtures/deepseek_v4/test_output_1.txt`, `tests/tokenizers_/test_deepseek_v4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56271 - [Frontend] Fix the parsing of missing `string=` in DeepSeek V4

- Link: https://github.com/vllm-project/vllm/pull/56271
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_deepseek_v4.py`, `tests/parser/engine/test_deepseek_v41.py`, `vllm/parser/deepseek_v4.py`, `vllm/parser/deepseek_v41.py`; associated commits `39e33db7f3b1`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +174/-29, 306 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/parser/engine/test_deepseek_v4.py` modified +70/-0 (70 lines); hunks: -119,6 +119,35 @@ def test_empty_args(self):; -232,6 +261,47 @@ def test_streaming_split_next_parameter_tag_is_buffered(; symbols: test_empty_args, _bare, test_missing_string_attribute_keeps_literal, test_missing_string_attribute_parses_json, touching `test_empty_args, _bare, test_missing_string_attribute_keeps_literal`; `vllm/parser/deepseek_v4.py` modified +20/-12 (32 lines); hunks: -56,18 +56,24; -80,6 +86,8 @@ def _dsml_arg_converter(; symbols: _param_patterns, _dsml_arg_converter, touching `_param_patterns, _dsml_arg_converter`; `vllm/parser/deepseek_v41.py` modified +2/-13 (15 lines); hunks: -5,11 +5,10; -21,17 +20,7; `tests/parser/engine/test_deepseek_v41.py` modified +2/-0 (2 lines); hunks: -137,10 +137,12 @@ def test_python_argument_conversion_and_partial_values():; symbols: test_python_argument_conversion_and_partial_values, touching `test_python_argument_conversion_and_partial_values`.
- Code diff details:
  - `tests/parser/engine/test_deepseek_v4.py` modified +70/-0 (70 lines); hunks: -119,6 +119,35 @@ def test_empty_args(self):; -232,6 +261,47 @@ def test_streaming_split_next_parameter_tag_is_buffered(; symbols: test_empty_args, _bare, test_missing_string_attribute_keeps_literal, test_missing_string_attribute_parses_json
  - `vllm/parser/deepseek_v4.py` modified +20/-12 (32 lines); hunks: -56,18 +56,24; -80,6 +86,8 @@ def _dsml_arg_converter(; symbols: _param_patterns, _dsml_arg_converter
  - `vllm/parser/deepseek_v41.py` modified +2/-13 (15 lines); hunks: -5,11 +5,10; -21,17 +20,7
  - `tests/parser/engine/test_deepseek_v41.py` modified +2/-0 (2 lines); hunks: -137,10 +137,12 @@ def test_python_argument_conversion_and_partial_values():; symbols: test_python_argument_conversion_and_partial_values
- Key code excerpts:

```diff
diff -- tests/parser/engine/test_deepseek_v4.py
@@ -119,6 +119,35 @@ def test_empty_args(self):
+    def _bare(self, name: str, value: str) -> str:
+        return f'<｜DSML｜parameter name="{name}">{value}{_PARAM_CLOSE}'
+    def test_missing_string_attribute_keeps_literal(self):
+        """The model sometimes omits ``string``; the parameter must survive."""
+        raw = self._bare("name", "alpha")
+        assert json.loads(_dsml_arg_converter(raw, partial=False)) == {"name": "alpha"}
diff -- vllm/parser/deepseek_v4.py
@@ -56,18 +56,24 @@
-_ESCAPED_DSML = re.escape(_DSML)
-_PARAM_RE = re.compile(
-    rf'<{_ESCAPED_DSML}parameter\s+name="([^"]+)"\s+string="(true|false)">'
-    rf"(.*?)"
-    rf"(?:</{_ESCAPED_DSML}parameter>|(?=<{_ESCAPED_DSML}parameter\s+name=))",
-    re.DOTALL,
diff -- vllm/parser/deepseek_v41.py
@@ -5,11 +5,10 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/parser/engine/test_deepseek_v4.py` modified +70/-0; `tests/parser/engine/test_deepseek_v41.py` modified +2/-0
  - runtime: `vllm/parser/deepseek_v4.py` modified +20/-12; `vllm/parser/deepseek_v41.py` modified +2/-13
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_deepseek_v4.py`, `tests/parser/engine/test_deepseek_v41.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56882 - [Bugfix][Multimodal] Preserve DeepSeek V4 image block spacing

- Link: https://github.com/vllm-project/vllm/pull/56882
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tokenizers_/fixtures/deepseek_v4/test_input_5.json`, `tests/tokenizers_/fixtures/deepseek_v4/test_output_5.txt`, `tests/tokenizers_/test_deepseek_v4.py`, `vllm/renderers/deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`; associated commits `cc09352e5c50`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +60/-12, 132 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/tokenizers_/fixtures/deepseek_v4/test_input_5.json` added +39/-0 (39 lines); hunks: -0,0 +1,39; `tests/tokenizers_/test_deepseek_v4.py` modified +7/-7 (14 lines); hunks: -54,13 +54,9 @@ def _load_reference_case(case_id: int):; -397,6 +393,7 @@ def test_deepseek_v4_maps_xhigh_to_high_reasoning_effort():; symbols: _load_reference_case, _render_reference_case, test_deepseek_v4_maps_xhigh_to_high_reasoning_effort, test_deepseek_v4_matches_reference_golden_fixtures, touching `_load_reference_case, _render_reference_case, test_deepseek_v4_maps_xhigh_to_high_reasoning_effort`; `tests/tokenizers_/fixtures/deepseek_v4/test_output_5.txt` added +9/-0 (9 lines); hunks: -0,0 +1,9; `vllm/tokenizers/deepseek_v4_encoding.py` modified +3/-3 (6 lines); hunks: -224,7 +224,7 @@ def flatten_content_blocks(content: Any) -> Any:; -267,7 +267,7 @@ def render_message(; symbols: flatten_content_blocks, render_message, touching `flatten_content_blocks, render_message`.
- Code diff details:
  - `tests/tokenizers_/fixtures/deepseek_v4/test_input_5.json` added +39/-0 (39 lines); hunks: -0,0 +1,39
  - `tests/tokenizers_/test_deepseek_v4.py` modified +7/-7 (14 lines); hunks: -54,13 +54,9 @@ def _load_reference_case(case_id: int):; -397,6 +393,7 @@ def test_deepseek_v4_maps_xhigh_to_high_reasoning_effort():; symbols: _load_reference_case, _render_reference_case, test_deepseek_v4_maps_xhigh_to_high_reasoning_effort, test_deepseek_v4_matches_reference_golden_fixtures
  - `tests/tokenizers_/fixtures/deepseek_v4/test_output_5.txt` added +9/-0 (9 lines); hunks: -0,0 +1,9
  - `vllm/tokenizers/deepseek_v4_encoding.py` modified +3/-3 (6 lines); hunks: -224,7 +224,7 @@ def flatten_content_blocks(content: Any) -> Any:; -267,7 +267,7 @@ def render_message(; symbols: flatten_content_blocks, render_message
  - `vllm/renderers/deepseek_v4.py` modified +2/-2 (4 lines); hunks: -40,7 +40,7 @@ def render_messages(; -67,7 +67,7 @@ async def render_messages_async(; symbols: render_messages, render_messages_async
- Key code excerpts:

```diff
diff -- tests/tokenizers_/fixtures/deepseek_v4/test_input_5.json
@@ -0,0 +1,39 @@
+{
+    "thinking_mode": "thinking",
+    "reasoning_effort": "max",
+    "messages": [
+        {
+            "role": "system",
diff -- tests/tokenizers_/test_deepseek_v4.py
@@ -54,13 +54,9 @@ def _load_reference_case(case_id: int):
-    conversation, _, _ = parse_chat_messages(
-        messages,
-        _model_config(),
-        content_format="string",
-    )
+    # Preserve content blocks for encoding without fetching fixture media.
diff -- tests/tokenizers_/fixtures/deepseek_v4/test_output_5.txt
@@ -0,0 +1,9 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/tokenizers_/fixtures/deepseek_v4/test_input_5.json` added +39/-0; `tests/tokenizers_/test_deepseek_v4.py` modified +7/-7; `tests/tokenizers_/fixtures/deepseek_v4/test_output_5.txt` added +9/-0
  - runtime: `vllm/tokenizers/deepseek_v4_encoding.py` modified +3/-3; `vllm/renderers/deepseek_v4.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/tokenizers_/fixtures/deepseek_v4/test_input_5.json`, `tests/tokenizers_/fixtures/deepseek_v4/test_output_5.txt`, `tests/tokenizers_/test_deepseek_v4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57465 - [DeepSeek V4] Fix fused MoE expert distribution

- Link: https://github.com/vllm-project/vllm/pull/57465
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/model.py`; associated commits `017dced6a6fd`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +5/-7, 26 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-7 (12 lines); hunks: -993,20 +993,18 @@ def _init_fused_moe_experts(; symbols: _init_fused_moe_experts, touching `_init_fused_moe_experts`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-7 (12 lines); hunks: -993,20 +993,18 @@ def _init_fused_moe_experts(; symbols: _init_fused_moe_experts
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -993,20 +993,18 @@ def _init_fused_moe_experts(
-        self.tp_rank = get_tensor_model_parallel_rank()
+        ep_group = get_ep_group()
+        ep_size = ep_group.world_size
+        ep_rank = ep_group.rank_in_group
-        assert self.n_physical_experts % self.tp_size == 0, (
-            f"n_physical_experts={self.n_physical_experts} must be divisible by "
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-7
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #57604 - [Perf][DSV4.1] Optimize MegaMoE staging and NVFP4 cache gathers

- Link: https://github.com/vllm-project/vllm/pull/57604
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`, `vllm/models/deepseek_v41/common/ops/cache_utils.py`, `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py`; associated commits `50d812b66a69`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +130/-62, 369 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +43/-24 (67 lines); hunks: -12,7 +12,7; -40,27 +40,32 @@ def _prepare_megamoe_inputs_kernel(; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs, touching `_prepare_megamoe_inputs_kernel, prepare_megamoe_inputs`; `tests/models/test_deepseek_v4_mega_moe.py` modified +13/-9 (22 lines); hunks: -686,12 +686,14 @@ def forward(self, hidden_states):; -846,6 +848,7 @@ def test_deepseek_v4_mega_moe_stages_shared_scale_tma_layout...; symbols: forward, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, test_deepseek_v4_mega_moe_stages_shared_scale_tma_layout, test_deepseek_v4_pwal_hook_finalizes_mega_moe_and_mhc_broadcast, touching `forward, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, test_deepseek_v4_mega_moe_stages_shared_scale_tma_layout`; `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +5/-1 (6 lines); hunks: -584,7 +584,11 @@ def dequantize_and_gather_k_cache_triton(; symbols: dequantize_and_gather_k_cache_triton, touching `dequantize_and_gather_k_cache_triton`; `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` modified +1/-1 (2 lines); hunks: -422,7 +422,7 @@ def _forward_prefill_mega(; symbols: _forward_prefill_mega, touching `_forward_prefill_mega`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +43/-24 (67 lines); hunks: -12,7 +12,7; -40,27 +40,32 @@ def _prepare_megamoe_inputs_kernel(; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +13/-9 (22 lines); hunks: -686,12 +686,14 @@ def forward(self, hidden_states):; -846,6 +848,7 @@ def test_deepseek_v4_mega_moe_stages_shared_scale_tma_layout...; symbols: forward, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, test_deepseek_v4_mega_moe_stages_shared_scale_tma_layout, test_deepseek_v4_pwal_hook_finalizes_mega_moe_and_mhc_broadcast
  - `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +5/-1 (6 lines); hunks: -584,7 +584,11 @@ def dequantize_and_gather_k_cache_triton(; symbols: dequantize_and_gather_k_cache_triton
  - `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` modified +1/-1 (2 lines); hunks: -422,7 +422,7 @@ def _forward_prefill_mega(; symbols: _forward_prefill_mega
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py
@@ -12,7 +12,7 @@
-@triton.jit
+@triton.jit(do_not_specialize=["num_tokens"])
@@ -40,27 +40,32 @@ def _prepare_megamoe_inputs_kernel(
+    num_tokens,
+    BLOCK_M: tl.constexpr,
-    token_id = tl.program_id(0)
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -686,12 +686,14 @@ def forward(self, hidden_states):
-def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact():
+@pytest.mark.parametrize("num_tokens", [7, 63, 64, 65, 127, 128, 135, 434, 16384])
+@pytest.mark.parametrize("hidden_size", [256, 5120])
+def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact(
+    num_tokens, hidden_size
+):
diff -- vllm/models/deepseek_v41/common/ops/cache_utils.py
@@ -584,7 +584,11 @@ def dequantize_and_gather_k_cache_triton(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +43/-24; `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +5/-1; `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` modified +1/-1
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +13/-9
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`, `tests/models/test_deepseek_v4_mega_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56227 - [Feat][Model] Support encoder-side SWA-bounded replay for DeepSeek-V4.1-Flash

- Link: https://github.com/vllm-project/vllm/pull/56227
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v41_replay_start.py`, `tests/v1/attention/test_deepseek_v4_swa_visible.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/sparse_mla.py` and 11 files; associated commits `1dc2d854c120`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 30 files, +1067/-38, 1949 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/nvidia/model_state.py` modified +143/-2 (145 lines); hunks: -2,15 +2,23; -41,14 +49,69 @@ def _gather_lookback_kernel(; symbols: _gather_lookback_kernel, _pad_replayed_slots_kernel, ReplayAttnMetadata, __init__, touching `_gather_lookback_kernel, _pad_replayed_slots_kernel, ReplayAttnMetadata`; `tests/models/test_deepseek_v41_replay_start.py` added +112/-0 (112 lines); hunks: -0,0 +1,112; symbols: state, prepare_attn, _batch, _prepare, touching `state, prepare_attn, _batch`; `vllm/models/deepseek_v41/attention.py` modified +20/-0 (20 lines); hunks: -512,6 +512,25 @@ def __init__(; -522,6 +541,7 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +13/-0 (13 lines); hunks: -978,6 +978,10 @@ def kernel(; -1124,6 +1128,8 @@ def build_flashinfer_mixed_sparse_indices(; symbols: kernel, build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel, touching `kernel, build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel`.
- Code diff details:
  - `vllm/models/deepseek_v41/nvidia/model_state.py` modified +143/-2 (145 lines); hunks: -2,15 +2,23; -41,14 +49,69 @@ def _gather_lookback_kernel(; symbols: _gather_lookback_kernel, _pad_replayed_slots_kernel, ReplayAttnMetadata, __init__
  - `tests/models/test_deepseek_v41_replay_start.py` added +112/-0 (112 lines); hunks: -0,0 +1,112; symbols: state, prepare_attn, _batch, _prepare
  - `vllm/models/deepseek_v41/attention.py` modified +20/-0 (20 lines); hunks: -512,6 +512,25 @@ def __init__(; -522,6 +541,7 @@ def __init__(; symbols: __init__
  - `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +13/-0 (13 lines); hunks: -978,6 +978,10 @@ def kernel(; -1124,6 +1128,8 @@ def build_flashinfer_mixed_sparse_indices(; symbols: kernel, build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel
  - `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +6/-1 (7 lines); hunks: -200,8 +200,11 @@ def build(; -375,6 +378,7 @@ def _build_sparse_index_metadata(; symbols: build, _build_sparse_index_metadata
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/nvidia/model_state.py
@@ -2,15 +2,23 @@
+import numpy as np
-from vllm.config import VllmConfig
+from vllm.config import CUDAGraphMode, VllmConfig
+from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder
+from vllm.v1.attention.backends.utils import PAD_SLOT_ID
+from vllm.v1.core.sched.output import NewRequestData
diff -- tests/models/test_deepseek_v41_replay_start.py
@@ -0,0 +1,112 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""SWA bounded replay in DeepseekV41ModelState: the batch's replay starts reach
+the sliding-window builders, and the replayed tokens' slots are padded in the
+prefix-cacheable groups only."""
+from dataclasses import replace
diff -- vllm/models/deepseek_v41/attention.py
@@ -512,6 +512,25 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/nvidia/model_state.py` modified +143/-2; `vllm/models/deepseek_v41/attention.py` modified +20/-0; `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +13/-0; `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +6/-1; `vllm/models/deepseek_v4/amd/rocm.py` modified +2/-0; `vllm/models/deepseek_v41/amd/rocm.py` modified +2/-0
  - tests: `tests/models/test_deepseek_v41_replay_start.py` added +112/-0
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v41_replay_start.py`, `tests/v1/attention/test_deepseek_v4_swa_visible.py`, `tests/v1/attention/test_dspark_noncausal_sparse_mla.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57434 - [ROCm][DSv4.1][Perf] Reuse the decode topk ragged metadata across layers

- Link: https://github.com/vllm-project/vllm/pull/57434
- Status/date: merged / 2026-09-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/amd/rocm.py`; associated commits `1596fa5f8825`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +66/-13, 118 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/amd/rocm.py` modified +66/-13 (79 lines); hunks: -13,7 +13,10; -391,6 +394,10 @@ def _copy_ragged_to_graph_buffers(; symbols: _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterSparseSWAMetadata, __init__, get_padded_num_q_heads, touching `_copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterSparseSWAMetadata, __init__`.
- Code diff details:
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +66/-13 (79 lines); hunks: -13,7 +13,10; -391,6 +394,10 @@ def _copy_ragged_to_graph_buffers(; symbols: _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterSparseSWAMetadata, __init__, get_padded_num_q_heads
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -13,7 +13,10 @@
-from vllm.models.deepseek_v41.attention import DeepseekV4Attention
+from vllm.models.deepseek_v41.attention import (
+    DeepseekV4Attention,
+    _replace_layer_index,
+)
@@ -391,6 +394,10 @@ def _copy_ragged_to_graph_buffers(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/amd/rocm.py` modified +66/-13
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v41/amd/rocm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #54894 - [ROCm][DSV4][Perf] Use FP8 WO_A output projection

- Link: https://github.com/vllm-project/vllm/pull/54894
- Status/date: merged / 2026-09-20
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_rocm_wo_a.py`, `vllm/models/deepseek_v4/amd/rocm.py`; associated commits `0e110f696db4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +217/-11, 264 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/amd/rocm.py` modified +173/-11 (184 lines); hunks: -7,6 +7,7; -42,6 +43,49; symbols: _wo_a_block_scale_to_e8m0, _trust_dsv4_extra_cache_nan_free, __init__, _prep, touching `_wo_a_block_scale_to_e8m0, _trust_dsv4_extra_cache_nan_free, __init__`; `tests/models/test_deepseek_v4_rocm_wo_a.py` added +44/-0 (44 lines); hunks: -0,0 +1,44; symbols: test_wo_a_block_scale_to_e8m0_from_float, test_wo_a_block_scale_to_e8m0_preserves_encoded_scales, test_wo_a_block_scale_to_e8m0_rejects_invalid_scales, touching `test_wo_a_block_scale_to_e8m0_from_float, test_wo_a_block_scale_to_e8m0_preserves_encoded_scales, test_wo_a_block_scale_to_e8m0_rejects_invalid_scales`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +173/-11 (184 lines); hunks: -7,6 +7,7; -42,6 +43,49; symbols: _wo_a_block_scale_to_e8m0, _trust_dsv4_extra_cache_nan_free, __init__, _prep
  - `tests/models/test_deepseek_v4_rocm_wo_a.py` added +44/-0 (44 lines); hunks: -0,0 +1,44; symbols: test_wo_a_block_scale_to_e8m0_from_float, test_wo_a_block_scale_to_e8m0_preserves_encoded_scales, test_wo_a_block_scale_to_e8m0_rejects_invalid_scales
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -7,6 +7,7 @@
+from vllm import envs
@@ -42,6 +43,49 @@
+def _wo_a_block_scale_to_e8m0(scale: torch.Tensor) -> torch.Tensor | None:
+    """Normalize checkpoint WO_A scales to raw OCP MX E8M0 bytes.
+    E8M0 is an unsigned exponent-only scale format with bias 127. A finite
+    encoded byte ``b`` in ``[0, 254]`` represents ``2 ** (b - 127)``;
diff -- tests/models/test_deepseek_v4_rocm_wo_a.py
@@ -0,0 +1,44 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+import torch
+from vllm.models.deepseek_v4.amd.rocm import _wo_a_block_scale_to_e8m0
+def test_wo_a_block_scale_to_e8m0_from_float():
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/amd/rocm.py` modified +173/-11
  - tests: `tests/models/test_deepseek_v4_rocm_wo_a.py` added +44/-0
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_rocm_wo_a.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56625 - [DSV4.1] Add encoder cuda graph support for deepseek-v4.1-flash

- Link: https://github.com/vllm-project/vllm/pull/56625
- Status/date: merged / 2026-09-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/vision.py`, `vllm/models/deepseek_v41/amd/vl_model.py`, `vllm/models/deepseek_v41/common/vl_cudagraph.py`, `vllm/models/deepseek_v41/nvidia/vl_model.py`; associated commits `27757dde020e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +599/-63, 809 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/common/vl_cudagraph.py` added +390/-0 (390 lines); hunks: -0,0 +1,390; symbols: DeepseekV4VLEncoderCudaGraphMixin, _encode_image, _build_image_span, _get_grid_list, touching `DeepseekV4VLEncoderCudaGraphMixin, _encode_image, _build_image_span`; `vllm/models/deepseek_v4/common/vision.py` modified +156/-7 (163 lines); hunks: -40,8 +40,7; -52,6 +51,13 @@ def get_vision_cos_sin(; symbols: get_vision_cos_sin, _compute_vision_cos_sin, apply_rotary, touching `get_vision_cos_sin, _compute_vision_cos_sin, apply_rotary`; `vllm/models/deepseek_v41/amd/vl_model.py` modified +12/-28 (40 lines); hunks: -26,6 +26,7; -44,15 +45,12; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, _parse_and_validate_image_input, _encode_image, touching `_make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, _parse_and_validate_image_input`; `vllm/models/deepseek_v41/nvidia/vl_model.py` modified +12/-28 (40 lines); hunks: -26,6 +26,7; -44,15 +45,12; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, _parse_and_validate_image_input, _encode_image, touching `_make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, _parse_and_validate_image_input`.
- Code diff details:
  - `vllm/models/deepseek_v41/common/vl_cudagraph.py` added +390/-0 (390 lines); hunks: -0,0 +1,390; symbols: DeepseekV4VLEncoderCudaGraphMixin, _encode_image, _build_image_span, _get_grid_list
  - `vllm/models/deepseek_v4/common/vision.py` modified +156/-7 (163 lines); hunks: -40,8 +40,7; -52,6 +51,13 @@ def get_vision_cos_sin(; symbols: get_vision_cos_sin, _compute_vision_cos_sin, apply_rotary
  - `vllm/models/deepseek_v41/amd/vl_model.py` modified +12/-28 (40 lines); hunks: -26,6 +26,7; -44,15 +45,12; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, _parse_and_validate_image_input, _encode_image
  - `vllm/models/deepseek_v41/nvidia/vl_model.py` modified +12/-28 (40 lines); hunks: -26,6 +26,7; -44,15 +45,12; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, _parse_and_validate_image_input, _encode_image
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/common/vl_cudagraph.py
@@ -0,0 +1,390 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Encoder CUDA graph support for the DeepSeek-V4.1 vision tower.
+Mixin implementing the ``SupportsEncoderCudaGraph`` protocol
+(``vllm/model_executor/models/interfaces.py``) on top of the shared
+``DeepseekV4ViT``/``DeepseekV4Aligner``. The captured graph packs a whole
diff -- vllm/models/deepseek_v4/common/vision.py
@@ -40,8 +40,7 @@
-@lru_cache(8)
-def get_vision_cos_sin(
+def _compute_vision_cos_sin(
@@ -52,6 +51,13 @@ def get_vision_cos_sin(
+@lru_cache(8)
+def get_vision_cos_sin(
diff -- vllm/models/deepseek_v41/amd/vl_model.py
@@ -26,6 +26,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/common/vl_cudagraph.py` added +390/-0; `vllm/models/deepseek_v4/common/vision.py` modified +156/-7; `vllm/models/deepseek_v41/amd/vl_model.py` modified +12/-28; `vllm/models/deepseek_v41/nvidia/vl_model.py` modified +12/-28
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`, `tests/models/multimodal/processing/test_tensor_schema.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57603 - [Perf][DSV4.1] Overlap mHC coefficients for small TP batches

- Link: https://github.com/vllm-project/vllm/pull/57603
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/mhc.py`; associated commits `9b49f9234431`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +357/-46, 645 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/nvidia/ops/mhc.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: supports_mhc_overlap, mhc_pre_delayed_overlap, touching `supports_mhc_overlap, mhc_pre_delayed_overlap`; `vllm/models/deepseek_v41/nvidia/model.py` modified +60/-16 (76 lines); hunks: -2,6 +2,7; -17,7 +18,11; symbols: __init__, forward, touching `__init__, forward`; `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +29/-1 (30 lines); hunks: -7,6 +7,7; -106,10 +107,37 @@ def mhc_shifted_post_pre(; symbols: mhc_shifted_post_pre, touching `mhc_shifted_post_pre`.
- Code diff details:
  - `vllm/models/deepseek_v41/nvidia/ops/mhc.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: supports_mhc_overlap, mhc_pre_delayed_overlap
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +60/-16 (76 lines); hunks: -2,6 +2,7; -17,7 +18,11; symbols: __init__, forward
  - `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +29/-1 (30 lines); hunks: -7,6 +7,7; -106,10 +107,37 @@ def mhc_shifted_post_pre(; symbols: mhc_shifted_post_pre
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/nvidia/ops/mhc.py
@@ -0,0 +1,117 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Split shifted mHC pre so coefficient generation can overlap the sublayer."""
+from functools import partial
+import torch
+from vllm.config import VllmConfig
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -2,6 +2,7 @@
+from functools import partial
@@ -17,7 +18,11 @@
-from vllm.forward_context import get_forward_context, is_forward_context_available
+from vllm.forward_context import (
+    get_forward_context,
+    in_piecewise_cudagraph,
diff -- vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py
@@ -7,6 +7,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/nvidia/ops/mhc.py` added +117/-0; `vllm/models/deepseek_v41/nvidia/model.py` modified +60/-16; `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +29/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57491 - [ROCm][DSv4.1] Keep the Engram tables in host memory on ROCm

- Link: https://github.com/vllm-project/vllm/pull/57491
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/amd/model.py`; associated commits `6dfd87f59658`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +20/-10, 96 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/amd/model.py` modified +5/-1 (6 lines); hunks: -61,9 +61,13.
- Code diff details:
  - `vllm/models/deepseek_v41/amd/model.py` modified +5/-1 (6 lines); hunks: -61,9 +61,13
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/amd/model.py
@@ -61,9 +61,13 @@
-from ..common.engram import Engram, EngramLayout, NgramHashState
+from ..common.engram import EngramLayout, NgramHashState
+# Engram host offload and its prefetch stream are neither ROCm- nor
+# NVIDIA-specific, so they are imported rather than duplicated.
+from ..nvidia.engram import Engram
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/amd/model.py` modified +5/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/test_engram.py`, `tests/test_config.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57643 - [Perf][DSV4.1] Fuse TP all-reduce with mHC input preparation

- Link: https://github.com/vllm-project/vllm/pull/57643
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/mhc.py`; associated commits `9a70c233cd5d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +549/-105, 890 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +156/-4 (160 lines); hunks: -1,15 +1,26; -29,6 +40,20 @@ def supports_mhc_overlap(vllm_config: VllmConfig) -> bool:; symbols: supports_mhc_overlap, supports_mhc_all_reduce, mhc_pre_delayed_overlap, touching `supports_mhc_overlap, supports_mhc_all_reduce, mhc_pre_delayed_overlap`; `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +0/-98 (98 lines); hunks: -5,10 +5,6; -88,97 +84,3 @@ def mhc_shifted_post_pre_deep_gemm(; symbols: mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre, touching `mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre`; `vllm/models/deepseek_v41/nvidia/model.py` modified +19/-2 (21 lines); hunks: -17,6 +17,7; -88,10 +89,11; symbols: __init__, forward, touching `__init__, forward`; `vllm/models/deepseek_v4/nvidia/model.py` modified +3/-0 (3 lines); hunks: -793,6 +793,7 @@ def __init__(; -804,6 +805,7 @@ def __init__(; symbols: __init__, _init_fused_moe_experts, touching `__init__, _init_fused_moe_experts`.
- Code diff details:
  - `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +156/-4 (160 lines); hunks: -1,15 +1,26; -29,6 +40,20 @@ def supports_mhc_overlap(vllm_config: VllmConfig) -> bool:; symbols: supports_mhc_overlap, supports_mhc_all_reduce, mhc_pre_delayed_overlap
  - `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +0/-98 (98 lines); hunks: -5,10 +5,6; -88,97 +84,3 @@ def mhc_shifted_post_pre_deep_gemm(; symbols: mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +19/-2 (21 lines); hunks: -17,6 +17,7; -88,10 +89,11; symbols: __init__, forward
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +3/-0 (3 lines); hunks: -793,6 +793,7 @@ def __init__(; -804,6 +805,7 @@ def __init__(; symbols: __init__, _init_fused_moe_experts
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/nvidia/ops/mhc.py
@@ -1,15 +1,26 @@
-"""Split shifted mHC pre so coefficient generation can overlap the sublayer."""
+"""Dispatch DSV4.1 mHC operations and overlap coefficient generation."""
+from typing import TYPE_CHECKING, cast
+from vllm.distributed import get_tp_group
+from vllm.model_executor.kernels.mhc.tilelang import (
+    mhc_fused_post_pre_delayed_tilelang,
diff -- vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py
@@ -5,10 +5,6 @@
-from vllm.model_executor.kernels.mhc.tilelang import (
-    mhc_fused_post_pre_delayed_tilelang,
-    mhc_post_tilelang,
-)
@@ -88,97 +84,3 @@ def mhc_shifted_post_pre_deep_gemm(
-def mhc_shifted_post_pre(
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -17,6 +17,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +156/-4; `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +0/-98; `vllm/models/deepseek_v41/nvidia/model.py` modified +19/-2; `vllm/models/deepseek_v4/nvidia/model.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `tests/distributed/test_custom_all_reduce.py`, `tests/distributed/test_engram_dp_shard.py`, `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57874 - [Bugfix][DSV4.1] Restrict mHC overlap to full CUDA graphs

- Link: https://github.com/vllm-project/vllm/pull/57874
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/mhc.py`; associated commits `67513c8b67e2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +45/-11, 108 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +16/-0 (16 lines); hunks: -28,6 +28,22 @@ def is_mega_mhc_supported(hidden_size: int, hc_mult: int) ->...; symbols: is_mega_mhc_supported, can_use_mega_mhc, mhc_shifted_post_pre_deep_gemm, touching `is_mega_mhc_supported, can_use_mega_mhc, mhc_shifted_post_pre_deep_gemm`; `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +6/-8 (14 lines); hunks: -16,7 +16,10; -223,13 +226,8 @@ def mhc_shifted_post_pre(; symbols: mhc_shifted_post_pre, touching `mhc_shifted_post_pre`; `vllm/models/deepseek_v41/nvidia/model.py` modified +2/-1 (3 lines); hunks: -385,8 +385,9 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +16/-0 (16 lines); hunks: -28,6 +28,22 @@ def is_mega_mhc_supported(hidden_size: int, hc_mult: int) ->...; symbols: is_mega_mhc_supported, can_use_mega_mhc, mhc_shifted_post_pre_deep_gemm
  - `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +6/-8 (14 lines); hunks: -16,7 +16,10; -223,13 +226,8 @@ def mhc_shifted_post_pre(; symbols: mhc_shifted_post_pre
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +2/-1 (3 lines); hunks: -385,8 +385,9 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py
@@ -28,6 +28,22 @@ def is_mega_mhc_supported(hidden_size: int, hc_mult: int) -> bool:
+def can_use_mega_mhc(
+    x: torch.Tensor,
+    residual: torch.Tensor,
+    pre_mix: torch.Tensor | None,
+    norm_weight: torch.Tensor | None,
+    capture_aux: bool,
diff -- vllm/models/deepseek_v41/nvidia/ops/mhc.py
@@ -16,7 +16,10 @@
-from .mega_mhc import is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm
+from .mega_mhc import (
+    can_use_mega_mhc,
+    mhc_shifted_post_pre_deep_gemm,
+)
@@ -223,13 +226,8 @@ def mhc_shifted_post_pre(
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -385,8 +385,9 @@ def forward(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +16/-0; `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +6/-8; `vllm/models/deepseek_v41/nvidia/model.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57906 - [Bugfix][ROCm][DSv4.1] Disable SWA bounded replay on ROCm

- Link: https://github.com/vllm-project/vllm/pull/57906
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/attention.py`; associated commits `04c1f4a40796`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +9/-0, 16 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/attention.py` modified +9/-0 (9 lines); hunks: -531,6 +531,15 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/deepseek_v41/attention.py` modified +9/-0 (9 lines); hunks: -531,6 +531,15 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/attention.py
@@ -531,6 +531,15 @@ def __init__(
+        if swa_bounded_replay and current_platform.is_rocm():
+            logger.warning_once(
+                "SWA bounded replay is off on ROCm (the sparse SWA metadata "
+                "builders forward replay_start, but the window clamp it relies "
+                "on lives in the FlashInfer and FlashMLA prefill kernels, so "
+                "the padded slots fault); the sliding-window cache takes part "
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/attention.py` modified +9/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v41/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #52362 - [ROCm][DSv4] Enable DSpark adaptive verification

- Link: https://github.com/vllm-project/vllm/pull/52362
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_dspark_rocm.py`, `tests/v1/attention/test_deepseek_v4_rocm_adaptive.py`, `vllm/models/deepseek_v4/amd/dspark.py`, `vllm/models/deepseek_v4/amd/rocm.py`; associated commits `e34685dfc0c9`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +725/-7, 895 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/test_deepseek_v4_dspark_rocm.py` added +130/-0 (130 lines); hunks: -0,0 +1,130; symbols: _make_uninitialized_model, _prepare_loader_model, _disable_distributed_loader_paths, test_deepseek_v4_rocm_dspark_maps_enabled_confidence_head, touching `_make_uninitialized_model, _prepare_loader_model, _disable_distributed_loader_paths`; `vllm/models/deepseek_v4/amd/dspark.py` modified +28/-5 (33 lines); hunks: -44,6 +44,7; -102,7 +103,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, markov_embed, markov_bias, compute_confidence, touching `__init__, markov_embed, markov_bias`; `vllm/models/deepseek_v4/amd/rocm.py` modified +31/-0 (31 lines); hunks: -8,6 +8,7; -26,6 +27,7; symbols: DeepseekV4ROCMAiterSparseSWAMetadata, DeepseekV4ROCMAiterMLASparseMetadataBuilder, get_cudagraph_support, __init__, touching `DeepseekV4ROCMAiterSparseSWAMetadata, DeepseekV4ROCMAiterMLASparseMetadataBuilder, get_cudagraph_support`; `tests/v1/attention/test_deepseek_v4_rocm_adaptive.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: _make_indexer_config, _mock_rocm_platform, _make_indexer_builder, test_deepseek_v4_rocm_adaptive_builders_support_varlen_full_graphs, touching `_make_indexer_config, _mock_rocm_platform, _make_indexer_builder`.
- Code diff details:
  - `tests/models/test_deepseek_v4_dspark_rocm.py` added +130/-0 (130 lines); hunks: -0,0 +1,130; symbols: _make_uninitialized_model, _prepare_loader_model, _disable_distributed_loader_paths, test_deepseek_v4_rocm_dspark_maps_enabled_confidence_head
  - `vllm/models/deepseek_v4/amd/dspark.py` modified +28/-5 (33 lines); hunks: -44,6 +44,7; -102,7 +103,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, markov_embed, markov_bias, compute_confidence
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +31/-0 (31 lines); hunks: -8,6 +8,7; -26,6 +27,7; symbols: DeepseekV4ROCMAiterSparseSWAMetadata, DeepseekV4ROCMAiterMLASparseMetadataBuilder, get_cudagraph_support, __init__
  - `tests/v1/attention/test_deepseek_v4_rocm_adaptive.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: _make_indexer_config, _mock_rocm_platform, _make_indexer_builder, test_deepseek_v4_rocm_adaptive_builders_support_varlen_full_graphs
- Key code excerpts:

```diff
diff -- tests/models/test_deepseek_v4_dspark_rocm.py
@@ -0,0 +1,130 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from types import SimpleNamespace
+import pytest
+import torch
+from vllm.models.deepseek_v4.amd import dspark as dspark_module
diff -- vllm/models/deepseek_v4/amd/dspark.py
@@ -44,6 +44,7 @@
+    DSparkConfidenceHead,
@@ -102,7 +103,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
-        # Heads: final norm + hc_head, and the Markov head
+        # Heads: final norm + hc_head, and the Markov + confidence heads
@@ -125,6 +126,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
+        self.confidence_head: DSparkConfidenceHead | None = None
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -8,6 +8,7 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/test_deepseek_v4_dspark_rocm.py` added +130/-0; `tests/v1/attention/test_deepseek_v4_rocm_adaptive.py` added +249/-0
  - runtime: `vllm/models/deepseek_v4/amd/dspark.py` modified +28/-5; `vllm/models/deepseek_v4/amd/rocm.py` modified +31/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`, `tests/models/test_deepseek_v4_dspark_rocm.py`, `tests/v1/attention/test_deepseek_v4_rocm_adaptive.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57919 - [ROCm][Bugfix] Explicitly reject FSE=1 with DPA+ETP deployment for DeepSeek-V4

- Link: https://github.com/vllm-project/vllm/pull/57919
- Status/date: merged / 2026-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_vl_rocm.py`, `vllm/models/deepseek_v4/amd/model.py`; associated commits `42a85a497619`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +43/-5, 93 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/amd/model.py` modified +21/-4 (25 lines); hunks: -12,7 +12,7; -504,11 +504,28 @@ def forward_modular(; symbols: forward_modular, _fuse_shared_experts_enabled, __init__, touching `forward_modular, _fuse_shared_experts_enabled, __init__`; `tests/models/test_deepseek_v4_vl_rocm.py` modified +22/-1 (23 lines); hunks: -7,6 +7,7; -100,7 +101,9 @@ def fake_factory(**kwargs):; symbols: fake_factory, test_rocm_fse_rejects_data_parallel_deployment, test_rocm_mtp_forwards_input_ids_for_vision_routing, touching `fake_factory, test_rocm_fse_rejects_data_parallel_deployment, test_rocm_mtp_forwards_input_ids_for_vision_routing`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/model.py` modified +21/-4 (25 lines); hunks: -12,7 +12,7; -504,11 +504,28 @@ def forward_modular(; symbols: forward_modular, _fuse_shared_experts_enabled, __init__
  - `tests/models/test_deepseek_v4_vl_rocm.py` modified +22/-1 (23 lines); hunks: -7,6 +7,7; -100,7 +101,9 @@ def fake_factory(**kwargs):; symbols: fake_factory, test_rocm_fse_rejects_data_parallel_deployment, test_rocm_mtp_forwards_input_ids_for_vision_routing
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -12,7 +12,7 @@
-from vllm.config import VllmConfig, get_current_vllm_config
+from vllm.config import ParallelConfig, VllmConfig
@@ -504,11 +504,28 @@ def forward_modular(
-def _fuse_shared_experts_enabled(config) -> bool:
+def _fuse_shared_experts_enabled(config, parallel_config: ParallelConfig) -> bool:
+    if (
diff -- tests/models/test_deepseek_v4_vl_rocm.py
@@ -7,6 +7,7 @@
+from vllm.config import ParallelConfig
@@ -100,7 +101,9 @@ def fake_factory(**kwargs):
-        model_config=SimpleNamespace(hf_config=config), quant_config=None
+        model_config=SimpleNamespace(hf_config=config),
+        quant_config=None,
+        parallel_config=ParallelConfig(),
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/amd/model.py` modified +21/-4
  - tests: `tests/models/test_deepseek_v4_vl_rocm.py` modified +22/-1
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_vl_rocm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57428 - [Kernel][DSV4.1] Fuse MXFP8 wo_b GEMM with sequence-parallel reduce-scatter

- Link: https://github.com/vllm-project/vllm/pull/57428
- Status/date: merged / 2026-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`, `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/nvidia/dspark.py`, `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py`, `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` and 7 files; associated commits `f92b78f6ef9c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 18 files, +1036/-623, 2293 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/model.py` modified +13/-44 (57 lines); hunks: -160,54 +160,23 @@ def maybe_init_gemm_rs_ar(vllm_config: VllmConfig, use_seq...; -276,7 +245,7 @@ def __init__(; symbols: maybe_init_gemm_rs_ar, __init__, forward, touching `maybe_init_gemm_rs_ar, __init__, forward`; `vllm/models/deepseek_v41/attention.py` modified +40/-0 (40 lines); hunks: -26,13 +26,15; -389,6 +391,9 @@ def __init__(; symbols: __init__, forward, bind_gemm_rs, _wo_b_proj, touching `__init__, forward, bind_gemm_rs`; `vllm/models/deepseek_v41/nvidia/model.py` modified +37/-1 (38 lines); hunks: -193,6 +193,30 @@ def _use_sequence_parallel(vllm_config: VllmConfig) -> bool:; -202,6 +226,7 @@ def __init__(; symbols: _use_sequence_parallel, maybe_init_gemm_rs, DeepseekV4DecoderLayer, __init__, touching `_use_sequence_parallel, maybe_init_gemm_rs, DeepseekV4DecoderLayer`; `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +6/-2 (8 lines); hunks: -1,5 +1,7; -31,7 +33,7 @@ def deep_gemm_fp8_o_proj(; symbols: deep_gemm_fp8_o_proj, touching `deep_gemm_fp8_o_proj`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/model.py` modified +13/-44 (57 lines); hunks: -160,54 +160,23 @@ def maybe_init_gemm_rs_ar(vllm_config: VllmConfig, use_seq...; -276,7 +245,7 @@ def __init__(; symbols: maybe_init_gemm_rs_ar, __init__, forward
  - `vllm/models/deepseek_v41/attention.py` modified +40/-0 (40 lines); hunks: -26,13 +26,15; -389,6 +391,9 @@ def __init__(; symbols: __init__, forward, bind_gemm_rs, _wo_b_proj
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +37/-1 (38 lines); hunks: -193,6 +193,30 @@ def _use_sequence_parallel(vllm_config: VllmConfig) -> bool:; -202,6 +226,7 @@ def __init__(; symbols: _use_sequence_parallel, maybe_init_gemm_rs, DeepseekV4DecoderLayer, __init__
  - `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +6/-2 (8 lines); hunks: -1,5 +1,7; -31,7 +33,7 @@ def deep_gemm_fp8_o_proj(; symbols: deep_gemm_fp8_o_proj
  - `vllm/models/deepseek_v41/nvidia/dspark.py` modified +7/-0 (7 lines); hunks: -58,6 +58,7; -113,12 +114,18 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -160,54 +160,23 @@ def maybe_init_gemm_rs_ar(vllm_config: VllmConfig, use_sequence_parallel: bool)
-    enabled = envs.VLLM_KIMI_K3_GEMM_AR if all_reduce else envs.VLLM_KIMI_K3_GEMM_RS
+    enabled = envs.VLLM_KIMI_K3_GEMM_AR if all_reduce else envs.VLLM_ENABLE_GEMM_RS
-    parallel_config = vllm_config.parallel_config
-    tp_size = parallel_config.tensor_parallel_size
-    if parallel_config.use_ubatching:
-        reason = "ubatching is enabled"
diff -- vllm/models/deepseek_v41/attention.py
@@ -26,13 +26,15 @@
+from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter
+    from vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar import GemmRsAr
@@ -389,6 +391,9 @@ def __init__(
+        # Set by ``bind_gemm_rs`` when the decoder layer runs sequence
+        # parallel and the fused GEMM + reduce-scatter kernel accepts wo_b.
+        self.gemm_rs: GemmRsAr | None = None
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -193,6 +193,30 @@ def _use_sequence_parallel(vllm_config: VllmConfig) -> bool:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +13/-44; `vllm/models/deepseek_v41/attention.py` modified +40/-0; `vllm/models/deepseek_v41/nvidia/model.py` modified +37/-1; `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +6/-2; `vllm/models/deepseek_v41/nvidia/dspark.py` modified +7/-0; `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/test_gemm_rs_ar.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57435 - [ROCm][DSv4.1][Perf] Fuse the inverse RoPE into the sparse decode reduce

- Link: https://github.com/vllm-project/vllm/pull/57435
- Status/date: merged / 2026-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/amd/rocm.py`; associated commits `ca831d1c5571`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +184/-13, 412 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/amd/rocm.py` modified +22/-3 (25 lines); hunks: -37,6 +37,7; -673,6 +674,7 @@ def _o_proj(self, attn_out: torch.Tensor, positions: torch.T...; symbols: _o_proj, forward_mqa, _decode_topk_ragged, _forward_decode, touching `_o_proj, forward_mqa, _decode_topk_ragged`.
- Code diff details:
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +22/-3 (25 lines); hunks: -37,6 +37,7; -673,6 +674,7 @@ def _o_proj(self, attn_out: torch.Tensor, positions: torch.T...; symbols: _o_proj, forward_mqa, _decode_topk_ragged, _forward_decode
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -37,6 +37,7 @@
+    rocm_inverse_rope_rows_,
@@ -673,6 +674,7 @@ def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tens
+            inverse_rope=False,
@@ -747,15 +749,28 @@ def forward_mqa(
+        rotated = 0
-            self._forward_decode(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/amd/rocm.py` modified +22/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57451 - [ROCm][DSv4][Perf] Fuse the inverse RoPE into the sparse decode reduce

- Link: https://github.com/vllm-project/vllm/pull/57451
- Status/date: merged / 2026-09-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/rocm.py`; associated commits `f9dce295c96f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +27/-3, 84 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/amd/rocm.py` modified +27/-3 (30 lines); hunks: -37,6 +37,7; -1175,6 +1176,7 @@ def _o_proj(self, o: torch.Tensor, positions: torch.Tensor...; symbols: _o_proj, forward_mqa, _forward_decode, touching `_o_proj, forward_mqa, _forward_decode`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +27/-3 (30 lines); hunks: -37,6 +37,7; -1175,6 +1176,7 @@ def _o_proj(self, o: torch.Tensor, positions: torch.Tensor...; symbols: _o_proj, forward_mqa, _forward_decode
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -37,6 +37,7 @@
+    rocm_inverse_rope_rows_,
@@ -1175,6 +1176,7 @@ def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
+                inverse_rope=False,
@@ -1245,9 +1247,15 @@ def forward_mqa(
+        # The fp8 wo_a path rotates inside inverse_rope_group_quant, so folding
+        # the rotation into the decode reduce would apply it twice. Only the
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/amd/rocm.py` modified +27/-3
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/rocm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44229 - [Feature][Frontend] Add DeepSeek-V4 FIM completion rendering

- Link: https://github.com/vllm-project/vllm/pull/44229
- Status/date: merged / 2026-09-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/renderers/deepseek_v4.py`; associated commits `d95a896ea612`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +243/-7, 323 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/renderers/deepseek_v4.py` modified +7/-0 (7 lines); hunks: -16,6 +16,10; -32,6 +36,9 @@ def __init__(; symbols: DeepseekV4Renderer, __init__, _apply_chat_template, render_completion_suffix, touching `DeepseekV4Renderer, __init__, _apply_chat_template`.
- Code diff details:
  - `vllm/renderers/deepseek_v4.py` modified +7/-0 (7 lines); hunks: -16,6 +16,10; -32,6 +36,9 @@ def __init__(; symbols: DeepseekV4Renderer, __init__, _apply_chat_template, render_completion_suffix
- Key code excerpts:

```diff
diff -- vllm/renderers/deepseek_v4.py
@@ -16,6 +16,10 @@
+_FIM_BEGIN = "<｜fim▁begin｜>"
+_FIM_HOLE = "<｜fim▁hole｜>"
+_FIM_END = "<｜fim▁end｜>"
@@ -32,6 +36,9 @@ def __init__(
+    def render_completion_suffix(self, prompt: str, suffix: str) -> str | None:
+        return f"{_FIM_BEGIN}{prompt}{_FIM_HOLE}{suffix}{_FIM_END}"
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/renderers/deepseek_v4.py` modified +7/-0
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/completion/test_completion_error.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58456 - [ROCm][DSv4.1][Perf] Emit MXFP8 from the sparse decode reduce and run wo_a as a grouped FP8 GEMM

- Link: https://github.com/vllm-project/vllm/pull/58456
- Status/date: merged / 2026-09-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/amd/rocm.py`; associated commits `721d0e5c112a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +664/-29, 957 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/amd/rocm.py` modified +136/-22 (158 lines); hunks: -13,6 +13,8; -37,7 +39,9; symbols: _split_qkv_and_norm, _o_proj, _alloc_attn_out, touching `_split_qkv_and_norm, _o_proj, _alloc_attn_out`.
- Code diff details:
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +136/-22 (158 lines); hunks: -13,6 +13,8; -37,7 +39,9; symbols: _split_qkv_and_norm, _o_proj, _alloc_attn_out
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -13,6 +13,8 @@
+from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
+from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
@@ -37,7 +39,9 @@
+    rocm_inverse_rope_mxfp8_rows,
+    rocm_mxfp8_wo_a_bmm,
@@ -663,7 +667,40 @@ def _split_qkv_and_norm(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/amd/rocm.py` modified +136/-22
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57679 - [Perf][DSv4.1] Restore the fused query RMSNorm + MXFP8 quantization path

- Link: https://github.com/vllm-project/vllm/pull/57679
- Status/date: merged / 2026-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/common/ops/query_quant.py`; associated commits `609731e8f56c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +8/-3, 40 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/common/ops/query_quant.py` modified +8/-3 (11 lines); hunks: -3,7 +3,10; -102,6 +105,7 @@ def fused_q_kv_rmsnorm_quant(; symbols: fused_q_kv_rmsnorm_quant, can_fuse_query_quant, touching `fused_q_kv_rmsnorm_quant, can_fuse_query_quant`.
- Code diff details:
  - `vllm/models/deepseek_v41/common/ops/query_quant.py` modified +8/-3 (11 lines); hunks: -3,7 +3,10; -102,6 +105,7 @@ def fused_q_kv_rmsnorm_quant(; symbols: fused_q_kv_rmsnorm_quant, can_fuse_query_quant
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/common/ops/query_quant.py
@@ -3,7 +3,10 @@
-from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
+from vllm.model_executor.layers.fusion.quant_activation import (
+    QuantizedActivation,
+    get_input_quant_key,
+)
@@ -102,6 +105,7 @@ def fused_q_kv_rmsnorm_quant(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/common/ops/query_quant.py` modified +8/-3
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v41/common/ops/query_quant.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #58621 - [Perf][DSv4] Fuse inverse RoPE + FP8 quant into FlashInfer sparse MLA

- Link: https://github.com/vllm-project/vllm/pull/58621
- Status/date: merged / 2026-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`; associated commits `5ff3bbfb089a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +583/-104, 984 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +116/-60 (176 lines); hunks: -9,6 +9,8; -17,6 +19,8; symbols: DeepseekV4FlashInferMLAAttention, get_padded_num_q_heads, _o_proj, touching `DeepseekV4FlashInferMLAAttention, get_padded_num_q_heads, _o_proj`; `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +83/-16 (99 lines); hunks: -5,11 +5,15; -29,7 +33,7 @@ def compute_fp8_einsum_recipe(; symbols: compute_fp8_einsum_recipe, deep_gemm_fp8_o_proj, touching `compute_fp8_einsum_recipe, deep_gemm_fp8_o_proj`; `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +72/-8 (80 lines); hunks: -1099,6 +1099,7 @@ def build_flashinfer_mixed_sparse_indices(; -1109,6 +1110,9 @@ def build_flashinfer_mixed_sparse_indices(; symbols: build_flashinfer_mixed_sparse_indices, kernel, touching `build_flashinfer_mixed_sparse_indices, kernel`; `vllm/models/deepseek_v4/attention.py` modified +37/-20 (57 lines); hunks: -43,6 +43,7; -171,14 +172,20 @@ def forward_mqa(; symbols: forward_mqa, _o_proj, _uses_fp8_ds_mla_layout, forward, touching `forward_mqa, _o_proj, _uses_fp8_ds_mla_layout`.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +116/-60 (176 lines); hunks: -9,6 +9,8; -17,6 +19,8; symbols: DeepseekV4FlashInferMLAAttention, get_padded_num_q_heads, _o_proj
  - `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +83/-16 (99 lines); hunks: -5,11 +5,15; -29,7 +33,7 @@ def compute_fp8_einsum_recipe(; symbols: compute_fp8_einsum_recipe, deep_gemm_fp8_o_proj
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +72/-8 (80 lines); hunks: -1099,6 +1099,7 @@ def build_flashinfer_mixed_sparse_indices(; -1109,6 +1110,9 @@ def build_flashinfer_mixed_sparse_indices(; symbols: build_flashinfer_mixed_sparse_indices, kernel
  - `vllm/models/deepseek_v4/attention.py` modified +37/-20 (57 lines); hunks: -43,6 +43,7; -171,14 +172,20 @@ def forward_mqa(; symbols: forward_mqa, _o_proj, _uses_fp8_ds_mla_layout, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
@@ -9,6 +9,8 @@
+from vllm.logger import init_logger
+from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
@@ -17,6 +19,8 @@
+    rope_quant_attn_out,
+    rope_quant_unsupported_reason,
@@ -37,6 +41,8 @@
diff -- vllm/models/deepseek_v4/nvidia/ops/o_proj.py
@@ -5,11 +5,15 @@
+from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
+from vllm.model_executor.layers.quantization.utils.quant_utils import (
+    kFp8Dynamic128Sym,
+)
-from vllm.utils.deep_gemm import fp8_einsum
+from vllm.utils.deep_gemm import fp8_einsum, get_tma_aligned_size
diff -- vllm/models/deepseek_v4/common/ops/cache_utils.py
@@ -1099,6 +1099,7 @@ def build_flashinfer_mixed_sparse_indices(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +116/-60; `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +83/-16; `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +72/-8; `vllm/models/deepseek_v4/attention.py` modified +37/-20
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_flashmla_sparse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58740 - [ROCm][CI] Test AMD DeepSeek V4 MoE routing against a PyTorch reference

- Link: https://github.com/vllm-project/vllm/pull/58740
- Status/date: merged / 2026-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v4_vl_rocm.py`; associated commits `e3dd5b5a758a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +126/-0, 133 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/test_deepseek_v4_vl_rocm.py` modified +126/-0 (126 lines); hunks: -16,6 +16,132; symbols: test_rocm_moe_routing_and_shared_experts_match_reference, expert_reference, test_rocm_packed_kv_cache_auto_uses_ds_mla_layout, touching `test_rocm_moe_routing_and_shared_experts_match_reference, expert_reference, test_rocm_packed_kv_cache_auto_uses_ds_mla_layout`.
- Code diff details:
  - `tests/models/test_deepseek_v4_vl_rocm.py` modified +126/-0 (126 lines); hunks: -16,6 +16,132; symbols: test_rocm_moe_routing_and_shared_experts_match_reference, expert_reference, test_rocm_packed_kv_cache_auto_uses_ds_mla_layout
- Key code excerpts:

```diff
diff -- tests/models/test_deepseek_v4_vl_rocm.py
@@ -16,6 +16,132 @@
+@pytest.mark.parametrize("hash_routing", [False, True], ids=["regular", "hash"])
+@pytest.mark.parametrize("vision", [False, True], ids=["text", "vision"])
+@torch.inference_mode()
+def test_rocm_moe_routing_and_shared_experts_match_reference(
+    hash_routing, vision, monkeypatch, default_vllm_config, dist_init
+) -> None:
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/test_deepseek_v4_vl_rocm.py` modified +126/-0
- Risk and verification: The diff ships test coverage in `tests/models/test_deepseek_v4_vl_rocm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58678 - [Perf][DSv4.1] Shard the Engram wkv projection across TP ranks

- Link: https://github.com/vllm-project/vllm/pull/58678
- Status/date: merged / 2026-09-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/common/engram.py`; associated commits `4bb804cc5bf0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +7/-3, 37 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/common/engram.py` modified +6/-2 (8 lines); hunks: -47,7 +47,7; -923,13 +923,17 @@ def __init__(; symbols: __init__, touching `__init__`; `tests/kernels/test_engram.py` modified +1/-1 (2 lines); hunks: -827,7 +827,7 @@ def test_engram_constructor_honors_offload(monkeypatch, back...; symbols: test_engram_constructor_honors_offload, touching `test_engram_constructor_honors_offload`.
- Code diff details:
  - `vllm/models/deepseek_v41/common/engram.py` modified +6/-2 (8 lines); hunks: -47,7 +47,7; -923,13 +923,17 @@ def __init__(; symbols: __init__
  - `tests/kernels/test_engram.py` modified +1/-1 (2 lines); hunks: -827,7 +827,7 @@ def test_engram_constructor_honors_offload(monkeypatch, back...; symbols: test_engram_constructor_honors_offload
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/common/engram.py
@@ -47,7 +47,7 @@
-from vllm.model_executor.layers.linear import ReplicatedLinear
+from vllm.model_executor.layers.linear import ColumnParallelLinear
@@ -923,13 +923,17 @@ def __init__(
-        self.wkv = ReplicatedLinear(
+        # Without sequence parallelism every TP rank holds every token, so
+        # shard the output columns instead of replicating the projection.
diff -- tests/kernels/test_engram.py
@@ -827,7 +827,7 @@ def test_engram_constructor_honors_offload(monkeypatch, backend, cpu_offload):
-        engram_ops, "ReplicatedLinear", lambda *a, **k: torch.nn.Identity()
+        engram_ops, "ColumnParallelLinear", lambda *a, **k: torch.nn.Identity()
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/common/engram.py` modified +6/-2
  - tests: `tests/kernels/test_engram.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/test_engram.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57071 - [Bugfix][ROCm] AMD-Quark mixed-precision DeepSeek-V4.1 support

- Link: https://github.com/vllm-project/vllm/pull/57071
- Status/date: merged / 2026-09-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/quant_config.py`, `vllm/models/deepseek_v41/amd/model.py`, `vllm/models/deepseek_v41/amd/vl_model.py`, `vllm/models/deepseek_v41/quant_config.py`; associated commits `5840d95284fe`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +647/-102, 1009 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/amd/vl_model.py` modified +26/-8 (34 lines); hunks: -79,20 +79,22 @@ class DeepseekV4VLImagePixelInputs(TensorSchema):; -131,6 +133,22 @@ class DeepseekV41ForCausalLM(; symbols: DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, attributes, touching `DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM`; `vllm/models/deepseek_v41/quant_config.py` modified +9/-7 (16 lines); hunks: -139,15 +139,17 @@ def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:; symbols: _is_quark_mxfp4_ocp, override_quantization_method, touching `_is_quark_mxfp4_ocp, override_quantization_method`; `vllm/models/deepseek_v4/quant_config.py` modified +7/-7 (14 lines); hunks: -135,15 +135,15 @@ def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:; symbols: _is_quark_mxfp4_ocp, override_quantization_method, touching `_is_quark_mxfp4_ocp, override_quantization_method`; `vllm/models/deepseek_v41/amd/model.py` modified +10/-0 (10 lines); hunks: -895,6 +895,11 @@ def _make_deepseek_v4_weights_mapper(; -908,6 +913,11 @@ def _make_deepseek_v4_weights_mapper(; symbols: _make_deepseek_v4_weights_mapper, touching `_make_deepseek_v4_weights_mapper`.
- Code diff details:
  - `vllm/models/deepseek_v41/amd/vl_model.py` modified +26/-8 (34 lines); hunks: -79,20 +79,22 @@ class DeepseekV4VLImagePixelInputs(TensorSchema):; -131,6 +133,22 @@ class DeepseekV41ForCausalLM(; symbols: DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, attributes
  - `vllm/models/deepseek_v41/quant_config.py` modified +9/-7 (16 lines); hunks: -139,15 +139,17 @@ def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:; symbols: _is_quark_mxfp4_ocp, override_quantization_method
  - `vllm/models/deepseek_v4/quant_config.py` modified +7/-7 (14 lines); hunks: -135,15 +135,15 @@ def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:; symbols: _is_quark_mxfp4_ocp, override_quantization_method
  - `vllm/models/deepseek_v41/amd/model.py` modified +10/-0 (10 lines); hunks: -895,6 +895,11 @@ def _make_deepseek_v4_weights_mapper(; -908,6 +913,11 @@ def _make_deepseek_v4_weights_mapper(; symbols: _make_deepseek_v4_weights_mapper
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/amd/vl_model.py
@@ -79,20 +79,22 @@ class DeepseekV4VLImagePixelInputs(TensorSchema):
+_VL_PREFIX_MAPPING: dict[str, str | None] = {
+    "layers.": "language_model.model.layers.",
+    "embed.": "language_model.model.embed.",
+    "norm.": "language_model.model.norm.",
+    "hc_head": "language_model.model.hc_head",
+    "mtp.": "language_model.model.mtp.",
diff -- vllm/models/deepseek_v41/quant_config.py
@@ -139,15 +139,17 @@ def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:
+        # Quark checkpoints are always handled by QuarkConfig, which knows
+        # how to dispatch each per-layer scheme (MXFP4, MXFP8, etc.).
+        # ``from_config`` below rewrites the config into a single global FP8
+        # scheme, which would discard per-layer specs — so never claim Quark.
+        if isinstance(hf_quant_cfg, dict) and (
+            hf_quant_cfg.get("quant_method") == "quark"
diff -- vllm/models/deepseek_v4/quant_config.py
@@ -135,15 +135,15 @@ def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/amd/vl_model.py` modified +26/-8; `vllm/models/deepseek_v41/quant_config.py` modified +9/-7; `vllm/models/deepseek_v4/quant_config.py` modified +7/-7; `vllm/models/deepseek_v41/amd/model.py` modified +10/-0
- Risk and verification: The diff ships test coverage in `tests/quantization/test_fp8.py`, `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58316 - [Bugfix][Frontend][Rust Frontend] Update DeepSeek V4.1 Flash reasoning effort mappings

- Link: https://github.com/vllm-project/vllm/pull/58316
- Status/date: merged / 2026-09-26
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt`, `tests/tokenizers_/test_deepseek_v41.py`, `vllm/tokenizers/deepseek_v41_encoding.py`; associated commits `ddd6fbca148a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +20/-20, 119 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/tokenizers_/test_deepseek_v41.py` modified +4/-4 (8 lines); hunks: -35,9 +35,9 @@ def test_reference_encoder_fixtures(case_id):; -138,7 +138,7 @@ def test_top_level_effort_overrides_template_effort():; symbols: test_reference_encoder_fixtures, test_top_level_effort_overrides_template_effort, test_encode_uses_one_bos_and_forwards_truncation, touching `test_reference_encoder_fixtures, test_top_level_effort_overrides_template_effort, test_encode_uses_one_bos_and_forwards_truncation`; `vllm/tokenizers/deepseek_v41_encoding.py` modified +2/-2 (4 lines); hunks: -181,8 +181,8 @@ def sort_tool_results_by_call_order(; symbols: sort_tool_results_by_call_order, touching `sort_tool_results_by_call_order`; `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` modified +1/-1 (2 lines); hunks: -1,4 +1,4; `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt` modified +1/-1 (2 lines); hunks: -1,3 +1,3.
- Code diff details:
  - `tests/tokenizers_/test_deepseek_v41.py` modified +4/-4 (8 lines); hunks: -35,9 +35,9 @@ def test_reference_encoder_fixtures(case_id):; -138,7 +138,7 @@ def test_top_level_effort_overrides_template_effort():; symbols: test_reference_encoder_fixtures, test_top_level_effort_overrides_template_effort, test_encode_uses_one_bos_and_forwards_truncation
  - `vllm/tokenizers/deepseek_v41_encoding.py` modified +2/-2 (4 lines); hunks: -181,8 +181,8 @@ def sort_tool_results_by_call_order(; symbols: sort_tool_results_by_call_order
  - `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` modified +1/-1 (2 lines); hunks: -1,4 +1,4
  - `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt` modified +1/-1 (2 lines); hunks: -1,3 +1,3
  - `rust/src/chat/src/renderer/deepseek_v41/fixtures/test_output_1.txt` modified +1/-1 (2 lines); hunks: -1,4 +1,4
- Key code excerpts:

```diff
diff -- tests/tokenizers_/test_deepseek_v41.py
@@ -35,9 +35,9 @@ def test_reference_encoder_fixtures(case_id):
-        (None, 50),
-        ("low", 25),
-        ("high", 50),
+        (None, 75),
+        ("low", 50),
+        ("high", 75),
diff -- vllm/tokenizers/deepseek_v41_encoding.py
@@ -181,8 +181,8 @@ def sort_tool_results_by_call_order(
-    "low": 25,
-    "high": 50,
+    "low": 50,
+    "high": 75,
diff -- tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt
@@ -1,4 +1,4 @@
-<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)
+<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)
```

- Extracted files (not manually reviewed):
  - tests: `tests/tokenizers_/test_deepseek_v41.py` modified +4/-4; `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` modified +1/-1; `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt` modified +1/-1
  - runtime: `vllm/tokenizers/deepseek_v41_encoding.py` modified +2/-2
  - other: `rust/src/chat/src/renderer/deepseek_v41/fixtures/test_output_1.txt` modified +1/-1; `rust/src/chat/src/renderer/deepseek_v41/fixtures/test_output_2.txt` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt`, `tests/tokenizers_/test_deepseek_v41.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58499 - [Bugfix][DSV4.1] Avoid host sync in ViT CUDA graph replay metadata

- Link: https://github.com/vllm-project/vllm/pull/58499
- Status/date: merged / 2026-09-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/common/vision.py`, `vllm/models/deepseek_v41/common/vl_cudagraph.py`; associated commits `7d8c5fe9a95d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +24/-15, 92 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/common/vl_cudagraph.py` modified +14/-8 (22 lines); hunks: -25,6 +25,7; -370,18 +371,23 @@ def postprocess_encoder_output(; symbols: postprocess_encoder_output, touching `postprocess_encoder_output`; `vllm/models/deepseek_v4/common/vision.py` modified +10/-7 (17 lines); hunks: -38,6 +38,7; -219,8 +220,8 @@ def forward(; symbols: _compute_vision_cos_sin, forward, build_packed_vit_metadata, build_packed_merge_metadata, touching `_compute_vision_cos_sin, forward, build_packed_vit_metadata`.
- Code diff details:
  - `vllm/models/deepseek_v41/common/vl_cudagraph.py` modified +14/-8 (22 lines); hunks: -25,6 +25,7; -370,18 +371,23 @@ def postprocess_encoder_output(; symbols: postprocess_encoder_output
  - `vllm/models/deepseek_v4/common/vision.py` modified +10/-7 (17 lines); hunks: -38,6 +38,7; -219,8 +220,8 @@ def forward(; symbols: _compute_vision_cos_sin, forward, build_packed_vit_metadata, build_packed_merge_metadata
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/common/vl_cudagraph.py
@@ -25,6 +25,7 @@
+from vllm.utils.torch_utils import async_tensor_h2d
@@ -370,18 +371,23 @@ def postprocess_encoder_output(
-        types = batch_mm_kwargs["types"].to(aligner_out.device)
+        types = batch_mm_kwargs["types"]
-        # Batched span assembly: one masked fill per role across all items.
-        dtype = aligner_out.dtype
diff -- vllm/models/deepseek_v4/common/vision.py
@@ -38,6 +38,7 @@
+from vllm.utils.torch_utils import async_tensor_h2d
@@ -219,8 +220,8 @@ def forward(
-        cos = cos.to(device=x.device)
-        sin = sin.to(device=x.device)
+        cos = async_tensor_h2d(cos, x.device)
+        sin = async_tensor_h2d(sin, x.device)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/common/vl_cudagraph.py` modified +14/-8; `vllm/models/deepseek_v4/common/vision.py` modified +10/-7
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/common/vision.py`, `vllm/models/deepseek_v41/common/vl_cudagraph.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #58586 - [Kernel][DSV4.1] Fuse MoE finalize into the TP all-reduce + mHC boundary

- Link: https://github.com/vllm-project/vllm/pull/58586
- Status/date: merged / 2026-09-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/__init__.py`, `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py`, `vllm/models/deepseek_v41/nvidia/ops/mhc.py`; associated commits `77871126f9b6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +1807/-309, 2312 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py` added +1134/-0 (1134 lines); hunks: -0,0 +1,1134; symbols: _group_leader_block_sum, _SharedOnlyPublishDeviceKernel, __init__, __call__, touching `_group_leader_block_sum, _SharedOnlyPublishDeviceKernel, __init__`; `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py` added +396/-0 (396 lines); hunks: -0,0 +1,396; symbols: _asm, load_global_u32x4, load_global_u32x2, load_global_bf16_as_f32, touching `_asm, load_global_u32x4, load_global_u32x2`; `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +73/-18 (91 lines); hunks: -9,10 +9,15; -24,9 +29,17; symbols: supports_mhc_overlap, supports_mhc_all_reduce, init_mhc_all_reduce, mhc_pre_delayed_overlap, touching `supports_mhc_overlap, supports_mhc_all_reduce, init_mhc_all_reduce`; `vllm/models/deepseek_v41/nvidia/model.py` modified +76/-4 (80 lines); hunks: -33,6 +33,14; -94,6 +102,7; symbols: __init__, defer_finalize, defers_finalize, forward_unfinalized, touching `__init__, defer_finalize, defers_finalize`.
- Code diff details:
  - `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py` added +1134/-0 (1134 lines); hunks: -0,0 +1,1134; symbols: _group_leader_block_sum, _SharedOnlyPublishDeviceKernel, __init__, __call__
  - `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py` added +396/-0 (396 lines); hunks: -0,0 +1,396; symbols: _asm, load_global_u32x4, load_global_u32x2, load_global_bf16_as_f32
  - `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +73/-18 (91 lines); hunks: -9,10 +9,15; -24,9 +29,17; symbols: supports_mhc_overlap, supports_mhc_all_reduce, init_mhc_all_reduce, mhc_pre_delayed_overlap
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +76/-4 (80 lines); hunks: -33,6 +33,14; -94,6 +102,7; symbols: __init__, defer_finalize, defers_finalize, forward_unfinalized
  - `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/__init__.py` added +8/-0 (8 lines); hunks: -0,0 +1,8
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py
@@ -0,0 +1,1134 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from FlashInfer's low-latency MNNVL CuTe DSL all-reduce
+# (flashinfer/comm/mnnvl_cutedsl/kernel_ll, flashinfer-ai/flashinfer@139af6f6).
+"""Lamport all-reduce fused with DSV4.1's mHC post, collapse and RMSNorm.
+FlashInfer's LL kernels, copied: a publish kernel (or its MoE finalize
diff -- vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py
@@ -0,0 +1,396 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from FlashInfer's flashinfer/comm/mnnvl_cutedsl/cute_dsl_primitives.py
+# and runtime.py (flashinfer-ai/flashinfer@139af6f6).
+"""PTX building blocks and launch helpers for the DSV4.1 LL all-reduce."""
+import cuda.bindings.driver as cuda
diff -- vllm/models/deepseek_v41/nvidia/ops/mhc.py
@@ -9,10 +9,15 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py` added +1134/-0; `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py` added +396/-0; `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +73/-18; `vllm/models/deepseek_v41/nvidia/model.py` modified +76/-4; `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/__init__.py` added +8/-0
- Risk and verification: The diff ships test coverage in `tests/distributed/test_custom_all_reduce.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58634 - [Perf][DSv4.1] Fuse small-batch WO-A with inverse RoPE and MXFP8 quant on SM100/SM103

- Link: https://github.com/vllm-project/vllm/pull/58634
- Status/date: merged / 2026-09-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v41/nvidia/flashmla.py`, `vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py`, `vllm/models/deepseek_v41/nvidia/ops/o_proj.py`; associated commits `a9eafde59cbd`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +759/-55, 899 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py` added +599/-0 (599 lines); hunks: -0,0 +1,599; symbols: _ue8m0_rp, _rope_fma, _scale, FusedWoAKernel, touching `_ue8m0_rp, _rope_fma, _scale`; `vllm/models/deepseek_v41/nvidia/ops/o_proj.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: _fused_wo_a_max_tokens, _can_fuse_wo_a, register_dsv41_o_proj_warmup, dsv41_o_proj, touching `_fused_wo_a_max_tokens, _can_fuse_wo_a, register_dsv41_o_proj_warmup`; `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +8/-34 (42 lines); hunks: -9,15 +9,16; -242,27 +243,14 @@ def get_padded_num_q_heads(cls, num_heads: int) -> int:; symbols: get_padded_num_q_heads, _o_proj, __init__, touching `get_padded_num_q_heads, _o_proj, __init__`; `vllm/models/deepseek_v41/nvidia/flashmla.py` modified +7/-19 (26 lines); hunks: -6,16 +6,17; -57,23 +58,10 @@ def __init__(self, *args, **kwargs) -> None:; symbols: __init__, _o_proj, get_padded_num_q_heads, touching `__init__, _o_proj, get_padded_num_q_heads`.
- Code diff details:
  - `vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py` added +599/-0 (599 lines); hunks: -0,0 +1,599; symbols: _ue8m0_rp, _rope_fma, _scale, FusedWoAKernel
  - `vllm/models/deepseek_v41/nvidia/ops/o_proj.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: _fused_wo_a_max_tokens, _can_fuse_wo_a, register_dsv41_o_proj_warmup, dsv41_o_proj
  - `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +8/-34 (42 lines); hunks: -9,15 +9,16; -242,27 +243,14 @@ def get_padded_num_q_heads(cls, num_heads: int) -> int:; symbols: get_padded_num_q_heads, _o_proj, __init__
  - `vllm/models/deepseek_v41/nvidia/flashmla.py` modified +7/-19 (26 lines); hunks: -6,16 +6,17; -57,23 +58,10 @@ def __init__(self, *args, **kwargs) -> None:; symbols: __init__, _o_proj, get_padded_num_q_heads
  - `vllm/models/deepseek_v41/attention.py` modified +2/-2 (4 lines); hunks: -697,7 +697,7 @@ def bind_gemm_rs(self) -> None:; -707,7 +707,7 @@ def _wo_b_proj(self, z: torch.Tensor) -> torch.Tensor:; symbols: bind_gemm_rs, _wo_b_proj
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py
@@ -0,0 +1,599 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DSV4.1 small-batch MXFP8 WO-A with inverse RoPE and quantization."""
+from dataclasses import dataclass
+from typing import Any
+import cutlass
diff -- vllm/models/deepseek_v41/nvidia/ops/o_proj.py
@@ -0,0 +1,95 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DSV4.1 output projection with a small-batch SM100/SM103 fusion."""
+import torch
+from torch import nn
+from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
diff -- vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py
@@ -9,15 +9,16 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py` added +599/-0; `vllm/models/deepseek_v41/nvidia/ops/o_proj.py` added +95/-0; `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +8/-34; `vllm/models/deepseek_v41/nvidia/flashmla.py` modified +7/-19; `vllm/models/deepseek_v41/attention.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_inv_rope_fp8_quant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57407 - [ROCm][Perf] Enable layer-aware CSA2 multi-stream overlap for DeepSeek-V4.1-Flash

- Link: https://github.com/vllm-project/vllm/pull/57407
- Status/date: merged / 2026-09-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/models/deepseek_v41/attention.py`; associated commits `44af287ebe38`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +496/-20, 592 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/amd/rocm.py` modified +189/-0 (189 lines); hunks: -29,6 +29,7; -529,6 +530,9 @@ def __init__(self, *args, **kwargs):; symbols: __init__, get_padded_num_q_heads, _fused_wqa_wkv_gemm, _run_parallel_input_projections, touching `__init__, get_padded_num_q_heads, _fused_wqa_wkv_gemm`; `vllm/models/deepseek_v41/attention.py` modified +32/-18 (50 lines); hunks: -478,7 +478,6 @@ def __init__(; -1381,6 +1380,37 @@ def _produce_k(; symbols: __init__, _produce_k, forward_q, forward, touching `__init__, _produce_k, forward_q`.
- Code diff details:
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +189/-0 (189 lines); hunks: -29,6 +29,7; -529,6 +530,9 @@ def __init__(self, *args, **kwargs):; symbols: __init__, get_padded_num_q_heads, _fused_wqa_wkv_gemm, _run_parallel_input_projections
  - `vllm/models/deepseek_v41/attention.py` modified +32/-18 (50 lines); hunks: -478,7 +478,6 @@ def __init__(; -1381,6 +1380,37 @@ def _produce_k(; symbols: __init__, _produce_k, forward_q, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -29,6 +29,7 @@
+from vllm.utils.multi_stream_utils import execute_in_parallel
@@ -529,6 +530,9 @@ def __init__(self, *args, **kwargs):
+        if self.compressor is None and self.indexer is None:
+            # Dense layers have nothing to overlap; keep the base serial path.
+            self.aux_stream_list = None
@@ -600,8 +604,193 @@ def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
diff -- vllm/models/deepseek_v41/attention.py
@@ -478,7 +478,6 @@ def __init__(
-        # Will be None on ROCm for now.
@@ -1381,6 +1380,37 @@ def _produce_k(
+    def forward_q(
+        self,
+        qr: torch.Tensor | QuantizedActivation,
+        qr_scale: torch.Tensor | None,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/amd/rocm.py` modified +189/-0; `vllm/models/deepseek_v41/attention.py` modified +32/-18
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57568 - [Bugfix][Spec Decode] Implement get_top_tokens() on the ROCm DeepSeek V4 MTP drafter

- Link: https://github.com/vllm-project/vllm/pull/57568
- Status/date: merged / 2026-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/mtp.py`; associated commits `f6877f563c5b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +200/-6, 237 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/amd/mtp.py` modified +37/-6 (43 lines); hunks: -240,13 +240,11 @@ def forward(; -260,14 +258,38 @@ def compute_logits(; symbols: forward, compute_logits, _pre_head, touching `forward, compute_logits, _pre_head`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/mtp.py` modified +37/-6 (43 lines); hunks: -240,13 +240,11 @@ def forward(; -260,14 +258,38 @@ def compute_logits(; symbols: forward, compute_logits, _pre_head
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/mtp.py
@@ -240,13 +240,11 @@ def forward(
-    def compute_logits(
+    def _pre_head(
+        mtp_layer: DeepSeekV4MultiTokenPredictorLayer,
-        spec_step_idx: int = 0,
-        current_step_idx = spec_step_idx % self.num_mtp_layers
-        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/amd/mtp.py` modified +37/-6
- Risk and verification: The diff ships test coverage in `tests/v1/spec_decode/test_dsv4_mtp_local_argmax.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58983 - [ROCm][Refactor] Move DeepSeek-V4/V4.1 multi-stream overlap gate to ROCm platform

- Link: https://github.com/vllm-project/vllm/pull/58983
- Status/date: merged / 2026-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v41/amd/rocm.py`; associated commits `7ba3df63cbe2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +40/-37, 126 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/amd/rocm.py` modified +6/-19 (25 lines); hunks: -649,23 +649,6 @@ def __init__(self, *args, **kwargs):; -709,7 +692,9 @@ def forward(; symbols: __init__, _enable_multi_stream_overlap, _run_sequential_pipeline, forward, touching `__init__, _enable_multi_stream_overlap, _run_sequential_pipeline`; `vllm/models/deepseek_v41/amd/rocm.py` modified +3/-18 (21 lines); hunks: -615,23 +615,6 @@ def _run_parallel_input_projections(; -641,7 +624,9 @@ def forward(; symbols: _run_parallel_input_projections, _enable_multi_stream_overlap, forward, touching `_run_parallel_input_projections, _enable_multi_stream_overlap, forward`; `vllm/platforms/rocm.py` modified +21/-0 (21 lines); hunks: -1106,6 +1106,27 @@ def opaque_attention_op(cls) -> bool:; symbols: opaque_attention_op, is_navi, enable_multi_stream_overlap, get_static_graph_wrapper_cls, touching `opaque_attention_op, is_navi, enable_multi_stream_overlap`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +6/-19 (25 lines); hunks: -649,23 +649,6 @@ def __init__(self, *args, **kwargs):; -709,7 +692,9 @@ def forward(; symbols: __init__, _enable_multi_stream_overlap, _run_sequential_pipeline, forward
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +3/-18 (21 lines); hunks: -615,23 +615,6 @@ def _run_parallel_input_projections(; -641,7 +624,9 @@ def forward(; symbols: _run_parallel_input_projections, _enable_multi_stream_overlap, forward
  - `vllm/platforms/rocm.py` modified +21/-0 (21 lines); hunks: -1106,6 +1106,27 @@ def opaque_attention_op(cls) -> bool:; symbols: opaque_attention_op, is_navi, enable_multi_stream_overlap, get_static_graph_wrapper_cls
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -649,23 +649,6 @@ def __init__(self, *args, **kwargs):
-    def _enable_multi_stream_overlap(self) -> bool:
-        """ROCm multi-stream gates: streams and capture region.
-        Dict metadata marks piecewise cudagraph, whose eager breaks rebuild
-        the attention inputs on the owning stream. Forking side streams
-        there would rely on runtime HIP event sync, which is unreliable in
-        this overlap on ROCm (event waits can hang), so multi-stream only
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -615,23 +615,6 @@ def _run_parallel_input_projections(
-    def _enable_multi_stream_overlap(self) -> bool:
-        """ROCm multi-stream gates: streams and capture region.
-        Dict metadata marks piecewise cudagraph, whose eager breaks rebuild
-        the attention inputs on the owning stream. Forking side streams
-        there would rely on runtime HIP event sync, which is unreliable in
-        this overlap on ROCm (event waits can hang), so multi-stream only
diff -- vllm/platforms/rocm.py
@@ -1106,6 +1106,27 @@ def opaque_attention_op(cls) -> bool:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/amd/rocm.py` modified +6/-19; `vllm/models/deepseek_v41/amd/rocm.py` modified +3/-18; `vllm/platforms/rocm.py` modified +21/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/platforms/interface.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #58655 - [ROCm][DSv4.1][Perf] Run the delayed mHC seams through aiter's fused Triton kernel

- Link: https://github.com/vllm-project/vllm/pull/58655
- Status/date: merged / 2026-09-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/amd/model.py`; associated commits `0af34418e99b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +324/-16, 473 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/amd/model.py` modified +29/-4 (33 lines); hunks: -22,7 +22,11; -251,6 +255,10 @@ def __init__(; symbols: __init__, _hc_collapse, forward, touching `__init__, _hc_collapse, forward`.
- Code diff details:
  - `vllm/models/deepseek_v41/amd/model.py` modified +29/-4 (33 lines); hunks: -22,7 +22,11; -251,6 +255,10 @@ def __init__(; symbols: __init__, _hc_collapse, forward
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/amd/model.py
@@ -22,7 +22,11 @@
-from vllm.model_executor.layers.mhc import MHCPostOp, MHCPreDelayedOp
+from vllm.model_executor.layers.mhc import (
+    HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM,
+    MHCPostOp,
+    MHCPreDelayedOp,
+)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/amd/model.py` modified +29/-4
- Risk and verification: The diff ships test coverage in `tests/kernels/mhc/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58671 - [ROCm][DSv4.1] Paged MXFP4 sparse indexer on aiter's MQA-logits kernel

- Link: https://github.com/vllm-project/vllm/pull/58671
- Status/date: merged / 2026-09-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/models/deepseek_v41/attention.py`; associated commits `63b9931b7399`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +2276/-9, 2437 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/amd/rocm.py` modified +149/-1 (150 lines); hunks: -2,8 +2,9; -15,8 +16,14; symbols: build, _aiter_indexer_cache_ops, rocm_mxfp4_indexer_k_store, rocm_mxfp4_indexer_q_quant, touching `build, _aiter_indexer_cache_ops, rocm_mxfp4_indexer_k_store`; `vllm/models/deepseek_v41/attention.py` modified +34/-6 (40 lines); hunks: -35,6 +35,9; -261,6 +264,13 @@ def _uses_fp8_ds_mla_layout(self) -> bool:; symbols: _uses_fp8_ds_mla_layout, _indexer_cls, __init__, touching `_uses_fp8_ds_mla_layout, _indexer_cls, __init__`; `vllm/config/attention.py` modified +4/-1 (5 lines); hunks: -93,7 +93,10 @@ class AttentionConfig:; symbols: AttentionConfig, GPU, touching `AttentionConfig, GPU`.
- Code diff details:
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +149/-1 (150 lines); hunks: -2,8 +2,9; -15,8 +16,14; symbols: build, _aiter_indexer_cache_ops, rocm_mxfp4_indexer_k_store, rocm_mxfp4_indexer_q_quant
  - `vllm/models/deepseek_v41/attention.py` modified +34/-6 (40 lines); hunks: -35,6 +35,9; -261,6 +264,13 @@ def _uses_fp8_ds_mla_layout(self) -> bool:; symbols: _uses_fp8_ds_mla_layout, _indexer_cls, __init__
  - `vllm/config/attention.py` modified +4/-1 (5 lines); hunks: -93,7 +93,10 @@ class AttentionConfig:; symbols: AttentionConfig, GPU
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -2,8 +2,9 @@
+from collections.abc import Callable
-from typing import cast
+from typing import Any, cast
@@ -15,8 +16,14 @@
+from vllm.model_executor.layers.rocm_paged_mxfp4_indexer import (
+    RocmSparseAttnIndexer,
diff -- vllm/models/deepseek_v41/attention.py
@@ -35,6 +35,9 @@
+    from vllm.model_executor.layers.rocm_paged_mxfp4_indexer import (
+        RocmSparseMQAIndexer,
+    )
@@ -261,6 +264,13 @@ def _uses_fp8_ds_mla_layout(self) -> bool:
+    def _indexer_cls(
+        self, k_cache: "DeepseekV4IndexerCache | None"
diff -- vllm/config/attention.py
@@ -93,7 +93,10 @@ class AttentionConfig:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/amd/rocm.py` modified +149/-1; `vllm/models/deepseek_v41/attention.py` modified +34/-6; `vllm/config/attention.py` modified +4/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py`, `tests/v1/attention/test_rocm_paged_mxfp4_indexer_plan.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #59119 - [DSv4.1] Avoid runtime recompiles of _ring_slot_mapping_kernel

- Link: https://github.com/vllm-project/vllm/pull/59119
- Status/date: merged / 2026-09-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/compressor.py`; associated commits `30f5c01d4ebb`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/compressor.py` modified +1/-1 (2 lines); hunks: -57,7 +57,7 @@ class CompressorMetadata:; symbols: CompressorMetadata, _ring_slot_mapping_kernel, touching `CompressorMetadata, _ring_slot_mapping_kernel`.
- Code diff details:
  - `vllm/models/deepseek_v41/compressor.py` modified +1/-1 (2 lines); hunks: -57,7 +57,7 @@ class CompressorMetadata:; symbols: CompressorMetadata, _ring_slot_mapping_kernel
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/compressor.py
@@ -57,7 +57,7 @@ class CompressorMetadata:
-@triton.jit
+@triton.jit(do_not_specialize=["block_table_stride", "num_actual_tokens", "num_tokens"])
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/compressor.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v41/compressor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #58132 - [Model] Decoder-side SWA bounded replay for DeepSeek-V4.1

- Link: https://github.com/vllm-project/vllm/pull/58132
- Status/date: merged / 2026-09-29
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/test_deepseek_v41_decoder_replay_layers.py`, `tests/models/test_deepseek_v41_replay_batch.py`, `vllm/models/deepseek_v41/decoder_replay_layers.py`, `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/model_state.py` and 6 files; associated commits `4b2e1cfa7b80`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +1001/-60, 1223 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/nvidia/model_state.py` modified +321/-2 (323 lines); hunks: -1,17 +1,24; -81,6 +88,61 @@ def _pad_replayed_slots_kernel(; symbols: _pad_replayed_slots_kernel, _gather_replay_batch_kernel, ReplayAttnMetadata, DeepseekV41ModelState, touching `_pad_replayed_slots_kernel, _gather_replay_batch_kernel, ReplayAttnMetadata`; `vllm/models/deepseek_v41/nvidia/model.py` modified +222/-40 (262 lines); hunks: -83,6 +83,7; -741,6 +742,35 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, forward, _mega_gate_metadata, _run_layers, touching `__init__, forward, _mega_gate_metadata`; `tests/models/test_deepseek_v41_replay_batch.py` added +259/-0 (259 lines); hunks: -0,0 +1,259; symbols: state, prepare_attn, _input_batch, _slot_mappings, touching `state, prepare_attn, _input_batch`; `tests/models/test_deepseek_v41_decoder_replay_layers.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: _context, _set_replay_batch, _states, test_run_gathers_states_and_realigns_shared_indexer_buffers, touching `_context, _set_replay_batch, _states`.
- Code diff details:
  - `vllm/models/deepseek_v41/nvidia/model_state.py` modified +321/-2 (323 lines); hunks: -1,17 +1,24; -81,6 +88,61 @@ def _pad_replayed_slots_kernel(; symbols: _pad_replayed_slots_kernel, _gather_replay_batch_kernel, ReplayAttnMetadata, DeepseekV41ModelState
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +222/-40 (262 lines); hunks: -83,6 +83,7; -741,6 +742,35 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, forward, _mega_gate_metadata, _run_layers
  - `tests/models/test_deepseek_v41_replay_batch.py` added +259/-0 (259 lines); hunks: -0,0 +1,259; symbols: state, prepare_attn, _input_batch, _slot_mappings
  - `tests/models/test_deepseek_v41_decoder_replay_layers.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: _context, _set_replay_batch, _states, test_run_gathers_states_and_realigns_shared_indexer_buffers
  - `vllm/models/deepseek_v41/decoder_replay_layers.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: DecoderReplayLayers, __init__, __call__
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/nvidia/model_state.py
@@ -1,17 +1,24 @@
+from dataclasses import replace
+import torch.distributed as dist
+from vllm.distributed.parallel_state import get_dp_group
+from vllm.forward_context import DPMetadata, create_forward_context
+from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
+from vllm.v1.worker.dp_utils import should_skip_dp_coordination
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -83,6 +83,7 @@
+from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
@@ -741,6 +742,35 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        # Decoder-side SWA bounded replay: in eager prefill steps the layers past
+        # the last KV source run on each request's trailing window only
+        # (decoder_replay_layers.py).
+        self.decoder_replay_layers: DecoderReplayLayers | None = None
diff -- tests/models/test_deepseek_v41_replay_batch.py
@@ -0,0 +1,259 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/nvidia/model_state.py` modified +321/-2; `vllm/models/deepseek_v41/nvidia/model.py` modified +222/-40; `vllm/models/deepseek_v41/decoder_replay_layers.py` added +67/-0; `vllm/models/deepseek_v41/nvidia/vl_model.py` modified +5/-0
  - tests: `tests/models/test_deepseek_v41_replay_batch.py` added +259/-0; `tests/models/test_deepseek_v41_decoder_replay_layers.py` added +104/-0
- Risk and verification: The diff ships test coverage in `tests/distributed/test_engram_dp_shard.py`, `tests/models/test_deepseek_v41_decoder_replay_layers.py`, `tests/models/test_deepseek_v41_replay_batch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57898 - [Bugfix] Profile maximum DeepSeek V4.1 vision features

- Link: https://github.com/vllm-project/vllm/pull/57898
- Status/date: merged / 2026-09-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/common/mm_preprocess.py`; associated commits `ac7f3e11ea88`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +21/-8, 37 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/common/mm_preprocess.py` modified +21/-8 (29 lines); hunks: -275,15 +275,28 @@ def get_image_size_with_most_features(self) -> ImageSize:; symbols: get_image_size_with_most_features, DeepseekV4VLDummyInputsBuilder, touching `get_image_size_with_most_features, DeepseekV4VLDummyInputsBuilder`.
- Code diff details:
  - `vllm/models/deepseek_v41/common/mm_preprocess.py` modified +21/-8 (29 lines); hunks: -275,15 +275,28 @@ def get_image_size_with_most_features(self) -> ImageSize:; symbols: get_image_size_with_most_features, DeepseekV4VLDummyInputsBuilder
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/common/mm_preprocess.py
@@ -275,15 +275,28 @@ def get_image_size_with_most_features(self) -> ImageSize:
-        # A square maximizes the ViT patch count (area) within the token
-        # budget; solve the budget-derived size directly to keep the dummy
-        # image small.
-        side = budget * patch_size * downsample_ratio
-        best_h, best_w = solve_resize_ratio(
-            side, side, patch_size, downsample_ratio, budget
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/common/mm_preprocess.py` modified +21/-8
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v41/common/mm_preprocess.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #58405 - [ROCm][DSv4][Perf] Use the shared prefill chunk plan in the ROCm sparse prefill

- Link: https://github.com/vllm-project/vllm/pull/58405
- Status/date: merged / 2026-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/amd/rocm.py`; associated commits `fcd317cf33b5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +11/-16, 63 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/amd/rocm.py` modified +11/-16 (27 lines); hunks: -1426,7 +1426,6 @@ def _forward_prefill(; -1458,26 +1457,22 @@ def _forward_prefill(; symbols: _forward_prefill, touching `_forward_prefill`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +11/-16 (27 lines); hunks: -1426,7 +1426,6 @@ def _forward_prefill(; -1458,26 +1457,22 @@ def _forward_prefill(; symbols: _forward_prefill
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -1426,7 +1426,6 @@ def _forward_prefill(
-        num_prefills = swa_metadata.num_prefills
@@ -1458,26 +1457,22 @@ def _forward_prefill(
-            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
-            N = 0
-        M = N + self.window_size + self.max_num_batched_tokens
-        num_chunks = (num_prefills + self.PREFILL_CHUNK_SIZE - 1) // (
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/amd/rocm.py` modified +11/-16
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/amd/rocm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #58539 - [ROCm][DSv4.1][Perf] Use the shared prefill chunk plan in the ROCm sparse prefill

- Link: https://github.com/vllm-project/vllm/pull/58539
- Status/date: merged / 2026-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/amd/rocm.py`; associated commits `8ee40690331c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +11/-10, 58 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/amd/rocm.py` modified +11/-10 (21 lines); hunks: -1379,7 +1379,6 @@ def _forward_prefill(; -1408,21 +1407,23 @@ def _forward_prefill(; symbols: _forward_prefill, touching `_forward_prefill`.
- Code diff details:
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +11/-10 (21 lines); hunks: -1379,7 +1379,6 @@ def _forward_prefill(; -1408,21 +1407,23 @@ def _forward_prefill(; symbols: _forward_prefill
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -1379,7 +1379,6 @@ def _forward_prefill(
-        num_prefills = swa_metadata.num_prefills
@@ -1408,21 +1407,23 @@ def _forward_prefill(
-        num_chunks = (num_prefills + self.PREFILL_CHUNK_SIZE - 1) // (
-            self.PREFILL_CHUNK_SIZE
+        chunk_plan = swa_metadata.get_prefill_chunk_plan(
+            compress_ratio=self.compress_ratio,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/amd/rocm.py` modified +11/-10
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v41/amd/rocm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #59373 - [BugFix][Multimodal] Pick worst-case DeepSeek-V4 VL dummy image size (#59271)

- Link: https://github.com/vllm-project/vllm/pull/59373
- Status/date: merged / 2026-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_deepseek_v4_vl.py`, `vllm/models/deepseek_v4/common/mm_preprocess.py`; associated commits `17e9295dd565`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +80/-8, 97 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/multimodal/processing/test_deepseek_v4_vl.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: _ConfigCtx, __init__, get_hf_config, _feature_counts, touching `_ConfigCtx, __init__, get_hf_config`; `vllm/models/deepseek_v4/common/mm_preprocess.py` modified +27/-8 (35 lines); hunks: -354,15 +354,34 @@ def get_image_size_with_most_features(self) -> ImageSize:; symbols: get_image_size_with_most_features, DeepseekV4VLDummyInputsBuilder, touching `get_image_size_with_most_features, DeepseekV4VLDummyInputsBuilder`.
- Code diff details:
  - `tests/models/multimodal/processing/test_deepseek_v4_vl.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: _ConfigCtx, __init__, get_hf_config, _feature_counts
  - `vllm/models/deepseek_v4/common/mm_preprocess.py` modified +27/-8 (35 lines); hunks: -354,15 +354,34 @@ def get_image_size_with_most_features(self) -> ImageSize:; symbols: get_image_size_with_most_features, DeepseekV4VLDummyInputsBuilder
- Key code excerpts:

```diff
diff -- tests/models/multimodal/processing/test_deepseek_v4_vl.py
@@ -0,0 +1,53 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DeepSeek-V4 VL processing helpers."""
+from PIL import Image
+from vllm.models.deepseek_v4.common.mm_preprocess import (
+    DeepseekV4VLProcessingInfo,
diff -- vllm/models/deepseek_v4/common/mm_preprocess.py
@@ -354,15 +354,34 @@ def get_image_size_with_most_features(self) -> ImageSize:
-        # A square maximizes the ViT patch count (area) within the token
-        # budget; solve the budget-derived size directly to keep the dummy
-        # image small.
+        max_wh_ratio = hf_config.vision_max_wh_ratio
+        # Every aligner row costs an extra IMAGE_NEW_LINE token, so height is
+        # more expensive than width and a square is *not* optimal: search the
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/multimodal/processing/test_deepseek_v4_vl.py` added +53/-0
  - runtime: `vllm/models/deepseek_v4/common/mm_preprocess.py` modified +27/-8
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_deepseek_v4_vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #59327 - [Perf][DSv4.1] Faster Engram host lookups: sorted rows, inline big lookups

- Link: https://github.com/vllm-project/vllm/pull/59327
- Status/date: merged / 2026-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/amd/model.py`, `vllm/models/deepseek_v41/common/engram.py`, `vllm/models/deepseek_v41/nvidia/model.py`; associated commits `4c2d277643e2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +547/-556, 1396 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/common/engram.py` modified +490/-22 (512 lines); hunks: -34,30 +34,60; -583,12 +613,14 @@ def _engram_head_shard_weight_loader(; symbols: _engram_lookup_thresholds, _is_prime, _engram_head_shard_weight_loader, _engram_lookup_kernel, touching `_engram_lookup_thresholds, _is_prime, _engram_head_shard_weight_loader`; `vllm/models/deepseek_v41/nvidia/engram.py` removed +0/-497 (497 lines); hunks: -1,497 +0,0; symbols: _allocate_huge_page_storage, engram_head_shard_rank, engram_gathered_num_tokens, gather_engram_hashes, touching `_allocate_huge_page_storage, engram_head_shard_rank, engram_gathered_num_tokens`; `vllm/models/deepseek_v41/nvidia/model.py` modified +7/-2 (9 lines); hunks: -98,9 +98,14; `vllm/models/deepseek_v41/amd/model.py` modified +1/-5 (6 lines); hunks: -65,13 +65,9.
- Code diff details:
  - `vllm/models/deepseek_v41/common/engram.py` modified +490/-22 (512 lines); hunks: -34,30 +34,60; -583,12 +613,14 @@ def _engram_head_shard_weight_loader(; symbols: _engram_lookup_thresholds, _is_prime, _engram_head_shard_weight_loader, _engram_lookup_kernel
  - `vllm/models/deepseek_v41/nvidia/engram.py` removed +0/-497 (497 lines); hunks: -1,497 +0,0; symbols: _allocate_huge_page_storage, engram_head_shard_rank, engram_gathered_num_tokens, gather_engram_hashes
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +7/-2 (9 lines); hunks: -98,9 +98,14
  - `vllm/models/deepseek_v41/amd/model.py` modified +1/-5 (6 lines); hunks: -65,13 +65,9
  - `tests/kernels/test_engram.py` modified +25/-25 (50 lines); hunks: -9,13 +9,10; -111,7 +108,7 @@ def test_fused_engram_post_wkv_matches_reference(; symbols: test_fused_engram_post_wkv_matches_reference, test_engram_rejects_empty_head_shards, test_engram_head_shards_reconstruct_checkpoint
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/common/engram.py
@@ -34,30 +34,60 @@
+import ctypes
+import mmap
+import os
+import tempfile
+from contextlib import ExitStack
+from vllm.compilation.breakable_cudagraph import eager_break_during_capture
diff -- vllm/models/deepseek_v41/nvidia/engram.py
@@ -1,497 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-"""NVIDIA Engram DP sharding, shared host storage, and asynchronous prefetch."""
-import ctypes
-import mmap
-import os
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -98,9 +98,14 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/common/engram.py` modified +490/-22; `vllm/models/deepseek_v41/nvidia/engram.py` removed +0/-497; `vllm/models/deepseek_v41/nvidia/model.py` modified +7/-2; `vllm/models/deepseek_v41/amd/model.py` modified +1/-5
  - tests: `tests/kernels/test_engram.py` modified +25/-25
- Risk and verification: The diff ships test coverage in `tests/distributed/test_engram_dp_shard.py`, `tests/kernels/test_engram.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #59159 - [Bugfix][XPU] Make DeepSeek V4 FP8 sparse decode graph-capturable

- Link: https://github.com/vllm-project/vllm/pull/59159
- Status/date: merged / 2026-10-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py`; associated commits `155d23cb008f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +36/-89, 155 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` modified +36/-89 (125 lines); hunks: -166,8 +166,7 @@ def xpu_sparse_decode_fp8(; -176,109 +175,57 @@ def xpu_sparse_decode_fp8(; symbols: xpu_sparse_decode_fp8, touching `xpu_sparse_decode_fp8`.
- Code diff details:
  - `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` modified +36/-89 (125 lines); hunks: -166,8 +166,7 @@ def xpu_sparse_decode_fp8(; -176,109 +175,57 @@ def xpu_sparse_decode_fp8(; symbols: xpu_sparse_decode_fp8
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py
@@ -166,8 +166,7 @@ def xpu_sparse_decode_fp8(
-    # Determine max topk and swa widths
-    if not swa_only and topk_indices is not None:
+    if not swa_only and topk_indices is not None and kv_cache is not None:
@@ -176,109 +175,57 @@ def xpu_sparse_decode_fp8(
+    use_topk = topk_idx_2d is not None
-    # Allocate flat workspace: [num_tokens * K_total, 512] bf16
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` modified +36/-89
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #58560 - [Bugfix][DSv4.1] Keep the compressor ring out of the null block

- Link: https://github.com/vllm-project/vllm/pull/58560
- Status/date: merged / 2026-10-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v41/compressor.py`; associated commits `49e0f4797853`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +20/-14, 67 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v41/compressor.py` modified +5/-1 (6 lines); hunks: -75,8 +75,12 @@ def _ring_slot_mapping_kernel(; symbols: _ring_slot_mapping_kernel, touching `_ring_slot_mapping_kernel`.
- Code diff details:
  - `vllm/models/deepseek_v41/compressor.py` modified +5/-1 (6 lines); hunks: -75,8 +75,12 @@ def _ring_slot_mapping_kernel(; symbols: _ring_slot_mapping_kernel
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v41/compressor.py
@@ -75,8 +75,12 @@ def _ring_slot_mapping_kernel(
+    # Block 0 is the null block. A request without a ring block (dummy or padding
+    # rows, which the runners fill with the null block) must not write.
-        slot_mapping_ptr + offsets, tl.where(valid, slot, -1), mask=offsets < num_tokens
+        slot_mapping_ptr + offsets,
+        tl.where(valid & (block != 0), slot, -1),
+        mask=offsets < num_tokens,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v41/compressor.py` modified +5/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
