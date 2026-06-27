# vllm DeepSeek V4 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml` | [#42111](https://github.com/vllm-project/vllm/pull/42111) |
| `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#42353](https://github.com/vllm-project/vllm/pull/42353), [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#45681](https://github.com/vllm-project/vllm/pull/45681) |
| `tests/models/test_deepseek_v4_mega_moe.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#43004](https://github.com/vllm-project/vllm/pull/43004), [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43632](https://github.com/vllm-project/vllm/pull/43632) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_1.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_2.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_3.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_4.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_1.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_2.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_3.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_4.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/test_deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#40982](https://github.com/vllm-project/vllm/pull/40982) |
| `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py` | [#43477](https://github.com/vllm-project/vllm/pull/43477) |
| `vllm/models/deepseek_v4/__init__.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953), [#43004](https://github.com/vllm-project/vllm/pull/43004), [#43077](https://github.com/vllm-project/vllm/pull/43077) |
| `vllm/models/deepseek_v4/amd/__init__.py` | [#43004](https://github.com/vllm-project/vllm/pull/43004) |
| `vllm/models/deepseek_v4/amd/model.py` | [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43385](https://github.com/vllm-project/vllm/pull/43385), [#43629](https://github.com/vllm-project/vllm/pull/43629), [#43679](https://github.com/vllm-project/vllm/pull/43679), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#44246](https://github.com/vllm-project/vllm/pull/44246), [#44262](https://github.com/vllm-project/vllm/pull/44262), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#45931](https://github.com/vllm-project/vllm/pull/45931) |
| `vllm/models/deepseek_v4/amd/mtp.py` | [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43385](https://github.com/vllm-project/vllm/pull/43385), [#43629](https://github.com/vllm-project/vllm/pull/43629), [#43679](https://github.com/vllm-project/vllm/pull/43679), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#44821](https://github.com/vllm-project/vllm/pull/44821), [#45931](https://github.com/vllm-project/vllm/pull/45931) |
| `vllm/models/deepseek_v4/amd/rocm.py` | [#43149](https://github.com/vllm-project/vllm/pull/43149), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43385](https://github.com/vllm-project/vllm/pull/43385), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#44699](https://github.com/vllm-project/vllm/pull/44699), [#45681](https://github.com/vllm-project/vllm/pull/45681) |
| `vllm/models/deepseek_v4/attention.py` | [#43039](https://github.com/vllm-project/vllm/pull/43039), [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43149](https://github.com/vllm-project/vllm/pull/43149), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43477](https://github.com/vllm-project/vllm/pull/43477), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#43891](https://github.com/vllm-project/vllm/pull/43891), [#44246](https://github.com/vllm-project/vllm/pull/44246), [#44561](https://github.com/vllm-project/vllm/pull/44561), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#45309](https://github.com/vllm-project/vllm/pull/45309), [#45972](https://github.com/vllm-project/vllm/pull/45972) |
| `vllm/models/deepseek_v4/common/__init__.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073) |
| `vllm/models/deepseek_v4/common/ops/__init__.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#43827](https://github.com/vllm-project/vllm/pull/43827) |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#45681](https://github.com/vllm-project/vllm/pull/45681) |
| `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710) |
| `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710) |
| `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` | [#42950](https://github.com/vllm-project/vllm/pull/42950), [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43477](https://github.com/vllm-project/vllm/pull/43477) |
| `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` | [#43746](https://github.com/vllm-project/vllm/pull/43746) |
| `vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073) |
| `vllm/models/deepseek_v4/common/ops/save_partial_states.py` | [#43710](https://github.com/vllm-project/vllm/pull/43710) |
| `vllm/models/deepseek_v4/common/rope.py` | [#44262](https://github.com/vllm-project/vllm/pull/44262) |
| `vllm/models/deepseek_v4/compressor.py` | [#42950](https://github.com/vllm-project/vllm/pull/42950), [#42953](https://github.com/vllm-project/vllm/pull/42953), [#43039](https://github.com/vllm-project/vllm/pull/43039), [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43477](https://github.com/vllm-project/vllm/pull/43477), [#43690](https://github.com/vllm-project/vllm/pull/43690), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43827](https://github.com/vllm-project/vllm/pull/43827) |
| `vllm/models/deepseek_v4/nvidia/__init__.py` | [#43004](https://github.com/vllm-project/vllm/pull/43004) |
| `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` | [#43477](https://github.com/vllm-project/vllm/pull/43477), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#44699](https://github.com/vllm-project/vllm/pull/44699), [#44892](https://github.com/vllm-project/vllm/pull/44892), [#45863](https://github.com/vllm-project/vllm/pull/45863) |
| `vllm/models/deepseek_v4/nvidia/flashmla.py` | [#43149](https://github.com/vllm-project/vllm/pull/43149), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#44569](https://github.com/vllm-project/vllm/pull/44569), [#44699](https://github.com/vllm-project/vllm/pull/44699), [#45061](https://github.com/vllm-project/vllm/pull/45061) |
| `vllm/models/deepseek_v4/nvidia/model.py` | [#42925](https://github.com/vllm-project/vllm/pull/42925), [#42950](https://github.com/vllm-project/vllm/pull/42950), [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43149](https://github.com/vllm-project/vllm/pull/43149), [#43162](https://github.com/vllm-project/vllm/pull/43162), [#43339](https://github.com/vllm-project/vllm/pull/43339), [#43477](https://github.com/vllm-project/vllm/pull/43477), [#43632](https://github.com/vllm-project/vllm/pull/43632), [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#43829](https://github.com/vllm-project/vllm/pull/43829), [#43891](https://github.com/vllm-project/vllm/pull/43891), ... (19 total) |
| `vllm/models/deepseek_v4/nvidia/mtp.py` | [#43077](https://github.com/vllm-project/vllm/pull/43077), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#43829](https://github.com/vllm-project/vllm/pull/43829), [#43905](https://github.com/vllm-project/vllm/pull/43905), [#44821](https://github.com/vllm-project/vllm/pull/44821) |
| `vllm/models/deepseek_v4/nvidia/ops/__init__.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073), [#43710](https://github.com/vllm-project/vllm/pull/43710) |
| `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073) |
| `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` | [#43073](https://github.com/vllm-project/vllm/pull/43073) |
| `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` | [#44569](https://github.com/vllm-project/vllm/pull/44569), [#45681](https://github.com/vllm-project/vllm/pull/45681) |
| `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` | [#43632](https://github.com/vllm-project/vllm/pull/43632) |
| `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` | [#43710](https://github.com/vllm-project/vllm/pull/43710), [#43827](https://github.com/vllm-project/vllm/pull/43827), [#44161](https://github.com/vllm-project/vllm/pull/44161), [#44236](https://github.com/vllm-project/vllm/pull/44236) |
| `vllm/models/deepseek_v4/quant_config.py` | [#42209](https://github.com/vllm-project/vllm/pull/42209), [#43004](https://github.com/vllm-project/vllm/pull/43004), [#44914](https://github.com/vllm-project/vllm/pull/44914) |
| `vllm/models/deepseek_v4/sparse_mla.py` | [#43477](https://github.com/vllm-project/vllm/pull/43477), [#44699](https://github.com/vllm-project/vllm/pull/44699), [#44892](https://github.com/vllm-project/vllm/pull/44892) |
| `vllm/models/deepseek_v4/xpu/__init__.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953) |
| `vllm/models/deepseek_v4/xpu/model.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953), [#44144](https://github.com/vllm-project/vllm/pull/44144) |
| `vllm/models/deepseek_v4/xpu/mtp.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953), [#43746](https://github.com/vllm-project/vllm/pull/43746), [#44821](https://github.com/vllm-project/vllm/pull/44821), [#45240](https://github.com/vllm-project/vllm/pull/45240) |
| `vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953) |
| `vllm/models/deepseek_v4/xpu/xpu_sparse.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953) |
| `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` | [#42953](https://github.com/vllm-project/vllm/pull/42953) |
| `vllm/renderers/deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/tokenizers/deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#40982](https://github.com/vllm-project/vllm/pull/40982) |
| `vllm/tokenizers/deepseek_v4_encoding.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/transformers_utils/configs/deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |

## PR 覆盖总览

- git 追溯 PR 数: 48
- 原文档显式引用补充 PR 数: 39
- 当前文档总 PR 数: 87
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
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

## 逐 PR diff 审计卡

### PR #40806 - [Bugfix] Fix the DSML token leakage in DSV4/3.2

- 链接: https://github.com/vllm-project/vllm/pull/40806
- 状态/时间: merged / 2026-04-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+76/-23，可读 patch 144 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix the DSML token leakage in DSV4/3.2」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Fix the DSML token leakage in DSV4/3.2」；主要实现面是 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0 (52 lines); hunks: -484,6 +484,58 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked, test_no_marker_leak_char_by_char，涉及 `test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked`；`vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23 (47 lines); hunks: -26,6 +26,7; -54,8 +55,8 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, extract_tool_calls, _reset_streaming_state, _extract_delta_tool_calls，涉及 `__init__, extract_tool_calls, _reset_streaming_state`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0 (52 lines); hunks: -484,6 +484,58 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked, test_no_marker_leak_char_by_char
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23 (47 lines); hunks: -26,6 +26,7; -54,8 +55,8 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, extract_tool_calls, _reset_streaming_state, _extract_delta_tool_calls
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_deepseekv32_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40860 - [Feat] DeepSeek V4 Rebased

- 链接: https://github.com/vllm-project/vllm/pull/40860
- 状态/时间: merged / 2026-04-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `tests/tokenizers_/fixtures/deepseek_v4/test_input_1.json`, `tests/tokenizers_/fixtures/deepseek_v4/test_input_2.json`, `tests/tokenizers_/fixtures/deepseek_v4/test_input_3.json` 等 16 个文件；关联提交 `4d51588e2381`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 150 个文件，+16313/-717，可读 patch 20516 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feat] DeepSeek V4 Rebased」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`, `tests/tokenizers_/test_deepseek_v4.py`；技术摘要: 覆盖「[Feat] DeepSeek V4 Rebased」；主要实现面是 `vllm/model_executor/models/deepseek_v4.py`, `vllm/tokenizers/deepseek_v4_encoding.py`, `tests/tokenizers_/test_deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method，涉及 `DeepseekV4FP8Config, __init__, get_name`；`vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format，涉及 `to_json, tools_from_openai_format, tool_calls_from_openai_format`；`tests/tokenizers_/test_deepseek_v4.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: FakeHfTokenizer, get_added_vocab, encode, _tokenizer，涉及 `FakeHfTokenizer, get_added_vocab, encode`；`tests/models/test_deepseek_v4_mega_moe.py` added +184/-0 (184 lines); hunks: -0,0 +1,184; symbols: test_deepseek_v4_mega_moe_expert_mapping, test_deepseek_v4_mega_moe_ue8m0_uint8_to_float, test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact，涉及 `test_deepseek_v4_mega_moe_expert_mapping, test_deepseek_v4_mega_moe_ue8m0_uint8_to_float, test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method
  - `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `tests/tokenizers_/test_deepseek_v4.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: FakeHfTokenizer, get_added_vocab, encode, _tokenizer
  - `tests/models/test_deepseek_v4_mega_moe.py` added +184/-0 (184 lines); hunks: -0,0 +1,184; symbols: test_deepseek_v4_mega_moe_expert_mapping, test_deepseek_v4_mega_moe_ue8m0_uint8_to_float, test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact
  - `tests/tokenizers_/fixtures/deepseek_v4/test_input_3.json` added +159/-0 (159 lines); hunks: -0,0 +1,159
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0; `vllm/tokenizers/deepseek_v4.py` added +90/-0
  - tests: `tests/tokenizers_/test_deepseek_v4.py` added +224/-0; `tests/models/test_deepseek_v4_mega_moe.py` added +184/-0; `tests/tokenizers_/fixtures/deepseek_v4/test_input_3.json` added +159/-0; `tests/tokenizers_/fixtures/deepseek_v4/test_input_1.json` added +81/-0; `tests/tokenizers_/fixtures/deepseek_v4/test_output_3.txt` added +38/-0
- 验证与风险: diff 自带测试面 `tests/compile/fusions_e2e/conftest.py`, `tests/kernels/attention/test_deepgemm_attention.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/moe/test_deepgemm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40760 - [New Model] Support DeepseekV4

- 链接: https://github.com/vllm-project/vllm/pull/40760
- 状态/时间: closed / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 158 个文件，+16968/-760，可读 patch 21398 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[New Model] Support DeepseekV4」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py`；技术摘要: 覆盖「[New Model] Support DeepseekV4」；主要实现面是 `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method，涉及 `DeepseekV4FP8Config, __init__, get_name`；`vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0 (1062 lines); hunks: -0,0 +1,1062; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does，涉及 `DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes`；`vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format，涉及 `to_json, tools_from_openai_format, tool_calls_from_openai_format`；`vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor，涉及 `DeepSeekV4MultiTokenPredictorLayer, __init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method
  - `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0 (1062 lines); hunks: -0,0 +1,1062; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does
  - `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor
  - `vllm/model_executor/layers/deepseek_compressor.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: CompressorBackend, __init__, get_name, get_supported_kernel_block_sizes
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0; `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0; `vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0; `vllm/model_executor/layers/deepseek_compressor.py` added +436/-0; `vllm/model_executor/layers/mhc.py` added +436/-0
- 验证与风险: diff 自带测试面 `tests/kernels/attention/test_use_trtllm_attention.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/moe/test_deepgemm.py`, `tests/kernels/moe/test_ocp_mx_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40950 - [DSV4] Add silu clamp limit to shared expert

- 链接: https://github.com/vllm-project/vllm/pull/40950
- 状态/时间: merged / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+269/-29，可读 patch 466 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Add silu clamp limit to shared expert」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/activation.py`, `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`；技术摘要: 覆盖「[DSV4] Add silu clamp limit to shared expert」；主要实现面是 `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/activation.py`, `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v4.py` modified +58/-3 (61 lines); hunks: -17,6 +17,7; -34,7 +35,10; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4FP8Config，涉及 `DeepseekV4MLP, __init__, forward`；`vllm/model_executor/layers/activation.py` modified +40/-0 (40 lines); hunks: -151,6 +151,46 @@ def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward_xpu, SiluAndMulWithClamp, __init__, forward_native，涉及 `forward_xpu, SiluAndMulWithClamp, __init__`；`vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` modified +1/-1 (2 lines); hunks: -45,7 +45,7 @@ def _gelu_and_mul(; symbols: _gelu_and_mul，涉及 `_gelu_and_mul`；`csrc/activation_kernels.cu` modified +82/-25 (107 lines); hunks: -11,29 +11,74; -58,8 +103,9 @@ __global__ void act_and_mul_kernel(。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v4.py` modified +58/-3 (61 lines); hunks: -17,6 +17,7; -34,7 +35,10; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4FP8Config
  - `vllm/model_executor/layers/activation.py` modified +40/-0 (40 lines); hunks: -151,6 +151,46 @@ def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward_xpu, SiluAndMulWithClamp, __init__, forward_native
  - `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` modified +1/-1 (2 lines); hunks: -45,7 +45,7 @@ def _gelu_and_mul(; symbols: _gelu_and_mul
  - `csrc/activation_kernels.cu` modified +82/-25 (107 lines); hunks: -11,29 +11,74; -58,8 +103,9 @@ __global__ void act_and_mul_kernel(
  - `tests/kernels/core/test_activation.py` modified +80/-0 (80 lines); hunks: -16,6 +16,7; -116,6 +117,85 @@ def _get_rtol(output) -> float:; symbols: _get_rtol, test_silu_and_mul_with_clamp
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +58/-3; `vllm/model_executor/layers/activation.py` modified +40/-0; `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` modified +1/-1
  - other: `csrc/activation_kernels.cu` modified +82/-25; `csrc/torch_bindings.cpp` modified +6/-0; `csrc/ops.h` modified +2/-0
  - tests: `tests/kernels/core/test_activation.py` modified +80/-0
- 验证与风险: diff 自带测试面 `tests/kernels/core/test_activation.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41006 - [Model][DSV4] Support base model

- 链接: https://github.com/vllm-project/vllm/pull/41006
- 状态/时间: merged / 2026-04-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+111/-23，可读 patch 223 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][DSV4] Support base model」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`；技术摘要: 覆盖「[Model][DSV4] Support base model」；主要实现面是 `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v4.py` modified +93/-19 (112 lines); hunks: -10,7 +10,7; -65,6 +65,8; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4FP8Config，涉及 `DeepseekV4MLP, __init__, forward`；`vllm/model_executor/models/deepseek_v4_mtp.py` modified +18/-4 (22 lines); hunks: -48,9 +48,14; -326,6 +331,15 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx，涉及 `_find_mtp_layer_idx`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v4.py` modified +93/-19 (112 lines); hunks: -10,7 +10,7; -65,6 +65,8; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4FP8Config
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +18/-4 (22 lines); hunks: -48,9 +48,14; -326,6 +331,15 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +93/-19; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +18/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41061 - [DSV4] Enable Multi-stream for Pre-Attn GEMM

- 链接: https://github.com/vllm-project/vllm/pull/41061
- 状态/时间: merged / 2026-04-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+187/-57，可读 patch 439 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Enable Multi-stream for Pre-Attn GEMM」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_compressor.py`；技术摘要: 覆盖「[DSV4] Enable Multi-stream for Pre-Attn GEMM」；主要实现面是 `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_compressor.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +111/-38 (149 lines); hunks: -4,8 +4,9; -16,6 +17,7; symbols: DeepseekV4MLAModules, __init__, forward，涉及 `DeepseekV4MLAModules, __init__, forward`；`vllm/model_executor/models/deepseek_v4.py` modified +10/-12 (22 lines); hunks: -54,7 +54,6; -872,7 +871,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/model_executor/layers/deepseek_compressor.py` modified +2/-7 (9 lines); hunks: -14,7 +14,6; -271,16 +270,12 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/utils/multi_stream_utils.py` modified +64/-0 (64 lines); hunks: -56,3 +56,67 @@ def maybe_execute_in_parallel(; symbols: maybe_execute_in_parallel, execute_in_parallel，涉及 `maybe_execute_in_parallel, execute_in_parallel`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +111/-38 (149 lines); hunks: -4,8 +4,9; -16,6 +17,7; symbols: DeepseekV4MLAModules, __init__, forward
  - `vllm/model_executor/models/deepseek_v4.py` modified +10/-12 (22 lines); hunks: -54,7 +54,6; -872,7 +871,7 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/layers/deepseek_compressor.py` modified +2/-7 (9 lines); hunks: -14,7 +14,6; -271,16 +270,12 @@ def __init__(; symbols: __init__, forward
  - `vllm/utils/multi_stream_utils.py` modified +64/-0 (64 lines); hunks: -56,3 +56,67 @@ def maybe_execute_in_parallel(; symbols: maybe_execute_in_parallel, execute_in_parallel
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +111/-38; `vllm/model_executor/models/deepseek_v4.py` modified +10/-12; `vllm/model_executor/layers/deepseek_compressor.py` modified +2/-7; `vllm/utils/multi_stream_utils.py` modified +64/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/deepseek_compressor.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41171 - [DSV4] Align aux stream API with DeepseekV4DecoderLayer

- 链接: https://github.com/vllm-project/vllm/pull/41171
- 状态/时间: merged / 2026-04-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-5，可读 patch 51 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Align aux stream API with DeepseekV4DecoderLayer」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepseek_v4_mtp.py`；技术摘要: 覆盖「[DSV4] Align aux stream API with DeepseekV4DecoderLayer」；主要实现面是 `vllm/model_executor/models/deepseek_v4_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v4_mtp.py` modified +7/-5 (12 lines); hunks: -35,7 +35,6; -65,6 +64,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +7/-5 (12 lines); hunks: -35,7 +35,6; -65,6 +64,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v4_mtp.py` modified +7/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v4_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41090 - [Bugfix] Fix Deepseek V4 import error due to AOT compile cache loading

- 链接: https://github.com/vllm-project/vllm/pull/41090
- 状态/时间: merged / 2026-04-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-5，可读 patch 24 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Deepseek V4 import error due to AOT compile cache loading」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v4.py`；技术摘要: 覆盖「[Bugfix] Fix Deepseek V4 import error due to AOT compile cache loading」；主要实现面是 `vllm/model_executor/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v4.py` modified +5/-5 (10 lines); hunks: -1098,6 +1098,11 @@ def __init__(; -1170,11 +1175,6 @@ def hc_pre(; symbols: __init__, hc_pre，涉及 `__init__, hc_pre`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v4.py` modified +5/-5 (10 lines); hunks: -1098,6 +1098,11 @@ def __init__(; -1170,11 +1175,6 @@ def hc_pre(; symbols: __init__, hc_pre
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +5/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41135 - [Bugfix] fix inductor error for dpsk v4

- 链接: https://github.com/vllm-project/vllm/pull/41135
- 状态/时间: merged / 2026-04-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+106/-36，可读 patch 172 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] fix inductor error for dpsk v4」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`；技术摘要: 覆盖「[Bugfix] fix inductor error for dpsk v4」；主要实现面是 `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` modified +106/-36 (142 lines); hunks: -10,6 +10,7; -180,34 +181,74 @@ def fused_inv_rope_fp8_quant(; symbols: fused_inv_rope_fp8_quant, _fused_inv_rope_fp8_quant_kernel_impl, _fused_inv_rope_fp8_quant_kernel_fake，涉及 `fused_inv_rope_fp8_quant, _fused_inv_rope_fp8_quant_kernel_impl, _fused_inv_rope_fp8_quant_kernel_fake`。
- 代码 diff 细节:
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` modified +106/-36 (142 lines); hunks: -10,6 +10,7; -180,34 +181,74 @@ def fused_inv_rope_fp8_quant(; symbols: fused_inv_rope_fp8_quant, _fused_inv_rope_fp8_quant_kernel_impl, _fused_inv_rope_fp8_quant_kernel_fake
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` modified +106/-36
- 验证与风险: runtime 路径改动集中在 `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #40982 - [DSV4] Support `max` reasoning effort

- 链接: https://github.com/vllm-project/vllm/pull/40982
- 状态/时间: merged / 2026-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py`；关联提交 `33f36d42605a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+126/-6，可读 patch 204 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Support `max` reasoning effort」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py`；技术摘要: 覆盖「[DSV4] Support `max` reasoning effort」；主要实现面是 `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tokenizers_/test_deepseek_v4.py` modified +66/-1 (67 lines); hunks: -182,7 +182,7 @@ def test_deepseek_v4_renders_parsed_history_tool_arguments():; -195,6 +195,58 @@ def test_deepseek_v4_accepts_openai_reasoning_effort_values...; symbols: test_deepseek_v4_renders_parsed_history_tool_arguments, test_deepseek_v4_accepts_openai_reasoning_effort_values, test_deepseek_v4_none_reasoning_effort_disables_thinking, test_deepseek_v4_maps_compatible_thinking_reasoning_effort_values，涉及 `test_deepseek_v4_renders_parsed_history_tool_arguments, test_deepseek_v4_accepts_openai_reasoning_effort_values, test_deepseek_v4_none_reasoning_effort_disables_thinking`；`vllm/tokenizers/deepseek_v4.py` modified +8/-2 (10 lines); hunks: -40,10 +40,16 @@ def apply_chat_template(; symbols: apply_chat_template，涉及 `apply_chat_template`。
- 代码 diff 细节:
  - `tests/tokenizers_/test_deepseek_v4.py` modified +66/-1 (67 lines); hunks: -182,7 +182,7 @@ def test_deepseek_v4_renders_parsed_history_tool_arguments():; -195,6 +195,58 @@ def test_deepseek_v4_accepts_openai_reasoning_effort_values...; symbols: test_deepseek_v4_renders_parsed_history_tool_arguments, test_deepseek_v4_accepts_openai_reasoning_effort_values, test_deepseek_v4_none_reasoning_effort_disables_thinking, test_deepseek_v4_maps_compatible_thinking_reasoning_effort_values
  - `vllm/tokenizers/deepseek_v4.py` modified +8/-2 (10 lines); hunks: -40,10 +40,16 @@ def apply_chat_template(; symbols: apply_chat_template
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/tokenizers_/test_deepseek_v4.py` modified +66/-1
  - runtime: `vllm/tokenizers/deepseek_v4.py` modified +8/-2
- 验证与风险: diff 自带测试面 `tests/entrypoints/openai/chat_completion/test_chat.py`, `tests/entrypoints/openai/parser/test_harmony_utils.py`, `tests/tokenizers_/test_deepseek_v4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41148 - [Bugfix] Fix repeated DSv4 RoPE cache initialization

- 链接: https://github.com/vllm-project/vllm/pull/41148
- 状态/时间: merged / 2026-04-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+11/-3，可读 patch 42 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix repeated DSv4 RoPE cache initialization」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/models/deepseek_v4.py`；技术摘要: 覆盖「[Bugfix] Fix repeated DSv4 RoPE cache initialization」；主要实现面是 `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +11/-2 (13 lines); hunks: -45,6 +45,7 @@ def __init__(; -65,7 +66,13 @@ def __init__(; symbols: __init__, _compute_inv_freq, DeepseekV4ScalingRotaryEmbedding，涉及 `__init__, _compute_inv_freq, DeepseekV4ScalingRotaryEmbedding`；`vllm/model_executor/models/deepseek_v4.py` modified +0/-1 (1 lines); hunks: -1027,7 +1027,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +11/-2 (13 lines); hunks: -45,6 +45,7 @@ def __init__(; -65,7 +66,13 @@ def __init__(; symbols: __init__, _compute_inv_freq, DeepseekV4ScalingRotaryEmbedding
  - `vllm/model_executor/models/deepseek_v4.py` modified +0/-1 (1 lines); hunks: -1027,7 +1027,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +11/-2; `vllm/model_executor/models/deepseek_v4.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41015 - [DSv4] Use `cvt` PTX for FP32->FP4 conversion

- 链接: https://github.com/vllm-project/vllm/pull/41015
- 状态/时间: merged / 2026-04-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+344/-62，可读 patch 509 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSv4] Use `cvt` PTX for FP32->FP4 conversion」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`；技术摘要: 覆盖「[DSv4] Use `cvt` PTX for FP32->FP4 conversion」；主要实现面是 `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/kernels/test_compressor_kv_cache.py` modified +228/-4 (232 lines); hunks: -3,12 +3,11; -21,6 +20,12; symbols: _ue8m0_reference, test_deepseek_v4_quant_magnitude_range, _reference_kv_compress_norm_rope, test_fused_kv_insert_indexer，涉及 `_ue8m0_reference, test_deepseek_v4_quant_magnitude_range, _reference_kv_compress_norm_rope`；`tests/kernels/test_fused_indexer_q_rope_quant.py` modified +90/-17 (107 lines); hunks: -30,13 +30,64; -49,22 +100,33 @@ def _reference(; symbols: quantize_to_mxfp4, _reference, test_fused_indexer_q_rope_quant_matches_unfused，涉及 `quantize_to_mxfp4, _reference, test_fused_indexer_q_rope_quant_matches_unfused`；`vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +20/-35 (55 lines); hunks: -24,36 +24,22 @@ def _get_cos_sin(; -65,17 +51,16 @@ def _quantize_mxfp4_pair(x_lo, x_hi):; symbols: _get_cos_sin, _e2m1_nibble, _fp32x2_to_fp4x2, _quantize_mxfp4_pair，涉及 `_get_cos_sin, _e2m1_nibble, _fp32x2_to_fp4x2`；`vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` modified +6/-6 (12 lines); hunks: -21,7 +21,7; -566,18 +566,18 @@ def _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(; symbols: _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn，涉及 `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn`。
- 代码 diff 细节:
  - `tests/kernels/test_compressor_kv_cache.py` modified +228/-4 (232 lines); hunks: -3,12 +3,11; -21,6 +20,12; symbols: _ue8m0_reference, test_deepseek_v4_quant_magnitude_range, _reference_kv_compress_norm_rope, test_fused_kv_insert_indexer
  - `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +90/-17 (107 lines); hunks: -30,13 +30,64; -49,22 +100,33 @@ def _reference(; symbols: quantize_to_mxfp4, _reference, test_fused_indexer_q_rope_quant_matches_unfused
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +20/-35 (55 lines); hunks: -24,36 +24,22 @@ def _get_cos_sin(; -65,17 +51,16 @@ def _quantize_mxfp4_pair(x_lo, x_hi):; symbols: _get_cos_sin, _e2m1_nibble, _fp32x2_to_fp4x2, _quantize_mxfp4_pair
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` modified +6/-6 (12 lines); hunks: -21,7 +21,7; -566,18 +566,18 @@ def _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(; symbols: _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/kernels/test_compressor_kv_cache.py` modified +228/-4; `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +90/-17
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +20/-35; `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` modified +6/-6
- 验证与风险: diff 自带测试面 `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41374 - [DSV4] Avoid redundant dtype conversion.

- 链接: https://github.com/vllm-project/vllm/pull/41374
- 状态/时间: merged / 2026-04-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-6，可读 patch 38 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Avoid redundant dtype conversion.」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepseek_v4.py`；技术摘要: 覆盖「[DSV4] Avoid redundant dtype conversion.」；主要实现面是 `vllm/model_executor/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v4.py` modified +11/-6 (17 lines); hunks: -854,10 +854,9 @@ def _init_fused_moe_experts(; -1225,7 +1224,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: st...; symbols: _init_fused_moe_experts, forward, __init__，涉及 `_init_fused_moe_experts, forward, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v4.py` modified +11/-6 (17 lines); hunks: -854,10 +854,9 @@ def _init_fused_moe_experts(; -1225,7 +1224,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: st...; symbols: _init_fused_moe_experts, forward, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +11/-6
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41255 - [Perf] Intergrate Tile Kernels `head_compute_mix_kernel` for Deepseek-V4

- 链接: https://github.com/vllm-project/vllm/pull/41255
- 状态/时间: merged / 2026-05-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+153/-9，可读 patch 180 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf] Intergrate Tile Kernels `head_compute_mix_kernel` for Deepseek-V4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py`；技术摘要: 覆盖「[Perf] Intergrate Tile Kernels `head_compute_mix_kernel` for Deepseek-V4」；主要实现面是 `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/mhc.py` modified +134/-0 (134 lines); hunks: -448,3 +448,137 @@ def _mhc_post_fake(; symbols: _mhc_post_fake, hc_head_fuse_tilelang, _hc_head_fused_kernel，涉及 `_mhc_post_fake, hc_head_fuse_tilelang, _hc_head_fused_kernel`；`vllm/model_executor/models/deepseek_v4.py` modified +19/-9 (28 lines); hunks: -7,7 +7,6; -1456,14 +1455,25 @@ def hc_head(; symbols: hc_head, _make_deepseek_v4_weights_mapper，涉及 `hc_head, _make_deepseek_v4_weights_mapper`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/mhc.py` modified +134/-0 (134 lines); hunks: -448,3 +448,137 @@ def _mhc_post_fake(; symbols: _mhc_post_fake, hc_head_fuse_tilelang, _hc_head_fused_kernel
  - `vllm/model_executor/models/deepseek_v4.py` modified +19/-9 (28 lines); hunks: -7,7 +7,6; -1456,14 +1455,25 @@ def hc_head(; symbols: hc_head, _make_deepseek_v4_weights_mapper
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/mhc.py` modified +134/-0; `vllm/model_executor/models/deepseek_v4.py` modified +19/-9
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41443 - [DSV4] Add knob to enable pre-attn gemm

- 链接: https://github.com/vllm-project/vllm/pull/41443
- 状态/时间: merged / 2026-05-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+24/-3，可读 patch 82 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Add knob to enable pre-attn gemm」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/envs.py`, `vllm/utils/multi_stream_utils.py`；技术摘要: 覆盖「[DSV4] Add knob to enable pre-attn gemm」；主要实现面是 `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/envs.py`, `vllm/utils/multi_stream_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +3/-0 (3 lines); hunks: -13,6 +13,7; -385,6 +386,8 @@ def fused_wqa_wkv() -> torch.Tensor:; symbols: fused_wqa_wkv，涉及 `fused_wqa_wkv`；`vllm/envs.py` modified +12/-0 (12 lines); hunks: -245,6 +245,7; -1669,6 +1670,17 @@ def _get_or_set_default() -> str:; symbols: _get_or_set_default，涉及 `_get_or_set_default`；`vllm/utils/multi_stream_utils.py` modified +9/-3 (12 lines); hunks: -64,6 +64,7 @@ def execute_in_parallel(; -74,8 +75,9 @@ def execute_in_parallel(; symbols: execute_in_parallel，涉及 `execute_in_parallel`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +3/-0 (3 lines); hunks: -13,6 +13,7; -385,6 +386,8 @@ def fused_wqa_wkv() -> torch.Tensor:; symbols: fused_wqa_wkv
  - `vllm/envs.py` modified +12/-0 (12 lines); hunks: -245,6 +245,7; -1669,6 +1670,17 @@ def _get_or_set_default() -> str:; symbols: _get_or_set_default
  - `vllm/utils/multi_stream_utils.py` modified +9/-3 (12 lines); hunks: -64,6 +64,7 @@ def execute_in_parallel(; -74,8 +75,9 @@ def execute_in_parallel(; symbols: execute_in_parallel
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +3/-0; `vllm/envs.py` modified +12/-0; `vllm/utils/multi_stream_utils.py` modified +9/-3
- 验证与风险: runtime 路径改动集中在 `vllm/envs.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/utils/multi_stream_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41522 - [DSV4] Guard megamoe flag with Pure TP

- 链接: https://github.com/vllm-project/vllm/pull/41522
- 状态/时间: merged / 2026-05-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+16/-10，可读 patch 42 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Guard megamoe flag with Pure TP」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepseek_v4.py`；技术摘要: 覆盖「[DSV4] Guard megamoe flag with Pure TP」；主要实现面是 `vllm/model_executor/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v4.py` modified +16/-10 (26 lines); hunks: -715,12 +715,15 @@ def __init__(; -1223,12 +1226,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v4.py` modified +16/-10 (26 lines); hunks: -715,12 +715,15 @@ def __init__(; -1223,12 +1226,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +16/-10
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #40871 - [New Model][ROCm] Add AMD support for DeepSeek V4

- 链接: https://github.com/vllm-project/vllm/pull/40871
- 状态/时间: merged / 2026-05-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 22 个文件，+939/-134，可读 patch 1657 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[New Model][ROCm] Add AMD support for DeepSeek V4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`；技术摘要: 覆盖「[New Model][ROCm] Add AMD support for DeepSeek V4」；主要实现面是 `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/mhc.py` modified +105/-2 (107 lines); hunks: -234,6 +234,39 @@ def mhc_pre(; -414,6 +447,14 @@ def mhc_post(; symbols: mhc_pre, mhc_post, hc_head_fuse_tilelang, _hc_head_fused_reference，涉及 `mhc_pre, mhc_post, hc_head_fuse_tilelang`；`vllm/model_executor/layers/deepseek_v4_attention.py` modified +73/-19 (92 lines); hunks: -28,6 +28,11; -53,6 +58,7; symbols: __init__, forward, attn_gemm_parallel_execute，涉及 `__init__, forward, attn_gemm_parallel_execute`；`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +79/-2 (81 lines); hunks: -18,6 +18,7; -64,6 +65,8 @@ class Mxfp4MoeBackend(Enum):; symbols: Mxfp4MoeBackend, _get_priority_backends, _return_or_raise, convert_weight_to_mxfp4_moe_kernel_format，涉及 `Mxfp4MoeBackend, _get_priority_backends, _return_or_raise`；`vllm/model_executor/layers/sparse_attn_indexer.py` modified +22/-8 (30 lines); hunks: -499,13 +499,31 @@ def forward_hip(; -522,8 +540,4 @@ def forward_hip(; symbols: forward_hip，涉及 `forward_hip`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/mhc.py` modified +105/-2 (107 lines); hunks: -234,6 +234,39 @@ def mhc_pre(; -414,6 +447,14 @@ def mhc_post(; symbols: mhc_pre, mhc_post, hc_head_fuse_tilelang, _hc_head_fused_reference
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +73/-19 (92 lines); hunks: -28,6 +28,11; -53,6 +58,7; symbols: __init__, forward, attn_gemm_parallel_execute
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +79/-2 (81 lines); hunks: -18,6 +18,7; -64,6 +65,8 @@ class Mxfp4MoeBackend(Enum):; symbols: Mxfp4MoeBackend, _get_priority_backends, _return_or_raise, convert_weight_to_mxfp4_moe_kernel_format
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +22/-8 (30 lines); hunks: -499,13 +499,31 @@ def forward_hip(; -522,8 +540,4 @@ def forward_hip(; symbols: forward_hip
  - `vllm/model_executor/kernels/linear/scaled_mm/aiter.py` modified +15/-0 (15 lines); hunks: -312,6 +312,21 @@ def apply_block_scaled_mm(; symbols: apply_block_scaled_mm
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/mhc.py` modified +105/-2; `vllm/model_executor/layers/deepseek_v4_attention.py` modified +73/-19; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +79/-2; `vllm/model_executor/layers/sparse_attn_indexer.py` modified +22/-8; `vllm/model_executor/kernels/linear/scaled_mm/aiter.py` modified +15/-0; `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +9/-0
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_topk_softplus_sqrt.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41801 - [Bugfix] DeepSeekV32/v4: respect string='true|false' attribute andunwrap arguments/input wrapper

- 链接: https://github.com/vllm-project/vllm/pull/41801
- 状态/时间: merged / 2026-05-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+224/-10，可读 patch 298 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] DeepSeekV32/v4: respect string='true|false' attribute andunwrap arguments/input wrapper」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`；技术摘要: 覆盖「[Bugfix] DeepSeekV32/v4: respect string='true|false' attribute andunwrap arguments/input wrapper」；主要实现面是 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +155/-2 (157 lines); hunks: -203,7 +203,14 @@ def test_type_conversion_in_non_streaming(self):; -212,6 +219,118 @@ def test_type_conversion_in_non_streaming(self):; symbols: test_type_conversion_in_non_streaming, test_string_attr_true_preserves_literal_despite_schema, test_string_attr_false_allows_schema_conversion, test_arguments_wrapper_repaired，涉及 `test_type_conversion_in_non_streaming, test_string_attr_true_preserves_literal_despite_schema, test_string_attr_false_allows_schema_conversion`；`vllm/tool_parsers/deepseekv32_tool_parser.py` modified +36/-8 (44 lines); hunks: -69,7 +69,7 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; -101,10 +101,12 @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, _generate_tool_call_id, _parse_invoke_params, _convert_param_value_checked，涉及 `__init__, _generate_tool_call_id, _parse_invoke_params`；`tests/tool_parsers/test_deepseekv4_tool_parser.py` modified +33/-0 (33 lines); hunks: -203,3 +203,36 @@ def test_get_vllm_registry_structural_tag_returns_structura...; symbols: test_get_vllm_registry_structural_tag_returns_structural_tag, test_extract_tool_calls_arguments_wrapper，涉及 `test_get_vllm_registry_structural_tag_returns_structural_tag, test_extract_tool_calls_arguments_wrapper`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +155/-2 (157 lines); hunks: -203,7 +203,14 @@ def test_type_conversion_in_non_streaming(self):; -212,6 +219,118 @@ def test_type_conversion_in_non_streaming(self):; symbols: test_type_conversion_in_non_streaming, test_string_attr_true_preserves_literal_despite_schema, test_string_attr_false_allows_schema_conversion, test_arguments_wrapper_repaired
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +36/-8 (44 lines); hunks: -69,7 +69,7 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; -101,10 +101,12 @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, _generate_tool_call_id, _parse_invoke_params, _convert_param_value_checked
  - `tests/tool_parsers/test_deepseekv4_tool_parser.py` modified +33/-0 (33 lines); hunks: -203,3 +203,36 @@ def test_get_vllm_registry_structural_tag_returns_structura...; symbols: test_get_vllm_registry_structural_tag_returns_structural_tag, test_extract_tool_calls_arguments_wrapper
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +155/-2; `tests/tool_parsers/test_deepseekv4_tool_parser.py` modified +33/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +36/-8
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41428 - [DSv4] Improved fused Indexer Q quant kernel

- 链接: https://github.com/vllm-project/vllm/pull/41428
- 状态/时间: merged / 2026-05-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+474/-25，可读 patch 527 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSv4] Improved fused Indexer Q quant kernel」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `vllm/utils/import_utils.py`；技术摘要: 覆盖「[DSv4] Improved fused Indexer Q quant kernel」；主要实现面是 `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `vllm/utils/import_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, _recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32，涉及 `fused_indexer_q_rope_quant_mxfp4_cutedsl, _recast_val, _fp32x2_to_bf16x2`；`vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +45/-24 (69 lines); hunks: -1,8 +1,10; -342,30 +344,49 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant，涉及 `fused_indexer_q_rope_quant`；`vllm/utils/import_utils.py` modified +5/-0 (5 lines); hunks: -469,3 +469,8 @@ def has_mori() -> bool:; symbols: has_mori, has_fbgemm_gpu, has_cutedsl，涉及 `has_mori, has_fbgemm_gpu, has_cutedsl`；`tests/kernels/test_fused_indexer_q_rope_quant.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7 @@ def _reference(; symbols: _reference，涉及 `_reference`。
- 代码 diff 细节:
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, _recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +45/-24 (69 lines); hunks: -1,8 +1,10; -342,30 +344,49 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant
  - `vllm/utils/import_utils.py` modified +5/-0 (5 lines); hunks: -469,3 +469,8 @@ def has_mori() -> bool:; symbols: has_mori, has_fbgemm_gpu, has_cutedsl
  - `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7 @@ def _reference(; symbols: _reference
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` added +423/-0; `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +45/-24; `vllm/utils/import_utils.py` modified +5/-0
  - tests: `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/kernels/test_fused_indexer_q_rope_quant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41957 - [Bugfix][PD] Fix DSv4 Disaggregated

- 链接: https://github.com/vllm-project/vllm/pull/41957
- 状态/时间: merged / 2026-05-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+49/-35，可读 patch 213 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][PD] Fix DSv4 Disaggregated」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py`, `tests/v1/kv_connector/unit/test_tp_mapping.py`；技术摘要: 覆盖「[Bugfix][PD] Fix DSv4 Disaggregated」；主要实现面是 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py`, `tests/v1/kv_connector/unit/test_tp_mapping.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py` modified +23/-23 (46 lines); hunks: -53,6 +53,7; -100,24 +101,24 @@ def _compute_desc_ids(; symbols: _compute_desc_ids, __init__, add_remote_agent, _validate_remote_agent_handshake，涉及 `_compute_desc_ids, __init__, add_remote_agent`；`vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py` modified +9/-9 (18 lines); hunks: -10,6 +10,7; -62,25 +63,24 @@ class TPMapping:; symbols: TPMapping, compute_tp_mapping，涉及 `TPMapping, compute_tp_mapping`；`tests/v1/kv_connector/unit/test_tp_mapping.py` modified +7/-2 (9 lines); hunks: -9,6 +9,8; -33,12 +35,15 @@ def _compute_mapping(; symbols: _compute_mapping，涉及 `_compute_mapping`；`vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py` modified +9/-0 (9 lines); hunks: -10,6 +10,7; -46,3 +47,11 @@ def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Sock...; symbols: zmq_ctx, get_representative_spec_type，涉及 `zmq_ctx, get_representative_spec_type`。
- 代码 diff 细节:
  - `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py` modified +23/-23 (46 lines); hunks: -53,6 +53,7; -100,24 +101,24 @@ def _compute_desc_ids(; symbols: _compute_desc_ids, __init__, add_remote_agent, _validate_remote_agent_handshake
  - `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py` modified +9/-9 (18 lines); hunks: -10,6 +10,7; -62,25 +63,24 @@ class TPMapping:; symbols: TPMapping, compute_tp_mapping
  - `tests/v1/kv_connector/unit/test_tp_mapping.py` modified +7/-2 (9 lines); hunks: -9,6 +9,8; -33,12 +35,15 @@ def _compute_mapping(; symbols: _compute_mapping
  - `vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py` modified +9/-0 (9 lines); hunks: -10,6 +10,7; -46,3 +47,11 @@ def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Sock...; symbols: zmq_ctx, get_representative_spec_type
  - `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1 (2 lines); hunks: -593,7 +593,7 @@ def describe(self, remote_engine_id: EngineId) -> str:; symbols: describe
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py` modified +23/-23; `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py` modified +9/-9; `vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py` modified +9/-0; `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1
  - tests: `tests/v1/kv_connector/unit/test_tp_mapping.py` modified +7/-2
- 验证与风险: diff 自带测试面 `tests/v1/kv_connector/unit/test_tp_mapping.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42169 - [Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len

- 链接: https://github.com/vllm-project/vllm/pull/42169
- 状态/时间: merged / 2026-05-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `csrc/topk.cu`；技术摘要: 覆盖「[Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len」；主要实现面是 `csrc/topk.cu`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `csrc/topk.cu` modified +2/-2 (4 lines); hunks: -20,7 +20,7 @@ void launch_persistent_topk(const torch::Tensor& logits,; -243,7 +243,7 @@ void persistent_topk(const torch::Tensor& logits, const torc...。
- 代码 diff 细节:
  - `csrc/topk.cu` modified +2/-2 (4 lines); hunks: -20,7 +20,7 @@ void launch_persistent_topk(const torch::Tensor& logits,; -243,7 +243,7 @@ void persistent_topk(const torch::Tensor& logits, const torc...
- 关键代码摘录:

```diff
diff -- csrc/topk.cu
@@ -20,7 +20,7 @@ void launch_persistent_topk(const torch::Tensor& logits,
-  const int64_t stride = logits.size(1);
+  const int64_t stride = logits.stride(0);
@@ -243,7 +243,7 @@ void persistent_topk(const torch::Tensor& logits, const torch::Tensor& lengths,
-  const int64_t stride = logits.size(1);
+  const int64_t stride = logits.stride(0);
```

- 已读文件:
  - other: `csrc/topk.cu` modified +2/-2
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #41694 - [DSV4] Add PP support for deepseek-v4

- 链接: https://github.com/vllm-project/vllm/pull/41694
- 状态/时间: merged / 2026-05-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+83/-22，可读 patch 216 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Add PP support for deepseek-v4」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/deepseek_v4.py`, `docs/models/supported_models.md`；技术摘要: 覆盖「[DSV4] Add PP support for deepseek-v4」；主要实现面是 `vllm/model_executor/models/deepseek_v4.py`, `docs/models/supported_models.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v4.py` modified +82/-21 (103 lines); hunks: -12,6 +12,7; -49,6 +50,7; symbols: __init__, embed_input_ids, make_empty_intermediate_tensors，涉及 `__init__, embed_input_ids, make_empty_intermediate_tensors`；`docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: -385,7 +385,7 @@ th {。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v4.py` modified +82/-21 (103 lines); hunks: -12,6 +12,7; -49,6 +50,7; symbols: __init__, embed_input_ids, make_empty_intermediate_tensors
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: -385,7 +385,7 @@ th {
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +82/-21
  - docs: `docs/models/supported_models.md` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41536 - add fused mhc_post_pre kernel

- 链接: https://github.com/vllm-project/vllm/pull/41536
- 状态/时间: merged / 2026-05-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+533/-11，可读 patch 592 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「add fused mhc_post_pre kernel」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py`, `tests/kernels/test_mhc_kernels.py`；技术摘要: 覆盖「add fused mhc_post_pre kernel」；主要实现面是 `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py`, `tests/kernels/test_mhc_kernels.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/mhc.py` modified +343/-0 (343 lines); hunks: -408,6 +408,131 @@ def mhc_post_tilelang(; -427,6 +552,218 @@ def mhc_post(; symbols: mhc_post_tilelang, mhc_fused_tilelang, mhc_post, mhc_fused_post_pre，涉及 `mhc_post_tilelang, mhc_fused_tilelang, mhc_post`；`vllm/model_executor/models/deepseek_v4.py` modified +48/-11 (59 lines); hunks: -1199,23 +1199,53 @@ def forward(; -1320,12 +1350,19 @@ def forward(; symbols: forward，涉及 `forward`；`tests/kernels/test_mhc_kernels.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: sinkhorn_normalize_ref, mhc_pre_ref, mhc_post_ref, test_mhc_fused_post_pre，涉及 `sinkhorn_normalize_ref, mhc_pre_ref, mhc_post_ref`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/mhc.py` modified +343/-0 (343 lines); hunks: -408,6 +408,131 @@ def mhc_post_tilelang(; -427,6 +552,218 @@ def mhc_post(; symbols: mhc_post_tilelang, mhc_fused_tilelang, mhc_post, mhc_fused_post_pre
  - `vllm/model_executor/models/deepseek_v4.py` modified +48/-11 (59 lines); hunks: -1199,23 +1199,53 @@ def forward(; -1320,12 +1350,19 @@ def forward(; symbols: forward
  - `tests/kernels/test_mhc_kernels.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: sinkhorn_normalize_ref, mhc_pre_ref, mhc_post_ref, test_mhc_fused_post_pre
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/mhc.py` modified +343/-0; `vllm/model_executor/models/deepseek_v4.py` modified +48/-11
  - tests: `tests/kernels/test_mhc_kernels.py` added +142/-0
- 验证与风险: diff 自带测试面 `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40392 - [Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA

- 链接: https://github.com/vllm-project/vllm/pull/40392
- 状态/时间: merged / 2026-05-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+966/-109，可读 patch 1331 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py`；技术摘要: 覆盖「[Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA」；主要实现面是 `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/attention/mla_attention.py` modified +19/-24 (43 lines); hunks: -345,6 +345,7 @@ def __init__(; -374,14 +375,21 @@ def __init__(; symbols: __init__, unified_mla_kv_cache_update, unified_mla_attention_with_output_fake，涉及 `__init__, unified_mla_kv_cache_update, unified_mla_attention_with_output_fake`；`vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +32/-9 (41 lines); hunks: -127,29 +127,52 @@ def forward_native(; symbols: forward_native, forward_static，涉及 `forward_native, forward_static`；`vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` modified +2/-4 (6 lines); hunks: -195,10 +195,8 @@ def forward_cuda(; symbols: forward_cuda, _apply_rotary_embedding，涉及 `forward_cuda, _apply_rotary_embedding`；`tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py` added +413/-0 (413 lines); hunks: -0,0 +1,413; symbols: MLARoPEKVCacheCatTestModel, __init__, build_attn_metadata, forward，涉及 `MLARoPEKVCacheCatTestModel, __init__, build_attn_metadata`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +19/-24 (43 lines); hunks: -345,6 +345,7 @@ def __init__(; -374,14 +375,21 @@ def __init__(; symbols: __init__, unified_mla_kv_cache_update, unified_mla_attention_with_output_fake
  - `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +32/-9 (41 lines); hunks: -127,29 +127,52 @@ def forward_native(; symbols: forward_native, forward_static
  - `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` modified +2/-4 (6 lines); hunks: -195,10 +195,8 @@ def forward_cuda(; symbols: forward_cuda, _apply_rotary_embedding
  - `tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py` added +413/-0 (413 lines); hunks: -0,0 +1,413; symbols: MLARoPEKVCacheCatTestModel, __init__, build_attn_metadata, forward
  - `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py` added +271/-0 (271 lines); hunks: -0,0 +1,271; symbols: fused_rope_unified_mla_kv_cache_update_impl, fused_rope_unified_mla_kv_cache_update_fake, MLARoPEKVCacheCatPattern, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/attention/mla_attention.py` modified +19/-24; `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +32/-9; `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` modified +2/-4; `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py` added +271/-0; `vllm/compilation/passes/fusion/matcher_utils.py` modified +84/-0; `vllm/compilation/passes/utility/fix_functionalization.py` modified +39/-0
  - tests: `tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py` added +413/-0
  - other: `csrc/cache_kernels_fused.cu` modified +75/-60
- 验证与风险: diff 自带测试面 `tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py`, `tests/compile/passes/test_rope_kvcache_fusion.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42236 - [DSv4] Improved dequant gather K cache kernel

- 链接: https://github.com/vllm-project/vllm/pull/42236
- 状态/时间: merged / 2026-05-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+658/-100，可读 patch 832 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSv4] Improved dequant gather K cache kernel」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py`, `tests/kernels/test_compressor_kv_cache.py`, `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py`；技术摘要: 覆盖「[DSv4] Improved dequant gather K cache kernel」；主要实现面是 `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py`, `tests/kernels/test_compressor_kv_cache.py`, `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py` added +334/-0 (334 lines); hunks: -0,0 +1,334; symbols: dequantize_and_gather_k_cache_cutedsl, DequantGatherKCacheKernel, __init__, __call__，涉及 `dequantize_and_gather_k_cache_cutedsl, DequantGatherKCacheKernel, __init__`；`tests/kernels/test_compressor_kv_cache.py` modified +141/-7 (148 lines); hunks: -3,11 +3,12; -134,7 +135,140 @@ def test_deepseek_v4_attention_quant_cache_roundtrip(num_t...; symbols: test_deepseek_v4_attention_quant_cache_roundtrip, _dequantize_and_gather_k_cache_reference, test_dequantize_and_gather_k_cache, test_indexer_gather_accepts_upper_bound_output，涉及 `test_deepseek_v4_attention_quant_cache_roundtrip, _dequantize_and_gather_k_cache_reference, test_dequantize_and_gather_k_cache`；`vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` added +145/-0 (145 lines); hunks: -0,0 +1,145; symbols: _recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32, _bf16x2_abs，涉及 `_recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32`；`vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +8/-92 (100 lines); hunks: -1,18 +1,22; -61,94 +65,6 @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, _recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32，涉及 `fused_indexer_q_rope_quant_mxfp4_cutedsl, _recast_val, _fp32x2_to_bf16x2`。
- 代码 diff 细节:
  - `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py` added +334/-0 (334 lines); hunks: -0,0 +1,334; symbols: dequantize_and_gather_k_cache_cutedsl, DequantGatherKCacheKernel, __init__, __call__
  - `tests/kernels/test_compressor_kv_cache.py` modified +141/-7 (148 lines); hunks: -3,11 +3,12; -134,7 +135,140 @@ def test_deepseek_v4_attention_quant_cache_roundtrip(num_t...; symbols: test_deepseek_v4_attention_quant_cache_roundtrip, _dequantize_and_gather_k_cache_reference, test_dequantize_and_gather_k_cache, test_indexer_gather_accepts_upper_bound_output
  - `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` added +145/-0 (145 lines); hunks: -0,0 +1,145; symbols: _recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32, _bf16x2_abs
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +8/-92 (100 lines); hunks: -1,18 +1,22; -61,94 +65,6 @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, _recast_val, _fp32x2_to_bf16x2, _bf16x2_to_fp32
  - `vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py` modified +30/-1 (31 lines); hunks: -17,6 +17,7; -303,7 +304,7 @@ def _dequantize_and_gather_k_kernel(; symbols: _dequantize_and_gather_k_kernel, dequantize_and_gather_k_cache, dequantize_and_gather_k_cache_triton
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py` added +334/-0; `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` added +145/-0; `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +8/-92; `vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py` modified +30/-1
  - tests: `tests/kernels/test_compressor_kv_cache.py` modified +141/-7
- 验证与风险: diff 自带测试面 `tests/kernels/test_compressor_kv_cache.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41812 - [ROCm][DSv4] implement flash sparse mla with triton kernels

- 链接: https://github.com/vllm-project/vllm/pull/41812
- 状态/时间: merged / 2026-05-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+1849/-212，可读 patch 2180 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][DSv4] implement flash sparse mla with triton kernels」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`；技术摘要: 覆盖「[ROCm][DSv4] implement flash sparse mla with triton kernels」；主要实现面是 `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +24/-46 (70 lines); hunks: -28,11 +28,7; -725,6 +721,12 @@ def __init__(; symbols: __init__, get_attn_backend, get_kv_cache_spec, forward，涉及 `__init__, get_attn_backend, get_kv_cache_spec`；`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +758/-164 (922 lines); hunks: -905,185 +905,757 @@ def rocm_inv_rope_einsum(; -1092,38 +1664,60 @@ def rocm_forward_decode_fallback(; symbols: rocm_inv_rope_einsum, rocm_ref_sparse_attn_prefill, _validate_dsv4_sparse_dims, _pack_dense_prefix_to_ragged_kernel，涉及 `rocm_inv_rope_einsum, rocm_ref_sparse_attn_prefill, _validate_dsv4_sparse_dims`；`vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` added +682/-0 (682 lines); hunks: -0,0 +1,682; symbols: _build_indptr_from_lengths, _compute_topk_lens_kernel, _pack_global_topk_ragged_kernel, compute_global_topk_ragged_indices_and_indptr，涉及 `_build_indptr_from_lengths, _compute_topk_lens_kernel, _pack_global_topk_ragged_kernel`；`tests/kernels/attention/test_rocm_triton_attn_dsv4.py` added +377/-0 (377 lines); hunks: -0,0 +1,377; symbols: _ref_global_topk_ragged, _ref_sparse_prefill_ragged, _pack_fp8_ds_mla_cache, _read_fp8_ds_mla_cache，涉及 `_ref_global_topk_ragged, _ref_sparse_prefill_ragged, _pack_fp8_ds_mla_cache`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +24/-46 (70 lines); hunks: -28,11 +28,7; -725,6 +721,12 @@ def __init__(; symbols: __init__, get_attn_backend, get_kv_cache_spec, forward
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +758/-164 (922 lines); hunks: -905,185 +905,757 @@ def rocm_inv_rope_einsum(; -1092,38 +1664,60 @@ def rocm_forward_decode_fallback(; symbols: rocm_inv_rope_einsum, rocm_ref_sparse_attn_prefill, _validate_dsv4_sparse_dims, _pack_dense_prefix_to_ragged_kernel
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` added +682/-0 (682 lines); hunks: -0,0 +1,682; symbols: _build_indptr_from_lengths, _compute_topk_lens_kernel, _pack_global_topk_ragged_kernel, compute_global_topk_ragged_indices_and_indptr
  - `tests/kernels/attention/test_rocm_triton_attn_dsv4.py` added +377/-0 (377 lines); hunks: -0,0 +1,377; symbols: _ref_global_topk_ragged, _ref_sparse_prefill_ragged, _pack_fp8_ds_mla_cache, _read_fp8_ds_mla_cache
  - `vllm/v1/attention/backends/mla/sparse_swa.py` modified +6/-0 (6 lines); hunks: -112,6 +112,12 @@ def get_supported_head_sizes(cls) -> list[int]:; symbols: get_supported_head_sizes, get_builder_cls
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +24/-46; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +758/-164; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` added +682/-0; `vllm/v1/attention/backends/mla/sparse_swa.py` modified +6/-0; `vllm/v1/attention/backends/mla/flashmla_sparse.py` modified +2/-2
  - tests: `tests/kernels/attention/test_rocm_triton_attn_dsv4.py` added +377/-0
- 验证与风险: diff 自带测试面 `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41946 - [Bugfix] [ROCm] [DSV4] [Perf] Add aiter mhc support

- 链接: https://github.com/vllm-project/vllm/pull/41946
- 状态/时间: merged / 2026-05-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+1920/-1033，可读 patch 3143 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] [ROCm] [DSV4] [Perf] Add aiter mhc support」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/kernels/mhc/tilelang.py`, `vllm/model_executor/kernels/mhc/triton.py`；技术摘要: 覆盖「[Bugfix] [ROCm] [DSV4] [Perf] Add aiter mhc support」；主要实现面是 `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/kernels/mhc/tilelang.py`, `vllm/model_executor/kernels/mhc/triton.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/mhc.py` modified +244/-988 (1232 lines); hunks: -1,1030 +1,286; symbols: compute_num_split, mhc_pre_big_fuse_tilelang, MHCPreOp, enabled，涉及 `compute_num_split, mhc_pre_big_fuse_tilelang, MHCPreOp`；`vllm/model_executor/kernels/mhc/tilelang.py` added +468/-0 (468 lines); hunks: -0,0 +1,468; symbols: mhc_pre_tilelang, _mhc_pre_tilelang_fake, mhc_post_tilelang, mhc_fused_post_pre_tilelang，涉及 `mhc_pre_tilelang, _mhc_pre_tilelang_fake, mhc_post_tilelang`；`vllm/model_executor/kernels/mhc/triton.py` added +174/-0 (174 lines); hunks: -0,0 +1,174; symbols: _rmsnorm_nw_kernel, rmsnorm_nw, _hc_head_reduce_store_kernel, hc_head_reduce_triton_kernel，涉及 `_rmsnorm_nw_kernel, rmsnorm_nw, _hc_head_reduce_store_kernel`；`vllm/model_executor/kernels/mhc/aiter.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: mhc_pre_aiter, _mhc_pre_aiter_fake, mhc_post_aiter, _mhc_post_aiter_fake，涉及 `mhc_pre_aiter, _mhc_pre_aiter_fake, mhc_post_aiter`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/mhc.py` modified +244/-988 (1232 lines); hunks: -1,1030 +1,286; symbols: compute_num_split, mhc_pre_big_fuse_tilelang, MHCPreOp, enabled
  - `vllm/model_executor/kernels/mhc/tilelang.py` added +468/-0 (468 lines); hunks: -0,0 +1,468; symbols: mhc_pre_tilelang, _mhc_pre_tilelang_fake, mhc_post_tilelang, mhc_fused_post_pre_tilelang
  - `vllm/model_executor/kernels/mhc/triton.py` added +174/-0 (174 lines); hunks: -0,0 +1,174; symbols: _rmsnorm_nw_kernel, rmsnorm_nw, _hc_head_reduce_store_kernel, hc_head_reduce_triton_kernel
  - `vllm/model_executor/kernels/mhc/aiter.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: mhc_pre_aiter, _mhc_pre_aiter_fake, mhc_post_aiter, _mhc_post_aiter_fake
  - `vllm/model_executor/kernels/mhc/torch.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: mhc_pre_torch, mhc_post_torch
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/mhc.py` modified +244/-988; `vllm/model_executor/kernels/mhc/tilelang.py` added +468/-0; `vllm/model_executor/kernels/mhc/triton.py` added +174/-0; `vllm/model_executor/kernels/mhc/aiter.py` added +138/-0; `vllm/model_executor/kernels/mhc/torch.py` added +106/-0; `vllm/model_executor/models/deepseek_v4.py` modified +59/-38
- 验证与风险: diff 自带测试面 `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42320 - [Bugfix] Fix DeepSeek V4 MTP HC state handling

- 链接: https://github.com/vllm-project/vllm/pull/42320
- 状态/时间: merged / 2026-05-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+8/-5，可读 patch 29 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix DeepSeek V4 MTP HC state handling」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`；技术摘要: 覆盖「[Bugfix] Fix DeepSeek V4 MTP HC state handling」；主要实现面是 `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -1203,10 +1203,10 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/model_executor/models/deepseek_v4_mtp.py` modified +4/-1 (5 lines); hunks: -141,9 +141,12 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -1203,10 +1203,10 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +4/-1 (5 lines); hunks: -141,9 +141,12 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +4/-4; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41778 - [MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell

- 链接: https://github.com/vllm-project/vllm/pull/41778
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+640/-89，可读 patch 975 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `benchmarks/attention_benchmarks/configs/mla_prefill.yaml`, `benchmarks/attention_benchmarks/configs/mla_decode.yaml`, `vllm/model_executor/layers/attention/mla_attention.py`；技术摘要: 覆盖「[MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell」；主要实现面是 `benchmarks/attention_benchmarks/configs/mla_prefill.yaml`, `benchmarks/attention_benchmarks/configs/mla_decode.yaml`, `vllm/model_executor/layers/attention/mla_attention.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `benchmarks/attention_benchmarks/configs/mla_prefill.yaml` modified +2/-0 (2 lines); hunks: -3,6 +3,7; -120,6 +121,7 @@ prefill_backends:；`benchmarks/attention_benchmarks/configs/mla_decode.yaml` modified +1/-0 (1 lines); hunks: -53,6 +53,7 @@ backends:；`vllm/model_executor/layers/attention/mla_attention.py` modified +1/-0 (1 lines); hunks: -1362,6 +1362,7 @@ def backend_supports_prefill_query_quantization() -> bool:; symbols: backend_supports_prefill_query_quantization，涉及 `backend_supports_prefill_query_quantization`；`vllm/v1/attention/backends/mla/tokenspeed_mla.py` added +277/-0 (277 lines); hunks: -0,0 +1,277; symbols: _get_workspace, TokenspeedMLAMetadataBuilder, TokenspeedMLABackend, get_supported_kernel_block_sizes，涉及 `_get_workspace, TokenspeedMLAMetadataBuilder, TokenspeedMLABackend`。
- 代码 diff 细节:
  - `benchmarks/attention_benchmarks/configs/mla_prefill.yaml` modified +2/-0 (2 lines); hunks: -3,6 +3,7; -120,6 +121,7 @@ prefill_backends:
  - `benchmarks/attention_benchmarks/configs/mla_decode.yaml` modified +1/-0 (1 lines); hunks: -53,6 +53,7 @@ backends:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +1/-0 (1 lines); hunks: -1362,6 +1362,7 @@ def backend_supports_prefill_query_quantization() -> bool:; symbols: backend_supports_prefill_query_quantization
  - `vllm/v1/attention/backends/mla/tokenspeed_mla.py` added +277/-0 (277 lines); hunks: -0,0 +1,277; symbols: _get_workspace, TokenspeedMLAMetadataBuilder, TokenspeedMLABackend, get_supported_kernel_block_sizes
  - `vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py` added +180/-0 (180 lines); hunks: -0,0 +1,180; symbols: TokenspeedMLAPrefillBackend, get_name, supports_compute_capability, is_available
- 关键代码摘录:

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

- 已读文件:
  - runtime: `benchmarks/attention_benchmarks/configs/mla_prefill.yaml` modified +2/-0; `benchmarks/attention_benchmarks/configs/mla_decode.yaml` modified +1/-0; `vllm/model_executor/layers/attention/mla_attention.py` modified +1/-0; `vllm/v1/attention/backends/mla/tokenspeed_mla.py` added +277/-0; `vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py` added +180/-0
  - other: `benchmarks/attention_benchmarks/mla_runner.py` modified +67/-63
  - tests: `tests/v1/attention/test_mla_backends.py` modified +66/-7; `tests/conftest.py` modified +22/-13
- 验证与风险: diff 自带测试面 `tests/conftest.py`, `tests/v1/attention/test_mla_backends.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42342 - [Bug] Fix DeepSeek V4 `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'LoadCacheMode'`

- 链接: https://github.com/vllm-project/vllm/pull/42342
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 7 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug] Fix DeepSeek V4 `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'LoadCacheMode'`」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `requirements/cuda.txt`；技术摘要: 覆盖「[Bug] Fix DeepSeek V4 `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'LoadCacheMode'`」；主要实现面是 `requirements/cuda.txt`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `requirements/cuda.txt` modified +1/-1 (2 lines); hunks: -21,5 +21,5 @@ nvidia-cudnn-frontend>=1.13.0,<1.19.0。
- 代码 diff 细节:
  - `requirements/cuda.txt` modified +1/-1 (2 lines); hunks: -21,5 +21,5 @@ nvidia-cudnn-frontend>=1.13.0,<1.19.0
- 关键代码摘录:

```diff
diff -- requirements/cuda.txt
@@ -21,5 +21,5 @@ nvidia-cudnn-frontend>=1.13.0,<1.19.0
-nvidia-cutlass-dsl[cu13]>=4.4.2
+nvidia-cutlass-dsl[cu13]==4.5.0
```

- 已读文件:
  - other: `requirements/cuda.txt` modified +1/-1
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #41263 - [DSV4] Fuse norm and router for low latency scenario

- 链接: https://github.com/vllm-project/vllm/pull/41263
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+815/-43，可读 patch 1013 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Fuse norm and router for low latency scenario」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`；技术摘要: 覆盖「[DSV4] Fuse norm and router for low latency scenario」；主要实现面是 `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: _dsv4_pro_norm_gate, _dsv4_pro_norm_gate_fake, NormGateLinear, __init__，涉及 `_dsv4_pro_norm_gate, _dsv4_pro_norm_gate_fake, NormGateLinear`；`vllm/model_executor/models/deepseek_v4.py` modified +44/-42 (86 lines); hunks: -23,11 +23,14; -755,23 +758,23 @@ def __init__(; symbols: __init__, _init_fused_moe_experts, forward，涉及 `__init__, _init_fused_moe_experts, forward`；`vllm/model_executor/models/deepseek_v4_mtp.py` modified +11/-1 (12 lines); hunks: -290,6 +290,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -437,7 +442,12 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: load_weights, _remap_weight_name, _find_mtp_layer_idx，涉及 `load_weights, _remap_weight_name, _find_mtp_layer_idx`；`csrc/moe/dsv4_norm_router_gemm_kernel.cu` added +249/-0 (249 lines); hunks: -0,0 +1,249。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: _dsv4_pro_norm_gate, _dsv4_pro_norm_gate_fake, NormGateLinear, __init__
  - `vllm/model_executor/models/deepseek_v4.py` modified +44/-42 (86 lines); hunks: -23,11 +23,14; -755,23 +758,23 @@ def __init__(; symbols: __init__, _init_fused_moe_experts, forward
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +11/-1 (12 lines); hunks: -290,6 +290,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -437,7 +442,12 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: load_weights, _remap_weight_name, _find_mtp_layer_idx
  - `csrc/moe/dsv4_norm_router_gemm_kernel.cu` added +249/-0 (249 lines); hunks: -0,0 +1,249
  - `benchmarks/kernels/benchmark_norm_router_gemm.py` added +183/-0 (183 lines); hunks: -0,0 +1,183; symbols: unfused_norm_router_gemm, fused_norm_router_gemm, _make_inputs, calculate_diff
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py` added +114/-0; `vllm/model_executor/models/deepseek_v4.py` modified +44/-42; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +11/-1; `vllm/_custom_ops.py` modified +30/-0
  - other: `csrc/moe/dsv4_norm_router_gemm_kernel.cu` added +249/-0; `benchmarks/kernels/benchmark_norm_router_gemm.py` added +183/-0; `csrc/moe/dsv4_norm_router_gemm_entry.cu` added +130/-0; `csrc/moe/dsv4_norm_router_gemm.h` added +30/-0
- 验证与风险: runtime 路径改动集中在 `vllm/_custom_ops.py`, `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py`, `vllm/model_executor/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42112 - [Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup

- 链接: https://github.com/vllm-project/vllm/pull/42112
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+9/-15，可读 patch 86 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/v1/attention/backends/mla/prefill/flashinfer.py`, `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py`；技术摘要: 覆盖「[Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup」；主要实现面是 `vllm/v1/attention/backends/mla/prefill/flashinfer.py`, `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/v1/attention/backends/mla/prefill/flashinfer.py` modified +6/-7 (13 lines); hunks: -77,6 +77,9 @@ def __init__(; -123,21 +126,17 @@ def prepare_metadata(; symbols: __init__, _ensure_chunks, prepare_metadata，涉及 `__init__, _ensure_chunks, prepare_metadata`；`vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` modified +3/-8 (11 lines); hunks: -61,15 +61,12 @@ def __init__(; -89,7 +86,6 @@ def run_prefill_new_tokens(; symbols: __init__, _get_workspace_buffer, prepare_metadata, run_prefill_new_tokens，涉及 `__init__, _get_workspace_buffer, prepare_metadata`。
- 代码 diff 细节:
  - `vllm/v1/attention/backends/mla/prefill/flashinfer.py` modified +6/-7 (13 lines); hunks: -77,6 +77,9 @@ def __init__(; -123,21 +126,17 @@ def prepare_metadata(; symbols: __init__, _ensure_chunks, prepare_metadata
  - `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` modified +3/-8 (11 lines); hunks: -61,15 +61,12 @@ def __init__(; -89,7 +86,6 @@ def run_prefill_new_tokens(; symbols: __init__, _get_workspace_buffer, prepare_metadata, run_prefill_new_tokens
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/v1/attention/backends/mla/prefill/flashinfer.py` modified +6/-7; `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` modified +3/-8
- 验证与风险: runtime 路径改动集中在 `vllm/v1/attention/backends/mla/prefill/flashinfer.py`, `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42604 - DeepSeekV4-Pro enable cuda graph full and piecewise mode

- 链接: https://github.com/vllm-project/vllm/pull/42604
- 状态/时间: merged / 2026-05-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+73/-3，可读 patch 125 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「DeepSeekV4-Pro enable cuda graph full and piecewise mode」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/mhc.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`；技术摘要: 覆盖「DeepSeekV4-Pro enable cuda graph full and piecewise mode」；主要实现面是 `vllm/model_executor/layers/mhc.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/mhc.py` modified +0/-3 (3 lines); hunks: -5,7 +5,6; -190,8 +189,6 @@ def forward_cuda(; symbols: forward_cuda, forward_hip，涉及 `forward_cuda, forward_hip`；`vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` modified +73/-0 (73 lines); hunks: -302,6 +302,30 @@ def combine_topk_swa_indices_ragged(; -317,6 +341,23 @@ class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSW...; symbols: combine_topk_swa_indices_ragged, _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata, DeepseekV4ROCMAiterSparseSWAMetadata，涉及 `combine_topk_swa_indices_ragged, _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/mhc.py` modified +0/-3 (3 lines); hunks: -5,7 +5,6; -190,8 +189,6 @@ def forward_cuda(; symbols: forward_cuda, forward_hip
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` modified +73/-0 (73 lines); hunks: -302,6 +302,30 @@ def combine_topk_swa_indices_ragged(; -317,6 +341,23 @@ class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSW...; symbols: combine_topk_swa_indices_ragged, _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata, DeepseekV4ROCMAiterSparseSWAMetadata
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/mhc.py` modified +0/-3; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` modified +73/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mhc.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42810 - [ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy

- 链接: https://github.com/vllm-project/vllm/pull/42810
- 状态/时间: merged / 2026-05-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+88/-177，可读 patch 364 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v4.py`；技术摘要: 覆盖「[ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy」；主要实现面是 `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/mhc.py` modified +48/-40 (88 lines); hunks: -61,31 +61,35 @@ def forward_hip(; -124,21 +128,25 @@ def forward_hip(; symbols: forward_hip, forward_native，涉及 `forward_hip, forward_native`；`vllm/model_executor/layers/sparse_attn_indexer.py` modified +5/-22 (27 lines); hunks: -505,27 +505,6 @@ def forward_hip(; -541,5 +520,9 @@ def forward_hip(; symbols: forward_hip，涉及 `forward_hip`；`vllm/model_executor/models/deepseek_v4.py` modified +2/-1 (3 lines); hunks: -1277,7 +1277,8 @@ def _forward_rocm(; symbols: _forward_rocm，涉及 `_forward_rocm`；`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +33/-114 (147 lines); hunks: -542,7 +542,11 @@ def rocm_fp8_mqa_logits(; -551,6 +555,12 @@ def _topk_indices_torch(logits: torch.Tensor, topk_tokens:...; symbols: rocm_fp8_mqa_logits, _topk_indices_torch，涉及 `rocm_fp8_mqa_logits, _topk_indices_torch`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/mhc.py` modified +48/-40 (88 lines); hunks: -61,31 +61,35 @@ def forward_hip(; -124,21 +128,25 @@ def forward_hip(; symbols: forward_hip, forward_native
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +5/-22 (27 lines); hunks: -505,27 +505,6 @@ def forward_hip(; -541,5 +520,9 @@ def forward_hip(; symbols: forward_hip
  - `vllm/model_executor/models/deepseek_v4.py` modified +2/-1 (3 lines); hunks: -1277,7 +1277,8 @@ def _forward_rocm(; symbols: _forward_rocm
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +33/-114 (147 lines); hunks: -542,7 +542,11 @@ def rocm_fp8_mqa_logits(; -551,6 +555,12 @@ def _topk_indices_torch(logits: torch.Tensor, topk_tokens:...; symbols: rocm_fp8_mqa_logits, _topk_indices_torch
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/mhc.py` modified +48/-40; `vllm/model_executor/layers/sparse_attn_indexer.py` modified +5/-22; `vllm/model_executor/models/deepseek_v4.py` modified +2/-1; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +33/-114
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41710 - fix: remove unused norm for dpskv4

- 链接: https://github.com/vllm-project/vllm/pull/41710
- 状态/时间: merged / 2026-05-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: remove unused norm for dpskv4」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/deepseek_v4_attention.py`；技术摘要: 覆盖「fix: remove unused norm for dpskv4」；主要实现面是 `vllm/model_executor/layers/deepseek_v4_attention.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-2 (3 lines); hunks: -47,7 +47,7; -1111,7 +1111,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-2 (3 lines); hunks: -47,7 +47,7; -1111,7 +1111,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -47,7 +47,7 @@
-from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
+from vllm.model_executor.layers.layernorm import RMSNorm
@@ -1111,7 +1111,6 @@ def __init__(
-        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/deepseek_v4_attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42930 - [Bugfix] Fix DSV4 MTP after ROCm mHC integration

- 链接: https://github.com/vllm-project/vllm/pull/42930
- 状态/时间: merged / 2026-05-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+17/-12，可读 patch 57 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix DSV4 MTP after ROCm mHC integration」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`；技术摘要: 覆盖「[Bugfix] Fix DSV4 MTP after ROCm mHC integration」；主要实现面是 `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v4.py` modified +12/-8 (20 lines); hunks: -1261,10 +1261,12 @@ def _forward_rocm(; -1288,10 +1290,12 @@ def forward(; symbols: _forward_rocm, forward，涉及 `_forward_rocm, forward`；`vllm/model_executor/models/deepseek_v4_mtp.py` modified +5/-4 (9 lines); hunks: -146,9 +146,10 @@ def forward(; -235,7 +236,7 @@ def compute_logits(; symbols: forward, compute_logits，涉及 `forward, compute_logits`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v4.py` modified +12/-8 (20 lines); hunks: -1261,10 +1261,12 @@ def _forward_rocm(; -1288,10 +1290,12 @@ def forward(; symbols: _forward_rocm, forward
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +5/-4 (9 lines); hunks: -146,9 +146,10 @@ def forward(; -235,7 +236,7 @@ def compute_logits(; symbols: forward, compute_logits
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +12/-8; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +5/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42541 - [Bugfix] fix swiglu limit issue for humming backend + deepseek v4

- 链接: https://github.com/vllm-project/vllm/pull/42541
- 状态/时间: merged / 2026-05-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+34/-6，可读 patch 94 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] fix swiglu limit issue for humming backend + deepseek v4」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`, `vllm/model_executor/layers/quantization/utils/humming_utils.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`；技术摘要: 覆盖「[Bugfix] fix swiglu limit issue for humming backend + deepseek v4」；主要实现面是 `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`, `vllm/model_executor/layers/quantization/utils/humming_utils.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py` modified +19/-4 (23 lines); hunks: -33,7 +33,10; -422,6 +425,18 @@ def is_supported_config(; symbols: is_supported_config, apply_activation, HummingIndexedExperts, finalize_weight_and_reduce_impl，涉及 `is_supported_config, apply_activation, HummingIndexedExperts`；`vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +9/-1 (10 lines); hunks: -164,7 +164,12 @@ def prepare_humming_moe_layer(layer: RoutedExperts, quant_c...; -211,4 +216,7 @@ def get_humming_moe_quant_config(layer: RoutedExperts):; symbols: prepare_humming_moe_layer, get_humming_moe_quant_config，涉及 `prepare_humming_moe_layer, get_humming_moe_quant_config`；`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +6/-1 (7 lines); hunks: -1567,7 +1567,12 @@ def make_mxfp4_moe_quant_config(; symbols: make_mxfp4_moe_quant_config，涉及 `make_mxfp4_moe_quant_config`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py` modified +19/-4 (23 lines); hunks: -33,7 +33,10; -422,6 +425,18 @@ def is_supported_config(; symbols: is_supported_config, apply_activation, HummingIndexedExperts, finalize_weight_and_reduce_impl
  - `vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +9/-1 (10 lines); hunks: -164,7 +164,12 @@ def prepare_humming_moe_layer(layer: RoutedExperts, quant_c...; -211,4 +216,7 @@ def get_humming_moe_quant_config(layer: RoutedExperts):; symbols: prepare_humming_moe_layer, get_humming_moe_quant_config
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +6/-1 (7 lines); hunks: -1567,7 +1567,12 @@ def make_mxfp4_moe_quant_config(; symbols: make_mxfp4_moe_quant_config
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py` modified +19/-4; `vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +9/-1; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/quantization/utils/humming_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43004 - [Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]

- 链接: https://github.com/vllm-project/vllm/pull/43004
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/amd/__init__.py`, `vllm/models/deepseek_v4/nvidia/__init__.py`, `vllm/models/deepseek_v4/quant_config.py`；关联提交 `287471b99442`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+189/-126，可读 patch 476 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `vllm/models/deepseek_v4/quant_config.py`, `vllm/models/deepseek_v4/__init__.py`, `tests/models/test_deepseek_v4_mega_moe.py`；技术摘要: 覆盖「[Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]」；主要实现面是 `vllm/models/deepseek_v4/quant_config.py`, `vllm/models/deepseek_v4/__init__.py`, `tests/models/test_deepseek_v4_mega_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/quant_config.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: DeepseekV4FP8Config, __init__, expert_dtype, is_scale_e8m0，涉及 `DeepseekV4FP8Config, __init__, expert_dtype`；`vllm/models/deepseek_v4/__init__.py` added +30/-0 (30 lines); hunks: -0,0 +1,30；`tests/models/test_deepseek_v4_mega_moe.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7；`vllm/model_executor/layers/quantization/__init__.py` modified +1/-1 (2 lines); hunks: -113,7 +113,7 @@ def get_quantization_config(quantization: str) -> type[Quant...; symbols: get_quantization_config，涉及 `get_quantization_config`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/quant_config.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: DeepseekV4FP8Config, __init__, expert_dtype, is_scale_e8m0
  - `vllm/models/deepseek_v4/__init__.py` added +30/-0 (30 lines); hunks: -0,0 +1,30
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7
  - `vllm/model_executor/layers/quantization/__init__.py` modified +1/-1 (2 lines); hunks: -113,7 +113,7 @@ def get_quantization_config(quantization: str) -> type[Quant...; symbols: get_quantization_config
  - `vllm/models/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/quant_config.py` added +106/-0; `vllm/models/deepseek_v4/__init__.py` added +30/-0; `vllm/model_executor/layers/quantization/__init__.py` modified +1/-1; `vllm/models/__init__.py` added +2/-0; `vllm/models/deepseek_v4/amd/__init__.py` added +2/-0; `vllm/models/deepseek_v4/nvidia/__init__.py` added +2/-0
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/test_deepseek_v4_mega_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43039 - [Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N]

- 链接: https://github.com/vllm-project/vllm/pull/43039
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/compressor.py`；关联提交 `87b08c5f6460`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+8/-11，可读 patch 62 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N]」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/compressor.py`；技术摘要: 覆盖「[Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N]」；主要实现面是 `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/compressor.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/attention.py` renamed +1/-1 (2 lines); hunks: -46,7 +46,6; -55,6 +54,7；`vllm/models/deepseek_v4/compressor.py` renamed +0/-0 (0 lines)。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/attention.py` renamed +1/-1 (2 lines); hunks: -46,7 +46,6; -55,6 +54,7
  - `vllm/models/deepseek_v4/compressor.py` renamed +0/-0 (0 lines)
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v4/attention.py
@@ -46,7 +46,6 @@
-from vllm.model_executor.layers.deepseek_compressor import DeepseekCompressor
@@ -55,6 +54,7 @@
+from vllm.models.deepseek_v4.compressor import DeepseekCompressor
```

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/attention.py` renamed +1/-1; `vllm/models/deepseek_v4/compressor.py` renamed +0/-0
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42899 - add cutedsl dsv4 indexer fp8 kernel

- 链接: https://github.com/vllm-project/vllm/pull/42899
- 状态/时间: merged / 2026-05-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+411/-60，可读 patch 562 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「add cutedsl dsv4 indexer fp8 kernel」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`；技术摘要: 覆盖「add cutedsl dsv4 indexer fp8 kernel」；主要实现面是 `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +311/-37 (348 lines); hunks: -14,6 +14,7; -65,8 +66,48 @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, IndexerQMxFp4Kernel, fused_indexer_q_rope_quant_fp8_cutedsl, IndexerQRopeQuantKernel，涉及 `fused_indexer_q_rope_quant_mxfp4_cutedsl, IndexerQMxFp4Kernel, fused_indexer_q_rope_quant_fp8_cutedsl`；`vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +37/-20 (57 lines); hunks: -398,24 +398,41 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant，涉及 `fused_indexer_q_rope_quant`；`tests/kernels/test_fused_indexer_q_rope_quant.py` modified +30/-3 (33 lines); hunks: -13,13 +13,17; -125,8 +129,14 @@ def _reference(; symbols: _reference, test_fused_indexer_q_rope_quant_matches_unfused，涉及 `_reference, test_fused_indexer_q_rope_quant_matches_unfused`；`vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` modified +33/-0 (33 lines); hunks: -117,6 +117,39 @@ def _fp8x4_to_bf16x4(x: Uint32, *, loc=None, ip=None) -> cu...; symbols: _fp8x4_to_bf16x4, _fp32x4_to_fp8x4, _fp32x8_to_fp4x8，涉及 `_fp8x4_to_bf16x4, _fp32x4_to_fp8x4, _fp32x8_to_fp4x8`。
- 代码 diff 细节:
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +311/-37 (348 lines); hunks: -14,6 +14,7; -65,8 +66,48 @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, IndexerQMxFp4Kernel, fused_indexer_q_rope_quant_fp8_cutedsl, IndexerQRopeQuantKernel
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +37/-20 (57 lines); hunks: -398,24 +398,41 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant
  - `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +30/-3 (33 lines); hunks: -13,13 +13,17; -125,8 +129,14 @@ def _reference(; symbols: _reference, test_fused_indexer_q_rope_quant_matches_unfused
  - `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` modified +33/-0 (33 lines); hunks: -117,6 +117,39 @@ def _fp8x4_to_bf16x4(x: Uint32, *, loc=None, ip=None) -> cu...; symbols: _fp8x4_to_bf16x4, _fp32x4_to_fp8x4, _fp32x8_to_fp4x8
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +311/-37; `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +37/-20; `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` modified +33/-0
  - tests: `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +30/-3
- 验证与风险: diff 自带测试面 `tests/kernels/test_fused_indexer_q_rope_quant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43073 - [Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N]

- 链接: https://github.com/vllm-project/vllm/pull/43073
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/__init__.py`, `vllm/models/deepseek_v4/common/ops/__init__.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py` 等 13 个文件；关联提交 `b14be81c1f63`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+34/-29，可读 patch 197 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N]」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/ops/__init__.py`, `vllm/models/deepseek_v4/attention.py`；技术摘要: 覆盖「[Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N]」；主要实现面是 `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/ops/__init__.py`, `vllm/models/deepseek_v4/attention.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/compressor.py` modified +6/-10 (16 lines); hunks: -11,9 +11,13; -23,14 +27,6；`vllm/models/deepseek_v4/nvidia/ops/__init__.py` added +8/-0 (8 lines); hunks: -0,0 +1,8；`vllm/models/deepseek_v4/attention.py` modified +3/-3 (6 lines); hunks: -19,16 +19,16；`vllm/models/deepseek_v4/common/ops/cache_utils.py` renamed +3/-1 (4 lines); hunks: -366,7 +366,9 @@ def dequantize_and_gather_k_cache(; symbols: dequantize_and_gather_k_cache，涉及 `dequantize_and_gather_k_cache`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/compressor.py` modified +6/-10 (16 lines); hunks: -11,9 +11,13; -23,14 +27,6
  - `vllm/models/deepseek_v4/nvidia/ops/__init__.py` added +8/-0 (8 lines); hunks: -0,0 +1,8
  - `vllm/models/deepseek_v4/attention.py` modified +3/-3 (6 lines); hunks: -19,16 +19,16
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` renamed +3/-1 (4 lines); hunks: -366,7 +366,9 @@ def dequantize_and_gather_k_cache(; symbols: dequantize_and_gather_k_cache
  - `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` renamed +2/-2 (4 lines); hunks: -346,7 +346,7 @@ def fused_indexer_q_rope_quant(; -400,7 +400,7 @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/compressor.py` modified +6/-10; `vllm/models/deepseek_v4/nvidia/ops/__init__.py` added +8/-0; `vllm/models/deepseek_v4/attention.py` modified +3/-3; `vllm/models/deepseek_v4/common/ops/cache_utils.py` renamed +3/-1; `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` renamed +2/-2; `vllm/models/deepseek_v4/common/__init__.py` added +2/-0
- 验证与风险: diff 自带测试面 `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42828 - [KVConnector][DSV4] HMA support for Mooncake store connector

- 链接: https://github.com/vllm-project/vllm/pull/42828
- 状态/时间: merged / 2026-05-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+1835/-446，可读 patch 3088 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[KVConnector][DSV4] HMA support for Mooncake store connector」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `tests/v1/kv_connector/unit/test_mooncake_store_worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`, `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py`；技术摘要: 覆盖「[KVConnector][DSV4] HMA support for Mooncake store connector」；主要实现面是 `tests/v1/kv_connector/unit/test_mooncake_store_worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`, `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/v1/kv_connector/unit/test_mooncake_store_worker.py` modified +357/-117 (474 lines); hunks: -19,28 +19,48; -55,18 +75,27 @@ def _make_store_recving_thread(; symbols: _default_send_coord, _make_store_sending_thread, _make_store_recving_thread, _make_load_req，涉及 `_default_send_coord, _make_store_sending_thread, _make_store_recving_thread`；`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` modified +237/-180 (417 lines); hunks: -10,6 +10,7; -36,15 +37,25; symbols: KVTransferThread, __init__, KVCacheStoreSendingThread，涉及 `KVTransferThread, __init__, KVCacheStoreSendingThread`；`tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py` added +342/-0 (342 lines); hunks: -0,0 +1,342; symbols: _DictStore, __init__, setup, register_buffer，涉及 `_DictStore, __init__, setup`；`tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py` added +302/-0 (302 lines); hunks: -0,0 +1,302; symbols: _make_coord, test_external_cached_block_pool_tautological_returns_present_for_any_hash, test_external_cached_block_pool_hit_all_groups, test_external_cached_block_pool_miss_one_group，涉及 `_make_coord, test_external_cached_block_pool_tautological_returns_present_for_any_hash, test_external_cached_block_pool_hit_all_groups`。
- 代码 diff 细节:
  - `tests/v1/kv_connector/unit/test_mooncake_store_worker.py` modified +357/-117 (474 lines); hunks: -19,28 +19,48; -55,18 +75,27 @@ def _make_store_recving_thread(; symbols: _default_send_coord, _make_store_sending_thread, _make_store_recving_thread, _make_load_req
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` modified +237/-180 (417 lines); hunks: -10,6 +10,7; -36,15 +37,25; symbols: KVTransferThread, __init__, KVCacheStoreSendingThread
  - `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py` added +342/-0 (342 lines); hunks: -0,0 +1,342; symbols: _DictStore, __init__, setup, register_buffer
  - `tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py` added +302/-0 (302 lines); hunks: -0,0 +1,302; symbols: _make_coord, test_external_cached_block_pool_tautological_returns_present_for_any_hash, test_external_cached_block_pool_hit_all_groups, test_external_cached_block_pool_miss_one_group
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py` added +290/-0 (290 lines); hunks: -0,0 +1,290; symbols: ExternalCachedBlockPool, __init__, get_cached_block, MooncakeStoreCoordinator
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/v1/kv_connector/unit/test_mooncake_store_worker.py` modified +357/-117; `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py` added +342/-0; `tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py` added +302/-0; `tests/v1/kv_connector/unit/test_mooncake_store_connector.py` modified +72/-94; `tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py` added +111/-0
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` modified +237/-180; `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py` added +290/-0; `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py` modified +47/-33
- 验证与风险: diff 自带测试面 `tests/v1/kv_connector/unit/test_mooncake_store_connector.py`, `tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py`, `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py`, `tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43077 - [Model Refactoring] Rename deepseek_v4.py to model.py [4/N]

- 链接: https://github.com/vllm-project/vllm/pull/43077
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/model.py` 等 6 个文件；关联提交 `07beaed8422d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+8/-8，可读 patch 46 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model Refactoring] Rename deepseek_v4.py to model.py [4/N]」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `vllm/models/deepseek_v4/__init__.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`；技术摘要: 覆盖「[Model Refactoring] Rename deepseek_v4.py to model.py [4/N]」；主要实现面是 `vllm/models/deepseek_v4/__init__.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/__init__.py` modified +4/-4 (8 lines); hunks: -17,11 +17,11；`tests/models/test_deepseek_v4_mega_moe.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7；`vllm/models/deepseek_v4/nvidia/mtp.py` renamed +1/-1 (2 lines); hunks: -40,7 +40,7；`vllm/models/deepseek_v4/amd/deepseek_v4_mtp.py` removed +0/-1 (1 lines); hunks: -1 +0,0。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/__init__.py` modified +4/-4 (8 lines); hunks: -17,11 +17,11
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7
  - `vllm/models/deepseek_v4/nvidia/mtp.py` renamed +1/-1 (2 lines); hunks: -40,7 +40,7
  - `vllm/models/deepseek_v4/amd/deepseek_v4_mtp.py` removed +0/-1 (1 lines); hunks: -1 +0,0
  - `vllm/models/deepseek_v4/amd/model.py` added +1/-0 (1 lines); hunks: -0,0 +1
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/__init__.py` modified +4/-4; `vllm/models/deepseek_v4/nvidia/mtp.py` renamed +1/-1; `vllm/models/deepseek_v4/amd/deepseek_v4_mtp.py` removed +0/-1; `vllm/models/deepseek_v4/amd/model.py` added +1/-0; `vllm/models/deepseek_v4/amd/mtp.py` added +1/-0; `vllm/models/deepseek_v4/nvidia/model.py` renamed +0/-0
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/test_deepseek_v4_mega_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42111 - [CI] Add DSV4-Flash to gsm8k moe-refactor/config-b200.txt

- 链接: https://github.com/vllm-project/vllm/pull/42111
- 状态/时间: merged / 2026-05-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml`；关联提交 `cd0ff26e7acf`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+12/-1，可读 patch 47 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add DSV4-Flash to gsm8k moe-refactor/config-b200.txt」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml`；技术摘要: 覆盖「[CI] Add DSV4-Flash to gsm8k moe-refactor/config-b200.txt」；主要实现面是 `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml` added +5/-0 (5 lines); hunks: -0,0 +1,5。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml` added +5/-0 (5 lines); hunks: -0,0 +1,5
- 关键代码摘录:

```diff
diff -- tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml
@@ -0,0 +1,5 @@
+model_name: "deepseek-ai/DeepSeek-V4-Flash"
+accuracy_threshold: 0.95
+num_questions: 1319
+num_fewshot: 5
+server_args: "--trust-remote-code --kv-cache-dtype fp8 --block-size 256 --enable-expert-parallel --tensor-parallel-size 2 --attention_config.use_fp4_indexer_cache=True --moe-backe
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml` added +5/-0
- 验证与风险: diff 自带测试面 `requirements/test/cuda.txt`, `requirements/test/rocm.txt`, `requirements/test/xpu.txt`, `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42209 - Add NVFP4 MOE support for Deepseek V4.

- 链接: https://github.com/vllm-project/vllm/pull/42209
- 状态/时间: merged / 2026-05-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/quant_config.py`；关联提交 `fb21d8b4f902`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+217/-17，可读 patch 488 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add NVFP4 MOE support for Deepseek V4.」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/quant_config.py`；技术摘要: 覆盖「Add NVFP4 MOE support for Deepseek V4.」；主要实现面是 `vllm/models/deepseek_v4/quant_config.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/quant_config.py` modified +53/-1 (54 lines); hunks: -2,6 +2,10; -14,6 +18,11; symbols: DeepseekV4FP8Config, __init__, is_scale_e8m0, _resolve_moe_overrides，涉及 `DeepseekV4FP8Config, __init__, is_scale_e8m0`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/quant_config.py` modified +53/-1 (54 lines); hunks: -2,6 +2,10; -14,6 +18,11; symbols: DeepseekV4FP8Config, __init__, is_scale_e8m0, _resolve_moe_overrides
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/quant_config.py` modified +53/-1
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_trtllm_nvfp4_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43149 - [Refactor] Extract DeepSeek V4 sparse MLA impl into model folder

- 链接: https://github.com/vllm-project/vllm/pull/43149
- 状态/时间: merged / 2026-05-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/model.py`；关联提交 `843715739b7b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+485/-402，可读 patch 1059 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Extract DeepSeek V4 sparse MLA impl into model folder」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/ops/attention.py`, `vllm/models/deepseek_v4/amd/rocm.py`；技术摘要: 覆盖「[Refactor] Extract DeepSeek V4 sparse MLA impl into model folder」；主要实现面是 `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/ops/attention.py`, `vllm/models/deepseek_v4/amd/rocm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/flashmla.py` added +402/-0 (402 lines); hunks: -0,0 +1,402; symbols: DeepseekV4SparseMLAAttentionImpl, forward_mqa, DeepseekV4FlashMLASparseBackend, get_supported_kernel_block_sizes，涉及 `DeepseekV4SparseMLAAttentionImpl, forward_mqa, DeepseekV4FlashMLASparseBackend`；`vllm/models/deepseek_v4/nvidia/ops/attention.py` renamed +23/-309 (332 lines); hunks: -20,9 +20,6; -62,28 +59,36; symbols: _select_v4_sparse_impl, wq_b_kv_insert, __init__, get_attn_backend，涉及 `_select_v4_sparse_impl, wq_b_kv_insert, __init__`；`vllm/models/deepseek_v4/amd/rocm.py` renamed +47/-59 (106 lines); hunks: -8,14 +8,15; -31,7 +32,9; symbols: _build_indptr_from_lengths, build, DeepseekV4ROCMAiterMLASparseImpl, DeepseekV4ROCMAiterMLASparseBackend，涉及 `_build_indptr_from_lengths, build, DeepseekV4ROCMAiterMLASparseImpl`；`vllm/models/deepseek_v4/nvidia/model.py` modified +1/-1 (2 lines); hunks: -56,7 +56,7。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/flashmla.py` added +402/-0 (402 lines); hunks: -0,0 +1,402; symbols: DeepseekV4SparseMLAAttentionImpl, forward_mqa, DeepseekV4FlashMLASparseBackend, get_supported_kernel_block_sizes
  - `vllm/models/deepseek_v4/nvidia/ops/attention.py` renamed +23/-309 (332 lines); hunks: -20,9 +20,6; -62,28 +59,36; symbols: _select_v4_sparse_impl, wq_b_kv_insert, __init__, get_attn_backend
  - `vllm/models/deepseek_v4/amd/rocm.py` renamed +47/-59 (106 lines); hunks: -8,14 +8,15; -31,7 +32,9; symbols: _build_indptr_from_lengths, build, DeepseekV4ROCMAiterMLASparseImpl, DeepseekV4ROCMAiterMLASparseBackend
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-1 (2 lines); hunks: -56,7 +56,7
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/flashmla.py` added +402/-0; `vllm/models/deepseek_v4/nvidia/ops/attention.py` renamed +23/-309; `vllm/models/deepseek_v4/amd/rocm.py` renamed +47/-59; `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`, `tests/kernels/test_fused_inv_rope_fp8_quant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42353 - DSv4 fused Q-norm kernel grid refactor

- 链接: https://github.com/vllm-project/vllm/pull/42353
- 状态/时间: merged / 2026-05-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`；关联提交 `f743254143f2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+330/-216，可读 patch 670 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「DSv4 fused Q-norm kernel grid refactor」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`；技术摘要: 覆盖「DSv4 fused Q-norm kernel grid refactor」；主要实现面是 `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +27/-24 (51 lines); hunks: -67,29 +67,26 @@ def apply_rope_gptj_last_k(; -99,11 +96,15 @@ def apply_rope_gptj_last_k(; symbols: apply_rope_gptj_last_k, rmsnorm_no_weight, _call_fused, test_q_path_matches_reference，涉及 `apply_rope_gptj_last_k, rmsnorm_no_weight, _call_fused`。
- 代码 diff 细节:
  - `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +27/-24 (51 lines); hunks: -67,29 +67,26 @@ def apply_rope_gptj_last_k(; -99,11 +96,15 @@ def apply_rope_gptj_last_k(; symbols: apply_rope_gptj_last_k, rmsnorm_no_weight, _call_fused, test_q_path_matches_reference
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +27/-24
- 验证与风险: diff 自带测试面 `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42950 - [XPU]fix: add XPU platform guards to DeepSeek-V4 ops

- 链接: https://github.com/vllm-project/vllm/pull/42950
- 状态/时间: merged / 2026-05-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/model.py`；关联提交 `8de5cabeb70d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+31/-18，可读 patch 133 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[XPU]fix: add XPU platform guards to DeepSeek-V4 ops」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/compressor.py`；技术摘要: 覆盖「[XPU]fix: add XPU platform guards to DeepSeek-V4 ops」；主要实现面是 `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/compressor.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-5 (10 lines); hunks: -1153,7 +1153,7 @@ def _forward_cuda(; -1193,8 +1193,8 @@ def forward(; symbols: _forward_cuda, _forward_rocm, _forward_native, forward，涉及 `_forward_cuda, _forward_rocm, _forward_native`；`vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +5/-1 (6 lines); hunks: -243,7 +243,11 @@ def _fused_inv_rope_fp8_quant_kernel_impl(; symbols: _fused_inv_rope_fp8_quant_kernel_impl，涉及 `_fused_inv_rope_fp8_quant_kernel_impl`；`vllm/models/deepseek_v4/compressor.py` modified +5/-1 (6 lines); hunks: -296,7 +296,11 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-5 (10 lines); hunks: -1153,7 +1153,7 @@ def _forward_cuda(; -1193,8 +1193,8 @@ def forward(; symbols: _forward_cuda, _forward_rocm, _forward_native, forward
  - `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +5/-1 (6 lines); hunks: -243,7 +243,11 @@ def _fused_inv_rope_fp8_quant_kernel_impl(; symbols: _fused_inv_rope_fp8_quant_kernel_impl
  - `vllm/models/deepseek_v4/compressor.py` modified +5/-1 (6 lines); hunks: -296,7 +296,11 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-5; `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +5/-1; `vllm/models/deepseek_v4/compressor.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/activation.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/compressor.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42925 - [DSV4] More multi-stream enablement for c4a

- 链接: https://github.com/vllm-project/vllm/pull/42925
- 状态/时间: merged / 2026-05-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/model.py`；关联提交 `367cb81966f9`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+61/-29，可读 patch 141 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] More multi-stream enablement for c4a」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `vllm/models/deepseek_v4/nvidia/model.py`；技术摘要: 覆盖「[DSV4] More multi-stream enablement for c4a」；主要实现面是 `vllm/models/deepseek_v4/nvidia/model.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/model.py` modified +7/-0 (7 lines); hunks: -935,6 +935,12 @@ def __init__(; -945,6 +951,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +7/-0 (7 lines); hunks: -935,6 +935,12 @@ def __init__(; -945,6 +951,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/ops/attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43385 - [ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP

- 链接: https://github.com/vllm-project/vllm/pull/43385
- 状态/时间: merged / 2026-05-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/amd/rocm.py`；关联提交 `1806d1adfc9b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+2340/-52，可读 patch 2496 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/amd/rocm.py`；技术摘要: 覆盖「[ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP」；主要实现面是 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/amd/rocm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/amd/model.py` added +1612/-0 (1612 lines); hunks: -0,0 +1,1612; symbols: DeepseekV4MLP, __init__, forward, _deepseek_v4_stage_mega_moe_inputs_kernel，涉及 `DeepseekV4MLP, __init__, forward`；`vllm/models/deepseek_v4/amd/mtp.py` added +520/-0 (520 lines); hunks: -0,0 +1,520; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor，涉及 `DeepSeekV4MultiTokenPredictorLayer, __init__, forward`；`vllm/models/deepseek_v4/amd/rocm.py` modified +134/-23 (157 lines); hunks: -44,6 +44,127 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> tor...; -704,38 +825,28 @@ def _forward_prefill(; symbols: _build_indptr_from_lengths, _combine_topk_swa_indices_kernel, combine_topk_swa_indices, _compute_topk_lens_kernel，涉及 `_build_indptr_from_lengths, _combine_topk_swa_indices_kernel, combine_topk_swa_indices`；`vllm/models/deepseek_v4/amd/model.py` removed +0/-1 (1 lines); hunks: -1 +0,0。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/amd/model.py` added +1612/-0 (1612 lines); hunks: -0,0 +1,1612; symbols: DeepseekV4MLP, __init__, forward, _deepseek_v4_stage_mega_moe_inputs_kernel
  - `vllm/models/deepseek_v4/amd/mtp.py` added +520/-0 (520 lines); hunks: -0,0 +1,520; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +134/-23 (157 lines); hunks: -44,6 +44,127 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> tor...; -704,38 +825,28 @@ def _forward_prefill(; symbols: _build_indptr_from_lengths, _combine_topk_swa_indices_kernel, combine_topk_swa_indices, _compute_topk_lens_kernel
  - `vllm/models/deepseek_v4/amd/model.py` removed +0/-1 (1 lines); hunks: -1 +0,0
  - `vllm/models/deepseek_v4/amd/mtp.py` removed +0/-1 (1 lines); hunks: -1 +0,0
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` added +1612/-0; `vllm/models/deepseek_v4/amd/mtp.py` added +520/-0; `vllm/models/deepseek_v4/amd/rocm.py` modified +134/-23; `vllm/models/deepseek_v4/amd/model.py` removed +0/-1; `vllm/models/deepseek_v4/amd/mtp.py` removed +0/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/model.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43632 - [DeepSeek V4] Move MegaMoE input prep kernel to nvidia/ops

- 链接: https://github.com/vllm-project/vllm/pull/43632
- 状态/时间: merged / 2026-05-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`；关联提交 `aa2b56ffb0c1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+177/-165，可读 patch 382 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek V4] Move MegaMoE input prep kernel to nvidia/ops」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `tests/models/test_deepseek_v4_mega_moe.py`；技术摘要: 覆盖「[DeepSeek V4] Move MegaMoE input prep kernel to nvidia/ops」；主要实现面是 `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `tests/models/test_deepseek_v4_mega_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` added +173/-0 (173 lines); hunks: -0,0 +1,173; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs，涉及 `_prepare_megamoe_inputs_kernel, prepare_megamoe_inputs`；`vllm/models/deepseek_v4/nvidia/model.py` modified +2/-163 (165 lines); hunks: -59,9 +59,9; -116,167 +116,6 @@ def forward(self, x):; symbols: forward, _deepseek_v4_stage_mega_moe_inputs_kernel, _stage_deepseek_v4_mega_moe_inputs, make_deepseek_v4_expert_params_mapping，涉及 `forward, _deepseek_v4_stage_mega_moe_inputs_kernel, _stage_deepseek_v4_mega_moe_inputs`；`tests/models/test_deepseek_v4_mega_moe.py` modified +2/-2 (4 lines); hunks: -8,9 +8,9; -164,7 +164,7 @@ def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise...; symbols: test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact，涉及 `test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` added +173/-0 (173 lines); hunks: -0,0 +1,173; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +2/-163 (165 lines); hunks: -59,9 +59,9; -116,167 +116,6 @@ def forward(self, x):; symbols: forward, _deepseek_v4_stage_mega_moe_inputs_kernel, _stage_deepseek_v4_mega_moe_inputs, make_deepseek_v4_expert_params_mapping
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +2/-2 (4 lines); hunks: -8,9 +8,9; -164,7 +164,7 @@ def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise...; symbols: test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` added +173/-0; `vllm/models/deepseek_v4/nvidia/model.py` modified +2/-163
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/models/test_deepseek_v4_mega_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43162 - [Feat][DSV4] Fuse q pad into deepseek v4 fused kernel

- 链接: https://github.com/vllm-project/vllm/pull/43162
- 状态/时间: merged / 2026-05-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py` 等 6 个文件；关联提交 `6ab6ffb428be`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+339/-151，可读 patch 888 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feat][DSV4] Fuse q pad into deepseek v4 fused kernel」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/amd/rocm.py`；技术摘要: 覆盖「[Feat][DSV4] Fuse q pad into deepseek v4 fused kernel」；主要实现面是 `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/amd/rocm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/attention.py` renamed +23/-40 (63 lines); hunks: -156,18 +156,6 @@ def __init__(; -263,6 +251,9 @@ def __init__(; symbols: __init__, attention_impl, wq_b_kv_insert，涉及 `__init__, attention_impl, wq_b_kv_insert`；`vllm/models/deepseek_v4/nvidia/flashmla.py` modified +23/-1 (24 lines); hunks: -28,7 +28,7; -63,6 +63,18 @@ def forward_mqa( # type: ignore[override]; symbols: forward_mqa, get_padded_num_q_heads, DeepseekV4FlashMLASparseBackend, DeepseekV4FlashMLASparseImpl，涉及 `forward_mqa, get_padded_num_q_heads, DeepseekV4FlashMLASparseBackend`；`vllm/models/deepseek_v4/amd/rocm.py` modified +5/-1 (6 lines); hunks: -32,7 +32,7; -592,6 +592,10 @@ class DeepseekV4ROCMAiterMLASparseImpl(DeepseekV4SparseMLAA...; symbols: DeepseekV4ROCMAiterMLASparseImpl, get_padded_num_q_heads, forward_mqa，涉及 `DeepseekV4ROCMAiterMLASparseImpl, get_padded_num_q_heads, forward_mqa`；`vllm/models/deepseek_v4/amd/model.py` modified +1/-1 (2 lines); hunks: -53,7 +53,7。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/attention.py` renamed +23/-40 (63 lines); hunks: -156,18 +156,6 @@ def __init__(; -263,6 +251,9 @@ def __init__(; symbols: __init__, attention_impl, wq_b_kv_insert
  - `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +23/-1 (24 lines); hunks: -28,7 +28,7; -63,6 +63,18 @@ def forward_mqa( # type: ignore[override]; symbols: forward_mqa, get_padded_num_q_heads, DeepseekV4FlashMLASparseBackend, DeepseekV4FlashMLASparseImpl
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +5/-1 (6 lines); hunks: -32,7 +32,7; -592,6 +592,10 @@ class DeepseekV4ROCMAiterMLASparseImpl(DeepseekV4SparseMLAA...; symbols: DeepseekV4ROCMAiterMLASparseImpl, get_padded_num_q_heads, forward_mqa
  - `vllm/models/deepseek_v4/amd/model.py` modified +1/-1 (2 lines); hunks: -53,7 +53,7
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-1 (2 lines); hunks: -54,7 +54,7
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/attention.py` renamed +23/-40; `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +23/-1; `vllm/models/deepseek_v4/amd/rocm.py` modified +5/-1; `vllm/models/deepseek_v4/amd/model.py` modified +1/-1; `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-1
  - tests: `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +72/-17
- 验证与风险: diff 自带测试面 `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `tests/kernels/test_fused_inv_rope_fp8_quant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43629 - [ROCm] Remove MegaMoE integration in deepseek v4

- 链接: https://github.com/vllm-project/vllm/pull/43629
- 状态/时间: merged / 2026-05-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`；关联提交 `c8414a82712b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-645，可读 patch 793 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm] Remove MegaMoE integration in deepseek v4」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`；技术摘要: 覆盖「[ROCm] Remove MegaMoE integration in deepseek v4」；主要实现面是 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/amd/model.py` modified +2/-623 (625 lines); hunks: -11,17 +11,12; -52,16 +47,13; symbols: DeepseekV4MLP, forward, _deepseek_v4_stage_mega_moe_inputs_kernel, _stage_deepseek_v4_mega_moe_inputs，涉及 `DeepseekV4MLP, forward, _deepseek_v4_stage_mega_moe_inputs_kernel`；`vllm/models/deepseek_v4/amd/mtp.py` modified +8/-22 (30 lines); hunks: -40,10 +40,7; -330,19 +327,13 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx, finalize_mega_moe_weights, _rewrite_spec_layer_name，涉及 `_find_mtp_layer_idx, finalize_mega_moe_weights, _rewrite_spec_layer_name`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/amd/model.py` modified +2/-623 (625 lines); hunks: -11,17 +11,12; -52,16 +47,13; symbols: DeepseekV4MLP, forward, _deepseek_v4_stage_mega_moe_inputs_kernel, _stage_deepseek_v4_mega_moe_inputs
  - `vllm/models/deepseek_v4/amd/mtp.py` modified +8/-22 (30 lines); hunks: -40,10 +40,7; -330,19 +327,13 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx, finalize_mega_moe_weights, _rewrite_spec_layer_name
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` modified +2/-623; `vllm/models/deepseek_v4/amd/mtp.py` modified +8/-22
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43690 - [DSv4] Drop _get_compressed_kv_buffer in DeepseekCompressor

- 链接: https://github.com/vllm-project/vllm/pull/43690
- 状态/时间: merged / 2026-05-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/compressor.py`；关联提交 `193ce8812eb4`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-33，可读 patch 59 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSv4] Drop _get_compressed_kv_buffer in DeepseekCompressor」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/models/deepseek_v4/compressor.py`；技术摘要: 覆盖「[DSv4] Drop _get_compressed_kv_buffer in DeepseekCompressor」；主要实现面是 `vllm/models/deepseek_v4/compressor.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/compressor.py` modified +5/-33 (38 lines); hunks: -173,33 +173,6 @@ def get_attn_backend(self) -> type[AttentionBackend]:; -276,11 +249,6 @@ def __init__(; symbols: get_attn_backend, DeepseekCompressor, _get_compressed_kv_buffer, __init__，涉及 `get_attn_backend, DeepseekCompressor, _get_compressed_kv_buffer`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/compressor.py` modified +5/-33 (38 lines); hunks: -173,33 +173,6 @@ def get_attn_backend(self) -> type[AttentionBackend]:; -276,11 +249,6 @@ def __init__(; symbols: get_attn_backend, DeepseekCompressor, _get_compressed_kv_buffer, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/compressor.py` modified +5/-33
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/compressor.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43710 - [DSv4] Refactor compressor & Fix ROCm compatibility

- 链接: https://github.com/vllm-project/vllm/pull/43710
- 状态/时间: merged / 2026-05-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/common/ops/__init__.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/common/ops/save_partial_states.py` 等 9 个文件；关联提交 `adaa5e455ad8`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+364/-239，可读 patch 753 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSv4] Refactor compressor & Fix ROCm compatibility」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/common/ops/save_partial_states.py`；技术摘要: 覆盖「[DSv4] Refactor compressor & Fix ROCm compatibility」；主要实现面是 `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/common/ops/save_partial_states.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/compressor.py` modified +68/-198 (266 lines); hunks: -13,15 +13,13; -173,6 +171,16 @@ def get_attn_backend(self) -> type[AttentionBackend]:; symbols: get_attn_backend, DeepseekCompressor, __init__, forward，涉及 `get_attn_backend, DeepseekCompressor, __init__`；`vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +78/-34 (112 lines); hunks: -11,12 +11,6; -25,43 +19,93; symbols: _get_sparse_attn_cutedsl_impls, compress_norm_rope_store_triton, _compress_kv_sparse_attn_cutedsl, _norm_rope_insert_sparse_attn_cutedsl，涉及 `_get_sparse_attn_cutedsl_impls, compress_norm_rope_store_triton, _compress_kv_sparse_attn_cutedsl`；`vllm/models/deepseek_v4/common/ops/save_partial_states.py` added +101/-0 (101 lines); hunks: -0,0 +1,101; symbols: save_partial_states, _save_partial_states_kernel，涉及 `save_partial_states, _save_partial_states_kernel`；`vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` renamed +95/-3 (98 lines); hunks: -8,6 +8,7; -1086,7 +1087,7 @@ def compile(; symbols: compile, _compress_kv_sparse_attn_cutedsl, compress_kv_sparse_attn_cutedsl, _norm_rope_insert_sparse_attn_cutedsl，涉及 `compile, _compress_kv_sparse_attn_cutedsl, compress_kv_sparse_attn_cutedsl`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/compressor.py` modified +68/-198 (266 lines); hunks: -13,15 +13,13; -173,6 +171,16 @@ def get_attn_backend(self) -> type[AttentionBackend]:; symbols: get_attn_backend, DeepseekCompressor, __init__, forward
  - `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +78/-34 (112 lines); hunks: -11,12 +11,6; -25,43 +19,93; symbols: _get_sparse_attn_cutedsl_impls, compress_norm_rope_store_triton, _compress_kv_sparse_attn_cutedsl, _norm_rope_insert_sparse_attn_cutedsl
  - `vllm/models/deepseek_v4/common/ops/save_partial_states.py` added +101/-0 (101 lines); hunks: -0,0 +1,101; symbols: save_partial_states, _save_partial_states_kernel
  - `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` renamed +95/-3 (98 lines); hunks: -8,6 +8,7; -1086,7 +1087,7 @@ def compile(; symbols: compile, _compress_kv_sparse_attn_cutedsl, compress_kv_sparse_attn_cutedsl, _norm_rope_insert_sparse_attn_cutedsl
  - `vllm/models/deepseek_v4/nvidia/ops/__init__.py` modified +16/-0 (16 lines); hunks: -6,3 +6,19
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/compressor.py` modified +68/-198; `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` modified +78/-34; `vllm/models/deepseek_v4/common/ops/save_partial_states.py` added +101/-0; `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` renamed +95/-3; `vllm/models/deepseek_v4/nvidia/ops/__init__.py` modified +16/-0; `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/common/ops/__init__.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43679 - [ROCm][DSV4] Enable Tilelang MHC replacing torch/triton mhc

- 链接: https://github.com/vllm-project/vllm/pull/43679
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`；关联提交 `0ba46d4b11d2`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+716/-99，可读 patch 1234 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][DSV4] Enable Tilelang MHC replacing torch/triton mhc」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`；技术摘要: 覆盖「[ROCm][DSV4] Enable Tilelang MHC replacing torch/triton mhc」；主要实现面是 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/amd/model.py` modified +11/-7 (18 lines); hunks: -54,6 +54,7; -473,6 +474,7 @@ def __init__(; symbols: DeepseekV4MLP, __init__, hc_pre, hc_post，涉及 `DeepseekV4MLP, __init__, hc_pre`；`vllm/models/deepseek_v4/amd/mtp.py` modified +3/-1 (4 lines); hunks: -39,6 +39,7; -118,6 +119,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/amd/model.py` modified +11/-7 (18 lines); hunks: -54,6 +54,7; -473,6 +474,7 @@ def __init__(; symbols: DeepseekV4MLP, __init__, hc_pre, hc_post
  - `vllm/models/deepseek_v4/amd/mtp.py` modified +3/-1 (4 lines); hunks: -39,6 +39,7; -118,6 +119,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` modified +11/-7; `vllm/models/deepseek_v4/amd/mtp.py` modified +3/-1
- 验证与风险: diff 自带测试面 `requirements/test/rocm.in`, `requirements/test/rocm.txt`, `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43829 - [DSV4] Remove AMD/XPU path in deepseek_v4/nvidia

- 链接: https://github.com/vllm-project/vllm/pull/43829
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`；关联提交 `a04afd76aa91`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+8/-78，可读 patch 171 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Remove AMD/XPU path in deepseek_v4/nvidia」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`；技术摘要: 覆盖「[DSV4] Remove AMD/XPU path in deepseek_v4/nvidia」；主要实现面是 `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/model.py` modified +3/-64 (67 lines); hunks: -60,7 +60,6; -262,13 +261,7 @@ def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:; symbols: _ue8m0_uint8_to_float, _check_runtime_supported, hc_post, _forward_cuda，涉及 `_ue8m0_uint8_to_float, _check_runtime_supported, hc_post`；`vllm/models/deepseek_v4/nvidia/mtp.py` modified +5/-14 (19 lines); hunks: -37,7 +37,6; -147,10 +146,9 @@ def forward(; symbols: forward, __init__，涉及 `forward, __init__`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +3/-64 (67 lines); hunks: -60,7 +60,6; -262,13 +261,7 @@ def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:; symbols: _ue8m0_uint8_to_float, _check_runtime_supported, hc_post, _forward_cuda
  - `vllm/models/deepseek_v4/nvidia/mtp.py` modified +5/-14 (19 lines); hunks: -37,7 +37,6; -147,10 +146,9 @@ def forward(; symbols: forward, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +3/-64; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +5/-14
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43746 - [Model Refactoring] Remove torch compile dependency in DSv4

- 链接: https://github.com/vllm-project/vllm/pull/43746
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/common/ops/__init__.py`, `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py`, `vllm/models/deepseek_v4/nvidia/model.py` 等 7 个文件；关联提交 `04cec9e4d846`, `9957e4d240aa`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+270/-24，可读 patch 424 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model Refactoring] Remove torch compile dependency in DSv4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`；技术摘要: 覆盖「[Model Refactoring] Remove torch compile dependency in DSv4」；主要实现面是 `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: _rmsnorm_row, _fused_mtp_input_rmsnorm_kernel, _mtp_shared_head_rmsnorm_kernel, mtp_shared_head_rmsnorm，涉及 `_rmsnorm_row, _fused_mtp_input_rmsnorm_kernel, _mtp_shared_head_rmsnorm_kernel`；`vllm/models/deepseek_v4/amd/mtp.py` modified +21/-10 (31 lines); hunks: -18,7 +18,6; -37,6 +36,10; symbols: forward, compute_logits, DeepSeekV4MTP, __init__，涉及 `forward, compute_logits, DeepSeekV4MTP`；`vllm/models/deepseek_v4/nvidia/mtp.py` modified +21/-10 (31 lines); hunks: -18,7 +18,6; -37,6 +36,10; symbols: forward, compute_logits, DeepSeekV4MTP, __init__，涉及 `forward, compute_logits, DeepSeekV4MTP`；`vllm/models/deepseek_v4/common/ops/__init__.py` modified +3/-0 (3 lines); hunks: -9,6 +9,7; -19,7 +20,9。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: _rmsnorm_row, _fused_mtp_input_rmsnorm_kernel, _mtp_shared_head_rmsnorm_kernel, mtp_shared_head_rmsnorm
  - `vllm/models/deepseek_v4/amd/mtp.py` modified +21/-10 (31 lines); hunks: -18,7 +18,6; -37,6 +36,10; symbols: forward, compute_logits, DeepSeekV4MTP, __init__
  - `vllm/models/deepseek_v4/nvidia/mtp.py` modified +21/-10 (31 lines); hunks: -18,7 +18,6; -37,6 +36,10; symbols: forward, compute_logits, DeepSeekV4MTP, __init__
  - `vllm/models/deepseek_v4/common/ops/__init__.py` modified +3/-0 (3 lines); hunks: -9,6 +9,7; -19,7 +20,9
  - `vllm/models/deepseek_v4/amd/model.py` modified +0/-2 (2 lines); hunks: -8,7 +8,6; -605,7 +604,6 @@ def forward(; symbols: forward, DeepseekV4Model, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` added +203/-0; `vllm/models/deepseek_v4/amd/mtp.py` modified +21/-10; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +21/-10; `vllm/models/deepseek_v4/common/ops/__init__.py` modified +3/-0; `vllm/models/deepseek_v4/amd/model.py` modified +0/-2; `vllm/models/deepseek_v4/nvidia/model.py` modified +0/-2
- 验证与风险: runtime 路径改动集中在 `vllm/config/vllm.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43891 - [Model Refactoring] Remove unncessary torch op registration for DSv4

- 链接: https://github.com/vllm-project/vllm/pull/43891
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`；关联提交 `69b8956dcd5a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-110，可读 patch 188 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model Refactoring] Remove unncessary torch op registration for DSv4」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`；技术摘要: 覆盖「[Model Refactoring] Remove unncessary torch op registration for DSv4」；主要实现面是 `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/attention.py` modified +9/-59 (68 lines); hunks: -25,7 +25,6; -292,8 +291,10 @@ def forward(; symbols: forward, deepseek_v4_attention, deepseek_v4_attention_fake, deepseek_v4_fp8_einsum，涉及 `forward, deepseek_v4_attention, deepseek_v4_attention_fake`；`vllm/models/deepseek_v4/nvidia/model.py` modified +1/-51 (52 lines); hunks: -15,7 +15,6; -60,7 +59,6; symbols: DeepseekV4MLP, __init__, _map_global_expert_id, forward，涉及 `DeepseekV4MLP, __init__, _map_global_expert_id`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/attention.py` modified +9/-59 (68 lines); hunks: -25,7 +25,6; -292,8 +291,10 @@ def forward(; symbols: forward, deepseek_v4_attention, deepseek_v4_attention_fake, deepseek_v4_fp8_einsum
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-51 (52 lines); hunks: -15,7 +15,6; -60,7 +59,6; symbols: DeepseekV4MLP, __init__, _map_global_expert_id, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +9/-59; `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-51
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43905 - [DSv4] Move mHC tilelang kernels & Don't use CustomOP in dsv4/nvidia

- 链接: https://github.com/vllm-project/vllm/pull/43905
- 状态/时间: merged / 2026-05-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`；关联提交 `7bd45da5857d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+72/-102，可读 patch 380 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSv4] Move mHC tilelang kernels & Don't use CustomOP in dsv4/nvidia」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`；技术摘要: 覆盖「[DSv4] Move mHC tilelang kernels & Don't use CustomOP in dsv4/nvidia」；主要实现面是 `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/model.py` modified +20/-54 (74 lines); hunks: -15,6 +15,12; -28,12 +34,6; symbols: __init__, hc_pre, hc_post, forward，涉及 `__init__, hc_pre, hc_post`；`vllm/models/deepseek_v4/nvidia/mtp.py` modified +6/-7 (13 lines); hunks: -24,11 +24,14; -122,8 +125,6 @@ def __init__(; symbols: __init__, forward, compute_logits，涉及 `__init__, forward, compute_logits`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +20/-54 (74 lines); hunks: -15,6 +15,12; -28,12 +34,6; symbols: __init__, hc_pre, hc_post, forward
  - `vllm/models/deepseek_v4/nvidia/mtp.py` modified +6/-7 (13 lines); hunks: -24,11 +24,14; -122,8 +125,6 @@ def __init__(; symbols: __init__, forward, compute_logits
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +20/-54; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +6/-7
- 验证与风险: diff 自带测试面 `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #44161 - [Kernel][DSv4] Optimize sparse FP8 compressor kernels

- 链接: https://github.com/vllm-project/vllm/pull/44161
- 状态/时间: merged / 2026-06-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`；关联提交 `035733515f25`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+139/-91，可读 patch 312 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel][DSv4] Optimize sparse FP8 compressor kernels」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`；技术摘要: 覆盖「[Kernel][DSv4] Optimize sparse FP8 compressor kernels」；主要实现面是 `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +139/-91 (230 lines); hunks: -96,9 +96,16 @@ def __init__(; -156,8 +163,9 @@ def kernel(; symbols: __init__, kernel，涉及 `__init__, kernel`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +139/-91 (230 lines); hunks: -96,9 +96,16 @@ def __init__(; -156,8 +163,9 @@ def kernel(; symbols: __init__, kernel
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +139/-91
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44246 - [DSV4] Remove unncessary classes & functions

- 链接: https://github.com/vllm-project/vllm/pull/44246
- 状态/时间: merged / 2026-06-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`；关联提交 `8c3cc98cffd3`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+62/-124，可读 patch 362 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Remove unncessary classes & functions」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`；技术摘要: 覆盖「[DSV4] Remove unncessary classes & functions」；主要实现面是 `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/attention.py` modified +34/-88 (122 lines); hunks: -5,7 +5,6; -38,9 +37,8; symbols: _select_v4_sparse_impl, DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes，涉及 `_select_v4_sparse_impl, DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper`；`vllm/models/deepseek_v4/amd/model.py` modified +14/-18 (32 lines); hunks: -48,8 +48,7; -314,7 +313,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/models/deepseek_v4/nvidia/model.py` modified +14/-18 (32 lines); hunks: -54,8 +54,7; -697,7 +696,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/attention.py` modified +34/-88 (122 lines); hunks: -5,7 +5,6; -38,9 +37,8; symbols: _select_v4_sparse_impl, DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes
  - `vllm/models/deepseek_v4/amd/model.py` modified +14/-18 (32 lines); hunks: -48,8 +48,7; -314,7 +313,7 @@ def __init__(; symbols: __init__
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +14/-18 (32 lines); hunks: -54,8 +54,7; -697,7 +696,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +34/-88; `vllm/models/deepseek_v4/amd/model.py` modified +14/-18; `vllm/models/deepseek_v4/nvidia/model.py` modified +14/-18
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/model.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44262 - [DSV4] Refactor RoPE initialization

- 链接: https://github.com/vllm-project/vllm/pull/44262
- 状态/时间: merged / 2026-06-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/common/rope.py`, `vllm/models/deepseek_v4/nvidia/model.py`；关联提交 `517e74a9644f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+50/-40，可读 patch 133 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Refactor RoPE initialization」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `vllm/models/deepseek_v4/common/rope.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`；技术摘要: 覆盖「[DSV4] Refactor RoPE initialization」；主要实现面是 `vllm/models/deepseek_v4/common/rope.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/common/rope.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: build_deepseek_v4_rope，涉及 `build_deepseek_v4_rope`；`vllm/models/deepseek_v4/amd/model.py` modified +7/-20 (27 lines); hunks: -30,7 +30,6; -50,6 +49,7; symbols: __init__，涉及 `__init__`；`vllm/models/deepseek_v4/nvidia/model.py` modified +7/-20 (27 lines); hunks: -35,7 +35,6; -56,6 +55,7; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/common/rope.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: build_deepseek_v4_rope
  - `vllm/models/deepseek_v4/amd/model.py` modified +7/-20 (27 lines); hunks: -30,7 +30,6; -50,6 +49,7; symbols: __init__
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +7/-20 (27 lines); hunks: -35,7 +35,6; -56,6 +55,7; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/common/rope.py` added +36/-0; `vllm/models/deepseek_v4/amd/model.py` modified +7/-20; `vllm/models/deepseek_v4/nvidia/model.py` modified +7/-20
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/common/rope.py`, `vllm/models/deepseek_v4/nvidia/model.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43339 - [Feature] Support EPLB for DeepSeek v4 Mega Moe

- 链接: https://github.com/vllm-project/vllm/pull/43339
- 状态/时间: merged / 2026-06-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/model.py`；关联提交 `242709415287`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+232/-46，可读 patch 449 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Support EPLB for DeepSeek v4 Mega Moe」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `vllm/models/deepseek_v4/nvidia/model.py`；技术摘要: 覆盖「[Feature] Support EPLB for DeepSeek v4 Mega Moe」；主要实现面是 `vllm/models/deepseek_v4/nvidia/model.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/model.py` modified +211/-38 (249 lines); hunks: -1,7 +1,7; -15,6 +15,7; symbols: __init__, _map_global_expert_id，涉及 `__init__, _map_global_expert_id`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +211/-38 (249 lines); hunks: -1,7 +1,7; -15,6 +15,7; symbols: __init__, _map_global_expert_id
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +211/-38
- 验证与风险: runtime 路径改动集中在 `vllm/distributed/eplb/eplb_utils.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/utils/deep_gemm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44367 - [DSV4] Minor cleanup for DeepseekV4MegaMoEExperts

- 链接: https://github.com/vllm-project/vllm/pull/44367
- 状态/时间: merged / 2026-06-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/model.py`；关联提交 `b254e0456c98`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-18，可读 patch 34 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Minor cleanup for DeepseekV4MegaMoEExperts」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/models/deepseek_v4/nvidia/model.py`；技术摘要: 覆盖「[DSV4] Minor cleanup for DeepseekV4MegaMoEExperts」；主要实现面是 `vllm/models/deepseek_v4/nvidia/model.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-18 (19 lines); hunks: -420,25 +420,7 @@ def forward(; -484,6 +466,7 @@ def _run_mega_moe(; symbols: forward, _run_mega_moe，涉及 `forward, _run_mega_moe`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-18 (19 lines); hunks: -420,25 +420,7 @@ def forward(; -484,6 +466,7 @@ def _run_mega_moe(; symbols: forward, _run_mega_moe
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +1/-18
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/nvidia/model.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44356 - [Bugfix] Fix Deepseek v4 non-mega-moe model init error

- 链接: https://github.com/vllm-project/vllm/pull/44356
- 状态/时间: merged / 2026-06-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/model.py`；关联提交 `969aec4bc845`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-0，可读 patch 15 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Deepseek v4 non-mega-moe model init error」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/models/deepseek_v4/nvidia/model.py`；技术摘要: 覆盖「[Bugfix] Fix Deepseek v4 non-mega-moe model init error」；主要实现面是 `vllm/models/deepseek_v4/nvidia/model.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/model.py` modified +8/-0 (8 lines); hunks: -637,6 +637,14 @@ def _init_fused_moe_experts(; symbols: _init_fused_moe_experts，涉及 `_init_fused_moe_experts`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +8/-0 (8 lines); hunks: -637,6 +637,14 @@ def _init_fused_moe_experts(; symbols: _init_fused_moe_experts
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/nvidia/model.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44236 - fix: resolve CUTLASS fmin compatibility for DeepSeek-V4 init

- 链接: https://github.com/vllm-project/vllm/pull/44236
- 状态/时间: merged / 2026-06-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`；关联提交 `597bc1593635`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-4，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: resolve CUTLASS fmin compatibility for DeepSeek-V4 init」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`；技术摘要: 覆盖「fix: resolve CUTLASS fmin compatibility for DeepSeek-V4 init」；主要实现面是 `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +4/-4 (8 lines); hunks: -370,11 +370,11 @@ def kernel(; -1026,11 +1026,11 @@ def kernel(; symbols: kernel，涉及 `kernel`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +4/-4 (8 lines); hunks: -370,11 +370,11 @@ def kernel(; -1026,11 +1026,11 @@ def kernel(; symbols: kernel
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +4/-4
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43827 - [DSv4] Adding TRTLLM gen attention kernel

- 链接: https://github.com/vllm-project/vllm/pull/43827
- 状态/时间: merged / 2026-06-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/ops/__init__.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py` 等 9 个文件；关联提交 `b5235fca2eb7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+2971/-398，可读 patch 4003 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSv4] Adding TRTLLM gen attention kernel」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`；技术摘要: 覆盖「[DSv4] Adding TRTLLM gen attention kernel」；主要实现面是 `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +1102/-323 (1425 lines); hunks: -508,28 +508,34 @@ def compile(; -539,17 +545,31 @@ def __call__(; symbols: compile, SparseAttnCompressC128Block8Kernel, SparseAttnCompressNormRopeStoreFullC4Kernel, __init__，涉及 `compile, SparseAttnCompressC128Block8Kernel, SparseAttnCompressNormRopeStoreFullC4Kernel`；`vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` added +407/-0 (407 lines); hunks: -0,0 +1,407; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_name, get_impl_cls，涉及 `_get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_name`；`vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +305/-0 (305 lines); hunks: -592,3 +592,308 @@ def _combine_topk_swa_indices_kernel(; symbols: _combine_topk_swa_indices_kernel, build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel，涉及 `_combine_topk_swa_indices_kernel, build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel`；`vllm/models/deepseek_v4/attention.py` modified +141/-42 (183 lines); hunks: -55,9 +55,6; -73,21 +70,82; symbols: _select_v4_sparse_impl, _resolve_dsv4_backend, _resolve_dsv4_kv_cache_dtype, DeepseekV4MLA，涉及 `_select_v4_sparse_impl, _resolve_dsv4_backend, _resolve_dsv4_kv_cache_dtype`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +1102/-323 (1425 lines); hunks: -508,28 +508,34 @@ def compile(; -539,17 +545,31 @@ def __call__(; symbols: compile, SparseAttnCompressC128Block8Kernel, SparseAttnCompressNormRopeStoreFullC4Kernel, __init__
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` added +407/-0 (407 lines); hunks: -0,0 +1,407; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_name, get_impl_cls
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +305/-0 (305 lines); hunks: -592,3 +592,308 @@ def _combine_topk_swa_indices_kernel(; symbols: _combine_topk_swa_indices_kernel, build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel
  - `vllm/models/deepseek_v4/attention.py` modified +141/-42 (183 lines); hunks: -55,9 +55,6; -73,21 +70,82; symbols: _select_v4_sparse_impl, _resolve_dsv4_backend, _resolve_dsv4_kv_cache_dtype, DeepseekV4MLA
  - `vllm/models/deepseek_v4/compressor.py` modified +38/-19 (57 lines); hunks: -155,13 +155,17 @@ def __init__(; -333,26 +337,40 @@ def forward(; symbols: __init__, get_kv_cache_spec, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` modified +1102/-323; `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` added +407/-0; `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +305/-0; `vllm/models/deepseek_v4/attention.py` modified +141/-42; `vllm/models/deepseek_v4/compressor.py` modified +38/-19; `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +10/-1
- 验证与风险: diff 自带测试面 `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #44569 - [DSV4] Refactor DeepseekV4Attention

- 链接: https://github.com/vllm-project/vllm/pull/44569
- 状态/时间: merged / 2026-06-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py` 等 7 个文件；关联提交 `4efd6ffde094`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+521/-918，可读 patch 2210 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Refactor DeepseekV4Attention」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/model.py`；技术摘要: 覆盖「[DSV4] Refactor DeepseekV4Attention」；主要实现面是 `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/model.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/attention.py` modified +224/-345 (569 lines); hunks: -4,8 +4,9; -15,16 +16,16; symbols: _resolve_dsv4_backend, _select_v4_sparse_impl, _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention，涉及 `_resolve_dsv4_backend, _select_v4_sparse_impl, _resolve_dsv4_kv_cache_dtype`；`vllm/models/deepseek_v4/nvidia/flashmla.py` modified +72/-118 (190 lines); hunks: -1,22 +1,22; -28,63 +28,9; symbols: DeepseekV4SparseMLAAttentionImpl, forward_mqa, get_padded_num_q_heads, init_layer_buffers，涉及 `DeepseekV4SparseMLAAttentionImpl, forward_mqa, get_padded_num_q_heads`；`vllm/models/deepseek_v4/nvidia/model.py` modified +17/-163 (180 lines); hunks: -33,7 +33,6; -55,13 +54,14; symbols: DeepseekV4MLP, finalize_mega_moe_weights, DeepseekV4Attention, __init__，涉及 `DeepseekV4MLP, finalize_mega_moe_weights, DeepseekV4Attention`；`vllm/models/deepseek_v4/amd/model.py` modified +3/-161 (164 lines); hunks: -18,7 +18,6; -45,11 +44,7; symbols: forward, DeepseekV4Attention, __init__, DeepseekV4DecoderLayer，涉及 `forward, DeepseekV4Attention, __init__`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/attention.py` modified +224/-345 (569 lines); hunks: -4,8 +4,9; -15,16 +16,16; symbols: _resolve_dsv4_backend, _select_v4_sparse_impl, _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention
  - `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +72/-118 (190 lines); hunks: -1,22 +1,22; -28,63 +28,9; symbols: DeepseekV4SparseMLAAttentionImpl, forward_mqa, get_padded_num_q_heads, init_layer_buffers
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +17/-163 (180 lines); hunks: -33,7 +33,6; -55,13 +54,14; symbols: DeepseekV4MLP, finalize_mega_moe_weights, DeepseekV4Attention, __init__
  - `vllm/models/deepseek_v4/amd/model.py` modified +3/-161 (164 lines); hunks: -18,7 +18,6; -45,11 +44,7; symbols: forward, DeepseekV4Attention, __init__, DeepseekV4DecoderLayer
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +65/-68 (133 lines); hunks: -2,15 +2,15; -26,16 +26,12; symbols: _build_indptr_from_lengths, get_name, get_builder_cls, get_impl_cls
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +224/-345; `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +72/-118; `vllm/models/deepseek_v4/nvidia/model.py` modified +17/-163; `vllm/models/deepseek_v4/amd/model.py` modified +3/-161; `vllm/models/deepseek_v4/amd/rocm.py` modified +65/-68; `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +71/-62
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44561 - [DSV4] Move more ops out of eager breakpoint

- 链接: https://github.com/vllm-project/vllm/pull/44561
- 状态/时间: merged / 2026-06-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/attention.py`；关联提交 `02d2da0748a1`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+30/-14，可读 patch 67 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Move more ops out of eager breakpoint」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/models/deepseek_v4/attention.py`；技术摘要: 覆盖「[DSV4] Move more ops out of eager breakpoint」；主要实现面是 `vllm/models/deepseek_v4/attention.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/attention.py` modified +30/-14 (44 lines); hunks: -330,10 +330,34 @@ def forward(; -403,25 +427,17 @@ def fused_wqa_wkv() -> torch.Tensor:; symbols: forward, fused_wqa_wkv, attention_impl，涉及 `forward, fused_wqa_wkv, attention_impl`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/attention.py` modified +30/-14 (44 lines); hunks: -330,10 +330,34 @@ def forward(; -403,25 +427,17 @@ def fused_wqa_wkv() -> torch.Tensor:; symbols: forward, fused_wqa_wkv, attention_impl
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +30/-14
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44699 - [DSV4] Decouple DS V4 Sparse MLA Metadata from DS V3.2

- 链接: https://github.com/vllm-project/vllm/pull/44699
- 状态/时间: merged / 2026-06-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/sparse_mla.py`；关联提交 `2a983c79acdb`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+449/-333，可读 patch 984 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Decouple DS V4 Sparse MLA Metadata from DS V3.2」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/sparse_mla.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`；技术摘要: 覆盖「[DSV4] Decouple DS V4 Sparse MLA Metadata from DS V3.2」；主要实现面是 `vllm/models/deepseek_v4/sparse_mla.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/sparse_mla.py` added +416/-0 (416 lines); hunks: -0,0 +1,416; symbols: DeepseekV4FlashMLABackend, get_supported_kernel_block_sizes, get_name, get_builder_cls，涉及 `DeepseekV4FlashMLABackend, get_supported_kernel_block_sizes, get_name`；`vllm/models/deepseek_v4/nvidia/flashmla.py` modified +7/-39 (46 lines); hunks: -16,10 +16,9; -31,41 +30,10; symbols: DeepseekV4FlashMLASparseBackend, get_supported_kernel_block_sizes, get_name, get_supported_head_sizes，涉及 `DeepseekV4FlashMLASparseBackend, get_supported_kernel_block_sizes, get_name`；`vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +13/-10 (23 lines); hunks: -18,13 +18,15; -47,13 +49,14 @@ def _get_flashinfer_dsv4_workspace(device: torch.device) ->...; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, forward_mqa, _build_sparse_index_metadata，涉及 `_get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, forward_mqa`；`vllm/models/deepseek_v4/amd/rocm.py` modified +8/-10 (18 lines); hunks: -9,17 +9,15; -445,7 +443,7 @@ def _copy_ragged_to_graph_buffers(; symbols: _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata, DeepseekV4ROCMAiterSparseSWAMetadata, DeepseekV4ROCMAiterMLASparseMetadataBuilder，涉及 `_copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata, DeepseekV4ROCMAiterSparseSWAMetadata`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/sparse_mla.py` added +416/-0 (416 lines); hunks: -0,0 +1,416; symbols: DeepseekV4FlashMLABackend, get_supported_kernel_block_sizes, get_name, get_builder_cls
  - `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +7/-39 (46 lines); hunks: -16,10 +16,9; -31,41 +30,10; symbols: DeepseekV4FlashMLASparseBackend, get_supported_kernel_block_sizes, get_name, get_supported_head_sizes
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +13/-10 (23 lines); hunks: -18,13 +18,15; -47,13 +49,14 @@ def _get_flashinfer_dsv4_workspace(device: torch.device) ->...; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, forward_mqa, _build_sparse_index_metadata
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +8/-10 (18 lines); hunks: -9,17 +9,15; -445,7 +443,7 @@ def _copy_ragged_to_graph_buffers(; symbols: _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterMLASparseMetadata, DeepseekV4ROCMAiterSparseSWAMetadata, DeepseekV4ROCMAiterMLASparseMetadataBuilder
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/sparse_mla.py` added +416/-0; `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +7/-39; `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +13/-10; `vllm/models/deepseek_v4/amd/rocm.py` modified +8/-10
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/nvidia/flashmla.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42953 - feat: add DeepSeek-V4 XPU attention decode path

- 链接: https://github.com/vllm-project/vllm/pull/42953
- 状态/时间: merged / 2026-06-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/xpu/__init__.py`, `vllm/models/deepseek_v4/xpu/model.py`, `vllm/models/deepseek_v4/xpu/mtp.py` 等 8 个文件；关联提交 `eebce65756f0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+2759/-11，可读 patch 2844 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: add DeepSeek-V4 XPU attention decode path」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `vllm/models/deepseek_v4/xpu/model.py`, `vllm/models/deepseek_v4/xpu/mtp.py`, `vllm/models/deepseek_v4/xpu/xpu_sparse.py`；技术摘要: 覆盖「feat: add DeepSeek-V4 XPU attention decode path」；主要实现面是 `vllm/models/deepseek_v4/xpu/model.py`, `vllm/models/deepseek_v4/xpu/mtp.py`, `vllm/models/deepseek_v4/xpu/xpu_sparse.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/xpu/model.py` added +1340/-0 (1340 lines); hunks: -0,0 +1,1340; symbols: DeepseekV4MLP, __init__, forward, _deepseek_v4_stage_mega_moe_inputs_kernel，涉及 `DeepseekV4MLP, __init__, forward`；`vllm/models/deepseek_v4/xpu/mtp.py` added +511/-0 (511 lines); hunks: -0,0 +1,511; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor，涉及 `DeepSeekV4MultiTokenPredictorLayer, __init__, forward`；`vllm/models/deepseek_v4/xpu/xpu_sparse.py` added +350/-0 (350 lines); hunks: -0,0 +1,350; symbols: DeepseekV4XPUSparseBackend, get_name, DeepseekV4XPUAttention, __init__，涉及 `DeepseekV4XPUSparseBackend, get_name, DeepseekV4XPUAttention`；`vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` added +290/-0 (290 lines); hunks: -0,0 +1,290; symbols: _dequant_gather_slots_kernel, dequant_gather_slots, xpu_sparse_decode_fp8，涉及 `_dequant_gather_slots_kernel, dequant_gather_slots, xpu_sparse_decode_fp8`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/xpu/model.py` added +1340/-0 (1340 lines); hunks: -0,0 +1,1340; symbols: DeepseekV4MLP, __init__, forward, _deepseek_v4_stage_mega_moe_inputs_kernel
  - `vllm/models/deepseek_v4/xpu/mtp.py` added +511/-0 (511 lines); hunks: -0,0 +1,511; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor
  - `vllm/models/deepseek_v4/xpu/xpu_sparse.py` added +350/-0 (350 lines); hunks: -0,0 +1,350; symbols: DeepseekV4XPUSparseBackend, get_name, DeepseekV4XPUAttention, __init__
  - `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` added +290/-0 (290 lines); hunks: -0,0 +1,290; symbols: _dequant_gather_slots_kernel, dequant_gather_slots, xpu_sparse_decode_fp8
  - `vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py` added +159/-0 (159 lines); hunks: -0,0 +1,159; symbols: _xpu_qnorm_rope_kernel, xpu_qnorm_rope_kv_fp8_insert
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/xpu/model.py` added +1340/-0; `vllm/models/deepseek_v4/xpu/mtp.py` added +511/-0; `vllm/models/deepseek_v4/xpu/xpu_sparse.py` added +350/-0; `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` added +290/-0; `vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py` added +159/-0; `vllm/models/deepseek_v4/__init__.py` modified +9/-8
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/kernels/linear/scaled_mm/xpu.py`, `vllm/model_executor/layers/mhc.py`, `vllm/models/deepseek_v4/__init__.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44144 - [DSV4][XPU] Add MHC fused_post_pre support

- 链接: https://github.com/vllm-project/vllm/pull/44144
- 状态/时间: merged / 2026-06-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/xpu/model.py`；关联提交 `70db1488c5d5`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+112/-17，可读 patch 168 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4][XPU] Add MHC fused_post_pre support」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/xpu/model.py`；技术摘要: 覆盖「[DSV4][XPU] Add MHC fused_post_pre support」；主要实现面是 `vllm/models/deepseek_v4/xpu/model.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/xpu/model.py` modified +40/-14 (54 lines); hunks: -930,22 +930,48 @@ def forward(; -1096,6 +1122,10 @@ def forward(; symbols: forward, load_weights，涉及 `forward, load_weights`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/xpu/model.py` modified +40/-14 (54 lines); hunks: -930,22 +930,48 @@ def forward(; -1096,6 +1122,10 @@ def forward(; symbols: forward, load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/xpu/model.py` modified +40/-14
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mhc.py`, `vllm/models/deepseek_v4/xpu/model.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44914 - [Bug] Fix deepseek v4 OOM issue

- 链接: https://github.com/vllm-project/vllm/pull/44914
- 状态/时间: merged / 2026-06-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/quant_config.py`；关联提交 `d7607ad2730f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-3，可读 patch 33 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug] Fix deepseek v4 OOM issue」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/models/deepseek_v4/quant_config.py`；技术摘要: 覆盖「[Bug] Fix deepseek v4 OOM issue」；主要实现面是 `vllm/models/deepseek_v4/quant_config.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/quant_config.py` modified +10/-3 (13 lines); hunks: -7,7 +7,11; -129,7 +133,7 @@ def override_quantization_method(; symbols: override_quantization_method, get_quant_method, is_mxfp4_quant，涉及 `override_quantization_method, get_quant_method, is_mxfp4_quant`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/quant_config.py` modified +10/-3 (13 lines); hunks: -7,7 +7,11; -129,7 +133,7 @@ def override_quantization_method(; symbols: override_quantization_method, get_quant_method, is_mxfp4_quant
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/quant_config.py` modified +10/-3
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/quant_config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44821 - fix: prefix DeepSeek V4 MTP projections

- 链接: https://github.com/vllm-project/vllm/pull/44821
- 状态/时间: merged / 2026-06-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`, `vllm/models/deepseek_v4/xpu/mtp.py`；关联提交 `04cec9e4d846`, `4673ca1d7869`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-0，可读 patch 32 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: prefix DeepSeek V4 MTP projections」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`；技术摘要: 覆盖「fix: prefix DeepSeek V4 MTP projections」；主要实现面是 `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-0 (2 lines); hunks: -86,13 +86,15 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/models/deepseek_v4/nvidia/mtp.py` modified +2/-0 (2 lines); hunks: -92,13 +92,15 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-0 (2 lines); hunks: -86,13 +86,15 @@ def __init__(; symbols: __init__
  - `vllm/models/deepseek_v4/nvidia/mtp.py` modified +2/-0 (2 lines); hunks: -92,13 +92,15 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-0; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #45240 - [XPU][DeepSeek-V4] Fix MTP: sync with upstream fixes #44821 and #43746

- 链接: https://github.com/vllm-project/vllm/pull/45240
- 状态/时间: merged / 2026-06-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/xpu/mtp.py`；关联提交 `04cec9e4d846`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+29/-18，可读 patch 119 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[XPU][DeepSeek-V4] Fix MTP: sync with upstream fixes #44821 and #43746」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/models/deepseek_v4/xpu/mtp.py`；技术摘要: 覆盖「[XPU][DeepSeek-V4] Fix MTP: sync with upstream fixes #44821 and #43746」；主要实现面是 `vllm/models/deepseek_v4/xpu/mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/xpu/mtp.py` modified +29/-18 (47 lines); hunks: -18,7 +18,6; -39,6 +38,10; symbols: __init__, forward, compute_logits, DeepSeekV4MTP，涉及 `__init__, forward, compute_logits`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/xpu/mtp.py` modified +29/-18 (47 lines); hunks: -18,7 +18,6; -39,6 +38,10; symbols: __init__, forward, compute_logits, DeepSeekV4MTP
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/xpu/mtp.py` modified +29/-18
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/xpu/mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #45061 - [Perf] Optimize DSv4 prefill chunk planning, 4.0% E2E Throughput Improvement

- 链接: https://github.com/vllm-project/vllm/pull/45061
- 状态/时间: merged / 2026-06-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/flashmla.py`；关联提交 `e18fe932ca61`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+133/-23，可读 patch 272 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf] Optimize DSv4 prefill chunk planning, 4.0% E2E Throughput Improvement」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/nvidia/flashmla.py`；技术摘要: 覆盖「[Perf] Optimize DSv4 prefill chunk planning, 4.0% E2E Throughput Improvement」；主要实现面是 `vllm/models/deepseek_v4/nvidia/flashmla.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +12/-20 (32 lines); hunks: -246,7 +246,6 @@ def _forward_prefill(; -274,29 +273,22 @@ def _forward_prefill(; symbols: _forward_prefill，涉及 `_forward_prefill`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +12/-20 (32 lines); hunks: -246,7 +246,6 @@ def _forward_prefill(; -274,29 +273,22 @@ def _forward_prefill(; symbols: _forward_prefill
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/flashmla.py` modified +12/-20
- 验证与风险: diff 自带测试面 `tests/kernels/attention/test_flashmla_sparse.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #44892 - [DSV4][Minor] Fix supported KV cache dtypes

- 链接: https://github.com/vllm-project/vllm/pull/44892
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/sparse_mla.py`；关联提交 `f4359a70f9e0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+8/-8，可读 patch 44 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4][Minor] Fix supported KV cache dtypes」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/sparse_mla.py`；技术摘要: 覆盖「[DSV4][Minor] Fix supported KV cache dtypes」；主要实现面是 `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/sparse_mla.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +6/-5 (11 lines); hunks: -13,6 +13,7; -52,13 +53,13 @@ def _get_flashinfer_dsv4_workspace(device: torch.device) ->...; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_name，涉及 `_get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_name`；`vllm/models/deepseek_v4/sparse_mla.py` modified +0/-1 (1 lines); hunks: -46,7 +46,6 @@ class DeepseekV4FlashMLABackend(AttentionBackend):; symbols: DeepseekV4FlashMLABackend，涉及 `DeepseekV4FlashMLABackend`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +6/-5 (11 lines); hunks: -13,6 +13,7; -52,13 +53,13 @@ def _get_flashinfer_dsv4_workspace(device: torch.device) ->...; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_name
  - `vllm/models/deepseek_v4/sparse_mla.py` modified +0/-1 (1 lines); hunks: -46,7 +46,6 @@ class DeepseekV4FlashMLABackend(AttentionBackend):; symbols: DeepseekV4FlashMLABackend
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +6/-5; `vllm/models/deepseek_v4/sparse_mla.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v4/sparse_mla.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #45863 - [DSv4 Perf] DSv4 flashinfer sparse index cache for metadata, 2%~4% TTFT improvement

- 链接: https://github.com/vllm-project/vllm/pull/45863
- 状态/时间: merged / 2026-06-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`；关联提交 `0a7bacdcacc5`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+184/-18，可读 patch 227 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSv4 Perf] DSv4 flashinfer sparse index cache for metadata, 2%~4% TTFT improvement」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`；技术摘要: 覆盖「[DSv4 Perf] DSv4 flashinfer sparse index cache for metadata, 2%~4% TTFT improvement」；主要实现面是 `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +33/-17 (50 lines); hunks: -288,24 +288,40 @@ def _build_sparse_index_metadata(; symbols: _build_sparse_index_metadata, _forward，涉及 `_build_sparse_index_metadata, _forward`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +33/-17 (50 lines); hunks: -288,24 +288,40 @@ def _build_sparse_index_metadata(; symbols: _build_sparse_index_metadata, _forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +33/-17
- 验证与风险: diff 自带测试面 `tests/kernels/attention/test_flashmla_sparse.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45309 - [DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`, 26.8% ~ 27.9% E2E TTFT improvement

- 链接: https://github.com/vllm-project/vllm/pull/45309
- 状态/时间: merged / 2026-06-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/attention.py`；关联提交 `1797576237cc`, `2a47a9ff0f4f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+37/-30，可读 patch 104 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`, 26.8% ~ 27.9% E2E TTFT improvement」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/attention.py`；技术摘要: 覆盖「[DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`, 26.8% ~ 27.9% E2E TTFT improvement」；主要实现面是 `vllm/models/deepseek_v4/attention.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/attention.py` modified +37/-30 (67 lines); hunks: -14,7 +14,7; -331,8 +331,8 @@ def forward(; symbols: forward, fused_wqa_wkv, attention_impl，涉及 `forward, fused_wqa_wkv, attention_impl`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/attention.py` modified +37/-30 (67 lines); hunks: -14,7 +14,7; -331,8 +331,8 @@ def forward(; symbols: forward, fused_wqa_wkv, attention_impl
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +37/-30
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #45972 - Revert "[DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`" (#45309)

- 链接: https://github.com/vllm-project/vllm/pull/45972
- 状态/时间: merged / 2026-06-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/attention.py`；关联提交 `1797576237cc`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+30/-37，可读 patch 104 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "[DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`" (#45309)」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/attention.py`；技术摘要: 覆盖「Revert "[DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`" (#45309)」；主要实现面是 `vllm/models/deepseek_v4/attention.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/attention.py` modified +30/-37 (67 lines); hunks: -14,7 +14,7; -331,8 +331,8 @@ def forward(; symbols: forward, fused_wqa_wkv, attention_impl，涉及 `forward, fused_wqa_wkv, attention_impl`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/attention.py` modified +30/-37 (67 lines); hunks: -14,7 +14,7; -331,8 +331,8 @@ def forward(; symbols: forward, fused_wqa_wkv, attention_impl
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/attention.py` modified +30/-37
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #45681 - [ROCm][DSv4] Functional fixes for DeepSeek V4 on MI300X/MI325X

- 链接: https://github.com/vllm-project/vllm/pull/45681
- 状态/时间: merged / 2026-06-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`；关联提交 `afdcbd5d39ea`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+545/-52，可读 patch 953 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][DSv4] Functional fixes for DeepSeek V4 on MI300X/MI325X」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`；技术摘要: 覆盖「[ROCm][DSv4] Functional fixes for DeepSeek V4 on MI300X/MI325X」；主要实现面是 `vllm/models/deepseek_v4/common/ops/cache_utils.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +49/-6 (55 lines); hunks: -16,6 +16,10; -39,6 +43,7 @@ def quantize_and_insert_k_kernel(; symbols: quantize_and_insert_k_kernel, quantize_and_insert_k_cache，涉及 `quantize_and_insert_k_kernel, quantize_and_insert_k_cache`；`vllm/models/deepseek_v4/amd/rocm.py` modified +4/-0 (4 lines); hunks: -14,6 +14,7; -796,6 +797,7 @@ def _forward_prefill(; symbols: _forward_prefill，涉及 `_forward_prefill`；`vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +3/-1 (4 lines); hunks: -3,7 +3,9；`tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +147/-24 (171 lines); hunks: -19,17 +19,28; -81,10 +92,11 @@ def apply_rope_gptj_last_k(; symbols: apply_rope_gptj_last_k, _call_fused, _bf16_ulp_distance, key，涉及 `apply_rope_gptj_last_k, _call_fused, _bf16_ulp_distance`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +49/-6 (55 lines); hunks: -16,6 +16,10; -39,6 +43,7 @@ def quantize_and_insert_k_kernel(; symbols: quantize_and_insert_k_kernel, quantize_and_insert_k_cache
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +4/-0 (4 lines); hunks: -14,6 +14,7; -796,6 +797,7 @@ def _forward_prefill(; symbols: _forward_prefill
  - `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +3/-1 (4 lines); hunks: -3,7 +3,9
  - `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +147/-24 (171 lines); hunks: -19,17 +19,28; -81,10 +92,11 @@ def apply_rope_gptj_last_k(; symbols: apply_rope_gptj_last_k, _call_fused, _bf16_ulp_distance, key
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/common/ops/cache_utils.py` modified +49/-6; `vllm/models/deepseek_v4/amd/rocm.py` modified +4/-0; `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +3/-1
  - tests: `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +147/-24
- 验证与风险: diff 自带测试面 `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #46001 - [DeepSeek-V4] Support TEP=16 for the block-FP8 shared expert

- 链接: https://github.com/vllm-project/vllm/pull/46001
- 状态/时间: merged / 2026-06-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/model.py`；关联提交 `2a6c6b94293e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+44/-0，可读 patch 80 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek-V4] Support TEP=16 for the block-FP8 shared expert」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/nvidia/model.py`；技术摘要: 覆盖「[DeepSeek-V4] Support TEP=16 for the block-FP8 shared expert」；主要实现面是 `vllm/models/deepseek_v4/nvidia/model.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/model.py` modified +44/-0 (44 lines); hunks: -64,6 +64,7; -85,6 +86,15 @@ def __init__(; symbols: __init__, load_weights, _pad_shared_expert_weight，涉及 `__init__, load_weights, _pad_shared_expert_weight`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +44/-0 (44 lines); hunks: -64,6 +64,7; -85,6 +86,15 @@ def __init__(; symbols: __init__, load_weights, _pad_shared_expert_weight
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +44/-0
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/nvidia/model.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #45931 - [ROCm][DSV4] Disable TileLang MHC dispatch on gfx942

- 链接: https://github.com/vllm-project/vllm/pull/45931
- 状态/时间: merged / 2026-06-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`；关联提交 `89accad2cc96`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+56/-22，可读 patch 207 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][DSV4] Disable TileLang MHC dispatch on gfx942」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`；技术摘要: 覆盖「[ROCm][DSV4] Disable TileLang MHC dispatch on gfx942」；主要实现面是 `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/amd/model.py` modified +3/-3 (6 lines); hunks: -27,6 +27,7; -51,7 +52,6; symbols: DeepseekV4MLP, __init__, hc_pre，涉及 `DeepseekV4MLP, __init__, hc_pre`；`vllm/models/deepseek_v4/amd/mtp.py` modified +2/-3 (5 lines); hunks: -28,7 +28,7; -42,7 +42,6; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/amd/model.py` modified +3/-3 (6 lines); hunks: -27,6 +27,7; -51,7 +52,6; symbols: DeepseekV4MLP, __init__, hc_pre
  - `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-3 (5 lines); hunks: -28,7 +28,7; -42,7 +42,6; symbols: __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` modified +3/-3; `vllm/models/deepseek_v4/amd/mtp.py` modified +2/-3
- 验证与风险: diff 自带测试面 `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43477 - Enable DeepSeek V4 and GLM-5.1 on SM120

- 链接: https://github.com/vllm-project/vllm/pull/43477
- 状态/时间: merged / 2026-06-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py`, `vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` 等 7 个文件；关联提交 `44d95069e9d6`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 37 个文件，+2340/-469，可读 patch 3895 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable DeepSeek V4 and GLM-5.1 on SM120」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py`, `vllm/models/deepseek_v4/attention.py`；技术摘要: 覆盖「Enable DeepSeek V4 and GLM-5.1 on SM120」；主要实现面是 `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`, `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py`, `vllm/models/deepseek_v4/attention.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +473/-27 (500 lines); hunks: -1,13 +1,6; -18,6 +11,7; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_supported_kernel_block_sizes, get_name，涉及 `_get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_supported_kernel_block_sizes`；`vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py` added +226/-0 (226 lines); hunks: -0,0 +1,226; symbols: _compute_mhc_pre_num_split, _normalize_token_sizes, _select_mhc_warmup_token_sizes, _find_first_mhc_layer，涉及 `_compute_mhc_pre_num_split, _normalize_token_sizes, _select_mhc_warmup_token_sizes`；`vllm/models/deepseek_v4/attention.py` modified +34/-32 (66 lines); hunks: -62,23 +62,22; -100,18 +99,20 @@ class DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC):; symbols: _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention, dispatches, _o_proj，涉及 `_resolve_dsv4_kv_cache_dtype, DeepseekV4Attention, dispatches`；`vllm/model_executor/layers/attention/mla_attention.py` modified +29/-14 (43 lines); hunks: -208,6 +208,7; -319,6 +320,22 @@ def _detect_output_quant_key(; symbols: _detect_output_quant_key, _canonicalize_sparse_mla_kv_cache_dtype, MLAAttention, __init__，涉及 `_detect_output_quant_key, _canonicalize_sparse_mla_kv_cache_dtype, MLAAttention`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +473/-27 (500 lines); hunks: -1,13 +1,6; -18,6 +11,7; symbols: _get_flashinfer_dsv4_workspace, DeepseekV4FlashInferMLASparseBackend, get_supported_kernel_block_sizes, get_name
  - `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py` added +226/-0 (226 lines); hunks: -0,0 +1,226; symbols: _compute_mhc_pre_num_split, _normalize_token_sizes, _select_mhc_warmup_token_sizes, _find_first_mhc_layer
  - `vllm/models/deepseek_v4/attention.py` modified +34/-32 (66 lines); hunks: -62,23 +62,22; -100,18 +99,20 @@ class DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC):; symbols: _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention, dispatches, _o_proj
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +29/-14 (43 lines); hunks: -208,6 +208,7; -319,6 +320,22 @@ def _detect_output_quant_key(; symbols: _detect_output_quant_key, _canonicalize_sparse_mla_kv_cache_dtype, MLAAttention, __init__
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +27/-5 (32 lines); hunks: -60,9 +60,11; -736,14 +738,34 @@ def finalize_mega_moe_weights(self) -> None:; symbols: finalize_mega_moe_weights, _select_dsv4_attn_cls, for
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` modified +473/-27; `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py` added +226/-0; `vllm/models/deepseek_v4/attention.py` modified +34/-32; `vllm/model_executor/layers/attention/mla_attention.py` modified +29/-14; `vllm/models/deepseek_v4/nvidia/model.py` modified +27/-5; `vllm/models/deepseek_v4/compressor.py` modified +9/-9
- 验证与风险: diff 自带测试面 `tests/model_executor/test_flashinfer_autotune_cache.py`, `tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py`, `tests/v1/attention/test_sparse_mla_backends.py`, `tests/v1/spec_decode/test_acceptance_length.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #46428 - [Optimization] Skip DP padding tokens in MoE

- 链接: https://github.com/vllm-project/vllm/pull/46428
- 状态/时间: merged / 2026-06-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+154/-1，可读 patch 395 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Optimization] Skip DP padding tokens in MoE」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/model_executor/layers/fused_moe/modular_kernel.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`；技术摘要: 覆盖「[Optimization] Skip DP padding tokens in MoE」；主要实现面是 `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/model_executor/layers/fused_moe/modular_kernel.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/test_deepseek_v4_mega_moe.py` modified +80/-1 (81 lines); hunks: -46,7 +46,8 @@ def test_deepseek_v4_mega_moe_ue8m0_uint8_to_float():; -182,3 +183,81 @@ def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwis...; symbols: test_deepseek_v4_mega_moe_ue8m0_uint8_to_float, test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, test_deepseek_v4_mega_moe_fused_input_staging_masks_padding，涉及 `test_deepseek_v4_mega_moe_ue8m0_uint8_to_float, test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact`；`vllm/model_executor/layers/fused_moe/modular_kernel.py` modified +17/-0 (17 lines); hunks: -10,6 +10,7; -1133,6 +1134,22 @@ def _prepare(; symbols: _prepare，涉及 `_prepare`；`vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +11/-0 (11 lines); hunks: -19,6 +19,7 @@ def _prepare_megamoe_inputs_kernel(; -31,6 +32,7 @@ def _prepare_megamoe_inputs_kernel(; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs，涉及 `_prepare_megamoe_inputs_kernel, prepare_megamoe_inputs`；`vllm/models/deepseek_v4/nvidia/model.py` modified +10/-0 (10 lines); hunks: -8,6 +8,7; -16,6 +17,7; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +80/-1 (81 lines); hunks: -46,7 +46,8 @@ def test_deepseek_v4_mega_moe_ue8m0_uint8_to_float():; -182,3 +183,81 @@ def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwis...; symbols: test_deepseek_v4_mega_moe_ue8m0_uint8_to_float, test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, test_deepseek_v4_mega_moe_fused_input_staging_masks_padding
  - `vllm/model_executor/layers/fused_moe/modular_kernel.py` modified +17/-0 (17 lines); hunks: -10,6 +10,7; -1133,6 +1134,22 @@ def _prepare(; symbols: _prepare
  - `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +11/-0 (11 lines); hunks: -19,6 +19,7 @@ def _prepare_megamoe_inputs_kernel(; -31,6 +32,7 @@ def _prepare_megamoe_inputs_kernel(; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +10/-0 (10 lines); hunks: -8,6 +8,7; -16,6 +17,7; symbols: forward
  - `vllm/v1/worker/gpu/model_runner.py` modified +10/-0 (10 lines); hunks: -27,6 +27,7; -847,6 +848,13 @@ def prepare_inputs(; symbols: prepare_inputs, execute_model
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +80/-1
  - runtime: `vllm/model_executor/layers/fused_moe/modular_kernel.py` modified +17/-0; `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +11/-0; `vllm/models/deepseek_v4/nvidia/model.py` modified +10/-0; `vllm/v1/worker/gpu/model_runner.py` modified +10/-0; `vllm/forward_context.py` modified +9/-0; `vllm/v1/worker/gpu/input_batch.py` modified +7/-0
- 验证与风险: diff 自带测试面 `tests/models/test_deepseek_v4_mega_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40811 - [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4

- 链接: https://github.com/vllm-project/vllm/pull/40811
- 状态/时间: closed / 2026-06-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+777/-347，可读 patch 1666 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `csrc/persistent_topk.cuh`；技术摘要: 覆盖「[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4」；主要实现面是 `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `csrc/persistent_topk.cuh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/sparse_attn_indexer.py` modified +6/-0 (6 lines); hunks: -98,6 +98,7 @@ def sparse_attn_indexer(; -227,6 +228,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, __init__, forward_cuda，涉及 `sparse_attn_indexer, __init__, forward_cuda`；`vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-0 (1 lines); hunks: -1051,6 +1051,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`csrc/persistent_topk.cuh` modified +623/-232 (855 lines); hunks: -6,10 +6,12; -58,6 +60,76 @@ __device__ __forceinline__ auto convert_to_uint8(float x) ->...；`csrc/topk.cu` modified +143/-115 (258 lines); hunks: -1,5 +1,4; -13,131 +12,158。
- 代码 diff 细节:
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +6/-0 (6 lines); hunks: -98,6 +98,7 @@ def sparse_attn_indexer(; -227,6 +228,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, __init__, forward_cuda
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-0 (1 lines); hunks: -1051,6 +1051,7 @@ def __init__(; symbols: __init__, forward
  - `csrc/persistent_topk.cuh` modified +623/-232 (855 lines); hunks: -6,10 +6,12; -58,6 +60,76 @@ __device__ __forceinline__ auto convert_to_uint8(float x) ->...
  - `csrc/topk.cu` modified +143/-115 (258 lines); hunks: -1,5 +1,4; -13,131 +12,158
  - `vllm/utils/deep_gemm.py` modified +4/-0 (4 lines); hunks: -345,6 +345,7 @@ def fp8_fp4_mqa_logits(; -380,6 +381,7 @@ def fp8_fp4_mqa_logits(; symbols: fp8_fp4_mqa_logits, fp8_fp4_paged_mqa_logits
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/sparse_attn_indexer.py` modified +6/-0; `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-0; `vllm/utils/deep_gemm.py` modified +4/-0
  - other: `csrc/persistent_topk.cuh` modified +623/-232; `csrc/topk.cu` modified +143/-115
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/utils/deep_gemm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
