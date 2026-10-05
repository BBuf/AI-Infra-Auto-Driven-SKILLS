# vLLM Kimi K2/K2.5/K3/Linear/VL Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `benchmarks/kernels/benchmark_kimi_k3_attn_res.py` | [#54261](https://github.com/vllm-project/vllm/pull/54261) |
| `benchmarks/kernels/benchmark_kimi_k3_kda_decode.py` | [#50654](https://github.com/vllm-project/vllm/pull/50654), [#53396](https://github.com/vllm-project/vllm/pull/53396) |
| `benchmarks/kernels/benchmark_kimi_k3_kda_projection.py` | [#54697](https://github.com/vllm-project/vllm/pull/54697) |
| `benchmarks/kernels/benchmark_kimi_k3_latent_moe_tail.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#53152](https://github.com/vllm-project/vllm/pull/53152) |
| `benchmarks/kernels/benchmark_kimi_k3_sp_collectives.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089) |
| `tests/distributed/test_kimi_linear_context_parallel.py` | [#50484](https://github.com/vllm-project/vllm/pull/50484), [#50493](https://github.com/vllm-project/vllm/pull/50493) |
| `tests/evals/gsm8k/configs/Kimi-K3-pruned75-DSpark-AITER-TP4.yaml` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/Kimi-K3-pruned75-DSpark-TP4.yaml` | [#54817](https://github.com/vllm-project/vllm/pull/54817) |
| `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#51772](https://github.com/vllm-project/vllm/pull/51772), [#54896](https://github.com/vllm-project/vllm/pull/54896) |
| `tests/kernels/attention/test_kimi_k3_mla_key_concat_kv_cache.py` | [#50484](https://github.com/vllm-project/vllm/pull/50484) |
| `tests/models/kimi_k3/__init__.py` | [#51253](https://github.com/vllm-project/vllm/pull/51253) |
| `tests/models/kimi_k3/test_amd_attn_res.py` | [#50090](https://github.com/vllm-project/vllm/pull/50090), [#50593](https://github.com/vllm-project/vllm/pull/50593) |
| `tests/models/kimi_k3/test_amd_kda_checkpoint.py` | [#58344](https://github.com/vllm-project/vllm/pull/58344) |
| `tests/models/kimi_k3/test_amd_kda_chunk.py` | [#52606](https://github.com/vllm-project/vllm/pull/52606), [#53294](https://github.com/vllm-project/vllm/pull/53294), [#54038](https://github.com/vllm-project/vllm/pull/54038), [#58344](https://github.com/vllm-project/vllm/pull/58344) |
| `tests/models/kimi_k3/test_amd_kda_decode.py` | [#50654](https://github.com/vllm-project/vllm/pull/50654) |
| `tests/models/kimi_k3/test_amd_kda_direct_return.py` | [#50592](https://github.com/vllm-project/vllm/pull/50592) |
| `tests/models/kimi_k3/test_amd_latent_moe_runner.py` | [#51253](https://github.com/vllm-project/vllm/pull/51253), [#54956](https://github.com/vllm-project/vllm/pull/54956) |
| `tests/models/kimi_k3/test_amd_mla_direct_return.py` | [#50592](https://github.com/vllm-project/vllm/pull/50592) |
| `tests/models/kimi_k3/test_attn_res.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#50090](https://github.com/vllm-project/vllm/pull/50090), [#54261](https://github.com/vllm-project/vllm/pull/54261), [#58012](https://github.com/vllm-project/vllm/pull/58012) |
| `tests/models/kimi_k3/test_aux_attn_res_stream.py` | [#50487](https://github.com/vllm-project/vllm/pull/50487) |
| `tests/models/kimi_k3/test_eagle3.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#50487](https://github.com/vllm-project/vllm/pull/50487), [#52171](https://github.com/vllm-project/vllm/pull/52171), [#54482](https://github.com/vllm-project/vllm/pull/54482) |
| `tests/models/kimi_k3/test_gfx942_int4.py` | [#51274](https://github.com/vllm-project/vllm/pull/51274) |
| `tests/models/kimi_k3/test_kda.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#51682](https://github.com/vllm-project/vllm/pull/51682), [#51809](https://github.com/vllm-project/vllm/pull/51809), [#51855](https://github.com/vllm-project/vllm/pull/51855), [#53396](https://github.com/vllm-project/vllm/pull/53396), [#53581](https://github.com/vllm-project/vllm/pull/53581), [#54255](https://github.com/vllm-project/vllm/pull/54255), [#54859](https://github.com/vllm-project/vllm/pull/54859), [#56159](https://github.com/vllm-project/vllm/pull/56159), [#58045](https://github.com/vllm-project/vllm/pull/58045) |
| `tests/models/kimi_k3/test_kda_metadata.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#51483](https://github.com/vllm-project/vllm/pull/51483), [#51855](https://github.com/vllm-project/vllm/pull/51855), [#52388](https://github.com/vllm-project/vllm/pull/52388), [#53614](https://github.com/vllm-project/vllm/pull/53614), [#53766](https://github.com/vllm-project/vllm/pull/53766), [#54781](https://github.com/vllm-project/vllm/pull/54781), [#56159](https://github.com/vllm-project/vllm/pull/56159), [#58045](https://github.com/vllm-project/vllm/pull/58045) |
| `tests/models/kimi_k3/test_latent_moe_tail.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#53152](https://github.com/vllm-project/vllm/pull/53152), [#53327](https://github.com/vllm-project/vllm/pull/53327), [#54168](https://github.com/vllm-project/vllm/pull/54168) |
| `tests/models/kimi_k3/test_mla_prefill_context.py` | [#51772](https://github.com/vllm-project/vllm/pull/51772) |
| `tests/models/kimi_k3/test_nvidia_kda_direct_return.py` | [#50592](https://github.com/vllm-project/vllm/pull/50592) |
| `tests/models/kimi_k3/test_prefix_cache.py` | [#59229](https://github.com/vllm-project/vllm/pull/59229) |
| `tests/models/kimi_k3/test_sequence_parallel.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#50656](https://github.com/vllm-project/vllm/pull/50656), [#50912](https://github.com/vllm-project/vllm/pull/50912), [#52079](https://github.com/vllm-project/vllm/pull/52079) |
| `tests/models/kimi_k3/test_weight_loading.py` | [#53379](https://github.com/vllm-project/vllm/pull/53379) |
| `tests/reasoning/test_kimi_k2_reasoning_parser.py` | [#37438](https://github.com/vllm-project/vllm/pull/37438), [#41068](https://github.com/vllm-project/vllm/pull/41068), [#46610](https://github.com/vllm-project/vllm/pull/46610) |
| `tests/reasoning/test_kimi_k3_reasoning_parser.py` | [#50093](https://github.com/vllm-project/vllm/pull/50093), [#50886](https://github.com/vllm-project/vllm/pull/50886), [#58372](https://github.com/vllm-project/vllm/pull/58372) |
| `tests/renderers/test_kimi_k3.py` | [#50093](https://github.com/vllm-project/vllm/pull/50093) |
| `tests/tool_parsers/test_kimi_k2_tool_parser.py` | [#31207](https://github.com/vllm-project/vllm/pull/31207), [#38579](https://github.com/vllm-project/vllm/pull/38579) |
| `tests/tool_parsers/test_kimi_k3_named_tool_choice.py` | [#50093](https://github.com/vllm-project/vllm/pull/50093) |
| `tests/tool_use/test_kimi_k3_tool_parser.py` | [#50093](https://github.com/vllm-project/vllm/pull/50093), [#50420](https://github.com/vllm-project/vllm/pull/50420) |
| `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` | [#50000](https://github.com/vllm-project/vllm/pull/50000), [#50592](https://github.com/vllm-project/vllm/pull/50592), [#51862](https://github.com/vllm-project/vllm/pull/51862) |
| `vllm/model_executor/models/kimi_audio.py` | [#36127](https://github.com/vllm-project/vllm/pull/36127), [#36903](https://github.com/vllm-project/vllm/pull/36903) |
| `vllm/model_executor/models/kimi_k25.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131), [#33320](https://github.com/vllm-project/vllm/pull/33320), [#33346](https://github.com/vllm-project/vllm/pull/33346), [#33562](https://github.com/vllm-project/vllm/pull/33562), [#33876](https://github.com/vllm-project/vllm/pull/33876), [#34427](https://github.com/vllm-project/vllm/pull/34427), [#34501](https://github.com/vllm-project/vllm/pull/34501), [#36192](https://github.com/vllm-project/vllm/pull/36192), [#36361](https://github.com/vllm-project/vllm/pull/36361), [#37693](https://github.com/vllm-project/vllm/pull/37693), [#39344](https://github.com/vllm-project/vllm/pull/39344), [#42869](https://github.com/vllm-project/vllm/pull/42869), ... (15 total) |
| `vllm/model_executor/models/kimi_k25_vit.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131), [#33346](https://github.com/vllm-project/vllm/pull/33346), [#34501](https://github.com/vllm-project/vllm/pull/34501), [#42081](https://github.com/vllm-project/vllm/pull/42081), [#44493](https://github.com/vllm-project/vllm/pull/44493), [#50000](https://github.com/vllm-project/vllm/pull/50000), [#50400](https://github.com/vllm-project/vllm/pull/50400), [#51196](https://github.com/vllm-project/vllm/pull/51196), [#58527](https://github.com/vllm-project/vllm/pull/58527), [#58651](https://github.com/vllm-project/vllm/pull/58651) |
| `vllm/model_executor/models/kimi_vl.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387), [#16833](https://github.com/vllm-project/vllm/pull/16833), [#17156](https://github.com/vllm-project/vllm/pull/17156), [#21769](https://github.com/vllm-project/vllm/pull/21769), [#23114](https://github.com/vllm-project/vllm/pull/23114), [#23817](https://github.com/vllm-project/vllm/pull/23817), [#31738](https://github.com/vllm-project/vllm/pull/31738), [#41992](https://github.com/vllm-project/vllm/pull/41992) |
| `vllm/model_executor/models/moonvit.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387), [#23817](https://github.com/vllm-project/vllm/pull/23817), [#29309](https://github.com/vllm-project/vllm/pull/29309), [#31738](https://github.com/vllm-project/vllm/pull/31738), [#41992](https://github.com/vllm-project/vllm/pull/41992) |
| `vllm/model_executor/warmup/kimi_k3_triton_warmup.py` | [#50000](https://github.com/vllm-project/vllm/pull/50000), [#59257](https://github.com/vllm-project/vllm/pull/59257) |
| `vllm/models/kimi_k3/__init__.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#50131](https://github.com/vllm-project/vllm/pull/50131), [#51529](https://github.com/vllm-project/vllm/pull/51529) |
| `vllm/models/kimi_k3/amd/__init__.py` | [#50090](https://github.com/vllm-project/vllm/pull/50090) |
| `vllm/models/kimi_k3/amd/kda.py` | [#50592](https://github.com/vllm-project/vllm/pull/50592), [#50649](https://github.com/vllm-project/vllm/pull/50649), [#50654](https://github.com/vllm-project/vllm/pull/50654), [#51862](https://github.com/vllm-project/vllm/pull/51862), [#52606](https://github.com/vllm-project/vllm/pull/52606), [#53294](https://github.com/vllm-project/vllm/pull/53294), [#53581](https://github.com/vllm-project/vllm/pull/53581), [#54038](https://github.com/vllm-project/vllm/pull/54038), [#58045](https://github.com/vllm-project/vllm/pull/58045), [#58344](https://github.com/vllm-project/vllm/pull/58344) |
| `vllm/models/kimi_k3/amd/kda_metadata.py` | [#51862](https://github.com/vllm-project/vllm/pull/51862), [#58344](https://github.com/vllm-project/vllm/pull/58344) |
| `vllm/models/kimi_k3/amd/latent_moe_runner.py` | [#51253](https://github.com/vllm-project/vllm/pull/51253), [#53152](https://github.com/vllm-project/vllm/pull/53152), [#54956](https://github.com/vllm-project/vllm/pull/54956) |
| `vllm/models/kimi_k3/amd/linear.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#50592](https://github.com/vllm-project/vllm/pull/50592), [#50593](https://github.com/vllm-project/vllm/pull/50593), [#50649](https://github.com/vllm-project/vllm/pull/50649), [#50761](https://github.com/vllm-project/vllm/pull/50761), [#51253](https://github.com/vllm-project/vllm/pull/51253), [#52494](https://github.com/vllm-project/vllm/pull/52494) |
| `vllm/models/kimi_k3/amd/mla.py` | [#52494](https://github.com/vllm-project/vllm/pull/52494), [#57640](https://github.com/vllm-project/vllm/pull/57640) |
| `vllm/models/kimi_k3/amd/model.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089) |
| `vllm/models/kimi_k3/amd/mtp.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089) |
| `vllm/models/kimi_k3/amd/ops/__init__.py` | [#50090](https://github.com/vllm-project/vllm/pull/50090), [#50131](https://github.com/vllm-project/vllm/pull/50131) |
| `vllm/models/kimi_k3/amd/ops/attn_res.py` | [#50090](https://github.com/vllm-project/vllm/pull/50090), [#50593](https://github.com/vllm-project/vllm/pull/50593) |
| `vllm/models/kimi_k3/amd/ops/kda_checkpoint.py` | [#58344](https://github.com/vllm-project/vllm/pull/58344) |
| `vllm/models/kimi_k3/amd/ops/kda_chunk.py` | [#52606](https://github.com/vllm-project/vllm/pull/52606), [#53294](https://github.com/vllm-project/vllm/pull/53294), [#54038](https://github.com/vllm-project/vllm/pull/54038), [#56526](https://github.com/vllm-project/vllm/pull/56526), [#58344](https://github.com/vllm-project/vllm/pull/58344) |
| `vllm/models/kimi_k3/amd/ops/kda_decode.py` | [#50654](https://github.com/vllm-project/vllm/pull/50654) |
| `vllm/models/kimi_k3/amd/ops/kda_prefill.py` | [#52606](https://github.com/vllm-project/vllm/pull/52606), [#53294](https://github.com/vllm-project/vllm/pull/53294), [#54038](https://github.com/vllm-project/vllm/pull/54038), [#58344](https://github.com/vllm-project/vllm/pull/58344) |
| `vllm/models/kimi_k3/amd/ops/third_party/__init__.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089) |
| `vllm/models/kimi_k3/amd/ops/third_party/kda/__init__.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#58045](https://github.com/vllm-project/vllm/pull/58045) |
| `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#50649](https://github.com/vllm-project/vllm/pull/50649), [#51862](https://github.com/vllm-project/vllm/pull/51862), [#58769](https://github.com/vllm-project/vllm/pull/58769) |
| `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#58769](https://github.com/vllm-project/vllm/pull/58769) |
| `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra_token_parallel.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#58769](https://github.com/vllm-project/vllm/pull/58769) |
| `vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#51682](https://github.com/vllm-project/vllm/pull/51682), [#58045](https://github.com/vllm-project/vllm/pull/58045) |
| `vllm/models/kimi_k3/common/__init__.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089) |
| `vllm/models/kimi_k3/common/mm_preprocess.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089) |
| `vllm/models/kimi_k3/common/mtp.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089) |
| `vllm/models/kimi_k3/nvidia/__init__.py` | [#50090](https://github.com/vllm-project/vllm/pull/50090) |
| `vllm/models/kimi_k3/nvidia/dspark_mla.py` | [#50000](https://github.com/vllm-project/vllm/pull/50000), [#50089](https://github.com/vllm-project/vllm/pull/50089), [#50242](https://github.com/vllm-project/vllm/pull/50242), [#50585](https://github.com/vllm-project/vllm/pull/50585), [#52988](https://github.com/vllm-project/vllm/pull/52988), [#55356](https://github.com/vllm-project/vllm/pull/55356), [#58814](https://github.com/vllm-project/vllm/pull/58814) |
| `vllm/models/kimi_k3/nvidia/kda.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#50262](https://github.com/vllm-project/vllm/pull/50262), [#51311](https://github.com/vllm-project/vllm/pull/51311), [#51855](https://github.com/vllm-project/vllm/pull/51855), [#52079](https://github.com/vllm-project/vllm/pull/52079), [#53053](https://github.com/vllm-project/vllm/pull/53053), [#53132](https://github.com/vllm-project/vllm/pull/53132), [#53396](https://github.com/vllm-project/vllm/pull/53396), [#53581](https://github.com/vllm-project/vllm/pull/53581), [#53614](https://github.com/vllm-project/vllm/pull/53614), [#54255](https://github.com/vllm-project/vllm/pull/54255), [#54697](https://github.com/vllm-project/vllm/pull/54697), ... (15 total) |
| `vllm/models/kimi_k3/nvidia/kda_metadata.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#51483](https://github.com/vllm-project/vllm/pull/51483), [#51855](https://github.com/vllm-project/vllm/pull/51855), [#52388](https://github.com/vllm-project/vllm/pull/52388), [#52988](https://github.com/vllm-project/vllm/pull/52988), [#53614](https://github.com/vllm-project/vllm/pull/53614), [#54781](https://github.com/vllm-project/vllm/pull/54781), [#56159](https://github.com/vllm-project/vllm/pull/56159) |
| `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` | [#50678](https://github.com/vllm-project/vllm/pull/50678), [#53152](https://github.com/vllm-project/vllm/pull/53152), [#53327](https://github.com/vllm-project/vllm/pull/53327) |
| `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#53534](https://github.com/vllm-project/vllm/pull/53534), [#53942](https://github.com/vllm-project/vllm/pull/53942), [#54088](https://github.com/vllm-project/vllm/pull/54088), [#54167](https://github.com/vllm-project/vllm/pull/54167), [#54565](https://github.com/vllm-project/vllm/pull/54565), [#54606](https://github.com/vllm-project/vllm/pull/54606), [#54697](https://github.com/vllm-project/vllm/pull/54697), [#55426](https://github.com/vllm-project/vllm/pull/55426) |
| `vllm/models/kimi_k3/nvidia/mla.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#50404](https://github.com/vllm-project/vllm/pull/50404), [#50484](https://github.com/vllm-project/vllm/pull/50484), [#51772](https://github.com/vllm-project/vllm/pull/51772), [#51860](https://github.com/vllm-project/vllm/pull/51860), [#52079](https://github.com/vllm-project/vllm/pull/52079), [#52188](https://github.com/vllm-project/vllm/pull/52188), [#53053](https://github.com/vllm-project/vllm/pull/53053), [#54015](https://github.com/vllm-project/vllm/pull/54015), [#55242](https://github.com/vllm-project/vllm/pull/55242) |
| `vllm/models/kimi_k3/nvidia/model.py` | [#50000](https://github.com/vllm-project/vllm/pull/50000), [#50089](https://github.com/vllm-project/vllm/pull/50089), [#50383](https://github.com/vllm-project/vllm/pull/50383), [#50487](https://github.com/vllm-project/vllm/pull/50487), [#50500](https://github.com/vllm-project/vllm/pull/50500), [#50592](https://github.com/vllm-project/vllm/pull/50592), [#50656](https://github.com/vllm-project/vllm/pull/50656), [#50678](https://github.com/vllm-project/vllm/pull/50678), [#50912](https://github.com/vllm-project/vllm/pull/50912), [#51070](https://github.com/vllm-project/vllm/pull/51070), [#51131](https://github.com/vllm-project/vllm/pull/51131), [#51146](https://github.com/vllm-project/vllm/pull/51146), ... (23 total) |
| `vllm/models/kimi_k3/nvidia/mtp.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089), [#53942](https://github.com/vllm-project/vllm/pull/53942), [#54015](https://github.com/vllm-project/vllm/pull/54015) |
| `vllm/models/kimi_k3/nvidia/ops/__init__.py` | [#50090](https://github.com/vllm-project/vllm/pull/50090) |
| `vllm/models/kimi_k3/nvidia/ops/attn_res.py` | [#50090](https://github.com/vllm-project/vllm/pull/50090), [#50567](https://github.com/vllm-project/vllm/pull/50567), [#54261](https://github.com/vllm-project/vllm/pull/54261) |
| `vllm/models/kimi_k3/nvidia/ops/cute_dsl/__init__.py` | [#50089](https://github.com/vllm-project/vllm/pull/50089) |
| `vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py` | [#54697](https://github.com/vllm-project/vllm/pull/54697), [#55426](https://github.com/vllm-project/vllm/pull/55426) |
| ... | 34 more files omitted from table; all were used for git tracing. |

## PR Coverage Summary

- Git-traced PRs: 131
- Extra PRs preserved from existing docs: 10
- Total PRs in this document: 141
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-04-14 | [#16387](https://github.com/vllm-project/vllm/pull/16387) | merged | [Model][VLM] Add Kimi-VL model support | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `vllm/transformers_utils/configs/kimi_vl.py` |
| 2025-04-18 | [#16833](https://github.com/vllm-project/vllm/pull/16833) | merged | [Misc] Clean up Kimi-VL | `vllm/model_executor/models/kimi_vl.py` |
| 2025-04-25 | [#17156](https://github.com/vllm-project/vllm/pull/17156) | merged | fix float16 support for kimi-vl | `vllm/model_executor/models/kimi_vl.py` |
| 2025-08-05 | [#21769](https://github.com/vllm-project/vllm/pull/21769) | merged | Migrate KimiVLImagePixelInputs to TensorSchema | `vllm/model_executor/models/kimi_vl.py` |
| 2025-08-19 | [#23114](https://github.com/vllm-project/vllm/pull/23114) | merged | [Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506 | `vllm/model_executor/models/kimi_vl.py` |
| 2025-09-01 | [#23817](https://github.com/vllm-project/vllm/pull/23817) | merged | [Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506 | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py` |
| 2025-10-30 | [#27809](https://github.com/vllm-project/vllm/pull/27809) | merged | [Model] Introduce Kimi Linear to vLLM | `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py` |
| 2025-10-31 | [#27834](https://github.com/vllm-project/vllm/pull/27834) | merged | [Kimi-Linear] Correct prefixes and add compatibility to AWQ quants | `vllm/model_executor/models/kimi_linear.py` |
| 2025-10-31 | [#27885](https://github.com/vllm-project/vllm/pull/27885) | merged | fix incorrect type annotation in KimiMLP | `vllm/model_executor/models/kimi_linear.py` |
| 2025-11-24 | [#29309](https://github.com/vllm-project/vllm/pull/29309) | merged | [XPU]fix Kimi-VL-A3B-thinking on xpu | `vllm/model_executor/models/moonvit.py` |
| 2025-12-15 | [#30125](https://github.com/vllm-project/vllm/pull/30125) | merged | [CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it. | `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `vllm/attention/layers/mm_encoder_attention.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-12-30 | [#31207](https://github.com/vllm-project/vllm/pull/31207) | merged | fix: update kimi k2 tool parser logic | `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py` |
| 2026-01-06 | [#31738](https://github.com/vllm-project/vllm/pull/31738) | merged | [Models]: Use `MMEncoderAttention` for MoonViT | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py` |
| 2026-01-27 | [#33131](https://github.com/vllm-project/vllm/pull/33131) | merged | [Models] Kimi-K2.5 | `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/configs/kimi_k25.py` |
| 2026-01-29 | [#33320](https://github.com/vllm-project/vllm/pull/33320) | merged | [Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d… | `vllm/model_executor/models/kimi_k25.py` |
| 2026-01-30 | [#33346](https://github.com/vllm-project/vllm/pull/33346) | merged | [Models] Refactor Kimi-K2.5 weight loading | `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py` |
| 2026-02-02 | [#33562](https://github.com/vllm-project/vllm/pull/33562) | merged | [Bugfix] Enable Kimi k25 processor test | `vllm/model_executor/models/kimi_k25.py` |
| 2026-02-05 | [#33876](https://github.com/vllm-project/vllm/pull/33876) | merged | [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading | `vllm/model_executor/models/kimi_k25.py` |
| 2026-02-13 | [#34427](https://github.com/vllm-project/vllm/pull/34427) | merged | [Bugfix] Delete unused redundant code in Kimi-K2.5 | `vllm/model_executor/models/kimi_k25.py` |
| 2026-02-13 | [#34501](https://github.com/vllm-project/vllm/pull/34501) | merged | [Bugfix] Add quant_config in ViT of Kimi-K2.5 | `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py` |
| 2026-02-27 | [#33646](https://github.com/vllm-project/vllm/pull/33646) | merged | [Bugfix] Handle case when kimi ends reasoning with a tool call | `vllm/reasoning/kimi_k2_reasoning_parser.py` |
| 2026-03-06 | [#36192](https://github.com/vllm-project/vllm/pull/36192) | merged | [Security] Respect user trust_remote_code setting in NemotronVL and KimiK25 | `vllm/model_executor/models/kimi_k25.py` |
| 2026-03-11 | [#36127](https://github.com/vllm-project/vllm/pull/36127) | merged | [Model] Add support for moonshotai/Kimi-Audio-7B-Instruct | `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py` |
| 2026-03-11 | [#36361](https://github.com/vllm-project/vllm/pull/36361) | merged | Kimi k2.5 MLA based eagle3 | `vllm/model_executor/models/kimi_k25.py` |
| 2026-03-14 | [#36903](https://github.com/vllm-project/vllm/pull/36903) | merged | [Misc] Clean up Kimi-audio whisper encoder loading | `vllm/model_executor/models/kimi_audio.py` |
| 2026-03-18 | [#37371](https://github.com/vllm-project/vllm/pull/37371) | merged | standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01 | `vllm/model_executor/models/kimi_linear.py` |
| 2026-03-19 | [#37438](https://github.com/vllm-project/vllm/pull/37438) | merged | [Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support | `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/entrypoints/chat_utils.py`, `vllm/entrypoints/openai/chat_completion/serving.py` |
| 2026-03-20 | [#37693](https://github.com/vllm-project/vllm/pull/37693) | merged | [Model] Update Kimi-K25 and Isaac processors to fit HF-style | `vllm/transformers_utils/processors/kimi_k25.py`, `vllm/model_executor/models/kimi_k25.py` |
| 2026-04-12 | [#39344](https://github.com/vllm-project/vllm/pull/39344) | merged | fix(kimi_k25): resolve media_placeholder_token_id from tokenizer | `vllm/model_executor/models/kimi_k25.py` |
| 2026-04-19 | [#38579](https://github.com/vllm-project/vllm/pull/38579) | merged | [Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping | `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py` |
| 2026-05-04 | [#41068](https://github.com/vllm-project/vllm/pull/41068) | merged | [Bugfix] KimiK2ReasoningParser: guard against buffered end-token in streaming | `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py` |
| 2026-05-11 | [#42081](https://github.com/vllm-project/vllm/pull/42081) | merged | [Bug] Fix kimi dtype issue with `mm_projector_forward` | `vllm/model_executor/models/kimi_k25_vit.py` |
| 2026-05-14 | [#41778](https://github.com/vllm-project/vllm/pull/41778) | merged | [MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell | `benchmarks/attention_benchmarks/configs/mla_prefill.yaml`, `benchmarks/attention_benchmarks/configs/mla_decode.yaml`, `vllm/model_executor/layers/attention/mla_attention.py` |
| 2026-05-18 | [#42869](https://github.com/vllm-project/vllm/pull/42869) | merged | [BugFix] Kimi-K2.5: skip vision tower dtype conversion when using quantization | `vllm/model_executor/models/kimi_k25.py` |
| 2026-05-22 | [#41126](https://github.com/vllm-project/vllm/pull/41126) | merged | [Attention] Mamba attention module refactor | `vllm/model_executor/models/olmo_hybrid.py`, `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` |
| 2026-05-29 | [#43857](https://github.com/vllm-project/vllm/pull/43857) | merged | Add vLLM library info to Hugging Face Hub requests | `vllm/model_executor/model_loader/weight_utils.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/tokenizers/grok2.py` |
| 2026-06-04 | [#44493](https://github.com/vllm-project/vllm/pull/44493) | merged | [Bugfix]Fix Kimi-K2.5 FlashInfer ViT metadata | `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py` |
| 2026-06-04 | [#44539](https://github.com/vllm-project/vllm/pull/44539) | merged | [mamba] unify KDA conv states into one cache to match 2-state SSM layout | `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `vllm/model_executor/models/kimi_linear.py` |
| 2026-06-12 | [#45003](https://github.com/vllm-project/vllm/pull/45003) | merged | [Frontend] Support strict mode for tool calling | `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py` |
| 2026-06-17 | [#41992](https://github.com/vllm-project/vllm/pull/41992) | merged | [MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py` |
| 2026-06-21 | [#45424](https://github.com/vllm-project/vllm/pull/45424) | merged | [Core] Ensure memory is pinned prior to async h2d copy | `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/pooler/seqwise/methods.py`, `vllm/multimodal/inputs.py` |
| 2026-06-30 | [#46610](https://github.com/vllm-project/vllm/pull/46610) | merged | [Frontend] Add Streaming Parser Engine and new Kimi k2.5/k2.6/k2.7 Parser | `vllm/tool_parsers/kimi_k2_tool_parser.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`, `tests/reasoning/test_kimi_k2_reasoning_parser.py` |
| 2026-07-06 | [#47416](https://github.com/vllm-project/vllm/pull/47416) | merged | [perf]Add fused Kimi image preprocessing | `vllm/transformers_utils/processors/kimi_k25_vision_fused.py`, `vllm/model_executor/models/kimi_k25.py` |
| 2026-07-28 | [#50090](https://github.com/vllm-project/vllm/pull/50090) | merged | [Kimi-K3] Add AttnRes kernels | `vllm/models/kimi_k3/nvidia/ops/attn_res.py`, `tests/models/kimi_k3/test_attn_res.py`, `vllm/models/kimi_k3/amd/ops/attn_res.py` |
| 2026-07-28 | [#50131](https://github.com/vllm-project/vllm/pull/50131) | merged | [Bugfix] Add missing `vllm/models/kimi_k3/__init__.py` | `vllm/models/kimi_k3/__init__.py`, `vllm/models/kimi_k3/amd/ops/__init__.py` |
| 2026-07-29 | [#50089](https://github.com/vllm-project/vllm/pull/50089) | merged | [Model] Add Kimi K3 support: model files and kernels [1/N] | `vllm/models/kimi_k3/amd/linear.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py`, `vllm/models/kimi_k3/nvidia/kda.py` |
| 2026-07-29 | [#50093](https://github.com/vllm-project/vllm/pull/50093) | merged | [Model] Add Kimi K3 support: Python frontend [2/2] | `tests/tool_use/test_kimi_k3_tool_parser.py`, `vllm/tool_parsers/kimi_k3_tool_parser.py`, `vllm/reasoning/kimi_k3_reasoning_parser.py` |
| 2026-07-29 | [#50262](https://github.com/vllm-project/vllm/pull/50262) | merged | [ROCm][CI] Fix Kimi K3 KDA on ROCm | `vllm/models/kimi_k3/nvidia/kda.py` |
| 2026-07-30 | [#50000](https://github.com/vllm-project/vllm/pull/50000) | merged | [New model] Kimi K3 | `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/models/kimi_k25_vit.py` |
| 2026-07-31 | [#50420](https://github.com/vllm-project/vllm/pull/50420) | merged | [Frontend][Bugfix] Use default tool call IDs for Kimi K3 for conversation-level uniqueness | `tests/tool_use/test_kimi_k3_tool_parser.py`, `vllm/tool_parsers/kimi_k3_tool_parser.py` |
| 2026-07-31 | [#50242](https://github.com/vllm-project/vllm/pull/50242) | merged | K3 DSpark AR fusion | `vllm/models/kimi_k3/nvidia/dspark_mla.py` |
| 2026-07-31 | [#50500](https://github.com/vllm-project/vllm/pull/50500) | merged | [Compressed-Tensors] Support Kimi-K3 quantized models | `vllm/models/kimi_k3/nvidia/model.py` |
| 2026-08-02 | [#50761](https://github.com/vllm-project/vllm/pull/50761) | merged | [ROCm][Bugfix][Kimi-K3] Preserve MoE correction bias in FP32 | `vllm/models/kimi_k3/amd/linear.py` |
| 2026-08-03 | [#50383](https://github.com/vllm-project/vllm/pull/50383) | merged | Shard the K3 Latent-MoE up-projection on large batches | `vllm/models/kimi_k3/nvidia/model.py` |
| 2026-08-03 | [#50678](https://github.com/vllm-project/vllm/pull/50678) | merged | K3: Move LatentMoERunner | `vllm/models/kimi_k3/nvidia/model.py`, `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` |
| 2026-08-03 | [#50656](https://github.com/vllm-project/vllm/pull/50656) | merged | [Kimi-K3] Add option to shard the shared expert instead of replicating | `tests/models/kimi_k3/test_sequence_parallel.py`, `vllm/models/kimi_k3/nvidia/model.py` |
| 2026-08-04 | [#50567](https://github.com/vllm-project/vllm/pull/50567) | merged | [Bugfix][Kimi-K3] Enforce packed rows and op availability in AttnRes dispatch | `vllm/models/kimi_k3/nvidia/ops/attn_res.py` |
| 2026-08-04 | [#50886](https://github.com/vllm-project/vllm/pull/50886) | merged | [Bugfix][Reasoning] kimi_k3: O(delta) reasoning-end check on the decode path | `tests/reasoning/test_kimi_k3_reasoning_parser.py`, `vllm/reasoning/kimi_k3_reasoning_parser.py` |
| 2026-08-04 | [#50593](https://github.com/vllm-project/vllm/pull/50593) | merged | [Kimi-K3][AMD] Fuse AttnRes state updates and norms | `vllm/models/kimi_k3/amd/ops/attn_res.py`, `tests/models/kimi_k3/test_amd_attn_res.py`, `vllm/models/kimi_k3/amd/linear.py` |
| 2026-08-04 | [#50929](https://github.com/vllm-project/vllm/pull/50929) | merged | [MM][CG] Support ViT full CUDA graph for Kimi-K2.5 | `vllm/model_executor/models/kimi_k25.py` |
| 2026-08-04 | [#50912](https://github.com/vllm-project/vllm/pull/50912) | merged | [Kimi K3 Perf] option to shard the shared expert for non mega case, 16.98 GiB memory/GPU saved | `tests/models/kimi_k3/test_sequence_parallel.py`, `vllm/models/kimi_k3/nvidia/model.py` |
| 2026-08-05 | [#50404](https://github.com/vllm-project/vllm/pull/50404) | merged | [Model] Fix Kimi-K3 MLA with disabled context parallelism | `vllm/models/kimi_k3/nvidia/mla.py` |
| 2026-08-05 | [#50649](https://github.com/vllm-project/vllm/pull/50649) | merged | [ROCm][Bugfix] Kimi-K3 Fix KDA NaN on mixed batches and racy autotune config | `vllm/models/kimi_k3/amd/kda.py`, `vllm/models/kimi_k3/amd/linear.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` |
| 2026-08-05 | [#51131](https://github.com/vllm-project/vllm/pull/51131) | merged | [BugFix][K3] Skip moe_intermediate padding when EP is enabled | `vllm/models/kimi_k3/nvidia/model.py` |
| 2026-08-05 | [#51070](https://github.com/vllm-project/vllm/pull/51070) | merged | [K3 Perf] Combine multiple all gather together for SP, 1.5~3x kernel level performance improvement | `vllm/models/kimi_k3/nvidia/model.py` |
| 2026-08-06 | [#51146](https://github.com/vllm-project/vllm/pull/51146) | merged | K3: remove the add operation for megamoe path | `vllm/models/kimi_k3/nvidia/model.py` |
| 2026-08-06 | [#51249](https://github.com/vllm-project/vllm/pull/51249) | merged | [Bugfix][Model] Add missing fused_qkv_a_proj to Kimi-Linear packed_modules_mapping | `vllm/models/kimi_k3/nvidia/model.py` |
| 2026-08-07 | [#51253](https://github.com/vllm-project/vllm/pull/51253) | merged | [ROCm][Perf] Kimi-K3 Shard Latent MoE up-projection for ROCm path | `tests/models/kimi_k3/test_amd_latent_moe_runner.py`, `vllm/models/kimi_k3/amd/latent_moe_runner.py`, `tests/models/kimi_k3/__init__.py` |
| 2026-08-07 | [#50585](https://github.com/vllm-project/vllm/pull/50585) | merged | [K3 Perf] Optimize k3 dspark fused kv, 4.5~4.6x kernel performance improvement | `vllm/models/kimi_k3/nvidia/dspark_mla.py`, `tests/models/test_dspark_mla.py` |
| 2026-08-08 | [#51196](https://github.com/vllm-project/vllm/pull/51196) | merged | [Kimi][MM] disable kimi_vit's dynamic torch.compile for TPU | `vllm/model_executor/models/kimi_k25_vit.py` |
| 2026-08-09 | [#51529](https://github.com/vllm-project/vllm/pull/51529) | merged | [K3] Allow tpu to import kimi_k3.common | `vllm/models/kimi_k3/__init__.py` |
| 2026-08-10 | [#51682](https://github.com/vllm-project/vllm/pull/51682) | merged | [Bugfix][Kimi-K3] Give the AMD packed KDA decode kernel the state-index stride | `tests/models/kimi_k3/test_kda.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py` |
| 2026-08-10 | [#50484](https://github.com/vllm-project/vllm/pull/50484) | merged | [Kimi-K3] DCP support | `vllm/models/kimi_k3/nvidia/mla.py`, `tests/kernels/attention/test_kimi_k3_mla_key_concat_kv_cache.py`, `tests/distributed/test_kimi_linear_context_parallel.py` |
| 2026-08-12 | [#50654](https://github.com/vllm-project/vllm/pull/50654) | merged | [ROCm][Perf] Kimi-K3 Fused kernel for KDA decode | `tests/models/kimi_k3/test_amd_kda_decode.py`, `vllm/models/kimi_k3/amd/ops/kda_decode.py`, `vllm/models/kimi_k3/amd/kda.py` |
| 2026-08-12 | [#51860](https://github.com/vllm-project/vllm/pull/51860) | merged | [ROCm][K3] Dequantize the fp8 decode query for MLA backends without quant-query support - TRITON_MLA | `vllm/models/kimi_k3/nvidia/mla.py` |
| 2026-08-12 | [#51311](https://github.com/vllm-project/vllm/pull/51311) | merged | [K3 Perf] Flash kda out kernel for prefill, 1.1~1.4x kernel performance improvement | `vllm/models/kimi_k3/nvidia/kda.py` |
| 2026-08-13 | [#51772](https://github.com/vllm-project/vllm/pull/51772) | merged | [Attention][MLA] Fuse Kimi-K3 chunked-context K/V packing | `tests/models/kimi_k3/test_mla_prefill_context.py`, `vllm/models/kimi_k3/nvidia/mla.py`, `vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py` |
| 2026-08-13 | [#51862](https://github.com/vllm-project/vllm/pull/51862) | merged | [ROCm][Perf] Kimi-K3 Remove prefill pipeline stall in chunk KDA | `vllm/models/kimi_k3/amd/kda_metadata.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` |
| 2026-08-13 | [#52079](https://github.com/vllm-project/vllm/pull/52079) | merged | [Kimi-K3] Add GEMM-RS for sequence parallelism | `vllm/models/kimi_k3/nvidia/model.py`, `vllm/models/kimi_k3/nvidia/mla.py`, `vllm/models/kimi_k3/nvidia/kda.py` |
| 2026-08-13 | [#52171](https://github.com/vllm-project/vllm/pull/52171) | merged | [Bugfix] Declare SupportsEagle3 on KimiLinearForCausalLM | `tests/models/kimi_k3/test_eagle3.py`, `vllm/models/kimi_k3/nvidia/model.py` |
| 2026-08-14 | [#50487](https://github.com/vllm-project/vllm/pull/50487) | merged | [Model][Spec Decode] Tap the pre-norm AttnRes mixture as the Kimi K3 DFlash aux state | `tests/models/kimi_k3/test_aux_attn_res_stream.py`, `vllm/models/kimi_k3/nvidia/model.py`, `tests/models/kimi_k3/test_eagle3.py` |
| 2026-08-15 | [#52445](https://github.com/vllm-project/vllm/pull/52445) | merged | [Bugfix][Model] Kimi-K3 MegaMoE: pass situ_beta/situ_linear_beta to fp8_fp4_mega_moe | `vllm/models/kimi_k3/nvidia/model.py` |
| 2026-08-17 | [#51809](https://github.com/vllm-project/vllm/pull/51809) | merged | [XPU] Enable Kimi K3 KDA kernel tests on XPU | `tests/models/kimi_k3/test_kda.py`, `vllm/model_executor/layers/mamba/ops/gather_initial_states.py` |
| 2026-08-17 | [#51855](https://github.com/vllm-project/vllm/pull/51855) | merged | [K3] support recoverssm for K3 | `vllm/models/kimi_k3/nvidia/ops/recoverssm.py`, `tests/models/kimi_k3/test_kda.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py` |
| 2026-08-17 | [#52188](https://github.com/vllm-project/vllm/pull/52188) | merged | [Spec decode] Support Kimi-K3 DCP with DSpark | `vllm/models/kimi_k3/nvidia/mla.py`, `vllm/v1/attention/backends/mla/flashinfer_mla.py`, `vllm/v1/attention/backends/mla/tokenspeed_mla.py` |
| 2026-08-18 | [#50493](https://github.com/vllm-project/vllm/pull/50493) | merged | [Kimi-K3] support DCP partial prefix cache hit | `tests/distributed/test_kimi_linear_context_parallel.py`, `vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/core/kv_cache_coordinator.py` |
| 2026-08-20 | [#50400](https://github.com/vllm-project/vllm/pull/50400) | merged | [Kernel][Kimi] fused vision q/k roper kernel | `vllm/model_executor/models/kimi_k25_vit.py` |
| 2026-08-21 | [#53053](https://github.com/vllm-project/vllm/pull/53053) | merged | [Kimi-K3] Extend GEMM-RS to GEMM-AR | `vllm/models/kimi_k3/nvidia/model.py`, `vllm/models/kimi_k3/nvidia/kda.py`, `vllm/models/kimi_k3/nvidia/mla.py` |
| 2026-08-21 | [#53132](https://github.com/vllm-project/vllm/pull/53132) | merged | Support kimi k3 nvfp4 checkpoint | `vllm/models/kimi_k3/nvidia/kda.py` |
| 2026-08-21 | [#52606](https://github.com/vllm-project/vllm/pull/52606) | merged | [ROCm][Perf] Kimi-K3 Fused kernels for KDA prefill | `tests/models/kimi_k3/test_amd_kda_chunk.py`, `vllm/models/kimi_k3/amd/ops/kda_chunk.py`, `vllm/models/kimi_k3/amd/ops/kda_prefill.py` |
| 2026-08-21 | [#53152](https://github.com/vllm-project/vllm/pull/53152) | merged | [K3 Perf] Fuse MXFP4 top-k finalization into latent-tail, ~5% E2E latency reduction | `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py`, `tests/models/kimi_k3/test_latent_moe_tail.py`, `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` |
| 2026-08-21 | [#53294](https://github.com/vllm-project/vllm/pull/53294) | merged | Revert "[ROCm][Perf] Kimi-K3 Fused kernels for KDA prefill" | `tests/models/kimi_k3/test_amd_kda_chunk.py`, `vllm/models/kimi_k3/amd/ops/kda_chunk.py`, `vllm/models/kimi_k3/amd/ops/kda_prefill.py` |
| 2026-08-22 | [#53327](https://github.com/vllm-project/vllm/pull/53327) | merged | [Bugfix][Kimi K3] Enable deferred MoE finalization before weight loading | `tests/models/kimi_k3/test_latent_moe_tail.py`, `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` |
| 2026-08-24 | [#53581](https://github.com/vllm-project/vllm/pull/53581) | merged | [Bugfix][Kimi K3] Skip absent metadata during CUDA graph profiling | `tests/models/kimi_k3/test_kda.py`, `vllm/models/kimi_k3/amd/kda.py`, `vllm/models/kimi_k3/nvidia/kda.py` |
| 2026-08-25 | [#52388](https://github.com/vllm-project/vllm/pull/52388) | merged | [K3 Perf] Optimize k3 mamba metadata preparation, 6.6~7.6x kernel performance improvement | `tests/models/kimi_k3/test_kda_metadata.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py` |
| 2026-08-25 | [#53766](https://github.com/vllm-project/vllm/pull/53766) | merged | [CI Bug] Fix kimi test `AssertionError: Aligned Mamba state indices must be precomputed` | `tests/models/kimi_k3/test_kda_metadata.py` |
| 2026-08-25 | [#53534](https://github.com/vllm-project/vllm/pull/53534) | merged | [Kimi K3][Kernel] Enable low-latency decode GEMM dispatch on SM100 | `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` |
| 2026-08-26 | [#53942](https://github.com/vllm-project/vllm/pull/53942) | merged | [Kimi K3 Perf] Optimize `eh_proj` linear calculation, 12.9 ~ 25.2% kernel performance improvement | `vllm/models/kimi_k3/nvidia/mtp.py`, `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` |
| 2026-08-27 | [#53396](https://github.com/vllm-project/vllm/pull/53396) | merged | [Kimi K3][Kernel] Support DS conv-state layout in fused KDA decode kernel | `tests/models/kimi_k3/test_kda.py`, `vllm/models/kimi_k3/nvidia/kda.py`, `benchmarks/kernels/benchmark_kimi_k3_kda_decode.py` |
| 2026-08-27 | [#54088](https://github.com/vllm-project/vllm/pull/54088) | merged | [Kimi Perf] Tune hopper low latency gemm kernel, 4%~97% performance improvement | `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` |
| 2026-08-28 | [#54015](https://github.com/vllm-project/vllm/pull/54015) | merged | [Kimi-K3] Merge MLA gate into QKV-A projection | `vllm/models/kimi_k3/nvidia/mla.py`, `vllm/models/kimi_k3/nvidia/model.py`, `vllm/models/kimi_k3/nvidia/mtp.py` |
| 2026-08-28 | [#54167](https://github.com/vllm-project/vllm/pull/54167) | merged | [Kimi-K3][Bugfix] Fix low-latency GEMM fallback initialization | `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` |
| 2026-08-28 | [#54168](https://github.com/vllm-project/vllm/pull/54168) | merged | [Kimi-K3][Kernel] Optimize the low-M fused latent MoE tail | `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py`, `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/primitives.py`, `tests/models/kimi_k3/test_latent_moe_tail.py` |
| 2026-08-31 | [#54482](https://github.com/vllm-project/vllm/pull/54482) | merged | [CI/Build] Fix Kimi K3 Eagle3 test fixture | `tests/models/kimi_k3/test_eagle3.py` |
| 2026-08-31 | [#54261](https://github.com/vllm-project/vllm/pull/54261) | merged | [Kimi-K3][Perf] Make native CUDA AttnRes the SM100 default | `tests/models/kimi_k3/test_attn_res.py`, `vllm/models/kimi_k3/nvidia/ops/attn_res.py`, `benchmarks/kernels/benchmark_kimi_k3_attn_res.py` |
| 2026-09-01 | [#54781](https://github.com/vllm-project/vllm/pull/54781) | merged | [Kimi Bug] Fix `cannot access local variable 'active_non_spec_mask_cpu'` | `tests/models/kimi_k3/test_kda_metadata.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py` |
| 2026-09-02 | [#54697](https://github.com/vllm-project/vllm/pull/54697) | merged | [Kimi-K3] Overlap low-M TP8 KDA projections | `vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py`, `vllm/models/kimi_k3/nvidia/low_latency_gemm.py`, `vllm/models/kimi_k3/nvidia/kda.py` |
| 2026-09-02 | [#54859](https://github.com/vllm-project/vllm/pull/54859) | merged | [Kimi-K3] Bump FlashKDA to fix unstable inverse | `tests/models/kimi_k3/test_kda.py` |
| 2026-09-02 | [#54817](https://github.com/vllm-project/vllm/pull/54817) | merged | [CI] Add Kimi-K3-pruned75-DSpark-TP4 gsm8k eval | `tests/evals/gsm8k/configs/Kimi-K3-pruned75-DSpark-TP4.yaml`, `vllm/v1/attention/backends/mla/prefill/flashinfer.py` |
| 2026-09-02 | [#54565](https://github.com/vllm-project/vllm/pull/54565) | merged | [K3 Perf] Enable DSV3 GEMM for inner-contiguous and row-strided tensors, 12%~81% kernel performance improvement | `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` |
| 2026-09-03 | [#54606](https://github.com/vllm-project/vllm/pull/54606) | merged | [Kernel] Enable Kimi-K3 SiTU on the CuteDSL MoE backend and the SM107 low-latency GEMM plan | `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` |
| 2026-09-03 | [#54896](https://github.com/vllm-project/vllm/pull/54896) | merged | [Perf][Kimi-K3] Cut MLA decode concat/cache epilogue latency | `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py` |
| 2026-09-04 | [#52494](https://github.com/vllm-project/vllm/pull/52494) | merged | [AMD][kimik3][ROCm][Perf] Fuse MLA q/kv RMSNorm in AMD Kimi-K3 MLA wrapper | `vllm/models/kimi_k3/amd/mla.py`, `vllm/models/kimi_k3/amd/linear.py` |
| 2026-09-05 | [#55242](https://github.com/vllm-project/vllm/pull/55242) | merged | [Perf] Kimi K3 nvfp4 Align in_proj weights by 128 to avoid elementwise copy | `vllm/models/kimi_k3/nvidia/kda.py`, `vllm/models/kimi_k3/nvidia/mla.py`, `vllm/models/kimi_k3/nvidia/model.py` |
| 2026-09-06 | [#53614](https://github.com/vllm-project/vllm/pull/53614) | merged | [Kimi K3] Support internal prefix checkpoints with partial prefix caching and spec-decoding | `tests/models/kimi_k3/test_kda_metadata.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py`, `vllm/models/kimi_k3/nvidia/kda.py` |
| 2026-09-08 | [#53379](https://github.com/vllm-project/vllm/pull/53379) | merged | [Bugfix] Fix Kimi K3 loading with interleaved weight streams | `tests/models/kimi_k3/test_weight_loading.py`, `vllm/models/kimi_k3/nvidia/model.py` |
| 2026-09-08 | [#55924](https://github.com/vllm-project/vllm/pull/55924) | merged | [Kimi Bug] Fix kda ima `Triton Error [CUDA]: an illegal memory access was encountered` | `vllm/models/kimi_k3/nvidia/kda.py` |
| 2026-09-10 | [#54038](https://github.com/vllm-project/vllm/pull/54038) | merged | [ROCm][Perf] Kimi-K3 Fused kernels for KDA prefill reland | `tests/models/kimi_k3/test_amd_kda_chunk.py`, `vllm/models/kimi_k3/amd/ops/kda_chunk.py`, `vllm/models/kimi_k3/amd/ops/kda_prefill.py` |
| 2026-09-10 | [#56159](https://github.com/vllm-project/vllm/pull/56159) | merged | [Kimi K3 Perf] Avoid KDA mixed-batch gather/scatter, 5.2%~7.7% E2E Throughput Improvement | `vllm/models/kimi_k3/nvidia/kda.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py`, `tests/models/kimi_k3/test_kda.py` |
| 2026-09-11 | [#55356](https://github.com/vllm-project/vllm/pull/55356) | merged | [Kimi Perf] Group fp8 mla cahche insertion, 4~6x kernel level performance improvement for small batch | `vllm/models/kimi_k3/nvidia/dspark_mla.py` |
| 2026-09-11 | [#55426](https://github.com/vllm-project/vllm/pull/55426) | merged | [Bugfix][Kimi-K3] Fix KDA projection overlap on Hopper | `vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py`, `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` |
| 2026-09-12 | [#56526](https://github.com/vllm-project/vllm/pull/56526) | merged | [ROCm][Kimi-K3] Fix non-contiguous state_indices crash and GPU-sync assert in fused KDA/MLA prefill | `vllm/models/kimi_k3/amd/ops/kda_chunk.py` |
| 2026-09-16 | [#51483](https://github.com/vllm-project/vllm/pull/51483) | merged | [Bugfix][Kimi-K3] Do not classify a stateless first chunk as a decode | `tests/models/kimi_k3/test_kda_metadata.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py` |
| 2026-09-17 | [#57098](https://github.com/vllm-project/vllm/pull/57098) | merged | [Kimi K3 Bug] Fix kimi k3 reasoning parser | `vllm/reasoning/kimi_k3_reasoning_parser.py` |
| 2026-09-18 | [#57430](https://github.com/vllm-project/vllm/pull/57430) | merged | [Quantization] Support kimi-k3 routed expert quant | `vllm/models/kimi_k3/nvidia/model.py` |
| 2026-09-21 | [#50592](https://github.com/vllm-project/vllm/pull/50592) | merged | [Kimi-K3][AMD] Return KDA and MLA projection outputs directly | `tests/models/kimi_k3/test_amd_kda_direct_return.py`, `vllm/models/kimi_k3/amd/linear.py`, `tests/models/kimi_k3/test_amd_mla_direct_return.py` |
| 2026-09-23 | [#52988](https://github.com/vllm-project/vllm/pull/52988) | merged | [Spec decode] Support variable-length decode for Kimi-K3 adaptive ver | `vllm/models/kimi_k3/nvidia/dspark_mla.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py` |
| 2026-09-23 | [#58012](https://github.com/vllm-project/vllm/pull/58012) | merged | [CI][ROCm] Add an MI355 Kimi-K3 unit test group | `tests/models/kimi_k3/test_attn_res.py`, `vllm/platforms/rocm.py` |
| 2026-09-25 | [#58527](https://github.com/vllm-project/vllm/pull/58527) | merged | [Kimi-K3][Perf] Dispatch GEMM for vision patch embedder | `vllm/model_executor/models/kimi_k25_vit.py` |
| 2026-09-25 | [#58045](https://github.com/vllm-project/vllm/pull/58045) | merged | [ROCm][Kimi-K3] Optimize low-concurrency speculative KDA | `vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py`, `tests/models/kimi_k3/test_kda.py`, `tests/models/kimi_k3/test_kda_metadata.py` |
| 2026-09-25 | [#58372](https://github.com/vllm-project/vllm/pull/58372) | merged | [Bugfix][Reasoning] Count Kimi K3 reasoning tokens | `tests/reasoning/test_kimi_k3_reasoning_parser.py`, `vllm/reasoning/kimi_k3_reasoning_parser.py` |
| 2026-09-28 | [#58651](https://github.com/vllm-project/vllm/pull/58651) | merged | [KimiViT][Perf] Fuse per-layer QK RoPE into one in-place kernel | `vllm/model_executor/models/kimi_k25_vit.py` |
| 2026-09-28 | [#54956](https://github.com/vllm-project/vllm/pull/54956) | merged | [ROCm][Perf] Kimi-K3 Enable sharded latent MoE up-projection under EP | `tests/models/kimi_k3/test_amd_latent_moe_runner.py`, `vllm/models/kimi_k3/amd/latent_moe_runner.py` |
| 2026-09-28 | [#58814](https://github.com/vllm-project/vllm/pull/58814) | merged | [Bugfix][Kimi-K3] Refresh DSpark context KV cache pointers after the KV cache is re-bound | `vllm/models/kimi_k3/nvidia/dspark_mla.py` |
| 2026-09-30 | [#57640](https://github.com/vllm-project/vllm/pull/57640) | merged | [ROCm][Kimi-K3][Perf] Fuse MLA decode KV-cache write and Q-prep via AITER | `vllm/models/kimi_k3/amd/mla.py` |
| 2026-10-01 | [#54255](https://github.com/vllm-project/vllm/pull/54255) | merged | [Kimi-K3] Add FlashInfer speculative KDA backend | `tests/models/kimi_k3/test_kda.py`, `vllm/models/kimi_k3/nvidia/kda.py` |
| 2026-10-01 | [#59257](https://github.com/vllm-project/vllm/pull/59257) | merged | [Bugfix] Gate Kimi-K3 KDA warmup on sys.modules to skip Kimi import for non-Kimi models | `vllm/model_executor/warmup/kimi_k3_triton_warmup.py` |
| 2026-10-01 | [#58769](https://github.com/vllm-project/vllm/pull/58769) | merged | [ROCm][Triton] Migrate Kimi-K3 kernels from make_block_ptr to tensor … | `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra_token_parallel.py` |
| 2026-10-02 | [#58344](https://github.com/vllm-project/vllm/pull/58344) | merged | [ROCm][Perf] Kimi-K3 enable prefill checkpoints on ROCm | `tests/models/kimi_k3/test_amd_kda_checkpoint.py`, `tests/models/kimi_k3/test_amd_kda_chunk.py`, `vllm/models/kimi_k3/amd/kda.py` |
| 2026-10-02 | [#59229](https://github.com/vllm-project/vllm/pull/59229) | merged | [CI][Kimi-K3] Test prefix cache reuse with KV offload, P/D and DCP | `tests/models/kimi_k3/test_prefix_cache.py` |
| 2026-10-03 | [#51274](https://github.com/vllm-project/vllm/pull/51274) | merged | [ROCm][Kimi-K3] Add opt-in gfx942 MXFP4-to-int4 conversion | `tests/models/kimi_k3/test_gfx942_int4.py`, `vllm/model_executor/layers/quantization/mxfp4.py`, `vllm/model_executor/layers/quantization/online/base.py` |

## Per-PR Diff Audit Cards

### PR #16387 - [Model][VLM] Add Kimi-VL model support

- Link: https://github.com/vllm-project/vllm/pull/16387
- Status/date: merged / 2025-04-14
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/16387 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`, `vllm/transformers_utils/configs/kimi_vl.py`, `vllm/transformers_utils/configs/moonvit.py`; associated commits `b1308b84a3a6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +1436/-14, 1618 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][VLM] Add Kimi-VL model support"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `vllm/transformers_utils/configs/kimi_vl.py`; technical summary: Covers "[Model][VLM] Add Kimi-VL model support"; the main implementation surface is `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `vllm/transformers_utils/configs/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` added +628/-0 (628 lines); hunks: -0,0 +1,628; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope, touching `multihead_attention, sdpa_attention, _apply_rope_input_validation`; `vllm/model_executor/models/kimi_vl.py` added +608/-0 (608 lines); hunks: -0,0 +1,608; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward, touching `MaxImageTokenMeta, KimiVLMultiModalProjector, __init__`; `vllm/transformers_utils/configs/kimi_vl.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: KimiVLConfig, __init__, touching `KimiVLConfig, __init__`; `vllm/transformers_utils/configs/moonvit.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: MoonViTConfig, __init__, touching `MoonViTConfig, __init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` added +628/-0 (628 lines); hunks: -0,0 +1,628; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope
  - `vllm/model_executor/models/kimi_vl.py` added +608/-0 (608 lines); hunks: -0,0 +1,608; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward
  - `vllm/transformers_utils/configs/kimi_vl.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: KimiVLConfig, __init__
  - `vllm/transformers_utils/configs/moonvit.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: MoonViTConfig, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -0,0 +1,628 @@
+# SPDX-License-Identifier: Apache-2.0
+# ruff: noqa: E501
+# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
+# This file is meant to be used in kimi_vl.py only
+# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/model_executor/models/kimi_vl.py
@@ -0,0 +1,608 @@
+# SPDX-License-Identifier: Apache-2.0
+# ruff: noqa: E501
+# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
+# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
+#
+# The code is based on llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py), but modified for KimiVL.
diff -- vllm/transformers_utils/configs/kimi_vl.py
@@ -0,0 +1,36 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` added +628/-0; `vllm/model_executor/models/kimi_vl.py` added +608/-0; `vllm/transformers_utils/configs/kimi_vl.py` added +36/-0; `vllm/transformers_utils/configs/moonvit.py` added +32/-0
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16833 - [Misc] Clean up Kimi-VL

- Link: https://github.com/vllm-project/vllm/pull/16833
- Status/date: merged / 2025-04-18
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/16833 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`; associated commits `aadb6565628c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +20/-44, 139 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Clean up Kimi-VL"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[Misc] Clean up Kimi-VL"; the main implementation surface is `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_vl.py` modified +17/-40 (57 lines); hunks: -56,7 +56,6; -70,22 +69,20; symbols: KimiVLProcessingInfo, get_hf_config, get_supported_mm_limits, get_num_image_tokens, touching `KimiVLProcessingInfo, get_hf_config, get_supported_mm_limits`.
- Code diff details:
  - `vllm/model_executor/models/kimi_vl.py` modified +17/-40 (57 lines); hunks: -56,7 +56,6; -70,22 +69,20; symbols: KimiVLProcessingInfo, get_hf_config, get_supported_mm_limits, get_num_image_tokens
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_vl.py
@@ -56,7 +56,6 @@
-from vllm.logger import init_logger
@@ -70,22 +69,20 @@
-from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
-                                    NestedTensors)
+from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
+                                    MultiModalKwargs, NestedTensors)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +17/-40
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17156 - fix float16 support for kimi-vl

- Link: https://github.com/vllm-project/vllm/pull/17156
- Status/date: merged / 2025-04-25
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/17156 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`; associated commits `69bff9bc8934`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix float16 support for kimi-vl"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "fix float16 support for kimi-vl"; the main implementation surface is `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_vl.py` modified +1/-2 (3 lines); hunks: -340,8 +340,7 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, touching `_parse_and_validate_image_input`.
- Code diff details:
  - `vllm/model_executor/models/kimi_vl.py` modified +1/-2 (3 lines); hunks: -340,8 +340,7 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_vl.py
@@ -340,8 +340,7 @@ def _parse_and_validate_image_input(
-        # fp32 -> bf16
-        pixel_values = pixel_values.to(torch.bfloat16)
+        pixel_values = pixel_values.to(self.vision_tower.dtype)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21769 - Migrate KimiVLImagePixelInputs to TensorSchema

- Link: https://github.com/vllm-project/vllm/pull/21769
- Status/date: merged / 2025-08-05
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/21769 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`; associated commits `05fae021750b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +15/-9, 55 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Migrate KimiVLImagePixelInputs to TensorSchema"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "Migrate KimiVLImagePixelInputs to TensorSchema"; the main implementation surface is `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_vl.py` modified +15/-9 (24 lines); hunks: -46,7 +46,7; -79,6 +79,7; symbols: forward, KimiVLImagePixelInputs, _parse_and_validate_image_input, touching `forward, KimiVLImagePixelInputs, _parse_and_validate_image_input`.
- Code diff details:
  - `vllm/model_executor/models/kimi_vl.py` modified +15/-9 (24 lines); hunks: -46,7 +46,7; -79,6 +79,7; symbols: forward, KimiVLImagePixelInputs, _parse_and_validate_image_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_vl.py
@@ -46,7 +46,7 @@
-from typing import Any, Literal, Optional, TypedDict, Union
+from typing import Annotated, Any, Literal, Optional, Union
@@ -79,6 +79,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
@@ -118,15 +119,22 @@ def forward(self, image_features: torch.Tensor) -> torch.Tensor:
-class KimiVLImagePixelInputs(TypedDict):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +15/-9
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23114 - [Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506

- Link: https://github.com/vllm-project/vllm/pull/23114
- Status/date: merged / 2025-08-19
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/23114 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`; associated commits `fda9537c5e61`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-13, 77 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506"; the main implementation surface is `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_vl.py` modified +17/-12 (29 lines); hunks: -54,16 +54,16; -81,7 +81,7; symbols: get_replacement, KimiVLForConditionalGeneration, get_placeholder_str, __init__, touching `get_replacement, KimiVLForConditionalGeneration, get_placeholder_str`.
- Code diff details:
  - `vllm/model_executor/models/kimi_vl.py` modified +17/-12 (29 lines); hunks: -54,16 +54,16; -81,7 +81,7; symbols: get_replacement, KimiVLForConditionalGeneration, get_placeholder_str, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_vl.py
@@ -54,16 +54,16 @@
-from vllm.distributed import (get_tensor_model_parallel_rank,
-                              get_tensor_model_parallel_world_size)
+from vllm.distributed import get_pp_group
-from vllm.model_executor.models.interfaces import SupportsMultiModal
+from vllm.model_executor.models.interfaces import (SupportsMultiModal,
+                                                   SupportsPP)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +17/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23817 - [Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506

- Link: https://github.com/vllm-project/vllm/pull/23817
- Status/date: merged / 2025-09-01
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/23817 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`; associated commits `a0e0efd6bdcf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +157/-62, 478 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506"; the main implementation surface is `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` modified +55/-22 (77 lines); hunks: -42,7 +42,6; -55,6 +54,8; symbols: MLP2, __init__, forward, MoonVitEncoderLayer, touching `MLP2, __init__, forward`; `vllm/model_executor/models/kimi_vl.py` modified +39/-15 (54 lines); hunks: -56,6 +56,7; -76,6 +77,7; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward, touching `MaxImageTokenMeta, KimiVLMultiModalProjector, __init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` modified +55/-22 (77 lines); hunks: -42,7 +42,6; -55,6 +54,8; symbols: MLP2, __init__, forward, MoonVitEncoderLayer
  - `vllm/model_executor/models/kimi_vl.py` modified +39/-15 (54 lines); hunks: -56,6 +56,7; -76,6 +77,7; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -42,7 +42,6 @@
-import math
@@ -55,6 +54,8 @@
+from vllm.model_executor.layers.linear import ReplicatedLinear
+from vllm.model_executor.models.utils import maybe_prefix
@@ -383,21 +384,30 @@ class MLP2(nn.Module):
-    def __init__(self, dims: list[int], activation, bias=True):
diff -- vllm/model_executor/models/kimi_vl.py
@@ -56,6 +56,7 @@
+from vllm.model_executor.layers.linear import ReplicatedLinear
@@ -76,6 +77,7 @@
+from vllm.multimodal.utils import run_dp_sharded_mrope_vision_model
@@ -93,29 +95,35 @@ class MaxImageTokenMeta:
-    def __init__(self, config: KimiVLConfig):
+    def __init__(self, config: KimiVLConfig, \
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +55/-22; `vllm/model_executor/models/kimi_vl.py` modified +39/-15
- Risk and verification: The diff ships test coverage in `tests/multimodal/test_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27809 - [Model] Introduce Kimi Linear to vLLM

- Link: https://github.com/vllm-project/vllm/pull/27809
- Status/date: merged / 2025-10-30
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/27809 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py`; associated commits `4e68cc9b6aa2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +1326/-49, 1510 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Introduce Kimi Linear to vLLM"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py`; technical summary: Covers "[Model] Introduce Kimi Linear to vLLM"; the main implementation surface is `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_linear.py` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: KimiMLP, __init__, forward, KimiMoE, touching `KimiMLP, __init__, forward`; `vllm/transformers_utils/configs/kimi_linear.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: KimiLinearConfig, __init__, is_mla, is_moe, touching `KimiLinearConfig, __init__, is_mla`.
- Code diff details:
  - `vllm/model_executor/models/kimi_linear.py` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: KimiMLP, __init__, forward, KimiMoE
  - `vllm/transformers_utils/configs/kimi_linear.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: KimiLinearConfig, __init__, is_mla, is_moe
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -0,0 +1,663 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable
+from typing import Any
+import torch
+from torch import nn
diff -- vllm/transformers_utils/configs/kimi_linear.py
@@ -0,0 +1,144 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from transformers.configuration_utils import PretrainedConfig
+from vllm.logger import init_logger
+logger = init_logger(__name__)
+class KimiLinearConfig(PretrainedConfig):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_linear.py` added +663/-0; `vllm/transformers_utils/configs/kimi_linear.py` added +144/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27834 - [Kimi-Linear] Correct prefixes and add compatibility to AWQ quants

- Link: https://github.com/vllm-project/vllm/pull/27834
- Status/date: merged / 2025-10-31
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/27834 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_linear.py`; associated commits `e5ef4dfc11ab`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kimi-Linear] Correct prefixes and add compatibility to AWQ quants"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_linear.py`; technical summary: Covers "[Kimi-Linear] Correct prefixes and add compatibility to AWQ quants"; the main implementation surface is `vllm/model_executor/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_linear.py` modified +2/-1 (3 lines); hunks: -155,6 +155,7 @@ def __init__(; -340,7 +341,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/kimi_linear.py` modified +2/-1 (3 lines); hunks: -155,6 +155,7 @@ def __init__(; -340,7 +341,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -155,6 +155,7 @@ def __init__(
+                prefix=f"{prefix}.shared_experts",
@@ -340,7 +341,7 @@ def __init__(
-                prefix=f"{prefix}.mlp",
+                prefix=f"{prefix}.block_sparse_moe",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_linear.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27885 - fix incorrect type annotation in KimiMLP

- Link: https://github.com/vllm-project/vllm/pull/27885
- Status/date: merged / 2025-10-31
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/27885 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_linear.py`; associated commits `bc306fe5e978`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix incorrect type annotation in KimiMLP"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_linear.py`; technical summary: Covers "fix incorrect type annotation in KimiMLP"; the main implementation surface is `vllm/model_executor/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_linear.py` modified +1/-2 (3 lines); hunks: -22,7 +22,6; -61,7 +60,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_linear.py` modified +1/-2 (3 lines); hunks: -22,7 +22,6; -61,7 +60,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -22,7 +22,6 @@
-    QKVParallelLinear,
@@ -61,7 +60,7 @@ def __init__(
-        quant_config: QKVParallelLinear | None = None,
+        quant_config: QuantizationConfig | None = None,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_linear.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29309 - [XPU]fix Kimi-VL-A3B-thinking on xpu

- Link: https://github.com/vllm-project/vllm/pull/29309
- Status/date: merged / 2025-11-24
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/29309 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/moonvit.py`; associated commits `3cfa63ad9916`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +14/-6, 52 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU]fix Kimi-VL-A3B-thinking on xpu"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/moonvit.py`; technical summary: Covers "[XPU]fix Kimi-VL-A3B-thinking on xpu"; the main implementation surface is `vllm/model_executor/models/moonvit.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` modified +14/-6 (20 lines); hunks: -56,10 +56,13; -106,10 +109,10 @@ def multihead_attention(; symbols: multihead_attention, Rope2DPosEmb, __init__, touching `multihead_attention, Rope2DPosEmb, __init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` modified +14/-6 (20 lines); hunks: -56,10 +56,13; -106,10 +109,10 @@ def multihead_attention(; symbols: multihead_attention, Rope2DPosEmb, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -56,10 +56,13 @@
+from vllm.platforms import current_platform
+elif current_platform.is_xpu():
+    from vllm.attention.utils.fa_utils import flash_attn_varlen_func
@@ -106,10 +109,10 @@ def multihead_attention(
-        q_cu_seqlens,
-        k_cu_seqlens,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +14/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/moonvit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30125 - [CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it.

- Link: https://github.com/vllm-project/vllm/pull/30125
- Status/date: merged / 2025-12-15
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/30125 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 24 files, +1264/-853, 3625 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it."; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `vllm/attention/layers/mm_encoder_attention.py`, `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it."; the main implementation surface is `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `vllm/attention/layers/mm_encoder_attention.py`, `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0 (434 lines); hunks: -0,0 +1,434; symbols: build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt, build_qwen2_5_video_prompt, touching `build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt`; `vllm/attention/layers/mm_encoder_attention.py` added +284/-0 (284 lines); hunks: -0,0 +1,284; symbols: maybe_get_vit_flash_attn_backend, MMEncoderAttention, __init__, enabled, touching `maybe_get_vit_flash_attn_backend, MMEncoderAttention, __init__`; `vllm/model_executor/models/qwen2_vl.py` modified +47/-96 (143 lines); hunks: -33,7 +33,6; -45,10 +44,8; symbols: __init__, split_qkv, forward, touching `__init__, split_qkv, forward`; `vllm/model_executor/models/glm4_1v.py` modified +46/-91 (137 lines); hunks: -47,8 +47,10; -191,10 +193,15 @@ def __init__(; symbols: __init__, split_qkv, forward, touching `__init__, split_qkv, forward`.
- Code diff details:
  - `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0 (434 lines); hunks: -0,0 +1,434; symbols: build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt, build_qwen2_5_video_prompt
  - `vllm/attention/layers/mm_encoder_attention.py` added +284/-0 (284 lines); hunks: -0,0 +1,284; symbols: maybe_get_vit_flash_attn_backend, MMEncoderAttention, __init__, enabled
  - `vllm/model_executor/models/qwen2_vl.py` modified +47/-96 (143 lines); hunks: -33,7 +33,6; -45,10 +44,8; symbols: __init__, split_qkv, forward
  - `vllm/model_executor/models/glm4_1v.py` modified +46/-91 (137 lines); hunks: -47,8 +47,10; -191,10 +193,15 @@ def __init__(; symbols: __init__, split_qkv, forward
  - `vllm/model_executor/models/dots_ocr.py` modified +46/-83 (129 lines); hunks: -5,15 +5,14; -254,11 +253,15 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- tests/models/multimodal/generation/test_vit_backend_functionality.py
@@ -0,0 +1,434 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Consolidated test for ViT attention backend functionality across multiple models.
+This test validates that each multimodal model can successfully generate outputs
+using different ViT attention backends. Tests are parametrized by model and backend.
diff -- vllm/attention/layers/mm_encoder_attention.py
@@ -0,0 +1,284 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Callable
+import torch
+from vllm.attention.backends.registry import AttentionBackendEnum
+from vllm.attention.ops.vit_attn_wrappers import (
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -33,7 +33,6 @@
```

- Reviewed files:
  - tests: `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0
  - runtime: `vllm/attention/layers/mm_encoder_attention.py` added +284/-0; `vllm/model_executor/models/qwen2_vl.py` modified +47/-96; `vllm/model_executor/models/glm4_1v.py` modified +46/-91; `vllm/model_executor/models/dots_ocr.py` modified +46/-83; `vllm/model_executor/models/siglip2navit.py` modified +45/-84; `vllm/model_executor/models/qwen2_5_vl.py` modified +48/-76
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_backend_functionality.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31207 - fix: update kimi k2 tool parser logic

- Link: https://github.com/vllm-project/vllm/pull/31207
- Status/date: merged / 2025-12-30
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/31207 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`; associated commits `358bfd315cad`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +211/-202, 511 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: update kimi k2 tool parser logic"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`; technical summary: Covers "fix: update kimi k2 tool parser logic"; the main implementation surface is `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +192/-191 (383 lines); hunks: -44,6 +44,33 @@ def assert_tool_calls(; -346,61 +373,32 @@ def test_token_leak_between_section_and_tool_begin(kimi_k2...; symbols: assert_tool_calls, run_streaming_sequence, test_extract_tool_calls_no_tools, test_token_leak_between_section_and_tool_begin, touching `assert_tool_calls, run_streaming_sequence, test_extract_tool_calls_no_tools`; `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +19/-11 (30 lines); hunks: -122,7 +122,6 @@ def _check_and_strip_markers(self, text: str) -> tuple[str,...; -238,6 +237,7 @@ def extract_tool_calls_streaming(; symbols: _check_and_strip_markers, _reset_section_state, extract_tool_calls_streaming, touching `_check_and_strip_markers, _reset_section_state, extract_tool_calls_streaming`.
- Code diff details:
  - `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +192/-191 (383 lines); hunks: -44,6 +44,33 @@ def assert_tool_calls(; -346,61 +373,32 @@ def test_token_leak_between_section_and_tool_begin(kimi_k2...; symbols: assert_tool_calls, run_streaming_sequence, test_extract_tool_calls_no_tools, test_token_leak_between_section_and_tool_begin
  - `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +19/-11 (30 lines); hunks: -122,7 +122,6 @@ def _check_and_strip_markers(self, text: str) -> tuple[str,...; -238,6 +237,7 @@ def extract_tool_calls_streaming(; symbols: _check_and_strip_markers, _reset_section_state, extract_tool_calls_streaming
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_kimi_k2_tool_parser.py
@@ -44,6 +44,33 @@ def assert_tool_calls(
+def run_streaming_sequence(parser, deltas):
+    """Helper to simulate a streaming sequence and return results."""
+    previous_text = ""
+    previous_token_ids: list[int] = []
+    results = []
+    for delta_text, delta_token_ids in deltas:
diff -- vllm/tool_parsers/kimi_k2_tool_parser.py
@@ -122,7 +122,6 @@ def _check_and_strip_markers(self, text: str) -> tuple[str, bool, bool]:
@@ -238,6 +237,7 @@ def extract_tool_calls_streaming(
@@ -252,13 +252,18 @@ def extract_tool_calls_streaming(
-                remaining = buffered_text
-                # Return remaining text as reasoning content if non-empty
-                if remaining.strip():
-                    return DeltaMessage(content=remaining)
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +192/-191
  - runtime: `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +19/-11
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_kimi_k2_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31738 - [Models]: Use `MMEncoderAttention` for MoonViT

- Link: https://github.com/vllm-project/vllm/pull/31738
- Status/date: merged / 2026-01-06
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/31738 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`; associated commits `7101e0851f73`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +72/-158, 345 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models]: Use `MMEncoderAttention` for MoonViT"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[Models]: Use `MMEncoderAttention` for MoonViT"; the main implementation surface is `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` modified +71/-157 (228 lines); hunks: -51,118 +51,20; -411,11 +313,19 @@ def __init__(; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, __init__, touching `multihead_attention, sdpa_attention, _apply_rope_input_validation`; `vllm/model_executor/models/kimi_vl.py` modified +1/-1 (2 lines); hunks: -325,7 +325,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` modified +71/-157 (228 lines); hunks: -51,118 +51,20; -411,11 +313,19 @@ def __init__(; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, __init__
  - `vllm/model_executor/models/kimi_vl.py` modified +1/-1 (2 lines); hunks: -325,7 +325,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -51,118 +51,20 @@
-from transformers.utils import is_flash_attn_2_available
+from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
+from vllm.config import MultiModalConfig
+from vllm.distributed import divide, get_tensor_model_parallel_world_size
-from vllm.model_executor.layers.linear import ReplicatedLinear
+from vllm.model_executor.layers.linear import (
diff -- vllm/model_executor/models/kimi_vl.py
@@ -325,7 +325,7 @@ def __init__(
-            self.use_data_parallel,
+            multimodal_config=model_config.multimodal_config,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +71/-157; `vllm/model_executor/models/kimi_vl.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33131 - [Models] Kimi-K2.5

- Link: https://github.com/vllm-project/vllm/pull/33131
- Status/date: merged / 2026-01-27
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/33131 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`, `vllm/transformers_utils/configs/kimi_k25.py`; associated commits `b539f988e1ee`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +1799/-8, 2011 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Kimi-K2.5"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/configs/kimi_k25.py`; technical summary: Covers "[Models] Kimi-K2.5"; the main implementation surface is `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/configs/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: _apply_rope_input_validation, get_rope_shape_decorate, wrapper, get_rope_shape, touching `_apply_rope_input_validation, get_rope_shape_decorate, wrapper`; `vllm/model_executor/models/kimi_k25.py` added +581/-0 (581 lines); hunks: -0,0 +1,581; symbols: MaxImageTokenMeta, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor, __init__, touching `MaxImageTokenMeta, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor`; `vllm/transformers_utils/configs/kimi_k25.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: KimiK25VisionConfig, __init__, KimiK25Config, hidden_size, touching `KimiK25VisionConfig, __init__, KimiK25Config`; `vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: KimiK2ReasoningParser, __init__, is_reasoning_end, is_reasoning_end_streaming, touching `KimiK2ReasoningParser, __init__, is_reasoning_end`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: _apply_rope_input_validation, get_rope_shape_decorate, wrapper, get_rope_shape
  - `vllm/model_executor/models/kimi_k25.py` added +581/-0 (581 lines); hunks: -0,0 +1,581; symbols: MaxImageTokenMeta, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor, __init__
  - `vllm/transformers_utils/configs/kimi_k25.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: KimiK25VisionConfig, __init__, KimiK25Config, hidden_size
  - `vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: KimiK2ReasoningParser, __init__, is_reasoning_end, is_reasoning_end_streaming
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -0,0 +1,678 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Vision tower implementation for Kimi-K2.5 model.
+This module provides the vision encoder components for Kimi-K2.5,
+including 3D patch embedding, RoPE position embedding, and
diff -- vllm/model_executor/models/kimi_k25.py
@@ -0,0 +1,581 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+"""
+Kimi-K2.5 Model Implementation for vLLM.
+Kimi-K2.5 extends Kimi-K2 with vision support
diff -- vllm/transformers_utils/configs/kimi_k25.py
@@ -0,0 +1,129 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0; `vllm/model_executor/models/kimi_k25.py` added +581/-0; `vllm/transformers_utils/configs/kimi_k25.py` added +129/-0; `vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33320 - [Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d…

- Link: https://github.com/vllm-project/vllm/pull/33320
- Status/date: merged / 2026-01-29
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/33320 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `17b17c068453`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d…"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d…"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -58,6 +58,7; -320,7 +321,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -58,6 +58,7; -320,7 +321,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -58,6 +58,7 @@
+from vllm.platforms import current_platform
@@ -320,7 +321,7 @@ def __init__(
-        self.device = torch.cuda.current_device()
+        self.device = current_platform.current_device()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33346 - [Models] Refactor Kimi-K2.5 weight loading

- Link: https://github.com/vllm-project/vllm/pull/33346
- Status/date: merged / 2026-01-30
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/33346 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `8bfc8d5600ed`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +40/-176, 282 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Refactor Kimi-K2.5 weight loading"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; technical summary: Covers "[Models] Refactor Kimi-K2.5 weight loading"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +38/-174 (212 lines); hunks: -23,16 +23,7; -64,7 +55,12; symbols: KimiK25ForConditionalGeneration, get_placeholder_str, __init__, _parse_and_validate_media_input, touching `KimiK25ForConditionalGeneration, get_placeholder_str, __init__`; `vllm/model_executor/models/kimi_k25_vit.py` modified +2/-2 (4 lines); hunks: -660,13 +660,13 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +38/-174 (212 lines); hunks: -23,16 +23,7; -64,7 +55,12; symbols: KimiK25ForConditionalGeneration, get_placeholder_str, __init__, _parse_and_validate_media_input
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +2/-2 (4 lines); hunks: -660,13 +660,13 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -23,16 +23,7 @@
-from vllm.distributed import get_pp_group
-from vllm.model_executor.layers.fused_moe import SharedFusedMoE
-from vllm.model_executor.layers.logits_processor import LogitsProcessor
-from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
-from vllm.model_executor.model_loader.weight_utils import (
-    default_weight_loader,
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -660,13 +660,13 @@ def __init__(
-            prefix=maybe_prefix(prefix, "linear_1"),
+            prefix=f"{prefix}.linear_1",
-            prefix=maybe_prefix(prefix, "linear_2"),
+            prefix=f"{prefix}.linear_2",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +38/-174; `vllm/model_executor/models/kimi_k25_vit.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33562 - [Bugfix] Enable Kimi k25 processor test

- Link: https://github.com/vllm-project/vllm/pull/33562
- Status/date: merged / 2026-02-02
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/33562 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `4061dcf4c51a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +96/-12, 221 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Enable Kimi k25 processor test"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix] Enable Kimi k25 processor test"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +27/-5 (32 lines); hunks: -96,16 +96,20 @@ class MoonshotKimiVAutoProcessor(ProcessorMixin):; -122,13 +126,30 @@ def __call__(; symbols: MoonshotKimiVAutoProcessor, __init__, __call__, touching `MoonshotKimiVAutoProcessor, __init__, __call__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +27/-5 (32 lines); hunks: -96,16 +96,20 @@ class MoonshotKimiVAutoProcessor(ProcessorMixin):; -122,13 +126,30 @@ def __call__(; symbols: MoonshotKimiVAutoProcessor, __init__, __call__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -96,16 +96,20 @@ class MoonshotKimiVAutoProcessor(ProcessorMixin):
-    def __init__(self, media_processor=None, tokenizer=None):
+    def __init__(
+        self, media_processor=None, tokenizer=None, media_token_id: int | None = None
+    ):
+        self.media_token_id = media_token_id
+        assert self.media_token_id is not None
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +27/-5
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33876 - [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading

- Link: https://github.com/vllm-project/vllm/pull/33876
- Status/date: merged / 2026-02-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `a2522839d87d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +15/-5, 53 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading"; model line: Kimi K2/K2.5/K3/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunks: -24,7 +24,11; -302,7 +306,9 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, touching `split_video_chunks, KimiK25ForConditionalGeneration, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunks: -24,7 +24,11; -302,7 +306,9 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -24,7 +24,11 @@
-from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
+from vllm.model_executor.models.interfaces import (
+    SupportsMultiModal,
+    SupportsPP,
+    SupportsQuant,
+)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +14/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34427 - [Bugfix] Delete unused redundant code in Kimi-K2.5

- Link: https://github.com/vllm-project/vllm/pull/34427
- Status/date: merged / 2026-02-13
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/34427 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `62788f99a4d0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-5, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Delete unused redundant code in Kimi-K2.5"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix] Delete unused redundant code in Kimi-K2.5"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +0/-5 (5 lines); hunks: -11,7 +11,6; -378,10 +377,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +0/-5 (5 lines); hunks: -11,7 +11,6; -378,10 +377,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -11,7 +11,6 @@
-import copy
@@ -378,10 +377,6 @@ def __init__(
-        sub_vllm_config = copy.deepcopy(vllm_config)
-        sub_vllm_config.model_config.hf_config = (
-            sub_vllm_config.model_config.hf_config.text_config
-        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +0/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34501 - [Bugfix] Add quant_config in ViT of Kimi-K2.5

- Link: https://github.com/vllm-project/vllm/pull/34501
- Status/date: merged / 2026-02-13
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/34501 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `4a9952ec1b15`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-0, 158 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Add quant_config in ViT of Kimi-K2.5"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix] Add quant_config in ViT of Kimi-K2.5"; the main implementation surface is `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25_vit.py` modified +15/-0 (15 lines); hunks: -28,6 +28,7; -304,6 +305,7 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/model_executor/models/kimi_k25.py` modified +11/-0 (11 lines); hunks: -23,6 +23,10; -361,6 +365,7 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, _parse_and_validate_media_input, touching `__init__, _maybe_ignore_quant_config, _parse_and_validate_media_input`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +15/-0 (15 lines); hunks: -28,6 +28,7; -304,6 +305,7 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/kimi_k25.py` modified +11/-0 (11 lines); hunks: -23,6 +23,10; -361,6 +365,7 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, _parse_and_validate_media_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -28,6 +28,7 @@
+from vllm.model_executor.layers.quantization import QuantizationConfig
@@ -304,6 +305,7 @@ def __init__(
+        quant_config: QuantizationConfig | None = None,
@@ -314,13 +316,15 @@ def __init__(
+            quant_config=quant_config,
+            quant_config=quant_config,
diff -- vllm/model_executor/models/kimi_k25.py
@@ -23,6 +23,10 @@
+from vllm.model_executor.layers.quantization import QuantizationConfig
+from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
+    CompressedTensorsConfig,
+)
@@ -361,6 +365,7 @@ def __init__(
+                quant_config=self._maybe_ignore_quant_config(quant_config),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` modified +15/-0; `vllm/model_executor/models/kimi_k25.py` modified +11/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33646 - [Bugfix] Handle case when kimi ends reasoning with a tool call

- Link: https://github.com/vllm-project/vllm/pull/33646
- Status/date: merged / 2026-02-27
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/33646 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/reasoning/kimi_k2_reasoning_parser.py`; associated commits `9251ed5c4fc6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +230/-2, 240 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Handle case when kimi ends reasoning with a tool call"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/reasoning/kimi_k2_reasoning_parser.py`; technical summary: Covers "[Bugfix] Handle case when kimi ends reasoning with a tool call"; the main implementation surface is `vllm/reasoning/kimi_k2_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/reasoning/kimi_k2_reasoning_parser.py` added +228/-0 (228 lines); hunks: -0,0 +1,228; symbols: KimiK2ReasoningParser, __init__, _is_identity_mode, is_reasoning_end, touching `KimiK2ReasoningParser, __init__, _is_identity_mode`.
- Code diff details:
  - `vllm/reasoning/kimi_k2_reasoning_parser.py` added +228/-0 (228 lines); hunks: -0,0 +1,228; symbols: KimiK2ReasoningParser, __init__, _is_identity_mode, is_reasoning_end
- Key code excerpts:

```diff
diff -- vllm/reasoning/kimi_k2_reasoning_parser.py
@@ -0,0 +1,228 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Sequence
+from transformers import PreTrainedTokenizerBase
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionRequest,
```

- Reviewed files:
  - runtime: `vllm/reasoning/kimi_k2_reasoning_parser.py` added +228/-0
- Risk and verification: Runtime changes concentrate in `vllm/reasoning/__init__.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36192 - [Security] Respect user trust_remote_code setting in NemotronVL and KimiK25

- Link: https://github.com/vllm-project/vllm/pull/36192
- Status/date: merged / 2026-03-06
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/36192 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `00bd08edeee5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-2, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Security] Respect user trust_remote_code setting in NemotronVL and KimiK25"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Security] Respect user trust_remote_code setting in NemotronVL and KimiK25"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -174,7 +174,8 @@ def __init__(self, ctx: InputProcessingContext) -> None:; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -174,7 +174,8 @@ def __init__(self, ctx: InputProcessingContext) -> None:; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -174,7 +174,8 @@ def __init__(self, ctx: InputProcessingContext) -> None:
-            self.ctx.model_config.model, trust_remote_code=True
+            self.ctx.model_config.model,
+            trust_remote_code=self.ctx.model_config.trust_remote_code,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36127 - [Model] Add support for moonshotai/Kimi-Audio-7B-Instruct

- Link: https://github.com/vllm-project/vllm/pull/36127
- Status/date: merged / 2026-03-11
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/36127 gh: Not Found (HTTP 404)`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja`, `vllm/transformers_utils/processors/kimi_audio.py`; associated commits `42fadebecb79`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +1446/-29, 1583 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add support for moonshotai/Kimi-Audio-7B-Instruct"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py`; technical summary: Covers "[Model] Add support for moonshotai/Kimi-Audio-7B-Instruct"; the main implementation surface is `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_audio.py` added +725/-0 (725 lines); hunks: -0,0 +1,725; symbols: _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__, KimiAudioProcessingInfo, touching `_get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__`; `vllm/tokenizers/kimi_audio.py` added +410/-0 (410 lines); hunks: -0,0 +1,410; symbols: _load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained, __init__, touching `_load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained`; `vllm/transformers_utils/processors/kimi_audio.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: _get_feat_extract_output_lengths, KimiAudioProcessor, __init__, check_argument_for_proper_class, touching `_get_feat_extract_output_lengths, KimiAudioProcessor, __init__`; `vllm/renderers/kimi_audio.py` added +49/-0 (49 lines); hunks: -0,0 +1,49; symbols: KimiAudioRenderer, from_config, touching `KimiAudioRenderer, from_config`.
- Code diff details:
  - `vllm/model_executor/models/kimi_audio.py` added +725/-0 (725 lines); hunks: -0,0 +1,725; symbols: _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__, KimiAudioProcessingInfo
  - `vllm/tokenizers/kimi_audio.py` added +410/-0 (410 lines); hunks: -0,0 +1,410; symbols: _load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained, __init__
  - `vllm/transformers_utils/processors/kimi_audio.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: _get_feat_extract_output_lengths, KimiAudioProcessor, __init__, check_argument_for_proper_class
  - `vllm/renderers/kimi_audio.py` added +49/-0 (49 lines); hunks: -0,0 +1,49; symbols: KimiAudioRenderer, from_config
  - `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja` added +13/-0 (13 lines); hunks: -0,0 +1,13
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_audio.py
@@ -0,0 +1,725 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only Kimi-Audio model compatible with HuggingFace weights."""
+import os
+from collections.abc import Iterable, Mapping, Sequence
+from typing import Any, ClassVar, Literal
diff -- vllm/tokenizers/kimi_audio.py
@@ -0,0 +1,410 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tokenizer for Kimi-Audio using TikToken."""
+import contextlib
+import json
+from pathlib import Path
diff -- vllm/transformers_utils/processors/kimi_audio.py
@@ -0,0 +1,163 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_audio.py` added +725/-0; `vllm/tokenizers/kimi_audio.py` added +410/-0; `vllm/transformers_utils/processors/kimi_audio.py` added +163/-0; `vllm/renderers/kimi_audio.py` added +49/-0; `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja` added +13/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`, `tests/models/test_initialization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36361 - Kimi k2.5 MLA based eagle3

- Link: https://github.com/vllm-project/vllm/pull/36361
- Status/date: merged / 2026-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `557389473755`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +499/-8, 649 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Kimi k2.5 MLA based eagle3"; model line: Kimi K2/K2.5/K3/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "Kimi k2.5 MLA based eagle3"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +14/-1 (15 lines); hunks: -28,6 +28,8; -311,7 +313,12 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, set_aux_hidden_state_layers, touching `split_video_chunks, KimiK25ForConditionalGeneration, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-1 (15 lines); hunks: -28,6 +28,8; -311,7 +313,12 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, set_aux_hidden_state_layers
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -28,6 +28,8 @@
+    SupportsEagle,
+    SupportsEagle3,
@@ -311,7 +313,12 @@ def split_video_chunks(self, video):
-    nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant
+    nn.Module,
+    SupportsMultiModal,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +14/-1
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36903 - [Misc] Clean up Kimi-audio whisper encoder loading

- Link: https://github.com/vllm-project/vllm/pull/36903
- Status/date: merged / 2026-03-14
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/36903 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_audio.py`; associated commits `a8e8d62dd80f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +89/-116, 382 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Clean up Kimi-audio whisper encoder loading"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_audio.py`; technical summary: Covers "[Misc] Clean up Kimi-audio whisper encoder loading"; the main implementation surface is `vllm/model_executor/models/kimi_audio.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_audio.py` modified +61/-111 (172 lines); hunks: -3,25 +3,21; -64,15 +60,6; symbols: _get_whisper_local_path, _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__, touching `_get_whisper_local_path, _get_feat_extract_output_lengths, KimiAudioWhisperEncoder`.
- Code diff details:
  - `vllm/model_executor/models/kimi_audio.py` modified +61/-111 (172 lines); hunks: -3,25 +3,21; -64,15 +60,6; symbols: _get_whisper_local_path, _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_audio.py
@@ -3,25 +3,21 @@
-import os
-from huggingface_hub import snapshot_download
-from safetensors import safe_open
-from vllm.model_executor.model_loader.weight_utils import (
-    default_weight_loader,
-)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_audio.py` modified +61/-111
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/model_loader/default_loader.py`, `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/kimi_audio.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37371 - standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01

- Link: https://github.com/vllm-project/vllm/pull/37371
- Status/date: merged / 2026-03-18
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/37371 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_linear.py`; associated commits `17808394bc48`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +235/-219, 527 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_linear.py`; technical summary: Covers "standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01"; the main implementation surface is `vllm/model_executor/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_linear.py` modified +97/-88 (185 lines); hunks: -46,6 +46,7; -472,94 +473,7 @@ def forward(; symbols: forward, KimiLinearForCausalLM, __init__, embed_input_ids, touching `forward, KimiLinearForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_linear.py` modified +97/-88 (185 lines); hunks: -46,6 +46,7; -472,94 +473,7 @@ def forward(; symbols: forward, KimiLinearForCausalLM, __init__, embed_input_ids
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -46,6 +46,7 @@
+    AutoWeightsLoader,
@@ -472,94 +473,7 @@ def forward(
-class KimiLinearForCausalLM(
-    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
-):
-    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_linear.py` modified +97/-88
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/models/minimax_text_01.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37438 - [Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support

- Link: https://github.com/vllm-project/vllm/pull/37438
- Status/date: merged / 2026-03-19
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/37438 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_kimi_k2_reasoning_parser.py`; associated commits `c63ca2b2e696`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +173/-18, 227 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/entrypoints/chat_utils.py`, `vllm/entrypoints/openai/chat_completion/serving.py`; technical summary: Covers "[Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support"; the main implementation surface is `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/entrypoints/chat_utils.py`, `vllm/entrypoints/openai/chat_completion/serving.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled, test_extract_reasoning_with_think_tags, touching `kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled`; `vllm/entrypoints/chat_utils.py` modified +14/-0 (14 lines); hunks: -1660,6 +1660,20 @@ def get_history_tool_calls_cnt(conversation: list[Convers...; symbols: get_history_tool_calls_cnt, get_tool_call_id_type, make_tool_call_id, touching `get_history_tool_calls_cnt, get_tool_call_id_type, make_tool_call_id`; `vllm/entrypoints/openai/chat_completion/serving.py` modified +2/-9 (11 lines); hunks: -19,6 +19,7; -152,15 +153,7 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/entrypoints/openai/responses/serving.py` modified +2/-9 (11 lines); hunks: -46,6 +46,7; -241,15 +242,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled, test_extract_reasoning_with_think_tags
  - `vllm/entrypoints/chat_utils.py` modified +14/-0 (14 lines); hunks: -1660,6 +1660,20 @@ def get_history_tool_calls_cnt(conversation: list[Convers...; symbols: get_history_tool_calls_cnt, get_tool_call_id_type, make_tool_call_id
  - `vllm/entrypoints/openai/chat_completion/serving.py` modified +2/-9 (11 lines); hunks: -19,6 +19,7; -152,15 +153,7 @@ def __init__(; symbols: __init__
  - `vllm/entrypoints/openai/responses/serving.py` modified +2/-9 (11 lines); hunks: -46,6 +46,7; -241,15 +242,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tests/reasoning/test_kimi_k2_reasoning_parser.py
@@ -0,0 +1,155 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
+from vllm.entrypoints.openai.engine.protocol import DeltaMessage
+from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
diff -- vllm/entrypoints/chat_utils.py
@@ -1660,6 +1660,20 @@ def get_history_tool_calls_cnt(conversation: list[ConversationMessage]):
+_KIMI_MODEL_TYPES = ("kimi_k2", "kimi_k25")
+def get_tool_call_id_type(model_config: ModelConfig) -> str:
+    """Return the tool-call ID type for a given model configuration."""
+    hf_overrides = getattr(model_config, "hf_overrides", None)
+    if model_config.hf_text_config.model_type in _KIMI_MODEL_TYPES or (
+        isinstance(hf_overrides, dict)
diff -- vllm/entrypoints/openai/chat_completion/serving.py
@@ -19,6 +19,7 @@
```

- Reviewed files:
  - tests: `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0
  - runtime: `vllm/entrypoints/chat_utils.py` modified +14/-0; `vllm/entrypoints/openai/chat_completion/serving.py` modified +2/-9; `vllm/entrypoints/openai/responses/serving.py` modified +2/-9
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_kimi_k2_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37693 - [Model] Update Kimi-K25 and Isaac processors to fit HF-style

- Link: https://github.com/vllm-project/vllm/pull/37693
- Status/date: merged / 2026-03-20
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/37693 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/processors/kimi_k25.py`; associated commits `37aadf623786`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +128/-95, 366 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Update Kimi-K25 and Isaac processors to fit HF-style"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/transformers_utils/processors/kimi_k25.py`, `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Model] Update Kimi-K25 and Isaac processors to fit HF-style"; the main implementation surface is `vllm/transformers_utils/processors/kimi_k25.py`, `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/processors/kimi_k25.py` modified +54/-38 (92 lines); hunks: -1,38 +1,41; -42,31 +45,44 @@ def __call__(; symbols: KimiK25Processor, __init__, __call__, touching `KimiK25Processor, __init__, __call__`; `vllm/model_executor/models/kimi_k25.py` modified +16/-18 (34 lines); hunks: -104,19 +104,25 @@ class KimiK25ProcessingInfo(BaseProcessingInfo):; -132,20 +138,15 @@ def get_supported_mm_limits(self) -> Mapping[str, int | No...; symbols: KimiK25ProcessingInfo, __init__, get_hf_processor, get_supported_mm_limits, touching `KimiK25ProcessingInfo, __init__, get_hf_processor`.
- Code diff details:
  - `vllm/transformers_utils/processors/kimi_k25.py` modified +54/-38 (92 lines); hunks: -1,38 +1,41; -42,31 +45,44 @@ def __call__(; symbols: KimiK25Processor, __init__, __call__
  - `vllm/model_executor/models/kimi_k25.py` modified +16/-18 (34 lines); hunks: -104,19 +104,25 @@ class KimiK25ProcessingInfo(BaseProcessingInfo):; -132,20 +138,15 @@ def get_supported_mm_limits(self) -> Mapping[str, int | No...; symbols: KimiK25ProcessingInfo, __init__, get_hf_processor, get_supported_mm_limits
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/processors/kimi_k25.py
@@ -1,38 +1,41 @@
-import torch
-from transformers import BatchFeature
+from transformers import BaseImageProcessor, BatchFeature, TensorType
+from vllm.tokenizers.hf import HfTokenizer
-    attributes = ["tokenizer"]
-    tokenizer_class = "AutoTokenizer"
diff -- vllm/model_executor/models/kimi_k25.py
@@ -104,19 +104,25 @@ class KimiK25ProcessingInfo(BaseProcessingInfo):
-        self.hf_config = self.get_hf_config()
-        self.media_token_id = self.hf_config.media_placeholder_token_id
-        media_processor = cached_get_image_processor(
+        self.hf_config = hf_config = self.get_hf_config()
+        tokenizer = self.get_tokenizer()
+        image_processor = cached_get_image_processor(
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/kimi_k25.py` modified +54/-38; `vllm/model_executor/models/kimi_k25.py` modified +16/-18
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/isaac.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/processors/isaac.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39344 - fix(kimi_k25): resolve media_placeholder_token_id from tokenizer

- Link: https://github.com/vllm-project/vllm/pull/39344
- Status/date: merged / 2026-04-12
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/39344 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `17e787a7792b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +24/-3, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(kimi_k25): resolve media_placeholder_token_id from tokenizer"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "fix(kimi_k25): resolve media_placeholder_token_id from tokenizer"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +24/-3 (27 lines); hunks: -113,7 +113,29 @@ def __init__(self, ctx: InputProcessingContext) -> None:; -232,8 +254,7 @@ def _get_prompt_updates(; symbols: __init__, _get_prompt_updates, get_replacement, touching `__init__, _get_prompt_updates, get_replacement`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +24/-3 (27 lines); hunks: -113,7 +113,29 @@ def __init__(self, ctx: InputProcessingContext) -> None:; -232,8 +254,7 @@ def _get_prompt_updates(; symbols: __init__, _get_prompt_updates, get_replacement
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -113,7 +113,29 @@ def __init__(self, ctx: InputProcessingContext) -> None:
-        self.media_token_id = media_token_id = hf_config.media_placeholder_token_id
+        # Resolve token ID from the tokenizer because transformers v5
+        # may remap token IDs vs config.json.
+        config_token_id = hf_config.media_placeholder_token_id
+        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
+        is_valid_resolved = isinstance(resolved_token_id, int) and (
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +24/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38579 - [Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping

- Link: https://github.com/vllm-project/vllm/pull/38579
- Status/date: merged / 2026-04-19
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/38579 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`; associated commits `03ce1c6ed908`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +684/-1405, 2206 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`; technical summary: Covers "[Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping"; the main implementation surface is `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +525/-921 (1446 lines); hunks: -3,14 +3,20; -20,959 +26,557 @@ def kimi_k2_tokenizer():; symbols: kimi_k2_tokenizer, kimi_k2_tool_parser, parser, assert_tool_calls, touching `kimi_k2_tokenizer, kimi_k2_tool_parser, parser`; `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +159/-484 (643 lines); hunks: -1,6 +1,5; -17,137 +16,59; symbols: KimiK2ToolParser, __init__, _check_and_strip_markers, _reset_section_state, touching `KimiK2ToolParser, __init__, _check_and_strip_markers`.
- Code diff details:
  - `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +525/-921 (1446 lines); hunks: -3,14 +3,20; -20,959 +26,557 @@ def kimi_k2_tokenizer():; symbols: kimi_k2_tokenizer, kimi_k2_tool_parser, parser, assert_tool_calls
  - `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +159/-484 (643 lines); hunks: -1,6 +1,5; -17,137 +16,59; symbols: KimiK2ToolParser, __init__, _check_and_strip_markers, _reset_section_state
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_kimi_k2_tool_parser.py
@@ -3,14 +3,20 @@
+from unittest.mock import MagicMock
-from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
+from tests.tool_parsers.utils import (
+    run_tool_extraction,
+    run_tool_extraction_streaming,
+)
diff -- vllm/tool_parsers/kimi_k2_tool_parser.py
@@ -1,6 +1,5 @@
-# code modified from deepseekv3_tool_parser.py
@@ -17,137 +16,59 @@
+from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
+from vllm.tool_parsers.utils import partial_tag_overlap
-        self.current_tool_name_sent: bool = False
+        # Streaming state
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +525/-921
  - runtime: `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +159/-484
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_kimi_k2_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41068 - [Bugfix] KimiK2ReasoningParser: guard against buffered end-token in streaming

- Link: https://github.com/vllm-project/vllm/pull/41068
- Status/date: merged / 2026-05-04
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/41068 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`; associated commits `712ad0286c9a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +70/-0, 102 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] KimiK2ReasoningParser: guard against buffered end-token in streaming"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`; technical summary: Covers "[Bugfix] KimiK2ReasoningParser: guard against buffered end-token in streaming"; the main implementation surface is `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_kimi_k2_reasoning_parser.py` modified +63/-0 (63 lines); hunks: -1,6 +1,8; -12,6 +14,20; symbols: mock_kimi_k2_tokenizer, kimi_k2_tokenizer, test_streaming_tool_section_ends_reasoning, test_streaming_end_token_id_buffered, touching `mock_kimi_k2_tokenizer, kimi_k2_tokenizer, test_streaming_tool_section_ends_reasoning`; `vllm/reasoning/kimi_k2_reasoning_parser.py` modified +7/-0 (7 lines); hunks: -221,6 +221,10 @@ def extract_reasoning_streaming(; -229,6 +233,9 @@ def extract_reasoning_streaming(; symbols: extract_reasoning_streaming, touching `extract_reasoning_streaming`.
- Code diff details:
  - `tests/reasoning/test_kimi_k2_reasoning_parser.py` modified +63/-0 (63 lines); hunks: -1,6 +1,8; -12,6 +14,20; symbols: mock_kimi_k2_tokenizer, kimi_k2_tokenizer, test_streaming_tool_section_ends_reasoning, test_streaming_end_token_id_buffered
  - `vllm/reasoning/kimi_k2_reasoning_parser.py` modified +7/-0 (7 lines); hunks: -221,6 +221,10 @@ def extract_reasoning_streaming(; -229,6 +233,9 @@ def extract_reasoning_streaming(; symbols: extract_reasoning_streaming
- Key code excerpts:

```diff
diff -- tests/reasoning/test_kimi_k2_reasoning_parser.py
@@ -1,6 +1,8 @@
+from unittest.mock import MagicMock
@@ -12,6 +14,20 @@
+@pytest.fixture
+def mock_kimi_k2_tokenizer():
+    tokenizer = MagicMock()
+    tokenizer.get_vocab.return_value = {
diff -- vllm/reasoning/kimi_k2_reasoning_parser.py
@@ -221,6 +221,10 @@ def extract_reasoning_streaming(
+            if self._end_token not in delta_text:
+                # Token ID arrived before text was flushed (stop-sequence buffering).
+                # Wait for the next delta when the text becomes visible.
+                return None
@@ -229,6 +233,9 @@ def extract_reasoning_streaming(
+            if self._tool_section_start_token not in delta_text:
```

- Reviewed files:
  - tests: `tests/reasoning/test_kimi_k2_reasoning_parser.py` modified +63/-0
  - runtime: `vllm/reasoning/kimi_k2_reasoning_parser.py` modified +7/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_kimi_k2_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42081 - [Bug] Fix kimi dtype issue with `mm_projector_forward`

- Link: https://github.com/vllm-project/vllm/pull/42081
- Status/date: merged / 2026-05-11
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/42081 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `3f9c0c25b331`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix kimi dtype issue with `mm_projector_forward`"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25_vit.py`; technical summary: Covers "[Bug] Fix kimi dtype issue with `mm_projector_forward`"; the main implementation surface is `vllm/model_executor/models/kimi_k25_vit.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25_vit.py` modified +3/-0 (3 lines); hunks: -618,6 +618,9 @@ def mm_projector_forward(mm_projector: torch.nn.Module, vt_o...; symbols: mm_projector_forward, touching `mm_projector_forward`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +3/-0 (3 lines); hunks: -618,6 +618,9 @@ def mm_projector_forward(mm_projector: torch.nn.Module, vt_o...; symbols: mm_projector_forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -618,6 +618,9 @@ def mm_projector_forward(mm_projector: torch.nn.Module, vt_output: list[torch.Te
+    projector_dtype = mm_projector.pre_norm.weight.dtype
+    if batched.dtype != projector_dtype:
+        batched = batched.to(projector_dtype)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41778 - [MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell

- Link: https://github.com/vllm-project/vllm/pull/41778
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +640/-89, 975 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell"; model line: Kimi K2/K2.5/K3/Linear/VL; category: docs/tests/CI; main diff: `benchmarks/attention_benchmarks/configs/mla_prefill.yaml`, `benchmarks/attention_benchmarks/configs/mla_decode.yaml`, `vllm/model_executor/layers/attention/mla_attention.py`; technical summary: Covers "[MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell"; the main implementation surface is `benchmarks/attention_benchmarks/configs/mla_prefill.yaml`, `benchmarks/attention_benchmarks/configs/mla_decode.yaml`, `vllm/model_executor/layers/attention/mla_attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #42869 - [BugFix] Kimi-K2.5: skip vision tower dtype conversion when using quantization

- Link: https://github.com/vllm-project/vllm/pull/42869
- Status/date: merged / 2026-05-18
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/42869 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `23c15acd770c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-3, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Kimi-K2.5: skip vision tower dtype conversion when using quantization"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[BugFix] Kimi-K2.5: skip vision tower dtype conversion when using quantization"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +6/-3 (9 lines); hunks: -339,9 +339,12 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +6/-3 (9 lines); hunks: -339,9 +339,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -339,9 +339,12 @@ def __init__(
-            self.vision_tower = self.vision_tower.to(
-                device=self.device, dtype=model_config.dtype
-            )
+            if self._maybe_ignore_quant_config(quant_config) is not None:
+                self.vision_tower = self.vision_tower.to(device=self.device)
+            else:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +6/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41126 - [Attention] Mamba attention module refactor

- Link: https://github.com/vllm-project/vllm/pull/41126
- Status/date: merged / 2026-05-22
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/41126 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +765/-774, 1913 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Attention] Mamba attention module refactor"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/olmo_hybrid.py`, `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`; technical summary: Covers "[Attention] Mamba attention module refactor"; the main implementation surface is `vllm/model_executor/models/olmo_hybrid.py`, `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/olmo_hybrid.py` modified +6/-645 (651 lines); hunks: -26,73 +26,47; -107,502 +81,6; symbols: _make_fused_conv1d_weight_loader, weight_loader, OlmoHybridGatedDeltaNet, mamba_type, touching `_make_fused_conv1d_weight_loader, weight_loader, OlmoHybridGatedDeltaNet`; `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` added +634/-0 (634 lines); hunks: -0,0 +1,634; symbols: OlmoHybridGatedDeltaNetAttention, get_state_shape, __init__, rearrange_mixed_qkv, touching `OlmoHybridGatedDeltaNetAttention, get_state_shape, __init__`; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` renamed +26/-45 (71 lines); hunks: -5,39 +5,37; -83,11 +81,8 @@ def kda_attention_fake(; symbols: kda_attention_fake, KimiDeltaAttention, mamba_type, KimiGatedDeltaNetAttention, touching `kda_attention_fake, KimiDeltaAttention, mamba_type`; `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` renamed +19/-52 (71 lines); hunks: -5,7 +5,6; -15,8 +14,6; symbols: forward_native, GatedDeltaNetAttention, mamba_type, get_state_dtype, touching `forward_native, GatedDeltaNetAttention, mamba_type`.
- Code diff details:
  - `vllm/model_executor/models/olmo_hybrid.py` modified +6/-645 (651 lines); hunks: -26,73 +26,47; -107,502 +81,6; symbols: _make_fused_conv1d_weight_loader, weight_loader, OlmoHybridGatedDeltaNet, mamba_type
  - `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` added +634/-0 (634 lines); hunks: -0,0 +1,634; symbols: OlmoHybridGatedDeltaNetAttention, get_state_shape, __init__, rearrange_mixed_qkv
  - `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` renamed +26/-45 (71 lines); hunks: -5,39 +5,37; -83,11 +81,8 @@ def kda_attention_fake(; symbols: kda_attention_fake, KimiDeltaAttention, mamba_type, KimiGatedDeltaNetAttention
  - `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` renamed +19/-52 (71 lines); hunks: -5,7 +5,6; -15,8 +14,6; symbols: forward_native, GatedDeltaNetAttention, mamba_type, get_state_dtype
  - `vllm/model_executor/layers/mamba/gdn/base.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: GatedDeltaNetAttention, for, __init__, mamba_type
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/olmo_hybrid.py
@@ -26,73 +26,47 @@
-from einops import rearrange
-from transformers.activations import ACT2FN
-    CacheConfig,
-    ModelConfig,
-    SpeculativeConfig,
-    get_current_vllm_config,
diff -- vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py
@@ -0,0 +1,634 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+from einops import rearrange
+from torch import nn
+from vllm.config import (
diff -- vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py
@@ -5,39 +5,37 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/olmo_hybrid.py` modified +6/-645; `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` added +634/-0; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` renamed +26/-45; `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` renamed +19/-52; `vllm/model_executor/layers/mamba/gdn/base.py` added +58/-0; `vllm/model_executor/models/kimi_linear.py` modified +13/-27
- Risk and verification: Runtime changes concentrate in `vllm/config/compilation.py`, `vllm/model_executor/layers/mamba/gdn/__init__.py`, `vllm/model_executor/layers/mamba/gdn/base.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43857 - Add vLLM library info to Hugging Face Hub requests

- Link: https://github.com/vllm-project/vllm/pull/43857
- Status/date: merged / 2026-05-29
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/43857 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +78/-43, 467 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add vLLM library info to Hugging Face Hub requests"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/model_loader/weight_utils.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/tokenizers/grok2.py`; technical summary: Covers "Add vLLM library info to Hugging Face Hub requests"; the main implementation surface is `vllm/model_executor/model_loader/weight_utils.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/tokenizers/grok2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/model_loader/weight_utils.py` modified +7/-7 (14 lines); hunks: -23,7 +23,6; -46,6 +45,7; symbols: get_quant_config, get_sparse_attention_config, download_weights_from_hf, touching `get_quant_config, get_sparse_attention_config, download_weights_from_hf`; `vllm/tokenizers/kimi_audio.py` modified +4/-4 (8 lines); hunks: -10,13 +10,13; -78,7 +78,7 @@ def from_pretrained(; symbols: from_pretrained, touching `from_pretrained`; `vllm/tokenizers/grok2.py` modified +3/-3 (6 lines); hunks: -8,7 +8,6; -20,6 +19,7; symbols: _maybe_load_tokenizer_config, from_pretrained, touching `_maybe_load_tokenizer_config, from_pretrained`; `vllm/model_executor/model_loader/bitsandbytes_loader.py` modified +2/-3 (5 lines); hunks: -10,7 +10,6; -48,6 +47,7; symbols: _get_weight_files, touching `_get_weight_files`.
- Code diff details:
  - `vllm/model_executor/model_loader/weight_utils.py` modified +7/-7 (14 lines); hunks: -23,7 +23,6; -46,6 +45,7; symbols: get_quant_config, get_sparse_attention_config, download_weights_from_hf
  - `vllm/tokenizers/kimi_audio.py` modified +4/-4 (8 lines); hunks: -10,13 +10,13; -78,7 +78,7 @@ def from_pretrained(; symbols: from_pretrained
  - `vllm/tokenizers/grok2.py` modified +3/-3 (6 lines); hunks: -8,7 +8,6; -20,6 +19,7; symbols: _maybe_load_tokenizer_config, from_pretrained
  - `vllm/model_executor/model_loader/bitsandbytes_loader.py` modified +2/-3 (5 lines); hunks: -10,7 +10,6; -48,6 +47,7; symbols: _get_weight_files
  - `vllm/model_executor/model_loader/gguf_loader.py` modified +2/-2 (4 lines); hunks: -8,7 +8,6; -27,6 +26,7; symbols: _prepare_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/model_loader/weight_utils.py
@@ -23,7 +23,6 @@
-from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
@@ -46,6 +45,7 @@
+from vllm.transformers_utils.repo_utils import hf_api, hf_fs
@@ -373,7 +373,7 @@ def get_quant_config(
-            hf_folder = snapshot_download(
+            hf_folder = hf_api().snapshot_download(
diff -- vllm/tokenizers/kimi_audio.py
@@ -10,13 +10,13 @@
-from huggingface_hub import hf_hub_download
+from vllm.transformers_utils.repo_utils import hf_api
@@ -78,7 +78,7 @@ def from_pretrained(
-                vocab_path = hf_hub_download(
+                vocab_path = hf_api().hf_hub_download(
@@ -87,7 +87,7 @@ def from_pretrained(
diff -- vllm/tokenizers/grok2.py
@@ -8,7 +8,6 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/model_loader/weight_utils.py` modified +7/-7; `vllm/tokenizers/kimi_audio.py` modified +4/-4; `vllm/tokenizers/grok2.py` modified +3/-3; `vllm/model_executor/model_loader/bitsandbytes_loader.py` modified +2/-3; `vllm/model_executor/model_loader/gguf_loader.py` modified +2/-2; `vllm/model_executor/model_loader/tensorizer.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/lora/test_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #44493 - [Bugfix]Fix Kimi-K2.5 FlashInfer ViT metadata

- Link: https://github.com/vllm-project/vllm/pull/44493
- Status/date: merged / 2026-06-04
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/44493 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `1bdc60ed53ad`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +109/-28, 260 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix]Fix Kimi-K2.5 FlashInfer ViT metadata"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix]Fix Kimi-K2.5 FlashInfer ViT metadata"; the main implementation surface is `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25_vit.py` modified +108/-27 (135 lines); hunks: -154,9 +154,12 @@ def __init__(; -218,7 +221,9 @@ def __init__(; symbols: __init__, reset_parameters, forward, touching `__init__, reset_parameters, forward`; `vllm/model_executor/models/kimi_k25.py` modified +1/-1 (2 lines); hunks: -235,7 +235,7 @@ def _get_mm_fields_config(; symbols: _get_mm_fields_config, _call_hf_processor, touching `_get_mm_fields_config, _call_hf_processor`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +108/-27 (135 lines); hunks: -154,9 +154,12 @@ def __init__(; -218,7 +221,9 @@ def __init__(; symbols: __init__, reset_parameters, forward
  - `vllm/model_executor/models/kimi_k25.py` modified +1/-1 (2 lines); hunks: -235,7 +235,7 @@ def _get_mm_fields_config(; symbols: _get_mm_fields_config, _call_hf_processor
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -154,9 +154,12 @@ def __init__(
-    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
+    def forward(
+        self, x: torch.Tensor, grid_thws: torch.Tensor | list[list[int]]
+    ) -> torch.Tensor:
-        for t, h, w in grid_thws.tolist():
+        grid_thw_list = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
diff -- vllm/model_executor/models/kimi_k25.py
@@ -235,7 +235,7 @@ def _get_mm_fields_config(
-            grid_thws=MultiModalFieldConfig.batched("vision_chunk"),
+            grid_thws=MultiModalFieldConfig.batched("vision_chunk", keep_on_cpu=True),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` modified +108/-27; `vllm/model_executor/models/kimi_k25.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44539 - [mamba] unify KDA conv states into one cache to match 2-state SSM layout

- Link: https://github.com/vllm-project/vllm/pull/44539
- Status/date: merged / 2026-06-04
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/44539 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +16/-30, 120 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[mamba] unify KDA conv states into one cache to match 2-state SSM layout"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `vllm/model_executor/models/kimi_linear.py`; technical summary: Covers "[mamba] unify KDA conv states into one cache to match 2-state SSM layout"; the main implementation surface is `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `vllm/model_executor/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/mamba/mamba_utils.py` modified +7/-19 (26 lines); hunks: -120,9 +120,9 @@ def kda_state_dtype(; -243,7 +243,7 @@ def kda_state_shape(; symbols: kda_state_dtype, MambaStateShapeCalculator, kda_state_shape, gated_delta_net_state_copy_func, touching `kda_state_dtype, MambaStateShapeCalculator, kda_state_shape`; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` modified +6/-6 (12 lines); hunks: -85,7 +85,7 @@ def kda_attention_fake(; -94,7 +94,7 @@ def get_state_dtype(; symbols: kda_attention_fake, KimiGatedDeltaNetAttention, get_state_dtype, get_state_shape, touching `kda_attention_fake, KimiGatedDeltaNetAttention, get_state_dtype`; `vllm/model_executor/models/kimi_linear.py` modified +3/-5 (8 lines); hunks: -600,15 +600,15 @@ def forward(; -628,9 +628,7 @@ def get_mamba_state_shape_from_config(; symbols: forward, get_mamba_state_dtype_from_config, get_mamba_state_shape_from_config, get_mamba_state_copy_func, touching `forward, get_mamba_state_dtype_from_config, get_mamba_state_shape_from_config`.
- Code diff details:
  - `vllm/model_executor/layers/mamba/mamba_utils.py` modified +7/-19 (26 lines); hunks: -120,9 +120,9 @@ def kda_state_dtype(; -243,7 +243,7 @@ def kda_state_shape(; symbols: kda_state_dtype, MambaStateShapeCalculator, kda_state_shape, gated_delta_net_state_copy_func
  - `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` modified +6/-6 (12 lines); hunks: -85,7 +85,7 @@ def kda_attention_fake(; -94,7 +94,7 @@ def get_state_dtype(; symbols: kda_attention_fake, KimiGatedDeltaNetAttention, get_state_dtype, get_state_shape
  - `vllm/model_executor/models/kimi_linear.py` modified +3/-5 (8 lines); hunks: -600,15 +600,15 @@ def forward(; -628,9 +628,7 @@ def get_mamba_state_shape_from_config(; symbols: forward, get_mamba_state_dtype_from_config, get_mamba_state_shape_from_config, get_mamba_state_copy_func
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/mamba/mamba_utils.py
@@ -120,9 +120,9 @@ def kda_state_dtype(
-    ):
+    ) -> tuple[torch.dtype, torch.dtype]:
-        return (state_dtype, state_dtype, state_dtype, torch.float32)
+        return (state_dtype, torch.float32)
@@ -243,7 +243,7 @@ def kda_state_shape(
-    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int, int]]:
diff -- vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py
@@ -85,7 +85,7 @@ def kda_attention_fake(
-    ) -> tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype]:
+    ) -> tuple[torch.dtype, torch.dtype]:
@@ -94,7 +94,7 @@ def get_state_dtype(
-    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
+    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
@@ -300,13 +300,13 @@ def _forward(
diff -- vllm/model_executor/models/kimi_linear.py
@@ -600,15 +600,15 @@ def forward(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/mamba/mamba_utils.py` modified +7/-19; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` modified +6/-6; `vllm/model_executor/models/kimi_linear.py` modified +3/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45003 - [Frontend] Support strict mode for tool calling

- Link: https://github.com/vllm-project/vllm/pull/45003
- Status/date: merged / 2026-06-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 29 files, +672/-1936, 3162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] Support strict mode for tool calling"; model line: Kimi K2/K2.5/K3/Linear/VL; category: docs/tests/CI; main diff: `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`; technical summary: Covers "[Frontend] Support strict mode for tool calling"; the main implementation surface is `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300 (1300 lines); hunks: -1,1300 +0,0; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks, touching `StreamingXMLToolCallParser, __init__, reset_streaming_state`; `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240 (414 lines); hunks: -1,14 +1,15; -24,23 +25,51; symbols: register_model_structural_tag, register_vllm_structural_tag, decorator, get_model_structural_tag, touching `register_model_structural_tag, register_vllm_structural_tag, decorator`; `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins, test_get_model_structural_tag_supports_vllm_hermes, touching `sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins`; `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190 (203 lines); hunks: -3,6 +3,7; -19,15 +20,12; symbols: qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized, assert_tool_calls, touching `qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized`.
- Code diff details:
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300 (1300 lines); hunks: -1,1300 +0,0; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks
  - `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240 (414 lines); hunks: -1,14 +1,15; -24,23 +25,51; symbols: register_model_structural_tag, register_vllm_structural_tag, decorator, get_model_structural_tag
  - `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins, test_get_model_structural_tag_supports_vllm_hermes
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190 (203 lines); hunks: -3,6 +3,7; -19,15 +20,12; symbols: qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized, assert_tool_calls
  - `tests/tool_parsers/test_qwen3xml_tool_parser.py` removed +0/-72 (72 lines); hunks: -1,72 +0,0; symbols: TestQwen3xmlToolParser, test_config
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/qwen3xml_tool_parser.py
@@ -1,1300 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-import json
-from collections.abc import Sequence
-from typing import Any
-from xml.parsers.expat import ParserCreate
diff -- vllm/tool_parsers/structural_tag_registry.py
@@ -1,14 +1,15 @@
-# Model-specific structural tag builders adapted from XGrammar's
-# builtin structural tag implementations:
-# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/builtin_structural_tag.py
-from xgrammar import StructuralTag
+from xgrammar import StructuralTag, normalize_tool_choice
+from xgrammar import get_model_structural_tag as get_xgrammar_model_structural_tag
diff -- tests/tool_parsers/test_structural_tag_registry.py
@@ -0,0 +1,314 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300; `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240; `vllm/tool_parsers/abstract_tool_parser.py` modified +36/-28; `vllm/entrypoints/serve/render/serving.py` modified +24/-28; `vllm/tool_parsers/deepseekv4_tool_parser.py` modified +1/-15
  - tests: `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0; `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190; `tests/tool_parsers/test_qwen3xml_tool_parser.py` removed +0/-72
- Risk and verification: The diff ships test coverage in `requirements/test/rocm.txt`, `tests/entrypoints/openai/chat_completion/test_completion_with_function_calling.py`, `tests/entrypoints/openai/responses/conftest.py`, `tests/parser/test_parse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41992 - [MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL

- Link: https://github.com/vllm-project/vllm/pull/41992
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`; associated commits `fa85ead2f378`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +498/-39, 726 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL"; model line: Kimi K2/K2.5/K3/Linear/VL; category: performance/backend optimization; main diff: `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL"; the main implementation surface is `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` modified +266/-37 (303 lines); hunks: -45,7 +45,9; -110,23 +112,42 @@ def __init__(; symbols: __init__, reset_parameters, forward, get_pos_embeds, touching `__init__, reset_parameters, forward`; `vllm/model_executor/models/kimi_vl.py` modified +195/-2 (197 lines); hunks: -56,7 +56,11; -79,6 +83,7; symbols: get_replacement, KimiVLForConditionalGeneration, __init__, get_encoder_cudagraph_config, touching `get_replacement, KimiVLForConditionalGeneration, __init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` modified +266/-37 (303 lines); hunks: -45,7 +45,9; -110,23 +112,42 @@ def __init__(; symbols: __init__, reset_parameters, forward, get_pos_embeds
  - `vllm/model_executor/models/kimi_vl.py` modified +195/-2 (197 lines); hunks: -56,7 +56,11; -79,6 +83,7; symbols: get_replacement, KimiVLForConditionalGeneration, __init__, get_encoder_cudagraph_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -45,7 +45,9 @@
+from typing import Any
+import numpy as np
@@ -110,23 +112,42 @@ def __init__(
-    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
-        pos_embs = []
-        for shape in grid_hws.tolist():
diff -- vllm/model_executor/models/kimi_vl.py
@@ -56,7 +56,11 @@
-from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
+from vllm.model_executor.models.interfaces import (
+    SupportsEncoderCudaGraph,
+    SupportsMultiModal,
+    SupportsPP,
+)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +266/-37; `vllm/model_executor/models/kimi_vl.py` modified +195/-2
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45424 - [Core] Ensure memory is pinned prior to async h2d copy

- Link: https://github.com/vllm-project/vllm/pull/45424
- Status/date: merged / 2026-06-21
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/45424 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 49 files, +254/-264, 1718 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Core] Ensure memory is pinned prior to async h2d copy"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/pooler/seqwise/methods.py`, `vllm/multimodal/inputs.py`; technical summary: Covers "[Core] Ensure memory is pinned prior to async h2d copy"; the main implementation surface is `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/pooler/seqwise/methods.py`, `vllm/multimodal/inputs.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/attention/mla_attention.py` modified +10/-8 (18 lines); hunks: -1684,12 +1684,13 @@ def build(; -1746,12 +1747,13 @@ def build(; symbols: build, touching `build`; `vllm/model_executor/layers/pooler/seqwise/methods.py` modified +8/-8 (16 lines); hunks: -10,6 +10,7; -74,15 +75,14 @@ def forward(; symbols: forward, touching `forward`; `vllm/multimodal/inputs.py` modified +14/-2 (16 lines); hunks: -488,7 +488,13 @@ def _reduce_data(; -538,7 +544,13 @@ def _reduce_data(; symbols: _reduce_data, touching `_reduce_data`; `vllm/model_executor/models/moonvit.py` modified +3/-2 (5 lines); hunks: -66,6 +66,7; -758,7 +759,7 @@ def prepare_encoder_metadata(; symbols: _apply_rope_input_validation, prepare_encoder_metadata, touching `_apply_rope_input_validation, prepare_encoder_metadata`.
- Code diff details:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +10/-8 (18 lines); hunks: -1684,12 +1684,13 @@ def build(; -1746,12 +1747,13 @@ def build(; symbols: build
  - `vllm/model_executor/layers/pooler/seqwise/methods.py` modified +8/-8 (16 lines); hunks: -10,6 +10,7; -74,15 +75,14 @@ def forward(; symbols: forward
  - `vllm/multimodal/inputs.py` modified +14/-2 (16 lines); hunks: -488,7 +488,13 @@ def _reduce_data(; -538,7 +544,13 @@ def _reduce_data(; symbols: _reduce_data
  - `vllm/model_executor/models/moonvit.py` modified +3/-2 (5 lines); hunks: -66,6 +66,7; -758,7 +759,7 @@ def prepare_encoder_metadata(; symbols: _apply_rope_input_validation, prepare_encoder_metadata
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-3 (5 lines); hunks: -83,9 +83,8; -825,7 +824,7 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen, invert_permutation
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -1684,12 +1684,13 @@ def build(
-                chunk_starts = (
+                chunk_starts = torch.empty(
+                    num_chunks, num_prefills, dtype=torch.int32, pin_memory=True
+                ).copy_(
+                    .multiply_(max_context_chunk)
-                    .expand(-1, num_prefills)
diff -- vllm/model_executor/layers/pooler/seqwise/methods.py
@@ -10,6 +10,7 @@
+from vllm.utils.torch_utils import async_tensor_h2d
@@ -74,15 +75,14 @@ def forward(
-        # Build segment_ids on CPU so repeat_interleave doesn't need to sync
-        # GPU->CPU to learn its data-dependent output length, then upload
-        # non-blocking. eg. [2, 1, 3] -> [0, 0, 1, 2, 2, 2]
+        prompt_lens = async_tensor_h2d(
diff -- vllm/multimodal/inputs.py
@@ -488,7 +488,13 @@ def _reduce_data(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/attention/mla_attention.py` modified +10/-8; `vllm/model_executor/layers/pooler/seqwise/methods.py` modified +8/-8; `vllm/multimodal/inputs.py` modified +14/-2; `vllm/model_executor/models/moonvit.py` modified +3/-2; `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-3; `vllm/model_executor/layers/attention/mm_encoder_attention.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/v1/logits_processors/test_correctness.py`, `tests/v1/streaming_input/test_gpu_model_runner_streaming.py`, `tests/v1/worker/test_gpu_input_batch.py`, `tests/v1/worker/test_gpu_model_runner.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #46610 - [Frontend] Add Streaming Parser Engine and new Kimi k2.5/k2.6/k2.7 Parser

- Link: https://github.com/vllm-project/vllm/pull/46610
- Status/date: merged / 2026-06-30
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/46610 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `vllm/parser/kimi_k2.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`; associated commits `2bc20e8abaf7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +397/-570, 1069 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] Add Streaming Parser Engine and new Kimi k2.5/k2.6/k2.7 Parser"; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `vllm/tool_parsers/kimi_k2_tool_parser.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`, `tests/reasoning/test_kimi_k2_reasoning_parser.py`; technical summary: Covers "[Frontend] Add Streaming Parser Engine and new Kimi k2.5/k2.6/k2.7 Parser"; the main implementation surface is `vllm/tool_parsers/kimi_k2_tool_parser.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`, `tests/reasoning/test_kimi_k2_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +4/-262 (266 lines); hunks: -1,278 +1,20; symbols: KimiK2ToolParser, __init__, adjust_request, extract_tool_calls, touching `KimiK2ToolParser, __init__, adjust_request`; `vllm/reasoning/kimi_k2_reasoning_parser.py` modified +3/-240 (243 lines); hunks: -1,245 +1,8; symbols: KimiK2ReasoningParser, __init__, reasoning_start_str, reasoning_end_str, touching `KimiK2ReasoningParser, __init__, reasoning_start_str`; `tests/reasoning/test_kimi_k2_reasoning_parser.py` modified +5/-67 (72 lines); hunks: -7,7 +7,6; -33,20 +32,6 @@ def kimi_k2_tokenizer():; symbols: kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled, test_extract_reasoning_with_think_tags, touching `kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled`; `vllm/parser/kimi_k2.py` added +285/-0 (285 lines); hunks: -0,0 +1,285; symbols: kimi_k2_config, KimiK2Parser, __init__, _extract_tool_id_and_name, touching `kimi_k2_config, KimiK2Parser, __init__`.
- Code diff details:
  - `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +4/-262 (266 lines); hunks: -1,278 +1,20; symbols: KimiK2ToolParser, __init__, adjust_request, extract_tool_calls
  - `vllm/reasoning/kimi_k2_reasoning_parser.py` modified +3/-240 (243 lines); hunks: -1,245 +1,8; symbols: KimiK2ReasoningParser, __init__, reasoning_start_str, reasoning_end_str
  - `tests/reasoning/test_kimi_k2_reasoning_parser.py` modified +5/-67 (72 lines); hunks: -7,7 +7,6; -33,20 +32,6 @@ def kimi_k2_tokenizer():; symbols: kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled, test_extract_reasoning_with_think_tags
  - `vllm/parser/kimi_k2.py` added +285/-0 (285 lines); hunks: -0,0 +1,285; symbols: kimi_k2_config, KimiK2Parser, __init__, _extract_tool_id_and_name
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/kimi_k2_tool_parser.py
@@ -1,278 +1,20 @@
-from collections.abc import Sequence
-import regex as re
-from vllm.entrypoints.openai.engine.protocol import (
-    DeltaFunctionCall,
-    DeltaMessage,
-    DeltaToolCall,
diff -- vllm/reasoning/kimi_k2_reasoning_parser.py
@@ -1,245 +1,8 @@
-from collections.abc import Iterable, Sequence
-from typing import TYPE_CHECKING
+from vllm.parser.engine.registered_adapters import KimiK2ParserReasoningAdapter
-from transformers import PreTrainedTokenizerBase
+KimiK2ReasoningParser = KimiK2ParserReasoningAdapter
-from vllm.entrypoints.openai.engine.protocol import DeltaMessage
diff -- tests/reasoning/test_kimi_k2_reasoning_parser.py
@@ -7,7 +7,6 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +4/-262; `vllm/reasoning/kimi_k2_reasoning_parser.py` modified +3/-240; `vllm/parser/kimi_k2.py` added +285/-0
  - tests: `tests/reasoning/test_kimi_k2_reasoning_parser.py` modified +5/-67
- Risk and verification: The diff ships test coverage in `tests/parser/engine/trace_builder.py`, `tests/reasoning/test_kimi_k2_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #47416 - [perf]Add fused Kimi image preprocessing

- Link: https://github.com/vllm-project/vllm/pull/47416
- Status/date: merged / 2026-07-06
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/47416 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/processors/kimi_k25_vision_fused.py`; associated commits `5ad11172b791`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +402/-2, 462 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[perf]Add fused Kimi image preprocessing"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `vllm/transformers_utils/processors/kimi_k25_vision_fused.py`, `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[perf]Add fused Kimi image preprocessing"; the main implementation surface is `vllm/transformers_utils/processors/kimi_k25_vision_fused.py`, `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/processors/kimi_k25_vision_fused.py` added +352/-0 (352 lines); hunks: -0,0 +1,352; symbols: _write_fused_patches, navit_resize_image, navit_resize_video, _to_pil, touching `_write_fused_patches, navit_resize_image, navit_resize_video`; `vllm/model_executor/models/kimi_k25.py` modified +10/-0 (10 lines); hunks: -56,6 +56,10; -108,10 +112,16 @@ def __init__(self, ctx: InputProcessingContext) -> None:; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/transformers_utils/processors/kimi_k25_vision_fused.py` added +352/-0 (352 lines); hunks: -0,0 +1,352; symbols: _write_fused_patches, navit_resize_image, navit_resize_video, _to_pil
  - `vllm/model_executor/models/kimi_k25.py` modified +10/-0 (10 lines); hunks: -56,6 +56,10; -108,10 +112,16 @@ def __init__(self, ctx: InputProcessingContext) -> None:; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/processors/kimi_k25_vision_fused.py
@@ -0,0 +1,352 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Optimized CPU image processor for Kimi-K2.5/K2.6 vision chunks."""
+import io
+import json
+import math
diff -- vllm/model_executor/models/kimi_k25.py
@@ -56,6 +56,10 @@
+from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
+    KimiK25FusedVisionProcessor,
+)
+from vllm.utils.import_utils import is_numba_available
@@ -108,10 +112,16 @@ def __init__(self, ctx: InputProcessingContext) -> None:
+        processor_cls = KimiK25FusedVisionProcessor if is_numba_available() else None
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/kimi_k25_vision_fused.py` added +352/-0; `vllm/model_executor/models/kimi_k25.py` modified +10/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/processor.py`, `vllm/transformers_utils/processors/kimi_k25_vision_fused.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50090 - [Kimi-K3] Add AttnRes kernels

- Link: https://github.com/vllm-project/vllm/pull/50090
- Status/date: merged / 2026-07-28
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_amd_attn_res.py`, `tests/models/kimi_k3/test_attn_res.py`, `vllm/models/kimi_k3/amd/__init__.py`, `vllm/models/kimi_k3/amd/ops/__init__.py`, `vllm/models/kimi_k3/amd/ops/attn_res.py` and 8 files; associated commits `61ac36802103`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 14 files, +1719/-0, 1776 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/ops/attn_res.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: get_attn_res_triton_warmup_profiles, _attn_res_kernel, attn_res, touching `get_attn_res_triton_warmup_profiles, _attn_res_kernel, attn_res`; `tests/models/kimi_k3/test_attn_res.py` added +193/-0 (193 lines); hunks: -0,0 +1,193; symbols: _randn_with_row_padding, _reference, test_attn_res, test_attn_res_block_counts, touching `_randn_with_row_padding, _reference, test_attn_res`; `vllm/models/kimi_k3/amd/ops/attn_res.py` added +132/-0 (132 lines); hunks: -0,0 +1,132; symbols: _attn_res_kernel, attn_res, touching `_attn_res_kernel, attn_res`; `tests/models/kimi_k3/test_amd_attn_res.py` added +102/-0 (102 lines); hunks: -0,0 +1,102; symbols: _randn_with_row_padding, _reference, test_amd_attn_res_matches_reference, touching `_randn_with_row_padding, _reference, test_amd_attn_res_matches_reference`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/ops/attn_res.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: get_attn_res_triton_warmup_profiles, _attn_res_kernel, attn_res
  - `tests/models/kimi_k3/test_attn_res.py` added +193/-0 (193 lines); hunks: -0,0 +1,193; symbols: _randn_with_row_padding, _reference, test_attn_res, test_attn_res_block_counts
  - `vllm/models/kimi_k3/amd/ops/attn_res.py` added +132/-0 (132 lines); hunks: -0,0 +1,132; symbols: _attn_res_kernel, attn_res
  - `tests/models/kimi_k3/test_amd_attn_res.py` added +102/-0 (102 lines); hunks: -0,0 +1,102; symbols: _randn_with_row_padding, _reference, test_amd_attn_res_matches_reference
  - `vllm/models/kimi_k3/nvidia/ops/__init__.py` added +6/-0 (6 lines); hunks: -0,0 +1,6
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/ops/attn_res.py
@@ -0,0 +1,245 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
+#
+# This file contains code adapted from the flash-linear-attention project.
+# The original source code was licensed under the MIT license and included
diff -- tests/models/kimi_k3/test_attn_res.py
@@ -0,0 +1,193 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+import torch
+import torch.nn.functional as F
+from vllm.models.kimi_k3.nvidia.ops import attn_res
diff -- vllm/models/kimi_k3/amd/ops/attn_res.py
@@ -0,0 +1,132 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/ops/attn_res.py` added +245/-0; `vllm/models/kimi_k3/amd/ops/attn_res.py` added +132/-0; `vllm/models/kimi_k3/nvidia/ops/__init__.py` added +6/-0; `vllm/models/kimi_k3/amd/__init__.py` added +2/-0; `vllm/models/kimi_k3/nvidia/__init__.py` added +2/-0; `vllm/models/kimi_k3/amd/ops/__init__.py` added +0/-0
  - tests: `tests/models/kimi_k3/test_attn_res.py` added +193/-0; `tests/models/kimi_k3/test_amd_attn_res.py` added +102/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_amd_attn_res.py`, `tests/models/kimi_k3/test_attn_res.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50131 - [Bugfix] Add missing `vllm/models/kimi_k3/__init__.py`

- Link: https://github.com/vllm-project/vllm/pull/50131
- Status/date: merged / 2026-07-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/__init__.py`, `vllm/models/kimi_k3/amd/ops/__init__.py`; associated commits `62d8db7c05af`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +4/-0, 6 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2; `vllm/models/kimi_k3/amd/ops/__init__.py` modified +2/-0 (2 lines); hunks: -0,0 +1,2.
- Code diff details:
  - `vllm/models/kimi_k3/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
  - `vllm/models/kimi_k3/amd/ops/__init__.py` modified +2/-0 (2 lines); hunks: -0,0 +1,2
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/__init__.py
@@ -0,0 +1,2 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
diff -- vllm/models/kimi_k3/amd/ops/__init__.py
@@ -0,0 +1,2 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/__init__.py` added +2/-0; `vllm/models/kimi_k3/amd/ops/__init__.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/__init__.py`, `vllm/models/kimi_k3/amd/ops/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50089 - [Model] Add Kimi K3 support: model files and kernels [1/N]

- Link: https://github.com/vllm-project/vllm/pull/50089
- Status/date: merged / 2026-07-29
- Trace source: `git log --name-only -- <model-files>` found it through `benchmarks/kernels/benchmark_kimi_k3_latent_moe_tail.py`, `benchmarks/kernels/benchmark_kimi_k3_sp_collectives.py`, `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py`, `tests/models/kimi_k3/test_attn_res.py`, `tests/models/kimi_k3/test_eagle3.py` and 47 files; associated commits `7c6729b76959`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 109 files, +19973/-773, 22075 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/amd/linear.py` added +1065/-0 (1065 lines); hunks: -0,0 +1,1065; symbols: KimiMLP, __init__, forward, KimiRoutedOutputTransform, touching `KimiMLP, __init__, forward`; `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` added +935/-0 (935 lines); hunks: -0,0 +1,935; symbols: recompute_w_u_fwd_kernel, recompute_w_u_fwd, chunk_gla_fwd_kernel_o, chunk_gla_fwd_o_gk, touching `recompute_w_u_fwd_kernel, recompute_w_u_fwd, chunk_gla_fwd_kernel_o`; `vllm/models/kimi_k3/nvidia/kda.py` added +775/-0 (775 lines); hunks: -0,0 +1,775; symbols: a_log_weight_loader, loader, _KimiGDNMergedColumnParallelLinear, __init__, touching `a_log_weight_loader, loader, _KimiGDNMergedColumnParallelLinear`; `tests/models/kimi_k3/test_kda.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: test_gather_initial_states_correctness, naive_recurrent_kda, assert_close, test_chunk_kda, touching `test_gather_initial_states_correctness, naive_recurrent_kda, assert_close`.
- Code diff details:
  - `vllm/models/kimi_k3/amd/linear.py` added +1065/-0 (1065 lines); hunks: -0,0 +1,1065; symbols: KimiMLP, __init__, forward, KimiRoutedOutputTransform
  - `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` added +935/-0 (935 lines); hunks: -0,0 +1,935; symbols: recompute_w_u_fwd_kernel, recompute_w_u_fwd, chunk_gla_fwd_kernel_o, chunk_gla_fwd_o_gk
  - `vllm/models/kimi_k3/nvidia/kda.py` added +775/-0 (775 lines); hunks: -0,0 +1,775; symbols: a_log_weight_loader, loader, _KimiGDNMergedColumnParallelLinear, __init__
  - `tests/models/kimi_k3/test_kda.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: test_gather_initial_states_correctness, naive_recurrent_kda, assert_close, test_chunk_kda
  - `vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py` added +671/-0 (671 lines); hunks: -0,0 +1,671; symbols: _kda_gate_beta_fwd_kernel, _fused_kda_gate_beta, fused_recurrent_kda_fwd_kernel, get_fused_recurrent_kda_fwd_warmup_profiles
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/amd/linear.py
@@ -0,0 +1,1065 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable
+from typing import Any
+import torch
+from torch import nn
diff -- vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py
@@ -0,0 +1,935 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
+#
+# This file contains code copied from the flash-linear-attention project.
+# The original source code was licensed under the MIT license and included
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -0,0 +1,775 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/amd/linear.py` added +1065/-0; `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` added +935/-0; `vllm/models/kimi_k3/nvidia/kda.py` added +775/-0; `vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py` added +671/-0; `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py` added +662/-0; `vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py` added +621/-0
  - tests: `tests/models/kimi_k3/test_kda.py` added +757/-0
- Risk and verification: The diff ships test coverage in `tests/distributed/test_custom_all_reduce.py`, `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py`, `tests/kernels/core/test_activation.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50093 - [Model] Add Kimi K3 support: Python frontend [2/2]

- Link: https://github.com/vllm-project/vllm/pull/50093
- Status/date: merged / 2026-07-29
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_kimi_k3_reasoning_parser.py`, `tests/renderers/test_kimi_k3.py`, `tests/tool_parsers/test_kimi_k3_named_tool_choice.py`, `tests/tool_use/test_kimi_k3_tool_parser.py`, `vllm/parser/kimi_k3.py` and 8 files; associated commits `f5a7cce9b6a6`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 20 files, +3056/-5, 3209 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/tool_use/test_kimi_k3_tool_parser.py` added +600/-0 (600 lines); hunks: -0,0 +1,600; symbols: DummyTokenizer, get_vocab, encode, KimiK3DelegatingParser, touching `DummyTokenizer, get_vocab, encode`; `vllm/tool_parsers/kimi_k3_tool_parser.py` added +405/-0 (405 lines); hunks: -0,0 +1,405; symbols: _partial_tag_overlap, KimiK3ToolParser, __init__, adjust_request, touching `_partial_tag_overlap, KimiK3ToolParser, __init__`; `vllm/reasoning/kimi_k3_reasoning_parser.py` added +371/-0 (371 lines); hunks: -0,0 +1,371; symbols: _subseq_index, KimiK3ReasoningParser, __init__, reasoning_start_str, touching `_subseq_index, KimiK3ReasoningParser, __init__`; `tests/reasoning/test_kimi_k3_reasoning_parser.py` added +250/-0 (250 lines); hunks: -0,0 +1,250; symbols: DummyTokenizer, get_vocab, encode, ReasoningOnlyParser, touching `DummyTokenizer, get_vocab, encode`.
- Code diff details:
  - `tests/tool_use/test_kimi_k3_tool_parser.py` added +600/-0 (600 lines); hunks: -0,0 +1,600; symbols: DummyTokenizer, get_vocab, encode, KimiK3DelegatingParser
  - `vllm/tool_parsers/kimi_k3_tool_parser.py` added +405/-0 (405 lines); hunks: -0,0 +1,405; symbols: _partial_tag_overlap, KimiK3ToolParser, __init__, adjust_request
  - `vllm/reasoning/kimi_k3_reasoning_parser.py` added +371/-0 (371 lines); hunks: -0,0 +1,371; symbols: _subseq_index, KimiK3ReasoningParser, __init__, reasoning_start_str
  - `tests/reasoning/test_kimi_k3_reasoning_parser.py` added +250/-0 (250 lines); hunks: -0,0 +1,250; symbols: DummyTokenizer, get_vocab, encode, ReasoningOnlyParser
  - `tests/tool_parsers/test_kimi_k3_named_tool_choice.py` added +62/-0 (62 lines); hunks: -0,0 +1,62; symbols: _DummyTokenizer, get_vocab, encode, _request
- Key code excerpts:

```diff
diff -- tests/tool_use/test_kimi_k3_tool_parser.py
@@ -0,0 +1,600 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+import pytest
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionRequest,
diff -- vllm/tool_parsers/kimi_k3_tool_parser.py
@@ -0,0 +1,405 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tool-call parser for the Kimi K3 (XTML) chat format.
+This turns the generated XTML ``response`` and ``tools`` channels back into
+OpenAI-compatible ``content`` and ``tool_calls``.
+K3 assistant tool calls live in a nested ``tools`` channel::
diff -- vllm/reasoning/kimi_k3_reasoning_parser.py
@@ -0,0 +1,371 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/tool_use/test_kimi_k3_tool_parser.py` added +600/-0; `tests/reasoning/test_kimi_k3_reasoning_parser.py` added +250/-0; `tests/tool_parsers/test_kimi_k3_named_tool_choice.py` added +62/-0; `tests/renderers/test_kimi_k3.py` added +353/-0
  - runtime: `vllm/tool_parsers/kimi_k3_tool_parser.py` added +405/-0; `vllm/reasoning/kimi_k3_reasoning_parser.py` added +371/-0; `vllm/renderers/kimi_k3.py` added +220/-0; `vllm/parser/kimi_k3.py` added +135/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_kimi_k3_reasoning_parser.py`, `tests/renderers/test_kimi_k3.py`, `tests/tool_parsers/test_kimi_k3_named_tool_choice.py`, `tests/tool_parsers/test_structural_tag_registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50262 - [ROCm][CI] Fix Kimi K3 KDA on ROCm

- Link: https://github.com/vllm-project/vllm/pull/50262
- Status/date: merged / 2026-07-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/kda.py`; associated commits `381b69162020`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +8/-2, 26 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/kda.py` modified +2/-0 (2 lines); hunks: -164,6 +164,8 @@ def is_flashkda_supported(; symbols: is_flashkda_supported, touching `is_flashkda_supported`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +2/-0 (2 lines); hunks: -164,6 +164,8 @@ def is_flashkda_supported(; symbols: is_flashkda_supported
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -164,6 +164,8 @@ def is_flashkda_supported(
+    if not current_platform.is_cuda():
+        return False
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/kda.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/kda.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50000 - [New model] Kimi K3

- Link: https://github.com/vllm-project/vllm/pull/50000
- Status/date: merged / 2026-07-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/warmup/kimi_k3_triton_warmup.py`, `vllm/models/kimi_k3/nvidia/dspark_mla.py`, `vllm/models/kimi_k3/nvidia/model.py` and 6 files; associated commits `aeeb36b1f171`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 82 files, +2931/-1349, 6277 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` modified +482/-291 (773 lines); hunks: -1,32 +1,33; -36,49 +37,118; symbols: kda_attention, kda_attention_fake, a_log_weight_loader, loader, touching `kda_attention, kda_attention_fake, a_log_weight_loader`; `vllm/model_executor/models/kimi_linear.py` removed +0/-646 (646 lines); hunks: -1,646 +0,0; symbols: KimiMLP, __init__, forward, KimiMoE, touching `KimiMLP, __init__, forward`; `vllm/model_executor/models/kimi_k25_vit.py` modified +204/-25 (229 lines); hunks: -34,6 +34,7; -154,9 +155,7 @@ def __init__(; symbols: __init__, reset_parameters, forward, get_pos_embeds, touching `__init__, reset_parameters, forward`; `vllm/model_executor/warmup/kimi_k3_triton_warmup.py` added +182/-0 (182 lines); hunks: -0,0 +1,182; symbols: _get_kda_layer, _warm_attn_res, _warm_recurrent_kda, kimi_k3_triton_warmup, touching `_get_kda_layer, _warm_attn_res, _warm_recurrent_kda`.
- Code diff details:
  - `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` modified +482/-291 (773 lines); hunks: -1,32 +1,33; -36,49 +37,118; symbols: kda_attention, kda_attention_fake, a_log_weight_loader, loader
  - `vllm/model_executor/models/kimi_linear.py` removed +0/-646 (646 lines); hunks: -1,646 +0,0; symbols: KimiMLP, __init__, forward, KimiMoE
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +204/-25 (229 lines); hunks: -34,6 +34,7; -154,9 +155,7 @@ def __init__(; symbols: __init__, reset_parameters, forward, get_pos_embeds
  - `vllm/model_executor/warmup/kimi_k3_triton_warmup.py` added +182/-0 (182 lines); hunks: -0,0 +1,182; symbols: _get_kda_layer, _warm_attn_res, _warm_recurrent_kda, kimi_k3_triton_warmup
  - `tests/models/test_dspark_mla.py` added +143/-0 (143 lines); hunks: -0,0 +1,143; symbols: test_dspark_mla_uses_compile_free_model_entrypoint, test_dspark_mla_checkpoint_weight_mapping, test_dspark_mla_shares_frozen_target_weights_and_skips_training_head, test_dspark_markov_head_is_replicated
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py
@@ -1,32 +1,33 @@
+from collections.abc import Callable
+from torch.nn.parameter import Parameter
-from vllm.config import VllmConfig, get_current_vllm_config
-from vllm.distributed import (
-    divide,
-)
diff -- vllm/model_executor/models/kimi_linear.py
@@ -1,646 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-from collections.abc import Iterable
-import torch
-from torch import nn
-from vllm.compilation.decorators import support_torch_compile
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -34,6 +34,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` modified +482/-291; `vllm/model_executor/models/kimi_linear.py` removed +0/-646; `vllm/model_executor/models/kimi_k25_vit.py` modified +204/-25; `vllm/model_executor/warmup/kimi_k3_triton_warmup.py` added +182/-0; `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +1/-21; `vllm/transformers_utils/configs/kimi_linear.py` modified +17/-4
  - tests: `tests/models/test_dspark_mla.py` added +143/-0
- Risk and verification: The diff ships test coverage in `requirements/test/cuda.txt`, `tests/kernels/moe/test_deepgemm.py`, `tests/models/registry.py`, `tests/models/test_dspark_mla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50420 - [Frontend][Bugfix] Use default tool call IDs for Kimi K3 for conversation-level uniqueness

- Link: https://github.com/vllm-project/vllm/pull/50420
- Status/date: merged / 2026-07-31
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_use/test_kimi_k3_tool_parser.py`, `vllm/tool_parsers/kimi_k3_tool_parser.py`; associated commits `ab98034d4ccc`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +55/-53, 275 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/tool_use/test_kimi_k3_tool_parser.py` modified +9/-3 (12 lines); hunks: -141,7 +141,6 @@ def test_extract_tool_calls_with_response_and_typed_argument...; -168,7 +167,6 @@ def test_delegating_parser_preserves_tool_calls_after_reason...; symbols: test_extract_tool_calls_with_response_and_typed_arguments, test_delegating_parser_preserves_tool_calls_after_reasoning, test_streaming_split_markers_do_not_leak, test_tool_call_ids_are_unique_across_messages, touching `test_extract_tool_calls_with_response_and_typed_arguments, test_delegating_parser_preserves_tool_calls_after_reasoning, test_streaming_split_markers_do_not_leak`; `vllm/tool_parsers/kimi_k3_tool_parser.py` modified +0/-10 (10 lines); hunks: -218,7 +218,6 @@ def _decode_call(self, attrs: str, body: str) -> ToolCall |...; -234,16 +233,7 @@ def _decode_call(self, attrs: str, body: str) -> ToolCall |...; symbols: _decode_call, touching `_decode_call`.
- Code diff details:
  - `tests/tool_use/test_kimi_k3_tool_parser.py` modified +9/-3 (12 lines); hunks: -141,7 +141,6 @@ def test_extract_tool_calls_with_response_and_typed_argument...; -168,7 +167,6 @@ def test_delegating_parser_preserves_tool_calls_after_reason...; symbols: test_extract_tool_calls_with_response_and_typed_arguments, test_delegating_parser_preserves_tool_calls_after_reasoning, test_streaming_split_markers_do_not_leak, test_tool_call_ids_are_unique_across_messages
  - `vllm/tool_parsers/kimi_k3_tool_parser.py` modified +0/-10 (10 lines); hunks: -218,7 +218,6 @@ def _decode_call(self, attrs: str, body: str) -> ToolCall |...; -234,16 +233,7 @@ def _decode_call(self, attrs: str, body: str) -> ToolCall |...; symbols: _decode_call
- Key code excerpts:

```diff
diff -- tests/tool_use/test_kimi_k3_tool_parser.py
@@ -141,7 +141,6 @@ def test_extract_tool_calls_with_response_and_typed_arguments():
-    assert tool_call.id == "calc:0"
@@ -168,7 +167,6 @@ def test_delegating_parser_preserves_tool_calls_after_reasoning():
-    assert tool_calls[0].id == "calc:0"
@@ -364,11 +362,19 @@ def test_streaming_split_markers_do_not_leak():
-    assert tool_deltas[0].id == "calc:0"
+def test_tool_call_ids_are_unique_across_messages():
diff -- vllm/tool_parsers/kimi_k3_tool_parser.py
@@ -218,7 +218,6 @@ def _decode_call(self, attrs: str, body: str) -> ToolCall | None:
-        tool_index = call_attrs.get("index", "")
@@ -234,16 +233,7 @@ def _decode_call(self, attrs: str, body: str) -> ToolCall | None:
-        tool_call_id = tool_name
-        if tool_index:
-            try:
-                tool_call_id = f"{tool_name}:{int(tool_index) - 1}"
```

- Extracted files (not manually reviewed):
  - tests: `tests/tool_use/test_kimi_k3_tool_parser.py` modified +9/-3
  - runtime: `vllm/tool_parsers/kimi_k3_tool_parser.py` modified +0/-10
- Risk and verification: The diff ships test coverage in `tests/tool_use/test_kimi_k3_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50242 - K3 DSpark AR fusion

- Link: https://github.com/vllm-project/vllm/pull/50242
- Status/date: merged / 2026-07-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/dspark_mla.py`; associated commits `92643d68f551`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +24/-8, 104 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +17/-3 (20 lines); hunks: -20,6 +20,7; -54,11 +55,15 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +17/-3 (20 lines); hunks: -20,6 +20,7; -54,11 +55,15 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/dspark_mla.py
@@ -20,6 +20,7 @@
+from vllm.models.common.ops import fused_allreduce_rms_norm
@@ -54,11 +55,15 @@ def __init__(
+        # Both row-parallel outputs stay un-reduced; their all-reduces are fused
+        # into the RMSNorm that follows via fused_allreduce_rms_norm.
+        self.self_attn.o_proj.reduce_results = False
+            reduce_results=False,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +17/-3
- Risk and verification: Runtime changes concentrate in `vllm/models/common/ops/__init__.py`, `vllm/models/common/ops/fused_allreduce_rms_norm.py`, `vllm/models/deepseek_v32/amd/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50500 - [Compressed-Tensors] Support Kimi-K3 quantized models

- Link: https://github.com/vllm-project/vllm/pull/50500
- Status/date: merged / 2026-07-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/model.py`; associated commits `d87d2ca74791`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +8/-1, 23 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/model.py` modified +8/-1 (9 lines); hunks: -951,7 +951,13 @@ def forward(; -1221,6 +1227,7 @@ def load_weights(; symbols: forward, KimiLinearModel, __init__, load_weights, touching `forward, KimiLinearModel, __init__`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/model.py` modified +8/-1 (9 lines); hunks: -951,7 +951,13 @@ def forward(; -1221,6 +1227,7 @@ def load_weights(; symbols: forward, KimiLinearModel, __init__, load_weights
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -951,7 +951,13 @@ def forward(
-class KimiLinearModel(nn.Module, EagleModelMixin):
+class KimiLinearModel(nn.Module, EagleModelMixin, SupportsQuant):
+    packed_modules_mapping = {
+        "gate_up_proj": ["gate_proj", "up_proj"],
+        "in_proj_qkvgfab": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj"],
+        "conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +8/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50761 - [ROCm][Bugfix][Kimi-K3] Preserve MoE correction bias in FP32

- Link: https://github.com/vllm-project/vllm/pull/50761
- Status/date: merged / 2026-08-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/amd/linear.py`; associated commits `c6668106760e`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +4/-1, 12 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/amd/linear.py` modified +4/-1 (5 lines); hunks: -210,7 +210,10 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/kimi_k3/amd/linear.py` modified +4/-1 (5 lines); hunks: -210,7 +210,10 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/amd/linear.py
@@ -210,7 +210,10 @@ def __init__(
-        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(num_experts))
+        # Preserve FP32 checkpoint values and match FP32 router logits.
+        self.gate.e_score_correction_bias = nn.Parameter(
+            torch.empty(num_experts, dtype=torch.float32)
+        )
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/amd/linear.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/amd/linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50383 - Shard the K3 Latent-MoE up-projection on large batches

- Link: https://github.com/vllm-project/vllm/pull/50383
- Status/date: merged / 2026-08-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/model.py`; associated commits `4635cc3e8f60`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +176/-72, 316 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/model.py` modified +2/-14 (16 lines); hunks: -472,8 +472,8 @@ def __init__(; -553,13 +553,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/model.py` modified +2/-14 (16 lines); hunks: -472,8 +472,8 @@ def __init__(; -553,13 +553,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -472,8 +472,8 @@ def __init__(
-                hidden_size=config.hidden_size,
-                intermediate_size=shared_intermediate_size,
+                hidden_size=config.hidden_size,  # 7618
+                intermediate_size=shared_intermediate_size,  # 3072*2
@@ -553,13 +553,6 @@ def __init__(
-            # The tail-fusion kernels are tcgen05-based, so they require an
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +2/-14
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/runner/latent_moe_runner.py`, `vllm/models/kimi_k3/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50678 - K3: Move LatentMoERunner

- Link: https://github.com/vllm-project/vllm/pull/50678
- Status/date: merged / 2026-08-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/latent_moe_runner.py`, `vllm/models/kimi_k3/nvidia/model.py`; associated commits `b9d1e2437e1a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +4/-5, 35 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/model.py` modified +3/-3 (6 lines); hunks: -30,9 +30,6; -97,6 +94,9; `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` renamed +1/-2 (3 lines); hunks: -11,13 +11,12.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/model.py` modified +3/-3 (6 lines); hunks: -30,9 +30,6; -97,6 +94,9
  - `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` renamed +1/-2 (3 lines); hunks: -11,13 +11,12
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -30,9 +30,6 @@
-from vllm.model_executor.layers.fused_moe.runner.latent_moe_runner import (
-    LatentMoERunner,
-)
@@ -97,6 +94,9 @@
+from vllm.models.kimi_k3.nvidia.latent_moe_runner import (
+    LatentMoERunner,
diff -- vllm/models/kimi_k3/nvidia/latent_moe_runner.py
@@ -11,13 +11,12 @@
+from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner, _unpack
-from .moe_runner import MoERunner, _unpack
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +3/-3; `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` renamed +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/latent_moe_runner.py`, `vllm/models/kimi_k3/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50656 - [Kimi-K3] Add option to shard the shared expert instead of replicating

- Link: https://github.com/vllm-project/vllm/pull/50656
- Status/date: merged / 2026-08-03
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_sequence_parallel.py`, `vllm/models/kimi_k3/nvidia/model.py`; associated commits `5df9999fcfaa`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +154/-3, 242 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_sequence_parallel.py` modified +79/-0 (79 lines); hunks: -10,6 +10,7; -265,6 +266,84 @@ def test_kimi_mtp_restores_sequence_parallel_output(monkeyp...; symbols: test_kimi_mtp_restores_sequence_parallel_output, test_shard_sequence_parallel_mlp_gating, test_sharded_sequence_parallel_mlp_matches_replicated, test_sp_all_gather_uses_custom_kernel, touching `test_kimi_mtp_restores_sequence_parallel_output, test_shard_sequence_parallel_mlp_gating, test_sharded_sequence_parallel_mlp_matches_replicated`; `vllm/models/kimi_k3/nvidia/model.py` modified +62/-3 (65 lines); hunks: -129,7 +129,42; -138,27 +173,38 @@ def __init__(; symbols: shard_sequence_parallel_mlp, KimiMLP, __init__, touching `shard_sequence_parallel_mlp, KimiMLP, __init__`.
- Code diff details:
  - `tests/models/kimi_k3/test_sequence_parallel.py` modified +79/-0 (79 lines); hunks: -10,6 +10,7; -265,6 +266,84 @@ def test_kimi_mtp_restores_sequence_parallel_output(monkeyp...; symbols: test_kimi_mtp_restores_sequence_parallel_output, test_shard_sequence_parallel_mlp_gating, test_sharded_sequence_parallel_mlp_matches_replicated, test_sp_all_gather_uses_custom_kernel
  - `vllm/models/kimi_k3/nvidia/model.py` modified +62/-3 (65 lines); hunks: -129,7 +129,42; -138,27 +173,38 @@ def __init__(; symbols: shard_sequence_parallel_mlp, KimiMLP, __init__
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_sequence_parallel.py
@@ -10,6 +10,7 @@
+from vllm.model_executor.layers.activation import SiluAndMul
@@ -265,6 +266,84 @@ def test_kimi_mtp_restores_sequence_parallel_output(monkeypatch):
+@pytest.mark.parametrize(
+    ("enabled", "use_sequence_parallel", "eligible", "tp_size", "expected"),
+    [
+        (True, True, True, 8, True),
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -129,7 +129,42 @@
+def shard_sequence_parallel_mlp(
+    hidden_size: int,
+    intermediate_size: int,
+    use_sequence_parallel: bool,
+    eligible: bool,
+) -> bool:
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_sequence_parallel.py` modified +79/-0
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +62/-3
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_sequence_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50567 - [Bugfix][Kimi-K3] Enforce packed rows and op availability in AttnRes dispatch

- Link: https://github.com/vllm-project/vllm/pull/50567
- Status/date: merged / 2026-08-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/ops/attn_res.py`; associated commits `41ba11b8413c`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +11/-1, 33 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/ops/attn_res.py` modified +4/-1 (5 lines); hunks: -187,14 +187,17 @@ def attn_res(; symbols: attn_res, touching `attn_res`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/ops/attn_res.py` modified +4/-1 (5 lines); hunks: -187,14 +187,17 @@ def attn_res(; symbols: attn_res
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/ops/attn_res.py
@@ -187,14 +187,17 @@ def attn_res(
-    # Triton handles block boundaries and final pre-norm output.
+    # Triton handles block boundaries and final pre-norm output. The native op
+    # is only compiled for SM100 under CUDA >= 13, so a device check alone is
+    # not enough to know it exists.
+        and hasattr(torch.ops._C, "kimi_k3_attn_res")
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/ops/attn_res.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/ops/attn_res.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50886 - [Bugfix][Reasoning] kimi_k3: O(delta) reasoning-end check on the decode path

- Link: https://github.com/vllm-project/vllm/pull/50886
- Status/date: merged / 2026-08-04
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_kimi_k3_reasoning_parser.py`, `vllm/reasoning/kimi_k3_reasoning_parser.py`; associated commits `adbf08d977fb`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +157/-10, 202 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/reasoning/test_kimi_k3_reasoning_parser.py` modified +95/-0 (95 lines); hunks: -248,3 +248,98 @@ def test_adjust_request_keeps_xtml_markers_contiguous():; symbols: test_adjust_request_keeps_xtml_markers_contiguous, _reference_is_reasoning_end, last, test_is_reasoning_end_streaming_only_scans_the_step_window, touching `test_adjust_request_keeps_xtml_markers_contiguous, _reference_is_reasoning_end, last`; `vllm/reasoning/kimi_k3_reasoning_parser.py` modified +62/-10 (72 lines); hunks: -22,7 +22,7; -36,17 +36,47; symbols: _match_at, _subseq_index, _newest_marker, KimiK3ReasoningParser, touching `_match_at, _subseq_index, _newest_marker`.
- Code diff details:
  - `tests/reasoning/test_kimi_k3_reasoning_parser.py` modified +95/-0 (95 lines); hunks: -248,3 +248,98 @@ def test_adjust_request_keeps_xtml_markers_contiguous():; symbols: test_adjust_request_keeps_xtml_markers_contiguous, _reference_is_reasoning_end, last, test_is_reasoning_end_streaming_only_scans_the_step_window
  - `vllm/reasoning/kimi_k3_reasoning_parser.py` modified +62/-10 (72 lines); hunks: -22,7 +22,7; -36,17 +36,47; symbols: _match_at, _subseq_index, _newest_marker, KimiK3ReasoningParser
- Key code excerpts:

```diff
diff -- tests/reasoning/test_kimi_k3_reasoning_parser.py
@@ -248,3 +248,98 @@ def test_adjust_request_keeps_xtml_markers_contiguous():
+OPEN_IDS = [1, 2, 3]
+CLOSE_IDS = [4, 2, 3]
+def _reference_is_reasoning_end(input_ids: list[int]) -> bool:
+    """Full-sequence reference for the streaming check to be measured against.
+    Two independent last-occurrence scans, i.e. the straightforward reading of
+    "reasoning ended iff the newest think marker is a close marker".
diff -- vllm/reasoning/kimi_k3_reasoning_parser.py
@@ -22,7 +22,7 @@
-from collections.abc import Sequence
+from collections.abc import Iterable, Sequence
@@ -36,17 +36,47 @@
+def _match_at(haystack: Sequence[int], i: int, needle: Sequence[int]) -> bool:
+    """Whether *needle* occurs in *haystack* starting at *i*.
+    Compares element by element rather than slicing: ``haystack`` is usually a
```

- Extracted files (not manually reviewed):
  - tests: `tests/reasoning/test_kimi_k3_reasoning_parser.py` modified +95/-0
  - runtime: `vllm/reasoning/kimi_k3_reasoning_parser.py` modified +62/-10
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_kimi_k3_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50593 - [Kimi-K3][AMD] Fuse AttnRes state updates and norms

- Link: https://github.com/vllm-project/vllm/pull/50593
- Status/date: merged / 2026-08-04
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_amd_attn_res.py`, `vllm/models/kimi_k3/amd/linear.py`, `vllm/models/kimi_k3/amd/ops/attn_res.py`; associated commits `7ac2ec758208`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +208/-49, 393 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/amd/ops/attn_res.py` modified +101/-40 (141 lines); hunks: -7,6 +7,7; -15,69 +16,119; symbols: _attn_res_kernel, attn_res, touching `_attn_res_kernel, attn_res`; `tests/models/kimi_k3/test_amd_attn_res.py` modified +91/-0 (91 lines); hunks: -88,15 +88,106 @@ def test_amd_attn_res_matches_reference(; symbols: test_amd_attn_res_matches_reference, test_amd_attn_res_fused_contract, touching `test_amd_attn_res_matches_reference, test_amd_attn_res_fused_contract`; `vllm/models/kimi_k3/amd/linear.py` modified +16/-9 (25 lines); hunks: -141,17 +141,22 @@ def _apply_attn_res(; -610,19 +615,20 @@ def forward_attn_residual(; symbols: _apply_attn_res, forward_attn_residual, touching `_apply_attn_res, forward_attn_residual`.
- Code diff details:
  - `vllm/models/kimi_k3/amd/ops/attn_res.py` modified +101/-40 (141 lines); hunks: -7,6 +7,7; -15,69 +16,119; symbols: _attn_res_kernel, attn_res
  - `tests/models/kimi_k3/test_amd_attn_res.py` modified +91/-0 (91 lines); hunks: -88,15 +88,106 @@ def test_amd_attn_res_matches_reference(; symbols: test_amd_attn_res_matches_reference, test_amd_attn_res_fused_contract
  - `vllm/models/kimi_k3/amd/linear.py` modified +16/-9 (25 lines); hunks: -141,17 +141,22 @@ def _apply_attn_res(; -610,19 +615,20 @@ def forward_attn_residual(; symbols: _apply_attn_res, forward_attn_residual
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/amd/ops/attn_res.py
@@ -7,6 +7,7 @@
@@ -15,69 +16,119 @@
+    delta_ptr,
+    output_norm_weight_ptr,
+    stride_delta_m: tl.constexpr,
+    block_write_idx: tl.constexpr,
+    output_norm_eps: tl.constexpr,
diff -- tests/models/kimi_k3/test_amd_attn_res.py
@@ -88,15 +88,106 @@ def test_amd_attn_res_matches_reference(
+        None,
+        None,
+        -1,
+        0.0,
+@pytest.mark.parametrize(
+    (
diff -- vllm/models/kimi_k3/amd/linear.py
@@ -141,17 +141,22 @@ def _apply_attn_res(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/amd/ops/attn_res.py` modified +101/-40; `vllm/models/kimi_k3/amd/linear.py` modified +16/-9
  - tests: `tests/models/kimi_k3/test_amd_attn_res.py` modified +91/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_amd_attn_res.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50929 - [MM][CG] Support ViT full CUDA graph for Kimi-K2.5

- Link: https://github.com/vllm-project/vllm/pull/50929
- Status/date: merged / 2026-08-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `7cab4368f222`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +247/-1, 287 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/kimi_k25.py` modified +247/-1 (248 lines); hunks: -8,7 +8,7; -25,10 +25,19; symbols: KimiK25ForConditionalGeneration, set_aux_hidden_state_layers, get_eagle3_aux_hidden_state_layers, _encoder_cudagraph_pad_totals, touching `KimiK25ForConditionalGeneration, set_aux_hidden_state_layers, get_eagle3_aux_hidden_state_layers`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +247/-1 (248 lines); hunks: -8,7 +8,7; -25,10 +25,19; symbols: KimiK25ForConditionalGeneration, set_aux_hidden_state_layers, get_eagle3_aux_hidden_state_layers, _encoder_cudagraph_pad_totals
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -8,7 +8,7 @@
-from typing import Annotated, Any, Literal
+from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal
@@ -25,10 +25,19 @@
+    SupportsEncoderCudaGraph,
+if TYPE_CHECKING:
+    from vllm.v1.worker.encoder_cudagraph_defs import (
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +247/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50912 - [Kimi K3 Perf] option to shard the shared expert for non mega case, 16.98 GiB memory/GPU saved

- Link: https://github.com/vllm-project/vllm/pull/50912
- Status/date: merged / 2026-08-04
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_sequence_parallel.py`, `vllm/models/kimi_k3/nvidia/model.py`; associated commits `d31de3c421c0`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +10/-18, 98 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_sequence_parallel.py` modified +6/-9 (15 lines); hunks: -267,21 +267,19 @@ def test_kimi_mtp_restores_sequence_parallel_output(monkey...; -295,7 +293,6 @@ def test_shard_sequence_parallel_mlp_gating(; symbols: test_kimi_mtp_restores_sequence_parallel_output, test_shard_sequence_parallel_mlp_gating, touching `test_kimi_mtp_restores_sequence_parallel_output, test_shard_sequence_parallel_mlp_gating`; `vllm/models/kimi_k3/nvidia/model.py` modified +1/-9 (10 lines); hunks: -133,15 +133,14 @@ def shard_sequence_parallel_mlp(; -173,7 +172,6 @@ def __init__(; symbols: shard_sequence_parallel_mlp, __init__, touching `shard_sequence_parallel_mlp, __init__`.
- Code diff details:
  - `tests/models/kimi_k3/test_sequence_parallel.py` modified +6/-9 (15 lines); hunks: -267,21 +267,19 @@ def test_kimi_mtp_restores_sequence_parallel_output(monkey...; -295,7 +293,6 @@ def test_shard_sequence_parallel_mlp_gating(; symbols: test_kimi_mtp_restores_sequence_parallel_output, test_shard_sequence_parallel_mlp_gating
  - `vllm/models/kimi_k3/nvidia/model.py` modified +1/-9 (10 lines); hunks: -133,15 +133,14 @@ def shard_sequence_parallel_mlp(; -173,7 +172,6 @@ def __init__(; symbols: shard_sequence_parallel_mlp, __init__
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_sequence_parallel.py
@@ -267,21 +267,19 @@ def test_kimi_mtp_restores_sequence_parallel_output(monkeypatch):
-    ("enabled", "use_sequence_parallel", "eligible", "tp_size", "expected"),
+    ("enabled", "use_sequence_parallel", "tp_size", "expected"),
-        (True, True, True, 8, True),
-        (False, True, True, 8, False),  # opt-in only
-        (True, False, True, 8, False),  # replication only exists under SP
-        (True, True, False, 8, False),  # FusedMoE path owns the reduction
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -133,15 +133,14 @@ def shard_sequence_parallel_mlp(
-    eligible: bool,
-    if not (use_sequence_parallel and eligible and enabled):
+    if not (use_sequence_parallel and enabled):
@@ -173,7 +172,6 @@ def __init__(
-        can_shard_sequence_parallel: bool = False,
@@ -184,7 +182,6 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_sequence_parallel.py` modified +6/-9
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +1/-9
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_sequence_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50404 - [Model] Fix Kimi-K3 MLA with disabled context parallelism

- Link: https://github.com/vllm-project/vllm/pull/50404
- Status/date: merged / 2026-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/mla.py`; associated commits `c416f15710bb`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +5/-0, 12 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/mla.py` modified +5/-0 (5 lines); hunks: -301,6 +301,11 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/mla.py` modified +5/-0 (5 lines); hunks: -301,6 +301,11 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/mla.py
@@ -301,6 +301,11 @@ def __init__(
+        if getattr(self.impl, "dcp_world_size", -1) < 1:
+            # FlashAttention requires the cp_world_size is positive and the cp_rank
+            # is non negative; manually set here if not set by caller (-1 is unset)
+            self.impl.dcp_world_size = 1
+            self.impl.dcp_rank = 0
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/mla.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50649 - [ROCm][Bugfix] Kimi-K3 Fix KDA NaN on mixed batches and racy autotune config

- Link: https://github.com/vllm-project/vllm/pull/50649
- Status/date: merged / 2026-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/amd/kda.py`, `vllm/models/kimi_k3/amd/linear.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py`; associated commits `f5cd862dbdaf`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +556/-7, 599 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/amd/kda.py` added +530/-0 (530 lines); hunks: -0,0 +1,530; symbols: KimiK3DeltaAttention, get_state_dtype, get_state_shape, __init__, touching `KimiK3DeltaAttention, get_state_dtype, get_state_shape`; `vllm/models/kimi_k3/amd/linear.py` modified +18/-6 (24 lines); hunks: -28,7 +28,7; -61,6 +61,7; symbols: __init__, touching `__init__`; `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` modified +8/-1 (9 lines); hunks: -28,6 +28,13; -39,7 +46,7.
- Code diff details:
  - `vllm/models/kimi_k3/amd/kda.py` added +530/-0 (530 lines); hunks: -0,0 +1,530; symbols: KimiK3DeltaAttention, get_state_dtype, get_state_shape, __init__
  - `vllm/models/kimi_k3/amd/linear.py` modified +18/-6 (24 lines); hunks: -28,7 +28,7; -61,6 +61,7; symbols: __init__
  - `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` modified +8/-1 (9 lines); hunks: -28,6 +28,13; -39,7 +46,7
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/amd/kda.py
@@ -0,0 +1,530 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+from einops import rearrange
+from torch import nn
+from vllm.compilation.breakable_cudagraph import eager_break_during_capture
diff -- vllm/models/kimi_k3/amd/linear.py
@@ -28,7 +28,7 @@
-    KimiGatedDeltaNetAttention,
+    KimiGatedDeltaNetAttention as KimiLinearGatedDeltaNetAttention,
@@ -61,6 +61,7 @@
+from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
@@ -472,11 +473,22 @@ def __init__(
-            self.self_attn = KimiGatedDeltaNetAttention(
diff -- vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py
@@ -28,6 +28,13 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/amd/kda.py` added +530/-0; `vllm/models/kimi_k3/amd/linear.py` modified +18/-6; `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` modified +8/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/amd/kda.py`, `vllm/models/kimi_k3/amd/linear.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51131 - [BugFix][K3] Skip moe_intermediate padding when EP is enabled

- Link: https://github.com/vllm-project/vllm/pull/51131
- Status/date: merged / 2026-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/model.py`; associated commits `beca88e59ea7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/model.py` modified +1/-1 (2 lines); hunks: -494,7 +494,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/model.py` modified +1/-1 (2 lines); hunks: -494,7 +494,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -494,7 +494,7 @@ def __init__(
-        if self.tp_size > 1:
+        if self.tp_size > 1 and not vllm_config.parallel_config.enable_expert_parallel:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51070 - [K3 Perf] Combine multiple all gather together for SP, 1.5~3x kernel level performance improvement

- Link: https://github.com/vllm-project/vllm/pull/51070
- Status/date: merged / 2026-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/model.py`; associated commits `877975897d68`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +21/-15, 64 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/model.py` modified +21/-15 (36 lines); hunks: -1118,13 +1118,6 @@ def forward(; -1135,6 +1128,14 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/model.py` modified +21/-15 (36 lines); hunks: -1118,13 +1118,6 @@ def forward(; -1135,6 +1128,14 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -1118,13 +1118,6 @@ def forward(
-        aux_hidden_states: list[torch.Tensor] = []
-        if self.start_layer in self.aux_hidden_state_layers:
-            if self.use_attn_res or residual is None:
-                aux_hidden_states.append(hidden_states)
-            else:
-                aux_hidden_states.append(hidden_states + residual)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +21/-15
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51146 - K3: remove the add operation for megamoe path

- Link: https://github.com/vllm-project/vllm/pull/51146
- Status/date: merged / 2026-08-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/model.py`; associated commits `f85c1d2f84f5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +27/-8, 58 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/model.py` modified +27/-8 (35 lines); hunks: -242,9 +242,23 @@ def __init__(; -523,8 +537,8 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/model.py` modified +27/-8 (35 lines); hunks: -242,9 +242,23 @@ def __init__(; -523,8 +537,8 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -242,9 +242,23 @@ def __init__(
-    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
+    def forward(
+        self,
+        hidden_states: torch.Tensor,
+        residual: torch.Tensor | None = None,
+    ) -> torch.Tensor:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +27/-8
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51249 - [Bugfix][Model] Add missing fused_qkv_a_proj to Kimi-Linear packed_modules_mapping

- Link: https://github.com/vllm-project/vllm/pull/51249
- Status/date: merged / 2026-08-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/model.py`; associated commits `5fba75aefeed`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/model.py` modified +1/-0 (1 lines); hunks: -1014,6 +1014,7 @@ class KimiLinearModel(nn.Module, EagleModelMixin, Supports...; symbols: KimiLinearModel, __init__, touching `KimiLinearModel, __init__`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/model.py` modified +1/-0 (1 lines); hunks: -1014,6 +1014,7 @@ class KimiLinearModel(nn.Module, EagleModelMixin, Supports...; symbols: KimiLinearModel, __init__
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -1014,6 +1014,7 @@ class KimiLinearModel(nn.Module, EagleModelMixin, SupportsQuant):
+        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51253 - [ROCm][Perf] Kimi-K3 Shard Latent MoE up-projection for ROCm path

- Link: https://github.com/vllm-project/vllm/pull/51253
- Status/date: merged / 2026-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/__init__.py`, `tests/models/kimi_k3/test_amd_latent_moe_runner.py`, `vllm/models/kimi_k3/amd/latent_moe_runner.py`, `vllm/models/kimi_k3/amd/linear.py`; associated commits `43d691ec6b1d`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +453/-0, 470 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_amd_latent_moe_runner.py` added +297/-0 (297 lines); hunks: -0,0 +1,297; symbols: _build_transform, _tail_runner, method, _all_reduced, touching `_build_transform, _tail_runner, method`; `vllm/models/kimi_k3/amd/latent_moe_runner.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: ROCmLatentMoERunner, __init__, _shard_up_proj_tail, forward, touching `ROCmLatentMoERunner, __init__, _shard_up_proj_tail`; `tests/models/kimi_k3/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2; `vllm/models/kimi_k3/amd/linear.py` modified +2/-0 (2 lines); hunks: -62,6 +62,7; -289,6 +290,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tests/models/kimi_k3/test_amd_latent_moe_runner.py` added +297/-0 (297 lines); hunks: -0,0 +1,297; symbols: _build_transform, _tail_runner, method, _all_reduced
  - `vllm/models/kimi_k3/amd/latent_moe_runner.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: ROCmLatentMoERunner, __init__, _shard_up_proj_tail, forward
  - `tests/models/kimi_k3/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
  - `vllm/models/kimi_k3/amd/linear.py` modified +2/-0 (2 lines); hunks: -62,6 +62,7; -289,6 +290,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_amd_latent_moe_runner.py
@@ -0,0 +1,297 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""The ROCm latent-MoE tail must equal the replicated up-projection.
+A wrong shard offset or a dropped accumulation still runs and still produces
+plausible text, so these pin the arithmetic rather than the behaviour.
+"""
diff -- vllm/models/kimi_k3/amd/latent_moe_runner.py
@@ -0,0 +1,152 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+from vllm.distributed import (
+    get_tensor_model_parallel_rank,
+    tensor_model_parallel_all_reduce,
diff -- tests/models/kimi_k3/__init__.py
@@ -0,0 +1,2 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_amd_latent_moe_runner.py` added +297/-0; `tests/models/kimi_k3/__init__.py` added +2/-0
  - runtime: `vllm/models/kimi_k3/amd/latent_moe_runner.py` added +152/-0; `vllm/models/kimi_k3/amd/linear.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/__init__.py`, `tests/models/kimi_k3/test_amd_latent_moe_runner.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50585 - [K3 Perf] Optimize k3 dspark fused kv, 4.5~4.6x kernel performance improvement

- Link: https://github.com/vllm-project/vllm/pull/50585
- Status/date: merged / 2026-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/dspark_mla.py`; associated commits `56a4b63d44c7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +104/-88, 302 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +56/-87 (143 lines); hunks: -6,12 +6,14; -24,6 +26,29; symbols: _duplicate_context_kv_weights, K3DSparkDecoderLayer, __init__, precompute_and_store_context_kv, touching `_duplicate_context_kv_weights, K3DSparkDecoderLayer, __init__`; `tests/models/test_dspark_mla.py` modified +48/-1 (49 lines); hunks: -107,6 +107,7 @@ def fail_collective(*args, **kwargs):; -116,8 +117,13 @@ def make_markov_head(*args, **kwargs):; symbols: fail_collective, test_k3_dspark_uses_replicated_markov_head, DummyModule, __init__, touching `fail_collective, test_k3_dspark_uses_replicated_markov_head, DummyModule`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +56/-87 (143 lines); hunks: -6,12 +6,14; -24,6 +26,29; symbols: _duplicate_context_kv_weights, K3DSparkDecoderLayer, __init__, precompute_and_store_context_kv
  - `tests/models/test_dspark_mla.py` modified +48/-1 (49 lines); hunks: -107,6 +107,7 @@ def fail_collective(*args, **kwargs):; -116,8 +117,13 @@ def make_markov_head(*args, **kwargs):; symbols: fail_collective, test_k3_dspark_uses_replicated_markov_head, DummyModule, __init__
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/dspark_mla.py
@@ -6,12 +6,14 @@
-import torch.nn.functional as F
-from vllm.model_executor.layers.linear import ReplicatedLinear
+from vllm.model_executor.layers.linear import (
+    MergedColumnParallelLinear,
+    ReplicatedLinear,
+)
diff -- tests/models/test_dspark_mla.py
@@ -107,6 +107,7 @@ def fail_collective(*args, **kwargs):
+    context_kv_proj_calls = []
@@ -116,8 +117,13 @@ def make_markov_head(*args, **kwargs):
+    def make_context_kv_proj(*args, **kwargs):
+        context_kv_proj_calls.append((args, kwargs))
+        return DummyModule()
+    monkeypatch.setattr(dspark_mla, "MergedColumnParallelLinear", make_context_kv_proj)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +56/-87
  - tests: `tests/models/test_dspark_mla.py` modified +48/-1
- Risk and verification: The diff ships test coverage in `tests/models/test_dspark_mla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51196 - [Kimi][MM] disable kimi_vit's dynamic torch.compile for TPU

- Link: https://github.com/vllm-project/vllm/pull/51196
- Status/date: merged / 2026-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `7f58e8294a19`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/kimi_k25_vit.py` modified +1/-1 (2 lines); hunks: -61,7 +61,7 @@ def wrapper(org, interpolation_mode, shape):; symbols: wrapper, get_rope_shape, touching `wrapper, get_rope_shape`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +1/-1 (2 lines); hunks: -61,7 +61,7 @@ def wrapper(org, interpolation_mode, shape):; symbols: wrapper, get_rope_shape
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -61,7 +61,7 @@ def wrapper(org, interpolation_mode, shape):
-@torch.compile(dynamic=True)
+@torch.compile(dynamic=True, disable=current_platform.simple_compile_backend == "tpu")
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51529 - [K3] Allow tpu to import kimi_k3.common

- Link: https://github.com/vllm-project/vllm/pull/51529
- Status/date: merged / 2026-08-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/__init__.py`; associated commits `04d13b5d6537`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +9/-2, 23 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/__init__.py` modified +9/-2 (11 lines); hunks: -13,13 +13,20.
- Code diff details:
  - `vllm/models/kimi_k3/__init__.py` modified +9/-2 (11 lines); hunks: -13,13 +13,20
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/__init__.py
@@ -13,13 +13,20 @@
-if TYPE_CHECKING or not current_platform.is_rocm():
+# TPU plugins import the shared ``common`` modules through this package, but
+# register their own model classes. Do not eagerly import a GPU implementation.
+if TYPE_CHECKING:
-else:
+elif current_platform.device_type == "tpu":
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/__init__.py` modified +9/-2
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51682 - [Bugfix][Kimi-K3] Give the AMD packed KDA decode kernel the state-index stride

- Link: https://github.com/vllm-project/vllm/pull/51682
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_kda.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py`; associated commits `0e2d78028c47`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +18/-4, 83 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_kda.py` modified +13/-1 (14 lines); hunks: -15,6 +15,9; -31,6 +34,13; symbols: test_gather_initial_states_correctness, test_chunk_kda_fused_gate_cumsum_matches_unfused, test_packed_kda_decode_correctness, touching `test_gather_initial_states_correctness, test_chunk_kda_fused_gate_cumsum_matches_unfused, test_packed_kda_decode_correctness`; `vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py` modified +5/-3 (8 lines); hunks: -459,6 +459,7 @@ def fused_recurrent_kda_packed_decode_kernel(; -476,7 +477,7 @@ def fused_recurrent_kda_packed_decode_kernel(; symbols: fused_recurrent_kda_packed_decode_kernel, fused_recurrent_kda_packed_decode, touching `fused_recurrent_kda_packed_decode_kernel, fused_recurrent_kda_packed_decode`.
- Code diff details:
  - `tests/models/kimi_k3/test_kda.py` modified +13/-1 (14 lines); hunks: -15,6 +15,9; -31,6 +34,13; symbols: test_gather_initial_states_correctness, test_chunk_kda_fused_gate_cumsum_matches_unfused, test_packed_kda_decode_correctness
  - `vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py` modified +5/-3 (8 lines); hunks: -459,6 +459,7 @@ def fused_recurrent_kda_packed_decode_kernel(; -476,7 +477,7 @@ def fused_recurrent_kda_packed_decode_kernel(; symbols: fused_recurrent_kda_packed_decode_kernel, fused_recurrent_kda_packed_decode
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_kda.py
@@ -15,6 +15,9 @@
+from vllm.models.kimi_k3.amd.ops.third_party.kda import (
+    fused_recurrent_kda_packed_decode as fused_recurrent_kda_packed_decode_amd,
+)
@@ -31,6 +34,13 @@
+# The AMD and NVIDIA copies of the KDA kernels are vendored separately and are
+# free to diverge, so the shared-semantics tests below run against both.
diff -- vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py
@@ -459,6 +459,7 @@ def fused_recurrent_kda_packed_decode_kernel(
+    stride_state_indices,
@@ -476,7 +477,7 @@ def fused_recurrent_kda_packed_decode_kernel(
-    state_idx = tl.load(state_indices + i_n).to(tl.int64)
+    state_idx = tl.load(state_indices + i_n * stride_state_indices).to(tl.int64)
@@ -560,8 +561,8 @@ def fused_recurrent_kda_packed_decode(
-    if state_indices.ndim != 1 or state_indices.stride(0) != 1:
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_kda.py` modified +13/-1
  - runtime: `vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py` modified +5/-3
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50484 - [Kimi-K3] DCP support

- Link: https://github.com/vllm-project/vllm/pull/50484
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/distributed/test_kimi_linear_context_parallel.py`, `tests/kernels/attention/test_kimi_k3_mla_key_concat_kv_cache.py`, `vllm/models/kimi_k3/nvidia/mla.py`; associated commits `63ac04a61e62`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 19 files, +3529/-82, 4066 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/mla.py` modified +69/-11 (80 lines); hunks: -24,8 +24,8; -35,7 +35,11; symbols: __init__, _attention, _decode_concat_cache, _forward_prefill_fused, touching `__init__, _attention, _decode_concat_cache`; `tests/kernels/attention/test_kimi_k3_mla_key_concat_kv_cache.py` added +246/-0 (246 lines); hunks: -0,0 +1,246; symbols: _inputs, make, _slot_mapping, _owned, touching `_inputs, make, _slot_mapping`; `tests/distributed/test_kimi_linear_context_parallel.py` added +128/-0 (128 lines); hunks: -0,0 +1,128; symbols: _make_tiny_overrides, _run_tiny_model, test_kimi_linear_dcp_tiny, touching `_make_tiny_overrides, _run_tiny_model, test_kimi_linear_dcp_tiny`; `vllm/v1/attention/backends/mla/flashinfer_mla.py` modified +4/-2 (6 lines); hunks: -309,9 +309,11 @@ def forward_mqa(; -326,7 +328,7 @@ def forward_mqa(; symbols: forward_mqa, touching `forward_mqa`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/mla.py` modified +69/-11 (80 lines); hunks: -24,8 +24,8; -35,7 +35,11; symbols: __init__, _attention, _decode_concat_cache, _forward_prefill_fused
  - `tests/kernels/attention/test_kimi_k3_mla_key_concat_kv_cache.py` added +246/-0 (246 lines); hunks: -0,0 +1,246; symbols: _inputs, make, _slot_mapping, _owned
  - `tests/distributed/test_kimi_linear_context_parallel.py` added +128/-0 (128 lines); hunks: -0,0 +1,128; symbols: _make_tiny_overrides, _run_tiny_model, test_kimi_linear_dcp_tiny
  - `vllm/v1/attention/backends/mla/flashinfer_mla.py` modified +4/-2 (6 lines); hunks: -309,9 +309,11 @@ def forward_mqa(; -326,7 +328,7 @@ def forward_mqa(; symbols: forward_mqa
  - `vllm/v1/attention/backends/mla/tokenspeed_mla.py` modified +2/-1 (3 lines); hunks: -289,8 +289,9 @@ def forward_mqa(; symbols: forward_mqa
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/mla.py
@@ -24,8 +24,8 @@
-Out of scope (extension points, not wired here): context parallelism (DCP/PCP),
-sparse/indexer MLA, and the ROCm/aiter fp8/fp4 BMM fast paths.
+Out of scope (extension points, not wired here): prefill context parallelism
+(PCP), sparse/indexer MLA, and the ROCm/aiter fp8/fp4 BMM fast paths.
@@ -35,7 +35,11 @@
-from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
diff -- tests/kernels/attention/test_kimi_k3_mla_key_concat_kv_cache.py
@@ -0,0 +1,246 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+import torch
+import vllm._custom_ops as ops
+from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
diff -- tests/distributed/test_kimi_linear_context_parallel.py
@@ -0,0 +1,128 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/mla.py` modified +69/-11; `vllm/v1/attention/backends/mla/flashinfer_mla.py` modified +4/-2; `vllm/v1/attention/backends/mla/tokenspeed_mla.py` modified +2/-1
  - tests: `tests/kernels/attention/test_kimi_k3_mla_key_concat_kv_cache.py` added +246/-0; `tests/distributed/test_kimi_linear_context_parallel.py` added +128/-0
- Risk and verification: The diff ships test coverage in `tests/distributed/test_dcp_a2a.py`, `tests/distributed/test_dcp_direct_a2a_lse_reduce.py`, `tests/distributed/test_kimi_linear_context_parallel.py`, `tests/kernels/attention/test_kimi_k3_mla_key_concat_kv_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50654 - [ROCm][Perf] Kimi-K3 Fused kernel for KDA decode

- Link: https://github.com/vllm-project/vllm/pull/50654
- Status/date: merged / 2026-08-12
- Trace source: `git log --name-only -- <model-files>` found it through `benchmarks/kernels/benchmark_kimi_k3_kda_decode.py`, `tests/models/kimi_k3/test_amd_kda_decode.py`, `vllm/models/kimi_k3/amd/kda.py`, `vllm/models/kimi_k3/amd/ops/kda_decode.py`; associated commits `7f9173dfa24d`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +1592/-9, 1666 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_amd_kda_decode.py` added +291/-0 (291 lines); hunks: -0,0 +1,291; symbols: _on_gfx950, _requires_kernel, KdaDecodeInputs, __init__, touching `_on_gfx950, _requires_kernel, KdaDecodeInputs`; `vllm/models/kimi_k3/amd/ops/kda_decode.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: is_fused_kda_decode_supported, make_decode_conv1d_weight_loader, weight_loader, make_decode_norm_weight_loader, touching `is_fused_kda_decode_supported, make_decode_conv1d_weight_loader, weight_loader`; `vllm/models/kimi_k3/amd/kda.py` modified +89/-9 (98 lines); hunks: -5,10 +5,12; -38,6 +40,11; symbols: KimiK3DeltaAttention, get_state_dtype, __init__, _forward, touching `KimiK3DeltaAttention, get_state_dtype, __init__`; `benchmarks/kernels/benchmark_kimi_k3_kda_decode.py` added +291/-0 (291 lines); hunks: -0,0 +1,291; symbols: _bench, _bench_graph_layers, _bench_graph, Inputs, touching `_bench, _bench_graph_layers, _bench_graph`.
- Code diff details:
  - `tests/models/kimi_k3/test_amd_kda_decode.py` added +291/-0 (291 lines); hunks: -0,0 +1,291; symbols: _on_gfx950, _requires_kernel, KdaDecodeInputs, __init__
  - `vllm/models/kimi_k3/amd/ops/kda_decode.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: is_fused_kda_decode_supported, make_decode_conv1d_weight_loader, weight_loader, make_decode_norm_weight_loader
  - `vllm/models/kimi_k3/amd/kda.py` modified +89/-9 (98 lines); hunks: -5,10 +5,12; -38,6 +40,11; symbols: KimiK3DeltaAttention, get_state_dtype, __init__, _forward
  - `benchmarks/kernels/benchmark_kimi_k3_kda_decode.py` added +291/-0 (291 lines); hunks: -0,0 +1,291; symbols: _bench, _bench_graph_layers, _bench_graph, Inputs
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_amd_kda_decode.py
@@ -0,0 +1,291 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""The ROCm fused KDA decode kernel must match the Triton chain it replaces.
+The fused kernel folds the packed causal conv1d update, the gated delta-rule
+recurrence and the gated output RMSNorm into one launch, and updates both the
+conv state and the recurrent state in place. Every one of those outputs is
diff -- vllm/models/kimi_k3/amd/ops/kda_decode.py
@@ -0,0 +1,99 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""ROCm entry points for the fused Kimi-K3 KDA decode kernel.
+The kernel in ``csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel_rocm.cu``
+replaces, for a pure non-speculative decode batch, the three Triton launches
+and two copies the AMD KDA layer otherwise runs per layer: the packed causal
diff -- vllm/models/kimi_k3/amd/kda.py
@@ -5,10 +5,12 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_amd_kda_decode.py` added +291/-0
  - runtime: `vllm/models/kimi_k3/amd/ops/kda_decode.py` added +99/-0; `vllm/models/kimi_k3/amd/kda.py` modified +89/-9
  - other: `benchmarks/kernels/benchmark_kimi_k3_kda_decode.py` added +291/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_amd_kda_decode.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51860 - [ROCm][K3] Dequantize the fp8 decode query for MLA backends without quant-query support - TRITON_MLA

- Link: https://github.com/vllm-project/vllm/pull/51860
- Status/date: merged / 2026-08-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/mla.py`; associated commits `b745d08de14f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +7/-5, 29 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/mla.py` modified +7/-5 (12 lines); hunks: -664,14 +664,10 @@ def _decode_concat_cache(; -683,6 +679,12 @@ def _decode_concat_cache(; symbols: _decode_concat_cache, touching `_decode_concat_cache`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/mla.py` modified +7/-5 (12 lines); hunks: -664,14 +664,10 @@ def _decode_concat_cache(; -683,6 +679,12 @@ def _decode_concat_cache(; symbols: _decode_concat_cache
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/mla.py
@@ -664,14 +664,10 @@ def _decode_concat_cache(
-            assert self.impl.supports_quant_query_input, (  # type: ignore[attr-defined]
-                "Kimi-K3 fp8 KV cache decode requires a backend that accepts an "
-                "fp8 (quantized) query input."
-            )
-            return fused_mla_decode_q_concat_kv_cache_insert(
+            mqa_q = fused_mla_decode_q_concat_kv_cache_insert(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/mla.py` modified +7/-5
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51311 - [K3 Perf] Flash kda out kernel for prefill, 1.1~1.4x kernel performance improvement

- Link: https://github.com/vllm-project/vllm/pull/51311
- Status/date: merged / 2026-08-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/kda.py`; associated commits `fe889ac92554`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +36/-15, 96 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/kda.py` modified +36/-15 (51 lines); hunks: -46,6 +46,7; -187,20 +188,12 @@ def _flashkda_prefill(; symbols: _flashkda_prefill, __init__, _prefill_conv, touching `_flashkda_prefill, __init__, _prefill_conv`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +36/-15 (51 lines); hunks: -46,6 +46,7; -187,20 +188,12 @@ def _flashkda_prefill(; symbols: _flashkda_prefill, __init__, _prefill_conv
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -46,6 +46,7 @@
+from vllm.v1.worker.workspace import current_workspace_manager
@@ -187,20 +188,12 @@ def _flashkda_prefill(
+    out: torch.Tensor,
+    final_state: torch.Tensor,
+    workspace: torch.Tensor,
-    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/kda.py` modified +36/-15
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/kda.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51772 - [Attention][MLA] Fuse Kimi-K3 chunked-context K/V packing

- Link: https://github.com/vllm-project/vllm/pull/51772
- Status/date: merged / 2026-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py`, `tests/models/kimi_k3/test_mla_prefill_context.py`, `vllm/models/kimi_k3/nvidia/mla.py`, `vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py`; associated commits `903d2efe7eb6`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 15 files, +996/-35, 1326 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_mla_prefill_context.py` added +319/-0 (319 lines); hunks: -0,0 +1,319; symbols: _RecordingPrefillBackend, __init__, get_name, supports_out, touching `_RecordingPrefillBackend, __init__, get_name`; `vllm/models/kimi_k3/nvidia/mla.py` modified +192/-6 (198 lines); hunks: -8,8 +8,9; -34,6 +35,7; symbols: _decode_concat_cache, _compute_prefill_context, run_chunk, _gather_context_latent, touching `_decode_concat_cache, _compute_prefill_context, run_chunk`; `vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py` modified +56/-0 (56 lines); hunks: -12,6 +12,9; -156,6 +159,59 @@ def fused_mla_qkv_quant_kv_cache_fp8_insert(; symbols: fused_mla_qkv_quant_kv_cache_fp8_insert, _empty_full_key, fused_mla_kv_concat, fused_mla_kv_concat_quant_fp8, touching `fused_mla_qkv_quant_kv_cache_fp8_insert, _empty_full_key, fused_mla_kv_concat`; `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py` modified +68/-2 (70 lines); hunks: -9,6 +9,8; -25,8 +27,8; symbols: _randn, _rope_cache, _assert_fp8_close, _strided_context_inputs, touching `_randn, _rope_cache, _assert_fp8_close`.
- Code diff details:
  - `tests/models/kimi_k3/test_mla_prefill_context.py` added +319/-0 (319 lines); hunks: -0,0 +1,319; symbols: _RecordingPrefillBackend, __init__, get_name, supports_out
  - `vllm/models/kimi_k3/nvidia/mla.py` modified +192/-6 (198 lines); hunks: -8,8 +8,9; -34,6 +35,7; symbols: _decode_concat_cache, _compute_prefill_context, run_chunk, _gather_context_latent
  - `vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py` modified +56/-0 (56 lines); hunks: -12,6 +12,9; -156,6 +159,59 @@ def fused_mla_qkv_quant_kv_cache_fp8_insert(; symbols: fused_mla_qkv_quant_kv_cache_fp8_insert, _empty_full_key, fused_mla_kv_concat, fused_mla_kv_concat_quant_fp8
  - `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py` modified +68/-2 (70 lines); hunks: -9,6 +9,8; -25,8 +27,8; symbols: _randn, _rope_cache, _assert_fp8_close, _strided_context_inputs
  - `vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py` modified +4/-1 (5 lines); hunks: -166,11 +166,13 @@ def run_prefill_context_chunk(; -187,6 +189,7 @@ def run_prefill_context_chunk(; symbols: run_prefill_context_chunk
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_mla_prefill_context.py
@@ -0,0 +1,319 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""K3's fused chunked-context prefill must match the generic MLA impl.
+The layer owns its own context loop so it can fuse the per-chunk K/V pack and
+skip re-quantizing an already-quantized query. That is only safe if it feeds the
+prefill backend exactly what ``MLACommonBaseImpl._compute_prefill_context``
diff -- vllm/models/kimi_k3/nvidia/mla.py
@@ -8,8 +8,9 @@
-                    (+ chunked-context merge); dispatched by cache dtype
-                    (bf16 / plain fp8 / fp8_ds_mla)
+                    (+ chunked-context merge, whose per-chunk gather -> kv_b_proj
+                    -> fused K/V pack loop this layer owns); dispatched by cache
+                    dtype (bf16 / plain fp8 / fp8_ds_mla)
@@ -34,6 +35,7 @@
diff -- vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py
@@ -12,6 +12,9 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_mla_prefill_context.py` added +319/-0; `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py` modified +68/-2
  - runtime: `vllm/models/kimi_k3/nvidia/mla.py` modified +192/-6; `vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py` modified +56/-0; `vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py` modified +4/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py`, `tests/models/kimi_k3/test_mla_prefill_context.py`, `tests/v1/attention/test_mla_prefill_registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51862 - [ROCm][Perf] Kimi-K3 Remove prefill pipeline stall in chunk KDA

- Link: https://github.com/vllm-project/vllm/pull/51862
- Status/date: merged / 2026-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `vllm/models/kimi_k3/amd/kda.py`, `vllm/models/kimi_k3/amd/kda_metadata.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py`; associated commits `f96261637c05`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +188/-43, 338 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/amd/kda_metadata.py` added +109/-0 (109 lines); hunks: -0,0 +1,109; symbols: _chunk_metadata_kernel, prepare_chunk_metadata_device, KimiK3ROCmKDAMetadataBuilder, _build_chunk_metadata, touching `_chunk_metadata_kernel, prepare_chunk_metadata_device, KimiK3ROCmKDAMetadataBuilder`; `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` modified +14/-5 (19 lines); hunks: -601,6 +601,7 @@ def _chunk_kda_fwd_with_cumulative_g(; -638,6 +639,7 @@ def _chunk_kda_fwd_with_cumulative_g(; symbols: _chunk_kda_fwd_with_cumulative_g, chunk_kda_with_fused_gate_fwd, chunk_kda_with_fused_gate, touching `_chunk_kda_fwd_with_cumulative_g, chunk_kda_with_fused_gate_fwd, chunk_kda_with_fused_gate`; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` modified +11/-2 (13 lines); hunks: -2,6 +2,7; -393,8 +394,16 @@ def _forward(; symbols: _forward, touching `_forward`; `vllm/models/kimi_k3/amd/kda.py` modified +7/-0 (7 lines); hunks: -40,6 +40,7; -52,12 +53,16; symbols: KimiK3DeltaAttention, get_attn_backend, get_state_dtype, _prefill_conv, touching `KimiK3DeltaAttention, get_attn_backend, get_state_dtype`.
- Code diff details:
  - `vllm/models/kimi_k3/amd/kda_metadata.py` added +109/-0 (109 lines); hunks: -0,0 +1,109; symbols: _chunk_metadata_kernel, prepare_chunk_metadata_device, KimiK3ROCmKDAMetadataBuilder, _build_chunk_metadata
  - `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` modified +14/-5 (19 lines); hunks: -601,6 +601,7 @@ def _chunk_kda_fwd_with_cumulative_g(; -638,6 +639,7 @@ def _chunk_kda_fwd_with_cumulative_g(; symbols: _chunk_kda_fwd_with_cumulative_g, chunk_kda_with_fused_gate_fwd, chunk_kda_with_fused_gate
  - `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` modified +11/-2 (13 lines); hunks: -2,6 +2,7; -393,8 +394,16 @@ def _forward(; symbols: _forward
  - `vllm/models/kimi_k3/amd/kda.py` modified +7/-0 (7 lines); hunks: -40,6 +40,7; -52,12 +53,16; symbols: KimiK3DeltaAttention, get_attn_backend, get_state_dtype, _prefill_conv
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/amd/kda_metadata.py
@@ -0,0 +1,109 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""ROCm Kimi-K3 specialization of GDN attention metadata.
+The request classification and cudagraph staging intentionally mirror
+``GDNAttentionMetadataBuilder``. Only the FLA chunk metadata is built
+differently on device rather than on the host.
diff -- vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py
@@ -601,6 +601,7 @@ def _chunk_kda_fwd_with_cumulative_g(
+    chunk_offsets: torch.Tensor | None = None,
@@ -638,6 +639,7 @@ def _chunk_kda_fwd_with_cumulative_g(
+        chunk_offsets=chunk_offsets,
@@ -711,13 +713,15 @@ def chunk_kda_with_fused_gate_fwd(
+    chunk_indices: torch.Tensor | None = None,
+    chunk_offsets: torch.Tensor | None = None,
diff -- vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py
@@ -2,6 +2,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/amd/kda_metadata.py` added +109/-0; `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` modified +14/-5; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` modified +11/-2; `vllm/models/kimi_k3/amd/kda.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `vllm/models/kimi_k3/amd/kda.py`, `vllm/models/kimi_k3/amd/kda_metadata.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #52079 - [Kimi-K3] Add GEMM-RS for sequence parallelism

- Link: https://github.com/vllm-project/vllm/pull/52079
- Status/date: merged / 2026-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_sequence_parallel.py`, `vllm/models/kimi_k3/nvidia/kda.py`, `vllm/models/kimi_k3/nvidia/mla.py`, `vllm/models/kimi_k3/nvidia/model.py`; associated commits `6014f9e67291`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +1591/-13, 1827 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/model.py` modified +82/-2 (84 lines); hunks: -133,21 +133,60 @@ def shard_sequence_parallel_mlp(; -172,6 +211,8 @@ def __init__(; symbols: shard_sequence_parallel_mlp, maybe_init_gemm_rs, KimiMLP, __init__, touching `shard_sequence_parallel_mlp, maybe_init_gemm_rs, KimiMLP`; `vllm/models/kimi_k3/nvidia/mla.py` modified +18/-4 (22 lines); hunks: -134,6 +134,7 @@ def __init__(; -270,6 +271,16 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/models/kimi_k3/nvidia/kda.py` modified +17/-1 (18 lines); hunks: -305,6 +305,7 @@ def __init__(; -470,7 +471,16 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `tests/models/kimi_k3/test_sequence_parallel.py` modified +9/-6 (15 lines); hunks: -267,19 +267,21 @@ def test_kimi_mtp_restores_sequence_parallel_output(monkey...; -293,6 +295,7 @@ def test_shard_sequence_parallel_mlp_gating(; symbols: test_kimi_mtp_restores_sequence_parallel_output, test_shard_sequence_parallel_mlp_gating, touching `test_kimi_mtp_restores_sequence_parallel_output, test_shard_sequence_parallel_mlp_gating`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/model.py` modified +82/-2 (84 lines); hunks: -133,21 +133,60 @@ def shard_sequence_parallel_mlp(; -172,6 +211,8 @@ def __init__(; symbols: shard_sequence_parallel_mlp, maybe_init_gemm_rs, KimiMLP, __init__
  - `vllm/models/kimi_k3/nvidia/mla.py` modified +18/-4 (22 lines); hunks: -134,6 +134,7 @@ def __init__(; -270,6 +271,16 @@ def __init__(; symbols: __init__, forward
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +17/-1 (18 lines); hunks: -305,6 +305,7 @@ def __init__(; -470,7 +471,16 @@ def __init__(; symbols: __init__, forward
  - `tests/models/kimi_k3/test_sequence_parallel.py` modified +9/-6 (15 lines); hunks: -267,19 +267,21 @@ def test_kimi_mtp_restores_sequence_parallel_output(monkey...; -293,6 +295,7 @@ def test_shard_sequence_parallel_mlp_gating(; symbols: test_kimi_mtp_restores_sequence_parallel_output, test_shard_sequence_parallel_mlp_gating
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -133,21 +133,60 @@ def shard_sequence_parallel_mlp(
+    eligible: bool,
-    if not (use_sequence_parallel and enabled):
+    if not (use_sequence_parallel and eligible and enabled):
+def maybe_init_gemm_rs(vllm_config: VllmConfig, use_sequence_parallel: bool) -> bool:
+    if not envs.VLLM_KIMI_K3_GEMM_RS:
+        return False
diff -- vllm/models/kimi_k3/nvidia/mla.py
@@ -134,6 +134,7 @@ def __init__(
+        run_gemm_rs: bool = False,
@@ -270,6 +271,16 @@ def __init__(
+        self.run_gemm_rs = run_gemm_rs
+        if self.run_gemm_rs:
+            from vllm.models.kimi_k3.nvidia.ops.cute_dsl.gemm_rs import get_gemm_rs
+            self.run_gemm_rs = get_gemm_rs().can_run(self.o_proj)
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -305,6 +305,7 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +82/-2; `vllm/models/kimi_k3/nvidia/mla.py` modified +18/-4; `vllm/models/kimi_k3/nvidia/kda.py` modified +17/-1
  - tests: `tests/models/kimi_k3/test_sequence_parallel.py` modified +9/-6
- Risk and verification: The diff ships test coverage in `tests/kernels/test_kimi_k3_gemm_rs.py`, `tests/models/kimi_k3/test_sequence_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52171 - [Bugfix] Declare SupportsEagle3 on KimiLinearForCausalLM

- Link: https://github.com/vllm-project/vllm/pull/52171
- Status/date: merged / 2026-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_eagle3.py`, `vllm/models/kimi_k3/nvidia/model.py`; associated commits `83d4c6196ad9`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +10/-1, 32 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_eagle3.py` modified +9/-0 (9 lines); hunks: -10,6 +10,7; -25,6 +26,14 @@ def test_kimi_k3_advertises_eagle3_support():; symbols: test_kimi_k3_advertises_eagle3_support, test_kimi_linear_advertises_eagle3_support, test_kimi_k3_uses_shared_eagle3_layer_configuration, touching `test_kimi_k3_advertises_eagle3_support, test_kimi_linear_advertises_eagle3_support, test_kimi_k3_uses_shared_eagle3_layer_configuration`; `vllm/models/kimi_k3/nvidia/model.py` modified +1/-1 (2 lines); hunks: -1397,7 +1397,7 @@ def finalize_mega_moe_weights(self) -> None:; symbols: finalize_mega_moe_weights, KimiLinearForCausalLM, __init__, touching `finalize_mega_moe_weights, KimiLinearForCausalLM, __init__`.
- Code diff details:
  - `tests/models/kimi_k3/test_eagle3.py` modified +9/-0 (9 lines); hunks: -10,6 +10,7; -25,6 +26,14 @@ def test_kimi_k3_advertises_eagle3_support():; symbols: test_kimi_k3_advertises_eagle3_support, test_kimi_linear_advertises_eagle3_support, test_kimi_k3_uses_shared_eagle3_layer_configuration
  - `vllm/models/kimi_k3/nvidia/model.py` modified +1/-1 (2 lines); hunks: -1397,7 +1397,7 @@ def finalize_mega_moe_weights(self) -> None:; symbols: finalize_mega_moe_weights, KimiLinearForCausalLM, __init__
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_eagle3.py
@@ -10,6 +10,7 @@
+    KimiLinearForCausalLM,
@@ -25,6 +26,14 @@ def test_kimi_k3_advertises_eagle3_support():
+def test_kimi_linear_advertises_eagle3_support():
+    # The text-only architecture serves the same inner KimiLinearModel, which
+    # already carries the EagleModelMixin tap machinery - only the interface
+    # declaration was missing, so EAGLE3-family speculative decoding (dspark)
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -1397,7 +1397,7 @@ def finalize_mega_moe_weights(self) -> None:
-    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
+    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid, SupportsEagle3
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_eagle3.py` modified +9/-0
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_eagle3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50487 - [Model][Spec Decode] Tap the pre-norm AttnRes mixture as the Kimi K3 DFlash aux state

- Link: https://github.com/vllm-project/vllm/pull/50487
- Status/date: merged / 2026-08-14
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_aux_attn_res_stream.py`, `tests/models/kimi_k3/test_eagle3.py`, `vllm/models/kimi_k3/nvidia/model.py`; associated commits `03a8d0b1ede6`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +350/-1, 391 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_aux_attn_res_stream.py` added +197/-0 (197 lines); hunks: -0,0 +1,197; symbols: _weights, _stub_model, recorder, _fake_attn_res, touching `_weights, _stub_model, recorder`; `vllm/models/kimi_k3/nvidia/model.py` modified +83/-1 (84 lines); hunks: -1198,6 +1198,85 @@ def make_empty_intermediate_tensors(; -1265,7 +1344,10 @@ def forward(; symbols: make_empty_intermediate_tensors, _set_aux_hidden_state_layers, _aux_attn_res_stream, _capture_aux_hidden_stream, touching `make_empty_intermediate_tensors, _set_aux_hidden_state_layers, _aux_attn_res_stream`; `tests/models/kimi_k3/test_eagle3.py` modified +62/-0 (62 lines); hunks: -19,6 +19,7 @@ def _make_kimi_linear_model() -> KimiLinearModel:; -137,3 +138,64 @@ def test_kimi_linear_forward_extracts_attn_res_aux_hidden_s...; symbols: _make_kimi_linear_model, test_kimi_linear_forward_extracts_attn_res_aux_hidden_states, test_attn_res_stream_capture_receives_the_layer_outputs_in_order, touching `_make_kimi_linear_model, test_kimi_linear_forward_extracts_attn_res_aux_hidden_states, test_attn_res_stream_capture_receives_the_layer_outputs_in_order`.
- Code diff details:
  - `tests/models/kimi_k3/test_aux_attn_res_stream.py` added +197/-0 (197 lines); hunks: -0,0 +1,197; symbols: _weights, _stub_model, recorder, _fake_attn_res
  - `vllm/models/kimi_k3/nvidia/model.py` modified +83/-1 (84 lines); hunks: -1198,6 +1198,85 @@ def make_empty_intermediate_tensors(; -1265,7 +1344,10 @@ def forward(; symbols: make_empty_intermediate_tensors, _set_aux_hidden_state_layers, _aux_attn_res_stream, _capture_aux_hidden_stream
  - `tests/models/kimi_k3/test_eagle3.py` modified +62/-0 (62 lines); hunks: -19,6 +19,7 @@ def _make_kimi_linear_model() -> KimiLinearModel:; -137,3 +138,64 @@ def test_kimi_linear_forward_extracts_attn_res_aux_hidden_s...; symbols: _make_kimi_linear_model, test_kimi_linear_forward_extracts_attn_res_aux_hidden_states, test_attn_res_stream_capture_receives_the_layer_outputs_in_order
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_aux_attn_res_stream.py
@@ -0,0 +1,197 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Which value the DFlash drafter is fed under AttnRes.
+`_capture_aux_hidden_stream` picks the weights it mixes against from one of
+three places depending on where the tapped layer sits, and returns the plain
+running prefix when the feature is off. The mixture itself is the kernel's
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -1198,6 +1198,85 @@ def make_empty_intermediate_tensors(
+    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
+        super()._set_aux_hidden_state_layers(layers)
+        if self.use_attn_res:
+            # Emitted once, at configuration time. Which layers are tapped and
+            # which convention is in force are the two things you need to
+            # confirm from a running process, and neither is recoverable from
diff -- tests/models/kimi_k3/test_eagle3.py
@@ -19,6 +19,7 @@ def _make_kimi_linear_model() -> KimiLinearModel:
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_aux_attn_res_stream.py` added +197/-0; `tests/models/kimi_k3/test_eagle3.py` modified +62/-0
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +83/-1
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_aux_attn_res_stream.py`, `tests/models/kimi_k3/test_eagle3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52445 - [Bugfix][Model] Kimi-K3 MegaMoE: pass situ_beta/situ_linear_beta to fp8_fp4_mega_moe

- Link: https://github.com/vllm-project/vllm/pull/52445
- Status/date: merged / 2026-08-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/model.py`; associated commits `ed0f4750f876`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +2/-2, 11 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/model.py` modified +2/-2 (4 lines); hunks: -486,8 +486,8 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/model.py` modified +2/-2 (4 lines); hunks: -486,8 +486,8 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -486,8 +486,8 @@ def forward(
-            activation_beta=self.activation_beta,
-            activation_linear_beta=self.activation_linear_beta,
+            situ_beta=self.activation_beta,
+            situ_linear_beta=self.activation_linear_beta,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51809 - [XPU] Enable Kimi K3 KDA kernel tests on XPU

- Link: https://github.com/vllm-project/vllm/pull/51809
- Status/date: merged / 2026-08-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_kda.py`; associated commits `cc7cf71fc819`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +8/-2, 26 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_kda.py` modified +7/-1 (8 lines); hunks: -30,9 +30,15; `vllm/model_executor/layers/mamba/ops/gather_initial_states.py` modified +1/-1 (2 lines); hunks: -49,7 +49,7 @@ def gather_initial_states(; symbols: gather_initial_states, touching `gather_initial_states`.
- Code diff details:
  - `tests/models/kimi_k3/test_kda.py` modified +7/-1 (8 lines); hunks: -30,9 +30,15
  - `vllm/model_executor/layers/mamba/ops/gather_initial_states.py` modified +1/-1 (2 lines); hunks: -49,7 +49,7 @@ def gather_initial_states(; symbols: gather_initial_states
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_kda.py
@@ -30,9 +30,15 @@
+from vllm.platforms import current_platform
-DEVICE = "cuda"
+DEVICE = current_platform.device_type
+pytestmark = pytest.mark.skipif(
+    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
+    reason="The KDA kernels require a CUDA-alike or XPU device.",
diff -- vllm/model_executor/layers/mamba/ops/gather_initial_states.py
@@ -49,7 +49,7 @@ def gather_initial_states(
-    assert state.is_cuda
+    assert state.is_cuda or state.is_xpu
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_kda.py` modified +7/-1
  - runtime: `vllm/model_executor/layers/mamba/ops/gather_initial_states.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51855 - [K3] support recoverssm for K3

- Link: https://github.com/vllm-project/vllm/pull/51855
- Status/date: merged / 2026-08-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_kda.py`, `tests/models/kimi_k3/test_kda_metadata.py`, `vllm/models/kimi_k3/nvidia/kda.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py`, `vllm/models/kimi_k3/nvidia/model.py` and 6 files; associated commits `70afdedc1081`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 19 files, +2235/-75, 2787 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/ops/recoverssm.py` added +1067/-0 (1067 lines); hunks: -0,0 +1,1067; symbols: _kda_gate, _kda_recurrent_step, _kda_recoverssm_verify_kernel, _prepare_commit_plan_kernel, touching `_kda_gate, _kda_recurrent_step, _kda_recoverssm_verify_kernel`; `tests/models/kimi_k3/test_kda.py` modified +342/-0 (342 lines); hunks: -6,6 +6,8; -22,6 +24,12; symbols: test_kda_recoverssm_config_state_layout, test_gather_initial_states_correctness, test_kda_spec_decode_correctness, test_kda_recoverssm_verify_and_group_commit, touching `test_kda_recoverssm_config_state_layout, test_gather_initial_states_correctness, test_kda_spec_decode_correctness`; `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +176/-12 (188 lines); hunks: -6,13 +6,18; -22,13 +27,22; symbols: _metadata_launch_pdl, stage_spec_decode_metadata, KimiK3KDAMetadata, KDARecoverSSMAlignMetadata, touching `_metadata_launch_pdl, stage_spec_decode_metadata, KimiK3KDAMetadata`; `tests/models/kimi_k3/test_kda_metadata.py` modified +145/-4 (149 lines); hunks: -2,6 +2,7; -26,6 +27,9; symbols: _assert_matches_shared_gdn, _make_builder, test_mixed_regular_and_spec_decode_excludes_request_padding, touching `_assert_matches_shared_gdn, _make_builder, test_mixed_regular_and_spec_decode_excludes_request_padding`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/ops/recoverssm.py` added +1067/-0 (1067 lines); hunks: -0,0 +1,1067; symbols: _kda_gate, _kda_recurrent_step, _kda_recoverssm_verify_kernel, _prepare_commit_plan_kernel
  - `tests/models/kimi_k3/test_kda.py` modified +342/-0 (342 lines); hunks: -6,6 +6,8; -22,6 +24,12; symbols: test_kda_recoverssm_config_state_layout, test_gather_initial_states_correctness, test_kda_spec_decode_correctness, test_kda_recoverssm_verify_and_group_commit
  - `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +176/-12 (188 lines); hunks: -6,13 +6,18; -22,13 +27,22; symbols: _metadata_launch_pdl, stage_spec_decode_metadata, KimiK3KDAMetadata, KDARecoverSSMAlignMetadata
  - `tests/models/kimi_k3/test_kda_metadata.py` modified +145/-4 (149 lines); hunks: -2,6 +2,7; -26,6 +27,9; symbols: _assert_matches_shared_gdn, _make_builder, test_mixed_regular_and_spec_decode_excludes_request_padding
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +73/-22 (95 lines); hunks: -282,23 +282,37 @@ def get_attn_backend(self) -> type[AttentionBackend]:; -308,6 +322,12 @@ def __init__(; symbols: get_attn_backend, get_state_dtype, get_state_shape, __init__
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/ops/recoverssm.py
@@ -0,0 +1,1067 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Kimi-K3 RecoverSSM speculative verify and accepted-state recovery."""
+from collections.abc import Sequence
+from dataclasses import dataclass
+from typing import Any
diff -- tests/models/kimi_k3/test_kda.py
@@ -6,6 +6,8 @@
+from types import SimpleNamespace
@@ -22,6 +24,12 @@
+from vllm.models.kimi_k3.nvidia.model import KimiLinearForCausalLM
+from vllm.models.kimi_k3.nvidia.ops import recoverssm as recoverssm_ops
+from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
+    KDARecoverSSMCommitContext,
diff -- vllm/models/kimi_k3/nvidia/kda_metadata.py
@@ -6,13 +6,18 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/ops/recoverssm.py` added +1067/-0; `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +176/-12; `vllm/models/kimi_k3/nvidia/kda.py` modified +73/-22; `vllm/models/kimi_k3/nvidia/model.py` modified +28/-5; `vllm/v1/worker/gpu/model_states/recoverssm.py` added +101/-0
  - tests: `tests/models/kimi_k3/test_kda.py` modified +342/-0; `tests/models/kimi_k3/test_kda_metadata.py` modified +145/-4
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda.py`, `tests/models/kimi_k3/test_kda_metadata.py`, `tests/models/test_registry.py`, `tests/test_config.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52188 - [Spec decode] Support Kimi-K3 DCP with DSpark

- Link: https://github.com/vllm-project/vllm/pull/52188
- Status/date: merged / 2026-08-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/mla.py`; associated commits `d1e3eee6fb8e`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 15 files, +341/-51, 710 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/mla.py` modified +0/-5 (5 lines); hunks: -341,11 +341,6 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/v1/attention/backends/mla/flashinfer_mla.py` modified +115/-7 (122 lines); hunks: -1,7 +1,8; -15,6 +16,7; symbols: _get_multi_ctas_kv_counter_buffer, FlashInferMLAMetadataBuilder, FlashInferMLADecodeMetadata, FlashInferMLAMetadata, touching `_get_multi_ctas_kv_counter_buffer, FlashInferMLAMetadataBuilder, FlashInferMLADecodeMetadata`; `vllm/v1/attention/backends/mla/tokenspeed_mla.py` modified +1/-0 (1 lines); hunks: -61,6 +61,7 @@ class TokenspeedMLAMetadataBuilder(MLACommonMetadataBuilder[ML...; symbols: TokenspeedMLAMetadataBuilder, __init__, touching `TokenspeedMLAMetadataBuilder, __init__`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/mla.py` modified +0/-5 (5 lines); hunks: -341,11 +341,6 @@ def __init__(; symbols: __init__
  - `vllm/v1/attention/backends/mla/flashinfer_mla.py` modified +115/-7 (122 lines); hunks: -1,7 +1,8; -15,6 +16,7; symbols: _get_multi_ctas_kv_counter_buffer, FlashInferMLAMetadataBuilder, FlashInferMLADecodeMetadata, FlashInferMLAMetadata
  - `vllm/v1/attention/backends/mla/tokenspeed_mla.py` modified +1/-0 (1 lines); hunks: -61,6 +61,7 @@ class TokenspeedMLAMetadataBuilder(MLACommonMetadataBuilder[ML...; symbols: TokenspeedMLAMetadataBuilder, __init__
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/mla.py
@@ -341,11 +341,6 @@ def __init__(
-        assert self.dcp_world_size <= 1 or self.rotary_emb is None, (
-            "Kimi-K3 MultiHeadLatentAttention does not support RoPE with decode "
-            "context parallelism because gathered queries require gathered "
-            "positions."
-        )
diff -- vllm/v1/attention/backends/mla/flashinfer_mla.py
@@ -1,7 +1,8 @@
-from typing import ClassVar
+from dataclasses import dataclass
+from typing import TYPE_CHECKING, ClassVar
@@ -15,6 +16,7 @@
+    MLACommonDecodeMetadata,
@@ -30,6 +32,10 @@
diff -- vllm/v1/attention/backends/mla/tokenspeed_mla.py
@@ -61,6 +61,7 @@ class TokenspeedMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
+    supports_non_causal_multi_token_dcp: ClassVar[bool] = True
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/mla.py` modified +0/-5; `vllm/v1/attention/backends/mla/flashinfer_mla.py` modified +115/-7; `vllm/v1/attention/backends/mla/tokenspeed_mla.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/transformers_utils/test_dspark_mla_config.py`, `tests/v1/attention/test_flashinfer_mla_dcp.py`, `tests/v1/attention/test_mla_backends.py`, `tests/v1/spec_decode/test_dflash_prepare_inputs.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50493 - [Kimi-K3] support DCP partial prefix cache hit

- Link: https://github.com/vllm-project/vllm/pull/50493
- Status/date: merged / 2026-08-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/distributed/test_kimi_linear_context_parallel.py`; associated commits `0db502c8d8a6`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +523/-37, 715 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/distributed/test_kimi_linear_context_parallel.py` modified +111/-0 (111 lines); hunks: -2,12 +2,15; -126,3 +129,111 @@ def test_kimi_linear_dcp_tiny(; symbols: test_kimi_linear_dcp_tiny, _make_tiny_k3_config, _run_k3_partial_prefix_reuse, test_kimi_k3_dcp_partial_prefix_reuse, touching `test_kimi_linear_dcp_tiny, _make_tiny_k3_config, _run_k3_partial_prefix_reuse`; `vllm/v1/worker/gpu/model_runner.py` modified +6/-10 (16 lines); hunks: -59,7 +59,6; -513,17 +512,14 @@ def initialize_kv_cache(self, kv_cache_config: KVCacheConf...; symbols: initialize_kv_cache, touching `initialize_kv_cache`; `vllm/v1/core/kv_cache_coordinator.py` modified +11/-4 (15 lines); hunks: -619,15 +619,22 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/v1/worker/gpu/block_table.py` modified +8/-1 (9 lines); hunks: -116,8 +116,15 @@ def append_block_ids(; symbols: append_block_ids, apply_staged_writes, touching `append_block_ids, apply_staged_writes`.
- Code diff details:
  - `tests/distributed/test_kimi_linear_context_parallel.py` modified +111/-0 (111 lines); hunks: -2,12 +2,15; -126,3 +129,111 @@ def test_kimi_linear_dcp_tiny(; symbols: test_kimi_linear_dcp_tiny, _make_tiny_k3_config, _run_k3_partial_prefix_reuse, test_kimi_k3_dcp_partial_prefix_reuse
  - `vllm/v1/worker/gpu/model_runner.py` modified +6/-10 (16 lines); hunks: -59,7 +59,6; -513,17 +512,14 @@ def initialize_kv_cache(self, kv_cache_config: KVCacheConf...; symbols: initialize_kv_cache
  - `vllm/v1/core/kv_cache_coordinator.py` modified +11/-4 (15 lines); hunks: -619,15 +619,22 @@ def __init__(; symbols: __init__
  - `vllm/v1/worker/gpu/block_table.py` modified +8/-1 (9 lines); hunks: -116,8 +116,15 @@ def append_block_ids(; symbols: append_block_ids, apply_staged_writes
- Key code excerpts:

```diff
diff -- tests/distributed/test_kimi_linear_context_parallel.py
@@ -2,12 +2,15 @@
+from pathlib import Path
+from vllm.config import CUDAGraphMode
+from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
@@ -126,3 +129,111 @@ def test_kimi_linear_dcp_tiny(
+def _make_tiny_k3_config(model_dir: Path) -> str:
+    config = KimiK3Config(
diff -- vllm/v1/worker/gpu/model_runner.py
@@ -59,7 +59,6 @@
-from vllm.utils.math_utils import cdiv
@@ -513,17 +512,14 @@ def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
-            # When using DCP, each request's KV cache is sharded among different ranks.
-            # As a result, one block on the current rank covers `block_size * cp_size`
-            # tokens in the full, global (unsharded) sequence.
-            max_num_blocks = cdiv(
diff -- vllm/v1/core/kv_cache_coordinator.py
@@ -619,15 +619,22 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - tests: `tests/distributed/test_kimi_linear_context_parallel.py` modified +111/-0
  - runtime: `vllm/v1/worker/gpu/model_runner.py` modified +6/-10; `vllm/v1/core/kv_cache_coordinator.py` modified +11/-4; `vllm/v1/worker/gpu/block_table.py` modified +8/-1
- Risk and verification: The diff ships test coverage in `tests/distributed/test_kimi_linear_context_parallel.py`, `tests/v1/attention/test_mla_backends.py`, `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py`, `tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50400 - [Kernel][Kimi] fused vision q/k roper kernel

- Link: https://github.com/vllm-project/vllm/pull/50400
- Status/date: merged / 2026-08-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `c8de519917ce`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +14/-26, 80 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/kimi_k25_vit.py` modified +14/-26 (40 lines); hunks: -29,6 +29,7; -79,29 +80,6 @@ def get_rope_shape(org, interpolation_mode, shape):; symbols: get_rope_shape, apply_rope, get_1d_sincos_pos_embed_from_grid, __init__, touching `get_rope_shape, apply_rope, get_1d_sincos_pos_embed_from_grid`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +14/-26 (40 lines); hunks: -29,6 +29,7; -79,29 +80,6 @@ def get_rope_shape(org, interpolation_mode, shape):; symbols: get_rope_shape, apply_rope, get_1d_sincos_pos_embed_from_grid, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -29,6 +29,7 @@
+from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
@@ -79,29 +80,6 @@ def get_rope_shape(org, interpolation_mode, shape):
-def apply_rope(
-    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
-) -> tuple[torch.Tensor, torch.Tensor]:
-    """
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` modified +14/-26
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #53053 - [Kimi-K3] Extend GEMM-RS to GEMM-AR

- Link: https://github.com/vllm-project/vllm/pull/53053
- Status/date: merged / 2026-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/kda.py`, `vllm/models/kimi_k3/nvidia/mla.py`, `vllm/models/kimi_k3/nvidia/model.py`; associated commits `2785c72a1497`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +481/-232, 1343 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/model.py` modified +60/-36 (96 lines); hunks: -154,15 +154,18 @@ def shard_sequence_parallel_mlp(; -178,17 +181,32 @@ def maybe_init_gemm_rs(vllm_config: VllmConfig, use_sequen...; symbols: shard_sequence_parallel_mlp, maybe_init_gemm_rs, maybe_init_gemm_rs_ar, __init__, touching `shard_sequence_parallel_mlp, maybe_init_gemm_rs, maybe_init_gemm_rs_ar`; `vllm/models/kimi_k3/nvidia/kda.py` modified +13/-13 (26 lines); hunks: -319,7 +319,7 @@ def __init__(; -491,14 +491,18 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/models/kimi_k3/nvidia/mla.py` modified +13/-13 (26 lines); hunks: -134,7 +134,7 @@ def __init__(; -271,14 +271,18 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/model.py` modified +60/-36 (96 lines); hunks: -154,15 +154,18 @@ def shard_sequence_parallel_mlp(; -178,17 +181,32 @@ def maybe_init_gemm_rs(vllm_config: VllmConfig, use_sequen...; symbols: shard_sequence_parallel_mlp, maybe_init_gemm_rs, maybe_init_gemm_rs_ar, __init__
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +13/-13 (26 lines); hunks: -319,7 +319,7 @@ def __init__(; -491,14 +491,18 @@ def __init__(; symbols: __init__, forward
  - `vllm/models/kimi_k3/nvidia/mla.py` modified +13/-13 (26 lines); hunks: -134,7 +134,7 @@ def __init__(; -271,14 +271,18 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -154,15 +154,18 @@ def shard_sequence_parallel_mlp(
-def maybe_init_gemm_rs(vllm_config: VllmConfig, use_sequence_parallel: bool) -> bool:
-    if not envs.VLLM_KIMI_K3_GEMM_RS:
+def maybe_init_gemm_rs_ar(vllm_config: VllmConfig, use_sequence_parallel: bool) -> bool:
+    # Both feature flags may be enabled; the worker's static SP topology binds
+    # its singleton to exactly one mode.
+    all_reduce = not use_sequence_parallel
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -319,7 +319,7 @@ def __init__(
-        run_gemm_rs: bool = False,
+        run_gemm_rs_ar: bool = False,
@@ -491,14 +491,18 @@ def __init__(
-        self.run_gemm_rs = run_gemm_rs
-        if self.run_gemm_rs:
-            from vllm.models.kimi_k3.nvidia.ops.cute_dsl.gemm_rs import get_gemm_rs
diff -- vllm/models/kimi_k3/nvidia/mla.py
@@ -134,7 +134,7 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +60/-36; `vllm/models/kimi_k3/nvidia/kda.py` modified +13/-13; `vllm/models/kimi_k3/nvidia/mla.py` modified +13/-13
- Risk and verification: The diff ships test coverage in `tests/kernels/test_kimi_k3_gemm_rs_ar.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53132 - Support kimi k3 nvfp4 checkpoint

- Link: https://github.com/vllm-project/vllm/pull/53132
- Status/date: merged / 2026-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/kda.py`; associated commits `f8e060271381`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +151/-35, 366 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/kda.py` modified +11/-3 (14 lines); hunks: -36,7 +36,7; -102,7 +102,11 @@ def weight_loader(; symbols: weight_loader, weight_loader_v2, touching `weight_loader, weight_loader_v2`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +11/-3 (14 lines); hunks: -36,7 +36,7; -102,7 +102,11 @@ def weight_loader(; symbols: weight_loader, weight_loader_v2
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -36,7 +36,7 @@
-from vllm.model_executor.parameter import BasevLLMParameter
+from vllm.model_executor.parameter import BasevLLMParameter, BlockQuantScaleParameter
@@ -102,7 +102,11 @@ def weight_loader(
-        if loaded_shard_id == self.replicated_shard_id:
+        replicate_block_scale = (
+            isinstance(param, BlockQuantScaleParameter)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/kda.py` modified +11/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_trtllm_nvfp4_moe.py`, `tests/quantization/test_modelopt.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52606 - [ROCm][Perf] Kimi-K3 Fused kernels for KDA prefill

- Link: https://github.com/vllm-project/vllm/pull/52606
- Status/date: merged / 2026-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_amd_kda_chunk.py`, `vllm/models/kimi_k3/amd/kda.py`, `vllm/models/kimi_k3/amd/ops/kda_chunk.py`, `vllm/models/kimi_k3/amd/ops/kda_prefill.py`; associated commits `463aa5e30fe0`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +2716/-18, 2868 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_amd_kda_chunk.py` added +426/-0 (426 lines); hunks: -0,0 +1,426; symbols: _on_gfx950, _requires_kernel, _inputs, _run, touching `_on_gfx950, _requires_kernel, _inputs`; `vllm/models/kimi_k3/amd/ops/kda_chunk.py` added +251/-0 (251 lines); hunks: -0,0 +1,251; symbols: is_fused_kda_chunk_supported, can_use_fused_kda_chunk, fused_kda_prologue, _like, touching `is_fused_kda_chunk_supported, can_use_fused_kda_chunk, fused_kda_prologue`; `vllm/models/kimi_k3/amd/ops/kda_prefill.py` added +126/-0 (126 lines); hunks: -0,0 +1,126; symbols: chunk_kda_prefill, touching `chunk_kda_prefill`; `vllm/models/kimi_k3/amd/kda.py` modified +38/-10 (48 lines); hunks: -41,13 +41,16; -210,10 +213,21 @@ def __init__(; symbols: __init__, _prefill_conv, touching `__init__, _prefill_conv`.
- Code diff details:
  - `tests/models/kimi_k3/test_amd_kda_chunk.py` added +426/-0 (426 lines); hunks: -0,0 +1,426; symbols: _on_gfx950, _requires_kernel, _inputs, _run
  - `vllm/models/kimi_k3/amd/ops/kda_chunk.py` added +251/-0 (251 lines); hunks: -0,0 +1,251; symbols: is_fused_kda_chunk_supported, can_use_fused_kda_chunk, fused_kda_prologue, _like
  - `vllm/models/kimi_k3/amd/ops/kda_prefill.py` added +126/-0 (126 lines); hunks: -0,0 +1,126; symbols: chunk_kda_prefill
  - `vllm/models/kimi_k3/amd/kda.py` modified +38/-10 (48 lines); hunks: -41,13 +41,16; -210,10 +213,21 @@ def __init__(; symbols: __init__, _prefill_conv
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_amd_kda_chunk.py
@@ -0,0 +1,426 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""The fused ROCm KDA chunk kernel must match the Triton chunk path.
+The fused kernel reassociates nothing, but it keeps the chunk states in
+registers and rounds them to bfloat16 only where the Triton path does, so the
+outputs agree to bfloat16 rounding and the final states nearly exactly.
diff -- vllm/models/kimi_k3/amd/ops/kda_chunk.py
@@ -0,0 +1,251 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""ROCm entry point for the fused Kimi-K3 KDA chunk kernel.
+The kernel in ``csrc/libtorch_stable/kimi_k3/fused_kda_chunk_kernel_rocm.cu``
+replaces the chunk-state recurrence and the output GEMM of the Triton chunk
+path with a single launch that keeps the per-chunk state in registers, so the
diff -- vllm/models/kimi_k3/amd/ops/kda_prefill.py
@@ -0,0 +1,126 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_amd_kda_chunk.py` added +426/-0
  - runtime: `vllm/models/kimi_k3/amd/ops/kda_chunk.py` added +251/-0; `vllm/models/kimi_k3/amd/ops/kda_prefill.py` added +126/-0; `vllm/models/kimi_k3/amd/kda.py` modified +38/-10
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_amd_kda_chunk.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53152 - [K3 Perf] Fuse MXFP4 top-k finalization into latent-tail, ~5% E2E latency reduction

- Link: https://github.com/vllm-project/vllm/pull/53152
- Status/date: merged / 2026-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `benchmarks/kernels/benchmark_kimi_k3_latent_moe_tail.py`, `tests/models/kimi_k3/test_latent_moe_tail.py`, `vllm/models/kimi_k3/amd/latent_moe_runner.py`, `vllm/models/kimi_k3/nvidia/latent_moe_runner.py`, `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py` and 6 files; associated commits `7a2fdbaac449`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 17 files, +720/-139, 1654 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py` modified +160/-23 (183 lines); hunks: -16,6 +16,8; -104,6 +106,7 @@ def __init__(; symbols: __init__, __call__, touching `__init__, __call__`; `tests/models/kimi_k3/test_latent_moe_tail.py` modified +152/-0 (152 lines); hunks: -13,13 +13,65; -131,6 +183,99 @@ def _run_latent_moe_tail_test(; symbols: _make_deferred_routed_output, _run_latent_moe_tail_test, _test_deferred_finalize_parity_worker, _run_deferred_finalize_parity_test, touching `_make_deferred_routed_output, _run_latent_moe_tail_test, _test_deferred_finalize_parity_worker`; `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` modified +51/-3 (54 lines); hunks: -1,6 +1,7; -11,6 +12,7; symbols: __init__, _get_zero_residual, _select_tail_tier, _small_batch_tail, touching `__init__, _get_zero_residual, _select_tail_tier`; `vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py` modified +29/-14 (43 lines); hunks: -9,17 +9,18; -35,6 +36,7 @@ class KimiK3LatentMoETailContract:; symbols: KimiK3LatentMoETailContract, KimiK3LatentMoETailOp, _contract_and_group, initialize, touching `KimiK3LatentMoETailContract, KimiK3LatentMoETailOp, _contract_and_group`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py` modified +160/-23 (183 lines); hunks: -16,6 +16,8; -104,6 +106,7 @@ def __init__(; symbols: __init__, __call__
  - `tests/models/kimi_k3/test_latent_moe_tail.py` modified +152/-0 (152 lines); hunks: -13,13 +13,65; -131,6 +183,99 @@ def _run_latent_moe_tail_test(; symbols: _make_deferred_routed_output, _run_latent_moe_tail_test, _test_deferred_finalize_parity_worker, _run_deferred_finalize_parity_test
  - `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` modified +51/-3 (54 lines); hunks: -1,6 +1,7; -11,6 +12,7; symbols: __init__, _get_zero_residual, _select_tail_tier, _small_batch_tail
  - `vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py` modified +29/-14 (43 lines); hunks: -9,17 +9,18; -35,6 +36,7 @@ class KimiK3LatentMoETailContract:; symbols: KimiK3LatentMoETailContract, KimiK3LatentMoETailOp, _contract_and_group, initialize
  - `vllm/models/kimi_k3/amd/latent_moe_runner.py` modified +4/-3 (7 lines); hunks: -1,13 +1,15; -139,8 +141,7 @@ def _fused_forward(; symbols: _fused_forward
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py
@@ -16,6 +16,8 @@
+from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
@@ -104,6 +106,7 @@ def __init__(
+        top_k: int = 0,
@@ -153,6 +156,7 @@ def __init__(
+        self.top_k = top_k
@@ -171,6 +175,8 @@ def __call__(
diff -- tests/models/kimi_k3/test_latent_moe_tail.py
@@ -13,13 +13,65 @@
+from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
+TOP_K = 8
+def _make_deferred_routed_output(
+    num_tokens: int,
+    device: torch.device,
+) -> tuple[UnfinalizedMoEOutput, torch.Tensor]:
diff -- vllm/models/kimi_k3/nvidia/latent_moe_runner.py
@@ -1,6 +1,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py` modified +160/-23; `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` modified +51/-3; `vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py` modified +29/-14; `vllm/models/kimi_k3/amd/latent_moe_runner.py` modified +4/-3
  - tests: `tests/models/kimi_k3/test_latent_moe_tail.py` modified +152/-0
  - other: `benchmarks/kernels/benchmark_kimi_k3_latent_moe_tail.py` modified +89/-17
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_latent_moe_tail.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53294 - Revert "[ROCm][Perf] Kimi-K3 Fused kernels for KDA prefill"

- Link: https://github.com/vllm-project/vllm/pull/53294
- Status/date: merged / 2026-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_amd_kda_chunk.py`, `vllm/models/kimi_k3/amd/kda.py`, `vllm/models/kimi_k3/amd/ops/kda_chunk.py`, `vllm/models/kimi_k3/amd/ops/kda_prefill.py`; associated commits `592e06f2ae11`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +18/-2716, 2868 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_amd_kda_chunk.py` removed +0/-426 (426 lines); hunks: -1,426 +0,0; symbols: _on_gfx950, _requires_kernel, _inputs, _run, touching `_on_gfx950, _requires_kernel, _inputs`; `vllm/models/kimi_k3/amd/ops/kda_chunk.py` removed +0/-251 (251 lines); hunks: -1,251 +0,0; symbols: is_fused_kda_chunk_supported, can_use_fused_kda_chunk, fused_kda_prologue, _like, touching `is_fused_kda_chunk_supported, can_use_fused_kda_chunk, fused_kda_prologue`; `vllm/models/kimi_k3/amd/ops/kda_prefill.py` removed +0/-126 (126 lines); hunks: -1,126 +0,0; symbols: chunk_kda_prefill, touching `chunk_kda_prefill`; `vllm/models/kimi_k3/amd/kda.py` modified +10/-38 (48 lines); hunks: -41,16 +41,13; -213,21 +210,10 @@ def __init__(; symbols: __init__, _prefill_conv, touching `__init__, _prefill_conv`.
- Code diff details:
  - `tests/models/kimi_k3/test_amd_kda_chunk.py` removed +0/-426 (426 lines); hunks: -1,426 +0,0; symbols: _on_gfx950, _requires_kernel, _inputs, _run
  - `vllm/models/kimi_k3/amd/ops/kda_chunk.py` removed +0/-251 (251 lines); hunks: -1,251 +0,0; symbols: is_fused_kda_chunk_supported, can_use_fused_kda_chunk, fused_kda_prologue, _like
  - `vllm/models/kimi_k3/amd/ops/kda_prefill.py` removed +0/-126 (126 lines); hunks: -1,126 +0,0; symbols: chunk_kda_prefill
  - `vllm/models/kimi_k3/amd/kda.py` modified +10/-38 (48 lines); hunks: -41,16 +41,13; -213,21 +210,10 @@ def __init__(; symbols: __init__, _prefill_conv
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_amd_kda_chunk.py
@@ -1,426 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-"""The fused ROCm KDA chunk kernel must match the Triton chunk path.
-The fused kernel reassociates nothing, but it keeps the chunk states in
-registers and rounds them to bfloat16 only where the Triton path does, so the
-outputs agree to bfloat16 rounding and the final states nearly exactly.
diff -- vllm/models/kimi_k3/amd/ops/kda_chunk.py
@@ -1,251 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-"""ROCm entry point for the fused Kimi-K3 KDA chunk kernel.
-The kernel in ``csrc/libtorch_stable/kimi_k3/fused_kda_chunk_kernel_rocm.cu``
-replaces the chunk-state recurrence and the output GEMM of the Triton chunk
-path with a single launch that keeps the per-chunk state in registers, so the
diff -- vllm/models/kimi_k3/amd/ops/kda_prefill.py
@@ -1,126 +0,0 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_amd_kda_chunk.py` removed +0/-426
  - runtime: `vllm/models/kimi_k3/amd/ops/kda_chunk.py` removed +0/-251; `vllm/models/kimi_k3/amd/ops/kda_prefill.py` removed +0/-126; `vllm/models/kimi_k3/amd/kda.py` modified +10/-38
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_amd_kda_chunk.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53327 - [Bugfix][Kimi K3] Enable deferred MoE finalization before weight loading

- Link: https://github.com/vllm-project/vllm/pull/53327
- Status/date: merged / 2026-08-22
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_latent_moe_tail.py`, `vllm/models/kimi_k3/nvidia/latent_moe_runner.py`; associated commits `e9d1398d9edf`; preserved from an explicit existing history/skill citation
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +90/-9, 137 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_latent_moe_tail.py` modified +79/-0 (79 lines); hunks: -1,6 +1,8; -13,8 +15,13; symbols: test_deferred_finalize_enabled_before_moe_kernel_setup, FakeMoEConfig, use_deferred_moe_finalize, fake_runner_init, touching `test_deferred_finalize_enabled_before_moe_kernel_setup, FakeMoEConfig, use_deferred_moe_finalize`; `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` modified +11/-9 (20 lines); hunks: -113,16 +113,12 @@ def __init__(; -145,6 +141,12 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tests/models/kimi_k3/test_latent_moe_tail.py` modified +79/-0 (79 lines); hunks: -1,6 +1,8; -13,8 +15,13; symbols: test_deferred_finalize_enabled_before_moe_kernel_setup, FakeMoEConfig, use_deferred_moe_finalize, fake_runner_init
  - `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` modified +11/-9 (20 lines); hunks: -113,16 +113,12 @@ def __init__(; -145,6 +141,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_latent_moe_tail.py
@@ -1,6 +1,8 @@
+from types import SimpleNamespace
@@ -13,8 +15,13 @@
+from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
+    TrtLlmMxfp4ExpertsMonolithic,
+)
+from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
diff -- vllm/models/kimi_k3/nvidia/latent_moe_runner.py
@@ -113,16 +113,12 @@ def __init__(
-            moe_kernel = self._quant_method.moe_kernel
+            # The kernel instance is built after weight loading, while the tail
+            # must register its CuTeDSL warmup units during runner construction.
-                (
-                    experts_cls is TrtLlmMxfp4ExpertsMonolithic
-                    or experts_cls is TrtLlmNvFp4ExpertsMonolithic
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_latent_moe_tail.py` modified +79/-0
  - runtime: `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` modified +11/-9
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_latent_moe_tail.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53581 - [Bugfix][Kimi K3] Skip absent metadata during CUDA graph profiling

- Link: https://github.com/vllm-project/vllm/pull/53581
- Status/date: merged / 2026-08-24
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_kda.py`, `vllm/models/kimi_k3/amd/kda.py`, `vllm/models/kimi_k3/nvidia/kda.py`; associated commits `4c56e62c85ce`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +20/-2, 50 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_kda.py` modified +14/-0 (14 lines); hunks: -20,6 +20,7; -59,6 +60,19; symbols: test_kda_warmup_skips_missing_metadata, test_kda_recoverssm_config_state_layout, touching `test_kda_warmup_skips_missing_metadata, test_kda_recoverssm_config_state_layout`; `vllm/models/kimi_k3/amd/kda.py` modified +3/-1 (4 lines); hunks: -313,7 +313,9 @@ def _forward(; symbols: _forward, touching `_forward`; `vllm/models/kimi_k3/nvidia/kda.py` modified +3/-1 (4 lines); hunks: -658,7 +658,9 @@ def _forward(; symbols: _forward, touching `_forward`.
- Code diff details:
  - `tests/models/kimi_k3/test_kda.py` modified +14/-0 (14 lines); hunks: -20,6 +20,7; -59,6 +60,19; symbols: test_kda_warmup_skips_missing_metadata, test_kda_recoverssm_config_state_layout
  - `vllm/models/kimi_k3/amd/kda.py` modified +3/-1 (4 lines); hunks: -313,7 +313,9 @@ def _forward(; symbols: _forward
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +3/-1 (4 lines); hunks: -658,7 +658,9 @@ def _forward(; symbols: _forward
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_kda.py
@@ -20,6 +20,7 @@
+from vllm.models.kimi_k3.nvidia import kda as nvidia_kda
@@ -59,6 +60,19 @@
+def test_kda_warmup_skips_missing_metadata(monkeypatch):
+    monkeypatch.setattr(
+        nvidia_kda,
+        "get_forward_context",
diff -- vllm/models/kimi_k3/amd/kda.py
@@ -313,7 +313,9 @@ def _forward(
-        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
+        attn_metadata_narrowed = attn_metadata_raw.get(self.prefix)
+        if attn_metadata_narrowed is None:
+            return
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -658,7 +658,9 @@ def _forward(
-        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
+        attn_metadata_narrowed = attn_metadata_raw.get(self.prefix)
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_kda.py` modified +14/-0
  - runtime: `vllm/models/kimi_k3/amd/kda.py` modified +3/-1; `vllm/models/kimi_k3/nvidia/kda.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52388 - [K3 Perf] Optimize k3 mamba metadata preparation, 6.6~7.6x kernel performance improvement

- Link: https://github.com/vllm-project/vllm/pull/52388
- Status/date: merged / 2026-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_kda_metadata.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py`; associated commits `41729fc53b02`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +187/-6, 256 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_kda_metadata.py` modified +37/-0 (37 lines); hunks: -330,6 +330,13 @@ def test_recoverssm_spec_uses_one_state_slot_and_current_wi...; -506,6 +513,36 @@ def test_kimi_k3_kda_backend_uses_private_metadata_builder():; symbols: test_recoverssm_spec_uses_one_state_slot_and_current_window, test_kimi_k3_kda_backend_uses_private_metadata_builder, test_kimi_k3_metadata_uses_precomputed_aligned_state_indices, test_stage_spec_decode_metadata_matches_pytorch, touching `test_recoverssm_spec_uses_one_state_slot_and_current_window, test_kimi_k3_kda_backend_uses_private_metadata_builder, test_kimi_k3_metadata_uses_precomputed_aligned_state_indices`; `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +18/-6 (24 lines); hunks: -306,6 +306,8 @@ def commit_recoverssm_state(; -367,12 +369,22 @@ def build( # type: ignore[override]; symbols: commit_recoverssm_state, KimiK3KDAMetadataBuilder, __init__, build, touching `commit_recoverssm_state, KimiK3KDAMetadataBuilder, __init__`.
- Code diff details:
  - `tests/models/kimi_k3/test_kda_metadata.py` modified +37/-0 (37 lines); hunks: -330,6 +330,13 @@ def test_recoverssm_spec_uses_one_state_slot_and_current_wi...; -506,6 +513,36 @@ def test_kimi_k3_kda_backend_uses_private_metadata_builder():; symbols: test_recoverssm_spec_uses_one_state_slot_and_current_window, test_kimi_k3_kda_backend_uses_private_metadata_builder, test_kimi_k3_metadata_uses_precomputed_aligned_state_indices, test_stage_spec_decode_metadata_matches_pytorch
  - `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +18/-6 (24 lines); hunks: -306,6 +306,8 @@ def commit_recoverssm_state(; -367,12 +369,22 @@ def build( # type: ignore[override]; symbols: commit_recoverssm_state, KimiK3KDAMetadataBuilder, __init__, build
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_kda_metadata.py
@@ -330,6 +330,13 @@ def test_recoverssm_spec_uses_one_state_slot_and_current_window(
+    if mamba_cache_mode == "align":
+        builder.mamba_aligned_state_indices = mamba_get_block_table_tensor(
+            common_attn_metadata.block_table_tensor,
+            common_attn_metadata.seq_lens,
+            builder.kv_cache_spec,
+            mamba_cache_mode,
diff -- vllm/models/kimi_k3/nvidia/kda_metadata.py
@@ -306,6 +306,8 @@ def commit_recoverssm_state(
+    mamba_aligned_state_indices: torch.Tensor | None = None
@@ -367,12 +369,22 @@ def build(  # type: ignore[override]
-        block_table_tensor = _mamba_get_block_table_tensor(
-            m.block_table_tensor,
-            m.seq_lens,
-            self.kv_cache_spec,
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_kda_metadata.py` modified +37/-0
  - runtime: `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +18/-6
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda_metadata.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53766 - [CI Bug] Fix kimi test `AssertionError: Aligned Mamba state indices must be precomputed`

- Link: https://github.com/vllm-project/vllm/pull/53766
- Status/date: merged / 2026-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_kda_metadata.py`; associated commits `0e30bd62fedf`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +20/-4, 50 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_kda_metadata.py` modified +20/-4 (24 lines); hunks: -130,14 +130,22 @@ def test_internal_checkpoint_metadata_targets_last_aligned...; -159,14 +167,22 @@ def test_internal_checkpoint_metadata_skips_unaligned_offs...; symbols: test_internal_checkpoint_metadata_targets_last_aligned_boundary, test_internal_checkpoint_metadata_skips_unaligned_offset, touching `test_internal_checkpoint_metadata_targets_last_aligned_boundary, test_internal_checkpoint_metadata_skips_unaligned_offset`.
- Code diff details:
  - `tests/models/kimi_k3/test_kda_metadata.py` modified +20/-4 (24 lines); hunks: -130,14 +130,22 @@ def test_internal_checkpoint_metadata_targets_last_aligned...; -159,14 +167,22 @@ def test_internal_checkpoint_metadata_skips_unaligned_offs...; symbols: test_internal_checkpoint_metadata_targets_last_aligned_boundary, test_internal_checkpoint_metadata_skips_unaligned_offset
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_kda_metadata.py
@@ -130,14 +130,22 @@ def test_internal_checkpoint_metadata_targets_last_aligned_boundary():
-    actual = _make_builder(
+    builder = _make_builder(
-    ).build(0, common_attn_metadata)
+    )
+    assert isinstance(builder, KimiK3KDAMetadataBuilder)
+    builder.mamba_aligned_state_indices = mamba_get_block_table_tensor(
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_kda_metadata.py` modified +20/-4
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda_metadata.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53534 - [Kimi K3][Kernel] Enable low-latency decode GEMM dispatch on SM100

- Link: https://github.com/vllm-project/vllm/pull/53534
- Status/date: merged / 2026-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/low_latency_gemm.py`; associated commits `bc2d63e650f6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +323/-7, 406 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +228/-6 (234 lines); hunks: -1,12 +1,19; -280,6 +287,204 @@ def _cute(; symbols: _cute, _backend_for, _is_sm103, _low_latency_table, touching `_cute, _backend_for, _is_sm103`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +228/-6 (234 lines); hunks: -1,12 +1,19; -280,6 +287,204 @@ def _cute(; symbols: _cute, _backend_for, _is_sm103, _low_latency_table
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/low_latency_gemm.py
@@ -1,12 +1,19 @@
-"""Kimi-K3 decode GEMM selection for unquantized BF16 on SM103.
+"""Kimi-K3 decode GEMM selection for unquantized BF16 on SM103 and SM100.
+The two supported capabilities carry separate measured tables:
+:data:`KIMI_K3_PROJECTIONS` was tuned on B300 (SM103),
+:data:`KIMI_K3_PROJECTIONS_SM100` on B200 (SM100). The per-(shape, M) winners
+genuinely differ between the two parts (e.g. 3584x7168 favors dsv3 at M2..8 on
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +228/-6
- Risk and verification: The diff ships test coverage in `tests/kernels/test_bf16_skinny_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53942 - [Kimi K3 Perf] Optimize `eh_proj` linear calculation, 12.9 ~ 25.2% kernel performance improvement

- Link: https://github.com/vllm-project/vllm/pull/53942
- Status/date: merged / 2026-08-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/low_latency_gemm.py`, `vllm/models/kimi_k3/nvidia/mtp.py`; associated commits `76cfe1cd88d3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +26/-2, 70 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/mtp.py` modified +9/-1 (10 lines); hunks: -16,6 +16,7; -80,7 +81,14 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +8/-0 (8 lines); hunks: -194,6 +194,14 @@ def _cute(; symbols: _cute, touching `_cute`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/mtp.py` modified +9/-1 (10 lines); hunks: -16,6 +16,7; -80,7 +81,14 @@ def __init__(; symbols: __init__
  - `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +8/-0 (8 lines); hunks: -194,6 +194,14 @@ def _cute(; symbols: _cute
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/mtp.py
@@ -16,6 +16,7 @@
+from vllm.model_executor.layers.linear import ReplicatedLinear
@@ -80,7 +81,14 @@ def __init__(
-        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
+        self.eh_proj = ReplicatedLinear(
+            config.hidden_size * 2,
+            config.hidden_size,
diff -- vllm/models/kimi_k3/nvidia/low_latency_gemm.py
@@ -194,6 +194,14 @@ def _cute(
+    (7168, 14336): ProjectionSpec(
+        7168,
+        14336,
+        cute_configs=(
+            (1, SkinnyGemmConfig(1, 256, 2, vector_width=4, static_k=14336)),
+            (2, _cute(2, 224, 4, 2)),
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/mtp.py` modified +9/-1; `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +8/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_bf16_skinny_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53396 - [Kimi K3][Kernel] Support DS conv-state layout in fused KDA decode kernel

- Link: https://github.com/vllm-project/vllm/pull/53396
- Status/date: merged / 2026-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `benchmarks/kernels/benchmark_kimi_k3_kda_decode.py`, `tests/models/kimi_k3/test_kda.py`, `vllm/models/kimi_k3/nvidia/kda.py`; associated commits `aa640684cc95`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +213/-106, 649 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_kda.py` modified +37/-15 (52 lines); hunks: -895,13 +895,18 @@ def test_kda_recoverssm_verify_and_group_commit(; -910,6 +915,7 @@ def test_fused_kda_decode_correctness(; symbols: test_kda_recoverssm_verify_and_group_commit, test_fused_kda_decode_correctness, touching `test_kda_recoverssm_verify_and_group_commit, test_fused_kda_decode_correctness`; `vllm/models/kimi_k3/nvidia/kda.py` modified +2/-1 (3 lines); hunks: -153,14 +153,15 @@ def is_fused_kda_decode_supported(; symbols: is_fused_kda_decode_supported, touching `is_fused_kda_decode_supported`; `benchmarks/kernels/benchmark_kimi_k3_kda_decode.py` modified +25/-8 (33 lines); hunks: -81,7 +81,7 @@ def _bench_graph(fn, repeats: int = NUM_KDA_LAYERS) -> float:; -98,9 +98,15 @@ def __init__(self, num_tokens: int, num_heads: int) -> None:; symbols: _bench_graph, Inputs, __init__, touching `_bench_graph, Inputs, __init__`.
- Code diff details:
  - `tests/models/kimi_k3/test_kda.py` modified +37/-15 (52 lines); hunks: -895,13 +895,18 @@ def test_kda_recoverssm_verify_and_group_commit(; -910,6 +915,7 @@ def test_fused_kda_decode_correctness(; symbols: test_kda_recoverssm_verify_and_group_commit, test_fused_kda_decode_correctness
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +2/-1 (3 lines); hunks: -153,14 +153,15 @@ def is_fused_kda_decode_supported(; symbols: is_fused_kda_decode_supported
  - `benchmarks/kernels/benchmark_kimi_k3_kda_decode.py` modified +25/-8 (33 lines); hunks: -81,7 +81,7 @@ def _bench_graph(fn, repeats: int = NUM_KDA_LAYERS) -> float:; -98,9 +98,15 @@ def __init__(self, num_tokens: int, num_heads: int) -> None:; symbols: _bench_graph, Inputs, __init__
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_kda.py
@@ -895,13 +895,18 @@ def test_kda_recoverssm_verify_and_group_commit(
-    ("num_heads", "num_seqs", "lower_bound", "fuse_output_norm"),
+    ("num_heads", "num_seqs", "lower_bound", "fuse_output_norm", "conv_layout"),
-        (12, 1, -5.0, True),
-        (12, 4, None, False),
-        (24, 4, None, False),
-        (48, 1, -5.0, True),
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -153,14 +153,15 @@ def is_fused_kda_decode_supported(
+    # The fused kernel handles both conv-state cache layouts (SD and DS); the
+    # inner strides are selected from the tensor at launch time.
-        or is_conv_state_dim_first()
diff -- benchmarks/kernels/benchmark_kimi_k3_kda_decode.py
@@ -81,7 +81,7 @@ def _bench_graph(fn, repeats: int = NUM_KDA_LAYERS) -> float:
-    def __init__(self, num_tokens: int, num_heads: int) -> None:
+    def __init__(self, num_tokens: int, num_heads: int, conv_layout: str) -> None:
@@ -98,9 +98,15 @@ def __init__(self, num_tokens: int, num_heads: int) -> None:
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_kda.py` modified +37/-15
  - runtime: `vllm/models/kimi_k3/nvidia/kda.py` modified +2/-1
  - other: `benchmarks/kernels/benchmark_kimi_k3_kda_decode.py` modified +25/-8
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #54088 - [Kimi Perf] Tune hopper low latency gemm kernel, 4%~97% performance improvement

- Link: https://github.com/vllm-project/vllm/pull/54088
- Status/date: merged / 2026-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/low_latency_gemm.py`; associated commits `9818bb3db8fd`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +175/-35, 337 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +98/-7 (105 lines); hunks: -1,19 +1,18; -493,6 +492,96 @@ def _cute(; symbols: _cute, _sm90_spec, _backend_for, _low_latency_table, touching `_cute, _sm90_spec, _backend_for`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +98/-7 (105 lines); hunks: -1,19 +1,18; -493,6 +492,96 @@ def _cute(; symbols: _cute, _sm90_spec, _backend_for, _low_latency_table
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/low_latency_gemm.py
@@ -1,19 +1,18 @@
-"""Kimi-K3 decode GEMM selection for unquantized BF16 on SM103 and SM100.
+"""Kimi-K3 decode GEMM selection for unquantized BF16 on SM90/SM100/SM103.
-The two supported capabilities carry separate measured tables:
+The supported capabilities carry separate measured tables:
-:data:`KIMI_K3_PROJECTIONS_SM100` on B200 (SM100). The per-(shape, M) winners
-genuinely differ between the two parts (e.g. 3584x7168 favors dsv3 at M2..8 on
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +98/-7
- Risk and verification: The diff ships test coverage in `tests/kernels/test_bf16_skinny_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #54015 - [Kimi-K3] Merge MLA gate into QKV-A projection

- Link: https://github.com/vllm-project/vllm/pull/54015
- Status/date: merged / 2026-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/mla.py`, `vllm/models/kimi_k3/nvidia/model.py`, `vllm/models/kimi_k3/nvidia/mtp.py`; associated commits `6ec92bcbc8ef`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +280/-103, 533 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/mla.py` modified +167/-94 (261 lines); hunks: -5,7 +5,8; -23,7 +24,8; symbols: _gate_sigmoid_mul, __init__, touching `_gate_sigmoid_mul, __init__`; `vllm/models/kimi_k3/nvidia/model.py` modified +12/-4 (16 lines); hunks: -1120,6 +1120,7 @@ class KimiLinearModel(nn.Module, EagleModelMixin, Supports...; -1460,10 +1461,17 @@ def load_weights(; symbols: KimiLinearModel, __init__, load_weights, touching `KimiLinearModel, __init__, load_weights`; `vllm/models/kimi_k3/nvidia/mtp.py` modified +11/-4 (15 lines); hunks: -272,10 +272,17 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/mla.py` modified +167/-94 (261 lines); hunks: -5,7 +5,8; -23,7 +24,8; symbols: _gate_sigmoid_mul, __init__
  - `vllm/models/kimi_k3/nvidia/model.py` modified +12/-4 (16 lines); hunks: -1120,6 +1120,7 @@ class KimiLinearModel(nn.Module, EagleModelMixin, Supports...; -1460,10 +1461,17 @@ def load_weights(; symbols: KimiLinearModel, __init__, load_weights
  - `vllm/models/kimi_k3/nvidia/mtp.py` modified +11/-4 (15 lines); hunks: -272,10 +272,17 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/mla.py
@@ -5,7 +5,8 @@
-      -> fused pre-attention ops (fused_qkv_a_proj / norms / q_b_proj)
+      -> fused pre-attention ops (fused_qkv_a_proj / norms / q_b_proj;
+         with an output gate the rows use fused_qkv_a_g_proj instead)
@@ -23,7 +24,8 @@
-layers, enabled for DSpark) and an optional sigmoid output gate (``g_proj``).
+layers, enabled for DSpark) and an optional sigmoid output gate (``g_proj``,
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -1120,6 +1120,7 @@ class KimiLinearModel(nn.Module, EagleModelMixin, SupportsQuant):
+        "fused_qkv_a_g_proj": ["q_a_proj", "kv_a_proj_with_mqa", "g_proj"],
@@ -1460,10 +1461,17 @@ def load_weights(
-            stacked_params_mapping += [
-                (".fused_qkv_a_proj", ".q_a_proj", 0),
-                (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
-            ]
diff -- vllm/models/kimi_k3/nvidia/mtp.py
@@ -272,10 +272,17 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/mla.py` modified +167/-94; `vllm/models/kimi_k3/nvidia/model.py` modified +12/-4; `vllm/models/kimi_k3/nvidia/mtp.py` modified +11/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/linear.py`, `vllm/models/kimi_k3/nvidia/mla.py`, `vllm/models/kimi_k3/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #54167 - [Kimi-K3][Bugfix] Fix low-latency GEMM fallback initialization

- Link: https://github.com/vllm-project/vllm/pull/54167
- Status/date: merged / 2026-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/low_latency_gemm.py`; associated commits `f956e1c343dc`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +7/-13, 34 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +1/-0 (1 lines); hunks: -725,6 +725,7 @@ class _KimiK3LowLatencyApply:; symbols: _KimiK3LowLatencyApply, __init__, apply, touching `_KimiK3LowLatencyApply, __init__, apply`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +1/-0 (1 lines); hunks: -725,6 +725,7 @@ class _KimiK3LowLatencyApply:; symbols: _KimiK3LowLatencyApply, __init__, apply
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/low_latency_gemm.py
@@ -725,6 +725,7 @@ class _KimiK3LowLatencyApply:
+        super().__init__()
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_bf16_skinny_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #54168 - [Kimi-K3][Kernel] Optimize the low-M fused latent MoE tail

- Link: https://github.com/vllm-project/vllm/pull/54168
- Status/date: merged / 2026-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_latent_moe_tail.py`, `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py`, `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/fused_add_multicast_skinny_gemm.py`, `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/lamport_copy.py`, `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/primitives.py` and 6 files; associated commits `d9dabfa351ac`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +653/-194, 1203 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py` modified +327/-161 (488 lines); hunks: -23,21 +23,27; -92,7 +98,7 @@ def _select_routed_schedule(; symbols: _mapping, _select_routed_schedule, AllReduceRMSNormWithReduceScatterEarlyExit, __init__, touching `_mapping, _select_routed_schedule, AllReduceRMSNormWithReduceScatterEarlyExit`; `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/primitives.py` modified +183/-0 (183 lines); hunks: -59,6 +59,31 @@ def to_cute_dynamic_m(; -96,6 +121,104 @@ def load_global_u32x4(; symbols: to_cute_dynamic_m, fma_f32_bf16, load_global_u32x4, _make_top16_bf16_finalize_asm, touching `to_cute_dynamic_m, fma_f32_bf16, load_global_u32x4`; `tests/models/kimi_k3/test_latent_moe_tail.py` modified +107/-2 (109 lines); hunks: -153,6 +153,57 @@ def _make_deferred_routed_output(; -197,7 +248,8 @@ def _test_latent_moe_tail_worker(; symbols: _make_deferred_routed_output, _make_bf16_top16_deferred_output, _test_latent_moe_tail_worker, _test_deferred_finalize_parity_worker, touching `_make_deferred_routed_output, _make_bf16_top16_deferred_output, _test_latent_moe_tail_worker`; `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/fused_add_multicast_skinny_gemm.py` modified +24/-10 (34 lines); hunks: -19,6 +19,7; -38,6 +39,13 @@ class SkinnyConfig:; symbols: SkinnyConfig, config_for_m, _as_cute, kernel, touching `SkinnyConfig, config_for_m, _as_cute`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py` modified +327/-161 (488 lines); hunks: -23,21 +23,27; -92,7 +98,7 @@ def _select_routed_schedule(; symbols: _mapping, _select_routed_schedule, AllReduceRMSNormWithReduceScatterEarlyExit, __init__
  - `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/primitives.py` modified +183/-0 (183 lines); hunks: -59,6 +59,31 @@ def to_cute_dynamic_m(; -96,6 +121,104 @@ def load_global_u32x4(; symbols: to_cute_dynamic_m, fma_f32_bf16, load_global_u32x4, _make_top16_bf16_finalize_asm
  - `tests/models/kimi_k3/test_latent_moe_tail.py` modified +107/-2 (109 lines); hunks: -153,6 +153,57 @@ def _make_deferred_routed_output(; -197,7 +248,8 @@ def _test_latent_moe_tail_worker(; symbols: _make_deferred_routed_output, _make_bf16_top16_deferred_output, _test_latent_moe_tail_worker, _test_deferred_finalize_parity_worker
  - `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/fused_add_multicast_skinny_gemm.py` modified +24/-10 (34 lines); hunks: -19,6 +19,7; -38,6 +39,13 @@ class SkinnyConfig:; symbols: SkinnyConfig, config_for_m, _as_cute, kernel
  - `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/lamport_copy.py` modified +11/-20 (31 lines); hunks: -40,8 +40,13 @@ def __call__(; -54,14 +59,15 @@ def kernel(; symbols: __call__, kernel
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py
@@ -23,21 +23,27 @@
-    block_sum_specialized,
+    finalize_top16_bf16,
+    load_shared_f32x2,
+    load_shared_f32x4,
+    stmc_bf16x8,
+    warp_sum_specialized,
diff -- vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/primitives.py
@@ -59,6 +59,31 @@ def to_cute_dynamic_m(
+@dsl_user_op
+def fma_f32_bf16(
+    a: BFloat16,
+    b: BFloat16,
+    acc: Float32,
+    *,
diff -- tests/models/kimi_k3/test_latent_moe_tail.py
@@ -153,6 +153,57 @@ def _make_deferred_routed_output(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py` modified +327/-161; `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/primitives.py` modified +183/-0; `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/fused_add_multicast_skinny_gemm.py` modified +24/-10; `vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/lamport_copy.py` modified +11/-20; `vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py` modified +1/-1
  - tests: `tests/models/kimi_k3/test_latent_moe_tail.py` modified +107/-2
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_latent_moe_tail.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #54482 - [CI/Build] Fix Kimi K3 Eagle3 test fixture

- Link: https://github.com/vllm-project/vllm/pull/54482
- Status/date: merged / 2026-08-31
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_eagle3.py`; associated commits `648b7468b8e1`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_eagle3.py` modified +1/-0 (1 lines); hunks: -42,6 +42,7 @@ def test_kimi_k3_uses_shared_eagle3_layer_configuration():; symbols: test_kimi_k3_uses_shared_eagle3_layer_configuration, touching `test_kimi_k3_uses_shared_eagle3_layer_configuration`.
- Code diff details:
  - `tests/models/kimi_k3/test_eagle3.py` modified +1/-0 (1 lines); hunks: -42,6 +42,7 @@ def test_kimi_k3_uses_shared_eagle3_layer_configuration():; symbols: test_kimi_k3_uses_shared_eagle3_layer_configuration
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_eagle3.py
@@ -42,6 +42,7 @@ def test_kimi_k3_uses_shared_eagle3_layer_configuration():
+        forward=lambda input_ids, positions: None,
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_eagle3.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_eagle3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #54261 - [Kimi-K3][Perf] Make native CUDA AttnRes the SM100 default

- Link: https://github.com/vllm-project/vllm/pull/54261
- Status/date: merged / 2026-08-31
- Trace source: `git log --name-only -- <model-files>` found it through `benchmarks/kernels/benchmark_kimi_k3_attn_res.py`, `tests/models/kimi_k3/test_attn_res.py`, `vllm/models/kimi_k3/nvidia/ops/attn_res.py`; associated commits `d6d665854314`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +415/-94, 797 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_attn_res.py` modified +53/-0 (53 lines); hunks: -1,6 +1,8; -12,6 +14,7; symbols: _randn_with_row_padding, _reference, test_attn_res_without_output_norm, test_sm100_variants_do_not_fall_back_to_triton, touching `_randn_with_row_padding, _reference, test_attn_res_without_output_norm`; `vllm/models/kimi_k3/nvidia/ops/attn_res.py` modified +9/-8 (17 lines); hunks: -186,16 +186,16 @@ def attn_res(; -207,6 +207,7 @@ def attn_res(; symbols: attn_res, touching `attn_res`; `benchmarks/kernels/benchmark_kimi_k3_attn_res.py` added +235/-0 (235 lines); hunks: -0,0 +1,235; symbols: Inputs, Result, parse_args, make_inputs, touching `Inputs, Result, parse_args`.
- Code diff details:
  - `tests/models/kimi_k3/test_attn_res.py` modified +53/-0 (53 lines); hunks: -1,6 +1,8; -12,6 +14,7; symbols: _randn_with_row_padding, _reference, test_attn_res_without_output_norm, test_sm100_variants_do_not_fall_back_to_triton
  - `vllm/models/kimi_k3/nvidia/ops/attn_res.py` modified +9/-8 (17 lines); hunks: -186,16 +186,16 @@ def attn_res(; -207,6 +207,7 @@ def attn_res(; symbols: attn_res
  - `benchmarks/kernels/benchmark_kimi_k3_attn_res.py` added +235/-0 (235 lines); hunks: -0,0 +1,235; symbols: Inputs, Result, parse_args, make_inputs
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_attn_res.py
@@ -1,6 +1,8 @@
+import importlib
@@ -12,6 +14,7 @@
+attn_res_module = importlib.import_module("vllm.models.kimi_k3.nvidia.ops.attn_res")
@@ -60,6 +63,7 @@ def _reference(
+        pytest.param(1, 4, 0, False, True, "nvidia", id="nvidia-single-token"),
@@ -194,6 +198,55 @@ def test_attn_res_without_output_norm():
diff -- vllm/models/kimi_k3/nvidia/ops/attn_res.py
@@ -186,16 +186,16 @@ def attn_res(
-    # The in-tree NVIDIA kernel covers the common fused-add + output-norm path;
-    # Triton handles block boundaries and final pre-norm output. The native op
-    # is only compiled for SM100 under CUDA >= 13, so a device check alone is
-    # not enough to know it exists.
+    # The native kernel covers every Kimi-K3 AttnRes variant on dense SM100
+    # inputs. The op is only compiled under CUDA >= 13, so a device check alone
diff -- benchmarks/kernels/benchmark_kimi_k3_attn_res.py
@@ -0,0 +1,235 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_attn_res.py` modified +53/-0
  - runtime: `vllm/models/kimi_k3/nvidia/ops/attn_res.py` modified +9/-8
  - other: `benchmarks/kernels/benchmark_kimi_k3_attn_res.py` added +235/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_attn_res.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #54781 - [Kimi Bug] Fix `cannot access local variable 'active_non_spec_mask_cpu'`

- Link: https://github.com/vllm-project/vllm/pull/54781
- Status/date: merged / 2026-09-01
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_kda_metadata.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py`; associated commits `fc72fc39ace2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +145/-1, 186 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_kda_metadata.py` modified +143/-1 (144 lines); hunks: -2,7 +2,8; -21,6 +22,7; symbols: _make_builder, test_kda_recoverssm_startup_metadata_flow_without_model, test_internal_checkpoint_metadata_targets_last_aligned_boundary, touching `_make_builder, test_kda_recoverssm_startup_metadata_flow_without_model, test_internal_checkpoint_metadata_targets_last_aligned_boundary`; `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +2/-0 (2 lines); hunks: -409,6 +409,8 @@ def build( # type: ignore[override]; symbols: build, touching `build`.
- Code diff details:
  - `tests/models/kimi_k3/test_kda_metadata.py` modified +143/-1 (144 lines); hunks: -2,7 +2,8; -21,6 +22,7; symbols: _make_builder, test_kda_recoverssm_startup_metadata_flow_without_model, test_internal_checkpoint_metadata_targets_last_aligned_boundary
  - `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +2/-0 (2 lines); hunks: -409,6 +409,8 @@ def build( # type: ignore[override]; symbols: build
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_kda_metadata.py
@@ -2,7 +2,8 @@
-from unittest.mock import Mock
+from types import SimpleNamespace
+from unittest.mock import Mock, patch
@@ -21,6 +22,7 @@
+from vllm.models.kimi_k3.nvidia.model import KimiLinearForCausalLM
@@ -30,11 +32,13 @@
diff -- vllm/models/kimi_k3/nvidia/kda_metadata.py
@@ -409,6 +409,8 @@ def build(  # type: ignore[override]
+                if num_spec_decodes == 0:
+                    spec_sequence_masks_cpu = None
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_kda_metadata.py` modified +143/-1
  - runtime: `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda_metadata.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #54697 - [Kimi-K3] Overlap low-M TP8 KDA projections

- Link: https://github.com/vllm-project/vllm/pull/54697
- Status/date: merged / 2026-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `benchmarks/kernels/benchmark_kimi_k3_kda_projection.py`, `vllm/models/kimi_k3/nvidia/kda.py`, `vllm/models/kimi_k3/nvidia/low_latency_gemm.py`, `vllm/models/kimi_k3/nvidia/model.py`, `vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py`; associated commits `3ba9907a1db2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +1092/-12, 1197 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py` added +507/-0 (507 lines); hunks: -0,0 +1,507; symbols: _fma_f32_bf16, _KdaSkinnyNGemm, __init__, __call__, touching `_fma_f32_bf16, _KdaSkinnyNGemm, __init__`; `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +195/-0 (195 lines); hunks: -35,12 +35,32; -683,6 +703,137 @@ def _run_plan(; symbols: ProjectionSpec, _run_plan, run_kda_projection_overlap, run_qkvg, touching `ProjectionSpec, _run_plan, run_kda_projection_overlap`; `vllm/models/kimi_k3/nvidia/kda.py` modified +43/-12 (55 lines); hunks: -399,6 +399,7 @@ def __init__(; -421,6 +422,11 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/models/kimi_k3/nvidia/model.py` modified +1/-0 (1 lines); hunks: -895,6 +895,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py` added +507/-0 (507 lines); hunks: -0,0 +1,507; symbols: _fma_f32_bf16, _KdaSkinnyNGemm, __init__, __call__
  - `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +195/-0 (195 lines); hunks: -35,12 +35,32; -683,6 +703,137 @@ def _run_plan(; symbols: ProjectionSpec, _run_plan, run_kda_projection_overlap, run_qkvg
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +43/-12 (55 lines); hunks: -399,6 +399,7 @@ def __init__(; -421,6 +422,11 @@ def __init__(; symbols: __init__, forward
  - `vllm/models/kimi_k3/nvidia/model.py` modified +1/-0 (1 lines); hunks: -895,6 +895,7 @@ def __init__(; symbols: __init__
  - `benchmarks/kernels/benchmark_kimi_k3_kda_projection.py` added +221/-0 (221 lines); hunks: -0,0 +1,221; symbols: _capture, _benchmark_graph, _validate, _sequential_projection
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py
@@ -0,0 +1,507 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from
+# https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh
+"""Kimi-K3 TP8 skinny GEMMs for the KDA F_A/beta and F_B projections."""
+from __future__ import annotations
diff -- vllm/models/kimi_k3/nvidia/low_latency_gemm.py
@@ -35,12 +35,32 @@
+from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
+# TP8 KDA projection split, measured together under CUDA graph capture on B300.
+# Q/K/V/G stay on the graph's main stream while F_A/beta and F_B run on the
+# model's auxiliary stream.
+KDA_M1_QKVG_CONFIG = SkinnyGemmConfig(1, 64, 4, 2, 8)
+KDA_M1_FAB_CONFIG = SkinnyGemmConfig(1, 224, 1, 2, 8)
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -399,6 +399,7 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py` added +507/-0; `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +195/-0; `vllm/models/kimi_k3/nvidia/kda.py` modified +43/-12; `vllm/models/kimi_k3/nvidia/model.py` modified +1/-0
  - other: `benchmarks/kernels/benchmark_kimi_k3_kda_projection.py` added +221/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_bf16_skinny_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #54859 - [Kimi-K3] Bump FlashKDA to fix unstable inverse

- Link: https://github.com/vllm-project/vllm/pull/54859
- Status/date: merged / 2026-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_kda.py`; associated commits `f4e613614628`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +51/-1, 66 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_kda.py` modified +50/-0 (50 lines); hunks: -1083,6 +1083,56 @@ def test_fused_kda_decode_rejects_speculative_conv_state():; symbols: test_fused_kda_decode_rejects_speculative_conv_state, test_flashkda_near_collinear_keys_remain_finite, test_flashkda_correctness, touching `test_fused_kda_decode_rejects_speculative_conv_state, test_flashkda_near_collinear_keys_remain_finite, test_flashkda_correctness`.
- Code diff details:
  - `tests/models/kimi_k3/test_kda.py` modified +50/-0 (50 lines); hunks: -1083,6 +1083,56 @@ def test_fused_kda_decode_rejects_speculative_conv_state():; symbols: test_fused_kda_decode_rejects_speculative_conv_state, test_flashkda_near_collinear_keys_remain_finite, test_flashkda_correctness
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_kda.py
@@ -1083,6 +1083,56 @@ def test_fused_kda_decode_rejects_speculative_conv_state():
+@torch.inference_mode()
+def test_flashkda_near_collinear_keys_remain_finite():
+    """Guard against unstable inversion of near-collinear key blocks."""
+    lower_bound = -5.0
+    if not is_flashkda_supported(128, torch.bfloat16, lower_bound):
+        pytest.skip("FlashKDA is not supported on this platform")
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_kda.py` modified +50/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #54817 - [CI] Add Kimi-K3-pruned75-DSpark-TP4 gsm8k eval

- Link: https://github.com/vllm-project/vllm/pull/54817
- Status/date: merged / 2026-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/Kimi-K3-pruned75-DSpark-TP4.yaml`; associated commits `872084fb773b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +64/-30, 124 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/evals/gsm8k/configs/Kimi-K3-pruned75-DSpark-TP4.yaml` added +26/-0 (26 lines); hunks: -0,0 +1,26; `vllm/v1/attention/backends/mla/prefill/flashinfer.py` modified +33/-28 (61 lines); hunks: -7,6 +7,7; -161,39 +162,43 @@ def prepare_metadata(; symbols: prepare_metadata, supports_out, touching `prepare_metadata, supports_out`.
- Code diff details:
  - `tests/evals/gsm8k/configs/Kimi-K3-pruned75-DSpark-TP4.yaml` added +26/-0 (26 lines); hunks: -0,0 +1,26
  - `vllm/v1/attention/backends/mla/prefill/flashinfer.py` modified +33/-28 (61 lines); hunks: -7,6 +7,7; -161,39 +162,43 @@ def prepare_metadata(; symbols: prepare_metadata, supports_out
- Key code excerpts:

```diff
diff -- tests/evals/gsm8k/configs/Kimi-K3-pruned75-DSpark-TP4.yaml
@@ -0,0 +1,26 @@
+model_name: "mgoin/Kimi-K3-pruned75"
+accuracy_threshold: 0.33
+min_acceptance_length: 4.8
+num_questions: 1319
+num_fewshot: 5
+startup_max_wait_seconds: 1800
diff -- vllm/v1/attention/backends/mla/prefill/flashinfer.py
@@ -7,6 +7,7 @@
+from vllm.utils.gpu_sync_debug import gpu_sync_allowed
@@ -161,39 +162,43 @@ def prepare_metadata(
-        self._prefill_main.plan(
-            qo_indptr=qo_indptr,
-            kv_indptr=kv_indptr,
-            num_qo_heads=num_qo_heads,
```

- Extracted files (not manually reviewed):
  - tests: `tests/evals/gsm8k/configs/Kimi-K3-pruned75-DSpark-TP4.yaml` added +26/-0
  - runtime: `vllm/v1/attention/backends/mla/prefill/flashinfer.py` modified +33/-28
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-confidence-TEP4.yaml`, `tests/evals/gsm8k/configs/Kimi-K3-pruned75-DSpark-TP4.yaml`, `tests/evals/gsm8k/configs/models-spec-decode.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #54565 - [K3 Perf] Enable DSV3 GEMM for inner-contiguous and row-strided tensors, 12%~81% kernel performance improvement

- Link: https://github.com/vllm-project/vllm/pull/54565
- Status/date: merged / 2026-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/low_latency_gemm.py`; associated commits `a56654d6de06`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +48/-35, 224 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +7/-2 (9 lines); hunks: -643,7 +643,9 @@ def _is_packed_row_major(tensor: torch.Tensor) -> bool:; -656,7 +658,8 @@ def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:; symbols: _is_packed_row_major, _runtime_ok, _residual_ok, _run_plan, touching `_is_packed_row_major, _runtime_ok, _residual_ok`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +7/-2 (9 lines); hunks: -643,7 +643,9 @@ def _is_packed_row_major(tensor: torch.Tensor) -> bool:; -656,7 +658,8 @@ def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:; symbols: _is_packed_row_major, _runtime_ok, _residual_ok, _run_plan
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/low_latency_gemm.py
@@ -643,7 +643,9 @@ def _is_packed_row_major(tensor: torch.Tensor) -> bool:
-        _is_packed_row_major(x)
+        x.dim() == 2
+        and x.stride(1) == 1
+        and (x.shape[0] == 1 or x.stride(0) % 8 == 0)
@@ -656,7 +658,8 @@ def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
-        residual.dim() == 2
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +7/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/test_bf16_skinny_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #54606 - [Kernel] Enable Kimi-K3 SiTU on the CuteDSL MoE backend and the SM107 low-latency GEMM plan

- Link: https://github.com/vllm-project/vllm/pull/54606
- Status/date: merged / 2026-09-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/low_latency_gemm.py`; associated commits `d410fc12f30b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +30/-4, 83 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +11/-3 (14 lines); hunks: -1,6 +1,6; -12,7 +12,10; symbols: _is_sm103, _is_sm107, _low_latency_table, touching `_is_sm103, _is_sm107, _low_latency_table`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +11/-3 (14 lines); hunks: -1,6 +1,6; -12,7 +12,10; symbols: _is_sm103, _is_sm107, _low_latency_table
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/low_latency_gemm.py
@@ -1,6 +1,6 @@
-"""Kimi-K3 decode GEMM selection for unquantized BF16 on SM90/SM100/SM103.
+"""Kimi-K3 decode GEMM selection for unquantized BF16 on SM90/SM100/SM103/SM107.
@@ -12,7 +12,10 @@
-genuinely differ between the parts, so the tables must not be merged.
+genuinely differ between the parts, so the tables must not be merged. SM107
+(Rubin) reuses the SM103 table: the plan was validated end-to-end on SM107
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +11/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py`, `vllm/models/kimi_k3/nvidia/low_latency_gemm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #54896 - [Perf][Kimi-K3] Cut MLA decode concat/cache epilogue latency

- Link: https://github.com/vllm-project/vllm/pull/54896
- Status/date: merged / 2026-09-03
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py`; associated commits `9509fc8ae61a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +121/-36, 261 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py` modified +23/-0 (23 lines); hunks: -285,3 +285,26 @@ def test_decode_epilogue_preserves_nope_path() -> None:; symbols: test_decode_epilogue_preserves_nope_path, test_decode_epilogue_row_split_boundary, touching `test_decode_epilogue_preserves_nope_path, test_decode_epilogue_row_split_boundary`.
- Code diff details:
  - `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py` modified +23/-0 (23 lines); hunks: -285,3 +285,26 @@ def test_decode_epilogue_preserves_nope_path() -> None:; symbols: test_decode_epilogue_preserves_nope_path, test_decode_epilogue_row_split_boundary
- Key code excerpts:

```diff
diff -- tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py
@@ -285,3 +285,26 @@ def test_decode_epilogue_preserves_nope_path() -> None:
+@torch.inference_mode()
+@pytest.mark.parametrize("num_tokens", [64, 65])
+def test_decode_epilogue_row_split_boundary(num_tokens: int) -> None:
+    """Both sides of the M-based warps-per-row dispatch produce the same rows."""
+    torch.manual_seed(3)
+    num_blocks = (num_tokens + _BLOCK_SIZE - 1) // _BLOCK_SIZE
```

- Extracted files (not manually reviewed):
  - tests: `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py` modified +23/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52494 - [AMD][kimik3][ROCm][Perf] Fuse MLA q/kv RMSNorm in AMD Kimi-K3 MLA wrapper

- Link: https://github.com/vllm-project/vllm/pull/52494
- Status/date: merged / 2026-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/amd/linear.py`, `vllm/models/kimi_k3/amd/mla.py`; associated commits `3ff4f02dfe69`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +123/-2, 147 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/amd/mla.py` added +120/-0 (120 lines); hunks: -0,0 +1,120; symbols: KimiK3MultiHeadLatentAttentionWrapper, __init__, _normalize_q_kv, forward, touching `KimiK3MultiHeadLatentAttentionWrapper, __init__, _normalize_q_kv`; `vllm/models/kimi_k3/amd/linear.py` modified +3/-2 (5 lines); hunks: -36,7 +36,7; -63,6 +63,7; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/kimi_k3/amd/mla.py` added +120/-0 (120 lines); hunks: -0,0 +1,120; symbols: KimiK3MultiHeadLatentAttentionWrapper, __init__, _normalize_q_kv, forward
  - `vllm/models/kimi_k3/amd/linear.py` modified +3/-2 (5 lines); hunks: -36,7 +36,7; -63,6 +63,7; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/amd/mla.py
@@ -0,0 +1,120 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""AMD-specific MLA wrapper for Kimi-K3."""
+from typing import cast
+import torch
+from vllm._aiter_ops import rocm_aiter_ops
diff -- vllm/models/kimi_k3/amd/linear.py
@@ -36,7 +36,7 @@
-from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
+from vllm.model_executor.layers.mla import MLAModules
@@ -63,6 +63,7 @@
+from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper
@@ -438,7 +439,7 @@ def __init__(
-        self.mla_attn = MultiHeadLatentAttentionWrapper(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/amd/mla.py` added +120/-0; `vllm/models/kimi_k3/amd/linear.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/amd/linear.py`, `vllm/models/kimi_k3/amd/mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #55242 - [Perf] Kimi K3 nvfp4 Align in_proj weights by 128 to avoid elementwise copy

- Link: https://github.com/vllm-project/vllm/pull/55242
- Status/date: merged / 2026-09-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/kda.py`, `vllm/models/kimi_k3/nvidia/mla.py`, `vllm/models/kimi_k3/nvidia/model.py`; associated commits `16328c7a775f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +53/-21, 195 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/kda.py` modified +14/-8 (22 lines); hunks: -33,6 +33,9; -428,8 +431,7 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/models/kimi_k3/nvidia/mla.py` modified +1/-4 (5 lines); hunks: -306,10 +306,7 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/models/kimi_k3/nvidia/model.py` modified +1/-4 (5 lines); hunks: -283,10 +283,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +14/-8 (22 lines); hunks: -33,6 +33,9; -428,8 +431,7 @@ def __init__(; symbols: __init__
  - `vllm/models/kimi_k3/nvidia/mla.py` modified +1/-4 (5 lines); hunks: -306,10 +306,7 @@ def __init__(; symbols: __init__
  - `vllm/models/kimi_k3/nvidia/model.py` modified +1/-4 (5 lines); hunks: -283,10 +283,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -33,6 +33,9 @@
+from vllm.model_executor.layers.quantization.modelopt import (
+    ModelOptMixedPrecisionConfig,
+)
@@ -428,8 +431,7 @@ def __init__(
-        # Keep f_a before the narrow beta shard, then pad each TP-local row
-        # to select the aligned BF16 GEMM path.
diff -- vllm/models/kimi_k3/nvidia/mla.py
@@ -306,10 +306,7 @@ def __init__(
-                logger.warning_once(
-                    "GEMM-RS/AR is disabled for %s due to an incompatible projection.",
-                    prefix,
-                )
+                gemm_rs_ar.warn_incompatible_projection()
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -283,10 +283,7 @@ def __init__(
-                logger.warning_once(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/kda.py` modified +14/-8; `vllm/models/kimi_k3/nvidia/mla.py` modified +1/-4; `vllm/models/kimi_k3/nvidia/model.py` modified +1/-4
- Risk and verification: The diff ships test coverage in `tests/quantization/test_modelopt.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53614 - [Kimi K3] Support internal prefix checkpoints with partial prefix caching and spec-decoding

- Link: https://github.com/vllm-project/vllm/pull/53614
- Status/date: merged / 2026-09-06
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_kda_metadata.py`, `vllm/models/kimi_k3/nvidia/kda.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py`; associated commits `144e79c8106d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +481/-106, 1033 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_kda_metadata.py` modified +71/-1 (72 lines); hunks: -92,6 +92,10 @@ def _make_builder(; -102,20 +106,29 @@ def _make_builder(; symbols: _make_builder, test_kda_recoverssm_startup_metadata_flow_without_model, test_internal_checkpoint_metadata_targets_last_aligned_boundary, touching `_make_builder, test_kda_recoverssm_startup_metadata_flow_without_model, test_internal_checkpoint_metadata_targets_last_aligned_boundary`; `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +35/-15 (50 lines); hunks: -20,6 +20,7; -36,17 +37,18; symbols: _metadata_launch_pdl, build, touching `_metadata_launch_pdl, build`; `vllm/models/kimi_k3/nvidia/kda.py` modified +3/-0 (3 lines); hunks: -609,6 +609,9 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> Mamb...; symbols: get_kv_cache_spec, forward, touching `get_kv_cache_spec, forward`.
- Code diff details:
  - `tests/models/kimi_k3/test_kda_metadata.py` modified +71/-1 (72 lines); hunks: -92,6 +92,10 @@ def _make_builder(; -102,20 +106,29 @@ def _make_builder(; symbols: _make_builder, test_kda_recoverssm_startup_metadata_flow_without_model, test_internal_checkpoint_metadata_targets_last_aligned_boundary
  - `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +35/-15 (50 lines); hunks: -20,6 +20,7; -36,17 +37,18; symbols: _metadata_launch_pdl, build
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +3/-0 (3 lines); hunks: -609,6 +609,9 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> Mamb...; symbols: get_kv_cache_spec, forward
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_kda_metadata.py
@@ -92,6 +92,10 @@ def _make_builder(
+    mamba_block_size: int = BLOCK_SIZE,
+    prefix_match_unit: int | None = None,
+    use_eagle: bool = False,
+    disable_eagle_block_drop: bool = False,
@@ -102,20 +106,29 @@ def _make_builder(
+        if use_eagle:
diff -- vllm/models/kimi_k3/nvidia/kda_metadata.py
@@ -20,6 +20,7 @@
+from vllm.utils.math_utils import cdiv
@@ -36,17 +37,18 @@
-from vllm.v1.kv_cache_interface import MambaSpec
+from vllm.v1.kv_cache_interface import (
+    MambaSpec,
+    get_mamba_prefill_checkpoint_position,
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -609,6 +609,9 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> MambaSpec:
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_kda_metadata.py` modified +71/-1
  - runtime: `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +35/-15; `vllm/models/kimi_k3/nvidia/kda.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda_metadata.py`, `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py`, `tests/v1/core/test_mamba_align_chunk_split.py`, `tests/v1/core/test_prefix_caching.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53379 - [Bugfix] Fix Kimi K3 loading with interleaved weight streams

- Link: https://github.com/vllm-project/vllm/pull/53379
- Status/date: merged / 2026-09-08
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_weight_loading.py`, `vllm/models/kimi_k3/nvidia/model.py`; associated commits `f998862d46e7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +80/-2, 98 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_weight_loading.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: _FakeKimiLinearModel, __init__, load_weights, finalize_mega_moe_weights, touching `_FakeKimiLinearModel, __init__, load_weights`; `vllm/models/kimi_k3/nvidia/model.py` modified +8/-2 (10 lines); hunks: -1705,12 +1705,15 @@ def compute_logits(; -2155,3 +2158,6 @@ def get_mamba_state_copy_func(cls):; symbols: compute_logits, load_weights, process_weights_after_loading, get_spec_layer_idx_from_weight_name, touching `compute_logits, load_weights, process_weights_after_loading`.
- Code diff details:
  - `tests/models/kimi_k3/test_weight_loading.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: _FakeKimiLinearModel, __init__, load_weights, finalize_mega_moe_weights
  - `vllm/models/kimi_k3/nvidia/model.py` modified +8/-2 (10 lines); hunks: -1705,12 +1705,15 @@ def compute_logits(; -2155,3 +2158,6 @@ def get_mamba_state_copy_func(cls):; symbols: compute_logits, load_weights, process_weights_after_loading, get_spec_layer_idx_from_weight_name
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_weight_loading.py
@@ -0,0 +1,72 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from types import SimpleNamespace
+import pytest
+import torch
+from torch import nn
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -1705,12 +1705,15 @@ def compute_logits(
-        loaded = loader.load_weights(weights)
+        return loader.load_weights(weights)
+    def process_weights_after_loading(self) -> None:
+        # A parent AutoWeightsLoader may invoke load_weights repeatedly for
+        # non-contiguous streamed prefixes. Finalize only after the full stream.
-        return loaded
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_weight_loading.py` added +72/-0
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +8/-2
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_weight_loading.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #55924 - [Kimi Bug] Fix kda ima `Triton Error [CUDA]: an illegal memory access was encountered`

- Link: https://github.com/vllm-project/vllm/pull/55924
- Status/date: merged / 2026-09-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/kda.py`; associated commits `bfb443a6b6f6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/kda.py` modified +1/-1 (2 lines); hunks: -415,7 +415,7 @@ def _store_cache_checkpoints_kernel(; symbols: _store_cache_checkpoints_kernel, touching `_store_cache_checkpoints_kernel`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +1/-1 (2 lines); hunks: -415,7 +415,7 @@ def _store_cache_checkpoints_kernel(; symbols: _store_cache_checkpoints_kernel
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -415,7 +415,7 @@ def _store_cache_checkpoints_kernel(
-    state_idx = tl.load(checkpoint_state_indices_ptr + seq_idx)
+    state_idx = tl.load(checkpoint_state_indices_ptr + seq_idx).to(tl.int64)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/kda.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/kda.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #54038 - [ROCm][Perf] Kimi-K3 Fused kernels for KDA prefill reland

- Link: https://github.com/vllm-project/vllm/pull/54038
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_amd_kda_chunk.py`, `vllm/models/kimi_k3/amd/kda.py`, `vllm/models/kimi_k3/amd/ops/kda_chunk.py`, `vllm/models/kimi_k3/amd/ops/kda_prefill.py`; associated commits `9521c60bdc0c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +3818/-33, 1639 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_amd_kda_chunk.py` added +859/-0 (859 lines); hunks: -0,0 +1,859; symbols: _on_gfx950, _requires_kernel, _inputs, _run, touching `_on_gfx950, _requires_kernel, _inputs`; `vllm/models/kimi_k3/amd/ops/kda_chunk.py` added +294/-0 (294 lines); hunks: -0,0 +1,294; symbols: is_fused_kda_chunk_supported, can_use_fused_kda_chunk, fused_kda_prologue, _like, touching `is_fused_kda_chunk_supported, can_use_fused_kda_chunk, fused_kda_prologue`; `vllm/models/kimi_k3/amd/ops/kda_prefill.py` added +179/-0 (179 lines); hunks: -0,0 +1,179; symbols: chunk_kda_prefill, touching `chunk_kda_prefill`; `vllm/models/kimi_k3/amd/kda.py` modified +44/-23 (67 lines); hunks: -35,19 +35,19; -210,10 +210,21 @@ def __init__(; symbols: __init__, _prefill_conv, touching `__init__, _prefill_conv`.
- Code diff details:
  - `tests/models/kimi_k3/test_amd_kda_chunk.py` added +859/-0 (859 lines); hunks: -0,0 +1,859; symbols: _on_gfx950, _requires_kernel, _inputs, _run
  - `vllm/models/kimi_k3/amd/ops/kda_chunk.py` added +294/-0 (294 lines); hunks: -0,0 +1,294; symbols: is_fused_kda_chunk_supported, can_use_fused_kda_chunk, fused_kda_prologue, _like
  - `vllm/models/kimi_k3/amd/ops/kda_prefill.py` added +179/-0 (179 lines); hunks: -0,0 +1,179; symbols: chunk_kda_prefill
  - `vllm/models/kimi_k3/amd/kda.py` modified +44/-23 (67 lines); hunks: -35,19 +35,19; -210,10 +210,21 @@ def __init__(; symbols: __init__, _prefill_conv
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_amd_kda_chunk.py
@@ -0,0 +1,859 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""The fused ROCm KDA chunk kernel must match the Triton chunk path.
+The fused kernel reassociates nothing, but it keeps the chunk states in
+registers and rounds them to bfloat16 only where the Triton path does, so the
+outputs agree to bfloat16 rounding and the final states nearly exactly.
diff -- vllm/models/kimi_k3/amd/ops/kda_chunk.py
@@ -0,0 +1,294 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""ROCm entry point for the fused Kimi-K3 KDA chunk kernel.
+The kernel in ``csrc/libtorch_stable/kimi_k3/fused_kda_chunk_kernel_rocm.cu``
+replaces the chunk-state recurrence and the output GEMM of the Triton chunk
+path with a single launch that keeps the per-chunk state in registers, so the
diff -- vllm/models/kimi_k3/amd/ops/kda_prefill.py
@@ -0,0 +1,179 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_amd_kda_chunk.py` added +859/-0
  - runtime: `vllm/models/kimi_k3/amd/ops/kda_chunk.py` added +294/-0; `vllm/models/kimi_k3/amd/ops/kda_prefill.py` added +179/-0; `vllm/models/kimi_k3/amd/kda.py` modified +44/-23
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_amd_kda_chunk.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56159 - [Kimi K3 Perf] Avoid KDA mixed-batch gather/scatter, 5.2%~7.7% E2E Throughput Improvement

- Link: https://github.com/vllm-project/vllm/pull/56159
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_kda.py`, `tests/models/kimi_k3/test_kda_metadata.py`, `vllm/models/kimi_k3/nvidia/kda.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py`, `vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk.py` and 6 files; associated commits `86aca6619161`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +66/-5, 261 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/kda.py` modified +33/-3 (36 lines); hunks: -903,6 +903,8 @@ def _forward(; -981,6 +983,18 @@ def _forward(; symbols: _forward, _prefill_conv, touching `_forward, _prefill_conv`; `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +13/-0 (13 lines); hunks: -270,6 +270,8 @@ class KDACheckpointMetadata:; -422,6 +424,8 @@ def build( # type: ignore[override]; symbols: KDACheckpointMetadata, KimiK3KDAMetadata, build, touching `KDACheckpointMetadata, KimiK3KDAMetadata, build`; `tests/models/kimi_k3/test_kda.py` modified +7/-0 (7 lines); hunks: -330,6 +330,7 @@ def test_chunk_kda_fused_gate_cumsum_matches_unfused(; -343,8 +344,10 @@ def test_chunk_kda_fused_gate_cumsum_matches_unfused(; symbols: test_chunk_kda_fused_gate_cumsum_matches_unfused, test_packed_kda_decode_correctness, touching `test_chunk_kda_fused_gate_cumsum_matches_unfused, test_packed_kda_decode_correctness`; `vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk.py` modified +6/-1 (7 lines); hunks: -598,6 +598,7 @@ def _chunk_kda_fwd_with_cumulative_g(; -642,7 +643,7 @@ def _chunk_kda_fwd_with_cumulative_g(; symbols: _chunk_kda_fwd_with_cumulative_g, chunk_kda_with_fused_gate_fwd, chunk_kda_with_fused_gate, touching `_chunk_kda_fwd_with_cumulative_g, chunk_kda_with_fused_gate_fwd, chunk_kda_with_fused_gate`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +33/-3 (36 lines); hunks: -903,6 +903,8 @@ def _forward(; -981,6 +983,18 @@ def _forward(; symbols: _forward, _prefill_conv
  - `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +13/-0 (13 lines); hunks: -270,6 +270,8 @@ class KDACheckpointMetadata:; -422,6 +424,8 @@ def build( # type: ignore[override]; symbols: KDACheckpointMetadata, KimiK3KDAMetadata, build
  - `tests/models/kimi_k3/test_kda.py` modified +7/-0 (7 lines); hunks: -330,6 +330,7 @@ def test_chunk_kda_fused_gate_cumsum_matches_unfused(; -343,8 +344,10 @@ def test_chunk_kda_fused_gate_cumsum_matches_unfused(; symbols: test_chunk_kda_fused_gate_cumsum_matches_unfused, test_packed_kda_decode_correctness
  - `vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk.py` modified +6/-1 (7 lines); hunks: -598,6 +598,7 @@ def _chunk_kda_fwd_with_cumulative_g(; -642,7 +643,7 @@ def _chunk_kda_fwd_with_cumulative_g(; symbols: _chunk_kda_fwd_with_cumulative_g, chunk_kda_with_fused_gate_fwd, chunk_kda_with_fused_gate
  - `tests/models/kimi_k3/test_kda_metadata.py` modified +4/-0 (4 lines); hunks: -512,6 +512,8 @@ def test_mixed_regular_and_spec_decode_uses_packed_decode_me...; -539,6 +541,8 @@ def test_mixed_regular_and_spec_decode_excludes_request_padd...; symbols: test_mixed_regular_and_spec_decode_uses_packed_decode_metadata, test_mixed_regular_and_spec_decode_excludes_request_padding
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -903,6 +903,8 @@ def _forward(
+        spec_token_start = m.spec_token_start
+        non_spec_token_start = m.non_spec_token_start
@@ -981,6 +983,18 @@ def _forward(
+            elif spec_token_start is not None:
+                assert non_spec_token_start is not None
+                spec_end = spec_token_start + m.num_spec_decode_tokens
diff -- vllm/models/kimi_k3/nvidia/kda_metadata.py
@@ -270,6 +270,8 @@ class KDACheckpointMetadata:
+    spec_token_start: int | None = None
+    non_spec_token_start: int | None = None
@@ -422,6 +424,8 @@ def build(  # type: ignore[override]
+        spec_token_start = None
+        non_spec_token_start = None
@@ -518,6 +522,13 @@ def build(  # type: ignore[override]
diff -- tests/models/kimi_k3/test_kda.py
@@ -330,6 +330,7 @@ def test_chunk_kda_fused_gate_cumsum_matches_unfused(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/kda.py` modified +33/-3; `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +13/-0; `vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk.py` modified +6/-1; `vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py` modified +3/-1
  - tests: `tests/models/kimi_k3/test_kda.py` modified +7/-0; `tests/models/kimi_k3/test_kda_metadata.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda.py`, `tests/models/kimi_k3/test_kda_metadata.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #55356 - [Kimi Perf] Group fp8 mla cahche insertion, 4~6x kernel level performance improvement for small batch

- Link: https://github.com/vllm-project/vllm/pull/55356
- Status/date: merged / 2026-09-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/dspark_mla.py`; associated commits `1e1060f9988f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +216/-39, 385 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +21/-5 (26 lines); hunks: -28,6 +28,10; -233,6 +237,13 @@ def _build_fused_context_kv_metadata(self) -> None:; symbols: _duplicate_context_kv_weights, _build_fused_context_kv_metadata, _precompute_fused_context_kv, touching `_duplicate_context_kv_weights, _build_fused_context_kv_metadata, _precompute_fused_context_kv`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +21/-5 (26 lines); hunks: -28,6 +28,10; -233,6 +237,13 @@ def _build_fused_context_kv_metadata(self) -> None:; symbols: _duplicate_context_kv_weights, _build_fused_context_kv_metadata, _precompute_fused_context_kv
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/dspark_mla.py
@@ -28,6 +28,10 @@
+_GROUPED_KV_CACHE_DTYPES = frozenset(
+    {"auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"}
+)
@@ -233,6 +237,13 @@ def _build_fused_context_kv_metadata(self) -> None:
+        self._context_kv_scales: torch.Tensor | None = None
+        if attn0.kv_cache_dtype in _GROUPED_KV_CACHE_DTYPES and is_quantized_kv_cache(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +21/-5
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #55426 - [Bugfix][Kimi-K3] Fix KDA projection overlap on Hopper

- Link: https://github.com/vllm-project/vllm/pull/55426
- Status/date: merged / 2026-09-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/low_latency_gemm.py`, `vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py`; associated commits `0c1e89ceb92b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +239/-43, 406 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py` modified +91/-14 (105 lines); hunks: -18,6 +18,8; -91,14 +93,61 @@ def _fma_f32_bf16(; symbols: _fma_f32_bf16, _fma_f32_bf16_portable, _has_mixed_precision_bf16_fma, _KdaSkinnyNGemm, touching `_fma_f32_bf16, _fma_f32_bf16_portable, _has_mixed_precision_bf16_fma`; `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +47/-28 (75 lines); hunks: -65,6 +65,20; -749,14 +763,16 @@ def run_qkvg() -> torch.Tensor:; symbols: _kda_qkvg_flashinfer_backend, ProjectionSpec, run_qkvg, run_fab_fb, touching `_kda_qkvg_flashinfer_backend, ProjectionSpec, run_qkvg`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py` modified +91/-14 (105 lines); hunks: -18,6 +18,8; -91,14 +93,61 @@ def _fma_f32_bf16(; symbols: _fma_f32_bf16, _fma_f32_bf16_portable, _has_mixed_precision_bf16_fma, _KdaSkinnyNGemm
  - `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +47/-28 (75 lines); hunks: -65,6 +65,20; -749,14 +763,16 @@ def run_qkvg() -> torch.Tensor:; symbols: _kda_qkvg_flashinfer_backend, ProjectionSpec, run_qkvg, run_fab_fb
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py
@@ -18,6 +18,8 @@
+from vllm.platforms import current_platform
@@ -91,14 +93,61 @@ def _fma_f32_bf16(
+@dsl_user_op
+def _fma_f32_bf16_portable(
+    a: BFloat16,
+    b: BFloat16,
diff -- vllm/models/kimi_k3/nvidia/low_latency_gemm.py
@@ -65,6 +65,20 @@
+def _kda_qkvg_flashinfer_backend() -> Literal["cute-dsl"] | None:
+    """Return the tested FlashInfer backend for the low-M QKVG branch.
+    FlashInfer's ``cute-dsl`` BF16 GEMM supports SM100 and SM103, but rejects
+    SM90 during backend validation. Other capabilities use ``torch.mm`` for
+    QKVG while retaining the concurrent F_A/beta and F_B branch.
+    """
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py` modified +91/-14; `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` modified +47/-28
- Risk and verification: The diff ships test coverage in `tests/kernels/test_bf16_skinny_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56526 - [ROCm][Kimi-K3] Fix non-contiguous state_indices crash and GPU-sync assert in fused KDA/MLA prefill

- Link: https://github.com/vllm-project/vllm/pull/56526
- Status/date: merged / 2026-09-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/amd/ops/kda_chunk.py`; associated commits `06e57f622cf5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +5/-3, 30 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/amd/ops/kda_chunk.py` modified +2/-2 (4 lines); hunks: -286,9 +286,9 @@ def fused_kda_chunk(; symbols: fused_kda_chunk, touching `fused_kda_chunk`.
- Code diff details:
  - `vllm/models/kimi_k3/amd/ops/kda_chunk.py` modified +2/-2 (4 lines); hunks: -286,9 +286,9 @@ def fused_kda_chunk(; symbols: fused_kda_chunk
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/amd/ops/kda_chunk.py
@@ -286,9 +286,9 @@ def fused_kda_chunk(
-        else checkpoint_state_indices.to(torch.int32),
+        else checkpoint_state_indices.to(torch.int32).contiguous(),
-        None if state_indices is None else state_indices.to(torch.int32),
+        None if state_indices is None else state_indices.to(torch.int32).contiguous(),
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/amd/ops/kda_chunk.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/amd/ops/kda_chunk.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51483 - [Bugfix][Kimi-K3] Do not classify a stateless first chunk as a decode

- Link: https://github.com/vllm-project/vllm/pull/51483
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_kda_metadata.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py`; associated commits `f8b5c11468f6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +113/-2, 128 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_kda_metadata.py` modified +88/-0 (88 lines); hunks: -934,3 +934,91 @@ def test_aligned_block_table_matches_shared_gdn():; symbols: test_aligned_block_table_matches_shared_gdn, _build_non_spec, test_one_token_first_chunk_excludes_padding, test_one_token_chunk_classification, touching `test_aligned_block_table_matches_shared_gdn, _build_non_spec, test_one_token_first_chunk_excludes_padding`; `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +25/-2 (27 lines); hunks: -427,10 +427,33 @@ def build( # type: ignore[override]; symbols: build, touching `build`.
- Code diff details:
  - `tests/models/kimi_k3/test_kda_metadata.py` modified +88/-0 (88 lines); hunks: -934,3 +934,91 @@ def test_aligned_block_table_matches_shared_gdn():; symbols: test_aligned_block_table_matches_shared_gdn, _build_non_spec, test_one_token_first_chunk_excludes_padding, test_one_token_chunk_classification
  - `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +25/-2 (27 lines); hunks: -427,10 +427,33 @@ def build( # type: ignore[override]; symbols: build
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_kda_metadata.py
@@ -934,3 +934,91 @@ def test_aligned_block_table_matches_shared_gdn():
+def _build_non_spec(batch, is_prefilling, full_cuda_graph=False):
+    common_attn_metadata = create_common_attn_metadata(
+        batch, BLOCK_SIZE, DEVICE
+    ).replace(
+        is_prefilling=None
+        if is_prefilling is None
diff -- vllm/models/kimi_k3/nvidia/kda_metadata.py
@@ -427,10 +427,33 @@ def build(  # type: ignore[override]
-            # The runner orders ordinary decodes before prefills.
+            # V2 already excludes prefills from full decode graphs via has_prefill.
+            # Classify first chunks as prefills to mask recycled state;
+            # resumed one-token chunks can still use the decode kernels.
+            assert m.seq_lens_cpu_upper_bound is not None
+            query_lens_cpu = query_start_loc_cpu.diff()
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_kda_metadata.py` modified +88/-0
  - runtime: `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +25/-2
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda_metadata.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57098 - [Kimi K3 Bug] Fix kimi k3 reasoning parser

- Link: https://github.com/vllm-project/vllm/pull/57098
- Status/date: merged / 2026-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/reasoning/kimi_k3_reasoning_parser.py`; associated commits `f3aa88d23095`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +10/-5, 34 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/reasoning/kimi_k3_reasoning_parser.py` modified +10/-5 (15 lines); hunks: -256,11 +256,13 @@ def extract_reasoning(; -270,11 +272,14 @@ def extract_reasoning(; symbols: extract_reasoning, touching `extract_reasoning`.
- Code diff details:
  - `vllm/reasoning/kimi_k3_reasoning_parser.py` modified +10/-5 (15 lines); hunks: -256,11 +256,13 @@ def extract_reasoning(; -270,11 +272,14 @@ def extract_reasoning(; symbols: extract_reasoning
- Key code excerpts:

```diff
diff -- vllm/reasoning/kimi_k3_reasoning_parser.py
@@ -256,11 +256,13 @@ def extract_reasoning(
-        Handles three shapes:
-          * no think channel at all   -> ``(None, model_output)`` (all content)
+        Handles four shapes:
+          * response opener without think markers -> no think channel; all content
+          * neither marker present     -> truncated reasoning after a consumed
+            generation prefix
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/reasoning/kimi_k3_reasoning_parser.py` modified +10/-5
- Risk and verification: Runtime changes concentrate in `vllm/reasoning/kimi_k3_reasoning_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #57430 - [Quantization] Support kimi-k3 routed expert quant

- Link: https://github.com/vllm-project/vllm/pull/57430
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/model.py`; associated commits `76d517fd1279`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/model.py` modified +2/-2 (4 lines); hunks: -677,7 +677,7 @@ def __init__(; -694,7 +694,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/model.py` modified +2/-2 (4 lines); hunks: -677,7 +677,7 @@ def __init__(; -694,7 +694,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -677,7 +677,7 @@ def __init__(
-                quant_config=None,
+                quant_config=quant_config,
@@ -694,7 +694,7 @@ def __init__(
-                quant_config=None,
+                quant_config=quant_config,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50592 - [Kimi-K3][AMD] Return KDA and MLA projection outputs directly

- Link: https://github.com/vllm-project/vllm/pull/50592
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_amd_kda_direct_return.py`, `tests/models/kimi_k3/test_amd_mla_direct_return.py`, `tests/models/kimi_k3/test_nvidia_kda_direct_return.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `vllm/models/kimi_k3/amd/kda.py` and 7 files; associated commits `382970ee6ca4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +170/-33, 337 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_amd_kda_direct_return.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: _ReturningAttention, __init__, forward, _TupleProjection, touching `_ReturningAttention, __init__, forward`; `vllm/models/kimi_k3/amd/linear.py` modified +25/-16 (41 lines); hunks: -456,9 +456,8 @@ def forward(; -582,25 +581,29 @@ def _run_self_attn(; symbols: forward, KimiDecoderLayer, _run_self_attn, forward_attn_residual, touching `forward, KimiDecoderLayer, _run_self_attn`; `tests/models/kimi_k3/test_amd_mla_direct_return.py` added +38/-0 (38 lines); hunks: -0,0 +1,38; symbols: _DirectMLA, __init__, forward, _make_layer, touching `_DirectMLA, __init__, forward`; `tests/models/kimi_k3/test_nvidia_kda_direct_return.py` added +31/-0 (31 lines); hunks: -0,0 +1,31; symbols: _ReturningSharedKDA, __init__, forward, test_low_rank_kda_caller_returns_projection_storage_directly, touching `_ReturningSharedKDA, __init__, forward`.
- Code diff details:
  - `tests/models/kimi_k3/test_amd_kda_direct_return.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: _ReturningAttention, __init__, forward, _TupleProjection
  - `vllm/models/kimi_k3/amd/linear.py` modified +25/-16 (41 lines); hunks: -456,9 +456,8 @@ def forward(; -582,25 +581,29 @@ def _run_self_attn(; symbols: forward, KimiDecoderLayer, _run_self_attn, forward_attn_residual
  - `tests/models/kimi_k3/test_amd_mla_direct_return.py` added +38/-0 (38 lines); hunks: -0,0 +1,38; symbols: _DirectMLA, __init__, forward, _make_layer
  - `tests/models/kimi_k3/test_nvidia_kda_direct_return.py` added +31/-0 (31 lines); hunks: -0,0 +1,31; symbols: _ReturningSharedKDA, __init__, forward, test_low_rank_kda_caller_returns_projection_storage_directly
  - `vllm/models/kimi_k3/nvidia/model.py` modified +0/-11 (11 lines); hunks: -922,14 +922,12 @@ def __init__(; -959,7 +957,6 @@ def __init__(; symbols: __init__, _run_self_attn
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_amd_kda_direct_return.py
@@ -0,0 +1,72 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from types import SimpleNamespace
+import torch
+from torch import nn
+from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
diff -- vllm/models/kimi_k3/amd/linear.py
@@ -456,9 +456,8 @@ def forward(
-        output: torch.Tensor,
-    ) -> None:
-        output[:] = self.mla_attn(positions, hidden_states)
+    ) -> torch.Tensor:
+        return self.mla_attn(positions, hidden_states)
@@ -582,25 +581,29 @@ def _run_self_attn(
diff -- tests/models/kimi_k3/test_amd_mla_direct_return.py
@@ -0,0 +1,38 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_amd_kda_direct_return.py` added +72/-0; `tests/models/kimi_k3/test_amd_mla_direct_return.py` added +38/-0; `tests/models/kimi_k3/test_nvidia_kda_direct_return.py` added +31/-0
  - runtime: `vllm/models/kimi_k3/amd/linear.py` modified +25/-16; `vllm/models/kimi_k3/nvidia/model.py` modified +0/-11; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` modified +2/-3; `vllm/models/kimi_k3/amd/kda.py` modified +2/-3
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_amd_kda_direct_return.py`, `tests/models/kimi_k3/test_amd_mla_direct_return.py`, `tests/models/kimi_k3/test_nvidia_kda_direct_return.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52988 - [Spec decode] Support variable-length decode for Kimi-K3 adaptive ver

- Link: https://github.com/vllm-project/vllm/pull/52988
- Status/date: merged / 2026-09-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/dspark_mla.py`, `vllm/models/kimi_k3/nvidia/kda_metadata.py`; associated commits `88aa0d287dd3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 13 files, +203/-22, 510 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +29/-1 (30 lines); hunks: -15,7 +15,10; -188,6 +191,18 @@ def __init__(; symbols: __init__, mapper, markov_embed, markov_bias, touching `__init__, mapper, markov_embed`; `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +17/-1 (18 lines); hunks: -25,7 +25,7; -305,6 +305,10 @@ def commit_recoverssm_state(; symbols: commit_recoverssm_state, KimiK3KDAMetadataBuilder, __init__, build, touching `commit_recoverssm_state, KimiK3KDAMetadataBuilder, __init__`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +29/-1 (30 lines); hunks: -15,7 +15,10; -188,6 +191,18 @@ def __init__(; symbols: __init__, mapper, markov_embed, markov_bias
  - `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +17/-1 (18 lines); hunks: -25,7 +25,7; -305,6 +305,10 @@ def commit_recoverssm_state(; symbols: commit_recoverssm_state, KimiK3KDAMetadataBuilder, __init__, build
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/dspark_mla.py
@@ -15,7 +15,10 @@
-from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
+from vllm.model_executor.models.qwen3_dspark import (
+    DSparkConfidenceHead,
+    DSparkMarkovHead,
+)
@@ -188,6 +191,18 @@ def __init__(
diff -- vllm/models/kimi_k3/nvidia/kda_metadata.py
@@ -25,7 +25,7 @@
-from vllm.v1.attention.backend import CommonAttentionMetadata
+from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
@@ -305,6 +305,10 @@ def commit_recoverssm_state(
+    # Overrides GDN's UNIFORM_BATCH: adaptive verification requires ALWAYS from every
+    # builder, and KDA reads per-request offsets off device within a fixed k+1 window,
+    # so one k+1 graph replays any 1..k+1 mix.
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +29/-1; `vllm/models/kimi_k3/nvidia/kda_metadata.py` modified +17/-1
- Risk and verification: The diff ships test coverage in `tests/v1/attention/test_flashinfer_mla_dcp.py`, `tests/v1/attention/test_mla_backends.py`, `tests/v1/attention/test_rocm_aiter_mla_mtp_split.py`, `tests/v1/worker/test_mamba_hybrid_model_state.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58012 - [CI][ROCm] Add an MI355 Kimi-K3 unit test group

- Link: https://github.com/vllm-project/vllm/pull/58012
- Status/date: merged / 2026-09-23
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_attn_res.py`; associated commits `c4cfd7007bad`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +102/-0, 158 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_attn_res.py` modified +23/-0 (23 lines); hunks: -17,6 +17,26; -79,6 +99,7 @@ def test_attn_res(; symbols: _on_rocm_below_10, _skip_on_old_rocm, _randn_with_row_padding, test_attn_res, touching `_on_rocm_below_10, _skip_on_old_rocm, _randn_with_row_padding`; `vllm/platforms/rocm.py` modified +20/-0 (20 lines); hunks: -1,6 +1,7; -361,6 +362,25 @@ def get_cdna_version() -> int:; symbols: get_cdna_version, get_rocm_version, touching `get_cdna_version, get_rocm_version`.
- Code diff details:
  - `tests/models/kimi_k3/test_attn_res.py` modified +23/-0 (23 lines); hunks: -17,6 +17,26; -79,6 +99,7 @@ def test_attn_res(; symbols: _on_rocm_below_10, _skip_on_old_rocm, _randn_with_row_padding, test_attn_res
  - `vllm/platforms/rocm.py` modified +20/-0 (20 lines); hunks: -1,6 +1,7; -361,6 +362,25 @@ def get_cdna_version() -> int:; symbols: get_cdna_version, get_rocm_version
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_attn_res.py
@@ -17,6 +17,26 @@
+def _on_rocm_below_10() -> bool:
+    if not current_platform.is_rocm():
+        return False
+    from vllm.platforms.rocm import get_rocm_version
+    return (get_rocm_version() or (0,)) < (10,)
+# The Triton bundled with ROCm < 10 (3.7.x) crashes in the AMD
diff -- vllm/platforms/rocm.py
@@ -1,6 +1,7 @@
+import importlib.metadata
@@ -361,6 +362,25 @@ def get_cdna_version() -> int:
+@cache
+def get_rocm_version() -> tuple[int, ...] | None:
+    """Return the installed ROCm release as (major, minor, patch), or None."""
+    # ROCm 10+ ships as the `rocm` pip SDK; older releases install to /opt/rocm.
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_attn_res.py` modified +23/-0
  - runtime: `vllm/platforms/rocm.py` modified +20/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_attn_res.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58527 - [Kimi-K3][Perf] Dispatch GEMM for vision patch embedder

- Link: https://github.com/vllm-project/vllm/pull/58527
- Status/date: merged / 2026-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `ade1056ee5af`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +30/-17, 80 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/kimi_k25_vit.py` modified +4/-16 (20 lines); hunks: -21,6 +21,7; -194,7 +195,7 @@ def __init__(; symbols: __init__, forward, _proj, Rope2DPosEmbRepeated, touching `__init__, forward, _proj`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +4/-16 (20 lines); hunks: -21,6 +21,7; -194,7 +195,7 @@ def __init__(; symbols: __init__, forward, _proj, Rope2DPosEmbRepeated
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -21,6 +21,7 @@
+from vllm.model_executor.layers.conv import Conv2dLayer
@@ -194,7 +195,7 @@ def __init__(
-        self.proj = nn.Conv2d(
+        self.proj = Conv2dLayer(
@@ -220,26 +221,13 @@ def forward(
-        x = self._proj(x).view(x.size(0), -1)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` modified +4/-16
- Risk and verification: The diff ships test coverage in `tests/model_executor/layers/test_conv.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58045 - [ROCm][Kimi-K3] Optimize low-concurrency speculative KDA

- Link: https://github.com/vllm-project/vllm/pull/58045
- Status/date: merged / 2026-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_kda.py`, `tests/models/kimi_k3/test_kda_metadata.py`, `vllm/models/kimi_k3/amd/kda.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/__init__.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py`; associated commits `1417022c3dbb`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +201/-27, 479 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py` modified +103/-15 (118 lines); hunks: -164,15 +164,24 @@ def fused_recurrent_kda_fwd_kernel(; -208,6 +217,14 @@ def fused_recurrent_kda_fwd_kernel(; symbols: fused_recurrent_kda_fwd_kernel, touching `fused_recurrent_kda_fwd_kernel`; `tests/models/kimi_k3/test_kda.py` modified +56/-9 (65 lines); hunks: -22,6 +22,9; -44,10 +47,12; symbols: test_kda_warmup_skips_missing_metadata, test_packed_kda_decode_correctness, test_kda_spec_decode_correctness, touching `test_kda_warmup_skips_missing_metadata, test_packed_kda_decode_correctness, test_kda_spec_decode_correctness`; `tests/models/kimi_k3/test_kda_metadata.py` modified +19/-0 (19 lines); hunks: -49,6 +49,7; -409,6 +410,7 @@ def test_internal_checkpoint_metadata_skips_unaligned_offset():; symbols: test_internal_checkpoint_metadata_skips_unaligned_offset, touching `test_internal_checkpoint_metadata_skips_unaligned_offset`; `vllm/models/kimi_k3/amd/kda.py` modified +9/-0 (9 lines); hunks: -438,6 +438,14 @@ def _forward(; -457,6 +465,7 @@ def _forward(; symbols: _forward, touching `_forward`.
- Code diff details:
  - `vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py` modified +103/-15 (118 lines); hunks: -164,15 +164,24 @@ def fused_recurrent_kda_fwd_kernel(; -208,6 +217,14 @@ def fused_recurrent_kda_fwd_kernel(; symbols: fused_recurrent_kda_fwd_kernel
  - `tests/models/kimi_k3/test_kda.py` modified +56/-9 (65 lines); hunks: -22,6 +22,9; -44,10 +47,12; symbols: test_kda_warmup_skips_missing_metadata, test_packed_kda_decode_correctness, test_kda_spec_decode_correctness
  - `tests/models/kimi_k3/test_kda_metadata.py` modified +19/-0 (19 lines); hunks: -49,6 +49,7; -409,6 +410,7 @@ def test_internal_checkpoint_metadata_skips_unaligned_offset():; symbols: test_internal_checkpoint_metadata_skips_unaligned_offset
  - `vllm/models/kimi_k3/amd/kda.py` modified +9/-0 (9 lines); hunks: -438,6 +438,14 @@ def _forward(; -457,6 +465,7 @@ def _forward(; symbols: _forward
  - `vllm/models/kimi_k3/amd/ops/third_party/kda/__init__.py` modified +4/-3 (7 lines); hunks: -16,9 +16,10
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py
@@ -164,15 +164,24 @@ def fused_recurrent_kda_fwd_kernel(
+    IS_SINGLE_SEQUENCE: tl.constexpr,
+    SEQUENCE_LENGTH: tl.constexpr,
-    i_n, i_h = i_nh // H, i_nh % H
-    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
-    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
-    sequence_length = eos - bos
diff -- tests/models/kimi_k3/test_kda.py
@@ -22,6 +22,9 @@
+from vllm.models.kimi_k3.amd.ops.third_party.kda import (
+    fused_recurrent_kda as fused_recurrent_kda_amd,
+)
@@ -44,10 +47,12 @@
-    fused_recurrent_kda,
+from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
diff -- tests/models/kimi_k3/test_kda_metadata.py
@@ -49,6 +49,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py` modified +103/-15; `vllm/models/kimi_k3/amd/kda.py` modified +9/-0; `vllm/models/kimi_k3/amd/ops/third_party/kda/__init__.py` modified +4/-3
  - tests: `tests/models/kimi_k3/test_kda.py` modified +56/-9; `tests/models/kimi_k3/test_kda_metadata.py` modified +19/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda.py`, `tests/models/kimi_k3/test_kda_metadata.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58372 - [Bugfix][Reasoning] Count Kimi K3 reasoning tokens

- Link: https://github.com/vllm-project/vllm/pull/58372
- Status/date: merged / 2026-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_kimi_k3_reasoning_parser.py`, `vllm/reasoning/kimi_k3_reasoning_parser.py`; associated commits `9bf44c40134c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +91/-4, 130 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/reasoning/test_kimi_k3_reasoning_parser.py` modified +51/-4 (55 lines); hunks: -19,6 +19,9; -119,6 +122,54 @@ def test_is_reasoning_end_ignores_stale_close_from_prior_tu...; symbols: DummyTokenizer, test_is_reasoning_end_ignores_stale_close_from_prior_turn, test_count_reasoning_tokens_matches_think_channel, test_count_reasoning_tokens_is_zero_when_thinking_disabled, touching `DummyTokenizer, test_is_reasoning_end_ignores_stale_close_from_prior_turn, test_count_reasoning_tokens_matches_think_channel`; `vllm/reasoning/kimi_k3_reasoning_parser.py` modified +40/-0 (40 lines); hunks: -134,6 +134,9 @@ def __init__(self, tokenizer: PreTrainedTokenizerBase, *args...; -211,6 +214,43 @@ def extract_content_ids(self, input_ids: list[int]) -> list...; symbols: __init__, extract_content_ids, count_reasoning_tokens, _strip_content_wrapper, touching `__init__, extract_content_ids, count_reasoning_tokens`.
- Code diff details:
  - `tests/reasoning/test_kimi_k3_reasoning_parser.py` modified +51/-4 (55 lines); hunks: -19,6 +19,9; -119,6 +122,54 @@ def test_is_reasoning_end_ignores_stale_close_from_prior_tu...; symbols: DummyTokenizer, test_is_reasoning_end_ignores_stale_close_from_prior_turn, test_count_reasoning_tokens_matches_think_channel, test_count_reasoning_tokens_is_zero_when_thinking_disabled
  - `vllm/reasoning/kimi_k3_reasoning_parser.py` modified +40/-0 (40 lines); hunks: -134,6 +134,9 @@ def __init__(self, tokenizer: PreTrainedTokenizerBase, *args...; -211,6 +214,43 @@ def extract_content_ids(self, input_ids: list[int]) -> list...; symbols: __init__, extract_content_ids, count_reasoning_tokens, _strip_content_wrapper
- Key code excerpts:

```diff
diff -- tests/reasoning/test_kimi_k3_reasoning_parser.py
@@ -19,6 +19,9 @@
+OPEN_IDS = [1, 2, 3]
+CLOSE_IDS = [4, 2, 3]
+RESPONSE_OPEN_IDS = [ord(ch) for ch in RESPONSE_OPEN]
@@ -119,6 +122,54 @@ def test_is_reasoning_end_ignores_stale_close_from_prior_turn():
+@pytest.mark.parametrize(
+    ("token_ids", "expected"),
diff -- vllm/reasoning/kimi_k3_reasoning_parser.py
@@ -134,6 +134,9 @@ def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
+        self._response_open_ids = tokenizer.encode(
+            self._response_open, add_special_tokens=False
+        )
@@ -211,6 +214,43 @@ def extract_content_ids(self, input_ids: list[int]) -> list[int]:
+    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
+        if not self._thinking_enabled:
```

- Extracted files (not manually reviewed):
  - tests: `tests/reasoning/test_kimi_k3_reasoning_parser.py` modified +51/-4
  - runtime: `vllm/reasoning/kimi_k3_reasoning_parser.py` modified +40/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_kimi_k3_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58651 - [KimiViT][Perf] Fuse per-layer QK RoPE into one in-place kernel

- Link: https://github.com/vllm-project/vllm/pull/58651
- Status/date: merged / 2026-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `39d49539de86`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +140/-7, 174 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/kimi_k25_vit.py` modified +16/-7 (23 lines); hunks: -30,13 +30,17; -451,14 +455,19 @@ def attention_qkvpacked(; symbols: attention_qkvpacked, touching `attention_qkvpacked`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +16/-7 (23 lines); hunks: -30,13 +30,17; -451,14 +455,19 @@ def attention_qkvpacked(; symbols: attention_qkvpacked
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -30,13 +30,17 @@
+from vllm.model_executor.layers.rotary_embedding.packed_qk_rope import (
+    packed_qk_rope_,
+)
+from vllm.triton_utils import HAS_TRITON
@@ -451,14 +455,19 @@ def attention_qkvpacked(
-        xq, xk, xv = torch.unbind(xqkv, dim=-3)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` modified +16/-7
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_apply_rotary_emb.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #54956 - [ROCm][Perf] Kimi-K3 Enable sharded latent MoE up-projection under EP

- Link: https://github.com/vllm-project/vllm/pull/54956
- Status/date: merged / 2026-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_amd_latent_moe_runner.py`, `vllm/models/kimi_k3/amd/latent_moe_runner.py`; associated commits `a7ff44355c1c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +97/-16, 237 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_amd_latent_moe_runner.py` modified +95/-15 (110 lines); hunks: -23,6 +23,7; -56,21 +57,26 @@ def _build_transform(device: torch.device) -> KimiRoutedOutp...; symbols: _build_transform, _tail_runner, method, _rank_partials, touching `_build_transform, _tail_runner, method`; `vllm/models/kimi_k3/amd/latent_moe_runner.py` modified +2/-1 (3 lines); hunks: -6,6 +6,7; -33,7 +34,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tests/models/kimi_k3/test_amd_latent_moe_runner.py` modified +95/-15 (110 lines); hunks: -23,6 +23,7; -56,21 +57,26 @@ def _build_transform(device: torch.device) -> KimiRoutedOutp...; symbols: _build_transform, _tail_runner, method, _rank_partials
  - `vllm/models/kimi_k3/amd/latent_moe_runner.py` modified +2/-1 (3 lines); hunks: -6,6 +6,7; -33,7 +34,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_amd_latent_moe_runner.py
@@ -23,6 +23,7 @@
+from vllm.models.kimi_k3.amd import latent_moe_runner
@@ -56,21 +57,26 @@ def _build_transform(device: torch.device) -> KimiRoutedOutputTransform:
-    transform: KimiRoutedOutputTransform, tp_size: int
+    transform: KimiRoutedOutputTransform, tp_world: int, use_ep: bool = False
+    ``tp_world`` is the size of the TP process group, which is what the hidden
+    dim is split by. Under expert parallelism the MoE config reports
diff -- vllm/models/kimi_k3/amd/latent_moe_runner.py
@@ -6,6 +6,7 @@
+    get_tensor_model_parallel_world_size,
@@ -33,7 +34,7 @@ def __init__(
-        tp_size = self.moe_config.tp_size
+        tp_size = get_tensor_model_parallel_world_size()
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_amd_latent_moe_runner.py` modified +95/-15
  - runtime: `vllm/models/kimi_k3/amd/latent_moe_runner.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_amd_latent_moe_runner.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58814 - [Bugfix][Kimi-K3] Refresh DSpark context KV cache pointers after the KV cache is re-bound

- Link: https://github.com/vllm-project/vllm/pull/58814
- Status/date: merged / 2026-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/nvidia/dspark_mla.py`; associated commits `94d146292479`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +6/-1, 14 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +6/-1 (7 lines); hunks: -376,7 +376,12 @@ def _has_uniform_block_layout(; symbols: _has_uniform_block_layout, touching `_has_uniform_block_layout`.
- Code diff details:
  - `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +6/-1 (7 lines); hunks: -376,7 +376,12 @@ def _has_uniform_block_layout(; symbols: _has_uniform_block_layout
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/nvidia/dspark_mla.py
@@ -376,7 +376,12 @@ def _has_uniform_block_layout(
-        if not hasattr(self, "_layers_share_kv_block_layout"):
+        key = tuple(cl.kv_cache.data_ptr() for cl in cache_layers)
+        if getattr(self, "_kv_cache_ptrs_key", None) != key:
+            assert not torch.cuda.is_current_stream_capturing()
+            self._kv_cache_ptrs_key = key
+            if hasattr(self, "_context_cache_ptrs"):
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/nvidia/dspark_mla.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/nvidia/dspark_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #57640 - [ROCm][Kimi-K3][Perf] Fuse MLA decode KV-cache write and Q-prep via AITER

- Link: https://github.com/vllm-project/vllm/pull/57640
- Status/date: merged / 2026-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/amd/mla.py`; associated commits `8f24ab30a36d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +170/-8, 208 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/amd/mla.py` modified +170/-8 (178 lines); hunks: -7,16 +7,47; -38,6 +69,107 @@ def _normalize_q_kv(; symbols: KimiK3MultiHeadLatentAttentionWrapper, __init__, _fused_qk_prep_supported, _normalize_q_kv, touching `KimiK3MultiHeadLatentAttentionWrapper, __init__, _fused_qk_prep_supported`.
- Code diff details:
  - `vllm/models/kimi_k3/amd/mla.py` modified +170/-8 (178 lines); hunks: -7,16 +7,47; -38,6 +69,107 @@ def _normalize_q_kv(; symbols: KimiK3MultiHeadLatentAttentionWrapper, __init__, _fused_qk_prep_supported, _normalize_q_kv
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/amd/mla.py
@@ -7,16 +7,47 @@
+from vllm.model_executor.layers.attention.attention import get_attention_context
+from vllm.platforms import current_platform
+_OPT_KV_LORA_RANK = 512
+_OPT_ROT_DIM = 64
+_OPT_MIN_SIZE = 2048
-    """Kimi-K3 MLA wrapper with eager AITER q/kv RMSNorm fusion."""
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/amd/mla.py` modified +170/-8
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/amd/mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #54255 - [Kimi-K3] Add FlashInfer speculative KDA backend

- Link: https://github.com/vllm-project/vllm/pull/54255
- Status/date: merged / 2026-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_kda.py`, `vllm/models/kimi_k3/nvidia/kda.py`; associated commits `766b4ae54c49`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +383/-9, 479 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_kda.py` modified +241/-0 (241 lines); hunks: -30,13 +30,17; -90,6 +94,25 @@ def test_kda_warmup_skips_missing_metadata(monkeypatch):; symbols: test_kda_warmup_skips_missing_metadata, test_resolve_kda_spec_decode_backend, test_kda_recoverssm_config_state_layout, test_fused_kda_decode_correctness, touching `test_kda_warmup_skips_missing_metadata, test_resolve_kda_spec_decode_backend, test_kda_recoverssm_config_state_layout`; `vllm/models/kimi_k3/nvidia/kda.py` modified +130/-9 (139 lines); hunks: -55,8 +55,10; -214,6 +216,73 @@ def is_flashinfer_fused_kda_decode_supported(; symbols: is_flashinfer_fused_kda_decode_supported, is_flashinfer_fused_kda_spec_decode_supported, resolve_kda_spec_decode_backend, resolve_kda_decode_backend, touching `is_flashinfer_fused_kda_decode_supported, is_flashinfer_fused_kda_spec_decode_supported, resolve_kda_spec_decode_backend`.
- Code diff details:
  - `tests/models/kimi_k3/test_kda.py` modified +241/-0 (241 lines); hunks: -30,13 +30,17; -90,6 +94,25 @@ def test_kda_warmup_skips_missing_metadata(monkeypatch):; symbols: test_kda_warmup_skips_missing_metadata, test_resolve_kda_spec_decode_backend, test_kda_recoverssm_config_state_layout, test_fused_kda_decode_correctness
  - `vllm/models/kimi_k3/nvidia/kda.py` modified +130/-9 (139 lines); hunks: -55,8 +55,10; -214,6 +216,73 @@ def is_flashinfer_fused_kda_decode_supported(; symbols: is_flashinfer_fused_kda_decode_supported, is_flashinfer_fused_kda_spec_decode_supported, resolve_kda_spec_decode_backend, resolve_kda_decode_backend
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_kda.py
@@ -30,13 +30,17 @@
+    KimiK3DeltaAttention,
+    is_flashinfer_fused_kda_spec_decode_supported,
+    resolve_kda_spec_decode_backend,
+from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata
@@ -90,6 +94,25 @@ def test_kda_warmup_skips_missing_metadata(monkeypatch):
+def test_resolve_kda_spec_decode_backend(monkeypatch: pytest.MonkeyPatch):
diff -- vllm/models/kimi_k3/nvidia/kda.py
@@ -55,8 +55,10 @@
+    flashinfer_packed_fused_kda_decode,
+    has_flashinfer_packed_fused_kda_decode,
@@ -214,6 +216,73 @@ def is_flashinfer_fused_kda_decode_supported(
+def is_flashinfer_fused_kda_spec_decode_supported(
+    num_heads: int,
+    head_dim: int,
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_kda.py` modified +241/-0
  - runtime: `vllm/models/kimi_k3/nvidia/kda.py` modified +130/-9
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_kda.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #59257 - [Bugfix] Gate Kimi-K3 KDA warmup on sys.modules to skip Kimi import for non-Kimi models

- Link: https://github.com/vllm-project/vllm/pull/59257
- Status/date: merged / 2026-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/warmup/kimi_k3_triton_warmup.py`; associated commits `08d77cad7be4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +6/-2, 29 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/warmup/kimi_k3_triton_warmup.py` modified +6/-2 (8 lines); hunks: -4,6 +4,7; -19,7 +20,10; symbols: _get_kda_layer, touching `_get_kda_layer`.
- Code diff details:
  - `vllm/model_executor/warmup/kimi_k3_triton_warmup.py` modified +6/-2 (8 lines); hunks: -4,6 +4,7; -19,7 +20,10; symbols: _get_kda_layer
- Key code excerpts:

```diff
diff -- vllm/model_executor/warmup/kimi_k3_triton_warmup.py
@@ -4,6 +4,7 @@
+import sys
@@ -19,7 +20,10 @@
-    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
+    # Kimi model construction already imports kda. Avoid importing it here.
+    kda = sys.modules.get("vllm.models.kimi_k3.nvidia.kda")
+    if kda is None:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/warmup/kimi_k3_triton_warmup.py` modified +6/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/warmup/kimi_k3_triton_warmup.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #58769 - [ROCm][Triton] Migrate Kimi-K3 kernels from make_block_ptr to tensor …

- Link: https://github.com/vllm-project/vllm/pull/58769
- Status/date: merged / 2026-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra_token_parallel.py`; associated commits `d848c4ed3d0c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +112/-351, 655 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py` modified +59/-165 (224 lines); hunks: -12,7 +12,11; -125,33 +129,21 @@ def chunk_kda_fwd_kernel_inter_solve_fused(; symbols: chunk_kda_fwd_kernel_inter_solve_fused, touching `chunk_kda_fwd_kernel_inter_solve_fused`; `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` modified +47/-175 (222 lines); hunks: -17,7 +17,11; -90,86 +94,35 @@ def recompute_w_u_fwd_kernel(; symbols: recompute_w_u_fwd_kernel, recompute_w_u_fwd, chunk_gla_fwd_kernel_o, chunk_gla_fwd_o_gk, touching `recompute_w_u_fwd_kernel, recompute_w_u_fwd, chunk_gla_fwd_kernel_o`; `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra_token_parallel.py` modified +6/-11 (17 lines); hunks: -12,7 +12,7; -108,20 +108,15 @@ def chunk_kda_fwd_kernel_intra_token_parallel(; symbols: chunk_kda_fwd_kernel_intra_token_parallel, touching `chunk_kda_fwd_kernel_intra_token_parallel`.
- Code diff details:
  - `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py` modified +59/-165 (224 lines); hunks: -12,7 +12,11; -125,33 +129,21 @@ def chunk_kda_fwd_kernel_inter_solve_fused(; symbols: chunk_kda_fwd_kernel_inter_solve_fused
  - `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` modified +47/-175 (222 lines); hunks: -17,7 +17,11; -90,86 +94,35 @@ def recompute_w_u_fwd_kernel(; symbols: recompute_w_u_fwd_kernel, recompute_w_u_fwd, chunk_gla_fwd_kernel_o, chunk_gla_fwd_o_gk
  - `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra_token_parallel.py` modified +6/-11 (17 lines); hunks: -12,7 +12,7; -108,20 +108,15 @@ def chunk_kda_fwd_kernel_intra_token_parallel(; symbols: chunk_kda_fwd_kernel_intra_token_parallel
- Key code excerpts:

```diff
diff -- vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py
@@ -12,7 +12,11 @@
-from vllm.third_party.flash_linear_attention.ops.op import exp2, gather
+from vllm.third_party.flash_linear_attention.ops.op import (
+    exp2,
+    gather,
+    make_tensor_descriptor,
+)
diff -- vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py
@@ -17,7 +17,11 @@
-from vllm.third_party.flash_linear_attention.ops.op import exp2, log
+from vllm.third_party.flash_linear_attention.ops.op import (
+    exp2,
+    log,
+    make_tensor_descriptor,
+)
diff -- vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra_token_parallel.py
@@ -12,7 +12,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py` modified +59/-165; `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` modified +47/-175; `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra_token_parallel.py` modified +6/-11
- Risk and verification: Runtime changes concentrate in `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py`, `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra_token_parallel.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #58344 - [ROCm][Perf] Kimi-K3 enable prefill checkpoints on ROCm

- Link: https://github.com/vllm-project/vllm/pull/58344
- Status/date: merged / 2026-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_amd_kda_checkpoint.py`, `tests/models/kimi_k3/test_amd_kda_chunk.py`, `vllm/models/kimi_k3/amd/kda.py`, `vllm/models/kimi_k3/amd/kda_metadata.py`, `vllm/models/kimi_k3/amd/ops/kda_checkpoint.py` and 7 files; associated commits `01549796563f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +748/-14, 900 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_amd_kda_checkpoint.py` added +521/-0 (521 lines); hunks: -0,0 +1,521; symbols: _StubSpeculativeConfig, __init__, use_eagle_block_drop, _builder, touching `_StubSpeculativeConfig, __init__, use_eagle_block_drop`; `tests/models/kimi_k3/test_amd_kda_chunk.py` modified +56/-6 (62 lines); hunks: -13,17 +13,17; -701,6 +701,56 @@ def test_cold_rows_start_from_zero_not_from_cache_junk(use_...; symbols: _on_gfx950, _fused_chunk_available, test_cold_rows_start_from_zero_not_from_cache_junk, test_state_indices_accept_a_block_table_column, touching `_on_gfx950, _fused_chunk_available, test_cold_rows_start_from_zero_not_from_cache_junk`; `vllm/models/kimi_k3/amd/kda.py` modified +56/-6 (62 lines); hunks: -1,6 +1,8; -38,7 +40,11; symbols: get_state_shape, get_kv_cache_spec, __init__, _prefill_conv, touching `get_state_shape, get_kv_cache_spec, __init__`; `vllm/models/kimi_k3/amd/kda_metadata.py` modified +56/-0 (56 lines); hunks: -7,16 +7,23; -86,6 +93,55 @@ def prepare_chunk_metadata_device(; symbols: prepare_chunk_metadata_device, KimiK3ROCmKDAMetadataBuilder, build, _build_checkpoint_metadata, touching `prepare_chunk_metadata_device, KimiK3ROCmKDAMetadataBuilder, build`.
- Code diff details:
  - `tests/models/kimi_k3/test_amd_kda_checkpoint.py` added +521/-0 (521 lines); hunks: -0,0 +1,521; symbols: _StubSpeculativeConfig, __init__, use_eagle_block_drop, _builder
  - `tests/models/kimi_k3/test_amd_kda_chunk.py` modified +56/-6 (62 lines); hunks: -13,17 +13,17; -701,6 +701,56 @@ def test_cold_rows_start_from_zero_not_from_cache_junk(use_...; symbols: _on_gfx950, _fused_chunk_available, test_cold_rows_start_from_zero_not_from_cache_junk, test_state_indices_accept_a_block_table_column
  - `vllm/models/kimi_k3/amd/kda.py` modified +56/-6 (62 lines); hunks: -1,6 +1,8; -38,7 +40,11; symbols: get_state_shape, get_kv_cache_spec, __init__, _prefill_conv
  - `vllm/models/kimi_k3/amd/kda_metadata.py` modified +56/-0 (56 lines); hunks: -7,16 +7,23; -86,6 +93,55 @@ def prepare_chunk_metadata_device(; symbols: prepare_chunk_metadata_device, KimiK3ROCmKDAMetadataBuilder, build, _build_checkpoint_metadata
  - `vllm/models/kimi_k3/amd/ops/kda_checkpoint.py` added +41/-0 (41 lines); hunks: -0,0 +1,41; symbols: KimiK3ROCmKDAPrefillCheckpointExporter, export
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_amd_kda_checkpoint.py
@@ -0,0 +1,521 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""The ROCm KDA prefill checkpoint: where it is taken and what it stores.
+Two things have to line up for a checkpoint to be safe: the metadata must name
+a position the chunk kernel will actually write, and the conv window saved
+there must be the one a resumed prefill would have left.
diff -- tests/models/kimi_k3/test_amd_kda_chunk.py
@@ -13,17 +13,17 @@
-def _on_gfx950() -> bool:
-    if not current_platform.is_rocm():
+def _fused_chunk_available() -> bool:
+    if not (current_platform.is_rocm() and torch.cuda.is_available()):
-    from vllm.platforms.rocm import on_gfx950
+    from vllm.models.kimi_k3.amd.ops.kda_chunk import is_fused_kda_chunk_supported
diff -- vllm/models/kimi_k3/amd/kda.py
@@ -1,6 +1,8 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_amd_kda_checkpoint.py` added +521/-0; `tests/models/kimi_k3/test_amd_kda_chunk.py` modified +56/-6
  - runtime: `vllm/models/kimi_k3/amd/kda.py` modified +56/-6; `vllm/models/kimi_k3/amd/kda_metadata.py` modified +56/-0; `vllm/models/kimi_k3/amd/ops/kda_checkpoint.py` added +41/-0; `vllm/models/kimi_k3/amd/ops/kda_chunk.py` modified +11/-1; `vllm/models/kimi_k3/amd/ops/kda_prefill.py` modified +7/-1
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_amd_kda_checkpoint.py`, `tests/models/kimi_k3/test_amd_kda_chunk.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #59229 - [CI][Kimi-K3] Test prefix cache reuse with KV offload, P/D and DCP

- Link: https://github.com/vllm-project/vllm/pull/59229
- Status/date: merged / 2026-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_prefix_cache.py`; associated commits `d648847fe451`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +396/-1, 412 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_prefix_cache.py` added +375/-0 (375 lines); hunks: -0,0 +1,375; symbols: Mode, name, Instance, num_gpus, touching `Mode, name, Instance`.
- Code diff details:
  - `tests/models/kimi_k3/test_prefix_cache.py` added +375/-0 (375 lines); hunks: -0,0 +1,375; symbols: Mode, name, Instance, num_gpus
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_prefix_cache.py
@@ -0,0 +1,375 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Kimi-K3 multi-turn prefix cache reuse with KV offload, P/D and DCP."""
+import contextlib
+import json
+import random
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_prefix_cache.py` added +375/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_prefix_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51274 - [ROCm][Kimi-K3] Add opt-in gfx942 MXFP4-to-int4 conversion

- Link: https://github.com/vllm-project/vllm/pull/51274
- Status/date: merged / 2026-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/kimi_k3/test_gfx942_int4.py`; associated commits `b0e21b308352`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +346/-2, 434 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/kimi_k3/test_gfx942_int4.py` added +108/-0 (108 lines); hunks: -0,0 +1,108; symbols: _FakeRocmModule, _FakeAiterOpsModule, _vllm_config, test_moe_weight_override_is_int4, touching `_FakeRocmModule, _FakeAiterOpsModule, _vllm_config`; `vllm/model_executor/layers/quantization/mxfp4.py` modified +224/-2 (226 lines); hunks: -1,6 +1,7; -22,6 +23,7; symbols: apply_monolithic, _use_k3_situ_aiter, _use_k3_situ_int4_gfx942, _moe_weight_override_is_int4, touching `apply_monolithic, _use_k3_situ_aiter, _use_k3_situ_int4_gfx942`; `vllm/model_executor/layers/quantization/online/base.py` modified +4/-0 (4 lines); hunks: -65,6 +65,7; -231,6 +232,9 @@ def _get_method_cls(; symbols: _get_method_cls, touching `_get_method_cls`; `vllm/config/quantization.py` modified +3/-0 (3 lines); hunks: -22,6 +22,7; -40,6 +41,8.
- Code diff details:
  - `tests/models/kimi_k3/test_gfx942_int4.py` added +108/-0 (108 lines); hunks: -0,0 +1,108; symbols: _FakeRocmModule, _FakeAiterOpsModule, _vllm_config, test_moe_weight_override_is_int4
  - `vllm/model_executor/layers/quantization/mxfp4.py` modified +224/-2 (226 lines); hunks: -1,6 +1,7; -22,6 +23,7; symbols: apply_monolithic, _use_k3_situ_aiter, _use_k3_situ_int4_gfx942, _moe_weight_override_is_int4
  - `vllm/model_executor/layers/quantization/online/base.py` modified +4/-0 (4 lines); hunks: -65,6 +65,7; -231,6 +232,9 @@ def _get_method_cls(; symbols: _get_method_cls
  - `vllm/config/quantization.py` modified +3/-0 (3 lines); hunks: -22,6 +22,7; -40,6 +41,8
- Key code excerpts:

```diff
diff -- tests/models/kimi_k3/test_gfx942_int4.py
@@ -0,0 +1,108 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import sys
+from collections.abc import Callable
+from types import ModuleType, SimpleNamespace
+from unittest.mock import Mock, patch
diff -- vllm/model_executor/layers/quantization/mxfp4.py
@@ -1,6 +1,7 @@
+import os
@@ -22,6 +23,7 @@
+    backend_to_kernel_cls,
@@ -477,16 +479,97 @@ def apply_monolithic(
+def _use_k3_situ_aiter(moe: FusedMoEConfig) -> bool:
+    """Route Kimi-K3 weight-only MXFP4 SiTU MoE to AITER A16W4 on gfx950."""
diff -- vllm/model_executor/layers/quantization/online/base.py
@@ -65,6 +65,7 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/kimi_k3/test_gfx942_int4.py` added +108/-0
  - runtime: `vllm/model_executor/layers/quantization/mxfp4.py` modified +224/-2; `vllm/model_executor/layers/quantization/online/base.py` modified +4/-0; `vllm/config/quantization.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `tests/models/kimi_k3/test_gfx942_int4.py`, `tests/quantization/test_quantization_config_args.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
