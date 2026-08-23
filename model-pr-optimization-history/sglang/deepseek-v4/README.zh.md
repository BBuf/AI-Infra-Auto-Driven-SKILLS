# sglang DeepSeek V4 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` | [#34809](https://github.com/sgl-project/sglang/pull/34809), [#34926](https://github.com/sgl-project/sglang/pull/34926), [#35224](https://github.com/sgl-project/sglang/pull/35224), [#35918](https://github.com/sgl-project/sglang/pull/35918) |
| `docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` | [#34809](https://github.com/sgl-project/sglang/pull/34809) |
| `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` | [#34044](https://github.com/sgl-project/sglang/pull/34044), [#34333](https://github.com/sgl-project/sglang/pull/34333), [#34809](https://github.com/sgl-project/sglang/pull/34809), [#35224](https://github.com/sgl-project/sglang/pull/35224), [#35918](https://github.com/sgl-project/sglang/pull/35918) |
| `python/sglang/kernels/aot/csrc/elementwise/deepseek_v4_topk.cu` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c128.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c128_online.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c128_online_v2.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c128_v2.cuh` | [#26671](https://github.com/sgl-project/sglang/pull/26671), [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c4.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c4_v2.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630), [#34189](https://github.com/sgl-project/sglang/pull/34189) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/common.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630), [#34277](https://github.com/sgl-project/sglang/pull/34277) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630), [#31563](https://github.com/sgl-project/sglang/pull/31563) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/hash_topk.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/online_c128_mtp.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/paged_mqa_metadata.cuh` | [#25855](https://github.com/sgl-project/sglang/pull/25855), [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/rope.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/store.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v1.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630), [#32910](https://github.com/sgl-project/sglang/pull/32910) |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/compress.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/compress_v2.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp8_utils.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kvcacheio.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/ops/attention/deepseek_v4_rope.py` | [#29630](https://github.com/sgl-project/sglang/pull/29630), [#31931](https://github.com/sgl-project/sglang/pull/31931) |
| `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#25144](https://github.com/sgl-project/sglang/pull/25144), [#25771](https://github.com/sgl-project/sglang/pull/25771), [#25820](https://github.com/sgl-project/sglang/pull/25820), [#29569](https://github.com/sgl-project/sglang/pull/29569), [#29775](https://github.com/sgl-project/sglang/pull/29775), [#29982](https://github.com/sgl-project/sglang/pull/29982), [#30237](https://github.com/sgl-project/sglang/pull/30237), [#33532](https://github.com/sgl-project/sglang/pull/33532), [#34926](https://github.com/sgl-project/sglang/pull/34926) |
| `python/sglang/srt/configs/deepseek_v4.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882) |
| `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24692](https://github.com/sgl-project/sglang/pull/24692), [#24890](https://github.com/sgl-project/sglang/pull/24890), [#25195](https://github.com/sgl-project/sglang/pull/25195), [#26209](https://github.com/sgl-project/sglang/pull/26209), [#26239](https://github.com/sgl-project/sglang/pull/26239), [#26471](https://github.com/sgl-project/sglang/pull/26471), [#26499](https://github.com/sgl-project/sglang/pull/26499), [#27059](https://github.com/sgl-project/sglang/pull/27059), [#27380](https://github.com/sgl-project/sglang/pull/27380), [#27914](https://github.com/sgl-project/sglang/pull/27914), [#29619](https://github.com/sgl-project/sglang/pull/29619), ... (26 total) |
| `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` | [#24933](https://github.com/sgl-project/sglang/pull/24933), [#26208](https://github.com/sgl-project/sglang/pull/26208), [#26383](https://github.com/sgl-project/sglang/pull/26383), [#26499](https://github.com/sgl-project/sglang/pull/26499), [#27152](https://github.com/sgl-project/sglang/pull/27152), [#27380](https://github.com/sgl-project/sglang/pull/27380), [#27928](https://github.com/sgl-project/sglang/pull/27928), [#28520](https://github.com/sgl-project/sglang/pull/28520), [#28920](https://github.com/sgl-project/sglang/pull/28920), [#29362](https://github.com/sgl-project/sglang/pull/29362), [#29420](https://github.com/sgl-project/sglang/pull/29420), [#29630](https://github.com/sgl-project/sglang/pull/29630), ... (16 total) |
| `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24933](https://github.com/sgl-project/sglang/pull/24933), [#25144](https://github.com/sgl-project/sglang/pull/25144), [#26471](https://github.com/sgl-project/sglang/pull/26471), [#30333](https://github.com/sgl-project/sglang/pull/30333), [#31747](https://github.com/sgl-project/sglang/pull/31747), [#33676](https://github.com/sgl-project/sglang/pull/33676) |
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24691](https://github.com/sgl-project/sglang/pull/24691), [#24704](https://github.com/sgl-project/sglang/pull/24704), [#24890](https://github.com/sgl-project/sglang/pull/24890), [#24933](https://github.com/sgl-project/sglang/pull/24933), [#25144](https://github.com/sgl-project/sglang/pull/25144), [#25884](https://github.com/sgl-project/sglang/pull/25884), [#25889](https://github.com/sgl-project/sglang/pull/25889), [#25898](https://github.com/sgl-project/sglang/pull/25898), [#26208](https://github.com/sgl-project/sglang/pull/26208), [#26209](https://github.com/sgl-project/sglang/pull/26209), [#26471](https://github.com/sgl-project/sglang/pull/26471), ... (25 total) |
| `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` | [#26383](https://github.com/sgl-project/sglang/pull/26383), [#32577](https://github.com/sgl-project/sglang/pull/32577), [#34926](https://github.com/sgl-project/sglang/pull/34926) |
| `python/sglang/srt/models/deepseek_v4.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24704](https://github.com/sgl-project/sglang/pull/24704), [#24890](https://github.com/sgl-project/sglang/pull/24890), [#24933](https://github.com/sgl-project/sglang/pull/24933), [#24947](https://github.com/sgl-project/sglang/pull/24947), [#25144](https://github.com/sgl-project/sglang/pull/25144), [#25195](https://github.com/sgl-project/sglang/pull/25195), [#25391](https://github.com/sgl-project/sglang/pull/25391), [#25396](https://github.com/sgl-project/sglang/pull/25396), [#25729](https://github.com/sgl-project/sglang/pull/25729), [#25733](https://github.com/sgl-project/sglang/pull/25733), [#25763](https://github.com/sgl-project/sglang/pull/25763), ... (63 total) |
| `python/sglang/srt/models/deepseek_v4_dspark.py` | [#27657](https://github.com/sgl-project/sglang/pull/27657), [#29630](https://github.com/sgl-project/sglang/pull/29630), [#30240](https://github.com/sgl-project/sglang/pull/30240), [#30964](https://github.com/sgl-project/sglang/pull/30964), [#33312](https://github.com/sgl-project/sglang/pull/33312), [#33676](https://github.com/sgl-project/sglang/pull/33676), [#33865](https://github.com/sgl-project/sglang/pull/33865) |
| `python/sglang/srt/models/deepseek_v4_nextn.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24934](https://github.com/sgl-project/sglang/pull/24934), [#24947](https://github.com/sgl-project/sglang/pull/24947), [#25810](https://github.com/sgl-project/sglang/pull/25810), [#25976](https://github.com/sgl-project/sglang/pull/25976), [#26238](https://github.com/sgl-project/sglang/pull/26238), [#28980](https://github.com/sgl-project/sglang/pull/28980), [#31700](https://github.com/sgl-project/sglang/pull/31700), [#33532](https://github.com/sgl-project/sglang/pull/33532) |
| `python/sglang/test/kernels/deepseek_v4/__init__.py` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/test/kernels/deepseek_v4/common.py` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `test/registered/amd/test_deepseek_v4_flash_fp4.py` | [#24825](https://github.com/sgl-project/sglang/pull/24825), [#25039](https://github.com/sgl-project/sglang/pull/25039), [#26662](https://github.com/sgl-project/sglang/pull/26662), [#27149](https://github.com/sgl-project/sglang/pull/27149), [#28290](https://github.com/sgl-project/sglang/pull/28290), [#28920](https://github.com/sgl-project/sglang/pull/28920) |
| `test/registered/amd/test_deepseek_v4_flash_fp8.py` | [#24825](https://github.com/sgl-project/sglang/pull/24825), [#25039](https://github.com/sgl-project/sglang/pull/25039), [#26662](https://github.com/sgl-project/sglang/pull/26662), [#27149](https://github.com/sgl-project/sglang/pull/27149), [#28290](https://github.com/sgl-project/sglang/pull/28290), [#28920](https://github.com/sgl-project/sglang/pull/28920) |
| `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py` | [#29362](https://github.com/sgl-project/sglang/pull/29362), [#34926](https://github.com/sgl-project/sglang/pull/34926) |
| `test/registered/amd/test_deepseek_v4_pro_fp4.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24203](https://github.com/sgl-project/sglang/pull/24203), [#24825](https://github.com/sgl-project/sglang/pull/24825), [#25039](https://github.com/sgl-project/sglang/pull/25039), [#26662](https://github.com/sgl-project/sglang/pull/26662), [#27149](https://github.com/sgl-project/sglang/pull/27149), [#28290](https://github.com/sgl-project/sglang/pull/28290), [#28920](https://github.com/sgl-project/sglang/pull/28920) |
| `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` | [#27928](https://github.com/sgl-project/sglang/pull/27928), [#28920](https://github.com/sgl-project/sglang/pull/28920) |
| `test/registered/amd/test_deepseek_v4_pro_fp4_cp_tbo.py` | [#33480](https://github.com/sgl-project/sglang/pull/33480) |
| `test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py` | [#30964](https://github.com/sgl-project/sglang/pull/30964) |
| `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` | [#28520](https://github.com/sgl-project/sglang/pull/28520), [#28920](https://github.com/sgl-project/sglang/pull/28920) |
| `test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py` | [#29362](https://github.com/sgl-project/sglang/pull/29362) |
| `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py` | [#30238](https://github.com/sgl-project/sglang/pull/30238) |
| `test/registered/amd/test_deepseek_v4_pro_fp8.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24203](https://github.com/sgl-project/sglang/pull/24203), [#24825](https://github.com/sgl-project/sglang/pull/24825), [#25039](https://github.com/sgl-project/sglang/pull/25039), [#26662](https://github.com/sgl-project/sglang/pull/26662), [#27149](https://github.com/sgl-project/sglang/pull/27149), [#28290](https://github.com/sgl-project/sglang/pull/28290), [#28920](https://github.com/sgl-project/sglang/pull/28920) |
| `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` | [#25195](https://github.com/sgl-project/sglang/pull/25195), [#29775](https://github.com/sgl-project/sglang/pull/29775), [#29885](https://github.com/sgl-project/sglang/pull/29885), [#30365](https://github.com/sgl-project/sglang/pull/30365) |
| `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` | [#24947](https://github.com/sgl-project/sglang/pull/24947), [#26609](https://github.com/sgl-project/sglang/pull/26609), [#28098](https://github.com/sgl-project/sglang/pull/28098), [#29569](https://github.com/sgl-project/sglang/pull/29569), [#30898](https://github.com/sgl-project/sglang/pull/30898), [#33532](https://github.com/sgl-project/sglang/pull/33532), [#33865](https://github.com/sgl-project/sglang/pull/33865), [#35918](https://github.com/sgl-project/sglang/pull/35918) |
| `test/registered/gb300/test_deepseek_v4_pro_fp4_balanced.py` | 无直接 PR 号提交 |
| `test/registered/gb300/test_deepseek_v4_pro_fp4_high_throughput.py` | 无直接 PR 号提交 |
| `test/registered/gb300/test_deepseek_v4_pro_fp4_low_latency.py` | 无直接 PR 号提交 |
| `test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py` | [#34189](https://github.com/sgl-project/sglang/pull/34189) |
| `test/registered/kernels/ops/attention/test_deepseek_v4_compress_state_runtime_shapes.py` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` | [#25195](https://github.com/sgl-project/sglang/pull/25195), [#26141](https://github.com/sgl-project/sglang/pull/26141), [#26609](https://github.com/sgl-project/sglang/pull/26609), [#26766](https://github.com/sgl-project/sglang/pull/26766), [#28098](https://github.com/sgl-project/sglang/pull/28098), [#30898](https://github.com/sgl-project/sglang/pull/30898), [#31125](https://github.com/sgl-project/sglang/pull/31125), [#35919](https://github.com/sgl-project/sglang/pull/35919) |
| `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` | [#26141](https://github.com/sgl-project/sglang/pull/26141), [#26609](https://github.com/sgl-project/sglang/pull/26609), [#27867](https://github.com/sgl-project/sglang/pull/27867), [#28098](https://github.com/sgl-project/sglang/pull/28098) |
| `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` | [#26609](https://github.com/sgl-project/sglang/pull/26609), [#26766](https://github.com/sgl-project/sglang/pull/26766), [#28098](https://github.com/sgl-project/sglang/pull/28098), [#35918](https://github.com/sgl-project/sglang/pull/35918) |
| `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py` | [#26609](https://github.com/sgl-project/sglang/pull/26609), [#28098](https://github.com/sgl-project/sglang/pull/28098), [#29016](https://github.com/sgl-project/sglang/pull/29016), [#34926](https://github.com/sgl-project/sglang/pull/34926) |
| `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_1p1d_16p_in8k_out1k_50ms.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms.py` | [#35162](https://github.com/sgl-project/sglang/pull/35162) |
| `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py` | 无直接 PR 号提交 |
| `test/registered/unit/models/test_deepseek_v4_amd_fused_mhc.py` | [#32577](https://github.com/sgl-project/sglang/pull/32577) |
| `test/registered/unit/models/test_deepseek_v4_amd_wo_a_bf16.py` | [#33313](https://github.com/sgl-project/sglang/pull/33313) |
| `test/registered/unit/models/test_deepseek_v4_rope_policy.py` | [#34788](https://github.com/sgl-project/sglang/pull/34788) |
| `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py` | [#27349](https://github.com/sgl-project/sglang/pull/27349), [#33312](https://github.com/sgl-project/sglang/pull/33312) |

## PR 覆盖总览

- git 追溯 PR 数: 125
- 原文档显式引用补充 PR 数: 83
- 当前文档总 PR 数: 208
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#23605](https://github.com/sgl-project/sglang/pull/23605) | merged | Add DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx` |
| 2026-04-24 | [#23617](https://github.com/sgl-project/sglang/pull/23617) | merged | Further update Deepseek V4 docs | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-24 | [#23628](https://github.com/sgl-project/sglang/pull/23628) | merged | docs: note H200 DeepSeek-V4 checkpoint | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23622](https://github.com/sgl-project/sglang/pull/23622) | merged | Again update DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23634](https://github.com/sgl-project/sglang/pull/23634) | merged | Update pro fp8 checkpoint in DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23684](https://github.com/sgl-project/sglang/pull/23684) | merged | docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23689](https://github.com/sgl-project/sglang/pull/23689) | merged | docs(DeepSeek-V4): mark b200\|small\|pd-disagg + h200\|small\|{cp,pd-disagg} verified | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23691](https://github.com/sgl-project/sglang/pull/23691) | merged | docs(DeepSeek-V4): mark gb300\|{small,big}\|{cp,pd-disagg} verified + GB300-specific fixes | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23690](https://github.com/sgl-project/sglang/pull/23690) | merged | Small udpate gb300 recipe for deepseek v4 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23697](https://github.com/sgl-project/sglang/pull/23697) | merged | update: b300 container for dsv4 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23698](https://github.com/sgl-project/sglang/pull/23698) | merged | docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23715](https://github.com/sgl-project/sglang/pull/23715) | merged | docs(DeepSeek-V4): mark h200\|big\|pd-disagg verified + recipe fixes | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23728](https://github.com/sgl-project/sglang/pull/23728) | merged | ci: add docker release workflow for deepseek_v4 branch | `.github/workflows/release-docker-deepseek-v4.yml` |
| 2026-04-25 | [#23730](https://github.com/sgl-project/sglang/pull/23730) | merged | [CI] release-docker-deepseek-v4: select which flavors to push | `.github/workflows/release-docker-deepseek-v4.yml` |
| 2026-04-26 | [#23725](https://github.com/sgl-project/sglang/pull/23725) | merged | docs(DeepSeek-V4): add GB200 platform to cookbook recipe | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-26 | [#23742](https://github.com/sgl-project/sglang/pull/23742) | merged | docs(DeepSeek-V4): add h200\|big verified recipes + tune H200 Pro parameters | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-26 | [#23737](https://github.com/sgl-project/sglang/pull/23737) | merged | docs(DeepSeek-V4): mark gb200\|big\|low-latency verified | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-26 | [#23778](https://github.com/sgl-project/sglang/pull/23778) | merged | ci(deepseek-v4): add b300/grace-blackwell dev-branch build options | `.github/workflows/release-docker-deepseek-v4.yml` |
| 2026-04-27 | [#23787](https://github.com/sgl-project/sglang/pull/23787) | merged | amd/deepseek_v4 integration 1/N - 0426 | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py`, `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` |
| 2026-04-27 | [#23776](https://github.com/sgl-project/sglang/pull/23776) | merged | [DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-27 | [#23817](https://github.com/sgl-project/sglang/pull/23817) | merged | docs: verify GB300 Pro DeepSeek V4 recipes | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-27 | [#23810](https://github.com/sgl-project/sglang/pull/23810) | merged | Add benchmarking scripts for deepseek v4 | `scripts/bench_gpqa_aime.py` |
| 2026-04-27 | [#23832](https://github.com/sgl-project/sglang/pull/23832) | merged | amd/deepseek_v4 integration 2/N - cuda graph 0426 | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/compressed/indexer.py`, `python/sglang/srt/layers/attention/compressed/metadata.py` |
| 2026-04-27 | [#23756](https://github.com/sgl-project/sglang/pull/23756) | merged | feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch | `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/environ.py` |
| 2026-04-28 | [#23883](https://github.com/sgl-project/sglang/pull/23883) | merged | Enable DeepGemm warmup in DeepSeek-V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-28 | [#23943](https://github.com/sgl-project/sglang/pull/23943) | merged | [Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-29 | [#23980](https://github.com/sgl-project/sglang/pull/23980) | merged | docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-29 | [#24035](https://github.com/sgl-project/sglang/pull/24035) | merged | [minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4 | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-04 | [#24203](https://github.com/sgl-project/sglang/pull/24203) | merged | [AMD] Deepseek v4 Flash / Pro nightly tests for MI35x ROCm 7.2 | `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py` |
| 2026-05-05 | [#24367](https://github.com/sgl-project/sglang/pull/24367) | merged | [docs] Update B300 Pro cookbook with accuracy-verified serving configs | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-05-08 | [#23882](https://github.com/sgl-project/sglang/pull/23882) | merged | Deepseek V4 | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4_nextn.py` |
| 2026-05-09 | [#24793](https://github.com/sgl-project/sglang/pull/24793) | merged | [DSV4] Cherry pick missing commits from deepseek_v4 branch and enhance tests | `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/unit/entrypoints/openai/test_protocol.py` |
| 2026-05-10 | [#24775](https://github.com/sgl-project/sglang/pull/24775) | merged | Optimize MHC pipeline: DeepGemm, fused norm, fused hc_head | `python/sglang/srt/layers/mhc.py`, `python/sglang/srt/layers/mhc_head.py`, `python/sglang/srt/models/deepseek_v4.py` |
| 2026-05-11 | [#24825](https://github.com/sgl-project/sglang/pull/24825) | merged | [AMD] DSv4 nightly hotfix + schedule-aware --continue-on-error in AMD CI | `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py` |
| 2026-05-12 | [#24949](https://github.com/sgl-project/sglang/pull/24949) | merged | Deepseek-v4-Pro share expert tp1 | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/environ.py` |
| 2026-05-13 | [#25039](https://github.com/sgl-project/sglang/pull/25039) | merged | [AMD] Disable unittest fail-fast for deepseekv4 perf test | `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py` |
| 2026-05-13 | [#25152](https://github.com/sgl-project/sglang/pull/25152) | merged | docs: prepend SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 for H200 FP8 Flash max-throughput | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-05-13 | [#24897](https://github.com/sgl-project/sglang/pull/24897) | merged | Port fused SiLU+clamp+FP8 quant from DSV4 dev branch | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-05-13 | [#24890](https://github.com/sgl-project/sglang/pull/24890) | merged | Port KV Compression V2 from deepseek_v4_dev | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/jit_kernel/deepseek_v4.py` |
| 2026-05-13 | [#24816](https://github.com/sgl-project/sglang/pull/24816) | merged | Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4 | `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py` |
| 2026-05-13 | [#25001](https://github.com/sgl-project/sglang/pull/25001) | merged | [LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` |
| 2026-05-13 | [#24986](https://github.com/sgl-project/sglang/pull/24986) | merged | [rebase]Deepseek_v4 support w4(mxfp4)a16 on hopper | `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/mxfp4.py` |
| 2026-05-14 | [#24925](https://github.com/sgl-project/sglang/pull/24925) | merged | [attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell) | `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py` |
| 2026-05-14 | [#25052](https://github.com/sgl-project/sglang/pull/25052) | merged | DeepSeek V4 w4a4 MegaMoE | `python/sglang/srt/layers/moe/mega_moe.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py` |
| 2026-05-14 | [#25243](https://github.com/sgl-project/sglang/pull/25243) | merged | [Docs] update dsv4 cookbook with H100 deployment commands | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-15 | [#24691](https://github.com/sgl-project/sglang/pull/24691) | merged | [UnifiedTree]: Support HiCache For DeepSeek_V4 | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-05-15 | [#25369](https://github.com/sgl-project/sglang/pull/25369) | merged | Add hicache feature in dsv4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-16 | [#25419](https://github.com/sgl-project/sglang/pull/25419) | merged | Port SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN from deepseek_v4_dev | `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/environ.py` |
| 2026-05-16 | [#24704](https://github.com/sgl-project/sglang/pull/24704) | merged | feat: add Pipeline Parallelism (PP) and PD support for DeepSeek-V4 | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-05-16 | [#25477](https://github.com/sgl-project/sglang/pull/25477) | merged | [BugFix]: Fix DeepSeek V4 HiCache layer count logic | `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` |
| 2026-05-16 | [#25410](https://github.com/sgl-project/sglang/pull/25410) | merged | [Docs] Update DeepSeek V4 cookbook to use the latest docker image | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-16 | [#25412](https://github.com/sgl-project/sglang/pull/25412) | merged | [Doc] DSV4 cookbook: clean up env vars, add MegaMoE toggle, unify docker image | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-05-17 | [#25506](https://github.com/sgl-project/sglang/pull/25506) | merged | [Doc] Fix several places for dpsk v4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-18 | [#25569](https://github.com/sgl-project/sglang/pull/25569) | merged | Add DeepSeekV4 fused MoE Triton autotune support | `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `benchmark/kernels/fused_moe_triton/common_utils.py` |
| 2026-05-18 | [#24933](https://github.com/sgl-project/sglang/pull/24933) | merged | Amd/deepseek v4 rebase main 0509 | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` |
| 2026-05-19 | [#25282](https://github.com/sgl-project/sglang/pull/25282) | merged | [UnifiedTree] Support deepseek v4 host pool layout | `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/test/kl_multiturn_utils.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` |
| 2026-05-19 | [#25733](https://github.com/sgl-project/sglang/pull/25733) | merged | [Bug] Fix V4-Pro NaN on Blackwell by converting fp8_einsum input scale to ue8m0 | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-05-19 | [#25396](https://github.com/sgl-project/sglang/pull/25396) | merged | fix: fix deepseek v4 CP error | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-05-19 | [#25729](https://github.com/sgl-project/sglang/pull/25729) | merged | fix(dsv4): upgrade forward metadata on main stream for large PP size | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-05-19 | [#24934](https://github.com/sgl-project/sglang/pull/24934) | merged | DeepSeek V4 MTP Support CP | `python/sglang/srt/models/deepseek_v4_nextn.py` |
| 2026-05-20 | [#25771](https://github.com/sgl-project/sglang/pull/25771) | merged | fix(dsv4): drop stale pp_size=1 guard for V4 PD disaggregation | `python/sglang/srt/arg_groups/deepseek_v4_hook.py` |
| 2026-05-20 | [#25821](https://github.com/sgl-project/sglang/pull/25821) | merged | [Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-05-21 | [#25810](https://github.com/sgl-project/sglang/pull/25810) | merged | perf(dsv4): add MHC token-count prewarm | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py` |
| 2026-05-21 | [#25889](https://github.com/sgl-project/sglang/pull/25889) | merged | [Fix] DSV4 cached_loc invalidated when SWA mapping is rebuilt | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-05-21 | [#25884](https://github.com/sgl-project/sglang/pull/25884) | merged | [Refactor] major JIT kernel clean up for dsv4 | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/jit_kernel/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-05-21 | [#26004](https://github.com/sgl-project/sglang/pull/26004) | merged | Default MegaMoE to W4A8 for Max-Throughput recipe | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-05-21 | [#25923](https://github.com/sgl-project/sglang/pull/25923) | merged | [Docs] DeepSeek-V4: switch H200 FP4 Pro to flashinfer_mxfp4, Flash Balanced too | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-22 | [#26057](https://github.com/sgl-project/sglang/pull/26057) | merged | [docs] DeepSeek-V4 cookbook: split Quantization axis, add H100 SGLang FP8 | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-05-22 | [#25128](https://github.com/sgl-project/sglang/pull/25128) | merged | [Intel GPU] 1/N Fix tilelang import in deepseek v4 rope as optional | `python/sglang/srt/layers/deepseek_v4_rope.py` |
| 2026-05-23 | [#26141](https://github.com/sgl-project/sglang/pull/26141) | merged | Add non-MTP DSV4 test coverage | `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` |
| 2026-05-23 | [#26164](https://github.com/sgl-project/sglang/pull/26164) | merged | [docs] DeepSeek-V4 cookbook: balanced MegaMoE cap, H200 Pro FP4 mem-frac, nsa-* compat, PD-disagg fixes | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-05-23 | [#25898](https://github.com/sgl-project/sglang/pull/25898) | merged | [AMD] Dsv4/pr1 fix run time issue | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-05-24 | [#25948](https://github.com/sgl-project/sglang/pull/25948) | merged | [dsv4] support eplb | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-05-25 | [#26239](https://github.com/sgl-project/sglang/pull/26239) | merged | [dsv4] fix multi-step draft on non-cuda-graph path | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-05-26 | [#25391](https://github.com/sgl-project/sglang/pull/25391) | merged | Support DeepSeek V4 DeepEP Waterfill | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-05-26 | [#26208](https://github.com/sgl-project/sglang/pull/26208) | merged | [AMD] Dsv4/pr2 compressor opt | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-05-26 | [#26413](https://github.com/sgl-project/sglang/pull/26413) | merged | [docs] DeepSeek-V4 cookbook: note cu129 image for GB200 Pro DeepEP backend | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-05-27 | [#26451](https://github.com/sgl-project/sglang/pull/26451) | merged | [docs] Fix V4 Pro balanced recipe | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-27 | [#26499](https://github.com/sgl-project/sglang/pull/26499) | merged | [Kernel] Import flash_mla kernels from sglang kernel for deepseek v4 | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` |
| 2026-05-27 | [#26383](https://github.com/sgl-project/sglang/pull/26383) | merged | [AMD][DSV4] DSV4 MTP graph + sparse triton attn optimizations | `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py` |
| 2026-05-28 | [#26238](https://github.com/sgl-project/sglang/pull/26238) | merged | refactor(dsv4): route MHC prenorm through DeepGEMM wrapper | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py` |
| 2026-05-28 | [#26609](https://github.com/sgl-project/sglang/pull/26609) | merged | [CI] Clean DeepSeek V4 tests and installation scripts | `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` |
| 2026-05-29 | [#26668](https://github.com/sgl-project/sglang/pull/26668) | merged | [Doc] Update benchmark instruction for dsv4 | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-29 | [#26662](https://github.com/sgl-project/sglang/pull/26662) | merged | [AMD][CI] Update v4 CI setting and move the task to main branch | `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py` |
| 2026-05-30 | [#25976](https://github.com/sgl-project/sglang/pull/25976) | merged | [DeepSeek-V4] Add mhc_fused_post_pre kernel | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py` |
| 2026-06-01 | [#24692](https://github.com/sgl-project/sglang/pull/24692) | merged | feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-06-01 | [#24947](https://github.com/sgl-project/sglang/pull/24947) | merged | DeepSeek V4: Support context parallelism with fused MoE (non-DeepEP) | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` |
| 2026-06-01 | [#26968](https://github.com/sgl-project/sglang/pull/26968) | merged | docs: update RTX PRO 6000 deployment snippet | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-06-02 | [#26931](https://github.com/sgl-project/sglang/pull/26931) | merged | [AMD] dpsk-v4 swa loc cache support | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-06-02 | [#26209](https://github.com/sgl-project/sglang/pull/26209) | merged | Add FP4 Indexer for DeepSeek V4 | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-06-03 | [#27049](https://github.com/sgl-project/sglang/pull/27049) | merged | docs: add DeepSeek-V4 EPLB Waterfill tips | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-06-04 | [#27035](https://github.com/sgl-project/sglang/pull/27035) | merged | docs: add DeepSeek V4 FP4 indexer usage | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/docs/advanced_features/server_arguments.mdx` |
| 2026-06-05 | [#24880](https://github.com/sgl-project/sglang/pull/24880) | merged | [PD & HiSparse] Add DeepSeek V4 support for HiSparse direct Prefill-to-Decode DRAM | `python/sglang/srt/mem_cache/hisparse_memory_pool.py`, `python/sglang/jit_kernel/tests/test_hisparse.py`, `test/registered/disaggregation/test_disaggregation_dsv4.py` |
| 2026-06-05 | [#27404](https://github.com/sgl-project/sglang/pull/27404) | merged | Remove DeepSeek V4 release Docker workflow | `.github/workflows/release-docker-deepseek-v4.yml` |
| 2026-06-06 | [#27152](https://github.com/sgl-project/sglang/pull/27152) | merged | [bugfix][AMD] AttributeError and warp mask bugs in DeepSeek V4 FP4 indexer | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` |
| 2026-06-07 | [#27191](https://github.com/sgl-project/sglang/pull/27191) | merged | Fix DeepSeek V4 DP reduce scatter when use attention DP + MoE TP | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-06-08 | [#26885](https://github.com/sgl-project/sglang/pull/26885) | merged | Cookbook renovation | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/_playground.jsx` |
| 2026-06-08 | [#27289](https://github.com/sgl-project/sglang/pull/27289) | merged | [ROCm] dsv4: remove the redundant fp8 scale transpose-copy on decode | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-06-08 | [#25195](https://github.com/sgl-project/sglang/pull/25195) | merged | [BCG] Support breakable CUDA graph for DeepSeek V4 DP attention | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` |
| 2026-06-10 | [#27380](https://github.com/sgl-project/sglang/pull/27380) | merged | [AMD] Add unified kv attention support in dpsk-v4 | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-06-10 | [#27529](https://github.com/sgl-project/sglang/pull/27529) | merged | [AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase | `python/sglang/srt/layers/attention/dsv4/compressor.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` |
| 2026-06-10 | [#27830](https://github.com/sgl-project/sglang/pull/27830) | merged | [Docs] Restore right-hand ToC on the DeepSeek-V4 cookbook page | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-06-11 | [#27747](https://github.com/sgl-project/sglang/pull/27747) | merged | fix: DSV4 BCG compress-prefill plan OOB on underfilled (tiny) prefill replay | `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh` |
| 2026-06-11 | [#27919](https://github.com/sgl-project/sglang/pull/27919) | merged | Revert "[AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase" | `python/sglang/srt/layers/attention/dsv4/compressor.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` |
| 2026-06-11 | [#27964](https://github.com/sgl-project/sglang/pull/27964) | merged | [Spec] Retire Spec V1 | `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py` |
| 2026-06-12 | [#27973](https://github.com/sgl-project/sglang/pull/27973) | merged | [DSV4] Use int64 for compressor out_loc tensors | `python/sglang/srt/layers/attention/dsv4/compressor_v2.py`, `python/sglang/srt/layers/attention/dsv4/metadata_kernel.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh` |
| 2026-06-12 | [#27149](https://github.com/sgl-project/sglang/pull/27149) | merged | [AMD] [CI] Add dsv4 accuracy PR gate to pr-test-amd-rocm720 | `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py` |
| 2026-06-13 | [#28098](https://github.com/sgl-project/sglang/pull/28098) | merged | Add DeepSeek V4 MTP acceptance length checks | `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` |
| 2026-06-16 | [#27954](https://github.com/sgl-project/sglang/pull/27954) | merged | [dsv4] Pad MLA decode q-heads to 64 (not full n_heads) for FlashMLA head64 kernel | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-06-16 | [#26471](https://github.com/sgl-project/sglang/pull/26471) | merged | DeepSeek-V4 Online Compress support MTP | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` |
| 2026-06-16 | [#28392](https://github.com/sgl-project/sglang/pull/28392) | merged | [AMD] Annotate ATOM source for imported v4 unified attention kernels | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-06-16 | [#28290](https://github.com/sgl-project/sglang/pull/28290) | merged | [AMD] Test DeepSeek V4 FlashMLA backend variants nightly | `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py` |
| 2026-06-16 | [#27928](https://github.com/sgl-project/sglang/pull/27928) | merged | [AMD] Feat: Add prefill context parallel support for deepseek v4 unified kv attention | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` |
| 2026-06-17 | [#28423](https://github.com/sgl-project/sglang/pull/28423) | merged | [AMD] Update v4 amd cookbook | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-06-17 | [#27277](https://github.com/sgl-project/sglang/pull/27277) | merged | Deepseek v4: support mixed dtype compression states | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-06-17 | [#28520](https://github.com/sgl-project/sglang/pull/28520) | merged | [AMD] Fix deepseek-v4 mtp accept length issue | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` |
| 2026-06-18 | [#28613](https://github.com/sgl-project/sglang/pull/28613) | merged | docs: add DeepSeek-V4 compressed state dtype tip | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-06-18 | [#28590](https://github.com/sgl-project/sglang/pull/28590) | merged | [Docs] DeepSeek-V4 cookbook: drop --disable-flashinfer-autotune from GB300 Flash low-latency | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` |
| 2026-06-18 | [#25144](https://github.com/sgl-project/sglang/pull/25144) | merged | [NPU] Add Ascend NPU support for DeepSeek-V4 | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` |
| 2026-06-18 | [#26766](https://github.com/sgl-project/sglang/pull/26766) | merged | [DeepSeek-V4] Fuse UE8M0 scale rounding into FP8 group quantization | `python/sglang/srt/models/deepseek_v4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` |
| 2026-06-22 | [#25820](https://github.com/sgl-project/sglang/pull/25820) | merged | [NVIDIA] Support NVFP4 MoE for DeepSeek-V4 | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py` |
| 2026-06-22 | [#28920](https://github.com/sgl-project/sglang/pull/28920) | merged | [AMD] deepseek-v4 clean env vars | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`, `test/registered/amd/test_deepseek_v4_flash_fp4.py` |
| 2026-06-22 | [#28941](https://github.com/sgl-project/sglang/pull/28941) | merged | [AMD] Fix nightly-8-gpu-mi35x-deepseek-v4-flash-rocm720 OOM issue | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-06-23 | [#28981](https://github.com/sgl-project/sglang/pull/28981) | merged | [AMD] Update v4 cookbook to clean env vars | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-06-23 | [#28938](https://github.com/sgl-project/sglang/pull/28938) | merged | [AMD] Improve performance of dsv4 in high concurrency | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-06-24 | [#28455](https://github.com/sgl-project/sglang/pull/28455) | merged | [AMD] Fix DeepSeek-V4 fp8 KV path on gfx942 (e4m3fnuz) | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-06-24 | [#28952](https://github.com/sgl-project/sglang/pull/28952) | merged | Add DeepSeek V4 Flash demo notebook | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-06-25 | [#29261](https://github.com/sgl-project/sglang/pull/29261) | merged | [Docs] Fix broken links in cookbook | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` |
| 2026-06-25 | [#28103](https://github.com/sgl-project/sglang/pull/28103) | merged | Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test | `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml` |
| 2026-06-25 | [#29103](https://github.com/sgl-project/sglang/pull/29103) | merged | [AMD] Feat/dsv4 aiter reduce scatter decode | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-06-26 | [#27783](https://github.com/sgl-project/sglang/pull/27783) | merged | [Intel GPU] DeepSeek V4 3/N: Support hc_split_sinkhorn on XPU using sgl_kernel | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-06-27 | [#29106](https://github.com/sgl-project/sglang/pull/29106) | merged | Fix DeepSeek V4 PP HiCache SWA allocation and layer mapping | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-06-28 | [#29502](https://github.com/sgl-project/sglang/pull/29502) | merged | [CI] Fix GB300 DSV4 Pro FP4 nightly | `test/registered/gb300/test_deepseek_v4_pro_fp4.py` |
| 2026-06-30 | [#29420](https://github.com/sgl-project/sglang/pull/29420) | merged | [AMD][DSV4] Remove per-batch D2H syncs in MTP to avoid bubbles between 2 batches | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` |
| 2026-06-30 | [#28980](https://github.com/sgl-project/sglang/pull/28980) | merged | [NPU] Support DeepSeek V4 Flash MTP on Ascend | `python/sglang/srt/models/deepseek_v4_nextn.py`, `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-01 | [#29827](https://github.com/sgl-project/sglang/pull/29827) | merged | [Doc] Tiny update dsv4 doc | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-07-01 | [#29775](https://github.com/sgl-project/sglang/pull/29775) | merged | [DeepSeek V4] Enable FlashMLA sparse prefill by default | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py` |
| 2026-07-02 | [#29885](https://github.com/sgl-project/sglang/pull/29885) | merged | [DeepSeek V4] Cover both dense and sparse prefill paths in the compress attention unittest | `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` |
| 2026-07-02 | [#29982](https://github.com/sgl-project/sglang/pull/29982) | merged | [AMD][DeepSeek V4] Fix default FlashMLA sparse prefill off on ROCm/HIP | `python/sglang/srt/arg_groups/deepseek_v4_hook.py` |
| 2026-07-03 | [#29619](https://github.com/sgl-project/sglang/pull/29619) | merged | [DeepSeek-V4] Add an opt-in non-paged indexer for long-context prefill | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-07-03 | [#29988](https://github.com/sgl-project/sglang/pull/29988) | merged | [dsv4] Trigger MHC prenorm prewarm at weight-load time with rank sync | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-03 | [#27349](https://github.com/sgl-project/sglang/pull/27349) | merged | Support DSV4 shared expert fusion for DeepEP and MegaMOE | `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py`, `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-03 | [#27914](https://github.com/sgl-project/sglang/pull/27914) | merged | [Intel GPU] DeepSeek V4 6/N: use sgl-kernel implemetation of flash_mla_with_kvcache on XPU | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-07-06 | [#29362](https://github.com/sgl-project/sglang/pull/29362) | merged | [AMD ]Feat/dsv4 ep tbo prefill | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py` |
| 2026-07-06 | [#30237](https://github.com/sgl-project/sglang/pull/30237) | merged | [AMD][DeepSeek V4] Set SGLANG_OPT_FLASHMLA_SPARSE_PREFILL to false on hip code path | `python/sglang/srt/arg_groups/deepseek_v4_hook.py` |
| 2026-07-07 | [#27867](https://github.com/sgl-project/sglang/pull/27867) | merged | [DSv4] Loading Time Weight Dequant | `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/configs/model_config.py` |
| 2026-07-07 | [#30333](https://github.com/sgl-project/sglang/pull/30333) | merged | [AMD] Fix DeepSeek V4 MTP accuracy issue | `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` |
| 2026-07-08 | [#27926](https://github.com/sgl-project/sglang/pull/27926) | merged | [DSV4] perf: Make FP8 quant output tensor contiguous | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-09 | [#29417](https://github.com/sgl-project/sglang/pull/29417) | merged | [AMD] Enable unified-KV HiCache on DeepSeek-V4 | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-07-09 | [#30695](https://github.com/sgl-project/sglang/pull/30695) | merged | [Refactor] Make DeepSeek-V4 attention backend tolerate an absent CPU seq_lens mirror | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-07-10 | [#30711](https://github.com/sgl-project/sglang/pull/30711) | merged | [Refactor] Split DeepSeek-V4 MQALayer into a reusable attention base | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-13 | [#30898](https://github.com/sgl-project/sglang/pull/30898) | merged | Enable breakable prefill CUDA graph for DP attention | `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/forward_batch_info.py` |
| 2026-07-14 | [#31125](https://github.com/sgl-project/sglang/pull/31125) | merged | Disable flaky DSV4-Flash FP4 BCG determinism test (nondeterminism from #30898 idle-rank dummy extend) | `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` |
| 2026-07-15 | [#30365](https://github.com/sgl-project/sglang/pull/30365) | merged | [DSV4] Remove per-step seqlen D2H from speculative to make overlap scheduler work | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` |
| 2026-07-15 | [#30792](https://github.com/sgl-project/sglang/pull/30792) | merged | [Kernel] Migrate DSA + DSV4 attention kernels to sglang.kernels (RFC #29630, Phase 2.5, 5/7) | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-15 | [#30651](https://github.com/sgl-project/sglang/pull/30651) | merged | cookbook(deepseek-v4): add MORI disagg backend for AMD + bump MI355X image | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/_playground.jsx` |
| 2026-07-16 | [#28983](https://github.com/sgl-project/sglang/pull/28983) | merged | perf(deepseek_v4): enable SGLANG_OPT_FP8_WO_A_GEMM on sm90 (Hopper) | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-16 | [#31373](https://github.com/sgl-project/sglang/pull/31373) | merged | [Docs] Align B200 DeepSeek-V4-Pro balanced recipe with MegaMoE | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` |
| 2026-07-16 | [#31122](https://github.com/sgl-project/sglang/pull/31122) | merged | [Docs] Add AMD-specific HiCache config for DeepSeek V4 playground | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/_playground.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-07-16 | [#30238](https://github.com/sgl-project/sglang/pull/30238) | merged | [AMD] Support two batch overlap with MTP on DeepSeekV4 | `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py` |
| 2026-07-16 | [#25763](https://github.com/sgl-project/sglang/pull/25763) | merged | [Feature] Support DeepSeek-V4 Wint4Abf16 and Win4Afp8. | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-17 | [#31452](https://github.com/sgl-project/sglang/pull/31452) | merged | [Docs] Tune DeepSeek-V4 HiCache for MI355X PD | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/_playground.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-07-18 | [#30272](https://github.com/sgl-project/sglang/pull/30272) | merged | Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2 | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-07-19 | [#31705](https://github.com/sgl-project/sglang/pull/31705) | merged | [DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-07-21 | [#31363](https://github.com/sgl-project/sglang/pull/31363) | merged | docs(cookbook): re-benchmark DeepSeek-V4 on sglang 0.5.15 | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/_deployment.jsx` |
| 2026-07-23 | [#27657](https://github.com/sgl-project/sglang/pull/27657) | merged | [DeepSeek V4] CP decode opt: slice repeat attention weights to local TP partition | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py` |
| 2026-07-23 | [#29569](https://github.com/sgl-project/sglang/pull/29569) | merged | [DSV4] Support megamoe for CP | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` |
| 2026-07-24 | [#27059](https://github.com/sgl-project/sglang/pull/27059) | merged | Add FP4 Indexer for DeepSeek V4 on SM120 | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-07-24 | [#31087](https://github.com/sgl-project/sglang/pull/31087) | merged | [RL] DSV4: dispatch indexer topk_transform_512 through DSATopKBackend | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-07-24 | [#31086](https://github.com/sgl-project/sglang/pull/31086) | merged | [RL] DSV4: add env to quantize SWA KV cache from bf16-rounded values | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-26 | [#30954](https://github.com/sgl-project/sglang/pull/30954) | merged | [SM120] Allow fused MHC opt-in with standalone TileLang pre disabled | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-28 | [#31931](https://github.com/sgl-project/sglang/pull/31931) | merged | [NPU] Optimize DeepSeek-V4 performance | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/kernels/ops/attention/deepseek_v4_rope.py` |
| 2026-07-29 | [#31747](https://github.com/sgl-project/sglang/pull/31747) | merged | [AMD] DSv4: bring HIP compress-state pool into the memory_saver KV_CACHE region | `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-07-29 | [#31563](https://github.com/sgl-project/sglang/pull/31563) | merged | fix mqa preshuffle layout issue for deepseek v4 | `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh` |
| 2026-07-30 | [#30240](https://github.com/sgl-project/sglang/pull/30240) | merged | Fix DeepSeek V4 loading with RunAI Model Streamer. | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py` |
| 2026-08-02 | [#31727](https://github.com/sgl-project/sglang/pull/31727) | merged | [AMD] Fix DeepSeek-V4 fused-RMS FP8 scale metadata on gfx950 | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-08-03 | [#32910](https://github.com/sgl-project/sglang/pull/32910) | merged | [DeepSeek-V4] Fix nvcc 13 crash building the topk_v2 kernel | `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` |
| 2026-08-04 | [#30741](https://github.com/sgl-project/sglang/pull/30741) | merged | Prewarm DSV4 MHC post kernel at model load | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-08-05 | [#31865](https://github.com/sgl-project/sglang/pull/31865) | merged | [XPU] DeepSeek V4: use sgl-kernel-xpu implemetation of flash_mla_sparse_fwd for prefill | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-08-07 | [#33616](https://github.com/sgl-project/sglang/pull/33616) | merged | feat: Add flashinfer mHC fusion for DSV4 | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-08-07 | [#33532](https://github.com/sgl-project/sglang/pull/33532) | merged | [CP]: Support CP V2 Strategy for dsv4 | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py` |
| 2026-08-07 | [#34044](https://github.com/sgl-project/sglang/pull/34044) | merged | docs(cookbook): DeepSeek-V4-Flash-0731 — drop chunked-prefill/autotune flags on B300 low-latency | `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` |
| 2026-08-08 | [#30964](https://github.com/sgl-project/sglang/pull/30964) | merged | [AMD] Support DeepSeek V4 DSpark on AMD HIP platform | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py` |
| 2026-08-09 | [#34189](https://github.com/sgl-project/sglang/pull/34189) | merged | [DSV4] Fix silent KV corruption when speculative draft tokens > 4 | `test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py`, `python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-08-10 | [#26671](https://github.com/sgl-project/sglang/pull/26671) | merged | [JIT Kernel][DSv4] Optimize epilogue of c128 | `python/sglang/kernels/jit/csrc/deepseek_v4/c128_v2.cuh` |
| 2026-08-10 | [#33312](https://github.com/sgl-project/sglang/pull/33312) | merged | Fix DSV4 DSpark shared expert loading | `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py`, `python/sglang/srt/models/deepseek_v4_dspark.py` |
| 2026-08-11 | [#31700](https://github.com/sgl-project/sglang/pull/31700) | merged | Fix DeepSeek-V4/DeepSeek-V4-Pro DP-attention gather semantics | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py` |
| 2026-08-11 | [#33662](https://github.com/sgl-project/sglang/pull/33662) | merged | [DSV4] Avoid host syncs in EAGLE prefill | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-08-11 | [#34333](https://github.com/sgl-project/sglang/pull/34333) | merged | docs: remove DSV4 low-latency chunked prefill size | `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` |
| 2026-08-11 | [#33865](https://github.com/sgl-project/sglang/pull/33865) | merged | Fix DSpark + DeepSeek V4 prefill CP compatibility | `python/sglang/srt/models/deepseek_v4_dspark.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` |
| 2026-08-11 | [#29070](https://github.com/sgl-project/sglang/pull/29070) | merged | [DSV4] perf: Enable alt stream during BCG prefill | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-08-13 | [#34597](https://github.com/sgl-project/sglang/pull/34597) | merged | [AMD] Run V4 MTP target-verify through the decode kernel | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` |
| 2026-08-14 | [#25855](https://github.com/sgl-project/sglang/pull/25855) | merged | perf(jit_kernel/deepseek_v4): optimize paged_mqa_metadata | `python/sglang/kernels/jit/csrc/deepseek_v4/paged_mqa_metadata.cuh` |
| 2026-08-14 | [#34788](https://github.com/sgl-project/sglang/pull/34788) | merged | [Fix] Restore layer-level DSV4 RoPE policy | `test/registered/unit/models/test_deepseek_v4_rope_policy.py`, `python/sglang/srt/models/deepseek_v4.py` |
| 2026-08-14 | [#34809](https://github.com/sgl-project/sglang/pull/34809) | merged | [Cookbook] Add DeepSeek-V4-Pro-0813 (Pro Official) serving recipes | `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-08-17 | [#34277](https://github.com/sgl-project/sglang/pull/34277) | merged | [DSV4] Emit TMA-aligned UE8M0 scales for FP8 einsum | `python/sglang/kernels/jit/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh` |
| 2026-08-17 | [#33480](https://github.com/sgl-project/sglang/pull/33480) | merged | [AMD] Support prefill context parallel two batch overlap for DeepSeek V4 | `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp_tbo.py` |
| 2026-08-17 | [#33676](https://github.com/sgl-project/sglang/pull/33676) | merged | [NPU] Support DeepSeek-V4 DSpark and refactor DSV4 cache management | `python/sglang/srt/models/deepseek_v4_dspark.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` |
| 2026-08-17 | [#34926](https://github.com/sgl-project/sglang/pull/34926) | merged | Clean deprecated DeepSeek V4 Environs | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py` |
| 2026-08-18 | [#35224](https://github.com/sgl-project/sglang/pull/35224) | merged | [Docs] Enable PD disaggregation for DSV4 low-latency recipes | `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-08-18 | [#34890](https://github.com/sgl-project/sglang/pull/34890) | merged | [Perf] Hoist DSv4 draft-extend SWA write locs; unify SWA graph buffer naming | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-08-18 | [#35162](https://github.com/sgl-project/sglang/pull/35162) | merged | Add deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms | `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms.py` |
| 2026-08-19 | [#33313](https://github.com/sgl-project/sglang/pull/33313) | merged | [AMD] DeepSeek-V4: route decode wo_a bf16 batched matmul to aiter batched_gemm_bf16 | `test/registered/unit/models/test_deepseek_v4_amd_wo_a_bf16.py`, `python/sglang/srt/models/deepseek_v4.py` |
| 2026-08-20 | [#32327](https://github.com/sgl-project/sglang/pull/32327) | merged | [DeepSeek-V4] Add Q8KV8 sparse MLA prefill runtime backend | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-08-21 | [#34973](https://github.com/sgl-project/sglang/pull/34973) | merged | [AMD] DSv4: fuse the qk-norm-rope pair on the MTP target-verify path | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-08-21 | [#35919](https://github.com/sgl-project/sglang/pull/35919) | merged | [DeepSeek V4] Default FP4 checkpoints to FlashInfer MXFP4 MoE | `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `python/sglang/srt/arg_groups/overrides.py`, `python/sglang/srt/server_args.py` |
| 2026-08-22 | [#35918](https://github.com/sgl-project/sglang/pull/35918) | merged | [DeepSeek V4] Add W4A4 MegaMoE server flag | `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` |
| 2026-08-22 | [#32577](https://github.com/sgl-project/sglang/pull/32577) | merged | [AMD] DeepSeek-V4: add aiter fused mHC post+pre with cross-layer boundary dispatch | `test/registered/unit/models/test_deepseek_v4_amd_fused_mhc.py`, `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py`, `python/sglang/srt/models/deepseek_v4.py` |

## 逐 PR diff 审计卡

### PR #23605 - Add DeepSeek V4 cookbook

- 链接: https://github.com/sgl-project/sglang/pull/23605
- 状态/时间: merged / 2026-04-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+1024/-1，可读 patch 1041 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add DeepSeek V4 cookbook」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`；技术摘要: 覆盖「Add DeepSeek V4 cookbook」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0 (569 lines); hunks: -0,0 +1,569；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0 (453 lines); hunks: -0,0 +1,453；`docs_new/cookbook/autoregressive/intro.mdx` modified +1/-1 (2 lines); hunks: -16,7 +16,7 @@ metatags:；`docs_new/docs.json` modified +1/-0 (1 lines); hunks: -940,6 +940,7。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0 (569 lines); hunks: -0,0 +1,569
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0 (453 lines); hunks: -0,0 +1,453
  - `docs_new/cookbook/autoregressive/intro.mdx` modified +1/-1 (2 lines); hunks: -16,7 +16,7 @@ metatags:
  - `docs_new/docs.json` modified +1/-0 (1 lines); hunks: -940,6 +940,7
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -0,0 +1,569 @@
+export const DeepSeekV4Deployment = () => {
+  // DeepSeek-V4 deployment matrix (small / real checkpoint):
+  //   Hardware × Recipe → concrete launch command.
+  //
+  //   Hardware (quantization determined by GPU generation):
+  //     B200  → FP4 weights, Flash TP=4 / Pro TP=8 single-node
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -0,0 +1,453 @@
+---
+title: DeepSeek-V4
+metatags:
+    description: "Deploy DeepSeek-V4 with SGLang — a next-generation MoE model from DeepSeek. Blackwell deployments use the FP4 checkpoint; Hopper deployments use the FP8 checkpoi
+tag: NEW
+---
diff -- docs_new/cookbook/autoregressive/intro.mdx
@@ -16,7 +16,7 @@ metatags:
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0; `docs_new/cookbook/autoregressive/intro.mdx` modified +1/-1; `docs_new/docs.json` modified +1/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/docs.json`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23617 - Further update Deepseek V4 docs

- 链接: https://github.com/sgl-project/sglang/pull/23617
- 状态/时间: merged / 2026-04-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-6，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Further update Deepseek V4 docs」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「Further update Deepseek V4 docs」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6 (11 lines); hunks: -137,12 +137,11 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6 (11 lines); hunks: -137,12 +137,11 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -137,12 +137,11 @@ export const DeepSeekV4Deployment = () => {
-    // H200 needs a separate FP8-only Instruct ckpt (Flash / Pro public repos
-    // ship FP4-mixed weights). That ckpt is still being uploaded, so we emit a
-    // placeholder that fails loudly on copy-paste instead of silently pulling
-    // the wrong weights. Replace with the real slug once Hopper ckpts are public.
-    "h200|small":  { slug: "<TO_BE_UPLOADED_DeepSeek-V4-Flash-hopper>", tp: 4,  multinode: false },
-    "h200|big":    { slug: "<TO_BE_UPLOADED_DeepSeek-V4-Pro-hopper>",   tp: 16, multinode: true, nnodes: 2 },
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23628 - docs: note H200 DeepSeek-V4 checkpoint

- 链接: https://github.com/sgl-project/sglang/pull/23628
- 状态/时间: merged / 2026-04-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-0，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: note H200 DeepSeek-V4 checkpoint」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「docs: note H200 DeepSeek-V4 checkpoint」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -99,6 +99,10 @@ Please refer to the [official SGLang installation guide](../....。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -99,6 +99,10 @@ Please refer to the [official SGLang installation guide](../....
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -99,6 +99,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+<Note>
+For H200 GPU deployments, use the SGLang checkpoint under `sgl-project`, not the default DeepSeek checkpoint.
+</Note>
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23622 - Again update DeepSeek V4 cookbook

- 链接: https://github.com/sgl-project/sglang/pull/23622
- 状态/时间: merged / 2026-04-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+32/-9，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Again update DeepSeek V4 cookbook」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「Again update DeepSeek V4 cookbook」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9 (28 lines); hunks: -42,11 +42,11 @@ export const DeepSeekV4Deployment = () => {; -161,7 +161,16 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../....。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9 (28 lines); hunks: -42,11 +42,11 @@ export const DeepSeekV4Deployment = () => {; -161,7 +161,16 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../....
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -42,11 +42,11 @@ export const DeepSeekV4Deployment = () => {
-        { id: "low-latency",    label: "Low-Latency",      default: true,  subtitle: "MTP 3/4" },
-        { id: "balanced",       label: "Balanced",         default: false, subtitle: "MTP 1/2 + DeepEP" },
-        { id: "max-throughput", label: "Max-Throughput",   default: false, subtitle: "DP + DeepEP" },
-        { id: "cp",             label: "Context-Parallel", default: false, subtitle: "long prompts" },
-        { id: "pd-disagg",      label: "PD-Disagg",        default: false, subtitle: "1P + 1D + router" },
+        { id: "low-latency",    label: "Low-Latency",      default: true  },
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+For how to actually launch one of these images, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). A minimal example (substitute the
+'''bash Command
+docker run --gpus all \
+    --shm-size 32g \
+    -p 30000:30000 \
+    -v ~/.cache/huggingface:/root/.cache/huggingface \
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23634 - Update pro fp8 checkpoint in DeepSeek V4 cookbook

- 链接: https://github.com/sgl-project/sglang/pull/23634
- 状态/时间: merged / 2026-04-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update pro fp8 checkpoint in DeepSeek V4 cookbook」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「Update pro fp8 checkpoint in DeepSeek V4 cookbook」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2 (4 lines); hunks: -139,9 +139,9 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2 (4 lines); hunks: -139,9 +139,9 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -139,9 +139,9 @@ export const DeepSeekV4Deployment = () => {
-    // repackagings; Flash is public, Pro is still being uploaded.
+    // repackagings for both variants.
-    "h200|big":    { slug: "<TO_BE_UPLOADED_DeepSeek-V4-Pro-FP8>",     tp: 16, multinode: true, nnodes: 2 },
+    "h200|big":    { slug: "sgl-project/DeepSeek-V4-Pro-FP8",          tp: 16, multinode: true, nnodes: 2 },
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23684 - docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models

- 链接: https://github.com/sgl-project/sglang/pull/23684
- 状态/时间: merged / 2026-04-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-0，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -147,6 +147,10 @@ The generator currently picks values on the **conservative*...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -147,6 +147,10 @@ The generator currently picks values on the **conservative*...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -147,6 +147,10 @@ The generator currently picks values on the **conservative** side (mirroring an
+**Base model usage**
+In order to use base models, please enable `SGLANG_FIX_DSV4_BASE_MODEL_LOAD=1` and use latest code, before the next round of testing matrix is finished.
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23689 - docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified

- 链接: https://github.com/sgl-project/sglang/pull/23689
- 状态/时间: merged / 2026-04-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+22/-1，可读 patch 59 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0 (14 lines); hunks: -164,14 +164,26 @@ export const DeepSeekV4Deployment = () => {; -387,6 +399,7 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1 (9 lines); hunks: -145,7 +145,14 @@ The generator currently picks values on the **conservative*...。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0 (14 lines); hunks: -164,14 +164,26 @@ export const DeepSeekV4Deployment = () => {; -387,6 +399,7 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1 (9 lines); hunks: -145,7 +145,14 @@ The generator currently picks values on the **conservative*...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -164,14 +164,26 @@ export const DeepSeekV4Deployment = () => {
+    "b200|small|pd-disagg",
+    "h200|small|cp",
+    "h200|small|pd-disagg",
+    // h200|big|pd-disagg: pending verification (needs 4-node H200 cluster with
+    //   shared IB fabric: 2-node prefill + 2-node decode).
+  // Recipes whose command is intentionally not yet provided (e.g. blocked by an
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -145,7 +145,14 @@ The generator currently picks values on the **conservative** side (mirroring an
-The H200 image and checkpoint are currently being uploaded — public path coming shortly.
+H200 image (`lmsysorg/sglang:deepseek-v4-hopper`) and FP8 checkpoints
+(`sgl-project/DeepSeek-V4-Flash-FP8`, `sgl-project/DeepSeek-V4-Pro-FP8`) are
+publicly available.
+PD-Disagg recipes on H200 may require `docker run --privileged --ulimit memlock=-1`
+(or `--device /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK`) so mooncake
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23691 - docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes

- 链接: https://github.com/sgl-project/sglang/pull/23691
- 状态/时间: merged / 2026-04-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+56/-5，可读 patch 113 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +49/-5 (54 lines); hunks: -176,6 +176,10 @@ export const DeepSeekV4Deployment = () => {; -372,7 +376,17 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0 (7 lines); hunks: -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpo...。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +49/-5 (54 lines); hunks: -176,6 +176,10 @@ export const DeepSeekV4Deployment = () => {; -372,7 +376,17 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0 (7 lines); hunks: -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpo...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -176,6 +176,10 @@ export const DeepSeekV4Deployment = () => {
+    "gb300|small|cp",
+    "gb300|big|cp",
+    "gb300|small|pd-disagg",
+    "gb300|big|pd-disagg",
@@ -372,7 +376,17 @@ export const DeepSeekV4Deployment = () => {
-      flags.push("  --mem-fraction-static 0.78");
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpoints.
+**GB300 PD-Disagg cross-pod MNNVL**
+On some GB300 clusters with cross-pod KV transfer over NVLink, mooncake may
+fail with `nvlink_transport.cpp:497 Requested address ... not found!`. If
+this happens, prepend `MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1`
+to both prefill and decode `sglang serve` commands.
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +49/-5; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23690 - Small udpate gb300 recipe for deepseek v4

- 链接: https://github.com/sgl-project/sglang/pull/23690
- 状态/时间: merged / 2026-04-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Small udpate gb300 recipe for deepseek v4」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「Small udpate gb300 recipe for deepseek v4」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -172,6 +172,9 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -172,6 +172,9 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -172,6 +172,9 @@ export const DeepSeekV4Deployment = () => {
+    "gb300|small|low-latency",
+    "gb300|small|balanced",
+    "gb300|small|max-throughput",
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23697 - update: b300 container for dsv4

- 链接: https://github.com/sgl-project/sglang/pull/23697
- 状态/时间: merged / 2026-04-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+11/-2，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「update: b300 container for dsv4」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「update: b300 container for dsv4」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +7/-2 (9 lines); hunks: -26,6 +26,7 @@ export const DeepSeekV4Deployment = () => {; -222,7 +223,9 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../....。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +7/-2 (9 lines); hunks: -26,6 +26,7 @@ export const DeepSeekV4Deployment = () => {; -222,7 +223,9 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../....
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -26,6 +26,7 @@ export const DeepSeekV4Deployment = () => {
+        { id: "b300",  label: "B300 (FP4)",  default: false  },
@@ -222,7 +223,9 @@ export const DeepSeekV4Deployment = () => {
-    const { hardware, modelSize, recipe, reasoningParser, toolcall } = values;
+    const { hardware: rawHardware, modelSize, recipe, reasoningParser, toolcall } = values;
+    // B300 usage is identical to B200 — alias so we don't duplicate every spec entry.
+    const hardware = rawHardware === "b300" ? "b200" : rawHardware;
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+    <tr>
+      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>NVIDIA B300</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>lmsysorg/sglang:deepseek-v4-b300</code></td>
+    </tr>
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +7/-2; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23698 - docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9

- 链接: https://github.com/sgl-project/sglang/pull/23698
- 状态/时间: merged / 2026-04-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-3，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-3 (8 lines); hunks: -495,11 +495,13 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-3 (8 lines); hunks: -495,11 +495,13 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -495,11 +495,13 @@ export const DeepSeekV4Deployment = () => {
-        // OOM during CG capture. Verified working on 2026-04-25 (journal
-        // 2026-04-25-001 Cell D, Δ10).
+        // OOM during CG capture. mem-frac sweep at 0.83 / 0.87 / 0.89 / 0.91
+        // all pass static smoke; 0.9 picked as the default — leaves
+        // ~14 GB / GPU post-CG headroom for mooncake transfer + activation
+        // peaks while giving ~1M-token KV pool.
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23715 - docs(DeepSeek-V4): mark h200|big|pd-disagg verified + recipe fixes

- 链接: https://github.com/sgl-project/sglang/pull/23715
- 状态/时间: merged / 2026-04-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+31/-4，可读 patch 59 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(DeepSeek-V4): mark h200|big|pd-disagg verified + recipe fixes」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「docs(DeepSeek-V4): mark h200|big|pd-disagg verified + recipe fixes」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +31/-4 (35 lines); hunks: -178,8 +178,7 @@ export const DeepSeekV4Deployment = () => {; -480,6 +479,12 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +31/-4 (35 lines); hunks: -178,8 +178,7 @@ export const DeepSeekV4Deployment = () => {; -480,6 +479,12 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -178,8 +178,7 @@ export const DeepSeekV4Deployment = () => {
-    // h200|big|pd-disagg: pending verification (needs 4-node H200 cluster with
-    //   shared IB fabric: 2-node prefill + 2-node decode).
+    "h200|big|pd-disagg",
@@ -480,6 +479,12 @@ export const DeepSeekV4Deployment = () => {
+      // H200 Pro PD: tp=16 multinode + DeepEP needs the dispatch buffer cap on
+      // BOTH prefill + decode (matches production playground LWS for the same
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +31/-4
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23728 - ci: add docker release workflow for deepseek_v4 branch

- 链接: https://github.com/sgl-project/sglang/pull/23728
- 状态/时间: merged / 2026-04-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+93/-0，可读 patch 94 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: add docker release workflow for deepseek_v4 branch」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `.github/workflows/release-docker-deepseek-v4.yml`；技术摘要: 覆盖「ci: add docker release workflow for deepseek_v4 branch」；主要实现面是 `.github/workflows/release-docker-deepseek-v4.yml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.github/workflows/release-docker-deepseek-v4.yml` added +93/-0 (93 lines); hunks: -0,0 +1,93。
- 代码 diff 细节:
  - `.github/workflows/release-docker-deepseek-v4.yml` added +93/-0 (93 lines); hunks: -0,0 +1,93
- 关键代码摘录:

```diff
diff -- .github/workflows/release-docker-deepseek-v4.yml
@@ -0,0 +1,93 @@
+name: Build and Push DeepSeek-V4 Docker Images
+# Builds the 4 Dockerfiles added in #23600 from the deepseek_v4 branch and
+# pushes them to Docker Hub. Each Dockerfile is single-arch and does its own
+# `git clone -b deepseek_v4` inside, so no build context source is required
+# beyond the Dockerfiles themselves and `--no-cache` is mandatory.
+on:
```

- 已读文件:
  - ci: `.github/workflows/release-docker-deepseek-v4.yml` added +93/-0
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #23730 - [CI] release-docker-deepseek-v4: select which flavors to push

- 链接: https://github.com/sgl-project/sglang/pull/23730
- 状态/时间: merged / 2026-04-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+56/-18，可读 patch 92 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] release-docker-deepseek-v4: select which flavors to push」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `.github/workflows/release-docker-deepseek-v4.yml`；技术摘要: 覆盖「[CI] release-docker-deepseek-v4: select which flavors to push」；主要实现面是 `.github/workflows/release-docker-deepseek-v4.yml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.github/workflows/release-docker-deepseek-v4.yml` modified +56/-18 (74 lines); hunks: -12,35 +12,73 @@ on:。
- 代码 diff 细节:
  - `.github/workflows/release-docker-deepseek-v4.yml` modified +56/-18 (74 lines); hunks: -12,35 +12,73 @@ on:
- 关键代码摘录:

```diff
diff -- .github/workflows/release-docker-deepseek-v4.yml
@@ -12,35 +12,73 @@ on:
+      build_hopper:
+        description: "Build and push the Hopper (H200) image."
+        required: false
+        type: boolean
+        default: true
+      build_blackwell:
```

- 已读文件:
  - ci: `.github/workflows/release-docker-deepseek-v4.yml` modified +56/-18
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #23725 - docs(DeepSeek-V4): add GB200 platform to cookbook recipe

- 链接: https://github.com/sgl-project/sglang/pull/23725
- 状态/时间: merged / 2026-04-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+58/-8，可读 patch 195 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(DeepSeek-V4): add GB200 platform to cookbook recipe」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「docs(DeepSeek-V4): add GB200 platform to cookbook recipe」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +52/-6 (58 lines); hunks: -4,6 +4,7 @@ export const DeepSeekV4Deployment = () => {; -27,6 +28,7 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-2 (8 lines); hunks: -29,13 +29,13 @@ tag: NEW; -88,6 +88,10 @@ Please refer to the [official SGLang installation guide](../....。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +52/-6 (58 lines); hunks: -4,6 +4,7 @@ export const DeepSeekV4Deployment = () => {; -27,6 +28,7 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-2 (8 lines); hunks: -29,13 +29,13 @@ tag: NEW; -88,6 +88,10 @@ Please refer to the [official SGLang installation guide](../....
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -4,6 +4,7 @@ export const DeepSeekV4Deployment = () => {
+  //     GB200 → FP4 weights, Flash TP=4 / Pro TP=8 2-node
@@ -27,6 +28,7 @@ export const DeepSeekV4Deployment = () => {
+        { id: "gb200", label: "GB200 (FP4)", default: false },
@@ -138,6 +140,8 @@ export const DeepSeekV4Deployment = () => {
+    "gb200|small": { slug: "deepseek-ai/DeepSeek-V4-Flash", tp: 4,  multinode: false },
+    "gb200|big":   { slug: "deepseek-ai/DeepSeek-V4-Pro",   tp: 8,  multinode: true, nnodes: 2 },
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -29,13 +29,13 @@ tag: NEW
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>single-node serving: B200 / GB300 / H200 on 4 GPUs</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>single-node serving: B200 / GB200 / GB300 / H200 on 4 GPUs</td>
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB300 4 GPU / H200 16 GPU (2 nodes)</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB200 8 GPU (2 nodes) / GB300 4 GPU / H200 16 GPU (2 nodes)</td>
@@ -88,6 +88,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+    <tr>
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +52/-6; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23742 - docs(DeepSeek-V4): add h200|big verified recipes + tune H200 Pro parameters

- 链接: https://github.com/sgl-project/sglang/pull/23742
- 状态/时间: merged / 2026-04-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+22/-8，可读 patch 83 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(DeepSeek-V4): add h200|big verified recipes + tune H200 Pro parameters」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「docs(DeepSeek-V4): add h200|big verified recipes + tune H200 Pro parameters」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +22/-8 (30 lines); hunks: -184,6 +184,9 @@ export const DeepSeekV4Deployment = () => {; -272,7 +275,9 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +22/-8 (30 lines); hunks: -184,6 +184,9 @@ export const DeepSeekV4Deployment = () => {; -272,7 +275,9 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -184,6 +184,9 @@ export const DeepSeekV4Deployment = () => {
+    "h200|big|low-latency",
+    "h200|big|balanced",
+    "h200|big|max-throughput",
@@ -272,7 +275,9 @@ export const DeepSeekV4Deployment = () => {
-        recipeEnv.push("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256");
+        recipeEnv.push(isBig
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +22/-8
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23737 - docs(DeepSeek-V4): mark gb200|big|low-latency verified

- 链接: https://github.com/sgl-project/sglang/pull/23737
- 状态/时间: merged / 2026-04-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(DeepSeek-V4): mark gb200|big|low-latency verified」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「docs(DeepSeek-V4): mark gb200|big|low-latency verified」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +1/-0 (1 lines); hunks: -174,6 +174,7 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +1/-0 (1 lines); hunks: -174,6 +174,7 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -174,6 +174,7 @@ export const DeepSeekV4Deployment = () => {
+    "gb300|big|low-latency",
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +1/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23778 - ci(deepseek-v4): add b300/grace-blackwell dev-branch build options

- 链接: https://github.com/sgl-project/sglang/pull/23778
- 状态/时间: merged / 2026-04-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+23/-5，可读 patch 58 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci(deepseek-v4): add b300/grace-blackwell dev-branch build options」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `.github/workflows/release-docker-deepseek-v4.yml`；技术摘要: 覆盖「ci(deepseek-v4): add b300/grace-blackwell dev-branch build options」；主要实现面是 `.github/workflows/release-docker-deepseek-v4.yml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.github/workflows/release-docker-deepseek-v4.yml` modified +23/-5 (28 lines); hunks: -32,6 +32,16 @@ on:; -50,19 +60,27 @@ jobs:。
- 代码 diff 细节:
  - `.github/workflows/release-docker-deepseek-v4.yml` modified +23/-5 (28 lines); hunks: -32,6 +32,16 @@ on:; -50,19 +60,27 @@ jobs:
- 关键代码摘录:

```diff
diff -- .github/workflows/release-docker-deepseek-v4.yml
@@ -32,6 +32,16 @@ on:
+      build_b300_dev:
+        description: "Build and push the B300 image from the deepseek_v4_dev branch."
+        required: false
+        type: boolean
+        default: true
+      build_grace_blackwell_dev:
```

- 已读文件:
  - ci: `.github/workflows/release-docker-deepseek-v4.yml` modified +23/-5
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #23787 - amd/deepseek_v4 integration 1/N - 0426

- 链接: https://github.com/sgl-project/sglang/pull/23787
- 状态/时间: merged / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 128 个文件，+18341/-879，可读 patch 18279 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「amd/deepseek_v4 integration 1/N - 0426」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py`, `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`；技术摘要: 覆盖「amd/deepseek_v4 integration 1/N - 0426」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py`, `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` added +2803/-0 (2803 lines)；`python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py` added +1330/-0 (1330 lines); hunks: -0,0 +1,1330; symbols: _copy_metadata, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadataRadix，涉及 `_copy_metadata, _create_flashmla_metadata, _create_dummy_paged_compress_data`；`python/sglang/srt/entrypoints/openai/encoding_dsv4.py` added +840/-0 (840 lines); hunks: -0,0 +1,840; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format，涉及 `to_json, tools_from_openai_format, tool_calls_from_openai_format`；`python/sglang/srt/layers/mhc.py` added +686/-0 (686 lines); hunks: -0,0 +1,686; symbols: hc_split_sinkhorn_kernel, hc_split_sinkhorn_kernel_, hc_split_sinkhorn, mhc_pre_big_fuse_tilelang，涉及 `hc_split_sinkhorn_kernel, hc_split_sinkhorn_kernel_, hc_split_sinkhorn`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` added +2803/-0 (2803 lines)
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py` added +1330/-0 (1330 lines); hunks: -0,0 +1,1330; symbols: _copy_metadata, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadataRadix
  - `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` added +840/-0 (840 lines); hunks: -0,0 +1,840; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `python/sglang/srt/layers/mhc.py` added +686/-0 (686 lines); hunks: -0,0 +1,686; symbols: hc_split_sinkhorn_kernel, hc_split_sinkhorn_kernel_, hc_split_sinkhorn, mhc_pre_big_fuse_tilelang
  - `python/sglang/srt/layers/attention/compressed/indexer.py` added +616/-0 (616 lines); hunks: -0,0 +1,616; symbols: fp8_paged_mqa_logits_torch, topk_transform_512_pytorch_vectorized, _fused_scale_kernel, fused_scale
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py
@@ -0,0 +1,1330 @@
+"""
+Some comments on the common terms used in DeepSeekV4Backend:
+topk_lengths:
+    NOTE: TL;DR: topk_lengths == seq_lens
+    The FlashMLA sparse decode kernel will attend to `k` tokens for each query.
+    `topk_lengths` indicates how many tokens each query will attend to.
diff -- python/sglang/srt/entrypoints/openai/encoding_dsv4.py
@@ -0,0 +1,840 @@
+# Adapted from the DeepSeek-V4 release reference implementation.
+"""
+DeepSeek-V4 Encoding
+A self-contained implementation for encoding/decoding DeepSeek-V4 chat messages
+with tool calling, thinking mode, and quick instruction task support.
+"""
diff -- python/sglang/srt/layers/mhc.py
@@ -0,0 +1,686 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` added +2803/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py` added +1330/-0; `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` added +840/-0; `python/sglang/srt/layers/mhc.py` added +686/-0; `python/sglang/srt/layers/attention/compressed/indexer.py` added +616/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` added +591/-0
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_activation.py`, `python/sglang/srt/flashmla_tests/__init__.py`, `python/sglang/srt/flashmla_tests/kernelkit/.gitignore`, `python/sglang/srt/flashmla_tests/kernelkit/__init__.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23776 - [DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP

- 链接: https://github.com/sgl-project/sglang/pull/23776
- 状态/时间: merged / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-0，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v2.py`；技术摘要: 覆盖「[DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP」；主要实现面是 `python/sglang/srt/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +10/-0 (10 lines); hunks: -227,9 +227,11 @@ def __init__(; -283,6 +285,12 @@ def forward(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +10/-0 (10 lines); hunks: -227,9 +227,11 @@ def __init__(; -283,6 +285,12 @@ def forward(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -227,9 +227,11 @@ def __init__(
+        swiglu_limit: Optional[float] = None,
+        self.swiglu_limit = swiglu_limit
@@ -283,6 +285,12 @@ def forward(
+        if self.swiglu_limit is not None:
+            _g, _u = gate_up.chunk(2, dim=-1)
+            _lim = float(self.swiglu_limit)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +10/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23817 - docs: verify GB300 Pro DeepSeek V4 recipes

- 链接: https://github.com/sgl-project/sglang/pull/23817
- 状态/时间: merged / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-0，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: verify GB300 Pro DeepSeek V4 recipes」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「docs: verify GB300 Pro DeepSeek V4 recipes」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +6/-0 (6 lines); hunks: -182,7 +182,9 @@ export const DeepSeekV4Deployment = () => {; -365,6 +367,8 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +6/-0 (6 lines); hunks: -182,7 +182,9 @@ export const DeepSeekV4Deployment = () => {; -365,6 +367,8 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -182,7 +182,9 @@ export const DeepSeekV4Deployment = () => {
+    "gb300|big|balanced",
+    "gb300|big|max-throughput",
@@ -365,6 +367,8 @@ export const DeepSeekV4Deployment = () => {
+      } else if (isBig && hardware === "gb300") {
+        flags.push("  --mem-fraction-static 0.9");
@@ -401,6 +405,8 @@ export const DeepSeekV4Deployment = () => {
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +6/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23810 - Add benchmarking scripts for deepseek v4

- 链接: https://github.com/sgl-project/sglang/pull/23810
- 状态/时间: merged / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+243/-0，可读 patch 244 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add benchmarking scripts for deepseek v4」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `scripts/bench_gpqa_aime.py`；技术摘要: 覆盖「Add benchmarking scripts for deepseek v4」；主要实现面是 `scripts/bench_gpqa_aime.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `scripts/bench_gpqa_aime.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: _venv_cmd, get_timestamp, get_random_int, setup_ns，涉及 `_venv_cmd, get_timestamp, get_random_int`。
- 代码 diff 细节:
  - `scripts/bench_gpqa_aime.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: _venv_cmd, get_timestamp, get_random_int, setup_ns
- 关键代码摘录:

```diff
diff -- scripts/bench_gpqa_aime.py
@@ -0,0 +1,243 @@
+# This script should be used inside the container. Before testing anything, please
+# 1. install typer
+# 2. set the following environment variables:
+# - HOST: the host to connect to (default 127.0.0.1)
+# - PORT: the port to connect to (default 30010)
+# - HF_TOKEN: needed for `setup-ns`
```

- 已读文件:
  - other: `scripts/bench_gpqa_aime.py` added +243/-0
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #23832 - amd/deepseek_v4 integration 2/N - cuda graph 0426

- 链接: https://github.com/sgl-project/sglang/pull/23832
- 状态/时间: merged / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 26 个文件，+534/-92，可读 patch 973 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「amd/deepseek_v4 integration 2/N - cuda graph 0426」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/compressed/indexer.py`, `python/sglang/srt/layers/attention/compressed/metadata.py`；技术摘要: 覆盖「amd/deepseek_v4 integration 2/N - cuda graph 0426」；主要实现面是 `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/compressed/indexer.py`, `python/sglang/srt/layers/attention/compressed/metadata.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +395/-1 (396 lines); hunks: -1,5 +1,5; -27,6 +27,7; symbols: fast_log2_ceil, tilelang_sparse_fwd, _next_power_of_2, _padded_H，涉及 `fast_log2_ceil, tilelang_sparse_fwd, _next_power_of_2`；`python/sglang/srt/layers/attention/compressed/indexer.py` modified +78/-76 (154 lines); hunks: -1,6 +1,6; -37,6 +37,8; symbols: fp8_paged_mqa_logits_torch，涉及 `fp8_paged_mqa_logits_torch`；`python/sglang/srt/layers/attention/compressed/metadata.py` modified +12/-11 (23 lines); hunks: -169,18 +169,19 @@ def max_seq_len(self) -> int:; symbols: max_seq_len, copy_，涉及 `max_seq_len, copy_`；`python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-1 (10 lines); hunks: -1152,7 +1152,9 @@ def run_once():; -1162,6 +1164,9 @@ def run_once():; symbols: run_once, replay_prepare，涉及 `run_once, replay_prepare`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +395/-1 (396 lines); hunks: -1,5 +1,5; -27,6 +27,7; symbols: fast_log2_ceil, tilelang_sparse_fwd, _next_power_of_2, _padded_H
  - `python/sglang/srt/layers/attention/compressed/indexer.py` modified +78/-76 (154 lines); hunks: -1,6 +1,6; -37,6 +37,8; symbols: fp8_paged_mqa_logits_torch
  - `python/sglang/srt/layers/attention/compressed/metadata.py` modified +12/-11 (23 lines); hunks: -169,18 +169,19 @@ def max_seq_len(self) -> int:; symbols: max_seq_len, copy_
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-1 (10 lines); hunks: -1152,7 +1152,9 @@ def run_once():; -1162,6 +1164,9 @@ def run_once():; symbols: run_once, replay_prepare
  - `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py` modified +7/-0 (7 lines); hunks: -13,6 +13,10 @@ def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):; -32,6 +36,9 @@ def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):; symbols: flash_mla_with_kvcache_entrypoint
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/nsa/tilelang_kernel.py
@@ -1,5 +1,5 @@
-from typing import Optional, Tuple
+from typing import Any, Optional, Tuple
@@ -27,6 +27,7 @@
+INT32 = "int32"
@@ -1375,3 +1376,396 @@ def tilelang_sparse_fwd(
+def _next_power_of_2(x: int) -> int:
diff -- python/sglang/srt/layers/attention/compressed/indexer.py
@@ -1,6 +1,6 @@
-from typing import TYPE_CHECKING, Any, List, Optional, Tuple
+from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
@@ -37,6 +37,8 @@
+_arange_cache: Dict[str, torch.Tensor] = {}
@@ -48,6 +50,8 @@ def fp8_paged_mqa_logits_torch(
+    """Vectorized implementation that avoids .item() and Python loops,
diff -- python/sglang/srt/layers/attention/compressed/metadata.py
@@ -169,18 +169,19 @@ def max_seq_len(self) -> int:
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +395/-1; `python/sglang/srt/layers/attention/compressed/indexer.py` modified +78/-76; `python/sglang/srt/layers/attention/compressed/metadata.py` modified +12/-11; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-1; `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py` modified +7/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +4/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/attention/base_attn_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23756 - feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch

- 链接: https://github.com/sgl-project/sglang/pull/23756
- 状态/时间: merged / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+47/-12，可读 patch 90 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/environ.py`；技术摘要: 覆盖「feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch」；主要实现面是 `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/environ.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +46/-12 (58 lines); hunks: -22,7 +22,7; -44,14 +44,43 @@ def update_deep_gemm_config(gpu_id: int, server_args: Server...; symbols: update_deep_gemm_config, _compile_deep_gemm_one_type_all，涉及 `update_deep_gemm_config, _compile_deep_gemm_one_type_all`；`python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -336,6 +336,7 @@ class Envs:; symbols: Envs，涉及 `Envs`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +46/-12 (58 lines); hunks: -22,7 +22,7; -44,14 +44,43 @@ def update_deep_gemm_config(gpu_id: int, server_args: Server...; symbols: update_deep_gemm_config, _compile_deep_gemm_one_type_all
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -336,6 +336,7 @@ class Envs:; symbols: Envs
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py
@@ -22,7 +22,7 @@
-_BUILTIN_M_LIST = list(range(1, 1024 * 16 + 1))
+_BUILTIN_M_LIST: List[int] = []
@@ -44,14 +44,43 @@ def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
-    # Generate m_max
-    m_max = 1024 * 16
-    if server_args.chunked_prefill_size < 1:
diff -- python/sglang/srt/environ.py
@@ -336,6 +336,7 @@ class Envs:
+    SGLANG_JIT_DEEPGEMM_FAST_WARMUP = EnvBool(False)
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +46/-12; `python/sglang/srt/environ.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/environ.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23883 - Enable DeepGemm warmup in DeepSeek-V4 cookbook

- 链接: https://github.com/sgl-project/sglang/pull/23883
- 状态/时间: merged / 2026-04-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-5，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable DeepGemm warmup in DeepSeek-V4 cookbook」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「Enable DeepGemm warmup in DeepSeek-V4 cookbook」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-5 (8 lines); hunks: -255,7 +255,6 @@ export const DeepSeekV4Deployment = () => {; -461,8 +460,8 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-5 (8 lines); hunks: -255,7 +255,6 @@ export const DeepSeekV4Deployment = () => {; -461,8 +460,8 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -255,7 +255,6 @@ export const DeepSeekV4Deployment = () => {
-    const COMMON_ENV = ["SGLANG_JIT_DEEPGEMM_PRECOMPILE=0"];
@@ -461,8 +460,8 @@ export const DeepSeekV4Deployment = () => {
-    // Assemble: [HW env] [recipe env] [common env] \ sglang serve \ flags...
-    const envAll = [...HW_ENV, ...recipeEnv, ...COMMON_ENV];
+    // Assemble: [HW env] [recipe env] \ sglang serve \ flags...
+    const envAll = [...HW_ENV, ...recipeEnv];
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-5
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23943 - [Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe

- 链接: https://github.com/sgl-project/sglang/pull/23943
- 状态/时间: merged / 2026-04-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+32/-0，可读 patch 39 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「[Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +32/-0 (32 lines); hunks: -482,6 +482,38 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +32/-0 (32 lines); hunks: -482,6 +482,38 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -482,6 +482,38 @@ export const DeepSeekV4Deployment = () => {
+    // H200 Pro low-latency: show BOTH a single-node (TP=8 marlin) variant
+    // and the existing multi-node (TP=16 DP-attn + DeepEP) variant.
+    if (hardware === "h200" && isBig && recipe === "low-latency") {
+      const singleFlags = [
+        "  --trust-remote-code",
+        "  --model-path deepseek-ai/DeepSeek-V4-Pro",
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +32/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23980 - docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4

- 链接: https://github.com/sgl-project/sglang/pull/23980
- 状态/时间: merged / 2026-04-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+84/-8，可读 patch 162 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-3 (82 lines); hunks: -31,6 +31,7 @@ export const DeepSeekV4Deployment = () => {; -70,7 +71,19 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +5/-5 (10 lines); hunks: -1,7 +1,7; -35,7 +35,7 @@ tag: NEW。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-3 (82 lines); hunks: -31,6 +31,7 @@ export const DeepSeekV4Deployment = () => {; -70,7 +71,19 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +5/-5 (10 lines); hunks: -1,7 +1,7; -35,7 +35,7 @@ tag: NEW
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -31,6 +31,7 @@ export const DeepSeekV4Deployment = () => {
+        { id: "h200-fp4", label: "H200 (FP4)", default: false },
@@ -70,7 +71,19 @@ export const DeepSeekV4Deployment = () => {
-  const resolveItems = (option) => option.items;
+  // Recipes that are not supported on the H200 (FP4) Marlin path.
+  const H200_FP4_UNSUPPORTED_RECIPES = new Set(["cp", "pd-disagg"]);
+  const resolveItems = (option, vals) => {
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -1,7 +1,7 @@
-    description: "Deploy DeepSeek-V4 with SGLang — a next-generation MoE model from DeepSeek. Blackwell deployments use the FP4 checkpoint; Hopper deployments use the FP8 checkpoi
+    description: "Deploy DeepSeek-V4 with SGLang — a next-generation MoE model from DeepSeek."
@@ -35,7 +35,7 @@ tag: NEW
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB200 8 GPU (2 nodes) / GB300 4 GPU / H200 16 GPU (2 nodes)</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB200 8 GPU (2 nodes) / GB300 4 GPU / H200 8 GPU(fp4)/16 GPU(fp8)</t
@@ -153,9 +153,9 @@ The generator currently picks values on the **conservative** side (mirroring an
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-3; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +5/-5
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #24035 - [minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4

- 链接: https://github.com/sgl-project/sglang/pull/24035
- 状态/时间: merged / 2026-04-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-3，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-3 (3 lines); hunks: -120,9 +120,6 @@ docker run --gpus all \。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-3 (3 lines); hunks: -120,9 +120,6 @@ docker run --gpus all \
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -120,9 +120,6 @@ docker run --gpus all \
-<Note>
-For H200 GPU deployments, use the SGLang checkpoint under `sgl-project`, not the default DeepSeek checkpoint.
-</Note>
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #24203 - [AMD] Deepseek v4 Flash / Pro nightly tests for MI35x ROCm 7.2

- 链接: https://github.com/sgl-project/sglang/pull/24203
- 状态/时间: merged / 2026-05-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`；关联提交 `5eff3c489a71`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+972/-0，可读 patch 997 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Deepseek v4 Flash / Pro nightly tests for MI35x ROCm 7.2」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`；技术摘要: 覆盖「[AMD] Deepseek v4 Flash / Pro nightly tests for MI35x ROCm 7.2」；主要实现面是 `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/test_deepseek_v4_pro_fp4.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: TestDeepseekV4ProFp4, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV4ProFp4, setUpClass, tearDownClass`；`test/registered/amd/test_deepseek_v4_pro_fp8.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: TestDeepseekV4ProFp8, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV4ProFp8, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: TestDeepseekV4ProFp4, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/test_deepseek_v4_pro_fp8.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: TestDeepseekV4ProFp8, setUpClass, tearDownClass, test_a_gsm8k
- 关键代码摘录:

```diff
diff -- test/registered/amd/test_deepseek_v4_pro_fp4.py
@@ -0,0 +1,209 @@
+"""MI35x DeepSeek-V4-Pro FP4 Test (8-GPU)
+Combined accuracy + performance test for DeepSeek-V4-Pro (1.6T) FP4 on
+MI35x ROCm 7.2.
+- Accuracy: GSM8K few-shot eval
+- Performance: bench_one_batch_server with input_len=8192, output_len=1024 (bs=1)
+Both tests share a single launched server.
diff -- test/registered/amd/test_deepseek_v4_pro_fp8.py
@@ -0,0 +1,209 @@
+"""MI35x DeepSeek-V4-Pro FP8 Test (8-GPU)
+Combined accuracy + performance test for DeepSeek-V4-Pro (1.6T) FP8 on
+MI35x ROCm 7.2.
+- Accuracy: GSM8K few-shot eval
+- Performance: bench_one_batch_server with input_len=8192, output_len=1024 (bs=1)
+Both tests share a single launched server.
```

- 已读文件:
  - tests: `test/registered/amd/test_deepseek_v4_pro_fp4.py` added +209/-0; `test/registered/amd/test_deepseek_v4_pro_fp8.py` added +209/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_fp4.py`, `test/registered/amd/test_deepseek_v4_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24367 - [docs] Update B300 Pro cookbook with accuracy-verified serving configs

- 链接: https://github.com/sgl-project/sglang/pull/24367
- 状态/时间: merged / 2026-05-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+108/-11，可读 patch 195 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Update B300 Pro cookbook with accuracy-verified serving configs」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「[docs] Update B300 Pro cookbook with accuracy-verified serving configs」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +108/-11 (119 lines); hunks: -351,13 +351,41 @@ export const DeepSeekV4Deployment = () => {; -367,6 +395,26 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +108/-11 (119 lines); hunks: -351,13 +351,41 @@ export const DeepSeekV4Deployment = () => {; -367,6 +395,26 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -351,13 +351,41 @@ export const DeepSeekV4Deployment = () => {
+      // B200/B300 Pro accuracy-verified env vars.
+      if (isBig && hardware === "b200") {
+        recipeEnv.push(
+          "SGLANG_JIT_DEEPGEMM_PRECOMPILE=0",
+          "SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1",
+          "SGLANG_OPT_USE_JIT_NORM=1",
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +108/-11
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23882 - Deepseek V4

- 链接: https://github.com/sgl-project/sglang/pull/23882
- 状态/时间: merged / 2026-05-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/configs/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` 等 9 个文件；关联提交 `35870d55aca7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 154 个文件，+24534/-712，可读 patch 27836 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Deepseek V4」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`；技术摘要: 覆盖「Deepseek V4」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` added +1528/-0 (1528 lines); hunks: -0,0 +1,1528; symbols: _rms_normalize_kernel, rms_normalize_triton, MQALayer, __init__，涉及 `_rms_normalize_kernel, rms_normalize_triton, MQALayer`；`python/sglang/srt/layers/attention/deepseek_v4_backend.py` added +1255/-0 (1255 lines); hunks: -0,0 +1,1255; symbols: _pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadata，涉及 `_pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data`；`python/sglang/srt/models/deepseek_v4_nextn.py` added +216/-0 (216 lines); hunks: -0,0 +1,216; symbols: DeepseekV4ModelNextN, __init__, hc_head, forward，涉及 `DeepseekV4ModelNextN, __init__, hc_head`；`python/sglang/srt/configs/deepseek_v4.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: try_detect_fp4_experts, DeepSeekV4Config，涉及 `try_detect_fp4_experts, DeepSeekV4Config`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` added +1528/-0 (1528 lines); hunks: -0,0 +1,1528; symbols: _rms_normalize_kernel, rms_normalize_triton, MQALayer, __init__
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` added +1255/-0 (1255 lines); hunks: -0,0 +1,1255; symbols: _pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadata
  - `python/sglang/srt/models/deepseek_v4_nextn.py` added +216/-0 (216 lines); hunks: -0,0 +1,216; symbols: DeepseekV4ModelNextN, __init__, hc_head, forward
  - `python/sglang/srt/configs/deepseek_v4.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: try_detect_fp4_experts, DeepSeekV4Config
  - `python/sglang/jit_kernel/deepseek_v4.py` added +908/-0 (908 lines); hunks: -0,0 +1,908; symbols: make_name, _jit_common_module, _jit_compress_128_online_plan_module, _jit_compress_128_online_module
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -0,0 +1,1528 @@
+from __future__ import annotations
+import concurrent.futures
+import logging
+from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Set, Tuple
+import torch
+import torch.nn as nn
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -0,0 +1,1255 @@
+from __future__ import annotations
+import enum
+import functools
+import logging
+from dataclasses import dataclass, field
+from typing import (
diff -- python/sglang/srt/models/deepseek_v4_nextn.py
@@ -0,0 +1,216 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` added +1528/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` added +1255/-0; `python/sglang/srt/models/deepseek_v4_nextn.py` added +216/-0; `python/sglang/srt/configs/deepseek_v4.py` added +110/-0; `python/sglang/jit_kernel/deepseek_v4.py` added +908/-0; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` added +738/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/test_utils.py`, `test/manual/dsv4/__init__.py`, `test/manual/dsv4/_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24793 - [DSV4] Cherry pick missing commits from deepseek_v4 branch and enhance tests

- 链接: https://github.com/sgl-project/sglang/pull/24793
- 状态/时间: merged / 2026-05-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+481/-87，可读 patch 873 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Cherry pick missing commits from deepseek_v4 branch and enhance tests」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/unit/entrypoints/openai/test_protocol.py`；技术摘要: 覆盖「[DSV4] Cherry pick missing commits from deepseek_v4 branch and enhance tests」；主要实现面是 `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/unit/entrypoints/openai/test_protocol.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/function_call/test_function_call_parser.py` modified +111/-1 (112 lines); hunks: -31,7 +31,7; -1686,6 +1686,26 @@ def test_get_model_structural_tag(self):; symbols: TestPythonicDetector, test_get_model_structural_tag, test_self_closing_zero_arg_invoke, TestDeepSeekV4Detector，涉及 `TestPythonicDetector, test_get_model_structural_tag, test_self_closing_zero_arg_invoke`；`python/sglang/srt/function_call/deepseekv32_detector.py` modified +26/-10 (36 lines); hunks: -81,8 +81,13 @@ def __init__(self):; -92,6 +97,20 @@ def has_tool_call(self, text: str) -> bool:; symbols: __init__, has_tool_call, _unpack_invoke_match, _parse_parameters_from_xml，涉及 `__init__, has_tool_call, _unpack_invoke_match`；`test/registered/unit/entrypoints/openai/test_protocol.py` modified +31/-0 (31 lines); hunks: -220,6 +220,37 @@ def test_chat_completion_reasoning_effort_none_from_reasoni...; symbols: test_chat_completion_reasoning_effort_none_from_reasoning_dict, test_chat_completion_reasoning_effort_max, test_chat_completion_json_format，涉及 `test_chat_completion_reasoning_effort_none_from_reasoning_dict, test_chat_completion_reasoning_effort_max, test_chat_completion_json_format`；`python/sglang/srt/entrypoints/openai/protocol.py` modified +5/-2 (7 lines); hunks: -633,13 +633,16 @@ class ChatCompletionRequest(BaseModel):; symbols: ChatCompletionRequest，涉及 `ChatCompletionRequest`。
- 代码 diff 细节:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +111/-1 (112 lines); hunks: -31,7 +31,7; -1686,6 +1686,26 @@ def test_get_model_structural_tag(self):; symbols: TestPythonicDetector, test_get_model_structural_tag, test_self_closing_zero_arg_invoke, TestDeepSeekV4Detector
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +26/-10 (36 lines); hunks: -81,8 +81,13 @@ def __init__(self):; -92,6 +97,20 @@ def has_tool_call(self, text: str) -> bool:; symbols: __init__, has_tool_call, _unpack_invoke_match, _parse_parameters_from_xml
  - `test/registered/unit/entrypoints/openai/test_protocol.py` modified +31/-0 (31 lines); hunks: -220,6 +220,37 @@ def test_chat_completion_reasoning_effort_none_from_reasoni...; symbols: test_chat_completion_reasoning_effort_none_from_reasoning_dict, test_chat_completion_reasoning_effort_max, test_chat_completion_json_format
  - `python/sglang/srt/entrypoints/openai/protocol.py` modified +5/-2 (7 lines); hunks: -633,13 +633,16 @@ class ChatCompletionRequest(BaseModel):; symbols: ChatCompletionRequest
  - `scripts/ci/cuda/ci_install_dsv4_dep.sh` added +161/-0 (161 lines); hunks: -0,0 +1,161
- 关键代码摘录:

```diff
diff -- test/registered/unit/function_call/test_function_call_parser.py
@@ -31,7 +31,7 @@
-register_cpu_ci(15, "stage-a-test-cpu")
+register_cpu_ci(est_time=15, suite="stage-a-test-cpu")
@@ -1686,6 +1686,26 @@ def test_get_model_structural_tag(self):
+    def test_self_closing_zero_arg_invoke(self):
+        """V32 inherits the same regex; verify self-closing parses to empty
+        params here too (V32 model rarely emits this shape, but the parser
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -81,8 +81,13 @@ def __init__(self):
+        # Long-form `<｜DSML｜invoke name="x">...</｜DSML｜invoke>` and the
+        # self-closing `<｜DSML｜invoke name="x"/>` shape V4 emits for zero-arg
+        # tools. The `end` group is empty when the closer hasn't streamed in.
-            r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)(</｜DSML｜invoke>|$)'
+            r'<｜DSML｜invoke\s+name="(?P<name>[^"]+)"\s*'
+            r"(?:(?P<self_close>/>)"
diff -- test/registered/unit/entrypoints/openai/test_protocol.py
@@ -220,6 +220,37 @@ def test_chat_completion_reasoning_effort_none_from_reasoning_dict(self):
```

- 已读文件:
  - tests: `test/registered/unit/function_call/test_function_call_parser.py` modified +111/-1; `test/registered/unit/entrypoints/openai/test_protocol.py` modified +31/-0; `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py` renamed +65/-15; `test/registered/dsv4/test_deepseek_v4_flash_fp8_h200.py` renamed +28/-16
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +26/-10; `python/sglang/srt/entrypoints/openai/protocol.py` modified +5/-2; `python/sglang/srt/model_loader/weight_utils.py` modified +33/-3
  - other: `scripts/ci/cuda/ci_install_dsv4_dep.sh` added +161/-0
- 验证与风险: diff 自带测试面 `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp8_h200.py`, `test/registered/unit/entrypoints/openai/test_protocol.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24775 - Optimize MHC pipeline: DeepGemm, fused norm, fused hc_head

- 链接: https://github.com/sgl-project/sglang/pull/24775
- 状态/时间: merged / 2026-05-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+512/-73，可读 patch 699 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Optimize MHC pipeline: DeepGemm, fused norm, fused hc_head」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/mhc.py`, `python/sglang/srt/layers/mhc_head.py`, `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「Optimize MHC pipeline: DeepGemm, fused norm, fused hc_head」；主要实现面是 `python/sglang/srt/layers/mhc.py`, `python/sglang/srt/layers/mhc_head.py`, `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/mhc.py` modified +319/-64 (383 lines); hunks: -7,6 +7,7; -138,12 +139,15 @@ def mhc_pre_big_fuse_tilelang(; symbols: mhc_pre_big_fuse_tilelang, mhc_pre_gemm_sqrsum_splitk_stage_1, _compute_num_split_for_mhc_pre, mhc_pre_big_fuse_with_norm_tilelang，涉及 `mhc_pre_big_fuse_tilelang, mhc_pre_gemm_sqrsum_splitk_stage_1, _compute_num_split_for_mhc_pre`；`python/sglang/srt/layers/mhc_head.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: _hc_head_kernel, fused_hc_head，涉及 `_hc_head_kernel, fused_hc_head`；`python/sglang/srt/models/deepseek_v4.py` modified +40/-9 (49 lines); hunks: -653,7 +653,11 @@ def hc_pre(; -671,11 +675,16 @@ def hc_pre_torch_impl(x, hc_fn):; symbols: hc_pre, hc_pre_torch_impl，涉及 `hc_pre, hc_pre_torch_impl`；`scripts/ci/utils/slash_command_handler.py` modified +2/-0 (2 lines); hunks: -424,6 +424,8 @@ def handle_rerun_stage(; symbols: handle_rerun_stage，涉及 `handle_rerun_stage`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/mhc.py` modified +319/-64 (383 lines); hunks: -7,6 +7,7; -138,12 +139,15 @@ def mhc_pre_big_fuse_tilelang(; symbols: mhc_pre_big_fuse_tilelang, mhc_pre_gemm_sqrsum_splitk_stage_1, _compute_num_split_for_mhc_pre, mhc_pre_big_fuse_with_norm_tilelang
  - `python/sglang/srt/layers/mhc_head.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: _hc_head_kernel, fused_hc_head
  - `python/sglang/srt/models/deepseek_v4.py` modified +40/-9 (49 lines); hunks: -653,7 +653,11 @@ def hc_pre(; -671,11 +675,16 @@ def hc_pre_torch_impl(x, hc_fn):; symbols: hc_pre, hc_pre_torch_impl
  - `scripts/ci/utils/slash_command_handler.py` modified +2/-0 (2 lines); hunks: -424,6 +424,8 @@ def handle_rerun_stage(; symbols: handle_rerun_stage
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/mhc.py
@@ -7,6 +7,7 @@
+from sglang.srt.environ import envs
@@ -138,12 +139,15 @@ def mhc_pre_big_fuse_tilelang(
+    gemm_last_dim: int = -1,
+    if gemm_last_dim < 0:
+        gemm_last_dim = hc_mult3
-    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]
diff -- python/sglang/srt/layers/mhc_head.py
@@ -0,0 +1,151 @@
+"""Fused triton kernel for the DSV4 hc_head LM-head mixer.
+Reference torch implementation (deepseek_v4.py DeepseekV4Model.hc_head):
+    shape, dtype = x.size(), x.dtype
+    x = x.flatten(1).float()
+    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
+    mixes = F.linear(x, hc_fn) * rsqrt
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -653,7 +653,11 @@ def hc_pre(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/mhc.py` modified +319/-64; `python/sglang/srt/layers/mhc_head.py` added +151/-0; `python/sglang/srt/models/deepseek_v4.py` modified +40/-9
  - other: `scripts/ci/utils/slash_command_handler.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/mhc.py`, `python/sglang/srt/layers/mhc_head.py`, `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24825 - [AMD] DSv4 nightly hotfix + schedule-aware --continue-on-error in AMD CI

- 链接: https://github.com/sgl-project/sglang/pull/24825
- 状态/时间: merged / 2026-05-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`；关联提交 `22543b198254`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+110/-110，可读 patch 990 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] DSv4 nightly hotfix + schedule-aware --continue-on-error in AMD CI」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`；技术摘要: 覆盖「[AMD] DSv4 nightly hotfix + schedule-aware --continue-on-error in AMD CI」；主要实现面是 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/test_deepseek_v4_flash_fp4.py` renamed +1/-1 (2 lines); hunks: -82,7 +82,7 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`；`test/registered/amd/test_deepseek_v4_flash_fp8.py` renamed +1/-1 (2 lines); hunks: -82,7 +82,7 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`；`test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +1/-1 (2 lines); hunks: -84,7 +84,7 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`；`test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +1/-1 (2 lines); hunks: -84,7 +84,7 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`。
- 代码 diff 细节:
  - `test/registered/amd/test_deepseek_v4_flash_fp4.py` renamed +1/-1 (2 lines); hunks: -82,7 +82,7 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/amd/test_deepseek_v4_flash_fp8.py` renamed +1/-1 (2 lines); hunks: -82,7 +82,7 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +1/-1 (2 lines); hunks: -84,7 +84,7 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +1/-1 (2 lines); hunks: -84,7 +84,7 @@ def setUpClass(cls):; symbols: setUpClass
- 关键代码摘录:

```diff
diff -- test/registered/amd/test_deepseek_v4_flash_fp4.py
@@ -82,7 +82,7 @@ def setUpClass(cls):
-            "dsv4",
+            "compressed",
diff -- test/registered/amd/test_deepseek_v4_flash_fp8.py
@@ -82,7 +82,7 @@ def setUpClass(cls):
-            "dsv4",
+            "compressed",
diff -- test/registered/amd/test_deepseek_v4_pro_fp4.py
@@ -84,7 +84,7 @@ def setUpClass(cls):
-            "dsv4",
+            "compressed",
diff -- test/registered/amd/test_deepseek_v4_pro_fp8.py
@@ -84,7 +84,7 @@ def setUpClass(cls):
-            "dsv4",
+            "compressed",
```

- 已读文件:
  - tests: `test/registered/amd/test_deepseek_v4_flash_fp4.py` renamed +1/-1; `test/registered/amd/test_deepseek_v4_flash_fp8.py` renamed +1/-1; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +1/-1; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24949 - Deepseek-v4-Pro share expert tp1

- 链接: https://github.com/sgl-project/sglang/pull/24949
- 状态/时间: merged / 2026-05-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+31/-17，可读 patch 112 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Deepseek-v4-Pro share expert tp1」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/environ.py`；技术摘要: 覆盖「Deepseek-v4-Pro share expert tp1」；主要实现面是 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/environ.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +26/-14 (40 lines); hunks: -534,6 +534,7 @@ def __init__(; -543,7 +544,19 @@ def __init__(; symbols: __init__, forward_normal_dual_stream，涉及 `__init__, forward_normal_dual_stream`；`python/sglang/srt/model_executor/model_runner.py` modified +4/-2 (6 lines); hunks: -1155,8 +1155,10 @@ def check_quantized_moe_compatibility(self):; symbols: check_quantized_moe_compatibility，涉及 `check_quantized_moe_compatibility`；`python/sglang/srt/environ.py` modified +1/-1 (2 lines); hunks: -611,7 +611,7 @@ class Envs:; symbols: Envs，涉及 `Envs`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +26/-14 (40 lines); hunks: -534,6 +534,7 @@ def __init__(; -543,7 +544,19 @@ def __init__(; symbols: __init__, forward_normal_dual_stream
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-2 (6 lines); hunks: -1155,8 +1155,10 @@ def check_quantized_moe_compatibility(self):; symbols: check_quantized_moe_compatibility
  - `python/sglang/srt/environ.py` modified +1/-1 (2 lines); hunks: -611,7 +611,7 @@ class Envs:; symbols: Envs
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -534,6 +534,7 @@ def __init__(
+        self._shared_expert_tp1 = False
@@ -543,7 +544,19 @@ def __init__(
-            # disable tp for shared experts when enable deepep moe, or with fp4 allgather
+            # Disable TP for shared experts for A2A/FP4 allgather paths, or when
+            # explicitly requested for DSV4 checkpoints whose shared scales are
+            # not divisible by the global TP size.
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -1155,8 +1155,10 @@ def check_quantized_moe_compatibility(self):
-                moe_intermediate_size // moe_tp_size
-            ) % weight_block_size_n != 0 and not _use_aiter:
+                not envs.SGLANG_SHARED_EXPERT_TP1.get()
+                and (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0
+                and not _use_aiter
+            ):
diff -- python/sglang/srt/environ.py
@@ -611,7 +611,7 @@ class Envs:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +26/-14; `python/sglang/srt/model_executor/model_runner.py` modified +4/-2; `python/sglang/srt/environ.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/environ.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25039 - [AMD] Disable unittest fail-fast for deepseekv4 perf test

- 链接: https://github.com/sgl-project/sglang/pull/25039
- 状态/时间: merged / 2026-05-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`；关联提交 `72b266d59b39`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+56/-8，可读 patch 176 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Disable unittest fail-fast for deepseekv4 perf test」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`；技术摘要: 覆盖「[AMD] Disable unittest fail-fast for deepseekv4 perf test」；主要实现面是 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +14/-2 (16 lines); hunks: -38,24 +38,28; -204,4 +208,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k，涉及 `test_b_perf_8k_1k`；`test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +14/-2 (16 lines); hunks: -38,24 +38,28; -204,4 +208,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k，涉及 `test_b_perf_8k_1k`；`test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +14/-2 (16 lines); hunks: -40,24 +40,28; -206,4 +210,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k，涉及 `test_b_perf_8k_1k`；`test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +14/-2 (16 lines); hunks: -40,24 +40,28; -206,4 +210,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k，涉及 `test_b_perf_8k_1k`。
- 代码 diff 细节:
  - `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +14/-2 (16 lines); hunks: -38,24 +38,28; -204,4 +208,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +14/-2 (16 lines); hunks: -38,24 +38,28; -204,4 +208,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +14/-2 (16 lines); hunks: -40,24 +40,28; -206,4 +210,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +14/-2 (16 lines); hunks: -40,24 +40,28; -206,4 +210,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k
- 关键代码摘录:

```diff
diff -- test/registered/amd/test_deepseek_v4_flash_fp4.py
@@ -38,24 +38,28 @@
-    "SGLANG_OPT_USE_FUSED_COMPRESS": "false",
+    "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
+    "SGLANG_OPT_USE_TRITON_SWA_PREPARE": "true",
+    "SGLANG_OPT_USE_AITER_MHC_PRE": "true",
+    "SGLANG_OPT_USE_AITER_MHC_POST": "true",
+    "AITER_BF16_FP8_MOE_BOUND": "1",
diff -- test/registered/amd/test_deepseek_v4_flash_fp8.py
@@ -38,24 +38,28 @@
-    "SGLANG_OPT_USE_FUSED_COMPRESS": "false",
+    "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
+    "SGLANG_OPT_USE_TRITON_SWA_PREPARE": "true",
+    "SGLANG_OPT_USE_AITER_MHC_PRE": "true",
+    "SGLANG_OPT_USE_AITER_MHC_POST": "true",
+    "AITER_BF16_FP8_MOE_BOUND": "1",
diff -- test/registered/amd/test_deepseek_v4_pro_fp4.py
@@ -40,24 +40,28 @@
```

- 已读文件:
  - tests: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +14/-2; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +14/-2; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +14/-2; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +14/-2
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25152 - docs: prepend SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 for H200 FP8 Flash max-throughput

- 链接: https://github.com/sgl-project/sglang/pull/25152
- 状态/时间: merged / 2026-05-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: prepend SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 for H200 FP8 Flash max-throughput」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「docs: prepend SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 for H200 FP8 Flash max-throughput」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -391,6 +391,9 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -391,6 +391,9 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -391,6 +391,9 @@ export const DeepSeekV4Deployment = () => {
+        if (!isBig) {
+          recipeEnv.push("SGLANG_JIT_DEEPGEMM_PRECOMPILE=0");
+        }
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #24897 - Port fused SiLU+clamp+FP8 quant from DSV4 dev branch

- 链接: https://github.com/sgl-project/sglang/pull/24897
- 状态/时间: merged / 2026-05-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+51/-6，可读 patch 79 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Port fused SiLU+clamp+FP8 quant from DSV4 dev branch」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v2.py`；技术摘要: 覆盖「Port fused SiLU+clamp+FP8 quant from DSV4 dev branch」；主要实现面是 `python/sglang/srt/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunks: -27,6 +27,10; -107,6 +111,9; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunks: -27,6 +27,10; -107,6 +111,9; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -27,6 +27,10 @@
+from sglang.jit_kernel.deepseek_v4 import (
+    silu_and_mul_clamp,
+    silu_and_mul_contig_post_quant,
+)
@@ -107,6 +111,9 @@
+from sglang.srt.layers.quantization.fp8_kernel import (
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +51/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24890 - Port KV Compression V2 from deepseek_v4_dev

- 链接: https://github.com/sgl-project/sglang/pull/24890
- 状态/时间: merged / 2026-05-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `e2290b155aa0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 23 个文件，+5201/-438，可读 patch 6145 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Port KV Compression V2 from deepseek_v4_dev」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/jit_kernel/deepseek_v4.py`；技术摘要: 覆盖「Port KV Compression V2 from deepseek_v4_dev」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/jit_kernel/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +91/-80 (171 lines); hunks: -11,7 +11,11; -25,7 +29,6; symbols: __init__, _compute_q_a, _compute_q_b, _compute_kv_to_cache，涉及 `__init__, _compute_q_a, _compute_q_b`；`python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-5 (20 lines); hunks: -20,11 +20,21；`python/sglang/jit_kernel/deepseek_v4.py` modified +127/-2 (129 lines); hunks: -195,6 +195,52 @@ def _jit_fused_store_module(; -571,6 +617,26 @@ def compress_fused_norm_rope_inplace(; symbols: _jit_fused_store_module, _jit_main_q_norm_rope_module, _jit_main_k_norm_rope_flashmla_module, _jit_main_q_indexer_rope_hadamard_quant_module，涉及 `_jit_fused_store_module, _jit_main_q_norm_rope_module, _jit_main_k_norm_rope_flashmla_module`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +37/-2 (39 lines); hunks: -6,7 +6,7; -630,7 +630,12 @@ def set_swa_key_buffer(; symbols: set_swa_key_buffer, get_extra_key_buffer, get_extra_key_page_size, set_extra_key_buffer，涉及 `set_swa_key_buffer, get_extra_key_buffer, get_extra_key_page_size`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +91/-80 (171 lines); hunks: -11,7 +11,11; -25,7 +29,6; symbols: __init__, _compute_q_a, _compute_q_b, _compute_kv_to_cache
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-5 (20 lines); hunks: -20,11 +20,21
  - `python/sglang/jit_kernel/deepseek_v4.py` modified +127/-2 (129 lines); hunks: -195,6 +195,52 @@ def _jit_fused_store_module(; -571,6 +617,26 @@ def compress_fused_norm_rope_inplace(; symbols: _jit_fused_store_module, _jit_main_q_norm_rope_module, _jit_main_k_norm_rope_flashmla_module, _jit_main_q_indexer_rope_hadamard_quant_module
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +37/-2 (39 lines); hunks: -6,7 +6,7; -630,7 +630,12 @@ def set_swa_key_buffer(; symbols: set_swa_key_buffer, get_extra_key_buffer, get_extra_key_page_size, set_extra_key_buffer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -11,7 +11,11 @@
-from sglang.jit_kernel.deepseek_v4 import fused_rope, rmsnorm_self
+from sglang.jit_kernel.deepseek_v4 import (
+    fused_norm_rope_inplace,
+    fused_q_norm_rope,
+    fused_rope_inplace,
+)
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -20,11 +20,21 @@
-from sglang.srt.layers.attention.dsv4.compressor import (
-    CompressorBackendMixin,
-    FusedCompressMetadata,
-    create_paged_compressor_data,
-)
+if envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
diff -- python/sglang/jit_kernel/deepseek_v4.py
@@ -195,6 +195,52 @@ def _jit_fused_store_module(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +91/-80; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-5; `python/sglang/jit_kernel/deepseek_v4.py` modified +127/-2; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +37/-2
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/deepseek_v4/__init__.py`, `python/sglang/jit_kernel/tests/deepseek_v4/common.py`, `python/sglang/jit_kernel/tests/deepseek_v4/test_c128_v2.py`, `python/sglang/jit_kernel/tests/deepseek_v4/test_c4_v2.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24816 - Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4

- 链接: https://github.com/sgl-project/sglang/pull/24816
- 状态/时间: merged / 2026-05-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+1542/-3，可读 patch 1649 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py`；技术摘要: 覆盖「Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4」；主要实现面是 `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py` added +544/-0 (544 lines); hunks: -0,0 +1,544; symbols: _MockLayer, _MockTopKOutput, __init__, _make_random_mxfp4，涉及 `_MockLayer, _MockTopKOutput, __init__`；`python/sglang/srt/layers/quantization/mxfp4.py` modified +269/-1 (270 lines); hunks: -16,12 +16,18; -62,7 +68,27; symbols: __init__, create_weights, process_weights_after_loading，涉及 `__init__, create_weights, process_weights_after_loading`；`python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: Mxfp4FlashinferCutlassMoEMethod, __init__, create_weights, create_moe_runner，涉及 `Mxfp4FlashinferCutlassMoEMethod, __init__, create_weights`；`python/sglang/srt/layers/moe/topk.py` modified +12/-0 (12 lines); hunks: -243,6 +243,18 @@ class BypassedTopKOutput(NamedTuple):; symbols: BypassedTopKOutput, format, to_standard，涉及 `BypassedTopKOutput, format, to_standard`。
- 代码 diff 细节:
  - `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py` added +544/-0 (544 lines); hunks: -0,0 +1,544; symbols: _MockLayer, _MockTopKOutput, __init__, _make_random_mxfp4
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +269/-1 (270 lines); hunks: -16,12 +16,18; -62,7 +68,27; symbols: __init__, create_weights, process_weights_after_loading
  - `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: Mxfp4FlashinferCutlassMoEMethod, __init__, create_weights, create_moe_runner
  - `python/sglang/srt/layers/moe/topk.py` modified +12/-0 (12 lines); hunks: -243,6 +243,18 @@ class BypassedTopKOutput(NamedTuple):; symbols: BypassedTopKOutput, format, to_standard
  - `python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py` modified +9/-1 (10 lines); hunks: -445,12 +445,20 @@ def maybe_fuse_routed_scale_and_shared_add(; symbols: maybe_fuse_routed_scale_and_shared_add
- 关键代码摘录:

```diff
diff -- test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py
@@ -0,0 +1,544 @@
+"""Unit test for the SM90 cutlass MXFP4 path in :class:`Mxfp4MoEMethod`.
+Builds a single-layer GPT-OSS-style MoE with random MXFP4 weights, drives the
+SGLang plumbing (``_process_weights_for_sm90_cutlass`` + ``_apply_sm90_cutlass``)
+and compares against a direct FlashInfer ``cutlass_fused_moe`` call with the
+same inputs. Both paths invoke the same SM90 kernel from FlashInfer PR #3084,
+so outputs must be bit-exact.
diff -- python/sglang/srt/layers/quantization/mxfp4.py
@@ -16,12 +16,18 @@
+import os
+# Silence the TRT-LLM cutlass autotune trace embedded inside FlashInfer's
+# cutlass_fused_moe. Its C++ logger reads TLLM_LOG_LEVEL on first kernel launch;
+# setdefault preserves any explicit user override.
+os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")
@@ -62,7 +68,27 @@
diff -- python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py
@@ -0,0 +1,263 @@
```

- 已读文件:
  - tests: `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py` added +544/-0; `python/sglang/test/bench_mxfp4_sm90_kernels.py` added +366/-0; `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py` modified +70/-1
  - runtime: `python/sglang/srt/layers/quantization/mxfp4.py` modified +269/-1; `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py` added +263/-0; `python/sglang/srt/layers/moe/topk.py` modified +12/-0; `python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py` modified +9/-1; `python/sglang/srt/layers/quantization/fp8.py` modified +9/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/bench_mxfp4_sm90_kernels.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25001 - [LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support

- 链接: https://github.com/sgl-project/sglang/pull/25001
- 状态/时间: merged / 2026-05-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+1013/-0，可读 patch 1081 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`；技术摘要: 覆盖「[LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support」；主要实现面是 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-0 (15 lines); hunks: -13,6 +13,15; -350,6 +359,8 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core，涉及 `forward_absorb_prepare, forward_absorb_core`；`python/sglang/srt/models/deepseek_v2.py` modified +4/-0 (4 lines); hunks: -1687,11 +1687,15 @@ def prepare_qkv_latent(; symbols: prepare_qkv_latent，涉及 `prepare_qkv_latent`；`python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` added +849/-0 (849 lines); hunks: -0,0 +1,849; symbols: _num_segments, _max_segment_len, _segment_grid_size, _step_a_q_kernel，涉及 `_num_segments, _max_segment_len, _segment_grid_size`；`python/sglang/srt/lora/deepseek_mla_correction.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: is_kv_b_lora_active, _get_state, apply_q_correction, apply_v_correction，涉及 `is_kv_b_lora_active, _get_state, apply_q_correction`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-0 (15 lines); hunks: -13,6 +13,15; -350,6 +359,8 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/models/deepseek_v2.py` modified +4/-0 (4 lines); hunks: -1687,11 +1687,15 @@ def prepare_qkv_latent(; symbols: prepare_qkv_latent
  - `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` added +849/-0 (849 lines); hunks: -0,0 +1,849; symbols: _num_segments, _max_segment_len, _segment_grid_size, _step_a_q_kernel
  - `python/sglang/srt/lora/deepseek_mla_correction.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: is_kv_b_lora_active, _get_state, apply_q_correction, apply_v_correction
  - `python/sglang/srt/lora/utils.py` modified +14/-0 (14 lines); hunks: -134,6 +134,18 @@ def get_hidden_dim(; -274,6 +286,8 @@ def get_target_module_name(full_module_name: str, target_mod...; symbols: get_hidden_dim, get_target_module_name
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -13,6 +13,15 @@
+from sglang.srt.lora.deepseek_mla_correction import (
+    apply_q_correction as apply_kv_b_lora_q_correction,
+)
+from sglang.srt.lora.deepseek_mla_correction import (
+    apply_v_correction as apply_kv_b_lora_v_correction,
+)
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1687,11 +1687,15 @@ def prepare_qkv_latent(
+        # When the module is wrapped with LoRA, the fused GEMM fast-path would
+        # bypass the adapter because it reads weight.T directly.
+        lora_active = getattr(self.fused_qkv_a_proj_with_mqa, "set_lora", False)
+            and not lora_active
diff -- python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py
@@ -0,0 +1,849 @@
+"""Triton kernels for absorbed-MLA ``kv_b_proj`` LoRA correction.
+The absorbed-MLA path bypasses ``kv_b_proj.forward()`` and folds the K/V
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-0; `python/sglang/srt/models/deepseek_v2.py` modified +4/-0; `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` added +849/-0; `python/sglang/srt/lora/deepseek_mla_correction.py` added +117/-0; `python/sglang/srt/lora/utils.py` modified +14/-0; `python/sglang/srt/lora/triton_ops/__init__.py` modified +10/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/lora/deepseek_mla_correction.py`, `python/sglang/srt/lora/triton_ops/__init__.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24986 - [rebase]Deepseek_v4 support w4(mxfp4)a16 on hopper

- 链接: https://github.com/sgl-project/sglang/pull/24986
- 状态/时间: merged / 2026-05-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+146/-36，可读 patch 295 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[rebase]Deepseek_v4 support w4(mxfp4)a16 on hopper」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/mxfp4.py`；技术摘要: 覆盖「[rebase]Deepseek_v4 support w4(mxfp4)a16 on hopper」；主要实现面是 `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/mxfp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py` modified +57/-12 (69 lines); hunks: -8,7 +8,7; -38,17 +38,62 @@ def create_weights(; symbols: create_weights, process_weights_after_loading, apply，涉及 `create_weights, process_weights_after_loading, apply`；`python/sglang/srt/layers/quantization/marlin_utils_fp4.py` modified +32/-16 (48 lines); hunks: -52,22 +52,38 @@ def _normalize_scale_tensor(; -129,19 +145,19 @@ def _permute_bias(bias: torch.Tensor | None) -> torch.Tens...; symbols: _normalize_scale_tensor, _get_optional_param, prepare_moe_mxfp4_layer_for_marlin, _permute_bias，涉及 `_normalize_scale_tensor, _get_optional_param, prepare_moe_mxfp4_layer_for_marlin`；`python/sglang/srt/layers/quantization/mxfp4.py` modified +40/-1 (41 lines); hunks: -35,6 +35,7; -342,6 +343,7 @@ def __init__(; symbols: __init__, create_weights, process_weights_after_loading, create_moe_runner，涉及 `__init__, create_weights, process_weights_after_loading`；`python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +3/-7 (10 lines); hunks: -119,13 +119,9 @@ def fused_marlin_moe(; symbols: fused_marlin_moe，涉及 `fused_marlin_moe`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py` modified +57/-12 (69 lines); hunks: -8,7 +8,7; -38,17 +38,62 @@ def create_weights(; symbols: create_weights, process_weights_after_loading, apply
  - `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` modified +32/-16 (48 lines); hunks: -52,22 +52,38 @@ def _normalize_scale_tensor(; -129,19 +145,19 @@ def _permute_bias(bias: torch.Tensor | None) -> torch.Tens...; symbols: _normalize_scale_tensor, _get_optional_param, prepare_moe_mxfp4_layer_for_marlin, _permute_bias
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +40/-1 (41 lines); hunks: -35,6 +35,7; -342,6 +343,7 @@ def __init__(; symbols: __init__, create_weights, process_weights_after_loading, create_moe_runner
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +3/-7 (10 lines); hunks: -119,13 +119,9 @@ def fused_marlin_moe(; symbols: fused_marlin_moe
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` modified +10/-0 (10 lines); hunks: -1006,6 +1006,16 @@ void moe_wna16_marlin_gemm(
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py
@@ -8,7 +8,7 @@
-from sglang.srt.utils import log_info_on_rank0
+from sglang.srt.utils import log_info_on_rank0, set_weight_attrs
@@ -38,17 +38,62 @@ def create_weights(
-        # Delegate to the underlying FP8 method for weight creation —
-        # the raw weight shapes are the same; only post-loading processing differs.
-        self._fp8.create_weights(
diff -- python/sglang/srt/layers/quantization/marlin_utils_fp4.py
@@ -52,22 +52,38 @@ def _normalize_scale_tensor(
+def _get_optional_param(layer: torch.nn.Module, *names: str) -> torch.Tensor | None:
+    for name in names:
+        value = getattr(layer, name, None)
+        if value is not None:
+            return value
+    return None
diff -- python/sglang/srt/layers/quantization/mxfp4.py
@@ -35,6 +35,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py` modified +57/-12; `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` modified +32/-16; `python/sglang/srt/layers/quantization/mxfp4.py` modified +40/-1; `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +3/-7; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` modified +10/-0
  - tests: `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py` modified +2/-0; `test/registered/dsv4/test_deepseek_v4_flash_fp8_h200.py` modified +2/-0
- 验证与风险: diff 自带测试面 `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp8_h200.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24925 - [attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell)

- 链接: https://github.com/sgl-project/sglang/pull/24925
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+462/-92，可读 patch 726 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell)」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`；技术摘要: 覆盖「[attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell)」；主要实现面是 `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` added +247/-0 (247 lines); hunks: -0,0 +1,247; symbols: _get_tokenspeed_workspace, TokenspeedMLABackend, __init__, _ensure_workspace，涉及 `_get_tokenspeed_workspace, TokenspeedMLABackend, __init__`；`python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +132/-91 (223 lines); hunks: -755,6 +755,109 @@ def unpad_draft_extend_output(; -838,46 +941,13 @@ def forward_decode(; symbols: unpad_draft_extend_output, _compute_decode_bmm1_scale, _run_decode_kernel, _run_prefill_kernel，涉及 `unpad_draft_extend_output, _compute_decode_bmm1_scale, _run_decode_kernel`；`python/sglang/srt/layers/attention/attention_registry.py` modified +11/-0 (11 lines); hunks: -62,6 +62,17 @@ def create_trtllm_mla_backend(runner):; symbols: create_trtllm_mla_backend, create_tokenspeed_mla_backend, create_aiter_backend，涉及 `create_trtllm_mla_backend, create_tokenspeed_mla_backend, create_aiter_backend`；`python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +7/-0 (7 lines); hunks: -134,6 +134,12 @@ def handle_attention_trtllm_mla(attn, forward_batch):; -183,6 +189,7 @@ def handle_attention_intel_xpu(attn, forward_batch):; symbols: handle_attention_trtllm_mla, handle_attention_tokenspeed_mla, handle_attention_aiter, handle_attention_intel_xpu，涉及 `handle_attention_trtllm_mla, handle_attention_tokenspeed_mla, handle_attention_aiter`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` added +247/-0 (247 lines); hunks: -0,0 +1,247; symbols: _get_tokenspeed_workspace, TokenspeedMLABackend, __init__, _ensure_workspace
  - `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +132/-91 (223 lines); hunks: -755,6 +755,109 @@ def unpad_draft_extend_output(; -838,46 +941,13 @@ def forward_decode(; symbols: unpad_draft_extend_output, _compute_decode_bmm1_scale, _run_decode_kernel, _run_prefill_kernel
  - `python/sglang/srt/layers/attention/attention_registry.py` modified +11/-0 (11 lines); hunks: -62,6 +62,17 @@ def create_trtllm_mla_backend(runner):; symbols: create_trtllm_mla_backend, create_tokenspeed_mla_backend, create_aiter_backend
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +7/-0 (7 lines); hunks: -134,6 +134,12 @@ def handle_attention_trtllm_mla(attn, forward_batch):; -183,6 +189,7 @@ def handle_attention_intel_xpu(attn, forward_batch):; symbols: handle_attention_trtllm_mla, handle_attention_tokenspeed_mla, handle_attention_aiter, handle_attention_intel_xpu
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-0 (2 lines); hunks: -244,6 +244,7; -256,6 +257,7
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/tokenspeed_mla_backend.py
@@ -0,0 +1,247 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/sglang/srt/layers/attention/trtllm_mla_backend.py
@@ -755,6 +755,109 @@ def unpad_draft_extend_output(
+    def _compute_decode_bmm1_scale(self, layer: RadixAttention) -> float:
+        """BMM1 scale ``q_scale * k_scale * softmax_scale``. k_scale only
+        applies when the KV cache stores FP8."""
+        q_scale = 1.0
+        if self.data_type == torch.float8_e4m3fn:
+            k_scale = (
diff -- python/sglang/srt/layers/attention/attention_registry.py
@@ -62,6 +62,17 @@ def create_trtllm_mla_backend(runner):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` added +247/-0; `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +132/-91; `python/sglang/srt/layers/attention/attention_registry.py` modified +11/-0; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +7/-0; `python/sglang/srt/model_executor/model_runner.py` modified +2/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/pyproject.toml`, `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25052 - DeepSeek V4 w4a4 MegaMoE

- 链接: https://github.com/sgl-project/sglang/pull/25052
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+212/-60，可读 patch 328 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「DeepSeek V4 w4a4 MegaMoE」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/mega_moe.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py`；技术摘要: 覆盖「DeepSeek V4 w4a4 MegaMoE」；主要实现面是 `python/sglang/srt/layers/moe/mega_moe.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/mega_moe.py` modified +52/-10 (62 lines); hunks: -15,6 +15,7; -34,6 +35,26; symbols: _apply_mega_moe_dg_env, _get_mega_moe_symm_buffer, _run_mega_routed，涉及 `_apply_mega_moe_dg_env, _get_mega_moe_symm_buffer, _run_mega_routed`；`test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py` added +148/-0 (148 lines); hunks: -0,0 +1,148; symbols: _gsm8k_check, TestDSV4FlashFP4B200W4A8MegaMoE, setUpClass, tearDownClass，涉及 `_gsm8k_check, TestDSV4FlashFP4B200W4A8MegaMoE, setUpClass`；`test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py` modified +0/-49 (49 lines); hunks: -31,14 +31,6; -138,46 +130,5 @@ def test_gsm8k(self):; symbols: _gsm8k_check, test_gsm8k, TestDSV4FlashFP4B200MegaMoE, setUpClass，涉及 `_gsm8k_check, test_gsm8k, TestDSV4FlashFP4B200MegaMoE`；`python/sglang/srt/environ.py` modified +11/-0 (11 lines); hunks: -595,6 +595,17 @@ class Envs:; symbols: Envs，涉及 `Envs`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/mega_moe.py` modified +52/-10 (62 lines); hunks: -15,6 +15,7; -34,6 +35,26; symbols: _apply_mega_moe_dg_env, _get_mega_moe_symm_buffer, _run_mega_routed
  - `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py` added +148/-0 (148 lines); hunks: -0,0 +1,148; symbols: _gsm8k_check, TestDSV4FlashFP4B200W4A8MegaMoE, setUpClass, tearDownClass
  - `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py` modified +0/-49 (49 lines); hunks: -31,14 +31,6; -138,46 +130,5 @@ def test_gsm8k(self):; symbols: _gsm8k_check, test_gsm8k, TestDSV4FlashFP4B200MegaMoE, setUpClass
  - `python/sglang/srt/environ.py` modified +11/-0 (11 lines); hunks: -595,6 +595,17 @@ class Envs:; symbols: Envs
  - `python/pyproject.toml` modified +1/-1 (2 lines); hunks: -59,7 +59,7 @@ dependencies = [
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/mega_moe.py
@@ -15,6 +15,7 @@
+import os
@@ -34,6 +35,26 @@
+_MEGA_MOE_DG_ENV_APPLIED = False
+def _apply_mega_moe_dg_env() -> None:
+    """Forward sglang's FP4/MXF4 opt-in flags to DeepGEMM via env vars.
+    DeepGEMM reads `DG_USE_FP4_ACTS` (and `DG_USE_MXF4_KIND`) at host-function
diff -- test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py
@@ -0,0 +1,148 @@
+"""B200 per-commit CI: DeepSeek-V4-Flash FP4 (LowLatency recipe).
+Launches TP=4 with flashinfer_mxfp4 MoE runner + EAGLE speculative decoding.
+Runs 12 ServerSanity probes (correctness, streaming, concurrency, determinism)
+plus a GSM8K accuracy gate.
+Registry: stage-c-test-dsv4-4-gpu-b200 (per-commit, 4x B200)
+"""
diff -- test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py
@@ -31,14 +31,6 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/mega_moe.py` modified +52/-10; `python/sglang/srt/environ.py` modified +11/-0; `python/pyproject.toml` modified +1/-1
  - tests: `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py` added +148/-0; `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py` modified +0/-49
- 验证与风险: diff 自带测试面 `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25243 - [Docs] update dsv4 cookbook with H100 deployment commands

- 链接: https://github.com/sgl-project/sglang/pull/25243
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+83/-9，可读 patch 153 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] update dsv4 cookbook with H100 deployment commands」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[Docs] update dsv4 cookbook with H100 deployment commands」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-9 (88 lines); hunks: -7,6 +7,7 @@ export const DeepSeekV4Deployment = () => {; -32,6 +33,7 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -100,6 +100,10 @@ Please refer to the [official SGLang installation guide](.....。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-9 (88 lines); hunks: -7,6 +7,7 @@ export const DeepSeekV4Deployment = () => {; -32,6 +33,7 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -100,6 +100,10 @@ Please refer to the [official SGLang installation guide](.....
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -7,6 +7,7 @@ export const DeepSeekV4Deployment = () => {
+  //     H100  → FP4 weights (Marlin), Flash TP=8 single-node / Pro TP=16 2-node
@@ -32,6 +33,7 @@ export const DeepSeekV4Deployment = () => {
+        { id: "h100", label: "H100 (FP4)", default: false },
@@ -71,14 +73,17 @@ export const DeepSeekV4Deployment = () => {
-  // Recipes that are not supported on the H200 (FP4) Marlin path.
-  const H200_FP4_UNSUPPORTED_RECIPES = new Set(["cp", "pd-disagg"]);
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -100,6 +100,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+    <tr>
+      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>NVIDIA H100</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>lmsysorg/sglang:dev</code></td>
+    </tr>
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-9; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #24691 - [UnifiedTree]: Support HiCache For DeepSeek_V4

- 链接: https://github.com/sgl-project/sglang/pull/24691
- 状态/时间: merged / 2026-05-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；关联提交 `d9fa84b25b79`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+1221/-154，可读 patch 1970 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[UnifiedTree]: Support HiCache For DeepSeek_V4」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「[UnifiedTree]: Support HiCache For DeepSeek_V4」；主要实现面是 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +12/-1 (13 lines); hunks: -605,21 +605,28 @@ def _init_compressed_layer_mapping(self):; -635,7 +642,8 @@ def get_extra_key_page_size(self, layer_id: int) -> int:; symbols: _init_compressed_layer_mapping, wait_layer_transfer, get_attention_compress_states, get_indexer_compress_states，涉及 `_init_compressed_layer_mapping, wait_layer_transfer, get_attention_compress_states`。
- 代码 diff 细节:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +12/-1 (13 lines); hunks: -605,21 +605,28 @@ def _init_compressed_layer_mapping(self):; -635,7 +642,8 @@ def get_extra_key_page_size(self, layer_id: int) -> int:; symbols: _init_compressed_layer_mapping, wait_layer_transfer, get_attention_compress_states, get_indexer_compress_states
- 关键代码摘录:

```diff
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -605,21 +605,28 @@ def _init_compressed_layer_mapping(self):
+    def wait_layer_transfer(self, layer_id: int) -> None:
+        if self.layer_transfer_counter is not None:
+            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
+        self.wait_layer_transfer(layer_id)
+        self.wait_layer_transfer(layer_id)
+        self.wait_layer_transfer(layer_id)
```

- 已读文件:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +12/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/kl_multiturn_utils.py`, `python/sglang/test/kl_test_utils.py`, `test/registered/radix_cache/test_unified_radix_cache_kl.py`, `test/registered/radix_cache/test_unified_radix_hicache_kl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25369 - Add hicache feature in dsv4 cookbook

- 链接: https://github.com/sgl-project/sglang/pull/25369
- 状态/时间: merged / 2026-05-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+50/-4，可读 patch 95 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add hicache feature in dsv4 cookbook」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「Add hicache feature in dsv4 cookbook」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +39/-4 (43 lines); hunks: -71,6 +71,14 @@ export const DeepSeekV4Deployment = () => {; -295,7 +303,7 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +11/-0 (11 lines); hunks: -334,6 +334,17 @@ print()。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +39/-4 (43 lines); hunks: -71,6 +71,14 @@ export const DeepSeekV4Deployment = () => {; -295,7 +303,7 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +11/-0 (11 lines); hunks: -334,6 +334,17 @@ print()
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -71,6 +71,14 @@ export const DeepSeekV4Deployment = () => {
+    hicache: {
+      name: "hicache",
+      title: "HiCache",
+      items: [
+        { id: "disabled", label: "Disabled", default: true  },
+        { id: "l2",       label: "L2",       default: false, subtitle: "GPU+CPU" },
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -334,6 +334,17 @@ print()
+#### 4.2.3 HiCache (Hierarchical KV Caching)
+HiCache enables multi-tier KV cache offloading (GPU → CPU → Storage), significantly expanding effective context capacity for long-context and multi-turn scenarios. Combined with U
+To enable HiCache, use the **HiCache** toggle in the [command generator above](#3-model-deployment):
+- **L2 (GPU + CPU):** Offloads cold KV pages to CPU memory. Enables `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` for intelligent hierarchical prefix caching.
+- **L3 (GPU + CPU + Storage):** Coming soon.
+For more details, see the [HiCache documentation](../../../docs/advanced_features/hicache).
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +39/-4; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +11/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25419 - Port SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN from deepseek_v4_dev

- 链接: https://github.com/sgl-project/sglang/pull/25419
- 状态/时间: merged / 2026-05-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-1，可读 patch 23 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Port SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN from deepseek_v4_dev」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/environ.py`；技术摘要: 覆盖「Port SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN from deepseek_v4_dev」；主要实现面是 `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/environ.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/managers/schedule_batch.py` modified +5/-1 (6 lines); hunks: -2724,9 +2724,13 @@ def _evict_swa(self, req: Req, pre_len: int):; symbols: _evict_swa，涉及 `_evict_swa`；`python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -596,6 +596,7 @@ class Envs:; symbols: Envs，涉及 `Envs`。
- 代码 diff 细节:
  - `python/sglang/srt/managers/schedule_batch.py` modified +5/-1 (6 lines); hunks: -2724,9 +2724,13 @@ def _evict_swa(self, req: Req, pre_len: int):; symbols: _evict_swa
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -596,6 +596,7 @@ class Envs:; symbols: Envs
- 关键代码摘录:

```diff
diff -- python/sglang/srt/managers/schedule_batch.py
@@ -2724,9 +2724,13 @@ def _evict_swa(self, req: Req, pre_len: int):
+        if envs.SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN.get():
+            evict_threshold = pre_len - sliding_window_size
+        else:
+            evict_threshold = pre_len - sliding_window_size - self.tree_cache.page_size
-            pre_len - sliding_window_size - self.tree_cache.page_size,
+            evict_threshold,
diff -- python/sglang/srt/environ.py
@@ -596,6 +596,7 @@ class Envs:
+    SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN = EnvBool(False)
```

- 已读文件:
  - runtime: `python/sglang/srt/managers/schedule_batch.py` modified +5/-1; `python/sglang/srt/environ.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/environ.py`, `python/sglang/srt/managers/schedule_batch.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24704 - feat: add Pipeline Parallelism (PP) and PD support for DeepSeek-V4

- 链接: https://github.com/sgl-project/sglang/pull/24704
- 状态/时间: merged / 2026-05-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `162540e0a8d3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+341/-103，可读 patch 750 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: add Pipeline Parallelism (PP) and PD support for DeepSeek-V4」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「feat: add Pipeline Parallelism (PP) and PD support for DeepSeek-V4」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +99/-39 (138 lines); hunks: -2,7 +2,16; -49,7 +58,7; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +71/-51 (122 lines); hunks: -401,6 +401,19 @@ def __init__(; -412,8 +425,8 @@ def __init__(; symbols: __init__, register_mapping, get_state_buf_infos，涉及 `__init__, register_mapping, get_state_buf_infos`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +99/-39 (138 lines); hunks: -2,7 +2,16; -49,7 +58,7; symbols: __init__, forward
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +71/-51 (122 lines); hunks: -401,6 +401,19 @@ def __init__(; -412,8 +425,8 @@ def __init__(; symbols: __init__, register_mapping, get_state_buf_infos
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -2,7 +2,16 @@
-from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Set, Tuple
+from typing import (
+    TYPE_CHECKING,
+    Iterable,
+    List,
+    Literal,
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -401,6 +401,19 @@ def __init__(
+        # Determine this PP stage's absolute layer range
+        if (
+            start_layer is not None
+            and end_layer is not None
+            and len(compression_ratios) >= end_layer
+        ):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +99/-39; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +71/-51
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/disaggregation/base/conn.py`, `python/sglang/srt/disaggregation/common/conn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25477 - [BugFix]: Fix DeepSeek V4 HiCache layer count logic

- 链接: https://github.com/sgl-project/sglang/pull/25477
- 状态/时间: merged / 2026-05-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+161/-144，可读 patch 349 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix]: Fix DeepSeek V4 HiCache layer count logic」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`；技术摘要: 覆盖「[BugFix]: Fix DeepSeek V4 HiCache layer count logic」；主要实现面是 `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: TestUnifiedMambaHiCache, setUpClass, tearDownClass, _assert_dsv4_decode_cached_tokens，涉及 `TestUnifiedMambaHiCache, setUpClass, tearDownClass`；`test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py` renamed +0/-141 (141 lines); hunks: -13,162 +13,21; symbols: TestUnifiedMambaHiCache, setUpClass, tearDownClass, _assert_dsv4_decode_cached_tokens，涉及 `TestUnifiedMambaHiCache, setUpClass, tearDownClass`；`python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +6/-3 (9 lines); hunks: -283,7 +283,8 @@ def build_deepseek_v4_hicache_stack(; -293,7 +294,9 @@ def build_deepseek_v4_hicache_stack(; symbols: build_deepseek_v4_hicache_stack, attach_hybrid_pool_to_unified_cache，涉及 `build_deepseek_v4_hicache_stack, attach_hybrid_pool_to_unified_cache`。
- 代码 diff 细节:
  - `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: TestUnifiedMambaHiCache, setUpClass, tearDownClass, _assert_dsv4_decode_cached_tokens
  - `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py` renamed +0/-141 (141 lines); hunks: -13,162 +13,21; symbols: TestUnifiedMambaHiCache, setUpClass, tearDownClass, _assert_dsv4_decode_cached_tokens
  - `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +6/-3 (9 lines); hunks: -283,7 +283,8 @@ def build_deepseek_v4_hicache_stack(; -293,7 +294,9 @@ def build_deepseek_v4_hicache_stack(; symbols: build_deepseek_v4_hicache_stack, attach_hybrid_pool_to_unified_cache
- 关键代码摘录:

```diff
diff -- test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py
@@ -0,0 +1,155 @@
+import unittest
+from test_unified_radix_cache_kl import UnifiedRadixTreeTestMixin
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.kl_multiturn_utils import (
+    get_input_ids,
diff -- test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py
@@ -13,162 +13,21 @@
-from test_unified_radix_cache_kl import UnifiedRadixTreeTestMixin
-from sglang.test.kl_multiturn_utils import (
-    get_input_ids,
-    make_mamba_decode_assert,
-    make_mamba_prefill_assert,
-)
diff -- python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py
@@ -283,7 +283,8 @@ def build_deepseek_v4_hicache_stack(
```

- 已读文件:
  - tests: `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` added +155/-0; `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py` renamed +0/-141
  - runtime: `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +6/-3
- 验证与风险: diff 自带测试面 `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25410 - [Docs] Update DeepSeek V4 cookbook to use the latest docker image

- 链接: https://github.com/sgl-project/sglang/pull/25410
- 状态/时间: merged / 2026-05-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-41，可读 patch 63 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Update DeepSeek V4 cookbook to use the latest docker image」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[Docs] Update DeepSeek V4 cookbook to use the latest docker image」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-41 (47 lines); hunks: -66,48 +66,13 @@ SGLang offers multiple installation methods. Choose based on...; -116,7 +81,7 @@ docker run --gpus all \。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-41 (47 lines); hunks: -66,48 +66,13 @@ SGLang offers multiple installation methods. Choose based on...; -116,7 +81,7 @@ docker run --gpus all \
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -66,48 +66,13 @@ SGLang offers multiple installation methods. Choose based on your hardware platf
-**Docker Images by Hardware Platform:**
+**Docker Image:** Use `lmsysorg/sglang:latest` for all supported hardware platforms (B300 / B200 / GB200 / GB300 / H200 / H100).
-<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
-  <colgroup>
-    <col style={{width: "55%"}} />
-    <col style={{width: "45%"}} />
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-41
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25412 - [Doc] DSV4 cookbook: clean up env vars, add MegaMoE toggle, unify docker image

- 链接: https://github.com/sgl-project/sglang/pull/25412
- 状态/时间: merged / 2026-05-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+38/-83，可读 patch 185 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc] DSV4 cookbook: clean up env vars, add MegaMoE toggle, unify docker image」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「[Doc] DSV4 cookbook: clean up env vars, add MegaMoE toggle, unify docker image」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +38/-83 (121 lines); hunks: -79,6 +79,15 @@ export const DeepSeekV4Deployment = () => {; -303,7 +312,7 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +38/-83 (121 lines); hunks: -79,6 +79,15 @@ export const DeepSeekV4Deployment = () => {; -303,7 +312,7 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -79,6 +79,15 @@ export const DeepSeekV4Deployment = () => {
+    megamoe: {
+      name: "megamoe",
+      title: "MegaMoE",
+      items: [
+        { id: "disabled", label: "Disabled", default: true  },
+        { id: "w4a8",     label: "W4A8",     default: false },
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +38/-83
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25506 - [Doc] Fix several places for dpsk v4 cookbook

- 链接: https://github.com/sgl-project/sglang/pull/25506
- 状态/时间: merged / 2026-05-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+47/-1，可读 patch 83 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc] Fix several places for dpsk v4 cookbook」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[Doc] Fix several places for dpsk v4 cookbook」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +26/-0 (26 lines); hunks: -96,6 +96,15 @@ export const DeepSeekV4Deployment = () => {; -104,6 +113,14 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +21/-1 (22 lines); hunks: -120,14 +120,34 @@ The generator currently picks values on the **conservative...。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +26/-0 (26 lines); hunks: -96,6 +96,15 @@ export const DeepSeekV4Deployment = () => {; -104,6 +113,14 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +21/-1 (22 lines); hunks: -120,14 +120,34 @@ The generator currently picks values on the **conservative...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -96,6 +96,15 @@ export const DeepSeekV4Deployment = () => {
+  // MegaMoE is only supported on Blackwell with DeepEP-based recipes
+  // (balanced / max-throughput / pd-disagg). It's disabled on Hopper
+  // (H100 / H200 / H200-FP4) and on low-latency / cp recipes.
+  const MEGAMOE_UNSUPPORTED_RECIPES = new Set(["low-latency", "cp"]);
+  const MEGAMOE_UNSUPPORTED_HARDWARE = new Set(["h100", "h200", "h200-fp4"]);
+  const isMegamoeUnsupported = (vals) =>
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -120,14 +120,34 @@ The generator currently picks values on the **conservative** side (mirroring an
-- Original FP4 checkpoints: To run original FP4 checkpoints, apply the w4a16 MoE kernels (marlin) as in interactive command generator. For this option we only support TP method. C
+- Original FP4 checkpoints: To run original FP4 checkpoints, we provide two different options for w4a16 MoE kernels: Marlin (`--moe-runner-backend marlin`) and Flashinfer (`--moe-
+**MegaMoE**
+MegaMoE fuses expert dispatch + GEMM into a single kernel for higher throughput
+on MoE layers. To enable it, use the **MegaMoE** toggle in the
+[command generator above](#3-model-deployment) — the generator will swap
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +26/-0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +21/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25569 - Add DeepSeekV4 fused MoE Triton autotune support

- 链接: https://github.com/sgl-project/sglang/pull/25569
- 状态/时间: merged / 2026-05-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-0，可读 patch 29 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add DeepSeekV4 fused MoE Triton autotune support」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `benchmark/kernels/fused_moe_triton/common_utils.py`；技术摘要: 覆盖「Add DeepSeekV4 fused MoE Triton autotune support」；主要实现面是 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `benchmark/kernels/fused_moe_triton/common_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +5/-0 (5 lines); hunks: -35,6 +35,7; -174,8 +175,12 @@ def prepare(i: int):; symbols: prepare, run，涉及 `prepare, run`；`benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunks: -85,6 +85,7 @@ def get_model_config(; symbols: get_model_config，涉及 `get_model_config`。
- 代码 diff 细节:
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +5/-0 (5 lines); hunks: -35,6 +35,7; -174,8 +175,12 @@ def prepare(i: int):; symbols: prepare, run
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunks: -85,6 +85,7 @@ def get_model_config(; symbols: get_model_config
- 关键代码摘录:

```diff
diff -- benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py
@@ -35,6 +35,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_config
@@ -174,8 +175,12 @@ def prepare(i: int):
+        model_config = get_config(args.model, trust_remote_code=True)
+        architecture = model_config.architectures[0]
+        is_dsv4 = architecture == "DeepseekV4ForCausalLM"
+            swiglu_limit=10.0 if is_dsv4 else None,
diff -- benchmark/kernels/fused_moe_triton/common_utils.py
@@ -85,6 +85,7 @@ def get_model_config(
+        "DeepseekV4ForCausalLM",
```

- 已读文件:
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +5/-0; `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #24933 - Amd/deepseek v4 rebase main 0509

- 链接: https://github.com/sgl-project/sglang/pull/24933
- 状态/时间: merged / 2026-05-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `866793c502b7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+3678/-70，可读 patch 4186 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Amd/deepseek v4 rebase main 0509」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`；技术摘要: 覆盖「Amd/deepseek v4 rebase main 0509」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` added +1265/-0 (1265 lines); hunks: -0,0 +1,1265; symbols: _pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadata，涉及 `_pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data`；`python/sglang/srt/models/deepseek_v4.py` modified +53/-5 (58 lines); hunks: -58,6 +58,7; -76,6 +77,12; symbols: __init__, _forward_prepare_multi_stream, _forward_prepare，涉及 `__init__, _forward_prepare_multi_stream, _forward_prepare`；`python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +88/-21 (109 lines); hunks: -7,8 +7,11; -22,16 +25,55 @@ def kv(self) -> torch.Tensor:; symbols: KVAndScore, kv, score, shape，涉及 `KVAndScore, kv, score`；`python/sglang/jit_kernel/deepseek_v4.py` modified +26/-0 (26 lines); hunks: -13,6 +13,13; -644,6 +651,23 @@ def fused_rope(; symbols: fused_rope, _dispatch_bf16_fp32_backend，涉及 `fused_rope, _dispatch_bf16_fp32_backend`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` added +1265/-0 (1265 lines); hunks: -0,0 +1,1265; symbols: _pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadata
  - `python/sglang/srt/models/deepseek_v4.py` modified +53/-5 (58 lines); hunks: -58,6 +58,7; -76,6 +77,12; symbols: __init__, _forward_prepare_multi_stream, _forward_prepare
  - `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +88/-21 (109 lines); hunks: -7,8 +7,11; -22,16 +25,55 @@ def kv(self) -> torch.Tensor:; symbols: KVAndScore, kv, score, shape
  - `python/sglang/jit_kernel/deepseek_v4.py` modified +26/-0 (26 lines); hunks: -13,6 +13,13; -644,6 +651,23 @@ def fused_rope(; symbols: fused_rope, _dispatch_bf16_fp32_backend
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +13/-4 (17 lines); hunks: -18,11 +18,13; -144,6 +146,9 @@ def set_key_buffer_fused(; symbols: get_compress_state_ring_size, set_key_buffer_fused, get_key_buffer, set_kv_buffer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -0,0 +1,1265 @@
+from __future__ import annotations
+import enum
+import functools
+import logging
+from dataclasses import dataclass, field
+from typing import (
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -58,6 +58,7 @@
+from sglang.srt.layers.rotary_embedding import get_rope_wrapper
@@ -76,6 +77,12 @@
+if not _is_hip:
+    from sglang.srt.layers.utils.cp_utils import (
+        prepare_context_parallel_metadata,
+    )
diff -- python/sglang/srt/mem_cache/deepseek_v4_compress_state.py
@@ -7,8 +7,11 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` added +1265/-0; `python/sglang/srt/models/deepseek_v4.py` modified +53/-5; `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +88/-21; `python/sglang/jit_kernel/deepseek_v4.py` modified +26/-0; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +13/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/deepseek_v4.py`, `python/sglang/srt/environ.py`, `python/sglang/srt/layers/attention/attention_registry.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25282 - [UnifiedTree] Support deepseek v4 host pool layout

- 链接: https://github.com/sgl-project/sglang/pull/25282
- 状态/时间: merged / 2026-05-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+401/-114，可读 patch 809 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[UnifiedTree] Support deepseek v4 host pool layout」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/test/kl_multiturn_utils.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`；技术摘要: 覆盖「[UnifiedTree] Support deepseek v4 host pool layout」；主要实现面是 `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/test/kl_multiturn_utils.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/mem_cache/memory_pool_host.py` modified +333/-92 (425 lines); hunks: -1754,6 +1754,7 @@ def __init__(; -1769,7 +1770,7 @@ def __init__(; symbols: __init__, _to_page_indices, _check_io_backend，涉及 `__init__, _to_page_indices, _check_io_backend`；`python/sglang/test/kl_multiturn_utils.py` modified +40/-19 (59 lines); hunks: -2,6 +2,7; -145,30 +146,45 @@ def _interleave_order(n: int, branches_per_group: int) ->...; symbols: _interleave_order, _generate_maybe_interleaved, test_input_output_logprobs_match_decode_cache_hit_helper，涉及 `_interleave_order, _generate_maybe_interleaved, test_input_output_logprobs_match_decode_cache_hit_helper`；`test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` modified +17/-3 (20 lines); hunks: -92,8 +92,13 @@ def _assert_dsv4_decode_cached_tokens(result, history_len, ou...; -129,15 +134,15 @@ def setUpClass(cls):; symbols: _assert_dsv4_decode_cached_tokens, TestUnifiedDeepSeekV4FlashHiCache, setUpClass, tearDownClass，涉及 `_assert_dsv4_decode_cached_tokens, TestUnifiedDeepSeekV4FlashHiCache, setUpClass`；`python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +7/-0 (7 lines); hunks: -325,6 +325,7 @@ def build_deepseek_v4_hicache_stack(; -357,6 +358,7 @@ def build_deepseek_v4_hicache_stack(; symbols: build_deepseek_v4_hicache_stack，涉及 `build_deepseek_v4_hicache_stack`。
- 代码 diff 细节:
  - `python/sglang/srt/mem_cache/memory_pool_host.py` modified +333/-92 (425 lines); hunks: -1754,6 +1754,7 @@ def __init__(; -1769,7 +1770,7 @@ def __init__(; symbols: __init__, _to_page_indices, _check_io_backend
  - `python/sglang/test/kl_multiturn_utils.py` modified +40/-19 (59 lines); hunks: -2,6 +2,7; -145,30 +146,45 @@ def _interleave_order(n: int, branches_per_group: int) ->...; symbols: _interleave_order, _generate_maybe_interleaved, test_input_output_logprobs_match_decode_cache_hit_helper
  - `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` modified +17/-3 (20 lines); hunks: -92,8 +92,13 @@ def _assert_dsv4_decode_cached_tokens(result, history_len, ou...; -129,15 +134,15 @@ def setUpClass(cls):; symbols: _assert_dsv4_decode_cached_tokens, TestUnifiedDeepSeekV4FlashHiCache, setUpClass, tearDownClass
  - `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +7/-0 (7 lines); hunks: -325,6 +325,7 @@ def build_deepseek_v4_hicache_stack(; -357,6 +358,7 @@ def build_deepseek_v4_hicache_stack(; symbols: build_deepseek_v4_hicache_stack
  - `test/registered/radix_cache/test_unified_radix_cache_kl.py` modified +4/-0 (4 lines); hunks: -49,6 +49,8 @@ class UnifiedRadixTreeTestMixin:; -163,6 +165,8 @@ def test_multiturn_decode_cache_hit_branching(self):; symbols: UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching
- 关键代码摘录:

```diff
diff -- python/sglang/srt/mem_cache/memory_pool_host.py
@@ -1754,6 +1754,7 @@ def __init__(
+        layout: str = "layer_first",
@@ -1769,7 +1770,7 @@ def __init__(
-        self.layout = "layer_first"
+        self.layout = layout
@@ -1789,26 +1790,62 @@ def __init__(
-        self.kv_buffer = [
diff -- python/sglang/test/kl_multiturn_utils.py
@@ -2,6 +2,7 @@
+import time
@@ -145,30 +146,45 @@ def _interleave_order(n: int, branches_per_group: int) -> list[int] | None:
-    base_url, inputs, max_new_tokens, order=None, sampling_temperature: float = 1
+    base_url,
+    inputs,
+    max_new_tokens,
diff -- test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py
@@ -92,8 +92,13 @@ def _assert_dsv4_decode_cached_tokens(result, history_len, output_len, label):
```

- 已读文件:
  - runtime: `python/sglang/srt/mem_cache/memory_pool_host.py` modified +333/-92; `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +7/-0
  - tests: `python/sglang/test/kl_multiturn_utils.py` modified +40/-19; `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` modified +17/-3; `test/registered/radix_cache/test_unified_radix_cache_kl.py` modified +4/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/kl_multiturn_utils.py`, `test/registered/radix_cache/test_unified_radix_cache_kl.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25733 - [Bug] Fix V4-Pro NaN on Blackwell by converting fp8_einsum input scale to ue8m0

- 链接: https://github.com/sgl-project/sglang/pull/25733
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `79ea30d1f134`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug] Fix V4-Pro NaN on Blackwell by converting fp8_einsum input scale to ue8m0」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[Bug] Fix V4-Pro NaN on Blackwell by converting fp8_einsum input scale to ue8m0」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +1/-0 (1 lines); hunks: -623,6 +623,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +1/-0 (1 lines); hunks: -623,6 +623,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -623,6 +623,7 @@ def forward(
+            o_s = deep_gemm.ceil_to_ue8m0(o_s)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25396 - fix: fix deepseek v4 CP error

- 链接: https://github.com/sgl-project/sglang/pull/25396
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `7e0818038a45`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: fix deepseek v4 CP error」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「fix: fix deepseek v4 CP error」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +1/-0 (1 lines); hunks: -388,6 +388,7 @@ def _compute_kv_bf16(; symbols: _compute_kv_bf16，涉及 `_compute_kv_bf16`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +1/-0 (1 lines); hunks: -388,6 +388,7 @@ def _compute_kv_bf16(; symbols: _compute_kv_bf16
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -388,6 +388,7 @@ def _compute_kv_bf16(
+        kv = kv.contiguous()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25729 - fix(dsv4): upgrade forward metadata on main stream for large PP size

- 链接: https://github.com/sgl-project/sglang/pull/25729
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `8322fe09a7b6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-0，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(dsv4): upgrade forward metadata on main stream for large PP size」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「fix(dsv4): upgrade forward metadata on main stream for large PP size」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +4/-0 (4 lines); hunks: -1045,6 +1045,10 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +4/-0 (4 lines); hunks: -1045,6 +1045,10 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -1045,6 +1045,10 @@ def forward(
+        # Upgrade lazy raw metadata on the main stream once before any layer
+        # forks alt-streams; later per-layer calls become no-ops.
+        forward_batch.attn_backend._maybe_upgrade_forward_metadata()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24934 - DeepSeek V4 MTP Support CP

- 链接: https://github.com/sgl-project/sglang/pull/24934
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4_nextn.py`；关联提交 `425dffbde339`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+105/-0，可读 patch 163 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「DeepSeek V4 MTP Support CP」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek_v4_nextn.py`；技术摘要: 覆盖「DeepSeek V4 MTP Support CP」；主要实现面是 `python/sglang/srt/models/deepseek_v4_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4_nextn.py` modified +59/-0 (59 lines); hunks: -7,9 +7,17; -18,6 +26,12; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +59/-0 (59 lines); hunks: -7,9 +7,17; -18,6 +26,12; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4_nextn.py
@@ -7,9 +7,17 @@
+from sglang.srt.layers.attention.nsa.utils import (
+    can_nsa_cp_split,
+    is_nsa_enable_prefill_cp,
+    is_nsa_prefill_cp_round_robin_split,
+    nsa_use_prefill_cp,
+)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4_nextn.py` modified +59/-0
- 验证与风险: diff 自带测试面 `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25771 - fix(dsv4): drop stale pp_size=1 guard for V4 PD disaggregation

- 链接: https://github.com/sgl-project/sglang/pull/25771
- 状态/时间: merged / 2026-05-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`；关联提交 `ca29c2b0e79e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-7，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(dsv4): drop stale pp_size=1 guard for V4 PD disaggregation」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/arg_groups/deepseek_v4_hook.py`；技术摘要: 覆盖「fix(dsv4): drop stale pp_size=1 guard for V4 PD disaggregation」；主要实现面是 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +0/-7 (7 lines); hunks: -51,13 +51,6 @@ def apply_deepseek_v4_defaults(server_args: "ServerArgs", mod...; symbols: apply_deepseek_v4_defaults, validate_deepseek_v4_cp，涉及 `apply_deepseek_v4_defaults, validate_deepseek_v4_cp`。
- 代码 diff 细节:
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +0/-7 (7 lines); hunks: -51,13 +51,6 @@ def apply_deepseek_v4_defaults(server_args: "ServerArgs", mod...; symbols: apply_deepseek_v4_defaults, validate_deepseek_v4_cp
- 关键代码摘录:

```diff
diff -- python/sglang/srt/arg_groups/deepseek_v4_hook.py
@@ -51,13 +51,6 @@ def apply_deepseek_v4_defaults(server_args: "ServerArgs", model_arch: str) -> No
-    if server_args.disaggregation_mode != "null" and server_args.pp_size > 1:
-        # get_mla_kv_ptrs_with_pp cannot slice V4's buffer-type-organized
-        # flat KV ptrs by PP layer range.
-        raise ValueError(
-            f"V4 PD disaggregation requires pp_size=1, got pp_size={server_args.pp_size}."
-        )
```

- 已读文件:
  - runtime: `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +0/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25821 - [Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename

- 链接: https://github.com/sgl-project/sglang/pull/25821
- 状态/时间: merged / 2026-05-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 162 个文件，+11303/-10745，可读 patch 15980 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`；技术摘要: 覆盖「[Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename」；主要实现面是 `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587 (2595 lines)；`python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0 (2589 lines)；`python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518 (2539 lines)；`python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0 (2528 lines)。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587 (2595 lines)
  - `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0 (2589 lines)
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518 (2539 lines)
  - `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0 (2528 lines)
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +8/-1744 (1752 lines); hunks: -1,1746 +1,10; symbols: BaseIndexerMetadata, get_seqlens_int32, get_page_table_64, get_page_table_1
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -1,1746 +1,10 @@
-from __future__ import annotations
+# [Deprecated] Re-export shim for backward compatibility. Use dsa.dsa_indexer instead.
+import warnings
-import contextlib
-import logging
-from abc import ABC, abstractmethod
diff -- python/sglang/srt/layers/attention/dsa/dsa_indexer.py
@@ -0,0 +1,1746 @@
+from __future__ import annotations
+import contextlib
+import logging
+from abc import ABC, abstractmethod
+from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
+import torch
diff -- python/sglang/srt/layers/attention/nsa/index_buf_accessor.py
@@ -1,814 +1,10 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587; `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0; `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518; `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +8/-1744; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` added +1746/-0
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py`, `python/sglang/jit_kernel/tests/test_fused_store_index_cache.py`, `python/sglang/jit_kernel/tests/test_set_mla_kv_buffer.py`, `python/sglang/test/nightly_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25810 - perf(dsv4): add MHC token-count prewarm

- 链接: https://github.com/sgl-project/sglang/pull/25810
- 状态/时间: merged / 2026-05-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`；关联提交 `3a6de13cd822`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+141/-1，可读 patch 198 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「perf(dsv4): add MHC token-count prewarm」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`；技术摘要: 覆盖「perf(dsv4): add MHC token-count prewarm」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +110/-0 (110 lines); hunks: -2,6 +2,7; -696,6 +697,70 @@ def __init__(; symbols: __init__, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets, hc_pre，涉及 `__init__, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets`；`python/sglang/srt/models/deepseek_v4_nextn.py` modified +5/-0 (5 lines); hunks: -108,6 +108,11 @@ def hc_head(; symbols: hc_head, prewarm_mhc_token_count_buckets, forward，涉及 `hc_head, prewarm_mhc_token_count_buckets, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +110/-0 (110 lines); hunks: -2,6 +2,7; -696,6 +697,70 @@ def __init__(; symbols: __init__, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets, hc_pre
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +5/-0 (5 lines); hunks: -108,6 +108,11 @@ def hc_head(; symbols: hc_head, prewarm_mhc_token_count_buckets, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -2,6 +2,7 @@
+import time
@@ -696,6 +697,70 @@ def __init__(
+    def prewarm_mhc_token_counts(
+        self, token_counts: Tuple[int, ...], device: torch.device
+    ) -> None:
+        paths = (
diff -- python/sglang/srt/models/deepseek_v4_nextn.py
@@ -108,6 +108,11 @@ def hc_head(
+    def prewarm_mhc_token_count_buckets(
+        self, max_num_tokens: int, device: torch.device
+    ) -> Tuple[int, ...]:
+        return self.decoder.prewarm_mhc_token_count_buckets(max_num_tokens, device)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +110/-0; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +5/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/mhc.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25889 - [Fix] DSV4 cached_loc invalidated when SWA mapping is rebuilt

- 链接: https://github.com/sgl-project/sglang/pull/25889
- 状态/时间: merged / 2026-05-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；关联提交 `888a8794ef3d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+462/-0，可读 patch 472 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] DSV4 cached_loc invalidated when SWA mapping is rebuilt」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「[Fix] DSV4 cached_loc invalidated when SWA mapping is rebuilt」；主要实现面是 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +4/-0 (4 lines); hunks: -492,6 +492,10 @@ def __init__(; symbols: __init__, register_mapping, invalidate_loc_cache, get_ring_size，涉及 `__init__, register_mapping, invalidate_loc_cache`。
- 代码 diff 细节:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +4/-0 (4 lines); hunks: -492,6 +492,10 @@ def __init__(; symbols: __init__, register_mapping, invalidate_loc_cache, get_ring_size
- 关键代码摘录:

```diff
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -492,6 +492,10 @@ def __init__(
+        self.cached_loc = None  # mapping replaced; discard any cached translation
+    def invalidate_loc_cache(self) -> None:
+        self.cached_loc = None
```

- 已读文件:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +4/-0
- 验证与风险: diff 自带测试面 `test/manual/core/test_dsv4_cached_loc_invalidation.py`, `test/manual/core/test_dsv4_hicache_swa_translation_cache.py`, `test/manual/core/test_dsv4_stale_loc_crash.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25884 - [Refactor] major JIT kernel clean up for dsv4

- 链接: https://github.com/sgl-project/sglang/pull/25884
- 状态/时间: merged / 2026-05-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `19f55c0e6d6f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 23 个文件，+1093/-1399，可读 patch 2663 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] major JIT kernel clean up for dsv4」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/jit_kernel/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「[Refactor] major JIT kernel clean up for dsv4」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/jit_kernel/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7；`python/sglang/jit_kernel/deepseek_v4.py` removed +0/-1036 (1036 lines); hunks: -1,1036 +0,0; symbols: make_name, _jit_common_module, _jit_compress_128_online_plan_module, _jit_compress_128_online_module，涉及 `make_name, _jit_common_module, _jit_compress_128_online_plan_module`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7
  - `python/sglang/jit_kernel/deepseek_v4.py` removed +0/-1036 (1036 lines); hunks: -1,1036 +0,0; symbols: make_name, _jit_common_module, _jit_compress_128_online_plan_module, _jit_compress_128_online_module
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -21,7 +21,7 @@
-from sglang.jit_kernel.deepseek_v4 import (
+from sglang.jit_kernel.dsv4 import (
diff -- python/sglang/jit_kernel/deepseek_v4.py
@@ -1,1036 +0,0 @@
-from __future__ import annotations
-from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional, Tuple, Union
-import torch
-import triton
-import triton.language as tl
-from sglang.jit_kernel.utils import (
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -6,7 +6,7 @@
-from sglang.jit_kernel.deepseek_v4 import fused_k_norm_rope_flashmla, fused_store_cache
+from sglang.jit_kernel.dsv4 import fused_k_norm_rope_flashmla, fused_store_cache
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +1/-1; `python/sglang/jit_kernel/deepseek_v4.py` removed +0/-1036; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/csrc/deepseek_v4/topk_1024.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/topk_v1.cuh`, `python/sglang/jit_kernel/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26004 - Default MegaMoE to W4A8 for Max-Throughput recipe

- 链接: https://github.com/sgl-project/sglang/pull/26004
- 状态/时间: merged / 2026-05-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+13/-2，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Default MegaMoE to W4A8 for Max-Throughput recipe」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「Default MegaMoE to W4A8 for Max-Throughput recipe」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +13/-2 (15 lines); hunks: -177,6 +177,16 @@ export const DeepSeekV4Deployment = () => {; -605,7 +615,8 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +13/-2 (15 lines); hunks: -177,6 +177,16 @@ export const DeepSeekV4Deployment = () => {; -605,7 +615,8 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -177,6 +177,16 @@ export const DeepSeekV4Deployment = () => {
+      // Switching to max-throughput on supported hardware: default MegaMoE to
+      // W4A8 if it's currently disabled (best throughput config).
+      if (
+        (optionName === "recipe" || optionName === "hardware") &&
+        next.recipe === "max-throughput" &&
+        next.megamoe === "disabled" &&
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +13/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25923 - [Docs] DeepSeek-V4: switch H200 FP4 Pro to flashinfer_mxfp4, Flash Balanced too

- 链接: https://github.com/sgl-project/sglang/pull/25923
- 状态/时间: merged / 2026-05-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+16/-5，可读 patch 47 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] DeepSeek-V4: switch H200 FP4 Pro to flashinfer_mxfp4, Flash Balanced too」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[Docs] DeepSeek-V4: switch H200 FP4 Pro to flashinfer_mxfp4, Flash Balanced too」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +15/-4 (19 lines); hunks: -360,22 +360,33 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -120,7 +120,7 @@ The generator currently picks values on the **conservative**...。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +15/-4 (19 lines); hunks: -360,22 +360,33 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -120,7 +120,7 @@ The generator currently picks values on the **conservative**...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -360,22 +360,33 @@ export const DeepSeekV4Deployment = () => {
-    // H200 (FP4) Marlin path: dedicated branch — Hopper runs the FP4-mixed
-    // Instruct repos through the Marlin MoE runner, so it doesn't share envs
-    // or flags with either the FP8 H200 path or the Blackwell paths.
+    // H200 (FP4) path: dedicated branch — Hopper runs the FP4-mixed Instruct
+    // repos through one of two w4a16 MoE runners (Marlin or Flashinfer mxfp4),
+    // so it doesn't share envs or flags with either the FP8 H200 path or the
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -120,7 +120,7 @@ The generator currently picks values on the **conservative** side (mirroring an
-- Original FP4 checkpoints: To run original FP4 checkpoints, we provide two different options for w4a16 MoE kernels: Marlin (`--moe-runner-backend marlin`) and Flashinfer (`--moe-
+- Original FP4 checkpoints: To run original FP4 checkpoints, we provide two different options for w4a16 MoE kernels: Marlin (`--moe-runner-backend marlin`) and Flashinfer (`--moe-
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +15/-4; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #26057 - [docs] DeepSeek-V4 cookbook: split Quantization axis, add H100 SGLang FP8

- 链接: https://github.com/sgl-project/sglang/pull/26057
- 状态/时间: merged / 2026-05-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+388/-93，可读 patch 722 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] DeepSeek-V4 cookbook: split Quantization axis, add H100 SGLang FP8」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「[docs] DeepSeek-V4 cookbook: split Quantization axis, add H100 SGLang FP8」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +273/-68 (341 lines); hunks: -35,7 +35,7 @@ tag: NEW; -182,7 +182,7 @@ curl http://localhost:30000/v1/chat/completions \；`docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +115/-25 (140 lines); hunks: -27,13 +27,12 @@ export const DeepSeekV4Deployment = () => {; -44,6 +43,14 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +273/-68 (341 lines); hunks: -35,7 +35,7 @@ tag: NEW; -182,7 +182,7 @@ curl http://localhost:30000/v1/chat/completions \
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +115/-25 (140 lines); hunks: -27,13 +27,12 @@ export const DeepSeekV4Deployment = () => {; -44,6 +43,14 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -35,7 +35,7 @@ tag: NEW
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB200 8 GPU (2 nodes) / GB300 4 GPU / H200 8 GPU(fp4)/16 GPU(fp8)</t
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB200 8 GPU (2 nodes) / GB300 4 GPU / H200 8 GPU (FP4) or 16 GPU (SG
@@ -182,7 +182,7 @@ curl http://localhost:30000/v1/chat/completions \
-**Streaming with Thinking Process:**
+<Accordion title="Streaming with Thinking Process (Python)">
@@ -227,17 +227,36 @@ for chunk in response:
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -27,13 +27,12 @@ export const DeepSeekV4Deployment = () => {
-        { id: "b200",  label: "B200 (FP4)",  default: true  },
-        { id: "b300",  label: "B300 (FP4)",  default: false  },
-        { id: "gb200", label: "GB200 (FP4)", default: false },
-        { id: "gb300", label: "GB300 (FP4)", default: false },
-        { id: "h200",  label: "H200 (FP8)",  default: false },
-        { id: "h200-fp4", label: "H200 (FP4)", default: false },
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +273/-68; `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +115/-25
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25128 - [Intel GPU] 1/N Fix tilelang import in deepseek v4 rope as optional

- 链接: https://github.com/sgl-project/sglang/pull/25128
- 状态/时间: merged / 2026-05-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-6，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Intel GPU] 1/N Fix tilelang import in deepseek v4 rope as optional」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/deepseek_v4_rope.py`；技术摘要: 覆盖「[Intel GPU] 1/N Fix tilelang import in deepseek v4 rope as optional」；主要实现面是 `python/sglang/srt/layers/deepseek_v4_rope.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/deepseek_v4_rope.py` modified +9/-6 (15 lines); hunks: -2,17 +2,20。
- 代码 diff 细节:
  - `python/sglang/srt/layers/deepseek_v4_rope.py` modified +9/-6 (15 lines); hunks: -2,17 +2,20
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/deepseek_v4_rope.py
@@ -2,17 +2,20 @@
-import tilelang
-tilelang.set_log_level("WARNING")
+try:
+    import tilelang
-pass_configs = {
-    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/deepseek_v4_rope.py` modified +9/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/deepseek_v4_rope.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26141 - Add non-MTP DSV4 test coverage

- 链接: https://github.com/sgl-project/sglang/pull/26141
- 状态/时间: merged / 2026-05-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`；关联提交 `7b7f1067bdb0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+68/-0，可读 patch 81 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add non-MTP DSV4 test coverage」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`；技术摘要: 覆盖「Add non-MTP DSV4 test coverage」；主要实现面是 `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +36/-0 (36 lines); hunks: -120,6 +120,42 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4NonMTPB200, setUpClass, TestDSV4FlashFP4B200Balanced_CP，涉及 `tearDownClass, TestDSV4FlashFP4NonMTPB200, setUpClass`；`test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +32/-0 (32 lines); hunks: -131,5 +131,37 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4NonMTPH200, setUpClass，涉及 `tearDownClass, TestDSV4FlashFP4NonMTPH200, setUpClass`。
- 代码 diff 细节:
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +36/-0 (36 lines); hunks: -120,6 +120,42 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4NonMTPB200, setUpClass, TestDSV4FlashFP4B200Balanced_CP
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +32/-0 (32 lines); hunks: -131,5 +131,37 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4NonMTPH200, setUpClass
- 关键代码摘录:

```diff
diff -- test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py
@@ -120,6 +120,42 @@ def tearDownClass(cls):
+class TestDSV4FlashFP4NonMTPB200(
+    BasicDecodeCorrectnessMixin, GSM8KMixin, CustomTestCase
+):
+    """Non-MTP recipe: TP=4, DP=4, DeepEP, no speculative decoding."""
+    gsm8k_accuracy_thres = 0.93
+    @classmethod
diff -- test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py
@@ -131,5 +131,37 @@ def tearDownClass(cls):
+class TestDSV4FlashFP4NonMTPH200(
+    BasicDecodeCorrectnessMixin, GSM8KMixin, CustomTestCase
+):
+    """LowLatency recipe without MTP: TP=4, Marlin FP4, no speculative decoding."""
+    gsm8k_accuracy_thres = 0.93
+    @classmethod
```

- 已读文件:
  - tests: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +36/-0; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +32/-0
- 验证与风险: diff 自带测试面 `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26164 - [docs] DeepSeek-V4 cookbook: balanced MegaMoE cap, H200 Pro FP4 mem-frac, nsa-* compat, PD-disagg fixes

- 链接: https://github.com/sgl-project/sglang/pull/26164
- 状态/时间: merged / 2026-05-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+66/-7，可读 patch 139 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] DeepSeek-V4 cookbook: balanced MegaMoE cap, H200 Pro FP4 mem-frac, nsa-* compat, PD-disagg fixes」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「[docs] DeepSeek-V4 cookbook: balanced MegaMoE cap, H200 Pro FP4 mem-frac, nsa-* compat, PD-disagg fixes」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +66/-7 (73 lines); hunks: -119,15 +119,23 @@ export const DeepSeekV4Deployment = () => {; -155,11 +163,20 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +66/-7 (73 lines); hunks: -119,15 +119,23 @@ export const DeepSeekV4Deployment = () => {; -155,11 +163,20 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -119,15 +119,23 @@ export const DeepSeekV4Deployment = () => {
-  // MegaMoE is only supported on Blackwell with DeepEP-based recipes
-  // (balanced / max-throughput / pd-disagg). It's disabled on Hopper
-  // (H100 / H200, both FP4 and FP8) and on low-latency / cp recipes.
-  const MEGAMOE_UNSUPPORTED_RECIPES = new Set(["low-latency", "cp"]);
+  // MegaMoE is only wired into the deepep-replacing recipes on Blackwell
+  // (balanced / max-throughput). Disabled on Hopper (H100 / H200, both FP4
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +66/-7
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25898 - [AMD] Dsv4/pr1 fix run time issue

- 链接: https://github.com/sgl-project/sglang/pull/25898
- 状态/时间: merged / 2026-05-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `af8f66940e9b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 32 个文件，+2523/-129，可读 patch 3203 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Dsv4/pr1 fix run time issue」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「[AMD] Dsv4/pr1 fix run time issue」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +153/-28 (181 lines); hunks: -96,6 +96,8; -105,6 +107,29; symbols: _fused_rmsnorm_fp8_quant, __init__, _forward_prepare_multi_stream，涉及 `_fused_rmsnorm_fp8_quant, __init__, _forward_prepare_multi_stream`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-0 (1 lines); hunks: -578,6 +578,7 @@ def _init_paged_compress_states(self, enable_memory_saver: b...; symbols: _init_paged_compress_states，涉及 `_init_paged_compress_states`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +153/-28 (181 lines); hunks: -96,6 +96,8; -105,6 +107,29; symbols: _fused_rmsnorm_fp8_quant, __init__, _forward_prepare_multi_stream
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-0 (1 lines); hunks: -578,6 +578,7 @@ def _init_paged_compress_states(self, enable_memory_saver: b...; symbols: _init_paged_compress_states
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -96,6 +96,8 @@
+    get_bool_env_var,
+    is_gfx95_supported,
@@ -105,6 +107,29 @@
+_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
+_is_gfx95_supported = is_gfx95_supported()
+if _use_aiter:
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -578,6 +578,7 @@ def _init_paged_compress_states(self, enable_memory_saver: bool):
+                swa_page_size=self.swa_page_size,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +153/-28; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25948 - [dsv4] support eplb

- 链接: https://github.com/sgl-project/sglang/pull/25948
- 状态/时间: merged / 2026-05-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `7f45bcdd2ab8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+18/-6，可读 patch 60 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[dsv4] support eplb」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[dsv4] support eplb」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +14/-6 (20 lines); hunks: -3,6 +3,7; -33,6 +34,7; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +14/-6 (20 lines); hunks: -3,6 +3,7; -33,6 +34,7; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -3,6 +3,7 @@
+from contextlib import nullcontext
@@ -33,6 +34,7 @@
+from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
@@ -1134,13 +1136,19 @@ def forward(
-            hidden_states = layer(
-                positions=positions,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +14/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/hash_topk.py`, `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26239 - [dsv4] fix multi-step draft on non-cuda-graph path

- 链接: https://github.com/sgl-project/sglang/pull/26239
- 状态/时间: merged / 2026-05-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；关联提交 `ed179bf9b297`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+47/-7，可读 patch 93 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[dsv4] fix multi-step draft on non-cuda-graph path」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；技术摘要: 覆盖「[dsv4] fix multi-step draft on non-cuda-graph path」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +13/-1 (14 lines); hunks: -53,6 +53,7; -676,11 +677,22 @@ def init_forward_metadata(self, forward_batch: ForwardBatc...; symbols: init_forward_metadata，涉及 `init_forward_metadata`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +13/-1 (14 lines); hunks: -53,6 +53,7; -676,11 +677,22 @@ def init_forward_metadata(self, forward_batch: ForwardBatc...; symbols: init_forward_metadata
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -53,6 +53,7 @@
+from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
@@ -676,11 +677,22 @@ def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
+            # DSv4 bakes this step's KV write target (c4/c128) into metadata,
+            # so slice the shared multi-step out_cache_loc now rather than at
+            # forward time.
+            out_cache_loc = forward_batch.out_cache_loc
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +13/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/speculative/eagle_utils.py`, `python/sglang/srt/speculative/eagle_worker_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25391 - Support DeepSeek V4 DeepEP Waterfill

- 链接: https://github.com/sgl-project/sglang/pull/25391
- 状态/时间: merged / 2026-05-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `59cad671e2a8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+58/-16，可读 patch 134 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support DeepSeek V4 DeepEP Waterfill」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「Support DeepSeek V4 DeepEP Waterfill」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +16/-0 (16 lines); hunks: -1396,6 +1396,22 @@ def determine_num_fused_shared_experts(self):; symbols: determine_num_fused_shared_experts，涉及 `determine_num_fused_shared_experts`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +16/-0 (16 lines); hunks: -1396,6 +1396,22 @@ def determine_num_fused_shared_experts(self):; symbols: determine_num_fused_shared_experts
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -1396,6 +1396,22 @@ def determine_num_fused_shared_experts(self):
+        # Waterfill needs shared-experts fusion so it can dispatch shared
+        # expert tokens to least-loaded EP ranks.
+        if get_global_server_args().enable_deepep_waterfill:
+            if self.config.n_shared_experts != 1:
+                raise ValueError(
+                    "DeepEP Waterfill for DeepSeek V4 expects exactly one shared "
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +16/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/hash_topk.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/model_executor/model_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26208 - [AMD] Dsv4/pr2 compressor opt

- 链接: https://github.com/sgl-project/sglang/pull/26208
- 状态/时间: merged / 2026-05-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `3f5e2c768825`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 31 个文件，+8829/-149，可读 patch 6378 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Dsv4/pr2 compressor opt」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「[AMD] Dsv4/pr2 compressor opt」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +152/-10 (162 lines); hunks: -536,6 +536,118 @@ def _forward_prepare_multi_stream(; -695,14 +807,24 @@ def forward(; symbols: _forward_prepare_multi_stream, _forward_prepare_multi_stream_hip, _forward_prepare, forward，涉及 `_forward_prepare_multi_stream, _forward_prepare_multi_stream_hip, _forward_prepare`；`python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +13/-5 (18 lines); hunks: -20,11 +20,19；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-1 (7 lines); hunks: -470,8 +470,13 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +152/-10 (162 lines); hunks: -536,6 +536,118 @@ def _forward_prepare_multi_stream(; -695,14 +807,24 @@ def forward(; symbols: _forward_prepare_multi_stream, _forward_prepare_multi_stream_hip, _forward_prepare, forward
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +13/-5 (18 lines); hunks: -20,11 +20,19
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-1 (7 lines); hunks: -470,8 +470,13 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -536,6 +536,118 @@ def _forward_prepare_multi_stream(
+    def _forward_prepare_multi_stream_hip(
+        self,
+        x: torch.Tensor,
+        positions: torch.Tensor,
+        forward_batch: ForwardBatch,
+        attn_backend,
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -20,11 +20,19 @@
-from sglang.srt.layers.attention.dsv4.compressor import (
-    CompressorBackendMixin,
-    FusedCompressMetadata,
-    create_paged_compressor_data,
-)
+if envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -470,8 +470,13 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +152/-10; `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +13/-5; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-1
- 验证与风险: diff 自带测试面 `sgl-kernel/tests/test_dsv4_norm_rope.py`, `test/manual/dsv4/test_fused_compress_attn_hip.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26413 - [docs] DeepSeek-V4 cookbook: note cu129 image for GB200 Pro DeepEP backend

- 链接: https://github.com/sgl-project/sglang/pull/26413
- 状态/时间: merged / 2026-05-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+14/-0，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] DeepSeek-V4 cookbook: note cu129 image for GB200 Pro DeepEP backend」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「[docs] DeepSeek-V4 cookbook: note cu129 image for GB200 Pro DeepEP backend」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0 (14 lines); hunks: -909,6 +909,20 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0 (14 lines); hunks: -909,6 +909,20 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -909,6 +909,20 @@ export const DeepSeekV4Deployment = () => {
+    // GB200 Pro with MegaMoE disabled runs the DeepEP a2a backend, which is
+    // currently only packaged in the CUDA 12.9 image — the default `:latest`
+    // ships CUDA 13 and does not include a compatible DeepEP build.
+    if (
+      hardware === "gb200" &&
+      isBig &&
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #26451 - [docs] Fix V4 Pro balanced recipe

- 链接: https://github.com/sgl-project/sglang/pull/26451
- 状态/时间: merged / 2026-05-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-12，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Fix V4 Pro balanced recipe」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[docs] Fix V4 Pro balanced recipe」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +4/-10 (14 lines); hunks: -119,11 +119,11 @@ export const DeepSeekV4Deployment = () => {; -864,12 +864,6 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +2/-2 (4 lines); hunks: -145,8 +145,8 @@ Two variants are exposed:。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +4/-10 (14 lines); hunks: -119,11 +119,11 @@ export const DeepSeekV4Deployment = () => {; -864,12 +864,6 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +2/-2 (4 lines); hunks: -145,8 +145,8 @@ Two variants are exposed:
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -119,11 +119,11 @@ export const DeepSeekV4Deployment = () => {
-  // MegaMoE is only wired into the deepep-replacing recipes on Blackwell
-  // (balanced / max-throughput). Disabled on Hopper (H100 / H200, both FP4
-  // and FP8), on low-latency / cp recipes, and on PD-Disagg (the cookbook's
+  // MegaMoE is only wired into the max-throughput recipe on Blackwell.
+  // Disabled on Hopper (H100 / H200, both FP4 and FP8), on
+  // low-latency / balanced / cp recipes, and on PD-Disagg (the cookbook's
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -145,8 +145,8 @@ Two variants are exposed:
-- MegaMoE is **not** supported on Hopper (H100 / H200) nor on the `low-latency` / `cp` settings. When running MegaMoE, don't set `--moe-runner-backend` manually.
-- Adjust `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` based on your workload and memory usage. Setting higher number of tokens for MegaMoE requires more HBM space. (reco
+- MegaMoE is **not** supported on Hopper (H100 / H200) nor on the `low-latency` / `balanced` / `cp` settings — it is only wired into the `max-throughput` recipe on Blackwell. When
+- Adjust `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` based on your workload and memory usage. Setting higher number of tokens for MegaMoE requires more HBM space. (reco
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +4/-10; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +2/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #26499 - [Kernel] Import flash_mla kernels from sglang kernel for deepseek v4

- 链接: https://github.com/sgl-project/sglang/pull/26499
- 状态/时间: merged / 2026-05-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`；关联提交 `e06058ed624f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+6/-6，可读 patch 54 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Import flash_mla kernels from sglang kernel for deepseek v4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`；技术摘要: 覆盖「[Kernel] Import flash_mla kernels from sglang kernel for deepseek v4」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +3/-3 (6 lines); hunks: -58,7 +58,7; -82,7 +82,7 @@ def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED...; symbols: _pad_last_dim, _create_flashmla_metadata, forward，涉及 `_pad_last_dim, _create_flashmla_metadata, forward`；`python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +2/-2 (4 lines); hunks: -55,7 +55,7; -83,7 +83,7 @@ def _create_flashmla_metadata():; symbols: _create_flashmla_metadata，涉及 `_create_flashmla_metadata`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +3/-3 (6 lines); hunks: -58,7 +58,7; -82,7 +82,7 @@ def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED...; symbols: _pad_last_dim, _create_flashmla_metadata, forward
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +2/-2 (4 lines); hunks: -55,7 +55,7; -83,7 +83,7 @@ def _create_flashmla_metadata():; symbols: _create_flashmla_metadata
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -58,7 +58,7 @@
-    from flash_mla.flash_mla_interface import FlashMLASchedMeta
+    from sgl_kernel.flash_mla import FlashMLASchedMeta
@@ -82,7 +82,7 @@ def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
-    import flash_mla
+    import sgl_kernel.flash_mla as flash_mla
@@ -1045,7 +1045,7 @@ def forward(
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -55,7 +55,7 @@
-    from flash_mla.flash_mla_interface import FlashMLASchedMeta
+    from sgl_kernel.flash_mla import FlashMLASchedMeta
@@ -83,7 +83,7 @@ def _create_flashmla_metadata():
-    import flash_mla
+    import sgl_kernel.flash_mla as flash_mla
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +3/-3; `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/hip_flash_mla.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26383 - [AMD][DSV4] DSV4 MTP graph + sparse triton attn optimizations

- 链接: https://github.com/sgl-project/sglang/pull/26383
- 状态/时间: merged / 2026-05-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `deaba74745d7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+659/-65，可读 patch 950 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][DSV4] DSV4 MTP graph + sparse triton attn optimizations」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[AMD][DSV4] DSV4 MTP graph + sparse triton attn optimizations」；主要实现面是 `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` added +158/-0 (158 lines); hunks: -0,0 +1,158; symbols: _get_triton_mhc_post_pre_ops, _get_fused_hc_post_pre_buffers, try_fused_hc_post_pre，涉及 `_get_triton_mhc_post_pre_ops, _get_fused_hc_post_pre_buffers, try_fused_hc_post_pre`；`python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +35/-26 (61 lines); hunks: -51,6 +51,7; -500,32 +501,21 @@ def init_forward_metadata_target_verify(; symbols: init_forward_metadata_target_verify, init_forward_metadata_target_verify_old, make_forward_metadata_from_raw_verify, init_forward_metadata，涉及 `init_forward_metadata_target_verify, init_forward_metadata_target_verify_old, make_forward_metadata_from_raw_verify`；`python/sglang/srt/models/deepseek_v4.py` modified +52/-7 (59 lines); hunks: -87,6 +87,9; -133,6 +136,28 @@ def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):; symbols: _fused_rmsnorm_fp8_quant, _freqs_cis_to_cos_sin, __init__, _forward_prepare_multi_stream_hip，涉及 `_fused_rmsnorm_fp8_quant, _freqs_cis_to_cos_sin, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` added +158/-0 (158 lines); hunks: -0,0 +1,158; symbols: _get_triton_mhc_post_pre_ops, _get_fused_hc_post_pre_buffers, try_fused_hc_post_pre
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +35/-26 (61 lines); hunks: -51,6 +51,7; -500,32 +501,21 @@ def init_forward_metadata_target_verify(; symbols: init_forward_metadata_target_verify, init_forward_metadata_target_verify_old, make_forward_metadata_from_raw_verify, init_forward_metadata
  - `python/sglang/srt/models/deepseek_v4.py` modified +52/-7 (59 lines); hunks: -87,6 +87,9; -133,6 +136,28 @@ def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):; symbols: _fused_rmsnorm_fp8_quant, _freqs_cis_to_cos_sin, __init__, _forward_prepare_multi_stream_hip
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py
@@ -0,0 +1,158 @@
+import logging
+from typing import Optional, Tuple
+import torch
+import triton
+from sglang.srt.environ import envs
+logger = logging.getLogger(__name__)
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -51,6 +51,7 @@
+from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
@@ -500,32 +501,21 @@ def init_forward_metadata_target_verify(
+        extend_seq_lens: Optional[torch.Tensor] = None,
-        if envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
-            assert out_cache_loc is not None
-            if not hasattr(self, "extend_seq_lens_buffer"):
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -87,6 +87,9 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` added +158/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +35/-26; `python/sglang/srt/models/deepseek_v4.py` modified +52/-7
- 验证与风险: diff 自带测试面 `test/registered/ops/test_aiter_greedy_sample_amd.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26238 - refactor(dsv4): route MHC prenorm through DeepGEMM wrapper

- 链接: https://github.com/sgl-project/sglang/pull/26238
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`；关联提交 `eae03ce3b2a8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+67/-148，可读 patch 345 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「refactor(dsv4): route MHC prenorm through DeepGEMM wrapper」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`；技术摘要: 覆盖「refactor(dsv4): route MHC prenorm through DeepGEMM wrapper」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +4/-112 (116 lines); hunks: -2,7 +2,6; -821,70 +820,6 @@ def __init__(; symbols: __init__, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets, hc_pre，涉及 `__init__, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets`；`python/sglang/srt/models/deepseek_v4_nextn.py` modified +0/-5 (5 lines); hunks: -129,11 +129,6 @@ def hc_head(; symbols: hc_head, prewarm_mhc_token_count_buckets, forward，涉及 `hc_head, prewarm_mhc_token_count_buckets, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +4/-112 (116 lines); hunks: -2,7 +2,6; -821,70 +820,6 @@ def __init__(; symbols: __init__, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets, hc_pre
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +0/-5 (5 lines); hunks: -129,11 +129,6 @@ def hc_head(; symbols: hc_head, prewarm_mhc_token_count_buckets, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -2,7 +2,6 @@
-import time
@@ -821,70 +820,6 @@ def __init__(
-    def prewarm_mhc_token_counts(
-        self, token_counts: Tuple[int, ...], device: torch.device
-    ) -> None:
-        paths = (
diff -- python/sglang/srt/models/deepseek_v4_nextn.py
@@ -129,11 +129,6 @@ def hc_head(
-    def prewarm_mhc_token_count_buckets(
-        self, max_num_tokens: int, device: torch.device
-    ) -> Tuple[int, ...]:
-        return self.decoder.prewarm_mhc_token_count_buckets(max_num_tokens, device)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +4/-112; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +0/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/mhc.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26609 - [CI] Clean DeepSeek V4 tests and installation scripts

- 链接: https://github.com/sgl-project/sglang/pull/26609
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py`；关联提交 `435c4ffb3081`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+66/-198，可读 patch 432 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Clean DeepSeek V4 tests and installation scripts」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`；技术摘要: 覆盖「[CI] Clean DeepSeek V4 tests and installation scripts」；主要实现面是 `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +2/-2 (4 lines); hunks: -4,7 +4,7; -20,7 +20,7；`test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +2/-2 (4 lines); hunks: -4,7 +4,7; -20,7 +20,7; symbols: _flashinfer_has_sm90_cutlass_mxfp4，涉及 `_flashinfer_has_sm90_cutlass_mxfp4`；`test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +2/-2 (4 lines); hunks: -4,7 +4,7; -20,7 +20,7；`test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py` modified +2/-2 (4 lines); hunks: -5,7 +5,7; -21,7 +21,7。
- 代码 diff 细节:
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +2/-2 (4 lines); hunks: -4,7 +4,7; -20,7 +20,7
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +2/-2 (4 lines); hunks: -4,7 +4,7; -20,7 +20,7; symbols: _flashinfer_has_sm90_cutlass_mxfp4
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +2/-2 (4 lines); hunks: -4,7 +4,7; -20,7 +20,7
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py` modified +2/-2 (4 lines); hunks: -5,7 +5,7; -21,7 +21,7
  - `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7
- 关键代码摘录:

```diff
diff -- test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py
@@ -4,7 +4,7 @@
-Registry: base-c-test-dsv4-4-gpu-b200 (per-commit, 4x B200)
+Registry: base-c-test-deepep-4-gpu-b200 (per-commit, 4x B200)
@@ -20,7 +20,7 @@
-register_cuda_ci(est_time=465, stage="base-c", runner_config="dsv4-4-gpu-b200")
+register_cuda_ci(est_time=465, stage="base-c", runner_config="deepep-4-gpu-b200")
diff -- test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py
@@ -4,7 +4,7 @@
-Registry: base-c-test-dsv4-8-gpu-h200 (per-commit, 8x H200 — only 4 used by TP=4)
+Registry: base-c-test-deepep-8-gpu-h200 (per-commit, 8x H200 — only 4 used by TP=4)
@@ -20,7 +20,7 @@
-register_cuda_ci(est_time=370, stage="base-c", runner_config="dsv4-8-gpu-h200")
+register_cuda_ci(est_time=370, stage="base-c", runner_config="deepep-8-gpu-h200")
diff -- test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py
@@ -4,7 +4,7 @@
-Registry: base-c-test-dsv4-4-gpu-b200 (per-commit, 4x B200)
+Registry: extra-b-test-deepep-4-gpu-b200 (label-gated, 4x B200)
```

- 已读文件:
  - tests: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +2/-2; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +2/-2; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +2/-2; `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py` modified +2/-2; `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/disaggregation/test_disaggregation_dsv4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26668 - [Doc] Update benchmark instruction for dsv4

- 链接: https://github.com/sgl-project/sglang/pull/26668
- 状态/时间: merged / 2026-05-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+28/-36，可读 patch 83 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc] Update benchmark instruction for dsv4」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[Doc] Update benchmark instruction for dsv4」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +28/-36 (64 lines); hunks: -363,6 +363,10 @@ For more details, see the [HiCache documentation](../../../...; -383,47 +387,35 @@ python3 -m sglang.test.few_shot_gsm8k --num-questions 200...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +28/-36 (64 lines); hunks: -363,6 +363,10 @@ For more details, see the [HiCache documentation](../../../...; -383,47 +387,35 @@ python3 -m sglang.test.few_shot_gsm8k --num-questions 200...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -363,6 +363,10 @@ For more details, see the [HiCache documentation](../../../docs/advanced_feature
+For accuracy benchmarking on DeepSeek-V4 models, please make sure that:
+- `SGLANG_DEFAULT_THINKING=1 SGLANG_REASONING_EFFORT=max` are set when launching model.
+- For GPQA and AIME25 benchmarks, run at least 16 turns to reduce randomness.
@@ -383,47 +387,35 @@ python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --port 30000
-#### 5.1.2 MMLU Benchmark
+#### 5.1.2 GPQA Diamond Benchmark
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +28/-36
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #26662 - [AMD][CI] Update v4 CI setting and move the task to main branch

- 链接: https://github.com/sgl-project/sglang/pull/26662
- 状态/时间: merged / 2026-05-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`；关联提交 `6e9bd82714cb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+97/-160，可读 patch 398 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][CI] Update v4 CI setting and move the task to main branch」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`；技术摘要: 覆盖「[AMD][CI] Update v4 CI setting and move the task to main branch」；主要实现面是 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +22/-24 (46 lines); hunks: -35,38 +35,32; -86,11 +80,15 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`；`test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +22/-24 (46 lines); hunks: -35,38 +35,32; -86,11 +80,15 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`；`test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +22/-24 (46 lines); hunks: -37,38 +37,32; -88,11 +82,15 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`；`test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +22/-24 (46 lines); hunks: -37,38 +37,32; -88,11 +82,15 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`。
- 代码 diff 细节:
  - `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +22/-24 (46 lines); hunks: -35,38 +35,32; -86,11 +80,15 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +22/-24 (46 lines); hunks: -35,38 +35,32; -86,11 +80,15 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +22/-24 (46 lines); hunks: -37,38 +37,32; -88,11 +82,15 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +22/-24 (46 lines); hunks: -37,38 +37,32; -88,11 +82,15 @@ def setUpClass(cls):; symbols: setUpClass
- 关键代码摘录:

```diff
diff -- test/registered/amd/test_deepseek_v4_flash_fp4.py
@@ -35,38 +35,32 @@
-# Common DeepSeek-V4 env vars (AMD ROCm 7.2 path: tilelang + AITER + ROCm700A).
-# Source of truth: python/run_dsv4.sh.
+# Common DeepSeek-V4 env vars (AMD ROCm 7.2 path: AITER indexer + triton attn + ROCm700A).
-    "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
-    "SGLANG_OPT_USE_OLD_COMPRESSOR": "true",
-    "SGLANG_OPT_USE_TILELANG_SWA_PREPARE": "false",
diff -- test/registered/amd/test_deepseek_v4_flash_fp8.py
@@ -35,38 +35,32 @@
-# Common DeepSeek-V4 env vars (AMD ROCm 7.2 path: tilelang + AITER + ROCm700A).
-# Source of truth: python/run_dsv4.sh.
+# Common DeepSeek-V4 env vars (AMD ROCm 7.2 path: AITER indexer + triton attn + ROCm700A).
-    "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
-    "SGLANG_OPT_USE_OLD_COMPRESSOR": "true",
-    "SGLANG_OPT_USE_TILELANG_SWA_PREPARE": "false",
diff -- test/registered/amd/test_deepseek_v4_pro_fp4.py
@@ -37,38 +37,32 @@
```

- 已读文件:
  - tests: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +22/-24; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +22/-24; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +22/-24; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +22/-24
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25976 - [DeepSeek-V4] Add mhc_fused_post_pre kernel

- 链接: https://github.com/sgl-project/sglang/pull/25976
- 状态/时间: merged / 2026-05-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`；关联提交 `7c5708cba734`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+875/-48，可读 patch 1065 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek-V4] Add mhc_fused_post_pre kernel」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`；技术摘要: 覆盖「[DeepSeek-V4] Add mhc_fused_post_pre kernel」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +261/-47 (308 lines); hunks: -2,6 +2,7; -61,6 +62,7; symbols: _is_fused_mhc_post_pre_enabled, __init__, refresh_mhc_norm_weight_cache, prewarm_mhc_token_counts，涉及 `_is_fused_mhc_post_pre_enabled, __init__, refresh_mhc_norm_weight_cache`；`python/sglang/srt/models/deepseek_v4_nextn.py` modified +5/-1 (6 lines); hunks: -170,13 +170,17 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +261/-47 (308 lines); hunks: -2,6 +2,7; -61,6 +62,7; symbols: _is_fused_mhc_post_pre_enabled, __init__, refresh_mhc_norm_weight_cache, prewarm_mhc_token_counts
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +5/-1 (6 lines); hunks: -170,13 +170,17 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -2,6 +2,7 @@
+import time
@@ -61,6 +62,7 @@
+from sglang.srt.layers.mhc import mhc_fused_post_pre
@@ -110,6 +112,18 @@
+_MHC_POST_MULT_VALUE = 2.0
+def _is_fused_mhc_post_pre_enabled() -> bool:
diff -- python/sglang/srt/models/deepseek_v4_nextn.py
@@ -170,13 +170,17 @@ def forward(
-        hidden_states = self.decoder(
+        hidden_states, residual, post, comb = self.decoder(
+        if residual is not None:
+            # NextN has a single decoder layer, so no later layer can consume a
+            # deferred fused hc_post state.
+            hidden_states = self.decoder.hc_post(hidden_states, residual, post, comb)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +261/-47; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +5/-1
- 验证与风险: diff 自带测试面 `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24692 - feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference

- 链接: https://github.com/sgl-project/sglang/pull/24692
- 状态/时间: merged / 2026-06-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；关联提交 `524ba10eda1b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+2105/-22，可读 patch 2268 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；技术摘要: 覆盖「feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +41/-18 (59 lines); hunks: -56,13 +56,16; -82,6 +85,8 @@ def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED...; symbols: _pad_last_dim, _create_flashmla_metadata, forward，涉及 `_pad_last_dim, _create_flashmla_metadata, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +41/-18 (59 lines); hunks: -56,13 +56,16; -82,6 +85,8 @@ def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED...; symbols: _pad_last_dim, _create_flashmla_metadata, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -56,13 +56,16 @@
+from sglang.srt.utils.common import is_sm120_supported
+_is_sm120 = is_sm120_supported()
@@ -82,6 +85,8 @@ def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
+    if _is_sm120:
+        return None
@@ -1045,24 +1050,42 @@ def forward(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +41/-18
- 验证与风险: diff 自带测试面 `test/registered/kernels/test_sm120_flash_mla.py`, `test/registered/kernels/test_sm120_paged_mqa_logits.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24947 - DeepSeek V4: Support context parallelism with fused MoE (non-DeepEP)

- 链接: https://github.com/sgl-project/sglang/pull/24947
- 状态/时间: merged / 2026-06-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`；关联提交 `5700790c0593`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+1982/-24，可读 patch 2117 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「DeepSeek V4: Support context parallelism with fused MoE (non-DeepEP)」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`；技术摘要: 覆盖「DeepSeek V4: Support context parallelism with fused MoE (non-DeepEP)」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +18/-9 (27 lines); hunks: -44,6 +44,10; -68,6 +72,7; symbols: forward，涉及 `forward`；`python/sglang/srt/models/deepseek_v4_nextn.py` modified +3/-0 (3 lines); hunks: -28,6 +28,7; -169,6 +170,8 @@ def forward(; symbols: forward，涉及 `forward`；`test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +45/-0 (45 lines); hunks: -81,5 +81,50 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4B200Balanced_CP_NonDeepEP, setUpClass，涉及 `tearDownClass, TestDSV4FlashFP4B200Balanced_CP_NonDeepEP, setUpClass`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +18/-9 (27 lines); hunks: -44,6 +44,10; -68,6 +72,7; symbols: forward
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +3/-0 (3 lines); hunks: -28,6 +28,7; -169,6 +170,8 @@ def forward(; symbols: forward
  - `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +45/-0 (45 lines); hunks: -81,5 +81,50 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4B200Balanced_CP_NonDeepEP, setUpClass
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -44,6 +44,10 @@
+from sglang.srt.layers.communicator_dsa_cp import (
+    dsa_cp_gather_hidden_states,
+    dsa_cp_reduce_scatter_hidden_states,
+)
@@ -68,6 +72,7 @@
+    cp_round_robin_input_ids,
diff -- python/sglang/srt/models/deepseek_v4_nextn.py
@@ -28,6 +28,7 @@
+    cp_round_robin_input_ids,
@@ -169,6 +170,8 @@ def forward(
+            input_ids = cp_round_robin_input_ids(input_ids)
+            input_ids_global = input_ids
diff -- test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py
@@ -81,5 +81,50 @@ def tearDownClass(cls):
+class TestDSV4FlashFP4B200Balanced_CP_NonDeepEP(
+    BasicDecodeCorrectnessMixin,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +18/-9; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +3/-0
  - tests: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +45/-0
- 验证与风险: diff 自带测试面 `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26968 - docs: update RTX PRO 6000 deployment snippet

- 链接: https://github.com/sgl-project/sglang/pull/26968
- 状态/时间: merged / 2026-06-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+33/-29，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: update RTX PRO 6000 deployment snippet」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；技术摘要: 覆盖「docs: update RTX PRO 6000 deployment snippet」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +33/-29 (62 lines); hunks: -33,7 +33,7 @@ export const DeepSeekV4Deployment = () => {; -125,7 +125,7 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +33/-29 (62 lines); hunks: -33,7 +33,7 @@ export const DeepSeekV4Deployment = () => {; -125,7 +125,7 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -33,7 +33,7 @@ export const DeepSeekV4Deployment = () => {
-        { id: "sm120", label: "RTX PRO 6000 (SM120)", default: false },
+        { id: "rtx6000", label: "RTX PRO 6000", default: false },
@@ -125,7 +125,7 @@ export const DeepSeekV4Deployment = () => {
-  const MEGAMOE_UNSUPPORTED_HARDWARE = new Set(["h100", "h200", "sm120"]);
+  const MEGAMOE_UNSUPPORTED_HARDWARE = new Set(["h100", "h200", "rtx6000"]);
@@ -134,7 +134,9 @@ export const DeepSeekV4Deployment = () => {
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +33/-29
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #26931 - [AMD] dpsk-v4 swa loc cache support

- 链接: https://github.com/sgl-project/sglang/pull/26931
- 状态/时间: merged / 2026-06-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `d15a2dc72c81`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+13/-16，可读 patch 64 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] dpsk-v4 swa loc cache support」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「[AMD] dpsk-v4 swa loc cache support」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -641,8 +641,8 @@ def _forward_prepare_multi_stream_hip(; -731,8 +731,8 @@ def _forward_prepare(; symbols: _forward_prepare_multi_stream_hip, _forward_prepare，涉及 `_forward_prepare_multi_stream_hip, _forward_prepare`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +9/-12 (21 lines); hunks: -512,6 +512,13 @@ def translate_loc_from_full_to_swa(self, kv_indices: torch....; -758,12 +765,7 @@ def set_swa_key_buffer_radix_fused(; symbols: translate_loc_from_full_to_swa, get_cached_swa_loc, get_contiguous_buf_infos, set_swa_key_buffer_radix_fused，涉及 `translate_loc_from_full_to_swa, get_cached_swa_loc, get_contiguous_buf_infos`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -641,8 +641,8 @@ def _forward_prepare_multi_stream_hip(; -731,8 +731,8 @@ def _forward_prepare(; symbols: _forward_prepare_multi_stream_hip, _forward_prepare
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +9/-12 (21 lines); hunks: -512,6 +512,13 @@ def translate_loc_from_full_to_swa(self, kv_indices: torch....; -758,12 +765,7 @@ def set_swa_key_buffer_radix_fused(; symbols: translate_loc_from_full_to_swa, get_cached_swa_loc, get_contiguous_buf_infos, set_swa_key_buffer_radix_fused
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -641,8 +641,8 @@ def _forward_prepare_multi_stream_hip(
-            swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(
-                forward_batch.out_cache_loc
+            swa_loc = token_to_kv_pool.get_cached_swa_loc(
+                forward_batch.out_cache_loc, self.layer_id
@@ -731,8 +731,8 @@ def _forward_prepare(
-            swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -512,6 +512,13 @@ def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
+    def get_cached_swa_loc(self, raw_loc: torch.Tensor, layer_id: int) -> torch.Tensor:
+        if self._should_cache_swa:
+            if layer_id == self.start_layer or self.cached_loc is None:
+                self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)
+            return self.cached_loc
+        return self.translate_loc_from_full_to_swa(raw_loc)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +4/-4; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +9/-12
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26209 - Add FP4 Indexer for DeepSeek V4

- 链接: https://github.com/sgl-project/sglang/pull/26209
- 状态/时间: merged / 2026-06-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；关联提交 `301bcf08726b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+1177/-33，可读 patch 1505 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add FP4 Indexer for DeepSeek V4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「Add FP4 Indexer for DeepSeek V4」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +3/-0 (3 lines); hunks: -366,6 +366,9 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +34/-3 (37 lines); hunks: -271,13 +271,17 @@ def __init__(; -346,6 +350,23 @@ def set_index_fused(; symbols: __init__, get_bytes_per_token, _create_buffer, set_index_fused，涉及 `__init__, get_bytes_per_token, _create_buffer`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +3/-0 (3 lines); hunks: -366,6 +366,9 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +34/-3 (37 lines); hunks: -271,13 +271,17 @@ def __init__(; -346,6 +350,23 @@ def set_index_fused(; symbols: __init__, get_bytes_per_token, _create_buffer, set_index_fused
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -366,6 +366,9 @@ def __init__(
+        self.enable_deepseek_v4_fp4_indexer: bool = (
+            model_runner.server_args.enable_deepseek_v4_fp4_indexer
+        )
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -271,13 +271,17 @@ def __init__(
+        self.use_fp4_indexer = get_global_server_args().enable_deepseek_v4_fp4_indexer
+    def get_bytes_per_token(self) -> int:
+        if self.use_fp4_indexer:
+            return self.index_head_dim // 2 + 4
+        return self.index_head_dim + 4
-        num_scales_per_token = self.index_head_dim // self.quant_block_size
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +3/-0; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +34/-3
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/deepseek_v4/test_fp4_indexer.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27049 - docs: add DeepSeek-V4 EPLB Waterfill tips

- 链接: https://github.com/sgl-project/sglang/pull/27049
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+41/-0，可读 patch 48 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: add DeepSeek-V4 EPLB Waterfill tips」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「docs: add DeepSeek-V4 EPLB Waterfill tips」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +41/-0 (41 lines); hunks: -115,6 +115,47 @@ The generator currently picks values on the **conservative*...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +41/-0 (41 lines); hunks: -115,6 +115,47 @@ The generator currently picks values on the **conservative*...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -115,6 +115,47 @@ The generator currently picks values on the **conservative** side (mirroring an
+**EPLB + DeepEP Waterfill (Experimental)**
+For recorded/static EPLB reproduction, first record an expert-distribution file by following
+[Capture expert selection distribution in MoE models](../../../docs/basic_usage/native_api.mdx#capture-expert-selection-distribution-in-moe-models).
+For reproduction runs, use the generated `expert_distribution_recorder_*.pt` as
+the initial expert location. **Please checkout to latest main branch for this feature.**
+For non-PD reproduction, use:
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +41/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27035 - docs: add DeepSeek V4 FP4 indexer usage

- 链接: https://github.com/sgl-project/sglang/pull/27035
- 状态/时间: merged / 2026-06-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+18/-0，可读 patch 32 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: add DeepSeek V4 FP4 indexer usage」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/docs/advanced_features/server_arguments.mdx`；技术摘要: 覆盖「docs: add DeepSeek V4 FP4 indexer usage」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/docs/advanced_features/server_arguments.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +12/-0 (12 lines); hunks: -156,6 +156,18 @@ MegaMoE is not supported with this DeepEP Waterfill recipe...；`docs_new/docs/advanced_features/server_arguments.mdx` modified +6/-0 (6 lines); hunks: -1218,6 +1218,12 @@ Please consult the documentation below and [server_args.p...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +12/-0 (12 lines); hunks: -156,6 +156,18 @@ MegaMoE is not supported with this DeepEP Waterfill recipe...
  - `docs_new/docs/advanced_features/server_arguments.mdx` modified +6/-0 (6 lines); hunks: -1218,6 +1218,12 @@ Please consult the documentation below and [server_args.p...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -156,6 +156,18 @@ MegaMoE is not supported with this DeepEP Waterfill recipe yet. Waterfill routes
+**FP4 Indexer (Experimental)**
+DeepSeek-V4 uses the default indexer path unless `--enable-deepseek-v4-fp4-indexer` is set. Enable this flag to use the experimental FP4 C4 indexer on SM100 GPUs with DeepGEMM FP4
+'''bash Command
+# Please use latest main branch for this feature
+sglang serve deepseek-ai/DeepSeek-V4-Flash \
+  --tp 4 \
diff -- docs_new/docs/advanced_features/server_arguments.mdx
@@ -1218,6 +1218,12 @@ Please consult the documentation below and [server_args.py](https://github.com/s
+    <tr>
+      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`--enable-deepseek-v4-fp4-indexer`</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Enable the experimental FP4 C4 indexer path for DeepSeek V4. When unset, SGLang keeps the defaul
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>`False`</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>bool flag (set to enable)</td>
+    </tr>
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +12/-0; `docs_new/docs/advanced_features/server_arguments.mdx` modified +6/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/docs/advanced_features/server_arguments.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #24880 - [PD & HiSparse] Add DeepSeek V4 support for HiSparse direct Prefill-to-Decode DRAM

- 链接: https://github.com/sgl-project/sglang/pull/24880
- 状态/时间: merged / 2026-06-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+477/-308，可读 patch 1103 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[PD & HiSparse] Add DeepSeek V4 support for HiSparse direct Prefill-to-Decode DRAM」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/mem_cache/hisparse_memory_pool.py`, `python/sglang/jit_kernel/tests/test_hisparse.py`, `test/registered/disaggregation/test_disaggregation_dsv4.py`；技术摘要: 覆盖「[PD & HiSparse] Add DeepSeek V4 support for HiSparse direct Prefill-to-Decode DRAM」；主要实现面是 `python/sglang/srt/mem_cache/hisparse_memory_pool.py`, `python/sglang/jit_kernel/tests/test_hisparse.py`, `test/registered/disaggregation/test_disaggregation_dsv4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/mem_cache/hisparse_memory_pool.py` modified +37/-129 (166 lines); hunks: -4,7 +4,6; -17,7 +16,6; symbols: free, DeepSeekV4SingleKVPoolHost, __init__, clear，涉及 `free, DeepSeekV4SingleKVPoolHost, __init__`；`python/sglang/jit_kernel/tests/test_hisparse.py` modified +128/-1 (129 lines); hunks: -3,7 +3,11; -26,6 +30,12; symbols: _host_cache, _dsv4_token_pattern, _write_dsv4_token, _read_dsv4_token，涉及 `_host_cache, _dsv4_token_pattern, _write_dsv4_token`；`test/registered/disaggregation/test_disaggregation_dsv4.py` modified +105/-1 (106 lines); hunks: -11,11 +11,15; -123,5 +127,105 @@ def start_decode(cls):; symbols: start_decode, TestDisaggregationDSV4HiSparseMooncake, setUpClass, start_prefill，涉及 `start_decode, TestDisaggregationDSV4HiSparseMooncake, setUpClass`；`python/sglang/jit_kernel/csrc/deepseek_v4/hisparse_transfer.cuh` removed +0/-82 (82 lines); hunks: -1,82 +0,0。
- 代码 diff 细节:
  - `python/sglang/srt/mem_cache/hisparse_memory_pool.py` modified +37/-129 (166 lines); hunks: -4,7 +4,6; -17,7 +16,6; symbols: free, DeepSeekV4SingleKVPoolHost, __init__, clear
  - `python/sglang/jit_kernel/tests/test_hisparse.py` modified +128/-1 (129 lines); hunks: -3,7 +3,11; -26,6 +30,12; symbols: _host_cache, _dsv4_token_pattern, _write_dsv4_token, _read_dsv4_token
  - `test/registered/disaggregation/test_disaggregation_dsv4.py` modified +105/-1 (106 lines); hunks: -11,11 +11,15; -123,5 +127,105 @@ def start_decode(cls):; symbols: start_decode, TestDisaggregationDSV4HiSparseMooncake, setUpClass, start_prefill
  - `python/sglang/jit_kernel/csrc/deepseek_v4/hisparse_transfer.cuh` removed +0/-82 (82 lines); hunks: -1,82 +0,0
  - `python/sglang/jit_kernel/csrc/hisparse.cuh` modified +61/-4 (65 lines); hunks: -52,6 +52,62 @@ transfer_item_warp(int32_t lane_id, const void* src_addr, voi...; -89,7 +145,7 @@ struct SmemLayout {
- 关键代码摘录:

```diff
diff -- python/sglang/srt/mem_cache/hisparse_memory_pool.py
@@ -4,7 +4,6 @@
-import psutil
@@ -17,7 +16,6 @@
-from sglang.srt.mem_cache.memory_pool_host import HiSparseHostPoolMixin
@@ -384,121 +382,6 @@ def free(self, free_index: torch.Tensor):
-class DeepSeekV4SingleKVPoolHost(HiSparseHostPoolMixin):
-    def __init__(
diff -- python/sglang/jit_kernel/tests/test_hisparse.py
@@ -3,7 +3,11 @@
-from sglang.jit_kernel.hisparse import load_cache_to_device_buffer_mla
+from sglang.jit_kernel.hisparse import (
+    load_cache_to_device_buffer_dsv4_mla,
+    load_cache_to_device_buffer_mla,
+    transfer_cache_dsv4_mla,
+)
diff -- test/registered/disaggregation/test_disaggregation_dsv4.py
@@ -11,11 +11,15 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/mem_cache/hisparse_memory_pool.py` modified +37/-129; `python/sglang/jit_kernel/csrc/deepseek_v4/hisparse_transfer.cuh` removed +0/-82; `python/sglang/jit_kernel/csrc/hisparse.cuh` modified +61/-4; `python/sglang/srt/mem_cache/memory_pool_host.py` modified +50/-8; `python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/kvcacheio.cuh` modified +14/-34; `python/sglang/srt/managers/hisparse_coordinator.py` modified +27/-17
  - tests: `python/sglang/jit_kernel/tests/test_hisparse.py` modified +128/-1; `test/registered/disaggregation/test_disaggregation_dsv4.py` modified +105/-1
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_hisparse.py`, `test/registered/disaggregation/test_disaggregation_dsv4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27404 - Remove DeepSeek V4 release Docker workflow

- 链接: https://github.com/sgl-project/sglang/pull/27404
- 状态/时间: merged / 2026-06-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-149，可读 patch 150 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Remove DeepSeek V4 release Docker workflow」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `.github/workflows/release-docker-deepseek-v4.yml`；技术摘要: 覆盖「Remove DeepSeek V4 release Docker workflow」；主要实现面是 `.github/workflows/release-docker-deepseek-v4.yml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.github/workflows/release-docker-deepseek-v4.yml` removed +0/-149 (149 lines); hunks: -1,149 +0,0。
- 代码 diff 细节:
  - `.github/workflows/release-docker-deepseek-v4.yml` removed +0/-149 (149 lines); hunks: -1,149 +0,0
- 关键代码摘录:

```diff
diff -- .github/workflows/release-docker-deepseek-v4.yml
@@ -1,149 +0,0 @@
-name: Build and Push DeepSeek-V4 Docker Images
-# Builds the 4 Dockerfiles added in #23600 from the deepseek_v4 branch and
-# pushes them to Docker Hub. Each Dockerfile is single-arch and does its own
-# `git clone -b deepseek_v4` inside, so no build context source is required
-# beyond the Dockerfiles themselves and `--no-cache` is mandatory.
-on:
```

- 已读文件:
  - ci: `.github/workflows/release-docker-deepseek-v4.yml` removed +0/-149
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #27152 - [bugfix][AMD] AttributeError and warp mask bugs in DeepSeek V4 FP4 indexer

- 链接: https://github.com/sgl-project/sglang/pull/27152
- 状态/时间: merged / 2026-06-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`；关联提交 `3030119ef7cb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+8/-2，可读 patch 24 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bugfix][AMD] AttributeError and warp mask bugs in DeepSeek V4 FP4 indexer」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`；技术摘要: 覆盖「[bugfix][AMD] AttributeError and warp mask bugs in DeepSeek V4 FP4 indexer」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +3/-1 (4 lines); hunks: -373,7 +373,9 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +3/-1 (4 lines); hunks: -373,7 +373,9 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -373,7 +373,9 @@ def __init__(
+        self.enable_deepseek_v4_fp4_indexer: bool = (
+            model_runner.server_args.enable_deepseek_v4_fp4_indexer
+        )
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27191 - Fix DeepSeek V4 DP reduce scatter when use attention DP + MoE TP

- 链接: https://github.com/sgl-project/sglang/pull/27191
- 状态/时间: merged / 2026-06-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `4c8a022f38e3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-2，可读 patch 33 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix DeepSeek V4 DP reduce scatter when use attention DP + MoE TP」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「Fix DeepSeek V4 DP reduce scatter when use attention DP + MoE TP」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +10/-2 (12 lines); hunks: -59,6 +59,7; -67,7 +68,7; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +10/-2 (12 lines); hunks: -59,6 +59,7; -67,7 +68,7; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -59,6 +59,7 @@
+    get_dp_global_num_tokens,
@@ -67,7 +68,7 @@
-from sglang.srt.layers.moe import get_moe_a2a_backend
+from sglang.srt.layers.moe import get_moe_a2a_backend, should_use_dp_reduce_scatterv
@@ -1430,7 +1431,14 @@ def forward(
-            dp_scatter(hidden_states, global_hidden_states, forward_batch)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +10/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26885 - Cookbook renovation

- 链接: https://github.com/sgl-project/sglang/pull/26885
- 状态/时间: merged / 2026-06-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+6692/-1693，可读 patch 8494 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Cookbook renovation」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/_playground.jsx`；技术摘要: 覆盖「Cookbook renovation」；主要实现面是 `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/_playground.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` added +1222/-0 (1222 lines); hunks: -0,0 +1,1222；`docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` added +306/-0 (306 lines); hunks: -0,0 +1,306；`docs_new/src/snippets/_playground.jsx` added +2048/-0 (2048 lines); hunks: -0,0 +1,2048；`docs_new/src/snippets/_deployment.jsx` added +1277/-0 (1277 lines); hunks: -0,0 +1,1277。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` added +1222/-0 (1222 lines); hunks: -0,0 +1,1222
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` added +306/-0 (306 lines); hunks: -0,0 +1,306
  - `docs_new/src/snippets/_playground.jsx` added +2048/-0 (2048 lines); hunks: -0,0 +1,2048
  - `docs_new/src/snippets/_deployment.jsx` added +1277/-0 (1277 lines); hunks: -0,0 +1,1277
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` removed +0/-1263 (1263 lines); hunks: -1,1263 +0,0
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -0,0 +1,1222 @@
+// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
+// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
+export const config = {
+  modelName: "DeepSeek-V4",
+  supportedHardware: [
+    "h100", "h200", "b200", "b300", "gb200", "gb300",
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx
@@ -0,0 +1,306 @@
+// DeepSeek-V4 per-cell benchmark numbers, keyed by the same `match` tuple as
+// deepseek-v4.jsx cells. See _deployment.jsx for the speed/accuracy schema.
+// Measured on sglang v0.5.12.post1.
+export const benchmarks = [
+  // ====================================================================
+  // B200 + FP4
diff -- docs_new/src/snippets/_playground.jsx
@@ -0,0 +1,2048 @@
```

- 已读文件:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` added +1222/-0; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` added +306/-0; `docs_new/src/snippets/_playground.jsx` added +2048/-0; `docs_new/src/snippets/_deployment.jsx` added +1277/-0; `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` removed +0/-1263; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +138/-430
  - ci: `.github/ISSUE_TEMPLATE/3-playground-verified-cell.yml` added +109/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/_deployment.jsx`, `docs_new/src/snippets/_playground.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27289 - [ROCm] dsv4: remove the redundant fp8 scale transpose-copy on decode

- 链接: https://github.com/sgl-project/sglang/pull/27289
- 状态/时间: merged / 2026-06-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `ea1d190ed026`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+20/-3，可读 patch 142 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm] dsv4: remove the redundant fp8 scale transpose-copy on decode」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[ROCm] dsv4: remove the redundant fp8 scale transpose-copy on decode」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +2/-0 (2 lines); hunks: -97,6 +97,7; -151,6 +152,7 @@ def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):; symbols: _fused_rmsnorm_fp8_quant，涉及 `_fused_rmsnorm_fp8_quant`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +2/-0 (2 lines); hunks: -97,6 +97,7; -151,6 +152,7 @@ def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):; symbols: _fused_rmsnorm_fp8_quant
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -97,6 +97,7 @@
+from sglang.srt.models.deepseek_common.utils import _use_aiter_bpreshuffle_gfx95
@@ -151,6 +152,7 @@ def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):
+        transpose_scale=_use_aiter_bpreshuffle_gfx95,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25195 - [BCG] Support breakable CUDA graph for DeepSeek V4 DP attention

- 链接: https://github.com/sgl-project/sglang/pull/25195
- 状态/时间: merged / 2026-06-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`；关联提交 `ca66e6fb5e5d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+726/-66，可读 patch 1223 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BCG] Support breakable CUDA graph for DeepSeek V4 DP attention」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`；技术摘要: 覆盖「[BCG] Support breakable CUDA graph for DeepSeek V4 DP attention」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +251/-26 (277 lines); hunks: -184,6 +184,47 @@ def copy_(self, other: DSV4AttnMetadata) -> None:; -312,6 +353,24 @@ def copy_(self, other: DSV4Metadata):; symbols: copy_, refresh_for_breakable_cuda_graph_replay_, init_compression_metadata，涉及 `copy_, refresh_for_breakable_cuda_graph_replay_, init_compression_metadata`；`python/sglang/srt/models/deepseek_v4.py` modified +86/-10 (96 lines); hunks: -27,6 +27,8; -81,6 +83,12; symbols: _freqs_cis_to_cos_sin, deepseek_v4_attention_with_output, _rms_normalize_kernel, forward，涉及 `_freqs_cis_to_cos_sin, deepseek_v4_attention_with_output, _rms_normalize_kernel`；`test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +166/-0 (166 lines); hunks: -338,6 +338,172 @@ def test_runner_mode_production_eagle_draft_extend_cuda_gr...; symbols: test_runner_mode_production_eagle_draft_extend_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract, _make_core_metadata, test_bcg_is_explicit_and_dsv4_backend_opt_in_only，涉及 `test_runner_mode_production_eagle_draft_extend_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract, _make_core_metadata`；`test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +51/-0 (51 lines); hunks: -156,5 +156,56 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4BreakableCudaGraphB200, setUpClass，涉及 `tearDownClass, TestDSV4FlashFP4BreakableCudaGraphB200, setUpClass`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +251/-26 (277 lines); hunks: -184,6 +184,47 @@ def copy_(self, other: DSV4AttnMetadata) -> None:; -312,6 +353,24 @@ def copy_(self, other: DSV4Metadata):; symbols: copy_, refresh_for_breakable_cuda_graph_replay_, init_compression_metadata
  - `python/sglang/srt/models/deepseek_v4.py` modified +86/-10 (96 lines); hunks: -27,6 +27,8; -81,6 +83,12; symbols: _freqs_cis_to_cos_sin, deepseek_v4_attention_with_output, _rms_normalize_kernel, forward
  - `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +166/-0 (166 lines); hunks: -338,6 +338,172 @@ def test_runner_mode_production_eagle_draft_extend_cuda_gr...; symbols: test_runner_mode_production_eagle_draft_extend_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract, _make_core_metadata, test_bcg_is_explicit_and_dsv4_backend_opt_in_only
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +51/-0 (51 lines); hunks: -156,5 +156,56 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4BreakableCudaGraphB200, setUpClass
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -184,6 +184,47 @@ def copy_(self, other: DSV4AttnMetadata) -> None:
+    def refresh_for_breakable_cuda_graph_replay_(self, other: DSV4AttnMetadata) -> None:
+        assert self.c4_sparse_topk == other.c4_sparse_topk
+        assert self.page_size == other.page_size
+        assert self.cuda_int32_kwargs == other.cuda_int32_kwargs
+        tensor_copy_fields = [
+            "raw_out_loc",
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -27,6 +27,8 @@
+from sglang.srt.compilation.compilation_config import register_split_op
+from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -81,6 +83,12 @@
+from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
+    eager_on_graph,
+)
diff -- test/registered/attention/unittests/dsv4/test_deepseek_v4.py
@@ -338,6 +338,172 @@ def test_runner_mode_production_eagle_draft_extend_cuda_graph_runner_cases(self)
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +251/-26; `python/sglang/srt/models/deepseek_v4.py` modified +86/-10
  - tests: `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +166/-0; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +51/-0
- 验证与风险: diff 自带测试面 `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27380 - [AMD] Add unified kv attention support in dpsk-v4

- 链接: https://github.com/sgl-project/sglang/pull/27380
- 状态/时间: merged / 2026-06-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `f2bcdb05086a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+2418/-84，可读 patch 2904 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add unified kv attention support in dpsk-v4」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；技术摘要: 覆盖「[AMD] Add unified kv attention support in dpsk-v4」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +244/-0 (244 lines); hunks: -113,11 +113,28 @@ class DSV4AttnMetadata:; -157,10 +174,23 @@ def copy_(self, other: DSV4AttnMetadata) -> None:; symbols: DSV4AttnMetadata, copy_, init_compression_metadata, init_flashmla_related，涉及 `DSV4AttnMetadata, copy_, init_compression_metadata`；`python/sglang/srt/models/deepseek_v4.py` modified +76/-27 (103 lines); hunks: -778,8 +778,17 @@ def _forward_prepare(; -797,15 +806,33 @@ def _forward_prepare(; symbols: _forward_prepare, forward，涉及 `_forward_prepare, forward`；`python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +1/-0 (1 lines); hunks: -238,6 +238,7 @@ def init_compression_metadata(self):; symbols: init_compression_metadata，涉及 `init_compression_metadata`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +149/-42 (191 lines); hunks: -374,6 +374,65 @@ class DeepSeekV4LayerItem(NamedTuple):; -395,6 +454,7 @@ def __init__(; symbols: DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool, __init__, get_unified_kv，涉及 `DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +244/-0 (244 lines); hunks: -113,11 +113,28 @@ class DSV4AttnMetadata:; -157,10 +174,23 @@ def copy_(self, other: DSV4AttnMetadata) -> None:; symbols: DSV4AttnMetadata, copy_, init_compression_metadata, init_flashmla_related
  - `python/sglang/srt/models/deepseek_v4.py` modified +76/-27 (103 lines); hunks: -778,8 +778,17 @@ def _forward_prepare(; -797,15 +806,33 @@ def _forward_prepare(; symbols: _forward_prepare, forward
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +1/-0 (1 lines); hunks: -238,6 +238,7 @@ def init_compression_metadata(self):; symbols: init_compression_metadata
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +149/-42 (191 lines); hunks: -374,6 +374,65 @@ class DeepSeekV4LayerItem(NamedTuple):; -395,6 +454,7 @@ def __init__(; symbols: DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool, __init__, get_unified_kv
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -113,11 +113,28 @@ class DSV4AttnMetadata:
+    c4_sparse_topk_lengths_raw: torch.Tensor = field(init=False)
+    c4_sparse_raw_indices: Optional[torch.Tensor] = field(init=False, default=None)
+    c128_topk_lengths_raw: Optional[torch.Tensor] = None
+    # unified_kv: per-forward prebuilt ragged decode index
+    unified_swa_indices: Optional[torch.Tensor] = None
+    unified_swa_indptr: Optional[torch.Tensor] = None
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -778,8 +778,17 @@ def _forward_prepare(
-        if self.use_fused_qk_norm_rope:
+        from sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate import (
+            is_unified_kv_triton,
+        )
+        unified = is_unified_kv_triton()
+        is_decode = forward_batch.forward_mode.is_decode_or_idle()
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -238,6 +238,7 @@ def init_compression_metadata(self):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +244/-0; `python/sglang/srt/models/deepseek_v4.py` modified +76/-27; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +1/-0; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +149/-42
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh`, `python/sglang/jit_kernel/dsv4/compress.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27529 - [AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase

- 链接: https://github.com/sgl-project/sglang/pull/27529
- 状态/时间: merged / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+177/-88，可读 patch 574 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/dsv4/compressor.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`；技术摘要: 覆盖「[AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase」；主要实现面是 `python/sglang/srt/layers/attention/dsv4/compressor.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +4/-0 (4 lines); hunks: -395,6 +395,10 @@ def apply_ape_hotfix(self):; symbols: apply_ape_hotfix, get_state_pool，涉及 `apply_ape_hotfix, get_state_pool`；`python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh` modified +93/-46 (139 lines); hunks: -74,23 +74,27 @@ struct C4Trait {; -102,28 +106,61 @@ SGL_DEVICE void c4_forward(；`python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +65/-39 (104 lines); hunks: -89,10 +89,10 @@ struct C128Trait {; -101,11 +101,15 @@ SGL_DEVICE void c128_forward(；`python/sglang/jit_kernel/dsv4/compress.py` modified +15/-3 (18 lines); hunks: -44,11 +44,14 @@ def _jit_compress_norm_rope_module(; -324,8 +327,17 @@ def compress_forward(; symbols: _jit_compress_norm_rope_module, _jit_compress_module, compress_forward，涉及 `_jit_compress_norm_rope_module, _jit_compress_module, compress_forward`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +4/-0 (4 lines); hunks: -395,6 +395,10 @@ def apply_ape_hotfix(self):; symbols: apply_ape_hotfix, get_state_pool
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh` modified +93/-46 (139 lines); hunks: -74,23 +74,27 @@ struct C4Trait {; -102,28 +106,61 @@ SGL_DEVICE void c4_forward(
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +65/-39 (104 lines); hunks: -89,10 +89,10 @@ struct C128Trait {; -101,11 +101,15 @@ SGL_DEVICE void c128_forward(
  - `python/sglang/jit_kernel/dsv4/compress.py` modified +15/-3 (18 lines); hunks: -44,11 +44,14 @@ def _jit_compress_norm_rope_module(; -324,8 +327,17 @@ def compress_forward(; symbols: _jit_compress_norm_rope_module, _jit_compress_module, compress_forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/dsv4/compressor.py
@@ -395,6 +395,10 @@ def apply_ape_hotfix(self):
+        if _use_aiter:
+            self.ape.data = self.ape.data.to(torch.bfloat16)
+            self.norm.weight.data = self.norm.weight.data.to(torch.bfloat16)
diff -- python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh
@@ -74,23 +74,27 @@ struct C4Trait {
-template <typename Trait, bool kUsePDL, typename InFloat, typename OutFloat>
+template <typename Trait, bool kUsePDL, typename BufFloat, typename InFloat, typename OutFloat>
-    const InFloat* kv_buf_0,  // overlap [4n - 4, 4n - 1]
-    const InFloat* kv_buf_1,  // normal [4n + 0, 4n + 3]
-    const InFloat* kv_src,    // ragged pointer at position = 4n + 3
+    const BufFloat* kv_buf_0,  // overlap [4n - 4, 4n - 1]
diff -- python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh
@@ -89,10 +89,10 @@ struct C128Trait {
-template <typename Trait, bool kUsePDL, typename InFloat, typename OutFloat>
+template <typename Trait, bool kUsePDL, typename BufFloat, typename InFloat, typename OutFloat>
-    const InFloat* kv_buf,  // [128n, 128n + 127]
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +4/-0; `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh` modified +93/-46; `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +65/-39; `python/sglang/jit_kernel/dsv4/compress.py` modified +15/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/dsv4/compress.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27830 - [Docs] Restore right-hand ToC on the DeepSeek-V4 cookbook page

- 链接: https://github.com/sgl-project/sglang/pull/27830
- 状态/时间: merged / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+12/-4，可读 patch 44 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Restore right-hand ToC on the DeepSeek-V4 cookbook page」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[Docs] Restore right-hand ToC on the DeepSeek-V4 cookbook page」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-1 (1 lines); hunks: -2,7 +2,6。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-1 (1 lines); hunks: -2,7 +2,6
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -2,7 +2,6 @@
-mode: wide
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27747 - fix: DSV4 BCG compress-prefill plan OOB on underfilled (tiny) prefill replay

- 链接: https://github.com/sgl-project/sglang/pull/27747
- 状态/时间: merged / 2026-06-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: DSV4 BCG compress-prefill plan OOB on underfilled (tiny) prefill replay」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`；技术摘要: 覆盖「fix: DSV4 BCG compress-prefill plan OOB on underfilled (tiny) prefill replay」；主要实现面是 `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh` modified +5/-1 (6 lines); hunks: -203,7 +203,11 @@ __global__ __launch_bounds__(1024, 1) //。
- 代码 diff 细节:
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh` modified +5/-1 (6 lines); hunks: -203,7 +203,11 @@ __global__ __launch_bounds__(1024, 1) //
- 关键代码摘录:

```diff
diff -- python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh
@@ -203,7 +203,11 @@ __global__ __launch_bounds__(1024, 1)  //
-    for (uint32_t k = tx; k < num_q; k += block_size) {
+    // num_q is the padded buffer size (graph bucket), not the work size: cap the
+    // loop at the real token count so batch_id = k / E stays < batch_size on an
+    // underfilled replay; Stage D pads [counter, num_q) with invalid.
+    const uint32_t num_real_q = params.batch_size * E;
+    for (uint32_t k = tx; k < num_real_q; k += block_size) {
```

- 已读文件:
  - runtime: `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27919 - Revert "[AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase"

- 链接: https://github.com/sgl-project/sglang/pull/27919
- 状态/时间: merged / 2026-06-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+88/-177，可读 patch 574 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "[AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase"」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/dsv4/compressor.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`；技术摘要: 覆盖「Revert "[AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase"」；主要实现面是 `python/sglang/srt/layers/attention/dsv4/compressor.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +0/-4 (4 lines); hunks: -395,10 +395,6 @@ def apply_ape_hotfix(self):; symbols: apply_ape_hotfix, get_state_pool，涉及 `apply_ape_hotfix, get_state_pool`；`python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh` modified +46/-93 (139 lines); hunks: -74,27 +74,23 @@ struct C4Trait {; -106,61 +102,28 @@ SGL_DEVICE void c4_forward(；`python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +39/-65 (104 lines); hunks: -89,10 +89,10 @@ struct C128Trait {; -101,15 +101,11 @@ SGL_DEVICE void c128_forward(；`python/sglang/jit_kernel/dsv4/compress.py` modified +3/-15 (18 lines); hunks: -44,14 +44,11 @@ def _jit_compress_norm_rope_module(; -327,17 +324,8 @@ def compress_forward(; symbols: _jit_compress_norm_rope_module, _jit_compress_module, compress_forward，涉及 `_jit_compress_norm_rope_module, _jit_compress_module, compress_forward`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +0/-4 (4 lines); hunks: -395,10 +395,6 @@ def apply_ape_hotfix(self):; symbols: apply_ape_hotfix, get_state_pool
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh` modified +46/-93 (139 lines); hunks: -74,27 +74,23 @@ struct C4Trait {; -106,61 +102,28 @@ SGL_DEVICE void c4_forward(
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +39/-65 (104 lines); hunks: -89,10 +89,10 @@ struct C128Trait {; -101,15 +101,11 @@ SGL_DEVICE void c128_forward(
  - `python/sglang/jit_kernel/dsv4/compress.py` modified +3/-15 (18 lines); hunks: -44,14 +44,11 @@ def _jit_compress_norm_rope_module(; -327,17 +324,8 @@ def compress_forward(; symbols: _jit_compress_norm_rope_module, _jit_compress_module, compress_forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/dsv4/compressor.py
@@ -395,10 +395,6 @@ def apply_ape_hotfix(self):
-        if _use_aiter:
-            self.ape.data = self.ape.data.to(torch.bfloat16)
-            self.norm.weight.data = self.norm.weight.data.to(torch.bfloat16)
diff -- python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh
@@ -74,27 +74,23 @@ struct C4Trait {
-template <typename Trait, bool kUsePDL, typename BufFloat, typename InFloat, typename OutFloat>
+template <typename Trait, bool kUsePDL, typename InFloat, typename OutFloat>
-    const BufFloat* kv_buf_0,  // overlap [4n - 4, 4n - 1]
-    const BufFloat* kv_buf_1,  // normal [4n + 0, 4n + 3]
-    const InFloat* kv_src,     // ragged pointer at position = 4n + 3
+    const InFloat* kv_buf_0,  // overlap [4n - 4, 4n - 1]
diff -- python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh
@@ -89,10 +89,10 @@ struct C128Trait {
-template <typename Trait, bool kUsePDL, typename BufFloat, typename InFloat, typename OutFloat>
+template <typename Trait, bool kUsePDL, typename InFloat, typename OutFloat>
-    const BufFloat* kv_buf,  // [128n, 128n + 127]
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +0/-4; `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh` modified +46/-93; `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +39/-65; `python/sglang/jit_kernel/dsv4/compress.py` modified +3/-15
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/dsv4/compress.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27964 - [Spec] Retire Spec V1

- 链接: https://github.com/sgl-project/sglang/pull/27964
- 状态/时间: merged / 2026-06-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 46 个文件，+111/-252，可读 patch 1422 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Spec] Retire Spec V1」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`；技术摘要: 覆盖「[Spec] Retire Spec V1」；主要实现面是 `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass，涉及 `TestDeepseekMTP, setUpClass, tearDownClass`；`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do；`python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family，涉及 `handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp`；`docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu...。
- 代码 diff 细节:
  - `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do
  - `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family
  - `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu...
  - `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10 (21 lines); hunks: -1,6 +1,5; -31,7 +30,8 @@ class TestEagleConstrainedDecoding(; symbols: TestEagleConstrainedDecoding, setUpClass, tearDownClass, TestEagleConstrainedDecodingV2
- 关键代码摘录:

```diff
diff -- test/registered/ep/test_deepep_large.py
@@ -3,7 +3,6 @@
-from sglang.srt.environ import envs
@@ -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):
-        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
-            cls.process = popen_launch_server(
-                cls.model,
-                cls.base_url,
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx
@@ -1108,7 +1108,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1227,7 +1226,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1351,7 +1349,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1476,7 +1473,6 @@ do
diff -- python/sglang/srt/arg_groups/speculative_hook.py
@@ -1,9 +1,8 @@
```

- 已读文件:
  - tests: `test/registered/ep/test_deepep_large.py` modified +43/-44; `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10; `python/sglang/test/server_fixtures/standalone_fixture.py` modified +7/-8; `python/sglang/test/server_fixtures/spec_eagle_fixture.py` modified +6/-6
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64; `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_optimization.mdx` modified +3/-8
  - runtime: `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26
- 验证与风险: diff 自带测试面 `python/sglang/test/server_fixtures/spec_eagle_fixture.py`, `python/sglang/test/server_fixtures/standalone_fixture.py`, `test/manual/dsv4/test_dsv4_flash_mtp_tp8.py`, `test/manual/dsv4/test_dsv4_pro_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27973 - [DSV4] Use int64 for compressor out_loc tensors

- 链接: https://github.com/sgl-project/sglang/pull/27973
- 状态/时间: merged / 2026-06-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+19/-22，可读 patch 144 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Use int64 for compressor out_loc tensors」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/dsv4/compressor_v2.py`, `python/sglang/srt/layers/attention/dsv4/metadata_kernel.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh`；技术摘要: 覆盖「[DSV4] Use int64 for compressor out_loc tensors」；主要实现面是 `python/sglang/srt/layers/attention/dsv4/compressor_v2.py`, `python/sglang/srt/layers/attention/dsv4/metadata_kernel.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` modified +2/-5 (7 lines); hunks: -519,12 +519,9 @@ def forward_unified(; symbols: forward_unified，涉及 `forward_unified`；`python/sglang/srt/layers/attention/dsv4/metadata_kernel.py` modified +2/-2 (4 lines); hunks: -107,12 +107,12 @@ def _init_compressed_attn_metadata_triton(; symbols: _init_compressed_attn_metadata_triton，涉及 `_init_compressed_attn_metadata_triton`；`python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh` modified +14/-14 (28 lines); hunks: -44,7 +44,7 @@ struct FusedNormRopeStoreParams {; -90,7 +90,7 @@ INDEXER_KERNEL void fused_norm_rope_indexer(const __grid_const...；`test/registered/jit/deepseek_v4/test_fp4_indexer.py` modified +1/-1 (2 lines); hunks: -148,7 +148,7 @@ def test_fp4_fused_norm_rope_store_layout(num_tokens: int) -...; symbols: test_fp4_fused_norm_rope_store_layout，涉及 `test_fp4_fused_norm_rope_store_layout`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` modified +2/-5 (7 lines); hunks: -519,12 +519,9 @@ def forward_unified(; symbols: forward_unified
  - `python/sglang/srt/layers/attention/dsv4/metadata_kernel.py` modified +2/-2 (4 lines); hunks: -107,12 +107,12 @@ def _init_compressed_attn_metadata_triton(; symbols: _init_compressed_attn_metadata_triton
  - `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh` modified +14/-14 (28 lines); hunks: -44,7 +44,7 @@ struct FusedNormRopeStoreParams {; -90,7 +90,7 @@ INDEXER_KERNEL void fused_norm_rope_indexer(const __grid_const...
  - `test/registered/jit/deepseek_v4/test_fp4_indexer.py` modified +1/-1 (2 lines); hunks: -148,7 +148,7 @@ def test_fp4_fused_norm_rope_store_layout(num_tokens: int) -...; symbols: test_fp4_fused_norm_rope_store_layout
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/dsv4/compressor_v2.py
@@ -519,12 +519,9 @@ def forward_unified(
-                    # The v2 compressor writes directly into the raw C4 KV tensor.
-                    # HiSparse C4 therefore needs the physical C4 location here.
-                    # The compress kernel requires an int32 write location.
-                    out_loc = compress_kv_pool.translate_loc_to_hisparse_device(
+                    out_loc = compress_kv_pool._translate_loc_to_hisparse_device(
-                    ).to(torch.int32)
diff -- python/sglang/srt/layers/attention/dsv4/metadata_kernel.py
@@ -107,12 +107,12 @@ def _init_compressed_attn_metadata_triton(
-    c4_out_loc = torch.empty(bs, dtype=torch.int32, device=device)
+    c4_out_loc = torch.empty(bs, dtype=torch.int64, device=device)
-    c128_out_loc = torch.empty(bs, dtype=torch.int32, device=device)
+    c128_out_loc = torch.empty(bs, dtype=torch.int64, device=device)
diff -- python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh
@@ -44,7 +44,7 @@ struct FusedNormRopeStoreParams {
-  const int32_t* __restrict__ out_loc;
+  const int64_t* __restrict__ out_loc;
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` modified +2/-5; `python/sglang/srt/layers/attention/dsv4/metadata_kernel.py` modified +2/-2; `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh` modified +14/-14
  - tests: `test/registered/jit/deepseek_v4/test_fp4_indexer.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/jit/deepseek_v4/test_fp4_indexer.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27149 - [AMD] [CI] Add dsv4 accuracy PR gate to pr-test-amd-rocm720

- 链接: https://github.com/sgl-project/sglang/pull/27149
- 状态/时间: merged / 2026-06-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`；关联提交 `1cd5cb1220b9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+165/-20，可读 patch 306 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [CI] Add dsv4 accuracy PR gate to pr-test-amd-rocm720」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`；技术摘要: 覆盖「[AMD] [CI] Add dsv4 accuracy PR gate to pr-test-amd-rocm720」；主要实现面是 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +5/-1 (6 lines); hunks: -44,7 +44,7; -131,6 +131,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k，涉及 `test_a_gsm8k, test_b_perf_8k_1k`；`test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +5/-1 (6 lines); hunks: -44,7 +44,7; -131,6 +131,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k，涉及 `test_a_gsm8k, test_b_perf_8k_1k`；`test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +5/-1 (6 lines); hunks: -46,7 +46,7; -133,6 +133,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k，涉及 `test_a_gsm8k, test_b_perf_8k_1k`；`test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +5/-1 (6 lines); hunks: -46,7 +46,7; -133,6 +133,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k，涉及 `test_a_gsm8k, test_b_perf_8k_1k`。
- 代码 diff 细节:
  - `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +5/-1 (6 lines); hunks: -44,7 +44,7; -131,6 +131,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +5/-1 (6 lines); hunks: -44,7 +44,7; -131,6 +131,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +5/-1 (6 lines); hunks: -46,7 +46,7; -133,6 +133,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +5/-1 (6 lines); hunks: -46,7 +46,7; -133,6 +133,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k
- 关键代码摘录:

```diff
diff -- test/registered/amd/test_deepseek_v4_flash_fp4.py
@@ -44,7 +44,7 @@
-    "SGLANG_HACK_FLASHMLA_BACKEND": "triton",
+    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
@@ -131,6 +131,10 @@ def test_a_gsm8k(self):
+    @unittest.skipIf(
+        os.environ.get("SGLANG_DSV4_ACCURACY_ONLY") == "1",
+        "SGLANG_DSV4_ACCURACY_ONLY=1: accuracy-only run (skipping perf)",
diff -- test/registered/amd/test_deepseek_v4_flash_fp8.py
@@ -44,7 +44,7 @@
-    "SGLANG_HACK_FLASHMLA_BACKEND": "triton",
+    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
@@ -131,6 +131,10 @@ def test_a_gsm8k(self):
+    @unittest.skipIf(
+        os.environ.get("SGLANG_DSV4_ACCURACY_ONLY") == "1",
+        "SGLANG_DSV4_ACCURACY_ONLY=1: accuracy-only run (skipping perf)",
diff -- test/registered/amd/test_deepseek_v4_pro_fp4.py
@@ -46,7 +46,7 @@
```

- 已读文件:
  - tests: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +5/-1; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +5/-1; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +5/-1; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +5/-1
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28098 - Add DeepSeek V4 MTP acceptance length checks

- 链接: https://github.com/sgl-project/sglang/pull/28098
- 状态/时间: merged / 2026-06-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py`；关联提交 `a14d1a565639`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+36/-3，可读 patch 222 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add DeepSeek V4 MTP acceptance length checks」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`；技术摘要: 覆盖「Add DeepSeek V4 MTP acceptance length checks」；主要实现面是 `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +7/-0 (7 lines); hunks: -14,6 +14,7; -33,13 +34,16; symbols: TestDSV4FlashFP4B200Balanced_CP, setUpClass, tearDownClass, TestDSV4FlashFP4B200Balanced_CP_NonDeepEP，涉及 `TestDSV4FlashFP4B200Balanced_CP, setUpClass, tearDownClass`；`test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +7/-0 (7 lines); hunks: -13,6 +13,7; -32,13 +33,16; symbols: TestDSV4FlashFP4B200, setUpClass, tearDownClass, TestDSV4FlashFP4B200Balanced，涉及 `TestDSV4FlashFP4B200, setUpClass, tearDownClass`；`test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +7/-0 (7 lines); hunks: -13,6 +13,7; -41,13 +42,16 @@ def _flashinfer_has_sm90_cutlass_mxfp4() -> bool:; symbols: _flashinfer_has_sm90_cutlass_mxfp4, TestDSV4FlashFP4H200, setUpClass, tearDownClass，涉及 `_flashinfer_has_sm90_cutlass_mxfp4, TestDSV4FlashFP4H200, setUpClass`；`test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +7/-0 (7 lines); hunks: -13,6 +13,7; -39,13 +40,16; symbols: TestDSV4FlashFP4B200W4A8MegaMoE, setUpClass, tearDownClass, TestDSV4FlashFP4B200W4A4MegaMoE，涉及 `TestDSV4FlashFP4B200W4A8MegaMoE, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +7/-0 (7 lines); hunks: -14,6 +14,7; -33,13 +34,16; symbols: TestDSV4FlashFP4B200Balanced_CP, setUpClass, tearDownClass, TestDSV4FlashFP4B200Balanced_CP_NonDeepEP
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +7/-0 (7 lines); hunks: -13,6 +13,7; -32,13 +33,16; symbols: TestDSV4FlashFP4B200, setUpClass, tearDownClass, TestDSV4FlashFP4B200Balanced
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +7/-0 (7 lines); hunks: -13,6 +13,7; -41,13 +42,16 @@ def _flashinfer_has_sm90_cutlass_mxfp4() -> bool:; symbols: _flashinfer_has_sm90_cutlass_mxfp4, TestDSV4FlashFP4H200, setUpClass, tearDownClass
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +7/-0 (7 lines); hunks: -13,6 +13,7; -39,13 +40,16; symbols: TestDSV4FlashFP4B200W4A8MegaMoE, setUpClass, tearDownClass, TestDSV4FlashFP4B200W4A4MegaMoE
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py` modified +4/-0 (4 lines); hunks: -14,6 +14,7; -29,13 +30,16; symbols: TestDSV4FlashFP8H200, setUpClass
- 关键代码摘录:

```diff
diff -- test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py
@@ -14,6 +14,7 @@
+from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
@@ -33,13 +34,16 @@
+    SpecDecodingMixin,
+    accept_length_thres = 1.8
+    bs_1_speed_thres = 100
@@ -82,13 +86,16 @@ def tearDownClass(cls):
diff -- test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py
@@ -13,6 +13,7 @@
+from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
@@ -32,13 +33,16 @@
+    SpecDecodingMixin,
+    accept_length_thres = 2.6
+    bs_1_speed_thres = 220
@@ -75,13 +79,16 @@ def tearDownClass(cls):
diff -- test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py
@@ -13,6 +13,7 @@
```

- 已读文件:
  - tests: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +7/-0; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +7/-0; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +7/-0; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +7/-0; `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py` modified +4/-0
- 验证与风险: diff 自带测试面 `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/disaggregation/test_disaggregation_dsv4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27954 - [dsv4] Pad MLA decode q-heads to 64 (not full n_heads) for FlashMLA head64 kernel

- 链接: https://github.com/sgl-project/sglang/pull/27954
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `b3be2e74026b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+21/-5，可读 patch 55 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[dsv4] Pad MLA decode q-heads to 64 (not full n_heads) for FlashMLA head64 kernel」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[dsv4] Pad MLA decode q-heads to 64 (not full n_heads) for FlashMLA head64 kernel」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +21/-5 (26 lines); hunks: -382,6 +382,9 @@ def __init__(; -898,10 +901,23 @@ def forward(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +21/-5 (26 lines); hunks: -382,6 +382,9 @@ def __init__(; -898,10 +901,23 @@ def forward(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -382,6 +382,9 @@ def __init__(
+        self._attn_sink_local: Optional[torch.Tensor] = (
+            self.attn_sink if attn_tp_size == 1 else None
+        )
@@ -898,10 +901,23 @@ def forward(
-            q_padded = x.new_empty(x.shape[0], self.n_heads, self.head_dim)
-            rank = self.tp_rank
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +21/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26471 - DeepSeek-V4 Online Compress support MTP

- 链接: https://github.com/sgl-project/sglang/pull/26471
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；关联提交 `063ab89ac168`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+1276/-49，可读 patch 1896 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「DeepSeek-V4 Online Compress support MTP」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`；技术摘要: 覆盖「DeepSeek-V4 Online Compress support MTP」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +159/-11 (170 lines); hunks: -35,6 +35,7; -79,6 +80,37; symbols: _get_logical_forward_mode, _get_target_verify_bs, _create_dummy_paged_compress_data, _copy_or_replace，涉及 `_get_logical_forward_mode, _get_target_verify_bs, _create_dummy_paged_compress_data`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +28/-1 (29 lines); hunks: -35,7 +35,8 @@ def get_compress_state_ring_size(; -458,6 +459,7 @@ def __init__(; symbols: get_compress_state_ring_size, __init__, _init_paged_compress_states, get_attention_compress_states，涉及 `get_compress_state_ring_size, __init__, _init_paged_compress_states`；`python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +12/-1 (13 lines); hunks: -88,18 +88,29 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +159/-11 (170 lines); hunks: -35,6 +35,7; -79,6 +80,37; symbols: _get_logical_forward_mode, _get_target_verify_bs, _create_dummy_paged_compress_data, _copy_or_replace
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +28/-1 (29 lines); hunks: -35,7 +35,8 @@ def get_compress_state_ring_size(; -458,6 +459,7 @@ def __init__(; symbols: get_compress_state_ring_size, __init__, _init_paged_compress_states, get_attention_compress_states
  - `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +12/-1 (13 lines); hunks: -88,18 +88,29 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -35,6 +35,7 @@
+from sglang.jit_kernel.dsv4.online_c128_mtp import OnlineC128MTPController
@@ -79,6 +80,37 @@
+def _get_logical_forward_mode(forward_batch: ForwardBatch) -> ForwardMode:
+    # IDLE is a real per-DP-rank mode. Do not let a stale _original_forward_mode
+    # from a reused/padded ForwardBatch turn an empty rank into TARGET_VERIFY.
+    if forward_batch.forward_mode.is_idle():
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -35,7 +35,8 @@ def get_compress_state_ring_size(
-        assert not is_speculative, "online c128 does not support MTP"
+        if is_speculative and not envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
+            raise AssertionError("online c128 does not support MTP")
@@ -458,6 +459,7 @@ def __init__(
+        online_mtp_max_draft_tokens: int = 0,
@@ -493,6 +495,12 @@ def __init__(
diff -- python/sglang/srt/mem_cache/deepseek_v4_compress_state.py
@@ -88,18 +88,29 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +159/-11; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +28/-1; `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +12/-1
- 验证与风险: diff 自带测试面 `test/registered/jit/benchmark/bench_online_c128_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28392 - [AMD] Annotate ATOM source for imported v4 unified attention kernels

- 链接: https://github.com/sgl-project/sglang/pull/28392
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；关联提交 `800aaefc9e9e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+10/-0，可读 patch 38 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Annotate ATOM source for imported v4 unified attention kernels」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「[AMD] Annotate ATOM source for imported v4 unified attention kernels」；主要实现面是 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-0 (1 lines); hunks: -375,6 +375,7 @@ class DeepSeekV4LayerItem(NamedTuple):; symbols: DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool，涉及 `DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool`。
- 代码 diff 细节:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-0 (1 lines); hunks: -375,6 +375,7 @@ class DeepSeekV4LayerItem(NamedTuple):; symbols: DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool
- 关键代码摘录:

```diff
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -375,6 +375,7 @@ class DeepSeekV4LayerItem(NamedTuple):
+# The following kv pool follows ATOM's unified_kv kernel layout.
```

- 已读文件:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/dsv4/unified_kv_kernels/paged_decode.py`, `python/sglang/srt/layers/attention/dsv4/unified_kv_kernels/paged_decode_indices.py`, `python/sglang/srt/layers/attention/dsv4/unified_kv_kernels/paged_prefill.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28290 - [AMD] Test DeepSeek V4 FlashMLA backend variants nightly

- 链接: https://github.com/sgl-project/sglang/pull/28290
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`；关联提交 `0fc2bc4a8bb4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+48/-18，可读 patch 204 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Test DeepSeek V4 FlashMLA backend variants nightly」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`；技术摘要: 覆盖「[AMD] Test DeepSeek V4 FlashMLA backend variants nightly」；主要实现面是 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +4/-3 (7 lines); hunks: -34,6 +34,7; -44,7 +45,7; symbols: test_a_gsm8k, test_b_perf_8k_1k，涉及 `test_a_gsm8k, test_b_perf_8k_1k`；`test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +4/-3 (7 lines); hunks: -34,6 +34,7; -44,7 +45,7; symbols: test_a_gsm8k, test_b_perf_8k_1k，涉及 `test_a_gsm8k, test_b_perf_8k_1k`；`test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +4/-3 (7 lines); hunks: -36,6 +36,7; -46,7 +47,7; symbols: test_a_gsm8k, test_b_perf_8k_1k，涉及 `test_a_gsm8k, test_b_perf_8k_1k`；`test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +4/-3 (7 lines); hunks: -36,6 +36,7; -46,7 +47,7; symbols: test_a_gsm8k, test_b_perf_8k_1k，涉及 `test_a_gsm8k, test_b_perf_8k_1k`。
- 代码 diff 细节:
  - `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +4/-3 (7 lines); hunks: -34,6 +34,7; -44,7 +45,7; symbols: test_a_gsm8k, test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +4/-3 (7 lines); hunks: -34,6 +34,7; -44,7 +45,7; symbols: test_a_gsm8k, test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +4/-3 (7 lines); hunks: -36,6 +36,7; -46,7 +47,7; symbols: test_a_gsm8k, test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +4/-3 (7 lines); hunks: -36,6 +36,7; -46,7 +47,7; symbols: test_a_gsm8k, test_b_perf_8k_1k
- 关键代码摘录:

```diff
diff -- test/registered/amd/test_deepseek_v4_flash_fp4.py
@@ -34,6 +34,7 @@
+FLASHMLA_BACKEND = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "unified_kv_triton")
@@ -44,7 +45,7 @@
-    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
+    "SGLANG_HACK_FLASHMLA_BACKEND": FLASHMLA_BACKEND,
@@ -126,7 +127,7 @@ def test_a_gsm8k(self):
-                f"### test_gsm8k (deepseek-v4-flash-fp4)\n"
diff -- test/registered/amd/test_deepseek_v4_flash_fp8.py
@@ -34,6 +34,7 @@
+FLASHMLA_BACKEND = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "unified_kv_triton")
@@ -44,7 +45,7 @@
-    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
+    "SGLANG_HACK_FLASHMLA_BACKEND": FLASHMLA_BACKEND,
@@ -126,7 +127,7 @@ def test_a_gsm8k(self):
-                f"### test_gsm8k (deepseek-v4-flash-fp8)\n"
diff -- test/registered/amd/test_deepseek_v4_pro_fp4.py
@@ -36,6 +36,7 @@
```

- 已读文件:
  - tests: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +4/-3; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +4/-3; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +4/-3; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +4/-3
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27928 - [AMD] Feat: Add prefill context parallel support for deepseek v4 unified kv attention

- 链接: https://github.com/sgl-project/sglang/pull/27928
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py`；关联提交 `a362ba9da37e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+213/-5，可读 patch 259 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Feat: Add prefill context parallel support for deepseek v4 unified kv attention」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py`；技术摘要: 覆盖「[AMD] Feat: Add prefill context parallel support for deepseek v4 unified kv attention」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +56/-5 (61 lines); hunks: -320,7 +320,7 @@ def apply_cp_reindex(self) -> None:; -342,6 +342,8 @@ def init_flashmla_related(self):; symbols: apply_cp_reindex, init_flashmla_related, _forward_unified_kv，涉及 `apply_cp_reindex, init_flashmla_related, _forward_unified_kv`；`python/sglang/srt/models/deepseek_v4.py` modified +13/-0 (13 lines); hunks: -819,6 +819,19 @@ def _forward_prepare(; symbols: _forward_prepare，涉及 `_forward_prepare`；`test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: TestDeepseekV4ProFp4CPInterleave, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV4ProFp4CPInterleave, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +56/-5 (61 lines); hunks: -320,7 +320,7 @@ def apply_cp_reindex(self) -> None:; -342,6 +342,8 @@ def init_flashmla_related(self):; symbols: apply_cp_reindex, init_flashmla_related, _forward_unified_kv
  - `python/sglang/srt/models/deepseek_v4.py` modified +13/-0 (13 lines); hunks: -819,6 +819,19 @@ def _forward_prepare(; symbols: _forward_prepare
  - `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: TestDeepseekV4ProFp4CPInterleave, setUpClass, tearDownClass, test_a_gsm8k
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -320,7 +320,7 @@ def apply_cp_reindex(self) -> None:
-    def init_flashmla_related(self):
+    def init_flashmla_related(self, is_prefill: bool = False):
@@ -342,6 +342,8 @@ def init_flashmla_related(self):
+        if is_prefill:
+            self.c4_sparse_raw_indices = torch.empty_like(self.c4_sparse_page_indices)
@@ -1187,6 +1189,49 @@ def _forward_unified_kv(
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -819,6 +819,19 @@ def _forward_prepare(
+                # HIP/ROCm-only: the unified_kv 2-source prefill path is exclusive
+                # to DeepseekV4HipRadixBackend. Guard with _is_hip so this CP
+                # all-gather never enters the NVIDIA (DeepseekV4AttnBackend) path.
+                if use_cp and _is_hip:
+                    # unified_kv + DSA CP: the 2-source prefill path needs the
+                    # FULL current-chunk KV (extend source + ring write), so
diff -- test/registered/amd/test_deepseek_v4_pro_fp4_cp.py
@@ -0,0 +1,144 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +56/-5; `python/sglang/srt/models/deepseek_v4.py` modified +13/-0
  - tests: `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` added +144/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28423 - [AMD] Update v4 amd cookbook

- 链接: https://github.com/sgl-project/sglang/pull/28423
- 状态/时间: merged / 2026-06-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+799/-6，可读 patch 871 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Update v4 amd cookbook」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[AMD] Update v4 amd cookbook」；主要实现面是 `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +724/-0 (724 lines); hunks: -7,6 +7,8 @@ export const config = {; -43,6 +45,10 @@ export const config = {；`docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +21/-0 (21 lines); hunks: -261,4 +261,25 @@ export const benchmarks = [；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +34/-4 (38 lines); hunks: -28,15 +28,15 @@ Then run the **Python** output of the command panel below in...; -47,6 +47,28 @@ docker run --gpus all \；`docs_new/src/snippets/_deployment.jsx` modified +20/-2 (22 lines); hunks: -504,9 +504,27 @@ export const Deployment = ({ config, benchmarks }) => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +724/-0 (724 lines); hunks: -7,6 +7,8 @@ export const config = {; -43,6 +45,10 @@ export const config = {
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +21/-0 (21 lines); hunks: -261,4 +261,25 @@ export const benchmarks = [
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +34/-4 (38 lines); hunks: -28,15 +28,15 @@ Then run the **Python** output of the command panel below in...; -47,6 +47,28 @@ docker run --gpus all \
  - `docs_new/src/snippets/_deployment.jsx` modified +20/-2 (22 lines); hunks: -504,9 +504,27 @@ export const Deployment = ({ config, benchmarks }) => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -7,6 +7,8 @@ export const config = {
+    // AMD ROCm — MI300X (Flash FP8) + MI355X (Flash/Pro, FP4/FP8).
+    "mi300x", "mi355x",
@@ -43,6 +45,10 @@ export const config = {
+    // AMD FP8 uses the sgl-project repackaging.
+    "mi300x|flash|fp8": "sgl-project/DeepSeek-V4-Flash-FP8",
+    "mi355x|flash|fp8": "sgl-project/DeepSeek-V4-Flash-FP8",
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx
@@ -261,4 +261,25 @@ export const benchmarks = [
+  // ====================================================================
+  // MI300X + FP8 (Flash)
+  { match: { hw: "mi300x", variant: "flash", quant: "fp8", strategy: "low-latency", nodes: "single" } },
+  { match: { hw: "mi300x", variant: "flash", quant: "fp8", strategy: "balanced", nodes: "single" } },
+  { match: { hw: "mi300x", variant: "flash", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
+  // MI355X + FP4 (Flash)
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -28,15 +28,15 @@ Then run the **Python** output of the command panel below in that environment.
```

- 已读文件:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +724/-0; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +21/-0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +34/-4; `docs_new/src/snippets/_deployment.jsx` modified +20/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/_deployment.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27277 - Deepseek v4: support mixed dtype compression states

- 链接: https://github.com/sgl-project/sglang/pull/27277
- 状态/时间: merged / 2026-06-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；关联提交 `8fd1694dd27f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+1411/-132，可读 patch 1954 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Deepseek v4: support mixed dtype compression states」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「Deepseek v4: support mixed dtype compression states」；主要实现面是 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-4 (10 lines); hunks: -448,7 +448,8 @@ def __init__(; -494,7 +495,8 @@ def __init__(; symbols: __init__, _init_paged_compress_states，涉及 `__init__, _init_paged_compress_states`。
- 代码 diff 细节:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-4 (10 lines); hunks: -448,7 +448,8 @@ def __init__(; -494,7 +495,8 @@ def __init__(; symbols: __init__, _init_paged_compress_states
- 关键代码摘录:

```diff
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -448,7 +448,8 @@ def __init__(
-        state_dtype: torch.dtype,
+        c4_state_dtype: torch.dtype,
+        c128_state_dtype: torch.dtype,
@@ -494,7 +495,8 @@ def __init__(
-        self.state_dtype = state_dtype
+        self.c4_state_dtype = c4_state_dtype
```

- 已读文件:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-4
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `test/registered/jit/test_deepseek_v4_compress_state_runtime_shapes.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28520 - [AMD] Fix deepseek-v4 mtp accept length issue

- 链接: https://github.com/sgl-project/sglang/pull/28520
- 状态/时间: merged / 2026-06-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`；关联提交 `f5b041622ba2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+249/-7，可读 patch 296 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix deepseek-v4 mtp accept length issue」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`；技术摘要: 覆盖「[AMD] Fix deepseek-v4 mtp accept length issue」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +20/-7 (27 lines); hunks: -1316,24 +1316,37 @@ def get_unified_swa_loc(self, forward_batch: ForwardBatc...; symbols: get_unified_swa_loc, store_cache，涉及 `get_unified_swa_loc, store_cache`；`test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` added +179/-0 (179 lines); hunks: -0,0 +1,179; symbols: TestDeepseekV4ProFp4MTP, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV4ProFp4MTP, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +20/-7 (27 lines); hunks: -1316,24 +1316,37 @@ def get_unified_swa_loc(self, forward_batch: ForwardBatc...; symbols: get_unified_swa_loc, store_cache
  - `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` added +179/-0 (179 lines); hunks: -0,0 +1,179; symbols: TestDeepseekV4ProFp4MTP, setUpClass, tearDownClass, test_a_gsm8k
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -1316,24 +1316,37 @@ def get_unified_swa_loc(self, forward_batch: ForwardBatch) -> torch.Tensor:
+        Cached swa_loc is computed once from committed positions, so every draft-decode
+        step would reuse the same ring slot and break the chain. Recompute from the live
+        per-step positions; only the draft path is affected, the rest keeps the fast path.
+        is_multistep_draft_decode = (
+            forward_batch.forward_mode.is_decode_or_idle()
+            and self.speculative_num_steps > 1
diff -- test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py
@@ -0,0 +1,179 @@
+"""MI35x DeepSeek-V4-Pro FP4 + MTP Test (8-GPU)
+- Accuracy: GSM8K few-shot eval
+- Acceptance: mtp acc length eval
+Registry: nightly-amd-8-gpu-mi35x-deepseek-v4-pro-mtp suite
+"""
+import os
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +20/-7
  - tests: `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` added +179/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28613 - docs: add DeepSeek-V4 compressed state dtype tip

- 链接: https://github.com/sgl-project/sglang/pull/28613
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+13/-0，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: add DeepSeek-V4 compressed state dtype tip」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「docs: add DeepSeek-V4 compressed state dtype tip」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -180,6 +180,19 @@ The generator currently picks values on the **conservative*...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -180,6 +180,19 @@ The generator currently picks values on the **conservative*...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -180,6 +180,19 @@ The generator currently picks values on the **conservative** side (mirroring an
+**Compressed attention state dtype**
+DeepSeek-V4 uses hybrid compressed attention for long-context efficiency. `SGLANG_DSV4_COMPRESS_STATE_DTYPE` controls the dtype of the C4 / C128 compressed attention state pools.
+'''bash Command
+SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16 \
+sglang serve \
+  --model-path deepseek-ai/DeepSeek-V4-Flash \
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28590 - [Docs] DeepSeek-V4 cookbook: drop --disable-flashinfer-autotune from GB300 Flash low-latency

- 链接: https://github.com/sgl-project/sglang/pull/28590
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+7/-1，可读 patch 22 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] DeepSeek-V4 cookbook: drop --disable-flashinfer-autotune from GB300 Flash low-latency」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；技术摘要: 覆盖「[Docs] DeepSeek-V4 cookbook: drop --disable-flashinfer-autotune from GB300 Flash low-latency」；主要实现面是 `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +7/-0 (7 lines); hunks: -140,6 +140,13 @@ export const benchmarks = [；`docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +0/-1 (1 lines); hunks: -743,7 +743,6 @@ sgl-eval run aime25 \\。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +7/-0 (7 lines); hunks: -140,6 +140,13 @@ export const benchmarks = [
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +0/-1 (1 lines); hunks: -743,7 +743,6 @@ sgl-eval run aime25 \\
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx
@@ -140,6 +140,13 @@ export const benchmarks = [
+    sglang_version: "0.5.13.post1",
+    speed: [
+      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
+        ttft_ms: 463, tpot_ms: 4.19, tokens_per_sec_per_gpu: 35 },
+      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
+        ttft_ms: 436, tpot_ms: 8.93, tokens_per_sec_per_gpu: 336 },
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -743,7 +743,6 @@ sgl-eval run aime25 \\
-        "--disable-flashinfer-autotune",
```

- 已读文件:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +7/-0; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +0/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25144 - [NPU] Add Ascend NPU support for DeepSeek-V4

- 链接: https://github.com/sgl-project/sglang/pull/25144
- 状态/时间: merged / 2026-06-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `9b10821c8e6e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 28 个文件，+4145/-144，可读 patch 4984 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Add Ascend NPU support for DeepSeek-V4」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`；技术摘要: 覆盖「[NPU] Add Ascend NPU support for DeepSeek-V4」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +103/-24 (127 lines); hunks: -29,6 +29,7; -47,10 +48,15; symbols: __init__, _forward_prepare，涉及 `__init__, _forward_prepare`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +129/-61 (190 lines); hunks: -568,48 +568,46 @@ def __init__(; -741,6 +739,99 @@ def get_state_buf_infos(self) -> Tuple[List[int], List[int]...; symbols: __init__, get_state_buf_infos, _make_kv_pool, _make_indexer_pool，涉及 `__init__, get_state_buf_infos, _make_kv_pool`；`python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +55/-9 (64 lines); hunks: -2,15 +2,21; -109,24 +115,46 @@ def __init__(; symbols: _lcm, __init__, _alloc_kv_score_buffer, state_cache_3d，涉及 `_lcm, __init__, _alloc_kv_score_buffer`；`python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +21/-10 (31 lines); hunks: -15,25 +15,36 @@ def apply_deepseek_v4_defaults(server_args: ServerArgs, mode...; symbols: apply_deepseek_v4_defaults，涉及 `apply_deepseek_v4_defaults`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +103/-24 (127 lines); hunks: -29,6 +29,7; -47,10 +48,15; symbols: __init__, _forward_prepare
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +129/-61 (190 lines); hunks: -568,48 +568,46 @@ def __init__(; -741,6 +739,99 @@ def get_state_buf_infos(self) -> Tuple[List[int], List[int]...; symbols: __init__, get_state_buf_infos, _make_kv_pool, _make_indexer_pool
  - `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +55/-9 (64 lines); hunks: -2,15 +2,21; -109,24 +115,46 @@ def __init__(; symbols: _lcm, __init__, _alloc_kv_score_buffer, state_cache_3d
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +21/-10 (31 lines); hunks: -15,25 +15,36 @@ def apply_deepseek_v4_defaults(server_args: ServerArgs, mode...; symbols: apply_deepseek_v4_defaults
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -29,6 +29,7 @@
+    get_tensor_model_parallel_world_size,
@@ -47,10 +48,15 @@
+from sglang.srt.layers.deepseek_v4_rope import (
+    v4_rope_inplace_npu,
+)
+    attn_tp_all_reduce,
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -568,48 +568,46 @@ def __init__(
-            self.swa_kv_pool = DeepSeekV4SingleKVPool(
-                swa_size,
-                swa_page_size,
-                dtype,
-                qk_nope_head_dim,
-                qk_rope_head_dim,
diff -- python/sglang/srt/mem_cache/deepseek_v4_compress_state.py
@@ -2,15 +2,21 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +103/-24; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +129/-61; `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +55/-9; `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +21/-10
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26766 - [DeepSeek-V4] Fuse UE8M0 scale rounding into FP8 group quantization

- 链接: https://github.com/sgl-project/sglang/pull/26766
- 状态/时间: merged / 2026-06-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`；关联提交 `bea282cede6c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+70/-23，可读 patch 181 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek-V4] Fuse UE8M0 scale rounding into FP8 group quantization」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`；技术摘要: 覆盖「[DeepSeek-V4] Fuse UE8M0 scale rounding into FP8 group quantization」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +1/-1 (2 lines); hunks: -1066,8 +1066,8 @@ def forward(; symbols: forward，涉及 `forward`；`test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +1/-1 (2 lines); hunks: -41,7 +41,7 @@ class TestDSV4FlashFP4B200(; symbols: TestDSV4FlashFP4B200，涉及 `TestDSV4FlashFP4B200`；`test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +1/-1 (2 lines); hunks: -95,7 +95,7 @@ class TestDSV4FlashFP4B200W4A4MegaMoE(; symbols: TestDSV4FlashFP4B200W4A4MegaMoE，涉及 `TestDSV4FlashFP4B200W4A4MegaMoE`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +1/-1 (2 lines); hunks: -1066,8 +1066,8 @@ def forward(; symbols: forward
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +1/-1 (2 lines); hunks: -41,7 +41,7 @@ class TestDSV4FlashFP4B200(; symbols: TestDSV4FlashFP4B200
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +1/-1 (2 lines); hunks: -95,7 +95,7 @@ class TestDSV4FlashFP4B200W4A4MegaMoE(; symbols: TestDSV4FlashFP4B200W4A4MegaMoE
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -1066,8 +1066,8 @@ def forward(
+                scale_ue8m0=True,
-            o_s = deep_gemm.ceil_to_ue8m0(o_s)
diff -- test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py
@@ -41,7 +41,7 @@ class TestDSV4FlashFP4B200(
-    accept_length_thres = 2.6
+    accept_length_thres = 2.8
diff -- test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py
@@ -95,7 +95,7 @@ class TestDSV4FlashFP4B200W4A4MegaMoE(
-    accept_length_thres = 2.6
+    accept_length_thres = 2.8
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +1/-1
  - tests: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +1/-1; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/jit/test_per_token_group_quant_8bit_v2.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25820 - [NVIDIA] Support NVFP4 MoE for DeepSeek-V4

- 链接: https://github.com/sgl-project/sglang/pull/25820
- 状态/时间: merged / 2026-06-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `c0bb04b67f26`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+385/-17，可读 patch 572 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NVIDIA] Support NVFP4 MoE for DeepSeek-V4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`；技术摘要: 覆盖「[NVIDIA] Support NVFP4 MoE for DeepSeek-V4」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +5/-1 (6 lines); hunks: -2293,7 +2293,11 @@ def auto_weight_loader(module):; symbols: auto_weight_loader，涉及 `auto_weight_loader`；`python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +11/-0 (11 lines); hunks: -46,6 +46,17 @@ def apply_deepseek_v4_defaults(server_args: "ServerArgs", mod...; symbols: apply_deepseek_v4_defaults, validate_deepseek_v4_cp，涉及 `apply_deepseek_v4_defaults, validate_deepseek_v4_cp`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +5/-1 (6 lines); hunks: -2293,7 +2293,11 @@ def auto_weight_loader(module):; symbols: auto_weight_loader
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +11/-0 (11 lines); hunks: -46,6 +46,17 @@ def apply_deepseek_v4_defaults(server_args: "ServerArgs", mod...; symbols: apply_deepseek_v4_defaults, validate_deepseek_v4_cp
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -2293,7 +2293,11 @@ def auto_weight_loader(module):
-        skipped_checking_patterns = ["attn_mqa.k_scale", "attn_mqa.v_scale"]
+        skipped_checking_patterns = [
+            "attn_mqa.k_scale",
+            "attn_mqa.v_scale",
+            "blockscale_swizzled",
+        ]
diff -- python/sglang/srt/arg_groups/deepseek_v4_hook.py
@@ -46,6 +46,17 @@ def apply_deepseek_v4_defaults(server_args: "ServerArgs", model_arch: str) -> No
+    # nvidia/DeepSeek-V4-Pro-NVFP4 uses flashinfer_trtllm_routed MoE runner backend.
+    if (
+        server_args.moe_runner_backend == "auto"
+        and server_args.get_model_config().nvfp4_moe_meta is not None
+    ):
+        server_args.moe_runner_backend = "flashinfer_trtllm_routed"
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +5/-1; `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +11/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/moe/hash_topk.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28920 - [AMD] deepseek-v4 clean env vars

- 链接: https://github.com/sgl-project/sglang/pull/28920
- 状态/时间: merged / 2026-06-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` 等 7 个文件；关联提交 `04d952ea102d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+29/-108，可读 patch 245 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] deepseek-v4 clean env vars」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`, `test/registered/amd/test_deepseek_v4_flash_fp4.py`；技术摘要: 覆盖「[AMD] deepseek-v4 clean env vars」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`, `test/registered/amd/test_deepseek_v4_flash_fp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +1/-3 (4 lines); hunks: -1472,13 +1472,11 @@ def forward(; symbols: forward，涉及 `forward`；`test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` modified +2/-17 (19 lines); hunks: -38,28 +38,13；`test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +2/-16 (18 lines); hunks: -36,26 +36,12；`test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +2/-16 (18 lines); hunks: -36,26 +36,12。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +1/-3 (4 lines); hunks: -1472,13 +1472,11 @@ def forward(; symbols: forward
  - `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` modified +2/-17 (19 lines); hunks: -38,28 +38,13
  - `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +2/-16 (18 lines); hunks: -36,26 +36,12
  - `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +2/-16 (18 lines); hunks: -36,26 +36,12
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +2/-16 (18 lines); hunks: -38,26 +38,12
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -1472,13 +1472,11 @@ def forward(
-            import os
-            backend = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "kernel")
+            backend = envs.SGLANG_HACK_FLASHMLA_BACKEND.get()
diff -- test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py
@@ -38,28 +38,13 @@
-# Common DeepSeek-V4 env vars (AMD ROCm 7.2 path: AITER indexer + triton attn + ROCm700A).
-    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "false",
-    "SGLANG_USE_AITER": "1",
-    "SGLANG_USE_ROCM700A": "1",
-    "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
-    "SGLANG_OPT_USE_FUSED_COMPRESS_TRITON": "true",
diff -- test/registered/amd/test_deepseek_v4_flash_fp4.py
@@ -36,26 +36,12 @@
-# Common DeepSeek-V4 env vars (AMD ROCm 7.2 path: AITER indexer + triton attn + ROCm700A).
-    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "false",
-    "SGLANG_USE_AITER": "1",
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +1/-3
  - tests: `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` modified +2/-17; `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +2/-16; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +2/-16; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +2/-16; `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` modified +2/-16; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +2/-16
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28941 - [AMD] Fix nightly-8-gpu-mi35x-deepseek-v4-flash-rocm720 OOM issue

- 链接: https://github.com/sgl-project/sglang/pull/28941
- 状态/时间: merged / 2026-06-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；关联提交 `cee1caaf476f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+22/-22，可读 patch 52 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix nightly-8-gpu-mi35x-deepseek-v4-flash-rocm720 OOM issue」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「[AMD] Fix nightly-8-gpu-mi35x-deepseek-v4-flash-rocm720 OOM issue」；主要实现面是 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +22/-22 (44 lines); hunks: -578,29 +578,29 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +22/-22 (44 lines); hunks: -578,29 +578,29 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -578,29 +578,29 @@ def __init__(
-        c4_kv_pool_type = DeepSeekV4SingleKVPool
-        if enable_hisparse:
-            c4_kv_pool_type = HiSparseC4DevicePool
-        self.c4_kv_pool = self._make_kv_pool(
-            size=c4_size,
-            page_size=c4_page_size,
```

- 已读文件:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +22/-22
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28981 - [AMD] Update v4 cookbook to clean env vars

- 链接: https://github.com/sgl-project/sglang/pull/28981
- 状态/时间: merged / 2026-06-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+18/-262，可读 patch 454 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Update v4 cookbook to clean env vars」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[AMD] Update v4 cookbook to clean env vars」；主要实现面是 `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +12/-257 (269 lines); hunks: -158,8 +158,8 @@ sgl-eval run aime25 \\; -1400,26 +1400,9 @@ sgl-eval run aime25 \\；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-5 (11 lines); hunks: -49,13 +49,14 @@ docker run --gpus all \; -65,7 +66,7 @@ docker run \。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +12/-257 (269 lines); hunks: -158,8 +158,8 @@ sgl-eval run aime25 \\; -1400,26 +1400,9 @@ sgl-eval run aime25 \\
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-5 (11 lines); hunks: -49,13 +49,14 @@ docker run --gpus all \; -65,7 +66,7 @@ docker run \
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -158,8 +158,8 @@ sgl-eval run aime25 \\
-    mi300x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi30x-20260615",
-    mi355x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260615",
+    mi300x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi30x-20260623",
+    mi355x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260623",
@@ -1400,26 +1400,9 @@ sgl-eval run aime25 \\
-        "SGLANG_DEFAULT_THINKING=1",
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -49,13 +49,14 @@ docker run --gpus all \
-AMD uses the daily-updated `lmsysorg/sglang-rocm` images:
+AMD uses the daily-updated `lmsysorg/sglang-rocm` images. You can find the latest images on [Docker Hub](https://hub.docker.com/r/lmsysorg/sglang-rocm/tags). We recommend the ROCm
-- **MI355X** → `lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260615`
-- **MI300X** → `lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi30x-20260615`
+For example:
+- **MI355X** → `lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260623`
```

- 已读文件:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +12/-257; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-5
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28938 - [AMD] Improve performance of dsv4 in high concurrency

- 链接: https://github.com/sgl-project/sglang/pull/28938
- 状态/时间: merged / 2026-06-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `af9027f6c938`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+111/-44，可读 patch 347 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Improve performance of dsv4 in high concurrency」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[AMD] Improve performance of dsv4 in high concurrency」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +29/-0 (29 lines); hunks: -157,6 +157,10 @@ def _is_fused_mhc_post_pre_enabled() -> bool:; -1580,6 +1584,22 @@ def forward(; symbols: _is_fused_mhc_post_pre_enabled, forward，涉及 `_is_fused_mhc_post_pre_enabled, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +29/-0 (29 lines); hunks: -157,6 +157,10 @@ def _is_fused_mhc_post_pre_enabled() -> bool:; -1580,6 +1584,22 @@ def forward(; symbols: _is_fused_mhc_post_pre_enabled, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -157,6 +157,10 @@ def _is_fused_mhc_post_pre_enabled() -> bool:
+# PoC: compute the (replicated TP1) shared expert on LOCAL hidden before the dp
+# gather instead of on the gathered global buffer. Requires
+# SGLANG_SHARED_EXPERT_TP1=1 (replicated shared expert). Default OFF.
+_SHARED_EXPERT_LOCAL = get_bool_env_var("SGLANG_DP_SHARED_EXPERT_LOCAL")
@@ -1580,6 +1584,22 @@ def forward(
+        # PoC (SGLANG_DP_SHARED_EXPERT_LOCAL): compute the replicated shared expert
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +29/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/deepseek_v4_rope.py`, `python/sglang/srt/layers/dp_attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28455 - [AMD] Fix DeepSeek-V4 fp8 KV path on gfx942 (e4m3fnuz)

- 链接: https://github.com/sgl-project/sglang/pull/28455
- 状态/时间: merged / 2026-06-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `5e6d7c1615a9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+52/-16，可读 patch 205 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix DeepSeek-V4 fp8 KV path on gfx942 (e4m3fnuz)」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[AMD] Fix DeepSeek-V4 fp8 KV path on gfx942 (e4m3fnuz)」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +10/-1 (11 lines); hunks: -127,6 +127,7; -151,6 +152,7 @@ def _is_fused_mhc_post_pre_enabled() -> bool:; symbols: _is_fused_mhc_post_pre_enabled, forward，涉及 `_is_fused_mhc_post_pre_enabled, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +10/-1 (11 lines); hunks: -127,6 +127,7; -151,6 +152,7 @@ def _is_fused_mhc_post_pre_enabled() -> bool:; symbols: _is_fused_mhc_post_pre_enabled, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -127,6 +127,7 @@
+    is_gfx942_supported,
@@ -151,6 +152,7 @@ def _is_fused_mhc_post_pre_enabled() -> bool:
+_is_gfx942_supported = is_gfx942_supported()
@@ -911,7 +913,14 @@ def forward(
-            q_padded = x.new_empty(x.shape[0], padded_num_heads, self.head_dim)
+            # Only [0:n_local_heads] is written below. Uninitialized padded TP
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +10/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/store.cuh`, `python/sglang/jit_kernel/csrc/dsa/fused_store_index_cache.cuh`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28952 - Add DeepSeek V4 Flash demo notebook

- 链接: https://github.com/sgl-project/sglang/pull/28952
- 状态/时间: merged / 2026-06-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+1108/-0，可读 patch 1116 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add DeepSeek V4 Flash demo notebook」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「Add DeepSeek V4 Flash demo notebook」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -88,6 +88,10 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -88,6 +88,10 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -88,6 +88,10 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai/deepseek-v4-benchm
+<Note>
+  For a runnable end-to-end example, see the [DeepSeek-V4-Flash demo notebook](/demo/deepseek_v4_flash).
+</Note>
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #29261 - [Docs] Fix broken links in cookbook

- 链接: https://github.com/sgl-project/sglang/pull/29261
- 状态/时间: merged / 2026-06-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+3/-3，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Fix broken links in cookbook」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；技术摘要: 覆盖「[Docs] Fix broken links in cookbook」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -89,7 +89,7 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai/...；`docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +1/-1 (2 lines); hunks: -127,7 +127,7 @@ Pick a weight format by hardware: **NVFP4** on NVIDIA Blackw...；`docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -89,7 +89,7 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai/...
  - `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +1/-1 (2 lines); hunks: -127,7 +127,7 @@ Pick a weight format by hardware: **NVFP4** on NVIDIA Blackw...
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -89,7 +89,7 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai/deepseek-v4-benchm
-  For a runnable end-to-end example, see the [DeepSeek-V4-Flash demo notebook](/demo/deepseek_v4_flash).
+  For a runnable end-to-end example, see the [DeepSeek-V4-Flash demo notebook](https://github.com/sgl-project/sglang/blob/main/docs_new/demo/deepseek_v4_flash.ipynb).
diff -- docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx
@@ -127,7 +127,7 @@ Pick a weight format by hardware: **NVFP4** on NVIDIA Blackwell (B200, GB200), *
-For general GLM-4.x family launch guidance (AMD ROCm notes and more), see [Launch GLM-4.5 / GLM-4.6 / GLM-4.7 with SGLang](../../../docs/basic_usage/glm45). Per-hardware bench com
+For general GLM-4.x family launch guidance (AMD ROCm notes and more), see [Launch GLM-4.5 / GLM-4.6 / GLM-4.7 with SGLang](/cookbook/autoregressive/GLM/GLM-4.5). Per-hardware benc
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx
@@ -52,7 +52,7 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=pytho
-For the full Docker setup and other installation methods, refer to the [official SGLang installation guide](../../../docs/get-started/installation).
+For the full Docker setup and other installation methods, refer to the [official SGLang installation guide](../../../docs/get-started/install).
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28103 - Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test

- 链接: https://github.com/sgl-project/sglang/pull/28103
- 状态/时间: merged / 2026-06-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+218/-19，可读 patch 334 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml`；技术摘要: 覆盖「Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test」；主要实现面是 `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/gb300/test_deepseek_v4_pro_fp4.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: TestDeepSeekV4ProFp4, test_deepseek_v4_pro_fp4，涉及 `TestDeepSeekV4ProFp4, test_deepseek_v4_pro_fp4`；`test/registered/gb300/test_kimi_k25_nvfp4.py` modified +26/-10 (36 lines); hunks: -6,9 +6,12; -19,30 +22,43; symbols: TestKimiK25Nvfp4, test_kimi_k25_nvfp4，涉及 `TestKimiK25Nvfp4, test_kimi_k25_nvfp4`；`.github/workflows/nightly-test-nvidia.yml` modified +18/-3 (21 lines); hunks: -539,7 +539,20 @@ jobs:; -549,8 +562,10 @@ jobs:；`test/run_suite.py` modified +8/-1 (9 lines); hunks: -121,8 +121,15。
- 代码 diff 细节:
  - `test/registered/gb300/test_deepseek_v4_pro_fp4.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: TestDeepSeekV4ProFp4, test_deepseek_v4_pro_fp4
  - `test/registered/gb300/test_kimi_k25_nvfp4.py` modified +26/-10 (36 lines); hunks: -6,9 +6,12; -19,30 +22,43; symbols: TestKimiK25Nvfp4, test_kimi_k25_nvfp4
  - `.github/workflows/nightly-test-nvidia.yml` modified +18/-3 (21 lines); hunks: -539,7 +539,20 @@ jobs:; -549,8 +562,10 @@ jobs:
  - `test/run_suite.py` modified +8/-1 (9 lines); hunks: -121,8 +121,15
  - `test/registered/gb300/test_glm5_fp8.py` modified +4/-1 (5 lines); hunks: -7,7 +7,10
- 关键代码摘录:

```diff
diff -- test/registered/gb300/test_deepseek_v4_pro_fp4.py
@@ -0,0 +1,152 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
diff -- test/registered/gb300/test_kimi_k25_nvfp4.py
@@ -6,9 +6,12 @@
-register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300", nightly=True)
+register_cuda_ci(
+    est_time=7200, suite="nightly-4-gpu-gb300-kimi-k25-nvfp4", nightly=True
+)
+DRAFT_MODEL_PATH = "lightseekorg/kimi-k2.5-eagle3-mla"
@@ -19,30 +22,43 @@
diff -- .github/workflows/nightly-test-nvidia.yml
@@ -539,7 +539,20 @@ jobs:
```

- 已读文件:
  - tests: `test/registered/gb300/test_deepseek_v4_pro_fp4.py` added +152/-0; `test/registered/gb300/test_kimi_k25_nvfp4.py` modified +26/-10; `test/run_suite.py` modified +8/-1; `test/registered/gb300/test_glm5_fp8.py` modified +4/-1; `test/registered/gb300/test_kimi_k25.py` modified +4/-1; `test/registered/gb300/test_qwen35_nvfp4.py` modified +4/-1
  - ci: `.github/workflows/nightly-test-nvidia.yml` modified +18/-3
- 验证与风险: diff 自带测试面 `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_glm5_fp8.py`, `test/registered/gb300/test_glm5_nvfp4.py`, `test/registered/gb300/test_kimi_k25.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29103 - [AMD] Feat/dsv4 aiter reduce scatter decode

- 链接: https://github.com/sgl-project/sglang/pull/29103
- 状态/时间: merged / 2026-06-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `b7d3c3016d8c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+124/-9，可读 patch 232 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Feat/dsv4 aiter reduce scatter decode」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[AMD] Feat/dsv4 aiter reduce scatter decode」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +33/-4 (37 lines); hunks: -57,6 +57,7; -1578,12 +1579,28 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +33/-4 (37 lines); hunks: -57,6 +57,7; -1578,12 +1579,28 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -57,6 +57,7 @@
+    dp_reduce_scatter_tensor,
@@ -1578,12 +1579,28 @@ def forward(
-        _use_gatherv_pair = (
+        _use_reduce_scatterv = (
+        # SGLANG_DP_USE_REDUCE_SCATTER: in the MAX_LEN decode path (equal per-rank
+        # padding, gatherv inactive, no EP), replace the MoE-internal post-experts
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +33/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/environ.py`, `python/sglang/srt/layers/dp_attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27783 - [Intel GPU] DeepSeek V4 3/N: Support hc_split_sinkhorn on XPU using sgl_kernel

- 链接: https://github.com/sgl-project/sglang/pull/27783
- 状态/时间: merged / 2026-06-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `dc113e8804df`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-6，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Intel GPU] DeepSeek V4 3/N: Support hc_split_sinkhorn on XPU using sgl_kernel」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[Intel GPU] DeepSeek V4 3/N: Support hc_split_sinkhorn on XPU using sgl_kernel」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +11/-6 (17 lines); hunks: -68,7 +68,6; -113,9 +112,12; symbols: hc_pre_torch_impl，涉及 `hc_pre_torch_impl`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +11/-6 (17 lines); hunks: -68,7 +68,6; -113,9 +112,12; symbols: hc_pre_torch_impl
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -68,7 +68,6 @@
-from sglang.srt.layers.mhc import mhc_fused_post_pre, npu_hc_pre
@@ -113,9 +112,12 @@
-from sglang.srt.models.deepseek_v2 import ParallelLMHead, _is_cuda, _is_hip, _is_npu
-from sglang.srt.models.triton_ops.deepseek_v4 import (
-    rms_normalize_triton as rms_normalize_triton,
+from sglang.srt.models.deepseek_v2 import (
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +11/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29106 - Fix DeepSeek V4 PP HiCache SWA allocation and layer mapping

- 链接: https://github.com/sgl-project/sglang/pull/29106
- 状态/时间: merged / 2026-06-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `c1b5c7e49959`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+86/-46，可读 patch 208 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix DeepSeek V4 PP HiCache SWA allocation and layer mapping」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「Fix DeepSeek V4 PP HiCache SWA allocation and layer mapping」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +2/-2 (4 lines); hunks: -698,7 +698,7 @@ def _forward_prepare_multi_stream_hip(; -799,7 +799,7 @@ def _forward_prepare(; symbols: _forward_prepare_multi_stream_hip, _forward_prepare，涉及 `_forward_prepare_multi_stream_hip, _forward_prepare`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +5/-1 (6 lines); hunks: -530,6 +530,7 @@ def __init__(; -572,7 +573,7 @@ def __init__(; symbols: __init__, _swa_local_layer_id, get_swa_raw_buffer, get_swa_key_buffer，涉及 `__init__, _swa_local_layer_id, get_swa_raw_buffer`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +2/-2 (4 lines); hunks: -698,7 +698,7 @@ def _forward_prepare_multi_stream_hip(; -799,7 +799,7 @@ def _forward_prepare(; symbols: _forward_prepare_multi_stream_hip, _forward_prepare
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +5/-1 (6 lines); hunks: -530,6 +530,7 @@ def __init__(; -572,7 +573,7 @@ def __init__(; symbols: __init__, _swa_local_layer_id, get_swa_raw_buffer, get_swa_key_buffer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -698,7 +698,7 @@ def _forward_prepare_multi_stream_hip(
-            swa_cache = token_to_kv_pool.swa_kv_pool.kv_buffer[self.layer_id]
+            swa_cache = token_to_kv_pool.get_swa_raw_buffer(self.layer_id)
@@ -799,7 +799,7 @@ def _forward_prepare(
-                swa_cache = token_to_kv_pool.swa_kv_pool.kv_buffer[self.layer_id]
+                swa_cache = token_to_kv_pool.get_swa_raw_buffer(self.layer_id)
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -530,6 +530,7 @@ def __init__(
+        stage_layer_num = len(stage_ratios)
@@ -572,7 +573,7 @@ def __init__(
-                layer_num=layer_num,
+                layer_num=stage_layer_num,
@@ -925,6 +926,9 @@ def _swa_local_layer_id(self, layer_id: int) -> int:
+    def get_swa_raw_buffer(self, layer_id: int) -> torch.Tensor:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +2/-2; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +5/-1
- 验证与风险: diff 自带测试面 `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4.py`, `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4_pp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29502 - [CI] Fix GB300 DSV4 Pro FP4 nightly

- 链接: https://github.com/sgl-project/sglang/pull/29502
- 状态/时间: merged / 2026-06-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Fix GB300 DSV4 Pro FP4 nightly」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `test/registered/gb300/test_deepseek_v4_pro_fp4.py`；技术摘要: 覆盖「[CI] Fix GB300 DSV4 Pro FP4 nightly」；主要实现面是 `test/registered/gb300/test_deepseek_v4_pro_fp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/gb300/test_deepseek_v4_pro_fp4.py` modified +1/-1 (2 lines); hunks: -69,7 +69,7。
- 代码 diff 细节:
  - `test/registered/gb300/test_deepseek_v4_pro_fp4.py` modified +1/-1 (2 lines); hunks: -69,7 +69,7
- 关键代码摘录:

```diff
diff -- test/registered/gb300/test_deepseek_v4_pro_fp4.py
@@ -69,7 +69,7 @@
-    "0.85",
+    "0.9",
```

- 已读文件:
  - tests: `test/registered/gb300/test_deepseek_v4_pro_fp4.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/gb300/test_deepseek_v4_pro_fp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29420 - [AMD][DSV4] Remove per-batch D2H syncs in MTP to avoid bubbles between 2 batches

- 链接: https://github.com/sgl-project/sglang/pull/29420
- 状态/时间: merged / 2026-06-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`；关联提交 `54e71506b32f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-1，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][DSV4] Remove per-batch D2H syncs in MTP to avoid bubbles between 2 batches」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`；技术摘要: 覆盖「[AMD][DSV4] Remove per-batch D2H syncs in MTP to avoid bubbles between 2 batches」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +9/-1 (10 lines); hunks: -589,11 +589,13 @@ def init_forward_metadata_target_verify(; -876,6 +878,9 @@ def init_forward_metadata_out_graph(; symbols: init_forward_metadata_target_verify, init_forward_metadata_out_graph, init_forward_metadata，涉及 `init_forward_metadata_target_verify, init_forward_metadata_out_graph, init_forward_metadata`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +9/-1 (10 lines); hunks: -589,11 +589,13 @@ def init_forward_metadata_target_verify(; -876,6 +878,9 @@ def init_forward_metadata_out_graph(; symbols: init_forward_metadata_target_verify, init_forward_metadata_out_graph, init_forward_metadata
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -589,11 +589,13 @@ def init_forward_metadata_target_verify(
+        seq_lens_cpu: Optional[List[int]] = None,
-        seq_lens_cpu = seq_lens.tolist()
+        if seq_lens_cpu is None:
+            seq_lens_cpu = seq_lens.tolist()
@@ -876,6 +878,9 @@ def init_forward_metadata_out_graph(
+                # CPU mirror already available here (== seq_lens, no D2H);
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +9/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28980 - [NPU] Support DeepSeek V4 Flash MTP on Ascend

- 链接: https://github.com/sgl-project/sglang/pull/28980
- 状态/时间: merged / 2026-06-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`；关联提交 `89620b9169e6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+852/-86，可读 patch 1365 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Support DeepSeek V4 Flash MTP on Ascend」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4_nextn.py`, `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[NPU] Support DeepSeek V4 Flash MTP on Ascend」；主要实现面是 `python/sglang/srt/models/deepseek_v4_nextn.py`, `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4_nextn.py` modified +6/-3 (9 lines); hunks: -23,6 +23,7; -91,15 +92,17 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -2145,16 +2145,16 @@ def remap_weight_name_to_dpsk_hf_format(; symbols: remap_weight_name_to_dpsk_hf_format，涉及 `remap_weight_name_to_dpsk_hf_format`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +6/-3 (9 lines); hunks: -23,6 +23,7; -91,15 +92,17 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -2145,16 +2145,16 @@ def remap_weight_name_to_dpsk_hf_format(; symbols: remap_weight_name_to_dpsk_hf_format
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4_nextn.py
@@ -23,6 +23,7 @@
+from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
@@ -91,15 +92,17 @@ def __init__(
-        layer_name = "decoder"
+        if isinstance(quant_config, ModelSlimConfig):
+            prefix = "mtp.0"
+        else:
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -2145,16 +2145,16 @@ def remap_weight_name_to_dpsk_hf_format(
-        if "self_attn" in name:
-            name = name.replace(".scale", ".weight_scale_inv")
+        if "self_attn" in name and name.endswith(".scale"):
+            name = name.removesuffix(".scale") + ".weight_scale_inv"
-        if "mlp" in name:
-            name = name.replace(".scale", ".weight_scale_inv")
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4_nextn.py` modified +6/-3; `python/sglang/srt/models/deepseek_v4.py` modified +4/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py`, `python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29827 - [Doc] Tiny update dsv4 doc

- 链接: https://github.com/sgl-project/sglang/pull/29827
- 状态/时间: merged / 2026-07-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc] Tiny update dsv4 doc」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[Doc] Tiny update dsv4 doc」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +2/-2 (4 lines); hunks: -32,7 +32,7 @@ For how to launch the image, see [Install → Method 3: Using Do...; -296,7 +296,7 @@ TCP, which can lead to garbled KV transfer on large checkpoi...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +2/-2 (4 lines); hunks: -32,7 +32,7 @@ For how to launch the image, see [Install → Method 3: Using Do...; -296,7 +296,7 @@ TCP, which can lead to garbled KV transfer on large checkpoi...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -32,7 +32,7 @@ For how to launch the image, see [Install → Method 3: Using Docker](../../../d
-A single image — `lmsysorg/sglang:latest` — covers the **datacenter GPUs** in this cookbook (B200 / B300 / GB200 / GB300 / H100 / H200). For **RTX PRO 6000 (SM120)**, use the nigh
+A single image — `lmsysorg/sglang:latest` — covers the **datacenter GPUs** in this cookbook (B200 / B300 / GB200 / GB300 / H100 / H200 / RTX PRO 6000).
@@ -296,7 +296,7 @@ TCP, which can lead to garbled KV transfer on large checkpoints.
-HiCache and MegaMoE are **not** supported on RTX PRO 6000. For Docker, use the nightly `lmsysorg/sglang:dev` image — SM120 support isn't in `lmsysorg/sglang:latest` yet (the Deplo
+HiCache and MegaMoE are **not** supported on RTX PRO 6000.
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +2/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #29775 - [DeepSeek V4] Enable FlashMLA sparse prefill by default

- 链接: https://github.com/sgl-project/sglang/pull/29775
- 状态/时间: merged / 2026-07-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`；关联提交 `c865347b98ae`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+150/-35，可读 patch 389 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek V4] Enable FlashMLA sparse prefill by default」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`；技术摘要: 覆盖「[DeepSeek V4] Enable FlashMLA sparse prefill by default」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-6 (21 lines); hunks: -55,6 +55,7; -373,8 +374,8 @@ class DSV4Metadata:; symbols: DSV4Metadata, refresh_for_breakable_cuda_graph_replay_, __init__, _move_to_device，涉及 `DSV4Metadata, refresh_for_breakable_cuda_graph_replay_, __init__`；`test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +78/-0 (78 lines); hunks: -17,6 +17,7; -266,6 +267,27 @@ def test_runner_mode_production_eagle_draft_cuda_graph_runn...; symbols: test_runner_mode_production_eagle_draft_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract, _make_sparse_prefill_cache, _make_core_metadata，涉及 `test_runner_mode_production_eagle_draft_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract, _make_sparse_prefill_cache`；`python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +7/-0 (7 lines); hunks: -3,6 +3,8; -93,6 +95,11 @@ def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:; symbols: validate_deepseek_v4_cp，涉及 `validate_deepseek_v4_cp`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-6 (21 lines); hunks: -55,6 +55,7; -373,8 +374,8 @@ class DSV4Metadata:; symbols: DSV4Metadata, refresh_for_breakable_cuda_graph_replay_, __init__, _move_to_device
  - `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +78/-0 (78 lines); hunks: -17,6 +17,7; -266,6 +267,27 @@ def test_runner_mode_production_eagle_draft_cuda_graph_runn...; symbols: test_runner_mode_production_eagle_draft_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract, _make_sparse_prefill_cache, _make_core_metadata
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +7/-0 (7 lines); hunks: -3,6 +3,8; -93,6 +95,11 @@ def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:; symbols: validate_deepseek_v4_cp
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -55,6 +55,7 @@
+    SparsePrefillWorkspace,
@@ -373,8 +374,8 @@ class DSV4Metadata:
-    # reused across every layer in the chunk. Reset to ``None`` on copy_ so
-    # cuda-graph replay rebuilds it for the next forward.
+    # reused across every layer in the chunk. Reset to ``None`` when graph
+    # metadata is refreshed so replay rebuilds it from the live batch.
diff -- test/registered/attention/unittests/dsv4/test_deepseek_v4.py
@@ -17,6 +17,7 @@
+from unittest import mock
@@ -266,6 +267,27 @@ def test_runner_mode_production_eagle_draft_cuda_graph_runner_cases(self):
+    @staticmethod
+    def _make_sparse_prefill_cache(max_seq_len):
+        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
+            SparsePrefillChunkCache,
diff -- python/sglang/srt/arg_groups/deepseek_v4_hook.py
@@ -3,6 +3,8 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-6; `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +7/-0
  - tests: `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +78/-0
- 验证与风险: diff 自带测试面 `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29885 - [DeepSeek V4] Cover both dense and sparse prefill paths in the compress attention unittest

- 链接: https://github.com/sgl-project/sglang/pull/29885
- 状态/时间: merged / 2026-07-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`；关联提交 `307094dc7d0a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+105/-17，可读 patch 205 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek V4] Cover both dense and sparse prefill paths in the compress attention unittest」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`；技术摘要: 覆盖「[DeepSeek V4] Cover both dense and sparse prefill paths in the compress attention unittest」；主要实现面是 `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +15/-1 (16 lines); hunks: -183,13 +183,27 @@ def test_runner_mode_cuda_graph_decode_cases(self):; symbols: test_runner_mode_cuda_graph_decode_cases, test_compress_attention_cases, test_compress_attention_cases_sparse_prefill, test_eagle_target_verify_chain_cases，涉及 `test_runner_mode_cuda_graph_decode_cases, test_compress_attention_cases, test_compress_attention_cases_sparse_prefill`。
- 代码 diff 细节:
  - `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +15/-1 (16 lines); hunks: -183,13 +183,27 @@ def test_runner_mode_cuda_graph_decode_cases(self):; symbols: test_runner_mode_cuda_graph_decode_cases, test_compress_attention_cases, test_compress_attention_cases_sparse_prefill, test_eagle_target_verify_chain_cases
- 关键代码摘录:

```diff
diff -- test/registered/attention/unittests/dsv4/test_deepseek_v4.py
@@ -183,13 +183,27 @@ def test_runner_mode_cuda_graph_decode_cases(self):
+        # Pinned to the dense extend path; the sparse prefill path is covered
+        # by test_compress_attention_cases_sparse_prefill below.
-                run_dsv4_compress_attention_case(self, case)
+                run_dsv4_compress_attention_case(self, case, sparse_prefill=False)
+    def test_compress_attention_cases_sparse_prefill(self):
+        # `_forward_prefill_sparse` extend path; decode never reaches it.
```

- 已读文件:
  - tests: `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +15/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29982 - [AMD][DeepSeek V4] Fix default FlashMLA sparse prefill off on ROCm/HIP

- 链接: https://github.com/sgl-project/sglang/pull/29982
- 状态/时间: merged / 2026-07-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`；关联提交 `8519be82e8ed`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+13/-0，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][DeepSeek V4] Fix default FlashMLA sparse prefill off on ROCm/HIP」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/arg_groups/deepseek_v4_hook.py`；技术摘要: 覆盖「[AMD][DeepSeek V4] Fix default FlashMLA sparse prefill off on ROCm/HIP」；主要实现面是 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +13/-0 (13 lines); hunks: -14,6 +14,19; symbols: apply_deepseek_v4_defaults，涉及 `apply_deepseek_v4_defaults`。
- 代码 diff 细节:
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +13/-0 (13 lines); hunks: -14,6 +14,19; symbols: apply_deepseek_v4_defaults
- 关键代码摘录:

```diff
diff -- python/sglang/srt/arg_groups/deepseek_v4_hook.py
@@ -14,6 +14,19 @@
+    from sglang.srt.utils import is_hip
+    # FlashMLA sparse prefill (SGLANG_OPT_FLASHMLA_SPARSE_PREFILL, default on)
+    # currently returns incorrect output for DeepSeek-V4-Flash on ROCm/HIP
+    # (MI355X), which breaks the disaggregation nightly. Keep the previous
+    # (dense prefill) behavior on ROCm until the sparse kernel is validated
+    # there; an explicit env var still overrides this.
```

- 已读文件:
  - runtime: `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +13/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29619 - [DeepSeek-V4] Add an opt-in non-paged indexer for long-context prefill

- 链接: https://github.com/sgl-project/sglang/pull/29619
- 状态/时间: merged / 2026-07-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；关联提交 `a6ee64d237a2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+468/-50，可读 patch 681 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek-V4] Add an opt-in non-paged indexer for long-context prefill」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「[DeepSeek-V4] Add an opt-in non-paged indexer for long-context prefill」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +11/-2 (13 lines); hunks: -550,11 +550,17 @@ def _make_target_verify_c128_metadata(; -637,7 +643,10 @@ def init_forward_metadata_prefill(; symbols: _make_target_verify_c128_metadata, init_forward_metadata_indexer, init_forward_metadata_decode, init_forward_metadata_prefill，涉及 `_make_target_verify_c128_metadata, init_forward_metadata_indexer, init_forward_metadata_decode`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +17/-4 (21 lines); hunks: -317,12 +317,19 @@ def get_index_k_with_scale_buffer(self, layer_id: int) ->...; -978,14 +985,20 @@ def get_index_k_with_scale_buffer(self, layer_id: int) ->...; symbols: get_index_k_with_scale_buffer, get_index_k_scale_buffer, set_index_k_scale_buffer，涉及 `get_index_k_with_scale_buffer, get_index_k_scale_buffer, set_index_k_scale_buffer`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +11/-2 (13 lines); hunks: -550,11 +550,17 @@ def _make_target_verify_c128_metadata(; -637,7 +643,10 @@ def init_forward_metadata_prefill(; symbols: _make_target_verify_c128_metadata, init_forward_metadata_indexer, init_forward_metadata_decode, init_forward_metadata_prefill
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +17/-4 (21 lines); hunks: -317,12 +317,19 @@ def get_index_k_with_scale_buffer(self, layer_id: int) ->...; -978,14 +985,20 @@ def get_index_k_with_scale_buffer(self, layer_id: int) ->...; symbols: get_index_k_with_scale_buffer, get_index_k_scale_buffer, set_index_k_scale_buffer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -550,11 +550,17 @@ def _make_target_verify_c128_metadata(
-    def init_forward_metadata_indexer(self, core_attn_metadata: DSV4AttnMetadata):
+    def init_forward_metadata_indexer(
+        self,
+        core_attn_metadata: DSV4AttnMetadata,
+        *,
+        use_prefill_cuda_graph: bool = False,
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -317,12 +317,19 @@ def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
-        seq_len: int,
+        seq_len_tensor: torch.Tensor,
+        seq_len_sum: int,
+        max_seq_len: int,
-            self, buf, seq_len=seq_len, page_indices=page_indices
+            self,
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +11/-2; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +17/-4
- 验证与风险: diff 自带测试面 `test/registered/unit/layers/test_dsv4_nonpaged_indexer.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29988 - [dsv4] Trigger MHC prenorm prewarm at weight-load time with rank sync

- 链接: https://github.com/sgl-project/sglang/pull/29988
- 状态/时间: merged / 2026-07-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `e81f05cf4f44`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+68/-180，可读 patch 318 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[dsv4] Trigger MHC prenorm prewarm at weight-load time with rank sync」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[dsv4] Trigger MHC prenorm prewarm at weight-load time with rank sync」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +64/-115 (179 lines); hunks: -1182,121 +1182,6 @@ def refresh_mhc_norm_weight_cache(self):; -1966,6 +1851,11 @@ def __init__(; symbols: refresh_mhc_norm_weight_cache, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets, hc_pre，涉及 `refresh_mhc_norm_weight_cache, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +64/-115 (179 lines); hunks: -1182,121 +1182,6 @@ def refresh_mhc_norm_weight_cache(self):; -1966,6 +1851,11 @@ def __init__(; symbols: refresh_mhc_norm_weight_cache, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets, hc_pre
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -1182,121 +1182,6 @@ def refresh_mhc_norm_weight_cache(self):
-    def prewarm_mhc_token_counts(
-        self, token_counts: Tuple[int, ...], device: torch.device
-    ) -> None:
-        paths = (
-            (
-                "attn",
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +64/-115
- 验证与风险: diff 自带测试面 `test/registered/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27349 - Support DSV4 shared expert fusion for DeepEP and MegaMOE

- 链接: https://github.com/sgl-project/sglang/pull/27349
- 状态/时间: merged / 2026-07-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py`；关联提交 `d364cd8ead47`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+532/-87，可读 patch 966 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support DSV4 shared expert fusion for DeepEP and MegaMOE」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py`, `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「Support DSV4 shared expert fusion for DeepEP and MegaMOE」；主要实现面是 `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py`, `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py` added +50/-0 (50 lines); hunks: -0,0 +1,50; symbols: TestDeepseekV4SharedExpertFusionPolicy, _make_model, test_disables_shared_fusion_without_enforce, test_enables_shared_fusion_when_enforced，涉及 `TestDeepseekV4SharedExpertFusionPolicy, _make_model, test_disables_shared_fusion_without_enforce`；`python/sglang/srt/models/deepseek_v4.py` modified +10/-13 (23 lines); hunks: -1975,28 +1975,25 @@ def determine_num_fused_shared_experts(self):; symbols: determine_num_fused_shared_experts, forward，涉及 `determine_num_fused_shared_experts, forward`。
- 代码 diff 细节:
  - `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py` added +50/-0 (50 lines); hunks: -0,0 +1,50; symbols: TestDeepseekV4SharedExpertFusionPolicy, _make_model, test_disables_shared_fusion_without_enforce, test_enables_shared_fusion_when_enforced
  - `python/sglang/srt/models/deepseek_v4.py` modified +10/-13 (23 lines); hunks: -1975,28 +1975,25 @@ def determine_num_fused_shared_experts(self):; symbols: determine_num_fused_shared_experts, forward
- 关键代码摘录:

```diff
diff -- test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py
@@ -0,0 +1,50 @@
+import unittest
+from types import SimpleNamespace
+from unittest.mock import patch
+from sglang.srt.models import deepseek_v4 as deepseek_v4_module
+from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
+from sglang.test.ci.ci_register import register_cpu_ci
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -1975,28 +1975,25 @@ def determine_num_fused_shared_experts(self):
-        # Waterfill needs shared-experts fusion so it can dispatch shared
-        # expert tokens to least-loaded EP ranks.
-        if get_global_server_args().enable_deepep_waterfill:
+        disable_reason = None
+        if get_global_server_args().enforce_shared_experts_fusion:
-                    "DeepEP Waterfill for DeepSeek V4 expects exactly one shared "
```

- 已读文件:
  - tests: `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py` added +50/-0
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +10/-13
- 验证与风险: diff 自带测试面 `test/registered/moe/test_fused_append_remap_per_rank_shared_slots.py`, `test/registered/moe/test_hash_topk.py`, `test/registered/unit/eplb/test_deepep_waterfill_eplb.py`, `test/registered/unit/layers/moe/test_fused_shared_expert_scaling.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27914 - [Intel GPU] DeepSeek V4 6/N: use sgl-kernel implemetation of flash_mla_with_kvcache on XPU

- 链接: https://github.com/sgl-project/sglang/pull/27914
- 状态/时间: merged / 2026-07-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；关联提交 `4dddb0432553`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-4，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Intel GPU] DeepSeek V4 6/N: use sgl-kernel implemetation of flash_mla_with_kvcache on XPU」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；技术摘要: 覆盖「[Intel GPU] DeepSeek V4 6/N: use sgl-kernel implemetation of flash_mla_with_kvcache on XPU」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +8/-4 (12 lines); hunks: -50,7 +50,7; -60,6 +60,7; symbols: _pad_last_dim, _create_flashmla_metadata, match_num_queries，涉及 `_pad_last_dim, _create_flashmla_metadata, match_num_queries`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +8/-4 (12 lines); hunks: -50,7 +50,7; -60,6 +60,7; symbols: _pad_last_dim, _create_flashmla_metadata, match_num_queries
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -50,7 +50,7 @@
-from sglang.srt.utils import ceil_align
+from sglang.srt.utils import ceil_align, is_xpu
@@ -60,6 +60,7 @@
+_is_xpu = is_xpu()
@@ -111,7 +112,7 @@ def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
-    if _is_sm120:
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +8/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29362 - [AMD ]Feat/dsv4 ep tbo prefill

- 链接: https://github.com/sgl-project/sglang/pull/29362
- 状态/时间: merged / 2026-07-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py`；关联提交 `81735ecf8099`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+1008/-31，可读 patch 1213 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD ]Feat/dsv4 ep tbo prefill」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py`；技术摘要: 覆盖「[AMD ]Feat/dsv4 ep tbo prefill」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +375/-24 (399 lines); hunks: -53,15 +53,21; -1108,6 +1114,21 @@ def forward(; symbols: forward, op_attn, DeepseekV4DecoderLayer, __init__，涉及 `forward, op_attn, DeepseekV4DecoderLayer`；`python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +8/-0 (8 lines); hunks: -406,6 +406,14 @@ def of(cls, forward_mode: ForwardMode) -> _GraphBucket:; symbols: of, DeepseekV4HipRadixBackend, __init__，涉及 `of, DeepseekV4HipRadixBackend, __init__`；`test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: TestDeepseekV4FlashFp8Tbo, setUpClass, tearDownClass, test_gsm8k_tbo，涉及 `TestDeepseekV4FlashFp8Tbo, setUpClass, tearDownClass`；`test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: TestDeepseekV4ProFp4Tbo, setUpClass, tearDownClass, test_gsm8k_tbo，涉及 `TestDeepseekV4ProFp4Tbo, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +375/-24 (399 lines); hunks: -53,15 +53,21; -1108,6 +1114,21 @@ def forward(; symbols: forward, op_attn, DeepseekV4DecoderLayer, __init__
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +8/-0 (8 lines); hunks: -406,6 +406,14 @@ def of(cls, forward_mode: ForwardMode) -> _GraphBucket:; symbols: of, DeepseekV4HipRadixBackend, __init__
  - `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: TestDeepseekV4FlashFp8Tbo, setUpClass, tearDownClass, test_gsm8k_tbo
  - `test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: TestDeepseekV4ProFp4Tbo, setUpClass, tearDownClass, test_gsm8k_tbo
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -53,15 +53,21 @@
+    _tbo_event,
+    dp_reduce_scatterv_async,
+    get_dp_tbo_comm_stream,
+    get_global_dp_buffer_len,
+    get_local_dp_buffer_len,
+    get_tbo_persistent_buffer,
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -406,6 +406,14 @@ def of(cls, forward_mode: ForwardMode) -> _GraphBucket:
+    # DSV4 TBO runs ONLY in eager prefill (prefill cuda-graph is disabled);
+    # decode/target-verify graphs are non-TBO (primary backend only). So the TBO
+    # child backends must not be driven through cuda-graph capture/replay — doing
+    # so rebuilds this backend's compressor/indexer metadata per replay step on
+    # both children and leaks ROCm HSA resources (HSA_STATUS_ERROR_OUT_OF_RESOURCES).
+    # TboAttnBackend reads this to skip children in the *_graph paths only.
diff -- test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py
@@ -0,0 +1,163 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +375/-24; `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +8/-0
  - tests: `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py` added +163/-0; `test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py` added +151/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py`, `test/registered/unit/server_args/test_server_args.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30237 - [AMD][DeepSeek V4] Set SGLANG_OPT_FLASHMLA_SPARSE_PREFILL to false on hip code path

- 链接: https://github.com/sgl-project/sglang/pull/30237
- 状态/时间: merged / 2026-07-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`；关联提交 `80decc78ec22`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][DeepSeek V4] Set SGLANG_OPT_FLASHMLA_SPARSE_PREFILL to false on hip code path」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/arg_groups/deepseek_v4_hook.py`；技术摘要: 覆盖「[AMD][DeepSeek V4] Set SGLANG_OPT_FLASHMLA_SPARSE_PREFILL to false on hip code path」；主要实现面是 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +2/-2 (4 lines); hunks: -20,8 +20,8 @@ def apply_deepseek_v4_defaults(server_args: ServerArgs, model_...; symbols: apply_deepseek_v4_defaults，涉及 `apply_deepseek_v4_defaults`。
- 代码 diff 细节:
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +2/-2 (4 lines); hunks: -20,8 +20,8 @@ def apply_deepseek_v4_defaults(server_args: ServerArgs, model_...; symbols: apply_deepseek_v4_defaults
- 关键代码摘录:

```diff
diff -- python/sglang/srt/arg_groups/deepseek_v4_hook.py
@@ -20,8 +20,8 @@ def apply_deepseek_v4_defaults(server_args: ServerArgs, model_arch: str) -> None
-    # there; an explicit env var still overrides this.
-    if is_hip() and not envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.is_set():
+    # there;
+    if is_hip():
```

- 已读文件:
  - runtime: `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27867 - [DSv4] Loading Time Weight Dequant

- 链接: https://github.com/sgl-project/sglang/pull/27867
- 状态/时间: merged / 2026-07-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`；关联提交 `627980596254`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+148/-3，可读 patch 234 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSv4] Loading Time Weight Dequant」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/configs/model_config.py`；技术摘要: 覆盖「[DSv4] Loading Time Weight Dequant」；主要实现面是 `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/configs/model_config.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +45/-2 (47 lines); hunks: -4,7 +4,10; -21,7 +24,7; symbols: _flashinfer_has_sm90_cutlass_mxfp4, tearDownClass, TestDSV4FlashFP4DequantTP8H200, setUpClass，涉及 `_flashinfer_has_sm90_cutlass_mxfp4, tearDownClass, TestDSV4FlashFP4DequantTP8H200`；`python/sglang/srt/layers/quantization/fp8.py` modified +95/-0 (95 lines); hunks: -144,6 +144,73 @@ def _require_fp4_dtype():; -162,6 +229,7 @@ def __init__(; symbols: _require_fp4_dtype, cast_e2m1fn_to_e4m3fn, Fp8Config, for，涉及 `_require_fp4_dtype, cast_e2m1fn_to_e4m3fn, Fp8Config`；`python/sglang/srt/configs/model_config.py` modified +6/-1 (7 lines); hunks: -325,7 +325,10 @@ def __init__(; -335,6 +338,8 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -853,6 +853,7 @@ class Envs:; symbols: Envs，涉及 `Envs`。
- 代码 diff 细节:
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +45/-2 (47 lines); hunks: -4,7 +4,10; -21,7 +24,7; symbols: _flashinfer_has_sm90_cutlass_mxfp4, tearDownClass, TestDSV4FlashFP4DequantTP8H200, setUpClass
  - `python/sglang/srt/layers/quantization/fp8.py` modified +95/-0 (95 lines); hunks: -144,6 +144,73 @@ def _require_fp4_dtype():; -162,6 +229,7 @@ def __init__(; symbols: _require_fp4_dtype, cast_e2m1fn_to_e4m3fn, Fp8Config, for
  - `python/sglang/srt/configs/model_config.py` modified +6/-1 (7 lines); hunks: -325,7 +325,10 @@ def __init__(; -335,6 +338,8 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -853,6 +853,7 @@ class Envs:; symbols: Envs
  - `python/sglang/srt/model_loader/loader.py` modified +1/-0 (1 lines); hunks: -246,6 +246,7 @@ def _get_quantization_config(; symbols: _get_quantization_config
- 关键代码摘录:

```diff
diff -- test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py
@@ -4,7 +4,10 @@
-Registry: base-c-test-deepep-8-gpu-h200 (per-commit, 8x H200 — only 4 used by TP=4)
+Also covers SGLANG_DSV4_FP4_DEQUANT=1 (TP=8): FP4 experts dequantized to FP8
+during loading and served through the plain FP8 MoE path.
+Registry: base-c-test-deepep-8-gpu-h200 (per-commit, 8x H200)
@@ -21,7 +24,7 @@
-register_cuda_ci(est_time=370, stage="base-c", runner_config="deepep-8-gpu-h200")
diff -- python/sglang/srt/layers/quantization/fp8.py
@@ -144,6 +144,73 @@ def _require_fp4_dtype():
+DSV4_DEQUANT_FP4_TABLE = torch.tensor(
+    [
+        0.0,
+        0.5,
+        1.0,
+        1.5,
diff -- python/sglang/srt/configs/model_config.py
@@ -325,7 +325,10 @@ def __init__(
```

- 已读文件:
  - tests: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +45/-2
  - runtime: `python/sglang/srt/layers/quantization/fp8.py` modified +95/-0; `python/sglang/srt/configs/model_config.py` modified +6/-1; `python/sglang/srt/environ.py` modified +1/-0; `python/sglang/srt/model_loader/loader.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30333 - [AMD] Fix DeepSeek V4 MTP accuracy issue

- 链接: https://github.com/sgl-project/sglang/pull/30333
- 状态/时间: merged / 2026-07-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`；关联提交 `9a6f8e599204`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-1，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix DeepSeek V4 MTP accuracy issue」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`；技术摘要: 覆盖「[AMD] Fix DeepSeek V4 MTP accuracy issue」；主要实现面是 `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +10/-1 (11 lines); hunks: -129,7 +129,16 @@ def __init__(; symbols: __init__, _alloc_kv_score_buffer，涉及 `__init__, _alloc_kv_score_buffer`。
- 代码 diff 细节:
  - `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +10/-1 (11 lines); hunks: -129,7 +129,16 @@ def __init__(; symbols: __init__, _alloc_kv_score_buffer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/mem_cache/deepseek_v4_compress_state.py
@@ -129,7 +129,16 @@ def __init__(
-            self.kv_score_buffer[-1].clear()
+            if _is_hip and ratio == 128:
+                # Request-scoped C128 state is addressed by req_pool_idx (or a
+                # per-request ring).  The pool is allocated with torch.empty(),
+                # so a cold server can otherwise read uninitialized partial
+                # states before a request slot has been written for the first
```

- 已读文件:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +10/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27926 - [DSV4] perf: Make FP8 quant output tensor contiguous

- 链接: https://github.com/sgl-project/sglang/pull/27926
- 状态/时间: merged / 2026-07-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `d7dcdf3efd2c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+479/-7，可读 patch 527 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] perf: Make FP8 quant output tensor contiguous」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[DSV4] perf: Make FP8 quant output tensor contiguous」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +3/-7 (10 lines); hunks: -24,6 +24,7; -70,7 +71,6; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +3/-7 (10 lines); hunks: -24,6 +24,7; -70,7 +71,6; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -24,6 +24,7 @@
+    sglang_per_token_group_quant_fp8_dsv4_wo_a,
@@ -70,7 +71,6 @@
-from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
@@ -1084,15 +1084,11 @@ def forward(
-            o_fp8, o_s = sglang_per_token_group_quant_fp8(
-                o.reshape(T * G, D).contiguous(),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +3/-7
- 验证与风险: diff 自带测试面 `test/registered/jit/deepseek_v4/test_fp8_wo_a.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29417 - [AMD] Enable unified-KV HiCache on DeepSeek-V4

- 链接: https://github.com/sgl-project/sglang/pull/29417
- 状态/时间: merged / 2026-07-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；关联提交 `8d0fd3415077`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+208/-99，可读 patch 483 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Enable unified-KV HiCache on DeepSeek-V4」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「[AMD] Enable unified-KV HiCache on DeepSeek-V4」；主要实现面是 `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +51/-4 (55 lines); hunks: -390,7 +390,7 @@ class DeepSeekV4LayerItem(NamedTuple):; -403,6 +403,7 @@ def __init__(; symbols: DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool, __init__，涉及 `DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +51/-4 (55 lines); hunks: -390,7 +390,7 @@ class DeepSeekV4LayerItem(NamedTuple):; -403,6 +403,7 @@ def __init__(; symbols: DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -390,7 +390,7 @@ class DeepSeekV4LayerItem(NamedTuple):
-    unified_kv[L]: ``[swa_pages + compress_pages, head_dim]`` bf16
+    unified_kv[L]: ``[swa_pages + padded_compress_rows, head_dim]`` bf16
@@ -403,6 +403,7 @@ def __init__(
+        page_size: int,
@@ -415,6 +416,7 @@ def __init__(
+        self.page_size = page_size
```

- 已读文件:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +51/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/managers/schedule_policy.py`, `python/sglang/srt/mem_cache/base_prefix_cache.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30695 - [Refactor] Make DeepSeek-V4 attention backend tolerate an absent CPU seq_lens mirror

- 链接: https://github.com/sgl-project/sglang/pull/30695
- 状态/时间: merged / 2026-07-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；关联提交 `504570f4250d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+26/-17，可读 patch 109 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Make DeepSeek-V4 attention backend tolerate an absent CPU seq_lens mirror」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；技术摘要: 覆盖「[Refactor] Make DeepSeek-V4 attention backend tolerate an absent CPU seq_lens mirror」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +26/-17 (43 lines); hunks: -457,6 +457,8 @@ class DeepseekV4AttnBackend(; -509,6 +511,11 @@ def __init__(; symbols: DeepseekV4AttnBackend, __init__, _move_to_device, init_forward_metadata_target_verify，涉及 `DeepseekV4AttnBackend, __init__, _move_to_device`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +26/-17 (43 lines); hunks: -457,6 +457,8 @@ class DeepseekV4AttnBackend(; -509,6 +511,11 @@ def __init__(; symbols: DeepseekV4AttnBackend, __init__, _move_to_device, init_forward_metadata_target_verify
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -457,6 +457,8 @@ class DeepseekV4AttnBackend(
+    needs_cpu_seq_lens: bool = False
@@ -509,6 +511,11 @@ def __init__(
+        # Draft-extend and online-c128 verify metadata are host-planned, so
+        # spec runs keep the relay publish (the mirror only exists under
+        # spec-v2; without spec the flag has no consumer either way).
+        if model_runner.server_args.speculative_algorithm is not None:
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +26/-17
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30711 - [Refactor] Split DeepSeek-V4 MQALayer into a reusable attention base

- 链接: https://github.com/sgl-project/sglang/pull/30711
- 状态/时间: merged / 2026-07-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `fef5eda4fb2c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+281/-142，可读 patch 564 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Split DeepSeek-V4 MQALayer into a reusable attention base」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[Refactor] Split DeepSeek-V4 MQALayer into a reusable attention base」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +281/-142 (423 lines); hunks: -8,7 +8,6; -158,6 +157,11; symbols: _is_fused_mhc_post_pre_enabled, _fused_rmsnorm_fp8_quant, make_hc_mixing_params, make_hc_head_params，涉及 `_is_fused_mhc_post_pre_enabled, _fused_rmsnorm_fp8_quant, make_hc_mixing_params`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +281/-142 (423 lines); hunks: -8,7 +8,6; -158,6 +157,11; symbols: _is_fused_mhc_post_pre_enabled, _fused_rmsnorm_fp8_quant, make_hc_mixing_params, make_hc_head_params
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -8,7 +8,6 @@
-    Literal,
@@ -158,6 +157,11 @@
+DEEPSEEK_V4_STACKED_PARAMS_MAPPING: List[Tuple[str, str, int]] = [
+    ("gate_up_proj", "gate_proj", 0),
+    ("gate_up_proj", "up_proj", 1),
+]
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +281/-142
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30898 - Enable breakable prefill CUDA graph for DP attention

- 链接: https://github.com/sgl-project/sglang/pull/30898
- 状态/时间: merged / 2026-07-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`；关联提交 `771e38633216`, `b94ac87e0c41`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+478/-20，可读 patch 731 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable breakable prefill CUDA graph for DP attention」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/forward_batch_info.py`；技术摘要: 覆盖「Enable breakable prefill CUDA graph for DP attention」；主要实现面是 `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/forward_batch_info.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +2/-0 (2 lines); hunks: -75,6 +75,8 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`；`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` modified +70/-8 (78 lines); hunks: -61,6 +61,8; -98,6 +100,7; symbols: __init__, _next_token_logits_buffer, _prefill_logits_buffer_rows, _capture_num_token_non_padded，涉及 `__init__, _next_token_logits_buffer, _prefill_logits_buffer_rows`；`python/sglang/srt/model_executor/forward_batch_info.py` modified +39/-1 (40 lines); hunks: -1177,6 +1177,26 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; -1233,7 +1253,13 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; symbols: prepare_mlp_sync_batch，涉及 `prepare_mlp_sync_batch`；`python/sglang/srt/model_executor/cuda_graph_buffer_registry.py` modified +10/-0 (10 lines); hunks: -788,6 +788,7 @@ def build_prefill_registry(; -876,6 +877,15 @@ def _bs(bs: int, _mt: int) -> Tuple[int, ...]:; symbols: build_prefill_registry, _bs，涉及 `build_prefill_registry, _bs`。
- 代码 diff 细节:
  - `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +2/-0 (2 lines); hunks: -75,6 +75,8 @@ def setUpClass(cls):; symbols: setUpClass
  - `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` modified +70/-8 (78 lines); hunks: -61,6 +61,8; -98,6 +100,7; symbols: __init__, _next_token_logits_buffer, _prefill_logits_buffer_rows, _capture_num_token_non_padded
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +39/-1 (40 lines); hunks: -1177,6 +1177,26 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; -1233,7 +1253,13 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; symbols: prepare_mlp_sync_batch
  - `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py` modified +10/-0 (10 lines); hunks: -788,6 +788,7 @@ def build_prefill_registry(; -876,6 +877,15 @@ def _bs(bs: int, _mt: int) -> Tuple[int, ...]:; symbols: build_prefill_registry, _bs
  - `python/sglang/srt/model_executor/runner_utils/buffers.py` modified +3/-1 (4 lines); hunks: -62,7 +62,6 @@ def foreach_copy(dsts: List[torch.Tensor], srcs: List[torch.Te...; -328,6 +327,7 @@ def populate_from_forward_batch(; symbols: foreach_copy, DecodeInputBuffers, populate_from_forward_batch, PrefillInputBuffers
- 关键代码摘录:

```diff
diff -- test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py
@@ -75,6 +75,8 @@ def setUpClass(cls):
+                "--mem-fraction-static",
+                "0.80",
diff -- python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py
@@ -61,6 +61,8 @@
+    compute_local_num_token_non_padded,
+    enable_num_token_non_padded,
@@ -98,6 +100,7 @@
+    require_gathered_buffer,
@@ -229,6 +232,7 @@ def __init__(self, model_runner: ModelRunner):
+            enable_num_token_non_padded=enable_num_token_non_padded(),
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -1177,6 +1177,26 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):
+        # Prefill breakable CUDA graph requires every DP rank to run the SAME
+        # captured shape. Under SUM_LEN each rank pads to its own local token
+        # count and can select a different capture bucket, so the in-graph DP
+        # collectives (all_gather / reduce_scatter) mismatch across ranks and
```

- 已读文件:
  - tests: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +2/-0
  - runtime: `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` modified +70/-8; `python/sglang/srt/model_executor/forward_batch_info.py` modified +39/-1; `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py` modified +10/-0; `python/sglang/srt/model_executor/runner_utils/buffers.py` modified +3/-1; `python/sglang/srt/server_args.py` modified +33/-5; `python/sglang/srt/managers/scheduler_components/dp_attn.py` modified +13/-5
- 验证与风险: diff 自带测试面 `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/dp_attn/test_dp_attention_bcg_kl.py`, `test/registered/unit/model_executor/test_cuda_graph_buffer_registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31125 - Disable flaky DSV4-Flash FP4 BCG determinism test (nondeterminism from #30898 idle-rank dummy extend)

- 链接: https://github.com/sgl-project/sglang/pull/31125
- 状态/时间: merged / 2026-07-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`；关联提交 `771e38633216`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-0，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Disable flaky DSV4-Flash FP4 BCG determinism test (nondeterminism from #30898 idle-rank dummy extend)」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`；技术摘要: 覆盖「Disable flaky DSV4-Flash FP4 BCG determinism test (nondeterminism from #30898 idle-rank dummy extend)」；主要实现面是 `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +11/-0 (11 lines); hunks: -170,6 +170,17 @@ class TestDSV4FlashFP4BreakableCudaGraphB200(; symbols: TestDSV4FlashFP4BreakableCudaGraphB200, test_determinism_temp_zero, setUpClass，涉及 `TestDSV4FlashFP4BreakableCudaGraphB200, test_determinism_temp_zero, setUpClass`。
- 代码 diff 细节:
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +11/-0 (11 lines); hunks: -170,6 +170,17 @@ class TestDSV4FlashFP4BreakableCudaGraphB200(; symbols: TestDSV4FlashFP4BreakableCudaGraphB200, test_determinism_temp_zero, setUpClass
- 关键代码摘录:

```diff
diff -- test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py
@@ -170,6 +170,17 @@ class TestDSV4FlashFP4BreakableCudaGraphB200(
+    @unittest.skip(
+        "Flaky: temp-0 outputs are nondeterministic under this recipe "
+        "(sparse-DP prefill replays the breakable CUDA graph with a "
+        "fabricated idle-rank dummy extend; its hidden states vary run to "
+        "run and perturb real tokens' logits through the shared EP grouped "
+        "GEMMs at capture buckets 4/16). Introduced by #30898; disabled "
```

- 已读文件:
  - tests: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +11/-0
- 验证与风险: diff 自带测试面 `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30365 - [DSV4] Remove per-step seqlen D2H from speculative to make overlap scheduler work

- 链接: https://github.com/sgl-project/sglang/pull/30365
- 状态/时间: merged / 2026-07-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`；关联提交 `a9cf5e68e688`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+115/-52，可读 patch 403 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Remove per-step seqlen D2H from speculative to make overlap scheduler work」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`；技术摘要: 覆盖「[DSV4] Remove per-step seqlen D2H from speculative to make overlap scheduler work」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +64/-48 (112 lines); hunks: -68,7 +68,7; -79,6 +79,7; symbols: __init__, _make_target_verify_c128_metadata, init_forward_metadata_target_verify，涉及 `__init__, _make_target_verify_c128_metadata, init_forward_metadata_target_verify`；`test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +29/-3 (32 lines); hunks: -26,7 +26,10; -35,6 +38,7; symbols: test_runner_mode_eagle_verify_cuda_graph_cases, test_eagle_draft_extend_without_cpu_seq_lens, test_runner_mode_production_eagle_draft_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract，涉及 `test_runner_mode_eagle_verify_cuda_graph_cases, test_eagle_draft_extend_without_cpu_seq_lens, test_runner_mode_production_eagle_draft_cuda_graph_runner_cases`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +64/-48 (112 lines); hunks: -68,7 +68,7; -79,6 +79,7; symbols: __init__, _make_target_verify_c128_metadata, init_forward_metadata_target_verify
  - `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +29/-3 (32 lines); hunks: -26,7 +26,10; -35,6 +38,7; symbols: test_runner_mode_eagle_verify_cuda_graph_cases, test_eagle_draft_extend_without_cpu_seq_lens, test_runner_mode_production_eagle_draft_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -68,7 +68,7 @@
-from sglang.srt.utils import ceil_align, is_xpu
+from sglang.srt.utils import ceil_align, is_cuda, is_xpu
@@ -79,6 +79,7 @@
+_is_cuda = is_cuda()
@@ -497,6 +498,7 @@ def __init__(
+        self.max_context_len = model_runner.model_config.context_len
diff -- test/registered/attention/unittests/dsv4/test_deepseek_v4.py
@@ -26,7 +26,10 @@
-_FLASH_MLA_AVAILABLE = importlib.util.find_spec("flash_mla") is not None
+_FLASH_MLA_AVAILABLE = (
+    importlib.util.find_spec("sgl_kernel") is not None
+    and importlib.util.find_spec("sgl_kernel.flash_mla") is not None
+)
@@ -35,6 +38,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +64/-48
  - tests: `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +29/-3
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/runner_modes/speculative_draft_runner.py`, `python/sglang/test/kits/attention_unittest/runner_modes/speculative_target_verify_runner.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30792 - [Kernel] Migrate DSA + DSV4 attention kernels to sglang.kernels (RFC #29630, Phase 2.5, 5/7)

- 链接: https://github.com/sgl-project/sglang/pull/30792
- 状态/时间: merged / 2026-07-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `ba5be86d42af`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 60 个文件，+662/-582，可读 patch 1730 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Migrate DSA + DSV4 attention kernels to sglang.kernels (RFC #29630, Phase 2.5, 5/7)」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[Kernel] Migrate DSA + DSV4 attention kernels to sglang.kernels (RFC #29630, Phase 2.5, 5/7)」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +11/-11 (22 lines); hunks: -18,6 +18,12; -31,12 +37,6; symbols: _attach_unified_kv_decode_streams, _attach_unified_kv_prefill_meta, _forward_unified_kv, forward，涉及 `_attach_unified_kv_decode_streams, _attach_unified_kv_prefill_meta, _forward_unified_kv`；`python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +9/-9 (18 lines); hunks: -19,6 +19,15; -31,22 +40,13；`python/sglang/srt/models/deepseek_v4.py` modified +2/-2 (4 lines); hunks: -876,7 +876,7 @@ def _forward_prepare(; -1130,7 +1130,7 @@ def forward(; symbols: _forward_prepare, forward，涉及 `_forward_prepare, forward`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +7/-7 (14 lines); hunks: -11,13 +11,13; -368,7 +368,7 @@ def set_index_fp4(; symbols: set_index_fp4, __init__，涉及 `set_index_fp4, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +11/-11 (22 lines); hunks: -18,6 +18,12; -31,12 +37,6; symbols: _attach_unified_kv_decode_streams, _attach_unified_kv_prefill_meta, _forward_unified_kv, forward
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +9/-9 (18 lines); hunks: -19,6 +19,15; -31,22 +40,13
  - `python/sglang/srt/models/deepseek_v4.py` modified +2/-2 (4 lines); hunks: -876,7 +876,7 @@ def _forward_prepare(; -1130,7 +1130,7 @@ def forward(; symbols: _forward_prepare, forward
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +7/-7 (14 lines); hunks: -11,13 +11,13; -368,7 +368,7 @@ def set_index_fp4(; symbols: set_index_fp4, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -18,6 +18,12 @@
+from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
+    init_compression_metadata as _init_compression_metadata_triton,
+)
+from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
+    quant_to_nope_fp8_rope_bf16_pack_triton,
+)
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -19,6 +19,15 @@
+from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
+    dequantize_k_cache_paged,
+)
+from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
+    init_compression_metadata as _init_compression_metadata_triton,
+)
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -876,7 +876,7 @@ def _forward_prepare(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +11/-11; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +9/-9; `python/sglang/srt/models/deepseek_v4.py` modified +2/-2; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +7/-7
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `test/manual/layers/attention/dsa/test_act_quant_triton.py`, `test/manual/layers/attention/dsa/test_get_k_scale_triton_kernel.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30651 - cookbook(deepseek-v4): add MORI disagg backend for AMD + bump MI355X image

- 链接: https://github.com/sgl-project/sglang/pull/30651
- 状态/时间: merged / 2026-07-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+52/-29，可读 patch 274 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「cookbook(deepseek-v4): add MORI disagg backend for AMD + bump MI355X image」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/_playground.jsx`；技术摘要: 覆盖「cookbook(deepseek-v4): add MORI disagg backend for AMD + bump MI355X image」；主要实现面是 `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/_playground.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +48/-29 (77 lines); hunks: -161,7 +161,7 @@ sgl-eval run aime25 \\; -289,6 +289,9 @@ sgl-eval run aime25 \\；`docs_new/src/snippets/_playground.jsx` modified +4/-0 (4 lines); hunks: -684,6 +684,10 @@ export const Playground = ({ config }) => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +48/-29 (77 lines); hunks: -161,7 +161,7 @@ sgl-eval run aime25 \\; -289,6 +289,9 @@ sgl-eval run aime25 \\
  - `docs_new/src/snippets/_playground.jsx` modified +4/-0 (4 lines); hunks: -684,6 +684,10 @@ export const Playground = ({ config }) => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -161,7 +161,7 @@ sgl-eval run aime25 \\
-    mi355x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260623",
+    mi355x: "lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260708",
@@ -289,6 +289,9 @@ sgl-eval run aime25 \\
+        // MORI-IO transport is AMD-only — hidden on every non-ROCm platform.
+        { id: "mori",     label: "MORI",
+          hide: { hw: ["h100", "h200", "b200", "b300", "gb200", "gb300", "rtx6000"] } },
diff -- docs_new/src/snippets/_playground.jsx
@@ -684,6 +684,10 @@ export const Playground = ({ config }) => {
+        if (next.transferBackend !== "mooncake" && fc.transferBackends
+            && h.isHidden(fc.transferBackends, next.transferBackend, base)) {
+          next.transferBackend = "mooncake"; changed = true;
+        }
```

- 已读文件:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +48/-29; `docs_new/src/snippets/_playground.jsx` modified +4/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/_playground.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28983 - perf(deepseek_v4): enable SGLANG_OPT_FP8_WO_A_GEMM on sm90 (Hopper)

- 链接: https://github.com/sgl-project/sglang/pull/28983
- 状态/时间: merged / 2026-07-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `dee91c51cf78`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+217/-18，可读 patch 306 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「perf(deepseek_v4): enable SGLANG_OPT_FP8_WO_A_GEMM on sm90 (Hopper)」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「perf(deepseek_v4): enable SGLANG_OPT_FP8_WO_A_GEMM on sm90 (Hopper)」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +54/-12 (66 lines); hunks: -32,6 +32,9; -495,10 +498,14 @@ def __init__(; symbols: __init__, forward, _setup_fp8_wo_a_scales，涉及 `__init__, forward, _setup_fp8_wo_a_scales`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +54/-12 (66 lines); hunks: -32,6 +32,9; -495,10 +498,14 @@ def __init__(; symbols: __init__, forward, _setup_fp8_wo_a_scales
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -32,6 +32,9 @@
+from sglang.kernels.ops.quantization.fp8_kernel import (
+    sglang_per_token_group_quant_fp8,
+)
@@ -495,10 +498,14 @@ def __init__(
+            from sglang.srt.layers import deep_gemm_wrapper
-            self.wo_a.weight_scale_inv.format_ue8m0 = True
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +54/-12
- 验证与风险: diff 自带测试面 `test/manual/dsv4/test_wo_a_fp8_sm90.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31373 - [Docs] Align B200 DeepSeek-V4-Pro balanced recipe with MegaMoE

- 链接: https://github.com/sgl-project/sglang/pull/31373
- 状态/时间: merged / 2026-07-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-4，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Align B200 DeepSeek-V4-Pro balanced recipe with MegaMoE」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；技术摘要: 覆盖「[Docs] Align B200 DeepSeek-V4-Pro balanced recipe with MegaMoE」；主要实现面是 `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +4/-4 (8 lines); hunks: -427,15 +427,16 @@ sgl-eval run aime25 \\; -444,7 +445,6 @@ sgl-eval run aime25 \\。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +4/-4 (8 lines); hunks: -427,15 +427,16 @@ sgl-eval run aime25 \\; -444,7 +445,6 @@ sgl-eval run aime25 \\
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -427,15 +427,16 @@ sgl-eval run aime25 \\
-      env: [],
+      env: [
+        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096",
+      ],
-        "--moe-runner-backend flashinfer_mxfp4",
-        "--disable-flashinfer-autotune",
```

- 已读文件:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +4/-4
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #31122 - [Docs] Add AMD-specific HiCache config for DeepSeek V4 playground

- 链接: https://github.com/sgl-project/sglang/pull/31122
- 状态/时间: merged / 2026-07-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+29/-7，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add AMD-specific HiCache config for DeepSeek V4 playground」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/_playground.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[Docs] Add AMD-specific HiCache config for DeepSeek V4 playground」；主要实现面是 `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/_playground.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +9/-3 (12 lines); hunks: -312,12 +312,18 @@ sgl-eval run aime25 \\；`docs_new/src/snippets/_playground.jsx` modified +16/-4 (20 lines); hunks: -870,19 +870,31 @@ export const Playground = ({ config }) => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -514,6 +514,10 @@ To enable HiCache, open the **HiCache** card in the [Playgr...。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +9/-3 (12 lines); hunks: -312,12 +312,18 @@ sgl-eval run aime25 \\
  - `docs_new/src/snippets/_playground.jsx` modified +16/-4 (20 lines); hunks: -870,19 +870,31 @@ export const Playground = ({ config }) => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -514,6 +514,10 @@ To enable HiCache, open the **HiCache** card in the [Playgr...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -312,12 +312,18 @@ sgl-eval run aime25 \\
+      // AMD ROCm (MI300X/MI325X/MI350X/MI355X): page_first_direct + direct io.
+      amdIo: { memLayout: "page_first_direct", ioBackend: "direct", ratio: 4 },
+      amdStorageFileOnly: true,
-        { id: "mooncake",  label: "Mooncake" },
-        { id: "hf3fs",     label: "HF3FS" },
-        { id: "nixl",      label: "NiXL" },
diff -- docs_new/src/snippets/_playground.jsx
@@ -870,19 +870,31 @@ export const Playground = ({ config }) => {
+          const isAmd = sel && /^mi\d/.test(sel.hw);
+          const ratio = (isAmd && fc.amdIo && fc.amdIo.ratio) || 2;
+          const useAmdIo = isAmd && fc.amdIo;
-            "--hicache-ratio 2",
-            "--hicache-size 0",
+            `--hicache-ratio ${ratio}`,
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -514,6 +514,10 @@ To enable HiCache, open the **HiCache** card in the [Playground above](#playgrou
```

- 已读文件:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +9/-3; `docs_new/src/snippets/_playground.jsx` modified +16/-4; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/_playground.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #30238 - [AMD] Support two batch overlap with MTP on DeepSeekV4

- 链接: https://github.com/sgl-project/sglang/pull/30238
- 状态/时间: merged / 2026-07-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py`；关联提交 `e2d021d4ab2f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+167/-1，可读 patch 190 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Support two batch overlap with MTP on DeepSeekV4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py`；技术摘要: 覆盖「[AMD] Support two batch overlap with MTP on DeepSeekV4」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +3/-1 (4 lines); hunks: -2087,7 +2087,9 @@ def _can_run_tbo(self, forward_batch: ForwardBatch) -> bool:; symbols: _can_run_tbo，涉及 `_can_run_tbo`；`test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py` added +149/-0 (149 lines); hunks: -0,0 +1,149; symbols: TestDeepseekV4ProFp4TboMTP, setUpClass, tearDownClass, test_gsm8k_tbo_mtp，涉及 `TestDeepseekV4ProFp4TboMTP, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +3/-1 (4 lines); hunks: -2087,7 +2087,9 @@ def _can_run_tbo(self, forward_batch: ForwardBatch) -> bool:; symbols: _can_run_tbo
  - `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py` added +149/-0 (149 lines); hunks: -0,0 +1,149; symbols: TestDeepseekV4ProFp4TboMTP, setUpClass, tearDownClass, test_gsm8k_tbo_mtp
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -2087,7 +2087,9 @@ def _can_run_tbo(self, forward_batch: ForwardBatch) -> bool:
-            and forward_batch.global_forward_mode.is_extend()
+            # MTP target-verify also reports is_extend(); only real prefill
+            # should enter the prefill TBO strategy.
+            and forward_batch.global_forward_mode.is_extend_without_speculative()
diff -- test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py
@@ -0,0 +1,149 @@
+"""MI35x DeepSeek-V4-Pro FP4 + non-EP DP two-batch-overlap (TBO) + MTP test (8-GPU)
+End-to-end accuracy test for DeepSeek-V4-Pro (1.6T) FP4 with the non-EP DP
+two-batch-overlap path on MI35x ROCm 7.2.
+Registry: nightly-amd-8-gpu-mi35x-deepseek-v4-pro-mtp suite
+"""
+import os
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +3/-1
  - tests: `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py` added +149/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25763 - [Feature] Support DeepSeek-V4 Wint4Abf16 and Win4Afp8.

- 链接: https://github.com/sgl-project/sglang/pull/25763
- 状态/时间: merged / 2026-07-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `bff489284b50`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+561/-34，可读 patch 790 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Support DeepSeek-V4 Wint4Abf16 and Win4Afp8.」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[Feature] Support DeepSeek-V4 Wint4Abf16 and Win4Afp8.」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +14/-7 (21 lines); hunks: -114,7 +114,10; -2672,7 +2675,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, auto_weight_loader，涉及 `load_weights, auto_weight_loader`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +14/-7 (21 lines); hunks: -114,7 +114,10; -2672,7 +2675,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, auto_weight_loader
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -114,7 +114,10 @@
-from sglang.srt.models.deepseek_common.utils import _use_aiter_bpreshuffle_gfx95
+from sglang.srt.models.deepseek_common.utils import (
+    _use_aiter_bpreshuffle_gfx95,
+    is_wint4afp8_or_wint4a16_config,
+)
@@ -2672,7 +2675,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +14/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/csrc/gemm/per_tensor_quant_fp8.cuh`, `python/sglang/jit_kernel/per_tensor_quant_fp8.py`, `python/sglang/kernels/ops/moe/ep_moe_kernels.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31452 - [Docs] Tune DeepSeek-V4 HiCache for MI355X PD

- 链接: https://github.com/sgl-project/sglang/pull/31452
- 状态/时间: merged / 2026-07-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+60/-10，可读 patch 133 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Tune DeepSeek-V4 HiCache for MI355X PD」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/_playground.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[Docs] Tune DeepSeek-V4 HiCache for MI355X PD」；主要实现面是 `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/_playground.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +27/-1 (28 lines); hunks: -161,7 +161,7 @@ sgl-eval run aime25 \\; -325,6 +325,32 @@ sgl-eval run aime25 \\；`docs_new/src/snippets/_playground.jsx` modified +30/-6 (36 lines); hunks: -1015,8 +1015,16 @@ export const Playground = ({ config }) => {; -1028,19 +1036,21 @@ export const Playground = ({ config }) => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +3/-3 (6 lines); hunks: -52,11 +52,11 @@ docker run --gpus all \; -66,7 +66,7 @@ docker run \。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +27/-1 (28 lines); hunks: -161,7 +161,7 @@ sgl-eval run aime25 \\; -325,6 +325,32 @@ sgl-eval run aime25 \\
  - `docs_new/src/snippets/_playground.jsx` modified +30/-6 (36 lines); hunks: -1015,8 +1015,16 @@ export const Playground = ({ config }) => {; -1028,19 +1036,21 @@ export const Playground = ({ config }) => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +3/-3 (6 lines); hunks: -52,11 +52,11 @@ docker run --gpus all \; -66,7 +66,7 @@ docker run \
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -161,7 +161,7 @@ sgl-eval run aime25 \\
-    mi355x: "lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260708",
+    mi355x: "lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260710",
@@ -325,6 +325,32 @@ sgl-eval run aime25 \\
+      roleOverrides: [
+        {
+          when: {
diff -- docs_new/src/snippets/_playground.jsx
@@ -1015,8 +1015,16 @@ export const Playground = ({ config }) => {
-          const ratio = (isAmd && fc.amdIo && fc.amdIo.ratio) || 2;
-          const useAmdIo = isAmd && fc.amdIo;
+          const pdMode = h.findFlagArg(flags, "--disaggregation-mode") || "off";
+          const pdBackend = h.findFlagArg(flags, "--disaggregation-transfer-backend");
+          const roleOverride = (fc.roleOverrides || []).find((item) => {
+            if (!item || item.mode !== pdMode) return false;
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -52,11 +52,11 @@ docker run --gpus all \
```

- 已读文件:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +27/-1; `docs_new/src/snippets/_playground.jsx` modified +30/-6; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +3/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/_playground.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #30272 - Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2

- 链接: https://github.com/sgl-project/sglang/pull/30272
- 状态/时间: merged / 2026-07-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；关联提交 `faf68940939a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+506/-237，可读 patch 1126 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；技术摘要: 覆盖「Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +8/-3 (11 lines); hunks: -1688,9 +1688,14 @@ def match_num_queries(x, value):; symbols: match_num_queries，涉及 `match_num_queries`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +8/-3 (11 lines); hunks: -1688,9 +1688,14 @@ def match_num_queries(x, value):; symbols: match_num_queries
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -1688,9 +1688,14 @@ def match_num_queries(x, value):
-            if forward_batch.forward_mode.is_extend_without_speculative() and (
-                q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD
-                or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
+            # sparse_prefill_fwd does not support SM120.
+            if (
+                forward_batch.forward_mode.is_extend_without_speculative()
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +8/-3
- 验证与风险: diff 自带测试面 `test/registered/unit/layers/quantization/test_mxfp4_sm120_cutlass.py`, `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py`, `test/registered/unit/test_model_overrides.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31705 - [DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph

- 链接: https://github.com/sgl-project/sglang/pull/31705
- 状态/时间: merged / 2026-07-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；关联提交 `688a6d23f144`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；技术摘要: 覆盖「[DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +2/-0 (2 lines); hunks: -94,6 +94,8 @@ def _get_logical_forward_mode(forward_batch: ForwardBatch) ->...; symbols: _get_logical_forward_mode，涉及 `_get_logical_forward_mode`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +2/-0 (2 lines); hunks: -94,6 +94,8 @@ def _get_logical_forward_mode(forward_batch: ForwardBatch) ->...; symbols: _get_logical_forward_mode
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -94,6 +94,8 @@ def _get_logical_forward_mode(forward_batch: ForwardBatch) -> ForwardMode:
+    if forward_batch.forward_mode == ForwardMode.EXTEND:
+        return forward_batch.forward_mode
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31363 - docs(cookbook): re-benchmark DeepSeek-V4 on sglang 0.5.15

- 链接: https://github.com/sgl-project/sglang/pull/31363
- 状态/时间: merged / 2026-07-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+231/-71，可读 patch 594 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): re-benchmark DeepSeek-V4 on sglang 0.5.15」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/_deployment.jsx`；技术摘要: 覆盖「docs(cookbook): re-benchmark DeepSeek-V4 on sglang 0.5.15」；主要实现面是 `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/_deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +209/-59 (268 lines); hunks: -1,113 +1,181; -118,6 +186,7 @@ export const benchmarks = [；`docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +3/-3 (6 lines); hunks: -4,7 +4,7; -80,7 +80,7 @@ export const config = {；`docs_new/src/snippets/_deployment.jsx` modified +3/-1 (4 lines); hunks: -25,6 +25,8; -577,7 +579,7 @@ export const Deployment = ({ config, benchmarks }) => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +209/-59 (268 lines); hunks: -1,113 +1,181; -118,6 +186,7 @@ export const benchmarks = [
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +3/-3 (6 lines); hunks: -4,7 +4,7; -80,7 +80,7 @@ export const config = {
  - `docs_new/src/snippets/_deployment.jsx` modified +3/-1 (4 lines); hunks: -25,6 +25,8; -577,7 +579,7 @@ export const Deployment = ({ config, benchmarks }) => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx
@@ -1,113 +1,181 @@
-// Measured on sglang v0.5.12.post1.
-// tokens_per_sec_per_gpu is total (input+output) tok/s/GPU: fp4/fp8 = measured
-// output/GPU × (isl+osl)/osl; nvfp4 was measured as total already.
+// Measured on sglang v0.5.15 / v0.5.15.post1 (per-cell sglang_version).
+// tokens_per_sec_per_gpu is total (input+output) tok/s/GPU = output/GPU × (isl+osl)/osl.
-    sglang_version: "0.5.12.post1",
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -4,7 +4,7 @@
-  latencyPercentile: "Mean", // temporary; re-measure to P50
+  latencyPercentile: "P50",
@@ -80,7 +80,7 @@ export const config = {
-  --warmup-requests 64`,
+  --warmup-requests 64 --flush-cache`,
@@ -1276,7 +1276,7 @@ sgl-eval run aime25 \\
diff -- docs_new/src/snippets/_deployment.jsx
@@ -25,6 +25,8 @@
```

- 已读文件:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +209/-59; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +3/-3; `docs_new/src/snippets/_deployment.jsx` modified +3/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/_deployment.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27657 - [DeepSeek V4] CP decode opt: slice repeat attention weights to local TP partition

- 链接: https://github.com/sgl-project/sglang/pull/27657
- 状态/时间: merged / 2026-07-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`；关联提交 `ebe3ab29e485`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+336/-49，可读 patch 533 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek V4] CP decode opt: slice repeat attention weights to local TP partition」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`；技术摘要: 覆盖「[DeepSeek V4] CP decode opt: slice repeat attention weights to local TP partition」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +60/-25 (85 lines); hunks: -4,7 +4,7; -60,6 +60,7; symbols: __init__, _local_attn_sink, maybe_use_decode_attn_tp, MQALayer，涉及 `__init__, _local_attn_sink, maybe_use_decode_attn_tp`；`python/sglang/srt/models/deepseek_v4_dspark.py` modified +2/-12 (14 lines); hunks: -121,17 +121,6 @@ def kv_proj_only(self, x: torch.Tensor) -> torch.Tensor:; -536,7 +525,8 @@ def forward(; symbols: kv_proj_only, _local_attn_sink, _store_block_kv, forward，涉及 `kv_proj_only, _local_attn_sink, _store_block_kv`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +60/-25 (85 lines); hunks: -4,7 +4,7; -60,6 +60,7; symbols: __init__, _local_attn_sink, maybe_use_decode_attn_tp, MQALayer
  - `python/sglang/srt/models/deepseek_v4_dspark.py` modified +2/-12 (14 lines); hunks: -121,17 +121,6 @@ def kv_proj_only(self, x: torch.Tensor) -> torch.Tensor:; -536,7 +525,8 @@ def forward(; symbols: kv_proj_only, _local_attn_sink, _store_block_kv, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -4,7 +4,7 @@
-from contextlib import nullcontext
+from contextlib import contextmanager, nullcontext
@@ -60,6 +60,7 @@
+from sglang.srt.layers.cp.cp_decode_attn_tp import get_cp_decode_attn_tp_ctx
@@ -453,9 +454,7 @@ def __init__(
-        self._attn_sink_local: Optional[torch.Tensor] = (
diff -- python/sglang/srt/models/deepseek_v4_dspark.py
@@ -121,17 +121,6 @@ def kv_proj_only(self, x: torch.Tensor) -> torch.Tensor:
-    def _local_attn_sink(self) -> torch.Tensor:
-        if self.attn_tp_size == 1:
-            return self.attn_sink
-        if self._attn_sink_local is None:
-            rank = self.attn_tp_rank
-            num_heads = self.n_local_heads
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +60/-25; `python/sglang/srt/models/deepseek_v4_dspark.py` modified +2/-12
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/cp/cp_decode_attn_tp.py`, `python/sglang/srt/layers/linear.py`, `python/sglang/srt/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29569 - [DSV4] Support megamoe for CP

- 链接: https://github.com/sgl-project/sglang/pull/29569
- 状态/时间: merged / 2026-07-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`；关联提交 `71fe41b6b3c7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+167/-7，可读 patch 235 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Support megamoe for CP」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`；技术摘要: 覆盖「[DSV4] Support megamoe for CP」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +6/-4 (10 lines); hunks: -1748,12 +1748,14 @@ def _run_moe_ffn_dp_sync(; symbols: _run_moe_ffn_dp_sync，涉及 `_run_moe_ffn_dp_sync`；`python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +98/-0 (98 lines); hunks: -11,6 +11,98; -85,6 +177,12 @@ def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:; symbols: validate_deepseek_v4_mega_moe_token_budget, apply_deepseek_v4_defaults, validate_deepseek_v4_cp，涉及 `validate_deepseek_v4_mega_moe_token_budget, apply_deepseek_v4_defaults, validate_deepseek_v4_cp`；`test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +56/-1 (57 lines); hunks: -37,8 +37,14; -92,6 +98,55 @@ def tearDownClass(cls):; symbols: TestDSV4FlashFP4B200Balanced_CP, TestDSV4FlashFP4B200Balanced_CP_DeepEP, tearDownClass, TestDSV4FlashFP4B200Balanced_CP_Megamoe，涉及 `TestDSV4FlashFP4B200Balanced_CP, TestDSV4FlashFP4B200Balanced_CP_DeepEP, tearDownClass`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +6/-4 (10 lines); hunks: -1748,12 +1748,14 @@ def _run_moe_ffn_dp_sync(; symbols: _run_moe_ffn_dp_sync
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +98/-0 (98 lines); hunks: -11,6 +11,98; -85,6 +177,12 @@ def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:; symbols: validate_deepseek_v4_mega_moe_token_budget, apply_deepseek_v4_defaults, validate_deepseek_v4_cp
  - `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +56/-1 (57 lines); hunks: -37,8 +37,14; -92,6 +98,55 @@ def tearDownClass(cls):; symbols: TestDSV4FlashFP4B200Balanced_CP, TestDSV4FlashFP4B200Balanced_CP_DeepEP, tearDownClass, TestDSV4FlashFP4B200Balanced_CP_Megamoe
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -1748,12 +1748,14 @@ def _run_moe_ffn_dp_sync(
-            if get_moe_a2a_backend().is_none():
+            moe_a2a_backend = get_moe_a2a_backend()
+            if moe_a2a_backend.is_none():
-                assert get_moe_a2a_backend().is_deepep(), (
-                    "CP requires DeepEP (moe_a2a_backend == deepep). "
-                    "Only DeepEP is tested with CP's per-rank token split."
diff -- python/sglang/srt/arg_groups/deepseek_v4_hook.py
@@ -11,6 +11,98 @@
+def validate_deepseek_v4_mega_moe_token_budget(
+    server_args: ServerArgs,
+) -> None:
+    """Ensure the DSV4 prefill budget fits MegaMoE's per-rank buffer."""
+    mega_moe_enabled = (
+        server_args.moe_a2a_backend == "megamoe"
diff -- test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py
@@ -37,8 +37,14 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +6/-4; `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +98/-0
  - tests: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +56/-1
- 验证与风险: diff 自带测试面 `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27059 - Add FP4 Indexer for DeepSeek V4 on SM120

- 链接: https://github.com/sgl-project/sglang/pull/27059
- 状态/时间: merged / 2026-07-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；关联提交 `1e69765bae5b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+76/-10，可读 patch 157 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add FP4 Indexer for DeepSeek V4 on SM120」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；技术摘要: 覆盖「Add FP4 Indexer for DeepSeek V4 on SM120」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +5/-0 (5 lines); hunks: -650,6 +650,11 @@ def init_forward_metadata_indexer(; symbols: init_forward_metadata_indexer，涉及 `init_forward_metadata_indexer`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +5/-0 (5 lines); hunks: -650,6 +650,11 @@ def init_forward_metadata_indexer(; symbols: init_forward_metadata_indexer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -650,6 +650,11 @@ def init_forward_metadata_indexer(
+            # The SM120 FP4 kernel schedules split_kv=128, while the generic
+            # JIT metadata planner encodes split_kv=256.
+            force_deep_gemm_metadata=(
+                self.enable_deepseek_v4_fp4_indexer and _is_sm120
+            ),
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +5/-0
- 验证与风险: diff 自带测试面 `test/registered/kernels/benchmark/attention/bench_dsv4_fp4_indexer.py`, `test/registered/unit/layers/test_dsv4_nonpaged_indexer.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31087 - [RL] DSV4: dispatch indexer topk_transform_512 through DSATopKBackend

- 链接: https://github.com/sgl-project/sglang/pull/31087
- 状态/时间: merged / 2026-07-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；关联提交 `f7986c8603f7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+96/-9，可读 patch 179 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[RL] DSV4: dispatch indexer topk_transform_512 through DSATopKBackend」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；技术摘要: 覆盖「[RL] DSV4: dispatch indexer topk_transform_512 through DSATopKBackend」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +4/-0 (4 lines); hunks: -40,6 +40,7; -528,6 +529,9 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +4/-0 (4 lines); hunks: -40,6 +40,7; -528,6 +529,9 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -40,6 +40,7 @@
+from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
@@ -528,6 +529,9 @@ def __init__(
+        self.dsa_topk_backend: DSATopKBackend = DSATopKBackend(
+            model_runner.server_args.dsa_topk_backend
+        )
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/dsv4/indexer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31086 - [RL] DSV4: add env to quantize SWA KV cache from bf16-rounded values

- 链接: https://github.com/sgl-project/sglang/pull/31086
- 状态/时间: merged / 2026-07-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `0a212c611909`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+17/-0，可读 patch 38 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[RL] DSV4: add env to quantize SWA KV cache from bf16-rounded values」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[RL] DSV4: add env to quantize SWA KV cache from bf16-rounded values」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +9/-0 (9 lines); hunks: -722,6 +722,15 @@ def _compute_kv_to_cache(; symbols: _compute_kv_to_cache，涉及 `_compute_kv_to_cache`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +9/-0 (9 lines); hunks: -722,6 +722,15 @@ def _compute_kv_to_cache(; symbols: _compute_kv_to_cache
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -722,6 +722,15 @@ def _compute_kv_to_cache(
+        if envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.get():
+            # Quantize the nope payload from bf16-rounded values (the fused
+            # kernel quantizes from fp32 registers; the bf16 rounding moves
+            # values across fp8 bins relative to bf16-sourced consumers).
+            kv = self._compute_kv_bf16(x, positions, qkv_a=qkv_a)
+            attn_backend.store_cache(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +9/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/environ.py`, `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30954 - [SM120] Allow fused MHC opt-in with standalone TileLang pre disabled

- 链接: https://github.com/sgl-project/sglang/pull/30954
- 状态/时间: merged / 2026-07-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `2cbddb842d67`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+84/-3，可读 patch 105 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[SM120] Allow fused MHC opt-in with standalone TileLang pre disabled」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[SM120] Allow fused MHC opt-in with standalone TileLang pre disabled」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +5/-3 (8 lines); hunks: -153,6 +153,7; -204,12 +205,13 @@ def _get_mhc_ops() -> MhcOps:; symbols: _get_mhc_ops, _is_fused_mhc_post_pre_enabled，涉及 `_get_mhc_ops, _is_fused_mhc_post_pre_enabled`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +5/-3 (8 lines); hunks: -153,6 +153,7; -204,12 +205,13 @@ def _get_mhc_ops() -> MhcOps:; symbols: _get_mhc_ops, _is_fused_mhc_post_pre_enabled
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -153,6 +153,7 @@
+from sglang.srt.utils.common import is_sm120_supported
@@ -204,12 +205,13 @@ def _get_mhc_ops() -> MhcOps:
-    # The fused path directly reuses TileLang mhc_post/mhc_pre kernels and their
-    # tensor layout assumptions, so keep it disabled when either dependency is off.
+    # SM120 disables the standalone TileLang pre path. mhc_fused_post_pre does
+    # not read that flag and dispatches independently for both small and large
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +5/-3
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_deepseek_v4_fused_mhc_policy.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31931 - [NPU] Optimize DeepSeek-V4 performance

- 链接: https://github.com/sgl-project/sglang/pull/31931
- 状态/时间: merged / 2026-07-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/ops/attention/deepseek_v4_rope.py`, `python/sglang/srt/models/deepseek_v4.py`；关联提交 `5558dbad0037`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 21 个文件，+1308/-520，可读 patch 2590 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Optimize DeepSeek-V4 performance」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/kernels/ops/attention/deepseek_v4_rope.py`；技术摘要: 覆盖「[NPU] Optimize DeepSeek-V4 performance」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/kernels/ops/attention/deepseek_v4_rope.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +59/-22 (81 lines); hunks: -23,9 +23,6; -47,6 +44,7; symbols: __init__, _get_npu_rope_position_cache, _compute_q_a, _forward_prepare，涉及 `__init__, _get_npu_rope_position_cache, _compute_q_a`；`python/sglang/kernels/ops/attention/deepseek_v4_rope.py` modified +0/-120 (120 lines); hunks: -649,123 +649,3 @@ def fused_norm_rope_inplace_triton(; symbols: fused_norm_rope_inplace_triton, _get_contig_freqs_real_imag, get_fused_compressor_rope_cos_sin, v4_rope_inplace_npu，涉及 `fused_norm_rope_inplace_triton, _get_contig_freqs_real_imag, get_fused_compressor_rope_cos_sin`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +59/-22 (81 lines); hunks: -23,9 +23,6; -47,6 +44,7; symbols: __init__, _get_npu_rope_position_cache, _compute_q_a, _forward_prepare
  - `python/sglang/kernels/ops/attention/deepseek_v4_rope.py` modified +0/-120 (120 lines); hunks: -649,123 +649,3 @@ def fused_norm_rope_inplace_triton(; symbols: fused_norm_rope_inplace_triton, _get_contig_freqs_real_imag, get_fused_compressor_rope_cos_sin, v4_rope_inplace_npu
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -23,9 +23,6 @@
-from sglang.kernels.ops.attention.deepseek_v4_rope import (
-    v4_rope_inplace_npu,
-)
@@ -47,6 +44,7 @@
+from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE
@@ -622,6 +620,11 @@ def __init__(
diff -- python/sglang/kernels/ops/attention/deepseek_v4_rope.py
@@ -649,123 +649,3 @@ def fused_norm_rope_inplace_triton(
-# Cache contiguous real/imag halves of each freqs_cis (its .real/.imag are
-# strided views, stride=2 on the interleaved layout), keyed by id.
-_NPU_ROPE_CONTIG_CACHE: dict[int, tuple] = {}
-def _get_contig_freqs_real_imag(
-    freqs_cis: torch.Tensor,
-) -> tuple[torch.Tensor, torch.Tensor]:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +59/-22; `python/sglang/kernels/ops/attention/deepseek_v4_rope.py` modified +0/-120
- 验证与风险: runtime 路径改动集中在 `python/sglang/kernels/ops/attention/deepseek_v4_rope.py`, `python/sglang/srt/disaggregation/ascend/conn.py`, `python/sglang/srt/disaggregation/decode.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31747 - [AMD] DSv4: bring HIP compress-state pool into the memory_saver KV_CACHE region

- 链接: https://github.com/sgl-project/sglang/pull/31747
- 状态/时间: merged / 2026-07-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；关联提交 `f5bcd00e1638`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+19/-27，可读 patch 61 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] DSv4: bring HIP compress-state pool into the memory_saver KV_CACHE region」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「[AMD] DSv4: bring HIP compress-state pool into the memory_saver KV_CACHE region」；主要实现面是 `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +18/-23 (41 lines); hunks: -153,30 +153,25 @@ def _alloc_kv_score_buffer(; symbols: _alloc_kv_score_buffer, state_cache_3d，涉及 `_alloc_kv_score_buffer, state_cache_3d`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-4 (5 lines); hunks: -646,10 +646,7 @@ def __init__(; symbols: __init__, get_unified_kv，涉及 `__init__, get_unified_kv`。
- 代码 diff 细节:
  - `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +18/-23 (41 lines); hunks: -153,30 +153,25 @@ def _alloc_kv_score_buffer(; symbols: _alloc_kv_score_buffer, state_cache_3d
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-4 (5 lines); hunks: -646,10 +646,7 @@ def __init__(; symbols: __init__, get_unified_kv
- 关键代码摘录:

```diff
diff -- python/sglang/srt/mem_cache/deepseek_v4_compress_state.py
@@ -153,30 +153,25 @@ def _alloc_kv_score_buffer(
-        if _is_hip:
-            self.kv_score_buffer = KVAndScore(
-                torch.empty((self._size, self.last_dim), dtype=dtype, device=device)
-            )
-        else:
-            self.memory_saver_adapter = TorchMemorySaverAdapter.create(
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -646,10 +646,7 @@ def __init__(
-        if _is_hip:
-            self._init_paged_compress_states(False)
-        else:
-            self._init_paged_compress_states(enable_memory_saver)
+        self._init_paged_compress_states(enable_memory_saver)
```

- 已读文件:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +18/-23; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31563 - fix mqa preshuffle layout issue for deepseek v4

- 链接: https://github.com/sgl-project/sglang/pull/31563
- 状态/时间: merged / 2026-07-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh`；关联提交 `1c6a0e91e125`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+92/-13，可读 patch 224 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix mqa preshuffle layout issue for deepseek v4」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh`；技术摘要: 覆盖「fix mqa preshuffle layout issue for deepseek v4」；主要实现面是 `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh` modified +23/-4 (27 lines); hunks: -64,7 +64,7 @@ enum class ForwardMode : bool {; -213,7 +213,19 @@ INDEXER_KERNEL void fused_norm_rope_indexer(const __grid_co...; symbols: ForwardMode，涉及 `ForwardMode`。
- 代码 diff 细节:
  - `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh` modified +23/-4 (27 lines); hunks: -64,7 +64,7 @@ enum class ForwardMode : bool {; -213,7 +213,19 @@ INDEXER_KERNEL void fused_norm_rope_indexer(const __grid_co...; symbols: ForwardMode
- 关键代码摘录:

```diff
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh
@@ -64,7 +64,7 @@ enum class ForwardMode : bool {
-template <typename DType, ForwardMode kMode, int32_t kPageBits, bool kUsePDL>
+template <typename DType, ForwardMode kMode, int32_t kPageBits, bool kUsePDL, int32_t kPreshuffleSize = 0>
@@ -213,7 +213,19 @@ INDEXER_KERNEL void fused_norm_rope_indexer(const __grid_constant__ FusedNormRop
-    result.store(value_ptr, lane_id);
+    if constexpr (kPreshuffleSize != 0) {
+      constexpr int32_t kTile = kPreshuffleSize;
```

- 已读文件:
  - runtime: `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh` modified +23/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh`, `python/sglang/kernels/ops/attention/dsa/index_buf_accessor.py`, `python/sglang/kernels/ops/attention/dsv4/compress.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30240 - Fix DeepSeek V4 loading with RunAI Model Streamer.

- 链接: https://github.com/sgl-project/sglang/pull/30240
- 状态/时间: merged / 2026-07-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`；关联提交 `b61cb5f9de87`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+185/-14，可读 patch 262 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix DeepSeek V4 loading with RunAI Model Streamer.」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`；技术摘要: 覆盖「Fix DeepSeek V4 loading with RunAI Model Streamer.」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +60/-10 (70 lines); hunks: -121,7 +121,10; -2777,13 +2780,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights, auto_weight_loader, _dequant_fp8, _clone_if_runai_streamed_tensor，涉及 `load_weights, auto_weight_loader, _dequant_fp8`；`python/sglang/srt/models/deepseek_v4_dspark.py` modified +2/-4 (6 lines); hunks: -30,7 +30,7; -761,9 +761,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +60/-10 (70 lines); hunks: -121,7 +121,10; -2777,13 +2780,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights, auto_weight_loader, _dequant_fp8, _clone_if_runai_streamed_tensor
  - `python/sglang/srt/models/deepseek_v4_dspark.py` modified +2/-4 (6 lines); hunks: -30,7 +30,7; -761,9 +761,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -121,7 +121,10 @@
-from sglang.srt.model_loader.weight_utils import default_weight_loader
+from sglang.srt.model_loader.weight_utils import (
+    RUNAI_STREAMER_TENSOR_ATTR,
+    default_weight_loader,
+)
@@ -2777,13 +2780,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
diff -- python/sglang/srt/models/deepseek_v4_dspark.py
@@ -30,7 +30,7 @@
-    _dequant_fp8_wo_a,
+    _dequant_fp8_wo_a_streaming,
@@ -761,9 +761,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
-        weights = list(weights)
-        if any(name.endswith(".wo_a.scale") for name, _ in weights):
-            weights = list(_dequant_fp8_wo_a(weights))
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +60/-10; `python/sglang/srt/models/deepseek_v4_dspark.py` modified +2/-4
- 验证与风险: diff 自带测试面 `test/registered/unit/model_loader/test_runai_model_streamer_loader.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31727 - [AMD] Fix DeepSeek-V4 fused-RMS FP8 scale metadata on gfx950

- 链接: https://github.com/sgl-project/sglang/pull/31727
- 状态/时间: merged / 2026-08-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `1685d29f2127`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+116/-0，可读 patch 169 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix DeepSeek-V4 fused-RMS FP8 scale metadata on gfx950」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[AMD] Fix DeepSeek-V4 fused-RMS FP8 scale metadata on gfx950」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +8/-0 (8 lines); hunks: -84,6 +84,9; -249,6 +252,11 @@ def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):; symbols: _fused_rmsnorm_fp8_quant，涉及 `_fused_rmsnorm_fp8_quant`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +8/-0 (8 lines); hunks: -84,6 +84,9; -249,6 +252,11 @@ def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):; symbols: _fused_rmsnorm_fp8_quant
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -84,6 +84,9 @@
+from sglang.srt.layers.quantization.fp8_utils import (
+    view_aiter_fused_rms_transposed_fp8_scale,
+)
@@ -249,6 +252,11 @@ def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):
+    if _use_aiter_bpreshuffle_gfx95:
+        x_quant = (
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +8/-0
- 验证与风险: diff 自带测试面 `test/registered/quant/test_fused_rms_fp8_group_quant.py`, `test/registered/unit/layers/test_fp8_bpreshuffle_scale.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32910 - [DeepSeek-V4] Fix nvcc 13 crash building the topk_v2 kernel

- 链接: https://github.com/sgl-project/sglang/pull/32910
- 状态/时间: merged / 2026-08-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh`；关联提交 `28a2472f9588`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-2，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek-V4] Fix nvcc 13 crash building the topk_v2 kernel」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh`；技术摘要: 覆盖「[DeepSeek-V4] Fix nvcc 13 crash building the topk_v2 kernel」；主要实现面是 `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` modified +7/-2 (9 lines); hunks: -220,8 +220,13 @@ CLUSTER_TOPK_KERNEL void topk_small_batch_kernel(const __gr...。
- 代码 diff 细节:
  - `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` modified +7/-2 (9 lines); hunks: -220,8 +220,13 @@ CLUSTER_TOPK_KERNEL void topk_small_batch_kernel(const __gr...
- 关键代码摘录:

```diff
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh
@@ -220,8 +220,13 @@ CLUSTER_TOPK_KERNEL void topk_small_batch_kernel(const __grid_constant__ TopKLau
-    problem.out = cluster.map_shared_rank(topk_indices, worker_rank);
-    Cluster::forward<kPDL>(problem, &smem);  // write to peer's output shared memory
+    // The mapped alias stays in a copy: the elected rank reads the very same
+    // bytes back through `topk_indices` below, and letting a shared::cluster
+    // address reach the `problem.out` that problem_transform loads makes cicc
+    // segfault on CUDA 13.x (issue #32830).
```

- 已读文件:
  - runtime: `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` modified +7/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30741 - Prewarm DSV4 MHC post kernel at model load

- 链接: https://github.com/sgl-project/sglang/pull/30741
- 状态/时间: merged / 2026-08-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `23ea7b648135`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+25/-10，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Prewarm DSV4 MHC post kernel at model load」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「Prewarm DSV4 MHC post kernel at model load」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +25/-10 (35 lines); hunks: -2752,8 +2752,8 @@ def remap_weight_name_to_dpsk_hf_format(; -2774,16 +2774,17 @@ def _prewarm_mhc_pre_kernels(self) -> None:; symbols: remap_weight_name_to_dpsk_hf_format, _prewarm_mhc_pre_kernels, _prewarm_mhc_kernels，涉及 `remap_weight_name_to_dpsk_hf_format, _prewarm_mhc_pre_kernels, _prewarm_mhc_kernels`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +25/-10 (35 lines); hunks: -2752,8 +2752,8 @@ def remap_weight_name_to_dpsk_hf_format(; -2774,16 +2774,17 @@ def _prewarm_mhc_pre_kernels(self) -> None:; symbols: remap_weight_name_to_dpsk_hf_format, _prewarm_mhc_pre_kernels, _prewarm_mhc_kernels
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -2752,8 +2752,8 @@ def remap_weight_name_to_dpsk_hf_format(
-    def _prewarm_mhc_pre_kernels(self) -> None:
-        """One-shot mhc_pre() JIT prewarm at load time, synced across ranks.
+    def _prewarm_mhc_kernels(self) -> None:
+        """One-shot MHC JIT prewarm at load time, synced across ranks.
@@ -2774,16 +2774,17 @@ def _prewarm_mhc_pre_kernels(self) -> None:
-        from sglang.kernels.ops.layernorm.mhc import prewarm_mhc_pre
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +25/-10
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31865 - [XPU] DeepSeek V4: use sgl-kernel-xpu implemetation of flash_mla_sparse_fwd for prefill

- 链接: https://github.com/sgl-project/sglang/pull/31865
- 状态/时间: merged / 2026-08-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；关联提交 `3b4fac5b9967`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[XPU] DeepSeek V4: use sgl-kernel-xpu implemetation of flash_mla_sparse_fwd for prefill」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；技术摘要: 覆盖「[XPU] DeepSeek V4: use sgl-kernel-xpu implemetation of flash_mla_sparse_fwd for prefill」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +4/-1 (5 lines); hunks: -1780,7 +1780,10 @@ def _forward_prefill_sparse(; symbols: _forward_prefill_sparse，涉及 `_forward_prefill_sparse`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +4/-1 (5 lines); hunks: -1780,7 +1780,10 @@ def _forward_prefill_sparse(; symbols: _forward_prefill_sparse
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -1780,7 +1780,10 @@ def _forward_prefill_sparse(
-        from sgl_kernel.flash_mla import flash_mla_sparse_fwd
+        if _is_xpu:
+            from sgl_kernel import flash_mla_sparse_fwd
+        else:
+            from sgl_kernel.flash_mla import flash_mla_sparse_fwd
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33616 - feat: Add flashinfer mHC fusion for DSV4

- 链接: https://github.com/sgl-project/sglang/pull/33616
- 状态/时间: merged / 2026-08-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `3ed2a0adf3d8`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+86/-0，可读 patch 114 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: Add flashinfer mHC fusion for DSV4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「feat: Add flashinfer mHC fusion for DSV4」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +85/-0 (85 lines); hunks: -219,6 +219,74 @@ def _is_fused_mhc_post_pre_enabled() -> bool:; -1513,6 +1581,18 @@ def hc_pre_torch_impl(x, hc_fn):; symbols: _is_fused_mhc_post_pre_enabled, _cuda_sm_count, _flashinfer_mhc_pre_num_splits, _flashinfer_hc_pre，涉及 `_is_fused_mhc_post_pre_enabled, _cuda_sm_count, _flashinfer_mhc_pre_num_splits`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +85/-0 (85 lines); hunks: -219,6 +219,74 @@ def _is_fused_mhc_post_pre_enabled() -> bool:; -1513,6 +1581,18 @@ def hc_pre_torch_impl(x, hc_fn):; symbols: _is_fused_mhc_post_pre_enabled, _cuda_sm_count, _flashinfer_mhc_pre_num_splits, _flashinfer_hc_pre
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -219,6 +219,74 @@ def _is_fused_mhc_post_pre_enabled() -> bool:
+# FlashInfer's mhc_pre_big_fuse only accepts these split-K counts.
+_FLASHINFER_MHC_PRE_SPLITS = (1, 2, 4, 8, 16)
+@functools.cache
+def _cuda_sm_count() -> int:
+    return torch.cuda.get_device_properties(0).multi_processor_count
+def _flashinfer_mhc_pre_num_splits(num_tokens: int, hc_hidden_size: int) -> int:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +85/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/environ.py`, `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33532 - [CP]: Support CP V2 Strategy for dsv4

- 链接: https://github.com/sgl-project/sglang/pull/33532
- 状态/时间: merged / 2026-08-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`；关联提交 `1480687cff5a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+161/-59，可读 patch 601 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CP]: Support CP V2 Strategy for dsv4」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`；技术摘要: 覆盖「[CP]: Support CP V2 Strategy for dsv4」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +32/-8 (40 lines); hunks: -58,6 +58,7; -277,11 +278,16 @@ def refresh_for_breakable_cuda_graph_replay_(self, other:...; symbols: refresh_for_breakable_cuda_graph_replay_, init_compression_metadata，涉及 `refresh_for_breakable_cuda_graph_replay_, init_compression_metadata`；`python/sglang/srt/models/deepseek_v4.py` modified +26/-14 (40 lines); hunks: -59,6 +59,11; -1073,9 +1078,8 @@ def _forward_prepare(; symbols: _forward_prepare, forward，涉及 `_forward_prepare, forward`；`python/sglang/srt/models/deepseek_v4_nextn.py` modified +18/-6 (24 lines); hunks: -13,6 +13,10; -115,6 +119,9 @@ def __init__(; symbols: __init__, get_input_embeddings, hc_head, forward，涉及 `__init__, get_input_embeddings, hc_head`；`test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +14/-11 (25 lines); hunks: -1,7 +1,7; -29,6 +29,7; symbols: setUpClass，涉及 `setUpClass`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +32/-8 (40 lines); hunks: -58,6 +58,7; -277,11 +278,16 @@ def refresh_for_breakable_cuda_graph_replay_(self, other:...; symbols: refresh_for_breakable_cuda_graph_replay_, init_compression_metadata
  - `python/sglang/srt/models/deepseek_v4.py` modified +26/-14 (40 lines); hunks: -59,6 +59,11; -1073,9 +1078,8 @@ def _forward_prepare(; symbols: _forward_prepare, forward
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +18/-6 (24 lines); hunks: -13,6 +13,10; -115,6 +119,9 @@ def __init__(; symbols: __init__, get_input_embeddings, hc_head, forward
  - `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +14/-11 (25 lines); hunks: -1,7 +1,7; -29,6 +29,7; symbols: setUpClass
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +1/-0 (1 lines); hunks: -167,6 +167,7 @@ def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:; symbols: validate_deepseek_v4_cp
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -58,6 +58,7 @@
+from sglang.srt.layers.cp.utils import is_cp_v2_active
@@ -277,11 +278,16 @@ def refresh_for_breakable_cuda_graph_replay_(self, other: DSV4AttnMetadata) -> N
-    def init_compression_metadata(self):
+    def init_compression_metadata(self, num_tokens: Optional[int] = None) -> None:
+        # CP-v2 pads causal metadata for per-rank partitioning, while cache-write
+        # locations remain one-per-logical-token. num_tokens tracks that unpadded
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -59,6 +59,11 @@
+from sglang.srt.layers.cp.utils import (
+    cp_materialize_global_token_order,
+    cp_round_robin_input_ids_v2,
+    is_cp_v2_active,
+)
@@ -1073,9 +1078,8 @@ def _forward_prepare(
diff -- python/sglang/srt/models/deepseek_v4_nextn.py
@@ -13,6 +13,10 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +32/-8; `python/sglang/srt/models/deepseek_v4.py` modified +26/-14; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +18/-6; `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +1/-0
  - tests: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +14/-11
- 验证与风险: diff 自带测试面 `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34044 - docs(cookbook): DeepSeek-V4-Flash-0731 — drop chunked-prefill/autotune flags on B300 low-latency

- 链接: https://github.com/sgl-project/sglang/pull/34044
- 状态/时间: merged / 2026-08-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；关联提交 `86f373daff12`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-3，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): DeepSeek-V4-Flash-0731 — drop chunked-prefill/autotune flags on B300 low-latency」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；技术摘要: 覆盖「docs(cookbook): DeepSeek-V4-Flash-0731 — drop chunked-prefill/autotune flags on B300 low-latency」；主要实现面是 `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +1/-3 (4 lines); hunks: -620,16 +620,14 @@ sgl-eval run aime25 \\。
- 代码 diff 细节:
  - `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +1/-3 (4 lines); hunks: -620,16 +620,14 @@ sgl-eval run aime25 \\
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -620,16 +620,14 @@ sgl-eval run aime25 \\
-      verified: false,
+      verified: true,
-        "--chunked-prefill-size 4096",
-        "--disable-flashinfer-autotune",
```

- 已读文件:
  - docs: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +1/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #30964 - [AMD] Support DeepSeek V4 DSpark on AMD HIP platform

- 链接: https://github.com/sgl-project/sglang/pull/30964
- 状态/时间: merged / 2026-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py`；关联提交 `ba7abd4f92de`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+586/-52，可读 patch 897 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Support DeepSeek V4 DSpark on AMD HIP platform」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py`；技术摘要: 覆盖「[AMD] Support DeepSeek V4 DSpark on AMD HIP platform」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +102/-27 (129 lines); hunks: -414,6 +414,7 @@ class DeepseekV4HipRadixBackend(; -456,6 +457,18 @@ def __init__(; symbols: DeepseekV4HipRadixBackend, __init__, init_forward_metadata_prefill，涉及 `DeepseekV4HipRadixBackend, __init__, init_forward_metadata_prefill`；`python/sglang/srt/models/deepseek_v4_dspark.py` modified +27/-1 (28 lines); hunks: -9,6 +9,9; -135,6 +138,20 @@ def _store_block_kv(; symbols: _store_block_kv, write_target_hidden_kv，涉及 `_store_block_kv, write_target_hidden_kv`；`test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: TestDSparkUnifiedKVKernelsAMD, test_build_unified_commit_inject_layout, test_scatter_bf16_into_unified, TestDeepseekV4DSparkUnifiedKVGSM8K，涉及 `TestDSparkUnifiedKVKernelsAMD, test_build_unified_commit_inject_layout, test_scatter_bf16_into_unified`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +28/-0 (28 lines); hunks: -1192,6 +1192,34 @@ def set_swa_key_buffer_radix_fused_norm_rope(; symbols: set_swa_key_buffer_radix_fused_norm_rope, set_unified_key_buffer_radix_fused_norm_rope, set_extra_key_buffer_fused，涉及 `set_swa_key_buffer_radix_fused_norm_rope, set_unified_key_buffer_radix_fused_norm_rope, set_extra_key_buffer_fused`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +102/-27 (129 lines); hunks: -414,6 +414,7 @@ class DeepseekV4HipRadixBackend(; -456,6 +457,18 @@ def __init__(; symbols: DeepseekV4HipRadixBackend, __init__, init_forward_metadata_prefill
  - `python/sglang/srt/models/deepseek_v4_dspark.py` modified +27/-1 (28 lines); hunks: -9,6 +9,9; -135,6 +138,20 @@ def _store_block_kv(; symbols: _store_block_kv, write_target_hidden_kv
  - `test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: TestDSparkUnifiedKVKernelsAMD, test_build_unified_commit_inject_layout, test_scatter_bf16_into_unified, TestDeepseekV4DSparkUnifiedKVGSM8K
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +28/-0 (28 lines); hunks: -1192,6 +1192,34 @@ def set_swa_key_buffer_radix_fused_norm_rope(; symbols: set_swa_key_buffer_radix_fused_norm_rope, set_unified_key_buffer_radix_fused_norm_rope, set_extra_key_buffer_fused
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -414,6 +414,7 @@ class DeepseekV4HipRadixBackend(
+    supports_ragged_verify_graph: bool = True
@@ -456,6 +457,18 @@ def __init__(
+        self.is_dspark_draft = (
+            getattr(model_runner, "is_draft_worker", False)
+            and model_runner.spec_algorithm.is_dspark()
+        )
diff -- python/sglang/srt/models/deepseek_v4_dspark.py
@@ -9,6 +9,9 @@
+from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
+    is_unified_kv_triton,
+)
@@ -135,6 +138,20 @@ def _store_block_kv(
+        if is_unified_kv_triton():
+            # unified_kv: SWA K lives in the shared bf16 ring (swa_kv_pool is
diff -- test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py
@@ -0,0 +1,201 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +102/-27; `python/sglang/srt/models/deepseek_v4_dspark.py` modified +27/-1; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +28/-0
  - tests: `test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py` added +201/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34189 - [DSV4] Fix silent KV corruption when speculative draft tokens > 4

- 链接: https://github.com/sgl-project/sglang/pull/34189
- 状态/时间: merged / 2026-08-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py`；关联提交 `4a5d7d3c67e6`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+314/-7，可读 patch 390 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Fix silent KV corruption when speculative draft tokens > 4」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py`, `python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`；技术摘要: 覆盖「[DSV4] Fix silent KV corruption when speculative draft tokens > 4」；主要实现面是 `test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py`, `python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py` added +217/-0 (217 lines); hunks: -0,0 +1,217; symbols: _window_size, _max_draft_tokens, _written_positions, TestCompressWritePlanDraftPad，涉及 `_window_size, _max_draft_tokens, _written_positions`；`python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh` modified +12/-4 (16 lines); hunks: -47,6 +47,8 @@ struct Prefill0Params {; -509,13 +511,19 @@ inline PrefillPlan plan_compress_prefill(；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +8/-0 (8 lines); hunks: -48,6 +48,14 @@ def get_compress_state_ring_size(; symbols: get_compress_state_ring_size, get_compress_state_write_pad, DeepSeekV4SingleKVPool, __init__，涉及 `get_compress_state_ring_size, get_compress_state_write_pad, DeepSeekV4SingleKVPool`。
- 代码 diff 细节:
  - `test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py` added +217/-0 (217 lines); hunks: -0,0 +1,217; symbols: _window_size, _max_draft_tokens, _written_positions, TestCompressWritePlanDraftPad
  - `python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh` modified +12/-4 (16 lines); hunks: -47,6 +47,8 @@ struct Prefill0Params {; -509,13 +511,19 @@ inline PrefillPlan plan_compress_prefill(
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +8/-0 (8 lines); hunks: -48,6 +48,14 @@ def get_compress_state_ring_size(; symbols: get_compress_state_ring_size, get_compress_state_write_pad, DeepSeekV4SingleKVPool, __init__
- 关键代码摘录:

```diff
diff -- test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py
@@ -0,0 +1,217 @@
+"""Kernel-level tests for the DSV4 compress write-plan (`plan_prefill`).
+`plan_w` decides which tokens' raw KV get persisted into the compress-state ring
+for a *future* compression window to read. A speculative verify batch plans from
+the optimistic `seq_len = prefix + num_draft_tokens` but rolls back to
+`prefix + accept_len`, so every committed token must stay resident whatever the
+accept length -- i.e. the plan must write all of `[prefix, seq_len)`.
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh
@@ -47,6 +47,8 @@ struct Prefill0Params {
+  /// \brief Trailing tokens the write plan keeps resident in the compress state ring.
+  /// Derived from the ring in `plan_compress_prefill`; see the bound there.
@@ -509,13 +511,19 @@ inline PrefillPlan plan_compress_prefill(
+  // Write pad: trailing tokens kept resident so a verify batch's committed tail survives
+  // any accept length. Zero without speculation -- nothing rolls back, and the ring is
+  // then exactly one window wide. Otherwise the ring bounds it: a write at `w` aliases
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -48,6 +48,14 @@ def get_compress_state_ring_size(
```

- 已读文件:
  - tests: `test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py` added +217/-0
  - runtime: `python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh` modified +12/-4; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +8/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py`, `test/registered/unit/mem_cache/test_dsv4_compress_write_pad.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26671 - [JIT Kernel][DSv4] Optimize epilogue of c128

- 链接: https://github.com/sgl-project/sglang/pull/26671
- 状态/时间: merged / 2026-08-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/jit/csrc/deepseek_v4/c128_v2.cuh`；关联提交 `7331287c1cad`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+96/-97，可读 patch 343 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[JIT Kernel][DSv4] Optimize epilogue of c128」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/kernels/jit/csrc/deepseek_v4/c128_v2.cuh`；技术摘要: 覆盖「[JIT Kernel][DSv4] Optimize epilogue of c128」；主要实现面是 `python/sglang/kernels/jit/csrc/deepseek_v4/c128_v2.cuh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/kernels/jit/csrc/deepseek_v4/c128_v2.cuh` modified +96/-97 (193 lines); hunks: -37,17 +37,37 @@ using PlanD = device::compress::DecodePlan;; -69,28 +89,6 @@ struct Compress128PrefillParams {。
- 代码 diff 细节:
  - `python/sglang/kernels/jit/csrc/deepseek_v4/c128_v2.cuh` modified +96/-97 (193 lines); hunks: -37,17 +37,37 @@ using PlanD = device::compress::DecodePlan;; -69,28 +89,6 @@ struct Compress128PrefillParams {
- 关键代码摘录:

```diff
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/c128_v2.cuh
@@ -37,17 +37,37 @@ using PlanD = device::compress::DecodePlan;
-/// \brief Each thread will handle this many elements (split along head_dim)
-constexpr int32_t kTileElements = 2;
-/// \brief Each warp will handle this many elements (split along 128)
-constexpr int32_t kElementsPerWarp = 8;
-constexpr uint32_t kNumWarps = 128 / kElementsPerWarp;
-constexpr uint32_t kBlockSize = device::kWarpThreads * kNumWarps;
```

- 已读文件:
  - runtime: `python/sglang/kernels/jit/csrc/deepseek_v4/c128_v2.cuh` modified +96/-97
- 验证与风险: runtime 路径改动集中在 `python/sglang/kernels/jit/csrc/deepseek_v4/c128_v2.cuh`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33312 - Fix DSV4 DSpark shared expert loading

- 链接: https://github.com/sgl-project/sglang/pull/33312
- 状态/时间: merged / 2026-08-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4_dspark.py`, `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py`；关联提交 `8c5d5f75bf8b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+122/-5，可读 patch 212 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix DSV4 DSpark shared expert loading」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`；技术摘要: 覆盖「Fix DSV4 DSpark shared expert loading」；主要实现面是 `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py` modified +104/-3 (107 lines); hunks: -1,18 +1,25; -31,13 +38,24 @@ def _publish(self, enforce):; symbols: TestDeepseekV4SharedExpertFusionPolicy, before, _publish, _install，涉及 `TestDeepseekV4SharedExpertFusionPolicy, before, _publish`；`python/sglang/srt/models/deepseek_v4_dspark.py` modified +17/-2 (19 lines); hunks: -20,6 +20,7; -32,6 +33,7; symbols: _run_ffn, DeepseekV4ForCausalLMDSpark, shared_experts_fusion_disable_reason, __init__，涉及 `_run_ffn, DeepseekV4ForCausalLMDSpark, shared_experts_fusion_disable_reason`。
- 代码 diff 细节:
  - `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py` modified +104/-3 (107 lines); hunks: -1,18 +1,25; -31,13 +38,24 @@ def _publish(self, enforce):; symbols: TestDeepseekV4SharedExpertFusionPolicy, before, _publish, _install
  - `python/sglang/srt/models/deepseek_v4_dspark.py` modified +17/-2 (19 lines); hunks: -20,6 +20,7; -32,6 +33,7; symbols: _run_ffn, DeepseekV4ForCausalLMDSpark, shared_experts_fusion_disable_reason, __init__
- 关键代码摘录:

```diff
diff -- test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py
@@ -1,18 +1,25 @@
+from unittest.mock import patch
+import torch
+from torch import nn
+from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
+from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
+from sglang.test.test_utils import CustomTestCase
diff -- python/sglang/srt/models/deepseek_v4_dspark.py
@@ -20,6 +20,7 @@
+from sglang.srt.layers.moe.utils import is_shared_experts_fusion_disabled
@@ -32,6 +33,7 @@
+    DeepseekV4ForCausalLM,
@@ -571,6 +573,12 @@ def _run_ffn(self, x: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor
+    @classmethod
+    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
```

- 已读文件:
  - tests: `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py` modified +104/-3
  - runtime: `python/sglang/srt/models/deepseek_v4_dspark.py` modified +17/-2
- 验证与风险: diff 自带测试面 `test/registered/unit/model_loader/test_runai_model_streamer_loader.py`, `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31700 - Fix DeepSeek-V4/DeepSeek-V4-Pro DP-attention gather semantics

- 链接: https://github.com/sgl-project/sglang/pull/31700
- 状态/时间: merged / 2026-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`；关联提交 `7c7326ccb328`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-4，可读 patch 46 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix DeepSeek-V4/DeepSeek-V4-Pro DP-attention gather semantics」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`；技术摘要: 覆盖「Fix DeepSeek-V4/DeepSeek-V4-Pro DP-attention gather semantics」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +7/-2 (9 lines); hunks: -1854,7 +1854,9 @@ def _run_moe_ffn_dp_sync(; -2372,7 +2374,10 @@ def forward(; symbols: _run_moe_ffn_dp_sync, forward，涉及 `_run_moe_ffn_dp_sync, forward`；`python/sglang/srt/models/deepseek_v4_nextn.py` modified +7/-2 (9 lines); hunks: -14,7 +14,7; -162,7 +162,12 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +7/-2 (9 lines); hunks: -1854,7 +1854,9 @@ def _run_moe_ffn_dp_sync(; -2372,7 +2374,10 @@ def forward(; symbols: _run_moe_ffn_dp_sync, forward
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +7/-2 (9 lines); hunks: -14,7 +14,7; -162,7 +162,12 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -1854,7 +1854,9 @@ def _run_moe_ffn_dp_sync(
-            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
+            # self_attn has already reduced across attention TP, so these hidden
+            # states are replicated and must not be summed by a partial gather.
+            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
@@ -2372,7 +2374,10 @@ def forward(
-            dp_gather_replicate(input_ids_global, input_ids[:, None], forward_batch)
diff -- python/sglang/srt/models/deepseek_v4_nextn.py
@@ -14,7 +14,7 @@
-    dp_gather_partial,
+    dp_gather_replicate,
@@ -162,7 +162,12 @@ def forward(
-            dp_gather_partial(input_ids_global, input_ids[:, None], forward_batch)
+            # Token IDs are replicated within an attention-TP group. Use replicate
+            # gather to avoid summing duplicated IDs when attention_tp_size > 1.
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +7/-2; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +7/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33662 - [DSV4] Avoid host syncs in EAGLE prefill

- 链接: https://github.com/sgl-project/sglang/pull/33662
- 状态/时间: merged / 2026-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；关联提交 `2d5009d130d3`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+17/-3，可读 patch 76 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Avoid host syncs in EAGLE prefill」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；技术摘要: 覆盖「[DSV4] Avoid host syncs in EAGLE prefill」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +9/-0 (9 lines); hunks: -1789,6 +1789,14 @@ def _forward_prefill_sparse(; -1801,6 +1809,7 @@ def _forward_prefill_sparse(; symbols: _forward_prefill_sparse，涉及 `_forward_prefill_sparse`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +9/-0 (9 lines); hunks: -1789,6 +1789,14 @@ def _forward_prefill_sparse(; -1801,6 +1809,7 @@ def _forward_prefill_sparse(; symbols: _forward_prefill_sparse
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -1789,6 +1789,14 @@ def _forward_prefill_sparse(
+            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
+            assert extend_seq_lens_cpu is not None
+            total_swa = sum(
+                min(int(seq_len), int(extend_len) + SWA_WINDOW - 1)
+                for seq_len, extend_len in zip(
+                    seq_lens_cpu.tolist(), extend_seq_lens_cpu, strict=True
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +9/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`, `python/sglang/srt/speculative/eagle_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34333 - docs: remove DSV4 low-latency chunked prefill size

- 链接: https://github.com/sgl-project/sglang/pull/34333
- 状态/时间: merged / 2026-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；关联提交 `a92bbf2f247f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-11，可读 patch 88 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: remove DSV4 low-latency chunked prefill size」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；技术摘要: 覆盖「docs: remove DSV4 low-latency chunked prefill size」；主要实现面是 `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +0/-11 (11 lines); hunks: -450,7 +450,6 @@ sgl-eval run aime25 \\; -471,7 +470,6 @@ sgl-eval run aime25 \\。
- 代码 diff 细节:
  - `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +0/-11 (11 lines); hunks: -450,7 +450,6 @@ sgl-eval run aime25 \\; -471,7 +470,6 @@ sgl-eval run aime25 \\
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -450,7 +450,6 @@ sgl-eval run aime25 \\
-        "--chunked-prefill-size 4096",
@@ -471,7 +470,6 @@ sgl-eval run aime25 \\
-        "--chunked-prefill-size 4096",
@@ -646,7 +644,6 @@ sgl-eval run aime25 \\
-        "--chunked-prefill-size 4096",
@@ -830,7 +827,6 @@ sgl-eval run aime25 \\
```

- 已读文件:
  - docs: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +0/-11
- 验证与风险: 该 PR 主要落在文档/示例 `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #33865 - Fix DSpark + DeepSeek V4 prefill CP compatibility

- 链接: https://github.com/sgl-project/sglang/pull/33865
- 状态/时间: merged / 2026-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`；关联提交 `9d4be40124cc`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+91/-21，可读 patch 178 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix DSpark + DeepSeek V4 prefill CP compatibility」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v4_dspark.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`；技术摘要: 覆盖「Fix DSpark + DeepSeek V4 prefill CP compatibility」；主要实现面是 `python/sglang/srt/models/deepseek_v4_dspark.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4_dspark.py` modified +18/-9 (27 lines); hunks: -319,6 +319,7 @@ def __init__(self, *, vocab_size: int, markov_rank: int) ->...; -338,16 +339,23 @@ def configure_tp_shard(self, *, lm_head: nn.Module) -> None:; symbols: __init__, configure_tp_shard, _apply_step_logits_sharded，涉及 `__init__, configure_tp_shard, _apply_step_logits_sharded`；`python/sglang/srt/models/deepseek_v4.py` modified +10/-7 (17 lines); hunks: -2492,12 +2492,6 @@ def forward(; -2548,12 +2542,21 @@ def forward(; symbols: forward，涉及 `forward`；`test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +44/-0 (44 lines); hunks: -198,5 +198,49 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4B200_CP_DSpark, setUpClass，涉及 `tearDownClass, TestDSV4FlashFP4B200_CP_DSpark, setUpClass`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4_dspark.py` modified +18/-9 (27 lines); hunks: -319,6 +319,7 @@ def __init__(self, *, vocab_size: int, markov_rank: int) ->...; -338,16 +339,23 @@ def configure_tp_shard(self, *, lm_head: nn.Module) -> None:; symbols: __init__, configure_tp_shard, _apply_step_logits_sharded
  - `python/sglang/srt/models/deepseek_v4.py` modified +10/-7 (17 lines); hunks: -2492,12 +2492,6 @@ def forward(; -2548,12 +2542,21 @@ def forward(; symbols: forward
  - `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +44/-0 (44 lines); hunks: -198,5 +198,49 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4B200_CP_DSpark, setUpClass
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4_dspark.py
@@ -319,6 +319,7 @@ def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
+        self._shard_group = None
@@ -338,16 +339,23 @@ def configure_tp_shard(self, *, lm_head: nn.Module) -> None:
-        attn_tp_size = get_parallel().attn_tp_group.world_size
-        if attn_tp_size != tp_size:
+        # Follow lm_head's group choice; attn_tp_group degenerates to size 1
+        # under prefill CP while lm_head still shards over the full TP group.
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -2492,12 +2492,6 @@ def forward(
-        if capture_dspark and use_prefill_cp:
-            raise NotImplementedError(
-                "DSpark aux hidden-state capture is not supported together with "
-                "DeepSeek-V4 prefill context parallelism (attn_cp_size > 1). Disable one "
-                "of them: DSpark static-verify is CP-off for v1."
-            )
diff -- test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py
@@ -198,5 +198,49 @@ def tearDownClass(cls):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4_dspark.py` modified +18/-9; `python/sglang/srt/models/deepseek_v4.py` modified +10/-7
  - tests: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +44/-0
- 验证与风险: diff 自带测试面 `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29070 - [DSV4] perf: Enable alt stream during BCG prefill

- 链接: https://github.com/sgl-project/sglang/pull/29070
- 状态/时间: merged / 2026-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `a0a76e448518`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] perf: Enable alt stream during BCG prefill」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[DSV4] perf: Enable alt stream during BCG prefill」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +4/-1 (5 lines); hunks: -1262,7 +1262,10 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +4/-1 (5 lines); hunks: -1262,7 +1262,10 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -1262,7 +1262,10 @@ def forward(
-            and x.shape[0] <= self._multi_stream_bs_limit
+            and (
+                is_in_breakable_cuda_graph()
+                or x.shape[0] <= self._multi_stream_bs_limit
+            )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34597 - [AMD] Run V4 MTP target-verify through the decode kernel

- 链接: https://github.com/sgl-project/sglang/pull/34597
- 状态/时间: merged / 2026-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`；关联提交 `34206c00175a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+43/-8，可读 patch 127 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Run V4 MTP target-verify through the decode kernel」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`；技术摘要: 覆盖「[AMD] Run V4 MTP target-verify through the decode kernel」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +43/-8 (51 lines); hunks: -105,6 +105,12 @@ class UnifiedKvMetadata:; -126,6 +132,7 @@ def copy_(self, other: UnifiedKvMetadata) -> None:; symbols: UnifiedKvMetadata, copy_, init_forward_metadata_prefill, init_forward_metadata_target_verify_old，涉及 `UnifiedKvMetadata, copy_, init_forward_metadata_prefill`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +43/-8 (51 lines); hunks: -105,6 +105,12 @@ class UnifiedKvMetadata:; -126,6 +132,7 @@ def copy_(self, other: UnifiedKvMetadata) -> None:; symbols: UnifiedKvMetadata, copy_, init_forward_metadata_prefill, init_forward_metadata_target_verify_old
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -105,6 +105,12 @@ class UnifiedKvMetadata:
+    # Per-token req-slot map used by the SWA ring store, precomputed once per
+    # step so the forward store does not recompute a repeat_interleave per layer.
+    # Read by the target-verify store (num_draft*bs tokens); for plain decode it
+    # equals req_pool_indices and is unused (the decode store reads that live).
+    verify_store_state_slot: Optional[torch.Tensor] = None
@@ -126,6 +132,7 @@ def copy_(self, other: UnifiedKvMetadata) -> None:
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +43/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25855 - perf(jit_kernel/deepseek_v4): optimize paged_mqa_metadata

- 链接: https://github.com/sgl-project/sglang/pull/25855
- 状态/时间: merged / 2026-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/jit/csrc/deepseek_v4/paged_mqa_metadata.cuh`；关联提交 `f2b2b567aacc`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+641/-58，可读 patch 769 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「perf(jit_kernel/deepseek_v4): optimize paged_mqa_metadata」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/kernels/jit/csrc/deepseek_v4/paged_mqa_metadata.cuh`；技术摘要: 覆盖「perf(jit_kernel/deepseek_v4): optimize paged_mqa_metadata」；主要实现面是 `python/sglang/kernels/jit/csrc/deepseek_v4/paged_mqa_metadata.cuh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/kernels/jit/csrc/deepseek_v4/paged_mqa_metadata.cuh` modified +297/-57 (354 lines); hunks: -1,93 +1,314; -98,21 +319,40 @@ struct IndexerMetadataKernel {。
- 代码 diff 细节:
  - `python/sglang/kernels/jit/csrc/deepseek_v4/paged_mqa_metadata.cuh` modified +297/-57 (354 lines); hunks: -1,93 +1,314; -98,21 +319,40 @@ struct IndexerMetadataKernel {
- 关键代码摘录:

```diff
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/paged_mqa_metadata.cuh
@@ -1,93 +1,314 @@
+// paged_mqa_metadata: batch-size-adaptive dispatch.
+//
+// Replaces upstream's single-block kernel (grid=1, Phase-3 lane-serial
+// advance, O(bs) dependent loads on the critical path) with three internal
+// kernels dispatched by batch_size, all sharing the same Phase-1/2 prefix
+// sum and a `num_sm + 1`-thread parallel upper_bound for Phase 3.
```

- 已读文件:
  - runtime: `python/sglang/kernels/jit/csrc/deepseek_v4/paged_mqa_metadata.cuh` modified +297/-57
- 验证与风险: diff 自带测试面 `test/registered/kernels/ops/attention/test_paged_mqa_metadata.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34788 - [Fix] Restore layer-level DSV4 RoPE policy

- 链接: https://github.com/sgl-project/sglang/pull/34788
- 状态/时间: merged / 2026-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `test/registered/unit/models/test_deepseek_v4_rope_policy.py`；关联提交 `0a6bbbe128dd`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+163/-8，可读 patch 211 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Restore layer-level DSV4 RoPE policy」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `test/registered/unit/models/test_deepseek_v4_rope_policy.py`, `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[Fix] Restore layer-level DSV4 RoPE policy」；主要实现面是 `test/registered/unit/models/test_deepseek_v4_rope_policy.py`, `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/models/test_deepseek_v4_rope_policy.py` added +145/-0 (145 lines); hunks: -0,0 +1,145; symbols: _ModuleStub, __init__, _RMSNormStub, _RoPEConsumerStub，涉及 `_ModuleStub, __init__, _RMSNormStub`；`python/sglang/srt/models/deepseek_v4.py` modified +18/-8 (26 lines); hunks: -607,15 +607,23 @@ def __init__(; -693,14 +701,16 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `test/registered/unit/models/test_deepseek_v4_rope_policy.py` added +145/-0 (145 lines); hunks: -0,0 +1,145; symbols: _ModuleStub, __init__, _RMSNormStub, _RoPEConsumerStub
  - `python/sglang/srt/models/deepseek_v4.py` modified +18/-8 (26 lines); hunks: -607,15 +607,23 @@ def __init__(; -693,14 +701,16 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- test/registered/unit/models/test_deepseek_v4_rope_policy.py
@@ -0,0 +1,145 @@
+"""Unit tests for DeepSeek-V4 layer-level RoPE selection."""
+import unittest
+from types import SimpleNamespace
+from unittest.mock import patch
+import torch
+from torch import nn
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -607,15 +607,23 @@ def __init__(
-        self.rope_scaling = rope_scaling
-        scaling = rope_scaling or {}
+        self.rope_scaling = dict(rope_scaling) if rope_scaling else None
+        scaling = self.rope_scaling or {}
+        # RoPE is selected at layer granularity in the reference model. Pure
+        # SWA layers use the main unscaled RoPE, while C4/C128 layers use the
```

- 已读文件:
  - tests: `test/registered/unit/models/test_deepseek_v4_rope_policy.py` added +145/-0
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +18/-8
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_deepseek_v4_rope_policy.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34809 - [Cookbook] Add DeepSeek-V4-Pro-0813 (Pro Official) serving recipes

- 链接: https://github.com/sgl-project/sglang/pull/34809
- 状态/时间: merged / 2026-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；关联提交 `463981922ce6`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+511/-9，可读 patch 623 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Cookbook] Add DeepSeek-V4-Pro-0813 (Pro Official) serving recipes」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[Cookbook] Add DeepSeek-V4-Pro-0813 (Pro Official) serving recipes」；主要实现面是 `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +439/-3 (442 lines); hunks: -26,6 +26,7 @@ export const config = {; -51,6 +52,7 @@ export const config = {；`docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +52/-0 (52 lines); hunks: -358,6 +358,58 @@ export const benchmarks = [；`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +20/-6 (26 lines); hunks: -1,6 +1,6; -127,7 +127,7 @@ import { Playground } from "/src/snippets/_playground.jsx";。
- 代码 diff 细节:
  - `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +439/-3 (442 lines); hunks: -26,6 +26,7 @@ export const config = {; -51,6 +52,7 @@ export const config = {
  - `docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +52/-0 (52 lines); hunks: -358,6 +358,58 @@ export const benchmarks = [
  - `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +20/-6 (26 lines); hunks: -1,6 +1,6; -127,7 +127,7 @@ import { Playground } from "/src/snippets/_playground.jsx";
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -26,6 +26,7 @@ export const config = {
+    { id: "pro-official", label: "Pro Official", subtitle: "1.6T · 0813" },
@@ -51,6 +52,7 @@ export const config = {
+    "pro-official|fp4": "deepseek-ai/DeepSeek-V4-Pro-0813",
@@ -83,6 +85,7 @@ export const config = {
+  --random-range-ratio 1.0 \\
@@ -289,11 +292,11 @@ sgl-eval run aime25 \\
diff -- docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx
@@ -358,6 +358,58 @@ export const benchmarks = [
+  // GB300 + FP4 — Pro Official (0813)
+  //
+  // 4xGB300, random 8192/1024 with --random-range-ratio 1.0 (a true fixed
+  // length; the 0.0 default samples uniformly and averages ~5100 in), 64 warmup
+  // requests, cache flushed per point. Each strategy carries its lowest and
+  // highest measured concurrency. TTFT/TPOT are bench_serving means, hence the
diff -- docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -1,6 +1,6 @@
```

- 已读文件:
  - docs: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +439/-3; `docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +52/-0; `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +20/-6
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #34277 - [DSV4] Emit TMA-aligned UE8M0 scales for FP8 einsum

- 链接: https://github.com/sgl-project/sglang/pull/34277
- 状态/时间: merged / 2026-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/jit/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh`；关联提交 `f3225bceb367`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+113/-36，可读 patch 306 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DSV4] Emit TMA-aligned UE8M0 scales for FP8 einsum」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/kernels/jit/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh`；技术摘要: 覆盖「[DSV4] Emit TMA-aligned UE8M0 scales for FP8 einsum」；主要实现面是 `python/sglang/kernels/jit/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/kernels/jit/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh` modified +48/-13 (61 lines); hunks: -2,9 +2,10; -34,6 +35,8 @@ constexpr uint32_t THREADS_PER_GROUP = 8;。
- 代码 diff 细节:
  - `python/sglang/kernels/jit/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh` modified +48/-13 (61 lines); hunks: -2,9 +2,10; -34,6 +35,8 @@ constexpr uint32_t THREADS_PER_GROUP = 8;
- 关键代码摘录:

```diff
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh
@@ -2,9 +2,10 @@
-// contiguous [T, G, D], group_size is fixed to 128, scales are fp32 UE8M0
-// power-of-two values, and output_s is a logical [T, G, D/128] view backed by
-// group-major [G, T, D/128] storage.
+// contiguous [T, G, D], group_size is fixed to 128, and output_s stores four
+// packed UE8M0 exponent bytes per int32. Its physical layout is
+// [G, ceil((D/128)/4), align_up(T, 4)]; the Python wrapper returns the logical
```

- 已读文件:
  - runtime: `python/sglang/kernels/jit/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh` modified +48/-13
- 验证与风险: diff 自带测试面 `test/registered/kernels/ops/attention/test_fp8_wo_a.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33480 - [AMD] Support prefill context parallel two batch overlap for DeepSeek V4

- 链接: https://github.com/sgl-project/sglang/pull/33480
- 状态/时间: merged / 2026-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp_tbo.py`；关联提交 `eb61cb28233b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+594/-19，可读 patch 884 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Support prefill context parallel two batch overlap for DeepSeek V4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp_tbo.py`；技术摘要: 覆盖「[AMD] Support prefill context parallel two batch overlap for DeepSeek V4」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp_tbo.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +235/-15 (250 lines); hunks: -66,6 +66,8; -96,6 +98,8; symbols: _forward_prepare, forward，涉及 `_forward_prepare, forward`；`test/registered/amd/test_deepseek_v4_pro_fp4_cp_tbo.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: TestDeepseekV4ProFp4CPInterleaveTbo, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV4ProFp4CPInterleaveTbo, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +235/-15 (250 lines); hunks: -66,6 +66,8; -96,6 +98,8; symbols: _forward_prepare, forward
  - `test/registered/amd/test_deepseek_v4_pro_fp4_cp_tbo.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: TestDeepseekV4ProFp4CPInterleaveTbo, setUpClass, tearDownClass, test_a_gsm8k
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -66,6 +66,8 @@
+    attn_cp_overlap_all_gather_into_tensor,
+    attn_cp_overlap_reduce_scatter_tensor,
@@ -96,6 +98,8 @@
+    cp_all_gather_rerange_finish,
+    cp_all_gather_rerange_launch,
@@ -1066,6 +1070,15 @@ def _forward_prepare(
diff -- test/registered/amd/test_deepseek_v4_pro_fp4_cp_tbo.py
@@ -0,0 +1,155 @@
+"""MI35x DeepSeek-V4-Pro FP4 prefill context-parallel (CP) + two-batch-overlap (TBO)
+accuracy test (8-GPU).
+Same launch conventions as test_deepseek_v4_pro_fp4_cp.py (prefill CP over the
+unified_kv backend via ``--enable-prefill-cp --cp-strategy interleave``), plus
+``--enable-two-batch-overlap``. This exercises the CP TBO op strategy
+(``op_cp_gather`` / ``op_cp_moe`` / ``op_cp_combine`` driven by
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +235/-15
  - tests: `test/registered/amd/test_deepseek_v4_pro_fp4_cp_tbo.py` added +155/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_pro_fp4_cp_tbo.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33676 - [NPU] Support DeepSeek-V4 DSpark and refactor DSV4 cache management

- 链接: https://github.com/sgl-project/sglang/pull/33676
- 状态/时间: merged / 2026-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`；关联提交 `b83d507cd711`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 37 个文件，+2017/-2025，可读 patch 5881 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Support DeepSeek-V4 DSpark and refactor DSV4 cache management」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek_v4_dspark.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`；技术摘要: 覆盖「[NPU] Support DeepSeek-V4 DSpark and refactor DSV4 cache management」；主要实现面是 `python/sglang/srt/models/deepseek_v4_dspark.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4_dspark.py` modified +171/-12 (183 lines); hunks: -18,12 +18,16; -53,7 +57,7; symbols: apply_rotary_emb, __init__, kv_proj_only, _compute_q，涉及 `apply_rotary_emb, __init__, kv_proj_only`；`python/sglang/srt/models/deepseek_v4.py` modified +123/-3 (126 lines); hunks: -730,7 +730,10 @@ def __init__(; -953,6 +956,108 @@ def _forward_prepare_multi_stream(; symbols: __init__, _forward_prepare_multi_stream, _forward_prepare_multi_stream_npu, _forward_prepare_multi_stream_hip，涉及 `__init__, _forward_prepare_multi_stream, _forward_prepare_multi_stream_npu`；`python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +13/-11 (24 lines); hunks: -8,11 +8,10; -95,11 +94,14 @@ def __init__(; symbols: _lcm, __init__, _alloc_kv_score_buffer，涉及 `_lcm, __init__, _alloc_kv_score_buffer`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4_dspark.py` modified +171/-12 (183 lines); hunks: -18,12 +18,16; -53,7 +57,7; symbols: apply_rotary_emb, __init__, kv_proj_only, _compute_q
  - `python/sglang/srt/models/deepseek_v4.py` modified +123/-3 (126 lines); hunks: -730,7 +730,10 @@ def __init__(; -953,6 +956,108 @@ def _forward_prepare_multi_stream(; symbols: __init__, _forward_prepare_multi_stream, _forward_prepare_multi_stream_npu, _forward_prepare_multi_stream_hip
  - `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +13/-11 (24 lines); hunks: -8,11 +8,10; -95,11 +94,14 @@ def __init__(; symbols: _lcm, __init__, _alloc_kv_score_buffer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4_dspark.py
@@ -18,12 +18,16 @@
+from sglang.srt.layers.dp_attention import is_dp_attention_enabled
-from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
+from sglang.srt.layers.vocab_parallel_embedding import (
+    ParallelLMHead,
+    VocabParallelEmbedding,
+)
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -730,7 +730,10 @@ def __init__(
-        if envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() and alt_streams is not None:
+        if alt_streams is not None and (
+            (_is_cuda and envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get())
+            or (_is_npu and envs.SGLANG_NPU_USE_MULTI_STREAM.get())
+        ):
@@ -953,6 +956,108 @@ def _forward_prepare_multi_stream(
diff -- python/sglang/srt/mem_cache/deepseek_v4_compress_state.py
@@ -8,11 +8,10 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4_dspark.py` modified +171/-12; `python/sglang/srt/models/deepseek_v4.py` modified +123/-3; `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +13/-11
- 验证与风险: diff 自带测试面 `test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py`, `test/registered/unit/spec/test_decode_bookkeeping_ownership.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34926 - Clean deprecated DeepSeek V4 Environs

- 链接: https://github.com/sgl-project/sglang/pull/34926
- 状态/时间: merged / 2026-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` 等 8 个文件；关联提交 `bc312d185dc1`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 28 个文件，+189/-643，可读 patch 1408 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Clean deprecated DeepSeek V4 Environs」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「Clean deprecated DeepSeek V4 Environs」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +38/-144 (182 lines); hunks: -25,9 +25,6; -68,7 +65,6; symbols: __init__, init_forward_metadata_decode, init_forward_metadata_prefill, init_forward_metadata_target_verify，涉及 `__init__, init_forward_metadata_decode, init_forward_metadata_prefill`；`python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +8/-49 (57 lines); hunks: -21,9 +21,6; -505,39 +502,10 @@ def init_forward_metadata_decode(; symbols: init_forward_metadata_decode, init_forward_metadata_prefill, init_forward_metadata_target_verify, store_cache，涉及 `init_forward_metadata_decode, init_forward_metadata_prefill, init_forward_metadata_target_verify`；`python/sglang/srt/models/deepseek_v4.py` modified +3/-7 (10 lines); hunks: -1779,7 +1779,7 @@ def hc_pre_torch_impl(x, hc_fn):; -1860,7 +1860,7 @@ def hc_post(; symbols: hc_pre_torch_impl, hc_post, _run_moe_ffn_dp_sync, _prewarm_mhc_kernels，涉及 `hc_pre_torch_impl, hc_post, _run_moe_ffn_dp_sync`；`python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` modified +0/-3 (3 lines); hunks: -4,8 +4,6; -104,7 +102,6 @@ def try_fused_hc_post_pre(; symbols: try_fused_hc_post_pre，涉及 `try_fused_hc_post_pre`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +38/-144 (182 lines); hunks: -25,9 +25,6; -68,7 +65,6; symbols: __init__, init_forward_metadata_decode, init_forward_metadata_prefill, init_forward_metadata_target_verify
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +8/-49 (57 lines); hunks: -21,9 +21,6; -505,39 +502,10 @@ def init_forward_metadata_decode(; symbols: init_forward_metadata_decode, init_forward_metadata_prefill, init_forward_metadata_target_verify, store_cache
  - `python/sglang/srt/models/deepseek_v4.py` modified +3/-7 (10 lines); hunks: -1779,7 +1779,7 @@ def hc_pre_torch_impl(x, hc_fn):; -1860,7 +1860,7 @@ def hc_post(; symbols: hc_pre_torch_impl, hc_post, _run_moe_ffn_dp_sync, _prewarm_mhc_kernels
  - `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` modified +0/-3 (3 lines); hunks: -4,8 +4,6; -104,7 +102,6 @@ def try_fused_hc_post_pre(; symbols: try_fused_hc_post_pre
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +1/-4 (5 lines); hunks: -15,10 +15,7 @@ def validate_deepseek_v4_mega_moe_token_budget(; symbols: validate_deepseek_v4_mega_moe_token_budget
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -25,9 +25,6 @@
-from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
-    quant_to_nope_fp8_rope_bf16_pack_triton,
-)
@@ -68,7 +65,6 @@
-    compute_ragged_extend_lengths,
@@ -586,9 +582,7 @@ def __init__(
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -21,9 +21,6 @@
-from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
-    quant_to_nope_fp8_rope_bf16_pack_triton,
-)
@@ -505,39 +502,10 @@ def init_forward_metadata_decode(
-        if envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
-            return DSV4RawDecodeMetadata(
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -1779,7 +1779,7 @@ def hc_pre_torch_impl(x, hc_fn):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +38/-144; `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +8/-49; `python/sglang/srt/models/deepseek_v4.py` modified +3/-7; `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` modified +0/-3; `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +1/-4
  - docs: `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-3
  - tests: `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py` modified +0/-3; `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py` modified +0/-2
- 验证与风险: diff 自带测试面 `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py`, `test/registered/dcp/test_kimi_linear_dcp_dspark4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py`, `test/registered/unit/layers/test_dsv4_nonpaged_indexer.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35224 - [Docs] Enable PD disaggregation for DSV4 low-latency recipes

- 链接: https://github.com/sgl-project/sglang/pull/35224
- 状态/时间: merged / 2026-08-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；关联提交 `53621818e414`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+36/-6，可读 patch 88 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Enable PD disaggregation for DSV4 low-latency recipes」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；技术摘要: 覆盖「[Docs] Enable PD disaggregation for DSV4 low-latency recipes」；主要实现面是 `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +0/-1 (1 lines); hunks: -319,7 +319,6 @@ sgl-eval run aime25 \\；`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -602,4 +602,4 @@ Larger blocks can improve decode latency when acceptance sta...。
- 代码 diff 细节:
  - `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +0/-1 (1 lines); hunks: -319,7 +319,6 @@ sgl-eval run aime25 \\
  - `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -602,4 +602,4 @@ Larger blocks can improve decode latency when acceptance sta...
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -319,7 +319,6 @@ sgl-eval run aime25 \\
-      showWhen: (base) => base.specAlgorithm !== "DSPARK",
diff -- docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -602,4 +602,4 @@ Larger blocks can improve decode latency when acceptance stays high, but they al
-DSpark currently requires CUDA, `pp_size == 1`, and DP Attention disabled. It is not compatible with PD disaggregation on current SGLang releases; turn DSpark off before selecting
+DSpark currently requires CUDA, `pp_size == 1`, and DP Attention disabled. It is not compatible with PD disaggregation on current SGLang releases; selecting a prefill or decode ro
```

- 已读文件:
  - docs: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +0/-1; `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs/scripts/check_cookbook_configs.mjs`, `docs/src/snippets/_playground.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #34890 - [Perf] Hoist DSv4 draft-extend SWA write locs; unify SWA graph buffer naming

- 链接: https://github.com/sgl-project/sglang/pull/34890
- 状态/时间: merged / 2026-08-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；关联提交 `0111b290312a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+68/-30，可读 patch 222 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf] Hoist DSv4 draft-extend SWA write locs; unify SWA graph buffer naming」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；技术摘要: 覆盖「[Perf] Hoist DSv4 draft-extend SWA write locs; unify SWA graph buffer naming」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +45/-9 (54 lines); hunks: -588,6 +588,7 @@ def __init__(; -1000,6 +1001,13 @@ def init_forward_metadata_draft_extend(; symbols: __init__, _move_to_device, init_forward_metadata_draft_extend, _fill_cuda_graph_swa_out_cache_loc，涉及 `__init__, _move_to_device, init_forward_metadata_draft_extend`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +45/-9 (54 lines); hunks: -588,6 +588,7 @@ def __init__(; -1000,6 +1001,13 @@ def init_forward_metadata_draft_extend(; symbols: __init__, _move_to_device, init_forward_metadata_draft_extend, _fill_cuda_graph_swa_out_cache_loc
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -588,6 +588,7 @@ def __init__(
+        self.cuda_graph_swa_out_cache_loc: Optional[torch.Tensor] = None
@@ -1000,6 +1001,13 @@ def init_forward_metadata_draft_extend(
+        swa_out_cache_loc = self._fill_cuda_graph_swa_out_cache_loc(out_cache_loc)
+        if swa_out_cache_loc is None and out_cache_loc is not None:
+            # Eager-only miss (no graph state / oversized batch): translate once
+            # per step instead of per layer at store time.
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +45/-9
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35162 - Add deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms

- 链接: https://github.com/sgl-project/sglang/pull/35162
- 状态/时间: merged / 2026-08-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms.py`；关联提交 `7605529bdf98`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+142/-0，可读 patch 143 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms.py`；技术摘要: 覆盖「Add deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms」；主要实现面是 `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: TestNPUDeepSeekV4FlashW8A88PIn32kOut1k50ms, test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms，涉及 `TestNPUDeepSeekV4FlashW8A88PIn32kOut1k50ms, test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms`。
- 代码 diff 细节:
  - `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: TestNPUDeepSeekV4FlashW8A88PIn32kOut1k50ms, test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms
- 关键代码摘录:

```diff
diff -- test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms.py
@@ -0,0 +1,142 @@
+import unittest
+from sglang.test.ascend.e2e.test_npu_performance_utils import (
+    AISBENCHMARK_DATASET_DEFAULT,
+    BENCHMARK_TOOL_DEFAULT,
+    DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH,
+    TestNpuPerformanceTestCaseBase,
```

- 已读文件:
  - tests: `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms.py` added +142/-0
- 验证与风险: diff 自带测试面 `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33313 - [AMD] DeepSeek-V4: route decode wo_a bf16 batched matmul to aiter batched_gemm_bf16

- 链接: https://github.com/sgl-project/sglang/pull/33313
- 状态/时间: merged / 2026-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`, `test/registered/unit/models/test_deepseek_v4_amd_wo_a_bf16.py`；关联提交 `f446e853e73f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+293/-1，可读 patch 316 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] DeepSeek-V4: route decode wo_a bf16 batched matmul to aiter batched_gemm_bf16」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/unit/models/test_deepseek_v4_amd_wo_a_bf16.py`, `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[AMD] DeepSeek-V4: route decode wo_a bf16 batched matmul to aiter batched_gemm_bf16」；主要实现面是 `test/registered/unit/models/test_deepseek_v4_amd_wo_a_bf16.py`, `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/models/test_deepseek_v4_amd_wo_a_bf16.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: TestWoABf16BatchedGemm, setUpClass, setUp, _rand，涉及 `TestWoABf16BatchedGemm, setUpClass, setUp`；`python/sglang/srt/models/deepseek_v4.py` modified +86/-1 (87 lines); hunks: -310,6 +310,89 @@ def _flashinfer_hc_pre(; -1581,7 +1664,9 @@ def forward(; symbols: _flashinfer_hc_pre, _wo_a_aiter_gemm_eligible, _apply_wo_a_bf16_matmul, _fused_rmsnorm_fp8_quant，涉及 `_flashinfer_hc_pre, _wo_a_aiter_gemm_eligible, _apply_wo_a_bf16_matmul`。
- 代码 diff 细节:
  - `test/registered/unit/models/test_deepseek_v4_amd_wo_a_bf16.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: TestWoABf16BatchedGemm, setUpClass, setUp, _rand
  - `python/sglang/srt/models/deepseek_v4.py` modified +86/-1 (87 lines); hunks: -310,6 +310,89 @@ def _flashinfer_hc_pre(; -1581,7 +1664,9 @@ def forward(; symbols: _flashinfer_hc_pre, _wo_a_aiter_gemm_eligible, _apply_wo_a_bf16_matmul, _fused_rmsnorm_fp8_quant
- 关键代码摘录:

```diff
diff -- test/registered/unit/models/test_deepseek_v4_amd_wo_a_bf16.py
@@ -0,0 +1,203 @@
+"""Tests for the DeepSeek-V4 decode ``wo_a`` bf16 batched-matmul routing.
+Covers ``deepseek_v4._apply_wo_a_bf16_matmul``, which (opt-in via
+``SGLANG_OPT_USE_AITER_BATCHED_GEMM`` on HIP/gfx95) routes the MLA output-absorb
+bf16 GEMM off the rocBLAS/Tensile ``Cijk_*`` batched GEMM onto aiter's tuned
+``batched_gemm_bf16``, with an einsum fallback.
+The opt-in flag and the aiter kernel are resolved ONCE at module import
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -310,6 +310,89 @@ def _flashinfer_hc_pre(
+def _wo_a_aiter_gemm_eligible(
+    flag: bool, use_aiter: bool, is_hip: bool, is_gfx95: bool
+) -> bool:
+    """Static eligibility for the aiter ``wo_a`` reroute.
+    Folds the opt-in flag, the global ``SGLANG_USE_AITER`` switch, and the
+    HIP/gfx95 platform gates into one predicate. Evaluated once at import (see
```

- 已读文件:
  - tests: `test/registered/unit/models/test_deepseek_v4_amd_wo_a_bf16.py` added +203/-0
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +86/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_deepseek_v4_amd_wo_a_bf16.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32327 - [DeepSeek-V4] Add Q8KV8 sparse MLA prefill runtime backend

- 链接: https://github.com/sgl-project/sglang/pull/32327
- 状态/时间: merged / 2026-08-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；关联提交 `9db4ba8da166`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+1337/-13，可读 patch 1499 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek-V4] Add Q8KV8 sparse MLA prefill runtime backend」；模型线: DeepSeek V4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`；技术摘要: 覆盖「[DeepSeek-V4] Add Q8KV8 sparse MLA prefill runtime backend」；主要实现面是 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +246/-2 (248 lines); hunks: -19,7 +19,11; -56,8 +60,12; symbols: __init__, match_num_queries, _forward_prefill_sparse, _prepare_q8kv8_q_and_sink，涉及 `__init__, match_num_queries, _forward_prefill_sparse`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +246/-2 (248 lines); hunks: -19,7 +19,11; -56,8 +60,12; symbols: __init__, match_num_queries, _forward_prefill_sparse, _prepare_q8kv8_q_and_sink
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -19,7 +19,11 @@
+    cast_q_fp8_for_q8kv8_prefill,
+    fp8_dtype,
+    gather_dequant_requant_fp8_paged,
+    q8kv8_padded_num_heads,
@@ -56,8 +60,12 @@
+    use_dsv4_q8kv8_sparse_prefill,
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +246/-2
- 验证与风险: diff 自带测试面 `test/registered/kernels/ops/attention/test_q8kv8_sparse_prefill_backend.py`, `test/registered/unit/server_args/test_server_args.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34973 - [AMD] DSv4: fuse the qk-norm-rope pair on the MTP target-verify path

- 链接: https://github.com/sgl-project/sglang/pull/34973
- 状态/时间: merged / 2026-08-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v4.py`；关联提交 `44c90c628264`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+49/-8，可读 patch 95 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] DSv4: fuse the qk-norm-rope pair on the MTP target-verify path」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[AMD] DSv4: fuse the qk-norm-rope pair on the MTP target-verify path」；主要实现面是 `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v4.py` modified +45/-8 (53 lines); hunks: -1295,11 +1295,19 @@ def _forward_prepare(; -1318,7 +1326,27 @@ def _forward_prepare(; symbols: _forward_prepare, forward，涉及 `_forward_prepare, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v4.py` modified +45/-8 (53 lines); hunks: -1295,11 +1295,19 @@ def _forward_prepare(; -1318,7 +1326,27 @@ def _forward_prepare(; symbols: _forward_prepare, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -1295,11 +1295,19 @@ def _forward_prepare(
-        do_fused_store = (unified and is_decode) or (
+        # The kernel is token-indexed (q, kv and positions are all length M), so
+        # a verify batch carrying several draft tokens per request is a shape it
+        # already handles. Only the cache store differs between decode and
+        # verify, and that half is left off below.
+        fuse_verify = (
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +45/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/environ.py`, `python/sglang/srt/models/deepseek_v4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35919 - [DeepSeek V4] Default FP4 checkpoints to FlashInfer MXFP4 MoE

- 链接: https://github.com/sgl-project/sglang/pull/35919
- 状态/时间: merged / 2026-08-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`；关联提交 `60ff1e33a51f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+106/-74，可读 patch 243 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek V4] Default FP4 checkpoints to FlashInfer MXFP4 MoE」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `python/sglang/srt/arg_groups/overrides.py`, `python/sglang/srt/server_args.py`；技术摘要: 覆盖「[DeepSeek V4] Default FP4 checkpoints to FlashInfer MXFP4 MoE」；主要实现面是 `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `python/sglang/srt/arg_groups/overrides.py`, `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +2/-3 (5 lines); hunks: -1,6 +1,7; -56,8 +57,6 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`；`python/sglang/srt/arg_groups/overrides.py` modified +21/-24 (45 lines); hunks: -1164,16 +1164,27 @@ def _deepseek_v4_overrides(server_args: Any, hf_config:...; -1921,20 +1932,6 @@ def _deepseek_v4_kv_cache_dtype(view: Any) -> dict:; symbols: _deepseek_v4_overrides, _deepseek_v4_kv_cache_dtype, _deepseek_v4_sm120_moe, _muse_glimmer_fp4_gemm_runner_overrides，涉及 `_deepseek_v4_overrides, _deepseek_v4_kv_cache_dtype, _deepseek_v4_sm120_moe`；`python/sglang/srt/server_args.py` modified +0/-9 (9 lines); hunks: -5540,15 +5540,6 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments，涉及 `_handle_model_specific_adjustments`。
- 代码 diff 细节:
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +2/-3 (5 lines); hunks: -1,6 +1,7; -56,8 +57,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `python/sglang/srt/arg_groups/overrides.py` modified +21/-24 (45 lines); hunks: -1164,16 +1164,27 @@ def _deepseek_v4_overrides(server_args: Any, hf_config:...; -1921,20 +1932,6 @@ def _deepseek_v4_kv_cache_dtype(view: Any) -> dict:; symbols: _deepseek_v4_overrides, _deepseek_v4_kv_cache_dtype, _deepseek_v4_sm120_moe, _muse_glimmer_fp4_gemm_runner_overrides
  - `python/sglang/srt/server_args.py` modified +0/-9 (9 lines); hunks: -5540,15 +5540,6 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- 关键代码摘录:

```diff
diff -- test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py
@@ -1,6 +1,7 @@
-Launches TP=4 with flashinfer_mxfp4 MoE runner + EAGLE speculative decoding.
+Launches TP=4 with the auto-selected flashinfer_mxfp4 MoE runner and EAGLE
+speculative decoding.
@@ -56,8 +57,6 @@ def setUpClass(cls):
-                "--moe-runner-backend",
-                "flashinfer_mxfp4",
diff -- python/sglang/srt/arg_groups/overrides.py
@@ -1164,16 +1164,27 @@ def _deepseek_v4_overrides(server_args: Any, hf_config: Any) -> dict:
-    # nvidia/DeepSeek-V4-Pro-NVFP4 uses flashinfer_trtllm_routed MoE runner backend.
-    if (
-        server_args.moe_runner_backend == "auto"
-        and server_args.get_model_config().nvfp4_moe_meta is not None
-    ):
-        overrides["moe_runner_backend"] = "flashinfer_trtllm_routed"
diff -- python/sglang/srt/server_args.py
@@ -5540,15 +5540,6 @@ def _handle_model_specific_adjustments(self):
```

- 已读文件:
  - tests: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +2/-3
  - runtime: `python/sglang/srt/arg_groups/overrides.py` modified +21/-24; `python/sglang/srt/server_args.py` modified +0/-9
- 验证与风险: diff 自带测试面 `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/unit/test_model_overrides.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35918 - [DeepSeek V4] Add W4A4 MegaMoE server flag

- 链接: https://github.com/sgl-project/sglang/pull/35918
- 状态/时间: merged / 2026-08-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`；关联提交 `3b5909de0ef5`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+147/-87，可读 patch 430 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek V4] Add W4A4 MegaMoE server flag」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`；技术摘要: 覆盖「[DeepSeek V4] Add W4A4 MegaMoE server flag」；主要实现面是 `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +2/-5 (7 lines); hunks: -258,11 +258,8 @@ sgl-eval run aime25 \\；`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-5 (9 lines); hunks: -332,14 +332,13 @@ HiCache and MegaMoE are **not** supported on RTX PRO 6000.；`test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +1/-2 (3 lines); hunks: -41,8 +41,6; -126,6 +124,7 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`；`test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +1/-2 (3 lines); hunks: -34,8 +34,6; -115,6 +113,7 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`。
- 代码 diff 细节:
  - `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +2/-5 (7 lines); hunks: -258,11 +258,8 @@ sgl-eval run aime25 \\
  - `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-5 (9 lines); hunks: -332,14 +332,13 @@ HiCache and MegaMoE are **not** supported on RTX PRO 6000.
  - `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +1/-2 (3 lines); hunks: -41,8 +41,6; -126,6 +124,7 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +1/-2 (3 lines); hunks: -34,8 +34,6; -115,6 +113,7 @@ def setUpClass(cls):; symbols: setUpClass
  - `python/sglang/srt/layers/moe/mega_moe.py` modified +1/-23 (24 lines); hunks: -42,26 +42,6; -74,8 +54,6 @@ def _get_mega_moe_symm_buffer(; symbols: _apply_mega_moe_dg_env, _get_mega_moe_symm_buffer, _run_mega_routed
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -258,11 +258,8 @@ sgl-eval run aime25 \\
-            env: [
-              "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
-              "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1",
-              "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1",
-            ] },
+            flags: ["--enable-w4a4-mxfp4-megamoe"],
diff -- docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -332,14 +332,13 @@ HiCache and MegaMoE are **not** supported on RTX PRO 6000.
-`--moe-a2a-backend megamoe` and add the relevant env vars automatically.
+`--moe-a2a-backend megamoe` and add the relevant launch settings automatically.
-- **W4A4** — adds `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1` and
-  `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1` to run the custom W4A4
-  kernel (FP4 activations). Higher throughput with negligible accuracy drop
-  (~89.5 GPQA on Pro).
diff -- test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py
@@ -41,8 +41,6 @@
```

- 已读文件:
  - docs: `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +2/-5; `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-5
  - tests: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +1/-2; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +1/-2
  - runtime: `python/sglang/srt/layers/moe/mega_moe.py` modified +1/-23; `python/sglang/srt/arg_groups/mega_moe_hook.py` added +38/-0; `python/sglang/srt/server_args.py` modified +10/-15; `python/sglang/srt/environ.py` modified +6/-10
- 验证与风险: diff 自带测试面 `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`, `test/registered/unit/server_args/test_server_args.py`, `test/registered/unit/test_environ.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32577 - [AMD] DeepSeek-V4: add aiter fused mHC post+pre with cross-layer boundary dispatch

- 链接: https://github.com/sgl-project/sglang/pull/32577
- 状态/时间: merged / 2026-08-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/unit/models/test_deepseek_v4_amd_fused_mhc.py`；关联提交 `d315eb725044`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+803/-130，可读 patch 1038 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] DeepSeek-V4: add aiter fused mHC post+pre with cross-layer boundary dispatch」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `test/registered/unit/models/test_deepseek_v4_amd_fused_mhc.py`, `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py`, `python/sglang/srt/models/deepseek_v4.py`；技术摘要: 覆盖「[AMD] DeepSeek-V4: add aiter fused mHC post+pre with cross-layer boundary dispatch」；主要实现面是 `test/registered/unit/models/test_deepseek_v4_amd_fused_mhc.py`, `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py`, `python/sglang/srt/models/deepseek_v4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/models/test_deepseek_v4_amd_fused_mhc.py` added +400/-0 (400 lines); hunks: -0,0 +1,400; symbols: TestAmdFusedMhcCrossLayerGating, test_tilelang_fuse_flag_enables_cross_layer_fusion, test_sm120_enables_fusion_with_tilelang_pre_disabled, test_no_sm120_still_requires_tilelang_pre，涉及 `TestAmdFusedMhcCrossLayerGating, test_tilelang_fuse_flag_enables_cross_layer_fusion, test_sm120_enables_fusion_with_tilelang_pre_disabled`；`python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` modified +259/-0 (259 lines); hunks: -4,13 +4,55; -153,3 +195,220 @@ def try_fused_hc_post_pre(; symbols: _is_fused_mhc_post_pre_enabled, _is_aiter_gfx95_mhc_available, _is_production_mhc_enabled, is_cross_layer_mhc_fusion_enabled，涉及 `_is_fused_mhc_post_pre_enabled, _is_aiter_gfx95_mhc_available, _is_production_mhc_enabled`；`python/sglang/srt/models/deepseek_v4.py` modified +144/-51 (195 lines); hunks: -138,7 +138,8; -168,7 +169,6; symbols: _get_mhc_ops, _is_fused_mhc_post_pre_enabled, __init__, forward，涉及 `_get_mhc_ops, _is_fused_mhc_post_pre_enabled, __init__`。
- 代码 diff 细节:
  - `test/registered/unit/models/test_deepseek_v4_amd_fused_mhc.py` added +400/-0 (400 lines); hunks: -0,0 +1,400; symbols: TestAmdFusedMhcCrossLayerGating, test_tilelang_fuse_flag_enables_cross_layer_fusion, test_sm120_enables_fusion_with_tilelang_pre_disabled, test_no_sm120_still_requires_tilelang_pre
  - `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` modified +259/-0 (259 lines); hunks: -4,13 +4,55; -153,3 +195,220 @@ def try_fused_hc_post_pre(; symbols: _is_fused_mhc_post_pre_enabled, _is_aiter_gfx95_mhc_available, _is_production_mhc_enabled, is_cross_layer_mhc_fusion_enabled
  - `python/sglang/srt/models/deepseek_v4.py` modified +144/-51 (195 lines); hunks: -138,7 +138,8; -168,7 +169,6; symbols: _get_mhc_ops, _is_fused_mhc_post_pre_enabled, __init__, forward
- 关键代码摘录:

```diff
diff -- test/registered/unit/models/test_deepseek_v4_amd_fused_mhc.py
@@ -0,0 +1,400 @@
+import unittest
+from unittest import mock
+from sglang.srt.environ import envs
+from sglang.srt.models.deepseek_common.amd import deepseek_v4_fused_mhc
+from sglang.test.ci.ci_register import register_cpu_ci
+register_cpu_ci(est_time=4, suite="base-a-test-cpu")
diff -- python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py
@@ -4,13 +4,55 @@
+from sglang.srt.environ import envs
+from sglang.srt.utils import get_bool_env_var, is_gfx95_supported, is_hip
+from sglang.srt.utils.common import is_sm120_supported
+_is_hip = is_hip()
+_is_gfx95_supported = is_gfx95_supported()
+# aiter fused mHC state. The kernel self-gates the fused-vs-unfused decision per
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -138,7 +138,8 @@
```

- 已读文件:
  - tests: `test/registered/unit/models/test_deepseek_v4_amd_fused_mhc.py` added +400/-0
  - runtime: `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` modified +259/-0; `python/sglang/srt/models/deepseek_v4.py` modified +144/-51
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_deepseek_v4_amd_fused_mhc.py`, `test/registered/unit/models/test_deepseek_v4_fused_mhc_policy.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
