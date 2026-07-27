# sglang DeepSeek V4 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` | [#23605](https://github.com/sgl-project/sglang/pull/23605), [#23622](https://github.com/sgl-project/sglang/pull/23622), [#23628](https://github.com/sgl-project/sglang/pull/23628), [#23684](https://github.com/sgl-project/sglang/pull/23684), [#23689](https://github.com/sgl-project/sglang/pull/23689), [#23691](https://github.com/sgl-project/sglang/pull/23691), [#23697](https://github.com/sgl-project/sglang/pull/23697), [#23725](https://github.com/sgl-project/sglang/pull/23725), [#23882](https://github.com/sgl-project/sglang/pull/23882), [#23980](https://github.com/sgl-project/sglang/pull/23980), [#24035](https://github.com/sgl-project/sglang/pull/24035), [#25243](https://github.com/sgl-project/sglang/pull/25243), ... (31 total) |
| `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` | [#25820](https://github.com/sgl-project/sglang/pull/25820), [#28423](https://github.com/sgl-project/sglang/pull/28423), [#28590](https://github.com/sgl-project/sglang/pull/28590), [#31363](https://github.com/sgl-project/sglang/pull/31363) |
| `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` | [#25820](https://github.com/sgl-project/sglang/pull/25820), [#28423](https://github.com/sgl-project/sglang/pull/28423), [#28590](https://github.com/sgl-project/sglang/pull/28590), [#28981](https://github.com/sgl-project/sglang/pull/28981), [#30272](https://github.com/sgl-project/sglang/pull/30272), [#30651](https://github.com/sgl-project/sglang/pull/30651), [#31122](https://github.com/sgl-project/sglang/pull/31122), [#31363](https://github.com/sgl-project/sglang/pull/31363), [#31373](https://github.com/sgl-project/sglang/pull/31373), [#31452](https://github.com/sgl-project/sglang/pull/31452) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c128.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c128_online.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c128_online_v2.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c128_v2.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c4.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c4_v2.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/common.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/hash_topk.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/online_c128_mtp.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/paged_mqa_metadata.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/rope.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/store.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v1.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/compress.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/compress_v2.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp8_utils.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kvcacheio.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/kernels/ops/attention/deepseek_v4_rope.py` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#25144](https://github.com/sgl-project/sglang/pull/25144), [#25771](https://github.com/sgl-project/sglang/pull/25771), [#25820](https://github.com/sgl-project/sglang/pull/25820), [#29569](https://github.com/sgl-project/sglang/pull/29569), [#29775](https://github.com/sgl-project/sglang/pull/29775), [#29982](https://github.com/sgl-project/sglang/pull/29982), [#30237](https://github.com/sgl-project/sglang/pull/30237) |
| `python/sglang/srt/configs/deepseek_v4.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882) |
| `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24692](https://github.com/sgl-project/sglang/pull/24692), [#24890](https://github.com/sgl-project/sglang/pull/24890), [#25195](https://github.com/sgl-project/sglang/pull/25195), [#26209](https://github.com/sgl-project/sglang/pull/26209), [#26239](https://github.com/sgl-project/sglang/pull/26239), [#26471](https://github.com/sgl-project/sglang/pull/26471), [#26499](https://github.com/sgl-project/sglang/pull/26499), [#27059](https://github.com/sgl-project/sglang/pull/27059), [#27380](https://github.com/sgl-project/sglang/pull/27380), [#27914](https://github.com/sgl-project/sglang/pull/27914), [#29619](https://github.com/sgl-project/sglang/pull/29619), ... (20 total) |
| `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` | [#24933](https://github.com/sgl-project/sglang/pull/24933), [#26208](https://github.com/sgl-project/sglang/pull/26208), [#26383](https://github.com/sgl-project/sglang/pull/26383), [#26499](https://github.com/sgl-project/sglang/pull/26499), [#27152](https://github.com/sgl-project/sglang/pull/27152), [#27380](https://github.com/sgl-project/sglang/pull/27380), [#27928](https://github.com/sgl-project/sglang/pull/27928), [#28520](https://github.com/sgl-project/sglang/pull/28520), [#28920](https://github.com/sgl-project/sglang/pull/28920), [#29362](https://github.com/sgl-project/sglang/pull/29362), [#29420](https://github.com/sgl-project/sglang/pull/29420), [#29630](https://github.com/sgl-project/sglang/pull/29630), ... (13 total) |
| `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24933](https://github.com/sgl-project/sglang/pull/24933), [#25144](https://github.com/sgl-project/sglang/pull/25144), [#26471](https://github.com/sgl-project/sglang/pull/26471), [#30333](https://github.com/sgl-project/sglang/pull/30333) |
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24691](https://github.com/sgl-project/sglang/pull/24691), [#24704](https://github.com/sgl-project/sglang/pull/24704), [#24890](https://github.com/sgl-project/sglang/pull/24890), [#24933](https://github.com/sgl-project/sglang/pull/24933), [#25144](https://github.com/sgl-project/sglang/pull/25144), [#25884](https://github.com/sgl-project/sglang/pull/25884), [#25889](https://github.com/sgl-project/sglang/pull/25889), [#25898](https://github.com/sgl-project/sglang/pull/25898), [#26208](https://github.com/sgl-project/sglang/pull/26208), [#26209](https://github.com/sgl-project/sglang/pull/26209), [#26471](https://github.com/sgl-project/sglang/pull/26471), ... (22 total) |
| `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` | [#26383](https://github.com/sgl-project/sglang/pull/26383) |
| `python/sglang/srt/models/deepseek_v4.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24704](https://github.com/sgl-project/sglang/pull/24704), [#24890](https://github.com/sgl-project/sglang/pull/24890), [#24933](https://github.com/sgl-project/sglang/pull/24933), [#24947](https://github.com/sgl-project/sglang/pull/24947), [#25144](https://github.com/sgl-project/sglang/pull/25144), [#25195](https://github.com/sgl-project/sglang/pull/25195), [#25391](https://github.com/sgl-project/sglang/pull/25391), [#25396](https://github.com/sgl-project/sglang/pull/25396), [#25729](https://github.com/sgl-project/sglang/pull/25729), [#25733](https://github.com/sgl-project/sglang/pull/25733), [#25763](https://github.com/sgl-project/sglang/pull/25763), ... (47 total) |
| `python/sglang/srt/models/deepseek_v4_dspark.py` | [#27657](https://github.com/sgl-project/sglang/pull/27657), [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/srt/models/deepseek_v4_nextn.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24934](https://github.com/sgl-project/sglang/pull/24934), [#24947](https://github.com/sgl-project/sglang/pull/24947), [#25810](https://github.com/sgl-project/sglang/pull/25810), [#25976](https://github.com/sgl-project/sglang/pull/25976), [#26238](https://github.com/sgl-project/sglang/pull/26238), [#28980](https://github.com/sgl-project/sglang/pull/28980) |
| `python/sglang/test/kernels/deepseek_v4/__init__.py` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `python/sglang/test/kernels/deepseek_v4/common.py` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `test/registered/amd/test_deepseek_v4_flash_fp4.py` | [#24825](https://github.com/sgl-project/sglang/pull/24825), [#25039](https://github.com/sgl-project/sglang/pull/25039), [#26662](https://github.com/sgl-project/sglang/pull/26662), [#27149](https://github.com/sgl-project/sglang/pull/27149), [#28290](https://github.com/sgl-project/sglang/pull/28290), [#28920](https://github.com/sgl-project/sglang/pull/28920) |
| `test/registered/amd/test_deepseek_v4_flash_fp8.py` | [#24825](https://github.com/sgl-project/sglang/pull/24825), [#25039](https://github.com/sgl-project/sglang/pull/25039), [#26662](https://github.com/sgl-project/sglang/pull/26662), [#27149](https://github.com/sgl-project/sglang/pull/27149), [#28290](https://github.com/sgl-project/sglang/pull/28290), [#28920](https://github.com/sgl-project/sglang/pull/28920) |
| `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py` | [#29362](https://github.com/sgl-project/sglang/pull/29362) |
| `test/registered/amd/test_deepseek_v4_pro_fp4.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24203](https://github.com/sgl-project/sglang/pull/24203), [#24825](https://github.com/sgl-project/sglang/pull/24825), [#25039](https://github.com/sgl-project/sglang/pull/25039), [#26662](https://github.com/sgl-project/sglang/pull/26662), [#27149](https://github.com/sgl-project/sglang/pull/27149), [#28290](https://github.com/sgl-project/sglang/pull/28290), [#28920](https://github.com/sgl-project/sglang/pull/28920) |
| `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` | [#27928](https://github.com/sgl-project/sglang/pull/27928), [#28920](https://github.com/sgl-project/sglang/pull/28920) |
| `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` | [#28520](https://github.com/sgl-project/sglang/pull/28520), [#28920](https://github.com/sgl-project/sglang/pull/28920) |
| `test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py` | [#29362](https://github.com/sgl-project/sglang/pull/29362) |
| `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py` | [#30238](https://github.com/sgl-project/sglang/pull/30238) |
| `test/registered/amd/test_deepseek_v4_pro_fp8.py` | [#23882](https://github.com/sgl-project/sglang/pull/23882), [#24203](https://github.com/sgl-project/sglang/pull/24203), [#24825](https://github.com/sgl-project/sglang/pull/24825), [#25039](https://github.com/sgl-project/sglang/pull/25039), [#26662](https://github.com/sgl-project/sglang/pull/26662), [#27149](https://github.com/sgl-project/sglang/pull/27149), [#28290](https://github.com/sgl-project/sglang/pull/28290), [#28920](https://github.com/sgl-project/sglang/pull/28920) |
| `test/registered/ascend/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py` | no direct PR-number commit |
| `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` | [#25195](https://github.com/sgl-project/sglang/pull/25195), [#29775](https://github.com/sgl-project/sglang/pull/29775), [#29885](https://github.com/sgl-project/sglang/pull/29885), [#30365](https://github.com/sgl-project/sglang/pull/30365) |
| `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` | [#24947](https://github.com/sgl-project/sglang/pull/24947), [#26609](https://github.com/sgl-project/sglang/pull/26609), [#28098](https://github.com/sgl-project/sglang/pull/28098), [#29569](https://github.com/sgl-project/sglang/pull/29569), [#30898](https://github.com/sgl-project/sglang/pull/30898) |
| `test/registered/gb300/test_deepseek_v4_pro_fp4.py` | [#28103](https://github.com/sgl-project/sglang/pull/28103), [#29502](https://github.com/sgl-project/sglang/pull/29502) |
| `test/registered/kernels/ops/attention/test_deepseek_v4_compress_state_runtime_shapes.py` | [#29630](https://github.com/sgl-project/sglang/pull/29630) |
| `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` | [#25195](https://github.com/sgl-project/sglang/pull/25195), [#26141](https://github.com/sgl-project/sglang/pull/26141), [#26609](https://github.com/sgl-project/sglang/pull/26609), [#26766](https://github.com/sgl-project/sglang/pull/26766), [#28098](https://github.com/sgl-project/sglang/pull/28098), [#30898](https://github.com/sgl-project/sglang/pull/30898), [#31125](https://github.com/sgl-project/sglang/pull/31125) |
| `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` | [#26141](https://github.com/sgl-project/sglang/pull/26141), [#26609](https://github.com/sgl-project/sglang/pull/26609), [#27867](https://github.com/sgl-project/sglang/pull/27867), [#28098](https://github.com/sgl-project/sglang/pull/28098) |
| `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` | [#26609](https://github.com/sgl-project/sglang/pull/26609), [#26766](https://github.com/sgl-project/sglang/pull/26766), [#28098](https://github.com/sgl-project/sglang/pull/28098) |
| `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py` | [#26609](https://github.com/sgl-project/sglang/pull/26609), [#28098](https://github.com/sgl-project/sglang/pull/28098) |
| `test/registered/unit/models/test_deepseek_v4_fused_mhc_policy.py` | [#30954](https://github.com/sgl-project/sglang/pull/30954) |
| `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py` | [#27349](https://github.com/sgl-project/sglang/pull/27349) |

## PR Coverage Summary

- Git-traced PRs: 122
- Extra PRs preserved from existing docs: 49
- Total PRs in this document: 171
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#23605](https://github.com/sgl-project/sglang/pull/23605) | merged | Add DeepSeek V4 cookbook | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23617](https://github.com/sgl-project/sglang/pull/23617) | merged | Further update Deepseek V4 docs | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-24 | [#23628](https://github.com/sgl-project/sglang/pull/23628) | merged | docs: note H200 DeepSeek-V4 checkpoint | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23622](https://github.com/sgl-project/sglang/pull/23622) | merged | Again update DeepSeek V4 cookbook | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23634](https://github.com/sgl-project/sglang/pull/23634) | merged | Update pro fp8 checkpoint in DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23684](https://github.com/sgl-project/sglang/pull/23684) | merged | docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23689](https://github.com/sgl-project/sglang/pull/23689) | merged | docs(DeepSeek-V4): mark b200\|small\|pd-disagg + h200\|small\|{cp,pd-disagg} verified | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23691](https://github.com/sgl-project/sglang/pull/23691) | merged | docs(DeepSeek-V4): mark gb300\|{small,big}\|{cp,pd-disagg} verified + GB300-specific fixes | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23690](https://github.com/sgl-project/sglang/pull/23690) | merged | Small udpate gb300 recipe for deepseek v4 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23697](https://github.com/sgl-project/sglang/pull/23697) | merged | update: b300 container for dsv4 | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23698](https://github.com/sgl-project/sglang/pull/23698) | merged | docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23715](https://github.com/sgl-project/sglang/pull/23715) | merged | docs(DeepSeek-V4): mark h200\|big\|pd-disagg verified + recipe fixes | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23728](https://github.com/sgl-project/sglang/pull/23728) | merged | ci: add docker release workflow for deepseek_v4 branch | `.github/workflows/release-docker-deepseek-v4.yml` |
| 2026-04-25 | [#23730](https://github.com/sgl-project/sglang/pull/23730) | merged | [CI] release-docker-deepseek-v4: select which flavors to push | `.github/workflows/release-docker-deepseek-v4.yml` |
| 2026-04-26 | [#23725](https://github.com/sgl-project/sglang/pull/23725) | merged | docs(DeepSeek-V4): add GB200 platform to cookbook recipe | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
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
| 2026-04-29 | [#23980](https://github.com/sgl-project/sglang/pull/23980) | merged | docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4 | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
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
| 2026-05-14 | [#25243](https://github.com/sgl-project/sglang/pull/25243) | merged | [Docs] update dsv4 cookbook with H100 deployment commands | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-15 | [#24691](https://github.com/sgl-project/sglang/pull/24691) | merged | [UnifiedTree]: Support HiCache For DeepSeek_V4 | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-05-15 | [#25369](https://github.com/sgl-project/sglang/pull/25369) | merged | Add hicache feature in dsv4 cookbook | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-16 | [#25419](https://github.com/sgl-project/sglang/pull/25419) | merged | Port SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN from deepseek_v4_dev | `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/environ.py` |
| 2026-05-16 | [#24704](https://github.com/sgl-project/sglang/pull/24704) | merged | feat: add Pipeline Parallelism (PP) and PD support for DeepSeek-V4 | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-05-16 | [#25477](https://github.com/sgl-project/sglang/pull/25477) | merged | [BugFix]: Fix DeepSeek V4 HiCache layer count logic | `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` |
| 2026-05-16 | [#25410](https://github.com/sgl-project/sglang/pull/25410) | merged | [Docs] Update DeepSeek V4 cookbook to use the latest docker image | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-16 | [#25412](https://github.com/sgl-project/sglang/pull/25412) | merged | [Doc] DSV4 cookbook: clean up env vars, add MegaMoE toggle, unify docker image | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-05-17 | [#25506](https://github.com/sgl-project/sglang/pull/25506) | merged | [Doc] Fix several places for dpsk v4 cookbook | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
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
| 2026-05-21 | [#25923](https://github.com/sgl-project/sglang/pull/25923) | merged | [Docs] DeepSeek-V4: switch H200 FP4 Pro to flashinfer_mxfp4, Flash Balanced too | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-22 | [#26057](https://github.com/sgl-project/sglang/pull/26057) | merged | [docs] DeepSeek-V4 cookbook: split Quantization axis, add H100 SGLang FP8 | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-22 | [#25128](https://github.com/sgl-project/sglang/pull/25128) | merged | [Intel GPU] 1/N Fix tilelang import in deepseek v4 rope as optional | `python/sglang/srt/layers/deepseek_v4_rope.py` |
| 2026-05-23 | [#26141](https://github.com/sgl-project/sglang/pull/26141) | merged | Add non-MTP DSV4 test coverage | `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` |
| 2026-05-23 | [#26164](https://github.com/sgl-project/sglang/pull/26164) | merged | [docs] DeepSeek-V4 cookbook: balanced MegaMoE cap, H200 Pro FP4 mem-frac, nsa-* compat, PD-disagg fixes | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-05-23 | [#25898](https://github.com/sgl-project/sglang/pull/25898) | merged | [AMD] Dsv4/pr1 fix run time issue | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-05-24 | [#25948](https://github.com/sgl-project/sglang/pull/25948) | merged | [dsv4] support eplb | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-05-25 | [#26239](https://github.com/sgl-project/sglang/pull/26239) | merged | [dsv4] fix multi-step draft on non-cuda-graph path | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-05-26 | [#25391](https://github.com/sgl-project/sglang/pull/25391) | merged | Support DeepSeek V4 DeepEP Waterfill | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-05-26 | [#26208](https://github.com/sgl-project/sglang/pull/26208) | merged | [AMD] Dsv4/pr2 compressor opt | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-05-26 | [#26413](https://github.com/sgl-project/sglang/pull/26413) | merged | [docs] DeepSeek-V4 cookbook: note cu129 image for GB200 Pro DeepEP backend | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-05-27 | [#26451](https://github.com/sgl-project/sglang/pull/26451) | merged | [docs] Fix V4 Pro balanced recipe | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
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
| 2026-06-04 | [#27035](https://github.com/sgl-project/sglang/pull/27035) | merged | docs: add DeepSeek V4 FP4 indexer usage | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
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
| 2026-06-22 | [#25820](https://github.com/sgl-project/sglang/pull/25820) | merged | [NVIDIA] Support NVFP4 MoE for DeepSeek-V4 | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `python/sglang/srt/models/deepseek_v4.py` |
| 2026-06-22 | [#28920](https://github.com/sgl-project/sglang/pull/28920) | merged | [AMD] deepseek-v4 clean env vars | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`, `test/registered/amd/test_deepseek_v4_flash_fp4.py` |
| 2026-06-22 | [#28941](https://github.com/sgl-project/sglang/pull/28941) | merged | [AMD] Fix nightly-8-gpu-mi35x-deepseek-v4-flash-rocm720 OOM issue | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-06-23 | [#28981](https://github.com/sgl-project/sglang/pull/28981) | merged | [AMD] Update v4 cookbook to clean env vars | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-06-23 | [#28938](https://github.com/sgl-project/sglang/pull/28938) | merged | [AMD] Improve performance of dsv4 in high concurrency | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-06-24 | [#28455](https://github.com/sgl-project/sglang/pull/28455) | merged | [AMD] Fix DeepSeek-V4 fp8 KV path on gfx942 (e4m3fnuz) | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-06-24 | [#28952](https://github.com/sgl-project/sglang/pull/28952) | merged | Add DeepSeek V4 Flash demo notebook | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-06-25 | [#29261](https://github.com/sgl-project/sglang/pull/29261) | merged | [Docs] Fix broken links in cookbook | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` |
| 2026-06-25 | [#28103](https://github.com/sgl-project/sglang/pull/28103) | merged | Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test | `test/registered/gb300/test_deepseek_v4_pro_fp4.py` |
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
| 2026-07-07 | [#27867](https://github.com/sgl-project/sglang/pull/27867) | merged | [DSv4] Loading Time Weight Dequant | `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` |
| 2026-07-07 | [#30333](https://github.com/sgl-project/sglang/pull/30333) | merged | [AMD] Fix DeepSeek V4 MTP accuracy issue | `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` |
| 2026-07-08 | [#27926](https://github.com/sgl-project/sglang/pull/27926) | merged | [DSV4] perf: Make FP8 quant output tensor contiguous | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-09 | [#29417](https://github.com/sgl-project/sglang/pull/29417) | merged | [AMD] Enable unified-KV HiCache on DeepSeek-V4 | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-07-09 | [#30695](https://github.com/sgl-project/sglang/pull/30695) | merged | [Refactor] Make DeepSeek-V4 attention backend tolerate an absent CPU seq_lens mirror | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-07-10 | [#30711](https://github.com/sgl-project/sglang/pull/30711) | merged | [Refactor] Split DeepSeek-V4 MQALayer into a reusable attention base | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-13 | [#30898](https://github.com/sgl-project/sglang/pull/30898) | merged | Enable breakable prefill CUDA graph for DP attention | `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` |
| 2026-07-14 | [#31125](https://github.com/sgl-project/sglang/pull/31125) | merged | Disable flaky DSV4-Flash FP4 BCG determinism test (nondeterminism from #30898 idle-rank dummy extend) | `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` |
| 2026-07-15 | [#30365](https://github.com/sgl-project/sglang/pull/30365) | merged | [DSV4] Remove per-step seqlen D2H from speculative to make overlap scheduler work | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` |
| 2026-07-15 | [#30792](https://github.com/sgl-project/sglang/pull/30792) | merged | [Kernel] Migrate DSA + DSV4 attention kernels to sglang.kernels (RFC #29630, Phase 2.5, 5/7) | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-15 | [#30651](https://github.com/sgl-project/sglang/pull/30651) | merged | cookbook(deepseek-v4): add MORI disagg backend for AMD + bump MI355X image | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` |
| 2026-07-16 | [#28983](https://github.com/sgl-project/sglang/pull/28983) | merged | perf(deepseek_v4): enable SGLANG_OPT_FP8_WO_A_GEMM on sm90 (Hopper) | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-16 | [#31373](https://github.com/sgl-project/sglang/pull/31373) | merged | [Docs] Align B200 DeepSeek-V4-Pro balanced recipe with MegaMoE | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` |
| 2026-07-16 | [#31122](https://github.com/sgl-project/sglang/pull/31122) | merged | [Docs] Add AMD-specific HiCache config for DeepSeek V4 playground | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-07-16 | [#30238](https://github.com/sgl-project/sglang/pull/30238) | merged | [AMD] Support two batch overlap with MTP on DeepSeekV4 | `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py` |
| 2026-07-16 | [#25763](https://github.com/sgl-project/sglang/pull/25763) | merged | [Feature] Support DeepSeek-V4 Wint4Abf16 and Win4Afp8. | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-17 | [#31452](https://github.com/sgl-project/sglang/pull/31452) | merged | [Docs] Tune DeepSeek-V4 HiCache for MI355X PD | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-07-18 | [#30272](https://github.com/sgl-project/sglang/pull/30272) | merged | Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2 | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-07-19 | [#31705](https://github.com/sgl-project/sglang/pull/31705) | merged | [DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-07-21 | [#31363](https://github.com/sgl-project/sglang/pull/31363) | merged | docs(cookbook): re-benchmark DeepSeek-V4 on sglang 0.5.15 | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` |
| 2026-07-23 | [#27657](https://github.com/sgl-project/sglang/pull/27657) | merged | [DeepSeek V4] CP decode opt: slice repeat attention weights to local TP partition | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py` |
| 2026-07-23 | [#29569](https://github.com/sgl-project/sglang/pull/29569) | merged | [DSV4] Support megamoe for CP | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` |
| 2026-07-24 | [#27059](https://github.com/sgl-project/sglang/pull/27059) | merged | Add FP4 Indexer for DeepSeek V4 on SM120 | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-07-24 | [#31087](https://github.com/sgl-project/sglang/pull/31087) | merged | [RL] DSV4: dispatch indexer topk_transform_512 through DSATopKBackend | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-07-24 | [#31086](https://github.com/sgl-project/sglang/pull/31086) | merged | [RL] DSV4: add env to quantize SWA KV cache from bf16-rounded values | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-07-26 | [#30954](https://github.com/sgl-project/sglang/pull/30954) | merged | [SM120] Allow fused MHC opt-in with standalone TileLang pre disabled | `test/registered/unit/models/test_deepseek_v4_fused_mhc_policy.py`, `python/sglang/srt/models/deepseek_v4.py` |

## Per-PR Diff Audit Cards

### PR #23605 - Add DeepSeek V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23605
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `492883c8ca66`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +1024/-1, 1041 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DeepSeek V4 cookbook"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "Add DeepSeek V4 cookbook"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0 (453 lines); hunks: -0,0 +1,453.
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0 (453 lines); hunks: -0,0 +1,453
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -0,0 +1,453 @@
+---
+title: DeepSeek-V4
+metatags:
+    description: "Deploy DeepSeek-V4 with SGLang — a next-generation MoE model from DeepSeek. Blackwell deployments use the FP4 checkpoint; Hopper deployments use the FP8 checkpoi
+tag: NEW
+---
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/docs.json`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23617 - Further update Deepseek V4 docs

- Link: https://github.com/sgl-project/sglang/pull/23617
- Status/date: merged / 2026-04-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-6, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Further update Deepseek V4 docs"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "Further update Deepseek V4 docs"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6 (11 lines); hunks: -137,12 +137,11 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6 (11 lines); hunks: -137,12 +137,11 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23628 - docs: note H200 DeepSeek-V4 checkpoint

- Link: https://github.com/sgl-project/sglang/pull/23628
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `1a37e57fb1ae`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-0, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: note H200 DeepSeek-V4 checkpoint"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs: note H200 DeepSeek-V4 checkpoint"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -99,6 +99,10 @@ Please refer to the [official SGLang installation guide](../.....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -99,6 +99,10 @@ Please refer to the [official SGLang installation guide](../....
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -99,6 +99,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+<Note>
+For H200 GPU deployments, use the SGLang checkpoint under `sgl-project`, not the default DeepSeek checkpoint.
+</Note>
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23622 - Again update DeepSeek V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23622
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `3a620cb761ff`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +32/-9, 73 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Again update DeepSeek V4 cookbook"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "Again update DeepSeek V4 cookbook"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../.....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../....
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+For how to actually launch one of these images, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). A minimal example (substitute the
+'''bash Command
+docker run --gpus all \
+    --shm-size 32g \
+    -p 30000:30000 \
+    -v ~/.cache/huggingface:/root/.cache/huggingface \
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23634 - Update pro fp8 checkpoint in DeepSeek V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23634
- Status/date: merged / 2026-04-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 12 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update pro fp8 checkpoint in DeepSeek V4 cookbook"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "Update pro fp8 checkpoint in DeepSeek V4 cookbook"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2 (4 lines); hunks: -139,9 +139,9 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2 (4 lines); hunks: -139,9 +139,9 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -139,9 +139,9 @@ export const DeepSeekV4Deployment = () => {
-    // repackagings; Flash is public, Pro is still being uploaded.
+    // repackagings for both variants.
-    "h200|big":    { slug: "<TO_BE_UPLOADED_DeepSeek-V4-Pro-FP8>",     tp: 16, multinode: true, nnodes: 2 },
+    "h200|big":    { slug: "sgl-project/DeepSeek-V4-Pro-FP8",          tp: 16, multinode: true, nnodes: 2 },
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23684 - docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models

- Link: https://github.com/sgl-project/sglang/pull/23684
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `fd401c2fb451`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-0, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models"; model line: DeepSeek V4; category: bug fix; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -147,6 +147,10 @@ The generator currently picks values on the **conservative*....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -147,6 +147,10 @@ The generator currently picks values on the **conservative*...
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -147,6 +147,10 @@ The generator currently picks values on the **conservative** side (mirroring an
+**Base model usage**
+In order to use base models, please enable `SGLANG_FIX_DSV4_BASE_MODEL_LOAD=1` and use latest code, before the next round of testing matrix is finished.
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23689 - docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified

- Link: https://github.com/sgl-project/sglang/pull/23689
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `d2c61acf2597`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +22/-1, 59 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1 (9 lines); hunks: -145,7 +145,14 @@ The generator currently picks values on the **conservative*....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1 (9 lines); hunks: -145,7 +145,14 @@ The generator currently picks values on the **conservative*...
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -145,7 +145,14 @@ The generator currently picks values on the **conservative** side (mirroring an
-The H200 image and checkpoint are currently being uploaded — public path coming shortly.
+H200 image (`lmsysorg/sglang:deepseek-v4-hopper`) and FP8 checkpoints
+(`sgl-project/DeepSeek-V4-Flash-FP8`, `sgl-project/DeepSeek-V4-Pro-FP8`) are
+publicly available.
+PD-Disagg recipes on H200 may require `docker run --privileged --ulimit memlock=-1`
+(or `--device /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK`) so mooncake
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23691 - docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes

- Link: https://github.com/sgl-project/sglang/pull/23691
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `8a395994edcf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +56/-5, 113 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes"; model line: DeepSeek V4; category: bug fix; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0 (7 lines); hunks: -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpo....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0 (7 lines); hunks: -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpo...
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpoints.
+**GB300 PD-Disagg cross-pod MNNVL**
+On some GB300 clusters with cross-pod KV transfer over NVLink, mooncake may
+fail with `nvlink_transport.cpp:497 Requested address ... not found!`. If
+this happens, prepend `MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1`
+to both prefill and decode `sglang serve` commands.
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23690 - Small udpate gb300 recipe for deepseek v4

- Link: https://github.com/sgl-project/sglang/pull/23690
- Status/date: merged / 2026-04-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Small udpate gb300 recipe for deepseek v4"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "Small udpate gb300 recipe for deepseek v4"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -172,6 +172,9 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -172,6 +172,9 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -172,6 +172,9 @@ export const DeepSeekV4Deployment = () => {
+    "gb300|small|low-latency",
+    "gb300|small|balanced",
+    "gb300|small|max-throughput",
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23697 - update: b300 container for dsv4

- Link: https://github.com/sgl-project/sglang/pull/23697
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `0d224e505333`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-2, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "update: b300 container for dsv4"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "update: b300 container for dsv4"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../.....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../....
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+    <tr>
+      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>NVIDIA B300</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>lmsysorg/sglang:deepseek-v4-b300</code></td>
+    </tr>
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23698 - docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9

- Link: https://github.com/sgl-project/sglang/pull/23698
- Status/date: merged / 2026-04-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-3, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-3 (8 lines); hunks: -495,11 +495,13 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-3 (8 lines); hunks: -495,11 +495,13 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-3
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23715 - docs(DeepSeek-V4): mark h200|big|pd-disagg verified + recipe fixes

- Link: https://github.com/sgl-project/sglang/pull/23715
- Status/date: merged / 2026-04-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +31/-4, 59 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): mark h200|big|pd-disagg verified + recipe fixes"; model line: DeepSeek V4; category: bug fix; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "docs(DeepSeek-V4): mark h200|big|pd-disagg verified + recipe fixes"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +31/-4 (35 lines); hunks: -178,8 +178,7 @@ export const DeepSeekV4Deployment = () => {; -480,6 +479,12 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +31/-4 (35 lines); hunks: -178,8 +178,7 @@ export const DeepSeekV4Deployment = () => {; -480,6 +479,12 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +31/-4
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23728 - ci: add docker release workflow for deepseek_v4 branch

- Link: https://github.com/sgl-project/sglang/pull/23728
- Status/date: merged / 2026-04-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +93/-0, 94 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "ci: add docker release workflow for deepseek_v4 branch"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `.github/workflows/release-docker-deepseek-v4.yml`; technical summary: Covers "ci: add docker release workflow for deepseek_v4 branch"; the main implementation surface is `.github/workflows/release-docker-deepseek-v4.yml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.github/workflows/release-docker-deepseek-v4.yml` added +93/-0 (93 lines); hunks: -0,0 +1,93.
- Code diff details:
  - `.github/workflows/release-docker-deepseek-v4.yml` added +93/-0 (93 lines); hunks: -0,0 +1,93
- Key code excerpts:

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

- Reviewed files:
  - ci: `.github/workflows/release-docker-deepseek-v4.yml` added +93/-0
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #23730 - [CI] release-docker-deepseek-v4: select which flavors to push

- Link: https://github.com/sgl-project/sglang/pull/23730
- Status/date: merged / 2026-04-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +56/-18, 92 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] release-docker-deepseek-v4: select which flavors to push"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `.github/workflows/release-docker-deepseek-v4.yml`; technical summary: Covers "[CI] release-docker-deepseek-v4: select which flavors to push"; the main implementation surface is `.github/workflows/release-docker-deepseek-v4.yml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.github/workflows/release-docker-deepseek-v4.yml` modified +56/-18 (74 lines); hunks: -12,35 +12,73 @@ on:.
- Code diff details:
  - `.github/workflows/release-docker-deepseek-v4.yml` modified +56/-18 (74 lines); hunks: -12,35 +12,73 @@ on:
- Key code excerpts:

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

- Reviewed files:
  - ci: `.github/workflows/release-docker-deepseek-v4.yml` modified +56/-18
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #23725 - docs(DeepSeek-V4): add GB200 platform to cookbook recipe

- Link: https://github.com/sgl-project/sglang/pull/23725
- Status/date: merged / 2026-04-26
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `049f1bf6fb42`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +58/-8, 195 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): add GB200 platform to cookbook recipe"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(DeepSeek-V4): add GB200 platform to cookbook recipe"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-2 (8 lines); hunks: -29,13 +29,13 @@ tag: NEW; -88,6 +88,10 @@ Please refer to the [official SGLang installation guide](../.....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-2 (8 lines); hunks: -29,13 +29,13 @@ tag: NEW; -88,6 +88,10 @@ Please refer to the [official SGLang installation guide](../....
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -29,13 +29,13 @@ tag: NEW
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>single-node serving: B200 / GB300 / H200 on 4 GPUs</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>single-node serving: B200 / GB200 / GB300 / H200 on 4 GPUs</td>
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB300 4 GPU / H200 16 GPU (2 nodes)</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB200 8 GPU (2 nodes) / GB300 4 GPU / H200 16 GPU (2 nodes)</td>
@@ -88,6 +88,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+    <tr>
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-2
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23742 - docs(DeepSeek-V4): add h200|big verified recipes + tune H200 Pro parameters

- Link: https://github.com/sgl-project/sglang/pull/23742
- Status/date: merged / 2026-04-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +22/-8, 83 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): add h200|big verified recipes + tune H200 Pro parameters"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "docs(DeepSeek-V4): add h200|big verified recipes + tune H200 Pro parameters"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +22/-8 (30 lines); hunks: -184,6 +184,9 @@ export const DeepSeekV4Deployment = () => {; -272,7 +275,9 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +22/-8 (30 lines); hunks: -184,6 +184,9 @@ export const DeepSeekV4Deployment = () => {; -272,7 +275,9 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +22/-8
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23737 - docs(DeepSeek-V4): mark gb200|big|low-latency verified

- Link: https://github.com/sgl-project/sglang/pull/23737
- Status/date: merged / 2026-04-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): mark gb200|big|low-latency verified"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "docs(DeepSeek-V4): mark gb200|big|low-latency verified"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +1/-0 (1 lines); hunks: -174,6 +174,7 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +1/-0 (1 lines); hunks: -174,6 +174,7 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -174,6 +174,7 @@ export const DeepSeekV4Deployment = () => {
+    "gb300|big|low-latency",
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +1/-0
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23778 - ci(deepseek-v4): add b300/grace-blackwell dev-branch build options

- Link: https://github.com/sgl-project/sglang/pull/23778
- Status/date: merged / 2026-04-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-5, 58 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "ci(deepseek-v4): add b300/grace-blackwell dev-branch build options"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `.github/workflows/release-docker-deepseek-v4.yml`; technical summary: Covers "ci(deepseek-v4): add b300/grace-blackwell dev-branch build options"; the main implementation surface is `.github/workflows/release-docker-deepseek-v4.yml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.github/workflows/release-docker-deepseek-v4.yml` modified +23/-5 (28 lines); hunks: -32,6 +32,16 @@ on:; -50,19 +60,27 @@ jobs:.
- Code diff details:
  - `.github/workflows/release-docker-deepseek-v4.yml` modified +23/-5 (28 lines); hunks: -32,6 +32,16 @@ on:; -50,19 +60,27 @@ jobs:
- Key code excerpts:

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

- Reviewed files:
  - ci: `.github/workflows/release-docker-deepseek-v4.yml` modified +23/-5
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #23787 - amd/deepseek_v4 integration 1/N - 0426

- Link: https://github.com/sgl-project/sglang/pull/23787
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 128 files, +18341/-879, 18279 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "amd/deepseek_v4 integration 1/N - 0426"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py`, `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`; technical summary: Covers "amd/deepseek_v4 integration 1/N - 0426"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py`, `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` added +2803/-0 (2803 lines); `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py` added +1330/-0 (1330 lines); hunks: -0,0 +1,1330; symbols: _copy_metadata, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadataRadix, touching `_copy_metadata, _create_flashmla_metadata, _create_dummy_paged_compress_data`; `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` added +840/-0 (840 lines); hunks: -0,0 +1,840; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`; `python/sglang/srt/layers/mhc.py` added +686/-0 (686 lines); hunks: -0,0 +1,686; symbols: hc_split_sinkhorn_kernel, hc_split_sinkhorn_kernel_, hc_split_sinkhorn, mhc_pre_big_fuse_tilelang, touching `hc_split_sinkhorn_kernel, hc_split_sinkhorn_kernel_, hc_split_sinkhorn`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` added +2803/-0 (2803 lines)
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py` added +1330/-0 (1330 lines); hunks: -0,0 +1,1330; symbols: _copy_metadata, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadataRadix
  - `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` added +840/-0 (840 lines); hunks: -0,0 +1,840; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `python/sglang/srt/layers/mhc.py` added +686/-0 (686 lines); hunks: -0,0 +1,686; symbols: hc_split_sinkhorn_kernel, hc_split_sinkhorn_kernel_, hc_split_sinkhorn, mhc_pre_big_fuse_tilelang
  - `python/sglang/srt/layers/attention/compressed/indexer.py` added +616/-0 (616 lines); hunks: -0,0 +1,616; symbols: fp8_paged_mqa_logits_torch, topk_transform_512_pytorch_vectorized, _fused_scale_kernel, fused_scale
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` added +2803/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py` added +1330/-0; `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` added +840/-0; `python/sglang/srt/layers/mhc.py` added +686/-0; `python/sglang/srt/layers/attention/compressed/indexer.py` added +616/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` added +591/-0
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_activation.py`, `python/sglang/srt/flashmla_tests/__init__.py`, `python/sglang/srt/flashmla_tests/kernelkit/.gitignore`, `python/sglang/srt/flashmla_tests/kernelkit/__init__.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23776 - [DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP

- Link: https://github.com/sgl-project/sglang/pull/23776
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-0, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "[DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +10/-0 (10 lines); hunks: -227,9 +227,11 @@ def __init__(; -283,6 +285,12 @@ def forward(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +10/-0 (10 lines); hunks: -227,9 +227,11 @@ def __init__(; -283,6 +285,12 @@ def forward(; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +10/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23817 - docs: verify GB300 Pro DeepSeek V4 recipes

- Link: https://github.com/sgl-project/sglang/pull/23817
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-0, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: verify GB300 Pro DeepSeek V4 recipes"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "docs: verify GB300 Pro DeepSeek V4 recipes"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +6/-0 (6 lines); hunks: -182,7 +182,9 @@ export const DeepSeekV4Deployment = () => {; -365,6 +367,8 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +6/-0 (6 lines); hunks: -182,7 +182,9 @@ export const DeepSeekV4Deployment = () => {; -365,6 +367,8 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +6/-0
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23810 - Add benchmarking scripts for deepseek v4

- Link: https://github.com/sgl-project/sglang/pull/23810
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +243/-0, 244 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add benchmarking scripts for deepseek v4"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `scripts/bench_gpqa_aime.py`; technical summary: Covers "Add benchmarking scripts for deepseek v4"; the main implementation surface is `scripts/bench_gpqa_aime.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `scripts/bench_gpqa_aime.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: _venv_cmd, get_timestamp, get_random_int, setup_ns, touching `_venv_cmd, get_timestamp, get_random_int`.
- Code diff details:
  - `scripts/bench_gpqa_aime.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: _venv_cmd, get_timestamp, get_random_int, setup_ns
- Key code excerpts:

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

- Reviewed files:
  - other: `scripts/bench_gpqa_aime.py` added +243/-0
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #23832 - amd/deepseek_v4 integration 2/N - cuda graph 0426

- Link: https://github.com/sgl-project/sglang/pull/23832
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 26 files, +534/-92, 973 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "amd/deepseek_v4 integration 2/N - cuda graph 0426"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/compressed/indexer.py`, `python/sglang/srt/layers/attention/compressed/metadata.py`; technical summary: Covers "amd/deepseek_v4 integration 2/N - cuda graph 0426"; the main implementation surface is `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/compressed/indexer.py`, `python/sglang/srt/layers/attention/compressed/metadata.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +395/-1 (396 lines); hunks: -1,5 +1,5; -27,6 +27,7; symbols: fast_log2_ceil, tilelang_sparse_fwd, _next_power_of_2, _padded_H, touching `fast_log2_ceil, tilelang_sparse_fwd, _next_power_of_2`; `python/sglang/srt/layers/attention/compressed/indexer.py` modified +78/-76 (154 lines); hunks: -1,6 +1,6; -37,6 +37,8; symbols: fp8_paged_mqa_logits_torch, touching `fp8_paged_mqa_logits_torch`; `python/sglang/srt/layers/attention/compressed/metadata.py` modified +12/-11 (23 lines); hunks: -169,18 +169,19 @@ def max_seq_len(self) -> int:; symbols: max_seq_len, copy_, touching `max_seq_len, copy_`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-1 (10 lines); hunks: -1152,7 +1152,9 @@ def run_once():; -1162,6 +1164,9 @@ def run_once():; symbols: run_once, replay_prepare, touching `run_once, replay_prepare`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +395/-1 (396 lines); hunks: -1,5 +1,5; -27,6 +27,7; symbols: fast_log2_ceil, tilelang_sparse_fwd, _next_power_of_2, _padded_H
  - `python/sglang/srt/layers/attention/compressed/indexer.py` modified +78/-76 (154 lines); hunks: -1,6 +1,6; -37,6 +37,8; symbols: fp8_paged_mqa_logits_torch
  - `python/sglang/srt/layers/attention/compressed/metadata.py` modified +12/-11 (23 lines); hunks: -169,18 +169,19 @@ def max_seq_len(self) -> int:; symbols: max_seq_len, copy_
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-1 (10 lines); hunks: -1152,7 +1152,9 @@ def run_once():; -1162,6 +1164,9 @@ def run_once():; symbols: run_once, replay_prepare
  - `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py` modified +7/-0 (7 lines); hunks: -13,6 +13,10 @@ def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):; -32,6 +36,9 @@ def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):; symbols: flash_mla_with_kvcache_entrypoint
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +395/-1; `python/sglang/srt/layers/attention/compressed/indexer.py` modified +78/-76; `python/sglang/srt/layers/attention/compressed/metadata.py` modified +12/-11; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-1; `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py` modified +7/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +4/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/attention/base_attn_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23756 - feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch

- Link: https://github.com/sgl-project/sglang/pull/23756
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +47/-12, 90 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/environ.py`; technical summary: Covers "feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch"; the main implementation surface is `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/environ.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +46/-12 (58 lines); hunks: -22,7 +22,7; -44,14 +44,43 @@ def update_deep_gemm_config(gpu_id: int, server_args: Server...; symbols: update_deep_gemm_config, _compile_deep_gemm_one_type_all, touching `update_deep_gemm_config, _compile_deep_gemm_one_type_all`; `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -336,6 +336,7 @@ class Envs:; symbols: Envs, touching `Envs`.
- Code diff details:
  - `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +46/-12 (58 lines); hunks: -22,7 +22,7; -44,14 +44,43 @@ def update_deep_gemm_config(gpu_id: int, server_args: Server...; symbols: update_deep_gemm_config, _compile_deep_gemm_one_type_all
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -336,6 +336,7 @@ class Envs:; symbols: Envs
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +46/-12; `python/sglang/srt/environ.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23883 - Enable DeepGemm warmup in DeepSeek-V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23883
- Status/date: merged / 2026-04-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-5, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable DeepGemm warmup in DeepSeek-V4 cookbook"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "Enable DeepGemm warmup in DeepSeek-V4 cookbook"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-5 (8 lines); hunks: -255,7 +255,6 @@ export const DeepSeekV4Deployment = () => {; -461,8 +460,8 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-5 (8 lines); hunks: -255,7 +255,6 @@ export const DeepSeekV4Deployment = () => {; -461,8 +460,8 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-5
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23943 - [Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe

- Link: https://github.com/sgl-project/sglang/pull/23943
- Status/date: merged / 2026-04-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +32/-0, 39 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "[Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +32/-0 (32 lines); hunks: -482,6 +482,38 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +32/-0 (32 lines); hunks: -482,6 +482,38 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +32/-0
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23980 - docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4

- Link: https://github.com/sgl-project/sglang/pull/23980
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `4e885baa9bf1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +84/-8, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +5/-5 (10 lines); hunks: -1,7 +1,7; -35,7 +35,7 @@ tag: NEW.
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +5/-5 (10 lines); hunks: -1,7 +1,7; -35,7 +35,7 @@ tag: NEW
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -1,7 +1,7 @@
-    description: "Deploy DeepSeek-V4 with SGLang — a next-generation MoE model from DeepSeek. Blackwell deployments use the FP4 checkpoint; Hopper deployments use the FP8 checkpoi
+    description: "Deploy DeepSeek-V4 with SGLang — a next-generation MoE model from DeepSeek."
@@ -35,7 +35,7 @@ tag: NEW
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB200 8 GPU (2 nodes) / GB300 4 GPU / H200 16 GPU (2 nodes)</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB200 8 GPU (2 nodes) / GB300 4 GPU / H200 8 GPU(fp4)/16 GPU(fp8)</t
@@ -153,9 +153,9 @@ The generator currently picks values on the **conservative** side (mirroring an
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +5/-5
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #24035 - [minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4

- Link: https://github.com/sgl-project/sglang/pull/24035
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `b3ead32d3ca2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-3, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-3 (3 lines); hunks: -120,9 +120,6 @@ docker run --gpus all \.
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-3 (3 lines); hunks: -120,9 +120,6 @@ docker run --gpus all \
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -120,9 +120,6 @@ docker run --gpus all \
-<Note>
-For H200 GPU deployments, use the SGLang checkpoint under `sgl-project`, not the default DeepSeek checkpoint.
-</Note>
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-3
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #24203 - [AMD] Deepseek v4 Flash / Pro nightly tests for MI35x ROCm 7.2

- Link: https://github.com/sgl-project/sglang/pull/24203
- Status/date: merged / 2026-05-04
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`; associated commits `5eff3c489a71`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +972/-0, 997 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Deepseek v4 Flash / Pro nightly tests for MI35x ROCm 7.2"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`; technical summary: Covers "[AMD] Deepseek v4 Flash / Pro nightly tests for MI35x ROCm 7.2"; the main implementation surface is `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/test_deepseek_v4_pro_fp4.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: TestDeepseekV4ProFp4, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV4ProFp4, setUpClass, tearDownClass`; `test/registered/amd/test_deepseek_v4_pro_fp8.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: TestDeepseekV4ProFp8, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV4ProFp8, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: TestDeepseekV4ProFp4, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/test_deepseek_v4_pro_fp8.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: TestDeepseekV4ProFp8, setUpClass, tearDownClass, test_a_gsm8k
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/amd/test_deepseek_v4_pro_fp4.py` added +209/-0; `test/registered/amd/test_deepseek_v4_pro_fp8.py` added +209/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_v4_fp4.py`, `test/registered/amd/test_deepseek_v4_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24367 - [docs] Update B300 Pro cookbook with accuracy-verified serving configs

- Link: https://github.com/sgl-project/sglang/pull/24367
- Status/date: merged / 2026-05-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +108/-11, 195 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] Update B300 Pro cookbook with accuracy-verified serving configs"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "[docs] Update B300 Pro cookbook with accuracy-verified serving configs"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +108/-11 (119 lines); hunks: -351,13 +351,41 @@ export const DeepSeekV4Deployment = () => {; -367,6 +395,26 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +108/-11 (119 lines); hunks: -351,13 +351,41 @@ export const DeepSeekV4Deployment = () => {; -367,6 +395,26 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +108/-11
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23882 - Deepseek V4

- Link: https://github.com/sgl-project/sglang/pull/23882
- Status/date: merged / 2026-05-08
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/configs/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` and 10 files; associated commits `35870d55aca7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 154 files, +24534/-712, 27836 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deepseek V4"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`; technical summary: Covers "Deepseek V4"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` added +1528/-0 (1528 lines); hunks: -0,0 +1,1528; symbols: _rms_normalize_kernel, rms_normalize_triton, MQALayer, __init__, touching `_rms_normalize_kernel, rms_normalize_triton, MQALayer`; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` added +1255/-0 (1255 lines); hunks: -0,0 +1,1255; symbols: _pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadata, touching `_pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data`; `python/sglang/srt/models/deepseek_v4_nextn.py` added +216/-0 (216 lines); hunks: -0,0 +1,216; symbols: DeepseekV4ModelNextN, __init__, hc_head, forward, touching `DeepseekV4ModelNextN, __init__, hc_head`; `python/sglang/srt/configs/deepseek_v4.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: try_detect_fp4_experts, DeepSeekV4Config, touching `try_detect_fp4_experts, DeepSeekV4Config`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` added +1528/-0 (1528 lines); hunks: -0,0 +1,1528; symbols: _rms_normalize_kernel, rms_normalize_triton, MQALayer, __init__
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` added +1255/-0 (1255 lines); hunks: -0,0 +1,1255; symbols: _pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadata
  - `python/sglang/srt/models/deepseek_v4_nextn.py` added +216/-0 (216 lines); hunks: -0,0 +1,216; symbols: DeepseekV4ModelNextN, __init__, hc_head, forward
  - `python/sglang/srt/configs/deepseek_v4.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: try_detect_fp4_experts, DeepSeekV4Config
  - `python/sglang/jit_kernel/deepseek_v4.py` added +908/-0 (908 lines); hunks: -0,0 +1,908; symbols: make_name, _jit_common_module, _jit_compress_128_online_plan_module, _jit_compress_128_online_module
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` added +1528/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` added +1255/-0; `python/sglang/srt/models/deepseek_v4_nextn.py` added +216/-0; `python/sglang/srt/configs/deepseek_v4.py` added +110/-0; `python/sglang/jit_kernel/deepseek_v4.py` added +908/-0; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` added +738/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/test_utils.py`, `test/manual/dsv4/__init__.py`, `test/manual/dsv4/_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24793 - [DSV4] Cherry pick missing commits from deepseek_v4 branch and enhance tests

- Link: https://github.com/sgl-project/sglang/pull/24793
- Status/date: merged / 2026-05-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +481/-87, 873 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Cherry pick missing commits from deepseek_v4 branch and enhance tests"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/unit/entrypoints/openai/test_protocol.py`; technical summary: Covers "[DSV4] Cherry pick missing commits from deepseek_v4 branch and enhance tests"; the main implementation surface is `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/unit/entrypoints/openai/test_protocol.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/function_call/test_function_call_parser.py` modified +111/-1 (112 lines); hunks: -31,7 +31,7; -1686,6 +1686,26 @@ def test_get_model_structural_tag(self):; symbols: TestPythonicDetector, test_get_model_structural_tag, test_self_closing_zero_arg_invoke, TestDeepSeekV4Detector, touching `TestPythonicDetector, test_get_model_structural_tag, test_self_closing_zero_arg_invoke`; `python/sglang/srt/function_call/deepseekv32_detector.py` modified +26/-10 (36 lines); hunks: -81,8 +81,13 @@ def __init__(self):; -92,6 +97,20 @@ def has_tool_call(self, text: str) -> bool:; symbols: __init__, has_tool_call, _unpack_invoke_match, _parse_parameters_from_xml, touching `__init__, has_tool_call, _unpack_invoke_match`; `test/registered/unit/entrypoints/openai/test_protocol.py` modified +31/-0 (31 lines); hunks: -220,6 +220,37 @@ def test_chat_completion_reasoning_effort_none_from_reasoni...; symbols: test_chat_completion_reasoning_effort_none_from_reasoning_dict, test_chat_completion_reasoning_effort_max, test_chat_completion_json_format, touching `test_chat_completion_reasoning_effort_none_from_reasoning_dict, test_chat_completion_reasoning_effort_max, test_chat_completion_json_format`; `python/sglang/srt/entrypoints/openai/protocol.py` modified +5/-2 (7 lines); hunks: -633,13 +633,16 @@ class ChatCompletionRequest(BaseModel):; symbols: ChatCompletionRequest, touching `ChatCompletionRequest`.
- Code diff details:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +111/-1 (112 lines); hunks: -31,7 +31,7; -1686,6 +1686,26 @@ def test_get_model_structural_tag(self):; symbols: TestPythonicDetector, test_get_model_structural_tag, test_self_closing_zero_arg_invoke, TestDeepSeekV4Detector
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +26/-10 (36 lines); hunks: -81,8 +81,13 @@ def __init__(self):; -92,6 +97,20 @@ def has_tool_call(self, text: str) -> bool:; symbols: __init__, has_tool_call, _unpack_invoke_match, _parse_parameters_from_xml
  - `test/registered/unit/entrypoints/openai/test_protocol.py` modified +31/-0 (31 lines); hunks: -220,6 +220,37 @@ def test_chat_completion_reasoning_effort_none_from_reasoni...; symbols: test_chat_completion_reasoning_effort_none_from_reasoning_dict, test_chat_completion_reasoning_effort_max, test_chat_completion_json_format
  - `python/sglang/srt/entrypoints/openai/protocol.py` modified +5/-2 (7 lines); hunks: -633,13 +633,16 @@ class ChatCompletionRequest(BaseModel):; symbols: ChatCompletionRequest
  - `scripts/ci/cuda/ci_install_dsv4_dep.sh` added +161/-0 (161 lines); hunks: -0,0 +1,161
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/unit/function_call/test_function_call_parser.py` modified +111/-1; `test/registered/unit/entrypoints/openai/test_protocol.py` modified +31/-0; `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py` renamed +65/-15; `test/registered/dsv4/test_deepseek_v4_flash_fp8_h200.py` renamed +28/-16
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +26/-10; `python/sglang/srt/entrypoints/openai/protocol.py` modified +5/-2; `python/sglang/srt/model_loader/weight_utils.py` modified +33/-3
  - other: `scripts/ci/cuda/ci_install_dsv4_dep.sh` added +161/-0
- Risk and verification: The diff ships test coverage in `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp8_h200.py`, `test/registered/unit/entrypoints/openai/test_protocol.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24775 - Optimize MHC pipeline: DeepGemm, fused norm, fused hc_head

- Link: https://github.com/sgl-project/sglang/pull/24775
- Status/date: merged / 2026-05-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +512/-73, 699 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimize MHC pipeline: DeepGemm, fused norm, fused hc_head"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/mhc.py`, `python/sglang/srt/layers/mhc_head.py`, `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "Optimize MHC pipeline: DeepGemm, fused norm, fused hc_head"; the main implementation surface is `python/sglang/srt/layers/mhc.py`, `python/sglang/srt/layers/mhc_head.py`, `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/mhc.py` modified +319/-64 (383 lines); hunks: -7,6 +7,7; -138,12 +139,15 @@ def mhc_pre_big_fuse_tilelang(; symbols: mhc_pre_big_fuse_tilelang, mhc_pre_gemm_sqrsum_splitk_stage_1, _compute_num_split_for_mhc_pre, mhc_pre_big_fuse_with_norm_tilelang, touching `mhc_pre_big_fuse_tilelang, mhc_pre_gemm_sqrsum_splitk_stage_1, _compute_num_split_for_mhc_pre`; `python/sglang/srt/layers/mhc_head.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: _hc_head_kernel, fused_hc_head, touching `_hc_head_kernel, fused_hc_head`; `python/sglang/srt/models/deepseek_v4.py` modified +40/-9 (49 lines); hunks: -653,7 +653,11 @@ def hc_pre(; -671,11 +675,16 @@ def hc_pre_torch_impl(x, hc_fn):; symbols: hc_pre, hc_pre_torch_impl, touching `hc_pre, hc_pre_torch_impl`; `scripts/ci/utils/slash_command_handler.py` modified +2/-0 (2 lines); hunks: -424,6 +424,8 @@ def handle_rerun_stage(; symbols: handle_rerun_stage, touching `handle_rerun_stage`.
- Code diff details:
  - `python/sglang/srt/layers/mhc.py` modified +319/-64 (383 lines); hunks: -7,6 +7,7; -138,12 +139,15 @@ def mhc_pre_big_fuse_tilelang(; symbols: mhc_pre_big_fuse_tilelang, mhc_pre_gemm_sqrsum_splitk_stage_1, _compute_num_split_for_mhc_pre, mhc_pre_big_fuse_with_norm_tilelang
  - `python/sglang/srt/layers/mhc_head.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: _hc_head_kernel, fused_hc_head
  - `python/sglang/srt/models/deepseek_v4.py` modified +40/-9 (49 lines); hunks: -653,7 +653,11 @@ def hc_pre(; -671,11 +675,16 @@ def hc_pre_torch_impl(x, hc_fn):; symbols: hc_pre, hc_pre_torch_impl
  - `scripts/ci/utils/slash_command_handler.py` modified +2/-0 (2 lines); hunks: -424,6 +424,8 @@ def handle_rerun_stage(; symbols: handle_rerun_stage
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/mhc.py` modified +319/-64; `python/sglang/srt/layers/mhc_head.py` added +151/-0; `python/sglang/srt/models/deepseek_v4.py` modified +40/-9
  - other: `scripts/ci/utils/slash_command_handler.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/mhc.py`, `python/sglang/srt/layers/mhc_head.py`, `python/sglang/srt/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24825 - [AMD] DSv4 nightly hotfix + schedule-aware --continue-on-error in AMD CI

- Link: https://github.com/sgl-project/sglang/pull/24825
- Status/date: merged / 2026-05-11
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`; associated commits `22543b198254`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +110/-110, 990 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] DSv4 nightly hotfix + schedule-aware --continue-on-error in AMD CI"; model line: DeepSeek V4; category: bug fix; main diff: `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`; technical summary: Covers "[AMD] DSv4 nightly hotfix + schedule-aware --continue-on-error in AMD CI"; the main implementation surface is `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/test_deepseek_v4_flash_fp4.py` renamed +1/-1 (2 lines); hunks: -82,7 +82,7 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/registered/amd/test_deepseek_v4_flash_fp8.py` renamed +1/-1 (2 lines); hunks: -82,7 +82,7 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +1/-1 (2 lines); hunks: -84,7 +84,7 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +1/-1 (2 lines); hunks: -84,7 +84,7 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`.
- Code diff details:
  - `test/registered/amd/test_deepseek_v4_flash_fp4.py` renamed +1/-1 (2 lines); hunks: -82,7 +82,7 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/amd/test_deepseek_v4_flash_fp8.py` renamed +1/-1 (2 lines); hunks: -82,7 +82,7 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +1/-1 (2 lines); hunks: -84,7 +84,7 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +1/-1 (2 lines); hunks: -84,7 +84,7 @@ def setUpClass(cls):; symbols: setUpClass
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/amd/test_deepseek_v4_flash_fp4.py` renamed +1/-1; `test/registered/amd/test_deepseek_v4_flash_fp8.py` renamed +1/-1; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +1/-1; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24949 - Deepseek-v4-Pro share expert tp1

- Link: https://github.com/sgl-project/sglang/pull/24949
- Status/date: merged / 2026-05-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +31/-17, 112 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deepseek-v4-Pro share expert tp1"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/environ.py`; technical summary: Covers "Deepseek-v4-Pro share expert tp1"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/environ.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +26/-14 (40 lines); hunks: -534,6 +534,7 @@ def __init__(; -543,7 +544,19 @@ def __init__(; symbols: __init__, forward_normal_dual_stream, touching `__init__, forward_normal_dual_stream`; `python/sglang/srt/model_executor/model_runner.py` modified +4/-2 (6 lines); hunks: -1155,8 +1155,10 @@ def check_quantized_moe_compatibility(self):; symbols: check_quantized_moe_compatibility, touching `check_quantized_moe_compatibility`; `python/sglang/srt/environ.py` modified +1/-1 (2 lines); hunks: -611,7 +611,7 @@ class Envs:; symbols: Envs, touching `Envs`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +26/-14 (40 lines); hunks: -534,6 +534,7 @@ def __init__(; -543,7 +544,19 @@ def __init__(; symbols: __init__, forward_normal_dual_stream
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-2 (6 lines); hunks: -1155,8 +1155,10 @@ def check_quantized_moe_compatibility(self):; symbols: check_quantized_moe_compatibility
  - `python/sglang/srt/environ.py` modified +1/-1 (2 lines); hunks: -611,7 +611,7 @@ class Envs:; symbols: Envs
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +26/-14; `python/sglang/srt/model_executor/model_runner.py` modified +4/-2; `python/sglang/srt/environ.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25039 - [AMD] Disable unittest fail-fast for deepseekv4 perf test

- Link: https://github.com/sgl-project/sglang/pull/25039
- Status/date: merged / 2026-05-13
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`; associated commits `72b266d59b39`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +56/-8, 176 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Disable unittest fail-fast for deepseekv4 perf test"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`; technical summary: Covers "[AMD] Disable unittest fail-fast for deepseekv4 perf test"; the main implementation surface is `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +14/-2 (16 lines); hunks: -38,24 +38,28; -204,4 +208,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k, touching `test_b_perf_8k_1k`; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +14/-2 (16 lines); hunks: -38,24 +38,28; -204,4 +208,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k, touching `test_b_perf_8k_1k`; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +14/-2 (16 lines); hunks: -40,24 +40,28; -206,4 +210,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k, touching `test_b_perf_8k_1k`; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +14/-2 (16 lines); hunks: -40,24 +40,28; -206,4 +210,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k, touching `test_b_perf_8k_1k`.
- Code diff details:
  - `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +14/-2 (16 lines); hunks: -38,24 +38,28; -204,4 +208,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +14/-2 (16 lines); hunks: -38,24 +38,28; -204,4 +208,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +14/-2 (16 lines); hunks: -40,24 +40,28; -206,4 +210,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +14/-2 (16 lines); hunks: -40,24 +40,28; -206,4 +210,12 @@ def test_b_perf_8k_1k(self):; symbols: test_b_perf_8k_1k
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +14/-2; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +14/-2; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +14/-2; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +14/-2
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25152 - docs: prepend SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 for H200 FP8 Flash max-throughput

- Link: https://github.com/sgl-project/sglang/pull/25152
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: prepend SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 for H200 FP8 Flash max-throughput"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "docs: prepend SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 for H200 FP8 Flash max-throughput"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -391,6 +391,9 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -391,6 +391,9 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -391,6 +391,9 @@ export const DeepSeekV4Deployment = () => {
+        if (!isBig) {
+          recipeEnv.push("SGLANG_JIT_DEEPGEMM_PRECOMPILE=0");
+        }
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #24897 - Port fused SiLU+clamp+FP8 quant from DSV4 dev branch

- Link: https://github.com/sgl-project/sglang/pull/24897
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +51/-6, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Port fused SiLU+clamp+FP8 quant from DSV4 dev branch"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "Port fused SiLU+clamp+FP8 quant from DSV4 dev branch"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunks: -27,6 +27,10; -107,6 +111,9; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunks: -27,6 +27,10; -107,6 +111,9; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +51/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24890 - Port KV Compression V2 from deepseek_v4_dev

- Link: https://github.com/sgl-project/sglang/pull/24890
- Status/date: merged / 2026-05-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `e2290b155aa0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 23 files, +5201/-438, 6145 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Port KV Compression V2 from deepseek_v4_dev"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/jit_kernel/deepseek_v4.py`; technical summary: Covers "Port KV Compression V2 from deepseek_v4_dev"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/jit_kernel/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +91/-80 (171 lines); hunks: -11,7 +11,11; -25,7 +29,6; symbols: __init__, _compute_q_a, _compute_q_b, _compute_kv_to_cache, touching `__init__, _compute_q_a, _compute_q_b`; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-5 (20 lines); hunks: -20,11 +20,21; `python/sglang/jit_kernel/deepseek_v4.py` modified +127/-2 (129 lines); hunks: -195,6 +195,52 @@ def _jit_fused_store_module(; -571,6 +617,26 @@ def compress_fused_norm_rope_inplace(; symbols: _jit_fused_store_module, _jit_main_q_norm_rope_module, _jit_main_k_norm_rope_flashmla_module, _jit_main_q_indexer_rope_hadamard_quant_module, touching `_jit_fused_store_module, _jit_main_q_norm_rope_module, _jit_main_k_norm_rope_flashmla_module`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +37/-2 (39 lines); hunks: -6,7 +6,7; -630,7 +630,12 @@ def set_swa_key_buffer(; symbols: set_swa_key_buffer, get_extra_key_buffer, get_extra_key_page_size, set_extra_key_buffer, touching `set_swa_key_buffer, get_extra_key_buffer, get_extra_key_page_size`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +91/-80 (171 lines); hunks: -11,7 +11,11; -25,7 +29,6; symbols: __init__, _compute_q_a, _compute_q_b, _compute_kv_to_cache
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-5 (20 lines); hunks: -20,11 +20,21
  - `python/sglang/jit_kernel/deepseek_v4.py` modified +127/-2 (129 lines); hunks: -195,6 +195,52 @@ def _jit_fused_store_module(; -571,6 +617,26 @@ def compress_fused_norm_rope_inplace(; symbols: _jit_fused_store_module, _jit_main_q_norm_rope_module, _jit_main_k_norm_rope_flashmla_module, _jit_main_q_indexer_rope_hadamard_quant_module
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +37/-2 (39 lines); hunks: -6,7 +6,7; -630,7 +630,12 @@ def set_swa_key_buffer(; symbols: set_swa_key_buffer, get_extra_key_buffer, get_extra_key_page_size, set_extra_key_buffer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +91/-80; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-5; `python/sglang/jit_kernel/deepseek_v4.py` modified +127/-2; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +37/-2
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/deepseek_v4/__init__.py`, `python/sglang/jit_kernel/tests/deepseek_v4/common.py`, `python/sglang/jit_kernel/tests/deepseek_v4/test_c128_v2.py`, `python/sglang/jit_kernel/tests/deepseek_v4/test_c4_v2.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24816 - Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4

- Link: https://github.com/sgl-project/sglang/pull/24816
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +1542/-3, 1649 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py`; technical summary: Covers "Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4"; the main implementation surface is `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py` added +544/-0 (544 lines); hunks: -0,0 +1,544; symbols: _MockLayer, _MockTopKOutput, __init__, _make_random_mxfp4, touching `_MockLayer, _MockTopKOutput, __init__`; `python/sglang/srt/layers/quantization/mxfp4.py` modified +269/-1 (270 lines); hunks: -16,12 +16,18; -62,7 +68,27; symbols: __init__, create_weights, process_weights_after_loading, touching `__init__, create_weights, process_weights_after_loading`; `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: Mxfp4FlashinferCutlassMoEMethod, __init__, create_weights, create_moe_runner, touching `Mxfp4FlashinferCutlassMoEMethod, __init__, create_weights`; `python/sglang/srt/layers/moe/topk.py` modified +12/-0 (12 lines); hunks: -243,6 +243,18 @@ class BypassedTopKOutput(NamedTuple):; symbols: BypassedTopKOutput, format, to_standard, touching `BypassedTopKOutput, format, to_standard`.
- Code diff details:
  - `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py` added +544/-0 (544 lines); hunks: -0,0 +1,544; symbols: _MockLayer, _MockTopKOutput, __init__, _make_random_mxfp4
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +269/-1 (270 lines); hunks: -16,12 +16,18; -62,7 +68,27; symbols: __init__, create_weights, process_weights_after_loading
  - `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: Mxfp4FlashinferCutlassMoEMethod, __init__, create_weights, create_moe_runner
  - `python/sglang/srt/layers/moe/topk.py` modified +12/-0 (12 lines); hunks: -243,6 +243,18 @@ class BypassedTopKOutput(NamedTuple):; symbols: BypassedTopKOutput, format, to_standard
  - `python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py` modified +9/-1 (10 lines); hunks: -445,12 +445,20 @@ def maybe_fuse_routed_scale_and_shared_add(; symbols: maybe_fuse_routed_scale_and_shared_add
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py` added +544/-0; `python/sglang/test/bench_mxfp4_sm90_kernels.py` added +366/-0; `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py` modified +70/-1
  - runtime: `python/sglang/srt/layers/quantization/mxfp4.py` modified +269/-1; `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py` added +263/-0; `python/sglang/srt/layers/moe/topk.py` modified +12/-0; `python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py` modified +9/-1; `python/sglang/srt/layers/quantization/fp8.py` modified +9/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/bench_mxfp4_sm90_kernels.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25001 - [LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support

- Link: https://github.com/sgl-project/sglang/pull/25001
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +1013/-0, 1081 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`; technical summary: Covers "[LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-0 (15 lines); hunks: -13,6 +13,15; -350,6 +359,8 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, touching `forward_absorb_prepare, forward_absorb_core`; `python/sglang/srt/models/deepseek_v2.py` modified +4/-0 (4 lines); hunks: -1687,11 +1687,15 @@ def prepare_qkv_latent(; symbols: prepare_qkv_latent, touching `prepare_qkv_latent`; `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` added +849/-0 (849 lines); hunks: -0,0 +1,849; symbols: _num_segments, _max_segment_len, _segment_grid_size, _step_a_q_kernel, touching `_num_segments, _max_segment_len, _segment_grid_size`; `python/sglang/srt/lora/deepseek_mla_correction.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: is_kv_b_lora_active, _get_state, apply_q_correction, apply_v_correction, touching `is_kv_b_lora_active, _get_state, apply_q_correction`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-0 (15 lines); hunks: -13,6 +13,15; -350,6 +359,8 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/models/deepseek_v2.py` modified +4/-0 (4 lines); hunks: -1687,11 +1687,15 @@ def prepare_qkv_latent(; symbols: prepare_qkv_latent
  - `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` added +849/-0 (849 lines); hunks: -0,0 +1,849; symbols: _num_segments, _max_segment_len, _segment_grid_size, _step_a_q_kernel
  - `python/sglang/srt/lora/deepseek_mla_correction.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: is_kv_b_lora_active, _get_state, apply_q_correction, apply_v_correction
  - `python/sglang/srt/lora/utils.py` modified +14/-0 (14 lines); hunks: -134,6 +134,18 @@ def get_hidden_dim(; -274,6 +286,8 @@ def get_target_module_name(full_module_name: str, target_mod...; symbols: get_hidden_dim, get_target_module_name
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-0; `python/sglang/srt/models/deepseek_v2.py` modified +4/-0; `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` added +849/-0; `python/sglang/srt/lora/deepseek_mla_correction.py` added +117/-0; `python/sglang/srt/lora/utils.py` modified +14/-0; `python/sglang/srt/lora/triton_ops/__init__.py` modified +10/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/lora/deepseek_mla_correction.py`, `python/sglang/srt/lora/triton_ops/__init__.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24986 - [rebase]Deepseek_v4 support w4(mxfp4)a16 on hopper

- Link: https://github.com/sgl-project/sglang/pull/24986
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +146/-36, 295 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[rebase]Deepseek_v4 support w4(mxfp4)a16 on hopper"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/mxfp4.py`; technical summary: Covers "[rebase]Deepseek_v4 support w4(mxfp4)a16 on hopper"; the main implementation surface is `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/mxfp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py` modified +57/-12 (69 lines); hunks: -8,7 +8,7; -38,17 +38,62 @@ def create_weights(; symbols: create_weights, process_weights_after_loading, apply, touching `create_weights, process_weights_after_loading, apply`; `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` modified +32/-16 (48 lines); hunks: -52,22 +52,38 @@ def _normalize_scale_tensor(; -129,19 +145,19 @@ def _permute_bias(bias: torch.Tensor | None) -> torch.Tens...; symbols: _normalize_scale_tensor, _get_optional_param, prepare_moe_mxfp4_layer_for_marlin, _permute_bias, touching `_normalize_scale_tensor, _get_optional_param, prepare_moe_mxfp4_layer_for_marlin`; `python/sglang/srt/layers/quantization/mxfp4.py` modified +40/-1 (41 lines); hunks: -35,6 +35,7; -342,6 +343,7 @@ def __init__(; symbols: __init__, create_weights, process_weights_after_loading, create_moe_runner, touching `__init__, create_weights, process_weights_after_loading`; `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +3/-7 (10 lines); hunks: -119,13 +119,9 @@ def fused_marlin_moe(; symbols: fused_marlin_moe, touching `fused_marlin_moe`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py` modified +57/-12 (69 lines); hunks: -8,7 +8,7; -38,17 +38,62 @@ def create_weights(; symbols: create_weights, process_weights_after_loading, apply
  - `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` modified +32/-16 (48 lines); hunks: -52,22 +52,38 @@ def _normalize_scale_tensor(; -129,19 +145,19 @@ def _permute_bias(bias: torch.Tensor | None) -> torch.Tens...; symbols: _normalize_scale_tensor, _get_optional_param, prepare_moe_mxfp4_layer_for_marlin, _permute_bias
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +40/-1 (41 lines); hunks: -35,6 +35,7; -342,6 +343,7 @@ def __init__(; symbols: __init__, create_weights, process_weights_after_loading, create_moe_runner
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +3/-7 (10 lines); hunks: -119,13 +119,9 @@ def fused_marlin_moe(; symbols: fused_marlin_moe
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` modified +10/-0 (10 lines); hunks: -1006,6 +1006,16 @@ void moe_wna16_marlin_gemm(
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py` modified +57/-12; `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` modified +32/-16; `python/sglang/srt/layers/quantization/mxfp4.py` modified +40/-1; `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +3/-7; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` modified +10/-0
  - tests: `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py` modified +2/-0; `test/registered/dsv4/test_deepseek_v4_flash_fp8_h200.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp8_h200.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24925 - [attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell)

- Link: https://github.com/sgl-project/sglang/pull/24925
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +462/-92, 726 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell)"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`; technical summary: Covers "[attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell)"; the main implementation surface is `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` added +247/-0 (247 lines); hunks: -0,0 +1,247; symbols: _get_tokenspeed_workspace, TokenspeedMLABackend, __init__, _ensure_workspace, touching `_get_tokenspeed_workspace, TokenspeedMLABackend, __init__`; `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +132/-91 (223 lines); hunks: -755,6 +755,109 @@ def unpad_draft_extend_output(; -838,46 +941,13 @@ def forward_decode(; symbols: unpad_draft_extend_output, _compute_decode_bmm1_scale, _run_decode_kernel, _run_prefill_kernel, touching `unpad_draft_extend_output, _compute_decode_bmm1_scale, _run_decode_kernel`; `python/sglang/srt/layers/attention/attention_registry.py` modified +11/-0 (11 lines); hunks: -62,6 +62,17 @@ def create_trtllm_mla_backend(runner):; symbols: create_trtllm_mla_backend, create_tokenspeed_mla_backend, create_aiter_backend, touching `create_trtllm_mla_backend, create_tokenspeed_mla_backend, create_aiter_backend`; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +7/-0 (7 lines); hunks: -134,6 +134,12 @@ def handle_attention_trtllm_mla(attn, forward_batch):; -183,6 +189,7 @@ def handle_attention_intel_xpu(attn, forward_batch):; symbols: handle_attention_trtllm_mla, handle_attention_tokenspeed_mla, handle_attention_aiter, handle_attention_intel_xpu, touching `handle_attention_trtllm_mla, handle_attention_tokenspeed_mla, handle_attention_aiter`.
- Code diff details:
  - `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` added +247/-0 (247 lines); hunks: -0,0 +1,247; symbols: _get_tokenspeed_workspace, TokenspeedMLABackend, __init__, _ensure_workspace
  - `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +132/-91 (223 lines); hunks: -755,6 +755,109 @@ def unpad_draft_extend_output(; -838,46 +941,13 @@ def forward_decode(; symbols: unpad_draft_extend_output, _compute_decode_bmm1_scale, _run_decode_kernel, _run_prefill_kernel
  - `python/sglang/srt/layers/attention/attention_registry.py` modified +11/-0 (11 lines); hunks: -62,6 +62,17 @@ def create_trtllm_mla_backend(runner):; symbols: create_trtllm_mla_backend, create_tokenspeed_mla_backend, create_aiter_backend
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +7/-0 (7 lines); hunks: -134,6 +134,12 @@ def handle_attention_trtllm_mla(attn, forward_batch):; -183,6 +189,7 @@ def handle_attention_intel_xpu(attn, forward_batch):; symbols: handle_attention_trtllm_mla, handle_attention_tokenspeed_mla, handle_attention_aiter, handle_attention_intel_xpu
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-0 (2 lines); hunks: -244,6 +244,7; -256,6 +257,7
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` added +247/-0; `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +132/-91; `python/sglang/srt/layers/attention/attention_registry.py` modified +11/-0; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +7/-0; `python/sglang/srt/model_executor/model_runner.py` modified +2/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/pyproject.toml`, `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25052 - DeepSeek V4 w4a4 MegaMoE

- Link: https://github.com/sgl-project/sglang/pull/25052
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +212/-60, 328 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DeepSeek V4 w4a4 MegaMoE"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/mega_moe.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py`; technical summary: Covers "DeepSeek V4 w4a4 MegaMoE"; the main implementation surface is `python/sglang/srt/layers/moe/mega_moe.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/mega_moe.py` modified +52/-10 (62 lines); hunks: -15,6 +15,7; -34,6 +35,26; symbols: _apply_mega_moe_dg_env, _get_mega_moe_symm_buffer, _run_mega_routed, touching `_apply_mega_moe_dg_env, _get_mega_moe_symm_buffer, _run_mega_routed`; `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py` added +148/-0 (148 lines); hunks: -0,0 +1,148; symbols: _gsm8k_check, TestDSV4FlashFP4B200W4A8MegaMoE, setUpClass, tearDownClass, touching `_gsm8k_check, TestDSV4FlashFP4B200W4A8MegaMoE, setUpClass`; `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py` modified +0/-49 (49 lines); hunks: -31,14 +31,6; -138,46 +130,5 @@ def test_gsm8k(self):; symbols: _gsm8k_check, test_gsm8k, TestDSV4FlashFP4B200MegaMoE, setUpClass, touching `_gsm8k_check, test_gsm8k, TestDSV4FlashFP4B200MegaMoE`; `python/sglang/srt/environ.py` modified +11/-0 (11 lines); hunks: -595,6 +595,17 @@ class Envs:; symbols: Envs, touching `Envs`.
- Code diff details:
  - `python/sglang/srt/layers/moe/mega_moe.py` modified +52/-10 (62 lines); hunks: -15,6 +15,7; -34,6 +35,26; symbols: _apply_mega_moe_dg_env, _get_mega_moe_symm_buffer, _run_mega_routed
  - `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py` added +148/-0 (148 lines); hunks: -0,0 +1,148; symbols: _gsm8k_check, TestDSV4FlashFP4B200W4A8MegaMoE, setUpClass, tearDownClass
  - `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py` modified +0/-49 (49 lines); hunks: -31,14 +31,6; -138,46 +130,5 @@ def test_gsm8k(self):; symbols: _gsm8k_check, test_gsm8k, TestDSV4FlashFP4B200MegaMoE, setUpClass
  - `python/sglang/srt/environ.py` modified +11/-0 (11 lines); hunks: -595,6 +595,17 @@ class Envs:; symbols: Envs
  - `python/pyproject.toml` modified +1/-1 (2 lines); hunks: -59,7 +59,7 @@ dependencies = [
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/mega_moe.py` modified +52/-10; `python/sglang/srt/environ.py` modified +11/-0; `python/pyproject.toml` modified +1/-1
  - tests: `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py` added +148/-0; `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py` modified +0/-49
- Risk and verification: The diff ships test coverage in `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25243 - [Docs] update dsv4 cookbook with H100 deployment commands

- Link: https://github.com/sgl-project/sglang/pull/25243
- Status/date: merged / 2026-05-14
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `1f119f6a4463`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +83/-9, 153 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] update dsv4 cookbook with H100 deployment commands"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[Docs] update dsv4 cookbook with H100 deployment commands"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -100,6 +100,10 @@ Please refer to the [official SGLang installation guide](......
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -100,6 +100,10 @@ Please refer to the [official SGLang installation guide](.....
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -100,6 +100,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+    <tr>
+      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>NVIDIA H100</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>lmsysorg/sglang:dev</code></td>
+    </tr>
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #24691 - [UnifiedTree]: Support HiCache For DeepSeek_V4

- Link: https://github.com/sgl-project/sglang/pull/24691
- Status/date: merged / 2026-05-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; associated commits `d9fa84b25b79`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +1221/-154, 1970 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[UnifiedTree]: Support HiCache For DeepSeek_V4"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "[UnifiedTree]: Support HiCache For DeepSeek_V4"; the main implementation surface is `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +12/-1 (13 lines); hunks: -605,21 +605,28 @@ def _init_compressed_layer_mapping(self):; -635,7 +642,8 @@ def get_extra_key_page_size(self, layer_id: int) -> int:; symbols: _init_compressed_layer_mapping, wait_layer_transfer, get_attention_compress_states, get_indexer_compress_states, touching `_init_compressed_layer_mapping, wait_layer_transfer, get_attention_compress_states`.
- Code diff details:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +12/-1 (13 lines); hunks: -605,21 +605,28 @@ def _init_compressed_layer_mapping(self):; -635,7 +642,8 @@ def get_extra_key_page_size(self, layer_id: int) -> int:; symbols: _init_compressed_layer_mapping, wait_layer_transfer, get_attention_compress_states, get_indexer_compress_states
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +12/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/kl_multiturn_utils.py`, `python/sglang/test/kl_test_utils.py`, `test/registered/radix_cache/test_unified_radix_cache_kl.py`, `test/registered/radix_cache/test_unified_radix_hicache_kl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25369 - Add hicache feature in dsv4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/25369
- Status/date: merged / 2026-05-15
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `c7e879e43f77`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +50/-4, 95 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add hicache feature in dsv4 cookbook"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "Add hicache feature in dsv4 cookbook"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +11/-0 (11 lines); hunks: -334,6 +334,17 @@ print().
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +11/-0 (11 lines); hunks: -334,6 +334,17 @@ print()
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -334,6 +334,17 @@ print()
+#### 4.2.3 HiCache (Hierarchical KV Caching)
+HiCache enables multi-tier KV cache offloading (GPU → CPU → Storage), significantly expanding effective context capacity for long-context and multi-turn scenarios. Combined with U
+To enable HiCache, use the **HiCache** toggle in the [command generator above](#3-model-deployment):
+- **L2 (GPU + CPU):** Offloads cold KV pages to CPU memory. Enables `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` for intelligent hierarchical prefix caching.
+- **L3 (GPU + CPU + Storage):** Coming soon.
+For more details, see the [HiCache documentation](../../../docs/advanced_features/hicache).
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +11/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #25419 - Port SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN from deepseek_v4_dev

- Link: https://github.com/sgl-project/sglang/pull/25419
- Status/date: merged / 2026-05-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-1, 23 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Port SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN from deepseek_v4_dev"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/environ.py`; technical summary: Covers "Port SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN from deepseek_v4_dev"; the main implementation surface is `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/environ.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/managers/schedule_batch.py` modified +5/-1 (6 lines); hunks: -2724,9 +2724,13 @@ def _evict_swa(self, req: Req, pre_len: int):; symbols: _evict_swa, touching `_evict_swa`; `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -596,6 +596,7 @@ class Envs:; symbols: Envs, touching `Envs`.
- Code diff details:
  - `python/sglang/srt/managers/schedule_batch.py` modified +5/-1 (6 lines); hunks: -2724,9 +2724,13 @@ def _evict_swa(self, req: Req, pre_len: int):; symbols: _evict_swa
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -596,6 +596,7 @@ class Envs:; symbols: Envs
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/managers/schedule_batch.py` modified +5/-1; `python/sglang/srt/environ.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/managers/schedule_batch.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24704 - feat: add Pipeline Parallelism (PP) and PD support for DeepSeek-V4

- Link: https://github.com/sgl-project/sglang/pull/24704
- Status/date: merged / 2026-05-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `162540e0a8d3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +341/-103, 750 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: add Pipeline Parallelism (PP) and PD support for DeepSeek-V4"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "feat: add Pipeline Parallelism (PP) and PD support for DeepSeek-V4"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +99/-39 (138 lines); hunks: -2,7 +2,16; -49,7 +58,7; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +71/-51 (122 lines); hunks: -401,6 +401,19 @@ def __init__(; -412,8 +425,8 @@ def __init__(; symbols: __init__, register_mapping, get_state_buf_infos, touching `__init__, register_mapping, get_state_buf_infos`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +99/-39 (138 lines); hunks: -2,7 +2,16; -49,7 +58,7; symbols: __init__, forward
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +71/-51 (122 lines); hunks: -401,6 +401,19 @@ def __init__(; -412,8 +425,8 @@ def __init__(; symbols: __init__, register_mapping, get_state_buf_infos
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +99/-39; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +71/-51
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/disaggregation/base/conn.py`, `python/sglang/srt/disaggregation/common/conn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25477 - [BugFix]: Fix DeepSeek V4 HiCache layer count logic

- Link: https://github.com/sgl-project/sglang/pull/25477
- Status/date: merged / 2026-05-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +161/-144, 349 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix]: Fix DeepSeek V4 HiCache layer count logic"; model line: DeepSeek V4; category: bug fix; main diff: `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`; technical summary: Covers "[BugFix]: Fix DeepSeek V4 HiCache layer count logic"; the main implementation surface is `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: TestUnifiedMambaHiCache, setUpClass, tearDownClass, _assert_dsv4_decode_cached_tokens, touching `TestUnifiedMambaHiCache, setUpClass, tearDownClass`; `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py` renamed +0/-141 (141 lines); hunks: -13,162 +13,21; symbols: TestUnifiedMambaHiCache, setUpClass, tearDownClass, _assert_dsv4_decode_cached_tokens, touching `TestUnifiedMambaHiCache, setUpClass, tearDownClass`; `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +6/-3 (9 lines); hunks: -283,7 +283,8 @@ def build_deepseek_v4_hicache_stack(; -293,7 +294,9 @@ def build_deepseek_v4_hicache_stack(; symbols: build_deepseek_v4_hicache_stack, attach_hybrid_pool_to_unified_cache, touching `build_deepseek_v4_hicache_stack, attach_hybrid_pool_to_unified_cache`.
- Code diff details:
  - `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: TestUnifiedMambaHiCache, setUpClass, tearDownClass, _assert_dsv4_decode_cached_tokens
  - `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py` renamed +0/-141 (141 lines); hunks: -13,162 +13,21; symbols: TestUnifiedMambaHiCache, setUpClass, tearDownClass, _assert_dsv4_decode_cached_tokens
  - `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +6/-3 (9 lines); hunks: -283,7 +283,8 @@ def build_deepseek_v4_hicache_stack(; -293,7 +294,9 @@ def build_deepseek_v4_hicache_stack(; symbols: build_deepseek_v4_hicache_stack, attach_hybrid_pool_to_unified_cache
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` added +155/-0; `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py` renamed +0/-141
  - runtime: `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +6/-3
- Risk and verification: The diff ships test coverage in `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25410 - [Docs] Update DeepSeek V4 cookbook to use the latest docker image

- Link: https://github.com/sgl-project/sglang/pull/25410
- Status/date: merged / 2026-05-16
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `9f26697d6a6a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-41, 63 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Update DeepSeek V4 cookbook to use the latest docker image"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[Docs] Update DeepSeek V4 cookbook to use the latest docker image"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-41 (47 lines); hunks: -66,48 +66,13 @@ SGLang offers multiple installation methods. Choose based on...; -116,7 +81,7 @@ docker run --gpus all \.
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-41 (47 lines); hunks: -66,48 +66,13 @@ SGLang offers multiple installation methods. Choose based on...; -116,7 +81,7 @@ docker run --gpus all \
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-41
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #25412 - [Doc] DSV4 cookbook: clean up env vars, add MegaMoE toggle, unify docker image

- Link: https://github.com/sgl-project/sglang/pull/25412
- Status/date: merged / 2026-05-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +38/-83, 185 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Doc] DSV4 cookbook: clean up env vars, add MegaMoE toggle, unify docker image"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "[Doc] DSV4 cookbook: clean up env vars, add MegaMoE toggle, unify docker image"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +38/-83 (121 lines); hunks: -79,6 +79,15 @@ export const DeepSeekV4Deployment = () => {; -303,7 +312,7 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +38/-83 (121 lines); hunks: -79,6 +79,15 @@ export const DeepSeekV4Deployment = () => {; -303,7 +312,7 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +38/-83
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #25506 - [Doc] Fix several places for dpsk v4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/25506
- Status/date: merged / 2026-05-17
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `6dcacb1159d6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +47/-1, 83 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Doc] Fix several places for dpsk v4 cookbook"; model line: DeepSeek V4; category: bug fix; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[Doc] Fix several places for dpsk v4 cookbook"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +21/-1 (22 lines); hunks: -120,14 +120,34 @@ The generator currently picks values on the **conservative....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +21/-1 (22 lines); hunks: -120,14 +120,34 @@ The generator currently picks values on the **conservative...
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -120,14 +120,34 @@ The generator currently picks values on the **conservative** side (mirroring an
-- Original FP4 checkpoints: To run original FP4 checkpoints, apply the w4a16 MoE kernels (marlin) as in interactive command generator. For this option we only support TP method. C
+- Original FP4 checkpoints: To run original FP4 checkpoints, we provide two different options for w4a16 MoE kernels: Marlin (`--moe-runner-backend marlin`) and Flashinfer (`--moe-
+**MegaMoE**
+MegaMoE fuses expert dispatch + GEMM into a single kernel for higher throughput
+on MoE layers. To enable it, use the **MegaMoE** toggle in the
+[command generator above](#3-model-deployment) — the generator will swap
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +21/-1
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #25569 - Add DeepSeekV4 fused MoE Triton autotune support

- Link: https://github.com/sgl-project/sglang/pull/25569
- Status/date: merged / 2026-05-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-0, 29 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DeepSeekV4 fused MoE Triton autotune support"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `benchmark/kernels/fused_moe_triton/common_utils.py`; technical summary: Covers "Add DeepSeekV4 fused MoE Triton autotune support"; the main implementation surface is `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `benchmark/kernels/fused_moe_triton/common_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +5/-0 (5 lines); hunks: -35,6 +35,7; -174,8 +175,12 @@ def prepare(i: int):; symbols: prepare, run, touching `prepare, run`; `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunks: -85,6 +85,7 @@ def get_model_config(; symbols: get_model_config, touching `get_model_config`.
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +5/-0 (5 lines); hunks: -35,6 +35,7; -174,8 +175,12 @@ def prepare(i: int):; symbols: prepare, run
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunks: -85,6 +85,7 @@ def get_model_config(; symbols: get_model_config
- Key code excerpts:

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

- Reviewed files:
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +5/-0; `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #24933 - Amd/deepseek v4 rebase main 0509

- Link: https://github.com/sgl-project/sglang/pull/24933
- Status/date: merged / 2026-05-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `866793c502b7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +3678/-70, 4186 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Amd/deepseek v4 rebase main 0509"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`; technical summary: Covers "Amd/deepseek v4 rebase main 0509"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` added +1265/-0 (1265 lines); hunks: -0,0 +1,1265; symbols: _pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadata, touching `_pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data`; `python/sglang/srt/models/deepseek_v4.py` modified +53/-5 (58 lines); hunks: -58,6 +58,7; -76,6 +77,12; symbols: __init__, _forward_prepare_multi_stream, _forward_prepare, touching `__init__, _forward_prepare_multi_stream, _forward_prepare`; `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +88/-21 (109 lines); hunks: -7,8 +7,11; -22,16 +25,55 @@ def kv(self) -> torch.Tensor:; symbols: KVAndScore, kv, score, shape, touching `KVAndScore, kv, score`; `python/sglang/jit_kernel/deepseek_v4.py` modified +26/-0 (26 lines); hunks: -13,6 +13,13; -644,6 +651,23 @@ def fused_rope(; symbols: fused_rope, _dispatch_bf16_fp32_backend, touching `fused_rope, _dispatch_bf16_fp32_backend`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` added +1265/-0 (1265 lines); hunks: -0,0 +1,1265; symbols: _pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadata
  - `python/sglang/srt/models/deepseek_v4.py` modified +53/-5 (58 lines); hunks: -58,6 +58,7; -76,6 +77,12; symbols: __init__, _forward_prepare_multi_stream, _forward_prepare
  - `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +88/-21 (109 lines); hunks: -7,8 +7,11; -22,16 +25,55 @@ def kv(self) -> torch.Tensor:; symbols: KVAndScore, kv, score, shape
  - `python/sglang/jit_kernel/deepseek_v4.py` modified +26/-0 (26 lines); hunks: -13,6 +13,13; -644,6 +651,23 @@ def fused_rope(; symbols: fused_rope, _dispatch_bf16_fp32_backend
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +13/-4 (17 lines); hunks: -18,11 +18,13; -144,6 +146,9 @@ def set_key_buffer_fused(; symbols: get_compress_state_ring_size, set_key_buffer_fused, get_key_buffer, set_kv_buffer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` added +1265/-0; `python/sglang/srt/models/deepseek_v4.py` modified +53/-5; `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +88/-21; `python/sglang/jit_kernel/deepseek_v4.py` modified +26/-0; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +13/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/deepseek_v4.py`, `python/sglang/srt/environ.py`, `python/sglang/srt/layers/attention/attention_registry.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25282 - [UnifiedTree] Support deepseek v4 host pool layout

- Link: https://github.com/sgl-project/sglang/pull/25282
- Status/date: merged / 2026-05-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +401/-114, 809 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[UnifiedTree] Support deepseek v4 host pool layout"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/test/kl_multiturn_utils.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`; technical summary: Covers "[UnifiedTree] Support deepseek v4 host pool layout"; the main implementation surface is `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/test/kl_multiturn_utils.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/mem_cache/memory_pool_host.py` modified +333/-92 (425 lines); hunks: -1754,6 +1754,7 @@ def __init__(; -1769,7 +1770,7 @@ def __init__(; symbols: __init__, _to_page_indices, _check_io_backend, touching `__init__, _to_page_indices, _check_io_backend`; `python/sglang/test/kl_multiturn_utils.py` modified +40/-19 (59 lines); hunks: -2,6 +2,7; -145,30 +146,45 @@ def _interleave_order(n: int, branches_per_group: int) ->...; symbols: _interleave_order, _generate_maybe_interleaved, test_input_output_logprobs_match_decode_cache_hit_helper, touching `_interleave_order, _generate_maybe_interleaved, test_input_output_logprobs_match_decode_cache_hit_helper`; `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` modified +17/-3 (20 lines); hunks: -92,8 +92,13 @@ def _assert_dsv4_decode_cached_tokens(result, history_len, ou...; -129,15 +134,15 @@ def setUpClass(cls):; symbols: _assert_dsv4_decode_cached_tokens, TestUnifiedDeepSeekV4FlashHiCache, setUpClass, tearDownClass, touching `_assert_dsv4_decode_cached_tokens, TestUnifiedDeepSeekV4FlashHiCache, setUpClass`; `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +7/-0 (7 lines); hunks: -325,6 +325,7 @@ def build_deepseek_v4_hicache_stack(; -357,6 +358,7 @@ def build_deepseek_v4_hicache_stack(; symbols: build_deepseek_v4_hicache_stack, touching `build_deepseek_v4_hicache_stack`.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool_host.py` modified +333/-92 (425 lines); hunks: -1754,6 +1754,7 @@ def __init__(; -1769,7 +1770,7 @@ def __init__(; symbols: __init__, _to_page_indices, _check_io_backend
  - `python/sglang/test/kl_multiturn_utils.py` modified +40/-19 (59 lines); hunks: -2,6 +2,7; -145,30 +146,45 @@ def _interleave_order(n: int, branches_per_group: int) ->...; symbols: _interleave_order, _generate_maybe_interleaved, test_input_output_logprobs_match_decode_cache_hit_helper
  - `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` modified +17/-3 (20 lines); hunks: -92,8 +92,13 @@ def _assert_dsv4_decode_cached_tokens(result, history_len, ou...; -129,15 +134,15 @@ def setUpClass(cls):; symbols: _assert_dsv4_decode_cached_tokens, TestUnifiedDeepSeekV4FlashHiCache, setUpClass, tearDownClass
  - `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +7/-0 (7 lines); hunks: -325,6 +325,7 @@ def build_deepseek_v4_hicache_stack(; -357,6 +358,7 @@ def build_deepseek_v4_hicache_stack(; symbols: build_deepseek_v4_hicache_stack
  - `test/registered/radix_cache/test_unified_radix_cache_kl.py` modified +4/-0 (4 lines); hunks: -49,6 +49,8 @@ class UnifiedRadixTreeTestMixin:; -163,6 +165,8 @@ def test_multiturn_decode_cache_hit_branching(self):; symbols: UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/memory_pool_host.py` modified +333/-92; `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +7/-0
  - tests: `python/sglang/test/kl_multiturn_utils.py` modified +40/-19; `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` modified +17/-3; `test/registered/radix_cache/test_unified_radix_cache_kl.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/kl_multiturn_utils.py`, `test/registered/radix_cache/test_unified_radix_cache_kl.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25733 - [Bug] Fix V4-Pro NaN on Blackwell by converting fp8_einsum input scale to ue8m0

- Link: https://github.com/sgl-project/sglang/pull/25733
- Status/date: merged / 2026-05-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `79ea30d1f134`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix V4-Pro NaN on Blackwell by converting fp8_einsum input scale to ue8m0"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[Bug] Fix V4-Pro NaN on Blackwell by converting fp8_einsum input scale to ue8m0"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +1/-0 (1 lines); hunks: -623,6 +623,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +1/-0 (1 lines); hunks: -623,6 +623,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -623,6 +623,7 @@ def forward(
+            o_s = deep_gemm.ceil_to_ue8m0(o_s)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25396 - fix: fix deepseek v4 CP error

- Link: https://github.com/sgl-project/sglang/pull/25396
- Status/date: merged / 2026-05-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `7e0818038a45`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: fix deepseek v4 CP error"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "fix: fix deepseek v4 CP error"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +1/-0 (1 lines); hunks: -388,6 +388,7 @@ def _compute_kv_bf16(; symbols: _compute_kv_bf16, touching `_compute_kv_bf16`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +1/-0 (1 lines); hunks: -388,6 +388,7 @@ def _compute_kv_bf16(; symbols: _compute_kv_bf16
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -388,6 +388,7 @@ def _compute_kv_bf16(
+        kv = kv.contiguous()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25729 - fix(dsv4): upgrade forward metadata on main stream for large PP size

- Link: https://github.com/sgl-project/sglang/pull/25729
- Status/date: merged / 2026-05-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `8322fe09a7b6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-0, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(dsv4): upgrade forward metadata on main stream for large PP size"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "fix(dsv4): upgrade forward metadata on main stream for large PP size"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +4/-0 (4 lines); hunks: -1045,6 +1045,10 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +4/-0 (4 lines); hunks: -1045,6 +1045,10 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -1045,6 +1045,10 @@ def forward(
+        # Upgrade lazy raw metadata on the main stream once before any layer
+        # forks alt-streams; later per-layer calls become no-ops.
+        forward_batch.attn_backend._maybe_upgrade_forward_metadata()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24934 - DeepSeek V4 MTP Support CP

- Link: https://github.com/sgl-project/sglang/pull/24934
- Status/date: merged / 2026-05-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4_nextn.py`; associated commits `425dffbde339`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +105/-0, 163 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DeepSeek V4 MTP Support CP"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v4_nextn.py`; technical summary: Covers "DeepSeek V4 MTP Support CP"; the main implementation surface is `python/sglang/srt/models/deepseek_v4_nextn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4_nextn.py` modified +59/-0 (59 lines); hunks: -7,9 +7,17; -18,6 +26,12; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +59/-0 (59 lines); hunks: -7,9 +7,17; -18,6 +26,12; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4_nextn.py` modified +59/-0
- Risk and verification: The diff ships test coverage in `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25771 - fix(dsv4): drop stale pp_size=1 guard for V4 PD disaggregation

- Link: https://github.com/sgl-project/sglang/pull/25771
- Status/date: merged / 2026-05-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/arg_groups/deepseek_v4_hook.py`; associated commits `ca29c2b0e79e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-7, 14 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(dsv4): drop stale pp_size=1 guard for V4 PD disaggregation"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/arg_groups/deepseek_v4_hook.py`; technical summary: Covers "fix(dsv4): drop stale pp_size=1 guard for V4 PD disaggregation"; the main implementation surface is `python/sglang/srt/arg_groups/deepseek_v4_hook.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +0/-7 (7 lines); hunks: -51,13 +51,6 @@ def apply_deepseek_v4_defaults(server_args: "ServerArgs", mod...; symbols: apply_deepseek_v4_defaults, validate_deepseek_v4_cp, touching `apply_deepseek_v4_defaults, validate_deepseek_v4_cp`.
- Code diff details:
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +0/-7 (7 lines); hunks: -51,13 +51,6 @@ def apply_deepseek_v4_defaults(server_args: "ServerArgs", mod...; symbols: apply_deepseek_v4_defaults, validate_deepseek_v4_cp
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +0/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/arg_groups/deepseek_v4_hook.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25821 - [Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename

- Link: https://github.com/sgl-project/sglang/pull/25821
- Status/date: merged / 2026-05-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 162 files, +11303/-10745, 15980 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; technical summary: Covers "[Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename"; the main implementation surface is `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587 (2595 lines); `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0 (2589 lines); `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518 (2539 lines); `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0 (2528 lines).
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587 (2595 lines)
  - `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0 (2589 lines)
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518 (2539 lines)
  - `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0 (2528 lines)
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +8/-1744 (1752 lines); hunks: -1,1746 +1,10; symbols: BaseIndexerMetadata, get_seqlens_int32, get_page_table_64, get_page_table_1
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587; `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0; `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518; `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +8/-1744; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` added +1746/-0
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py`, `python/sglang/jit_kernel/tests/test_fused_store_index_cache.py`, `python/sglang/jit_kernel/tests/test_set_mla_kv_buffer.py`, `python/sglang/test/nightly_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25810 - perf(dsv4): add MHC token-count prewarm

- Link: https://github.com/sgl-project/sglang/pull/25810
- Status/date: merged / 2026-05-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`; associated commits `3a6de13cd822`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +141/-1, 198 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "perf(dsv4): add MHC token-count prewarm"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`; technical summary: Covers "perf(dsv4): add MHC token-count prewarm"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +110/-0 (110 lines); hunks: -2,6 +2,7; -696,6 +697,70 @@ def __init__(; symbols: __init__, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets, hc_pre, touching `__init__, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets`; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +5/-0 (5 lines); hunks: -108,6 +108,11 @@ def hc_head(; symbols: hc_head, prewarm_mhc_token_count_buckets, forward, touching `hc_head, prewarm_mhc_token_count_buckets, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +110/-0 (110 lines); hunks: -2,6 +2,7; -696,6 +697,70 @@ def __init__(; symbols: __init__, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets, hc_pre
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +5/-0 (5 lines); hunks: -108,6 +108,11 @@ def hc_head(; symbols: hc_head, prewarm_mhc_token_count_buckets, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +110/-0; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/mhc.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25889 - [Fix] DSV4 cached_loc invalidated when SWA mapping is rebuilt

- Link: https://github.com/sgl-project/sglang/pull/25889
- Status/date: merged / 2026-05-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; associated commits `888a8794ef3d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +462/-0, 472 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] DSV4 cached_loc invalidated when SWA mapping is rebuilt"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "[Fix] DSV4 cached_loc invalidated when SWA mapping is rebuilt"; the main implementation surface is `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +4/-0 (4 lines); hunks: -492,6 +492,10 @@ def __init__(; symbols: __init__, register_mapping, invalidate_loc_cache, get_ring_size, touching `__init__, register_mapping, invalidate_loc_cache`.
- Code diff details:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +4/-0 (4 lines); hunks: -492,6 +492,10 @@ def __init__(; symbols: __init__, register_mapping, invalidate_loc_cache, get_ring_size
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -492,6 +492,10 @@ def __init__(
+        self.cached_loc = None  # mapping replaced; discard any cached translation
+    def invalidate_loc_cache(self) -> None:
+        self.cached_loc = None
```

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `test/manual/core/test_dsv4_cached_loc_invalidation.py`, `test/manual/core/test_dsv4_hicache_swa_translation_cache.py`, `test/manual/core/test_dsv4_stale_loc_crash.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25884 - [Refactor] major JIT kernel clean up for dsv4

- Link: https://github.com/sgl-project/sglang/pull/25884
- Status/date: merged / 2026-05-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `19f55c0e6d6f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 23 files, +1093/-1399, 2663 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] major JIT kernel clean up for dsv4"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/jit_kernel/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "[Refactor] major JIT kernel clean up for dsv4"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/jit_kernel/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7; `python/sglang/jit_kernel/deepseek_v4.py` removed +0/-1036 (1036 lines); hunks: -1,1036 +0,0; symbols: make_name, _jit_common_module, _jit_compress_128_online_plan_module, _jit_compress_128_online_module, touching `make_name, _jit_common_module, _jit_compress_128_online_plan_module`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7
  - `python/sglang/jit_kernel/deepseek_v4.py` removed +0/-1036 (1036 lines); hunks: -1,1036 +0,0; symbols: make_name, _jit_common_module, _jit_compress_128_online_plan_module, _jit_compress_128_online_module
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +1/-1; `python/sglang/jit_kernel/deepseek_v4.py` removed +0/-1036; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/deepseek_v4/topk_1024.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/topk_v1.cuh`, `python/sglang/jit_kernel/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26004 - Default MegaMoE to W4A8 for Max-Throughput recipe

- Link: https://github.com/sgl-project/sglang/pull/26004
- Status/date: merged / 2026-05-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +13/-2, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Default MegaMoE to W4A8 for Max-Throughput recipe"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "Default MegaMoE to W4A8 for Max-Throughput recipe"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +13/-2 (15 lines); hunks: -177,6 +177,16 @@ export const DeepSeekV4Deployment = () => {; -605,7 +615,8 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +13/-2 (15 lines); hunks: -177,6 +177,16 @@ export const DeepSeekV4Deployment = () => {; -605,7 +615,8 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +13/-2
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #25923 - [Docs] DeepSeek-V4: switch H200 FP4 Pro to flashinfer_mxfp4, Flash Balanced too

- Link: https://github.com/sgl-project/sglang/pull/25923
- Status/date: merged / 2026-05-21
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `17dadebd4e53`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +16/-5, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] DeepSeek-V4: switch H200 FP4 Pro to flashinfer_mxfp4, Flash Balanced too"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[Docs] DeepSeek-V4: switch H200 FP4 Pro to flashinfer_mxfp4, Flash Balanced too"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -120,7 +120,7 @@ The generator currently picks values on the **conservative**....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -120,7 +120,7 @@ The generator currently picks values on the **conservative**...
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -120,7 +120,7 @@ The generator currently picks values on the **conservative** side (mirroring an
-- Original FP4 checkpoints: To run original FP4 checkpoints, we provide two different options for w4a16 MoE kernels: Marlin (`--moe-runner-backend marlin`) and Flashinfer (`--moe-
+- Original FP4 checkpoints: To run original FP4 checkpoints, we provide two different options for w4a16 MoE kernels: Marlin (`--moe-runner-backend marlin`) and Flashinfer (`--moe-
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #26057 - [docs] DeepSeek-V4 cookbook: split Quantization axis, add H100 SGLang FP8

- Link: https://github.com/sgl-project/sglang/pull/26057
- Status/date: merged / 2026-05-22
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `88a37d740511`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +388/-93, 722 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] DeepSeek-V4 cookbook: split Quantization axis, add H100 SGLang FP8"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[docs] DeepSeek-V4 cookbook: split Quantization axis, add H100 SGLang FP8"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +273/-68 (341 lines); hunks: -35,7 +35,7 @@ tag: NEW; -182,7 +182,7 @@ curl http://localhost:30000/v1/chat/completions \.
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +273/-68 (341 lines); hunks: -35,7 +35,7 @@ tag: NEW; -182,7 +182,7 @@ curl http://localhost:30000/v1/chat/completions \
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -35,7 +35,7 @@ tag: NEW
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB200 8 GPU (2 nodes) / GB300 4 GPU / H200 8 GPU(fp4)/16 GPU(fp8)</t
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB200 8 GPU (2 nodes) / GB300 4 GPU / H200 8 GPU (FP4) or 16 GPU (SG
@@ -182,7 +182,7 @@ curl http://localhost:30000/v1/chat/completions \
-**Streaming with Thinking Process:**
+<Accordion title="Streaming with Thinking Process (Python)">
@@ -227,17 +227,36 @@ for chunk in response:
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +273/-68
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #25128 - [Intel GPU] 1/N Fix tilelang import in deepseek v4 rope as optional

- Link: https://github.com/sgl-project/sglang/pull/25128
- Status/date: merged / 2026-05-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-6, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Intel GPU] 1/N Fix tilelang import in deepseek v4 rope as optional"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/layers/deepseek_v4_rope.py`; technical summary: Covers "[Intel GPU] 1/N Fix tilelang import in deepseek v4 rope as optional"; the main implementation surface is `python/sglang/srt/layers/deepseek_v4_rope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/deepseek_v4_rope.py` modified +9/-6 (15 lines); hunks: -2,17 +2,20.
- Code diff details:
  - `python/sglang/srt/layers/deepseek_v4_rope.py` modified +9/-6 (15 lines); hunks: -2,17 +2,20
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/deepseek_v4_rope.py` modified +9/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/deepseek_v4_rope.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26141 - Add non-MTP DSV4 test coverage

- Link: https://github.com/sgl-project/sglang/pull/26141
- Status/date: merged / 2026-05-23
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`; associated commits `7b7f1067bdb0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +68/-0, 81 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add non-MTP DSV4 test coverage"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`; technical summary: Covers "Add non-MTP DSV4 test coverage"; the main implementation surface is `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +36/-0 (36 lines); hunks: -120,6 +120,42 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4NonMTPB200, setUpClass, TestDSV4FlashFP4B200Balanced_CP, touching `tearDownClass, TestDSV4FlashFP4NonMTPB200, setUpClass`; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +32/-0 (32 lines); hunks: -131,5 +131,37 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4NonMTPH200, setUpClass, touching `tearDownClass, TestDSV4FlashFP4NonMTPH200, setUpClass`.
- Code diff details:
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +36/-0 (36 lines); hunks: -120,6 +120,42 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4NonMTPB200, setUpClass, TestDSV4FlashFP4B200Balanced_CP
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +32/-0 (32 lines); hunks: -131,5 +131,37 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4NonMTPH200, setUpClass
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +36/-0; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +32/-0
- Risk and verification: The diff ships test coverage in `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26164 - [docs] DeepSeek-V4 cookbook: balanced MegaMoE cap, H200 Pro FP4 mem-frac, nsa-* compat, PD-disagg fixes

- Link: https://github.com/sgl-project/sglang/pull/26164
- Status/date: merged / 2026-05-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +66/-7, 139 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] DeepSeek-V4 cookbook: balanced MegaMoE cap, H200 Pro FP4 mem-frac, nsa-* compat, PD-disagg fixes"; model line: DeepSeek V4; category: bug fix; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "[docs] DeepSeek-V4 cookbook: balanced MegaMoE cap, H200 Pro FP4 mem-frac, nsa-* compat, PD-disagg fixes"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +66/-7 (73 lines); hunks: -119,15 +119,23 @@ export const DeepSeekV4Deployment = () => {; -155,11 +163,20 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +66/-7 (73 lines); hunks: -119,15 +119,23 @@ export const DeepSeekV4Deployment = () => {; -155,11 +163,20 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +66/-7
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #25898 - [AMD] Dsv4/pr1 fix run time issue

- Link: https://github.com/sgl-project/sglang/pull/25898
- Status/date: merged / 2026-05-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `af8f66940e9b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 32 files, +2523/-129, 3203 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Dsv4/pr1 fix run time issue"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "[AMD] Dsv4/pr1 fix run time issue"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +153/-28 (181 lines); hunks: -96,6 +96,8; -105,6 +107,29; symbols: _fused_rmsnorm_fp8_quant, __init__, _forward_prepare_multi_stream, touching `_fused_rmsnorm_fp8_quant, __init__, _forward_prepare_multi_stream`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-0 (1 lines); hunks: -578,6 +578,7 @@ def _init_paged_compress_states(self, enable_memory_saver: b...; symbols: _init_paged_compress_states, touching `_init_paged_compress_states`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +153/-28 (181 lines); hunks: -96,6 +96,8; -105,6 +107,29; symbols: _fused_rmsnorm_fp8_quant, __init__, _forward_prepare_multi_stream
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-0 (1 lines); hunks: -578,6 +578,7 @@ def _init_paged_compress_states(self, enable_memory_saver: b...; symbols: _init_paged_compress_states
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +153/-28; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25948 - [dsv4] support eplb

- Link: https://github.com/sgl-project/sglang/pull/25948
- Status/date: merged / 2026-05-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `7f45bcdd2ab8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-6, 60 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[dsv4] support eplb"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[dsv4] support eplb"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +14/-6 (20 lines); hunks: -3,6 +3,7; -33,6 +34,7; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +14/-6 (20 lines); hunks: -3,6 +3,7; -33,6 +34,7; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +14/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/hash_topk.py`, `python/sglang/srt/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26239 - [dsv4] fix multi-step draft on non-cuda-graph path

- Link: https://github.com/sgl-project/sglang/pull/26239
- Status/date: merged / 2026-05-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; associated commits `ed179bf9b297`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +47/-7, 93 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[dsv4] fix multi-step draft on non-cuda-graph path"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; technical summary: Covers "[dsv4] fix multi-step draft on non-cuda-graph path"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +13/-1 (14 lines); hunks: -53,6 +53,7; -676,11 +677,22 @@ def init_forward_metadata(self, forward_batch: ForwardBatc...; symbols: init_forward_metadata, touching `init_forward_metadata`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +13/-1 (14 lines); hunks: -53,6 +53,7; -676,11 +677,22 @@ def init_forward_metadata(self, forward_batch: ForwardBatc...; symbols: init_forward_metadata
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +13/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/speculative/eagle_utils.py`, `python/sglang/srt/speculative/eagle_worker_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25391 - Support DeepSeek V4 DeepEP Waterfill

- Link: https://github.com/sgl-project/sglang/pull/25391
- Status/date: merged / 2026-05-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `59cad671e2a8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +58/-16, 134 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support DeepSeek V4 DeepEP Waterfill"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "Support DeepSeek V4 DeepEP Waterfill"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +16/-0 (16 lines); hunks: -1396,6 +1396,22 @@ def determine_num_fused_shared_experts(self):; symbols: determine_num_fused_shared_experts, touching `determine_num_fused_shared_experts`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +16/-0 (16 lines); hunks: -1396,6 +1396,22 @@ def determine_num_fused_shared_experts(self):; symbols: determine_num_fused_shared_experts
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +16/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/hash_topk.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/model_executor/model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26208 - [AMD] Dsv4/pr2 compressor opt

- Link: https://github.com/sgl-project/sglang/pull/26208
- Status/date: merged / 2026-05-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `3f5e2c768825`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 31 files, +8829/-149, 6378 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Dsv4/pr2 compressor opt"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "[AMD] Dsv4/pr2 compressor opt"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +152/-10 (162 lines); hunks: -536,6 +536,118 @@ def _forward_prepare_multi_stream(; -695,14 +807,24 @@ def forward(; symbols: _forward_prepare_multi_stream, _forward_prepare_multi_stream_hip, _forward_prepare, forward, touching `_forward_prepare_multi_stream, _forward_prepare_multi_stream_hip, _forward_prepare`; `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +13/-5 (18 lines); hunks: -20,11 +20,19; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-1 (7 lines); hunks: -470,8 +470,13 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +152/-10 (162 lines); hunks: -536,6 +536,118 @@ def _forward_prepare_multi_stream(; -695,14 +807,24 @@ def forward(; symbols: _forward_prepare_multi_stream, _forward_prepare_multi_stream_hip, _forward_prepare, forward
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +13/-5 (18 lines); hunks: -20,11 +20,19
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-1 (7 lines); hunks: -470,8 +470,13 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +152/-10; `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +13/-5; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-1
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_dsv4_norm_rope.py`, `test/manual/dsv4/test_fused_compress_attn_hip.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26413 - [docs] DeepSeek-V4 cookbook: note cu129 image for GB200 Pro DeepEP backend

- Link: https://github.com/sgl-project/sglang/pull/26413
- Status/date: merged / 2026-05-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +14/-0, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] DeepSeek-V4 cookbook: note cu129 image for GB200 Pro DeepEP backend"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "[docs] DeepSeek-V4 cookbook: note cu129 image for GB200 Pro DeepEP backend"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0 (14 lines); hunks: -909,6 +909,20 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0 (14 lines); hunks: -909,6 +909,20 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #26451 - [docs] Fix V4 Pro balanced recipe

- Link: https://github.com/sgl-project/sglang/pull/26451
- Status/date: merged / 2026-05-27
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `d6032c04b665`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-12, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] Fix V4 Pro balanced recipe"; model line: DeepSeek V4; category: bug fix; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[docs] Fix V4 Pro balanced recipe"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +2/-2 (4 lines); hunks: -145,8 +145,8 @@ Two variants are exposed:.
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +2/-2 (4 lines); hunks: -145,8 +145,8 @@ Two variants are exposed:
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -145,8 +145,8 @@ Two variants are exposed:
-- MegaMoE is **not** supported on Hopper (H100 / H200) nor on the `low-latency` / `cp` settings. When running MegaMoE, don't set `--moe-runner-backend` manually.
-- Adjust `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` based on your workload and memory usage. Setting higher number of tokens for MegaMoE requires more HBM space. (reco
+- MegaMoE is **not** supported on Hopper (H100 / H200) nor on the `low-latency` / `balanced` / `cp` settings — it is only wired into the `max-throughput` recipe on Blackwell. When
+- Adjust `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` based on your workload and memory usage. Setting higher number of tokens for MegaMoE requires more HBM space. (reco
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +2/-2
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #26499 - [Kernel] Import flash_mla kernels from sglang kernel for deepseek v4

- Link: https://github.com/sgl-project/sglang/pull/26499
- Status/date: merged / 2026-05-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`; associated commits `e06058ed624f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +6/-6, 54 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Import flash_mla kernels from sglang kernel for deepseek v4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`; technical summary: Covers "[Kernel] Import flash_mla kernels from sglang kernel for deepseek v4"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +3/-3 (6 lines); hunks: -58,7 +58,7; -82,7 +82,7 @@ def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED...; symbols: _pad_last_dim, _create_flashmla_metadata, forward, touching `_pad_last_dim, _create_flashmla_metadata, forward`; `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +2/-2 (4 lines); hunks: -55,7 +55,7; -83,7 +83,7 @@ def _create_flashmla_metadata():; symbols: _create_flashmla_metadata, touching `_create_flashmla_metadata`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +3/-3 (6 lines); hunks: -58,7 +58,7; -82,7 +82,7 @@ def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED...; symbols: _pad_last_dim, _create_flashmla_metadata, forward
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +2/-2 (4 lines); hunks: -55,7 +55,7; -83,7 +83,7 @@ def _create_flashmla_metadata():; symbols: _create_flashmla_metadata
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +3/-3; `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/hip_flash_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26383 - [AMD][DSV4] DSV4 MTP graph + sparse triton attn optimizations

- Link: https://github.com/sgl-project/sglang/pull/26383
- Status/date: merged / 2026-05-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `deaba74745d7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +659/-65, 950 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][DSV4] DSV4 MTP graph + sparse triton attn optimizations"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[AMD][DSV4] DSV4 MTP graph + sparse triton attn optimizations"; the main implementation surface is `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` added +158/-0 (158 lines); hunks: -0,0 +1,158; symbols: _get_triton_mhc_post_pre_ops, _get_fused_hc_post_pre_buffers, try_fused_hc_post_pre, touching `_get_triton_mhc_post_pre_ops, _get_fused_hc_post_pre_buffers, try_fused_hc_post_pre`; `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +35/-26 (61 lines); hunks: -51,6 +51,7; -500,32 +501,21 @@ def init_forward_metadata_target_verify(; symbols: init_forward_metadata_target_verify, init_forward_metadata_target_verify_old, make_forward_metadata_from_raw_verify, init_forward_metadata, touching `init_forward_metadata_target_verify, init_forward_metadata_target_verify_old, make_forward_metadata_from_raw_verify`; `python/sglang/srt/models/deepseek_v4.py` modified +52/-7 (59 lines); hunks: -87,6 +87,9; -133,6 +136,28 @@ def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):; symbols: _fused_rmsnorm_fp8_quant, _freqs_cis_to_cos_sin, __init__, _forward_prepare_multi_stream_hip, touching `_fused_rmsnorm_fp8_quant, _freqs_cis_to_cos_sin, __init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` added +158/-0 (158 lines); hunks: -0,0 +1,158; symbols: _get_triton_mhc_post_pre_ops, _get_fused_hc_post_pre_buffers, try_fused_hc_post_pre
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +35/-26 (61 lines); hunks: -51,6 +51,7; -500,32 +501,21 @@ def init_forward_metadata_target_verify(; symbols: init_forward_metadata_target_verify, init_forward_metadata_target_verify_old, make_forward_metadata_from_raw_verify, init_forward_metadata
  - `python/sglang/srt/models/deepseek_v4.py` modified +52/-7 (59 lines); hunks: -87,6 +87,9; -133,6 +136,28 @@ def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):; symbols: _fused_rmsnorm_fp8_quant, _freqs_cis_to_cos_sin, __init__, _forward_prepare_multi_stream_hip
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` added +158/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +35/-26; `python/sglang/srt/models/deepseek_v4.py` modified +52/-7
- Risk and verification: The diff ships test coverage in `test/registered/ops/test_aiter_greedy_sample_amd.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26238 - refactor(dsv4): route MHC prenorm through DeepGEMM wrapper

- Link: https://github.com/sgl-project/sglang/pull/26238
- Status/date: merged / 2026-05-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`; associated commits `eae03ce3b2a8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +67/-148, 345 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "refactor(dsv4): route MHC prenorm through DeepGEMM wrapper"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`; technical summary: Covers "refactor(dsv4): route MHC prenorm through DeepGEMM wrapper"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +4/-112 (116 lines); hunks: -2,7 +2,6; -821,70 +820,6 @@ def __init__(; symbols: __init__, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets, hc_pre, touching `__init__, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets`; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +0/-5 (5 lines); hunks: -129,11 +129,6 @@ def hc_head(; symbols: hc_head, prewarm_mhc_token_count_buckets, forward, touching `hc_head, prewarm_mhc_token_count_buckets, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +4/-112 (116 lines); hunks: -2,7 +2,6; -821,70 +820,6 @@ def __init__(; symbols: __init__, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets, hc_pre
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +0/-5 (5 lines); hunks: -129,11 +129,6 @@ def hc_head(; symbols: hc_head, prewarm_mhc_token_count_buckets, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +4/-112; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +0/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/mhc.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26609 - [CI] Clean DeepSeek V4 tests and installation scripts

- Link: https://github.com/sgl-project/sglang/pull/26609
- Status/date: merged / 2026-05-28
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py`; associated commits `435c4ffb3081`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +66/-198, 432 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Clean DeepSeek V4 tests and installation scripts"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`; technical summary: Covers "[CI] Clean DeepSeek V4 tests and installation scripts"; the main implementation surface is `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +2/-2 (4 lines); hunks: -4,7 +4,7; -20,7 +20,7; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +2/-2 (4 lines); hunks: -4,7 +4,7; -20,7 +20,7; symbols: _flashinfer_has_sm90_cutlass_mxfp4, touching `_flashinfer_has_sm90_cutlass_mxfp4`; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +2/-2 (4 lines); hunks: -4,7 +4,7; -20,7 +20,7; `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py` modified +2/-2 (4 lines); hunks: -5,7 +5,7; -21,7 +21,7.
- Code diff details:
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +2/-2 (4 lines); hunks: -4,7 +4,7; -20,7 +20,7
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +2/-2 (4 lines); hunks: -4,7 +4,7; -20,7 +20,7; symbols: _flashinfer_has_sm90_cutlass_mxfp4
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +2/-2 (4 lines); hunks: -4,7 +4,7; -20,7 +20,7
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py` modified +2/-2 (4 lines); hunks: -5,7 +5,7; -21,7 +21,7
  - `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +2/-2; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +2/-2; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +2/-2; `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py` modified +2/-2; `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/disaggregation/test_disaggregation_dsv4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26668 - [Doc] Update benchmark instruction for dsv4

- Link: https://github.com/sgl-project/sglang/pull/26668
- Status/date: merged / 2026-05-29
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `69362cbc2c04`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +28/-36, 83 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Doc] Update benchmark instruction for dsv4"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[Doc] Update benchmark instruction for dsv4"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +28/-36 (64 lines); hunks: -363,6 +363,10 @@ For more details, see the [HiCache documentation](../../../...; -383,47 +387,35 @@ python3 -m sglang.test.few_shot_gsm8k --num-questions 200....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +28/-36 (64 lines); hunks: -363,6 +363,10 @@ For more details, see the [HiCache documentation](../../../...; -383,47 +387,35 @@ python3 -m sglang.test.few_shot_gsm8k --num-questions 200...
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +28/-36
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #26662 - [AMD][CI] Update v4 CI setting and move the task to main branch

- Link: https://github.com/sgl-project/sglang/pull/26662
- Status/date: merged / 2026-05-29
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`; associated commits `6e9bd82714cb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +97/-160, 398 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][CI] Update v4 CI setting and move the task to main branch"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`; technical summary: Covers "[AMD][CI] Update v4 CI setting and move the task to main branch"; the main implementation surface is `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +22/-24 (46 lines); hunks: -35,38 +35,32; -86,11 +80,15 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +22/-24 (46 lines); hunks: -35,38 +35,32; -86,11 +80,15 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +22/-24 (46 lines); hunks: -37,38 +37,32; -88,11 +82,15 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +22/-24 (46 lines); hunks: -37,38 +37,32; -88,11 +82,15 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`.
- Code diff details:
  - `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +22/-24 (46 lines); hunks: -35,38 +35,32; -86,11 +80,15 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +22/-24 (46 lines); hunks: -35,38 +35,32; -86,11 +80,15 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +22/-24 (46 lines); hunks: -37,38 +37,32; -88,11 +82,15 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +22/-24 (46 lines); hunks: -37,38 +37,32; -88,11 +82,15 @@ def setUpClass(cls):; symbols: setUpClass
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +22/-24; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +22/-24; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +22/-24; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +22/-24
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25976 - [DeepSeek-V4] Add mhc_fused_post_pre kernel

- Link: https://github.com/sgl-project/sglang/pull/25976
- Status/date: merged / 2026-05-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`; associated commits `7c5708cba734`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +875/-48, 1065 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek-V4] Add mhc_fused_post_pre kernel"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`; technical summary: Covers "[DeepSeek-V4] Add mhc_fused_post_pre kernel"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +261/-47 (308 lines); hunks: -2,6 +2,7; -61,6 +62,7; symbols: _is_fused_mhc_post_pre_enabled, __init__, refresh_mhc_norm_weight_cache, prewarm_mhc_token_counts, touching `_is_fused_mhc_post_pre_enabled, __init__, refresh_mhc_norm_weight_cache`; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +5/-1 (6 lines); hunks: -170,13 +170,17 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +261/-47 (308 lines); hunks: -2,6 +2,7; -61,6 +62,7; symbols: _is_fused_mhc_post_pre_enabled, __init__, refresh_mhc_norm_weight_cache, prewarm_mhc_token_counts
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +5/-1 (6 lines); hunks: -170,13 +170,17 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +261/-47; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +5/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24692 - feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference

- Link: https://github.com/sgl-project/sglang/pull/24692
- Status/date: merged / 2026-06-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; associated commits `524ba10eda1b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +2105/-22, 2268 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; technical summary: Covers "feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +41/-18 (59 lines); hunks: -56,13 +56,16; -82,6 +85,8 @@ def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED...; symbols: _pad_last_dim, _create_flashmla_metadata, forward, touching `_pad_last_dim, _create_flashmla_metadata, forward`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +41/-18 (59 lines); hunks: -56,13 +56,16; -82,6 +85,8 @@ def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED...; symbols: _pad_last_dim, _create_flashmla_metadata, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +41/-18
- Risk and verification: The diff ships test coverage in `test/registered/kernels/test_sm120_flash_mla.py`, `test/registered/kernels/test_sm120_paged_mqa_logits.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24947 - DeepSeek V4: Support context parallelism with fused MoE (non-DeepEP)

- Link: https://github.com/sgl-project/sglang/pull/24947
- Status/date: merged / 2026-06-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`; associated commits `5700790c0593`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +1982/-24, 2117 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DeepSeek V4: Support context parallelism with fused MoE (non-DeepEP)"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`; technical summary: Covers "DeepSeek V4: Support context parallelism with fused MoE (non-DeepEP)"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +18/-9 (27 lines); hunks: -44,6 +44,10; -68,6 +72,7; symbols: forward, touching `forward`; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +3/-0 (3 lines); hunks: -28,6 +28,7; -169,6 +170,8 @@ def forward(; symbols: forward, touching `forward`; `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +45/-0 (45 lines); hunks: -81,5 +81,50 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4B200Balanced_CP_NonDeepEP, setUpClass, touching `tearDownClass, TestDSV4FlashFP4B200Balanced_CP_NonDeepEP, setUpClass`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +18/-9 (27 lines); hunks: -44,6 +44,10; -68,6 +72,7; symbols: forward
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +3/-0 (3 lines); hunks: -28,6 +28,7; -169,6 +170,8 @@ def forward(; symbols: forward
  - `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +45/-0 (45 lines); hunks: -81,5 +81,50 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4B200Balanced_CP_NonDeepEP, setUpClass
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +18/-9; `python/sglang/srt/models/deepseek_v4_nextn.py` modified +3/-0
  - tests: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +45/-0
- Risk and verification: The diff ships test coverage in `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26968 - docs: update RTX PRO 6000 deployment snippet

- Link: https://github.com/sgl-project/sglang/pull/26968
- Status/date: merged / 2026-06-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +33/-29, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: update RTX PRO 6000 deployment snippet"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "docs: update RTX PRO 6000 deployment snippet"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +33/-29 (62 lines); hunks: -33,7 +33,7 @@ export const DeepSeekV4Deployment = () => {; -125,7 +125,7 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +33/-29 (62 lines); hunks: -33,7 +33,7 @@ export const DeepSeekV4Deployment = () => {; -125,7 +125,7 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +33/-29
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #26931 - [AMD] dpsk-v4 swa loc cache support

- Link: https://github.com/sgl-project/sglang/pull/26931
- Status/date: merged / 2026-06-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `d15a2dc72c81`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +13/-16, 64 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] dpsk-v4 swa loc cache support"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "[AMD] dpsk-v4 swa loc cache support"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -641,8 +641,8 @@ def _forward_prepare_multi_stream_hip(; -731,8 +731,8 @@ def _forward_prepare(; symbols: _forward_prepare_multi_stream_hip, _forward_prepare, touching `_forward_prepare_multi_stream_hip, _forward_prepare`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +9/-12 (21 lines); hunks: -512,6 +512,13 @@ def translate_loc_from_full_to_swa(self, kv_indices: torch....; -758,12 +765,7 @@ def set_swa_key_buffer_radix_fused(; symbols: translate_loc_from_full_to_swa, get_cached_swa_loc, get_contiguous_buf_infos, set_swa_key_buffer_radix_fused, touching `translate_loc_from_full_to_swa, get_cached_swa_loc, get_contiguous_buf_infos`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -641,8 +641,8 @@ def _forward_prepare_multi_stream_hip(; -731,8 +731,8 @@ def _forward_prepare(; symbols: _forward_prepare_multi_stream_hip, _forward_prepare
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +9/-12 (21 lines); hunks: -512,6 +512,13 @@ def translate_loc_from_full_to_swa(self, kv_indices: torch....; -758,12 +765,7 @@ def set_swa_key_buffer_radix_fused(; symbols: translate_loc_from_full_to_swa, get_cached_swa_loc, get_contiguous_buf_infos, set_swa_key_buffer_radix_fused
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +4/-4; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +9/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26209 - Add FP4 Indexer for DeepSeek V4

- Link: https://github.com/sgl-project/sglang/pull/26209
- Status/date: merged / 2026-06-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; associated commits `301bcf08726b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +1177/-33, 1505 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add FP4 Indexer for DeepSeek V4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "Add FP4 Indexer for DeepSeek V4"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +3/-0 (3 lines); hunks: -366,6 +366,9 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +34/-3 (37 lines); hunks: -271,13 +271,17 @@ def __init__(; -346,6 +350,23 @@ def set_index_fused(; symbols: __init__, get_bytes_per_token, _create_buffer, set_index_fused, touching `__init__, get_bytes_per_token, _create_buffer`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +3/-0 (3 lines); hunks: -366,6 +366,9 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +34/-3 (37 lines); hunks: -271,13 +271,17 @@ def __init__(; -346,6 +350,23 @@ def set_index_fused(; symbols: __init__, get_bytes_per_token, _create_buffer, set_index_fused
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +3/-0; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +34/-3
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/deepseek_v4/test_fp4_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27049 - docs: add DeepSeek-V4 EPLB Waterfill tips

- Link: https://github.com/sgl-project/sglang/pull/27049
- Status/date: merged / 2026-06-03
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `ac16dbf41250`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +41/-0, 48 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: add DeepSeek-V4 EPLB Waterfill tips"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs: add DeepSeek-V4 EPLB Waterfill tips"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +41/-0 (41 lines); hunks: -115,6 +115,47 @@ The generator currently picks values on the **conservative*....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +41/-0 (41 lines); hunks: -115,6 +115,47 @@ The generator currently picks values on the **conservative*...
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +41/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #27035 - docs: add DeepSeek V4 FP4 indexer usage

- Link: https://github.com/sgl-project/sglang/pull/27035
- Status/date: merged / 2026-06-04
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `10b6b45cad33`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-0, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: add DeepSeek V4 FP4 indexer usage"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs: add DeepSeek V4 FP4 indexer usage"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +12/-0 (12 lines); hunks: -156,6 +156,18 @@ MegaMoE is not supported with this DeepEP Waterfill recipe....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +12/-0 (12 lines); hunks: -156,6 +156,18 @@ MegaMoE is not supported with this DeepEP Waterfill recipe...
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -156,6 +156,18 @@ MegaMoE is not supported with this DeepEP Waterfill recipe yet. Waterfill routes
+**FP4 Indexer (Experimental)**
+DeepSeek-V4 uses the default indexer path unless `--enable-deepseek-v4-fp4-indexer` is set. Enable this flag to use the experimental FP4 C4 indexer on SM100 GPUs with DeepGEMM FP4
+'''bash Command
+# Please use latest main branch for this feature
+sglang serve deepseek-ai/DeepSeek-V4-Flash \
+  --tp 4 \
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +12/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/docs/advanced_features/server_arguments.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #24880 - [PD & HiSparse] Add DeepSeek V4 support for HiSparse direct Prefill-to-Decode DRAM

- Link: https://github.com/sgl-project/sglang/pull/24880
- Status/date: merged / 2026-06-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +477/-308, 1103 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[PD & HiSparse] Add DeepSeek V4 support for HiSparse direct Prefill-to-Decode DRAM"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `python/sglang/srt/mem_cache/hisparse_memory_pool.py`, `python/sglang/jit_kernel/tests/test_hisparse.py`, `test/registered/disaggregation/test_disaggregation_dsv4.py`; technical summary: Covers "[PD & HiSparse] Add DeepSeek V4 support for HiSparse direct Prefill-to-Decode DRAM"; the main implementation surface is `python/sglang/srt/mem_cache/hisparse_memory_pool.py`, `python/sglang/jit_kernel/tests/test_hisparse.py`, `test/registered/disaggregation/test_disaggregation_dsv4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/mem_cache/hisparse_memory_pool.py` modified +37/-129 (166 lines); hunks: -4,7 +4,6; -17,7 +16,6; symbols: free, DeepSeekV4SingleKVPoolHost, __init__, clear, touching `free, DeepSeekV4SingleKVPoolHost, __init__`; `python/sglang/jit_kernel/tests/test_hisparse.py` modified +128/-1 (129 lines); hunks: -3,7 +3,11; -26,6 +30,12; symbols: _host_cache, _dsv4_token_pattern, _write_dsv4_token, _read_dsv4_token, touching `_host_cache, _dsv4_token_pattern, _write_dsv4_token`; `test/registered/disaggregation/test_disaggregation_dsv4.py` modified +105/-1 (106 lines); hunks: -11,11 +11,15; -123,5 +127,105 @@ def start_decode(cls):; symbols: start_decode, TestDisaggregationDSV4HiSparseMooncake, setUpClass, start_prefill, touching `start_decode, TestDisaggregationDSV4HiSparseMooncake, setUpClass`; `python/sglang/jit_kernel/csrc/deepseek_v4/hisparse_transfer.cuh` removed +0/-82 (82 lines); hunks: -1,82 +0,0.
- Code diff details:
  - `python/sglang/srt/mem_cache/hisparse_memory_pool.py` modified +37/-129 (166 lines); hunks: -4,7 +4,6; -17,7 +16,6; symbols: free, DeepSeekV4SingleKVPoolHost, __init__, clear
  - `python/sglang/jit_kernel/tests/test_hisparse.py` modified +128/-1 (129 lines); hunks: -3,7 +3,11; -26,6 +30,12; symbols: _host_cache, _dsv4_token_pattern, _write_dsv4_token, _read_dsv4_token
  - `test/registered/disaggregation/test_disaggregation_dsv4.py` modified +105/-1 (106 lines); hunks: -11,11 +11,15; -123,5 +127,105 @@ def start_decode(cls):; symbols: start_decode, TestDisaggregationDSV4HiSparseMooncake, setUpClass, start_prefill
  - `python/sglang/jit_kernel/csrc/deepseek_v4/hisparse_transfer.cuh` removed +0/-82 (82 lines); hunks: -1,82 +0,0
  - `python/sglang/jit_kernel/csrc/hisparse.cuh` modified +61/-4 (65 lines); hunks: -52,6 +52,62 @@ transfer_item_warp(int32_t lane_id, const void* src_addr, voi...; -89,7 +145,7 @@ struct SmemLayout {
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/hisparse_memory_pool.py` modified +37/-129; `python/sglang/jit_kernel/csrc/deepseek_v4/hisparse_transfer.cuh` removed +0/-82; `python/sglang/jit_kernel/csrc/hisparse.cuh` modified +61/-4; `python/sglang/srt/mem_cache/memory_pool_host.py` modified +50/-8; `python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/kvcacheio.cuh` modified +14/-34; `python/sglang/srt/managers/hisparse_coordinator.py` modified +27/-17
  - tests: `python/sglang/jit_kernel/tests/test_hisparse.py` modified +128/-1; `test/registered/disaggregation/test_disaggregation_dsv4.py` modified +105/-1
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_hisparse.py`, `test/registered/disaggregation/test_disaggregation_dsv4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27404 - Remove DeepSeek V4 release Docker workflow

- Link: https://github.com/sgl-project/sglang/pull/27404
- Status/date: merged / 2026-06-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-149, 150 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove DeepSeek V4 release Docker workflow"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `.github/workflows/release-docker-deepseek-v4.yml`; technical summary: Covers "Remove DeepSeek V4 release Docker workflow"; the main implementation surface is `.github/workflows/release-docker-deepseek-v4.yml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.github/workflows/release-docker-deepseek-v4.yml` removed +0/-149 (149 lines); hunks: -1,149 +0,0.
- Code diff details:
  - `.github/workflows/release-docker-deepseek-v4.yml` removed +0/-149 (149 lines); hunks: -1,149 +0,0
- Key code excerpts:

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

- Reviewed files:
  - ci: `.github/workflows/release-docker-deepseek-v4.yml` removed +0/-149
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #27152 - [bugfix][AMD] AttributeError and warp mask bugs in DeepSeek V4 FP4 indexer

- Link: https://github.com/sgl-project/sglang/pull/27152
- Status/date: merged / 2026-06-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`; associated commits `3030119ef7cb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +8/-2, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bugfix][AMD] AttributeError and warp mask bugs in DeepSeek V4 FP4 indexer"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`; technical summary: Covers "[bugfix][AMD] AttributeError and warp mask bugs in DeepSeek V4 FP4 indexer"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +3/-1 (4 lines); hunks: -373,7 +373,9 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +3/-1 (4 lines); hunks: -373,7 +373,9 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -373,7 +373,9 @@ def __init__(
+        self.enable_deepseek_v4_fp4_indexer: bool = (
+            model_runner.server_args.enable_deepseek_v4_fp4_indexer
+        )
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27191 - Fix DeepSeek V4 DP reduce scatter when use attention DP + MoE TP

- Link: https://github.com/sgl-project/sglang/pull/27191
- Status/date: merged / 2026-06-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `4c8a022f38e3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-2, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix DeepSeek V4 DP reduce scatter when use attention DP + MoE TP"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "Fix DeepSeek V4 DP reduce scatter when use attention DP + MoE TP"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +10/-2 (12 lines); hunks: -59,6 +59,7; -67,7 +68,7; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +10/-2 (12 lines); hunks: -59,6 +59,7; -67,7 +68,7; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +10/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26885 - Cookbook renovation

- Link: https://github.com/sgl-project/sglang/pull/26885
- Status/date: merged / 2026-06-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +6692/-1693, 8494 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Cookbook renovation"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/_playground.jsx`; technical summary: Covers "Cookbook renovation"; the main implementation surface is `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/_playground.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` added +1222/-0 (1222 lines); hunks: -0,0 +1,1222; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` added +306/-0 (306 lines); hunks: -0,0 +1,306; `docs_new/src/snippets/_playground.jsx` added +2048/-0 (2048 lines); hunks: -0,0 +1,2048; `docs_new/src/snippets/_deployment.jsx` added +1277/-0 (1277 lines); hunks: -0,0 +1,1277.
- Code diff details:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` added +1222/-0 (1222 lines); hunks: -0,0 +1,1222
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` added +306/-0 (306 lines); hunks: -0,0 +1,306
  - `docs_new/src/snippets/_playground.jsx` added +2048/-0 (2048 lines); hunks: -0,0 +1,2048
  - `docs_new/src/snippets/_deployment.jsx` added +1277/-0 (1277 lines); hunks: -0,0 +1,1277
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` removed +0/-1263 (1263 lines); hunks: -1,1263 +0,0
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` added +1222/-0; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` added +306/-0; `docs_new/src/snippets/_playground.jsx` added +2048/-0; `docs_new/src/snippets/_deployment.jsx` added +1277/-0; `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` removed +0/-1263; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +138/-430
  - ci: `.github/ISSUE_TEMPLATE/3-playground-verified-cell.yml` added +109/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/_deployment.jsx`, `docs_new/src/snippets/_playground.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #27289 - [ROCm] dsv4: remove the redundant fp8 scale transpose-copy on decode

- Link: https://github.com/sgl-project/sglang/pull/27289
- Status/date: merged / 2026-06-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `ea1d190ed026`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +20/-3, 142 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] dsv4: remove the redundant fp8 scale transpose-copy on decode"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[ROCm] dsv4: remove the redundant fp8 scale transpose-copy on decode"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +2/-0 (2 lines); hunks: -97,6 +97,7; -151,6 +152,7 @@ def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):; symbols: _fused_rmsnorm_fp8_quant, touching `_fused_rmsnorm_fp8_quant`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +2/-0 (2 lines); hunks: -97,6 +97,7; -151,6 +152,7 @@ def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):; symbols: _fused_rmsnorm_fp8_quant
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -97,6 +97,7 @@
+from sglang.srt.models.deepseek_common.utils import _use_aiter_bpreshuffle_gfx95
@@ -151,6 +152,7 @@ def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):
+        transpose_scale=_use_aiter_bpreshuffle_gfx95,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25195 - [BCG] Support breakable CUDA graph for DeepSeek V4 DP attention

- Link: https://github.com/sgl-project/sglang/pull/25195
- Status/date: merged / 2026-06-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`; associated commits `ca66e6fb5e5d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +726/-66, 1223 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BCG] Support breakable CUDA graph for DeepSeek V4 DP attention"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`; technical summary: Covers "[BCG] Support breakable CUDA graph for DeepSeek V4 DP attention"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +251/-26 (277 lines); hunks: -184,6 +184,47 @@ def copy_(self, other: DSV4AttnMetadata) -> None:; -312,6 +353,24 @@ def copy_(self, other: DSV4Metadata):; symbols: copy_, refresh_for_breakable_cuda_graph_replay_, init_compression_metadata, touching `copy_, refresh_for_breakable_cuda_graph_replay_, init_compression_metadata`; `python/sglang/srt/models/deepseek_v4.py` modified +86/-10 (96 lines); hunks: -27,6 +27,8; -81,6 +83,12; symbols: _freqs_cis_to_cos_sin, deepseek_v4_attention_with_output, _rms_normalize_kernel, forward, touching `_freqs_cis_to_cos_sin, deepseek_v4_attention_with_output, _rms_normalize_kernel`; `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +166/-0 (166 lines); hunks: -338,6 +338,172 @@ def test_runner_mode_production_eagle_draft_extend_cuda_gr...; symbols: test_runner_mode_production_eagle_draft_extend_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract, _make_core_metadata, test_bcg_is_explicit_and_dsv4_backend_opt_in_only, touching `test_runner_mode_production_eagle_draft_extend_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract, _make_core_metadata`; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +51/-0 (51 lines); hunks: -156,5 +156,56 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4BreakableCudaGraphB200, setUpClass, touching `tearDownClass, TestDSV4FlashFP4BreakableCudaGraphB200, setUpClass`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +251/-26 (277 lines); hunks: -184,6 +184,47 @@ def copy_(self, other: DSV4AttnMetadata) -> None:; -312,6 +353,24 @@ def copy_(self, other: DSV4Metadata):; symbols: copy_, refresh_for_breakable_cuda_graph_replay_, init_compression_metadata
  - `python/sglang/srt/models/deepseek_v4.py` modified +86/-10 (96 lines); hunks: -27,6 +27,8; -81,6 +83,12; symbols: _freqs_cis_to_cos_sin, deepseek_v4_attention_with_output, _rms_normalize_kernel, forward
  - `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +166/-0 (166 lines); hunks: -338,6 +338,172 @@ def test_runner_mode_production_eagle_draft_extend_cuda_gr...; symbols: test_runner_mode_production_eagle_draft_extend_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract, _make_core_metadata, test_bcg_is_explicit_and_dsv4_backend_opt_in_only
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +51/-0 (51 lines); hunks: -156,5 +156,56 @@ def tearDownClass(cls):; symbols: tearDownClass, TestDSV4FlashFP4BreakableCudaGraphB200, setUpClass
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +251/-26; `python/sglang/srt/models/deepseek_v4.py` modified +86/-10
  - tests: `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +166/-0; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +51/-0
- Risk and verification: The diff ships test coverage in `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27380 - [AMD] Add unified kv attention support in dpsk-v4

- Link: https://github.com/sgl-project/sglang/pull/27380
- Status/date: merged / 2026-06-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `f2bcdb05086a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +2418/-84, 2904 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add unified kv attention support in dpsk-v4"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; technical summary: Covers "[AMD] Add unified kv attention support in dpsk-v4"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +244/-0 (244 lines); hunks: -113,11 +113,28 @@ class DSV4AttnMetadata:; -157,10 +174,23 @@ def copy_(self, other: DSV4AttnMetadata) -> None:; symbols: DSV4AttnMetadata, copy_, init_compression_metadata, init_flashmla_related, touching `DSV4AttnMetadata, copy_, init_compression_metadata`; `python/sglang/srt/models/deepseek_v4.py` modified +76/-27 (103 lines); hunks: -778,8 +778,17 @@ def _forward_prepare(; -797,15 +806,33 @@ def _forward_prepare(; symbols: _forward_prepare, forward, touching `_forward_prepare, forward`; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +1/-0 (1 lines); hunks: -238,6 +238,7 @@ def init_compression_metadata(self):; symbols: init_compression_metadata, touching `init_compression_metadata`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +149/-42 (191 lines); hunks: -374,6 +374,65 @@ class DeepSeekV4LayerItem(NamedTuple):; -395,6 +454,7 @@ def __init__(; symbols: DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool, __init__, get_unified_kv, touching `DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool, __init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +244/-0 (244 lines); hunks: -113,11 +113,28 @@ class DSV4AttnMetadata:; -157,10 +174,23 @@ def copy_(self, other: DSV4AttnMetadata) -> None:; symbols: DSV4AttnMetadata, copy_, init_compression_metadata, init_flashmla_related
  - `python/sglang/srt/models/deepseek_v4.py` modified +76/-27 (103 lines); hunks: -778,8 +778,17 @@ def _forward_prepare(; -797,15 +806,33 @@ def _forward_prepare(; symbols: _forward_prepare, forward
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +1/-0 (1 lines); hunks: -238,6 +238,7 @@ def init_compression_metadata(self):; symbols: init_compression_metadata
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +149/-42 (191 lines); hunks: -374,6 +374,65 @@ class DeepSeekV4LayerItem(NamedTuple):; -395,6 +454,7 @@ def __init__(; symbols: DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool, __init__, get_unified_kv
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +244/-0; `python/sglang/srt/models/deepseek_v4.py` modified +76/-27; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +1/-0; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +149/-42
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh`, `python/sglang/jit_kernel/dsv4/compress.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27529 - [AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase

- Link: https://github.com/sgl-project/sglang/pull/27529
- Status/date: merged / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +177/-88, 574 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/layers/attention/dsv4/compressor.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`; technical summary: Covers "[AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase"; the main implementation surface is `python/sglang/srt/layers/attention/dsv4/compressor.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +4/-0 (4 lines); hunks: -395,6 +395,10 @@ def apply_ape_hotfix(self):; symbols: apply_ape_hotfix, get_state_pool, touching `apply_ape_hotfix, get_state_pool`; `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh` modified +93/-46 (139 lines); hunks: -74,23 +74,27 @@ struct C4Trait {; -102,28 +106,61 @@ SGL_DEVICE void c4_forward(; `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +65/-39 (104 lines); hunks: -89,10 +89,10 @@ struct C128Trait {; -101,11 +101,15 @@ SGL_DEVICE void c128_forward(; `python/sglang/jit_kernel/dsv4/compress.py` modified +15/-3 (18 lines); hunks: -44,11 +44,14 @@ def _jit_compress_norm_rope_module(; -324,8 +327,17 @@ def compress_forward(; symbols: _jit_compress_norm_rope_module, _jit_compress_module, compress_forward, touching `_jit_compress_norm_rope_module, _jit_compress_module, compress_forward`.
- Code diff details:
  - `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +4/-0 (4 lines); hunks: -395,6 +395,10 @@ def apply_ape_hotfix(self):; symbols: apply_ape_hotfix, get_state_pool
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh` modified +93/-46 (139 lines); hunks: -74,23 +74,27 @@ struct C4Trait {; -102,28 +106,61 @@ SGL_DEVICE void c4_forward(
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +65/-39 (104 lines); hunks: -89,10 +89,10 @@ struct C128Trait {; -101,11 +101,15 @@ SGL_DEVICE void c128_forward(
  - `python/sglang/jit_kernel/dsv4/compress.py` modified +15/-3 (18 lines); hunks: -44,11 +44,14 @@ def _jit_compress_norm_rope_module(; -324,8 +327,17 @@ def compress_forward(; symbols: _jit_compress_norm_rope_module, _jit_compress_module, compress_forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +4/-0; `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh` modified +93/-46; `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +65/-39; `python/sglang/jit_kernel/dsv4/compress.py` modified +15/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/dsv4/compress.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27830 - [Docs] Restore right-hand ToC on the DeepSeek-V4 cookbook page

- Link: https://github.com/sgl-project/sglang/pull/27830
- Status/date: merged / 2026-06-10
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `99258b2f1e1c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +12/-4, 44 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Restore right-hand ToC on the DeepSeek-V4 cookbook page"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[Docs] Restore right-hand ToC on the DeepSeek-V4 cookbook page"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-1 (1 lines); hunks: -2,7 +2,6.
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-1 (1 lines); hunks: -2,7 +2,6
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -2,7 +2,6 @@
-mode: wide
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-1
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #27747 - fix: DSV4 BCG compress-prefill plan OOB on underfilled (tiny) prefill replay

- Link: https://github.com/sgl-project/sglang/pull/27747
- Status/date: merged / 2026-06-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: DSV4 BCG compress-prefill plan OOB on underfilled (tiny) prefill replay"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`; technical summary: Covers "fix: DSV4 BCG compress-prefill plan OOB on underfilled (tiny) prefill replay"; the main implementation surface is `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh` modified +5/-1 (6 lines); hunks: -203,7 +203,11 @@ __global__ __launch_bounds__(1024, 1) //.
- Code diff details:
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh` modified +5/-1 (6 lines); hunks: -203,7 +203,11 @@ __global__ __launch_bounds__(1024, 1) //
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27919 - Revert "[AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase"

- Link: https://github.com/sgl-project/sglang/pull/27919
- Status/date: merged / 2026-06-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +88/-177, 574 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase""; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/layers/attention/dsv4/compressor.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`; technical summary: Covers "Revert "[AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase""; the main implementation surface is `python/sglang/srt/layers/attention/dsv4/compressor.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +0/-4 (4 lines); hunks: -395,10 +395,6 @@ def apply_ape_hotfix(self):; symbols: apply_ape_hotfix, get_state_pool, touching `apply_ape_hotfix, get_state_pool`; `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh` modified +46/-93 (139 lines); hunks: -74,27 +74,23 @@ struct C4Trait {; -106,61 +102,28 @@ SGL_DEVICE void c4_forward(; `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +39/-65 (104 lines); hunks: -89,10 +89,10 @@ struct C128Trait {; -101,15 +101,11 @@ SGL_DEVICE void c128_forward(; `python/sglang/jit_kernel/dsv4/compress.py` modified +3/-15 (18 lines); hunks: -44,14 +44,11 @@ def _jit_compress_norm_rope_module(; -327,17 +324,8 @@ def compress_forward(; symbols: _jit_compress_norm_rope_module, _jit_compress_module, compress_forward, touching `_jit_compress_norm_rope_module, _jit_compress_module, compress_forward`.
- Code diff details:
  - `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +0/-4 (4 lines); hunks: -395,10 +395,6 @@ def apply_ape_hotfix(self):; symbols: apply_ape_hotfix, get_state_pool
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh` modified +46/-93 (139 lines); hunks: -74,27 +74,23 @@ struct C4Trait {; -106,61 +102,28 @@ SGL_DEVICE void c4_forward(
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +39/-65 (104 lines); hunks: -89,10 +89,10 @@ struct C128Trait {; -101,15 +101,11 @@ SGL_DEVICE void c128_forward(
  - `python/sglang/jit_kernel/dsv4/compress.py` modified +3/-15 (18 lines); hunks: -44,14 +44,11 @@ def _jit_compress_norm_rope_module(; -327,17 +324,8 @@ def compress_forward(; symbols: _jit_compress_norm_rope_module, _jit_compress_module, compress_forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +0/-4; `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh` modified +46/-93; `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +39/-65; `python/sglang/jit_kernel/dsv4/compress.py` modified +3/-15
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/dsv4/compress.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27964 - [Spec] Retire Spec V1

- Link: https://github.com/sgl-project/sglang/pull/27964
- Status/date: merged / 2026-06-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 46 files, +111/-252, 1422 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Spec] Retire Spec V1"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`; technical summary: Covers "[Spec] Retire Spec V1"; the main implementation surface is `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass, touching `TestDeepseekMTP, setUpClass, tearDownClass`; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do; `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family, touching `handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp`; `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu....
- Code diff details:
  - `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do
  - `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family
  - `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu...
  - `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10 (21 lines); hunks: -1,6 +1,5; -31,7 +30,8 @@ class TestEagleConstrainedDecoding(; symbols: TestEagleConstrainedDecoding, setUpClass, tearDownClass, TestEagleConstrainedDecodingV2
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/ep/test_deepep_large.py` modified +43/-44; `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10; `python/sglang/test/server_fixtures/standalone_fixture.py` modified +7/-8; `python/sglang/test/server_fixtures/spec_eagle_fixture.py` modified +6/-6
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64; `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_optimization.mdx` modified +3/-8
  - runtime: `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26
- Risk and verification: The diff ships test coverage in `python/sglang/test/server_fixtures/spec_eagle_fixture.py`, `python/sglang/test/server_fixtures/standalone_fixture.py`, `test/manual/dsv4/test_dsv4_flash_mtp_tp8.py`, `test/manual/dsv4/test_dsv4_pro_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27973 - [DSV4] Use int64 for compressor out_loc tensors

- Link: https://github.com/sgl-project/sglang/pull/27973
- Status/date: merged / 2026-06-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +19/-22, 144 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Use int64 for compressor out_loc tensors"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/dsv4/compressor_v2.py`, `python/sglang/srt/layers/attention/dsv4/metadata_kernel.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh`; technical summary: Covers "[DSV4] Use int64 for compressor out_loc tensors"; the main implementation surface is `python/sglang/srt/layers/attention/dsv4/compressor_v2.py`, `python/sglang/srt/layers/attention/dsv4/metadata_kernel.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` modified +2/-5 (7 lines); hunks: -519,12 +519,9 @@ def forward_unified(; symbols: forward_unified, touching `forward_unified`; `python/sglang/srt/layers/attention/dsv4/metadata_kernel.py` modified +2/-2 (4 lines); hunks: -107,12 +107,12 @@ def _init_compressed_attn_metadata_triton(; symbols: _init_compressed_attn_metadata_triton, touching `_init_compressed_attn_metadata_triton`; `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh` modified +14/-14 (28 lines); hunks: -44,7 +44,7 @@ struct FusedNormRopeStoreParams {; -90,7 +90,7 @@ INDEXER_KERNEL void fused_norm_rope_indexer(const __grid_const...; `test/registered/jit/deepseek_v4/test_fp4_indexer.py` modified +1/-1 (2 lines); hunks: -148,7 +148,7 @@ def test_fp4_fused_norm_rope_store_layout(num_tokens: int) -...; symbols: test_fp4_fused_norm_rope_store_layout, touching `test_fp4_fused_norm_rope_store_layout`.
- Code diff details:
  - `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` modified +2/-5 (7 lines); hunks: -519,12 +519,9 @@ def forward_unified(; symbols: forward_unified
  - `python/sglang/srt/layers/attention/dsv4/metadata_kernel.py` modified +2/-2 (4 lines); hunks: -107,12 +107,12 @@ def _init_compressed_attn_metadata_triton(; symbols: _init_compressed_attn_metadata_triton
  - `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh` modified +14/-14 (28 lines); hunks: -44,7 +44,7 @@ struct FusedNormRopeStoreParams {; -90,7 +90,7 @@ INDEXER_KERNEL void fused_norm_rope_indexer(const __grid_const...
  - `test/registered/jit/deepseek_v4/test_fp4_indexer.py` modified +1/-1 (2 lines); hunks: -148,7 +148,7 @@ def test_fp4_fused_norm_rope_store_layout(num_tokens: int) -...; symbols: test_fp4_fused_norm_rope_store_layout
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` modified +2/-5; `python/sglang/srt/layers/attention/dsv4/metadata_kernel.py` modified +2/-2; `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh` modified +14/-14
  - tests: `test/registered/jit/deepseek_v4/test_fp4_indexer.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/jit/deepseek_v4/test_fp4_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27149 - [AMD] [CI] Add dsv4 accuracy PR gate to pr-test-amd-rocm720

- Link: https://github.com/sgl-project/sglang/pull/27149
- Status/date: merged / 2026-06-12
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`; associated commits `1cd5cb1220b9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +165/-20, 306 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] [CI] Add dsv4 accuracy PR gate to pr-test-amd-rocm720"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`; technical summary: Covers "[AMD] [CI] Add dsv4 accuracy PR gate to pr-test-amd-rocm720"; the main implementation surface is `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +5/-1 (6 lines); hunks: -44,7 +44,7; -131,6 +131,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k, touching `test_a_gsm8k, test_b_perf_8k_1k`; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +5/-1 (6 lines); hunks: -44,7 +44,7; -131,6 +131,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k, touching `test_a_gsm8k, test_b_perf_8k_1k`; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +5/-1 (6 lines); hunks: -46,7 +46,7; -133,6 +133,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k, touching `test_a_gsm8k, test_b_perf_8k_1k`; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +5/-1 (6 lines); hunks: -46,7 +46,7; -133,6 +133,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k, touching `test_a_gsm8k, test_b_perf_8k_1k`.
- Code diff details:
  - `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +5/-1 (6 lines); hunks: -44,7 +44,7; -131,6 +131,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +5/-1 (6 lines); hunks: -44,7 +44,7; -131,6 +131,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +5/-1 (6 lines); hunks: -46,7 +46,7; -133,6 +133,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +5/-1 (6 lines); hunks: -46,7 +46,7; -133,6 +133,10 @@ def test_a_gsm8k(self):; symbols: test_a_gsm8k, test_b_perf_8k_1k
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +5/-1; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +5/-1; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +5/-1; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +5/-1
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28098 - Add DeepSeek V4 MTP acceptance length checks

- Link: https://github.com/sgl-project/sglang/pull/28098
- Status/date: merged / 2026-06-13
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py`; associated commits `a14d1a565639`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +36/-3, 222 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DeepSeek V4 MTP acceptance length checks"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`; technical summary: Covers "Add DeepSeek V4 MTP acceptance length checks"; the main implementation surface is `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +7/-0 (7 lines); hunks: -14,6 +14,7; -33,13 +34,16; symbols: TestDSV4FlashFP4B200Balanced_CP, setUpClass, tearDownClass, TestDSV4FlashFP4B200Balanced_CP_NonDeepEP, touching `TestDSV4FlashFP4B200Balanced_CP, setUpClass, tearDownClass`; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +7/-0 (7 lines); hunks: -13,6 +13,7; -32,13 +33,16; symbols: TestDSV4FlashFP4B200, setUpClass, tearDownClass, TestDSV4FlashFP4B200Balanced, touching `TestDSV4FlashFP4B200, setUpClass, tearDownClass`; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +7/-0 (7 lines); hunks: -13,6 +13,7; -41,13 +42,16 @@ def _flashinfer_has_sm90_cutlass_mxfp4() -> bool:; symbols: _flashinfer_has_sm90_cutlass_mxfp4, TestDSV4FlashFP4H200, setUpClass, tearDownClass, touching `_flashinfer_has_sm90_cutlass_mxfp4, TestDSV4FlashFP4H200, setUpClass`; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +7/-0 (7 lines); hunks: -13,6 +13,7; -39,13 +40,16; symbols: TestDSV4FlashFP4B200W4A8MegaMoE, setUpClass, tearDownClass, TestDSV4FlashFP4B200W4A4MegaMoE, touching `TestDSV4FlashFP4B200W4A8MegaMoE, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +7/-0 (7 lines); hunks: -14,6 +14,7; -33,13 +34,16; symbols: TestDSV4FlashFP4B200Balanced_CP, setUpClass, tearDownClass, TestDSV4FlashFP4B200Balanced_CP_NonDeepEP
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +7/-0 (7 lines); hunks: -13,6 +13,7; -32,13 +33,16; symbols: TestDSV4FlashFP4B200, setUpClass, tearDownClass, TestDSV4FlashFP4B200Balanced
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +7/-0 (7 lines); hunks: -13,6 +13,7; -41,13 +42,16 @@ def _flashinfer_has_sm90_cutlass_mxfp4() -> bool:; symbols: _flashinfer_has_sm90_cutlass_mxfp4, TestDSV4FlashFP4H200, setUpClass, tearDownClass
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +7/-0 (7 lines); hunks: -13,6 +13,7; -39,13 +40,16; symbols: TestDSV4FlashFP4B200W4A8MegaMoE, setUpClass, tearDownClass, TestDSV4FlashFP4B200W4A4MegaMoE
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py` modified +4/-0 (4 lines); hunks: -14,6 +14,7; -29,13 +30,16; symbols: TestDSV4FlashFP8H200, setUpClass
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +7/-0; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +7/-0; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +7/-0; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +7/-0; `test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/disaggregation/test_disaggregation_dsv4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27954 - [dsv4] Pad MLA decode q-heads to 64 (not full n_heads) for FlashMLA head64 kernel

- Link: https://github.com/sgl-project/sglang/pull/27954
- Status/date: merged / 2026-06-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `b3be2e74026b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +21/-5, 55 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[dsv4] Pad MLA decode q-heads to 64 (not full n_heads) for FlashMLA head64 kernel"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[dsv4] Pad MLA decode q-heads to 64 (not full n_heads) for FlashMLA head64 kernel"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +21/-5 (26 lines); hunks: -382,6 +382,9 @@ def __init__(; -898,10 +901,23 @@ def forward(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +21/-5 (26 lines); hunks: -382,6 +382,9 @@ def __init__(; -898,10 +901,23 @@ def forward(; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +21/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26471 - DeepSeek-V4 Online Compress support MTP

- Link: https://github.com/sgl-project/sglang/pull/26471
- Status/date: merged / 2026-06-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; associated commits `063ab89ac168`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +1276/-49, 1896 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DeepSeek-V4 Online Compress support MTP"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`; technical summary: Covers "DeepSeek-V4 Online Compress support MTP"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +159/-11 (170 lines); hunks: -35,6 +35,7; -79,6 +80,37; symbols: _get_logical_forward_mode, _get_target_verify_bs, _create_dummy_paged_compress_data, _copy_or_replace, touching `_get_logical_forward_mode, _get_target_verify_bs, _create_dummy_paged_compress_data`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +28/-1 (29 lines); hunks: -35,7 +35,8 @@ def get_compress_state_ring_size(; -458,6 +459,7 @@ def __init__(; symbols: get_compress_state_ring_size, __init__, _init_paged_compress_states, get_attention_compress_states, touching `get_compress_state_ring_size, __init__, _init_paged_compress_states`; `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +12/-1 (13 lines); hunks: -88,18 +88,29 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +159/-11 (170 lines); hunks: -35,6 +35,7; -79,6 +80,37; symbols: _get_logical_forward_mode, _get_target_verify_bs, _create_dummy_paged_compress_data, _copy_or_replace
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +28/-1 (29 lines); hunks: -35,7 +35,8 @@ def get_compress_state_ring_size(; -458,6 +459,7 @@ def __init__(; symbols: get_compress_state_ring_size, __init__, _init_paged_compress_states, get_attention_compress_states
  - `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +12/-1 (13 lines); hunks: -88,18 +88,29 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +159/-11; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +28/-1; `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +12/-1
- Risk and verification: The diff ships test coverage in `test/registered/jit/benchmark/bench_online_c128_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28392 - [AMD] Annotate ATOM source for imported v4 unified attention kernels

- Link: https://github.com/sgl-project/sglang/pull/28392
- Status/date: merged / 2026-06-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; associated commits `800aaefc9e9e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +10/-0, 38 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Annotate ATOM source for imported v4 unified attention kernels"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "[AMD] Annotate ATOM source for imported v4 unified attention kernels"; the main implementation surface is `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-0 (1 lines); hunks: -375,6 +375,7 @@ class DeepSeekV4LayerItem(NamedTuple):; symbols: DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool, touching `DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool`.
- Code diff details:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-0 (1 lines); hunks: -375,6 +375,7 @@ class DeepSeekV4LayerItem(NamedTuple):; symbols: DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -375,6 +375,7 @@ class DeepSeekV4LayerItem(NamedTuple):
+# The following kv pool follows ATOM's unified_kv kernel layout.
```

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/dsv4/unified_kv_kernels/paged_decode.py`, `python/sglang/srt/layers/attention/dsv4/unified_kv_kernels/paged_decode_indices.py`, `python/sglang/srt/layers/attention/dsv4/unified_kv_kernels/paged_prefill.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28290 - [AMD] Test DeepSeek V4 FlashMLA backend variants nightly

- Link: https://github.com/sgl-project/sglang/pull/28290
- Status/date: merged / 2026-06-16
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`; associated commits `0fc2bc4a8bb4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +48/-18, 204 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Test DeepSeek V4 FlashMLA backend variants nightly"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`; technical summary: Covers "[AMD] Test DeepSeek V4 FlashMLA backend variants nightly"; the main implementation surface is `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +4/-3 (7 lines); hunks: -34,6 +34,7; -44,7 +45,7; symbols: test_a_gsm8k, test_b_perf_8k_1k, touching `test_a_gsm8k, test_b_perf_8k_1k`; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +4/-3 (7 lines); hunks: -34,6 +34,7; -44,7 +45,7; symbols: test_a_gsm8k, test_b_perf_8k_1k, touching `test_a_gsm8k, test_b_perf_8k_1k`; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +4/-3 (7 lines); hunks: -36,6 +36,7; -46,7 +47,7; symbols: test_a_gsm8k, test_b_perf_8k_1k, touching `test_a_gsm8k, test_b_perf_8k_1k`; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +4/-3 (7 lines); hunks: -36,6 +36,7; -46,7 +47,7; symbols: test_a_gsm8k, test_b_perf_8k_1k, touching `test_a_gsm8k, test_b_perf_8k_1k`.
- Code diff details:
  - `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +4/-3 (7 lines); hunks: -34,6 +34,7; -44,7 +45,7; symbols: test_a_gsm8k, test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +4/-3 (7 lines); hunks: -34,6 +34,7; -44,7 +45,7; symbols: test_a_gsm8k, test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +4/-3 (7 lines); hunks: -36,6 +36,7; -46,7 +47,7; symbols: test_a_gsm8k, test_b_perf_8k_1k
  - `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +4/-3 (7 lines); hunks: -36,6 +36,7; -46,7 +47,7; symbols: test_a_gsm8k, test_b_perf_8k_1k
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +4/-3; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +4/-3; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +4/-3; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +4/-3
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27928 - [AMD] Feat: Add prefill context parallel support for deepseek v4 unified kv attention

- Link: https://github.com/sgl-project/sglang/pull/27928
- Status/date: merged / 2026-06-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py`; associated commits `a362ba9da37e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +213/-5, 259 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Feat: Add prefill context parallel support for deepseek v4 unified kv attention"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py`; technical summary: Covers "[AMD] Feat: Add prefill context parallel support for deepseek v4 unified kv attention"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +56/-5 (61 lines); hunks: -320,7 +320,7 @@ def apply_cp_reindex(self) -> None:; -342,6 +342,8 @@ def init_flashmla_related(self):; symbols: apply_cp_reindex, init_flashmla_related, _forward_unified_kv, touching `apply_cp_reindex, init_flashmla_related, _forward_unified_kv`; `python/sglang/srt/models/deepseek_v4.py` modified +13/-0 (13 lines); hunks: -819,6 +819,19 @@ def _forward_prepare(; symbols: _forward_prepare, touching `_forward_prepare`; `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: TestDeepseekV4ProFp4CPInterleave, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV4ProFp4CPInterleave, setUpClass, tearDownClass`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +56/-5 (61 lines); hunks: -320,7 +320,7 @@ def apply_cp_reindex(self) -> None:; -342,6 +342,8 @@ def init_flashmla_related(self):; symbols: apply_cp_reindex, init_flashmla_related, _forward_unified_kv
  - `python/sglang/srt/models/deepseek_v4.py` modified +13/-0 (13 lines); hunks: -819,6 +819,19 @@ def _forward_prepare(; symbols: _forward_prepare
  - `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: TestDeepseekV4ProFp4CPInterleave, setUpClass, tearDownClass, test_a_gsm8k
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +56/-5; `python/sglang/srt/models/deepseek_v4.py` modified +13/-0
  - tests: `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` added +144/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28423 - [AMD] Update v4 amd cookbook

- Link: https://github.com/sgl-project/sglang/pull/28423
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; associated commits `0d651e653b98`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +799/-6, 871 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Update v4 amd cookbook"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[AMD] Update v4 amd cookbook"; the main implementation surface is `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +724/-0 (724 lines); hunks: -7,6 +7,8 @@ export const config = {; -43,6 +45,10 @@ export const config = {; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +21/-0 (21 lines); hunks: -261,4 +261,25 @@ export const benchmarks = [; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +34/-4 (38 lines); hunks: -28,15 +28,15 @@ Then run the **Python** output of the command panel below in...; -47,6 +47,28 @@ docker run --gpus all \.
- Code diff details:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +724/-0 (724 lines); hunks: -7,6 +7,8 @@ export const config = {; -43,6 +45,10 @@ export const config = {
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +21/-0 (21 lines); hunks: -261,4 +261,25 @@ export const benchmarks = [
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +34/-4 (38 lines); hunks: -28,15 +28,15 @@ Then run the **Python** output of the command panel below in...; -47,6 +47,28 @@ docker run --gpus all \
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +724/-0; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +21/-0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +34/-4
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/_deployment.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #27277 - Deepseek v4: support mixed dtype compression states

- Link: https://github.com/sgl-project/sglang/pull/27277
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; associated commits `8fd1694dd27f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +1411/-132, 1954 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deepseek v4: support mixed dtype compression states"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "Deepseek v4: support mixed dtype compression states"; the main implementation surface is `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-4 (10 lines); hunks: -448,7 +448,8 @@ def __init__(; -494,7 +495,8 @@ def __init__(; symbols: __init__, _init_paged_compress_states, touching `__init__, _init_paged_compress_states`.
- Code diff details:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-4 (10 lines); hunks: -448,7 +448,8 @@ def __init__(; -494,7 +495,8 @@ def __init__(; symbols: __init__, _init_paged_compress_states
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-4
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `test/registered/jit/test_deepseek_v4_compress_state_runtime_shapes.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28520 - [AMD] Fix deepseek-v4 mtp accept length issue

- Link: https://github.com/sgl-project/sglang/pull/28520
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`; associated commits `f5b041622ba2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +249/-7, 296 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix deepseek-v4 mtp accept length issue"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`; technical summary: Covers "[AMD] Fix deepseek-v4 mtp accept length issue"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +20/-7 (27 lines); hunks: -1316,24 +1316,37 @@ def get_unified_swa_loc(self, forward_batch: ForwardBatc...; symbols: get_unified_swa_loc, store_cache, touching `get_unified_swa_loc, store_cache`; `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` added +179/-0 (179 lines); hunks: -0,0 +1,179; symbols: TestDeepseekV4ProFp4MTP, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV4ProFp4MTP, setUpClass, tearDownClass`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +20/-7 (27 lines); hunks: -1316,24 +1316,37 @@ def get_unified_swa_loc(self, forward_batch: ForwardBatc...; symbols: get_unified_swa_loc, store_cache
  - `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` added +179/-0 (179 lines); hunks: -0,0 +1,179; symbols: TestDeepseekV4ProFp4MTP, setUpClass, tearDownClass, test_a_gsm8k
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +20/-7
  - tests: `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` added +179/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28613 - docs: add DeepSeek-V4 compressed state dtype tip

- Link: https://github.com/sgl-project/sglang/pull/28613
- Status/date: merged / 2026-06-18
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `b55cf4382d0d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +13/-0, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: add DeepSeek-V4 compressed state dtype tip"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs: add DeepSeek-V4 compressed state dtype tip"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -180,6 +180,19 @@ The generator currently picks values on the **conservative*....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -180,6 +180,19 @@ The generator currently picks values on the **conservative*...
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #28590 - [Docs] DeepSeek-V4 cookbook: drop --disable-flashinfer-autotune from GB300 Flash low-latency

- Link: https://github.com/sgl-project/sglang/pull/28590
- Status/date: merged / 2026-06-18
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; associated commits `3f6687330462`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-1, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] DeepSeek-V4 cookbook: drop --disable-flashinfer-autotune from GB300 Flash low-latency"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; technical summary: Covers "[Docs] DeepSeek-V4 cookbook: drop --disable-flashinfer-autotune from GB300 Flash low-latency"; the main implementation surface is `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +7/-0 (7 lines); hunks: -140,6 +140,13 @@ export const benchmarks = [; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +0/-1 (1 lines); hunks: -743,7 +743,6 @@ sgl-eval run aime25 \\.
- Code diff details:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +7/-0 (7 lines); hunks: -140,6 +140,13 @@ export const benchmarks = [
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +0/-1 (1 lines); hunks: -743,7 +743,6 @@ sgl-eval run aime25 \\
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +7/-0; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +0/-1
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #25144 - [NPU] Add Ascend NPU support for DeepSeek-V4

- Link: https://github.com/sgl-project/sglang/pull/25144
- Status/date: merged / 2026-06-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `9b10821c8e6e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 28 files, +4145/-144, 4984 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Add Ascend NPU support for DeepSeek-V4"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`; technical summary: Covers "[NPU] Add Ascend NPU support for DeepSeek-V4"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +103/-24 (127 lines); hunks: -29,6 +29,7; -47,10 +48,15; symbols: __init__, _forward_prepare, touching `__init__, _forward_prepare`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +129/-61 (190 lines); hunks: -568,48 +568,46 @@ def __init__(; -741,6 +739,99 @@ def get_state_buf_infos(self) -> Tuple[List[int], List[int]...; symbols: __init__, get_state_buf_infos, _make_kv_pool, _make_indexer_pool, touching `__init__, get_state_buf_infos, _make_kv_pool`; `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +55/-9 (64 lines); hunks: -2,15 +2,21; -109,24 +115,46 @@ def __init__(; symbols: _lcm, __init__, _alloc_kv_score_buffer, state_cache_3d, touching `_lcm, __init__, _alloc_kv_score_buffer`; `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +21/-10 (31 lines); hunks: -15,25 +15,36 @@ def apply_deepseek_v4_defaults(server_args: ServerArgs, mode...; symbols: apply_deepseek_v4_defaults, touching `apply_deepseek_v4_defaults`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +103/-24 (127 lines); hunks: -29,6 +29,7; -47,10 +48,15; symbols: __init__, _forward_prepare
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +129/-61 (190 lines); hunks: -568,48 +568,46 @@ def __init__(; -741,6 +739,99 @@ def get_state_buf_infos(self) -> Tuple[List[int], List[int]...; symbols: __init__, get_state_buf_infos, _make_kv_pool, _make_indexer_pool
  - `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +55/-9 (64 lines); hunks: -2,15 +2,21; -109,24 +115,46 @@ def __init__(; symbols: _lcm, __init__, _alloc_kv_score_buffer, state_cache_3d
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +21/-10 (31 lines); hunks: -15,25 +15,36 @@ def apply_deepseek_v4_defaults(server_args: ServerArgs, mode...; symbols: apply_deepseek_v4_defaults
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +103/-24; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +129/-61; `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +55/-9; `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +21/-10
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26766 - [DeepSeek-V4] Fuse UE8M0 scale rounding into FP8 group quantization

- Link: https://github.com/sgl-project/sglang/pull/26766
- Status/date: merged / 2026-06-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`; associated commits `bea282cede6c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +70/-23, 181 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek-V4] Fuse UE8M0 scale rounding into FP8 group quantization"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`; technical summary: Covers "[DeepSeek-V4] Fuse UE8M0 scale rounding into FP8 group quantization"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +1/-1 (2 lines); hunks: -1066,8 +1066,8 @@ def forward(; symbols: forward, touching `forward`; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +1/-1 (2 lines); hunks: -41,7 +41,7 @@ class TestDSV4FlashFP4B200(; symbols: TestDSV4FlashFP4B200, touching `TestDSV4FlashFP4B200`; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +1/-1 (2 lines); hunks: -95,7 +95,7 @@ class TestDSV4FlashFP4B200W4A4MegaMoE(; symbols: TestDSV4FlashFP4B200W4A4MegaMoE, touching `TestDSV4FlashFP4B200W4A4MegaMoE`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +1/-1 (2 lines); hunks: -1066,8 +1066,8 @@ def forward(; symbols: forward
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +1/-1 (2 lines); hunks: -41,7 +41,7 @@ class TestDSV4FlashFP4B200(; symbols: TestDSV4FlashFP4B200
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +1/-1 (2 lines); hunks: -95,7 +95,7 @@ class TestDSV4FlashFP4B200W4A4MegaMoE(; symbols: TestDSV4FlashFP4B200W4A4MegaMoE
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +1/-1
  - tests: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +1/-1; `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/jit/test_per_token_group_quant_8bit_v2.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25820 - [NVIDIA] Support NVFP4 MoE for DeepSeek-V4

- Link: https://github.com/sgl-project/sglang/pull/25820
- Status/date: merged / 2026-06-22
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `c0bb04b67f26`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +385/-17, 572 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NVIDIA] Support NVFP4 MoE for DeepSeek-V4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[NVIDIA] Support NVFP4 MoE for DeepSeek-V4"; the main implementation surface is `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +181/-0 (181 lines); hunks: -23,6 +23,7 @@ export const config = {; -38,8 +39,10 @@ export const config = {; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +50/-0 (50 lines); hunks: -136,6 +136,31 @@ export const benchmarks = [; -171,6 +196,31 @@ export const benchmarks = [; `python/sglang/srt/models/deepseek_v4.py` modified +5/-1 (6 lines); hunks: -2293,7 +2293,11 @@ def auto_weight_loader(module):; symbols: auto_weight_loader, touching `auto_weight_loader`; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +25/-0 (25 lines); hunks: -212,6 +212,31 @@ sglang serve \.
- Code diff details:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +181/-0 (181 lines); hunks: -23,6 +23,7 @@ export const config = {; -38,8 +39,10 @@ export const config = {
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +50/-0 (50 lines); hunks: -136,6 +136,31 @@ export const benchmarks = [; -171,6 +196,31 @@ export const benchmarks = [
  - `python/sglang/srt/models/deepseek_v4.py` modified +5/-1 (6 lines); hunks: -2293,7 +2293,11 @@ def auto_weight_loader(module):; symbols: auto_weight_loader
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +25/-0 (25 lines); hunks: -212,6 +212,31 @@ sglang serve \
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +11/-0 (11 lines); hunks: -46,6 +46,17 @@ def apply_deepseek_v4_defaults(server_args: "ServerArgs", mod...; symbols: apply_deepseek_v4_defaults, validate_deepseek_v4_cp
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -23,6 +23,7 @@ export const config = {
+    { id: "nvfp4", label: "NVFP4" },
@@ -38,8 +39,10 @@ export const config = {
+    "flash|nvfp4": "nvidia/DeepSeek-V4-Flash-NVFP4",
+    "pro|nvfp4": "nvidia/DeepSeek-V4-Pro-NVFP4",
@@ -582,6 +585,95 @@ sgl-eval run aime25 \\
+    // ====================================================================
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx
@@ -136,6 +136,31 @@ export const benchmarks = [
+  // GB200 + NVFP4
+  // ====================================================================
+  {
+    match: { hw: "gb200", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
+    sglang_version: "PR #25820",
+    speed: [
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -2293,7 +2293,11 @@ def auto_weight_loader(module):
```

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +181/-0; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +50/-0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +25/-0
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +5/-1; `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +11/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/moe/hash_topk.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28920 - [AMD] deepseek-v4 clean env vars

- Link: https://github.com/sgl-project/sglang/pull/28920
- Status/date: merged / 2026-06-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` and 7 files; associated commits `04d952ea102d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +29/-108, 245 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] deepseek-v4 clean env vars"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`, `test/registered/amd/test_deepseek_v4_flash_fp4.py`; technical summary: Covers "[AMD] deepseek-v4 clean env vars"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py`, `test/registered/amd/test_deepseek_v4_flash_fp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +1/-3 (4 lines); hunks: -1472,13 +1472,11 @@ def forward(; symbols: forward, touching `forward`; `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` modified +2/-17 (19 lines); hunks: -38,28 +38,13; `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +2/-16 (18 lines); hunks: -36,26 +36,12; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +2/-16 (18 lines); hunks: -36,26 +36,12.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +1/-3 (4 lines); hunks: -1472,13 +1472,11 @@ def forward(; symbols: forward
  - `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` modified +2/-17 (19 lines); hunks: -38,28 +38,13
  - `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +2/-16 (18 lines); hunks: -36,26 +36,12
  - `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +2/-16 (18 lines); hunks: -36,26 +36,12
  - `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +2/-16 (18 lines); hunks: -38,26 +38,12
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +1/-3
  - tests: `test/registered/amd/test_deepseek_v4_pro_fp4_mtp.py` modified +2/-17; `test/registered/amd/test_deepseek_v4_flash_fp4.py` modified +2/-16; `test/registered/amd/test_deepseek_v4_flash_fp8.py` modified +2/-16; `test/registered/amd/test_deepseek_v4_pro_fp4.py` modified +2/-16; `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` modified +2/-16; `test/registered/amd/test_deepseek_v4_pro_fp8.py` modified +2/-16
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_v4_flash_fp4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8.py`, `test/registered/amd/test_deepseek_v4_pro_fp4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28941 - [AMD] Fix nightly-8-gpu-mi35x-deepseek-v4-flash-rocm720 OOM issue

- Link: https://github.com/sgl-project/sglang/pull/28941
- Status/date: merged / 2026-06-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; associated commits `cee1caaf476f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +22/-22, 52 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix nightly-8-gpu-mi35x-deepseek-v4-flash-rocm720 OOM issue"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "[AMD] Fix nightly-8-gpu-mi35x-deepseek-v4-flash-rocm720 OOM issue"; the main implementation surface is `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +22/-22 (44 lines); hunks: -578,29 +578,29 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +22/-22 (44 lines); hunks: -578,29 +578,29 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +22/-22
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28981 - [AMD] Update v4 cookbook to clean env vars

- Link: https://github.com/sgl-project/sglang/pull/28981
- Status/date: merged / 2026-06-23
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; associated commits `7e6587c94a1d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-262, 454 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Update v4 cookbook to clean env vars"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[AMD] Update v4 cookbook to clean env vars"; the main implementation surface is `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +12/-257 (269 lines); hunks: -158,8 +158,8 @@ sgl-eval run aime25 \\; -1400,26 +1400,9 @@ sgl-eval run aime25 \\; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-5 (11 lines); hunks: -49,13 +49,14 @@ docker run --gpus all \; -65,7 +66,7 @@ docker run \.
- Code diff details:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +12/-257 (269 lines); hunks: -158,8 +158,8 @@ sgl-eval run aime25 \\; -1400,26 +1400,9 @@ sgl-eval run aime25 \\
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-5 (11 lines); hunks: -49,13 +49,14 @@ docker run --gpus all \; -65,7 +66,7 @@ docker run \
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +12/-257; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-5
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #28938 - [AMD] Improve performance of dsv4 in high concurrency

- Link: https://github.com/sgl-project/sglang/pull/28938
- Status/date: merged / 2026-06-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `af9027f6c938`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +111/-44, 347 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Improve performance of dsv4 in high concurrency"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[AMD] Improve performance of dsv4 in high concurrency"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +29/-0 (29 lines); hunks: -157,6 +157,10 @@ def _is_fused_mhc_post_pre_enabled() -> bool:; -1580,6 +1584,22 @@ def forward(; symbols: _is_fused_mhc_post_pre_enabled, forward, touching `_is_fused_mhc_post_pre_enabled, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +29/-0 (29 lines); hunks: -157,6 +157,10 @@ def _is_fused_mhc_post_pre_enabled() -> bool:; -1580,6 +1584,22 @@ def forward(; symbols: _is_fused_mhc_post_pre_enabled, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +29/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/deepseek_v4_rope.py`, `python/sglang/srt/layers/dp_attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28455 - [AMD] Fix DeepSeek-V4 fp8 KV path on gfx942 (e4m3fnuz)

- Link: https://github.com/sgl-project/sglang/pull/28455
- Status/date: merged / 2026-06-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `5e6d7c1615a9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +52/-16, 205 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix DeepSeek-V4 fp8 KV path on gfx942 (e4m3fnuz)"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[AMD] Fix DeepSeek-V4 fp8 KV path on gfx942 (e4m3fnuz)"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +10/-1 (11 lines); hunks: -127,6 +127,7; -151,6 +152,7 @@ def _is_fused_mhc_post_pre_enabled() -> bool:; symbols: _is_fused_mhc_post_pre_enabled, forward, touching `_is_fused_mhc_post_pre_enabled, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +10/-1 (11 lines); hunks: -127,6 +127,7; -151,6 +152,7 @@ def _is_fused_mhc_post_pre_enabled() -> bool:; symbols: _is_fused_mhc_post_pre_enabled, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +10/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/store.cuh`, `python/sglang/jit_kernel/csrc/dsa/fused_store_index_cache.cuh`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28952 - Add DeepSeek V4 Flash demo notebook

- Link: https://github.com/sgl-project/sglang/pull/28952
- Status/date: merged / 2026-06-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `e97cc339e30c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +1108/-0, 1116 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DeepSeek V4 Flash demo notebook"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "Add DeepSeek V4 Flash demo notebook"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -88,6 +88,10 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -88,6 +88,10 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai...
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -88,6 +88,10 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai/deepseek-v4-benchm
+<Note>
+  For a runnable end-to-end example, see the [DeepSeek-V4-Flash demo notebook](/demo/deepseek_v4_flash).
+</Note>
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #29261 - [Docs] Fix broken links in cookbook

- Link: https://github.com/sgl-project/sglang/pull/29261
- Status/date: merged / 2026-06-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +3/-3, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Fix broken links in cookbook"; model line: DeepSeek V4; category: bug fix; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`; technical summary: Covers "[Docs] Fix broken links in cookbook"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -89,7 +89,7 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai/...; `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +1/-1 (2 lines); hunks: -127,7 +127,7 @@ Pick a weight format by hardware: **NVFP4** on NVIDIA Blackw...; `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -89,7 +89,7 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai/...
  - `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +1/-1 (2 lines); hunks: -127,7 +127,7 @@ Pick a weight format by hardware: **NVFP4** on NVIDIA Blackw...
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#...
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #28103 - Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test

- Link: https://github.com/sgl-project/sglang/pull/28103
- Status/date: merged / 2026-06-25
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/gb300/test_deepseek_v4_pro_fp4.py`; associated commits `3344b73c80b3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +218/-19, 334 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/gb300/test_deepseek_v4_pro_fp4.py`; technical summary: Covers "Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test"; the main implementation surface is `test/registered/gb300/test_deepseek_v4_pro_fp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/gb300/test_deepseek_v4_pro_fp4.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: TestDeepSeekV4ProFp4, test_deepseek_v4_pro_fp4, touching `TestDeepSeekV4ProFp4, test_deepseek_v4_pro_fp4`.
- Code diff details:
  - `test/registered/gb300/test_deepseek_v4_pro_fp4.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: TestDeepSeekV4ProFp4, test_deepseek_v4_pro_fp4
- Key code excerpts:

```diff
diff -- test/registered/gb300/test_deepseek_v4_pro_fp4.py
@@ -0,0 +1,152 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
```

- Reviewed files:
  - tests: `test/registered/gb300/test_deepseek_v4_pro_fp4.py` added +152/-0
- Risk and verification: The diff ships test coverage in `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_glm5_fp8.py`, `test/registered/gb300/test_glm5_nvfp4.py`, `test/registered/gb300/test_kimi_k25.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29103 - [AMD] Feat/dsv4 aiter reduce scatter decode

- Link: https://github.com/sgl-project/sglang/pull/29103
- Status/date: merged / 2026-06-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `b7d3c3016d8c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +124/-9, 232 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Feat/dsv4 aiter reduce scatter decode"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[AMD] Feat/dsv4 aiter reduce scatter decode"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +33/-4 (37 lines); hunks: -57,6 +57,7; -1578,12 +1579,28 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +33/-4 (37 lines); hunks: -57,6 +57,7; -1578,12 +1579,28 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +33/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/environ.py`, `python/sglang/srt/layers/dp_attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27783 - [Intel GPU] DeepSeek V4 3/N: Support hc_split_sinkhorn on XPU using sgl_kernel

- Link: https://github.com/sgl-project/sglang/pull/27783
- Status/date: merged / 2026-06-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `dc113e8804df`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-6, 45 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Intel GPU] DeepSeek V4 3/N: Support hc_split_sinkhorn on XPU using sgl_kernel"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[Intel GPU] DeepSeek V4 3/N: Support hc_split_sinkhorn on XPU using sgl_kernel"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +11/-6 (17 lines); hunks: -68,7 +68,6; -113,9 +112,12; symbols: hc_pre_torch_impl, touching `hc_pre_torch_impl`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +11/-6 (17 lines); hunks: -68,7 +68,6; -113,9 +112,12; symbols: hc_pre_torch_impl
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +11/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29106 - Fix DeepSeek V4 PP HiCache SWA allocation and layer mapping

- Link: https://github.com/sgl-project/sglang/pull/29106
- Status/date: merged / 2026-06-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `c1b5c7e49959`
- Diff scope read: GitHub Pull Request files API returned 5 files, +86/-46, 208 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix DeepSeek V4 PP HiCache SWA allocation and layer mapping"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "Fix DeepSeek V4 PP HiCache SWA allocation and layer mapping"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +2/-2 (4 lines); hunks: -698,7 +698,7 @@ def _forward_prepare_multi_stream_hip(; -799,7 +799,7 @@ def _forward_prepare(; symbols: _forward_prepare_multi_stream_hip, _forward_prepare, touching `_forward_prepare_multi_stream_hip, _forward_prepare`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +5/-1 (6 lines); hunks: -530,6 +530,7 @@ def __init__(; -572,7 +573,7 @@ def __init__(; symbols: __init__, _swa_local_layer_id, get_swa_raw_buffer, get_swa_key_buffer, touching `__init__, _swa_local_layer_id, get_swa_raw_buffer`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +2/-2 (4 lines); hunks: -698,7 +698,7 @@ def _forward_prepare_multi_stream_hip(; -799,7 +799,7 @@ def _forward_prepare(; symbols: _forward_prepare_multi_stream_hip, _forward_prepare
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +5/-1 (6 lines); hunks: -530,6 +530,7 @@ def __init__(; -572,7 +573,7 @@ def __init__(; symbols: __init__, _swa_local_layer_id, get_swa_raw_buffer, get_swa_key_buffer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +2/-2; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +5/-1
- Risk and verification: The diff ships test coverage in `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4.py`, `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4_pp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29502 - [CI] Fix GB300 DSV4 Pro FP4 nightly

- Link: https://github.com/sgl-project/sglang/pull/29502
- Status/date: merged / 2026-06-28
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/gb300/test_deepseek_v4_pro_fp4.py`; associated commits `ae09b8302fa1`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Fix GB300 DSV4 Pro FP4 nightly"; model line: DeepSeek V4; category: bug fix; main diff: `test/registered/gb300/test_deepseek_v4_pro_fp4.py`; technical summary: Covers "[CI] Fix GB300 DSV4 Pro FP4 nightly"; the main implementation surface is `test/registered/gb300/test_deepseek_v4_pro_fp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/gb300/test_deepseek_v4_pro_fp4.py` modified +1/-1 (2 lines); hunks: -69,7 +69,7.
- Code diff details:
  - `test/registered/gb300/test_deepseek_v4_pro_fp4.py` modified +1/-1 (2 lines); hunks: -69,7 +69,7
- Key code excerpts:

```diff
diff -- test/registered/gb300/test_deepseek_v4_pro_fp4.py
@@ -69,7 +69,7 @@
-    "0.85",
+    "0.9",
```

- Reviewed files:
  - tests: `test/registered/gb300/test_deepseek_v4_pro_fp4.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/gb300/test_deepseek_v4_pro_fp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29420 - [AMD][DSV4] Remove per-batch D2H syncs in MTP to avoid bubbles between 2 batches

- Link: https://github.com/sgl-project/sglang/pull/29420
- Status/date: merged / 2026-06-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`; associated commits `54e71506b32f`
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-1, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][DSV4] Remove per-batch D2H syncs in MTP to avoid bubbles between 2 batches"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`; technical summary: Covers "[AMD][DSV4] Remove per-batch D2H syncs in MTP to avoid bubbles between 2 batches"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +9/-1 (10 lines); hunks: -589,11 +589,13 @@ def init_forward_metadata_target_verify(; -876,6 +878,9 @@ def init_forward_metadata_out_graph(; symbols: init_forward_metadata_target_verify, init_forward_metadata_out_graph, init_forward_metadata, touching `init_forward_metadata_target_verify, init_forward_metadata_out_graph, init_forward_metadata`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +9/-1 (10 lines); hunks: -589,11 +589,13 @@ def init_forward_metadata_target_verify(; -876,6 +878,9 @@ def init_forward_metadata_out_graph(; symbols: init_forward_metadata_target_verify, init_forward_metadata_out_graph, init_forward_metadata
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +9/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28980 - [NPU] Support DeepSeek V4 Flash MTP on Ascend

- Link: https://github.com/sgl-project/sglang/pull/28980
- Status/date: merged / 2026-06-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`; associated commits `89620b9169e6`
- Diff scope read: GitHub Pull Request files API returned 13 files, +852/-86, 1365 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Support DeepSeek V4 Flash MTP on Ascend"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4_nextn.py`, `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[NPU] Support DeepSeek V4 Flash MTP on Ascend"; the main implementation surface is `python/sglang/srt/models/deepseek_v4_nextn.py`, `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4_nextn.py` modified +6/-3 (9 lines); hunks: -23,6 +23,7; -91,15 +92,17 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -2145,16 +2145,16 @@ def remap_weight_name_to_dpsk_hf_format(; symbols: remap_weight_name_to_dpsk_hf_format, touching `remap_weight_name_to_dpsk_hf_format`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4_nextn.py` modified +6/-3 (9 lines); hunks: -23,6 +23,7; -91,15 +92,17 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -2145,16 +2145,16 @@ def remap_weight_name_to_dpsk_hf_format(; symbols: remap_weight_name_to_dpsk_hf_format
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4_nextn.py` modified +6/-3; `python/sglang/srt/models/deepseek_v4.py` modified +4/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py`, `python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29827 - [Doc] Tiny update dsv4 doc

- Link: https://github.com/sgl-project/sglang/pull/29827
- Status/date: merged / 2026-07-01
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `677a11bfa960`
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Doc] Tiny update dsv4 doc"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[Doc] Tiny update dsv4 doc"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +2/-2 (4 lines); hunks: -32,7 +32,7 @@ For how to launch the image, see [Install → Method 3: Using Do...; -296,7 +296,7 @@ TCP, which can lead to garbled KV transfer on large checkpoi....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +2/-2 (4 lines); hunks: -32,7 +32,7 @@ For how to launch the image, see [Install → Method 3: Using Do...; -296,7 +296,7 @@ TCP, which can lead to garbled KV transfer on large checkpoi...
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -32,7 +32,7 @@ For how to launch the image, see [Install → Method 3: Using Docker](../../../d
-A single image — `lmsysorg/sglang:latest` — covers the **datacenter GPUs** in this cookbook (B200 / B300 / GB200 / GB300 / H100 / H200). For **RTX PRO 6000 (SM120)**, use the nigh
+A single image — `lmsysorg/sglang:latest` — covers the **datacenter GPUs** in this cookbook (B200 / B300 / GB200 / GB300 / H100 / H200 / RTX PRO 6000).
@@ -296,7 +296,7 @@ TCP, which can lead to garbled KV transfer on large checkpoints.
-HiCache and MegaMoE are **not** supported on RTX PRO 6000. For Docker, use the nightly `lmsysorg/sglang:dev` image — SM120 support isn't in `lmsysorg/sglang:latest` yet (the Deplo
+HiCache and MegaMoE are **not** supported on RTX PRO 6000.
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +2/-2
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #29775 - [DeepSeek V4] Enable FlashMLA sparse prefill by default

- Link: https://github.com/sgl-project/sglang/pull/29775
- Status/date: merged / 2026-07-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`; associated commits `c865347b98ae`
- Diff scope read: GitHub Pull Request files API returned 5 files, +150/-35, 389 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek V4] Enable FlashMLA sparse prefill by default"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`; technical summary: Covers "[DeepSeek V4] Enable FlashMLA sparse prefill by default"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-6 (21 lines); hunks: -55,6 +55,7; -373,8 +374,8 @@ class DSV4Metadata:; symbols: DSV4Metadata, refresh_for_breakable_cuda_graph_replay_, __init__, _move_to_device, touching `DSV4Metadata, refresh_for_breakable_cuda_graph_replay_, __init__`; `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +78/-0 (78 lines); hunks: -17,6 +17,7; -266,6 +267,27 @@ def test_runner_mode_production_eagle_draft_cuda_graph_runn...; symbols: test_runner_mode_production_eagle_draft_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract, _make_sparse_prefill_cache, _make_core_metadata, touching `test_runner_mode_production_eagle_draft_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract, _make_sparse_prefill_cache`; `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +7/-0 (7 lines); hunks: -3,6 +3,8; -93,6 +95,11 @@ def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:; symbols: validate_deepseek_v4_cp, touching `validate_deepseek_v4_cp`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-6 (21 lines); hunks: -55,6 +55,7; -373,8 +374,8 @@ class DSV4Metadata:; symbols: DSV4Metadata, refresh_for_breakable_cuda_graph_replay_, __init__, _move_to_device
  - `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +78/-0 (78 lines); hunks: -17,6 +17,7; -266,6 +267,27 @@ def test_runner_mode_production_eagle_draft_cuda_graph_runn...; symbols: test_runner_mode_production_eagle_draft_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract, _make_sparse_prefill_cache, _make_core_metadata
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +7/-0 (7 lines); hunks: -3,6 +3,8; -93,6 +95,11 @@ def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:; symbols: validate_deepseek_v4_cp
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-6; `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +7/-0
  - tests: `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +78/-0
- Risk and verification: The diff ships test coverage in `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29885 - [DeepSeek V4] Cover both dense and sparse prefill paths in the compress attention unittest

- Link: https://github.com/sgl-project/sglang/pull/29885
- Status/date: merged / 2026-07-02
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`; associated commits `307094dc7d0a`
- Diff scope read: GitHub Pull Request files API returned 2 files, +105/-17, 205 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek V4] Cover both dense and sparse prefill paths in the compress attention unittest"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`; technical summary: Covers "[DeepSeek V4] Cover both dense and sparse prefill paths in the compress attention unittest"; the main implementation surface is `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +15/-1 (16 lines); hunks: -183,13 +183,27 @@ def test_runner_mode_cuda_graph_decode_cases(self):; symbols: test_runner_mode_cuda_graph_decode_cases, test_compress_attention_cases, test_compress_attention_cases_sparse_prefill, test_eagle_target_verify_chain_cases, touching `test_runner_mode_cuda_graph_decode_cases, test_compress_attention_cases, test_compress_attention_cases_sparse_prefill`.
- Code diff details:
  - `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +15/-1 (16 lines); hunks: -183,13 +183,27 @@ def test_runner_mode_cuda_graph_decode_cases(self):; symbols: test_runner_mode_cuda_graph_decode_cases, test_compress_attention_cases, test_compress_attention_cases_sparse_prefill, test_eagle_target_verify_chain_cases
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +15/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29982 - [AMD][DeepSeek V4] Fix default FlashMLA sparse prefill off on ROCm/HIP

- Link: https://github.com/sgl-project/sglang/pull/29982
- Status/date: merged / 2026-07-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/arg_groups/deepseek_v4_hook.py`; associated commits `8519be82e8ed`
- Diff scope read: GitHub Pull Request files API returned 1 files, +13/-0, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][DeepSeek V4] Fix default FlashMLA sparse prefill off on ROCm/HIP"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/arg_groups/deepseek_v4_hook.py`; technical summary: Covers "[AMD][DeepSeek V4] Fix default FlashMLA sparse prefill off on ROCm/HIP"; the main implementation surface is `python/sglang/srt/arg_groups/deepseek_v4_hook.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +13/-0 (13 lines); hunks: -14,6 +14,19; symbols: apply_deepseek_v4_defaults, touching `apply_deepseek_v4_defaults`.
- Code diff details:
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +13/-0 (13 lines); hunks: -14,6 +14,19; symbols: apply_deepseek_v4_defaults
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +13/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/arg_groups/deepseek_v4_hook.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29619 - [DeepSeek-V4] Add an opt-in non-paged indexer for long-context prefill

- Link: https://github.com/sgl-project/sglang/pull/29619
- Status/date: merged / 2026-07-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; associated commits `a6ee64d237a2`
- Diff scope read: GitHub Pull Request files API returned 6 files, +468/-50, 681 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek-V4] Add an opt-in non-paged indexer for long-context prefill"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "[DeepSeek-V4] Add an opt-in non-paged indexer for long-context prefill"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +11/-2 (13 lines); hunks: -550,11 +550,17 @@ def _make_target_verify_c128_metadata(; -637,7 +643,10 @@ def init_forward_metadata_prefill(; symbols: _make_target_verify_c128_metadata, init_forward_metadata_indexer, init_forward_metadata_decode, init_forward_metadata_prefill, touching `_make_target_verify_c128_metadata, init_forward_metadata_indexer, init_forward_metadata_decode`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +17/-4 (21 lines); hunks: -317,12 +317,19 @@ def get_index_k_with_scale_buffer(self, layer_id: int) ->...; -978,14 +985,20 @@ def get_index_k_with_scale_buffer(self, layer_id: int) ->...; symbols: get_index_k_with_scale_buffer, get_index_k_scale_buffer, set_index_k_scale_buffer, touching `get_index_k_with_scale_buffer, get_index_k_scale_buffer, set_index_k_scale_buffer`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +11/-2 (13 lines); hunks: -550,11 +550,17 @@ def _make_target_verify_c128_metadata(; -637,7 +643,10 @@ def init_forward_metadata_prefill(; symbols: _make_target_verify_c128_metadata, init_forward_metadata_indexer, init_forward_metadata_decode, init_forward_metadata_prefill
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +17/-4 (21 lines); hunks: -317,12 +317,19 @@ def get_index_k_with_scale_buffer(self, layer_id: int) ->...; -978,14 +985,20 @@ def get_index_k_with_scale_buffer(self, layer_id: int) ->...; symbols: get_index_k_with_scale_buffer, get_index_k_scale_buffer, set_index_k_scale_buffer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +11/-2; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +17/-4
- Risk and verification: The diff ships test coverage in `test/registered/unit/layers/test_dsv4_nonpaged_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29988 - [dsv4] Trigger MHC prenorm prewarm at weight-load time with rank sync

- Link: https://github.com/sgl-project/sglang/pull/29988
- Status/date: merged / 2026-07-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `e81f05cf4f44`
- Diff scope read: GitHub Pull Request files API returned 3 files, +68/-180, 318 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[dsv4] Trigger MHC prenorm prewarm at weight-load time with rank sync"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[dsv4] Trigger MHC prenorm prewarm at weight-load time with rank sync"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +64/-115 (179 lines); hunks: -1182,121 +1182,6 @@ def refresh_mhc_norm_weight_cache(self):; -1966,6 +1851,11 @@ def __init__(; symbols: refresh_mhc_norm_weight_cache, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets, hc_pre, touching `refresh_mhc_norm_weight_cache, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +64/-115 (179 lines); hunks: -1182,121 +1182,6 @@ def refresh_mhc_norm_weight_cache(self):; -1966,6 +1851,11 @@ def __init__(; symbols: refresh_mhc_norm_weight_cache, prewarm_mhc_token_counts, prewarm_mhc_token_count_buckets, hc_pre
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +64/-115
- Risk and verification: The diff ships test coverage in `test/registered/kernels/test_mhc_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27349 - Support DSV4 shared expert fusion for DeepEP and MegaMOE

- Link: https://github.com/sgl-project/sglang/pull/27349
- Status/date: merged / 2026-07-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`, `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py`; associated commits `d364cd8ead47`
- Diff scope read: GitHub Pull Request files API returned 13 files, +532/-87, 966 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support DSV4 shared expert fusion for DeepEP and MegaMOE"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py`, `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "Support DSV4 shared expert fusion for DeepEP and MegaMOE"; the main implementation surface is `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py`, `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py` added +50/-0 (50 lines); hunks: -0,0 +1,50; symbols: TestDeepseekV4SharedExpertFusionPolicy, _make_model, test_disables_shared_fusion_without_enforce, test_enables_shared_fusion_when_enforced, touching `TestDeepseekV4SharedExpertFusionPolicy, _make_model, test_disables_shared_fusion_without_enforce`; `python/sglang/srt/models/deepseek_v4.py` modified +10/-13 (23 lines); hunks: -1975,28 +1975,25 @@ def determine_num_fused_shared_experts(self):; symbols: determine_num_fused_shared_experts, forward, touching `determine_num_fused_shared_experts, forward`.
- Code diff details:
  - `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py` added +50/-0 (50 lines); hunks: -0,0 +1,50; symbols: TestDeepseekV4SharedExpertFusionPolicy, _make_model, test_disables_shared_fusion_without_enforce, test_enables_shared_fusion_when_enforced
  - `python/sglang/srt/models/deepseek_v4.py` modified +10/-13 (23 lines); hunks: -1975,28 +1975,25 @@ def determine_num_fused_shared_experts(self):; symbols: determine_num_fused_shared_experts, forward
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py` added +50/-0
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +10/-13
- Risk and verification: The diff ships test coverage in `test/registered/moe/test_fused_append_remap_per_rank_shared_slots.py`, `test/registered/moe/test_hash_topk.py`, `test/registered/unit/eplb/test_deepep_waterfill_eplb.py`, `test/registered/unit/layers/moe/test_fused_shared_expert_scaling.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27914 - [Intel GPU] DeepSeek V4 6/N: use sgl-kernel implemetation of flash_mla_with_kvcache on XPU

- Link: https://github.com/sgl-project/sglang/pull/27914
- Status/date: merged / 2026-07-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; associated commits `4dddb0432553`
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-4, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Intel GPU] DeepSeek V4 6/N: use sgl-kernel implemetation of flash_mla_with_kvcache on XPU"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; technical summary: Covers "[Intel GPU] DeepSeek V4 6/N: use sgl-kernel implemetation of flash_mla_with_kvcache on XPU"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +8/-4 (12 lines); hunks: -50,7 +50,7; -60,6 +60,7; symbols: _pad_last_dim, _create_flashmla_metadata, match_num_queries, touching `_pad_last_dim, _create_flashmla_metadata, match_num_queries`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +8/-4 (12 lines); hunks: -50,7 +50,7; -60,6 +60,7; symbols: _pad_last_dim, _create_flashmla_metadata, match_num_queries
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +8/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29362 - [AMD ]Feat/dsv4 ep tbo prefill

- Link: https://github.com/sgl-project/sglang/pull/29362
- Status/date: merged / 2026-07-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py`; associated commits `81735ecf8099`
- Diff scope read: GitHub Pull Request files API returned 11 files, +1008/-31, 1213 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD ]Feat/dsv4 ep tbo prefill"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py`; technical summary: Covers "[AMD ]Feat/dsv4 ep tbo prefill"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +375/-24 (399 lines); hunks: -53,15 +53,21; -1108,6 +1114,21 @@ def forward(; symbols: forward, op_attn, DeepseekV4DecoderLayer, __init__, touching `forward, op_attn, DeepseekV4DecoderLayer`; `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +8/-0 (8 lines); hunks: -406,6 +406,14 @@ def of(cls, forward_mode: ForwardMode) -> _GraphBucket:; symbols: of, DeepseekV4HipRadixBackend, __init__, touching `of, DeepseekV4HipRadixBackend, __init__`; `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: TestDeepseekV4FlashFp8Tbo, setUpClass, tearDownClass, test_gsm8k_tbo, touching `TestDeepseekV4FlashFp8Tbo, setUpClass, tearDownClass`; `test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: TestDeepseekV4ProFp4Tbo, setUpClass, tearDownClass, test_gsm8k_tbo, touching `TestDeepseekV4ProFp4Tbo, setUpClass, tearDownClass`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +375/-24 (399 lines); hunks: -53,15 +53,21; -1108,6 +1114,21 @@ def forward(; symbols: forward, op_attn, DeepseekV4DecoderLayer, __init__
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +8/-0 (8 lines); hunks: -406,6 +406,14 @@ def of(cls, forward_mode: ForwardMode) -> _GraphBucket:; symbols: of, DeepseekV4HipRadixBackend, __init__
  - `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: TestDeepseekV4FlashFp8Tbo, setUpClass, tearDownClass, test_gsm8k_tbo
  - `test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: TestDeepseekV4ProFp4Tbo, setUpClass, tearDownClass, test_gsm8k_tbo
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +375/-24; `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +8/-0
  - tests: `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py` added +163/-0; `test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py` added +151/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py`, `test/registered/unit/server_args/test_server_args.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30237 - [AMD][DeepSeek V4] Set SGLANG_OPT_FLASHMLA_SPARSE_PREFILL to false on hip code path

- Link: https://github.com/sgl-project/sglang/pull/30237
- Status/date: merged / 2026-07-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/arg_groups/deepseek_v4_hook.py`; associated commits `80decc78ec22`
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][DeepSeek V4] Set SGLANG_OPT_FLASHMLA_SPARSE_PREFILL to false on hip code path"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/arg_groups/deepseek_v4_hook.py`; technical summary: Covers "[AMD][DeepSeek V4] Set SGLANG_OPT_FLASHMLA_SPARSE_PREFILL to false on hip code path"; the main implementation surface is `python/sglang/srt/arg_groups/deepseek_v4_hook.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +2/-2 (4 lines); hunks: -20,8 +20,8 @@ def apply_deepseek_v4_defaults(server_args: ServerArgs, model_...; symbols: apply_deepseek_v4_defaults, touching `apply_deepseek_v4_defaults`.
- Code diff details:
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +2/-2 (4 lines); hunks: -20,8 +20,8 @@ def apply_deepseek_v4_defaults(server_args: ServerArgs, model_...; symbols: apply_deepseek_v4_defaults
- Key code excerpts:

```diff
diff -- python/sglang/srt/arg_groups/deepseek_v4_hook.py
@@ -20,8 +20,8 @@ def apply_deepseek_v4_defaults(server_args: ServerArgs, model_arch: str) -> None
-    # there; an explicit env var still overrides this.
-    if is_hip() and not envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.is_set():
+    # there;
+    if is_hip():
```

- Reviewed files:
  - runtime: `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/arg_groups/deepseek_v4_hook.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27867 - [DSv4] Loading Time Weight Dequant

- Link: https://github.com/sgl-project/sglang/pull/27867
- Status/date: merged / 2026-07-07
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`; associated commits `627980596254`
- Diff scope read: GitHub Pull Request files API returned 5 files, +148/-3, 234 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4] Loading Time Weight Dequant"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`; technical summary: Covers "[DSv4] Loading Time Weight Dequant"; the main implementation surface is `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +45/-2 (47 lines); hunks: -4,7 +4,10; -21,7 +24,7; symbols: _flashinfer_has_sm90_cutlass_mxfp4, tearDownClass, TestDSV4FlashFP4DequantTP8H200, setUpClass, touching `_flashinfer_has_sm90_cutlass_mxfp4, tearDownClass, TestDSV4FlashFP4DequantTP8H200`.
- Code diff details:
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +45/-2 (47 lines); hunks: -4,7 +4,10; -21,7 +24,7; symbols: _flashinfer_has_sm90_cutlass_mxfp4, tearDownClass, TestDSV4FlashFP4DequantTP8H200, setUpClass
- Key code excerpts:

```diff
diff -- test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py
@@ -4,7 +4,10 @@
-Registry: base-c-test-deepep-8-gpu-h200 (per-commit, 8x H200 — only 4 used by TP=4)
+Also covers SGLANG_DSV4_FP4_DEQUANT=1 (TP=8): FP4 experts dequantized to FP8
+during loading and served through the plain FP8 MoE path.
+Registry: base-c-test-deepep-8-gpu-h200 (per-commit, 8x H200)
@@ -21,7 +24,7 @@
-register_cuda_ci(est_time=370, stage="base-c", runner_config="deepep-8-gpu-h200")
```

- Reviewed files:
  - tests: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py` modified +45/-2
- Risk and verification: The diff ships test coverage in `test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30333 - [AMD] Fix DeepSeek V4 MTP accuracy issue

- Link: https://github.com/sgl-project/sglang/pull/30333
- Status/date: merged / 2026-07-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`; associated commits `9a6f8e599204`
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-1, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix DeepSeek V4 MTP accuracy issue"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`; technical summary: Covers "[AMD] Fix DeepSeek V4 MTP accuracy issue"; the main implementation surface is `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +10/-1 (11 lines); hunks: -129,7 +129,16 @@ def __init__(; symbols: __init__, _alloc_kv_score_buffer, touching `__init__, _alloc_kv_score_buffer`.
- Code diff details:
  - `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +10/-1 (11 lines); hunks: -129,7 +129,16 @@ def __init__(; symbols: __init__, _alloc_kv_score_buffer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` modified +10/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27926 - [DSV4] perf: Make FP8 quant output tensor contiguous

- Link: https://github.com/sgl-project/sglang/pull/27926
- Status/date: merged / 2026-07-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `d7dcdf3efd2c`
- Diff scope read: GitHub Pull Request files API returned 5 files, +479/-7, 527 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] perf: Make FP8 quant output tensor contiguous"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[DSV4] perf: Make FP8 quant output tensor contiguous"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +3/-7 (10 lines); hunks: -24,6 +24,7; -70,7 +71,6; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +3/-7 (10 lines); hunks: -24,6 +24,7; -70,7 +71,6; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +3/-7
- Risk and verification: The diff ships test coverage in `test/registered/jit/deepseek_v4/test_fp8_wo_a.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29417 - [AMD] Enable unified-KV HiCache on DeepSeek-V4

- Link: https://github.com/sgl-project/sglang/pull/29417
- Status/date: merged / 2026-07-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; associated commits `8d0fd3415077`
- Diff scope read: GitHub Pull Request files API returned 8 files, +208/-99, 483 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Enable unified-KV HiCache on DeepSeek-V4"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "[AMD] Enable unified-KV HiCache on DeepSeek-V4"; the main implementation surface is `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +51/-4 (55 lines); hunks: -390,7 +390,7 @@ class DeepSeekV4LayerItem(NamedTuple):; -403,6 +403,7 @@ def __init__(; symbols: DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool, __init__, touching `DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool, __init__`.
- Code diff details:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +51/-4 (55 lines); hunks: -390,7 +390,7 @@ class DeepSeekV4LayerItem(NamedTuple):; -403,6 +403,7 @@ def __init__(; symbols: DeepSeekV4LayerItem, DeepSeekV4UnifiedKVPool, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +51/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/managers/schedule_policy.py`, `python/sglang/srt/mem_cache/base_prefix_cache.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30695 - [Refactor] Make DeepSeek-V4 attention backend tolerate an absent CPU seq_lens mirror

- Link: https://github.com/sgl-project/sglang/pull/30695
- Status/date: merged / 2026-07-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; associated commits `504570f4250d`
- Diff scope read: GitHub Pull Request files API returned 1 files, +26/-17, 109 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Make DeepSeek-V4 attention backend tolerate an absent CPU seq_lens mirror"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; technical summary: Covers "[Refactor] Make DeepSeek-V4 attention backend tolerate an absent CPU seq_lens mirror"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +26/-17 (43 lines); hunks: -457,6 +457,8 @@ class DeepseekV4AttnBackend(; -509,6 +511,11 @@ def __init__(; symbols: DeepseekV4AttnBackend, __init__, _move_to_device, init_forward_metadata_target_verify, touching `DeepseekV4AttnBackend, __init__, _move_to_device`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +26/-17 (43 lines); hunks: -457,6 +457,8 @@ class DeepseekV4AttnBackend(; -509,6 +511,11 @@ def __init__(; symbols: DeepseekV4AttnBackend, __init__, _move_to_device, init_forward_metadata_target_verify
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +26/-17
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30711 - [Refactor] Split DeepSeek-V4 MQALayer into a reusable attention base

- Link: https://github.com/sgl-project/sglang/pull/30711
- Status/date: merged / 2026-07-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `fef5eda4fb2c`
- Diff scope read: GitHub Pull Request files API returned 1 files, +281/-142, 564 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Split DeepSeek-V4 MQALayer into a reusable attention base"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[Refactor] Split DeepSeek-V4 MQALayer into a reusable attention base"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +281/-142 (423 lines); hunks: -8,7 +8,6; -158,6 +157,11; symbols: _is_fused_mhc_post_pre_enabled, _fused_rmsnorm_fp8_quant, make_hc_mixing_params, make_hc_head_params, touching `_is_fused_mhc_post_pre_enabled, _fused_rmsnorm_fp8_quant, make_hc_mixing_params`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +281/-142 (423 lines); hunks: -8,7 +8,6; -158,6 +157,11; symbols: _is_fused_mhc_post_pre_enabled, _fused_rmsnorm_fp8_quant, make_hc_mixing_params, make_hc_head_params
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +281/-142
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30898 - Enable breakable prefill CUDA graph for DP attention

- Link: https://github.com/sgl-project/sglang/pull/30898
- Status/date: merged / 2026-07-13
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`; associated commits `771e38633216`, `b94ac87e0c41`
- Diff scope read: GitHub Pull Request files API returned 9 files, +478/-20, 731 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable breakable prefill CUDA graph for DP attention"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`; technical summary: Covers "Enable breakable prefill CUDA graph for DP attention"; the main implementation surface is `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +2/-0 (2 lines); hunks: -75,6 +75,8 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`.
- Code diff details:
  - `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +2/-0 (2 lines); hunks: -75,6 +75,8 @@ def setUpClass(cls):; symbols: setUpClass
- Key code excerpts:

```diff
diff -- test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py
@@ -75,6 +75,8 @@ def setUpClass(cls):
+                "--mem-fraction-static",
+                "0.80",
```

- Reviewed files:
  - tests: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/dp_attn/test_dp_attention_bcg_kl.py`, `test/registered/unit/model_executor/test_cuda_graph_buffer_registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31125 - Disable flaky DSV4-Flash FP4 BCG determinism test (nondeterminism from #30898 idle-rank dummy extend)

- Link: https://github.com/sgl-project/sglang/pull/31125
- Status/date: merged / 2026-07-14
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`; associated commits `771e38633216`
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-0, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Disable flaky DSV4-Flash FP4 BCG determinism test (nondeterminism from #30898 idle-rank dummy extend)"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`; technical summary: Covers "Disable flaky DSV4-Flash FP4 BCG determinism test (nondeterminism from #30898 idle-rank dummy extend)"; the main implementation surface is `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +11/-0 (11 lines); hunks: -170,6 +170,17 @@ class TestDSV4FlashFP4BreakableCudaGraphB200(; symbols: TestDSV4FlashFP4BreakableCudaGraphB200, test_determinism_temp_zero, setUpClass, touching `TestDSV4FlashFP4BreakableCudaGraphB200, test_determinism_temp_zero, setUpClass`.
- Code diff details:
  - `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +11/-0 (11 lines); hunks: -170,6 +170,17 @@ class TestDSV4FlashFP4BreakableCudaGraphB200(; symbols: TestDSV4FlashFP4BreakableCudaGraphB200, test_determinism_temp_zero, setUpClass
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py` modified +11/-0
- Risk and verification: The diff ships test coverage in `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30365 - [DSV4] Remove per-step seqlen D2H from speculative to make overlap scheduler work

- Link: https://github.com/sgl-project/sglang/pull/30365
- Status/date: merged / 2026-07-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`; associated commits `a9cf5e68e688`
- Diff scope read: GitHub Pull Request files API returned 5 files, +115/-52, 403 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Remove per-step seqlen D2H from speculative to make overlap scheduler work"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`; technical summary: Covers "[DSV4] Remove per-step seqlen D2H from speculative to make overlap scheduler work"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +64/-48 (112 lines); hunks: -68,7 +68,7; -79,6 +79,7; symbols: __init__, _make_target_verify_c128_metadata, init_forward_metadata_target_verify, touching `__init__, _make_target_verify_c128_metadata, init_forward_metadata_target_verify`; `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +29/-3 (32 lines); hunks: -26,7 +26,10; -35,6 +38,7; symbols: test_runner_mode_eagle_verify_cuda_graph_cases, test_eagle_draft_extend_without_cpu_seq_lens, test_runner_mode_production_eagle_draft_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract, touching `test_runner_mode_eagle_verify_cuda_graph_cases, test_eagle_draft_extend_without_cpu_seq_lens, test_runner_mode_production_eagle_draft_cuda_graph_runner_cases`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +64/-48 (112 lines); hunks: -68,7 +68,7; -79,6 +79,7; symbols: __init__, _make_target_verify_c128_metadata, init_forward_metadata_target_verify
  - `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +29/-3 (32 lines); hunks: -26,7 +26,10; -35,6 +38,7; symbols: test_runner_mode_eagle_verify_cuda_graph_cases, test_eagle_draft_extend_without_cpu_seq_lens, test_runner_mode_production_eagle_draft_cuda_graph_runner_cases, TestDSV4BreakableCudaGraphMetadataContract
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +64/-48
  - tests: `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +29/-3
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/runner_modes/speculative_draft_runner.py`, `python/sglang/test/kits/attention_unittest/runner_modes/speculative_target_verify_runner.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30792 - [Kernel] Migrate DSA + DSV4 attention kernels to sglang.kernels (RFC #29630, Phase 2.5, 5/7)

- Link: https://github.com/sgl-project/sglang/pull/30792
- Status/date: merged / 2026-07-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `ba5be86d42af`
- Diff scope read: GitHub Pull Request files API returned 60 files, +662/-582, 1730 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Migrate DSA + DSV4 attention kernels to sglang.kernels (RFC #29630, Phase 2.5, 5/7)"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[Kernel] Migrate DSA + DSV4 attention kernels to sglang.kernels (RFC #29630, Phase 2.5, 5/7)"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +11/-11 (22 lines); hunks: -18,6 +18,12; -31,12 +37,6; symbols: _attach_unified_kv_decode_streams, _attach_unified_kv_prefill_meta, _forward_unified_kv, forward, touching `_attach_unified_kv_decode_streams, _attach_unified_kv_prefill_meta, _forward_unified_kv`; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +9/-9 (18 lines); hunks: -19,6 +19,15; -31,22 +40,13; `python/sglang/srt/models/deepseek_v4.py` modified +2/-2 (4 lines); hunks: -876,7 +876,7 @@ def _forward_prepare(; -1130,7 +1130,7 @@ def forward(; symbols: _forward_prepare, forward, touching `_forward_prepare, forward`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +7/-7 (14 lines); hunks: -11,13 +11,13; -368,7 +368,7 @@ def set_index_fp4(; symbols: set_index_fp4, __init__, touching `set_index_fp4, __init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +11/-11 (22 lines); hunks: -18,6 +18,12; -31,12 +37,6; symbols: _attach_unified_kv_decode_streams, _attach_unified_kv_prefill_meta, _forward_unified_kv, forward
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +9/-9 (18 lines); hunks: -19,6 +19,15; -31,22 +40,13
  - `python/sglang/srt/models/deepseek_v4.py` modified +2/-2 (4 lines); hunks: -876,7 +876,7 @@ def _forward_prepare(; -1130,7 +1130,7 @@ def forward(; symbols: _forward_prepare, forward
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +7/-7 (14 lines); hunks: -11,13 +11,13; -368,7 +368,7 @@ def set_index_fp4(; symbols: set_index_fp4, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +11/-11; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +9/-9; `python/sglang/srt/models/deepseek_v4.py` modified +2/-2; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +7/-7
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `test/manual/layers/attention/dsa/test_act_quant_triton.py`, `test/manual/layers/attention/dsa/test_get_k_scale_triton_kernel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30651 - cookbook(deepseek-v4): add MORI disagg backend for AMD + bump MI355X image

- Link: https://github.com/sgl-project/sglang/pull/30651
- Status/date: merged / 2026-07-15
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; associated commits `fbcbe0a986f1`
- Diff scope read: GitHub Pull Request files API returned 2 files, +52/-29, 274 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "cookbook(deepseek-v4): add MORI disagg backend for AMD + bump MI355X image"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; technical summary: Covers "cookbook(deepseek-v4): add MORI disagg backend for AMD + bump MI355X image"; the main implementation surface is `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +48/-29 (77 lines); hunks: -161,7 +161,7 @@ sgl-eval run aime25 \\; -289,6 +289,9 @@ sgl-eval run aime25 \\.
- Code diff details:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +48/-29 (77 lines); hunks: -161,7 +161,7 @@ sgl-eval run aime25 \\; -289,6 +289,9 @@ sgl-eval run aime25 \\
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -161,7 +161,7 @@ sgl-eval run aime25 \\
-    mi355x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260623",
+    mi355x: "lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260708",
@@ -289,6 +289,9 @@ sgl-eval run aime25 \\
+        // MORI-IO transport is AMD-only — hidden on every non-ROCm platform.
+        { id: "mori",     label: "MORI",
+          hide: { hw: ["h100", "h200", "b200", "b300", "gb200", "gb300", "rtx6000"] } },
```

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +48/-29
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/_playground.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #28983 - perf(deepseek_v4): enable SGLANG_OPT_FP8_WO_A_GEMM on sm90 (Hopper)

- Link: https://github.com/sgl-project/sglang/pull/28983
- Status/date: merged / 2026-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `dee91c51cf78`
- Diff scope read: GitHub Pull Request files API returned 4 files, +217/-18, 306 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "perf(deepseek_v4): enable SGLANG_OPT_FP8_WO_A_GEMM on sm90 (Hopper)"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "perf(deepseek_v4): enable SGLANG_OPT_FP8_WO_A_GEMM on sm90 (Hopper)"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +54/-12 (66 lines); hunks: -32,6 +32,9; -495,10 +498,14 @@ def __init__(; symbols: __init__, forward, _setup_fp8_wo_a_scales, touching `__init__, forward, _setup_fp8_wo_a_scales`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +54/-12 (66 lines); hunks: -32,6 +32,9; -495,10 +498,14 @@ def __init__(; symbols: __init__, forward, _setup_fp8_wo_a_scales
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +54/-12
- Risk and verification: The diff ships test coverage in `test/manual/dsv4/test_wo_a_fp8_sm90.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31373 - [Docs] Align B200 DeepSeek-V4-Pro balanced recipe with MegaMoE

- Link: https://github.com/sgl-project/sglang/pull/31373
- Status/date: merged / 2026-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; associated commits `a61482134186`
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-4, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Align B200 DeepSeek-V4-Pro balanced recipe with MegaMoE"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; technical summary: Covers "[Docs] Align B200 DeepSeek-V4-Pro balanced recipe with MegaMoE"; the main implementation surface is `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +4/-4 (8 lines); hunks: -427,15 +427,16 @@ sgl-eval run aime25 \\; -444,7 +445,6 @@ sgl-eval run aime25 \\.
- Code diff details:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +4/-4 (8 lines); hunks: -427,15 +427,16 @@ sgl-eval run aime25 \\; -444,7 +445,6 @@ sgl-eval run aime25 \\
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +4/-4
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #31122 - [Docs] Add AMD-specific HiCache config for DeepSeek V4 playground

- Link: https://github.com/sgl-project/sglang/pull/31122
- Status/date: merged / 2026-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; associated commits `3264477a0716`
- Diff scope read: GitHub Pull Request files API returned 3 files, +29/-7, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Add AMD-specific HiCache config for DeepSeek V4 playground"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[Docs] Add AMD-specific HiCache config for DeepSeek V4 playground"; the main implementation surface is `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +9/-3 (12 lines); hunks: -312,12 +312,18 @@ sgl-eval run aime25 \\; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -514,6 +514,10 @@ To enable HiCache, open the **HiCache** card in the [Playgr....
- Code diff details:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +9/-3 (12 lines); hunks: -312,12 +312,18 @@ sgl-eval run aime25 \\
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -514,6 +514,10 @@ To enable HiCache, open the **HiCache** card in the [Playgr...
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -312,12 +312,18 @@ sgl-eval run aime25 \\
+      // AMD ROCm (MI300X/MI325X/MI350X/MI355X): page_first_direct + direct io.
+      amdIo: { memLayout: "page_first_direct", ioBackend: "direct", ratio: 4 },
+      amdStorageFileOnly: true,
-        { id: "mooncake",  label: "Mooncake" },
-        { id: "hf3fs",     label: "HF3FS" },
-        { id: "nixl",      label: "NiXL" },
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -514,6 +514,10 @@ To enable HiCache, open the **HiCache** card in the [Playground above](#playgrou
+For AMD devices,
+- **L2 (GPU + CPU)** — leave Storage on `auto` (default). Cold KV pages spill to CPU pinned memory only. Use `direct` IO backend + `page_first_direct` or `layer-first` mem-layout.
+- **L3 (GPU + CPU + Storage)** — pick a Storage backend (`file`); the Playground emits the canonical `page_first_direct` mem-layout + `direct` IO backend + `wait_complete` prefetc
```

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +9/-3; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/_playground.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #30238 - [AMD] Support two batch overlap with MTP on DeepSeekV4

- Link: https://github.com/sgl-project/sglang/pull/30238
- Status/date: merged / 2026-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py`; associated commits `e2d021d4ab2f`
- Diff scope read: GitHub Pull Request files API returned 3 files, +167/-1, 190 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Support two batch overlap with MTP on DeepSeekV4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py`; technical summary: Covers "[AMD] Support two batch overlap with MTP on DeepSeekV4"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +3/-1 (4 lines); hunks: -2087,7 +2087,9 @@ def _can_run_tbo(self, forward_batch: ForwardBatch) -> bool:; symbols: _can_run_tbo, touching `_can_run_tbo`; `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py` added +149/-0 (149 lines); hunks: -0,0 +1,149; symbols: TestDeepseekV4ProFp4TboMTP, setUpClass, tearDownClass, test_gsm8k_tbo_mtp, touching `TestDeepseekV4ProFp4TboMTP, setUpClass, tearDownClass`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +3/-1 (4 lines); hunks: -2087,7 +2087,9 @@ def _can_run_tbo(self, forward_batch: ForwardBatch) -> bool:; symbols: _can_run_tbo
  - `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py` added +149/-0 (149 lines); hunks: -0,0 +1,149; symbols: TestDeepseekV4ProFp4TboMTP, setUpClass, tearDownClass, test_gsm8k_tbo_mtp
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +3/-1
  - tests: `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py` added +149/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25763 - [Feature] Support DeepSeek-V4 Wint4Abf16 and Win4Afp8.

- Link: https://github.com/sgl-project/sglang/pull/25763
- Status/date: merged / 2026-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `bff489284b50`
- Diff scope read: GitHub Pull Request files API returned 12 files, +561/-34, 790 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Support DeepSeek-V4 Wint4Abf16 and Win4Afp8."; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[Feature] Support DeepSeek-V4 Wint4Abf16 and Win4Afp8."; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +14/-7 (21 lines); hunks: -114,7 +114,10; -2672,7 +2675,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, auto_weight_loader, touching `load_weights, auto_weight_loader`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +14/-7 (21 lines); hunks: -114,7 +114,10; -2672,7 +2675,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, auto_weight_loader
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +14/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/gemm/per_tensor_quant_fp8.cuh`, `python/sglang/jit_kernel/per_tensor_quant_fp8.py`, `python/sglang/kernels/ops/moe/ep_moe_kernels.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31452 - [Docs] Tune DeepSeek-V4 HiCache for MI355X PD

- Link: https://github.com/sgl-project/sglang/pull/31452
- Status/date: merged / 2026-07-17
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; associated commits `27a52d2530a1`
- Diff scope read: GitHub Pull Request files API returned 3 files, +60/-10, 133 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Tune DeepSeek-V4 HiCache for MI355X PD"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[Docs] Tune DeepSeek-V4 HiCache for MI355X PD"; the main implementation surface is `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +27/-1 (28 lines); hunks: -161,7 +161,7 @@ sgl-eval run aime25 \\; -325,6 +325,32 @@ sgl-eval run aime25 \\; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +3/-3 (6 lines); hunks: -52,11 +52,11 @@ docker run --gpus all \; -66,7 +66,7 @@ docker run \.
- Code diff details:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +27/-1 (28 lines); hunks: -161,7 +161,7 @@ sgl-eval run aime25 \\; -325,6 +325,32 @@ sgl-eval run aime25 \\
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +3/-3 (6 lines); hunks: -52,11 +52,11 @@ docker run --gpus all \; -66,7 +66,7 @@ docker run \
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -161,7 +161,7 @@ sgl-eval run aime25 \\
-    mi355x: "lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260708",
+    mi355x: "lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260710",
@@ -325,6 +325,32 @@ sgl-eval run aime25 \\
+      roleOverrides: [
+        {
+          when: {
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -52,11 +52,11 @@ docker run --gpus all \
-- **MI355X** → `lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260623`
+- **MI355X** → `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260710`
-docker pull lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260623
+docker pull lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260710
@@ -66,7 +66,7 @@ docker run \
-    lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260623 \
```

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +27/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +3/-3
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/_playground.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #30272 - Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2

- Link: https://github.com/sgl-project/sglang/pull/30272
- Status/date: merged / 2026-07-18
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; associated commits `faf68940939a`
- Diff scope read: GitHub Pull Request files API returned 18 files, +506/-237, 1126 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +8/-3 (11 lines); hunks: -1688,9 +1688,14 @@ def match_num_queries(x, value):; symbols: match_num_queries, touching `match_num_queries`; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +3/-4 (7 lines); hunks: -1421,7 +1421,6 @@ sgl-eval run aime25 \\; -1430,9 +1429,9 @@ sgl-eval run aime25 \\; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +3/-4 (7 lines); hunks: -149,7 +149,7 @@ import { Playground } from "/src/snippets/_playground.jsx";; -293,9 +293,8 @@ TCP, which can lead to garbled KV transfer on large checkpoi....
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +8/-3 (11 lines); hunks: -1688,9 +1688,14 @@ def match_num_queries(x, value):; symbols: match_num_queries
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +3/-4 (7 lines); hunks: -1421,7 +1421,6 @@ sgl-eval run aime25 \\; -1430,9 +1429,9 @@ sgl-eval run aime25 \\
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +3/-4 (7 lines); hunks: -149,7 +149,7 @@ import { Playground } from "/src/snippets/_playground.jsx";; -293,9 +293,8 @@ TCP, which can lead to garbled KV transfer on large checkpoi...
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -1688,9 +1688,14 @@ def match_num_queries(x, value):
-            if forward_batch.forward_mode.is_extend_without_speculative() and (
-                q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD
-                or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
+            # sparse_prefill_fwd does not support SM120.
+            if (
+                forward_batch.forward_mode.is_extend_without_speculative()
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -1421,7 +1421,6 @@ sgl-eval run aime25 \\
-    // (V4-Pro doesn't fit on 8× 96 GB); TP-only, Marlin MoE runner.
@@ -1430,9 +1429,9 @@ sgl-eval run aime25 \\
-        "--tp 4",
-        "--moe-runner-backend marlin",
-        "--mem-fraction-static 0.70",
+        "--tp 2",
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -149,7 +149,7 @@ import { Playground } from "/src/snippets/_playground.jsx";
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +8/-3
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +3/-4; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +3/-4
- Risk and verification: The diff ships test coverage in `test/registered/unit/layers/quantization/test_mxfp4_sm120_cutlass.py`, `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py`, `test/registered/unit/test_model_overrides.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31705 - [DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph

- Link: https://github.com/sgl-project/sglang/pull/31705
- Status/date: merged / 2026-07-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; associated commits `688a6d23f144`
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; technical summary: Covers "[DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +2/-0 (2 lines); hunks: -94,6 +94,8 @@ def _get_logical_forward_mode(forward_batch: ForwardBatch) ->...; symbols: _get_logical_forward_mode, touching `_get_logical_forward_mode`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +2/-0 (2 lines); hunks: -94,6 +94,8 @@ def _get_logical_forward_mode(forward_batch: ForwardBatch) ->...; symbols: _get_logical_forward_mode
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -94,6 +94,8 @@ def _get_logical_forward_mode(forward_batch: ForwardBatch) -> ForwardMode:
+    if forward_batch.forward_mode == ForwardMode.EXTEND:
+        return forward_batch.forward_mode
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31363 - docs(cookbook): re-benchmark DeepSeek-V4 on sglang 0.5.15

- Link: https://github.com/sgl-project/sglang/pull/31363
- Status/date: merged / 2026-07-21
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; associated commits `4a55fdba0b7e`
- Diff scope read: GitHub Pull Request files API returned 8 files, +231/-71, 594 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): re-benchmark DeepSeek-V4 on sglang 0.5.15"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; technical summary: Covers "docs(cookbook): re-benchmark DeepSeek-V4 on sglang 0.5.15"; the main implementation surface is `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +209/-59 (268 lines); hunks: -1,113 +1,181; -118,6 +186,7 @@ export const benchmarks = [; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +3/-3 (6 lines); hunks: -4,7 +4,7; -80,7 +80,7 @@ export const config = {.
- Code diff details:
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +209/-59 (268 lines); hunks: -1,113 +1,181; -118,6 +186,7 @@ export const benchmarks = [
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +3/-3 (6 lines); hunks: -4,7 +4,7; -80,7 +80,7 @@ export const config = {
- Key code excerpts:

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
```

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` modified +209/-59; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +3/-3
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/_deployment.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #27657 - [DeepSeek V4] CP decode opt: slice repeat attention weights to local TP partition

- Link: https://github.com/sgl-project/sglang/pull/27657
- Status/date: merged / 2026-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`; associated commits `ebe3ab29e485`
- Diff scope read: GitHub Pull Request files API returned 6 files, +336/-49, 533 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek V4] CP decode opt: slice repeat attention weights to local TP partition"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`; technical summary: Covers "[DeepSeek V4] CP decode opt: slice repeat attention weights to local TP partition"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_dspark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +60/-25 (85 lines); hunks: -4,7 +4,7; -60,6 +60,7; symbols: __init__, _local_attn_sink, maybe_use_decode_attn_tp, MQALayer, touching `__init__, _local_attn_sink, maybe_use_decode_attn_tp`; `python/sglang/srt/models/deepseek_v4_dspark.py` modified +2/-12 (14 lines); hunks: -121,17 +121,6 @@ def kv_proj_only(self, x: torch.Tensor) -> torch.Tensor:; -536,7 +525,8 @@ def forward(; symbols: kv_proj_only, _local_attn_sink, _store_block_kv, forward, touching `kv_proj_only, _local_attn_sink, _store_block_kv`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +60/-25 (85 lines); hunks: -4,7 +4,7; -60,6 +60,7; symbols: __init__, _local_attn_sink, maybe_use_decode_attn_tp, MQALayer
  - `python/sglang/srt/models/deepseek_v4_dspark.py` modified +2/-12 (14 lines); hunks: -121,17 +121,6 @@ def kv_proj_only(self, x: torch.Tensor) -> torch.Tensor:; -536,7 +525,8 @@ def forward(; symbols: kv_proj_only, _local_attn_sink, _store_block_kv, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +60/-25; `python/sglang/srt/models/deepseek_v4_dspark.py` modified +2/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/cp/cp_decode_attn_tp.py`, `python/sglang/srt/layers/linear.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29569 - [DSV4] Support megamoe for CP

- Link: https://github.com/sgl-project/sglang/pull/29569
- Status/date: merged / 2026-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`; associated commits `71fe41b6b3c7`
- Diff scope read: GitHub Pull Request files API returned 5 files, +167/-7, 235 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Support megamoe for CP"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`; technical summary: Covers "[DSV4] Support megamoe for CP"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +6/-4 (10 lines); hunks: -1748,12 +1748,14 @@ def _run_moe_ffn_dp_sync(; symbols: _run_moe_ffn_dp_sync, touching `_run_moe_ffn_dp_sync`; `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +98/-0 (98 lines); hunks: -11,6 +11,98; -85,6 +177,12 @@ def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:; symbols: validate_deepseek_v4_mega_moe_token_budget, apply_deepseek_v4_defaults, validate_deepseek_v4_cp, touching `validate_deepseek_v4_mega_moe_token_budget, apply_deepseek_v4_defaults, validate_deepseek_v4_cp`; `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +56/-1 (57 lines); hunks: -37,8 +37,14; -92,6 +98,55 @@ def tearDownClass(cls):; symbols: TestDSV4FlashFP4B200Balanced_CP, TestDSV4FlashFP4B200Balanced_CP_DeepEP, tearDownClass, TestDSV4FlashFP4B200Balanced_CP_Megamoe, touching `TestDSV4FlashFP4B200Balanced_CP, TestDSV4FlashFP4B200Balanced_CP_DeepEP, tearDownClass`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +6/-4 (10 lines); hunks: -1748,12 +1748,14 @@ def _run_moe_ffn_dp_sync(; symbols: _run_moe_ffn_dp_sync
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +98/-0 (98 lines); hunks: -11,6 +11,98; -85,6 +177,12 @@ def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:; symbols: validate_deepseek_v4_mega_moe_token_budget, apply_deepseek_v4_defaults, validate_deepseek_v4_cp
  - `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +56/-1 (57 lines); hunks: -37,8 +37,14; -92,6 +98,55 @@ def tearDownClass(cls):; symbols: TestDSV4FlashFP4B200Balanced_CP, TestDSV4FlashFP4B200Balanced_CP_DeepEP, tearDownClass, TestDSV4FlashFP4B200Balanced_CP_Megamoe
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +6/-4; `python/sglang/srt/arg_groups/deepseek_v4_hook.py` modified +98/-0
  - tests: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` modified +56/-1
- Risk and verification: The diff ships test coverage in `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27059 - Add FP4 Indexer for DeepSeek V4 on SM120

- Link: https://github.com/sgl-project/sglang/pull/27059
- Status/date: merged / 2026-07-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; associated commits `1e69765bae5b`
- Diff scope read: GitHub Pull Request files API returned 5 files, +76/-10, 157 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add FP4 Indexer for DeepSeek V4 on SM120"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; technical summary: Covers "Add FP4 Indexer for DeepSeek V4 on SM120"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +5/-0 (5 lines); hunks: -650,6 +650,11 @@ def init_forward_metadata_indexer(; symbols: init_forward_metadata_indexer, touching `init_forward_metadata_indexer`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +5/-0 (5 lines); hunks: -650,6 +650,11 @@ def init_forward_metadata_indexer(; symbols: init_forward_metadata_indexer
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -650,6 +650,11 @@ def init_forward_metadata_indexer(
+            # The SM120 FP4 kernel schedules split_kv=128, while the generic
+            # JIT metadata planner encodes split_kv=256.
+            force_deep_gemm_metadata=(
+                self.enable_deepseek_v4_fp4_indexer and _is_sm120
+            ),
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +5/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/benchmark/attention/bench_dsv4_fp4_indexer.py`, `test/registered/unit/layers/test_dsv4_nonpaged_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31087 - [RL] DSV4: dispatch indexer topk_transform_512 through DSATopKBackend

- Link: https://github.com/sgl-project/sglang/pull/31087
- Status/date: merged / 2026-07-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; associated commits `f7986c8603f7`
- Diff scope read: GitHub Pull Request files API returned 2 files, +96/-9, 179 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[RL] DSV4: dispatch indexer topk_transform_512 through DSATopKBackend"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; technical summary: Covers "[RL] DSV4: dispatch indexer topk_transform_512 through DSATopKBackend"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +4/-0 (4 lines); hunks: -40,6 +40,7; -528,6 +529,9 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +4/-0 (4 lines); hunks: -40,6 +40,7; -528,6 +529,9 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -40,6 +40,7 @@
+from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
@@ -528,6 +529,9 @@ def __init__(
+        self.dsa_topk_backend: DSATopKBackend = DSATopKBackend(
+            model_runner.server_args.dsa_topk_backend
+        )
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/dsv4/indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31086 - [RL] DSV4: add env to quantize SWA KV cache from bf16-rounded values

- Link: https://github.com/sgl-project/sglang/pull/31086
- Status/date: merged / 2026-07-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `0a212c611909`
- Diff scope read: GitHub Pull Request files API returned 3 files, +17/-0, 38 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[RL] DSV4: add env to quantize SWA KV cache from bf16-rounded values"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[RL] DSV4: add env to quantize SWA KV cache from bf16-rounded values"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +9/-0 (9 lines); hunks: -722,6 +722,15 @@ def _compute_kv_to_cache(; symbols: _compute_kv_to_cache, touching `_compute_kv_to_cache`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +9/-0 (9 lines); hunks: -722,6 +722,15 @@ def _compute_kv_to_cache(; symbols: _compute_kv_to_cache
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +9/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30954 - [SM120] Allow fused MHC opt-in with standalone TileLang pre disabled

- Link: https://github.com/sgl-project/sglang/pull/30954
- Status/date: merged / 2026-07-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`, `test/registered/unit/models/test_deepseek_v4_fused_mhc_policy.py`; associated commits `2cbddb842d67`
- Diff scope read: GitHub Pull Request files API returned 2 files, +84/-3, 105 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[SM120] Allow fused MHC opt-in with standalone TileLang pre disabled"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `test/registered/unit/models/test_deepseek_v4_fused_mhc_policy.py`, `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[SM120] Allow fused MHC opt-in with standalone TileLang pre disabled"; the main implementation surface is `test/registered/unit/models/test_deepseek_v4_fused_mhc_policy.py`, `python/sglang/srt/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/models/test_deepseek_v4_fused_mhc_policy.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: TestDeepseekV4FusedMHCPolicy, _is_enabled, test_sm120_allows_fused_opt_in_with_standalone_pre_disabled, test_other_platform_still_requires_tilelang_pre, touching `TestDeepseekV4FusedMHCPolicy, _is_enabled, test_sm120_allows_fused_opt_in_with_standalone_pre_disabled`; `python/sglang/srt/models/deepseek_v4.py` modified +5/-3 (8 lines); hunks: -153,6 +153,7; -204,12 +205,13 @@ def _get_mhc_ops() -> MhcOps:; symbols: _get_mhc_ops, _is_fused_mhc_post_pre_enabled, touching `_get_mhc_ops, _is_fused_mhc_post_pre_enabled`.
- Code diff details:
  - `test/registered/unit/models/test_deepseek_v4_fused_mhc_policy.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: TestDeepseekV4FusedMHCPolicy, _is_enabled, test_sm120_allows_fused_opt_in_with_standalone_pre_disabled, test_other_platform_still_requires_tilelang_pre
  - `python/sglang/srt/models/deepseek_v4.py` modified +5/-3 (8 lines); hunks: -153,6 +153,7; -204,12 +205,13 @@ def _get_mhc_ops() -> MhcOps:; symbols: _get_mhc_ops, _is_fused_mhc_post_pre_enabled
- Key code excerpts:

```diff
diff -- test/registered/unit/models/test_deepseek_v4_fused_mhc_policy.py
@@ -0,0 +1,79 @@
+"""Unit tests for the DeepSeek-V4 fused-MHC enable policy."""
+import unittest
+from unittest.mock import patch
+import sglang.srt.models.deepseek_v4 as deepseek_v4
+from sglang.srt.environ import envs
+from sglang.test.ci.ci_register import register_cpu_ci
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -153,6 +153,7 @@
+from sglang.srt.utils.common import is_sm120_supported
@@ -204,12 +205,13 @@ def _get_mhc_ops() -> MhcOps:
-    # The fused path directly reuses TileLang mhc_post/mhc_pre kernels and their
-    # tensor layout assumptions, so keep it disabled when either dependency is off.
+    # SM120 disables the standalone TileLang pre path. mhc_fused_post_pre does
+    # not read that flag and dispatches independently for both small and large
```

- Reviewed files:
  - tests: `test/registered/unit/models/test_deepseek_v4_fused_mhc_policy.py` added +79/-0
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +5/-3
- Risk and verification: The diff ships test coverage in `test/registered/unit/models/test_deepseek_v4_fused_mhc_policy.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
