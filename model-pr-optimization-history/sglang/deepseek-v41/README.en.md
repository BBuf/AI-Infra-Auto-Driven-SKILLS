# SGLang DeepSeek V4.1 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` | [#38802](https://github.com/sgl-project/sglang/pull/38802) |
| `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` | [#38802](https://github.com/sgl-project/sglang/pull/38802), [#38839](https://github.com/sgl-project/sglang/pull/38839), [#38844](https://github.com/sgl-project/sglang/pull/38844), [#39929](https://github.com/sgl-project/sglang/pull/39929), [#41308](https://github.com/sgl-project/sglang/pull/41308) |
| `docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v4_flash.mdx` | no direct PR-number commit |
| `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx` | no direct PR-number commit |
| `docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` | no direct PR-number commit |
| `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` | no direct PR-number commit |
| `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` | [#38798](https://github.com/sgl-project/sglang/pull/38798), [#38802](https://github.com/sgl-project/sglang/pull/38802), [#38839](https://github.com/sgl-project/sglang/pull/38839), [#38844](https://github.com/sgl-project/sglang/pull/38844), [#38861](https://github.com/sgl-project/sglang/pull/38861), [#41308](https://github.com/sgl-project/sglang/pull/41308) |
| `examples/runtime/deepseek_v4/benchmark_deepseek_5090.py` | no direct PR-number commit |
| `python/sglang/kernels/aot/csrc/elementwise/deepseek_v4_topk.cu` | [#41020](https://github.com/sgl-project/sglang/pull/41020) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/block_amax.cuh` | [#39648](https://github.com/sgl-project/sglang/pull/39648) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c1.cuh` | [#39652](https://github.com/sgl-project/sglang/pull/39652), [#41019](https://github.com/sgl-project/sglang/pull/41019) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c128.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c128_online.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c128_online_v2.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c128_v2.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh` | [#39652](https://github.com/sgl-project/sglang/pull/39652), [#41660](https://github.com/sgl-project/sglang/pull/41660) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c4.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c4_v2.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh` | [#39652](https://github.com/sgl-project/sglang/pull/39652) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/candidate_block_table.cuh` | [#39648](https://github.com/sgl-project/sglang/pull/39648) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/common.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/flashmla_sched_meta.cuh` | [#39646](https://github.com/sgl-project/sglang/pull/39646) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope.cuh` | [#39656](https://github.com/sgl-project/sglang/pull/39656), [#41019](https://github.com/sgl-project/sglang/pull/41019) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope_hip.cuh` | [#41019](https://github.com/sgl-project/sglang/pull/41019) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/fp8_cvt.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh` | [#39652](https://github.com/sgl-project/sglang/pull/39652) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/hash_topk.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh` | [#39652](https://github.com/sgl-project/sglang/pull/39652), [#41019](https://github.com/sgl-project/sglang/pull/41019), [#41657](https://github.com/sgl-project/sglang/pull/41657) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/mhc_boundary_gfx95.cuh` | [#41021](https://github.com/sgl-project/sglang/pull/41021) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/mhc_post_combine_norm_prefill.cuh` | [#39704](https://github.com/sgl-project/sglang/pull/39704) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/mxfp8_gemv_gfx95.cuh` | [#41018](https://github.com/sgl-project/sglang/pull/41018) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/online_c128_mtp.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/paged_mqa_metadata.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/rope.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/store.cuh` | [#39652](https://github.com/sgl-project/sglang/pull/39652), [#41019](https://github.com/sgl-project/sglang/pull/41019) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh` | [#39648](https://github.com/sgl-project/sglang/pull/39648) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v1.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` | [#39648](https://github.com/sgl-project/sglang/pull/39648) |
| `python/sglang/kernels/jit/csrc/deepseek_v4/wo_a_fused.cuh` | [#39957](https://github.com/sgl-project/sglang/pull/39957) |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/compress.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/compress_v2.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp4_utils.cuh` | [#39652](https://github.com/sgl-project/sglang/pull/39652), [#41019](https://github.com/sgl-project/sglang/pull/41019) |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp8_utils.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh` | [#39652](https://github.com/sgl-project/sglang/pull/39652), [#41019](https://github.com/sgl-project/sglang/pull/41019) |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kvcacheio.cuh` | no direct PR-number commit |
| `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh` | [#39648](https://github.com/sgl-project/sglang/pull/39648) |
| `python/sglang/kernels/ops/attention/deepseek_v4_rope.py` | no direct PR-number commit |
| `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798), [#41308](https://github.com/sgl-project/sglang/pull/41308) |
| `python/sglang/srt/arg_groups/model_overrides/deepseek_v4.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798) |
| `python/sglang/srt/configs/deepseek_v4.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798) |
| `python/sglang/srt/configs/deepseek_v41.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798) |
| `python/sglang/srt/entrypoints/openai/encoding_dsv41.py` | [#39665](https://github.com/sgl-project/sglang/pull/39665), [#39929](https://github.com/sgl-project/sglang/pull/39929) |
| `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798), [#38947](https://github.com/sgl-project/sglang/pull/38947), [#39704](https://github.com/sgl-project/sglang/pull/39704), [#40217](https://github.com/sgl-project/sglang/pull/40217), [#40352](https://github.com/sgl-project/sglang/pull/40352), [#40637](https://github.com/sgl-project/sglang/pull/40637), [#41125](https://github.com/sgl-project/sglang/pull/41125), [#41291](https://github.com/sgl-project/sglang/pull/41291), [#41308](https://github.com/sgl-project/sglang/pull/41308), [#41660](https://github.com/sgl-project/sglang/pull/41660), [#42128](https://github.com/sgl-project/sglang/pull/42128), [#42273](https://github.com/sgl-project/sglang/pull/42273) |
| `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` | [#38947](https://github.com/sgl-project/sglang/pull/38947), [#41308](https://github.com/sgl-project/sglang/pull/41308), [#42014](https://github.com/sgl-project/sglang/pull/42014), [#42017](https://github.com/sgl-project/sglang/pull/42017) |
| `python/sglang/srt/layers/attention/deepseek_v4_trtllm_backend.py` | no direct PR-number commit |
| `python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798), [#41308](https://github.com/sgl-project/sglang/pull/41308) |
| `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798) |
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798), [#38947](https://github.com/sgl-project/sglang/pull/38947), [#41308](https://github.com/sgl-project/sglang/pull/41308), [#41337](https://github.com/sgl-project/sglang/pull/41337), [#41345](https://github.com/sgl-project/sglang/pull/41345), [#41658](https://github.com/sgl-project/sglang/pull/41658), [#42128](https://github.com/sgl-project/sglang/pull/42128) |
| `python/sglang/srt/mem_cache/dsv41_request_window.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798) |
| `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` | [#41308](https://github.com/sgl-project/sglang/pull/41308) |
| `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py` | [#41308](https://github.com/sgl-project/sglang/pull/41308), [#41970](https://github.com/sgl-project/sglang/pull/41970), [#42055](https://github.com/sgl-project/sglang/pull/42055) |
| `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_hip.py` | [#41308](https://github.com/sgl-project/sglang/pull/41308), [#42055](https://github.com/sgl-project/sglang/pull/42055) |
| `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_wo_a_fp8.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_v4.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798), [#39704](https://github.com/sgl-project/sglang/pull/39704), [#39957](https://github.com/sgl-project/sglang/pull/39957), [#41251](https://github.com/sgl-project/sglang/pull/41251), [#41308](https://github.com/sgl-project/sglang/pull/41308), [#41657](https://github.com/sgl-project/sglang/pull/41657), [#42055](https://github.com/sgl-project/sglang/pull/42055) |
| `python/sglang/srt/models/deepseek_v41_vit.py` | [#39668](https://github.com/sgl-project/sglang/pull/39668) |
| `python/sglang/srt/models/deepseek_v4_dspark.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798) |
| `python/sglang/srt/models/deepseek_v4_nextn.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798) |
| `python/sglang/srt/multimodal/deepseek_v41_image_processing.py` | [#39668](https://github.com/sgl-project/sglang/pull/39668) |
| `python/sglang/srt/multimodal/dsv41/__init__.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798) |
| `python/sglang/srt/multimodal/dsv41/vl_routing.py` | [#38798](https://github.com/sgl-project/sglang/pull/38798) |
| `python/sglang/srt/multimodal/processors/deepseek_v41.py` | [#39668](https://github.com/sgl-project/sglang/pull/39668) |
| `python/sglang/test/kernels/deepseek_v4/__init__.py` | no direct PR-number commit |
| `python/sglang/test/kernels/deepseek_v4/common.py` | no direct PR-number commit |
| `python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py` | [#41019](https://github.com/sgl-project/sglang/pull/41019) |
| `test/registered/amd/accuracy/mi35x/test_deepseek_v41_flash_eval_mi35x.py` | [#41476](https://github.com/sgl-project/sglang/pull/41476) |
| `test/registered/amd/accuracy/mi45x/test_deepseek_v4_flash_eval_mi45x.py` | no direct PR-number commit |
| ... | 44 more files omitted from table; all were used for git tracing. |

## PR Coverage Summary

- Git-traced PRs: 39
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 39
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-09-10 | [#38802](https://github.com/sgl-project/sglang/pull/38802) | merged | Add DeepSeek-V4.1 Flash cookbook | `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-09-10 | [#38839](https://github.com/sgl-project/sglang/pull/38839) | merged | Fix the DeepSeek-V4.1 reasoning example and make every NVIDIA cell start | `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` |
| 2026-09-10 | [#38844](https://github.com/sgl-project/sglang/pull/38844) | merged | [Cookbook] DeepSeek-V4.1: add the HiCache L2 knob to the Playground | `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` |
| 2026-09-10 | [#38861](https://github.com/sgl-project/sglang/pull/38861) | merged | Make the remaining DeepSeek-V4.1 NVIDIA cells start | `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` |
| 2026-09-10 | [#38947](https://github.com/sgl-project/sglang/pull/38947) | merged | [Refactor] Clarify DeepSeek V4 metadata names for V4.1 | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-09-16 | [#39646](https://github.com/sgl-project/sglang/pull/39646) | merged | dsv4.1: standalone kernels and Python wrappers | `python/sglang/kernels/jit/csrc/deepseek_v4/flashmla_sched_meta.cuh` |
| 2026-09-16 | [#39648](https://github.com/sgl-project/sglang/pull/39648) | merged | dsv4.1: Top-k kernels and candidate selection | `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` |
| 2026-09-16 | [#39652](https://github.com/sgl-project/sglang/pull/39652) | merged | dsv4.1: compression, KV I/O, and metadata kernels | `python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/c1.cuh`, `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh` |
| 2026-09-16 | [#39656](https://github.com/sgl-project/sglang/pull/39656) | merged | dsv4.1: RoPE and FP4 packing kernels | `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope.cuh` |
| 2026-09-17 | [#39668](https://github.com/sgl-project/sglang/pull/39668) | merged | dsv4.1: vision tower and image preprocessing | `python/sglang/srt/multimodal/deepseek_v41_image_processing.py`, `python/sglang/srt/models/deepseek_v41_vit.py`, `python/sglang/srt/multimodal/processors/deepseek_v41.py` |
| 2026-09-17 | [#39665](https://github.com/sgl-project/sglang/pull/39665) | merged | dsv4.1: chat encoding and tool parsing | `python/sglang/srt/entrypoints/openai/encoding_dsv41.py` |
| 2026-09-18 | [#38798](https://github.com/sgl-project/sglang/pull/38798) | merged | dsv4.1: remaining model and runtime integration | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py` |
| 2026-09-19 | [#39957](https://github.com/sgl-project/sglang/pull/39957) | merged | [DSV4.1] Big fused wo_a quant | `python/sglang/srt/models/deepseek_v4.py`, `test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py`, `python/sglang/kernels/jit/csrc/deepseek_v4/wo_a_fused.cuh` |
| 2026-09-19 | [#39704](https://github.com/sgl-project/sglang/pull/39704) | merged | [DSV4.1] Reduce mHC, metadata and small-batch router overhead | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/kernels/jit/csrc/deepseek_v4/mhc_post_combine_norm_prefill.cuh` |
| 2026-09-20 | [#40217](https://github.com/sgl-project/sglang/pull/40217) | merged | [DeepSeek-V4.1] Bound dense prefill indexer memory | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/unit/layers/test_dsv41_candidate_blocks.py` |
| 2026-09-22 | [#40352](https://github.com/sgl-project/sglang/pull/40352) | merged | [DSv4.1] Score prefill consumer index layers on candidate blocks with DeepGEMM | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py` |
| 2026-09-22 | [#40637](https://github.com/sgl-project/sglang/pull/40637) | merged | [Fix] Handle chunked paged MQA metadata in DSV4.1 eager forwards | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| 2026-09-24 | [#39929](https://github.com/sgl-project/sglang/pull/39929) | merged | [Bugfix] Align DeepSeek-V4.1 reasoning effort budgets | `python/sglang/srt/entrypoints/openai/encoding_dsv41.py`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` |
| 2026-09-25 | [#41018](https://github.com/sgl-project/sglang/pull/41018) | merged | dsv4.1-amd: gfx950 MXFP8 matmul kernels and fp8-grid producers | `python/sglang/kernels/jit/csrc/deepseek_v4/mxfp8_gemv_gfx95.cuh` |
| 2026-09-26 | [#41291](https://github.com/sgl-project/sglang/pull/41291) | merged | [DSv4.1] Move the ratio-1/2 index top-k ops into kernels/ops/attention/dsv4 | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/unit/layers/test_dsv41_candidate_blocks.py`, `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py` |
| 2026-09-26 | [#41125](https://github.com/sgl-project/sglang/pull/41125) | merged | [DSv4.1] Move the low-ratio index top-k into dsv4/low_ratio_indexer | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/unit/layers/test_dsv41_candidate_blocks.py`, `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py` |
| 2026-09-27 | [#41019](https://github.com/sgl-project/sglang/pull/41019) | merged | dsv4.1-amd: KV cache layouts, FP4 indexer, compressor and router kernels | `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope_hip.cuh`, `python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py`, `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh` |
| 2026-09-27 | [#41345](https://github.com/sgl-project/sglang/pull/41345) | merged | [DSV4.1][HiCache] fix: wait for the layer transfer before reading low-ratio index-K | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-09-28 | [#41020](https://github.com/sgl-project/sglang/pull/41020) | merged | dsv4.1-amd: gfx950 sparse decode attention and sorted top-k | `test/registered/kernels/ops/attention/dsv4/test_dsv41_decode_glue_hip.py`, `python/sglang/kernels/aot/csrc/elementwise/deepseek_v4_topk.cu` |
| 2026-09-28 | [#41021](https://github.com/sgl-project/sglang/pull/41021) | merged | dsv4.1-amd: fused mHC boundary and all-reduce + mHC post kernels | `python/sglang/kernels/jit/csrc/deepseek_v4/mhc_boundary_gfx95.cuh` |
| 2026-09-30 | [#41308](https://github.com/sgl-project/sglang/pull/41308) | merged | dsv4.1-amd: serve DeepSeek-V4.1 on gfx950 | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/unit/models/test_deepseek_v4_amd_tp4.py`, `python/sglang/srt/models/deepseek_v4.py` |
| 2026-10-01 | [#41337](https://github.com/sgl-project/sglang/pull/41337) | merged | [DSV4/DSA] Name the FlashMLA KV format and drop the V4.1 support probe | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-10-01 | [#41970](https://github.com/sgl-project/sglang/pull/41970) | merged | [AMD][V4.1][*/N] Switch the fp8 dense GEMMs on gfx950 to aiter's MXFP8 GEMM | `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py` |
| 2026-10-01 | [#42014](https://github.com/sgl-project/sglang/pull/42014) | merged | [AMD][V4.1][*/N] Build DSpark draft metadata inside the CUDA graph on ROCm | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` |
| 2026-10-01 | [#42017](https://github.com/sgl-project/sglang/pull/42017) | merged | [AMD][V4.1][*/N] OPUS sparse prefill on gfx950 through layout conversion | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/attention/unittests/dsv4/test_dsv41_opus_sparse_prefill_hip.py` |
| 2026-10-02 | [#42011](https://github.com/sgl-project/sglang/pull/42011) | merged | [AMD][V4.1][*/N] Fix shared-expert fusion accuracy and speed up MoE routing on ROCm | `test/registered/unit/models/test_deepseek_v4_mxfp4_shared_expert_requant.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/moe/topk.py` |
| 2026-10-02 | [#42128](https://github.com/sgl-project/sglang/pull/42128) | merged | [Fix][DSV4.1] SWA page size with bounded replay | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-10-02 | [#42055](https://github.com/sgl-project/sglang/pull/42055) | merged | [AMD][V4.1][*/N] Fuse MXFP8 activation quant into producer kernels on gfx950 | `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py` |
| 2026-10-03 | [#41251](https://github.com/sgl-project/sglang/pull/41251) | merged | [Perf] Optimize DeepSeek V4.1 Flash Hopper paths and Blackwell prefill selection | `python/sglang/srt/models/deepseek_v4.py` |
| 2026-10-03 | [#41657](https://github.com/sgl-project/sglang/pull/41657) | merged | [DSv4.1] Fold q_rope_store into fused_q_norm_rope | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh` |
| 2026-10-03 | [#41658](https://github.com/sgl-project/sglang/pull/41658) | merged | [DSv4.1] Faster fp4 index-K gather and combine_topk_swa_indices | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-10-03 | [#41660](https://github.com/sgl-project/sglang/pull/41660) | merged | [DSv4.1] Fused c1/c2 compress for eager extend, faster c2 decode | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh` |
| 2026-10-04 | [#41476](https://github.com/sgl-project/sglang/pull/41476) | merged | [AMD] Add DeepSeek-V4.1-Flash MI35x nightly accuracy test | `test/registered/amd/accuracy/mi35x/test_deepseek_v41_flash_eval_mi35x.py` |
| 2026-10-05 | [#42273](https://github.com/sgl-project/sglang/pull/42273) | merged | [Dsv4.1] Bounded replay with sparse mla path | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |

## Per-PR Diff Audit Cards

### PR #38802 - Add DeepSeek-V4.1 Flash cookbook

- Link: https://github.com/sgl-project/sglang/pull/38802
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`; associated commits `69777c4d36c0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +584/-2, 616 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` added +375/-0 (375 lines); hunks: -0,0 +1,375; `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` added +190/-0 (190 lines); hunks: -0,0 +1,190; `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-1 (1 lines); hunks: -1,7 +1,6.
- Code diff details:
  - `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` added +375/-0 (375 lines); hunks: -0,0 +1,375
  - `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` added +190/-0 (190 lines); hunks: -0,0 +1,190
  - `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-1 (1 lines); hunks: -1,7 +1,6
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx
@@ -0,0 +1,375 @@
+// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
+//
+// Cells marked `verified` are transcribed from recorded runs on that hardware with
+// real weights. DP-Attention, DeepEP and MegaMoE are absent by design: they have
+// never been enabled on this model. EP is set equal to TP on every shape here.
+export const config = {
diff -- docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx
@@ -0,0 +1,190 @@
+---
+title: DeepSeek-V4.1
+description: "Deploy DeepSeek-V4.1 Flash with SGLang — launch recipes, feature compatibility, and tuning notes for GB300, H200, B200, B300 and MI350X."
+tag: NEW
+---
+## Deployment
diff -- docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -1,7 +1,6 @@
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` added +375/-0; `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` added +190/-0; `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-1
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`, `docs/cookbook/autoregressive/intro.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #38839 - Fix the DeepSeek-V4.1 reasoning example and make every NVIDIA cell start

- Link: https://github.com/sgl-project/sglang/pull/38839
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`; associated commits `5caafd2118b7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +16/-5, 80 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` modified +11/-4 (15 lines); hunks: -235,8 +235,8 @@ export const config = {; -271,10 +271,14 @@ export const config = {; `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` modified +5/-1 (6 lines); hunks: -114,6 +114,8 @@ Overriding them is the most common cause of a disappointing...; -123,6 +125,7 @@ client = OpenAI(base_url="http://localhost:30000/v1", api_ke....
- Code diff details:
  - `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` modified +11/-4 (15 lines); hunks: -235,8 +235,8 @@ export const config = {; -271,10 +271,14 @@ export const config = {
  - `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` modified +5/-1 (6 lines); hunks: -114,6 +114,8 @@ Overriding them is the most common cause of a disappointing...; -123,6 +125,7 @@ client = OpenAI(base_url="http://localhost:30000/v1", api_ke...
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx
@@ -235,8 +235,8 @@ export const config = {
-    // ---------- B200 / B300: verification round open. Mirrors the GB300 recipe
-    // because the kernels dispatch by architecture family. ----------
+    // ---------- B200: verification round open. Mirrors the GB300 recipe because
+    // the kernels dispatch by architecture family. ----------
@@ -271,10 +271,14 @@ export const config = {
+    // ---------- B300: 4x B300, TP4 + EP4. Same recipe as GB300 — the kernels
diff -- docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx
@@ -114,6 +114,8 @@ Overriding them is the most common cause of a disappointing measurement: it leav
+The parser can only split a thinking block the model actually produced, and **thinking is off by default** (`SGLANG_DEFAULT_THINKING=false`). Sending `reasoning_effort` turns it o
@@ -123,6 +125,7 @@ client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
+    reasoning_effort="high",
@@ -131,7 +134,7 @@ print("Answer:", msg.content)
-Reasoning effort is part of the request contract for this model: send `reasoning_effort` on the request, either as a tier or as an integer budget. Tiers with no V4.1 counterpart (
+On the request, `reasoning_effort` accepts the tiers `low`, `high`, `xhigh` and `max`, or a float in `[0.0, 0.99]` that maps onto the model's 1–100 budget. An integer budget is re
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` modified +11/-4; `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` modified +5/-1
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #38844 - [Cookbook] DeepSeek-V4.1: add the HiCache L2 knob to the Playground

- Link: https://github.com/sgl-project/sglang/pull/38844
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`; associated commits `a37ded1693f6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +25/-0, 36 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` modified +13/-0 (13 lines); hunks: -114,6 +114,19 @@ export const config = {; `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` modified +12/-0 (12 lines); hunks: -188,3 +188,15 @@ Prefill/decode disaggregation is validated token-identical....
- Code diff details:
  - `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` modified +13/-0 (13 lines); hunks: -114,6 +114,19 @@ export const config = {
  - `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` modified +12/-0 (12 lines); hunks: -188,3 +188,15 @@ Prefill/decode disaggregation is validated token-identical...
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx
@@ -114,6 +114,19 @@ export const config = {
+    // GPU → CPU KV offload (L2 only; no storage tier). Hidden on MI350X: both
+    // ROCm cells run `--disable-radix-cache`, which the server rejects alongside
+    // `--enable-hierarchical-cache`.
+    hicache: {
+      excludesHw: ["mi350x"],
+      writePolicies: [
diff -- docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx
@@ -188,3 +188,15 @@ Prefill/decode disaggregation is validated token-identical against a single serv
+### 3.5 HiCache (Hierarchical KV Caching)
+HiCache extends RadixAttention with a hierarchy of KV cache tiers, significantly expanding effective context capacity for long-context and multi-turn scenarios.
+To enable HiCache, open the **HiCache** card in the [Playground above](#playground) and flip **Enable**: the Playground emits `--enable-hierarchical-cache` on top of the recipe's
+The Write policy knob controls the GPU → CPU write and defaults to `write_through` (the upstream default): every page is mirrored to the CPU tier as it is written. `write_through_
+The card is not offered on MI350X: the ROCm recipes run `--disable-radix-cache`, and the server rejects that alongside `--enable-hierarchical-cache`.
+Only the L2 tier is exposed here. For the storage (L3) tier and the canonical flag set, see the [HiCache best-practices recipe](../../../docs/advanced_features/hicache_best_practi
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` modified +13/-0; `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` modified +12/-0
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #38861 - Make the remaining DeepSeek-V4.1 NVIDIA cells start

- Link: https://github.com/sgl-project/sglang/pull/38861
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`; associated commits `4b7331fb7770`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +31/-14, 130 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` modified +31/-14 (45 lines); hunks: -1,8 +1,16; -169,6 +177,8 @@ export const config = {.
- Code diff details:
  - `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` modified +31/-14 (45 lines); hunks: -1,8 +1,16; -169,6 +177,8 @@ export const config = {
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx
@@ -1,8 +1,16 @@
-// real weights. DP-Attention, DeepEP and MegaMoE are absent by design: they have
-// never been enabled on this model. EP is set equal to TP on every shape here.
+// real weights.
+//
+// Every DSpark cell caps --cuda-graph-max-bs-decode: the derived batch list does
+// not fit while capturing the DSpark decode graphs, on any NVIDIA platform. The
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` modified +31/-14
- Risk and verification: This is mostly docs/examples in `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #38947 - [Refactor] Clarify DeepSeek V4 metadata names for V4.1

- Link: https://github.com/sgl-project/sglang/pull/38947
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`; associated commits `dc5f59c3a2c4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 16 files, +123/-220, 924 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +18/-49 (67 lines); hunks: -162,8 +162,7 @@ class DSV4AttnMetadata:; -181,7 +180,7 @@ class DSV4AttnMetadata:; symbols: DSV4AttnMetadata, positions, get_flashmla_metadata, copy_, touching `DSV4AttnMetadata, positions, get_flashmla_metadata`; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +14/-27 (41 lines); hunks: -179,8 +179,7 @@ class DSV4AttnMetadata:; -205,7 +204,7 @@ class DSV4AttnMetadata:; symbols: DSV4AttnMetadata, positions, get_flashmla_metadata, copy_, touching `DSV4AttnMetadata, positions, get_flashmla_metadata`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +17/-27 (44 lines); hunks: -42,9 +42,8 @@ def get_compress_state_ring_size(; -56,9 +55,8 @@ def get_compress_state_ring_size(; symbols: get_compress_state_ring_size, get_compress_state_write_pad, __init__, touching `get_compress_state_ring_size, get_compress_state_write_pad, __init__`; `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +5/-13 (18 lines); hunks: -365,7 +365,7 @@ def _make_core_metadata(self, base: int):; -377,10 +377,7 @@ def test_bcg_is_explicit_and_dsv4_backend_opt_in_only(self):; symbols: _make_core_metadata, test_bcg_is_explicit_and_dsv4_backend_opt_in_only, test_refresh_replay_metadata_preserves_captured_tensor_storage, test_sparse_prefill_c128_uses_live_extent, touching `_make_core_metadata, test_bcg_is_explicit_and_dsv4_backend_opt_in_only, test_refresh_replay_metadata_preserves_captured_tensor_storage`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +18/-49 (67 lines); hunks: -162,8 +162,7 @@ class DSV4AttnMetadata:; -181,7 +180,7 @@ class DSV4AttnMetadata:; symbols: DSV4AttnMetadata, positions, get_flashmla_metadata, copy_
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +14/-27 (41 lines); hunks: -179,8 +179,7 @@ class DSV4AttnMetadata:; -205,7 +204,7 @@ class DSV4AttnMetadata:; symbols: DSV4AttnMetadata, positions, get_flashmla_metadata, copy_
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +17/-27 (44 lines); hunks: -42,9 +42,8 @@ def get_compress_state_ring_size(; -56,9 +55,8 @@ def get_compress_state_ring_size(; symbols: get_compress_state_ring_size, get_compress_state_write_pad, __init__
  - `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +5/-13 (18 lines); hunks: -365,7 +365,7 @@ def _make_core_metadata(self, base: int):; -377,10 +377,7 @@ def test_bcg_is_explicit_and_dsv4_backend_opt_in_only(self):; symbols: _make_core_metadata, test_bcg_is_explicit_and_dsv4_backend_opt_in_only, test_refresh_replay_metadata_preserves_captured_tensor_storage, test_sparse_prefill_c128_uses_live_extent
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -162,8 +162,7 @@ class DSV4AttnMetadata:
-    # SWA KV-store write target (out_cache_loc translated to SWA space), computed
-    # once per iteration in make_core_attn_metadata and read by the store path.
+    # Shared by all layer stores; locations are in SWA space.
@@ -181,7 +180,7 @@ class DSV4AttnMetadata:
-    c1_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
+    c0_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -179,8 +179,7 @@ class DSV4AttnMetadata:
-    # SWA KV-store write target (out_cache_loc translated to SWA space), computed
-    # once per iteration in make_core_attn_metadata and read by the store path.
+    # Shared by all layer stores; locations are in SWA space.
@@ -205,7 +204,7 @@ class DSV4AttnMetadata:
-    c1_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
+    c0_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -42,9 +42,8 @@ def get_compress_state_ring_size(
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +18/-49; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +14/-27; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +17/-27
  - tests: `test/registered/attention/unittests/dsv4/test_deepseek_v4.py` modified +5/-13
- Risk and verification: The diff ships test coverage in `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`, `test/registered/unit/disaggregation/test_disaggregation_wire.py`, `test/registered/unit/layers/test_dsv4_nonpaged_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39646 - dsv4.1: standalone kernels and Python wrappers

- Link: https://github.com/sgl-project/sglang/pull/39646
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/kernels/jit/csrc/deepseek_v4/flashmla_sched_meta.cuh`; associated commits `91f691c49077`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 25 files, +2741/-19, 2848 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/kernels/jit/csrc/deepseek_v4/flashmla_sched_meta.cuh` added +276/-0 (276 lines); hunks: -0,0 +1,276.
- Code diff details:
  - `python/sglang/kernels/jit/csrc/deepseek_v4/flashmla_sched_meta.cuh` added +276/-0 (276 lines); hunks: -0,0 +1,276
- Key code excerpts:

```diff
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/flashmla_sched_meta.cuh
@@ -0,0 +1,276 @@
+#include <sgl_kernel/tensor.h>
+#include <sgl_kernel/utils.h>
+#include <sgl_kernel/utils.cuh>
+#include <tvm/ffi/container/tensor.h>
+#include <cstdint>
+// Tile-scheduler metadata for FlashMLA's split-KV decode.
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/kernels/jit/csrc/deepseek_v4/flashmla_sched_meta.cuh` added +276/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernel/attention/test_dsv4_q_rope_store.py`, `test/registered/kernel/attention/test_flashmla_sched_meta.py`, `test/registered/kernel/layernorm/test_mxfp8_epilogue.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39648 - dsv4.1: Top-k kernels and candidate selection

- Link: https://github.com/sgl-project/sglang/pull/39648
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/kernels/jit/csrc/deepseek_v4/block_amax.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/candidate_block_table.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh`, `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh`; associated commits `faaff1eca887`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 12 files, +1754/-526, 3053 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh` modified +338/-314 (652 lines); hunks: -7,11 +7,11; -23,52 +23,48; `python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh` added +440/-0 (440 lines); hunks: -0,0 +1,440; `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` modified +239/-162 (401 lines); hunks: -20,9 +20,9; -38,27 +38,14 @@ enum class TopKMode {; symbols: TopKMode, touching `TopKMode`; `python/sglang/kernels/jit/csrc/deepseek_v4/candidate_block_table.cuh` added +218/-0 (218 lines); hunks: -0,0 +1,218.
- Code diff details:
  - `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh` modified +338/-314 (652 lines); hunks: -7,11 +7,11; -23,52 +23,48
  - `python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh` added +440/-0 (440 lines); hunks: -0,0 +1,440
  - `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` modified +239/-162 (401 lines); hunks: -20,9 +20,9; -38,27 +38,14 @@ enum class TopKMode {; symbols: TopKMode
  - `python/sglang/kernels/jit/csrc/deepseek_v4/candidate_block_table.cuh` added +218/-0 (218 lines); hunks: -0,0 +1,218
  - `python/sglang/kernels/jit/csrc/deepseek_v4/block_amax.cuh` added +155/-0 (155 lines); hunks: -0,0 +1,155
- Key code excerpts:

```diff
diff -- python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh
@@ -7,11 +7,11 @@
-///  - the output is the page-table transform of the selected raw indices
-///    (`TopKProblem::emit` then `transform_output`).
+///  - the dispatcher optionally transforms selected raw indices through a page
+///    table after the device implementation writes them.
-///  - the cluster size is fixed at 8 (dynamic persistent clusters are hard).
+///  - the dispatcher selects cluster size 8 or 16 from the probed occupancy.
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh
@@ -0,0 +1,440 @@
+/**
+ * \brief DeepSeek-V4.1's bf16 top-k kernel for short rows (<= 16384 scores)
+ * Adapted from https://github.com/deepseek-ai/DeepSelect
+ * Plain SIMT (no tensor cores or clusters), tuned for 16384-wide rows with k = 512.
+ */
+#pragma once
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh
@@ -20,9 +20,9 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh` modified +338/-314; `python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh` added +440/-0; `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` modified +239/-162; `python/sglang/kernels/jit/csrc/deepseek_v4/candidate_block_table.cuh` added +218/-0; `python/sglang/kernels/jit/csrc/deepseek_v4/block_amax.cuh` added +155/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/dsa_metadata_kit.py`, `test/registered/kernels/benchmark/attention/bench_topk.py`, `test/registered/kernels/ops/attention/test_topk_v2.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39652 - dsv4.1: compression, KV I/O, and metadata kernels

- Link: https://github.com/sgl-project/sglang/pull/39652
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/kernels/jit/csrc/deepseek_v4/c1.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh` and 8 files; associated commits `13d593b6cf88`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 19 files, +2102/-279, 2789 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh` added +398/-0 (398 lines); hunks: -0,0 +1,398; `python/sglang/kernels/jit/csrc/deepseek_v4/c1.cuh` added +312/-0 (312 lines); hunks: -0,0 +1,312; `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh` added +212/-0 (212 lines); hunks: -0,0 +1,212; symbols: KVLayout, touching `KVLayout`; `python/sglang/kernels/jit/csrc/deepseek_v4/store.cuh` modified +137/-24 (161 lines); hunks: -8,13 +8,15; -29,12 +31,21 @@ struct FusedStoreCacheParam {.
- Code diff details:
  - `python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh` added +398/-0 (398 lines); hunks: -0,0 +1,398
  - `python/sglang/kernels/jit/csrc/deepseek_v4/c1.cuh` added +312/-0 (312 lines); hunks: -0,0 +1,312
  - `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh` added +212/-0 (212 lines); hunks: -0,0 +1,212; symbols: KVLayout
  - `python/sglang/kernels/jit/csrc/deepseek_v4/store.cuh` modified +137/-24 (161 lines); hunks: -8,13 +8,15; -29,12 +31,21 @@ struct FusedStoreCacheParam {
  - `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh` modified +50/-11 (61 lines); hunks: -9,6 +9,7; -386,14 +387,16 @@ constexpr int64_t kFp8TwoPoolRowBytes = 512;
- Key code excerpts:

```diff
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh
@@ -0,0 +1,398 @@
+#include <sgl_kernel/tensor.h>
+#include <sgl_kernel/utils.h>
+#include <sgl_kernel/math.cuh>
+#include <sgl_kernel/type.cuh>
+#include <sgl_kernel/utils.cuh>
+#include <sgl_kernel/vec.cuh>
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/c1.cuh
@@ -0,0 +1,312 @@
+#include <sgl_kernel/tensor.h>
+#include <sgl_kernel/utils.h>
+#include <sgl_kernel/math.cuh>
+#include <sgl_kernel/type.cuh>
+#include <sgl_kernel/utils.cuh>
+#include <sgl_kernel/vec.cuh>
diff -- python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh
@@ -0,0 +1,212 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh` added +398/-0; `python/sglang/kernels/jit/csrc/deepseek_v4/c1.cuh` added +312/-0; `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh` added +212/-0; `python/sglang/kernels/jit/csrc/deepseek_v4/store.cuh` modified +137/-24; `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh` modified +50/-11; `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp4_utils.cuh` added +60/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernel/attention/test_deepseek_v4_compress_plan_bounds.py`, `test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39656 - dsv4.1: RoPE and FP4 packing kernels

- Link: https://github.com/sgl-project/sglang/pull/39656
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope.cuh`; associated commits `c2443458e179`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +864/-5, 930 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope.cuh` added +441/-0 (441 lines); hunks: -0,0 +1,441.
- Code diff details:
  - `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope.cuh` added +441/-0 (441 lines); hunks: -0,0 +1,441
- Key code excerpts:

```diff
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope.cuh
@@ -0,0 +1,441 @@
+#include <sgl_kernel/tensor.h>
+#include <sgl_kernel/utils.h>
+#include <sgl_kernel/math.cuh>
+#include <sgl_kernel/type.cuh>
+#include <sgl_kernel/utils.cuh>
+#include <sgl_kernel/vec.cuh>
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope.cuh` added +441/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope.cuh`, `python/sglang/kernels/ops/attention/dsv4/fp4_indexer.py`, `python/sglang/kernels/ops/attention/dsv4/fp4_indexer_rope.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39668 - dsv4.1: vision tower and image preprocessing

- Link: https://github.com/sgl-project/sglang/pull/39668
- Status/date: merged / 2026-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v41_vit.py`, `python/sglang/srt/multimodal/deepseek_v41_image_processing.py`, `python/sglang/srt/multimodal/processors/deepseek_v41.py`; associated commits `c89c63fa3895`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +493/-2, 546 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/multimodal/deepseek_v41_image_processing.py` added +200/-0 (200 lines); hunks: -0,0 +1,200; symbols: num_image_tokens, llm_grid, solve_resize_ratio, safe_resize, touching `num_image_tokens, llm_grid, solve_resize_ratio`; `python/sglang/srt/models/deepseek_v41_vit.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: _rms_norm, get_vision_cos_sin, apply_rotary, PatchEmbed, touching `_rms_norm, get_vision_cos_sin, apply_rotary`; `python/sglang/srt/multimodal/processors/deepseek_v41.py` added +127/-0 (127 lines); hunks: -0,0 +1,127; symbols: DeepseekV41ImageProcessor, __init__, preprocess_fingerprint_payload, process_mm_data_async, touching `DeepseekV41ImageProcessor, __init__, preprocess_fingerprint_payload`.
- Code diff details:
  - `python/sglang/srt/multimodal/deepseek_v41_image_processing.py` added +200/-0 (200 lines); hunks: -0,0 +1,200; symbols: num_image_tokens, llm_grid, solve_resize_ratio, safe_resize
  - `python/sglang/srt/models/deepseek_v41_vit.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: _rms_norm, get_vision_cos_sin, apply_rotary, PatchEmbed
  - `python/sglang/srt/multimodal/processors/deepseek_v41.py` added +127/-0 (127 lines); hunks: -0,0 +1,127; symbols: DeepseekV41ImageProcessor, __init__, preprocess_fingerprint_payload, process_mm_data_async
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/deepseek_v41_image_processing.py
@@ -0,0 +1,200 @@
+"""Image preprocessing.
+An image becomes a `n_vit_h x n_vit_w` patch grid for the ViT and a `n_llm_h x n_llm_w` token grid
+after the 3x3 aligner downsample, which the LLM sees as
+    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
+Every one of those positions carries `image_token_id` in `input_ids`; only the token type tells them
+apart. The IMAGE slots are filled with aligner rows in reading order.
diff -- python/sglang/srt/models/deepseek_v41_vit.py
@@ -0,0 +1,151 @@
+"""DeepSeek-V4.1 vision tower and aligner."""
+from functools import lru_cache
+import torch
+import torch.nn.functional as F
+from torch import nn
+from sglang.srt.layers.attention.vision import (
diff -- python/sglang/srt/multimodal/processors/deepseek_v41.py
@@ -0,0 +1,127 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/multimodal/deepseek_v41_image_processing.py` added +200/-0; `python/sglang/srt/models/deepseek_v41_vit.py` added +151/-0; `python/sglang/srt/multimodal/processors/deepseek_v41.py` added +127/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/multimodal/test_processor_async_call_sites.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39665 - dsv4.1: chat encoding and tool parsing

- Link: https://github.com/sgl-project/sglang/pull/39665
- Status/date: merged / 2026-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/entrypoints/openai/encoding_dsv41.py`; associated commits `464fffbec8d0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 13 files, +1034/-58, 1332 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/entrypoints/openai/encoding_dsv41.py` added +705/-0 (705 lines); hunks: -0,0 +1,705; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, encode_arguments_to_dsml, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/encoding_dsv41.py` added +705/-0 (705 lines); hunks: -0,0 +1,705; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, encode_arguments_to_dsml
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/encoding_dsv41.py
@@ -0,0 +1,705 @@
+# Adapted from the DeepSeek-V4.1 release reference implementation.
+"""Encode DeepSeek-V4.1 chat messages.
+Mid-conversation system messages trigger the assistant generation header.
+"""
+import copy
+import json
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/entrypoints/openai/encoding_dsv41.py` added +705/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/entrypoints/openai/test_serving_chat.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38798 - dsv4.1: remaining model and runtime integration

- Link: https://github.com/sgl-project/sglang/pull/38798
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/arg_groups/model_overrides/deepseek_v4.py`, `python/sglang/srt/configs/deepseek_v4.py`, `python/sglang/srt/configs/deepseek_v41.py` and 19 files; associated commits `a6cf05817f11`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 103 files, +8802/-718, 11068 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +2070/-101 (2171 lines); `python/sglang/srt/models/deepseek_v4.py` modified +1520/-105 (1625 lines); hunks: -5,6 +5,7; -29,7 +30,13; symbols: _get_mhc_ops, wo_a_fp8_gemm_enabled, _wo_a_aiter_gemm_eligible, _apply_wo_a_bf16_matmul, touching `_get_mhc_ops, wo_a_fp8_gemm_enabled, _wo_a_aiter_gemm_eligible`; `python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py` added +266/-0 (266 lines); hunks: -0,0 +1,266; symbols: _rope_fq4, RMSNorm, __init__, forward, touching `_rope_fq4, RMSNorm, __init__`; `python/sglang/srt/models/deepseek_v4_dspark.py` modified +191/-18 (209 lines); hunks: -1,5 +1,6; -17,6 +18,7; symbols: _compute_q, forward, MarkovW2ShardGeometry, DSparkV4MarkovHead, touching `_compute_q, forward, MarkovW2ShardGeometry`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +2070/-101 (2171 lines)
  - `python/sglang/srt/models/deepseek_v4.py` modified +1520/-105 (1625 lines); hunks: -5,6 +5,7; -29,7 +30,13; symbols: _get_mhc_ops, wo_a_fp8_gemm_enabled, _wo_a_aiter_gemm_eligible, _apply_wo_a_bf16_matmul
  - `python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py` added +266/-0 (266 lines); hunks: -0,0 +1,266; symbols: _rope_fq4, RMSNorm, __init__, forward
  - `python/sglang/srt/models/deepseek_v4_dspark.py` modified +191/-18 (209 lines); hunks: -1,5 +1,6; -17,6 +18,7; symbols: _compute_q, forward, MarkovW2ShardGeometry, DSparkV4MarkovHead
  - `python/sglang/srt/multimodal/dsv41/vl_routing.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: _scale_fused_shared_weights, vision_topk
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -5,6 +5,7 @@
+from types import SimpleNamespace
@@ -29,7 +30,13 @@
+from sglang.kernels.ops.attention.dsv4.wo_a_bf16 import (
+    wo_a_bf16_gemv,
+    wo_a_bf16_small_batch,
+    wo_a_bf16_small_batch_mxfp8,
diff -- python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py
@@ -0,0 +1,266 @@
+"""DeepSeek V4.1 ratio-1/2 compressors and indexers.
+Only kv_source layers own compressed latents; later layers of the same ratio
+share that storage.
+"""
+from __future__ import annotations
+from typing import Optional, Tuple
diff -- python/sglang/srt/models/deepseek_v4_dspark.py
@@ -1,5 +1,6 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +2070/-101; `python/sglang/srt/models/deepseek_v4.py` modified +1520/-105; `python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py` added +266/-0; `python/sglang/srt/models/deepseek_v4_dspark.py` modified +191/-18; `python/sglang/srt/multimodal/dsv41/vl_routing.py` added +106/-0; `python/sglang/srt/configs/deepseek_v41.py` added +88/-0
  - tests: `test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `test/registered/attention/unittests/dsv4/test_deepseek_v4.py`, `test/registered/kernels/ops/attention/test_q8kv8_sparse_prefill_backend.py`, `test/registered/spec/dspark/test_dspark_draft_path_default.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39957 - [DSV4.1] Big fused wo_a quant

- Link: https://github.com/sgl-project/sglang/pull/39957
- Status/date: merged / 2026-09-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/kernels/jit/csrc/deepseek_v4/wo_a_fused.cuh`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py`; associated commits `d1acbe07467e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 12 files, +1070/-27, 1236 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/deepseek_v4.py` modified +53/-22 (75 lines); hunks: -30,7 +30,9; -443,6 +445,11 @@ def _wo_a_aiter_gemm_eligible(; symbols: _wo_a_aiter_gemm_eligible, _fused_wo_a_arch_supported, _apply_wo_a_bf16_matmul, __init__, touching `_wo_a_aiter_gemm_eligible, _fused_wo_a_arch_supported, _apply_wo_a_bf16_matmul`; `test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py` modified +1/-0 (1 lines); hunks: -78,6 +78,7 @@ def __init__(self, rank=3):; symbols: __init__, touching `__init__`; `python/sglang/kernels/jit/csrc/deepseek_v4/wo_a_fused.cuh` added +806/-0 (806 lines); hunks: -0,0 +1,806.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +53/-22 (75 lines); hunks: -30,7 +30,9; -443,6 +445,11 @@ def _wo_a_aiter_gemm_eligible(; symbols: _wo_a_aiter_gemm_eligible, _fused_wo_a_arch_supported, _apply_wo_a_bf16_matmul, __init__
  - `test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py` modified +1/-0 (1 lines); hunks: -78,6 +78,7 @@ def __init__(self, rank=3):; symbols: __init__
  - `python/sglang/kernels/jit/csrc/deepseek_v4/wo_a_fused.cuh` added +806/-0 (806 lines); hunks: -0,0 +1,806
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -30,7 +30,9 @@
-from sglang.kernels.ops.attention.dsv4.wo_a_bf16 import (
+from sglang.kernels.ops.attention.dsv4.wo_a import MAX_M as _FUSED_WO_A_MAX_TOKENS
+from sglang.kernels.ops.attention.dsv4.wo_a import (
+    fused_rope_wo_a_bf16,
@@ -443,6 +445,11 @@ def _wo_a_aiter_gemm_eligible(
+@functools.lru_cache(maxsize=1)
diff -- test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py
@@ -78,6 +78,7 @@ def __init__(self, rank=3):
+        self.use_fused_wo_a = False
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/wo_a_fused.cuh
@@ -0,0 +1,806 @@
+// Fused inverse-RoPE + grouped WO-A BF16 GEMM + MXFP8 quantization, SM100.
+//
+// Collapses three launches on the DSV4 decode/verify path (fused_rope_inplace,
+// _wo_a_partial, _wo_a_reduce_quant) into one cluster-launched kernel.
+//
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +53/-22; `python/sglang/kernels/jit/csrc/deepseek_v4/wo_a_fused.cuh` added +806/-0
  - tests: `test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/attention/test_wo_a_fused.py`, `test/registered/kernels/ops/layernorm/test_mxfp8_epilogue.py`, `test/registered/unit/layers/quantization/test_fp8_blockwise_linear_backends.py`, `test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39704 - [DSV4.1] Reduce mHC, metadata and small-batch router overhead

- Link: https://github.com/sgl-project/sglang/pull/39704
- Status/date: merged / 2026-09-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/kernels/jit/csrc/deepseek_v4/mhc_post_combine_norm_prefill.cuh`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `7fac84b6391b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 15 files, +1039/-59, 1456 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/deepseek_v4.py` modified +260/-37 (297 lines); hunks: -2494,24 +2494,41 @@ def forward(; -3149,6 +3166,7 @@ def _hc_combine(; symbols: forward, _hc_combine, combine_and_norm, _hc_mix_and_combine, touching `forward, _hc_combine, combine_and_norm`; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +1/-1 (2 lines); hunks: -4294,7 +4294,7 @@ def make_core_attn_metadata(; symbols: make_core_attn_metadata, touching `make_core_attn_metadata`; `python/sglang/kernels/jit/csrc/deepseek_v4/mhc_post_combine_norm_prefill.cuh` added +190/-0 (190 lines); hunks: -0,0 +1,190.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +260/-37 (297 lines); hunks: -2494,24 +2494,41 @@ def forward(; -3149,6 +3166,7 @@ def _hc_combine(; symbols: forward, _hc_combine, combine_and_norm, _hc_mix_and_combine
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +1/-1 (2 lines); hunks: -4294,7 +4294,7 @@ def make_core_attn_metadata(; symbols: make_core_attn_metadata
  - `python/sglang/kernels/jit/csrc/deepseek_v4/mhc_post_combine_norm_prefill.cuh` added +190/-0 (190 lines); hunks: -0,0 +1,190
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -2494,24 +2494,41 @@ def forward(
-        if mhc is not None:
+        if mhc is not None and mhc.overlap_only:
+            mhc.start_stats_before_all_reduce()
+            o = attn_tp_all_reduce(o)
+        elif mhc is not None:
-            o, mhc.output, mhc.normalized = all_reduce_mhc_norm(
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -4294,7 +4294,7 @@ def make_core_attn_metadata(
-            and 0 < seq_lens_casual.numel() <= 8
+            and 0 < seq_lens_casual.numel() <= 384
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/mhc_post_combine_norm_prefill.cuh
@@ -0,0 +1,190 @@
+// Prefill mHC post/combine/RMSNorm with the original BF16 intermediates.
+// Stage the collapsed row in shared memory to bound register use while preserving
+// the original Triton prefill normalization reduction and PTX arithmetic.
+#pragma once
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +260/-37; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +1/-1; `python/sglang/kernels/jit/csrc/deepseek_v4/mhc_post_combine_norm_prefill.cuh` added +190/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/attention/test_fp4_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40217 - [DeepSeek-V4.1] Bound dense prefill indexer memory

- Link: https://github.com/sgl-project/sglang/pull/40217
- Status/date: merged / 2026-09-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/unit/layers/test_dsv41_candidate_blocks.py`; associated commits `95521da18df4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +641/-94, 830 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +29/-84 (113 lines); hunks: -19,7 +19,6; -61,6 +60,7; symbols: _has_dense_fp4_indexer, _dense_fp4_mqa_logits, _low_ratio_source_projections, enter_late_layer_tail, touching `_has_dense_fp4_indexer, _dense_fp4_mqa_logits, _low_ratio_source_projections`; `test/registered/unit/layers/test_dsv41_candidate_blocks.py` added +86/-0 (86 lines); hunks: -0,0 +1,86; symbols: TestPrefillCandidateBlocks, test_causal_partial_blocks_and_forced_newest_block, test_underfilled_and_empty_candidates, test_replay_tail_keeps_request_boundaries_and_empty_tails, touching `TestPrefillCandidateBlocks, test_causal_partial_blocks_and_forced_newest_block, test_underfilled_and_empty_candidates`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +29/-84 (113 lines); hunks: -19,7 +19,6; -61,6 +60,7; symbols: _has_dense_fp4_indexer, _dense_fp4_mqa_logits, _low_ratio_source_projections, enter_late_layer_tail
  - `test/registered/unit/layers/test_dsv41_candidate_blocks.py` added +86/-0 (86 lines); hunks: -0,0 +1,86; symbols: TestPrefillCandidateBlocks, test_causal_partial_blocks_and_forced_newest_block, test_underfilled_and_empty_candidates, test_replay_tail_keeps_request_boundaries_and_empty_tails
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -19,7 +19,6 @@
-from sglang.kernels.ops.attention.dsv4 import topk_transform_ragged_v2
@@ -61,6 +60,7 @@
+    PrefillCandidateBlocks,
@@ -71,6 +71,7 @@
+from sglang.srt.layers.attention.dsv4.dense_prefill_indexer import dense_prefill_topk
@@ -313,21 +314,6 @@ def _has_dense_fp4_indexer() -> bool:
diff -- test/registered/unit/layers/test_dsv41_candidate_blocks.py
@@ -0,0 +1,86 @@
+import unittest
+import torch
+from sglang.srt.layers.attention.dsv4.candidate_indexer import (
+    PrefillCandidateBlocks,
+    candidate_block_mask,
+    select_candidate_block_ids,
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +29/-84
  - tests: `test/registered/unit/layers/test_dsv41_candidate_blocks.py` added +86/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/attention/test_dense_prefill_indexer.py`, `test/registered/unit/layers/test_dsv41_candidate_blocks.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40352 - [DSv4.1] Score prefill consumer index layers on candidate blocks with DeepGEMM

- Link: https://github.com/sgl-project/sglang/pull/40352
- Status/date: merged / 2026-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py`; associated commits `c79510cc2a33`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +753/-105, 1028 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +85/-76 (161 lines); hunks: -60,7 +60,9; -71,7 +73,7; symbols: _maybe_precompute_flashmla_sched_meta, _expand_index_page_table, init_forward_metadata_indexer, _low_ratio_prefill_indexer_metadata, touching `_maybe_precompute_flashmla_sched_meta, _expand_index_page_table, init_forward_metadata_indexer`; `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: Case, make_case, rows_of, reference_blocks, touching `Case, make_case, rows_of`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +85/-76 (161 lines); hunks: -60,7 +60,9; -71,7 +73,7; symbols: _maybe_precompute_flashmla_sched_meta, _expand_index_page_table, init_forward_metadata_indexer, _low_ratio_prefill_indexer_metadata
  - `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: Case, make_case, rows_of, reference_blocks
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -60,7 +60,9 @@
-    PrefillCandidateBlocks,
+    PrefillIndexerInputs,
+    cut_request_masks,
+    expand_index_page_table,
@@ -71,7 +73,7 @@
-from sglang.srt.layers.attention.dsv4.dense_prefill_indexer import dense_prefill_topk
diff -- test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py
@@ -0,0 +1,229 @@
+"""The DeepGEMM two-level indexer on the dense prefill path against the torch
+block selection and the dense implementation of the same protocol."""
+import unittest
+from typing import NamedTuple
+import msgspec
+import torch
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +85/-76
  - tests: `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py` added +229/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40637 - [Fix] Handle chunked paged MQA metadata in DSV4.1 eager forwards

- Link: https://github.com/sgl-project/sglang/pull/40637
- Status/date: merged / 2026-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; associated commits `4cbf290fb9e7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +281/-25, 399 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +38/-11 (49 lines); hunks: -3447,17 +3447,44 @@ def _low_ratio_index_topk_decode(self, layer, x, q_lora,...; symbols: _low_ratio_index_topk_decode, touching `_low_ratio_index_topk_decode`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +38/-11 (49 lines); hunks: -3447,17 +3447,44 @@ def _low_ratio_index_topk_decode(self, layer, x, q_lora,...; symbols: _low_ratio_index_topk_decode
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -3447,17 +3447,44 @@ def _low_ratio_index_topk_decode(self, layer, x, q_lora, pos, req=None) -> None:
-        logits = deep_gemm_fp4_paged_mqa_logits(
-            (q_fp4, q_sf),
-            k_cache,
-            weights,
-            metadata.compressed_seq_lens,
-            metadata.page_table,
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +38/-11
- Risk and verification: The diff ships test coverage in `test/registered/unit/layers/test_dsv4_nonpaged_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39929 - [Bugfix] Align DeepSeek-V4.1 reasoning effort budgets

- Link: https://github.com/sgl-project/sglang/pull/39929
- Status/date: merged / 2026-09-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`, `python/sglang/srt/entrypoints/openai/encoding_dsv41.py`; associated commits `6bbd689ab416`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +3/-3, 20 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/entrypoints/openai/encoding_dsv41.py` modified +2/-2 (4 lines); hunks: -66,8 +66,8; `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` modified +1/-1 (2 lines); hunks: -134,7 +134,7 @@ print("Answer:", msg.content).
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/encoding_dsv41.py` modified +2/-2 (4 lines); hunks: -66,8 +66,8
  - `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` modified +1/-1 (2 lines); hunks: -134,7 +134,7 @@ print("Answer:", msg.content)
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/encoding_dsv41.py
@@ -66,8 +66,8 @@
-    "low": 25,
-    "high": 50,
+    "low": 50,
+    "high": 75,
diff -- docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx
@@ -134,7 +134,7 @@ print("Answer:", msg.content)
-On the request, `reasoning_effort` accepts the tiers `low`, `high`, `xhigh` and `max`, or a float in `[0.0, 0.99]` that maps onto the model's 1–100 budget. An integer budget is re
+On the request, `reasoning_effort` accepts the tiers `low` (50), `high` (75), `xhigh` (75), and `max` (100), or a float in `[0.0, 0.99]` that maps onto the model's 1–100 budget. A
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/entrypoints/openai/encoding_dsv41.py` modified +2/-2
  - docs: `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/encoding_dsv41.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41018 - dsv4.1-amd: gfx950 MXFP8 matmul kernels and fp8-grid producers

- Link: https://github.com/sgl-project/sglang/pull/41018
- Status/date: merged / 2026-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/kernels/jit/csrc/deepseek_v4/mxfp8_gemv_gfx95.cuh`; associated commits `27e883a20d2d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 10 files, +2287/-1, 2321 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/kernels/jit/csrc/deepseek_v4/mxfp8_gemv_gfx95.cuh` added +376/-0 (376 lines); hunks: -0,0 +1,376.
- Code diff details:
  - `python/sglang/kernels/jit/csrc/deepseek_v4/mxfp8_gemv_gfx95.cuh` added +376/-0 (376 lines); hunks: -0,0 +1,376
- Key code excerpts:

```diff
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/mxfp8_gemv_gfx95.cuh
@@ -0,0 +1,376 @@
+// MXFP8 skinny GEMM for gfx950: out[M, N] bf16 = X[M, K] . W[N, K]^T, M <= 32, fp8 e4m3 operands,
+// one ue8m0 scale per 32 K on both sides, fp32 accumulation on v_mfma_scale_f32_16x16x128_f8f6f4.
+// The sum order is fixed by (tile, wave, step): repeated calls are bitwise equal, and a row's bits depend
+// on the config (its wave count), not on M.
+//
+// Operand layout of the 16x16x128 scaled MFMA (undocumented):
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/kernels/jit/csrc/deepseek_v4/mxfp8_gemv_gfx95.cuh` added +376/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/quantization/test_fp8_grid_producers_gfx95.py`, `test/registered/kernels/ops/quantization/test_mxfp8_amd_gfx95.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41291 - [DSv4.1] Move the ratio-1/2 index top-k ops into kernels/ops/attention/dsv4

- Link: https://github.com/sgl-project/sglang/pull/41291
- Status/date: merged / 2026-09-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py`, `test/registered/unit/layers/test_dsv41_candidate_blocks.py`; associated commits `0967a013c2ab`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +419/-329, 983 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-14 (29 lines); hunks: -19,6 +19,10; -30,6 +34,9; symbols: _low_ratio_index_topk_prefill_graph, _low_ratio_index_topk_decode, touching `_low_ratio_index_topk_prefill_graph, _low_ratio_index_topk_decode`; `test/registered/unit/layers/test_dsv41_candidate_blocks.py` modified +2/-2 (4 lines); hunks: -2,12 +2,12; `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py` modified +2/-4 (6 lines); hunks: -7,11 +7,9.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-14 (29 lines); hunks: -19,6 +19,10; -30,6 +34,9; symbols: _low_ratio_index_topk_prefill_graph, _low_ratio_index_topk_decode
  - `test/registered/unit/layers/test_dsv41_candidate_blocks.py` modified +2/-2 (4 lines); hunks: -2,12 +2,12
  - `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py` modified +2/-4 (6 lines); hunks: -7,11 +7,9
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -19,6 +19,10 @@
+from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
+    mask_topk_scores,
+    select_candidate_blocks,
+)
@@ -30,6 +34,9 @@
+from sglang.kernels.ops.attention.dsv4.index_logits import (
diff -- test/registered/unit/layers/test_dsv41_candidate_blocks.py
@@ -2,12 +2,12 @@
-from sglang.srt.layers.attention.dsv4.candidate_indexer import (
-    PrefillCandidateBlocks,
+from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
+from sglang.srt.layers.attention.dsv4.candidate_indexer import PrefillCandidateBlocks
diff -- test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py
@@ -7,11 +7,9 @@
+from sglang.kernels.ops.attention.dsv4.candidate_blocks import select_candidate_blocks
-from sglang.srt.layers.attention.dsv4.candidate_indexer import (
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +15/-14
  - tests: `test/registered/unit/layers/test_dsv41_candidate_blocks.py` modified +2/-2; `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py` modified +2/-4
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py`, `test/registered/unit/layers/test_dsv41_candidate_blocks.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41125 - [DSv4.1] Move the low-ratio index top-k into dsv4/low_ratio_indexer

- Link: https://github.com/sgl-project/sglang/pull/41125
- Status/date: merged / 2026-09-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py`, `test/registered/unit/layers/test_dsv41_candidate_blocks.py`; associated commits `8772916e06e6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 18 files, +2253/-1683, 4417 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +144/-560 (704 lines); hunks: -19,10 +19,6; -33,10 +29,6; symbols: _is_sm100_or_newer, _get_logical_forward_mode, _maybe_precompute_flashmla_sched_meta, _every_request_fits, touching `_is_sm100_or_newer, _get_logical_forward_mode, _maybe_precompute_flashmla_sched_meta`; `test/registered/unit/layers/test_dsv41_candidate_blocks.py` modified +83/-49 (132 lines); hunks: -3,82 +3,116; symbols: TestPrefillCandidateBlocks, id_sets, masked_topk, TestCandidateBlockIds, touching `TestPrefillCandidateBlocks, id_sets, masked_topk`; `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py` modified +119/-59 (178 lines); hunks: -7,11 +7,12; -35,7 +36,10; symbols: Case, make_case, rows_of, touching `Case, make_case, rows_of`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +144/-560 (704 lines); hunks: -19,10 +19,6; -33,10 +29,6; symbols: _is_sm100_or_newer, _get_logical_forward_mode, _maybe_precompute_flashmla_sched_meta, _every_request_fits
  - `test/registered/unit/layers/test_dsv41_candidate_blocks.py` modified +83/-49 (132 lines); hunks: -3,82 +3,116; symbols: TestPrefillCandidateBlocks, id_sets, masked_topk, TestCandidateBlockIds
  - `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py` modified +119/-59 (178 lines); hunks: -7,11 +7,12; -35,7 +36,10; symbols: Case, make_case, rows_of
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -19,10 +19,6 @@
-from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
-    mask_topk_scores,
-    select_candidate_blocks,
-)
@@ -33,10 +29,6 @@
-from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_decode
diff -- test/registered/unit/layers/test_dsv41_candidate_blocks.py
@@ -3,82 +3,116 @@
-    candidate_block_mask,
-    select_candidate_blocks,
+    topk_among_blocks,
-from sglang.srt.layers.attention.dsv4.candidate_indexer import PrefillCandidateBlocks
+from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import BlockIds
+from sglang.srt.layers.attention.dsv4.v41_indexer.types import get_tail_row_indices
diff -- test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py
@@ -7,11 +7,12 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +144/-560
  - tests: `test/registered/unit/layers/test_dsv41_candidate_blocks.py` modified +83/-49; `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py` modified +119/-59
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/attention/test_dense_prefill_indexer.py`, `test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py`, `test/registered/unit/layers/test_dsv41_candidate_blocks.py`, `test/registered/unit/layers/test_dsv4_nonpaged_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41019 - dsv4.1-amd: KV cache layouts, FP4 indexer, compressor and router kernels

- Link: https://github.com/sgl-project/sglang/pull/41019
- Status/date: merged / 2026-09-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/kernels/jit/csrc/deepseek_v4/c1.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope_hip.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh`, `python/sglang/kernels/jit/csrc/deepseek_v4/store.cuh` and 8 files; associated commits `effb75218808`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 26 files, +2766/-116, 3544 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope_hip.cuh` added +156/-0 (156 lines); hunks: -0,0 +1,156; `python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py` added +91/-0 (91 lines); hunks: -0,0 +1,91; symbols: quantize_to_e2m1_codes, dequantize_e2m1_codes, quantize_k_cache_v41_fp4, dequantize_k_cache_v41_fp4, touching `quantize_to_e2m1_codes, dequantize_e2m1_codes, quantize_k_cache_v41_fp4`; `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh` modified +79/-6 (85 lines); hunks: -15,6 +15,7; -258,6 +259,12 @@ struct FusedKNormRopeFlashMLAParams {; `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp4_utils.cuh` modified +52/-0 (52 lines); hunks: -23,18 +23,66 @@ constexpr float kAmaxFloor = 6.0f * 1.1754943508222875e-38f;; -50,8 +98,12 @@ SGL_DEVICE fp32x2_t block_scale(float amax) {.
- Code diff details:
  - `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope_hip.cuh` added +156/-0 (156 lines); hunks: -0,0 +1,156
  - `python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py` added +91/-0 (91 lines); hunks: -0,0 +1,91; symbols: quantize_to_e2m1_codes, dequantize_e2m1_codes, quantize_k_cache_v41_fp4, dequantize_k_cache_v41_fp4
  - `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh` modified +79/-6 (85 lines); hunks: -15,6 +15,7; -258,6 +259,12 @@ struct FusedKNormRopeFlashMLAParams {
  - `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp4_utils.cuh` modified +52/-0 (52 lines); hunks: -23,18 +23,66 @@ constexpr float kAmaxFloor = 6.0f * 1.1754943508222875e-38f;; -50,8 +98,12 @@ SGL_DEVICE fp32x2_t block_scale(float amax) {
  - `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh` modified +18/-21 (39 lines); hunks: -5,8 +5,10; -111,8 +113,6 @@ struct PagedKV {
- Key code excerpts:

```diff
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope_hip.cuh
@@ -0,0 +1,156 @@
+/// Index-K write of `fp4_indexer_rope.cuh` into the split FlyDSL layout: payload `[npages, 1, 4, kPageSize, 16]`
+/// (chunk `c` holds elements `[32c, 32c + 32)`) and ue8m0 exponents `[npages, 1, 4, kPageSize]` with
+/// the slot axis transposed as a 16 x 4 tile -- the bytes `store_fp4_index_k_cache_split` writes.
+#pragma once
+#ifndef USE_ROCM
+#error "fp4_indexer_rope_hip.cuh writes the FlyDSL index-K layout, which exists on ROCm only"
diff -- python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py
@@ -0,0 +1,91 @@
+"""Independent packed-FP4 (V41_FP4 layout) cache oracle for the V4.1 store and reader tests."""
+from typing import Optional
+import torch
+KV_DIM = 512
+FP4_GROUP = 16  # values per e4m3 scale
+E2M1_BYTES_PER_TOKEN = KV_DIM // 2
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh
@@ -15,6 +15,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope_hip.cuh` added +156/-0; `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh` modified +79/-6; `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp4_utils.cuh` modified +52/-0; `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh` modified +18/-21; `python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope.cuh` modified +13/-2; `python/sglang/kernels/jit/csrc/deepseek_v4/c1.cuh` modified +1/-1
  - tests: `python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py` added +91/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py`, `test/registered/kernels/ops/attention/dsv4/test_v41_kv_store.py`, `test/registered/kernels/ops/attention/test_fp4_indexer_hip.py`, `test/registered/kernels/ops/gemm/test_router_gemv_hip.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41345 - [DSV4.1][HiCache] fix: wait for the layer transfer before reading low-ratio index-K

- Link: https://github.com/sgl-project/sglang/pull/41345
- Status/date: merged / 2026-09-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; associated commits `38d865489af5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +8/-0, 35 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +2/-0 (2 lines); hunks: -1867,6 +1867,7 @@ def get_low_ratio_index_k_dequant(; -1877,6 +1878,7 @@ def get_low_ratio_index_k_fp4(; symbols: get_low_ratio_index_k_dequant, get_low_ratio_index_k_fp4, touching `get_low_ratio_index_k_dequant, get_low_ratio_index_k_fp4`.
- Code diff details:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +2/-0 (2 lines); hunks: -1867,6 +1867,7 @@ def get_low_ratio_index_k_dequant(; -1877,6 +1878,7 @@ def get_low_ratio_index_k_fp4(; symbols: get_low_ratio_index_k_dequant, get_low_ratio_index_k_fp4
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -1867,6 +1867,7 @@ def get_low_ratio_index_k_dequant(
+        self.wait_layer_transfer(layer_id)
@@ -1877,6 +1878,7 @@ def get_low_ratio_index_k_fp4(
+        self.wait_layer_transfer(layer_id)
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/mem_cache/test_dsv4_compressed_pools.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41020 - dsv4.1-amd: gfx950 sparse decode attention and sorted top-k

- Link: https://github.com/sgl-project/sglang/pull/41020
- Status/date: merged / 2026-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/kernels/aot/csrc/elementwise/deepseek_v4_topk.cu`, `test/registered/kernels/ops/attention/dsv4/test_dsv41_decode_glue_hip.py`; associated commits `096b066fb4f6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 17 files, +2569/-17, 2801 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/registered/kernels/ops/attention/dsv4/test_dsv41_decode_glue_hip.py` added +358/-0 (358 lines); hunks: -0,0 +1,358; symbols: index_slots, reference_position_mask, candidate_block_ids_to_mask, ids_to_position_mask, touching `index_slots, reference_position_mask, candidate_block_ids_to_mask`; `python/sglang/kernels/aot/csrc/elementwise/deepseek_v4_topk.cu` modified +172/-6 (178 lines); hunks: -46,6 +46,7 @@ constexpr size_t kSMEM = 48 * 1024; // bytes; -57,6 +58,8 @@ struct TopKParams {.
- Code diff details:
  - `test/registered/kernels/ops/attention/dsv4/test_dsv41_decode_glue_hip.py` added +358/-0 (358 lines); hunks: -0,0 +1,358; symbols: index_slots, reference_position_mask, candidate_block_ids_to_mask, ids_to_position_mask
  - `python/sglang/kernels/aot/csrc/elementwise/deepseek_v4_topk.cu` modified +172/-6 (178 lines); hunks: -46,6 +46,7 @@ constexpr size_t kSMEM = 48 * 1024; // bytes; -57,6 +58,8 @@ struct TopKParams {
- Key code excerpts:

```diff
diff -- test/registered/kernels/ops/attention/dsv4/test_dsv41_decode_glue_hip.py
@@ -0,0 +1,358 @@
+"""The V4.1 HIP decode glue is bitwise the torch chains it replaces and selects within each row's reach."""
+from __future__ import annotations
+import random
+import sys
+import pytest
+import torch
diff -- python/sglang/kernels/aot/csrc/elementwise/deepseek_v4_topk.cu
@@ -46,6 +46,7 @@ constexpr size_t kSMEM = 48 * 1024;  // bytes
+// seq_lens[b] must not exceed scores.size(1) or page_table.size(1) << page_bits: a row reads up to its length
@@ -57,6 +58,8 @@ struct TopKParams {
+  // Emit each row's picks in ascending order (see bitonic_sort_u32).
+  bool sort_output;
@@ -251,6 +254,116 @@ radix_topk(const float* __restrict__ input, int32_t* __restrict__ output, uint32
+#ifdef USE_ROCM
```

- Extracted files (not manually reviewed):
  - tests: `test/registered/kernels/ops/attention/dsv4/test_dsv41_decode_glue_hip.py` added +358/-0
  - runtime: `python/sglang/kernels/aot/csrc/elementwise/deepseek_v4_topk.cu` modified +172/-6
- Risk and verification: The diff ships test coverage in `python/sglang/kernels/aot/tests/test_topk.py`, `test/registered/kernels/ops/attention/dsv4/test_compact_attention_hip.py`, `test/registered/kernels/ops/attention/dsv4/test_dsv41_decode_glue_hip.py`, `test/registered/kernels/ops/attention/dsv4/test_swapab_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41021 - dsv4.1-amd: fused mHC boundary and all-reduce + mHC post kernels

- Link: https://github.com/sgl-project/sglang/pull/41021
- Status/date: merged / 2026-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/kernels/jit/csrc/deepseek_v4/mhc_boundary_gfx95.cuh`; associated commits `9654e5c4bcb2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +1990/-1, 2025 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/kernels/jit/csrc/deepseek_v4/mhc_boundary_gfx95.cuh` added +593/-0 (593 lines); hunks: -0,0 +1,593.
- Code diff details:
  - `python/sglang/kernels/jit/csrc/deepseek_v4/mhc_boundary_gfx95.cuh` added +593/-0 (593 lines); hunks: -0,0 +1,593
- Key code excerpts:

```diff
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/mhc_boundary_gfx95.cuh
@@ -0,0 +1,593 @@
+// gfx950 prefill-regime kernel for the fused mHC sublayer boundary: the operation sequence
+// of the Triton _hc_boundary_partial_kernel (mhc_boundary_hip.py) at its decode configuration,
+// so a row's fp32 result is bitwise identical in both regimes. Compiled with -ffp-contract=off:
+// the Triton binary contracts none of the hc_post multiplies and adds.
+#pragma once
+#ifndef USE_ROCM
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/kernels/jit/csrc/deepseek_v4/mhc_boundary_gfx95.cuh` added +593/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/communication/test_all_reduce_mhc_hip.py`, `test/registered/kernels/ops/layernorm/test_hc_boundary_hip.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41308 - dsv4.1-amd: serve DeepSeek-V4.1 on gfx950

- Link: https://github.com/sgl-project/sglang/pull/41308
- Status/date: merged / 2026-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`, `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`, `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` and 18 files; associated commits `3a398442bfcc`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 39 files, +5723/-365, 7675 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +1211/-198 (1409 lines); hunks: -4,37 +4,91; -74,6 +128,69 @@ def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGN...; symbols: _pad_last_dim, _fold_lengths_into_index_lists, _fold_lengths_for_aiter_sparse, _create_flashmla_metadata, touching `_pad_last_dim, _fold_lengths_into_index_lists, _fold_lengths_for_aiter_sparse`; `test/registered/unit/models/test_deepseek_v4_amd_tp4.py` added +454/-0 (454 lines); hunks: -0,0 +1,454; symbols: group, _Projection, __init__, __call__, touching `group, _Projection, __init__`; `python/sglang/srt/models/deepseek_v4.py` modified +194/-80 (274 lines); hunks: -425,7 +425,9 @@ def _wo_a_aiter_gemm_eligible(; -459,31 +461,48 @@ def _apply_wo_a_bf16_matmul(; symbols: _wo_a_aiter_gemm_eligible, _apply_wo_a_bf16_matmul, deepseek_v4_low_ratio_sources, __init__, touching `_wo_a_aiter_gemm_eligible, _apply_wo_a_bf16_matmul, deepseek_v4_low_ratio_sources`; `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_hip.py` added +247/-0 (247 lines); hunks: -0,0 +1,247; symbols: init_mqa_layer, wo_a_emits_fp8_grid, wo_a_split_k_allowed, use_fused_qk_norm_rope, touching `init_mqa_layer, wo_a_emits_fp8_grid, wo_a_split_k_allowed`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +1211/-198 (1409 lines); hunks: -4,37 +4,91; -74,6 +128,69 @@ def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGN...; symbols: _pad_last_dim, _fold_lengths_into_index_lists, _fold_lengths_for_aiter_sparse, _create_flashmla_metadata
  - `test/registered/unit/models/test_deepseek_v4_amd_tp4.py` added +454/-0 (454 lines); hunks: -0,0 +1,454; symbols: group, _Projection, __init__, __call__
  - `python/sglang/srt/models/deepseek_v4.py` modified +194/-80 (274 lines); hunks: -425,7 +425,9 @@ def _wo_a_aiter_gemm_eligible(; -459,31 +461,48 @@ def _apply_wo_a_bf16_matmul(; symbols: _wo_a_aiter_gemm_eligible, _apply_wo_a_bf16_matmul, deepseek_v4_low_ratio_sources, __init__
  - `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_hip.py` added +247/-0 (247 lines); hunks: -0,0 +1,247; symbols: init_mqa_layer, wo_a_emits_fp8_grid, wo_a_split_k_allowed, use_fused_qk_norm_rope
  - `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` modified +204/-2 (206 lines); hunks: -1,11 +1,22; -15,6 +26,22; symbols: apply_mhc_post_pre_boundary, hc_boundary, _can_fuse_mhc, _make_mhc_fusion
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -4,37 +4,91 @@
-from dataclasses import dataclass, field
+from dataclasses import dataclass, field, fields
+    ClassVar,
+    Tuple,
+from sglang.kernels.ops.attention.dsv4.attn_glue_hip import (
+    expand_index_page_table,
diff -- test/registered/unit/models/test_deepseek_v4_amd_tp4.py
@@ -0,0 +1,454 @@
+"""DeepSeek-V4.1 TP4 collectives on gfx950: fused all-reduce + mHC post under graph replay and bit-exact Engram reconstruction."""
+import os
+from contextlib import nullcontext
+from types import SimpleNamespace
+from unittest.mock import patch
+import pytest
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -425,7 +425,9 @@ def _wo_a_aiter_gemm_eligible(
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +1211/-198; `python/sglang/srt/models/deepseek_v4.py` modified +194/-80; `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_hip.py` added +247/-0; `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` modified +204/-2; `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py` added +205/-0
  - tests: `test/registered/unit/models/test_deepseek_v4_amd_tp4.py` added +454/-0; `test/registered/unit/models/test_deepseek_v4_amd_wo_a_bf16.py` modified +102/-1; `test/registered/unit/layers/test_dsv41_candidate_graph_variants.py` added +91/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_dsv4_hip_bcg_metadata.py`, `test/registered/attention/unittests/dsv4/test_dsv41_bcg_hip.py`, `test/registered/attention/unittests/dsv4/test_dsv41_fused_compress.py`, `test/registered/attention/unittests/dsv4/test_hip_flash_mla_aiter_sparse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41337 - [DSV4/DSA] Name the FlashMLA KV format and drop the V4.1 support probe

- Link: https://github.com/sgl-project/sglang/pull/41337
- Status/date: merged / 2026-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; associated commits `3ed6367d3f28`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +8/-23, 89 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-19 (20 lines); hunks: -159,16 +159,6 @@ def resolve_compressed_kv_layout(; -194,17 +184,9 @@ def select_dsv4_kv_layout() -> Tuple[KVLayout, Optional[str]]:; symbols: resolve_compressed_kv_layout, flashmla_supports_v41_kv_layouts, select_dsv4_kv_layout, touching `resolve_compressed_kv_layout, flashmla_supports_v41_kv_layouts, select_dsv4_kv_layout`.
- Code diff details:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-19 (20 lines); hunks: -159,16 +159,6 @@ def resolve_compressed_kv_layout(; -194,17 +184,9 @@ def select_dsv4_kv_layout() -> Tuple[KVLayout, Optional[str]]:; symbols: resolve_compressed_kv_layout, flashmla_supports_v41_kv_layouts, select_dsv4_kv_layout
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -159,16 +159,6 @@ def resolve_compressed_kv_layout(
-def flashmla_supports_v41_kv_layouts() -> bool:
-    """Whether the installed FlashMLA decode kernel reads the V41 / V41_FP4
-    formats; its docstring lists the bytes-per-token it detects."""
-    try:
-        from sgl_kernel.flash_mla import flash_mla_with_kvcache
-    except Exception:
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +1/-19
- Risk and verification: Runtime changes concentrate in `python/pyproject.toml`, `python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/environ.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41970 - [AMD][V4.1][*/N] Switch the fp8 dense GEMMs on gfx950 to aiter's MXFP8 GEMM

- Link: https://github.com/sgl-project/sglang/pull/41970
- Status/date: merged / 2026-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py`; associated commits `73ba6513f469`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +215/-31, 417 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py` modified +16/-10 (26 lines); hunks: -59,31 +59,37 @@ def fused_rmsnorm_fake_quant_eligible(; -99,7 +105,7 @@ def q_norm_fake_quant(attn, q_lora: torch.Tensor) -> Tuple[to...; symbols: fused_rmsnorm_fake_quant_eligible, _native_mxfp8_consumer, _mxfp8_consumer, q_norm_fake_quant, touching `fused_rmsnorm_fake_quant_eligible, _native_mxfp8_consumer, _mxfp8_consumer`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py` modified +16/-10 (26 lines); hunks: -59,31 +59,37 @@ def fused_rmsnorm_fake_quant_eligible(; -99,7 +105,7 @@ def q_norm_fake_quant(attn, q_lora: torch.Tensor) -> Tuple[to...; symbols: fused_rmsnorm_fake_quant_eligible, _native_mxfp8_consumer, _mxfp8_consumer, q_norm_fake_quant
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py
@@ -59,31 +59,37 @@ def fused_rmsnorm_fake_quant_eligible(
-    (V4.1), whose dense route takes the norm output already on the fp8 grid as an
-    Fp8GridActivation."""
+    (V4.1), whose native or aiter dense route takes the norm output already on the fp8
+    grid (Fp8GridActivation) or as fp8 + ue8m0 (Mxfp8Activation)."""
+    backend = resolve_block_fp8_mxfp8_backend()
-        and resolve_block_fp8_mxfp8_backend().is_gfx95_mxfp8_native()
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py` modified +16/-10
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/quantization/test_mxfp8_amd_gfx95.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42014 - [AMD][V4.1][*/N] Build DSpark draft metadata inside the CUDA graph on ROCm

- Link: https://github.com/sgl-project/sglang/pull/42014
- Status/date: merged / 2026-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`; associated commits `f45c004e18d2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +50/-8, 107 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +47/-8 (55 lines); hunks: -112,6 +112,8; -792,6 +794,20 @@ def _grouped_asm_enabled() -> bool:; symbols: _grouped_asm_enabled, DSV4RawDSparkDraftMetadata, copy_, _GraphBucket, touching `_grouped_asm_enabled, DSV4RawDSparkDraftMetadata, copy_`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +47/-8 (55 lines); hunks: -112,6 +112,8; -792,6 +794,20 @@ def _grouped_asm_enabled() -> bool:; symbols: _grouped_asm_enabled, DSV4RawDSparkDraftMetadata, copy_, _GraphBucket
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -112,6 +112,8 @@
+_DSPARK_DRAFT_RAW_METADATA = envs.SGLANG_HIP_DSPARK_DRAFT_RAW_METADATA.get()
@@ -792,6 +794,20 @@ def _grouped_asm_enabled() -> bool:
+@dataclass
+class DSV4RawDSparkDraftMetadata:
+    req_pool_indices: torch.Tensor
+    seq_lens: torch.Tensor
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +47/-8
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42017 - [AMD][V4.1][*/N] OPUS sparse prefill on gfx950 through layout conversion

- Link: https://github.com/sgl-project/sglang/pull/42017
- Status/date: merged / 2026-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `test/registered/attention/unittests/dsv4/test_dsv41_opus_sparse_prefill_hip.py`; associated commits `3c4e2187793d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +497/-3, 606 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +232/-3 (235 lines); hunks: -44,10 +44,16; -81,12 +87,18; symbols: _pad_last_dim, _OpusPrefillState, _fold_lengths_into_index_lists, init_flashmla_related, touching `_pad_last_dim, _OpusPrefillState, _fold_lengths_into_index_lists`; `test/registered/attention/unittests/dsv4/test_dsv41_opus_sparse_prefill_hip.py` added +161/-0 (161 lines); hunks: -0,0 +1,161; symbols: _opus_available, _padded_lists, _csr_reference, _attention_reference, touching `_opus_available, _padded_lists, _csr_reference`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +232/-3 (235 lines); hunks: -44,10 +44,16; -81,12 +87,18; symbols: _pad_last_dim, _OpusPrefillState, _fold_lengths_into_index_lists, init_flashmla_related
  - `test/registered/attention/unittests/dsv4/test_dsv41_opus_sparse_prefill_hip.py` added +161/-0 (161 lines); hunks: -0,0 +1,161; symbols: _opus_available, _padded_lists, _csr_reference, _attention_reference
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -44,10 +44,16 @@
+from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged
+from sglang.kernels.ops.attention.dsv4.opus_sparse_prefill_hip import (
+    Csr,
+    combined_to_csr,
+    opus_sparse_prefill,
+)
diff -- test/registered/attention/unittests/dsv4/test_dsv41_opus_sparse_prefill_hip.py
@@ -0,0 +1,161 @@
+"""The DSV4.1 OPUS sparse prefill path on gfx950: the -1 padded lists to CSR conversion, the
+two-source OPUS attention against an fp32 reference, and the -1 filled raw top-k buffers it reads."""
+import unittest
+import torch
+from sglang.srt.utils import is_gfx95_supported, is_hip
+from sglang.test.ci.ci_register import register_amd_ci
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` modified +232/-3
  - tests: `test/registered/attention/unittests/dsv4/test_dsv41_opus_sparse_prefill_hip.py` added +161/-0
- Risk and verification: The diff ships test coverage in `test/registered/attention/unittests/dsv4/test_dsv41_opus_sparse_prefill_hip.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42011 - [AMD][V4.1][*/N] Fix shared-expert fusion accuracy and speed up MoE routing on ROCm

- Link: https://github.com/sgl-project/sglang/pull/42011
- Status/date: merged / 2026-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/unit/models/test_deepseek_v4_mxfp4_shared_expert_requant.py`; associated commits `29f6d408c01c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +248/-14, 424 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/registered/unit/models/test_deepseek_v4_mxfp4_shared_expert_requant.py` modified +18/-3 (21 lines); hunks: -1,7 +1,9; -64,12 +66,25 @@ def test_packing_layout_contract(self):; symbols: test_packing_layout_contract, test_ties_round_to_even, test_roundtrip_error_is_mxfp4_sized, touching `test_packing_layout_contract, test_ties_round_to_even, test_roundtrip_error_is_mxfp4_sized`; `python/sglang/srt/layers/quantization/fp8_utils.py` modified +42/-0 (42 lines); hunks: -1729,6 +1729,10 @@ def quantize_block_fp8_weight_to_mxfp4(; -1747,6 +1751,44 @@ def quantize_block_fp8_weight_to_mxfp4(; symbols: quantize_block_fp8_weight_to_mxfp4, _quantize_block_fp8_weight_to_mxfp4_rne, requant_weight_ue8m0_inplace, touching `quantize_block_fp8_weight_to_mxfp4, _quantize_block_fp8_weight_to_mxfp4_rne, requant_weight_ue8m0_inplace`; `python/sglang/srt/layers/moe/topk.py` modified +34/-7 (41 lines); hunks: -1441,8 +1441,12 @@ def biased_topk_jit_kernel_impl(; -1455,6 +1459,7 @@ def biased_topk_jit_kernel_impl(; symbols: biased_topk_jit_kernel_impl, _post_process_topk_ids, select_experts, touching `biased_topk_jit_kernel_impl, _post_process_topk_ids, select_experts`; `python/sglang/srt/layers/moe/moe_runner/aiter.py` modified +4/-0 (4 lines); hunks: -12,6 +12,7; -43,6 +44,8; symbols: AiterQuantType, run, touching `AiterQuantType, run`.
- Code diff details:
  - `test/registered/unit/models/test_deepseek_v4_mxfp4_shared_expert_requant.py` modified +18/-3 (21 lines); hunks: -1,7 +1,9; -64,12 +66,25 @@ def test_packing_layout_contract(self):; symbols: test_packing_layout_contract, test_ties_round_to_even, test_roundtrip_error_is_mxfp4_sized
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +42/-0 (42 lines); hunks: -1729,6 +1729,10 @@ def quantize_block_fp8_weight_to_mxfp4(; -1747,6 +1751,44 @@ def quantize_block_fp8_weight_to_mxfp4(; symbols: quantize_block_fp8_weight_to_mxfp4, _quantize_block_fp8_weight_to_mxfp4_rne, requant_weight_ue8m0_inplace
  - `python/sglang/srt/layers/moe/topk.py` modified +34/-7 (41 lines); hunks: -1441,8 +1441,12 @@ def biased_topk_jit_kernel_impl(; -1455,6 +1459,7 @@ def biased_topk_jit_kernel_impl(; symbols: biased_topk_jit_kernel_impl, _post_process_topk_ids, select_experts
  - `python/sglang/srt/layers/moe/moe_runner/aiter.py` modified +4/-0 (4 lines); hunks: -12,6 +12,7; -43,6 +44,8; symbols: AiterQuantType, run
  - `python/sglang/kernels/ops/moe/rocm_router_gate.py` modified +17/-4 (21 lines); hunks: -224,11 +224,13 @@ def _router_gate_kernel(; -247,7 +249,12 @@ def _router_gate_kernel(; symbols: _router_gate_kernel, rocm_router_gate
- Key code excerpts:

```diff
diff -- test/registered/unit/models/test_deepseek_v4_mxfp4_shared_expert_requant.py
@@ -1,7 +1,9 @@
+from unittest import mock
+from sglang.srt.layers.quantization import fp8_utils
@@ -64,12 +66,25 @@ def test_packing_layout_contract(self):
-        # Zero padding is not asserted byte-exactly: the quantizer encodes 0.0
-        # as -0.0 (code 8), which the kernel decodes back to zero. Check the
-        # padding dequantizes to zero without pinning its sign bit.
diff -- python/sglang/srt/layers/quantization/fp8_utils.py
@@ -1729,6 +1729,10 @@ def quantize_block_fp8_weight_to_mxfp4(
+    if _is_hip:
+        return _quantize_block_fp8_weight_to_mxfp4_rne(
+            fp8_weight, fp8_scale, weight_block_size, mxfp4_block_size
+        )
@@ -1747,6 +1751,44 @@ def quantize_block_fp8_weight_to_mxfp4(
+def _quantize_block_fp8_weight_to_mxfp4_rne(
diff -- python/sglang/srt/layers/moe/topk.py
@@ -1441,8 +1441,12 @@ def biased_topk_jit_kernel_impl(
```

- Extracted files (not manually reviewed):
  - tests: `test/registered/unit/models/test_deepseek_v4_mxfp4_shared_expert_requant.py` modified +18/-3
  - runtime: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +42/-0; `python/sglang/srt/layers/moe/topk.py` modified +34/-7; `python/sglang/srt/layers/moe/moe_runner/aiter.py` modified +4/-0; `python/sglang/kernels/ops/moe/rocm_router_gate.py` modified +17/-4; `python/sglang/srt/environ.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/moe/test_rocm_router_gate.py`, `test/registered/unit/models/test_deepseek_v4_mxfp4_shared_expert_requant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42128 - [Fix][DSV4.1] SWA page size with bounded replay

- Link: https://github.com/sgl-project/sglang/pull/42128
- Status/date: merged / 2026-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; associated commits `ef867fa40d2e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +22/-2, 52 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +2/-2 (4 lines); hunks: -2420,7 +2420,7 @@ def _build_sparse_prefill_chunk_cache(; -3316,7 +3316,7 @@ def _forward_attention(; symbols: _build_sparse_prefill_chunk_cache, _forward_attention, touching `_build_sparse_prefill_chunk_cache, _forward_attention`; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +5/-0 (5 lines); hunks: -1858,6 +1858,11 @@ def get_swa_key_layout(self) -> KVLayout:; symbols: get_swa_key_layout, get_swa_key_page_size, get_swa_key_bytes_per_token, touching `get_swa_key_layout, get_swa_key_page_size, get_swa_key_bytes_per_token`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +2/-2 (4 lines); hunks: -2420,7 +2420,7 @@ def _build_sparse_prefill_chunk_cache(; -3316,7 +3316,7 @@ def _forward_attention(; symbols: _build_sparse_prefill_chunk_cache, _forward_attention
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +5/-0 (5 lines); hunks: -1858,6 +1858,11 @@ def get_swa_key_layout(self) -> KVLayout:; symbols: get_swa_key_layout, get_swa_key_page_size, get_swa_key_bytes_per_token
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -2420,7 +2420,7 @@ def _build_sparse_prefill_chunk_cache(
-            swa_page_size=self.token_to_kv_pool.swa_kv_pool.page_size,
+            swa_page_size=self.token_to_kv_pool.get_swa_key_page_size(),
@@ -3316,7 +3316,7 @@ def _forward_attention(
-            swa_kv_page_size = token_to_kv_pool.swa_kv_pool.page_size
+            swa_kv_page_size = token_to_kv_pool.get_swa_key_page_size()
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -1858,6 +1858,11 @@ def get_swa_key_layout(self) -> KVLayout:
+    def get_swa_key_page_size(self) -> int:
+        if self.request_window is not None:
+            return self.request_window.page_size
+        return self.swa_kv_pool.page_size
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +2/-2; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +5/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/mem_cache/test_dsv4_compressed_pools.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42055 - [AMD][V4.1][*/N] Fuse MXFP8 activation quant into producer kernels on gfx950

- Link: https://github.com/sgl-project/sglang/pull/42055
- Status/date: merged / 2026-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py`, `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_hip.py`, `python/sglang/srt/models/deepseek_v4.py`, `test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py`; associated commits `a74f259f3815`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +185/-33, 454 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py` modified +48/-2 (50 lines); hunks: -165,12 +165,36 @@ def post_attention_norm(layer, x: torch.Tensor, coefficien...; -196,15 +220,37 @@ def wo_b_takes_fp8_grid(attn) -> bool:; symbols: post_attention_norm, _ffn_norm_emits_mxfp8, live_rows, wo_b_takes_fp8_grid, touching `post_attention_norm, _ffn_norm_emits_mxfp8, live_rows`; `python/sglang/srt/models/deepseek_v4.py` modified +9/-2 (11 lines); hunks: -462,7 +462,8 @@ def _apply_wo_a_bf16_matmul(; -528,7 +529,7 @@ def _apply_wo_a_bf16_matmul(; symbols: _apply_wo_a_bf16_matmul, _apply_gguf_grouped_wo_a, forward, touching `_apply_wo_a_bf16_matmul, _apply_gguf_grouped_wo_a, forward`; `test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py` modified +2/-1 (3 lines); hunks: -120,7 +120,8 @@ def _forward_prepare(; symbols: _forward_prepare, touching `_forward_prepare`; `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_hip.py` modified +1/-0 (1 lines); hunks: -28,6 +28,7.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py` modified +48/-2 (50 lines); hunks: -165,12 +165,36 @@ def post_attention_norm(layer, x: torch.Tensor, coefficien...; -196,15 +220,37 @@ def wo_b_takes_fp8_grid(attn) -> bool:; symbols: post_attention_norm, _ffn_norm_emits_mxfp8, live_rows, wo_b_takes_fp8_grid
  - `python/sglang/srt/models/deepseek_v4.py` modified +9/-2 (11 lines); hunks: -462,7 +462,8 @@ def _apply_wo_a_bf16_matmul(; -528,7 +529,7 @@ def _apply_wo_a_bf16_matmul(; symbols: _apply_wo_a_bf16_matmul, _apply_gguf_grouped_wo_a, forward
  - `test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py` modified +2/-1 (3 lines); hunks: -120,7 +120,8 @@ def _forward_prepare(; symbols: _forward_prepare
  - `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_hip.py` modified +1/-0 (1 lines); hunks: -28,6 +28,7
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py
@@ -165,12 +165,36 @@ def post_attention_norm(layer, x: torch.Tensor, coefficients=None) -> torch.Tens
+    if _ffn_norm_emits_mxfp8(layer, x.shape[0]):
+        x_quant, out = rmsnorm_with_sinkhorn(
+            x, norm.weight.data, norm.variance_epsilon, coefficients, emit_fp8=True
+        )
+        # the shared expert's gate_up takes it in place of its own quant (_forward_shared_experts)
+        out._hip_mxfp8_operand = x_quant
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -462,7 +462,8 @@ def _apply_wo_a_bf16_matmul(
-) -> torch.Tensor | Mxfp8SwizzledInput:
+    emit_fp8: bool = False,
+) -> torch.Tensor | Mxfp8SwizzledInput | Fp8GridActivation | Mxfp8Activation:
@@ -528,7 +529,7 @@ def _apply_wo_a_bf16_matmul(
-        y = _hip.wo_a_fp8_grid_matmul(o, wo_a, fp8_grid)
+        y = _hip.wo_a_fp8_grid_matmul(o, wo_a, fp8_grid, emit_fp8=emit_fp8)
diff -- test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py
@@ -120,7 +120,8 @@ def _forward_prepare(
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py` modified +48/-2; `python/sglang/srt/models/deepseek_v4.py` modified +9/-2; `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_hip.py` modified +1/-0
  - tests: `test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41251 - [Perf] Optimize DeepSeek V4.1 Flash Hopper paths and Blackwell prefill selection

- Link: https://github.com/sgl-project/sglang/pull/41251
- Status/date: merged / 2026-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v4.py`; associated commits `391e665dab9b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 32 files, +2096/-189, 2810 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/deepseek_v4.py` modified +113/-82 (195 lines); hunks: -2802,12 +2802,21 @@ def refresh_mhc_norm_weight_cache(self):; -3321,7 +3330,10 @@ def combine_and_norm():; symbols: refresh_mhc_norm_weight_cache, combine_and_norm, _hc_mix_stats, _hc_mix_stats_impl, touching `refresh_mhc_norm_weight_cache, combine_and_norm, _hc_mix_stats`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +113/-82 (195 lines); hunks: -2802,12 +2802,21 @@ def refresh_mhc_norm_weight_cache(self):; -3321,7 +3330,10 @@ def combine_and_norm():; symbols: refresh_mhc_norm_weight_cache, combine_and_norm, _hc_mix_stats, _hc_mix_stats_impl
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -2802,12 +2802,21 @@ def refresh_mhc_norm_weight_cache(self):
-            and get_platform().is_sm100
+            and (get_platform().is_sm100 or get_platform().is_sm90)
+            if get_platform().is_sm90:
+                from sglang.kernels.ops.layernorm.mhc import split_bf16_hc_weight
+                # The compensated BF16 projection uses ordinary tensor cores;
+                # it does not require Blackwell or DeepGEMM's prenorm kernel.
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +113/-82
- Risk and verification: The diff ships test coverage in `test/registered/kernels/benchmark/attention/bench_fp4_index_logits.py`, `test/registered/kernels/ops/activation/test_activation.py`, `test/registered/kernels/ops/attention/test_fp4_indexer.py`, `test/registered/kernels/ops/attention/test_q8kv8_sparse_prefill_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41657 - [DSv4.1] Fold q_rope_store into fused_q_norm_rope

- Link: https://github.com/sgl-project/sglang/pull/41657
- Status/date: merged / 2026-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh`, `python/sglang/srt/models/deepseek_v4.py`; associated commits `379ec90fb24a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +64/-177, 349 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/deepseek_v4.py` modified +8/-20 (28 lines); hunks: -1338,27 +1338,15 @@ def _compute_q_b(; symbols: _compute_q_b, touching `_compute_q_b`; `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh` modified +38/-19 (57 lines); hunks: -80,7 +80,7 @@ struct FusedQNormRopeParams {; -129,24 +129,26 @@ Q_KERNEL void fused_q_norm_rope(const __grid_constant__ Fu....
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +8/-20 (28 lines); hunks: -1338,27 +1338,15 @@ def _compute_q_b(; symbols: _compute_q_b
  - `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh` modified +38/-19 (57 lines); hunks: -80,7 +80,7 @@ struct FusedQNormRopeParams {; -129,24 +129,26 @@ Q_KERNEL void fused_q_norm_rope(const __grid_constant__ Fu...
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -1338,27 +1338,15 @@ def _compute_q_b(
-            if (
-                (_is_cuda or _is_gfx95_supported)
-                and q_out is not None
-                and (
-                    0 < q.shape[0] <= 8
-                    or (
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh
@@ -80,7 +80,7 @@ struct FusedQNormRopeParams {
-template <typename DType, int64_t kHeadDim, int64_t kRopeDim, typename PosT, bool kUsePDL>
+template <typename DType, int64_t kHeadDim, int64_t kRopeDim, typename PosT, bool kUsePDL, bool kApplyNorm>
@@ -129,24 +129,26 @@ Q_KERNEL void fused_q_norm_rope(const __grid_constant__ FusedQNormRopeParams par
-  float sum_of_squares = 0.0f;
+  if constexpr (kApplyNorm) {
+    float sum_of_squares = 0.0f;
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +8/-20; `python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh` modified +38/-19
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/attention/test_dsv4_q_rope_store.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41658 - [DSv4.1] Faster fp4 index-K gather and combine_topk_swa_indices

- Link: https://github.com/sgl-project/sglang/pull/41658
- Status/date: merged / 2026-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; associated commits `d75d5b33f71a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +191/-51, 347 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-0 (6 lines); hunks: -699,6 +699,12 @@ def get_index_k_fp4(; symbols: get_index_k_fp4, touching `get_index_k_fp4`.
- Code diff details:
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-0 (6 lines); hunks: -699,6 +699,12 @@ def get_index_k_fp4(; symbols: get_index_k_fp4
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -699,6 +699,12 @@ def get_index_k_fp4(
+        if buf.is_cuda:
+            from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
+                gather_fp4_index_k,
+            )
+            return gather_fp4_index_k(buf, slots, page_size=self.page_size)
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +6/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/attention/test_combine_topk_swa_indices.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41660 - [DSv4.1] Fused c1/c2 compress for eager extend, faster c2 decode

- Link: https://github.com/sgl-project/sglang/pull/41660
- Status/date: merged / 2026-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; associated commits `5916999afbc5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +642/-74, 938 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +39/-6 (45 lines); hunks: -2803,6 +2803,22 @@ def _low_ratio_compress(self, layer, x, req, pos, forward...; -2872,13 +2888,18 @@ def _low_ratio_compress_decode(self, layer, x, req, pos)...; symbols: _low_ratio_compress, _low_ratio_compress_decode, _low_ratio_compress_fused, touching `_low_ratio_compress, _low_ratio_compress_decode, _low_ratio_compress_fused`; `python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh` modified +172/-38 (210 lines); hunks: -18,6 +18,8; -34,30 +36,40 @@ struct Compress2DecodeParams {; symbols: C2Mode, touching `C2Mode`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +39/-6 (45 lines); hunks: -2803,6 +2803,22 @@ def _low_ratio_compress(self, layer, x, req, pos, forward...; -2872,13 +2888,18 @@ def _low_ratio_compress_decode(self, layer, x, req, pos)...; symbols: _low_ratio_compress, _low_ratio_compress_decode, _low_ratio_compress_fused
  - `python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh` modified +172/-38 (210 lines); hunks: -18,6 +18,8; -34,30 +36,40 @@ struct Compress2DecodeParams {; symbols: C2Mode
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -2803,6 +2803,22 @@ def _low_ratio_compress(self, layer, x, req, pos, forward_batch) -> None:
+        elif (
+            layer.compressor.use_fused_compress
+            and forward_batch.forward_mode.is_extend_without_speculative()
+            and forward_batch.extend_start_loc is not None
+        ):
+            # DP padding extends extend_seq_lens but not extend_start_loc, whose
diff -- python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh
@@ -18,6 +18,8 @@
+using device::math::fast_mod_div_u32_t;
@@ -34,30 +36,40 @@ struct Compress2DecodeParams {
-  /// Positions per request slot in the pair-state ring.
-  uint32_t ring_size;
+  fast_mod_div_u32_t ring_size;            // Positions per request slot in the pair-state ring.
-/// \brief grid = num_tokens, block = kHeadDim / kC2VecSize.
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +39/-6; `python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh` modified +172/-38
- Risk and verification: The diff ships test coverage in `test/registered/kernels/benchmark/attention/bench_dsv4_c2_decode.py`, `test/registered/kernels/ops/attention/dsv4/test_c2_prefill_ring.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41476 - [AMD] Add DeepSeek-V4.1-Flash MI35x nightly accuracy test

- Link: https://github.com/sgl-project/sglang/pull/41476
- Status/date: merged / 2026-10-04
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi35x/test_deepseek_v41_flash_eval_mi35x.py`; associated commits `f2385314b2b3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +207/-120, 377 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/registered/amd/accuracy/mi35x/test_deepseek_v41_flash_eval_mi35x.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: TestDeepseekV41FlashEvalMI35x, setUpClass, test_deepseek_v41_flash_gsm8k_accuracy, touching `TestDeepseekV41FlashEvalMI35x, setUpClass, test_deepseek_v41_flash_gsm8k_accuracy`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_deepseek_v41_flash_eval_mi35x.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: TestDeepseekV41FlashEvalMI35x, setUpClass, test_deepseek_v41_flash_gsm8k_accuracy
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_deepseek_v41_flash_eval_mi35x.py
@@ -0,0 +1,154 @@
+"""MI35x DeepSeek-V4.1-Flash GSM8K Completion Evaluation Test (4-GPU)
+Tests deepseek-ai/DeepSeek-V4.1-Flash with the GSM8K few-shot benchmark on
+MI35x.
+Server arguments follow the MI350X High-Throughput cell of the cookbook recipe
+(docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx): TP4 + EP4 with
+DSpark speculative decoding, decode CUDA graphs up to the 256-request cap, and
```

- Extracted files (not manually reviewed):
  - tests: `test/registered/amd/accuracy/mi35x/test_deepseek_v41_flash_eval_mi35x.py` added +154/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_deepseek_v41_flash_eval_mi35x.py`, `test/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42273 - [Dsv4.1] Bounded replay with sparse mla path

- Link: https://github.com/sgl-project/sglang/pull/42273
- Status/date: merged / 2026-10-05
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/deepseek_v4_backend.py`; associated commits `f70e8c68fc76`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +247/-21, 433 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +24/-7 (31 lines); hunks: -232,6 +232,8 @@ def _maybe_precompute_flashmla_sched_meta(; -2382,11 +2384,15 @@ def _build_sparse_prefill_chunk_cache(; symbols: _maybe_precompute_flashmla_sched_meta, _build_sparse_prefill_chunk_cache, _build_forward_metadata, touching `_maybe_precompute_flashmla_sched_meta, _build_sparse_prefill_chunk_cache, _build_forward_metadata`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +24/-7 (31 lines); hunks: -232,6 +232,8 @@ def _maybe_precompute_flashmla_sched_meta(; -2382,11 +2384,15 @@ def _build_sparse_prefill_chunk_cache(; symbols: _maybe_precompute_flashmla_sched_meta, _build_sparse_prefill_chunk_cache, _build_forward_metadata
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -232,6 +232,8 @@ def _maybe_precompute_flashmla_sched_meta(
+    if 4 * (5 * b + 1 + num_sm_parts * META_INTS) > 48 * 1024:
+        return
@@ -2382,11 +2384,15 @@ def _build_sparse_prefill_chunk_cache(
-        # The chunk cache gathers the W-1 positions before the chunk; under the
-        # tail those are late-layer window slots this prefill never wrote.
-        assert self.forward_metadata.late_layer_tail is None
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +24/-7
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/attention/test_flashmla_sched_meta.py`, `test/registered/kernels/ops/attention/test_q8kv8_sparse_prefill_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
