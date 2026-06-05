# sglang DeepSeek V4 Model PR Optimization History

## 2026-06-05 PR Backfill Audit

Rechecked sglang upstream `origin/main@6cfdc1858` on 2026-06-05; 12 additional PR-numbered merge(s) touched the tracked implementation files after the previous freshness cutoff (2026-05-19). These are not yet reflected in the timeline / diff-audit cards below and should be folded in on the next full regeneration.

| Merged | PR | Title | Tracked files touched |
| --- | --- | --- | --- |
| 2026-06-04 | [#27035](https://github.com/sgl-project/sglang/pull/27035) | docs: add DeepSeek V4 FP4 indexer usage | `DeepSeek-V4.mdx` |
| 2026-06-03 | [#27049](https://github.com/sgl-project/sglang/pull/27049) | docs: add DeepSeek-V4 EPLB Waterfill tips | `DeepSeek-V4.mdx` |
| 2026-06-01 | [#26968](https://github.com/sgl-project/sglang/pull/26968) | docs: update RTX PRO 6000 deployment snippet | `deepseek-v4-deployment.jsx` |
| 2026-06-01 | [#24692](https://github.com/sgl-project/sglang/pull/24692) | feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference | `deepseek-v4-deployment.jsx` |
| 2026-05-28 | [#26668](https://github.com/sgl-project/sglang/pull/26668) | [Doc] Update benchmark instruction for dsv4 | `DeepSeek-V4.mdx` |
| 2026-05-26 | [#26451](https://github.com/sgl-project/sglang/pull/26451) | [docs] Fix V4 Pro balanced recipe | `DeepSeek-V4.mdx`, `deepseek-v4-deployment.jsx` |
| 2026-05-26 | [#26413](https://github.com/sgl-project/sglang/pull/26413) | [docs] DeepSeek-V4 cookbook: note cu129 image for GB200 Pro DeepEP backend | `deepseek-v4-deployment.jsx` |
| 2026-05-23 | [#26164](https://github.com/sgl-project/sglang/pull/26164) | [docs] DeepSeek-V4 cookbook: balanced MegaMoE cap, H200 Pro FP4 mem-frac, nsa-* compat, PD-disagg fixes | `deepseek-v4-deployment.jsx` |
| 2026-05-22 | [#26057](https://github.com/sgl-project/sglang/pull/26057) | [docs] DeepSeek-V4 cookbook: split Quantization axis, add H100 SGLang FP8 | `DeepSeek-V4.mdx`, `deepseek-v4-deployment.jsx` |
| 2026-05-21 | [#26004](https://github.com/sgl-project/sglang/pull/26004) | Default MegaMoE to W4A8 for Max-Throughput recipe | `deepseek-v4-deployment.jsx` |
| 2026-05-21 | [#25923](https://github.com/sgl-project/sglang/pull/25923) | [Docs] DeepSeek-V4: switch H200 FP4 Pro to flashinfer_mxfp4, Flash Balanced too | `DeepSeek-V4.mdx`, `deepseek-v4-deployment.jsx` |
| 2026-05-20 | [#25821](https://github.com/sgl-project/sglang/pull/25821) | [Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename | `deepseek-v4-deployment.jsx` |


## 2026-05-19 PR Backfill Audit

Rechecked sglang upstream `origin/main@78cb38ed5` and the GitHub Pull Request files API; this pass adds timeline entries and per-PR diff audit cards for `#24367`, `#24691`, `#24704`, `#24775`, `#24793`, `#24816`, `#24890`, `#24897`, `#24925`, `#24933`, `#24949`, `#24986`, `#25001`, `#25052`, `#25152`, `#25243`, `#25282`, `#25369`, `#25410`, `#25412`, `#25419`, `#25477`, `#25506`, `#25569`, `#25733`.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `.github/workflows/release-docker-deepseek-v4.yml` | [#23728](https://github.com/sgl-project/sglang/pull/23728), [#23730](https://github.com/sgl-project/sglang/pull/23730), [#23778](https://github.com/sgl-project/sglang/pull/23778) |
| `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` | [#23605](https://github.com/sgl-project/sglang/pull/23605), [#23622](https://github.com/sgl-project/sglang/pull/23622), [#23628](https://github.com/sgl-project/sglang/pull/23628), [#23684](https://github.com/sgl-project/sglang/pull/23684), [#23689](https://github.com/sgl-project/sglang/pull/23689), [#23691](https://github.com/sgl-project/sglang/pull/23691), [#23697](https://github.com/sgl-project/sglang/pull/23697), [#23725](https://github.com/sgl-project/sglang/pull/23725), [#23980](https://github.com/sgl-project/sglang/pull/23980), [#24035](https://github.com/sgl-project/sglang/pull/24035) |
| `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` | [#23605](https://github.com/sgl-project/sglang/pull/23605), [#23617](https://github.com/sgl-project/sglang/pull/23617), [#23622](https://github.com/sgl-project/sglang/pull/23622), [#23634](https://github.com/sgl-project/sglang/pull/23634), [#23689](https://github.com/sgl-project/sglang/pull/23689), [#23690](https://github.com/sgl-project/sglang/pull/23690), [#23691](https://github.com/sgl-project/sglang/pull/23691), [#23697](https://github.com/sgl-project/sglang/pull/23697), [#23698](https://github.com/sgl-project/sglang/pull/23698), [#23715](https://github.com/sgl-project/sglang/pull/23715), [#23725](https://github.com/sgl-project/sglang/pull/23725), [#23737](https://github.com/sgl-project/sglang/pull/23737), ... (17 total) |

## PR Coverage Summary

- Git-traced PRs: 23
- Extra PRs preserved from existing docs: 30
- Total PRs in this document: 53
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#23605](https://github.com/sgl-project/sglang/pull/23605) | merged | Add DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
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
| 2026-05-05 | [#24367](https://github.com/sgl-project/sglang/pull/24367) | merged | [docs] Update B300 Pro cookbook with accuracy-verified serving configs | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-05-09 | [#24793](https://github.com/sgl-project/sglang/pull/24793) | merged | [DSV4] Cherry pick missing commits from deepseek_v4 branch and enhance tests | `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/model_loader/weight_utils.py` |
| 2026-05-10 | [#24775](https://github.com/sgl-project/sglang/pull/24775) | merged | Optimize MHC pipeline: DeepGemm, fused norm, fused hc_head | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/mhc.py`, `python/sglang/srt/layers/mhc_head.py` |
| 2026-05-12 | [#24949](https://github.com/sgl-project/sglang/pull/24949) | merged | Deepseek-v4-Pro share expert tp1 | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/environ.py` |
| 2026-05-13 | [#24816](https://github.com/sgl-project/sglang/pull/24816) | merged | Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4 | `python/sglang/test/bench_mxfp4_sm90_kernels.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py` |
| 2026-05-13 | [#24890](https://github.com/sgl-project/sglang/pull/24890) | merged | Port KV Compression V2 from deepseek_v4_dev | `python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh` |
| 2026-05-13 | [#24897](https://github.com/sgl-project/sglang/pull/24897) | merged | Port fused SiLU+clamp+FP8 quant from DSV4 dev branch | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-05-13 | [#24986](https://github.com/sgl-project/sglang/pull/24986) | merged | [rebase]Deepseek_v4 support w4(mxfp4)a16 on hopper | `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/mxfp4.py` |
| 2026-05-13 | [#25001](https://github.com/sgl-project/sglang/pull/25001) | merged | [LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` |
| 2026-05-13 | [#25152](https://github.com/sgl-project/sglang/pull/25152) | merged | docs: prepend SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 for H200 FP8 Flash max-throughput | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-05-14 | [#24925](https://github.com/sgl-project/sglang/pull/24925) | merged | [attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell) | `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py` |
| 2026-05-14 | [#25052](https://github.com/sgl-project/sglang/pull/25052) | merged | DeepSeek V4 w4a4 MegaMoE | `python/sglang/srt/layers/moe/mega_moe.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py`, `python/sglang/srt/environ.py` |
| 2026-05-14 | [#25243](https://github.com/sgl-project/sglang/pull/25243) | merged | [Docs] update dsv4 cookbook with H100 deployment commands | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-15 | [#24691](https://github.com/sgl-project/sglang/pull/24691) | merged | [UnifiedTree]: Support HiCache For DeepSeek_V4 | `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`, `python/sglang/srt/mem_cache/unified_radix_cache.py` |
| 2026-05-15 | [#25369](https://github.com/sgl-project/sglang/pull/25369) | merged | Add hicache feature in dsv4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-16 | [#24704](https://github.com/sgl-project/sglang/pull/24704) | merged | feat: add Pipeline Parallelism (PP) and PD support for DeepSeek-V4 | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/disaggregation/common/conn.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 2026-05-16 | [#25410](https://github.com/sgl-project/sglang/pull/25410) | merged | [Docs] Update DeepSeek V4 cookbook to use the latest docker image | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-16 | [#25412](https://github.com/sgl-project/sglang/pull/25412) | merged | [Doc] DSV4 cookbook: clean up env vars, add MegaMoE toggle, unify docker image | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-05-16 | [#25419](https://github.com/sgl-project/sglang/pull/25419) | merged | Port SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN from deepseek_v4_dev | `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/environ.py` |
| 2026-05-16 | [#25477](https://github.com/sgl-project/sglang/pull/25477) | merged | [BugFix]: Fix DeepSeek V4 HiCache layer count logic | `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py` |
| 2026-05-17 | [#25506](https://github.com/sgl-project/sglang/pull/25506) | merged | [Doc] Fix several places for dpsk v4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-05-18 | [#24933](https://github.com/sgl-project/sglang/pull/24933) | merged | Amd/deepseek v4 rebase main 0509 | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsv4/compress_hip.py` |
| 2026-05-18 | [#25569](https://github.com/sgl-project/sglang/pull/25569) | merged | Add DeepSeekV4 fused MoE Triton autotune support | `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `benchmark/kernels/fused_moe_triton/common_utils.py` |
| 2026-05-19 | [#25282](https://github.com/sgl-project/sglang/pull/25282) | merged | [UnifiedTree] Support deepseek v4 host pool layout | `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/test/kl_multiturn_utils.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` |
| 2026-05-19 | [#25733](https://github.com/sgl-project/sglang/pull/25733) | merged | [Bug] Fix V4-Pro NaN on Blackwell by converting fp8_einsum input scale to ue8m0 | `python/sglang/srt/models/deepseek_v4.py` |

## Per-PR Diff Audit Cards

### PR #23605 - Add DeepSeek V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23605
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `492883c8ca66`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +1024/-1, 1041 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DeepSeek V4 cookbook"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "Add DeepSeek V4 cookbook"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0 (569 lines); hunks: -0,0 +1,569; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0 (453 lines); hunks: -0,0 +1,453.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0 (569 lines); hunks: -0,0 +1,569
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0 (453 lines); hunks: -0,0 +1,453
- Key code excerpts:

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
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/docs.json`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23617 - Further update Deepseek V4 docs

- Link: https://github.com/sgl-project/sglang/pull/23617
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `734e1e2965cb`; preserved from an explicit existing history/skill citation
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `3a620cb761ff`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +32/-9, 73 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Again update DeepSeek V4 cookbook"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "Again update DeepSeek V4 cookbook"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9 (28 lines); hunks: -42,11 +42,11 @@ export const DeepSeekV4Deployment = () => {; -161,7 +161,16 @@ export const DeepSeekV4Deployment = () => {; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../.....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9 (28 lines); hunks: -42,11 +42,11 @@ export const DeepSeekV4Deployment = () => {; -161,7 +161,16 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../....
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23634 - Update pro fp8 checkpoint in DeepSeek V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23634
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `92bb5c6bbee9`; preserved from an explicit existing history/skill citation
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `d2c61acf2597`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +22/-1, 59 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0 (14 lines); hunks: -164,14 +164,26 @@ export const DeepSeekV4Deployment = () => {; -387,6 +399,7 @@ export const DeepSeekV4Deployment = () => {; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1 (9 lines); hunks: -145,7 +145,14 @@ The generator currently picks values on the **conservative*....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0 (14 lines); hunks: -164,14 +164,26 @@ export const DeepSeekV4Deployment = () => {; -387,6 +399,7 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1 (9 lines); hunks: -145,7 +145,14 @@ The generator currently picks values on the **conservative*...
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23691 - docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes

- Link: https://github.com/sgl-project/sglang/pull/23691
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `8a395994edcf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +56/-5, 113 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes"; model line: DeepSeek V4; category: bug fix; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +49/-5 (54 lines); hunks: -176,6 +176,10 @@ export const DeepSeekV4Deployment = () => {; -372,7 +376,17 @@ export const DeepSeekV4Deployment = () => {; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0 (7 lines); hunks: -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpo....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +49/-5 (54 lines); hunks: -176,6 +176,10 @@ export const DeepSeekV4Deployment = () => {; -372,7 +376,17 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0 (7 lines); hunks: -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpo...
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +49/-5; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23690 - Small udpate gb300 recipe for deepseek v4

- Link: https://github.com/sgl-project/sglang/pull/23690
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `69485a176c87`; preserved from an explicit existing history/skill citation
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `0d224e505333`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-2, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "update: b300 container for dsv4"; model line: DeepSeek V4; category: model implementation change; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "update: b300 container for dsv4"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +7/-2 (9 lines); hunks: -26,6 +26,7 @@ export const DeepSeekV4Deployment = () => {; -222,7 +223,9 @@ export const DeepSeekV4Deployment = () => {; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../.....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +7/-2 (9 lines); hunks: -26,6 +26,7 @@ export const DeepSeekV4Deployment = () => {; -222,7 +223,9 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../....
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +7/-2; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23698 - docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9

- Link: https://github.com/sgl-project/sglang/pull/23698
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `880599cd430f`; preserved from an explicit existing history/skill citation
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `d4c16656262b`
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
- Trace source: `git log --name-only -- <model-files>` found it through `.github/workflows/release-docker-deepseek-v4.yml`; associated commits `0c826374a85a`
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
- Trace source: `git log --name-only -- <model-files>` found it through `.github/workflows/release-docker-deepseek-v4.yml`; associated commits `921e14dcac53`
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `049f1bf6fb42`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +58/-8, 195 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): add GB200 platform to cookbook recipe"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(DeepSeek-V4): add GB200 platform to cookbook recipe"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +52/-6 (58 lines); hunks: -4,6 +4,7 @@ export const DeepSeekV4Deployment = () => {; -27,6 +28,7 @@ export const DeepSeekV4Deployment = () => {; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-2 (8 lines); hunks: -29,13 +29,13 @@ tag: NEW; -88,6 +88,10 @@ Please refer to the [official SGLang installation guide](../.....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +52/-6 (58 lines); hunks: -4,6 +4,7 @@ export const DeepSeekV4Deployment = () => {; -27,6 +28,7 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-2 (8 lines); hunks: -29,13 +29,13 @@ tag: NEW; -88,6 +88,10 @@ Please refer to the [official SGLang installation guide](../....
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +52/-6; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-2
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23742 - docs(DeepSeek-V4): add h200|big verified recipes + tune H200 Pro parameters

- Link: https://github.com/sgl-project/sglang/pull/23742
- Status/date: merged / 2026-04-26
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `3cfd1561df78`; preserved from an explicit existing history/skill citation
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `3d95ca7546fb`; preserved from an explicit existing history/skill citation
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
- Trace source: `git log --name-only -- <model-files>` found it through `.github/workflows/release-docker-deepseek-v4.yml`; associated commits `977830e91e41`
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `c2ec64f243d4`; preserved from an explicit existing history/skill citation
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `3177fa795154`; preserved from an explicit existing history/skill citation
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `4e1ef6b3cf9b`
- Diff scope read: GitHub Pull Request files API returned 1 files, +32/-0, 39 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "[Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `4e885baa9bf1`
- Diff scope read: GitHub Pull Request files API returned 2 files, +84/-8, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-3 (82 lines); hunks: -31,6 +31,7 @@ export const DeepSeekV4Deployment = () => {; -70,7 +71,19 @@ export const DeepSeekV4Deployment = () => {; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +5/-5 (10 lines); hunks: -1,7 +1,7; -35,7 +35,7 @@ tag: NEW.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-3 (82 lines); hunks: -31,6 +31,7 @@ export const DeepSeekV4Deployment = () => {; -70,7 +71,19 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +5/-5 (10 lines); hunks: -1,7 +1,7; -35,7 +35,7 @@ tag: NEW
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-3; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +5/-5
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #24035 - [minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4

- Link: https://github.com/sgl-project/sglang/pull/24035
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `b3ead32d3ca2`
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-3, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
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

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.

### PR #24367 - [docs] Update B300 Pro cookbook with accuracy-verified serving configs

- Link: https://github.com/sgl-project/sglang/pull/24367
- Status/date: merged / 2026-05-05
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `6279aee7163a`.
- Diff scope read: GitHub Pull Request files API returned 1 files, +108/-11, 195 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] Update B300 Pro cookbook with accuracy-verified serving configs"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "[docs] Update B300 Pro cookbook with accuracy-verified serving configs" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +108/-11 (119 lines); hunks: -351,13 +351,41  @@ export const DeepSeekV4Deployment = () => {; -367,6 +395,26  @@ export const DeepSeekV4Deployment = () => {; symbols: DeepSeekV4Deployment, touching `DeepSeekV4Deployment`.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +108/-11 (119 lines); hunks: -351,13 +351,41  @@ export const DeepSeekV4Deployment = () => {; -367,6 +395,26  @@ export const DeepSeekV4Deployment = () => {; symbols: DeepSeekV4Deployment, touching `DeepSeekV4Deployment`
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
+          "SGLANG_OPT_USE_JIT_INDEXER_METADATA=1",
+          "SGLANG_OPT_USE_TOPK_V2=1",
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +108/-11
- Risk and verification: This PR mainly changes docs/tests/CI ``docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx``; verify commands, CI selectors, and model repo names still map to the current implementation.

### PR #24793 - [DSV4] Cherry pick missing commits from deepseek_v4 branch and enhance tests

- Link: https://github.com/sgl-project/sglang/pull/24793
- Status/date: merged / 2026-05-09
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `ef5e9f8abab1`.
- Diff scope read: GitHub Pull Request files API returned 15 files, +481/-87, 873 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Cherry pick missing commits from deepseek_v4 branch and enhance tests"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/model_loader/weight_utils.py`; technical summary: Covers "[DSV4] Cherry pick missing commits from deepseek_v4 branch and enhance tests" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +26/-10 (36 lines); hunks: -81,8 +81,13  @@ def __init__(self):; -92,6 +97,20  @@ def has_tool_call(self, text: str) -> bool:; symbols: __init__, has_tool_call, detect_and_parse, parse_streaming_increment, touching `__init__, has_tool_call, detect_and_parse`；`test/registered/unit/function_call/test_function_call_parser.py` modified +111/-1 (112 lines); hunks: -31,7 +31,7  @@ from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector; -1686,6 +1686,26  @@ def test_get_model_structural_tag(self):; symbols: test_get_model_structural_tag, touching `test_get_model_structural_tag`；`python/sglang/srt/model_loader/weight_utils.py` modified +33/-3 (36 lines); hunks: -875,11 +875,30  @@ def _run_prefetch() -> None:; -907,6 +926,8  @@ def safetensors_weights_iterator(; symbols: _run_prefetch, safetensors_weights_iterator, multi_thread_safetensors_weights_iterator, _load_file, touching `_run_prefetch, safetensors_weights_iterator, multi_thread_safetensors_weights_iterator`；`python/sglang/srt/entrypoints/openai/protocol.py` modified +5/-2 (7 lines); hunks: -633,13 +633,16  @@ class ChatCompletionRequest(BaseModel):; symbols: ChatCompletionRequest, touching `ChatCompletionRequest`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +26/-10 (36 lines); hunks: -81,8 +81,13  @@ def __init__(self):; -92,6 +97,20  @@ def has_tool_call(self, text: str) -> bool:; symbols: __init__, has_tool_call, detect_and_parse, parse_streaming_increment, touching `__init__, has_tool_call, detect_and_parse`
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +111/-1 (112 lines); hunks: -31,7 +31,7  @@ from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector; -1686,6 +1686,26  @@ def test_get_model_structural_tag(self):; symbols: test_get_model_structural_tag, touching `test_get_model_structural_tag`
  - `python/sglang/srt/model_loader/weight_utils.py` modified +33/-3 (36 lines); hunks: -875,11 +875,30  @@ def _run_prefetch() -> None:; -907,6 +926,8  @@ def safetensors_weights_iterator(; symbols: _run_prefetch, safetensors_weights_iterator, multi_thread_safetensors_weights_iterator, _load_file, touching `_run_prefetch, safetensors_weights_iterator, multi_thread_safetensors_weights_iterator`
  - `python/sglang/srt/entrypoints/openai/protocol.py` modified +5/-2 (7 lines); hunks: -633,13 +633,16  @@ class ChatCompletionRequest(BaseModel):; symbols: ChatCompletionRequest, touching `ChatCompletionRequest`
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: -788,6 +788,7  @@ class ServerArgs:; -6634,6 +6635,11  @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args, touching `ServerArgs, add_cli_args`
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -81,8 +81,13 @@ def __init__(self):
+        # Long-form `<｜DSML｜invoke name="x">...</｜DSML｜invoke>` and the
+        # self-closing `<｜DSML｜invoke name="x"/>` shape V4 emits for zero-arg
+        # tools. The `end` group is empty when the closer hasn't streamed in.
-            r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)(</｜DSML｜invoke>|$)'
+            r'<｜DSML｜invoke\s+name="(?P<name>[^"]+)"\s*'
+            r"(?:(?P<self_close>/>)"
+            r"|>(?P<body>.*?)(?P<end>(?:</｜DSML｜invoke>|$)))"
@@ -92,6 +97,20 @@ def has_tool_call(self, text: str) -> bool:
diff -- test/registered/unit/function_call/test_function_call_parser.py
@@ -31,7 +31,7 @@
-register_cpu_ci(15, "stage-a-test-cpu")
+register_cpu_ci(est_time=15, suite="stage-a-test-cpu")
@@ -1686,6 +1686,26 @@ def test_get_model_structural_tag(self):
+    def test_self_closing_zero_arg_invoke(self):
+        """V32 inherits the same regex; verify self-closing parses to empty
+        params here too (V32 model rarely emits this shape, but the parser
+        must agree with V4 since V4 inherits from V32)."""
+        submit_tool = Tool(
diff -- python/sglang/srt/model_loader/weight_utils.py
@@ -875,11 +875,30 @@ def _run_prefetch() -> None:
+def _drop_file_cache_after_load(path: str) -> None:
+    """Release of checkpoint pages after weights have been copied out. Used to avoid CPU OOM in RL."""
+    posix_fadvise = getattr(os, "posix_fadvise", None)
+    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
+    if posix_fadvise is None or dontneed is None:
+        return
+
+    fd = None
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +26/-10; `python/sglang/srt/model_loader/weight_utils.py` modified +33/-3; `python/sglang/srt/entrypoints/openai/protocol.py` modified +5/-2; `python/sglang/srt/server_args.py` modified +6/-0
  - tests: `test/registered/unit/function_call/test_function_call_parser.py` modified +111/-1; `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py` renamed +65/-15; `test/registered/dsv4/test_deepseek_v4_flash_fp8_h200.py` renamed +28/-16
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/model_loader/weight_utils.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #24775 - Optimize MHC pipeline: DeepGemm, fused norm, fused hc_head

- Link: https://github.com/sgl-project/sglang/pull/24775
- Status/date: merged / 2026-05-10
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `2f06867128e8`.
- Diff scope read: GitHub Pull Request files API returned 4 files, +512/-73, 699 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimize MHC pipeline: DeepGemm, fused norm, fused hc_head"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/mhc.py`, `python/sglang/srt/layers/mhc_head.py`; technical summary: Covers "Optimize MHC pipeline: DeepGemm, fused norm, fused hc_head" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +40/-9 (49 lines); hunks: -653,7 +653,11  @@ def hc_pre(; -671,11 +675,16  @@ def hc_pre_torch_impl(x, hc_fn):; symbols: hc_pre, hc_pre_torch_impl, forward, hc_head, touching `hc_pre, hc_pre_torch_impl, forward`；`python/sglang/srt/layers/mhc.py` modified +319/-64 (383 lines); hunks: -7,6 +7,7  @@ import torch; -138,12 +139,15  @@ def mhc_pre_big_fuse_tilelang(; symbols: mhc_pre_big_fuse_tilelang, mhc_pre_gemm_sqrsum_splitk_stage_1, mhc_pre, touching `mhc_pre_big_fuse_tilelang, mhc_pre_gemm_sqrsum_splitk_stage_1, mhc_pre`；`python/sglang/srt/layers/mhc_head.py` added +151/-0 (151 lines); hunks: -0,0 +1,151  @@ +"""Fused triton kernel for the DSV4 hc_head LM-head mixer.；`scripts/ci/utils/slash_command_handler.py` modified +2/-0 (2 lines); hunks: -424,6 +424,8  @@ def handle_rerun_stage(; symbols: handle_rerun_stage, touching `handle_rerun_stage`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +40/-9 (49 lines); hunks: -653,7 +653,11  @@ def hc_pre(; -671,11 +675,16  @@ def hc_pre_torch_impl(x, hc_fn):; symbols: hc_pre, hc_pre_torch_impl, forward, hc_head, touching `hc_pre, hc_pre_torch_impl, forward`
  - `python/sglang/srt/layers/mhc.py` modified +319/-64 (383 lines); hunks: -7,6 +7,7  @@ import torch; -138,12 +139,15  @@ def mhc_pre_big_fuse_tilelang(; symbols: mhc_pre_big_fuse_tilelang, mhc_pre_gemm_sqrsum_splitk_stage_1, mhc_pre, touching `mhc_pre_big_fuse_tilelang, mhc_pre_gemm_sqrsum_splitk_stage_1, mhc_pre`
  - `python/sglang/srt/layers/mhc_head.py` added +151/-0 (151 lines); hunks: -0,0 +1,151  @@ +"""Fused triton kernel for the DSV4 hc_head LM-head mixer.
  - `scripts/ci/utils/slash_command_handler.py` modified +2/-0 (2 lines); hunks: -424,6 +424,8  @@ def handle_rerun_stage(; symbols: handle_rerun_stage, touching `handle_rerun_stage`
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -653,7 +653,11 @@ def hc_pre(
+        norm: Optional[nn.Module] = None,
+        """If *norm* is given and the TileLang path is active, the returned
+        hidden_states are already post-norm (the norm is fused into the kernel)."""
+
@@ -671,11 +675,16 @@ def hc_pre_torch_impl(x, hc_fn):
-            return y, post, comb
+            return y, post, comb, False
+            norm_kwargs = {}
diff -- python/sglang/srt/layers/mhc.py
@@ -7,6 +7,7 @@
+from sglang.srt.environ import envs
@@ -138,12 +139,15 @@ def mhc_pre_big_fuse_tilelang(
+    gemm_last_dim: int = -1,
+    if gemm_last_dim < 0:
+        gemm_last_dim = hc_mult3
-    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]
+    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]
@@ -438,6 +442,186 @@ def mhc_pre_gemm_sqrsum_splitk_stage_1(
diff -- python/sglang/srt/layers/mhc_head.py
@@ -0,0 +1,151 @@
+"""Fused triton kernel for the DSV4 hc_head LM-head mixer.
+
+Reference torch implementation (deepseek_v4.py DeepseekV4Model.hc_head):
+
+    shape, dtype = x.size(), x.dtype
+    x = x.flatten(1).float()
+    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
+    mixes = F.linear(x, hc_fn) * rsqrt
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +40/-9; `python/sglang/srt/layers/mhc.py` modified +319/-64; `python/sglang/srt/layers/mhc_head.py` added +151/-0
  - ci: `scripts/ci/utils/slash_command_handler.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/mhc.py`, `python/sglang/srt/layers/mhc_head.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #24949 - Deepseek-v4-Pro share expert tp1

- Link: https://github.com/sgl-project/sglang/pull/24949
- Status/date: merged / 2026-05-12
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `ca3bc05fea1a`.
- Diff scope read: GitHub Pull Request files API returned 3 files, +31/-17, 112 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deepseek-v4-Pro share expert tp1"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/environ.py`; technical summary: Covers "Deepseek-v4-Pro share expert tp1" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +26/-14 (40 lines); hunks: -534,6 +534,7  @@ def __init__(; -543,7 +544,19  @@ def __init__(; symbols: __init__, forward_normal_dual_stream, _post_combine_hook, touching `__init__, forward_normal_dual_stream, _post_combine_hook`；`python/sglang/srt/model_executor/model_runner.py` modified +4/-2 (6 lines); hunks: -1155,8 +1155,10  @@ def check_quantized_moe_compatibility(self):; symbols: check_quantized_moe_compatibility, touching `check_quantized_moe_compatibility`；`python/sglang/srt/environ.py` modified +1/-1 (2 lines); hunks: -611,7 +611,7  @@ class Envs:; symbols: Envs, touching `Envs`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +26/-14 (40 lines); hunks: -534,6 +534,7  @@ def __init__(; -543,7 +544,19  @@ def __init__(; symbols: __init__, forward_normal_dual_stream, _post_combine_hook, touching `__init__, forward_normal_dual_stream, _post_combine_hook`
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-2 (6 lines); hunks: -1155,8 +1155,10  @@ def check_quantized_moe_compatibility(self):; symbols: check_quantized_moe_compatibility, touching `check_quantized_moe_compatibility`
  - `python/sglang/srt/environ.py` modified +1/-1 (2 lines); hunks: -611,7 +611,7  @@ class Envs:; symbols: Envs, touching `Envs`
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
+            _shared_expert_use_tp1 = (
+                get_moe_a2a_backend().is_deepep()
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
-
+    SGLANG_SHARED_EXPERT_TP1 = EnvBool(False)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +26/-14; `python/sglang/srt/model_executor/model_runner.py` modified +4/-2; `python/sglang/srt/environ.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/environ.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #24816 - Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4

- Link: https://github.com/sgl-project/sglang/pull/24816
- Status/date: merged / 2026-05-13
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `28758d37dd5c`.
- Diff scope read: GitHub Pull Request files API returned 8 files, +1542/-3, 1649 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/test/bench_mxfp4_sm90_kernels.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py`; technical summary: Covers "Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/test/bench_mxfp4_sm90_kernels.py` added +366/-0 (366 lines); hunks: -0,0 +1,366  @@ +"""Benchmark MXFP4 MoE kernels on H100/H200: SGLang Marlin vs FlashInfer cutlass.；`python/sglang/srt/layers/quantization/mxfp4.py` modified +269/-1 (270 lines); hunks: -16,12 +16,18  @@ from __future__ import annotations; -62,7 +68,27  @@ nvfp4_block_scale_interleave,; symbols: __init__, create_weights, swap_every_two_rows, axis, touching `__init__, create_weights, swap_every_two_rows`；`python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py` added +263/-0 (263 lines); hunks: -0,0 +1,263  @@ +"""DeepSeek-V4 MXFP4 expert backend backed by FlashInfer's SM90 cutlass；`python/sglang/srt/layers/moe/topk.py` modified +12/-0 (12 lines); hunks: -243,6 +243,18  @@ class BypassedTopKOutput(NamedTuple):; symbols: BypassedTopKOutput, touching `BypassedTopKOutput`.
- Code diff details:
  - `python/sglang/test/bench_mxfp4_sm90_kernels.py` added +366/-0 (366 lines); hunks: -0,0 +1,366  @@ +"""Benchmark MXFP4 MoE kernels on H100/H200: SGLang Marlin vs FlashInfer cutlass.
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +269/-1 (270 lines); hunks: -16,12 +16,18  @@ from __future__ import annotations; -62,7 +68,27  @@ nvfp4_block_scale_interleave,; symbols: __init__, create_weights, swap_every_two_rows, axis, touching `__init__, create_weights, swap_every_two_rows`
  - `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py` added +263/-0 (263 lines); hunks: -0,0 +1,263  @@ +"""DeepSeek-V4 MXFP4 expert backend backed by FlashInfer's SM90 cutlass
  - `python/sglang/srt/layers/moe/topk.py` modified +12/-0 (12 lines); hunks: -243,6 +243,18  @@ class BypassedTopKOutput(NamedTuple):; symbols: BypassedTopKOutput, touching `BypassedTopKOutput`
  - `python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py` modified +9/-1 (10 lines); hunks: -445,12 +445,20  @@ def maybe_fuse_routed_scale_and_shared_add(; symbols: maybe_fuse_routed_scale_and_shared_add, touching `maybe_fuse_routed_scale_and_shared_add`
- Key code excerpts:

```diff
diff -- python/sglang/test/bench_mxfp4_sm90_kernels.py
@@ -0,0 +1,366 @@
+"""Benchmark MXFP4 MoE kernels on H100/H200: SGLang Marlin vs FlashInfer cutlass.
+
+Compares per-call latency of:
+
+  * Marlin path  :  ``fused_marlin_moe(...)`` after Marlin weight repack
+  * FlashInfer   :  ``cutlass_fused_moe(use_w4_group_scaling=True, ...)``
+                    (PR #3084's SM90 mixed-input path)
+
diff -- python/sglang/srt/layers/quantization/mxfp4.py
@@ -16,12 +16,18 @@
+import os
+# Silence the TRT-LLM cutlass autotune trace embedded inside FlashInfer's
+# cutlass_fused_moe. Its C++ logger reads TLLM_LOG_LEVEL on first kernel launch;
+# setdefault preserves any explicit user override.
+os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")
+
@@ -62,7 +68,27 @@
-    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache
diff -- python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py
@@ -0,0 +1,263 @@
+"""DeepSeek-V4 MXFP4 expert backend backed by FlashInfer's SM90 cutlass
+mixed-input MoE GEMM (FlashInfer PR #3084).
+
+Sibling of :class:`Mxfp4MarlinMoEMethod` and :class:`Mxfp4FlashinferTrtllmMoEMethod`.
+Wired into :func:`Fp8MoEConfig.get_quant_method` when
+``is_fp4_experts=True`` and ``--moe-runner-backend flashinfer_mxfp4`` is
+selected on a Hopper (SM90) device. SM100 still routes to
+:class:`Mxfp4FlashinferTrtllmMoEMethod` (trtllm-gen).
```

- Reviewed files:
  - runtime: `python/sglang/test/bench_mxfp4_sm90_kernels.py` added +366/-0; `python/sglang/srt/layers/quantization/mxfp4.py` modified +269/-1; `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py` added +263/-0; `python/sglang/srt/layers/moe/topk.py` modified +12/-0
  - tests: `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py` added +544/-0; `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py` modified +70/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/test/bench_mxfp4_sm90_kernels.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #24890 - Port KV Compression V2 from deepseek_v4_dev

- Link: https://github.com/sgl-project/sglang/pull/24890
- Status/date: merged / 2026-05-13
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `e2290b155aa0`.
- Diff scope read: GitHub Pull Request files API returned 23 files, +5201/-438, 6145 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Port KV Compression V2 from deepseek_v4_dev"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh`; technical summary: Covers "Port KV Compression V2 from deepseek_v4_dev" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh` added +875/-0 (875 lines); hunks: -0,0 +1,875  @@ +#include <sgl_kernel/ffi.h>；`python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh` added +827/-0 (827 lines); hunks: -0,0 +1,827  @@ +#include <sgl_kernel/ffi.h>；`python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh` added +629/-0 (629 lines); hunks: -0,0 +1,629  @@ +#include <sgl_kernel/tensor.h>；`python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +208/-303 (511 lines); hunks: -1,3 +1,16  @@ +/**; -8,7 +21,7  @@ #include <sgl_kernel/vec.cuh>.
- Code diff details:
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh` added +875/-0 (875 lines); hunks: -0,0 +1,875  @@ +#include <sgl_kernel/ffi.h>
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh` added +827/-0 (827 lines); hunks: -0,0 +1,827  @@ +#include <sgl_kernel/ffi.h>
  - `python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh` added +629/-0 (629 lines); hunks: -0,0 +1,629  @@ +#include <sgl_kernel/tensor.h>
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +208/-303 (511 lines); hunks: -1,3 +1,16  @@ +/**; -8,7 +21,7  @@ #include <sgl_kernel/vec.cuh>
  - `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh` added +419/-0 (419 lines); hunks: -0,0 +1,419  @@ +#include <sgl_kernel/tensor.h>
- Key code excerpts:

```diff
diff -- python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh
@@ -0,0 +1,875 @@
+#include <sgl_kernel/ffi.h>
+#include <sgl_kernel/tensor.h>
+#include <sgl_kernel/utils.h>
+
+#include <sgl_kernel/runtime.cuh>
+#include <sgl_kernel/tile.cuh>
+#include <sgl_kernel/type.cuh>
+#include <sgl_kernel/utils.cuh>
diff -- python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh
@@ -0,0 +1,827 @@
+#include <sgl_kernel/ffi.h>
+#include <sgl_kernel/tensor.h>
+#include <sgl_kernel/utils.h>
+
+#include <sgl_kernel/utils.cuh>
+#include <sgl_kernel/warp.cuh>
+
+#include <sgl_kernel/deepseek_v4/compress_v2.cuh>
diff -- python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh
@@ -0,0 +1,629 @@
+#include <sgl_kernel/tensor.h>
+#include <sgl_kernel/utils.h>
+
+#include <sgl_kernel/math.cuh>
+#include <sgl_kernel/tile.cuh>
+#include <sgl_kernel/type.cuh>
+#include <sgl_kernel/utils.cuh>
+#include <sgl_kernel/vec.cuh>
```

- Reviewed files:
  - runtime: `python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh` added +875/-0; `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh` added +827/-0; `python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh` added +629/-0; `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` modified +208/-303
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #24897 - Port fused SiLU+clamp+FP8 quant from DSV4 dev branch

- Link: https://github.com/sgl-project/sglang/pull/24897
- Status/date: merged / 2026-05-13
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `d0913fca8dc5`.
- Diff scope read: GitHub Pull Request files API returned 1 files, +51/-6, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Port fused SiLU+clamp+FP8 quant from DSV4 dev branch"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "Port fused SiLU+clamp+FP8 quant from DSV4 dev branch" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunks: -27,6 +27,10  @@ from torch import nn; -107,6 +111,9  @@ ); symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunks: -27,6 +27,10  @@ from torch import nn; -107,6 +111,9  @@ ); symbols: forward, touching `forward`
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
+    create_per_token_group_quant_fp8_output_scale,
+)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +51/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #24986 - [rebase]Deepseek_v4 support w4(mxfp4)a16 on hopper

- Link: https://github.com/sgl-project/sglang/pull/24986
- Status/date: merged / 2026-05-13
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `37f18438c593`.
- Diff scope read: GitHub Pull Request files API returned 7 files, +146/-36, 295 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[rebase]Deepseek_v4 support w4(mxfp4)a16 on hopper"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/mxfp4.py`; technical summary: Covers "[rebase]Deepseek_v4 support w4(mxfp4)a16 on hopper" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py` modified +57/-12 (69 lines); hunks: -8,7 +8,7  @@ from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo; -38,17 +38,62  @@ def create_weights(; symbols: create_weights, apply, touching `create_weights, apply`；`python/sglang/srt/layers/quantization/marlin_utils_fp4.py` modified +32/-16 (48 lines); hunks: -52,22 +52,38  @@ def _normalize_scale_tensor(; -129,19 +145,19  @@ def _permute_bias(bias: torch.Tensor | None) -> torch.Tensor | None:; symbols: _normalize_scale_tensor, _permute_bias, touching `_normalize_scale_tensor, _permute_bias`；`python/sglang/srt/layers/quantization/mxfp4.py` modified +40/-1 (41 lines); hunks: -35,6 +35,7  @@ ); -342,6 +343,7  @@ def __init__(; symbols: __init__, create_weights, create_moe_runner, apply, touching `__init__, create_weights, create_moe_runner`；`python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` modified +10/-0 (10 lines); hunks: -1006,6 +1006,16  @@ void moe_wna16_marlin_gemm(.
- Code diff details:
  - `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py` modified +57/-12 (69 lines); hunks: -8,7 +8,7  @@ from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo; -38,17 +38,62  @@ def create_weights(; symbols: create_weights, apply, touching `create_weights, apply`
  - `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` modified +32/-16 (48 lines); hunks: -52,22 +52,38  @@ def _normalize_scale_tensor(; -129,19 +145,19  @@ def _permute_bias(bias: torch.Tensor | None) -> torch.Tensor | None:; symbols: _normalize_scale_tensor, _permute_bias, touching `_normalize_scale_tensor, _permute_bias`
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +40/-1 (41 lines); hunks: -35,6 +35,7  @@ ); -342,6 +343,7  @@ def __init__(; symbols: __init__, create_weights, create_moe_runner, apply, touching `__init__, create_weights, create_moe_runner`
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` modified +10/-0 (10 lines); hunks: -1006,6 +1006,16  @@ void moe_wna16_marlin_gemm(
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +3/-7 (10 lines); hunks: -119,13 +119,9  @@ def fused_marlin_moe(; symbols: fused_marlin_moe, touching `fused_marlin_moe`
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
-            layer,
-            num_experts,
diff -- python/sglang/srt/layers/quantization/marlin_utils_fp4.py
@@ -52,22 +52,38 @@ def _normalize_scale_tensor(
+def _get_optional_param(layer: torch.nn.Module, *names: str) -> torch.Tensor | None:
+    for name in names:
+        value = getattr(layer, name, None)
+        if value is not None:
+            return value
+    return None
+
+
diff -- python/sglang/srt/layers/quantization/mxfp4.py
@@ -35,6 +35,7 @@
+from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
@@ -342,6 +343,7 @@ def __init__(
+        self.use_marlin = get_moe_runner_backend().is_marlin()
@@ -507,6 +509,25 @@ def create_weights(
+        if self.use_marlin:
+            from sglang.srt.layers.quantization.marlin_utils import (
+                check_moe_marlin_supports_layer,
+            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py` modified +57/-12; `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` modified +32/-16; `python/sglang/srt/layers/quantization/mxfp4.py` modified +40/-1; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` modified +10/-0
  - tests: `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py` modified +2/-0; `test/registered/dsv4/test_deepseek_v4_flash_fp8_h200.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/mxfp4.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #25001 - [LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support

- Link: https://github.com/sgl-project/sglang/pull/25001
- Status/date: merged / 2026-05-13
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `01a225ac6f4a`.
- Diff scope read: GitHub Pull Request files API returned 7 files, +1013/-0, 1081 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`; technical summary: Covers "[LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-0 (15 lines); hunks: -13,6 +13,15  @@ per_tensor_quant_mla_fp8,; -350,6 +359,8  @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, touching `forward_absorb_prepare, forward_absorb_core`；`python/sglang/srt/models/deepseek_v2.py` modified +4/-0 (4 lines); hunks: -1687,11 +1687,15  @@ def prepare_qkv_latent(; symbols: prepare_qkv_latent, touching `prepare_qkv_latent`；`python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` added +849/-0 (849 lines); hunks: -0,0 +1,849  @@ +"""Triton kernels for absorbed-MLA ``kv_b_proj`` LoRA correction.；`python/sglang/srt/lora/deepseek_mla_correction.py` added +117/-0 (117 lines); hunks: -0,0 +1,117  @@ +"""LoRA correction for absorbed-MLA ``kv_b_proj``..
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-0 (15 lines); hunks: -13,6 +13,15  @@ per_tensor_quant_mla_fp8,; -350,6 +359,8  @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, touching `forward_absorb_prepare, forward_absorb_core`
  - `python/sglang/srt/models/deepseek_v2.py` modified +4/-0 (4 lines); hunks: -1687,11 +1687,15  @@ def prepare_qkv_latent(; symbols: prepare_qkv_latent, touching `prepare_qkv_latent`
  - `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` added +849/-0 (849 lines); hunks: -0,0 +1,849  @@ +"""Triton kernels for absorbed-MLA ``kv_b_proj`` LoRA correction.
  - `python/sglang/srt/lora/deepseek_mla_correction.py` added +117/-0 (117 lines); hunks: -0,0 +1,117  @@ +"""LoRA correction for absorbed-MLA ``kv_b_proj``.
  - `python/sglang/srt/lora/utils.py` modified +14/-0 (14 lines); hunks: -134,6 +134,18  @@ def get_hidden_dim(; -274,6 +286,8  @@ def get_target_module_name(full_module_name: str, target_modules: Set[str]) -> s; symbols: get_hidden_dim, get_target_module_name, touching `get_hidden_dim, get_target_module_name`
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
+from sglang.srt.lora.deepseek_mla_correction import (
+    is_kv_b_lora_active,
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1687,11 +1687,15 @@ def prepare_qkv_latent(
+        # When the module is wrapped with LoRA, the fused GEMM fast-path would
+        # bypass the adapter because it reads weight.T directly.
+        lora_active = getattr(self.fused_qkv_a_proj_with_mqa, "set_lora", False)
+            and not lora_active
diff -- python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py
@@ -0,0 +1,849 @@
+"""Triton kernels for absorbed-MLA ``kv_b_proj`` LoRA correction.
+
+The absorbed-MLA path bypasses ``kv_b_proj.forward()`` and folds the K/V
+sides as plain BMMs ``q_nope @ w_kc`` and ``attn_output @ w_vc``.  When a
+LoRA adapter is active on ``kv_b_proj`` we add the LoRA delta to
+``q_nope_out`` / ``attn_bmm_output`` manually.
+
+Using the standard LoRA factored math we *never* materialize ``B @ A``:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-0; `python/sglang/srt/models/deepseek_v2.py` modified +4/-0; `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` added +849/-0; `python/sglang/srt/lora/deepseek_mla_correction.py` added +117/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #25152 - docs: prepend SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 for H200 FP8 Flash max-throughput

- Link: https://github.com/sgl-project/sglang/pull/25152
- Status/date: merged / 2026-05-13
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `7d515c6d1f22`.
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: prepend SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 for H200 FP8 Flash max-throughput"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "docs: prepend SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 for H200 FP8 Flash max-throughput" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -391,6 +391,9  @@ export const DeepSeekV4Deployment = () => {; symbols: DeepSeekV4Deployment, touching `DeepSeekV4Deployment`.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -391,6 +391,9  @@ export const DeepSeekV4Deployment = () => {; symbols: DeepSeekV4Deployment, touching `DeepSeekV4Deployment`
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
- Risk and verification: This PR mainly changes docs/tests/CI ``docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx``; verify commands, CI selectors, and model repo names still map to the current implementation.

### PR #24925 - [attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell)

- Link: https://github.com/sgl-project/sglang/pull/24925
- Status/date: merged / 2026-05-14
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `7618ad707568`.
- Diff scope read: GitHub Pull Request files API returned 11 files, +462/-92, 726 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell)"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`; technical summary: Covers "[attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell)" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` added +247/-0 (247 lines); hunks: -0,0 +1,247  @@ +# Copyright (c) 2026 LightSeek Foundation；`python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +132/-91 (223 lines); hunks: -755,6 +755,109  @@ def unpad_draft_extend_output(; -838,46 +941,13  @@ def forward_decode(; symbols: unpad_draft_extend_output, forward_decode, forward_extend, touching `unpad_draft_extend_output, forward_decode, forward_extend`；`python/sglang/srt/layers/attention/attention_registry.py` modified +11/-0 (11 lines); hunks: -62,6 +62,17  @@ def create_trtllm_mla_backend(runner):; symbols: create_trtllm_mla_backend, touching `create_trtllm_mla_backend`；`python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +7/-0 (7 lines); hunks: -134,6 +134,12  @@ def handle_attention_trtllm_mla(attn, forward_batch):; -183,6 +189,7  @@ def handle_attention_intel_xpu(attn, forward_batch):; symbols: handle_attention_trtllm_mla, handle_attention_intel_xpu, touching `handle_attention_trtllm_mla, handle_attention_intel_xpu`.
- Code diff details:
  - `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` added +247/-0 (247 lines); hunks: -0,0 +1,247  @@ +# Copyright (c) 2026 LightSeek Foundation
  - `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +132/-91 (223 lines); hunks: -755,6 +755,109  @@ def unpad_draft_extend_output(; -838,46 +941,13  @@ def forward_decode(; symbols: unpad_draft_extend_output, forward_decode, forward_extend, touching `unpad_draft_extend_output, forward_decode, forward_extend`
  - `python/sglang/srt/layers/attention/attention_registry.py` modified +11/-0 (11 lines); hunks: -62,6 +62,17  @@ def create_trtllm_mla_backend(runner):; symbols: create_trtllm_mla_backend, touching `create_trtllm_mla_backend`
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +7/-0 (7 lines); hunks: -134,6 +134,12  @@ def handle_attention_trtllm_mla(attn, forward_batch):; -183,6 +189,7  @@ def handle_attention_intel_xpu(attn, forward_batch):; symbols: handle_attention_trtllm_mla, handle_attention_intel_xpu, touching `handle_attention_trtllm_mla, handle_attention_intel_xpu`
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +1/-1 (2 lines); hunks: -675,7 +675,7  @@ def _fuse_rope_for_trtllm_mla(; symbols: _fuse_rope_for_trtllm_mla, touching `_fuse_rope_for_trtllm_mla`
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
+# copies of the Software, and to permit persons to whom the Software is
+# furnished to do so, subject to the following conditions:
diff -- python/sglang/srt/layers/attention/trtllm_mla_backend.py
@@ -755,6 +755,109 @@ def unpad_draft_extend_output(
+    def _compute_decode_bmm1_scale(self, layer: RadixAttention) -> float:
+        """BMM1 scale ``q_scale * k_scale * softmax_scale``. k_scale only
+        applies when the KV cache stores FP8."""
+        q_scale = 1.0
+        if self.data_type == torch.float8_e4m3fn:
+            k_scale = (
+                layer.k_scale_float
+                if getattr(layer, "k_scale_float", None) is not None
diff -- python/sglang/srt/layers/attention/attention_registry.py
@@ -62,6 +62,17 @@ def create_trtllm_mla_backend(runner):
+@register_attention_backend("tokenspeed_mla")
+def create_tokenspeed_mla_backend(runner):
+    if not runner.use_mla_backend:
+        raise ValueError("tokenspeed_mla backend can only be used with MLA models.")
+    from sglang.srt.layers.attention.tokenspeed_mla_backend import (
+        TokenspeedMLABackend,
+    )
+
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` added +247/-0; `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +132/-91; `python/sglang/srt/layers/attention/attention_registry.py` modified +11/-0; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #25052 - DeepSeek V4 w4a4 MegaMoE

- Link: https://github.com/sgl-project/sglang/pull/25052
- Status/date: merged / 2026-05-14
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `b7f856df70c8`.
- Diff scope read: GitHub Pull Request files API returned 5 files, +212/-60, 328 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DeepSeek V4 w4a4 MegaMoE"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/mega_moe.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py`, `python/sglang/srt/environ.py`; technical summary: Covers "DeepSeek V4 w4a4 MegaMoE" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/layers/moe/mega_moe.py` modified +52/-10 (62 lines); hunks: -15,6 +15,7  @@ from __future__ import annotations; -34,6 +35,26  @@ _MEGA_MOE_SYMM_BUFFER: dict = {}; symbols: dict, _get_mega_moe_symm_buffer, _run_mega_routed, touching `dict, _get_mega_moe_symm_buffer, _run_mega_routed`；`test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py` added +148/-0 (148 lines); hunks: -0,0 +1,148  @@ +"""B200 per-commit CI: DeepSeek-V4-Flash FP4 (LowLatency recipe).；`python/sglang/srt/environ.py` modified +11/-0 (11 lines); hunks: -595,6 +595,17  @@ class Envs:; symbols: Envs, touching `Envs`；`python/pyproject.toml` modified +1/-1 (2 lines); hunks: -59,7 +59,7  @@ dependencies = [; symbols: dependencies, touching `dependencies`.
- Code diff details:
  - `python/sglang/srt/layers/moe/mega_moe.py` modified +52/-10 (62 lines); hunks: -15,6 +15,7  @@ from __future__ import annotations; -34,6 +35,26  @@ _MEGA_MOE_SYMM_BUFFER: dict = {}; symbols: dict, _get_mega_moe_symm_buffer, _run_mega_routed, touching `dict, _get_mega_moe_symm_buffer, _run_mega_routed`
  - `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py` added +148/-0 (148 lines); hunks: -0,0 +1,148  @@ +"""B200 per-commit CI: DeepSeek-V4-Flash FP4 (LowLatency recipe).
  - `python/sglang/srt/environ.py` modified +11/-0 (11 lines); hunks: -595,6 +595,17  @@ class Envs:; symbols: Envs, touching `Envs`
  - `python/pyproject.toml` modified +1/-1 (2 lines); hunks: -59,7 +59,7  @@ dependencies = [; symbols: dependencies, touching `dependencies`
  - `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py` modified +0/-49 (49 lines); hunks: -31,14 +31,6  @@ "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",; -138,46 +130,5  @@ def test_gsm8k(self):; symbols: test_gsm8k, touching `test_gsm8k`
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/mega_moe.py
@@ -15,6 +15,7 @@
+import os
@@ -34,6 +35,26 @@
+_MEGA_MOE_DG_ENV_APPLIED = False
+
+
+def _apply_mega_moe_dg_env() -> None:
+    """Forward sglang's FP4/MXF4 opt-in flags to DeepGEMM via env vars.
+
diff -- test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py
@@ -0,0 +1,148 @@
+"""B200 per-commit CI: DeepSeek-V4-Flash FP4 (LowLatency recipe).
+
+Launches TP=4 with flashinfer_mxfp4 MoE runner + EAGLE speculative decoding.
+Runs 12 ServerSanity probes (correctness, streaming, concurrency, determinism)
+plus a GSM8K accuracy gate.
+
+Registry: stage-c-test-dsv4-4-gpu-b200 (per-commit, 4x B200)
+"""
diff -- python/sglang/srt/environ.py
@@ -595,6 +595,17 @@ class Envs:
+
+    # When set, the mega-MoE x slot is packed E2M1 (FP4) instead of FP8 E4M3.
+    # Halves symm-buffer footprint and unlocks the MXF4 mainloop downstream.
+    # Setting this also exports DG_USE_FP4_ACTS=1 so DeepGEMM's symm-buffer
+    # sizing + fp8_fp4_mega_moe pick up the FP4 layout.
+    SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS = EnvBool(False)
+    # Switches the L1+L2 mainloops from kind::mxf8f6f4 (K=32 with-padding) to
+    # kind::mxf4 (K=64 dense) inside fp8_fp4_mega_moe. No effect unless
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/mega_moe.py` modified +52/-10; `python/sglang/srt/environ.py` modified +11/-0; `python/pyproject.toml` modified +1/-1
  - tests: `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py` added +148/-0; `test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py` modified +0/-49
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/mega_moe.py`, `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py`, `python/sglang/srt/environ.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #25243 - [Docs] update dsv4 cookbook with H100 deployment commands

- Link: https://github.com/sgl-project/sglang/pull/25243
- Status/date: merged / 2026-05-14
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `1f119f6a4463`.
- Diff scope read: GitHub Pull Request files API returned 2 files, +83/-9, 153 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] update dsv4 cookbook with H100 deployment commands"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[Docs] update dsv4 cookbook with H100 deployment commands" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-9 (88 lines); hunks: -7,6 +7,7  @@ export const DeepSeekV4Deployment = () => {; -32,6 +33,7  @@ export const DeepSeekV4Deployment = () => {; symbols: DeepSeekV4Deployment, touching `DeepSeekV4Deployment`；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -100,6 +100,10  @@ Please refer to the [official SGLang installation guide](../../../docs/get-start.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-9 (88 lines); hunks: -7,6 +7,7  @@ export const DeepSeekV4Deployment = () => {; -32,6 +33,7  @@ export const DeepSeekV4Deployment = () => {; symbols: DeepSeekV4Deployment, touching `DeepSeekV4Deployment`
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -100,6 +100,10  @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -7,6 +7,7 @@ export const DeepSeekV4Deployment = () => {
+  //     H100  → FP4 weights (Marlin), Flash TP=8 single-node / Pro TP=16 2-node
@@ -32,6 +33,7 @@ export const DeepSeekV4Deployment = () => {
+        { id: "h100", label: "H100 (FP4)", default: false },
@@ -71,14 +73,17 @@ export const DeepSeekV4Deployment = () => {
-  // Recipes that are not supported on the H200 (FP4) Marlin path.
-  const H200_FP4_UNSUPPORTED_RECIPES = new Set(["cp", "pd-disagg"]);
+  // Recipes that are not supported on the Marlin (FP4) Hopper paths
+  // (H200 FP4, H100 FP4).
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -100,6 +100,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+    <tr>
+      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>NVIDIA H100</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>lmsysorg/sglang:dev</code></td>
+    </tr>
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-9; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- Risk and verification: This PR mainly changes docs/tests/CI ``docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx``; verify commands, CI selectors, and model repo names still map to the current implementation.

### PR #24691 - [UnifiedTree]: Support HiCache For DeepSeek_V4

- Link: https://github.com/sgl-project/sglang/pull/24691
- Status/date: merged / 2026-05-15
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `d9fa84b25b79`.
- Diff scope read: GitHub Pull Request files API returned 11 files, +1221/-154, 1970 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[UnifiedTree]: Support HiCache For DeepSeek_V4"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`, `python/sglang/srt/mem_cache/unified_radix_cache.py`; technical summary: Covers "[UnifiedTree]: Support HiCache For DeepSeek_V4" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/mem_cache/memory_pool_host.py` modified +490/-3 (493 lines); hunks: -1656,16 +1656,503  @@ def get_page_buffer_meta(self, indices):; symbols: get_page_buffer_meta, touching `get_page_buffer_meta`；`python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +335/-38 (373 lines); hunks: -3,12 +3,15  @@ import logging; -71,7 +74,6  @@ def build_pool_entry(; symbols: build_pool_entry, build_kv_only_stack, build_hybrid_swa_stack, build_hybrid_mamba_stack, touching `build_pool_entry, build_kv_only_stack, build_hybrid_swa_stack`；`python/sglang/srt/mem_cache/unified_radix_cache.py` modified +67/-23 (90 lines); hunks: -22,7 +22,11  @@ MatchPrefixParams,; -221,9 +225,7  @@ def __init__(; symbols: __init__, init_hicache, write_backup, bool, touching `__init__, init_hicache, write_backup`；`python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` modified +59/-22 (81 lines); hunks: -23,6 +23,7  @@ from sglang.srt.mem_cache.hicache_storage import (; -50,12 +51,12  @@ def __init__(; symbols: __init__, cat_or_none, move_hybrid_indices, _page_transfer, touching `__init__, cat_or_none, move_hybrid_indices`.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool_host.py` modified +490/-3 (493 lines); hunks: -1656,16 +1656,503  @@ def get_page_buffer_meta(self, indices):; symbols: get_page_buffer_meta, touching `get_page_buffer_meta`
  - `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +335/-38 (373 lines); hunks: -3,12 +3,15  @@ import logging; -71,7 +74,6  @@ def build_pool_entry(; symbols: build_pool_entry, build_kv_only_stack, build_hybrid_swa_stack, build_hybrid_mamba_stack, touching `build_pool_entry, build_kv_only_stack, build_hybrid_swa_stack`
  - `python/sglang/srt/mem_cache/unified_radix_cache.py` modified +67/-23 (90 lines); hunks: -22,7 +22,11  @@ MatchPrefixParams,; -221,9 +225,7  @@ def __init__(; symbols: __init__, init_hicache, write_backup, bool, touching `__init__, init_hicache, write_backup`
  - `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` modified +59/-22 (81 lines); hunks: -23,6 +23,7  @@ from sglang.srt.mem_cache.hicache_storage import (; -50,12 +51,12  @@ def __init__(; symbols: __init__, cat_or_none, move_hybrid_indices, _page_transfer, touching `__init__, cat_or_none, move_hybrid_indices`
  - `python/sglang/test/kl_multiturn_utils.py` modified +58/-9 (67 lines); hunks: -104,6 +104,7  @@ def _replay_and_compare_kl(; -114,6 +115,7  @@ def _replay_and_compare_kl(; symbols: _replay_and_compare_kl, _interleave_order, test_input_output_logprobs_match_helper, test_input_output_logprobs_match_prefill_cache_hit_helper, touching `_replay_and_compare_kl, _interleave_order, test_input_output_logprobs_match_helper`
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/memory_pool_host.py
@@ -1656,16 +1656,503 @@ def get_page_buffer_meta(self, indices):
+# ---- V4 Compressed KV Host Pools ----
+
+
+class LogicalHostPool:
+    """Pure-logical anchor pool for V4 HiCache.
+
+    The pool manages page-aligned token slots but holds no KV tensor. V4
+    compressed side pools use these logical FULL indices as stable page anchors.
diff -- python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py
@@ -3,12 +3,15 @@
-from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName
+from sglang.srt.mem_cache.hicache_storage import PoolName, SidecarPoolSpec
+    DeepSeekV4PagedHostPool,
+    DeepSeekV4StateHostPool,
+    LogicalHostPool,
@@ -71,7 +74,6 @@ def build_pool_entry(
-    share_indices_with_anchor: bool = False,
@@ -83,7 +85,6 @@ def build_pool_entry(
diff -- python/sglang/srt/mem_cache/unified_radix_cache.py
@@ -22,7 +22,11 @@
-from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
+from sglang.srt.mem_cache.hicache_storage import (
+    PoolName,
+    PoolTransfer,
+    SidecarPoolSpec,
+)
@@ -221,9 +225,7 @@ def __init__(
-        self.hicache_anchor_kv_shared_indices_pools: list[
```

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/memory_pool_host.py` modified +490/-3; `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +335/-38; `python/sglang/srt/mem_cache/unified_radix_cache.py` modified +67/-23; `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` modified +59/-22
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`, `python/sglang/srt/mem_cache/unified_radix_cache.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #25369 - Add hicache feature in dsv4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/25369
- Status/date: merged / 2026-05-15
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `c7e879e43f77`.
- Diff scope read: GitHub Pull Request files API returned 2 files, +50/-4, 95 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add hicache feature in dsv4 cookbook"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "Add hicache feature in dsv4 cookbook" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +39/-4 (43 lines); hunks: -71,6 +71,14  @@ export const DeepSeekV4Deployment = () => {; -295,7 +303,7  @@ export const DeepSeekV4Deployment = () => {; symbols: DeepSeekV4Deployment, touching `DeepSeekV4Deployment`；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +11/-0 (11 lines); hunks: -334,6 +334,17  @@ print().
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +39/-4 (43 lines); hunks: -71,6 +71,14  @@ export const DeepSeekV4Deployment = () => {; -295,7 +303,7  @@ export const DeepSeekV4Deployment = () => {; symbols: DeepSeekV4Deployment, touching `DeepSeekV4Deployment`
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +11/-0 (11 lines); hunks: -334,6 +334,17  @@ print()
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -71,6 +71,14 @@ export const DeepSeekV4Deployment = () => {
+    hicache: {
+      name: "hicache",
+      title: "HiCache",
+      items: [
+        { id: "disabled", label: "Disabled", default: true  },
+        { id: "l2",       label: "L2",       default: false, subtitle: "GPU+CPU" },
+      ],
+    },
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -334,6 +334,17 @@ print()
+#### 4.2.3 HiCache (Hierarchical KV Caching)
+
+HiCache enables multi-tier KV cache offloading (GPU → CPU → Storage), significantly expanding effective context capacity for long-context and multi-turn scenarios. Combined with U
+
+To enable HiCache, use the **HiCache** toggle in the [command generator above](#3-model-deployment):
+
+- **L2 (GPU + CPU):** Offloads cold KV pages to CPU memory. Enables `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` for intelligent hierarchical prefix caching.
+- **L3 (GPU + CPU + Storage):** Coming soon.
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +39/-4; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +11/-0
- Risk and verification: This PR mainly changes docs/tests/CI ``docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx``; verify commands, CI selectors, and model repo names still map to the current implementation.

### PR #24704 - feat: add Pipeline Parallelism (PP) and PD support for DeepSeek-V4

- Link: https://github.com/sgl-project/sglang/pull/24704
- Status/date: merged / 2026-05-16
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `162540e0a8d3`.
- Diff scope read: GitHub Pull Request files API returned 9 files, +341/-103, 750 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: add Pipeline Parallelism (PP) and PD support for DeepSeek-V4"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/disaggregation/common/conn.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; technical summary: Covers "feat: add Pipeline Parallelism (PP) and PD support for DeepSeek-V4" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +99/-39 (138 lines); hunks: -2,7 +2,16  @@ import concurrent.futures; -49,7 +58,7  @@ from sglang.srt.layers.moe import get_moe_a2a_backend; symbols: __init__, forward, _setup_fp8_wo_a_scales, post_load_weights, touching `__init__, forward, _setup_fp8_wo_a_scales`；`python/sglang/srt/disaggregation/common/conn.py` modified +125/-7 (132 lines); hunks: -474,15 +474,133  @@ def get_mha_kv_ptrs_with_pp(; symbols: get_mha_kv_ptrs_with_pp, touching `get_mha_kv_ptrs_with_pp`；`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +71/-51 (122 lines); hunks: -401,6 +401,19  @@ def __init__(; -412,8 +425,8  @@ def __init__(; symbols: __init__, get_state_buf_infos, _init_paged_compress_states, get_indexer_compress_states, touching `__init__, get_state_buf_infos, _init_paged_compress_states`；`python/sglang/srt/disaggregation/base/conn.py` modified +12/-2 (14 lines); hunks: -47,11 +47,21  @@ class KVArgs:; symbols: KVArgs, touching `KVArgs`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +99/-39 (138 lines); hunks: -2,7 +2,16  @@ import concurrent.futures; -49,7 +58,7  @@ from sglang.srt.layers.moe import get_moe_a2a_backend; symbols: __init__, forward, _setup_fp8_wo_a_scales, post_load_weights, touching `__init__, forward, _setup_fp8_wo_a_scales`
  - `python/sglang/srt/disaggregation/common/conn.py` modified +125/-7 (132 lines); hunks: -474,15 +474,133  @@ def get_mha_kv_ptrs_with_pp(; symbols: get_mha_kv_ptrs_with_pp, touching `get_mha_kv_ptrs_with_pp`
  - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +71/-51 (122 lines); hunks: -401,6 +401,19  @@ def __init__(; -412,8 +425,8  @@ def __init__(; symbols: __init__, get_state_buf_infos, _init_paged_compress_states, get_indexer_compress_states, touching `__init__, get_state_buf_infos, _init_paged_compress_states`
  - `python/sglang/srt/disaggregation/base/conn.py` modified +12/-2 (14 lines); hunks: -47,11 +47,21  @@ class KVArgs:; symbols: KVArgs, touching `KVArgs`
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +12/-2 (14 lines); hunks: -174,6 +174,7  @@ def create(; -207,10 +208,16  @@ def create(; symbols: create, __init__, touching `create, __init__`
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
+    Optional,
+    Set,
diff -- python/sglang/srt/disaggregation/common/conn.py
@@ -474,15 +474,133 @@ def get_mha_kv_ptrs_with_pp(
+        # Fast path: both sides use exactly the same PP layout
+        if len(src_kv_ptrs) == len(dst_kv_ptrs):
+            return src_kv_ptrs, dst_kv_ptrs, len(src_kv_ptrs)
+
+        mla_ratios = getattr(self.kv_args, "mla_compression_ratios", None)
+        if mla_ratios:
+            # Compressed-MLA (e.g. DeepSeek V4): the flat list is organized
+            # by buffer type (compression-ratio bucket) rather than by
diff -- python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
@@ -401,6 +401,19 @@ def __init__(
+        # Determine this PP stage's absolute layer range
+        if (
+            start_layer is not None
+            and end_layer is not None
+            and len(compression_ratios) >= end_layer
+        ):
+            self._stage_start = start_layer
+            self._stage_end = end_layer
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +99/-39; `python/sglang/srt/disaggregation/common/conn.py` modified +125/-7; `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` modified +71/-51; `python/sglang/srt/disaggregation/base/conn.py` modified +12/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/disaggregation/common/conn.py`, `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #25410 - [Docs] Update DeepSeek V4 cookbook to use the latest docker image

- Link: https://github.com/sgl-project/sglang/pull/25410
- Status/date: merged / 2026-05-16
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `9f26697d6a6a`.
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-41, 63 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Update DeepSeek V4 cookbook to use the latest docker image"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[Docs] Update DeepSeek V4 cookbook to use the latest docker image" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-41 (47 lines); hunks: -66,48 +66,13  @@ SGLang offers multiple installation methods. Choose based on your hardware platf; -116,7 +81,7  @@ docker run --gpus all \.
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-41 (47 lines); hunks: -66,48 +66,13  @@ SGLang offers multiple installation methods. Choose based on your hardware platf; -116,7 +81,7  @@ docker run --gpus all \
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
-  </colgroup>
-  <thead>
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-41
- Risk and verification: This PR mainly changes docs/tests/CI ``docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx``; verify commands, CI selectors, and model repo names still map to the current implementation.

### PR #25412 - [Doc] DSV4 cookbook: clean up env vars, add MegaMoE toggle, unify docker image

- Link: https://github.com/sgl-project/sglang/pull/25412
- Status/date: merged / 2026-05-16
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `57eb5bdaf6cc`.
- Diff scope read: GitHub Pull Request files API returned 1 files, +38/-83, 185 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Doc] DSV4 cookbook: clean up env vars, add MegaMoE toggle, unify docker image"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "[Doc] DSV4 cookbook: clean up env vars, add MegaMoE toggle, unify docker image" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +38/-83 (121 lines); hunks: -79,6 +79,15  @@ export const DeepSeekV4Deployment = () => {; -303,7 +312,7  @@ export const DeepSeekV4Deployment = () => {; symbols: DeepSeekV4Deployment, touching `DeepSeekV4Deployment`.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +38/-83 (121 lines); hunks: -79,6 +79,15  @@ export const DeepSeekV4Deployment = () => {; -303,7 +312,7  @@ export const DeepSeekV4Deployment = () => {; symbols: DeepSeekV4Deployment, touching `DeepSeekV4Deployment`
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
+        { id: "w4a4",     label: "W4A4",     default: false, subtitle: "FP4 acts" },
+      ],
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +38/-83
- Risk and verification: This PR mainly changes docs/tests/CI ``docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx``; verify commands, CI selectors, and model repo names still map to the current implementation.

### PR #25419 - Port SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN from deepseek_v4_dev

- Link: https://github.com/sgl-project/sglang/pull/25419
- Status/date: merged / 2026-05-16
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `18c16f8660c3`.
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-1, 23 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Port SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN from deepseek_v4_dev"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/environ.py`; technical summary: Covers "Port SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN from deepseek_v4_dev" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/managers/schedule_batch.py` modified +5/-1 (6 lines); hunks: -2724,9 +2724,13  @@ def _evict_swa(self, req: Req, pre_len: int):; symbols: _evict_swa, touching `_evict_swa`；`python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -596,6 +596,7  @@ class Envs:; symbols: Envs, touching `Envs`.
- Code diff details:
  - `python/sglang/srt/managers/schedule_batch.py` modified +5/-1 (6 lines); hunks: -2724,9 +2724,13  @@ def _evict_swa(self, req: Req, pre_len: int):; symbols: _evict_swa, touching `_evict_swa`
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -596,6 +596,7  @@ class Envs:; symbols: Envs, touching `Envs`
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
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/environ.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #25477 - [BugFix]: Fix DeepSeek V4 HiCache layer count logic

- Link: https://github.com/sgl-project/sglang/pull/25477
- Status/date: merged / 2026-05-16
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `0be539024faf`.
- Diff scope read: GitHub Pull Request files API returned 3 files, +161/-144, 349 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix]: Fix DeepSeek V4 HiCache layer count logic"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py`; technical summary: Covers "[BugFix]: Fix DeepSeek V4 HiCache layer count logic" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +6/-3 (9 lines); hunks: -283,7 +283,8  @@ def build_deepseek_v4_hicache_stack(; -293,7 +294,9  @@ def build_deepseek_v4_hicache_stack(; symbols: build_deepseek_v4_hicache_stack, attach_hybrid_pool_to_unified_cache, touching `build_deepseek_v4_hicache_stack, attach_hybrid_pool_to_unified_cache`；`test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` added +155/-0 (155 lines); hunks: -0,0 +1,155  @@ +import unittest；`test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py` renamed +0/-141 (141 lines); hunks: -13,162 +13,21  @@ from urllib.parse import urlparse.
- Code diff details:
  - `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +6/-3 (9 lines); hunks: -283,7 +283,8  @@ def build_deepseek_v4_hicache_stack(; -293,7 +294,9  @@ def build_deepseek_v4_hicache_stack(; symbols: build_deepseek_v4_hicache_stack, attach_hybrid_pool_to_unified_cache, touching `build_deepseek_v4_hicache_stack, attach_hybrid_pool_to_unified_cache`
  - `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` added +155/-0 (155 lines); hunks: -0,0 +1,155  @@ +import unittest
  - `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py` renamed +0/-141 (141 lines); hunks: -13,162 +13,21  @@ from urllib.parse import urlparse
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py
@@ -283,7 +283,8 @@ def build_deepseek_v4_hicache_stack(
-    transfer_layer_num = len(kvcache.compression_ratios)
+    # TODO(hzh0425): Support PP for deepseek v4 with hicache
+    transfer_layer_num = kvcache.end_layer - kvcache.start_layer
@@ -293,7 +294,9 @@ def build_deepseek_v4_hicache_stack(
-    for layer_id, layer_item in enumerate(kvcache.layer_mapping):
+    for layer_id, layer_item in enumerate(
+        kvcache.layer_mapping[kvcache.start_layer : kvcache.end_layer]
+    ):
diff -- test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py
@@ -0,0 +1,155 @@
+import unittest
+
+from test_unified_radix_cache_kl import UnifiedRadixTreeTestMixin
+
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
-    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
-MAMBA_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
```

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +6/-3
  - tests: `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` added +155/-0; `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py` renamed +0/-141
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py`, `test/registered/radix_cache/test_unified_radix_cache_kl_hicache_nightly.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #25506 - [Doc] Fix several places for dpsk v4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/25506
- Status/date: merged / 2026-05-17
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `6dcacb1159d6`.
- Diff scope read: GitHub Pull Request files API returned 2 files, +47/-1, 83 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Doc] Fix several places for dpsk v4 cookbook"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[Doc] Fix several places for dpsk v4 cookbook" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +26/-0 (26 lines); hunks: -96,6 +96,15  @@ export const DeepSeekV4Deployment = () => {; -104,6 +113,14  @@ export const DeepSeekV4Deployment = () => {; symbols: DeepSeekV4Deployment, touching `DeepSeekV4Deployment`；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +21/-1 (22 lines); hunks: -120,14 +120,34  @@ The generator currently picks values on the **conservative** side (mirroring an.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +26/-0 (26 lines); hunks: -96,6 +96,15  @@ export const DeepSeekV4Deployment = () => {; -104,6 +113,14  @@ export const DeepSeekV4Deployment = () => {; symbols: DeepSeekV4Deployment, touching `DeepSeekV4Deployment`
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +21/-1 (22 lines); hunks: -120,14 +120,34  @@ The generator currently picks values on the **conservative** side (mirroring an
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -96,6 +96,15 @@ export const DeepSeekV4Deployment = () => {
+  // MegaMoE is only supported on Blackwell with DeepEP-based recipes
+  // (balanced / max-throughput / pd-disagg). It's disabled on Hopper
+  // (H100 / H200 / H200-FP4) and on low-latency / cp recipes.
+  const MEGAMOE_UNSUPPORTED_RECIPES = new Set(["low-latency", "cp"]);
+  const MEGAMOE_UNSUPPORTED_HARDWARE = new Set(["h100", "h200", "h200-fp4"]);
+  const isMegamoeUnsupported = (vals) =>
+    MEGAMOE_UNSUPPORTED_HARDWARE.has(vals.hardware) ||
+    MEGAMOE_UNSUPPORTED_RECIPES.has(vals.recipe);
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -120,14 +120,34 @@ The generator currently picks values on the **conservative** side (mirroring an
-- Original FP4 checkpoints: To run original FP4 checkpoints, apply the w4a16 MoE kernels (marlin) as in interactive command generator. For this option we only support TP method. C
+- Original FP4 checkpoints: To run original FP4 checkpoints, we provide two different options for w4a16 MoE kernels: Marlin (`--moe-runner-backend marlin`) and Flashinfer (`--moe-
+**MegaMoE**
+
+MegaMoE fuses expert dispatch + GEMM into a single kernel for higher throughput
+on MoE layers. To enable it, use the **MegaMoE** toggle in the
+[command generator above](#3-model-deployment) — the generator will swap
+`--moe-a2a-backend deepep` for `--moe-a2a-backend megamoe` and add the
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +26/-0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +21/-1
- Risk and verification: This PR mainly changes docs/tests/CI ``docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx``; verify commands, CI selectors, and model repo names still map to the current implementation.

### PR #24933 - Amd/deepseek v4 rebase main 0509

- Link: https://github.com/sgl-project/sglang/pull/24933
- Status/date: merged / 2026-05-18
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `866793c502b7`.
- Diff scope read: GitHub Pull Request files API returned 17 files, +3678/-70, 4186 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Amd/deepseek v4 rebase main 0509"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsv4/compress_hip.py`; technical summary: Covers "Amd/deepseek v4 rebase main 0509" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` added +1265/-0 (1265 lines); hunks: -0,0 +1,1265  @@ +from __future__ import annotations；`python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +1214/-2 (1216 lines); hunks: -1,5 +1,6  @@ +import functools; -10,6 +11,28  @@ tilelang.set_log_level("WARNING"); symbols: _is_fp8_fnuz, tilelang_sparse_fwd, touching `_is_fp8_fnuz, tilelang_sparse_fwd`；`python/sglang/srt/layers/attention/dsv4/compress_hip.py` added +455/-0 (455 lines); hunks: -0,0 +1,455  @@ +from __future__ import annotations；`python/sglang/srt/layers/attention/hip_flash_mla.py` added +197/-0 (197 lines); hunks: -0,0 +1,197  @@ +from typing import Any, Optional.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` added +1265/-0 (1265 lines); hunks: -0,0 +1,1265  @@ +from __future__ import annotations
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +1214/-2 (1216 lines); hunks: -1,5 +1,6  @@ +import functools; -10,6 +11,28  @@ tilelang.set_log_level("WARNING"); symbols: _is_fp8_fnuz, tilelang_sparse_fwd, touching `_is_fp8_fnuz, tilelang_sparse_fwd`
  - `python/sglang/srt/layers/attention/dsv4/compress_hip.py` added +455/-0 (455 lines); hunks: -0,0 +1,455  @@ +from __future__ import annotations
  - `python/sglang/srt/layers/attention/hip_flash_mla.py` added +197/-0 (197 lines); hunks: -0,0 +1,197  @@ +from typing import Any, Optional
  - `python/sglang/srt/layers/quantization/fp8.py` modified +143/-16 (159 lines); hunks: -84,6 +84,7  @@ get_bool_env_var,; -111,9 +112,21  @@ _is_fp8_fnuz = is_fp8_fnuz(); symbols: _is_fp8_fnuz, create_weights, process_weights_after_loading_block_quant, maybe_get_hip_aiter_quant_info, touching `_is_fp8_fnuz, create_weights, process_weights_after_loading_block_quant`
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -0,0 +1,1265 @@
+from __future__ import annotations
+
+import enum
+import functools
+import logging
+from dataclasses import dataclass, field
+from typing import (
+    TYPE_CHECKING,
diff -- python/sglang/srt/layers/attention/nsa/tilelang_kernel.py
@@ -1,5 +1,6 @@
+import functools
-from typing import Optional, Tuple
+from typing import Any, Optional, Tuple
@@ -10,6 +11,28 @@
+# Workaround a tilelang bug: BaseKernelAdapter._legalize_result_idx mutates the
+# `out_idx` list in place when normalising negative indices to positive ones.
+# That breaks any @tilelang.jit factory that compiles two prim_funcs with
+# different param counts (e.g. our unified single/dual partial kernel) — the
diff -- python/sglang/srt/layers/attention/dsv4/compress_hip.py
@@ -0,0 +1,455 @@
+from __future__ import annotations
+
+import os
+from functools import cached_property
+from typing import TYPE_CHECKING, Any
+
+import torch
+import torch.nn as nn
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` added +1265/-0; `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +1214/-2; `python/sglang/srt/layers/attention/dsv4/compress_hip.py` added +455/-0; `python/sglang/srt/layers/attention/hip_flash_mla.py` added +197/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsv4/compress_hip.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #25569 - Add DeepSeekV4 fused MoE Triton autotune support

- Link: https://github.com/sgl-project/sglang/pull/25569
- Status/date: merged / 2026-05-18
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `ba2ffcf156b5`.
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-0, 29 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DeepSeekV4 fused MoE Triton autotune support"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `benchmark/kernels/fused_moe_triton/common_utils.py`; technical summary: Covers "Add DeepSeekV4 fused MoE Triton autotune support" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +5/-0 (5 lines); hunks: -35,6 +35,7  @@ set_global_server_args_for_scheduler,; -174,8 +175,12  @@ def prepare(i: int):; symbols: prepare, touching `prepare`；`benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunks: -85,6 +85,7  @@ def get_model_config(; symbols: get_model_config, touching `get_model_config`.
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +5/-0 (5 lines); hunks: -35,6 +35,7  @@ set_global_server_args_for_scheduler,; -174,8 +175,12  @@ def prepare(i: int):; symbols: prepare, touching `prepare`
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunks: -85,6 +85,7  @@ def get_model_config(; symbols: get_model_config, touching `get_model_config`
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
  - runtime: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +5/-0; `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `benchmark/kernels/fused_moe_triton/common_utils.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #25282 - [UnifiedTree] Support deepseek v4 host pool layout

- Link: https://github.com/sgl-project/sglang/pull/25282
- Status/date: merged / 2026-05-19
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `c2a212bfe222`.
- Diff scope read: GitHub Pull Request files API returned 5 files, +401/-114, 809 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[UnifiedTree] Support deepseek v4 host pool layout"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/test/kl_multiturn_utils.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`; technical summary: Covers "[UnifiedTree] Support deepseek v4 host pool layout" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/mem_cache/memory_pool_host.py` modified +333/-92 (425 lines); hunks: -1754,6 +1754,7  @@ def __init__(; -1769,7 +1770,7  @@ def __init__(; symbols: __init__, _to_page_indices, init_kv_buffer, backup_from_device_all_layer, touching `__init__, _to_page_indices, init_kv_buffer`；`python/sglang/test/kl_multiturn_utils.py` modified +40/-19 (59 lines); hunks: -2,6 +2,7  @@ from __future__ import annotations; -145,30 +146,45  @@ def _interleave_order(n: int, branches_per_group: int) -> list[int] | None:; symbols: _interleave_order, test_input_output_logprobs_match_decode_cache_hit_helper, touching `_interleave_order, test_input_output_logprobs_match_decode_cache_hit_helper`；`python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +7/-0 (7 lines); hunks: -325,6 +325,7  @@ def build_deepseek_v4_hicache_stack(; -357,6 +358,7  @@ def build_deepseek_v4_hicache_stack(; symbols: build_deepseek_v4_hicache_stack, touching `build_deepseek_v4_hicache_stack`；`test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` modified +17/-3 (20 lines); hunks: -92,8 +92,13  @@ def _assert_dsv4_decode_cached_tokens(result, history_len, output_len, label):; -129,15 +134,15  @@ def setUpClass(cls):; symbols: _assert_dsv4_decode_cached_tokens, setUpClass, tearDownClass, touching `_assert_dsv4_decode_cached_tokens, setUpClass, tearDownClass`.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool_host.py` modified +333/-92 (425 lines); hunks: -1754,6 +1754,7  @@ def __init__(; -1769,7 +1770,7  @@ def __init__(; symbols: __init__, _to_page_indices, init_kv_buffer, backup_from_device_all_layer, touching `__init__, _to_page_indices, init_kv_buffer`
  - `python/sglang/test/kl_multiturn_utils.py` modified +40/-19 (59 lines); hunks: -2,6 +2,7  @@ from __future__ import annotations; -145,30 +146,45  @@ def _interleave_order(n: int, branches_per_group: int) -> list[int] | None:; symbols: _interleave_order, test_input_output_logprobs_match_decode_cache_hit_helper, touching `_interleave_order, test_input_output_logprobs_match_decode_cache_hit_helper`
  - `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +7/-0 (7 lines); hunks: -325,6 +325,7  @@ def build_deepseek_v4_hicache_stack(; -357,6 +358,7  @@ def build_deepseek_v4_hicache_stack(; symbols: build_deepseek_v4_hicache_stack, touching `build_deepseek_v4_hicache_stack`
  - `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` modified +17/-3 (20 lines); hunks: -92,8 +92,13  @@ def _assert_dsv4_decode_cached_tokens(result, history_len, output_len, label):; -129,15 +134,15  @@ def setUpClass(cls):; symbols: _assert_dsv4_decode_cached_tokens, setUpClass, tearDownClass, touching `_assert_dsv4_decode_cached_tokens, setUpClass, tearDownClass`
  - `test/registered/radix_cache/test_unified_radix_cache_kl.py` modified +4/-0 (4 lines); hunks: -49,6 +49,8  @@ class UnifiedRadixTreeTestMixin:; -163,6 +165,8  @@ def test_multiturn_decode_cache_hit_branching(self):; symbols: UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching, touching `UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching`
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
-            alloc_func(
-                (num_host_pages, self.item_bytes),
diff -- python/sglang/test/kl_multiturn_utils.py
@@ -2,6 +2,7 @@
+import time
@@ -145,30 +146,45 @@ def _interleave_order(n: int, branches_per_group: int) -> list[int] | None:
-    base_url, inputs, max_new_tokens, order=None, sampling_temperature: float = 1
+    base_url,
+    inputs,
+    max_new_tokens,
+    order=None,
+    sampling_temperature: float = 1,
diff -- python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py
@@ -325,6 +325,7 @@ def build_deepseek_v4_hicache_stack(
+        layout=server_args.hicache_mem_layout,
@@ -357,6 +358,7 @@ def build_deepseek_v4_hicache_stack(
+            layout=server_args.hicache_mem_layout,
@@ -368,6 +370,7 @@ def build_deepseek_v4_hicache_stack(
+            layout=server_args.hicache_mem_layout,
@@ -378,6 +381,7 @@ def build_deepseek_v4_hicache_stack(
+            layout=server_args.hicache_mem_layout,
@@ -388,6 +392,7 @@ def build_deepseek_v4_hicache_stack(
```

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/memory_pool_host.py` modified +333/-92; `python/sglang/test/kl_multiturn_utils.py` modified +40/-19; `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` modified +7/-0
  - tests: `test/registered/radix_cache/test_unified_radix_cache_kl_hicache.py` modified +17/-3; `test/registered/radix_cache/test_unified_radix_cache_kl.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/test/kl_multiturn_utils.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #25733 - [Bug] Fix V4-Pro NaN on Blackwell by converting fp8_einsum input scale to ue8m0

- Link: https://github.com/sgl-project/sglang/pull/25733
- Status/date: merged / 2026-05-19
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@78cb38ed5` history, and the GitHub Pull Request files API; associated commit `79ea30d1f134`.
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix V4-Pro NaN on Blackwell by converting fp8_einsum input scale to ue8m0"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v4.py`; technical summary: Covers "[Bug] Fix V4-Pro NaN on Blackwell by converting fp8_einsum input scale to ue8m0" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` modified +1/-0 (1 lines); hunks: -623,6 +623,7  @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` modified +1/-0 (1 lines); hunks: -623,6 +623,7  @@ def forward(; symbols: forward, touching `forward`
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -623,6 +623,7 @@ def forward(
+            o_s = deep_gemm.ceil_to_ue8m0(o_s)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v4.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.
