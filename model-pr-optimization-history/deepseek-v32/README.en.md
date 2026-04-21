# SGLang DeepSeek V3.2 Support and Optimization Timeline

This document is based on SGLang `origin/main` snapshot `929e00eea`, sgl-cookbook `origin/main` snapshot `8ec4d03`, and patch-level reading of DeepSeek V3.2 / DSA / NSA merged and open PRs. The scope covers DeepSeek-V3.2-Exp, DeepSeek-V3.2, DeepSeek-V3.2-Speciale, DeepSeek-V3.2-NVFP4, DeepSeek-V3.2-MXFP4, and their SGLang evolution across DSA/NSA, Indexer, KV cache, Context Parallel, MTP, IndexCache, DSML parser, AMD/NPU/Blackwell backends, tests, and docs.

Conclusion: as of `929e00eea`, DeepSeek V3.2 is a separate DeepSeek architecture path. It enters the DSA/NSA sparse-attention stack through `is_deepseek_nsa(config)`. The current main runtime entry is `DeepseekV32ForCausalLM`, and the core runtime surface is `nsa_indexer.py`, `nsa_backend.py`, `deepseek_v2.py`, `deepseek_nextn.py`, and `server_args.py`. Main already supports base DSA, NSA backend auto-selection, BF16/FP8 KV cache, MTP, CP, PP, DP attention, TRTLLM/FlashMLA/FA3/TileLang/AITER backends, NVFP4, AMD MXFP4/FP8 KV, NPU, HiSparse/HiCache, IndexCache, and the DSML tool parser. Additional runtime items include CP all-reduce fusion, `moe_dp_size=1` with `attention_cp_size`, adaptive EAGLE, PCG plus speculative decoding, shared `deepseek_nextn.py` changes, and thinking-token radix-cache stripping. The main open tracks are NSA PCG, spec-v2 adaptive spec, CPU/GPU sparse KV scheduling, TP-SP, V3.2 DCP, AMD CP round-robin, IndexCache/top-k backend work, short-sequence dense fallback, partial JSON parsing, and HiCache/3FS.

## 1. Timeline Overview

| Created | PR | State | Track | Effect |
| --- | ---: | --- | --- | --- |
| 2025-09-25 | [#10912](https://github.com/sgl-project/sglang/pull/10912) | merged | PD | Added PD support for hybrid models including DeepSeek V3.2 Exp. |
| 2025-09-29 | [#11061](https://github.com/sgl-project/sglang/pull/11061) | merged | bring-up | Supported DeepSeek V3.2 Exp and added NSA backend, Indexer, sparse attention plumbing, KV quant/dequant, memory pool, runner, and tests. |
| 2025-10-03 | [#11191](https://github.com/sgl-project/sglang/pull/11191) | open | sparse KV | Supports CPU/GPU KV-cache scheduling for GQA/DSA sparse attention. |
| 2025-10-12 | [#11510](https://github.com/sgl-project/sglang/pull/11510) | merged | bugfix | Fixed Qwen3/DSV3/DSV3.2 model support. |
| 2025-10-15 | [#11652](https://github.com/sgl-project/sglang/pull/11652) | merged | MTP | Added MTP for DSV3.2. |
| 2025-10-20 | [#11877](https://github.com/sgl-project/sglang/pull/11877) | merged | docs | Added DeepSeek V3.2 documentation. |
| 2025-10-21 | [#11936](https://github.com/sgl-project/sglang/pull/11936) | merged | NSA tests | Added V3.2 NSA backend testing. |
| 2025-10-24 | [#12044](https://github.com/sgl-project/sglang/pull/12044) | merged | Indexer | Enabled the mixed-type LayerNorm kernel for NSA Indexer. |
| 2025-10-24 | [#12065](https://github.com/sgl-project/sglang/pull/12065) | merged | CP | Added initial Context Parallel support for DeepSeek V3.2 DSA. |
| 2025-10-25 | [#12123](https://github.com/sgl-project/sglang/pull/12123) | merged | template | Fixed dict/string handling for tool arguments in DeepSeek templates. |
| 2025-10-28 | [#12296](https://github.com/sgl-project/sglang/pull/12296) | merged | docs | Updated `deepseek_v32.md`. |
| 2025-11-08 | [#12868](https://github.com/sgl-project/sglang/pull/12868) | merged | docs | Documented V3.2 MHA short-sequence prefill. |
| 2025-11-20 | [#13646](https://github.com/sgl-project/sglang/pull/13646) | merged | TP/DP attention | Enabled pure TP and partial DP attention. |
| 2025-11-23 | [#13812](https://github.com/sgl-project/sglang/pull/13812) | merged | Indexer perf | Optimized NSA Indexer K/S buffer access with fused Triton kernels. |
| 2025-11-26 | [#13959](https://github.com/sgl-project/sglang/pull/13959) | merged | CP perf | Optimized CP with fused MoE, multi-batch, and FP8 KV cache. |
| 2025-12-06 | [#14541](https://github.com/sgl-project/sglang/pull/14541) | merged | NPU CP | Added V3.2 CP support for NPU. |
| 2025-12-07 | [#14572](https://github.com/sgl-project/sglang/pull/14572) | merged | NPU perf | Added V3.2 NPU optimizations. |
| 2025-12-14 | [#15088](https://github.com/sgl-project/sglang/pull/15088) | merged | MTP tests | Added pure TP + MTP testing. |
| 2025-12-17 | [#15307](https://github.com/sgl-project/sglang/pull/15307) | merged | spec overlap | Supported overlap speculative decoding plus NSA. |
| 2025-12-18 | [#15381](https://github.com/sgl-project/sglang/pull/15381) | merged | NPU | Added NPU MLA prolog support. |
| 2025-12-27 | [#15938](https://github.com/sgl-project/sglang/pull/15938) | merged | env cleanup | Cleaned V3.2 environment variables. |
| 2025-12-30 | [#16119](https://github.com/sgl-project/sglang/pull/16119) | merged | CP bugfix | Fixed V3.2 CP issues. |
| 2025-12-30 | [#16156](https://github.com/sgl-project/sglang/pull/16156) | merged | CP guard | Added assertion for V3.2 CP in PD decode mode. |
| 2026-01-02 | [#16305](https://github.com/sgl-project/sglang/pull/16305) | merged | V32/CP updates | Added multiple DeepSeek V32 and CP updates. |
| 2026-01-02 | [#16306](https://github.com/sgl-project/sglang/pull/16306) | merged | refactor | Refactored DeepSeek attention backend handlers and forward-method definitions. |
| 2026-01-04 | [#16380](https://github.com/sgl-project/sglang/pull/16380) | merged | PP/CP | Supported and optimized PP when context pipeline is enabled. |
| 2026-01-07 | [#16637](https://github.com/sgl-project/sglang/pull/16637) | merged | Indexer overlap | Overlapped Indexer `weights_proj` during dual-stream decode. |
| 2026-01-11 | [#16907](https://github.com/sgl-project/sglang/pull/16907) | merged | AWQ loading | Fixed DeepSeek-V3.2-AWQ loading. |
| 2026-01-12 | [#16916](https://github.com/sgl-project/sglang/pull/16916) | merged | docs | Added V3.2 CP+PP documentation. |
| 2026-01-12 | [#16961](https://github.com/sgl-project/sglang/pull/16961) | merged | MTP perf | Optimized MTP decode CUDA batch sizes and NSA implementation. |
| 2026-01-13 | [#16990](https://github.com/sgl-project/sglang/pull/16990) | merged | NPU bugfix | Fixed V3.2 weight-cast bug. |
| 2026-01-13 | [#17007](https://github.com/sgl-project/sglang/pull/17007) | merged | NPU bugfix | Fixed V3.2 and DSVL2 NPU issues. |
| 2026-01-14 | [#17076](https://github.com/sgl-project/sglang/pull/17076) | merged | Indexer/FA3 | Fixed sliced indexer and FA3 padding when CUDA graph cannot run. |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | MoE tuning | Added H20/H20-3E fused MoE configs for V3.1/V3.2. |
| 2026-01-23 | [#17657](https://github.com/sgl-project/sglang/pull/17657) | merged | NVFP4 | Updated tests and docs for V3.2 NVFP4 checkpoint. |
| 2026-01-23 | [#17662](https://github.com/sgl-project/sglang/pull/17662) | merged | TRTLLM NSA | Fixed TRT-LLM NSA in target_verify/draft_extend. |
| 2026-01-25 | [#17688](https://github.com/sgl-project/sglang/pull/17688) | merged | Indexer overlap | Overlapped Indexer q/k projection and activation quantization. |
| 2026-01-26 | [#17783](https://github.com/sgl-project/sglang/pull/17783) | merged | AMD docs | Updated V3.2 AMD GPU docs and unified the ROCm TileLang build. |
| 2026-02-05 | [#18280](https://github.com/sgl-project/sglang/pull/18280) | merged | CP scale | Added CP support for `get_index_k_scale_buffer`. |
| 2026-02-07 | [#18389](https://github.com/sgl-project/sglang/pull/18389) | merged | NVFP4/TRTLLM | Added NSA TRTLLM sparse MLA FP8 support for DeepSeek V3.2 NVFP4. |
| 2026-02-10 | [#18553](https://github.com/sgl-project/sglang/pull/18553) | merged | bugfix | Fixed a V3.2 bug. |
| 2026-02-11 | [#18613](https://github.com/sgl-project/sglang/pull/18613) | merged | CP default | Changed default CP token split to `round-robin-split`. |
| 2026-02-16 | [#18876](https://github.com/sgl-project/sglang/pull/18876) | merged | MoE tune | Added DeepSeek3.2 and GLM-MoE-DSA into MoE tuning. |
| 2026-02-17 | [#18931](https://github.com/sgl-project/sglang/pull/18931) | merged | FP8 KV | Fixed NSA FP8 KV-cache path for both-TRTLLM MHA one-shot. |
| 2026-02-18 | [#18978](https://github.com/sgl-project/sglang/pull/18978) | merged | AMD MTP | Fixed MI35x V3.2 MTP nightly. |
| 2026-02-19 | [#19016](https://github.com/sgl-project/sglang/pull/19016) | merged | spec bugfix | Fixed NSA backend page-table overflow in target_verify. |
| 2026-02-20 | [#19041](https://github.com/sgl-project/sglang/pull/19041) | merged | quality | Avoided FP32 precision loss in `weights_proj`. |
| 2026-02-20 | [#19062](https://github.com/sgl-project/sglang/pull/19062) | merged | MTP/CP | Fixed MTP and CP compatibility. |
| 2026-02-21 | [#19122](https://github.com/sgl-project/sglang/pull/19122) | merged | MLA refactor | Migrated DeepSeek MLA forward methods. |
| 2026-02-22 | [#19148](https://github.com/sgl-project/sglang/pull/19148) | merged | JIT kernel | Added NSA fused-store Indexer K-cache JIT kernel. |
| 2026-02-25 | [#19319](https://github.com/sgl-project/sglang/pull/19319) | merged | 128K bugfix | Fixed `get_k_and_s_triton` for 128K sequence length. |
| 2026-02-25 | [#19367](https://github.com/sgl-project/sglang/pull/19367) | merged | MTP/CP | Fixed NSA CP position mismatch in EAGLE NextN. |
| 2026-02-26 | [#19428](https://github.com/sgl-project/sglang/pull/19428) | merged | qlora/ag | Added `mla_ag_after_qlora` for V3.2. |
| 2026-02-28 | [#19536](https://github.com/sgl-project/sglang/pull/19536) | merged | MTP metadata | Optimized NSA backend metadata under MTP. |
| 2026-03-05 | [#19945](https://github.com/sgl-project/sglang/pull/19945) | merged | AMD TileLang | Added TileLang sparse forward for V3.2 MI355/MI300. |
| 2026-03-07 | [#20086](https://github.com/sgl-project/sglang/pull/20086) | merged | NVFP4 default | Changed V3.2 NVFP4 TP4 defaults. |
| 2026-03-11 | [#20326](https://github.com/sgl-project/sglang/pull/20326) | merged | docs | Added DSA/NSA attention backend to the support matrix. |
| 2026-03-12 | [#20438](https://github.com/sgl-project/sglang/pull/20438) | merged | CP perf | Overlapped NSA-CP key all-gather with query computation. |
| 2026-03-13 | [#20492](https://github.com/sgl-project/sglang/pull/20492) | merged | EAGLE3/DP | Fixed DeepSeek Eagle3 in Attn-DP mode. |
| 2026-03-15 | [#20606](https://github.com/sgl-project/sglang/pull/20606) | merged | FP8 KV offset | Computed `topk_indices_offset` for flashmla_sparse with FP8 KV cache. |
| 2026-03-18 | [#20840](https://github.com/sgl-project/sglang/pull/20840) | merged | AMD accuracy | Fixed V3.2 accuracy on MI355. |
| 2026-03-20 | [#20984](https://github.com/sgl-project/sglang/pull/20984) | merged | FP4 test | Fixed DeepSeek V32 FP4 test. |
| 2026-03-20 | [#21003](https://github.com/sgl-project/sglang/pull/21003) | merged | revert | Reverted `#20984`. |
| 2026-03-23 | [#21192](https://github.com/sgl-project/sglang/pull/21192) | merged | CP tests | Fixed CP in-seq-split and updated tests. |
| 2026-03-24 | [#21249](https://github.com/sgl-project/sglang/pull/21249) | merged | CP/all-reduce | Supported all-reduce fusion with context parallel. |
| 2026-03-24 | [#21259](https://github.com/sgl-project/sglang/pull/21259) | merged | HiCache | Added mooncake backend support for DSA and mamba hybrid models. |
| 2026-03-24 | [#21337](https://github.com/sgl-project/sglang/pull/21337) | merged | B200+DP perf | Added a workaround for DSA performance drop on B200 + DP. |
| 2026-03-25 | [#21405](https://github.com/sgl-project/sglang/pull/21405) | merged | IndexCache | Enabled IndexCache for DeepSeek V3.2. |
| 2026-03-26 | [#21468](https://github.com/sgl-project/sglang/pull/21468) | merged | NPU docs | Updated V3.2 NPU deployment docs. |
| 2026-03-27 | [#21511](https://github.com/sgl-project/sglang/pull/21511) | merged | AMD FP8 KV | Enabled FP8 KV cache and FP8 attention kernel for NSA TileLang. |
| 2026-03-28 | [#21585](https://github.com/sgl-project/sglang/pull/21585) | merged | CI | Moved V3.2 CP test to the DeepEP suite. |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | MTP/spec | Added adaptive `speculative_num_steps` for EAGLE top-k=1. |
| 2026-03-31 | [#21783](https://github.com/sgl-project/sglang/pull/21783) | merged | TRTLLM prefill | Supported TRTLLM sparse MLA kernel for DSA prefill batches. |
| 2026-04-02 | [#21914](https://github.com/sgl-project/sglang/pull/21914) | merged | Blackwell default | Set TRTLLM kernels as the default for Blackwell DSA. |
| 2026-04-03 | [#22003](https://github.com/sgl-project/sglang/pull/22003) | merged | CP topology | Supported `moe_dp_size = 1` with different `attention_cp_size` values. |
| 2026-04-03 | [#22065](https://github.com/sgl-project/sglang/pull/22065) | merged | HiSparse guard | Temporarily allowed HiSparse only for DSA models. |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | PCG/spec | Allowed piecewise CUDA graph to run with speculative decoding. |
| 2026-04-06 | [#22179](https://github.com/sgl-project/sglang/pull/22179) | merged | docs | Improved DeepSeek V3.2 / GLM-5 docs. |
| 2026-04-07 | [#22232](https://github.com/sgl-project/sglang/pull/22232) | merged | Indexer perf | Reduced unnecessary kernels and copies in NSA Indexer. |
| 2026-04-07 | [#22258](https://github.com/sgl-project/sglang/pull/22258) | merged | AMD perf | Added BF16 passthrough from RMSNorm in AMD/HIP NSA to avoid FP8 dequantization. |
| 2026-04-08 | [#22372](https://github.com/sgl-project/sglang/pull/22372) | merged | Hopper FP8 KV | Added Hopper FP8 FlashMLA KV padding. |
| 2026-04-08 | [#22390](https://github.com/sgl-project/sglang/pull/22390) | merged | AR fusion | Enabled all-reduce fusion for DSA models. |
| 2026-04-09 | [#22424](https://github.com/sgl-project/sglang/pull/22424) | merged | AMD LayerNorm | Used AITER CK LayerNorm2D to reduce NSA Indexer kernel launches. |
| 2026-04-09 | [#22425](https://github.com/sgl-project/sglang/pull/22425) | merged | HiSparse CI | Added HiSparse-DSA nightly CI. |
| 2026-04-09 | [#22430](https://github.com/sgl-project/sglang/pull/22430) | merged | DSA bugfix | Fixed several DSA model bugs. |
| 2026-04-15 | [#22850](https://github.com/sgl-project/sglang/pull/22850) | merged | AMD Indexer perf | Fused weights projection and K-cache store to reduce NSA Indexer kernels. |
| 2026-04-16 | [#22914](https://github.com/sgl-project/sglang/pull/22914) | merged | CP refactor | Deduplicated NSA utils into CP utils. |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | reasoning cache | Explored parser-gated two-phase reasoning radix-cache stripping and closed. |
| 2026-04-20 | [#23219](https://github.com/sgl-project/sglang/pull/23219) | merged | shared NextN | Enabled MTP for GLM-5 MXFP4 by touching shared `deepseek_nextn.py`. |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | reasoning cache | Added opt-in thinking-token stripping from radix cache. |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | spec v2 | Extends adaptive speculative decoding to spec v2. |
| 2026-04-21 | [#23351](https://github.com/sgl-project/sglang/pull/23351) | open | PCG | Supports piecewise CUDA graph with NSA. |

## 1.1 V3.2-Related PRs Outside the Main Timeline

Additional PRs directly related to V3.2 / DSA / NSA / tool parsing / platform backends include:

- Early bring-up polish: `#11063`, `#11194`, `#11308`, `#11309`, `#11450`, `#11557`, `#11565`, `#11682`, `#11815`, and `#11835`. These cover the V3.2 tool template, fast-topk, basic tests, KV-cache estimation, NSA act-quant kernels, default config, `_get_logits_head_gate` torch.compile, Indexer cleanup, ragged fast-topk transform, and MTP CI.
- Short-sequence MHA / Indexer fixes: `#11892`, `#12094`, `#12582`, `#12583`, `#12645`, `#12788`, `#12816`, `#12964`, `#13022`, `#13459`, and `#13544`. These add adaptive MHA short-sequence prefill, Indexer `wk+weight_proj` fusion, top-k row starts, Indexer accuracy fixes, KV-buffer shape fixes, B200 short-sequence MHA, extend-without-spec logits skipping, MHA FP8, NSA `torch.cat` compile, FP32 Indexer weight projection, and centralized NSA dispatch.
- DSML / tool / parser path: `#14304`, `#14307`, `#14353`, `#14573`, `#14750`, `#15064`, `#15278`, `#16091`, `#17951`, `#18126`, and `#18174`. These cover OpenAI developer role, DS32 role support, encoder error handling, no-parameter function streaming, function-call parameter streamlining, default drop_thinking, streaming tool-call output, JSON argument streaming, tool-call nightly tests, `encode_messages` fixes, and malformed-JSON tolerance.
- NSA backend / metadata / sparse-cache work: `#14781`, `#14901`, `#15040`, `#15086`, `#15242`, `#15429`, `#16520`, `#16758`, `#16841`, `#17205`, `#17554`, and `#18319`. These cover multi-step speculative metadata, prefill TBO, paged-MQA logits metadata initialization, PP plus radix-cache assertion fixes, FlashMLA sparse FP8, V1 MTP fixes, BaseIndexerMetadata methods, TRTLLM NSA BF16 KV, AMD CUDA graph / FP8 RMSNorm, Indexer `weight_proj` MMA optimization, NSA multi-spec fused metadata kernels, and AMD TileLang default NSA dispatch.
- HiSparse / HiCache and platform fixes: `#14741`, `#17409`, `#17518`, `#17523`, `#17633`, `#18297`, `#18526`, `#20343`, `#21932`, and `#22238`. These connect sparse interface work, fused-MoE config lookup, AMD dtype mismatch, MI325 CI, Transformers v5 compatibility, AITER NSA CUDA graph, HiSparse, decode-backup scheduling, and HiSparse docs.
- Additional open PRs: `#14332`, `#14524`, `#15322`, `#18094`, `#18542`, `#19987`, `#20534`, `#21623`, `#22792`, and `#23268`. They track V32 tool parsing without DSML tags, NSA backend tests, `o_proj` TP, V3.2 PCG, EAGLE3 plus NSA CP aux-hidden-state issues, TileLang NSA FP8 KV, CP prefill-gather FP8 K/K-scale, `encoding_dsv32.py` tests, AITER `indexer_k_quant_and_cache`, and NPU NSA CP plus prefix-cache accuracy.
- Closed / superseded history: `#11109`, `#11596`, `#11761`, `#12017`, `#12052`, `#13531`, `#13546`, `#14619`, `#14904`, `#15051`, `#15217`, `#15310`, `#15807`, `#16079`, `#16881`, `#17024`, `#17199`, `#17310`, and `#17647`. Do not count them as current support, but keep them as context when tracing history or successor PRs.
- Runtime additions: `#21249` and `#22003` cover CP/all-reduce and topology constraints; `#21599`, `#22128`, and `#23336` cover adaptive speculative decoding and PCG combination; `#23219` is GLM-5 MXFP4-specific but touches shared `deepseek_nextn.py`, so it belongs to DSA/NextN-adjacent history; `#22950` and `#23315` distinguish closed/current thinking radix-cache stripping.

## 2. What V3.2 Really Is: The DSA/NSA Runtime Surface

The largest difference between V3.2 and V3/R1 is attention. When the model config satisfies `is_deepseek_nsa(config)`, SGLang treats it as a DSA/NSA model. Docs often call it DSA, while the code mostly uses NSA.

`#11061` is the foundation of V3.2 support. It did not merely add `DeepseekV32ForCausalLM`; it introduced the full sparse-attention path:

- model config detects DSA/NSA and reads fields such as `index_topk`, indexer head count, and index head dim.
- `server_args.py` sets `attention_backend = "nsa"` for DSA.
- `nsa_backend.py` adds `NativeSparseAttnBackend`.
- `nsa_indexer.py` adds the Indexer that generates sparse-attention top-k indices.
- `transform_index.py` and Triton/TileLang kernels convert top-k into paged/ragged structures required by backends.
- `quant_k_cache.py` / `dequant_k_cache.py` handle FP8 K cache.
- memory pool, model runner, CUDA graph, and forward-batch metadata all gained NSA-specific fields.

This is why V3.2 performance or correctness issues usually cannot be solved by reading only `deepseek_v2.py`. The real call chain is: model layer calls the Indexer to generate or reuse top-k, NSA backend builds metadata from top-k and cache seqlens, and execution is dispatched to TRTLLM, FlashMLA, FA3, TileLang, or AITER.

## 3. Server Args: Defaults Are Part of V3.2 Support

Current `server_args.py` has dedicated logic for DSA models:

- If no attention backend is specified, `attention_backend` becomes `nsa`.
- If `SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` is not user-set, it defaults to the model `index_topk`.
- DSA KV cache dtype defaults to `fp8_e4m3` on SM100 and `bfloat16` on other devices.
- The main DSA path allows only `bfloat16` or `fp8_e4m3` KV cache dtype.
- ROCm defaults NSA prefill/decode to TileLang.
- FP8 KV + SM100 defaults to TRTLLM.
- FP8 KV + Hopper can use `flashmla_kv`.
- BF16 KV + SM100 can use `flashmla_sparse` + `trtllm`, and BF16 KV + Hopper can use `flashmla_sparse` + `fa3`.

This means performance comparisons must record `--kv-cache-dtype`, `--nsa-prefill-backend`, `--nsa-decode-backend`, and hardware. If the user specifies an NSA backend but leaves KV dtype as `auto`, defaults can change the result.

## 4. NSA Indexer: The Most Heavily Optimized V3.2 Area

`nsa_indexer.py` is the core DSA hotspot. It:

- computes q/k-related projections from hidden states.
- computes indexing weights through `weights_proj`.
- applies LayerNorm, RoPE, and activation quantization.
- generates top-k sparse indices.
- handles key all-gather, rerange, and round-robin/in-seq split under CP.
- writes or quantizes K cache.

Indexer optimization has been dense:

- `#12044` enabled mixed-type LayerNorm kernel to avoid type conversion and extra kernels.
- `#13812` optimized K/S buffer access with fused Triton kernels.
- `#16637` overlapped `weights_proj` during dual-stream decode.
- `#17688` overlapped q/k projection and activation quantization.
- `#19041` avoided FP32 precision loss in `weights_proj`; it is a quality fix and also affects performance triage.
- `#19148` added JIT NSA fused store for Indexer K cache.
- `#19319` fixed `get_k_and_s_triton` for 128K sequence length.
- `#22232` reduced extra kernels and copies in the Indexer.
- `#22424` used AITER CK LayerNorm2D on AMD.
- `#22850` fused `weights_proj` and K-cache store on AMD.

If V3.2 has slow first token, abnormal decode kernel count, 128K long-context errors, FP8 KV scale issues, or too many Indexer kernels on AMD, the first place to read is usually `nsa_indexer.py` and the backend kernels.

## 5. NSA Backend: Metadata, Top-K Transform, and Sparse MLA

`NativeSparseAttnBackend` in `nsa_backend.py` converts Indexer results into metadata that attention backends can execute. Important fields include:

- `nsa_cache_seqlens_int32`: cache seqlens clipped to top-k.
- `nsa_cu_seqlens_q` / `nsa_cu_seqlens_k`: cumulative seqlens for prefill/decode.
- `nsa_seqlens_expanded`: expanded real seqlens.
- `nsa_extend_seq_lens_list`: CPU-side extend lengths.
- FlashMLA metadata, paged MQA schedule, page table, and top-k offsets.

`#11936` brought NSA backend into testing; `#18389` added TRTLLM sparse MLA FP8 for NVFP4; `#18931` fixed the both-TRTLLM MHA one-shot FP8 KV path; `#21783` added TRTLLM sparse MLA for DSA prefill batches; `#21914` made Blackwell default to TRTLLM kernels; `#22372` handled Hopper FP8 FlashMLA KV padding.

If top-k indices look correct but attention output is wrong, the issue is often not the Indexer. It is usually top-k transform, cache seqlens, page-table offsets, or FP8 KV padding.

## 6. Context Parallel, PP, and DP Attention

`#12065` started V3.2 CP. It changed server args, pynccl, parallel state, NSA utils/backend, communicator, DP attention, schedule policy, CUDA graph, forward batch, `deepseek_v2.py`, `deepseek_nextn.py`, docs, and tests. CP therefore spans scheduling, attention metadata, model forward, and communication.

Later CP evolution includes:

- `#13959` supported fused MoE, multi-batch, and FP8 KV cache.
- `#16119` fixed CP bugs.
- `#16156` asserted V3.2 CP in PD decode mode.
- `#16380` supported and optimized PP when context pipeline is enabled.
- `#18613` changed the default CP token split to `round-robin-split`.
- `#20438` overlapped NSA-CP key all-gather with query computation.
- `#21192` fixed CP `in-seq-split` and updated tests.
- `#21249` supported all-reduce fusion with CP by changing `communicator.py`, `flashinfer_comm_fusion.py`, `model_runner.py`, and server args.
- `#22003` let `moe_dp_size = 1` work with different `attention_cp_size` values; read `parallel_state.py`, `dp_attention.py`, and CP utils together.
- `#22914` deduplicated NSA utils into CP utils.

Current constraints to remember:

- `round-robin-split` is the default CP token split.
- `in-seq-split` requires DeepEP and requires `ep_size == tp_size`.
- CP is restricted in PD decode mode.
- CP only makes sense for V3.2/DSA cases where `is_deepseek_nsa(config)` is true.
- all-reduce fusion and CP should no longer be treated as categorically exclusive; check the active communicator and fusion backend.

Open `#20360` and `#20531` show that AMD CP round-robin and ragged gather still have edge cases. Open `#17185` and `#19609` point to TP `o_proj` and TP Indexer weight work under CP NSA.

## 7. MTP and Speculative Decoding: NSA Metadata Is in the Critical Path

V3.2 MTP starts with `#11652`, but stabilization required many follow-ups:

- `#15088` added pure TP + MTP testing.
- `#15307` supported overlap spec + NSA.
- `#16961` optimized MTP decode CUDA batch sizes and NSA implementation.
- `#17662` fixed TRTLLM NSA in `target_verify` / `draft_extend`.
- `#19016` fixed page-table overflow in speculative target_verify.
- `#19062` fixed MTP + CP compatibility.
- `#19367` fixed NSA CP position mismatch in EAGLE NextN.
- `#19536` optimized NSA backend metadata under MTP.
- `#20492` fixed DeepSeek Eagle3 in Attn-DP mode.
- `#21599` made EAGLE top-k=1 draft steps adaptive.
- `#22128` allowed PCG to coexist with speculative decoding.
- `#23219` is GLM-5 MXFP4 MTP work, but it edits shared `deepseek_nextn.py`, so read it as DSA/NextN-adjacent history.
- Open `#23336` carries adaptive spec into spec-v2 workers.

So V3.2 MTP bugs cannot be debugged by looking only at `deepseek_nextn.py`. They may come from NSA metadata precompute, target verify, draft extend, page table, CP positions, DP attention, or backend auto-selection. Open `#20809` still tracks adding `DeepseekV32ForCausalLM` to MTP draft model mapping.

## 8. Quantization and Platform Backends: NVFP4, AMD, NPU, HiSparse

V3.2 has several platform tracks:

NVFP4 / Blackwell:

- `#17657` updated V3.2 NVFP4 checkpoint tests and docs.
- `#18389` added NSA TRTLLM sparse MLA FP8 support.
- `#20086` changed V3.2 NVFP4 defaults under TP4.
- `#21914` made TRTLLM kernels the Blackwell default.

AMD / ROCm:

- `#17783` updated AMD GPU docs and unified the ROCm TileLang build.
- `#19945` added TileLang sparse forward on MI355/MI300.
- `#20840` fixed MI355 accuracy.
- `#21511` enabled FP8 KV cache and FP8 attention kernel for NSA TileLang.
- `#22258` passed BF16 through from RMSNorm to avoid FP8 dequantization.
- `#22424` used CK LayerNorm2D to reduce Indexer kernel launches.
- `#22850` fused weights projection and K-cache store.

NPU:

- `#14541` added V3.2 CP.
- `#14572` added NPU optimizations.
- `#15381` supported NPU MLA prolog.
- `#16990` fixed a weight-cast bug.
- `#17007` fixed V3.2 / DSVL2 NPU bugs.
- `#21468` updated NPU deployment docs.

HiSparse / HiCache:

- `#21259` made mooncake backend support DSA and mamba hybrid models.
- `#22065` restricted HiSparse checks to DSA models.
- `#22425` added HiSparse-DSA nightly CI.
- open `#23241` continues 3FS backend support for DSA/mamba.

## 9. IndexCache: Skipping Top-K in Some Layers

`#21405` enabled IndexCache for V3.2. Current `deepseek_v2.py` sets:

- `skip_topk`: whether the current layer skips top-k and reuses previous top-k.
- `next_skip_topk`: whether the next layer will reuse the current layer's top-k.
- `index_topk_freq`: frequency-based top-k computation.
- `index_topk_pattern`: explicit per-layer compute/skip pattern.

Without a pattern, the logic computes top-k periodically using `index_topk_freq`. With a pattern, `S` means skip and other characters mean compute. `test_deepseek_v32_indexcache.py` covers both `index_topk_freq=4` and a long `index_topk_pattern`, using GSM8K threshold `0.935` to ensure top-k reuse does not break accuracy.

IndexCache changes more than speed. It changes which layers compute top-k and which layers reuse it, so model accuracy validation is required.

## 10. DSML Tool Parser and Reasoning Interaction

Standard DeepSeek V3.2 uses `DeepSeekV32Detector`, with DSML tool-call format:

```text
<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>
```

The detector supports two parameter forms:

- XML parameter tags.
- Direct JSON inside the invoke block.

The streaming parser emits the tool name once, then computes argument diffs from previous arguments and the common prefix. `_parse_parameters_from_xml(..., allow_partial=True)` uses a partial JSON parser to handle unclosed JSON or unclosed parameter tags.

One cookbook detail matters: the DeepSeek-V3.2-Exp tool path may use `deepseekv31` with `tool_chat_template_deepseekv32.jinja`, while standard DeepSeek-V3.2 uses `--tool-call-parser deepseekv32` and removes the custom chat template. DeepSeek-V3.2-Speciale does not support tool calling.

Open `#21179` says the reasoning parser may swallow V3.2 tool-call markers. Open `#21546` fixes exception handling for malformed JSON during partial parsing.

## 10.1 Thinking Radix Cache: Cache Semantics Outside the Parser

For V3.2 reasoning/tool output, also check whether prefix cache reuses thinking tokens. `#22950` is the closed early parser-gated reasoning-cache strip design; current main is `#23315`, which adds an opt-in flag in `server_args.py` and supports stripping thinking tokens from radix-cache entries in `schedule_batch.py` and `mem_cache/common.py`.

This is a different layer from DSML parsing. If V3.2 tool-call markers are swallowed by the reasoning parser, inspect `deepseekv32_detector.py` / `reasoning_parser.py`; if multi-turn requests reuse an unintended `<think>` / `</think>` prefix, inspect the `#23315` radix-cache stripping path.

## 11. Current Validation Surface and Open PRs

Current validation surface:

- `test/registered/8-gpu-models/test_deepseek_v32.py`: DP8, DP8+MTP, TP8, TP8+MTP, with `deepseekv32` tool parser and `deepseek-v3` reasoning parser.
- `test_deepseek_v32_nsa_backends` inside the same file: tests `flashmla_sparse+flashmla_kv`, `fa3+fa3`, and FP8 KV cache on H200.
- B200 GPQA test inside the same file: reasoning mode, GPQA baseline `0.83`.
- `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`: `index_topk_pattern` and `index_topk_freq=4`.
- `test/manual/test_deepseek_chat_templates.py`: DeepSeek V3/V3.1/V3.2 template argument types.

Open PRs to track:

- `#11191`: DSA sparse attention and CPU/GPU KV-cache scheduling.
- `#12820`: TP-SP support for DeepSeek V2/V3/V3.2.
- `#16148`: V3.2 W4AFP8 MTP using FP8 draft model.
- `#17185`: TP `o_proj` linear in context-parallel NSA.
- `#17761`: missing Assistant token after V3.1/V3.2 tool output.
- `#18167`: V3.2 DCP.
- `#18275`: NPU allgather after qlora.
- `#18733`: V3.2 PD disaggregation test.
- `#19211`: extract `DeepseekV32Mixin` to reduce V3.2/NSA complexity inside `deepseek_v2.py`.
- `#19299`: O(1) expert weight matching in the DeepSeek weight loader.
- `#19609`: TP Indexer weight in NSA attention.
- `#19975`: context parallel for V3.2 on AMD.
- `#20360`: AMD CP round-robin-split output corruption.
- `#20531`: NSA Indexer ragged gather batch-view mismatch in CP round-robin.
- `#20809`: add `DeepseekV32ForCausalLM` to MTP draft model mapping.
- `#20880`: reject HiCache L3 at NSA model initialization.
- `#21179`: preserve V3.2 tool-call markers during reasoning parsing.
- `#21194`: AMD AITER gfx95 `PPMissingLayer` fix for DeepSeek paths.
- `#21506`: V3.2 NPU torch compile.
- `#21529`: ROCm DeepSeek-architecture MXFP4 / Quark W4A4.
- `#21530`: ROCm DeepSeek-variant fused MLA decode RoPE.
- `#21546`: catch MalformedJSON during partial function-call parsing.
- `#21889`: AMD TileLang NSA FP4 KV cache quantization.
- `#22268`: DeepSeek MLA `prepare_qkv_latent` bypassing LoRA adapters.
- `#22473`: dense MLA decode fallback for short sequences.
- `#22774`: MUSA backend support for DeepSeek V2/V3/R1-class layers.
- `#22851`: add `--nsa-topk-backend` and integrate FlashInfer/PyTorch top-k.
- `#22865`: extend the sparsity framework for non-NSA sparse algorithms.
- `#22938`: restore MI300X DeepSeek MLA paths after the MLA refactor.
- `#23195`: guard DeepSeek MLA `.weight` access for AWQ/compressed-tensors.
- `#23241`: 3FS backend support for DSA/mamba.
- `#23257`: `DeepseekV2MoE` double-reduce with CuteDSL EP plus DP attention.
- `#23336`: spec-v2 adaptive speculative decoding.
- `#23351`: NSA piecewise CUDA graph.
