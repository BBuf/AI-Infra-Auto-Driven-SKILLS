# DeepSeek V3.2 PR History

Snapshot:

- SGLang `origin/main`: `929e00eea`
- sgl-cookbook `origin/main`: `8ec4d03`
- Date: `2026-04-21`

This history covers DeepSeek V3.2 only. DeepSeek V3/R1 runtime internals and V3.1 parser/template differences are tracked by separate skills.

## Chronological Timeline

| Date | PR | State | Area | Main effect |
| --- | ---: | --- | --- | --- |
| 2025-09-25 | [#10912](https://github.com/sgl-project/sglang/pull/10912) | merged | PD | Added PD support for hybrid models including DeepSeek V3.2 Exp. |
| 2025-09-29 | [#11061](https://github.com/sgl-project/sglang/pull/11061) | merged | bring-up | Added DeepSeek V3.2 Exp, NSA backend, indexer, sparse attention plumbing, memory pool, model runner, and tests. |
| 2025-10-03 | [#11191](https://github.com/sgl-project/sglang/pull/11191) | open | sparse KV scheduling | Tracks CPU/GPU KV cache scheduling for GQA/DSA sparse attention. |
| 2025-10-12 | [#11510](https://github.com/sgl-project/sglang/pull/11510) | merged | bugfix | Fixed Qwen3/DSV3/DSV3.2 model support. |
| 2025-10-15 | [#11652](https://github.com/sgl-project/sglang/pull/11652) | merged | MTP | Added MTP for DSV3.2. |
| 2025-10-20 | [#11877](https://github.com/sgl-project/sglang/pull/11877) | merged | docs | Added DeepSeek V3.2 docs. |
| 2025-10-21 | [#11936](https://github.com/sgl-project/sglang/pull/11936) | merged | NSA tests | Added V3.2 NSA backend testing. |
| 2025-10-24 | [#12044](https://github.com/sgl-project/sglang/pull/12044) | merged | indexer | Enabled mixed-type LayerNorm kernel for NSA indexer. |
| 2025-10-24 | [#12065](https://github.com/sgl-project/sglang/pull/12065) | merged | CP | Added initial context parallel support for DeepSeek V3.2 DSA. |
| 2025-10-25 | [#12123](https://github.com/sgl-project/sglang/pull/12123) | merged | template | Fixed dict/string argument type handling in DeepSeek templates. |
| 2025-10-28 | [#12296](https://github.com/sgl-project/sglang/pull/12296) | merged | docs | Updated `deepseek_v32.md`. |
| 2025-11-08 | [#12868](https://github.com/sgl-project/sglang/pull/12868) | merged | docs | Documented MHA short-sequence prefill for V3.2. |
| 2025-11-20 | [#13646](https://github.com/sgl-project/sglang/pull/13646) | merged | TP/DP attention | Enabled pure TP and partial DP attention for V3.2. |
| 2025-11-23 | [#13812](https://github.com/sgl-project/sglang/pull/13812) | merged | indexer perf | Optimized NSA indexer K/S buffer access with fused Triton kernels. |
| 2025-11-26 | [#13959](https://github.com/sgl-project/sglang/pull/13959) | merged | CP perf | Optimized context parallel with fused MoE, multi-batch, and FP8 KV cache. |
| 2025-12-06 | [#14541](https://github.com/sgl-project/sglang/pull/14541) | merged | NPU CP | Added V3.2 CP support for NPU. |
| 2025-12-07 | [#14572](https://github.com/sgl-project/sglang/pull/14572) | merged | NPU perf | Added V3.2 NPU optimizations. |
| 2025-12-14 | [#15088](https://github.com/sgl-project/sglang/pull/15088) | merged | MTP tests | Added pure TP plus MTP test. |
| 2025-12-17 | [#15307](https://github.com/sgl-project/sglang/pull/15307) | merged | spec overlap | Supported overlap speculative decoding plus NSA. |
| 2025-12-18 | [#15381](https://github.com/sgl-project/sglang/pull/15381) | merged | NPU | Added NPU MLA prolog support for V3.2. |
| 2025-12-27 | [#15938](https://github.com/sgl-project/sglang/pull/15938) | merged | env cleanup | Cleaned V3.2 environment variables. |
| 2025-12-30 | [#16119](https://github.com/sgl-project/sglang/pull/16119) | merged | CP bugfix | Fixed V3.2 CP issues. |
| 2025-12-30 | [#16156](https://github.com/sgl-project/sglang/pull/16156) | merged | CP guard | Asserted V3.2 CP in PD decode mode. |
| 2026-01-02 | [#16305](https://github.com/sgl-project/sglang/pull/16305) | merged | V32/CP updates | Added multiple V3.2 and context-parallel updates. |
| 2026-01-02 | [#16306](https://github.com/sgl-project/sglang/pull/16306) | merged | refactor | Refactored DeepSeek attention backend handlers and forward definitions. |
| 2026-01-04 | [#16380](https://github.com/sgl-project/sglang/pull/16380) | merged | PP/CP | Supported and optimized pipeline parallelism when context pipeline is enabled. |
| 2026-01-07 | [#16637](https://github.com/sgl-project/sglang/pull/16637) | merged | indexer overlap | Overlapped indexer `weights_proj` during dual-stream decode. |
| 2026-01-11 | [#16907](https://github.com/sgl-project/sglang/pull/16907) | merged | AWQ loading | Fixed DeepSeek-V3.2-AWQ model loading. |
| 2026-01-12 | [#16916](https://github.com/sgl-project/sglang/pull/16916) | merged | docs | Added V3.2 CP+PP documentation. |
| 2026-01-12 | [#16961](https://github.com/sgl-project/sglang/pull/16961) | merged | MTP perf | Optimized MTP decode CUDA batch sizes and NSA implementation. |
| 2026-01-13 | [#16990](https://github.com/sgl-project/sglang/pull/16990) | merged | NPU bugfix | Fixed V3.2 weight-cast bug on Ascend. |
| 2026-01-13 | [#17007](https://github.com/sgl-project/sglang/pull/17007) | merged | NPU bugfix | Fixed V3.2 and DSVL2 NPU issues. |
| 2026-01-14 | [#17076](https://github.com/sgl-project/sglang/pull/17076) | merged | indexer/FA3 bugfix | Fixed sliced indexer and FA3 padding when CUDA graph cannot run. |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | MoE tuning | Added H20/H20-3E fused MoE configs for V3.1/V3.2. |
| 2026-01-23 | [#17657](https://github.com/sgl-project/sglang/pull/17657) | merged | NVFP4 | Updated tests and docs for V3.2 NVFP4 checkpoint. |
| 2026-01-23 | [#17662](https://github.com/sgl-project/sglang/pull/17662) | merged | TRTLLM NSA | Fixed TRT-LLM NSA in target_verify and draft_extend. |
| 2026-01-25 | [#17688](https://github.com/sgl-project/sglang/pull/17688) | merged | indexer overlap | Overlapped indexer q/k projection and activation quantization. |
| 2026-01-26 | [#17783](https://github.com/sgl-project/sglang/pull/17783) | merged | AMD docs | Updated V3.2 AMD GPU docs and unified ROCm TileLang build. |
| 2026-02-05 | [#18280](https://github.com/sgl-project/sglang/pull/18280) | merged | CP scale buffer | Added CP support for `get_index_k_scale_buffer`. |
| 2026-02-07 | [#18389](https://github.com/sgl-project/sglang/pull/18389) | merged | NVFP4/TRTLLM | Added NSA TRTLLM sparse MLA FP8 support for V3.2 NVFP4. |
| 2026-02-10 | [#18553](https://github.com/sgl-project/sglang/pull/18553) | merged | bugfix | Fixed V3.2 bug. |
| 2026-02-11 | [#18613](https://github.com/sgl-project/sglang/pull/18613) | merged | CP default | Changed default CP token split method to `round-robin-split`. |
| 2026-02-16 | [#18876](https://github.com/sgl-project/sglang/pull/18876) | merged | MoE tune | Added DeepSeek3.2 and GLM-MoE-DSA into MoE tuning. |
| 2026-02-17 | [#18931](https://github.com/sgl-project/sglang/pull/18931) | merged | FP8 KV | Fixed NSA FP8 KV cache path for both-TRTLLM MHA one-shot. |
| 2026-02-18 | [#18978](https://github.com/sgl-project/sglang/pull/18978) | merged | AMD MTP | Fixed MI35x V3.2 MTP nightly. |
| 2026-02-19 | [#19016](https://github.com/sgl-project/sglang/pull/19016) | merged | spec bugfix | Fixed NSA backend page-table overflow in target_verify. |
| 2026-02-20 | [#19041](https://github.com/sgl-project/sglang/pull/19041) | merged | quality | Avoided FP32 precision loss in `weights_proj`. |
| 2026-02-20 | [#19062](https://github.com/sgl-project/sglang/pull/19062) | merged | MTP/CP | Fixed MTP and CP compatibility. |
| 2026-02-21 | [#19122](https://github.com/sgl-project/sglang/pull/19122) | merged | MLA refactor | Migrated MLA forward method out of `deepseek_v2.py`. |
| 2026-02-22 | [#19148](https://github.com/sgl-project/sglang/pull/19148) | merged | JIT kernel | Added JIT NSA fused store for indexer K cache. |
| 2026-02-25 | [#19319](https://github.com/sgl-project/sglang/pull/19319) | merged | 128K bugfix | Fixed `get_k_and_s_triton` for 128K sequence case. |
| 2026-02-25 | [#19367](https://github.com/sgl-project/sglang/pull/19367) | merged | MTP/CP | Fixed NSA CP position mismatch in EAGLE NextN. |
| 2026-02-26 | [#19428](https://github.com/sgl-project/sglang/pull/19428) | merged | qlora/ag | Added `mla_ag_after_qlora` feature for V3.2. |
| 2026-02-28 | [#19536](https://github.com/sgl-project/sglang/pull/19536) | merged | MTP metadata | Optimized NSA backend metadata under MTP. |
| 2026-03-05 | [#19945](https://github.com/sgl-project/sglang/pull/19945) | merged | AMD TileLang | Added TileLang sparse forward for V3.2 MI355/MI300. |
| 2026-03-07 | [#20086](https://github.com/sgl-project/sglang/pull/20086) | merged | NVFP4 default | Changed V3.2 NVFP4 default setting on TP4. |
| 2026-03-11 | [#20326](https://github.com/sgl-project/sglang/pull/20326) | merged | docs | Added DSA/NSA attention backend to support matrix. |
| 2026-03-12 | [#20438](https://github.com/sgl-project/sglang/pull/20438) | merged | CP perf | Overlapped NSA-CP key all-gather with query computation. |
| 2026-03-13 | [#20492](https://github.com/sgl-project/sglang/pull/20492) | merged | EAGLE3/DP | Fixed DeepSeek EAGLE3 in attention-DP mode. |
| 2026-03-15 | [#20606](https://github.com/sgl-project/sglang/pull/20606) | merged | FP8 KV offset | Computed `topk_indices_offset` for flashmla_sparse with FP8 KV cache. |
| 2026-03-18 | [#20840](https://github.com/sgl-project/sglang/pull/20840) | merged | AMD accuracy | Fixed V3.2 accuracy on MI355. |
| 2026-03-20 | [#20984](https://github.com/sgl-project/sglang/pull/20984) | merged | FP4 test | Fixed V3.2 FP4 test. |
| 2026-03-20 | [#21003](https://github.com/sgl-project/sglang/pull/21003) | merged | revert | Reverted the V3.2 FP4 test fix. |
| 2026-03-23 | [#21192](https://github.com/sgl-project/sglang/pull/21192) | merged | CP tests | Fixed CP in-seq-split and updated tests. |
| 2026-03-24 | [#21249](https://github.com/sgl-project/sglang/pull/21249) | merged | CP/all-reduce | Supported all-reduce fusion with context parallel. |
| 2026-03-24 | [#21259](https://github.com/sgl-project/sglang/pull/21259) | merged | HiCache | Added mooncake backend support for DSA and mamba hybrid models. |
| 2026-03-24 | [#21337](https://github.com/sgl-project/sglang/pull/21337) | merged | B200+DP perf | Added workaround for DSA performance drop on B200 + DP. |
| 2026-03-25 | [#21405](https://github.com/sgl-project/sglang/pull/21405) | merged | IndexCache | Enabled IndexCache for DeepSeek V3.2. |
| 2026-03-26 | [#21468](https://github.com/sgl-project/sglang/pull/21468) | merged | NPU docs | Updated V3.2 NPU deployment docs. |
| 2026-03-27 | [#21511](https://github.com/sgl-project/sglang/pull/21511) | merged | AMD FP8 KV | Enabled FP8 KV cache and FP8 attention kernel for NSA TileLang. |
| 2026-03-28 | [#21585](https://github.com/sgl-project/sglang/pull/21585) | merged | CI | Moved V3.2 CP test to DeepEP suite. |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | MTP/spec | Added adaptive `speculative_num_steps` for EAGLE top-k=1. |
| 2026-03-31 | [#21783](https://github.com/sgl-project/sglang/pull/21783) | merged | TRTLLM prefill | Supported TRTLLM sparse MLA kernel for DSA prefill batches. |
| 2026-04-02 | [#21914](https://github.com/sgl-project/sglang/pull/21914) | merged | Blackwell default | Set TRTLLM kernels as default for Blackwell DSA. |
| 2026-04-03 | [#22003](https://github.com/sgl-project/sglang/pull/22003) | merged | CP topology | Supported `moe_dp_size = 1` across different `attention_cp_size` values. |
| 2026-04-03 | [#22065](https://github.com/sgl-project/sglang/pull/22065) | merged | HiSparse guard | Restricted HiSparse checks to DSA models. |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | PCG/spec | Allowed piecewise CUDA graph with speculative decoding. |
| 2026-04-06 | [#22179](https://github.com/sgl-project/sglang/pull/22179) | merged | docs | Improved DeepSeek V3.2 and GLM-5 docs. |
| 2026-04-07 | [#22232](https://github.com/sgl-project/sglang/pull/22232) | merged | indexer perf | Reduced unnecessary kernels and copies in NSA indexer. |
| 2026-04-07 | [#22258](https://github.com/sgl-project/sglang/pull/22258) | merged | AMD perf | Added BF16 passthrough from RMSNorm to avoid FP8 dequantization. |
| 2026-04-08 | [#22372](https://github.com/sgl-project/sglang/pull/22372) | merged | Hopper FP8 KV | Added Hopper FP8 FlashMLA KV padding. |
| 2026-04-08 | [#22390](https://github.com/sgl-project/sglang/pull/22390) | merged | all-reduce fusion | Enabled all-reduce fusion for DSA models. |
| 2026-04-09 | [#22424](https://github.com/sgl-project/sglang/pull/22424) | merged | AMD layernorm | Used AITER CK LayerNorm2D for NSA indexer. |
| 2026-04-09 | [#22425](https://github.com/sgl-project/sglang/pull/22425) | merged | HiSparse CI | Added HiSparse-DSA nightly CI. |
| 2026-04-09 | [#22430](https://github.com/sgl-project/sglang/pull/22430) | merged | DSA bugfix | Fixed several DSA model bugs. |
| 2026-04-15 | [#22850](https://github.com/sgl-project/sglang/pull/22850) | merged | AMD indexer perf | Fused weights projection and K-cache store to reduce NSA indexer kernels. |
| 2026-04-16 | [#22914](https://github.com/sgl-project/sglang/pull/22914) | merged | CP refactor | Deduplicated NSA utils into CP utils. |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | reasoning cache | Explored parser-gated two-phase radix-cache stripping for reasoning tokens. |
| 2026-04-20 | [#23219](https://github.com/sgl-project/sglang/pull/23219) | merged | shared NextN | Enabled MTP for GLM-5 MXFP4 by touching shared `deepseek_nextn.py` infrastructure. |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | reasoning cache | Added opt-in stripping of thinking tokens from radix cache. |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | spec v2 | Extends adaptive speculative decoding to spec v2 EAGLE workers. |
| 2026-04-21 | [#23351](https://github.com/sgl-project/sglang/pull/23351) | open | PCG | Adds piecewise CUDA graph support with NSA. |

## Additional PR Coverage

Additional all-state PR coverage includes V3.2-relevant work that the first timeline did not enumerate one by one:

- Early bring-up polish: [#11063](https://github.com/sgl-project/sglang/pull/11063), [#11194](https://github.com/sgl-project/sglang/pull/11194), [#11308](https://github.com/sgl-project/sglang/pull/11308), [#11309](https://github.com/sgl-project/sglang/pull/11309), [#11450](https://github.com/sgl-project/sglang/pull/11450), [#11557](https://github.com/sgl-project/sglang/pull/11557), [#11565](https://github.com/sgl-project/sglang/pull/11565), [#11682](https://github.com/sgl-project/sglang/pull/11682), [#11815](https://github.com/sgl-project/sglang/pull/11815), and [#11835](https://github.com/sgl-project/sglang/pull/11835).
- Short-sequence MHA / Indexer fixes: [#11892](https://github.com/sgl-project/sglang/pull/11892), [#12094](https://github.com/sgl-project/sglang/pull/12094), [#12582](https://github.com/sgl-project/sglang/pull/12582), [#12583](https://github.com/sgl-project/sglang/pull/12583), [#12645](https://github.com/sgl-project/sglang/pull/12645), [#12788](https://github.com/sgl-project/sglang/pull/12788), [#12816](https://github.com/sgl-project/sglang/pull/12816), [#12964](https://github.com/sgl-project/sglang/pull/12964), [#13022](https://github.com/sgl-project/sglang/pull/13022), [#13459](https://github.com/sgl-project/sglang/pull/13459), and [#13544](https://github.com/sgl-project/sglang/pull/13544).
- DSML/tool/parser path: [#14304](https://github.com/sgl-project/sglang/pull/14304), [#14307](https://github.com/sgl-project/sglang/pull/14307), [#14353](https://github.com/sgl-project/sglang/pull/14353), [#14573](https://github.com/sgl-project/sglang/pull/14573), [#14750](https://github.com/sgl-project/sglang/pull/14750), [#15064](https://github.com/sgl-project/sglang/pull/15064), [#15278](https://github.com/sgl-project/sglang/pull/15278), [#16091](https://github.com/sgl-project/sglang/pull/16091), [#17951](https://github.com/sgl-project/sglang/pull/17951), [#18126](https://github.com/sgl-project/sglang/pull/18126), and [#18174](https://github.com/sgl-project/sglang/pull/18174).
- NSA backend / metadata / sparse-cache work: [#14781](https://github.com/sgl-project/sglang/pull/14781), [#14901](https://github.com/sgl-project/sglang/pull/14901), [#15040](https://github.com/sgl-project/sglang/pull/15040), [#15086](https://github.com/sgl-project/sglang/pull/15086), [#15242](https://github.com/sgl-project/sglang/pull/15242), [#15429](https://github.com/sgl-project/sglang/pull/15429), [#16520](https://github.com/sgl-project/sglang/pull/16520), [#16758](https://github.com/sgl-project/sglang/pull/16758), [#16841](https://github.com/sgl-project/sglang/pull/16841), [#17205](https://github.com/sgl-project/sglang/pull/17205), [#17554](https://github.com/sgl-project/sglang/pull/17554), and [#18319](https://github.com/sgl-project/sglang/pull/18319).
- HiSparse/HiCache and platform fixes: [#14741](https://github.com/sgl-project/sglang/pull/14741), [#17409](https://github.com/sgl-project/sglang/pull/17409), [#17518](https://github.com/sgl-project/sglang/pull/17518), [#17523](https://github.com/sgl-project/sglang/pull/17523), [#17633](https://github.com/sgl-project/sglang/pull/17633), [#18297](https://github.com/sgl-project/sglang/pull/18297), [#18526](https://github.com/sgl-project/sglang/pull/18526), [#20343](https://github.com/sgl-project/sglang/pull/20343), [#21932](https://github.com/sgl-project/sglang/pull/21932), and [#22238](https://github.com/sgl-project/sglang/pull/22238).
- Additional open PRs: [#14332](https://github.com/sgl-project/sglang/pull/14332), [#14524](https://github.com/sgl-project/sglang/pull/14524), [#15322](https://github.com/sgl-project/sglang/pull/15322), [#18094](https://github.com/sgl-project/sglang/pull/18094), [#18542](https://github.com/sgl-project/sglang/pull/18542), [#19987](https://github.com/sgl-project/sglang/pull/19987), [#20534](https://github.com/sgl-project/sglang/pull/20534), [#21623](https://github.com/sgl-project/sglang/pull/21623), [#22792](https://github.com/sgl-project/sglang/pull/22792), and [#23268](https://github.com/sgl-project/sglang/pull/23268).
- Closed or superseded experiments to cite as history, not current support: [#11109](https://github.com/sgl-project/sglang/pull/11109), [#11596](https://github.com/sgl-project/sglang/pull/11596), [#11761](https://github.com/sgl-project/sglang/pull/11761), [#12017](https://github.com/sgl-project/sglang/pull/12017), [#12052](https://github.com/sgl-project/sglang/pull/12052), [#13531](https://github.com/sgl-project/sglang/pull/13531), [#13546](https://github.com/sgl-project/sglang/pull/13546), [#14619](https://github.com/sgl-project/sglang/pull/14619), [#14904](https://github.com/sgl-project/sglang/pull/14904), [#15051](https://github.com/sgl-project/sglang/pull/15051), [#15217](https://github.com/sgl-project/sglang/pull/15217), [#15310](https://github.com/sgl-project/sglang/pull/15310), [#15807](https://github.com/sgl-project/sglang/pull/15807), [#16079](https://github.com/sgl-project/sglang/pull/16079), [#16881](https://github.com/sgl-project/sglang/pull/16881), [#17024](https://github.com/sgl-project/sglang/pull/17024), [#17199](https://github.com/sgl-project/sglang/pull/17199), [#17310](https://github.com/sgl-project/sglang/pull/17310), and [#17647](https://github.com/sgl-project/sglang/pull/17647).
- Round-2 runtime additions: [#21249](https://github.com/sgl-project/sglang/pull/21249) and [#22003](https://github.com/sgl-project/sglang/pull/22003) are CP/all-reduce topology updates; [#21599](https://github.com/sgl-project/sglang/pull/21599), [#22128](https://github.com/sgl-project/sglang/pull/22128), and open [#23336](https://github.com/sgl-project/sglang/pull/23336) are speculative-decoding updates; [#23219](https://github.com/sgl-project/sglang/pull/23219) is GLM-5-specific but touches shared `deepseek_nextn.py`; [#22950](https://github.com/sgl-project/sglang/pull/22950) and [#23315](https://github.com/sgl-project/sglang/pull/23315) define the closed/current reasoning radix-cache split.

## Code-Level Narrative

### 1. Bring-up and server defaults

[#11061](https://github.com/sgl-project/sglang/pull/11061) is the foundational V3.2 Exp bring-up. It added the model-config detection, NSA backend, indexer, top-k transform, K-cache quant/dequant, memory-pool changes, model runner and CUDA-graph plumbing, `deepseek_v2.py` integration, `server_args.py` defaults, and tests.

Current `server_args.py` treats DSA specially. If `is_deepseek_nsa(hf_config)` is true, it sets the attention backend to `nsa`, sets the dense-attention threshold env var to the model `index_topk` when not user-set, chooses DSA KV cache dtype, and selects NSA prefill/decode backends by hardware and dtype.

### 2. NSA indexer and sparse attention backend

`nsa_indexer.py` is the core of DSA. It computes q/k projections, applies the indexer weights projection, produces top-k indices, handles CP all-gather and rerange, and can quantize/store K cache. Performance work repeatedly targeted this file:

- `#12044` enabled mixed-type LayerNorm.
- `#13812` fused K/S buffer access.
- `#16637` overlapped `weights_proj` in dual-stream decode.
- `#17688` overlapped q/k projection and activation quantization.
- `#19041` avoided FP32 precision loss in `weights_proj`.
- `#19148` added JIT fused K-cache store.
- `#22232` reduced extra kernels and copies.
- `#22424` used AITER CK LayerNorm2D on AMD.
- `#22850` fused AMD weights projection and K-cache store.

`nsa_backend.py` then consumes those indices and metadata. It owns `NativeSparseAttnBackend`, computes `nsa_cache_seqlens_int32`, `nsa_cu_seqlens_q/k`, picks paged or ragged top-k transforms, prepares FlashMLA metadata, handles FP8 K-cache dequantization, and dispatches to `trtllm`, `flashmla_sparse`, `flashmla_kv`, `fa3`, `tilelang`, or `aiter`.

### 3. Context parallel, PP, and DP attention

V3.2 context parallel started in `#12065`. It touched server args, pynccl, parallel state, NSA utils/backend, communicator, DP attention, schedule policy, cuda graph, forward-batch metadata, `deepseek_v2.py`, `deepseek_nextn.py`, docs, and tests.

The CP line then evolved through `#13959`, `#16119`, `#16156`, `#16305`, `#16380`, `#18613`, `#20438`, `#21192`, `#21249`, `#22003`, and `#22914`. Current important rules are:

- `round-robin-split` is the default CP token split method.
- `in-seq-split` requires DeepEP and `ep_size == tp_size`.
- CP is restricted in PD decode mode.
- CP positions must match EAGLE NextN.
- key all-gather can overlap query computation.
- all-reduce fusion can now be used with CP, so inspect `flashinfer_comm_fusion.py`, communicator setup, and `model_runner.py` before treating CP and all-reduce fusion as mutually exclusive.
- `moe_dp_size = 1` with nontrivial `attention_cp_size` is no longer automatically out of scope; check `parallel_state.py`, `dp_attention.py`, and CP utilities for the active topology constraints.
- utilities have been moved toward shared CP utils.

### 4. MTP and speculative decoding

`#11652` added MTP for V3.2, but several later PRs were needed because NSA changes the speculative decoding surface:

- `#15088` added pure TP + MTP testing.
- `#15307` supported overlap speculative decoding plus NSA.
- `#16961` optimized MTP decode batch sizes and NSA implementation.
- `#17662` fixed TRTLLM NSA in `target_verify` and `draft_extend`.
- `#19016` fixed page-table overflow in speculative target_verify.
- `#19062` fixed MTP and CP compatibility.
- `#19367` fixed NSA CP position mismatch in EAGLE NextN.
- `#19536` optimized NSA metadata under MTP.
- `#20492` fixed DeepSeek EAGLE3 in attention-DP mode.
- `#21599` added adaptive EAGLE top-k=1 draft steps, changing the assumption that speculative steps are static.
- `#22128` allowed piecewise CUDA graph with speculative decoding.
- `#23219` is GLM-5 MXFP4-specific, but it edits shared `deepseek_nextn.py`, so read it as DSA/NextN-adjacent history rather than V3.2 checkpoint support.
- Open `#23336` extends adaptive speculative decoding to spec v2 workers.

For V3.2, an MTP bug can also come from NSA metadata, CP positions, page-table offsets, or backend selection.

### 5. Quantized and platform tracks

V3.2 has several platform-specific tracks:

- NVFP4 Blackwell: `#17657`, `#18389`, and `#20086` added docs/tests, TRTLLM sparse MLA FP8 support, and TP4 defaults.
- AMD: `#17783`, `#19945`, `#20840`, `#21511`, `#22258`, and `#22850` cover ROCm docs, TileLang sparse forward, MI355 accuracy, FP8 KV cache, BF16 passthrough, and indexer kernel fusion.
- NPU: `#14541`, `#14572`, `#15381`, `#16990`, `#17007`, and `#21468` cover CP, NPU optimizations, MLA prolog, cast bugs, and deployment docs.
- HiSparse/HiCache: `#21259`, `#22065`, and `#22425` add DSA hybrid support, guard checks, and nightly CI.

### 6. IndexCache

[#21405](https://github.com/sgl-project/sglang/pull/21405) enabled IndexCache for V3.2. Current `deepseek_v2.py` sets `skip_topk` and `next_skip_topk` per layer. Without a pattern, it uses `index_topk_freq`; with `index_topk_pattern`, layers marked `S` skip top-k and reuse previous top-k indices. `test_deepseek_v32_indexcache.py` covers both `index_topk_freq=4` and a long explicit pattern.

### 7. DSML parser

Standard V3.2 uses `DeepSeekV32Detector`, which parses DSML:

```text
<｜DSML｜function_calls>
<｜DSML｜invoke name="tool">
<｜DSML｜parameter name="city" string="true">Beijing</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>
```

It also accepts direct JSON inside an invoke block. Streaming parsing emits the tool name once, keeps previous argument strings, and sends stable-prefix diffs. Open `#21179` and `#21546` show the remaining parser edge cases around reasoning-parser marker preservation and malformed partial JSON.

### 8. Reasoning radix-cache behavior

V3.2 thinking and DSML parsing can interact with prefix-cache reuse. Closed [#22950](https://github.com/sgl-project/sglang/pull/22950) tried parser-gated two-phase reasoning cache stripping across model config, scheduler, radix cache, and reasoning parser code. Merged [#23315](https://github.com/sgl-project/sglang/pull/23315) is the current path: it adds an opt-in server argument and changes `schedule_batch.py` plus `mem_cache/common.py` so thinking tokens can be stripped from radix-cache entries.

This is separate from DSML parsing. A tool-call marker preservation bug belongs to `deepseekv32_detector.py` / `reasoning_parser.py`; a thinking-prefix cache reuse bug belongs to the radix-cache stripping path.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG DeepSeek V3.2 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-09-25 | [#10912](https://github.com/sgl-project/sglang/pull/10912) | merged | [PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp) | attention/backend, scheduler/runtime, tests/benchmarks | `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/decode.py` |
| 2025-09-29 | [#11061](https://github.com/sgl-project/sglang/pull/11061) | merged | Support DeepSeek V3.2 Exp | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-09-29 | [#11063](https://github.com/sgl-project/sglang/pull/11063) | merged | Add DeepSeek-V3.2 Tool Call Template | docs/config | `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2025-09-30 | [#11109](https://github.com/sgl-project/sglang/pull/11109) | closed | [Draft] Support MTP for DeepSeek-V3.2 | attention/backend, scheduler/runtime, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-03 | [#11191](https://github.com/sgl-project/sglang/pull/11191) | open | [Feature] Support Sparse Attention and KV cache scheduling between CPU and GPU for GQA/DSA. | attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm100.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_bwd.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm90.py` |
| 2025-10-03 | [#11194](https://github.com/sgl-project/sglang/pull/11194) | merged | [Feature] Add a fast-topk to sgl-kernel for DeepSeek v3.2 | MoE/router, kernel, tests/benchmarks | `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py` |
| 2025-10-07 | [#11308](https://github.com/sgl-project/sglang/pull/11308) | merged | [CI] Add Basic Test for DeepSeek V3.2 | tests/benchmarks | `test/srt/test_deepseek_v32_basic.py`, `.github/workflows/pr-test.yml`, `scripts/ci/ci_install_dependency.sh` |
| 2025-10-07 | [#11309](https://github.com/sgl-project/sglang/pull/11309) | merged | [DeepSeek-V3.2] Include indexer kv cache when estimating kv cache size | scheduler/runtime | `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py` |
| 2025-10-11 | [#11450](https://github.com/sgl-project/sglang/pull/11450) | merged | [DPSKv3.2] Rewrite nsa tilelang act_quant kernel to triton | attention/backend, quantization, kernel, tests/benchmarks | `test/srt/layers/attention/nsa/test_act_quant_triton.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-12 | [#11510](https://github.com/sgl-project/sglang/pull/11510) | merged | [Bugfix] Fix Qwen3/DSV3/DSV3.2 model support | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks | `.github/workflows/pr-test-npu.yml`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/ascend_backend.py` |
| 2025-10-13 | [#11557](https://github.com/sgl-project/sglang/pull/11557) | merged | Fix DeepSeek-v3.2 default config (ValueError: not enough values to unpack (expected 4, got 3)) | misc | `python/sglang/srt/server_args.py` |
| 2025-10-13 | [#11565](https://github.com/sgl-project/sglang/pull/11565) | merged | [DSv32] Use torch.compile for _get_logits_head_gate | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-14 | [#11596](https://github.com/sgl-project/sglang/pull/11596) | closed | [Spec Decoding] Support MTP for dsv3.2 | attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `.github/workflows/pr-test-amd.yml`, `.github/workflows/release-docker-dev.yml` |
| 2025-10-15 | [#11652](https://github.com/sgl-project/sglang/pull/11652) | merged | [Spec Decoding] Support MTP for dsv3.2 | attention/backend, kernel, scheduler/runtime, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/speculative/draft_utils.py` |
| 2025-10-15 | [#11682](https://github.com/sgl-project/sglang/pull/11682) | merged | Cleaning indexer for DeepSeek V3.2 | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2025-10-17 | [#11761](https://github.com/sgl-project/sglang/pull/11761) | closed | (beta)support context parallel with deepseekv3.2-DSA | misc |  |
| 2025-10-19 | [#11815](https://github.com/sgl-project/sglang/pull/11815) | merged | [DeepseekV32] Add fast_topk_transform_ragged_fused kernel | MoE/router, kernel, tests/benchmarks | `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py` |
| 2025-10-19 | [#11835](https://github.com/sgl-project/sglang/pull/11835) | merged | [CI] Add CI test for DeepSeek V3.2 MTP | tests/benchmarks | `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `python/sglang/srt/server_args.py` |
| 2025-10-20 | [#11877](https://github.com/sgl-project/sglang/pull/11877) | merged | [Doc] Add documentation for DeepSeek V3.2 | docs/config | `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md`, `docs/basic_usage/deepseek_v32.md`, `docs/advanced_features/separate_reasoning.ipynb` |
| 2025-10-21 | [#11892](https://github.com/sgl-project/sglang/pull/11892) | merged | DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill | model wrapper, attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-21 | [#11936](https://github.com/sgl-project/sglang/pull/11936) | merged | [Test] Add dsv3.2 nsa backend testing | tests/benchmarks | `test/srt/test_deepseek_v32_nsabackend.py`, `test/srt/run_suite.py` |
| 2025-10-23 | [#12017](https://github.com/sgl-project/sglang/pull/12017) | closed | (beta)support context parallel with deepseekv3.2-DSA | model wrapper, attention/backend, scheduler/runtime | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/utils/common.py` |
| 2025-10-24 | [#12044](https://github.com/sgl-project/sglang/pull/12044) | merged | Enable mixed type LayerNorm kernel for NSA indexer | attention/backend, tests/benchmarks | `python/sglang/srt/layers/layernorm.py`, `python/sglang/test/test_layernorm.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-24 | [#12052](https://github.com/sgl-project/sglang/pull/12052) | closed | Fix Illegal Instruction/IMA errors when using DP attention with DeepSeek-V3.2 models | attention/backend | `python/sglang/srt/layers/dp_attention.py` |
| 2025-10-24 | [#12065](https://github.com/sgl-project/sglang/pull/12065) | merged | (1/n)support context parallel with deepseekv3.2-DSA | model wrapper, attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-24 | [#12094](https://github.com/sgl-project/sglang/pull/12094) | merged | Fuse wk and weight_proj in Indexer for DeepSeekV3.2-FP4 | model wrapper, attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-25 | [#12123](https://github.com/sgl-project/sglang/pull/12123) | merged | Fix DeepSeek chat templates to handle tool call arguments type checking (#11700) | tests/benchmarks, docs/config | `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-10-28 | [#12296](https://github.com/sgl-project/sglang/pull/12296) | merged | Update deepseek_v32.md | docs/config | `docs/basic_usage/deepseek_v32.md` |
| 2025-11-04 | [#12582](https://github.com/sgl-project/sglang/pull/12582) | merged | [sgl-kernel][Deepseek V3.2] Add row_starts to topk kernel | MoE/router, kernel, tests/benchmarks | `sgl-kernel/tests/test_topk.py`, `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/python/sgl_kernel/top_k.py` |
| 2025-11-04 | [#12583](https://github.com/sgl-project/sglang/pull/12583) | merged | [Deepseek V3.2] Fix accuracy bug in the Indexer | attention/backend, tests/benchmarks | `test/srt/test_deepseek_v32_nsabackend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/srt/test_deepseek_v32_mtp.py` |
| 2025-11-04 | [#12645](https://github.com/sgl-project/sglang/pull/12645) | merged | [Bug] Fix NSA Backend KV-Buffer Shape Mismatch in DeepSeek-V3.2 | scheduler/runtime | `python/sglang/srt/mem_cache/memory_pool.py` |
| 2025-11-06 | [#12788](https://github.com/sgl-project/sglang/pull/12788) | merged | [DeepSeek-V3.2][NSA] Enable MHA Pathway for Short Sequence Prefill on B200 (SM100) | model wrapper, attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-07 | [#12816](https://github.com/sgl-project/sglang/pull/12816) | merged | [Deepseek V3.2] Only skip Indexer logits computation when is_extend_without_speculative | model wrapper, attention/backend, scheduler/runtime | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/model_executor/forward_batch_info.py` |
| 2025-11-08 | [#12868](https://github.com/sgl-project/sglang/pull/12868) | merged | [Docs][DeepseekV3.2] Update deepseekv3.2 docs for mha short seq prefill | docs/config | `docs/basic_usage/deepseek_v32.md` |
| 2025-11-10 | [#12964](https://github.com/sgl-project/sglang/pull/12964) | merged | [DeepseekV3.2] Deepseek fp8 support for MHA path | model wrapper, attention/backend | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-11-10 | [#13022](https://github.com/sgl-project/sglang/pull/13022) | merged | [Deepseek V3.2] Use torch.compile to speed up torch.cat in nsa | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-11-17 | [#13459](https://github.com/sgl-project/sglang/pull/13459) | merged | [Deepseek V3.2] Change indexer weights_proj to fp32 | model wrapper, attention/backend, docs/config | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `docs/basic_usage/deepseek_v32.md` |
| 2025-11-18 | [#13531](https://github.com/sgl-project/sglang/pull/13531) | closed | DeepSeek V3.2 indexer RoPE fix | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-11-18 | [#13544](https://github.com/sgl-project/sglang/pull/13544) | merged | [DeepSeekV3.2] Centralize NSA dispatch logic in NativeSparseAttnBackend | model wrapper, attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-18 | [#13546](https://github.com/sgl-project/sglang/pull/13546) | closed | [Deepseek V3.2] Optimize use of dual_stream in nsa_indexer/attention | model wrapper, attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-20 | [#13646](https://github.com/sgl-project/sglang/pull/13646) | merged | [DeepSeekV3.2] Enable pure TP & Partial DP Attention | attention/backend, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/nightly/test_deepseek_v32_nsabackend.py` |
| 2025-11-23 | [#13812](https://github.com/sgl-project/sglang/pull/13812) | merged | [Performance] Optimize NSA Indexer K/S Buffer Access with Fused Triton Kernels | attention/backend, scheduler/runtime, tests/benchmarks | `test/manual/layers/attention/nsa/test_index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2025-11-26 | [#13959](https://github.com/sgl-project/sglang/pull/13959) | merged | [DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache | model wrapper, attention/backend, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-12-02 | [#14304](https://github.com/sgl-project/sglang/pull/14304) | merged | [FIX][DS32]openai protocol: support openai message role: developer | misc | `python/sglang/srt/entrypoints/openai/protocol.py` |
| 2025-12-02 | [#14307](https://github.com/sgl-project/sglang/pull/14307) | merged | [SMG][DS32][fix] support dsv32, add role developer | MoE/router | `sgl-model-gateway/src/protocols/chat.rs`, `sgl-model-gateway/src/routers/grpc/harmony/builder.rs`, `sgl-model-gateway/src/routers/http/pd_router.rs` |
| 2025-12-03 | [#14332](https://github.com/sgl-project/sglang/pull/14332) | open | feat: V32 tool call parsing for no-dsml tag | tests/benchmarks | `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-03 | [#14353](https://github.com/sgl-project/sglang/pull/14353) | merged | feat(dsv32): better error handling for DeepSeek-v3.2 encoder | misc | `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`, `python/sglang/srt/entrypoints/openai/serving_base.py` |
| 2025-12-06 | [#14524](https://github.com/sgl-project/sglang/pull/14524) | open | [Test] Add test suite for NSA backend | attention/backend, tests/benchmarks | `python/sglang/test/attention/test_nsa_backend.py` |
| 2025-12-06 | [#14541](https://github.com/sgl-project/sglang/pull/14541) | merged | [NPU]dsv3.2 cp for npu | attention/backend, tests/benchmarks | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2025-12-07 | [#14572](https://github.com/sgl-project/sglang/pull/14572) | merged | [NPU] optimization for dsv3.2 | model wrapper, attention/backend, MoE/router, quantization | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-12-07 | [#14573](https://github.com/sgl-project/sglang/pull/14573) | merged | [Tool Call] Fix DeepSeekV32Detector skipping functions with no params in streaming mode | tests/benchmarks | `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-08 | [#14619](https://github.com/sgl-project/sglang/pull/14619) | closed | [Sparse & HICache]: Enables hierarchical sparse KV cache management and scheduling for DeepSeek V32. | model wrapper, attention/backend, kernel, multimodal/processor, scheduler/runtime | `python/sglang/srt/mem_cache/sparsity/ops/triton_kernel.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py` |
| 2025-12-09 | [#14741](https://github.com/sgl-project/sglang/pull/14741) | merged | [1/N][Sparse With Hicache]: Add Sparse Interface | scheduler/runtime | `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py` |
| 2025-12-09 | [#14750](https://github.com/sgl-project/sglang/pull/14750) | merged | [Tool Call][DSV32] Streamline function call parameters | tests/benchmarks | `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-10 | [#14781](https://github.com/sgl-project/sglang/pull/14781) | merged | [Performance] optimize NSA backend metadata computation for multi-step speculative decoding | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2025-12-11 | [#14901](https://github.com/sgl-project/sglang/pull/14901) | merged | fix ds3.2 nsa backend prefill TBO | model wrapper, attention/backend, tests/benchmarks | `test/srt/ep/test_deepep_large.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py` |
| 2025-12-11 | [#14904](https://github.com/sgl-project/sglang/pull/14904) | closed | [DeepSeek V3.2] Proper drop_thinking logic | misc | `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2025-12-13 | [#15040](https://github.com/sgl-project/sglang/pull/15040) | merged | [DSv32] Move deep_gemm.get_paged_mqa_logits_metadata to init time as metadata | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-12-13 | [#15051](https://github.com/sgl-project/sglang/pull/15051) | closed | feat(ds32): support <function_call> tag for deepseek 3.2 tool call | misc | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-13 | [#15064](https://github.com/sgl-project/sglang/pull/15064) | merged | fix: dpskv32 chat history processing, default drop_thinking to true | misc | `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2025-12-13 | [#15086](https://github.com/sgl-project/sglang/pull/15086) | merged | [NSA] Fix NSA backend assertion error when running DeepSeek-V3.2 PP with radix-cache | attention/backend, quantization, scheduler/runtime | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` |
| 2025-12-14 | [#15088](https://github.com/sgl-project/sglang/pull/15088) | merged | [DeepSeekV3.2] Add pure TP+MTP test | tests/benchmarks, docs/config | `test/nightly/test_deepseek_v32_tp.py`, `docs/basic_usage/deepseek_v32.md` |
| 2025-12-16 | [#15217](https://github.com/sgl-project/sglang/pull/15217) | closed | fix(DeepSeek-V3.2 function_call): fix streaming content loss in DeepSeekV32Detector | misc | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-16 | [#15242](https://github.com/sgl-project/sglang/pull/15242) | merged | [sgl-kernel] Update flashmla to include fp8 sparse_mla optimizations | attention/backend, kernel | `sgl-kernel/cmake/flashmla.cmake` |
| 2025-12-16 | [#15278](https://github.com/sgl-project/sglang/pull/15278) | merged | feat: DeepSeek-V3.2 Streaming tool call output | tests/benchmarks | `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/function_call/test_function_call_parser.py` |
| 2025-12-17 | [#15307](https://github.com/sgl-project/sglang/pull/15307) | merged | [Deepseek V3.2] Support Overlap Spec + NSA | attention/backend, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-12-17 | [#15310](https://github.com/sgl-project/sglang/pull/15310) | closed | [Deepseek V3.2] Enable TRTLLM Allreduce Fusion | misc | `python/sglang/srt/server_args.py` |
| 2025-12-17 | [#15322](https://github.com/sgl-project/sglang/pull/15322) | open | dsv32 support o_proj tp | model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/distributed/parallel_state.py` |
| 2025-12-18 | [#15381](https://github.com/sgl-project/sglang/pull/15381) | merged | [NPU]DeepSeek-V3.2 support npu mlaprolog | attention/backend, quantization | `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`, `python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py` |
| 2025-12-19 | [#15429](https://github.com/sgl-project/sglang/pull/15429) | merged | [Deepseek V3.2] Fix Deepseek MTP in V1 mode | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-12-25 | [#15807](https://github.com/sgl-project/sglang/pull/15807) | closed | [2/N][Sparse With Hicache]: Support separating nsa memory management for KV cache and index_k in decode side. | attention/backend, scheduler/runtime | `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/mem_cache/common.py` |
| 2025-12-27 | [#15938](https://github.com/sgl-project/sglang/pull/15938) | merged | Clean Some Environment Variables for DeepSeek V32 | attention/backend, quantization, scheduler/runtime, docs/config | `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2025-12-29 | [#16079](https://github.com/sgl-project/sglang/pull/16079) | closed | [Performance] Change sparse MLA and dense MHA switching threshold DSv3.2 | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-12-29 | [#16091](https://github.com/sgl-project/sglang/pull/16091) | merged | [Tool Call] Stream DeepSeek-V3.2 function call parameters in JSON format. | tests/benchmarks | `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/function_call/test_function_call_parser.py` |
| 2025-12-30 | [#16119](https://github.com/sgl-project/sglang/pull/16119) | merged | [cp] bug fix for dsv3.2 cp | attention/backend | `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2025-12-30 | [#16156](https://github.com/sgl-project/sglang/pull/16156) | merged | [cp] assert dsv3.2 cp in pd decode mode | misc | `python/sglang/srt/server_args.py` |
| 2026-01-02 | [#16305](https://github.com/sgl-project/sglang/pull/16305) | merged | Multiple updates of DeepSeek V32 and context parallel | tests/benchmarks, docs/config | `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `docs/basic_usage/deepseek_v32.md` |
| 2026-01-02 | [#16306](https://github.com/sgl-project/sglang/pull/16306) | merged | [1/n]deepseek_v2.py Refactor: attention backend handlers and forward method definition | model wrapper, attention/backend | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` |
| 2026-01-04 | [#16380](https://github.com/sgl-project/sglang/pull/16380) | merged | [DeepSeek 3.2] Support and optimize pipeline parallelis when context pipeline enabled | attention/backend, scheduler/runtime | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/scheduler_pp_mixin.py` |
| 2026-01-05 | [#16520](https://github.com/sgl-project/sglang/pull/16520) | merged | fix: unimplemented methods in BaseIndexerMetadata | kernel, tests/benchmarks | `test/registered/kernels/test_nsa_indexer.py` |
| 2026-01-07 | [#16637](https://github.com/sgl-project/sglang/pull/16637) | merged | [DSv32] Overlap indexer weights_proj during dual_stream decode | model wrapper, attention/backend | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-08 | [#16758](https://github.com/sgl-project/sglang/pull/16758) | merged | [DeepSeek V3.2] Enable trtllm NSA with bf16 kvcache | attention/backend | `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-01-10 | [#16841](https://github.com/sgl-project/sglang/pull/16841) | merged | [AMD] enable CUDA graph for NSA backend and fix NSA FP8 fused RMSNorm group quant | model wrapper, attention/backend, kernel, scheduler/runtime | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` |
| 2026-01-11 | [#16881](https://github.com/sgl-project/sglang/pull/16881) | closed | [DSv32] Add returning DSA topk indices | model wrapper, attention/backend, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks | `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `python/sglang/srt/managers/detokenizer_manager.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py` |
| 2026-01-11 | [#16907](https://github.com/sgl-project/sglang/pull/16907) | merged | Fix model loading for DeepSeek-V3.2-AWQ | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-01-12 | [#16916](https://github.com/sgl-project/sglang/pull/16916) | merged | add doc for dsv32 cp+pp | docs/config | `docs/basic_usage/deepseek_v32.md` |
| 2026-01-12 | [#16961](https://github.com/sgl-project/sglang/pull/16961) | merged | [DeepSeek v3.2] Opt MTP decode cuda batch sizes and nsa implementation | attention/backend, kernel, scheduler/runtime | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| 2026-01-13 | [#16990](https://github.com/sgl-project/sglang/pull/16990) | merged | [Ascend] fix dsv3.2 weight cast bug | quantization | `python/sglang/srt/layers/quantization/unquant.py` |
| 2026-01-13 | [#17007](https://github.com/sgl-project/sglang/pull/17007) | merged | [NPU]bugfix: fix for dsv3.2 and dsvl2 | model wrapper, attention/backend, tests/benchmarks, docs/config | `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `test/registered/ascend/llm_models/test_ascend_deepseek_v3_2_exp_w8a8.py`, `test/registered/ascend/vlm_models/test_ascend_deepseek_vl2.py` |
| 2026-01-13 | [#17024](https://github.com/sgl-project/sglang/pull/17024) | closed | [PD] Fix DeepSeek V3.2 indexer cache transfer | misc | `python/sglang/srt/disaggregation/prefill.py` |
| 2026-01-14 | [#17076](https://github.com/sgl-project/sglang/pull/17076) | merged | [DeepSeek V3.2] [Bugfix] slice indexer and padding fa3 when can not run cuda graph | attention/backend, kernel, tests/benchmarks | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/kernels/test_nsa_indexer.py` |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab | MoE/router, quantization, kernel, tests/benchmarks, docs/config | `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` |
| 2026-01-16 | [#17199](https://github.com/sgl-project/sglang/pull/17199) | closed | [Feature] add feature mla_ag_after_qlora for dsv3.2 | model wrapper, attention/backend | `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-16 | [#17205](https://github.com/sgl-project/sglang/pull/17205) | merged | [OPT] DeepSeekV3.2: optimize indexer weight_proj-mma performance | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-18 | [#17310](https://github.com/sgl-project/sglang/pull/17310) | closed | [TileLang] Align TileLang NSA kernel with current TileLang and stabilize output | attention/backend, kernel | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` |
| 2026-01-20 | [#17409](https://github.com/sgl-project/sglang/pull/17409) | merged | [Fix]: correctly fetch ds32 config in tuning_fused_moe_triton | MoE/router, kernel, tests/benchmarks | `benchmark/kernels/fused_moe_triton/common_utils.py` |
| 2026-01-21 | [#17518](https://github.com/sgl-project/sglang/pull/17518) | merged | [HotFix]Fix dtype mismatch in nsa indexer on AMD device | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-21 | [#17523](https://github.com/sgl-project/sglang/pull/17523) | merged | [AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI | quantization, tests/benchmarks | `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` |
| 2026-01-22 | [#17554](https://github.com/sgl-project/sglang/pull/17554) | merged | Kernel: optimize decoding metadata in NSA multi-spec backend with fused kernels | attention/backend, kernel, tests/benchmarks | `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_metadata_copy.cuh`, `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py` |
| 2026-01-23 | [#17633](https://github.com/sgl-project/sglang/pull/17633) | merged | [AMD] CI - enable deepseekv3.2 on MI325-8gpu and merge perf/accuracy test suites into stage-b suites | attention/backend, MoE/router, kernel, tests/benchmarks | `.github/workflows/pr-test-amd.yml`, `scripts/ci/utils/slash_command_handler.py`, `test/registered/eval/test_moe_eval_accuracy_large.py` |
| 2026-01-23 | [#17647](https://github.com/sgl-project/sglang/pull/17647) | closed | [Perf] opt nsa backend init forward metada | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py` |
| 2026-01-23 | [#17657](https://github.com/sgl-project/sglang/pull/17657) | merged | [DeepSeek] Update tests and document for DeepSeek V3.2 NVFP4 checkpoint | quantization, tests/benchmarks, docs/config | `test/srt/test_deepseek_v32_fp4_4gpu.py`, `docs/basic_usage/deepseek_v32.md`, `test/srt/run_suite.py` |
| 2026-01-23 | [#17662](https://github.com/sgl-project/sglang/pull/17662) | merged | [DeepSeek-V3.2] Fix TRT-LLM NSA in target_verify/draft_extend | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-01-25 | [#17688](https://github.com/sgl-project/sglang/pull/17688) | merged | [DSv32] Overlap indexer qk projection and activation quant | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-26 | [#17783](https://github.com/sgl-project/sglang/pull/17783) | merged | [AMD] Update dsv3.2 AMD GPU docs and unify ROCm TileLang build | docs/config | `docker/rocm.Dockerfile`, `docs/basic_usage/deepseek_v32.md` |
| 2026-01-29 | [#17951](https://github.com/sgl-project/sglang/pull/17951) | merged | Add tool call tests for DeepSeek V3.2 in nightly CI | model wrapper, scheduler/runtime, tests/benchmarks | `python/sglang/test/tool_call_test_runner.py`, `python/sglang/test/run_combined_tests.py`, `test/registered/8-gpu-models/test_deepseek_v32.py` |
| 2026-02-02 | [#18094](https://github.com/sgl-project/sglang/pull/18094) | open | support deepseekv3.2-piecewise-cuda-graph | model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, docs/config | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2026-02-02 | [#18126](https://github.com/sgl-project/sglang/pull/18126) | merged | Fix dsv32 encode_messages | misc | `python/sglang/srt/parser/jinja_template_utils.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2026-02-03 | [#18174](https://github.com/sgl-project/sglang/pull/18174) | merged | [Bugfix] Catch errors when DeepSeek-V3.2 generates malformed JSON | misc | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2026-02-05 | [#18280](https://github.com/sgl-project/sglang/pull/18280) | merged | [DeepSeek v3.2][Bugfix] get_index_k_scale_buffer support cp | attention/backend, kernel, tests/benchmarks | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/kernels/test_nsa_indexer.py` |
| 2026-02-05 | [#18297](https://github.com/sgl-project/sglang/pull/18297) | merged | Deepseekv32 compatibility with transformers v5 | model wrapper, attention/backend, docs/config | `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-02-05 | [#18319](https://github.com/sgl-project/sglang/pull/18319) | merged | [AMD] Use `tilelang` as default NSA attention backend dispatch on AMD Instinct | attention/backend | `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-02-07 | [#18389](https://github.com/sgl-project/sglang/pull/18389) | merged | Nsa trtllm mla sparse fp8 support with Deepseek v3.2 NVFP4 | model wrapper, attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/utils.py` |
| 2026-02-10 | [#18526](https://github.com/sgl-project/sglang/pull/18526) | merged | [AMD] Enable cudagraph for aiter nsa backend and add aiter impl for nsa pr… | attention/backend, kernel | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py` |
| 2026-02-10 | [#18542](https://github.com/sgl-project/sglang/pull/18542) | open | fix: fixed aux hidden state index out of range when using eagle3 with nsa cp | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-10 | [#18553](https://github.com/sgl-project/sglang/pull/18553) | merged | Fix Bug on dsv3.2 | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/overlap_utils.py` |
| 2026-02-11 | [#18613](https://github.com/sgl-project/sglang/pull/18613) | merged | [V3.2] Change default CP token split method to `--round-robin-split` | docs/config | `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md` |
| 2026-02-16 | [#18876](https://github.com/sgl-project/sglang/pull/18876) | merged | Add DeepSeek3.2 and GlmMoeDsa into moe tune | MoE/router, kernel, tests/benchmarks | `benchmark/kernels/fused_moe_triton/common_utils.py` |
| 2026-02-17 | [#18931](https://github.com/sgl-project/sglang/pull/18931) | merged | Fix NSA FP8 KV cache path for both-trtllm MHA one-shot | model wrapper, attention/backend | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` |
| 2026-02-18 | [#18978](https://github.com/sgl-project/sglang/pull/18978) | merged | [AMD] Fix mi35x dsv32 mtp nightly | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-02-19 | [#19016](https://github.com/sgl-project/sglang/pull/19016) | merged | [FIX] NSA backend page_table overflow in speculative decoding target_verify | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-02-20 | [#19041](https://github.com/sgl-project/sglang/pull/19041) | merged | [DSv32] [GLM5] Improve Model Quality by Avoiding FP32 Precision Loss in `weights_proj` | attention/backend, kernel, tests/benchmarks | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py` |
| 2026-02-20 | [#19062](https://github.com/sgl-project/sglang/pull/19062) | merged | [DSv32] Fix MTP and CP compatibility | model wrapper | `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-02-21 | [#19122](https://github.com/sgl-project/sglang/pull/19122) | merged | [3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py | model wrapper, attention/backend, tests/benchmarks | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` |
| 2026-02-22 | [#19148](https://github.com/sgl-project/sglang/pull/19148) | merged | [DeepSeek-V3.2][JIT-kernel] Support nsa fuse store indexer k cache | attention/backend, kernel, scheduler/runtime | `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh`, `python/sglang/jit_kernel/fused_store_index_cache.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-02-25 | [#19319](https://github.com/sgl-project/sglang/pull/19319) | merged | [deepseekv3.2] fix get_k_and_s_triton kernel for 128K seqlen case bug | attention/backend, kernel, scheduler/runtime, tests/benchmarks | `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py` |
| 2026-02-25 | [#19367](https://github.com/sgl-project/sglang/pull/19367) | merged | Fix NSA CP positions mismatch in eagle NextN model | model wrapper | `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-02-26 | [#19428](https://github.com/sgl-project/sglang/pull/19428) | merged | [Feature] add feature mla_ag_after_qlora for dsv3.2 | model wrapper, attention/backend | `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-28 | [#19536](https://github.com/sgl-project/sglang/pull/19536) | merged | [Perf] Optimize NSA backend metadata under MTP | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py` |
| 2026-03-05 | [#19945](https://github.com/sgl-project/sglang/pull/19945) | merged | [AMD] Tilelang sparse fwd for dsv32 mi355/mi300 | attention/backend, kernel | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` |
| 2026-03-05 | [#19987](https://github.com/sgl-project/sglang/pull/19987) | closed | [AMD] Fix nightly GLM-5 failures: Fix NSA indexer tensor aliasing on ROCm during CUDA graph capture | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-03-07 | [#20086](https://github.com/sgl-project/sglang/pull/20086) | merged | [V32/GLM5] Change default setting of V32 nvfp4 on TP4 | misc | `python/sglang/srt/server_args.py` |
| 2026-03-11 | [#20326](https://github.com/sgl-project/sglang/pull/20326) | merged | [Doc] Add DSA/NSA attention backend to support matrix | attention/backend, docs/config | `docs/advanced_features/attention_backend.md` |
| 2026-03-11 | [#20343](https://github.com/sgl-project/sglang/pull/20343) | merged | HiSparse for Sparse Attention | attention/backend, kernel, multimodal/processor, scheduler/runtime | `python/sglang/srt/managers/hisparse_coordinator.py`, `python/sglang/jit_kernel/csrc/hisparse.cuh`, `python/sglang/srt/mem_cache/hisparse_memory_pool.py` |
| 2026-03-12 | [#20438](https://github.com/sgl-project/sglang/pull/20438) | merged | [Perf] Overlap NSA-CP key all-gather with query computation for DeepSeek-V3.2 | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-03-13 | [#20492](https://github.com/sgl-project/sglang/pull/20492) | merged | [BugFix] bug fix for DeepSeek eagle3 in Attn-DP mode | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-13 | [#20534](https://github.com/sgl-project/sglang/pull/20534) | open | Transfer FP8 K/K_scale for CP indexer prefill gather | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-03-15 | [#20606](https://github.com/sgl-project/sglang/pull/20606) | merged | FIX: (NSA) Compute topk_indices_offset when NSA prefill flashmla_sparse is used with FP8 KV cache | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-03-18 | [#20840](https://github.com/sgl-project/sglang/pull/20840) | merged | [AMD] Fix dpsk-v32 accuracy issue on mi355 | quantization | `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2026-03-20 | [#20984](https://github.com/sgl-project/sglang/pull/20984) | merged | Fix DeepSeek V32 FP4 test | quantization, tests/benchmarks | `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/test_utils.py` |
| 2026-03-20 | [#21003](https://github.com/sgl-project/sglang/pull/21003) | merged | Revert "Fix DeepSeek V32 FP4 test" | quantization, tests/benchmarks | `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/test_utils.py` |
| 2026-03-23 | [#21192](https://github.com/sgl-project/sglang/pull/21192) | merged | Fix CP in-seq-split method for DeepSeek V32 and update related tests | model wrapper, tests/benchmarks | `test/registered/cp/test_deepseek_v32_cp_single_node.py`, `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py`, `python/sglang/srt/server_args.py` |
| 2026-03-24 | [#21249](https://github.com/sgl-project/sglang/pull/21249) | merged | Support allreduce fusion with cp | attention/backend, scheduler/runtime | `python/sglang/srt/layers/flashinfer_comm_fusion.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/communicator.py` |
| 2026-03-24 | [#21259](https://github.com/sgl-project/sglang/pull/21259) | merged | [HiCache & HybridModel] mooncake backend support DSA & mamba model | scheduler/runtime, tests/benchmarks | `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` |
| 2026-03-24 | [#21337](https://github.com/sgl-project/sglang/pull/21337) | merged | Workaround of DSA performance drop on B200 + DP | misc | `python/sglang/srt/server_args.py` |
| 2026-03-25 | [#21405](https://github.com/sgl-project/sglang/pull/21405) | merged | Enable IndexCache for DeepSeek V3.2 | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks | `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-03-26 | [#21468](https://github.com/sgl-project/sglang/pull/21468) | merged | [NPU] Update DeepSeek-V3.2 model deployment instructions in documentation | docs/config | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-03-27 | [#21511](https://github.com/sgl-project/sglang/pull/21511) | merged | [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend | model wrapper, attention/backend, kernel, scheduler/runtime | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/mem_cache/utils.py` |
| 2026-03-28 | [#21585](https://github.com/sgl-project/sglang/pull/21585) | merged | [CI] Move v32 cp test to deepep running suite | tests/benchmarks | `test/registered/cp/test_deepseek_v32_cp_single_node.py` |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1 | kernel, scheduler/runtime, tests/benchmarks, docs/config | `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py` |
| 2026-03-29 | [#21623](https://github.com/sgl-project/sglang/pull/21623) | open | [Test] Add unit tests for encoding_dsv32.py | tests/benchmarks | `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py` |
| 2026-03-31 | [#21783](https://github.com/sgl-project/sglang/pull/21783) | merged | [DSA] Support trtllm sparse mla kernel for prefill batches | attention/backend, tests/benchmarks | `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/test/run_eval.py` |
| 2026-04-02 | [#21914](https://github.com/sgl-project/sglang/pull/21914) | merged | [DSA] Set trtllm kernels as default for Blackwell | misc | `python/sglang/srt/server_args.py` |
| 2026-04-02 | [#21932](https://github.com/sgl-project/sglang/pull/21932) | merged | [HiSparse] Optimize the scheduling of decode backup. | scheduler/runtime | `python/sglang/srt/managers/hisparse_coordinator.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2026-04-03 | [#22003](https://github.com/sgl-project/sglang/pull/22003) | merged | Support moe_dp_size = 1 for various attention_cp_size | model wrapper, attention/backend, MoE/router, tests/benchmarks | `python/sglang/srt/layers/communicator.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/layers/dp_attention.py` |
| 2026-04-03 | [#22065](https://github.com/sgl-project/sglang/pull/22065) | merged | [HiSparse]: Optimize server args checking-HiSparse is temporarily only available for DSA models. | misc | `python/sglang/srt/server_args.py` |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | Allow piecewise CUDA graph with speculative decoding | kernel, scheduler/runtime, tests/benchmarks | `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2026-04-06 | [#22179](https://github.com/sgl-project/sglang/pull/22179) | merged | [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation | docs/config | `docs/basic_usage/deepseek_v32.md` |
| 2026-04-07 | [#22232](https://github.com/sgl-project/sglang/pull/22232) | merged | Reduce unnecessary kernels and copies in the NSA indexer | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-07 | [#22238](https://github.com/sgl-project/sglang/pull/22238) | merged | [HiSparse]: Add readme docs for HiSparse Feature | docs/config | `docs/advanced_features/hisparse_guide.md`, `docs/basic_usage/deepseek_v32.md` |
| 2026-04-07 | [#22258](https://github.com/sgl-project/sglang/pull/22258) | merged | [AMD][HIP] NSA: bf16 passthrough from RMSNorm to eliminate FP8 dequantization | attention/backend | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-08 | [#22372](https://github.com/sgl-project/sglang/pull/22372) | merged | [DSA] Hopper FP8 FlashMLA KV padding | attention/backend, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py` |
| 2026-04-08 | [#22390](https://github.com/sgl-project/sglang/pull/22390) | merged | [DSA] Enable all reduce fusion for DSA models | misc | `python/sglang/srt/server_args.py` |
| 2026-04-09 | [#22424](https://github.com/sgl-project/sglang/pull/22424) | merged | [AMD] Use aiter CK layernorm2d for LayerNorm to reduce NSA indexer kernel launches | attention/backend | `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-09 | [#22425](https://github.com/sgl-project/sglang/pull/22425) | merged | [HiSparse]: Add HiSpares-DSA Model's nightly CI | model wrapper, tests/benchmarks | `test/registered/8-gpu-models/test_dsa_models_hisparse.py` |
| 2026-04-09 | [#22430](https://github.com/sgl-project/sglang/pull/22430) | merged | [Fix] Fix several bugs on DSA models | attention/backend | `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-04-14 | [#22792](https://github.com/sgl-project/sglang/pull/22792) | open | nsa indexer: use aiter indexer_k_quant_and_cache | model wrapper, attention/backend, quantization, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `scripts/ci/utils/diffusion/generate_diffusion_dashboard.py`, `python/tools/get_version_tag.py`, `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` |
| 2026-04-15 | [#22850](https://github.com/sgl-project/sglang/pull/22850) | merged | [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion) | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-16 | [#22914](https://github.com/sgl-project/sglang/pull/22914) | merged | [Refactor] Deduplicate NSA utils.py into cp_utils.py for context parallel | model wrapper, attention/backend, scheduler/runtime | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373) | scheduler/runtime, tests/benchmarks, docs/config | `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py` |
| 2026-04-20 | [#23219](https://github.com/sgl-project/sglang/pull/23219) | merged | [AMD] Enable MTP for GLM-5-mxfp4 model | model wrapper | `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-04-20 | [#23268](https://github.com/sgl-project/sglang/pull/23268) | open | 【NPU】【bugfix】accuracy fix when enable both nsa cp and prefixcache | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | Opt-in strip of thinking tokens from radix cache | scheduler/runtime, tests/benchmarks | `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py` |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | [SPEC V2][2/N] feat: adaptive spec support spec v2 | multimodal/processor, scheduler/runtime | `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py` |
| 2026-04-21 | [#23351](https://github.com/sgl-project/sglang/pull/23351) | open | Support piecewise CUDA graph with NSA | attention/backend, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/piecewise_cuda_graph/test_pcg_glm5_fp4.py`, `python/sglang/srt/layers/layernorm.py` |

## Diff Cards

### PR #10912 - [PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp)

- Link: https://github.com/sgl-project/sglang/pull/10912
- Status/date: `merged`, created 2025-09-25, merged 2025-10-16; author `ShangmingCai`.
- Diff scope read: `13` files, `+727/-186`; areas: attention/backend, scheduler/runtime, tests/benchmarks; keywords: kv, cache, spec, attention, config, scheduler, test, cuda, fp8, lora.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +248/-137 (385 lines); hunks: def __init__(; def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:; symbols: __init__, for, get_speculative_mamba2_params_all_layers, fork_from
  - `python/sglang/srt/disaggregation/mooncake/conn.py` modified +148/-17 (165 lines); hunks: class TransferKVChunk:; class TransferInfo:; symbols: TransferKVChunk:, TransferInfo:, from_zmq, from_zmq
  - `python/sglang/srt/disaggregation/decode.py` modified +113/-8 (121 lines); hunks: from collections import deque; ); symbols: clear, HybridMambaDecodeReqToTokenPool, __init__, clear
  - `test/srt/test_disaggregation_hybrid_attention.py` added +83/-0 (83 lines); hunks: +import os; symbols: TestDisaggregationHybridAttentionMamba, setUpClass, start_prefill, start_decode
  - `python/sglang/srt/disaggregation/prefill.py` modified +71/-1 (72 lines); hunks: RequestStage,; def _init_kv_manager(self) -> BaseKVManager:; symbols: _init_kv_manager, send_kv_chunk
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/decode.py`; keywords observed in patches: kv, cache, spec, attention, config, scheduler. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/decode.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11061 - Support DeepSeek V3.2 Exp

- Link: https://github.com/sgl-project/sglang/pull/11061
- Status/date: `merged`, created 2025-09-29, merged 2025-10-06; author `fzyzcjy`.
- Diff scope read: `29` files, `+4542/-141`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: quant, attention, cuda, fp8, cache, kv, topk, config, mla, triton.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` added +887/-0 (887 lines); hunks: +from __future__ import annotations; symbols: NSAFlashMLAMetadata:, slice, copy_, NSAMetadata:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` added +785/-0 (785 lines); hunks: +from typing import Optional, Tuple; symbols: fast_log2_ceil, fast_pow2, fast_round_scale, act_quant_kernel
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` added +761/-0 (761 lines); hunks: +from __future__ import annotations; symbols: BaseIndexerMetadata, get_seqlens_int32, get_page_table_64, get_seqlens_expanded
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` added +354/-0 (354 lines); hunks: +from typing import TYPE_CHECKING; symbols: GetK:, execute, slow, torch_fast
  - `python/sglang/srt/models/deepseek_v2.py` modified +329/-17 (346 lines); hunks: # Adapted from:; import torch; symbols: AttnForwardMethod, handle_attention_ascend, _get_sum_extend_prefix_lens, _is_extend_without_speculative
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: quant, attention, cuda, fp8, cache, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11063 - Add DeepSeek-V3.2 Tool Call Template

- Link: https://github.com/sgl-project/sglang/pull/11063
- Status/date: `merged`, created 2025-09-29, merged 2025-10-05; author `Xu-Wenqing`.
- Diff scope read: `1` files, `+100/-0`; areas: docs/config; keywords: kv.
- Code diff details:
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` added +100/-0 (100 lines); hunks: +{% if not add_generation_prompt is defined %}
- Optimization/support interpretation: The concrete diff surface is `examples/chat_template/tool_chat_template_deepseekv32.jinja`; keywords observed in patches: kv. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `examples/chat_template/tool_chat_template_deepseekv32.jinja`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11109 - [Draft] Support MTP for DeepSeek-V3.2

- Link: https://github.com/sgl-project/sglang/pull/11109
- Status/date: `closed`, created 2025-09-30, closed 2025-10-17; author `Fridge003`.
- Diff scope read: `4` files, `+180/-25`; areas: attention/backend, scheduler/runtime, docs/config; keywords: topk, attention, kv, eagle, flash, mla, spec, cache, config, cuda.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +146/-21 (167 lines); hunks: def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; def __init__(self, model_runner: ModelRunner):; symbols: compute_cu_seqlens, NativeSparseAttnBackend, __init__, __init__
  - `python/sglang/srt/speculative/eagle_worker.py` modified +18/-0 (18 lines); hunks: def _create_decode_backend(self):; def _create_draft_extend_backend(self):; symbols: _create_decode_backend, _create_draft_extend_backend, _create_flashmla_decode_backend, _create_nsa_decode_backend
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +11/-3 (14 lines); hunks: def _get_topk_ragged(; def _get_topk_ragged(; symbols: _get_topk_ragged, _get_topk_ragged, _forward
  - `python/sglang/srt/configs/model_config.py` modified +5/-1 (6 lines); hunks: def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: is_deepseek_nsa
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: topk, attention, kv, eagle, flash, mla. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11191 - [Feature] Support Sparse Attention and KV cache scheduling between CPU and GPU for GQA/DSA.

- Link: https://github.com/sgl-project/sglang/pull/11191
- Status/date: `open`, created 2025-10-03; author `yukavio`.
- Diff scope read: `52` files, `+18474/-70`; areas: attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: attention, flash, cuda, config, kv, cache, scheduler, mla, doc, fp8.
- Code diff details:
  - `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm100.py` added +2560/-0 (2560 lines)
  - `python/sglang/srt/sparse_attention/kernels/attention/flash_bwd.py` added +1547/-0 (1547 lines); hunks: +# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.; symbols: FlashAttentionBackwardSm80:, __init__, can_implement, _check_type
  - `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm90.py` added +1402/-0 (1402 lines); hunks: +# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.; symbols: FlashAttentionForwardSm90, __init__, _get_smem_layout_atom, _get_tiled_mma
  - `python/sglang/srt/sparse_attention/kernels/attention/interface.py` added +1266/-0 (1266 lines); hunks: +# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.; symbols: maybe_contiguous, _flash_attn_fwd, _flash_attn_bwd, FlashAttnFunc
  - `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd.py` added +1259/-0 (1259 lines); hunks: +# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.; symbols: FlashAttentionForwardBase:, __init__, can_implement, _check_type
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm100.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_bwd.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm90.py`; keywords observed in patches: attention, flash, cuda, config, kv, cache. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm100.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_bwd.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm90.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11194 - [Feature] Add a fast-topk to sgl-kernel for DeepSeek v3.2

- Link: https://github.com/sgl-project/sglang/pull/11194
- Status/date: `merged`, created 2025-10-03, merged 2025-10-05; author `DarkSharpness`.
- Diff scope read: `7` files, `+588/-1`; areas: MoE/router, kernel, tests/benchmarks; keywords: topk, cuda, mla, spec, awq, test.
- Code diff details:
  - `sgl-kernel/csrc/elementwise/topk.cu` added +422/-0 (422 lines); hunks: +/**; symbols: int, int, size_t, FastTopKParams
  - `sgl-kernel/tests/test_topk.py` added +120/-0 (120 lines); hunks: +import pytest; symbols: _ref_torch_impl, _ref_torch_transform_decode_impl, assert_equal, test_topk_kernel
  - `sgl-kernel/python/sgl_kernel/top_k.py` modified +29/-0 (29 lines); hunks: def fast_topk(values, topk, dim):; symbols: fast_topk, fast_topk_v2, fast_topk_transform_fused
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-0 (8 lines); hunks: void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor& output);
  - `sgl-kernel/csrc/common_extension.cc` modified +7/-0 (7 lines); hunks: TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py`; keywords observed in patches: topk, cuda, mla, spec, awq, test. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11308 - [CI] Add Basic Test for DeepSeek V3.2

- Link: https://github.com/sgl-project/sglang/pull/11308
- Status/date: `merged`, created 2025-10-07, merged 2025-10-13; author `Fridge003`.
- Diff scope read: `4` files, `+137/-4`; areas: tests/benchmarks; keywords: test, deepep, attention, cuda, flash, kv, mla, quant.
- Code diff details:
  - `test/srt/test_deepseek_v32_basic.py` added +78/-0 (78 lines); hunks: +import unittest; symbols: TestDeepseekV3Basic, setUpClass, tearDownClass, test_a_gsm8k
  - `.github/workflows/pr-test.yml` modified +30/-3 (33 lines); hunks: jobs:; jobs:
  - `scripts/ci/ci_install_dependency.sh` modified +26/-1 (27 lines); hunks: set -euxo pipefail; if [ "$IS_BLACKWELL" != "1" ]; then
  - `test/srt/run_suite.py` modified +3/-0 (3 lines); hunks: class TestFile:; symbols: TestFile:
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_deepseek_v32_basic.py`, `.github/workflows/pr-test.yml`, `scripts/ci/ci_install_dependency.sh`; keywords observed in patches: test, deepep, attention, cuda, flash, kv. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_deepseek_v32_basic.py`, `.github/workflows/pr-test.yml`, `scripts/ci/ci_install_dependency.sh`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11309 - [DeepSeek-V3.2] Include indexer kv cache when estimating kv cache size

- Link: https://github.com/sgl-project/sglang/pull/11309
- Status/date: `merged`, created 2025-10-07, merged 2025-10-09; author `trevor-m`.
- Diff scope read: `3` files, `+25/-7`; areas: scheduler/runtime; keywords: cache, kv, quant, attention, config, fp8, mla, spec.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +14/-4 (18 lines); hunks: def __init__(; def load_cpu_copy(self, kv_cache_cpu, indices):; symbols: __init__, get_kv_size_bytes, load_cpu_copy, NSATokenToKVPool
  - `python/sglang/srt/model_executor/model_runner.py` modified +11/-0 (11 lines); hunks: def profile_max_num_token(self, total_gpu_memory: int):; symbols: profile_max_num_token
  - `python/sglang/srt/server_args.py` modified +0/-3 (3 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: cache, kv, quant, attention, config, fp8. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11450 - [DPSKv3.2] Rewrite nsa tilelang act_quant kernel to triton

- Link: https://github.com/sgl-project/sglang/pull/11450
- Status/date: `merged`, created 2025-10-11, merged 2025-10-11; author `byjiang1996`.
- Diff scope read: `3` files, `+420/-1`; areas: attention/backend, quantization, kernel, tests/benchmarks; keywords: attention, quant, triton, fp8, benchmark, cuda, kv, test, vision.
- Code diff details:
  - `test/srt/layers/attention/nsa/test_act_quant_triton.py` added +281/-0 (281 lines); hunks: +"""; symbols: benchmark_kernel, check_accuracy, test_act_quant_comprehensive_benchmark
  - `python/sglang/srt/layers/attention/nsa/triton_kernel.py` added +136/-0 (136 lines); hunks: +from typing import Optional, Tuple; symbols: _act_quant_kernel, act_quant
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-1 (4 lines); hunks: def _forward(; symbols: _forward
- Optimization/support interpretation: The concrete diff surface is `test/srt/layers/attention/nsa/test_act_quant_triton.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, quant, triton, fp8, benchmark, cuda. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/layers/attention/nsa/test_act_quant_triton.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11510 - [Bugfix] Fix Qwen3/DSV3/DSV3.2 model support

- Link: https://github.com/sgl-project/sglang/pull/11510
- Status/date: `merged`, created 2025-10-12, merged 2025-10-16; author `iforgetmyname`.
- Diff scope read: `12` files, `+102/-33`; areas: model wrapper, attention/backend, scheduler/runtime, tests/benchmarks; keywords: cache, test, cuda, attention, config, vision, deepep, doc, mla, quant.
- Code diff details:
  - `.github/workflows/pr-test-npu.yml` modified +33/-13 (46 lines); hunks: jobs:; jobs:
  - `python/sglang/srt/server_args.py` modified +20/-0 (20 lines); hunks: def _handle_gpu_memory_settings(self, gpu_mem):; def _handle_gpu_memory_settings(self, gpu_mem):; symbols: _handle_gpu_memory_settings, _handle_gpu_memory_settings
  - `python/sglang/srt/layers/attention/ascend_backend.py` modified +17/-0 (17 lines); hunks: def forward_extend(; def forward_extend(; symbols: forward_extend, forward_extend, forward_decode_graph
  - `scripts/ci/npu_ci_install_dependency.sh` modified +13/-3 (16 lines); hunks: set -euo pipefail; TORCHVISION_VERSION=0.21.0
  - `docker/Dockerfile.npu` modified +10/-2 (12 lines); hunks: ARG PYTHON_VERSION=py3.11; RUN git clone --depth 1 https://github.com/vllm-project/vllm.git --branch $VLLM_
- Optimization/support interpretation: The concrete diff surface is `.github/workflows/pr-test-npu.yml`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/ascend_backend.py`; keywords observed in patches: cache, test, cuda, attention, config, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `.github/workflows/pr-test-npu.yml`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/ascend_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11557 - Fix DeepSeek-v3.2 default config (ValueError: not enough values to unpack (expected 4, got 3))

- Link: https://github.com/sgl-project/sglang/pull/11557
- Status/date: `merged`, created 2025-10-13, merged 2025-10-13; author `trevor-m`.
- Diff scope read: `1` files, `+1/-1`; areas: misc; keywords: attention, config, cuda, kv, spec.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`; keywords observed in patches: attention, config, cuda, kv, spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11565 - [DSv32] Use torch.compile for _get_logits_head_gate

- Link: https://github.com/sgl-project/sglang/pull/11565
- Status/date: `merged`, created 2025-10-13, merged 2025-10-14; author `trevor-m`.
- Diff scope read: `1` files, `+1/-0`; areas: attention/backend; keywords: attention.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-0 (1 lines); hunks: def _forward_fake(; symbols: _forward_fake, _get_logits_head_gate
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11596 - [Spec Decoding] Support MTP for dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/11596
- Status/date: `closed`, created 2025-10-14, closed 2025-10-15; author `Paiiiiiiiiiiiiii`.
- Diff scope read: `8` files, `+515/-534`; areas: attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: cache, kv, cuda, eagle, spec, topk, attention, config, mla, doc.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +396/-65 (461 lines); hunks: from sglang.srt.model_executor.model_runner import ModelRunner; def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; symbols: compute_cu_seqlens, NativeSparseAttnBackend, __init__, __init__
  - `.github/workflows/pr-test-amd.yml` removed +0/-352 (352 lines); hunks: -name: PR Test (AMD)
  - `.github/workflows/release-docker-dev.yml` removed +0/-108 (108 lines); hunks: -name: Build and Push Development Docker Images
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +82/-8 (90 lines); hunks: except ImportError as e:; from sglang.srt.model_executor.forward_batch_info import ForwardBatch; symbols: _get_topk_paged, _get_verify_topk_paged, _get_topk_ragged, _get_topk_ragged
  - `python/sglang/srt/speculative/draft_utils.py` modified +16/-0 (16 lines); hunks: def init_forward_metadata(*args, **kwargs):; def create_draft_extend_backend(self):; symbols: init_forward_metadata, create_draft_extend_backend, create_draft_extend_backend, _create_nsa_decode_backend
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `.github/workflows/pr-test-amd.yml`, `.github/workflows/release-docker-dev.yml`; keywords observed in patches: cache, kv, cuda, eagle, spec, topk. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `.github/workflows/pr-test-amd.yml`, `.github/workflows/release-docker-dev.yml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11652 - [Spec Decoding] Support MTP for dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/11652
- Status/date: `merged`, created 2025-10-15, merged 2025-10-19; author `Paiiiiiiiiiiiiii`.
- Diff scope read: `6` files, `+445/-79`; areas: attention/backend, kernel, scheduler/runtime, docs/config; keywords: kv, cache, cuda, eagle, spec, topk, attention, config, flash, mla.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +385/-68 (453 lines); hunks: from sglang.srt.model_executor.model_runner import ModelRunner; def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; symbols: compute_cu_seqlens, NativeSparseAttnBackend, __init__, __init__
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +23/-10 (33 lines); hunks: def _get_topk_paged(; def _get_topk_ragged(; symbols: _get_topk_paged, _get_topk_ragged, _get_topk_ragged, forward_indexer
  - `python/sglang/srt/speculative/draft_utils.py` modified +16/-0 (16 lines); hunks: def create_decode_backend(self):; def create_draft_extend_backend(self):; symbols: create_decode_backend, create_draft_extend_backend, create_draft_extend_backend, _create_nsa_decode_backend
  - `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` modified +8/-0 (8 lines); hunks: def __init__(self, eagle_worker: EAGLEWorker):; def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__, __init__, capture_one_batch_size, capture_one_batch_size
  - `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +8/-0 (8 lines); hunks: def __init__(self, eagle_worker: EAGLEWorker):; def capture_one_batch_size(self, bs: int, forward: Callable):; symbols: __init__, capture_one_batch_size, capture_one_batch_size, capture_one_batch_size
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/speculative/draft_utils.py`; keywords observed in patches: kv, cache, cuda, eagle, spec, topk. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/speculative/draft_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11682 - Cleaning indexer for DeepSeek V3.2

- Link: https://github.com/sgl-project/sglang/pull/11682
- Status/date: `merged`, created 2025-10-15, merged 2025-10-17; author `Fridge003`.
- Diff scope read: `2` files, `+3/-66`; areas: attention/backend; keywords: attention, topk, cuda, fp8, kv, lora, quant.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-65 (68 lines); hunks: except ImportError as e:; def __init__(; symbols: __init__, _forward_fake, _get_logits_head_gate, _get_topk_ragged
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +0/-1 (1 lines); hunks: # temp NSA debugging environ
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`; keywords observed in patches: attention, topk, cuda, fp8, kv, lora. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11761 - (beta)support context parallel with deepseekv3.2-DSA

- Link: https://github.com/sgl-project/sglang/pull/11761
- Status/date: `closed`, created 2025-10-17, closed 2025-10-23; author `lixiaolx`.
- Diff scope read: `0` files, `+0/-0`; areas: misc; keywords: n/a.
- Code diff details:
  - No patch file list returned.
- Optimization/support interpretation: The concrete diff surface is no returned patch files; keywords observed in patches: n/a. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises no returned patch files; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11815 - [DeepseekV32] Add fast_topk_transform_ragged_fused kernel

- Link: https://github.com/sgl-project/sglang/pull/11815
- Status/date: `merged`, created 2025-10-19, merged 2025-10-20; author `hlu1`.
- Diff scope read: `6` files, `+201/-20`; areas: MoE/router, kernel, tests/benchmarks; keywords: topk, cuda, kv, spec, mla, quant, test.
- Code diff details:
  - `sgl-kernel/csrc/elementwise/topk.cu` modified +81/-8 (89 lines); hunks: __device__ void naive_topk_transform(; __global__ __launch_bounds__(kThreadsPerBlock) // prefill; symbols: __launch_bounds__, __launch_bounds__
  - `sgl-kernel/tests/test_topk.py` modified +75/-4 (79 lines); hunks: +from typing import Optional; def _ref_torch_transform_decode_impl(; symbols: _ref_torch_impl, _ref_torch_transform_decode_impl, _ref_torch_transform_ragged_impl, assert_equal
  - `sgl-kernel/python/sgl_kernel/top_k.py` modified +24/-1 (25 lines); hunks: def fast_topk_transform_fused(; symbols: fast_topk_transform_fused, fast_topk_transform_ragged_fused
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +11/-6 (17 lines); hunks: void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor& output);
  - `sgl-kernel/python/sgl_kernel/__init__.py` modified +6/-1 (7 lines); hunks: def _find_cuda_home():; symbols: _find_cuda_home
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py`; keywords observed in patches: topk, cuda, kv, spec, mla, quant. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11835 - [CI] Add CI test for DeepSeek V3.2 MTP

- Link: https://github.com/sgl-project/sglang/pull/11835
- Status/date: `merged`, created 2025-10-19, merged 2025-10-20; author `Fridge003`.
- Diff scope read: `4` files, `+112/-3`; areas: tests/benchmarks; keywords: test, eagle, kv, spec, attention, awq, cache, quant, topk.
- Code diff details:
  - `test/srt/test_deepseek_v32_mtp.py` added +105/-0 (105 lines); hunks: +import unittest; symbols: TestDeepseekV32MTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/srt/test_deepseek_v32_basic.py` modified +3/-3 (6 lines); hunks: DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2-Exp"; def test_a_gsm8k(; symbols: TestDeepseekV3Basic, TestDeepseekV32Basic, setUpClass, test_a_gsm8k
  - `python/sglang/srt/server_args.py` modified +3/-0 (3 lines); hunks: def _handle_speculative_decoding(self):; symbols: _handle_speculative_decoding
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: class TestFile:; symbols: TestFile:
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: test, eagle, kv, spec, attention, awq. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11877 - [Doc] Add documentation for DeepSeek V3.2

- Link: https://github.com/sgl-project/sglang/pull/11877
- Status/date: `merged`, created 2025-10-20, merged 2025-10-25; author `Fridge003`.
- Diff scope read: `4` files, `+723/-3`; areas: docs/config; keywords: doc, spec, attention, config, cache, cuda, deepep, eagle, flash, kv.
- Code diff details:
  - `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md` added +570/-0 (570 lines); hunks: +# DeepSeekV32-Exp RBG Based PD Deploy
  - `docs/basic_usage/deepseek_v32.md` added +150/-0 (150 lines); hunks: +# DeepSeek V3.2 Usage
  - `docs/advanced_features/separate_reasoning.ipynb` modified +2/-2 (4 lines); hunks: "\| Model \| Reasoning tags \| Parser \| Notes \|
",; "- Both are handled by the same `deepseek-r1` parser
",
  - `docs/basic_usage/deepseek.md` modified +1/-1 (2 lines); hunks: python3 -m sglang.launch_server \
- Optimization/support interpretation: The concrete diff surface is `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md`, `docs/basic_usage/deepseek_v32.md`, `docs/advanced_features/separate_reasoning.ipynb`; keywords observed in patches: doc, spec, attention, config, cache, cuda. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md`, `docs/basic_usage/deepseek_v32.md`, `docs/advanced_features/separate_reasoning.ipynb`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11892 - DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill

- Link: https://github.com/sgl-project/sglang/pull/11892
- Status/date: `merged`, created 2025-10-21, merged 2025-11-06; author `YAMY1234`.
- Diff scope read: `3` files, `+188/-4`; areas: model wrapper, attention/backend; keywords: attention, cache, kv, mla, cuda, lora, quant, spec, topk, flash.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +84/-0 (84 lines); hunks: def _get_q_k_bf16(; def _get_topk_ragged(; symbols: _get_q_k_bf16, _get_k_bf16, _get_topk_paged, _get_topk_ragged
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +61/-2 (63 lines); hunks: "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."; def forward_extend(; symbols: forward_extend, _forward_flashmla_kv, _forward_standard_mha, _forward_tilelang
  - `python/sglang/srt/models/deepseek_v2.py` modified +43/-2 (45 lines); hunks: def handle_attention_aiter(attn, forward_batch):; def forward_normal_prepare(; symbols: handle_attention_aiter, handle_attention_nsa, forward_normal_prepare
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: attention, cache, kv, mla, cuda, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11936 - [Test] Add dsv3.2 nsa backend testing

- Link: https://github.com/sgl-project/sglang/pull/11936
- Status/date: `merged`, created 2025-10-21, merged 2025-10-26; author `Johnsonms`.
- Diff scope read: `2` files, `+125/-0`; areas: tests/benchmarks; keywords: test, attention, awq, flash, kv, mla, quant.
- Code diff details:
  - `test/srt/test_deepseek_v32_nsabackend.py` added +124/-0 (124 lines); hunks: +import unittest; symbols: TestDeepseekV32NasBackend_flashmla, setUpClass, tearDownClass, test_a_gsm8k
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: class TestFile:; symbols: TestFile:
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_deepseek_v32_nsabackend.py`, `test/srt/run_suite.py`; keywords observed in patches: test, attention, awq, flash, kv, mla. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_deepseek_v32_nsabackend.py`, `test/srt/run_suite.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12017 - (beta)support context parallel with deepseekv3.2-DSA

- Link: https://github.com/sgl-project/sglang/pull/12017
- Status/date: `closed`, created 2025-10-23, closed 2025-10-24; author `lixiaolx`.
- Diff scope read: `11` files, `+595/-81`; areas: model wrapper, attention/backend, scheduler/runtime; keywords: attention, config, cuda, kv, quant, cache, lora, moe, topk, expert.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +133/-50 (183 lines); hunks: get_attention_tp_rank,; ParallelLMHead,; symbols: forward, __init__, forward, forward
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +174/-5 (179 lines); hunks: except ImportError as e:; DUAL_STREAM_TOKEN_THRESHOLD = 1024 if is_cuda() else 0; symbols: BaseIndexerMetadata, __init__, _get_q_k_bf16, _get_q_k_bf16
  - `python/sglang/srt/utils/common.py` modified +121/-1 (122 lines); hunks: from sglang.srt.environ import envs; def require_mlp_tp_gather(server_args):; symbols: require_mlp_tp_gather, require_attn_tp_gather, decorator, calculate_cp_seq_idx
  - `python/sglang/srt/models/deepseek_nextn.py` modified +50/-9 (59 lines); hunks: """Inference-only DeepSeek NextN Speculative Decoding."""; enable_nextn_moe_bf16_cast_to_fp8,; symbols: __init__, forward, forward, __init__
  - `python/sglang/srt/layers/communicator.py` modified +25/-9 (34 lines); hunks: is_sm100_supported,; def _scattered_to_tp_attn_full(; symbols: _scattered_to_tp_attn_full, _scatter_hidden_states_and_residual, _scatter_hidden_states, _gather
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/utils/common.py`; keywords observed in patches: attention, config, cuda, kv, quant, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/utils/common.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12044 - Enable mixed type LayerNorm kernel for NSA indexer

- Link: https://github.com/sgl-project/sglang/pull/12044
- Status/date: `merged`, created 2025-10-24, merged 2025-11-04; author `akhilg-nv`.
- Diff scope read: `3` files, `+166/-25`; areas: attention/backend, tests/benchmarks; keywords: cuda, attention, flash, test.
- Code diff details:
  - `python/sglang/srt/layers/layernorm.py` modified +91/-3 (94 lines); hunks: import torch; _is_cpu_amx_available = cpu_has_amx_support(); symbols: forward_with_allreduce_fusion, LayerNorm, __init__, forward_cuda
  - `python/sglang/test/test_layernorm.py` modified +73/-1 (74 lines); hunks: import torch; def test_gemma_rms_norm(self):; symbols: test_gemma_rms_norm, TestLayerNorm, setUpClass, _run_layer_norm_test
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-21 (23 lines); hunks: from typing import TYPE_CHECKING, Any, Dict, Optional; def rotate_activation(x: torch.Tensor) -> torch.Tensor:; symbols: rotate_activation, V32LayerNorm, __init__, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/layernorm.py`, `python/sglang/test/test_layernorm.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: cuda, attention, flash, test. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/layernorm.py`, `python/sglang/test/test_layernorm.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12052 - Fix Illegal Instruction/IMA errors when using DP attention with DeepSeek-V3.2 models

- Link: https://github.com/sgl-project/sglang/pull/12052
- Status/date: `closed`, created 2025-10-24, closed 2025-10-25; author `YAMY1234`.
- Diff scope read: `1` files, `+18/-1`; areas: attention/backend; keywords: attention, scheduler.
- Code diff details:
  - `python/sglang/srt/layers/dp_attention.py` modified +18/-1 (19 lines); hunks: def _dp_gather_via_all_reduce(; symbols: _dp_gather_via_all_reduce
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/dp_attention.py`; keywords observed in patches: attention, scheduler. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/dp_attention.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12065 - (1/n)support context parallel with deepseekv3.2-DSA

- Link: https://github.com/sgl-project/sglang/pull/12065
- Status/date: `merged`, created 2025-10-24, merged 2025-11-17; author `lixiaolx`.
- Diff scope read: `17` files, `+1247/-54`; areas: model wrapper, attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: attention, kv, cuda, topk, spec, cache, config, moe, quant, expert.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +305/-0 (305 lines); hunks: # temp NSA debugging environ; def print_nsa_bool_env_vars():; symbols: print_nsa_bool_env_vars, compute_nsa_seqlens, is_nsa_enable_prefill_cp, NSAContextParallelMetadata:
  - `python/sglang/srt/layers/communicator_nsa_cp.py` added +284/-0 (284 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: nsa_enable_prefill_cp, NSACPLayerCommunicator, __init__, NSACPCommunicateSimpleFn
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +221/-8 (229 lines); hunks: from __future__ import annotations; except ImportError as e:; symbols: __init__, _get_q_k_bf16, _get_q_k_bf16, _forward_cuda_k_only
  - `python/sglang/srt/models/deepseek_v2.py` modified +134/-32 (166 lines); hunks: is_mla_preprocess_enabled,; def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_nsa, __init__, forward, forward
  - `test/srt/test_deepseek_v32_cp_single_node.py` added +99/-0 (99 lines); hunks: +"""; symbols: TestDeepseekV32CP, setUpClass, tearDownClass, test_a_gsm8k
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, kv, cuda, topk, spec, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12094 - Fuse wk and weight_proj in Indexer for DeepSeekV3.2-FP4

- Link: https://github.com/sgl-project/sglang/pull/12094
- Status/date: `merged`, created 2025-10-24, merged 2025-10-30; author `trevor-m`.
- Diff scope read: `2` files, `+110/-22`; areas: model wrapper, attention/backend; keywords: attention, config, lora, quant, cache, cuda, fp4, fp8, kv, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +45/-22 (67 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +65/-0 (65 lines); hunks: def add_forward_absorb_core_attention_backend(backend_name):; def __init__(; symbols: add_forward_absorb_core_attention_backend, is_nsa_indexer_wk_and_weights_proj_fused, AttnForwardMethod, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: attention, config, lora, quant, cache, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12123 - Fix DeepSeek chat templates to handle tool call arguments type checking (#11700)

- Link: https://github.com/sgl-project/sglang/pull/12123
- Status/date: `merged`, created 2025-10-25, merged 2025-10-30; author `Kangyan-Zhou`.
- Diff scope read: `4` files, `+331/-9`; areas: tests/benchmarks, docs/config; keywords: kv, test.
- Code diff details:
  - `test/srt/test_deepseek_chat_templates.py` added +319/-0 (319 lines); hunks: +"""; symbols: tool, TestDeepSeekChatTemplateToolCalls, setUpClass, _render_template
  - `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +4/-3 (7 lines); hunks: {%- set ns.is_tool = false -%}
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +4/-3 (7 lines); hunks: {%- set ns.is_first = false %}
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +4/-3 (7 lines); hunks: {%- set ns.is_first = false %}
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`; keywords observed in patches: kv, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12296 - Update deepseek_v32.md

- Link: https://github.com/sgl-project/sglang/pull/12296
- Status/date: `merged`, created 2025-10-28, merged 2025-10-28; author `hlu1`.
- Diff scope read: `1` files, `+4/-5`; areas: docs/config; keywords: attention, benchmark, cache, config, doc, eagle, flash, fp8, kv, mla.
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +4/-5 (9 lines); hunks: python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/deepseek_v32.md`; keywords observed in patches: attention, benchmark, cache, config, doc, eagle. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/deepseek_v32.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12582 - [sgl-kernel][Deepseek V3.2] Add row_starts to topk kernel

- Link: https://github.com/sgl-project/sglang/pull/12582
- Status/date: `merged`, created 2025-11-04, merged 2025-11-08; author `hlu1`.
- Diff scope read: `5` files, `+209/-61`; areas: MoE/router, kernel, tests/benchmarks; keywords: topk, cuda, kv, mla, spec, test.
- Code diff details:
  - `sgl-kernel/tests/test_topk.py` modified +85/-24 (109 lines); hunks: -from typing import Optional; ); symbols: _ref_torch_impl, _ref_torch_impl, _ref_torch_transform_decode_impl, _ref_torch_transform_ragged_impl
  - `sgl-kernel/csrc/elementwise/topk.cu` modified +51/-24 (75 lines); hunks: constexpr int kThreadsPerBlock = 1024;; __device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {; symbols: int, size_t, FastTopKParams, __launch_bounds__
  - `sgl-kernel/python/sgl_kernel/top_k.py` modified +61/-7 (68 lines); hunks: +from typing import Optional; def fast_topk(values, topk, dim):; symbols: fast_topk, fast_topk_v2, fast_topk_v2, fast_topk_transform_fused
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +9/-3 (12 lines); hunks: void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor& output);
  - `sgl-kernel/csrc/common_extension.cc` modified +3/-3 (6 lines); hunks: TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/tests/test_topk.py`, `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/python/sgl_kernel/top_k.py`; keywords observed in patches: topk, cuda, kv, mla, spec, test. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/tests/test_topk.py`, `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/python/sgl_kernel/top_k.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12583 - [Deepseek V3.2] Fix accuracy bug in the Indexer

- Link: https://github.com/sgl-project/sglang/pull/12583
- Status/date: `merged`, created 2025-11-04, merged 2025-11-12; author `hlu1`.
- Diff scope read: `6` files, `+96/-17`; areas: attention/backend, tests/benchmarks; keywords: test, attention, kv, fp8, topk, cache, cuda, spec.
- Code diff details:
  - `test/srt/test_deepseek_v32_nsabackend.py` modified +52/-2 (54 lines); hunks: def test_a_gsm8k(; def test_a_gsm8k(; symbols: test_a_gsm8k, test_a_gsm8k, TestDeepseekV32NasBackend_fp8kvcache, setUpClass
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +32/-8 (40 lines); hunks: def _get_topk_ragged(; def _get_topk_ragged(; symbols: _get_topk_ragged, _get_topk_ragged, _forward_cuda_k_only
  - `test/srt/test_deepseek_v32_mtp.py` modified +4/-4 (8 lines); hunks: write_github_step_summary,; def test_a_gsm8k(; symbols: TestDeepseekV32MTP, setUpClass, test_a_gsm8k, test_bs_1_speed
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +6/-1 (7 lines); hunks: def topk_transform(; def topk_transform(; symbols: topk_transform, topk_transform, topk_transform
  - `.github/workflows/pr-test.yml` modified +1/-1 (2 lines); hunks: jobs:
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_deepseek_v32_nsabackend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/srt/test_deepseek_v32_mtp.py`; keywords observed in patches: test, attention, kv, fp8, topk, cache. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_deepseek_v32_nsabackend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/srt/test_deepseek_v32_mtp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12645 - [Bug] Fix NSA Backend KV-Buffer Shape Mismatch in DeepSeek-V3.2

- Link: https://github.com/sgl-project/sglang/pull/12645
- Status/date: `merged`, created 2025-11-04, merged 2025-11-04; author `Johnsonms`.
- Diff scope read: `1` files, `+3/-1`; areas: scheduler/runtime; keywords: cache, fp8, kv, lora, mla, quant.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-1 (4 lines); hunks: def load_cpu_copy(self, kv_cache_cpu, indices):; def __init__(; symbols: load_cpu_copy, NSATokenToKVPool, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: cache, fp8, kv, lora, mla, quant. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12788 - [DeepSeek-V3.2][NSA] Enable MHA Pathway for Short Sequence Prefill on B200 (SM100)

- Link: https://github.com/sgl-project/sglang/pull/12788
- Status/date: `merged`, created 2025-11-06, merged 2025-11-07; author `YAMY1234`.
- Diff scope read: `2` files, `+53/-6`; areas: model wrapper, attention/backend; keywords: attention, cache, kv, topk, config, cuda, flash, mla, quant, spec.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +46/-2 (48 lines); hunks: import torch; from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache; symbols: NSAFlashMLAMetadata:, __init__, get_device_int32_arange, _forward_standard_mha
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-4 (11 lines); hunks: def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_nsa
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: attention, cache, kv, topk, config, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12816 - [Deepseek V3.2] Only skip Indexer logits computation when is_extend_without_speculative

- Link: https://github.com/sgl-project/sglang/pull/12816
- Status/date: `merged`, created 2025-11-07, merged 2025-11-07; author `hlu1`.
- Diff scope read: `3` files, `+20/-18`; areas: model wrapper, attention/backend, scheduler/runtime; keywords: spec, attention, cache, kv, quant, topk, cuda, flash, fp8, lora.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-14 (20 lines); hunks: def _get_sum_extend_prefix_lens(forward_batch):; def _handle_attention_backend(; symbols: _get_sum_extend_prefix_lens, _is_extend_without_speculative, _support_mha_one_shot, _handle_attention_backend
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-4 (11 lines); hunks: def _forward_cuda_k_only(; def forward_cuda(; symbols: _forward_cuda_k_only, forward_cuda
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +7/-0 (7 lines); hunks: def is_cpu_graph(self):; symbols: is_cpu_graph, is_split_prefill, is_extend_without_speculative, CaptureHiddenMode
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/model_executor/forward_batch_info.py`; keywords observed in patches: spec, attention, cache, kv, quant, topk. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/model_executor/forward_batch_info.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12868 - [Docs][DeepseekV3.2] Update deepseekv3.2 docs for mha short seq prefill

- Link: https://github.com/sgl-project/sglang/pull/12868
- Status/date: `merged`, created 2025-11-08, merged 2025-11-08; author `YAMY1234`.
- Diff scope read: `1` files, `+3/-2`; areas: docs/config; keywords: attention, benchmark, cache, config, doc, eagle, flash, fp8, kv, mla.
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +3/-2 (5 lines); hunks: python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/deepseek_v32.md`; keywords observed in patches: attention, benchmark, cache, config, doc, eagle. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/deepseek_v32.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12964 - [DeepseekV3.2] Deepseek fp8 support for MHA path

- Link: https://github.com/sgl-project/sglang/pull/12964
- Status/date: `merged`, created 2025-11-10, merged 2025-11-20; author `YAMY1234`.
- Diff scope read: `2` files, `+55/-9`; areas: model wrapper, attention/backend; keywords: attention, cache, fp8, kv, quant, topk, lora, mla, spec.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +46/-8 (54 lines); hunks: NPUFusedMLAPreprocess,; def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_nsa, handle_attention_nsa, forward_normal_prepare, _get_mla_kv_buffer
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +9/-1 (10 lines); hunks: def init_forward_metadata(self, forward_batch: ForwardBatch):; def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; keywords observed in patches: attention, cache, fp8, kv, quant, topk. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13022 - [Deepseek V3.2] Use torch.compile to speed up torch.cat in nsa

- Link: https://github.com/sgl-project/sglang/pull/13022
- Status/date: `merged`, created 2025-11-10, merged 2025-11-17; author `hlu1`.
- Diff scope read: `1` files, `+22/-1`; areas: attention/backend; keywords: attention, cache, flash, kv, mla, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +22/-1 (23 lines); hunks: class TopkTransformMethod(IntEnum):; def forward_extend(; symbols: TopkTransformMethod, _compiled_cat, _cat, NSAIndexerMetadata
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`; keywords observed in patches: attention, cache, flash, kv, mla, topk. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13459 - [Deepseek V3.2] Change indexer weights_proj to fp32

- Link: https://github.com/sgl-project/sglang/pull/13459
- Status/date: `merged`, created 2025-11-17, merged 2025-11-20; author `hlu1`.
- Diff scope read: `3` files, `+92/-124`; areas: model wrapper, attention/backend, docs/config; keywords: attention, config, fp8, kv, lora, quant, benchmark, cache, cuda, doc.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +26/-53 (79 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +0/-71 (71 lines); hunks: def add_forward_absorb_core_attention_backend(backend_name):; def __init__(; symbols: add_forward_absorb_core_attention_backend, is_nsa_indexer_wk_and_weights_proj_fused, AttnForwardMethod, __init__
  - `docs/basic_usage/deepseek_v32.md` modified +66/-0 (66 lines); hunks: Latency: 25.109 s; Repeat: 8, mean: 0.797; symbols: file:, chat_template_thinking
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `docs/basic_usage/deepseek_v32.md`; keywords observed in patches: attention, config, fp8, kv, lora, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `docs/basic_usage/deepseek_v32.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13531 - DeepSeek V3.2 indexer RoPE fix

- Link: https://github.com/sgl-project/sglang/pull/13531
- Status/date: `closed`, created 2025-11-18, closed 2025-11-18; author `Johnsonms`.
- Diff scope read: `1` files, `+1/-1`; areas: attention/backend; keywords: attention.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13544 - [DeepSeekV3.2] Centralize NSA dispatch logic in NativeSparseAttnBackend

- Link: https://github.com/sgl-project/sglang/pull/13544
- Status/date: `merged`, created 2025-11-18, merged 2025-11-25; author `YAMY1234`.
- Diff scope read: `2` files, `+74/-78`; areas: model wrapper, attention/backend; keywords: attention, cache, fp8, kv, mla, quant, spec, topk, benchmark, config.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +69/-42 (111 lines); hunks: NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,; def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; symbols: compute_cu_seqlens, NativeSparseAttnBackend, __init__, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-36 (41 lines); hunks: def handle_attention_aiter(attn, forward_batch):; symbols: handle_attention_aiter, handle_attention_nsa
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: attention, cache, fp8, kv, mla, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13546 - [Deepseek V3.2] Optimize use of dual_stream in nsa_indexer/attention

- Link: https://github.com/sgl-project/sglang/pull/13546
- Status/date: `closed`, created 2025-11-18, closed 2026-04-10; author `hlu1`.
- Diff scope read: `5` files, `+254/-161`; areas: model wrapper, attention/backend; keywords: attention, cache, kv, cuda, fp8, lora, mla, quant, topk, config.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +191/-130 (321 lines); hunks: from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple; from sglang.srt.layers import deep_gemm_wrapper; symbols: rotate_activation, V32LayerNorm, __init__, _forward_compiled
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +57/-25 (82 lines); hunks: from sglang.srt.environ import envs; def __init__(; symbols: __init__, forward_decode, forward_decode, _prepare_kv_cache
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer; def rebuild_cp_kv_cache(self, latent_cache, forward_batch, k_nope, k_pe):; symbols: rebuild_cp_kv_cache, forward
  - `python/sglang/srt/models/deepseek_nextn.py` modified +2/-2 (4 lines); hunks: from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder; def forward(; symbols: forward
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +1/-1 (2 lines); hunks: def cp_attn_tp_all_gather_reorganazied_into_tensor(; symbols: cp_attn_tp_all_gather_reorganazied_into_tensor, cp_all_gather_rerange_output, cp_all_gather_rearrange_output
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: attention, cache, kv, cuda, fp8, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13646 - [DeepSeekV3.2] Enable pure TP & Partial DP Attention

- Link: https://github.com/sgl-project/sglang/pull/13646
- Status/date: `merged`, created 2025-11-20, merged 2025-11-30; author `YAMY1234`.
- Diff scope read: `7` files, `+286/-24`; areas: attention/backend, tests/benchmarks, docs/config; keywords: attention, kv, flash, mla, spec, cache, config, test, cuda, eagle.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +95/-14 (109 lines); hunks: def _get_topk_paged(; def _get_topk_ragged(; symbols: _get_topk_paged, _should_chunk_mqa_logits, _get_topk_ragged, _get_topk_ragged
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +72/-7 (79 lines); hunks: def topk_transform(; def __init__(; symbols: topk_transform, __init__, forward_extend, forward_decode
  - `test/nightly/test_deepseek_v32_nsabackend.py` modified +57/-0 (57 lines); hunks: def test_a_gsm8k(; symbols: test_a_gsm8k, TestDeepseekV32NasBackend_pure_tp, setUpClass, tearDownClass
  - `test/manual/nightly/test_deepseek_v32_perf.py` modified +25/-0 (25 lines); hunks: def setUpClass(cls):; def setUpClass(cls):; symbols: setUpClass, setUpClass, setUpClass
  - `test/nightly/test_deepseek_v32_perf.py` modified +25/-0 (25 lines); hunks: def setUpClass(cls):; def setUpClass(cls):; symbols: setUpClass, setUpClass, setUpClass
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/nightly/test_deepseek_v32_nsabackend.py`; keywords observed in patches: attention, kv, flash, mla, spec, cache. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/nightly/test_deepseek_v32_nsabackend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13812 - [Performance] Optimize NSA Indexer K/S Buffer Access with Fused Triton Kernels

- Link: https://github.com/sgl-project/sglang/pull/13812
- Status/date: `merged`, created 2025-11-23, merged 2025-12-03; author `Johnsonms`.
- Diff scope read: `4` files, `+896/-8`; areas: attention/backend, scheduler/runtime, tests/benchmarks; keywords: fp8, triton, attention, kv, cache, cuda, quant, test, topk.
- Code diff details:
  - `test/manual/layers/attention/nsa/test_index_buf_accessor.py` added +554/-0 (554 lines); hunks: +"""; symbols: MockNSATokenToKVPool:, __init__, create_test_buffer, TestGetK:
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +318/-2 (320 lines); hunks: class GetK:; def torch_fast(; symbols: GetK:, execute, slow, torch_fast
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +22/-0 (22 lines); hunks: def get_index_k_scale_continuous(; symbols: get_index_k_scale_continuous, get_index_k_scale_buffer, set_index_k_scale_buffer
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-6 (8 lines); hunks: def _get_topk_ragged(; symbols: _get_topk_ragged
- Optimization/support interpretation: The concrete diff surface is `test/manual/layers/attention/nsa/test_index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: fp8, triton, attention, kv, cache, cuda. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/manual/layers/attention/nsa/test_index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13959 - [DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache

- Link: https://github.com/sgl-project/sglang/pull/13959
- Status/date: `merged`, created 2025-11-26, merged 2026-01-02; author `xu-yfei`.
- Diff scope read: `14` files, `+603/-264`; areas: model wrapper, attention/backend, tests/benchmarks, docs/config; keywords: attention, kv, cache, spec, topk, cuda, fp8, triton, mla, moe.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +209/-5 (214 lines); hunks: # temp NSA debugging environ; def is_nsa_enable_prefill_cp():; symbols: is_nsa_enable_prefill_cp, is_nsa_prefill_cp_in_seq_split, is_nsa_prefill_cp_round_robin_split, can_nsa_prefill_cp_round_robin_split
  - `python/sglang/srt/layers/communicator_nsa_cp.py` modified +60/-133 (193 lines); hunks: import torch; LayerScatterModes,; symbols: __init__, _post_init_communicate, get_fn, _scattered_to_tp_attn_full
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +149/-20 (169 lines); hunks: from dataclasses import dataclass; NSA_ENABLE_MTP_PRECOMPUTE_METADATA,; symbols: NSAMetadata:, TopkTransformMethod, get_seqlens_expanded, get_cu_seqlens_k
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +45/-68 (113 lines); hunks: NSA_DUAL_STREAM,; def get_seqlens_expanded(self) -> torch.Tensor:; symbols: get_seqlens_expanded, get_indexer_kvcache_range, get_indexer_seq_len_cpu, get_token_to_batch_idx
  - `test/manual/test_deepseek_v32_cp_single_node.py` modified +74/-0 (74 lines); hunks: def test_a_gsm8k(; symbols: test_a_gsm8k, TestDeepseekV32CPMode1, setUpClass, tearDownClass
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; keywords observed in patches: attention, kv, cache, spec, topk, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14304 - [FIX][DS32]openai protocol: support openai message role: developer

- Link: https://github.com/sgl-project/sglang/pull/14304
- Status/date: `merged`, created 2025-12-02, merged 2025-12-11; author `jimmy-evo`.
- Diff scope read: `1` files, `+4/-3`; areas: misc; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/protocol.py` modified +4/-3 (7 lines); hunks: class ToolCall(BaseModel):; symbols: ToolCall, ChatCompletionMessageGenericParam, _normalize_role
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/entrypoints/openai/protocol.py`; keywords observed in patches: n/a. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/entrypoints/openai/protocol.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14307 - [SMG][DS32][fix] support dsv32, add role developer

- Link: https://github.com/sgl-project/sglang/pull/14307
- Status/date: `merged`, created 2025-12-02, merged 2025-12-11; author `jimmy-evo`.
- Diff scope read: `3` files, `+36/-9`; areas: MoE/router; keywords: router.
- Code diff details:
  - `sgl-model-gateway/src/protocols/chat.rs` modified +12/-9 (21 lines); hunks: pub enum ChatMessage {; impl GenerationRequest for ChatCompletionRequest {
  - `sgl-model-gateway/src/routers/grpc/harmony/builder.rs` modified +20/-0 (20 lines); hunks: impl HarmonyBuilder {
  - `sgl-model-gateway/src/routers/http/pd_router.rs` modified +4/-0 (4 lines); hunks: impl RouterTrait for PDRouter {
- Optimization/support interpretation: The concrete diff surface is `sgl-model-gateway/src/protocols/chat.rs`, `sgl-model-gateway/src/routers/grpc/harmony/builder.rs`, `sgl-model-gateway/src/routers/http/pd_router.rs`; keywords observed in patches: router. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `sgl-model-gateway/src/protocols/chat.rs`, `sgl-model-gateway/src/routers/grpc/harmony/builder.rs`, `sgl-model-gateway/src/routers/http/pd_router.rs`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14332 - feat: V32 tool call parsing for no-dsml tag

- Link: https://github.com/sgl-project/sglang/pull/14332
- Status/date: `open`, created 2025-12-03; author `Eva20150932-atlascloud`.
- Diff scope read: `2` files, `+481/-44`; areas: tests/benchmarks; keywords: kv, benchmark, test.
- Code diff details:
  - `test/registered/function_call/test_function_call_parser.py` modified +334/-0 (334 lines); hunks: def setUp(self):; def test_streaming_json_format(self):; symbols: setUp, test_streaming_json_format, test_detect_and_parse_xml_format_without_dsml, test_detect_and_parse_json_format_without_dsml
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +147/-44 (191 lines); hunks: class DeepSeekV32Detector(BaseFormatDetector):; class DeepSeekV32Detector(BaseFormatDetector):; symbols: DeepSeekV32Detector, DeepSeekV32Detector, DeepSeekV32Detector, DeepSeekV32Detector
- Optimization/support interpretation: The concrete diff surface is `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`; keywords observed in patches: kv, benchmark, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14353 - feat(dsv32): better error handling for DeepSeek-v3.2 encoder

- Link: https://github.com/sgl-project/sglang/pull/14353
- Status/date: `merged`, created 2025-12-03, merged 2025-12-19; author `jimmy-evo`.
- Diff scope read: `2` files, `+53/-32`; areas: misc; keywords: kv, spec.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/encoding_dsv32.py` modified +45/-32 (77 lines); hunks: import re; def find_last_user_index(messages: List[Dict[str, Any]]) -> int:; symbols: DS32EncodingError, find_last_user_index, render_message, render_message
  - `python/sglang/srt/entrypoints/openai/serving_base.py` modified +8/-0 (8 lines); hunks: from fastapi import HTTPException, Request; async def handle_request(; symbols: handle_request
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`, `python/sglang/srt/entrypoints/openai/serving_base.py`; keywords observed in patches: kv, spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`, `python/sglang/srt/entrypoints/openai/serving_base.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14524 - [Test] Add test suite for NSA backend

- Link: https://github.com/sgl-project/sglang/pull/14524
- Status/date: `open`, created 2025-12-06; author `Johnsonms`.
- Diff scope read: `1` files, `+709/-0`; areas: attention/backend, tests/benchmarks; keywords: attention, cache, config, cuda, eagle, flash, kv, lora, mla, scheduler.
- Code diff details:
  - `python/sglang/test/attention/test_nsa_backend.py` added +709/-0 (709 lines); hunks: +import unittest; symbols: MockNSAConfig:, __init__, MockModelRunner:, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/test/attention/test_nsa_backend.py`; keywords observed in patches: attention, cache, config, cuda, eagle, flash. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/test/attention/test_nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14541 - [NPU]dsv3.2 cp for npu

- Link: https://github.com/sgl-project/sglang/pull/14541
- Status/date: `merged`, created 2025-12-06, merged 2025-12-11; author `liupeng374`.
- Diff scope read: `8` files, `+281/-134`; areas: attention/backend, tests/benchmarks; keywords: attention, cache, kv, cuda, topk, flash, lora, mla, config, quant.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +117/-94 (211 lines); hunks: cp_all_gather_rerange_output,; def forward_npu(; symbols: forward_npu, forward_npu, forward_npu, forward_npu
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +111/-22 (133 lines); hunks: is_mla_preprocess_enabled,; class ForwardMetadata:; symbols: ForwardMetadata:, update_verify_buffers_to_fill_after_draft, init_forward_metadata, init_forward_metadata_replay_cuda_graph
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +25/-4 (29 lines); hunks: class NSAContextParallelMetadata:; def prepare_input_dp_with_cp_dsa(; symbols: NSAContextParallelMetadata:, prepare_input_dp_with_cp_dsa
  - `python/sglang/srt/layers/communicator_nsa_cp.py` modified +7/-8 (15 lines); hunks: def _gather_hidden_states_and_residual(; symbols: _gather_hidden_states_and_residual, _scatter_hidden_states_and_residual
  - `python/sglang/srt/distributed/parallel_state.py` modified +3/-6 (9 lines); hunks: def cp_all_gather_into_tensor_async(; symbols: cp_all_gather_into_tensor_async, all_gather
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py`; keywords observed in patches: attention, cache, kv, cuda, topk, flash. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14572 - [NPU] optimization for dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/14572
- Status/date: `merged`, created 2025-12-07, merged 2025-12-12; author `ZhengdQin`.
- Diff scope read: `11` files, `+141/-68`; areas: model wrapper, attention/backend, MoE/router, quantization; keywords: config, kv, topk, attention, cuda, expert, lora, moe, quant, router.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +51/-15 (66 lines); hunks: from sglang.srt.layers.layernorm import LayerNorm; def forward_npu(; symbols: forward_npu, forward_npu
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +34/-18 (52 lines); hunks: def forward_dsa_prepare_npu(; def forward_dsa_core_npu(; symbols: forward_dsa_prepare_npu, forward_dsa_core_npu
  - `python/sglang/srt/models/deepseek_v2.py` modified +25/-4 (29 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, forward_deepep
  - `python/sglang/srt/hardware_backend/npu/moe/topk.py` modified +4/-11 (15 lines); hunks: def fused_topk_npu(; def fused_topk_npu(; symbols: fused_topk_npu, fused_topk_npu
  - `python/sglang/srt/layers/layernorm.py` modified +1/-13 (14 lines); hunks: def forward_npu(; symbols: forward_npu, forward_cpu
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: config, kv, topk, attention, cuda, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14573 - [Tool Call] Fix DeepSeekV32Detector skipping functions with no params in streaming mode

- Link: https://github.com/sgl-project/sglang/pull/14573
- Status/date: `merged`, created 2025-12-07, merged 2025-12-08; author `momaek`.
- Diff scope read: `2` files, `+144/-7`; areas: tests/benchmarks; keywords: kv, test.
- Code diff details:
  - `test/registered/function_call/test_function_call_parser.py` modified +142/-0 (142 lines); hunks: def test_streaming_json_format(self):; symbols: test_streaming_json_format, test_detect_and_parse_no_parameters, test_streaming_no_parameters, test_streaming_no_parameters_with_whitespace
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +2/-7 (9 lines); hunks: def parse_streaming_increment(; symbols: parse_streaming_increment
- Optimization/support interpretation: The concrete diff surface is `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`; keywords observed in patches: kv, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14619 - [Sparse & HICache]: Enables hierarchical sparse KV cache management and scheduling for DeepSeek V32.

- Link: https://github.com/sgl-project/sglang/pull/14619
- Status/date: `closed`, created 2025-12-08, closed 2026-03-23; author `hzh0425`.
- Diff scope read: `30` files, `+3077/-118`; areas: model wrapper, attention/backend, kernel, multimodal/processor, scheduler/runtime; keywords: cache, kv, attention, config, topk, triton, cuda, flash, spec, benchmark.
- Code diff details:
  - `python/sglang/srt/mem_cache/sparsity/ops/triton_kernel.py` added +622/-0 (622 lines); hunks: +import torch; symbols: nsa_sparse_diff_triton_kernel, invoke_nsa_sparse_diff_kernel, benchmark_kernel
  - `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py` added +383/-0 (383 lines); hunks: +from abc import ABC, abstractmethod; symbols: BaseSparseAlgorithm, for, provides, __init__
  - `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py` added +341/-0 (341 lines); hunks: +import logging; symbols: RequestTrackers:, __init__, register, clear
  - `python/sglang/srt/mem_cache/sparsity/core/sparse_kvcache_manager.py` added +237/-0 (237 lines); hunks: +from __future__ import annotations; symbols: SparseKVCacheManager:, __init__, transfer_sparse_top_k_cache, offload_sparse_decode_req_tokens
  - `python/sglang/srt/mem_cache/common.py` modified +195/-39 (234 lines); hunks: import triton; def evict_from_tree_cache(tree_cache: BasePrefixCache \| None, num_tokens: int):; symbols: evict_from_tree_cache, truncate_kv_cache_after_prefill, alloc_paged_token_slots_extend, alloc_paged_token_slots_decode
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/mem_cache/sparsity/ops/triton_kernel.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py`; keywords observed in patches: cache, kv, attention, config, topk, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/mem_cache/sparsity/ops/triton_kernel.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14741 - [1/N][Sparse With Hicache]: Add Sparse Interface

- Link: https://github.com/sgl-project/sglang/pull/14741
- Status/date: `merged`, created 2025-12-09, merged 2025-12-25; author `hzh0425`.
- Diff scope read: `4` files, `+642/-0`; areas: scheduler/runtime; keywords: cache, attention, config, kv, spec, topk, cuda, lora, triton.
- Code diff details:
  - `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py` added +383/-0 (383 lines); hunks: +from abc import ABC, abstractmethod; symbols: BaseSparseAlgorithm, for, provides, __init__
  - `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py` added +166/-0 (166 lines); hunks: +"""; symbols: QuestAlgorithm, __init__, _initialize_representation_pools, _compute_page_representations
  - `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py` added +80/-0 (80 lines); hunks: +from typing import Any, Optional; symbols: DeepSeekNSAAlgorithm, __init__, retrieve_topk, initialize_representation_pool
  - `python/sglang/srt/mem_cache/sparsity/algorithms/__init__.py` added +13/-0 (13 lines); hunks: +from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py`; keywords observed in patches: cache, attention, config, kv, spec, topk. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14750 - [Tool Call][DSV32] Streamline function call parameters

- Link: https://github.com/sgl-project/sglang/pull/14750
- Status/date: `merged`, created 2025-12-09, merged 2025-12-26; author `Muqi1029`.
- Diff scope read: `2` files, `+60/-29`; areas: tests/benchmarks; keywords: kv, spec, test.
- Code diff details:
  - `test/registered/function_call/test_function_call_parser.py` modified +37/-14 (51 lines); hunks: def setUp(self):; def test_streaming_xml_format(self):; symbols: setUp, test_detect_and_parse_xml_format, test_streaming_xml_format, test_streaming_xml_format
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +23/-15 (38 lines); hunks: import logging; ToolCallItem,; symbols: __init__, has_tool_call, _parse_parameters_from_xml, _parse_parameters_from_xml
- Optimization/support interpretation: The concrete diff surface is `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`; keywords observed in patches: kv, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14781 - [Performance] optimize NSA backend metadata computation for multi-step speculative decoding

- Link: https://github.com/sgl-project/sglang/pull/14781
- Status/date: `merged`, created 2025-12-10, merged 2025-12-18; author `Johnsonms`.
- Diff scope read: `3` files, `+440/-16`; areas: attention/backend; keywords: attention, cache, spec, cuda, flash, kv, mla, quant, topk, fp8.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py` added +324/-0 (324 lines); hunks: +"""Multi-step precompute utilities for Native Sparse Attention backend.; symbols: PrecomputedMetadata:, compute_cu_seqlens, NativeSparseAttnBackendMTPPrecomputeMixin:, providing
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +111/-16 (127 lines); hunks: from sglang.srt.environ import envs; def topk_transform(; symbols: topk_transform, compute_cu_seqlens, NativeSparseAttnBackend, NativeSparseAttnBackend
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +5/-0 (5 lines); hunks: NSA_QUANT_K_CACHE_FAST = get_bool_env_var("SGLANG_NSA_QUANT_K_CACHE_FAST", "true"); symbols: print_nsa_bool_env_vars
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py`; keywords observed in patches: attention, cache, spec, cuda, flash, kv. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14901 - fix ds3.2 nsa backend prefill TBO

- Link: https://github.com/sgl-project/sglang/pull/14901
- Status/date: `merged`, created 2025-12-11, merged 2025-12-21; author `Chen-0210`.
- Diff scope read: `5` files, `+76/-2`; areas: model wrapper, attention/backend, tests/benchmarks; keywords: attention, cuda, deepep, kv, moe, test, awq, fp8, mla, quant.
- Code diff details:
  - `test/srt/ep/test_deepep_large.py` modified +55/-0 (55 lines); hunks: from sglang.srt.utils import kill_process_tree; popen_launch_server,; symbols: TestDeepseek, test_gsm8k, TestDeepseekV32TBO, setUpClass
  - `python/sglang/srt/models/deepseek_v2.py` modified +8/-1 (9 lines); hunks: is_nsa_enable_prefill_cp,; def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_nsa, _get_mla_kv_buffer_from_fp8
  - `python/sglang/srt/server_args.py` modified +9/-0 (9 lines); hunks: def __post_init__(self):; def _handle_other_validations(self):; symbols: __post_init__, _handle_deprecated_args, _handle_other_validations, _handle_two_batch_overlap
  - `python/sglang/srt/layers/attention/tbo_backend.py` modified +3/-0 (3 lines); hunks: def forward_extend(self, *args, **kwargs):; symbols: forward_extend, forward_decode, get_indexer_metadata, _init_forward_metadata_cuda_graph_split
  - `test/srt/run_suite.py` modified +1/-1 (2 lines); hunks: # TestFile("ep/test_mooncake_ep_small.py", 450),
- Optimization/support interpretation: The concrete diff surface is `test/srt/ep/test_deepep_large.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: attention, cuda, deepep, kv, moe, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/ep/test_deepep_large.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14904 - [DeepSeek V3.2] Proper drop_thinking logic

- Link: https://github.com/sgl-project/sglang/pull/14904
- Status/date: `closed`, created 2025-12-11, closed 2025-12-13; author `vladnosiv`.
- Diff scope read: `1` files, `+5/-1`; areas: misc; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +5/-1 (6 lines); hunks: def _apply_jinja_template(; symbols: _apply_jinja_template
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/entrypoints/openai/serving_chat.py`; keywords observed in patches: n/a. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/entrypoints/openai/serving_chat.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15040 - [DSv32] Move deep_gemm.get_paged_mqa_logits_metadata to init time as metadata

- Link: https://github.com/sgl-project/sglang/pull/15040
- Status/date: `merged`, created 2025-12-13, merged 2025-12-19; author `qianlihuang`.
- Diff scope read: `2` files, `+91/-5`; areas: attention/backend; keywords: attention, topk, cache, cuda, flash, fp8, kv, mla.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +84/-1 (85 lines); hunks: from sglang.srt.layers.attention.trtllm_mla_backend import _concat_mla_absorb_q_general; class NSAMetadata:; symbols: NSAMetadata:, _cat, NSAIndexerMetadata, get_seqlens_int32
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-4 (11 lines); hunks: def _get_topk_paged(; symbols: _get_topk_paged
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, topk, cache, cuda, flash, fp8. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15051 - feat(ds32): support <function_call> tag for deepseek 3.2 tool call

- Link: https://github.com/sgl-project/sglang/pull/15051
- Status/date: `closed`, created 2025-12-13, closed 2025-12-16; author `jimmy-evo`.
- Diff scope read: `1` files, `+56/-10`; areas: misc; keywords: kv.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +56/-10 (66 lines); hunks: def __init__(self):; def _parse_parameters_from_xml(self, invoke_content: str) -> dict:; symbols: __init__, has_tool_call, _parse_parameters_from_xml, _parse_parameters_from_xml
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/deepseekv32_detector.py`; keywords observed in patches: kv. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/deepseekv32_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15064 - fix: dpskv32 chat history processing, default drop_thinking to true

- Link: https://github.com/sgl-project/sglang/pull/15064
- Status/date: `merged`, created 2025-12-13, merged 2025-12-13; author `JustinTong0323`.
- Diff scope read: `1` files, `+1/-3`; areas: misc; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +1/-3 (4 lines); hunks: def _apply_jinja_template(; symbols: _apply_jinja_template
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/entrypoints/openai/serving_chat.py`; keywords observed in patches: n/a. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/entrypoints/openai/serving_chat.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15086 - [NSA] Fix NSA backend assertion error when running DeepSeek-V3.2 PP with radix-cache

- Link: https://github.com/sgl-project/sglang/pull/15086
- Status/date: `merged`, created 2025-12-13, merged 2025-12-15; author `YAMY1234`.
- Diff scope read: `2` files, `+19/-5`; areas: attention/backend, quantization, scheduler/runtime; keywords: attention, cache, kv, quant, fp8, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +17/-2 (19 lines); hunks: def init_forward_metadata(self, forward_batch: ForwardBatch):; def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata
  - `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` modified +2/-3 (5 lines); hunks: def dequantize_k_cache_paged(; symbols: dequantize_k_cache_paged
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`; keywords observed in patches: attention, cache, kv, quant, fp8, topk. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15088 - [DeepSeekV3.2] Add pure TP+MTP test

- Link: https://github.com/sgl-project/sglang/pull/15088
- Status/date: `merged`, created 2025-12-14, merged 2025-12-17; author `ashtonchew`.
- Diff scope read: `2` files, `+107/-7`; areas: tests/benchmarks, docs/config; keywords: config, eagle, spec, topk, attention, cache, cuda, doc, flash, kv.
- Code diff details:
  - `test/nightly/test_deepseek_v32_tp.py` modified +100/-6 (106 lines); hunks: import unittest; write_github_step_summary,; symbols: test_a_gsm8k, TestDeepseekV32_TP_MTP, setUpClass, tearDownClass
  - `docs/basic_usage/deepseek_v32.md` modified +7/-1 (8 lines); hunks: python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8
- Optimization/support interpretation: The concrete diff surface is `test/nightly/test_deepseek_v32_tp.py`, `docs/basic_usage/deepseek_v32.md`; keywords observed in patches: config, eagle, spec, topk, attention, cache. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/nightly/test_deepseek_v32_tp.py`, `docs/basic_usage/deepseek_v32.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15217 - fix(DeepSeek-V3.2 function_call): fix streaming content loss in DeepSeekV32Detector

- Link: https://github.com/sgl-project/sglang/pull/15217
- Status/date: `closed`, created 2025-12-16, closed 2025-12-16; author `momaek`.
- Diff scope read: `1` files, `+3/-3`; areas: misc; keywords: kv.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +3/-3 (6 lines); hunks: def parse_streaming_increment(; symbols: parse_streaming_increment
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/deepseekv32_detector.py`; keywords observed in patches: kv. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/deepseekv32_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15242 - [sgl-kernel] Update flashmla to include fp8 sparse_mla optimizations

- Link: https://github.com/sgl-project/sglang/pull/15242
- Status/date: `merged`, created 2025-12-16, merged 2025-12-16; author `hlu1`.
- Diff scope read: `1` files, `+1/-1`; areas: attention/backend, kernel; keywords: flash, mla.
- Code diff details:
  - `sgl-kernel/cmake/flashmla.cmake` modified +1/-1 (2 lines); hunks: include(FetchContent)
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/cmake/flashmla.cmake`; keywords observed in patches: flash, mla. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/cmake/flashmla.cmake`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15278 - feat: DeepSeek-V3.2 Streaming tool call output

- Link: https://github.com/sgl-project/sglang/pull/15278
- Status/date: `merged`, created 2025-12-16, merged 2025-12-18; author `JustinTong0323`.
- Diff scope read: `2` files, `+111/-69`; areas: tests/benchmarks; keywords: kv, spec, test.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +95/-65 (160 lines); hunks: import json; ToolCallItem,; symbols: __init__, has_tool_call, _parse_parameters_from_xml, _parse_parameters_from_xml
  - `test/registered/function_call/test_function_call_parser.py` modified +16/-4 (20 lines); hunks: def setUp(self):; def test_streaming_xml_format(self):; symbols: setUp, test_detect_and_parse_xml_format, test_streaming_xml_format, test_streaming_json_format
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/function_call/test_function_call_parser.py`; keywords observed in patches: kv, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/function_call/test_function_call_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15307 - [Deepseek V3.2] Support Overlap Spec + NSA

- Link: https://github.com/sgl-project/sglang/pull/15307
- Status/date: `merged`, created 2025-12-17, merged 2025-12-17; author `b8zhong`.
- Diff scope read: `3` files, `+25/-8`; areas: attention/backend, docs/config; keywords: topk, attention, cuda, spec, cache, config, doc, eagle, flash, fp8.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +19/-6 (25 lines); hunks: def init_forward_metadata(self, forward_batch: ForwardBatch):; def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata, init_forward_metadata_capture_cuda_graph, init_forward_metadata_replay_cuda_graph
  - `docs/basic_usage/deepseek_v32.md` modified +4/-0 (4 lines); hunks: python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-2 (4 lines); hunks: def _get_topk_paged(; def forward_cuda(; symbols: _get_topk_paged, forward_cuda
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: topk, attention, cuda, spec, cache, config. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15310 - [Deepseek V3.2] Enable TRTLLM Allreduce Fusion

- Link: https://github.com/sgl-project/sglang/pull/15310
- Status/date: `closed`, created 2025-12-17, closed 2026-01-06; author `b8zhong`.
- Diff scope read: `1` files, `+1/-0`; areas: misc; keywords: kv, moe, spec.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +1/-0 (1 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`; keywords observed in patches: kv, moe, spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15322 - dsv32 support o_proj tp

- Link: https://github.com/sgl-project/sglang/pull/15322
- Status/date: `open`, created 2025-12-17; author `lawtherWu`.
- Diff scope read: `14` files, `+472/-23`; areas: model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime; keywords: moe, config, attention, cache, expert, quant, spec, topk, cuda, deepep.
- Code diff details:
  - `python/sglang/srt/layers/communicator.py` modified +179/-5 (184 lines); hunks: import torch; get_moe_a2a_backend,; symbols: enable_moe_dense_fully_dp, get_max_bs_across_dp, LayerCommunicator:, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +74/-14 (88 lines); hunks: from sglang.srt.distributed import (; from sglang.srt.layers.attention.tbo_backend import TboAttnBackend; symbols: forward_deepep, __init__, __init__, __init__
  - `python/sglang/srt/distributed/parallel_state.py` modified +85/-0 (85 lines); hunks: def get_moe_tp_group() -> GroupCoordinator:; def initialize_model_parallel(; symbols: get_moe_tp_group, get_o_proj_tp_group, get_o_proj_dp_group, initialize_model_parallel
  - `python/sglang/srt/layers/linear.py` modified +73/-1 (74 lines); hunks: from sglang.srt.distributed import (; def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):; symbols: weight_loader, weight_loader, extra_repr, TP2DPandTPRowParallelLinear
  - `python/sglang/srt/distributed/communication_op.py` modified +20/-1 (21 lines); hunks: import torch; def broadcast_tensor_dict(; symbols: tensor_model_parallel_all_reduce, broadcast_tensor_dict, o_proj_tensor_model_parallel_reduce_scatter_tensor, o_proj_tensor_model_parallel_all_reduce
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/distributed/parallel_state.py`; keywords observed in patches: moe, config, attention, cache, expert, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/distributed/parallel_state.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15381 - [NPU]DeepSeek-V3.2 support npu mlaprolog

- Link: https://github.com/sgl-project/sglang/pull/15381
- Status/date: `merged`, created 2025-12-18, merged 2026-01-26; author `lawtherWu`.
- Diff scope read: `5` files, `+195/-61`; areas: attention/backend, quantization; keywords: attention, kv, cache, lora, mla, quant, config, topk.
- Code diff details:
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +116/-55 (171 lines); hunks: +import re; def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu, forward_dsa_prepare_npu, forward_dsa_core_npu, npu_mla_preprocess
  - `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py` modified +63/-1 (64 lines); hunks: +import re; def __init__(; symbols: __init__, __init__, preprocess_weights, preprocess_weights
  - `python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py` modified +8/-2 (10 lines); hunks: def apply(; symbols: apply
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +3/-3 (6 lines); hunks: def forward_extend(; def forward_extend(; symbols: forward_extend, forward_extend
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +5/-0 (5 lines); hunks: def forward_npu(; def forward_npu(; symbols: forward_npu, forward_npu, forward_npu
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`, `python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py`; keywords observed in patches: attention, kv, cache, lora, mla, quant. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`, `python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15429 - [Deepseek V3.2] Fix Deepseek MTP in V1 mode

- Link: https://github.com/sgl-project/sglang/pull/15429
- Status/date: `merged`, created 2025-12-19, merged 2025-12-19; author `b8zhong`.
- Diff scope read: `1` files, `+1/-1`; areas: attention/backend; keywords: attention.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-1 (2 lines); hunks: def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`; keywords observed in patches: attention. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15807 - [2/N][Sparse With Hicache]: Support separating nsa memory management for KV cache and index_k in decode side.

- Link: https://github.com/sgl-project/sglang/pull/15807
- Status/date: `closed`, created 2025-12-25, closed 2026-03-23; author `hzh0425`.
- Diff scope read: `10` files, `+516/-39`; areas: attention/backend, scheduler/runtime; keywords: cache, kv, attention, cuda, mla, spec, topk, config, flash, fp8.
- Code diff details:
  - `python/sglang/srt/mem_cache/allocator.py` modified +156/-0 (156 lines); hunks: def get_cpu_copy(self, indices):; symbols: get_cpu_copy, load_cpu_copy, NSAHybridTokenToKVPoolAllocator, __init__
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +89/-0 (89 lines); hunks: ); class NSAMetadata:; symbols: NSAMetadata:, get_seqlens_int32, get_page_table_64, get_seqlens_expanded
  - `python/sglang/srt/mem_cache/common.py` modified +68/-5 (73 lines); hunks: import triton; def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:; symbols: alloc_for_decode, _alloc_for_nsa_index_k, release_kv_cache, release_kv_cache
  - `python/sglang/srt/model_executor/model_runner.py` modified +48/-17 (65 lines); hunks: def init_memory_pool(; def init_memory_pool(; symbols: init_memory_pool, init_memory_pool, init_memory_pool, init_memory_pool
  - `python/sglang/srt/disaggregation/decode.py` modified +56/-4 (60 lines); hunks: from sglang.srt.managers.utils import GenerationBatchResult; def clear(self):; symbols: clear, NSADecodeReqToTokenPool, __init__, write_index_token
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/mem_cache/common.py`; keywords observed in patches: cache, kv, attention, cuda, mla, spec. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/mem_cache/common.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15938 - Clean Some Environment Variables for DeepSeek V32

- Link: https://github.com/sgl-project/sglang/pull/15938
- Status/date: `merged`, created 2025-12-27, merged 2026-01-07; author `Fridge003`.
- Diff scope read: `8` files, `+39/-108`; areas: attention/backend, quantization, scheduler/runtime, docs/config; keywords: attention, cache, cuda, fp8, quant, topk, flash, mla, spec, config.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` modified +6/-42 (48 lines); hunks: import triton; def quantize_k_cache_separate(; symbols: quantize_k_cache, quantize_k_cache_separate, quantize_k_cache_separate, _quantize_k_cache_slow
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +8/-19 (27 lines); hunks: transform_index_page_table_prefill,; def topk_transform(; symbols: topk_transform, forward_extend, forward_extend, forward_extend
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +0/-24 (24 lines); hunks: get_attention_tp_size,; symbols: print_nsa_bool_env_vars, compute_nsa_seqlens
  - `python/sglang/srt/server_args.py` modified +8/-13 (21 lines); hunks: def _handle_model_specific_adjustments(self):; def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, _handle_model_specific_adjustments, _handle_model_specific_adjustments
  - `docs/references/environment_variables.md` modified +10/-0 (10 lines); hunks: SGLang supports various environment variables that can be used to configure its
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py`; keywords observed in patches: attention, cache, cuda, fp8, quant, topk. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16079 - [Performance] Change sparse MLA and dense MHA switching threshold DSv3.2

- Link: https://github.com/sgl-project/sglang/pull/16079
- Status/date: `closed`, created 2025-12-29, closed 2026-03-25; author `zhangxiaolei123456`.
- Diff scope read: `1` files, `+4/-2`; areas: attention/backend; keywords: attention, cache, config, flash, fp8, kv, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +4/-2 (6 lines); hunks: from sglang.srt.layers.dp_attention import get_attention_tp_size; def __init__(; symbols: __init__, set_nsa_prefill_impl
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`; keywords observed in patches: attention, cache, config, flash, fp8, kv. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16091 - [Tool Call] Stream DeepSeek-V3.2 function call parameters in JSON format.

- Link: https://github.com/sgl-project/sglang/pull/16091
- Status/date: `merged`, created 2025-12-29, merged 2026-03-03; author `Muqi1029`.
- Diff scope read: `2` files, `+31/-22`; areas: tests/benchmarks; keywords: kv, test.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +17/-21 (38 lines); hunks: def __init__(self):; def has_tool_call(self, text: str) -> bool:; symbols: __init__, has_tool_call, has_tool_call, _parse_parameters_from_xml
  - `test/registered/function_call/test_function_call_parser.py` modified +14/-1 (15 lines); hunks: def test_streaming_xml_format(self):; def test_streaming_xml_format(self):; symbols: test_streaming_xml_format, test_streaming_xml_format, test_streaming_json_format, test_streaming_json_format
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/function_call/test_function_call_parser.py`; keywords observed in patches: kv, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/function_call/test_function_call_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16119 - [cp] bug fix for dsv3.2 cp

- Link: https://github.com/sgl-project/sglang/pull/16119
- Status/date: `merged`, created 2025-12-30, merged 2025-12-30; author `liupeng374`.
- Diff scope read: `1` files, `+1/-1`; areas: attention/backend; keywords: attention, cache, kv, lora, mla.
- Code diff details:
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +1/-1 (2 lines); hunks: def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; keywords observed in patches: attention, cache, kv, lora, mla. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16156 - [cp] assert dsv3.2 cp in pd decode mode

- Link: https://github.com/sgl-project/sglang/pull/16156
- Status/date: `merged`, created 2025-12-30, merged 2025-12-31; author `liupeng374`.
- Diff scope read: `1` files, `+4/-0`; areas: misc; keywords: spec.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +4/-0 (4 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`; keywords observed in patches: spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16305 - Multiple updates of DeepSeek V32 and context parallel

- Link: https://github.com/sgl-project/sglang/pull/16305
- Status/date: `merged`, created 2026-01-02, merged 2026-01-02; author `Fridge003`.
- Diff scope read: `7` files, `+190/-35`; areas: tests/benchmarks, docs/config; keywords: test, attention, kv, spec, cache, deepep, eagle, fp8, moe, benchmark.
- Code diff details:
  - `test/srt/test_deepseek_v32_mtp.py` modified +81/-1 (82 lines); hunks: FULL_DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2-Exp"; def test_bs_1_speed(self):; symbols: TestDeepseekV32MTP, TestDeepseekV32DPMTP, setUpClass, test_bs_1_speed
  - `test/srt/test_deepseek_v32_basic.py` modified +56/-1 (57 lines); hunks: DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2-Exp"; def test_bs_1_speed(self):; symbols: TestDeepseekV32Basic, TestDeepseekV32DP, setUpClass, test_bs_1_speed
  - `docs/basic_usage/deepseek_v32.md` modified +30/-21 (51 lines); hunks: DeepSeek-V3.2-Speciale:
  - `python/sglang/srt/server_args.py` modified +16/-5 (21 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `test/srt/run_suite.py` modified +3/-2 (5 lines); hunks: TestFile("test_deepseek_v3_mtp.py", 275),
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `docs/basic_usage/deepseek_v32.md`; keywords observed in patches: test, attention, kv, spec, cache, deepep. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `docs/basic_usage/deepseek_v32.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16306 - [1/n]deepseek_v2.py Refactor: attention backend handlers and forward method definition

- Link: https://github.com/sgl-project/sglang/pull/16306
- Status/date: `merged`, created 2026-01-02, merged 2026-01-08; author `Fridge003`.
- Diff scope read: `5` files, `+255/-228`; areas: model wrapper, attention/backend; keywords: attention, cache, cuda, kv, mla, flash, fp8, quant, spec, triton.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +18/-228 (246 lines); hunks: import logging; from sglang.srt.layers.quantization.fp8 import Fp8Config; symbols: add_forward_absorb_core_attention_backend, AttnForwardMethod, _dispatch_mla_subtype, AttentionBackendRegistry:
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` added +182/-0 (182 lines); hunks: +from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph; symbols: AttentionBackendRegistry:, register, get_handler, _dispatch_mla_subtype
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` added +32/-0 (32 lines); hunks: +from enum import IntEnum, auto; symbols: AttnForwardMethod
  - `python/sglang/srt/models/deepseek_common/utils.py` added +23/-0 (23 lines); hunks: +from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
  - `python/sglang/srt/models/deepseek_common/__init__.py` added +0/-0 (0 lines)
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py`; keywords observed in patches: attention, cache, cuda, kv, mla, flash. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16380 - [DeepSeek 3.2] Support and optimize pipeline parallelis when context pipeline enabled

- Link: https://github.com/sgl-project/sglang/pull/16380
- Status/date: `merged`, created 2026-01-04, merged 2026-01-09; author `xu-yfei`.
- Diff scope read: `2` files, `+72/-36`; areas: attention/backend, scheduler/runtime; keywords: attention, cache, config, cuda, fp8, kv, lora, scheduler, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +57/-33 (90 lines); hunks: from __future__ import annotations; import torch_npu; symbols: __init__, __init__, _with_real_sm_count, _get_logits_head_gate
  - `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +15/-3 (18 lines); hunks: def event_loop_pp_disagg_decode(self: Scheduler):; def _pp_send_dict_to_next_stage(; symbols: event_loop_pp_disagg_decode, init_pp_loop_state, _pp_send_dict_to_next_stage, _pp_recv_proxy_tensors
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/scheduler_pp_mixin.py`; keywords observed in patches: attention, cache, config, cuda, fp8, kv. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/scheduler_pp_mixin.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16520 - fix: unimplemented methods in BaseIndexerMetadata

- Link: https://github.com/sgl-project/sglang/pull/16520
- Status/date: `merged`, created 2026-01-05, merged 2026-01-06; author `dougyster`.
- Diff scope read: `1` files, `+39/-1`; areas: kernel, tests/benchmarks; keywords: cache, kv, test, topk.
- Code diff details:
  - `test/registered/kernels/test_nsa_indexer.py` modified +39/-1 (40 lines); hunks: import unittest; def get_seqlens_expanded(self) -> torch.Tensor:; symbols: get_seqlens_expanded, get_indexer_kvcache_range, get_indexer_seq_len_cpu, get_token_to_batch_idx
- Optimization/support interpretation: The concrete diff surface is `test/registered/kernels/test_nsa_indexer.py`; keywords observed in patches: cache, kv, test, topk. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/kernels/test_nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16637 - [DSv32] Overlap indexer weights_proj during dual_stream decode

- Link: https://github.com/sgl-project/sglang/pull/16637
- Status/date: `merged`, created 2026-01-07, merged 2026-01-10; author `zianglih`.
- Diff scope read: `2` files, `+64/-22`; areas: model wrapper, attention/backend; keywords: cuda, lora, attention, cache, fp8, kv, quant, topk.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +34/-11 (45 lines); hunks: def forward_absorb_prepare(; def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_prepare, forward_absorb_prepare
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +30/-11 (41 lines); hunks: def _with_real_sm_count(self):; def forward_cuda(; symbols: _with_real_sm_count, _project_and_scale_head_gates, _get_logits_head_gate, forward_cuda
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: cuda, lora, attention, cache, fp8, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16758 - [DeepSeek V3.2] Enable trtllm NSA with bf16 kvcache

- Link: https://github.com/sgl-project/sglang/pull/16758
- Status/date: `merged`, created 2026-01-08, merged 2026-01-23; author `akhilg-nv`.
- Diff scope read: `2` files, `+118/-31`; areas: attention/backend; keywords: attention, cache, config, cuda, flash, kv, mla, fp4, fp8, lora.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +64/-28 (92 lines); hunks: "fa3",; class ServerArgs:; symbols: ServerArgs:, _generate_piecewise_cuda_graph_tokens, _set_default_nsa_kv_cache_dtype, _set_default_nsa_backends
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +54/-3 (57 lines); hunks: def topk_transform(; def __init__(; symbols: topk_transform, NativeSparseAttnBackend, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; keywords observed in patches: attention, cache, config, cuda, flash, kv. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16841 - [AMD] enable CUDA graph for NSA backend and fix NSA FP8 fused RMSNorm group quant

- Link: https://github.com/sgl-project/sglang/pull/16841
- Status/date: `merged`, created 2026-01-10, merged 2026-01-14; author `hubertlu-tw`.
- Diff scope read: `7` files, `+260/-81`; areas: model wrapper, attention/backend, kernel, scheduler/runtime; keywords: attention, cache, kv, fp8, cuda, triton, quant, spec, topk, lora.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +171/-55 (226 lines); hunks: from sglang.srt.utils import add_prefix, ceil_align, is_cuda, is_hip, is_npu; if TYPE_CHECKING:; symbols: BaseIndexerMetadata, get_page_table_64, get_page_table_1, get_seqlens_expanded
  - `python/sglang/srt/models/deepseek_v2.py` modified +58/-18 (76 lines); hunks: def forward_normal_prepare(; def forward_absorb_prepare(; symbols: forward_normal_prepare, forward_absorb_prepare
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +10/-3 (13 lines); hunks: import triton; def _set_k_and_s_triton(; symbols: _set_k_and_s_triton
  - `python/sglang/srt/server_args.py` modified +10/-3 (13 lines); hunks: def _handle_model_specific_adjustments(self):; def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, _handle_model_specific_adjustments
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +6/-2 (8 lines); hunks: set_mla_kv_buffer_triton,; GB = 1024 * 1024 * 1024; symbols: get_tensor_size_bytes, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`; keywords observed in patches: attention, cache, kv, fp8, cuda, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16881 - [DSv32] Add returning DSA topk indices

- Link: https://github.com/sgl-project/sglang/pull/16881
- Status/date: `closed`, created 2026-01-11, closed 2026-01-11; author `zianglih`.
- Diff scope read: `15` files, `+205/-2`; areas: model wrapper, attention/backend, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks; keywords: expert, topk, cache, config, lora, moe, scheduler, cuda, kv, processor.
- Code diff details:
  - `python/sglang/srt/layers/moe/routed_experts_capturer.py` modified +118/-0 (118 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, get_buffer_size_bytes, get_dsa_topk_indices_buffer_size_bytes
  - `python/sglang/srt/managers/detokenizer_manager.py` modified +17/-2 (19 lines); hunks: def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):; def _extract_routed_experts(self, recv_obj: BatchTokenIDOutput) -> List[List[int; symbols: _decode_batch_token_id_output, _extract_routed_experts, _extract_routed_experts, handle_batch_token_id_out
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +16/-0 (16 lines); hunks: def maybe_collect_routed_experts(self: Scheduler, req: Req):; def process_batch_result_prefill(; symbols: maybe_collect_routed_experts, maybe_collect_dsa_topk_indices, maybe_collect_customized_info, process_batch_result_prefill
  - `python/sglang/srt/managers/io_struct.py` modified +12/-0 (12 lines); hunks: class GenerateReqInput(BaseReq, APIServingTimingMixin):; def __getitem__(self, i):; symbols: GenerateReqInput, __getitem__, TokenizedGenerateReqInput, BatchTokenIDOutput
  - `python/sglang/srt/managers/schedule_batch.py` modified +11/-0 (11 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, reset_for_retract, ScheduleBatch
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `python/sglang/srt/managers/detokenizer_manager.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`; keywords observed in patches: expert, topk, cache, config, lora, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `python/sglang/srt/managers/detokenizer_manager.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16907 - Fix model loading for DeepSeek-V3.2-AWQ

- Link: https://github.com/sgl-project/sglang/pull/16907
- Status/date: `merged`, created 2026-01-11, merged 2026-02-15; author `bingps`.
- Diff scope read: `1` files, `+8/-4`; areas: model wrapper; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +8/-4 (12 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: n/a. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16916 - add doc for dsv32 cp+pp

- Link: https://github.com/sgl-project/sglang/pull/16916
- Status/date: `merged`, created 2026-01-12, merged 2026-01-12; author `whybeyoung`.
- Diff scope read: `1` files, `+114/-0`; areas: docs/config; keywords: benchmark, cache, config, cuda, doc, kv, moe, test.
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +114/-0 (114 lines); hunks: Latency: 29.545 s; Example usage:
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/deepseek_v32.md`; keywords observed in patches: benchmark, cache, config, cuda, doc, kv. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/deepseek_v32.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16961 - [DeepSeek v3.2] Opt MTP decode cuda batch sizes and nsa implementation

- Link: https://github.com/sgl-project/sglang/pull/16961
- Status/date: `merged`, created 2026-01-12, merged 2026-01-19; author `xu-yfei`.
- Diff scope read: `2` files, `+26/-12`; areas: attention/backend, kernel, scheduler/runtime; keywords: attention, cache, config, cuda, flash, kv, mla, moe, test.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +14/-5 (19 lines); hunks: def forward_extend(; def forward_extend(; symbols: forward_extend, forward_extend, forward_extend, forward_extend
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +12/-7 (19 lines); hunks: def set_torch_compile_config():; def get_batch_sizes_to_capture(model_runner: ModelRunner):; symbols: set_torch_compile_config, get_batch_sizes_to_capture, get_batch_sizes_to_capture, get_batch_sizes_to_capture
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`; keywords observed in patches: attention, cache, config, cuda, flash, kv. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16990 - [Ascend] fix dsv3.2 weight cast bug

- Link: https://github.com/sgl-project/sglang/pull/16990
- Status/date: `merged`, created 2026-01-13, merged 2026-01-13; author `MichelleWu351`.
- Diff scope read: `1` files, `+3/-2`; areas: quantization; keywords: flash, moe, quant.
- Code diff details:
  - `python/sglang/srt/layers/quantization/unquant.py` modified +3/-2 (5 lines); hunks: def create_weights(; def process_weights_after_loading(self, layer: torch.nn.Module) -> None:; symbols: create_weights, process_weights_after_loading, apply, process_weights_after_loading
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/unquant.py`; keywords observed in patches: flash, moe, quant. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/unquant.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17007 - [NPU]bugfix: fix for dsv3.2 and dsvl2

- Link: https://github.com/sgl-project/sglang/pull/17007
- Status/date: `merged`, created 2026-01-13, merged 2026-01-23; author `JiaruiChang5268`.
- Diff scope read: `5` files, `+129/-46`; areas: model wrapper, attention/backend, tests/benchmarks, docs/config; keywords: attention, cache, kv, config, lora, mla, quant, test, benchmark, cuda.
- Code diff details:
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +80/-46 (126 lines); hunks: def forward_mha_prepare_npu(; def forward_dsa_prepare_npu(; symbols: forward_mha_prepare_npu, forward_dsa_prepare_npu, forward_dsa_prepare_npu, forward_dsa_prepare_npu
  - `test/registered/ascend/llm_models/test_ascend_deepseek_v3_2_exp_w8a8.py` added +29/-0 (29 lines); hunks: +import unittest; symbols: TestDeepSeekV3_2ExpW8A8
  - `test/registered/ascend/vlm_models/test_ascend_deepseek_vl2.py` added +18/-0 (18 lines); hunks: +import unittest; symbols: TestGemmaModels, test_vlm_mmmu_benchmark
  - `python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunks: def _derive_model_shapes(self):; symbols: _derive_model_shapes
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-0 (1 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `test/registered/ascend/llm_models/test_ascend_deepseek_v3_2_exp_w8a8.py`, `test/registered/ascend/vlm_models/test_ascend_deepseek_vl2.py`; keywords observed in patches: attention, cache, kv, config, lora, mla. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `test/registered/ascend/llm_models/test_ascend_deepseek_v3_2_exp_w8a8.py`, `test/registered/ascend/vlm_models/test_ascend_deepseek_vl2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17024 - [PD] Fix DeepSeek V3.2 indexer cache transfer

- Link: https://github.com/sgl-project/sglang/pull/17024
- Status/date: `closed`, created 2026-01-13, closed 2026-03-19; author `ShangmingCai`.
- Diff scope read: `1` files, `+6/-10`; areas: misc; keywords: cache, kv.
- Code diff details:
  - `python/sglang/srt/disaggregation/prefill.py` modified +6/-10 (16 lines); hunks: def send_kv_chunk(; def send_kv_chunk(; symbols: send_kv_chunk, send_kv_chunk
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/disaggregation/prefill.py`; keywords observed in patches: cache, kv. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/disaggregation/prefill.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17076 - [DeepSeek V3.2] [Bugfix] slice indexer and padding fa3 when can not run cuda graph

- Link: https://github.com/sgl-project/sglang/pull/17076
- Status/date: `merged`, created 2026-01-14, merged 2026-02-02; author `xu-yfei`.
- Diff scope read: `4` files, `+58/-7`; areas: attention/backend, kernel, tests/benchmarks; keywords: attention, cache, kv, triton, fp8, test, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +28/-4 (32 lines); hunks: import triton.language as tl; def nsa_cp_round_robin_split_data(input_: Union[torch.Tensor, List]):; symbols: nsa_cp_round_robin_split_data, pad_nsa_cache_seqlens, cal_padded_tokens, pad_nsa_cache_seqlens
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +20/-2 (22 lines); hunks: def get_indexer_seq_len_cpu(self) -> torch.Tensor:; def _get_topk_paged(; symbols: get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx, _get_topk_paged
  - `test/registered/kernels/test_nsa_indexer.py` modified +7/-1 (8 lines); hunks: import unittest; def get_indexer_seq_len_cpu(self) -> torch.Tensor:; symbols: get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-0 (3 lines); hunks: def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:; symbols: get_indexer_kvcache_range, get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/kernels/test_nsa_indexer.py`; keywords observed in patches: attention, cache, kv, triton, fp8, test. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/kernels/test_nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17133 - [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab

- Link: https://github.com/sgl-project/sglang/pull/17133
- Status/date: `merged`, created 2026-01-15, merged 2026-01-16; author `xu-yfei`.
- Diff scope read: `6` files, `+959/-217`; areas: MoE/router, quantization, kernel, tests/benchmarks, docs/config; keywords: moe, triton, config, fp8, benchmark, cache, cuda, expert, quant, router.
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +337/-215 (552 lines); hunks: # Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py; sort_config,; symbols: MoeInputs:, KernelWrapper:, __init__, cuda_graph_wrapper
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`; keywords observed in patches: moe, triton, config, fp8, benchmark, cache. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17199 - [Feature] add feature mla_ag_after_qlora for dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/17199
- Status/date: `closed`, created 2026-01-16, closed 2026-02-26; author `JiaruiChang5268`.
- Diff scope read: `5` files, `+191/-82`; areas: model wrapper, attention/backend; keywords: lora, kv, attention, cache, mla, config, cuda, topk, fp8, quant.
- Code diff details:
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +119/-70 (189 lines); hunks: import torch_npu; def __init__(self, model_runner: ModelRunner):; symbols: __init__, init_forward_metadata, _generate_alibi_bias, generate_alibi_bias
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +34/-6 (40 lines); hunks: cp_split_and_rebuild_position,; def forward_mha_prepare_npu(; symbols: forward_mha_prepare_npu, forward_mha_prepare_npu, forward_dsa_prepare_npu, forward_dsa_prepare_npu
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +19/-3 (22 lines); hunks: from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode; def forward_npu(; symbols: forward_npu, forward_npu, forward_npu, forward_npu
  - `python/sglang/srt/models/deepseek_v2.py` modified +15/-2 (17 lines); hunks: _is_cpu = is_cpu(); def forward(; symbols: forward, forward, forward_prepare, forward_prepare
  - `python/sglang/srt/layers/communicator.py` modified +4/-1 (5 lines); hunks: _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip(); def __init__(self):; symbols: __init__, init_context, get_fn
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: lora, kv, attention, cache, mla, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17205 - [OPT] DeepSeekV3.2: optimize indexer weight_proj-mma performance

- Link: https://github.com/sgl-project/sglang/pull/17205
- Status/date: `merged`, created 2026-01-16, merged 2026-01-20; author `BJWang-ant`.
- Diff scope read: `1` files, `+5/-4`; areas: attention/backend; keywords: attention, config, quant.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +5/-4 (9 lines); hunks: def __init__(; def _with_real_sm_count(self):; symbols: __init__, _with_real_sm_count, _project_and_scale_head_gates, _get_logits_head_gate
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, config, quant. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17310 - [TileLang] Align TileLang NSA kernel with current TileLang and stabilize output

- Link: https://github.com/sgl-project/sglang/pull/17310
- Status/date: `closed`, created 2026-01-18, closed 2026-01-25; author `mmangkad`.
- Diff scope read: `1` files, `+56/-60`; areas: attention/backend, kernel; keywords: attention, config, fp8, kv, quant, spec, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +56/-60 (116 lines); hunks: pass_configs = {; def fast_round_scale(amax, fp8_max_inv):; symbols: fast_log2_ceil, fast_pow2, fast_round_scale, fast_round_scale
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`; keywords observed in patches: attention, config, fp8, kv, quant, spec. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17409 - [Fix]: correctly fetch ds32 config in tuning_fused_moe_triton

- Link: https://github.com/sgl-project/sglang/pull/17409
- Status/date: `merged`, created 2026-01-20, merged 2026-01-20; author `huangzhilin-hzl`.
- Diff scope read: `1` files, `+2/-2`; areas: MoE/router, kernel, tests/benchmarks; keywords: benchmark, config, expert, moe, topk, triton.
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +2/-2 (4 lines); hunks: from typing import Dict, List, TypedDict; def get_model_config(; symbols: BenchmarkConfig, get_model_config
- Optimization/support interpretation: The concrete diff surface is `benchmark/kernels/fused_moe_triton/common_utils.py`; keywords observed in patches: benchmark, config, expert, moe, topk, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `benchmark/kernels/fused_moe_triton/common_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17518 - [HotFix]Fix dtype mismatch in nsa indexer on AMD device

- Link: https://github.com/sgl-project/sglang/pull/17518
- Status/date: `merged`, created 2026-01-21, merged 2026-01-22; author `Fridge003`.
- Diff scope read: `1` files, `+1/-1`; areas: attention/backend; keywords: attention, cuda.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, cuda. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17523 - [AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI

- Link: https://github.com/sgl-project/sglang/pull/17523
- Status/date: `merged`, created 2026-01-21, merged 2026-01-28; author `michaelzhang-ai`.
- Diff scope read: `27` files, `+1540/-43`; areas: quantization, tests/benchmarks; keywords: test, benchmark, config, kv, attention, spec, cache, eagle, topk, cuda.
- Code diff details:
  - `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py` added +248/-0 (248 lines); hunks: +"""AMD DeepSeek-V3.2 GSM8K Completion Evaluation Test (8-GPU); symbols: ModelConfig:, __post_init__, get_display_name, get_one_example
  - `.github/workflows/nightly-test-amd.yml` modified +158/-35 (193 lines); hunks: on:; jobs:
  - `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` added +149/-0 (149 lines); hunks: +"""AMD Nightly performance benchmark for DeepSeek-V3.2 model (MTP variant).; symbols: generate_simple_markdown_report, TestNightlyDeepseekV32MTPPerformance, setUpClass, test_bench_one_batch
  - `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` added +142/-0 (142 lines); hunks: +"""MI35x DeepSeek-V3.2 TP+MTP GSM8K Accuracy Evaluation Test (8-GPU); symbols: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/accuracy/test_deepseek_v32_mtp_eval_amd.py` added +142/-0 (142 lines); hunks: +"""AMD DeepSeek-V3.2 TP+MTP GSM8K Accuracy Evaluation Test (8-GPU); symbols: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py`; keywords observed in patches: test, benchmark, config, kv, attention, spec. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17554 - Kernel: optimize decoding metadata in NSA multi-spec backend with fused kernels

- Link: https://github.com/sgl-project/sglang/pull/17554
- Status/date: `merged`, created 2026-01-22, merged 2026-02-14; author `Johnsonms`.
- Diff scope read: `7` files, `+2824/-54`; areas: attention/backend, kernel, tests/benchmarks; keywords: cache, mla, flash, attention, cuda, spec, config, benchmark, quant, scheduler.
- Code diff details:
  - `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py` added +1067/-0 (1067 lines); hunks: +"""; symbols: create_test_metadata, reference_copy_decode, reference_copy_target_verify, reference_copy_draft_extend
  - `python/sglang/jit_kernel/csrc/elementwise/fused_metadata_copy.cuh` added +722/-0 (722 lines); hunks: +/*; symbols: SourcePointers, DestinationPointers, FusedMetadataCopyParams, FusedMetadataCopyMultiParams
  - `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py` added +407/-0 (407 lines); hunks: +"""; symbols: verify_single_backend_fused_metadata_copy, check_tensor_equal, verify_multi_backend_fused_metadata_copy, check_tensor_equal
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +307/-51 (358 lines); hunks: compute_cu_seqlens,; # Reuse this workspace buffer across all NSA backend instances; symbols: NSAFlashMLAMetadata:, init_forward_metadata_replay_cuda_graph_from_precomputed, init_forward_metadata_replay_cuda_graph
  - `python/sglang/jit_kernel/fused_metadata_copy.py` added +316/-0 (316 lines); hunks: +"""; symbols: _jit_fused_metadata_copy_module, _jit_fused_metadata_copy_multi_module, fused_metadata_copy_cuda, parameters
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_metadata_copy.cuh`, `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py`; keywords observed in patches: cache, mla, flash, attention, cuda, spec. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_metadata_copy.cuh`, `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17633 - [AMD] CI - enable deepseekv3.2 on MI325-8gpu and merge perf/accuracy test suites into stage-b suites

- Link: https://github.com/sgl-project/sglang/pull/17633
- Status/date: `merged`, created 2026-01-23, merged 2026-01-28; author `yctseng0211`.
- Diff scope read: `9` files, `+88/-230`; areas: attention/backend, MoE/router, kernel, tests/benchmarks; keywords: test, attention, moe, triton.
- Code diff details:
  - `.github/workflows/pr-test-amd.yml` modified +47/-206 (253 lines); hunks: jobs:; jobs:
  - `scripts/ci/utils/slash_command_handler.py` modified +5/-8 (13 lines); hunks: def handle_rerun_stage(; symbols: handle_rerun_stage
  - `test/registered/eval/test_moe_eval_accuracy_large.py` modified +12/-0 (12 lines); hunks: python -m unittest test_moe_eval_accuracy_large.TestMoEEvalAccuracyLarge.test_mmlu; DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,; symbols: TestMoEEvalAccuracyLarge, setUpClass, setUpClass
  - `test/registered/amd/test_deepseek_v32_basic.py` modified +2/-6 (8 lines); hunks: write_github_step_summary,; def test_bs_1_speed(self):; symbols: test_bs_1_speed
  - `test/registered/amd/test_kimi_k2_instruct.py` modified +6/-2 (8 lines); hunks: from sglang.test.test_utils import (; def test_bs_1_speed(self):; symbols: test_bs_1_speed
- Optimization/support interpretation: The concrete diff surface is `.github/workflows/pr-test-amd.yml`, `scripts/ci/utils/slash_command_handler.py`, `test/registered/eval/test_moe_eval_accuracy_large.py`; keywords observed in patches: test, attention, moe, triton. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `.github/workflows/pr-test-amd.yml`, `scripts/ci/utils/slash_command_handler.py`, `test/registered/eval/test_moe_eval_accuracy_large.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17647 - [Perf] opt nsa backend init forward metada

- Link: https://github.com/sgl-project/sglang/pull/17647
- Status/date: `closed`, created 2026-01-23, closed 2026-03-01; author `Baidu-AIAK`.
- Diff scope read: `2` files, `+88/-64`; areas: attention/backend; keywords: attention, cuda, kv, triton, cache, mla, spec, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +27/-64 (91 lines); hunks: pad_nsa_cache_seqlens,; def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata, init_forward_metadata_replay_cuda_graph, init_forward_metadata_replay_cuda_graph
  - `python/sglang/srt/layers/attention/utils.py` modified +61/-0 (61 lines); hunks: def pad_sequence_with_mask(; symbols: pad_sequence_with_mask, seqlens_expand_kernel, seqlens_expand_triton
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py`; keywords observed in patches: attention, cuda, kv, triton, cache, mla. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17657 - [DeepSeek] Update tests and document for DeepSeek V3.2 NVFP4 checkpoint

- Link: https://github.com/sgl-project/sglang/pull/17657
- Status/date: `merged`, created 2026-01-23, merged 2026-01-27; author `Fridge003`.
- Diff scope read: `3` files, `+88/-0`; areas: quantization, tests/benchmarks, docs/config; keywords: fp4, fp8, flash, kv, moe, quant, test, attention, cache, config.
- Code diff details:
  - `test/srt/test_deepseek_v32_fp4_4gpu.py` added +79/-0 (79 lines); hunks: +import unittest; symbols: TestDeepseekV32FP4, setUpClass, tearDownClass, test_a_gsm8k
  - `docs/basic_usage/deepseek_v32.md` modified +8/-0 (8 lines); hunks: python3 -m sglang.launch_server \
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: TestFile("test_deepseek_v3_fp4_4gpu.py", 1500),
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_deepseek_v32_fp4_4gpu.py`, `docs/basic_usage/deepseek_v32.md`, `test/srt/run_suite.py`; keywords observed in patches: fp4, fp8, flash, kv, moe, quant. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/srt/test_deepseek_v32_fp4_4gpu.py`, `docs/basic_usage/deepseek_v32.md`, `test/srt/run_suite.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17662 - [DeepSeek-V3.2] Fix TRT-LLM NSA in target_verify/draft_extend

- Link: https://github.com/sgl-project/sglang/pull/17662
- Status/date: `merged`, created 2026-01-23, merged 2026-01-25; author `mmangkad`.
- Diff scope read: `1` files, `+18/-1`; areas: attention/backend; keywords: attention, cache, flash, kv, lora, mla, test, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +18/-1 (19 lines); hunks: def forward_extend(; def forward_decode(; symbols: forward_extend, forward_decode, _forward_trtllm, _forward_trtllm
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`; keywords observed in patches: attention, cache, flash, kv, lora, mla. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17688 - [DSv32] Overlap indexer qk projection and activation quant

- Link: https://github.com/sgl-project/sglang/pull/17688
- Status/date: `merged`, created 2026-01-25, merged 2026-01-28; author `zianglih`.
- Diff scope read: `1` files, `+4/-4`; areas: attention/backend; keywords: attention, cuda, fp8, lora, quant.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +4/-4 (8 lines); hunks: def forward_cuda(; symbols: forward_cuda
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, cuda, fp8, lora, quant. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17783 - [AMD] Update dsv3.2 AMD GPU docs and unify ROCm TileLang build

- Link: https://github.com/sgl-project/sglang/pull/17783
- Status/date: `merged`, created 2026-01-26, merged 2026-01-27; author `hubertlu-tw`.
- Diff scope read: `2` files, `+81/-88`; areas: docs/config; keywords: config, doc, test, cache, cuda, kv, quant.
- Code diff details:
  - `docker/rocm.Dockerfile` modified +71/-87 (158 lines); hunks: # Usage (to build SGLang ROCm docker image):; ARG LLVM_COMMIT="6520ace8227ffe2728148d5f3b9872a870b0a560"
  - `docs/basic_usage/deepseek_v32.md` modified +10/-1 (11 lines); hunks: Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](h; python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --
- Optimization/support interpretation: The concrete diff surface is `docker/rocm.Dockerfile`, `docs/basic_usage/deepseek_v32.md`; keywords observed in patches: config, doc, test, cache, cuda, kv. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docker/rocm.Dockerfile`, `docs/basic_usage/deepseek_v32.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17951 - Add tool call tests for DeepSeek V3.2 in nightly CI

- Link: https://github.com/sgl-project/sglang/pull/17951
- Status/date: `merged`, created 2026-01-29, merged 2026-01-29; author `harvenstar`.
- Diff scope read: `3` files, `+363/-5`; areas: model wrapper, scheduler/runtime, tests/benchmarks; keywords: test, spec, cuda, eagle, kv.
- Code diff details:
  - `python/sglang/test/tool_call_test_runner.py` added +320/-0 (320 lines); hunks: +import json; symbols: ToolCallTestParams:, ToolCallTestResult:, _call, _test_basic_format
  - `python/sglang/test/run_combined_tests.py` modified +32/-1 (33 lines); hunks: run_performance_test,; def run_combined_tests(; symbols: run_combined_tests, run_combined_tests, run_combined_tests, run_combined_tests
  - `test/registered/8-gpu-models/test_deepseek_v32.py` modified +11/-4 (15 lines); hunks: from sglang.test.performance_test_runner import PerformanceTestParams; '{"enable_multithread_load": true}',; symbols: test_deepseek_v32_all_variants, test_deepseek_v32_all_variants
- Optimization/support interpretation: The concrete diff surface is `python/sglang/test/tool_call_test_runner.py`, `python/sglang/test/run_combined_tests.py`, `test/registered/8-gpu-models/test_deepseek_v32.py`; keywords observed in patches: test, spec, cuda, eagle, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/test/tool_call_test_runner.py`, `python/sglang/test/run_combined_tests.py`, `test/registered/8-gpu-models/test_deepseek_v32.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18094 - support deepseekv3.2-piecewise-cuda-graph

- Link: https://github.com/sgl-project/sglang/pull/18094
- Status/date: `open`, created 2026-02-02; author `BJWang-ant`.
- Diff scope read: `15` files, `+243/-91`; areas: model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, docs/config; keywords: cuda, kv, attention, moe, topk, config, deepep, cache, expert, mla.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +148/-48 (196 lines); hunks: MaybeTboDeepEPDispatcher,; prepare_input_dp_with_cp_dsa,; symbols: forward, forward_deepep, _post_combine_hook, __init__
  - `python/sglang/srt/layers/radix_attention.py` modified +19/-19 (38 lines); hunks: def forward(; symbols: forward
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +12/-3 (15 lines); hunks: def forward_impl(self, hidden_states: torch.Tensor, topk_output: TopKOutput):; symbols: forward_impl, moe_forward_piecewise_cuda_graph_impl
  - `python/sglang/srt/layers/moe/topk.py` modified +12/-3 (15 lines); hunks: def is_power_of_two(n):; def biased_grouped_topk_gpu(; symbols: is_power_of_two, _mask_topk_ids_padded_region, _biased_grouped_topk_postprocess, biased_grouped_topk_gpu
  - `python/sglang/srt/layers/communicator.py` modified +7/-5 (12 lines); hunks: def prepare_attn(; symbols: prepare_attn, _tp_reduce_scatter
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; keywords observed in patches: cuda, kv, attention, moe, topk, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18126 - Fix dsv32 encode_messages

- Link: https://github.com/sgl-project/sglang/pull/18126
- Status/date: `merged`, created 2026-02-02, merged 2026-02-14; author `whybeyoung`.
- Diff scope read: `2` files, `+30/-5`; areas: misc; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/parser/jinja_template_utils.py` modified +16/-5 (21 lines); hunks: def process_content_for_template_format(; def process_content_for_template_format(; symbols: process_content_for_template_format, format, process_content_for_template_format, process_content_for_template_format
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +14/-0 (14 lines); hunks: def _apply_jinja_template(; symbols: _apply_jinja_template
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/parser/jinja_template_utils.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`; keywords observed in patches: n/a. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/parser/jinja_template_utils.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18174 - [Bugfix] Catch errors when DeepSeek-V3.2 generates malformed JSON

- Link: https://github.com/sgl-project/sglang/pull/18174
- Status/date: `merged`, created 2026-02-03, merged 2026-03-03; author `Muqi1029`.
- Diff scope read: `1` files, `+6/-3`; areas: misc; keywords: kv.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +6/-3 (9 lines); hunks: def _parse_parameters_from_xml(; symbols: _parse_parameters_from_xml
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/deepseekv32_detector.py`; keywords observed in patches: kv. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/deepseekv32_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18280 - [DeepSeek v3.2][Bugfix] get_index_k_scale_buffer support cp

- Link: https://github.com/sgl-project/sglang/pull/18280
- Status/date: `merged`, created 2026-02-05, merged 2026-03-17; author `xu-yfei`.
- Diff scope read: `4` files, `+22/-4`; areas: attention/backend, kernel, tests/benchmarks; keywords: attention, cache, kv, topk, fp8, test, triton.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +9/-3 (12 lines); hunks: def get_indexer_seq_len_cpu(self) -> torch.Tensor:; def _get_topk_ragged(; symbols: get_indexer_seq_len_cpu, get_indexer_seq_len, get_nsa_extend_len_cpu, _get_topk_ragged
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +8/-0 (8 lines); hunks: class NSAMetadata:; def get_cu_seqlens_k(self) -> torch.Tensor:; symbols: NSAMetadata:, get_cu_seqlens_k, get_indexer_kvcache_range, get_indexer_seq_len
  - `test/registered/kernels/test_nsa_indexer.py` modified +4/-0 (4 lines); hunks: def get_indexer_seq_len_cpu(self) -> torch.Tensor:; symbols: get_indexer_seq_len_cpu, get_indexer_seq_len, get_nsa_extend_len_cpu
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +1/-1 (2 lines); hunks: def _get_k_and_s_triton(; symbols: _get_k_and_s_triton
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/kernels/test_nsa_indexer.py`; keywords observed in patches: attention, cache, kv, topk, fp8, test. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/kernels/test_nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18297 - Deepseekv32 compatibility with transformers v5

- Link: https://github.com/sgl-project/sglang/pull/18297
- Status/date: `merged`, created 2026-02-05, merged 2026-02-10; author `JustinTong0323`.
- Diff scope read: `5` files, `+33/-19`; areas: model wrapper, attention/backend, docs/config; keywords: attention, config, mla, quant, cuda, kv, lora, moe, spec, topk.
- Code diff details:
  - `python/sglang/srt/configs/model_config.py` modified +13/-14 (27 lines); hunks: def _derive_model_shapes(self):; symbols: _derive_model_shapes
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-4 (17 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +4/-0 (4 lines); hunks: def __init__(; def set_nsa_prefill_impl(self, forward_batch: Optional[ForwardBatch] = None):; symbols: __init__, set_nsa_prefill_impl
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-1 (3 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: class Envs:; symbols: Envs:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; keywords observed in patches: attention, config, mla, quant, cuda, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18319 - [AMD] Use `tilelang` as default NSA attention backend dispatch on AMD Instinct

- Link: https://github.com/sgl-project/sglang/pull/18319
- Status/date: `merged`, created 2026-02-05, merged 2026-02-27; author `fxmarty-amd`.
- Diff scope read: `2` files, `+7/-2`; areas: attention/backend; keywords: attention, cache, flash, fp8, kv, mla.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +4/-1 (5 lines); hunks: def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:; symbols: _set_default_nsa_backends
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-1 (4 lines); hunks: def forward_extend(; symbols: forward_extend, forward_decode
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; keywords observed in patches: attention, cache, flash, fp8, kv, mla. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18389 - Nsa trtllm mla sparse fp8 support with Deepseek v3.2 NVFP4

- Link: https://github.com/sgl-project/sglang/pull/18389
- Status/date: `merged`, created 2026-02-07, merged 2026-02-16; author `rainj-me`.
- Diff scope read: `10` files, `+352/-183`; areas: model wrapper, attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: fp8, kv, mla, attention, cache, quant, lora, flash, cuda, triton.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +172/-66 (238 lines); hunks: nsa_cp_round_robin_split_q_seqs,; def __init__(; symbols: __init__, forward_extend, forward_extend, forward_extend
  - `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +13/-97 (110 lines); hunks: FlashInferMLAMultiStepDraftBackend,; logger = logging.getLogger(__name__); symbols: init_forward_metadata, init_mha_chunk_metadata, quantize_and_rope_for_fp8, pad_draft_extend_query
  - `python/sglang/srt/layers/attention/utils.py` modified +99/-0 (99 lines); hunks: import triton; def canonicalize_stride(tensor: torch.Tensor) -> torch.Tensor:; symbols: create_flashinfer_kv_indices_triton, canonicalize_stride, mla_quantize_and_rope_for_fp8, concat_mla_absorb_q_general
  - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +40/-0 (40 lines); hunks: def handle_max_mamba_cache(self: ModelRunner, total_rest_memory):; def init_memory_pool(self: ModelRunner, total_gpu_memory: int):; symbols: handle_max_mamba_cache, calculate_mla_kv_cache_dim, set_num_tokens_hybrid_swa, init_memory_pool
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +14/-18 (32 lines); hunks: def __init__(; def set_kv_buffer(; symbols: __init__, set_kv_buffer, set_mla_kv_buffer, set_kv_buffer
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/utils.py`; keywords observed in patches: fp8, kv, mla, attention, cache, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18526 - [AMD] Enable cudagraph for aiter nsa backend and add aiter impl for nsa pr…

- Link: https://github.com/sgl-project/sglang/pull/18526
- Status/date: `merged`, created 2026-02-10, merged 2026-02-27; author `wufann`.
- Diff scope read: `2` files, `+130/-3`; areas: attention/backend, kernel; keywords: attention, kv, topk, triton, cache, eagle, flash, mla, quant, spec.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +70/-3 (73 lines); hunks: _is_hip = is_hip(); def __init__(; symbols: __init__, forward_extend, _forward_aiter, _forward_aiter
  - `python/sglang/srt/layers/attention/nsa/triton_kernel.py` modified +60/-0 (60 lines); hunks: def act_quant(; symbols: act_quant, _get_valid_kv_indices_kernel, get_valid_kv_indices
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py`; keywords observed in patches: attention, kv, topk, triton, cache, eagle. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18542 - fix: fixed aux hidden state index out of range when using eagle3 with nsa cp

- Link: https://github.com/sgl-project/sglang/pull/18542
- Status/date: `open`, created 2026-02-10; author `echo-rain`.
- Diff scope read: `1` files, `+9/-1`; areas: model wrapper; keywords: cuda, moe.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: cuda, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18553 - Fix Bug on dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/18553
- Status/date: `merged`, created 2026-02-10, merged 2026-02-11; author `BourneSun0527`.
- Diff scope read: `2` files, `+16/-8`; areas: attention/backend; keywords: attention, eagle, fp8, quant, scheduler, spec.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +12/-6 (18 lines); hunks: import torch; def forward_npu(; symbols: forward_npu, forward_npu
  - `python/sglang/srt/managers/overlap_utils.py` modified +4/-2 (6 lines); hunks: import torch; symbols: _resolve_future_token_ids
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/overlap_utils.py`; keywords observed in patches: attention, eagle, fp8, quant, scheduler, spec. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/overlap_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18613 - [V3.2] Change default CP token split method to `--round-robin-split`

- Link: https://github.com/sgl-project/sglang/pull/18613
- Status/date: `merged`, created 2026-02-11, merged 2026-02-11; author `Fridge003`.
- Diff scope read: `3` files, `+5/-5`; areas: docs/config; keywords: cache, kv, attention, doc, fp8, moe, spec.
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +2/-2 (4 lines); hunks: DeepSeek-V3.2-Speciale:; Example:
  - `python/sglang/srt/server_args.py` modified +2/-2 (4 lines); hunks: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs:, add_cli_args
  - `docs/advanced_features/server_arguments.md` modified +1/-1 (2 lines); hunks: Please consult the documentation below and [server_args.py](https://github.com/s
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md`; keywords observed in patches: cache, kv, attention, doc, fp8, moe. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18876 - Add DeepSeek3.2 and GlmMoeDsa into moe tune

- Link: https://github.com/sgl-project/sglang/pull/18876
- Status/date: `merged`, created 2026-02-16, merged 2026-03-10; author `yuan-luo`.
- Diff scope read: `1` files, `+4/-0`; areas: MoE/router, kernel, tests/benchmarks; keywords: benchmark, config, expert, kv, moe, triton.
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +4/-0 (4 lines); hunks: def get_model_config(; def get_model_config(; symbols: get_model_config, get_model_config
- Optimization/support interpretation: The concrete diff surface is `benchmark/kernels/fused_moe_triton/common_utils.py`; keywords observed in patches: benchmark, config, expert, kv, moe, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `benchmark/kernels/fused_moe_triton/common_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18931 - Fix NSA FP8 KV cache path for both-trtllm MHA one-shot

- Link: https://github.com/sgl-project/sglang/pull/18931
- Status/date: `merged`, created 2026-02-17, merged 2026-02-20; author `mmangkad`.
- Diff scope read: `1` files, `+8/-1`; areas: model wrapper, attention/backend; keywords: attention, cache, fp8, kv, mla, quant, spec.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +8/-1 (9 lines); hunks: def forward_normal_prepare(; symbols: forward_normal_prepare
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`; keywords observed in patches: attention, cache, fp8, kv, mla, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18978 - [AMD] Fix mi35x dsv32 mtp nightly

- Link: https://github.com/sgl-project/sglang/pull/18978
- Status/date: `merged`, created 2026-02-18, merged 2026-02-19; author `bingxche`.
- Diff scope read: `1` files, `+1/-1`; areas: attention/backend; keywords: attention.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-1 (2 lines); hunks: # Control whether to use fused metadata copy kernel (default: enabled)
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`; keywords observed in patches: attention. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19016 - [FIX] NSA backend page_table overflow in speculative decoding target_verify

- Link: https://github.com/sgl-project/sglang/pull/19016
- Status/date: `merged`, created 2026-02-19, merged 2026-03-06; author `JustinTong0323`.
- Diff scope read: `1` files, `+3/-1`; areas: attention/backend; keywords: attention, cuda, spec.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-1 (4 lines); hunks: def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):; symbols: init_cuda_graph_state
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`; keywords observed in patches: attention, cuda, spec. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19041 - [DSv32] [GLM5] Improve Model Quality by Avoiding FP32 Precision Loss in `weights_proj`

- Link: https://github.com/sgl-project/sglang/pull/19041
- Status/date: `merged`, created 2026-02-20, merged 2026-02-22; author `zianglih`.
- Diff scope read: `4` files, `+48/-9`; areas: attention/backend, kernel, tests/benchmarks; keywords: config, cuda, attention, scheduler, test, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +17/-7 (24 lines); hunks: def _with_real_sm_count(self):; symbols: _with_real_sm_count, _project_and_scale_head_gates, _weights_proj_bf16_in_fp32_out, _project_and_scale_head_gates
  - `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +15/-0 (15 lines); hunks: class DeepGemmKernelType(IntEnum):; def create(kernel_type: DeepGemmKernelType, **kwargs):; symbols: DeepGemmKernelType, create, get_memory_requirement, execute
  - `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py` modified +14/-0 (14 lines); hunks: def gemm_nt_f8f8bf16(; symbols: gemm_nt_f8f8bf16, gemm_nt_bf16bf16f32, update_deep_gemm_config
  - `test/registered/kernels/test_nsa_indexer.py` modified +2/-2 (4 lines); hunks: from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler; "context_len": 2048,
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`; keywords observed in patches: config, cuda, attention, scheduler, test, topk. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19062 - [DSv32] Fix MTP and CP compatibility

- Link: https://github.com/sgl-project/sglang/pull/19062
- Status/date: `merged`, created 2026-02-20, merged 2026-02-21; author `vladnosiv`.
- Diff scope read: `1` files, `+5/-5`; areas: model wrapper; keywords: attention, config.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +5/-5 (10 lines); hunks: prepare_input_dp_with_cp_dsa,; def __init__(; symbols: __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_nextn.py`; keywords observed in patches: attention, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_nextn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19122 - [3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py

- Link: https://github.com/sgl-project/sglang/pull/19122
- Status/date: `merged`, created 2026-02-21, merged 2026-02-27; author `Fridge003`.
- Diff scope read: `9` files, `+906/-818`; areas: model wrapper, attention/backend, tests/benchmarks; keywords: attention, mla, cache, kv, fp8, lora, quant, cuda, triton, config.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +22/-811 (833 lines); hunks: from __future__ import annotations; MaybeTboDeepEPDispatcher,; symbols: DeepseekV2MLP, __init__, op_output, DeepseekV2AttentionMLA
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` added +492/-0 (492 lines); hunks: +from __future__ import annotations; symbols: DeepseekMLAForwardMixin:, init_mla_forward, forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` added +227/-0 (227 lines); hunks: +from __future__ import annotations; symbols: DeepseekMLARocmForwardMixin:, init_mla_fused_rope_rocm_forward, forward_absorb_fused_mla_rope_prepare, forward_absorb_fused_mla_rope_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` added +152/-0 (152 lines); hunks: +from __future__ import annotations; symbols: DeepseekMLACpuForwardMixin:, init_mla_fused_rope_cpu_forward, forward_absorb_fused_mla_rope_cpu_prepare, forward_absorb_fused_mla_rope_cpu_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py` modified +6/-0 (6 lines); hunks: from .forward_methods import AttnForwardMethod
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`; keywords observed in patches: attention, mla, cache, kv, fp8, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19148 - [DeepSeek-V3.2][JIT-kernel] Support nsa fuse store indexer k cache

- Link: https://github.com/sgl-project/sglang/pull/19148
- Status/date: `merged`, created 2026-02-22, merged 2026-02-26; author `yuan-luo`.
- Diff scope read: `4` files, `+307/-21`; areas: attention/backend, kernel, scheduler/runtime; keywords: fp8, cache, cuda, kv, quant, attention, topk.
- Code diff details:
  - `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh` added +124/-0 (124 lines); hunks: +#include <sgl_kernel/tensor.h>; symbols: FusedStoreCacheParam, void, int64_t, FusedStoreCacheIndexerKernel
  - `python/sglang/jit_kernel/fused_store_index_cache.py` added +103/-0 (103 lines); hunks: +"""; symbols: _jit_nsa_fused_store_module, can_use_nsa_fused_store, fused_store_index_k_cache
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +79/-21 (100 lines); hunks: import torch; def _forward_cuda_k_only(; symbols: _forward_cuda_k_only, forward_indexer, _store_index_k_cache, forward_cuda
  - `python/sglang/jit_kernel/utils.py` modified +1/-0 (1 lines); hunks: def __str__(self) -> str:; symbols: __str__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh`, `python/sglang/jit_kernel/fused_store_index_cache.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: fp8, cache, cuda, kv, quant, attention. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh`, `python/sglang/jit_kernel/fused_store_index_cache.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19319 - [deepseekv3.2] fix get_k_and_s_triton kernel for 128K seqlen case bug

- Link: https://github.com/sgl-project/sglang/pull/19319
- Status/date: `merged`, created 2026-02-25, merged 2026-03-11; author `BJWang-ant`.
- Diff scope read: `5` files, `+380/-81`; areas: attention/backend, kernel, scheduler/runtime, tests/benchmarks; keywords: triton, attention, kv, cache, cuda, fp8, test, topk.
- Code diff details:
  - `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py` added +191/-0 (191 lines); hunks: +import torch; symbols: golden_torch_gen, get_k_and_s_triton
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +105/-48 (153 lines); hunks: def execute(cls, *args, **kwargs):; def _get_s_triton_kernel(; symbols: execute, triton, _get_s_triton_kernel, _get_k_and_s_triton
  - `test/manual/layers/attention/nsa/test_index_buf_accessor.py` modified +46/-9 (55 lines); hunks: def test_get_k_and_s_correctness(; def test_get_k_and_s_correctness(; symbols: test_get_k_and_s_correctness, test_get_k_and_s_correctness, test_get_k_and_s_sequential_pages, test_get_k_and_s_sequential_pages
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +29/-22 (51 lines); hunks: def _should_chunk_mqa_logits(; def _get_topk_ragged(; symbols: _should_chunk_mqa_logits, _get_topk_ragged, _get_topk_ragged, _get_topk_ragged
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +9/-2 (11 lines); hunks: def get_index_k_scale_continuous(; def get_index_k_scale_buffer(; symbols: get_index_k_scale_continuous, get_index_k_scale_buffer, get_index_k_scale_buffer, set_index_k_scale_buffer
- Optimization/support interpretation: The concrete diff surface is `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py`; keywords observed in patches: triton, attention, kv, cache, cuda, fp8. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19367 - Fix NSA CP positions mismatch in eagle NextN model

- Link: https://github.com/sgl-project/sglang/pull/19367
- Status/date: `merged`, created 2026-02-25, merged 2026-02-26; author `alisonshao`.
- Diff scope read: `1` files, `+2/-0`; areas: model wrapper; keywords: expert.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +2/-0 (2 lines); hunks: can_cp_split,; def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_nextn.py`; keywords observed in patches: expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_nextn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19428 - [Feature] add feature mla_ag_after_qlora for dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/19428
- Status/date: `merged`, created 2026-02-26, merged 2026-03-02; author `JiaruiChang5268`.
- Diff scope read: `5` files, `+101/-9`; areas: model wrapper, attention/backend; keywords: lora, attention, kv, quant, cache, config, cuda, fp8, mla, topk.
- Code diff details:
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +35/-3 (38 lines); hunks: import torch; def forward_mha_prepare_npu(; symbols: forward_mha_prepare_npu, forward_mha_prepare_npu, forward_mla_prepare_npu, forward_mla_prepare_npu
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +33/-2 (35 lines); hunks: fused_store_index_k_cache,; is_nsa_enable_prefill_cp,; symbols: forward_npu, forward_npu, forward_npu, forward_npu
  - `python/sglang/srt/models/deepseek_v2.py` modified +26/-3 (29 lines); hunks: def forward(; def forward_prepare(; symbols: forward, forward_prepare, forward_prepare, __init__
  - `python/sglang/srt/layers/communicator.py` modified +5/-1 (6 lines); hunks: from sglang.srt.distributed.device_communicators.pynccl_allocator import (; _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip(); symbols: __init__, init_context, get_fn
  - `python/sglang/srt/environ.py` modified +2/-0 (2 lines); hunks: class Envs:; symbols: Envs:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: lora, attention, kv, quant, cache, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19536 - [Perf] Optimize NSA backend metadata under MTP

- Link: https://github.com/sgl-project/sglang/pull/19536
- Status/date: `merged`, created 2026-02-28, merged 2026-03-01; author `b8zhong`.
- Diff scope read: `2` files, `+85/-64`; areas: attention/backend; keywords: attention, cuda, kv, triton, cache, fp8, mla, quant, spec.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +24/-64 (88 lines); hunks: from sglang.srt.layers.attention.utils import (; def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata, init_forward_metadata_replay_cuda_graph, init_forward_metadata_replay_cuda_graph
  - `python/sglang/srt/layers/attention/utils.py` modified +61/-0 (61 lines); hunks: def pad_sequence_with_mask(; symbols: pad_sequence_with_mask, seqlens_expand_kernel, seqlens_expand_triton
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py`; keywords observed in patches: attention, cuda, kv, triton, cache, fp8. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19945 - [AMD] Tilelang sparse fwd for dsv32 mi355/mi300

- Link: https://github.com/sgl-project/sglang/pull/19945
- Status/date: `merged`, created 2026-03-05, merged 2026-03-24; author `1am9trash`.
- Diff scope read: `1` files, `+141/-95`; areas: attention/backend, kernel; keywords: attention, kv, mla, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +141/-95 (236 lines); hunks: def sparse_mla_fwd_decode_partial(; def sparse_mla_fwd_decode_partial(; symbols: sparse_mla_fwd_decode_partial, sparse_mla_fwd_decode_partial, main, main
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`; keywords observed in patches: attention, kv, mla, topk. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19987 - [AMD] Fix nightly GLM-5 failures: Fix NSA indexer tensor aliasing on ROCm during CUDA graph capture

- Link: https://github.com/sgl-project/sglang/pull/19987
- Status/date: `closed`, created 2026-03-05, closed 2026-03-05; author `michaelzhang-ai`.
- Diff scope read: `1` files, `+7/-0`; areas: attention/backend; keywords: attention, cuda.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-0 (7 lines); hunks: def _get_q_k_bf16(; symbols: _get_q_k_bf16
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, cuda. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20086 - [V32/GLM5] Change default setting of V32 nvfp4 on TP4

- Link: https://github.com/sgl-project/sglang/pull/20086
- Status/date: `merged`, created 2026-03-07, merged 2026-03-07; author `Fridge003`.
- Diff scope read: `1` files, `+15/-6`; areas: misc; keywords: attention, cache, flash, fp8, kv, mla.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +15/-6 (21 lines); hunks: def _set_default_nsa_kv_cache_dtype(self, major: int) -> str:; def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:; symbols: _set_default_nsa_kv_cache_dtype, _set_default_nsa_backends
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`; keywords observed in patches: attention, cache, flash, fp8, kv, mla. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20326 - [Doc] Add DSA/NSA attention backend to support matrix

- Link: https://github.com/sgl-project/sglang/pull/20326
- Status/date: `merged`, created 2026-03-11, merged 2026-03-11; author `mvanhorn`.
- Diff scope read: `1` files, `+19/-1`; areas: attention/backend, docs/config; keywords: attention, cache, cuda, doc, flash, fp8, kv, mla, spec.
- Code diff details:
  - `docs/advanced_features/attention_backend.md` modified +19/-1 (20 lines); hunks: Multimodal attention is selected by `--mm-attention-backend`. The "MultiModal" c; GDN models are hybrid: the full-attention layers still require a standard `--a
- Optimization/support interpretation: The concrete diff surface is `docs/advanced_features/attention_backend.md`; keywords observed in patches: attention, cache, cuda, doc, flash, fp8. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/advanced_features/attention_backend.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20343 - HiSparse for Sparse Attention

- Link: https://github.com/sgl-project/sglang/pull/20343
- Status/date: `merged`, created 2026-03-11, merged 2026-03-23; author `xiezhq-hermann`.
- Diff scope read: `20` files, `+1692/-59`; areas: attention/backend, kernel, multimodal/processor, scheduler/runtime; keywords: cache, cuda, kv, mla, config, attention, lora, quant, scheduler, test.
- Code diff details:
  - `python/sglang/srt/managers/hisparse_coordinator.py` added +596/-0 (596 lines); hunks: +# to be combined with the sparse coordinator class and sparse algorithm family; symbols: and, HiSparseAct, HiSparseCoordinator:, __init__
  - `python/sglang/jit_kernel/csrc/hisparse.cuh` added +390/-0 (390 lines); hunks: +#include <sgl_kernel/tensor.h>; symbols: int, int32_t, int32_t, void
  - `python/sglang/srt/mem_cache/hisparse_memory_pool.py` added +341/-0 (341 lines); hunks: +# mapping on device memory, host memory and memory allocator; symbols: HiSparseNSATokenToKVPool, __init__, register_mapping, translate_loc_to_hisparse_device
  - `python/sglang/srt/managers/scheduler.py` modified +85/-23 (108 lines); hunks: from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config; ); symbols: __init__, init_cache_with_memory_pool, handle_batch_embedding_request, stash_chunked_request
  - `python/sglang/jit_kernel/hisparse.py` added +88/-0 (88 lines); hunks: +from __future__ import annotations; symbols: _jit_sparse_module, load_cache_to_device_buffer_mla
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/managers/hisparse_coordinator.py`, `python/sglang/jit_kernel/csrc/hisparse.cuh`, `python/sglang/srt/mem_cache/hisparse_memory_pool.py`; keywords observed in patches: cache, cuda, kv, mla, config, attention. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/managers/hisparse_coordinator.py`, `python/sglang/jit_kernel/csrc/hisparse.cuh`, `python/sglang/srt/mem_cache/hisparse_memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20438 - [Perf] Overlap NSA-CP key all-gather with query computation for DeepSeek-V3.2

- Link: https://github.com/sgl-project/sglang/pull/20438
- Status/date: `merged`, created 2026-03-12, merged 2026-03-24; author `Baidu-AIAK`.
- Diff scope read: `1` files, `+19/-0`; areas: attention/backend; keywords: attention, cuda.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +19/-0 (19 lines); hunks: def _get_q_k_bf16(; symbols: _get_q_k_bf16
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, cuda. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20492 - [BugFix] bug fix for DeepSeek eagle3 in Attn-DP mode

- Link: https://github.com/sgl-project/sglang/pull/20492
- Status/date: `merged`, created 2026-03-13, merged 2026-03-19; author `khalil2ji3mp6`.
- Diff scope read: `1` files, `+2/-2`; areas: model wrapper; keywords: attention, expert, moe.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: get_moe_expert_parallel_world_size,; from sglang.srt.layers.dp_attention import (; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: attention, expert, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20534 - Transfer FP8 K/K_scale for CP indexer prefill gather

- Link: https://github.com/sgl-project/sglang/pull/20534
- Status/date: `open`, created 2026-03-13; author `huangzhilin-hzl`.
- Diff scope read: `1` files, `+35/-8`; areas: attention/backend; keywords: attention, cache, cuda, fp8, kv, quant.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +35/-8 (43 lines); hunks: def _get_q_k_bf16(; def _store_index_k_cache(; symbols: _get_q_k_bf16, _get_k_bf16, _store_index_k_cache
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, cache, cuda, fp8, kv, quant. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20606 - FIX: (NSA) Compute topk_indices_offset when NSA prefill flashmla_sparse is used with FP8 KV cache

- Link: https://github.com/sgl-project/sglang/pull/20606
- Status/date: `merged`, created 2026-03-15, merged 2026-03-26; author `JackChuang`.
- Diff scope read: `1` files, `+20/-4`; areas: attention/backend; keywords: attention, cache, flash, kv, mla, spec, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +20/-4 (24 lines); hunks: def topk_transform(; def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: topk_transform, init_forward_metadata, forward_extend, set_nsa_prefill_impl
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`; keywords observed in patches: attention, cache, flash, kv, mla, spec. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20840 - [AMD] Fix dpsk-v32 accuracy issue on mi355

- Link: https://github.com/sgl-project/sglang/pull/20840
- Status/date: `merged`, created 2026-03-18, merged 2026-03-18; author `1am9trash`.
- Diff scope read: `1` files, `+1/-0`; areas: quantization; keywords: fp8, quant, triton.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +1/-0 (1 lines); hunks: def use_aiter_triton_gemm_w8a8_tuned_gfx950(n: int, k: int) -> bool:; symbols: use_aiter_triton_gemm_w8a8_tuned_gfx950
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/fp8_utils.py`; keywords observed in patches: fp8, quant, triton. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/fp8_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20984 - Fix DeepSeek V32 FP4 test

- Link: https://github.com/sgl-project/sglang/pull/20984
- Status/date: `merged`, created 2026-03-20, merged 2026-03-20; author `Fridge003`.
- Diff scope read: `3` files, `+20/-1`; areas: quantization, tests/benchmarks; keywords: test, fp4, quant, cache.
- Code diff details:
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +9/-0 (9 lines); hunks: +import os; def setUpClass(cls):; symbols: setUpClass, setUpClass
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +9/-0 (9 lines); hunks: +import os; def setUpClass(cls):; symbols: setUpClass, setUpClass
  - `python/sglang/test/test_utils.py` modified +2/-1 (3 lines); hunks: def popen_launch_server(; def popen_launch_server(; symbols: popen_launch_server, popen_launch_server
- Optimization/support interpretation: The concrete diff surface is `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/test_utils.py`; keywords observed in patches: test, fp4, quant, cache. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/test_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21003 - Revert "Fix DeepSeek V32 FP4 test"

- Link: https://github.com/sgl-project/sglang/pull/21003
- Status/date: `merged`, created 2026-03-20, merged 2026-03-20; author `merrymercy`.
- Diff scope read: `3` files, `+1/-20`; areas: quantization, tests/benchmarks; keywords: test, fp4, quant, cache.
- Code diff details:
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-9 (9 lines); hunks: -import os; def setUpClass(cls):; symbols: setUpClass, setUpClass
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-9 (9 lines); hunks: -import os; def setUpClass(cls):; symbols: setUpClass, setUpClass
  - `python/sglang/test/test_utils.py` modified +1/-2 (3 lines); hunks: def popen_launch_server(; def popen_launch_server(; symbols: popen_launch_server, popen_launch_server
- Optimization/support interpretation: The concrete diff surface is `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/test_utils.py`; keywords observed in patches: test, fp4, quant, cache. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/test_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21192 - Fix CP in-seq-split method for DeepSeek V32 and update related tests

- Link: https://github.com/sgl-project/sglang/pull/21192
- Status/date: `merged`, created 2026-03-23, merged 2026-03-23; author `Fridge003`.
- Diff scope read: `7` files, `+162/-97`; areas: model wrapper, tests/benchmarks; keywords: test, cuda, kv, spec, attention, config, eagle, topk, benchmark, cache.
- Code diff details:
  - `test/registered/cp/test_deepseek_v32_cp_single_node.py` added +157/-0 (157 lines); hunks: +import unittest; symbols: TestDeepseekV32CPInSeqSplit, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py` removed +0/-92 (92 lines); hunks: -import unittest; symbols: TestDeepseekV32CPSingleNode, for, test_deepseek_v32_cp_variants
  - `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `test/manual/nightly/test_deepseek_v32_perf.py` modified +1/-1 (2 lines); hunks: from sglang.test.nightly_utils import NightlyBenchmarkRunner
  - `test/registered/8-gpu-models/test_deepseek_v32_basic.py` modified +1/-1 (2 lines); hunks: register_cuda_ci(est_time=360, suite="stage-c-test-8-gpu-h200"); symbols: TestDeepseekV32DP
- Optimization/support interpretation: The concrete diff surface is `test/registered/cp/test_deepseek_v32_cp_single_node.py`, `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: test, cuda, kv, spec, attention, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/cp/test_deepseek_v32_cp_single_node.py`, `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21249 - Support allreduce fusion with cp

- Link: https://github.com/sgl-project/sglang/pull/21249
- Status/date: `merged`, created 2026-03-24, merged 2026-04-20; author `Shunkangz`.
- Diff scope read: `4` files, `+201/-27`; areas: attention/backend, scheduler/runtime; keywords: attention, flash, cuda, moe, config, expert, spec.
- Code diff details:
  - `python/sglang/srt/layers/flashinfer_comm_fusion.py` modified +178/-22 (200 lines); hunks: from sglang.srt.distributed import (; logger = logging.getLogger(__name__); symbols: _always_disable_fabric, _FixedTorchDistBackend, __init__, bcast
  - `python/sglang/srt/model_executor/model_runner.py` modified +22/-0 (22 lines); hunks: def initialize(self, pre_model_load_memory: float):; def kernel_warmup(self):; symbols: initialize, kernel_warmup, _pre_initialize_flashinfer_allreduce_workspace, _should_run_flashinfer_autotune
  - `python/sglang/srt/layers/communicator.py` modified +1/-4 (5 lines); hunks: def apply_flashinfer_allreduce_fusion(batch_size: int):; def prepare_attn(; symbols: apply_flashinfer_allreduce_fusion, prepare_attn
  - `python/sglang/srt/server_args.py` modified +0/-1 (1 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/flashinfer_comm_fusion.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/communicator.py`; keywords observed in patches: attention, flash, cuda, moe, config, expert. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/flashinfer_comm_fusion.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/communicator.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21259 - [HiCache & HybridModel] mooncake backend support DSA & mamba model

- Link: https://github.com/sgl-project/sglang/pull/21259
- Status/date: `merged`, created 2026-03-24, merged 2026-04-14; author `huangtingwei9988`.
- Diff scope read: `8` files, `+760/-232`; areas: scheduler/runtime, tests/benchmarks; keywords: cache, kv, mla, config, attention, cuda, quant, test.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool_host.py` modified +230/-68 (298 lines); hunks: logger = logging.getLogger(__name__); def __init__(; symbols: synchronized, __init__, __init__, __init__
  - `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` modified +180/-36 (216 lines); hunks: import time; HiCacheStorage,; symbols: __init__, register_mem_pool_host, register_mem_host_pool_v2, _tag_keys
  - `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` added +212/-0 (212 lines); hunks: +from __future__ import annotations; symbols: build_nsa_hybrid_stack, layer_mapper, build_mamba_hybrid_stack, kv_layer_mapper
  - `python/sglang/srt/mem_cache/hiradix_cache.py` modified +83/-32 (115 lines); hunks: MatchPrefixParams,; from sglang.srt.mem_cache.memory_pool_host import (; symbols: __init__, __init__, __init__, attach_storage_backend
  - `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py` modified +9/-88 (97 lines); hunks: ); def __init__(self, params: CacheInitParams, server_args: ServerArgs):; symbols: __init__, kv_layer_mapper, mamba_layer_mapper, mamba_layer_mapper
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`; keywords observed in patches: cache, kv, mla, config, attention, cuda. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21337 - Workaround of DSA performance drop on B200 + DP

- Link: https://github.com/sgl-project/sglang/pull/21337
- Status/date: `merged`, created 2026-03-24, merged 2026-03-25; author `Fridge003`.
- Diff scope read: `1` files, `+11/-5`; areas: misc; keywords: cache, cuda, fp4, fp8, kv, quant, spec.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +11/-5 (16 lines); hunks: def _generate_piecewise_cuda_graph_tokens(self):; def _set_default_nsa_kv_cache_dtype(self, major: int) -> str:; symbols: _generate_piecewise_cuda_graph_tokens, _set_default_nsa_kv_cache_dtype, _set_default_nsa_kv_cache_dtype, _set_default_nsa_kv_cache_dtype
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`; keywords observed in patches: cache, cuda, fp4, fp8, kv, quant. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21405 - Enable IndexCache for DeepSeek V3.2

- Link: https://github.com/sgl-project/sglang/pull/21405
- Status/date: `merged`, created 2026-03-25, merged 2026-04-05; author `jinyouzhi`.
- Diff scope read: `4` files, `+196/-20`; areas: model wrapper, attention/backend, scheduler/runtime, tests/benchmarks; keywords: topk, cache, cuda, kv, attention, config, lora, mla, expert, fp4.
- Code diff details:
  - `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` added +117/-0 (117 lines); hunks: +import unittest; symbols: TestDeepseekV32IndexTopkPattern, setUpClass, tearDownClass, test_a_gsm8k
  - `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, op_prepare
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +27/-13 (40 lines); hunks: def forward_absorb_prepare(; def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla
  - `python/sglang/srt/models/deepseek_nextn.py` modified +1/-1 (2 lines); hunks: def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; keywords observed in patches: topk, cache, cuda, kv, attention, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21468 - [NPU] Update DeepSeek-V3.2 model deployment instructions in documentation

- Link: https://github.com/sgl-project/sglang/pull/21468
- Status/date: `merged`, created 2026-03-26, merged 2026-03-30; author `MichelleWu351`.
- Diff scope read: `1` files, `+96/-148`; areas: docs/config; keywords: attention, cache, config, cuda, deepep, doc, eagle, mla, moe, quant.
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +96/-148 (244 lines); hunks: you encounter issues or have any questions, please [open an issue](https://githu; We tested it based on the `RANDOM` dataset.
- Optimization/support interpretation: The concrete diff surface is `docs/platforms/ascend/ascend_npu_best_practice.md`; keywords observed in patches: attention, cache, config, cuda, deepep, doc. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/platforms/ascend/ascend_npu_best_practice.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21511 - [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend

- Link: https://github.com/sgl-project/sglang/pull/21511
- Status/date: `merged`, created 2026-03-27, merged 2026-04-03; author `1am9trash`.
- Diff scope read: `6` files, `+517/-77`; areas: model wrapper, attention/backend, kernel, scheduler/runtime; keywords: cache, kv, fp8, mla, attention, quant, lora, topk, triton, config.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +307/-42 (349 lines); hunks: +from functools import lru_cache; def fast_round_scale(amax, fp8_max_inv):; symbols: fast_round_scale, _pick_inner_iter, act_quant_kernel, main
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +79/-18 (97 lines); hunks: def forward_absorb_prepare(; def forward_absorb_core(; symbols: forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla, _skip_rope_for_nsa_tilelang_fused
  - `python/sglang/srt/mem_cache/utils.py` modified +87/-0 (87 lines); hunks: def set_mla_kv_buffer_triton(; symbols: set_mla_kv_buffer_triton, set_mla_kv_buffer_fp8_quant_kernel, set_mla_kv_buffer_triton_fp8_quant, set_mla_kv_scale_buffer_kernel
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +32/-15 (47 lines); hunks: quantize_k_cache,; _is_cpu = is_cpu(); symbols: get_tensor_size_bytes, set_mla_kv_buffer
  - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +11/-1 (12 lines); hunks: from sglang.srt.utils.common import (; def __post_init__(self):; symbols: __post_init__, ModelRunnerKVCacheMixin:, calculate_mla_kv_cache_dim
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/mem_cache/utils.py`; keywords observed in patches: cache, kv, fp8, mla, attention, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/mem_cache/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21585 - [CI] Move v32 cp test to deepep running suite

- Link: https://github.com/sgl-project/sglang/pull/21585
- Status/date: `merged`, created 2026-03-28, merged 2026-03-28; author `Fridge003`.
- Diff scope read: `1` files, `+1/-1`; areas: tests/benchmarks; keywords: cuda, deepep, test.
- Code diff details:
  - `test/registered/cp/test_deepseek_v32_cp_single_node.py` modified +1/-1 (2 lines); hunks: write_github_step_summary,
- Optimization/support interpretation: The concrete diff surface is `test/registered/cp/test_deepseek_v32_cp_single_node.py`; keywords observed in patches: cuda, deepep, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/cp/test_deepseek_v32_cp_single_node.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21599 - [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1

- Link: https://github.com/sgl-project/sglang/pull/21599
- Status/date: `merged`, created 2026-03-28, merged 2026-04-20; author `alphabetc1`.
- Diff scope read: `13` files, `+1296/-33`; areas: kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: spec, config, cuda, eagle, attention, topk, cache, kv, quant, test.
- Code diff details:
  - `benchmark/bench_adaptive_speculative.py` added +263/-0 (263 lines); hunks: +"""Benchmark adaptive speculative decoding against static baselines.; symbols: build_phase_plan, send_request, run_phase, summarize_phases
  - `test/registered/unit/spec/test_adaptive_spec_params.py` added +195/-0 (195 lines); hunks: +import unittest; symbols: TestAdaptiveSpeculativeParams, test_initial_steps_snap_to_nearest_candidate_preferring_larger_step, test_update_respects_warmup_and_interval, test_empty_batches_do_not_consume_warmup_or_shift_steps
  - `test/registered/spec/eagle/test_adaptive_speculative.py` added +170/-0 (170 lines); hunks: +import json; symbols: TestAdaptiveSpeculativeServer, setUpClass, tearDownClass, _get_internal_state
  - `python/sglang/srt/speculative/eagle_worker.py` modified +162/-4 (166 lines); hunks: import logging; alloc_token_slots,; symbols: __init__, __init__, init_cuda_graphs, apply_runtime_state
  - `docs/advanced_features/adaptive_speculative_decoding.md` added +156/-0 (156 lines); hunks: +# Adaptive Speculative Decoding
- Optimization/support interpretation: The concrete diff surface is `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py`; keywords observed in patches: spec, config, cuda, eagle, attention, topk. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21623 - [Test] Add unit tests for encoding_dsv32.py

- Link: https://github.com/sgl-project/sglang/pull/21623
- Status/date: `open`, created 2026-03-29; author `dondetir`.
- Diff scope read: `1` files, `+871/-0`; areas: tests/benchmarks; keywords: config, spec, test.
- Code diff details:
  - `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py` added +871/-0 (871 lines); hunks: +"""Unit tests for encoding_dsv32.py — no server, no model loading.; symbols: _make_tool, _make_tool_call, _parse_dsml_args, TestEncodeArgumentsToDsml
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py`; keywords observed in patches: config, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21783 - [DSA] Support trtllm sparse mla kernel for prefill batches

- Link: https://github.com/sgl-project/sglang/pull/21783
- Status/date: `merged`, created 2026-03-31, merged 2026-04-01; author `Fridge003`.
- Diff scope read: `3` files, `+12/-14`; areas: attention/backend, tests/benchmarks; keywords: attention, cache, flash, mla, topk, kv, spec, test.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +0/-11 (11 lines); hunks: def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:; def _handle_model_specific_adjustments(self):; symbols: _set_default_nsa_backends, _handle_model_specific_adjustments
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +9/-0 (9 lines); hunks: def forward_extend(; def _forward_trtllm(; symbols: forward_extend, _forward_trtllm, _forward_trtllm
  - `python/sglang/test/run_eval.py` modified +3/-3 (6 lines); hunks: def get_thinking_kwargs(args):; def run_eval(args):; symbols: get_thinking_kwargs, run_eval
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/test/run_eval.py`; keywords observed in patches: attention, cache, flash, mla, topk, kv. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/test/run_eval.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21914 - [DSA] Set trtllm kernels as default for Blackwell

- Link: https://github.com/sgl-project/sglang/pull/21914
- Status/date: `merged`, created 2026-04-02, merged 2026-04-02; author `Fridge003`.
- Diff scope read: `1` files, `+2/-7`; areas: misc; keywords: cache, fp4, fp8, kv, quant.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +2/-7 (9 lines); hunks: def _set_default_nsa_kv_cache_dtype(self, major: int, quantization: str) -> str:; def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:; symbols: _set_default_nsa_kv_cache_dtype, _set_default_nsa_backends
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`; keywords observed in patches: cache, fp4, fp8, kv, quant. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21932 - [HiSparse] Optimize the scheduling of decode backup.

- Link: https://github.com/sgl-project/sglang/pull/21932
- Status/date: `merged`, created 2026-04-02, merged 2026-04-07; author `huangtingwei9988`.
- Diff scope read: `2` files, `+42/-9`; areas: scheduler/runtime; keywords: cuda, topk.
- Code diff details:
  - `python/sglang/srt/managers/hisparse_coordinator.py` modified +36/-9 (45 lines); hunks: def __init__(; def _eager_backup_previous_token(; symbols: __init__, _eager_backup_previous_token, _eager_backup_previous_token, wait_for_pending_backup
  - `python/sglang/srt/model_executor/model_runner.py` modified +6/-0 (6 lines); hunks: def _forward_raw(; symbols: _forward_raw
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/managers/hisparse_coordinator.py`, `python/sglang/srt/model_executor/model_runner.py`; keywords observed in patches: cuda, topk. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/managers/hisparse_coordinator.py`, `python/sglang/srt/model_executor/model_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22003 - Support moe_dp_size = 1 for various attention_cp_size

- Link: https://github.com/sgl-project/sglang/pull/22003
- Status/date: `merged`, created 2026-04-03, merged 2026-04-20; author `Shunkangz`.
- Diff scope read: `8` files, `+276/-25`; areas: model wrapper, attention/backend, MoE/router, tests/benchmarks; keywords: moe, attention, cuda, config, expert, flash, fp4, spec, test.
- Code diff details:
  - `python/sglang/srt/layers/communicator.py` modified +164/-10 (174 lines); hunks: get_dp_global_num_tokens,; class ScatterMode(Enum):; symbols: ScatterMode, model_input_output, _compute_layer_input_mode, _compute_mlp_mode
  - `test/registered/4-gpu-models/test_qwen3_30b.py` modified +55/-0 (55 lines); hunks: def test_gsm8k(self):; symbols: test_gsm8k, TestQwen330BCP, setUpClass, tearDownClass
  - `python/sglang/srt/layers/dp_attention.py` modified +28/-0 (28 lines); hunks: get_attn_tensor_model_parallel_rank,; def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):; symbols: attn_cp_all_gather_into_tensor, get_moe_cp_group, get_moe_cp_rank, get_moe_cp_size
  - `python/sglang/srt/distributed/parallel_state.py` modified +13/-7 (20 lines); hunks: def initialize_model_parallel(; def initialize_model_parallel(; symbols: initialize_model_parallel, initialize_model_parallel, destroy_model_parallel, destroy_model_parallel
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: def __init__(; symbols: __init__, get_input_embeddings
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/communicator.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/layers/dp_attention.py`; keywords observed in patches: moe, attention, cuda, config, expert, flash. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/communicator.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/layers/dp_attention.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22065 - [HiSparse]: Optimize server args checking-HiSparse is temporarily only available for DSA models.

- Link: https://github.com/sgl-project/sglang/pull/22065
- Status/date: `merged`, created 2026-04-03, merged 2026-04-03; author `hzh0425`.
- Diff scope read: `1` files, `+8/-0`; areas: misc; keywords: attention, cache, config.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: def check_server_args(self):; symbols: check_server_args
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`; keywords observed in patches: attention, cache, config. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22128 - Allow piecewise CUDA graph with speculative decoding

- Link: https://github.com/sgl-project/sglang/pull/22128
- Status/date: `merged`, created 2026-04-05, merged 2026-04-17; author `narutolhy`.
- Diff scope read: `4` files, `+272/-18`; areas: kernel, scheduler/runtime, tests/benchmarks; keywords: cuda, spec, quant, attention, cache, config, eagle, expert, fp8, lora.
- Code diff details:
  - `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` added +243/-0 (243 lines); hunks: +"""Test piecewise CUDA graph coexisting with speculative decoding.; symbols: TestPCGWithMTP, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/server_args.py` modified +15/-18 (33 lines); hunks: def _handle_piecewise_cuda_graph(self):; symbols: _handle_piecewise_cuda_graph
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +10/-0 (10 lines); hunks: def can_run(self, forward_batch: ForwardBatch):; symbols: can_run
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunks: def init_piecewise_cuda_graphs(self):; symbols: init_piecewise_cuda_graphs
- Optimization/support interpretation: The concrete diff surface is `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`; keywords observed in patches: cuda, spec, quant, attention, cache, config. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22179 - [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation

- Link: https://github.com/sgl-project/sglang/pull/22179
- Status/date: `merged`, created 2026-04-06, merged 2026-04-06; author `mmangkad`.
- Diff scope read: `1` files, `+11/-12`; areas: docs/config; keywords: attention, benchmark, cache, config, deepep, doc, eagle, flash, fp8, kv.
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunks: DeepSeek-V3.2 model family equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attent
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/deepseek_v32.md`; keywords observed in patches: attention, benchmark, cache, config, deepep, doc. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/deepseek_v32.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22232 - Reduce unnecessary kernels and copies in the NSA indexer

- Link: https://github.com/sgl-project/sglang/pull/22232
- Status/date: `merged`, created 2026-04-07, merged 2026-04-07; author `1am9trash`.
- Diff scope read: `1` files, `+13/-5`; areas: attention/backend; keywords: attention, cuda, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +13/-5 (18 lines); hunks: def _weights_proj_bf16_in_fp32_out(self, x: torch.Tensor) -> torch.Tensor:; def _get_q_k_bf16(; symbols: _weights_proj_bf16_in_fp32_out, _project_and_scale_head_gates, _get_logits_head_gate, _get_q_k_bf16
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, cuda, topk. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22238 - [HiSparse]: Add readme docs for HiSparse Feature

- Link: https://github.com/sgl-project/sglang/pull/22238
- Status/date: `merged`, created 2026-04-07, merged 2026-04-07; author `hzh0425`.
- Diff scope read: `2` files, `+117/-0`; areas: docs/config; keywords: attention, config, cuda, doc, kv, cache, flash, mla, topk.
- Code diff details:
  - `docs/advanced_features/hisparse_guide.md` added +111/-0 (111 lines); hunks: +# HiSparse: Hierarchical Sparse Attention
  - `docs/basic_usage/deepseek_v32.md` modified +6/-0 (6 lines); hunks: python -m sglang.launch_server \
- Optimization/support interpretation: The concrete diff surface is `docs/advanced_features/hisparse_guide.md`, `docs/basic_usage/deepseek_v32.md`; keywords observed in patches: attention, config, cuda, doc, kv, cache. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/advanced_features/hisparse_guide.md`, `docs/basic_usage/deepseek_v32.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22258 - [AMD][HIP] NSA: bf16 passthrough from RMSNorm to eliminate FP8 dequantization

- Link: https://github.com/sgl-project/sglang/pull/22258
- Status/date: `merged`, created 2026-04-07, merged 2026-04-10; author `Jacob0226`.
- Diff scope read: `2` files, `+68/-25`; areas: attention/backend; keywords: attention, cuda, fp8, quant, lora.
- Code diff details:
  - `python/sglang/srt/layers/communicator.py` modified +38/-18 (56 lines); hunks: def __init__(self):; def prepare_attn(; symbols: __init__, init_context, prepare_attn, prepare_attn
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +30/-7 (37 lines); hunks: import contextlib; ceil_align,; symbols: _with_real_sm_count, _weights_proj_bf16_in_fp32_out, _weights_proj_bf16_in_fp32_out, _weights_proj_bf16_in_fp32_out
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, cuda, fp8, quant, lora. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22372 - [DSA] Hopper FP8 FlashMLA KV padding

- Link: https://github.com/sgl-project/sglang/pull/22372
- Status/date: `merged`, created 2026-04-08, merged 2026-04-12; author `mmangkad`.
- Diff scope read: `3` files, `+43/-8`; areas: attention/backend, docs/config; keywords: cache, flash, fp8, kv, mla, attention, doc, topk, benchmark, config.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +39/-4 (43 lines); hunks: def __init__(; def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: __init__, init_forward_metadata, init_forward_metadata, _forward_flashmla_kv
  - `docs/basic_usage/deepseek_v32.md` modified +2/-2 (4 lines); hunks: To serve GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
  - `python/sglang/srt/server_args.py` modified +2/-2 (4 lines); hunks: def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:; symbols: _set_default_nsa_backends
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py`; keywords observed in patches: cache, flash, fp8, kv, mla, attention. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22390 - [DSA] Enable all reduce fusion for DSA models

- Link: https://github.com/sgl-project/sglang/pull/22390
- Status/date: `merged`, created 2026-04-08, merged 2026-04-09; author `Fridge003`.
- Diff scope read: `1` files, `+2/-0`; areas: misc; keywords: kv, moe, spec.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`; keywords observed in patches: kv, moe, spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22424 - [AMD] Use aiter CK layernorm2d for LayerNorm to reduce NSA indexer kernel launches

- Link: https://github.com/sgl-project/sglang/pull/22424
- Status/date: `merged`, created 2026-04-09, merged 2026-04-09; author `1am9trash`.
- Diff scope read: `2` files, `+27/-3`; areas: attention/backend; keywords: attention, cuda, fp8, quant.
- Code diff details:
  - `python/sglang/srt/layers/layernorm.py` modified +15/-1 (16 lines); hunks: gemma_rmsnorm,; def forward_hip(; symbols: forward_hip, forward_npu
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +12/-2 (14 lines); hunks: from sglang.srt.layers.layernorm import LayerNorm; def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, cuda, fp8, quant. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22425 - [HiSparse]: Add HiSpares-DSA Model's nightly CI

- Link: https://github.com/sgl-project/sglang/pull/22425
- Status/date: `merged`, created 2026-04-09, merged 2026-04-09; author `hzh0425`.
- Diff scope read: `1` files, `+84/-0`; areas: model wrapper, tests/benchmarks; keywords: attention, cache, config, cuda, flash, fp8, kv, mla, test.
- Code diff details:
  - `test/registered/8-gpu-models/test_dsa_models_hisparse.py` added +84/-0 (84 lines); hunks: +import unittest; symbols: TestGLM5DPHiSparse, setUpClass, tearDownClass, test_a_gsm8k
- Optimization/support interpretation: The concrete diff surface is `test/registered/8-gpu-models/test_dsa_models_hisparse.py`; keywords observed in patches: attention, cache, config, cuda, flash, fp8. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/8-gpu-models/test_dsa_models_hisparse.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22430 - [Fix] Fix several bugs on DSA models

- Link: https://github.com/sgl-project/sglang/pull/22430
- Status/date: `merged`, created 2026-04-09, merged 2026-04-09; author `Fridge003`.
- Diff scope read: `2` files, `+5/-5`; areas: attention/backend; keywords: cache, flash, fp8, kv, mla, attention, topk.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +4/-2 (6 lines); hunks: def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:; symbols: _set_default_nsa_backends
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-3 (4 lines); hunks: def get_topk_transform_method(; symbols: get_topk_transform_method
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; keywords observed in patches: cache, flash, fp8, kv, mla, attention. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22792 - nsa indexer: use aiter indexer_k_quant_and_cache

- Link: https://github.com/sgl-project/sglang/pull/22792
- Status/date: `open`, created 2026-04-14; author `almaslof`.
- Diff scope read: `32` files, `+701/-165`; areas: model wrapper, attention/backend, quantization, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: attention, config, cache, cuda, kv, test, topk, lora, mla, quant.
- Code diff details:
  - `scripts/ci/utils/diffusion/generate_diffusion_dashboard.py` modified +208/-44 (252 lines); hunks: def generate_dashboard(; def generate_dashboard(; symbols: generate_dashboard, generate_dashboard, _chart_label, _chart_label
  - `python/tools/get_version_tag.py` added +171/-0 (171 lines); hunks: +#!/usr/bin/env python3; symbols: parse_version_tuple, run_git, get_exact_version_tag, get_latest_version_tag_describe
  - `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` added +117/-0 (117 lines); hunks: +import unittest; symbols: TestDeepseekV32IndexTopkPattern, setUpClass, tearDownClass, test_a_gsm8k
  - `python/sglang/srt/models/deepseek_v2.py` modified +53/-9 (62 lines); hunks: make_layers,; if _is_cuda:; symbols: forward, __init__, __init__, __init__
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +27/-13 (40 lines); hunks: def forward_absorb_prepare(; def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla
- Optimization/support interpretation: The concrete diff surface is `scripts/ci/utils/diffusion/generate_diffusion_dashboard.py`, `python/tools/get_version_tag.py`, `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`; keywords observed in patches: attention, config, cache, cuda, kv, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `scripts/ci/utils/diffusion/generate_diffusion_dashboard.py`, `python/tools/get_version_tag.py`, `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22850 - [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)

- Link: https://github.com/sgl-project/sglang/pull/22850
- Status/date: `merged`, created 2026-04-15, merged 2026-04-19; author `1am9trash`.
- Diff scope read: `1` files, `+24/-5`; areas: attention/backend; keywords: attention, cache, cuda, fp8, kv, quant.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunks: from sglang.srt.environ import envs; _is_npu = is_npu(); symbols: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, cache, cuda, fp8, kv, quant. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22914 - [Refactor] Deduplicate NSA utils.py into cp_utils.py for context parallel

- Link: https://github.com/sgl-project/sglang/pull/22914
- Status/date: `merged`, created 2026-04-16, merged 2026-04-20; author `Fridge003`.
- Diff scope read: `8` files, `+148/-402`; areas: model wrapper, attention/backend, scheduler/runtime; keywords: attention, cache, kv, config, cuda, expert, fp8, moe, quant, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +2/-353 (355 lines); hunks: -# temp NSA debugging environ; def pad_nsa_cache_seqlens(forward_batch: "ForwardBatch", nsa_cache_seqlens):; symbols: pad_nsa_cache_seqlens, NSAContextParallelMetadata:, can_cp_split, can_nsa_cp_split
  - `python/sglang/srt/layers/utils/cp_utils.py` modified +103/-12 (115 lines); hunks: import torch; def can_cp_split(seq_len: int, cp_size: int, forward_batch):; symbols: can_cp_split, cp_split_and_rebuild_data, cp_split_and_rebuild_data, cp_split_and_rebuild_position
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +20/-17 (37 lines); hunks: from sglang.srt.distributed.parallel_state import get_pp_group; def _get_q_k_bf16(; symbols: _get_q_k_bf16, _get_q_k_bf16, forward_cuda, forward_npu
  - `python/sglang/srt/models/deepseek_nextn.py` modified +11/-7 (18 lines); hunks: from sglang.srt.environ import envs; from sglang.srt.layers.logits_processor import LogitsProcessor; symbols: forward
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-7 (18 lines); hunks: from sglang.srt.layers.amx_utils import PackWeightMethod; from sglang.srt.layers.radix_attention import RadixAttention; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, cache, kv, config, cuda, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22950 - [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373)

- Link: https://github.com/sgl-project/sglang/pull/22950
- Status/date: `closed`, created 2026-04-16, closed 2026-04-21; author `Wen-xuan-Xu`.
- Diff scope read: `11` files, `+597/-64`; areas: scheduler/runtime, tests/benchmarks, docs/config; keywords: cache, kv, cuda, eagle, test, attention, spec.
- Code diff details:
  - `test/registered/unit/mem_cache/test_radix_cache_thinking.py` added +238/-0 (238 lines); hunks: +import unittest; symbols: _MockReqToTokenPool:, __init__, write, _MockAllocator:
  - `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py` added +220/-0 (220 lines); hunks: +import unittest; symbols: _MockReqToTokenPool:, __init__, write, _MockAllocator:
  - `python/sglang/srt/mem_cache/mamba_radix_cache.py` modified +62/-50 (112 lines); hunks: from numpy import float64; MatchPrefixParams,; symbols: cache_finished_req, _skip_cache_unfinished_req, _skip_cache_unfinished_req
  - `python/sglang/srt/mem_cache/radix_cache_cpp.py` modified +27/-14 (41 lines); hunks: MatchPrefixParams,; def cache_finished_req(self, req: Req, is_insert: bool = True):; symbols: cache_finished_req, cache_unfinished_req, cache_unfinished_req, pretty_print
  - `python/sglang/srt/mem_cache/common.py` modified +22/-0 (22 lines); hunks: def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:; symbols: alloc_for_decode, maybe_strip_thinking_tokens, release_kv_cache
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py`; keywords observed in patches: cache, kv, cuda, eagle, test, attention. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23219 - [AMD] Enable MTP for GLM-5-mxfp4 model

- Link: https://github.com/sgl-project/sglang/pull/23219
- Status/date: `merged`, created 2026-04-20, merged 2026-04-20; author `1am9trash`.
- Diff scope read: `1` files, `+41/-15`; areas: model wrapper; keywords: attention, config, fp8, processor, quant, spec.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunks: is_dp_attention_enabled,; def __init__(; symbols: __init__, forward, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_nextn.py`; keywords observed in patches: attention, config, fp8, processor, quant, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_nextn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23268 - 【NPU】【bugfix】accuracy fix when enable both nsa cp and prefixcache

- Link: https://github.com/sgl-project/sglang/pull/23268
- Status/date: `open`, created 2026-04-20; author `cen121212`.
- Diff scope read: `2` files, `+21/-5`; areas: attention/backend; keywords: attention, kv, lora, mla.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +18/-4 (22 lines); hunks: def forward_npu(; symbols: forward_npu
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +3/-1 (4 lines); hunks: def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; keywords observed in patches: attention, kv, lora, mla. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23315 - Opt-in strip of thinking tokens from radix cache

- Link: https://github.com/sgl-project/sglang/pull/23315
- Status/date: `merged`, created 2026-04-21, merged 2026-04-21; author `hnyls2002`.
- Diff scope read: `4` files, `+72/-4`; areas: scheduler/runtime, tests/benchmarks; keywords: cache, kv, spec, cuda, scheduler, test.
- Code diff details:
  - `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py` modified +52/-1 (53 lines); hunks: from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType; def test_cache_finished_req_insert(self):; symbols: test_cache_finished_req_insert, test_cache_finished_req_strips_thinking, test_cache_finished_req_no_insert
  - `python/sglang/srt/managers/schedule_batch.py` modified +9/-2 (11 lines); hunks: def output_ids_through_stop(self) -> List[int]:; def pop_overallocated_kv_cache(self) -> Tuple[int, int]:; symbols: output_ids_through_stop, _cache_commit_len, pop_committed_kv_cache, pop_overallocated_kv_cache
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs:, add_cli_args
  - `python/sglang/srt/mem_cache/common.py` modified +3/-1 (4 lines); hunks: def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = Tr; symbols: release_kv_cache
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: cache, kv, spec, cuda, scheduler, test. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23336 - [SPEC V2][2/N] feat: adaptive spec support spec v2

- Link: https://github.com/sgl-project/sglang/pull/23336
- Status/date: `open`, created 2026-04-21; author `alphabetc1`.
- Diff scope read: `6` files, `+193/-10`; areas: multimodal/processor, scheduler/runtime; keywords: spec, eagle, cuda, scheduler, attention, processor, config, kv, moe, topk.
- Code diff details:
  - `python/sglang/srt/speculative/eagle_worker_v2.py` modified +173/-0 (173 lines); hunks: from sglang.srt.managers.schedule_batch import ModelWorkerBatch; def __init__(; symbols: __init__, __init__, target_worker, forward_batch_generation
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +8/-4 (12 lines); hunks: def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):; def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):; symbols: prepare_for_decode, prepare_for_decode, prepare_for_v2_draft
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +10/-1 (11 lines); hunks: def _resolve_spec_overlap_token_ids(; symbols: _resolve_spec_overlap_token_ids
  - `python/sglang/srt/speculative/adaptive_spec_params.py` modified +0/-5 (5 lines); hunks: def adaptive_unsupported_reason(server_args: ServerArgs) -> str \| None:; symbols: adaptive_unsupported_reason
  - `python/sglang/srt/managers/utils.py` modified +1/-0 (1 lines); hunks: class GenerationBatchResult:; symbols: GenerationBatchResult:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`; keywords observed in patches: spec, eagle, cuda, scheduler, attention, processor. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23351 - Support piecewise CUDA graph with NSA

- Link: https://github.com/sgl-project/sglang/pull/23351
- Status/date: `open`, created 2026-04-21; author `nvjullin`.
- Diff scope read: `11` files, `+302/-56`; areas: attention/backend, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: cache, config, cuda, attention, quant, topk, fp8, moe, flash, fp4.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +163/-34 (197 lines); hunks: can_use_nsa_fused_store,; DUAL_STREAM_TOKEN_THRESHOLD = 1024 if _is_cuda else 0; symbols: k_cache_and_topk_result, _logits_head_gate_pcg_fake_impl, logits_head_gate_pcg, BaseIndexerMetadata
  - `test/registered/piecewise_cuda_graph/test_pcg_glm5_fp4.py` added +70/-0 (70 lines); hunks: +import unittest; symbols: TestPCGGlm5Fp4, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/layers/layernorm.py` modified +20/-1 (21 lines); hunks: if _is_cuda or _is_xpu:; symbols: _layernorm_fake_impl, layernorm
  - `python/sglang/srt/server_args.py` modified +4/-16 (20 lines); hunks: def _handle_piecewise_cuda_graph(self):; def _handle_moe_kernel_config(self):; symbols: _handle_piecewise_cuda_graph, _handle_multi_item_scoring, _handle_moe_kernel_config, _handle_a2a_moe
  - `python/sglang/srt/layers/radix_attention.py` modified +14/-0 (14 lines); hunks: def unified_attention_with_output(; def unified_attention_with_output(; symbols: unified_attention_with_output, unified_attention_with_output
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/piecewise_cuda_graph/test_pcg_glm5_fp4.py`, `python/sglang/srt/layers/layernorm.py`; keywords observed in patches: cache, config, cuda, attention, quant, topk. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/piecewise_cuda_graph/test_pcg_glm5_fp4.py`, `python/sglang/srt/layers/layernorm.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
