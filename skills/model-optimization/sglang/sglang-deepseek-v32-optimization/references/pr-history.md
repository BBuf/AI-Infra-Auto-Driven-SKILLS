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
