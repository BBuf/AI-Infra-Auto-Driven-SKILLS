# vLLM Torch Compile Fusion Patterns

Source inspection: **2026-11-08**, vLLM
`b022bb7a4873a47cdcdf474edf3bd499c221f40d`.
[Pass manager](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/compilation/passes/pass_manager.py)
and [configuration](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/config/compilation.py)
are the registration/gating authority. The previous MiniMax-specific pass is
not registered in this snapshot. Transformers norm canonicalization now adds
`AddRMSNormFusionPass` before AR+RMS matching and `RMSNormReshapeFusionPass`
afterward, exposing remaining norms to quant fusion. Current CUDA C++ sources
are under `csrc/libtorch_stable/`.

Use this file when the fuse-pattern table reports split kernels in a trace and
you need to decide whether the shape is already covered by vLLM's
`torch.compile` pattern matcher. Treat every row here as an upstream precedent
before calling a similar SGLang opportunity novel.

## Pass Registration

vLLM registers these passes from
`vllm/compilation/passes/pass_manager.py` through `PassConfig`.

| Toggle | Pass | Target shape |
| --- | --- | --- |
| `fuse_allreduce_rms` | `AllReduceFusionPass` or ROCm AITER variant | all-reduce followed by RMSNorm, optional residual add, optional FP8 / NVFP4 quant; current pass ordering runs AITER add-RMSNorm-pad before this fusion when available |
| `fuse_norm_quant` | `RMSNormQuantFusionPass` | RMSNorm or fused-add-RMSNorm followed by FP8 / FP4 quant |
| `fuse_norm_quant` + AITER | `RocmAiterRMSNormQuantFusionPass` | ROCm AITER RMSNorm / fused-add-RMSNorm followed by AITER or vLLM quant |
| `fuse_act_quant` | `ActivationQuantFusionPass` | SiLU-and-mul followed by FP8 / NVFP4 / block quant |
| `fuse_act_quant` + AITER | `RocmAiterSiluMulFp8GroupQuantFusionPass` | AITER SiLU-and-mul followed by FP8 group quant |
| `fuse_act_padding` + AITER | `RocmAiterTritonAddRMSNormPadFusionPass` | AITER fused-add-RMSNorm followed by padding into the next layout |
| `fuse_mla_dual_rms_norm` + AITER | `MLADualRMSNormFusionPass` | MLA paired Q and KV RMSNorms become `fused_mla_dual_rms_norm` |
| `fuse_rope_kvcache` | `RopeKVCacheFusionPass` | RoPE plus paged KV-cache update, after split cleanup passes |
| `fuse_rope_kvcache_cat_mla` | `MLARoPEKVCacheCatFusionPass` | MLA RoPE on `q_pe` / `k_pe` plus unified MLA KV-cache update through a fused concat/cache op |
| `fuse_attn_quant` | `AttnQuantFusionPass` | attention output followed by FP8 / NVFP4 quant |
| `fuse_attn_quant` | `MLAAttnQuantFusionPass` | MLA attention output followed by FP8 / NVFP4 / FP8 group quant |
| `enable_qk_norm_rope_fusion` | `QKNormRoPEFusionPass` | Q/K RMSNorm plus RoPE on packed QKV tensors |
| `fuse_qk_norm_rope_kvcache` | `QkNormRopeKvCacheFusionPass` | supported ROCm attention layers fuse Q/K RMSNorm, RoPE/MRoPE and unified KV-cache update; matched query quantization remains separate |
| Transformers backend + any norm/padding/AR fusion | `AddRMSNormFusionPass`, `RMSNormReshapeFusionPass` | canonicalizes residual add and reorders output reshapes around AR+RMS matching; #54461 covers rsqrt forms |
| `eliminate_noops` (default True) | `NoOpEliminationPass` | removes no-op graph nodes before fusion matching |
| RoPE/QK fusion prerequisites | `SplitCoalescingPass`, `ScatterSplitReplacementPass` | canonicalizes split/slice and functionalized scatter layouts; runs before the relevant fusion |
| Always after configured fusions | `PostCleanupPass`, `VllmIRLoweringPass`, `UnsafeCloneEliminationPass`, `FixFunctionalizationPass` | cleanup, provider lowering, clone removal, second cleanup, then final functionalization repair |

## Pattern Inventory

| Source file | Pattern classes | Trace clue | Replacement |
| --- | --- | --- | --- |
| `fusion/allreduce_rms_fusion.py` | `AllReduceRMSNormPattern`, `AllReduceFusedAddRMSNormPattern`, `AllReduceFusedRMSNormStaticQuantFP8Pattern`, `AllReduceFusedAddRMSNormStaticQuantFP8Pattern`, `AllReduceFusedRMSNormStaticQuantNVFP4Pattern`, `AllReduceFusedAddRMSNormStaticQuantNVFP4Pattern` | TP all-reduce directly before RMSNorm, residual-add RMSNorm, or quant | `flashinfer_trtllm_fused_allreduce_norm` with FlashInfer allreduce fusion pattern codes |
| `fusion/rms_quant_fusion.py` | `RMSNormStaticQuantPattern`, `FusedAddRMSNormStaticQuantPattern`, `RMSNormDynamicQuantPattern`, `FusedAddRMSNormDynamicQuantPattern`, `RMSNormGroupQuantPattern`, `FusedAddRMSNormGroupQuantPattern` | RMSNorm or fused-add-RMSNorm followed by static FP8, dynamic per-token FP8, FP8 group quant, or NVFP4 quant | `_C.rms_norm_*_quant`, `_C.fused_add_rms_norm_*_quant`, or per-block quant custom op |
| `fusion/rocm_aiter_fusion.py` | `AiterRMSNormDynamicQuantPattern`, `AiterFusedAddRMSNormDynamicQuantPattern`, `AiterRMSFp8GroupQuantPattern`, `AiterFusedAddRMSFp8GroupQuantPattern` | AITER RMSNorm/fused-add-RMSNorm followed by AITER or vLLM FP8 quant | AITER fused RMSNorm-quant custom ops |
| `fusion/act_quant_fusion.py` | `SiluMulFp8StaticQuantPattern`, `SiluMulNvfp4QuantPattern`, `SiluMulBlockQuantPattern` | SiLU-and-mul activation output immediately quantized | fused activation-plus-quant custom op |
| `fusion/rocm_aiter_fusion.py` | `AiterSiluMulFp8GroupQuantPattern` | AITER SiLU-and-mul followed by FP8 group quant | AITER `act_mul_fused_fp8_group_quant` |
| `fusion/rocm_aiter_fusion.py` | `AddAiterRMSNormPadPattern` | AITER fused-add-RMSNorm output padded before the next op | AITER add-RMSNorm-pad op |
| `fusion/rocm_aiter_fusion.py` | `MLADualRMSNormPattern` | MLA Q branch and KV branch each run RMSNorm | `torch.ops.vllm.fused_mla_dual_rms_norm` backed by AITER fused QK RMSNorm |
| `fusion/qk_norm_rope_fusion.py` | `QkNormRopePattern` | Q/K RMSNorm, split/getitem reshapes, then RoPE | `_C.fused_qk_norm_rope` |
| `fusion/qk_norm_rope_kvcache_fusion.py` | `QkNormRopeKvCachePattern` | Q/K RMSNorm and RoPE flow directly into unified KV-cache update, optionally with FP8 query quantization | `vllm.fused_qk_norm_rope_and_unified_kv_cache_update`, followed by the original Q-quant form when matched |
| `fusion/rope_kvcache_fusion.py` | `RopeReshapeKVCachePattern` | RoPE output followed by reshape/cache update | `vllm.fused_rope_and_unified_kv_cache_update` |
| `fusion/mla_rope_kvcache_cat_fusion.py` | `MLARoPEKVCacheCatPattern` | MLA RoPE on `q_pe` and `k_pe` flows into `unified_mla_kv_cache_update` | `vllm.fused_rope_unified_mla_kv_cache_update`, backed by `concat_and_cache_mla_rope_fused` |
| `fusion/attn_quant_fusion.py` | `AttnFp8StaticQuantPattern`, `AttnNvfp4QuantPattern` | attention output followed by FP8 static quant or NVFP4 quant | backend attention op with fused output quant when supported |
| `fusion/mla_attn_quant_fusion.py` | `MLAAttnFp8StaticQuantPattern`, `MLAAttnNvfp4QuantPattern`, `MLAAttnFp8GroupQuantPattern` | MLA attention output followed by static FP8, NVFP4, or FP8 group quant | MLA attention op with fused output quant when supported |
| `fusion/add_rms_fusion.py` | `RMSNormReshapePattern`, `FusedAddRMSNormReshapePattern` | residual add, norm and output reshapes in Transformers-backend models | canonical residual-add RMSNorm and reshape movement ahead of norms; before/after AR+norm respectively |
| `fusion/allreduce_rms_fusion.py` | `AllReduceGemmaRMSNormPattern`, `AllReduceFusedAddGemmaRMSNormPattern`, `AiterAllreduceFusedRMSNormPattern`, `AiterAllreduceFusedAddRMSNormPattern`, `AiterAllreduceFusedAddRMSNormOutputOnlyPattern`, `AiterAllreduceFusedRMSNormGroupQuantFP8Pattern`, `AiterAllreduceFusedAddRMSNormGroupQuantFP8Pattern`, `AiterAllreduceFusedAddRMSNormGroupQuantWithIndexerPattern` | all-reduce followed by Gemma (1+w) norm or ROCm residual norm/FP8 group quant/indexer fan-out | FlashInfer AR+Gemma norm or AITER AR+RMSNorm (+group quant and BF16 indexer fan-out) |
| `fusion/attn_quant_fusion.py` | `RocmAttnFp8StaticQuantPattern` | ROCm AITER attention output then static FP8 quant | AITER attention with quantized output |
| `fusion/qk_norm_rope_kvcache_fusion.py` | `QkNormMRopeKvCachePattern` | MRoPE plus QK norm, optional Q quant and unified KV update on ROCm | `fused_qk_norm_mrope_and_unified_kv_cache_update`, followed by separate Q quant when matched; raw K view must have no other observable consumer |
| `fusion/rocm_aiter_fusion.py` | `DoubleAiterRMSFp8GroupQuantPattern`, `DoubleAiterRMSFp8GroupQuantViewPattern`, `AiterRMSNormGatedFp8GroupQuantPattern`, `MLADualRMSPerTokenQuantPattern` | double/gated RMSNorm plus group FP8 quant, or MLA dual Q/KV norms plus per-token quant | AITER double/gated norm-quant and dual MLA norm-quant custom ops; inspect gfx950/provider guards |
| `fusion/add_rms_fusion.py` | `AddRMSNormPattern` | residual add directly before RMSNorm in Transformers backend | `vllm.ir.ops.fused_add_rms_norm` canonical form |
| `fusion/rms_quant_fusion.py` | `RMSNormQuantPattern`, `FusedAddRMSNormNvfp4QuantPattern` | residual-add RMSNorm followed by NVFP4 quant | `flashinfer_fused_add_rms_norm_nvfp4_quant` (#51925 merged 2026-09-08); base pattern dispatches by quant key |
| `fusion/rope_kvcache_fusion.py` | `RopeStaticQQuantKVCachePattern` | RoPE plus static FP8 Q quant feeding KV update | `fused_rope_and_unified_kv_cache_update` followed by separate `_C.static_scaled_fp8_quant` |

Base matcher classes such as `BasePattern` (allreduce),
`ActivationQuantPattern`, `RMSNormQuantPattern` and `AiterRMSNormQuantPattern`
share the concrete patterns’ dispatch logic; they are not additional GPU kernels.

## Enablement and provider interpretation

[Optimization-level defaults](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/config/vllm.py) resolve conditionally.
O1 enables eligible norm/activation quant, AITER norm-pad and MLA dual norms;
O2/O3 add eligible AR+norm, RoPE/cache, combined QK-norm/cache and MLA cache
fusion. **Attention-output quant is now enabled for quantized models at
O2/O3**, subject to backend and graph-splitting checks. It is no longer always
opt-in. The generic QK-norm/RoPE flag remains false in these level defaults.

For RoPE/cache, inspect the resolved configuration and graph, not only the
optimization-level table. In the normal default Dynamo split, current main
disables explicit ordinary/QK cache fusion and extracts both cache-update ops.
An explicit MLA flag can survive while its required op is outside the graph.
Custom rotary registration also precedes optimization-level resolution, so the
ordinary/QK default resolvers can remain false even with partition opt-ins.
The [complete compiler review](vllm-compiler-source-review.md) explains this
ordering and the proposed fix: #57349 is closed without merge and superseded
by still-open #57900. Neither is current-main behavior at this pinned SHA.

Ordinary and combined QK cache fusion use a default compile-range upper bound
of 256 tokens. The #57900 MI355X report improves 1–64-token latency but regresses
at 128/256; it leaves that bound unchanged. Its Qwen3 QK-norm and Kimi MLA
pattern-match reports have no corresponding performance measurements. A
remaining Q-quant kernel is expected in the quantized-query patterns above.

The [CUDA kernel proposal review](vllm-rope-cache-kernel-review.md) separates
#57271's ordinary full-width GQA writer from #58798's model-selected MiMo
partial-RoPE/value-scaling/DiffKV path. Both remain open. The former needs
short-slot and FP8 intermediate-rounding checks; the latter's old plain-QKV
split does not cover current MiMo's padded checkpoint-chunk layout. A
registered fusion or historical benchmark is not current-model admission.
The same review covers #52363's separate manual Llama/FlashAttention design:
descriptor-based admission, graph-owned Q and an in-op runtime threshold
avoid freezing dispatch when shape guards are skipped. Above 256 tokens,
compiled fallback adds Q/K copies; reported 257/512/1024-token cases regress
2–2.5%, while small operator gains coexist with near-equal batch-one E2E latency.

The old `enable_sp` and `fuse_gemm_comms` fields and their sequence-parallel /
collective pass files have been removed. Do not suggest those switches to a
current server. [Runtime MoE sequence parallelism](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/config/parallel.py)
is a separate backend/EP/TP/DP decision, not the removed Inductor transform.

AITER norm-pad runs before AR+norm; Transformers add/RMS canonicalization
runs before it and reshape canonicalization after it. Norm-pad, norm-quant
and activation-quant registration also accepts the RDNA AITER enablement;
that does not make every ROCm fusion available on RDNA.

Current model-specific additions:

| Pattern | Registration and per-layer guards | Replacement |
|---|---|---|
| [Muse sandwich norm](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/compilation/passes/fusion/muse_glimmer_sandwich_norm_fusion.py) | MuseGlimmer causal/conditional architectures, ROCm+AITER, gfx950, BF16 | Gemma-style post RMSNorm (1+w), residual add and next RMSNorm; returns both residual and normalized rows. |
| [Muse QK/RoPE/cache](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/compilation/passes/fusion/muse_glimmer_qk_rope_cache_fusion.py) | MuseGlimmer, ROCm+AITER; RoPE-enabled layer, AITER FlashAttention, KV dtype `auto`, head size 128 and backend support predicate | QK normalization/scale, RoPE and cache update after split/scatter cleanup. Model-specific registration can occur without the generic QK/cache toggle. |

These model-specific passes have explicit architecture/provider gates; do not
infer their status solely from generic `PassConfig` flags. The manager runs
post-cleanup → IR lowering → clone elimination → post-cleanup → functionalization.
Inspect IR provider selection as well as matched patterns: a native/Inductor
norm may appear as a generated Triton kernel. A missing custom-op name alone
is not evidence that norm fusion failed.

## Triage Rules

- If the trace shows split norm/add/quant, compare first against
  `RMSNormQuantFusionPass`, AITER variants, and `AllReduceFusionPass`.
- If the trace shows attention output followed by quant kernels, compare against
  `AttnQuantFusionPass` or `MLAAttnQuantFusionPass`, not only handwritten
  attention kernels.
- If the trace shows Q/K norm followed by RoPE or cache update, compare
  `QkNormRopeKvCacheFusionPass`, `QKNormRoPEFusionPass`,
  `RopeKVCacheFusionPass`, and the MLA-specific
  `MLARoPEKVCacheCatFusionPass`. The combined ROCm pass supersedes the first
  two separate stages when its platform, backend, graph-range, and head-size
  guards pass.
- For a TP/EP trace with visible collectives, inspect the current runtime
  backend and communication ownership. Use historical SP/AsyncTP rows only
  to understand older traces, not as current configuration advice.
- A missing vLLM compile fusion may be intentional when the graph range, backend
  support check, dtype, token count, or AITER / FlashInfer availability does not
  satisfy the pass-specific guard.
- Distinguish compiler fusion reachability from cache binding inside a captured
  handwritten kernel. The [MLA/DSA reviews](vllm-compiler-source-review.md)
  cover #49142's default proposal and #54851's conditional stale-write mechanism.
  PIECEWISE does not itself imply absent metadata on current MRV2; verify the
  forward context, persistent slot addresses and actual cache-write owner.
