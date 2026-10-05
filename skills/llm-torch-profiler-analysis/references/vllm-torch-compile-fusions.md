# vLLM Torch Compile Fusion Patterns

Source inspection: **2026-10-05**, vLLM
`0c16eee3f1ff777298cc894c3eeb85f3880c6d6a`.
[Pass manager](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/compilation/passes/pass_manager.py)
and [configuration](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/config/compilation.py)
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
| `enable_sp` | `SequenceParallelismPass` | all-reduce around residual/norm blocks becomes reduce-scatter, local work, and all-gather |
| `fuse_gemm_comms` | `AsyncTPPass` | GEMM plus reduce-scatter / all-gather overlap through symmetric-memory collectives |
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
| `fuse_qk_norm_rope_kvcache` | `QkNormRopeKvCacheFusionPass` | supported ROCm attention layers fuse Q/K RMSNorm, RoPE, unified KV-cache update, and optional query quantization |

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
| `fusion/qk_norm_rope_kvcache_fusion.py` | `QkNormRopeKvCachePattern` | Q/K RMSNorm and RoPE flow directly into unified KV-cache update, optionally with FP8 query quantization | `vllm.fused_qk_norm_rope_and_unified_kv_cache_update` |
| `fusion/rope_kvcache_fusion.py` | `RopeReshapeKVCachePattern` | RoPE output followed by reshape/cache update | `vllm.fused_rope_and_unified_kv_cache_update` |
| `fusion/mla_rope_kvcache_cat_fusion.py` | `MLARoPEKVCacheCatPattern` | MLA RoPE on `q_pe` and `k_pe` flows into `unified_mla_kv_cache_update` | `vllm.fused_rope_unified_mla_kv_cache_update`, backed by `concat_and_cache_mla_rope_fused` |
| `fusion/attn_quant_fusion.py` | `AttnFp8StaticQuantPattern`, `AttnNvfp4QuantPattern` | attention output followed by FP8 static quant or NVFP4 quant | backend attention op with fused output quant when supported |
| `fusion/mla_attn_quant_fusion.py` | `MLAAttnFp8StaticQuantPattern`, `MLAAttnNvfp4QuantPattern`, `MLAAttnFp8GroupQuantPattern` | MLA attention output followed by static FP8, NVFP4, or FP8 group quant | MLA attention op with fused output quant when supported |
| `fusion/sequence_parallelism.py` | `FirstAllReduceRMSNormPattern`, `MiddleAllReduceRMSNormPattern`, `FirstAllReduceRMSNormStaticFP8Pattern`, `MiddleAllReduceRMSNormStaticFP8Pattern` | all-reduce plus norm block in a full-graph TP model | sequence-parallel reduce-scatter, local norm, all-gather staging |
| `fusion/collective_fusion.py` | `GEMMReduceScatterPattern`, `AllGatherGEMMPattern`, `ScaledMMReduceScatterPattern`, `AllGatherScaledMMPattern`, `CutlassScaledMMReduceScatterPattern`, `AllGatherCutlassScaledMMPattern`, `FlashInferBMMFP8ReduceScatterPattern`, `FlashInferAllGatherBMMFP8Pattern` | matmul / scaled-mm / FlashInfer BMM adjacent to TP collectives | symmetric-memory fused matmul+reduce-scatter or all-gather+matmul |

| `fusion/act_quant_fusion.py` | `ActivationQuantPattern` | see existing family row above | same registered pass and provider; inspect pattern-specific guards |
| `fusion/add_rms_fusion.py` | `RMSNormReshapePattern`, `FusedAddRMSNormReshapePattern` | residual add, norm and output reshapes in Transformers-backend models | canonical residual-add RMSNorm and reshape movement ahead of norms; before/after AR+norm respectively |
| `fusion/allreduce_rms_fusion.py` | `AllReduceGemmaRMSNormPattern`, `AllReduceFusedAddGemmaRMSNormPattern`, `AiterAllreduceFusedRMSNormPattern`, `AiterAllreduceFusedAddRMSNormPattern`, `AiterAllreduceFusedAddRMSNormOutputOnlyPattern`, `AiterAllreduceFusedRMSNormGroupQuantFP8Pattern`, `AiterAllreduceFusedAddRMSNormGroupQuantFP8Pattern`, `AiterAllreduceFusedAddRMSNormGroupQuantWithIndexerPattern` | all-reduce followed by Gemma (1+w) norm or ROCm residual norm/FP8 group quant/indexer fan-out | FlashInfer AR+Gemma norm or AITER AR+RMSNorm (+group quant and BF16 indexer fan-out) |
| `fusion/attn_quant_fusion.py` | `RocmAttnFp8StaticQuantPattern` | ROCm AITER attention output then static FP8 quant | AITER attention with quantized output |
| `fusion/collective_fusion.py` | `RocmGEMMReduceScatterPattern`, `RocmAllGatherGEMMPattern`, `FlashInferAllGatherFP4Pattern` | ROCm BF16 GEMM adjacent to collectives or NVFP4 all-gather feeding GEMM | symmetric-memory ROCm GEMM RS/AG and fused_all_gather_flashinfer_fp4_matmul |
| `fusion/qk_norm_rope_kvcache_fusion.py` | `QkNormMRopeKvCachePattern` | MRoPE plus QK norm, optional Q quant and unified KV update on ROCm | fused_qk_norm_mrope_and_unified_kv_cache_update |
| `fusion/rocm_aiter_fusion.py` | `DoubleAiterRMSFp8GroupQuantPattern`, `DoubleAiterRMSFp8GroupQuantViewPattern`, `AiterRMSNormGatedFp8GroupQuantPattern`, `MLADualRMSPerTokenQuantPattern` | double/gated RMSNorm plus group FP8 quant, or MLA dual Q/KV norms plus per-token quant | AITER double/gated norm-quant and dual MLA norm-quant custom ops; inspect gfx950/provider guards |
| `fusion/sequence_parallelism.py` | `FirstAllReduceRMSNormStaticNVFP4Pattern`, `MiddleAllReduceRMSNormStaticNVFP4Pattern` | all-reduce and NVFP4 norm block | reduce-scatter/local norm-quant/all-gather staging |

| `fusion/add_rms_fusion.py` | `AddRMSNormPattern` | residual add directly before RMSNorm in Transformers backend | `vllm.ir.ops.fused_add_rms_norm` canonical form |
| `fusion/rms_quant_fusion.py` | `RMSNormQuantPattern`, `FusedAddRMSNormNvfp4QuantPattern` | residual-add RMSNorm followed by NVFP4 quant | `flashinfer_fused_add_rms_norm_nvfp4_quant` (#51925 merged 2026-09-08); base pattern dispatches by quant key |
| `fusion/rope_kvcache_fusion.py` | `RopeStaticQQuantKVCachePattern` | RoPE plus static FP8 Q quant feeding KV update | `fused_rope_and_unified_kv_cache_update` with Q quant |

Base matcher classes `BasePattern` (allreduce and collective files),
`ActivationQuantPattern`, `RMSNormQuantPattern` and `AiterRMSNormQuantPattern`
share the concrete patterns’ dispatch logic; they are not additional GPU kernels.

## Enablement and provider interpretation

`vllm/config/vllm.py::OPTIMIZATION_LEVEL_00..03` controls defaults (default O2).
O0 disables all fusion passes and graphs. O1 conditionally enables norm-quant,
activation-quant, AITER norm-pad and MLA dual norms, with PIECEWISE graphs.
O2/O3 additionally conditionally enable AR+norm, RoPE/cache, combined QK-norm
RoPE/cache and MLA RoPE/cache, with FULL_AND_PIECEWISE graphs. Platform,
provider, TP size and graph-splitting guards still apply. Attention-output quant,
QK-norm/RoPE, SP and AsyncTP remain opt-in at every level. Batch invariance
force-disables AR+norm, SP and AsyncTP. XPU also registers norm-quant,
activation-quant, QK-norm/RoPE and SP when their platform guards pass.

IR ops (`vllm/ir/ops/layernorm.py`, `activation.py`) are lowered **after** matching.
`kernel_config.ir_op_priority` chooses native/Inductor, vllm_c, AITER or Oink
providers. Compiled CUDA/ROCm defaults use native norms: look for Inductor
`triton_*_fused_*` kernels before concluding a custom norm-quant op is missing.
`VLLM_USE_OINK_OPS` changes the SM100 provider. Breakable-graph models skip
these compile passes: compare model-local eager AR+norm and
`vllm/model_executor/layers/fusion/fused_act_quant.py` instead.

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
- If the trace is a TP decode trace with visible collectives, check whether
  `enable_sp` and `fuse_gemm_comms` would transform the same region into
  sequence-parallel or AsyncTP overlap.
- A missing vLLM compile fusion may be intentional when the graph range, backend
  support check, dtype, token count, or AITER / FlashInfer availability does not
  satisfy the pass-specific guard.
