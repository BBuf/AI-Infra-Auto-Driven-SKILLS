---
name: sglang-deepseek-v3-r1-optimization
description: PR-backed and current-main optimization manual for DeepSeek V3 and DeepSeek R1 in SGLang. Use when Codex needs to recover, extend, or audit DeepSeek V3/R1 MLA, MoE, shared experts, FP8/FP4/W4AFP8/MXFP4/NVFP4 loading, MTP, DeepEP, DP attention, LoRA, backend selection, or validation lanes.
---

# SGLang DeepSeek V3/R1 Optimization

## Overview

This skill covers the DeepSeek V3/R1 optimization ladder that is active in SGLang main. It intentionally excludes the V3.1 parser delta and the V3.2 DSA/NSA sparse-attention stack, which have separate skills.

Current-main snapshot:

- SGLang `origin/main`: `929e00eea` on `2026-04-21`
- sgl-cookbook `origin/main`: `8ec4d03` on `2026-04-21`
- active runtime entry: `python/sglang/srt/models/deepseek_v2.py`
- DeepSeek V3/R1 entry class: `DeepseekV3ForCausalLM`
- NextN/MTP entry class: `DeepseekV3ForCausalLMNextN`

The historical evidence lives in:

- [references/pr-history.md](references/pr-history.md): chronological PR evidence and code-level notes
- [references/playbook.md](references/playbook.md): investigation order, symptom mapping, validation commands

## Before You Change Anything

Record the exact serving shape first:

- model: V3, V3-0324, R1, R1-0528, R1-distill, or vendor quantized checkpoint
- native FP8, BF16, INT8, AWQ, W4A8/W4AFP8, NVFP4, MXFP4, MXFP8, or LoRA
- TP / DP / EP / PP / PD topology
- DP attention enabled or not
- DeepEP mode: none, normal, low_latency, or auto
- MoE runner backend: triton, deep_gemm, flashinfer_trtllm, flashinfer_cutlass, flashinfer_cutedsl, cutlass_w4afp8, aiter, or auto
- MLA attention backend: fa3, flashinfer, flashmla, cutlass_mla, trtllm_mla, aiter, triton, CPU/XPU/NPU fallback
- MTP enabled or not, and whether the NextN layer is quantized differently from the target model
- parser pair: `--reasoning-parser deepseek-v3` for V3 thinking-style output, `--reasoning-parser deepseek-r1` for R1, and `--tool-call-parser deepseekv3` for V3/R1 tool calls
- exact registered/manual test lane and hardware

## Core Principle

Do not treat DeepSeek V3/R1 as only one optimization.

- The V3/R1 base path is an MLA plus MoE throughput problem.
- R1 adds a reasoning-parser contract and heavier quantized deployment tracks.
- W4AFP8 is its own loader, kernel, EP, TP, and DeepEP story.
- MTP is a NextN-model story; target model and draft layer can have different quantization or backend requirements.
- Shared expert fusion is powerful but topology-sensitive. On current main it is disabled under DeepEP unless `--enforce-shared-experts-fusion` is set.
- On Blackwell, server defaults may automatically select `trtllm_mla` and `flashinfer_trtllm`; on ROCm, AITER and TileLang paths are separate validation surfaces.

The optimization order matters:

1. confirm the loader and quant config
2. confirm MLA backend and KV cache dtype
3. confirm MoE runner and shared expert behavior
4. add or validate MTP
5. add DP attention, EP, PP, PD, or DeepEP only after the single-shape path is correct
6. harden PCG, deterministic, LoRA, and backend-specific tests

## Main Runtime Surfaces

Start from these files before changing behavior:

- `python/sglang/srt/models/deepseek_v2.py`
- `python/sglang/srt/models/deepseek_nextn.py`
- `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/`
- `python/sglang/srt/layers/attention/flashattention_backend.py`
- `python/sglang/srt/layers/radix_attention.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/layers/quantization/fp8_kernel.py`
- `python/sglang/srt/layers/quantization/deep_gemm.py`
- `python/sglang/compile_deep_gemm.py`
- `python/sglang/srt/layers/moe/topk.py`
- `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`
- `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`
- `python/sglang/srt/layers/moe/ep_moe/layer.py`
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- `python/sglang/srt/layers/quantization/w4afp8.py`
- `python/sglang/srt/layers/rotary_embedding.py`
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/srt/parser/reasoning_parser.py`
- `python/sglang/srt/function_call/deepseekv3_detector.py`
- `sgl-kernel/csrc/moe/moe_fused_gate.cu`
- `sgl-kernel/csrc/moe/moe_align_kernel.cu`
- `sgl-kernel/csrc/attention/merge_attn_states.cu`
- `sgl-kernel/include/sgl_kernel_ops.h`

## Open PRs to Track

Check these before declaring a V3/R1 gap:

- [#14194](https://github.com/sgl-project/sglang/pull/14194): DCP for `deepseek_v2`, open.
- [#15315](https://github.com/sgl-project/sglang/pull/15315) and [#15380](https://github.com/sgl-project/sglang/pull/15380): group GEMM work for DeepSeek-R1-W4AFP8, open.
- [#18892](https://github.com/sgl-project/sglang/pull/18892): JIT support for DeepSeek V3 GEMM, open.
- [#6011](https://github.com/sgl-project/sglang/pull/6011): FlashInfer MLA speculative-decoding custom mask, open.
- [#6738](https://github.com/sgl-project/sglang/pull/6738): partial MHA-kernel support in MLA forward when page size is greater than one, open.
- [#7005](https://github.com/sgl-project/sglang/pull/7005): FlowMLA zero-overhead DP MLA memory optimization, open.
- [#23336](https://github.com/sgl-project/sglang/pull/23336): adaptive speculative-num-steps support for spec v2 EAGLE workers, open.
- [#21526](https://github.com/sgl-project/sglang/pull/21526): ROCm AITER router GEMM regression for non-DSR1 MoE, open.
- [#21529](https://github.com/sgl-project/sglang/pull/21529): MXFP4 / Quark W4A4 support for DeepSeek architecture on ROCm, open.
- [#21530](https://github.com/sgl-project/sglang/pull/21530): ROCm fused MLA decode RoPE fix for DeepSeek variants, open.
- [#21531](https://github.com/sgl-project/sglang/pull/21531): migrate `dsv3_router_gemm` from AOT `sgl-kernel` to JIT, open.
- [#22268](https://github.com/sgl-project/sglang/pull/22268): fix LoRA adapters bypassed in DeepSeek MLA `prepare_qkv_latent`, open.
- [#22774](https://github.com/sgl-project/sglang/pull/22774): MUSA backend support for DeepSeek V2/V3/R1, open.
- [#22938](https://github.com/sgl-project/sglang/pull/22938): restore DeepSeek MLA MI300X paths after the MLA refactor, open.
- [#23195](https://github.com/sgl-project/sglang/pull/23195): guard `.weight` access in `DeepseekV2AttentionMLA` for AWQ/compressed-tensors, open.
- [#23257](https://github.com/sgl-project/sglang/pull/23257): double-reduce in `DeepseekV2MoE` with CuteDSL EP plus DP attention, open.

Known reverted track:

- [#14162](https://github.com/sgl-project/sglang/pull/14162) enabled FP8 communication for R1 W4A8 DeepEP low-latency, was reverted by [#21719](https://github.com/sgl-project/sglang/pull/21719), then relanded by [#22316](https://github.com/sgl-project/sglang/pull/22316).

Known exploratory or closed tracks:

- [#5432](https://github.com/sgl-project/sglang/pull/5432) introduced a DeepGEMM `group_gemm_masked` BMM path for MLA FP8 quantization. Treat it as an explored path, not as the default production H200 speed path.
- [#6151](https://github.com/sgl-project/sglang/pull/6151) explored hybrid attention backend wiring and closed without becoming the main V3/R1 path.
- [#22950](https://github.com/sgl-project/sglang/pull/22950) explored parser-gated two-phase reasoning radix-cache stripping and closed before becoming current support; read [#23315](https://github.com/sgl-project/sglang/pull/23315) for the merged path.
- [#22933](https://github.com/sgl-project/sglang/pull/22933) is current-main CPU shared-expert interface cleanup. It matters for CPU shared-expert parity, not for H200 GPU throughput.

## Runtime Addendum

Additional current-main runtime tracks should be checked in addition to the original H200 ladder:

- [#21599](https://github.com/sgl-project/sglang/pull/21599) adds adaptive `speculative_num_steps` for EAGLE top-k=1. For V3/R1 MTP, inspect `server_args.py`, speculative runtime params, and EAGLE worker state before assuming the number of draft steps is static.
- [#22128](https://github.com/sgl-project/sglang/pull/22128) allows piecewise CUDA graph to run with speculative decoding. When auditing PCG plus MTP failures, check `model_runner.py`, `piecewise_cuda_graph_runner.py`, and the server flag gate instead of treating the combination as unsupported.
- [#23315](https://github.com/sgl-project/sglang/pull/23315) adds opt-in stripping of thinking tokens from radix cache. This touches `schedule_batch.py`, `mem_cache/common.py`, and `server_args.py`; it matters for DeepSeek V3/R1 reasoning-parser cache reuse, especially when thinking tokens should not become a reusable prefix.
- [#22950](https://github.com/sgl-project/sglang/pull/22950) is the closed predecessor for that reasoning-cache behavior, and [#23336](https://github.com/sgl-project/sglang/pull/23336) is the open spec-v2 extension for adaptive speculative decoding.

## H200 Single-Node Optimization Findings

The single-node H200 optimization notes explicitly name a March-May 2025 H200 optimization ladder. A complete V3/R1 audit must include those PRs because many later abstractions hide the original performance reason.

Required H200 PR coverage:

- FP8 Block GEMM and DeepGEMM: [#3893](https://github.com/sgl-project/sglang/pull/3893), [#4165](https://github.com/sgl-project/sglang/pull/4165), [#4199](https://github.com/sgl-project/sglang/pull/4199), [#4613](https://github.com/sgl-project/sglang/pull/4613), [#5263](https://github.com/sgl-project/sglang/pull/5263), [#5310](https://github.com/sgl-project/sglang/pull/5310), [#5432](https://github.com/sgl-project/sglang/pull/5432), [#5473](https://github.com/sgl-project/sglang/pull/5473), [#5549](https://github.com/sgl-project/sglang/pull/5549), [#5628](https://github.com/sgl-project/sglang/pull/5628), [#6890](https://github.com/sgl-project/sglang/pull/6890).
- Fused MoE, top-k, and shared experts: [#4530](https://github.com/sgl-project/sglang/pull/4530), [#5086](https://github.com/sgl-project/sglang/pull/5086), [#5371](https://github.com/sgl-project/sglang/pull/5371), [#5571](https://github.com/sgl-project/sglang/pull/5571), [#5716](https://github.com/sgl-project/sglang/pull/5716), [#5740](https://github.com/sgl-project/sglang/pull/5740), [#6220](https://github.com/sgl-project/sglang/pull/6220), [#6970](https://github.com/sgl-project/sglang/pull/6970).
- MLA and attention backends: [#4472](https://github.com/sgl-project/sglang/pull/4472), [#4514](https://github.com/sgl-project/sglang/pull/4514), [#4831](https://github.com/sgl-project/sglang/pull/4831), [#5113](https://github.com/sgl-project/sglang/pull/5113), [#5210](https://github.com/sgl-project/sglang/pull/5210), [#5381](https://github.com/sgl-project/sglang/pull/5381), [#5385](https://github.com/sgl-project/sglang/pull/5385), [#5390](https://github.com/sgl-project/sglang/pull/5390), [#5578](https://github.com/sgl-project/sglang/pull/5578), [#5619](https://github.com/sgl-project/sglang/pull/5619), [#5748](https://github.com/sgl-project/sglang/pull/5748), [#5977](https://github.com/sgl-project/sglang/pull/5977), [#6034](https://github.com/sgl-project/sglang/pull/6034), [#6109](https://github.com/sgl-project/sglang/pull/6109).
- MTP and backend interaction: [#3582](https://github.com/sgl-project/sglang/pull/3582), [#4218](https://github.com/sgl-project/sglang/pull/4218), [#4631](https://github.com/sgl-project/sglang/pull/4631), [#5707](https://github.com/sgl-project/sglang/pull/5707), [#5793](https://github.com/sgl-project/sglang/pull/5793), [#5952](https://github.com/sgl-project/sglang/pull/5952), [#6081](https://github.com/sgl-project/sglang/pull/6081), [#6109](https://github.com/sgl-project/sglang/pull/6109).

When updating this skill, explicitly mark whether an H200 optimization is still current-main default, current-main optional, hardware-specific, or only an explored/closed path.

## Evolution Path

### Stage V3R1-H200: Single-node H200 performance ladder

The H200 ladder is the missing context behind many later V3/R1 defaults.

- FP8 Block GEMM evolved from Triton/Cutlass experiments into DeepGEMM-backed paths. Current main exposes DeepGEMM through `fp8_kernel.py`, `deep_gemm.py`, and `compile_deep_gemm.py`; Hopper/Blackwell defaults should be checked with `SGLANG_ENABLE_JIT_DEEPGEMM`.
- Fused MoE acceleration starts before the current `fused_topk_deepseek` abstraction. `moe_fused_gate.cu`, `moe_align_kernel.cu`, `per_token_group_quant_8bit`, routed scaling fusion, and shared-expert fusion all belong to this stage.
- MLA backend selection was built across FlashMLA, Cutlass MLA, FA3 MLA, and later MHA-chunked prefill paths. Do not conclude that a backend is current based only on an early support PR; check `server_args.py`, `attention_backend_handler.py`, and `docs/basic_usage/deepseek_v3.md`.
- Small model-file optimizations matter: DeepSeek CUDA RoPE, removing `forward_absorb` copies, fusing `q_a_proj` with `kv_a_proj_with_mqa`, fusing MLA KV-cache writes, overlapping q/k norm, and removing scalar/allocator overhead all live on the hot path.
- The closed hybrid-attention PR [#6151](https://github.com/sgl-project/sglang/pull/6151) should be cited as non-mainline context, not as shipped V3/R1 support.

Success check:

- every H200-note PR is either in the timeline, in a closed/exploratory note, or explicitly marked out of scope
- current source paths have replaced historical file names where refactors moved the code
- DeepGEMM, FA3/FlashMLA/Cutlass MLA, fused MoE, shared experts, and MTP interactions are checked together

### Stage V3R1-0: Basic V3/R1 support is not enough

Early DeepSeek support can launch, but the optimized path needs hardware-specific MLA, MoE, and quant handling.

- AMD bring-up: [#2601](https://github.com/sgl-project/sglang/pull/2601), [#2667](https://github.com/sgl-project/sglang/pull/2667)
- docs and launch recipes: [#3314](https://github.com/sgl-project/sglang/pull/3314), [#3522](https://github.com/sgl-project/sglang/pull/3522), [#4079](https://github.com/sgl-project/sglang/pull/4079)
- current usage doc: `docs/basic_usage/deepseek_v3.md`

Success check:

- `DeepseekV3ForCausalLM` is selected
- the loader recognizes the quant config
- launch docs and current server defaults agree

### Stage V3R1-1: MLA backend and FP8 correctness

DeepSeek V3/R1 performance depends on MLA path selection, weight absorption, KV-cache dtype, DeepGEMM, and backend fallback.

- block-wise FP8 and BMM paths feed the MLA absorbed path
- `server_args.py` selects `trtllm_mla` on SM100 when no attention backend is set
- `deepseek_weight_loader.py` requantizes or dequantizes `kv_b_proj` according to quant format
- deterministic mode forces supported DeepSeek attention backends

Key PRs:

- [#9744](https://github.com/sgl-project/sglang/pull/9744)
- [#11708](https://github.com/sgl-project/sglang/pull/11708)
- [#14897](https://github.com/sgl-project/sglang/pull/14897)
- [#15531](https://github.com/sgl-project/sglang/pull/15531)
- [#19122](https://github.com/sgl-project/sglang/pull/19122)
- [#23195](https://github.com/sgl-project/sglang/pull/23195), open

### Stage V3R1-2: MoE routing and shared experts

The main model has `256` routed experts plus one shared expert. Current main can remap `mlp.shared_experts` into expert slot `256` when shared-expert fusion is active.

- `DeepseekV2MoE` computes `num_fused_shared_experts`
- `TopK` is configured with grouped top-k, correction bias, routed scaling, and optional fused shared expert slots
- `determine_num_fused_shared_experts()` disables fusion for incompatible shapes, W4AFP8 shared/routed mismatch, TBO/SBO, or DeepEP unless explicitly enforced
- DeepEP fusion expands the local expert layout from `256` routed experts to `256 + EP_size` slots

Key PRs:

- [#15347](https://github.com/sgl-project/sglang/pull/15347)
- [#17707](https://github.com/sgl-project/sglang/pull/17707)
- [#18451](https://github.com/sgl-project/sglang/pull/18451)
- [#20089](https://github.com/sgl-project/sglang/pull/20089)
- [#21657](https://github.com/sgl-project/sglang/pull/21657)
- [#21531](https://github.com/sgl-project/sglang/pull/21531), open

### Stage V3R1-3: MTP and NextN

DeepSeek V3/R1 MTP uses the NextN model as an EAGLE draft path.

- target model: `DeepseekV3ForCausalLM`
- draft model: `DeepseekV3ForCausalLMNextN`
- `deepseek_nextn.py` handles the single NextN layer, shared head/embed reuse, quant override, and AMD R1 MXFP4 naming
- current main allows quantized target and BF16 or differently quantized NextN layers, but validate the exact backend pair

Key PRs:

- [#7376](https://github.com/sgl-project/sglang/pull/7376)
- [#13548](https://github.com/sgl-project/sglang/pull/13548)
- [#18607](https://github.com/sgl-project/sglang/pull/18607)
- [#19425](https://github.com/sgl-project/sglang/pull/19425)

Validation:

- `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`
- `test/registered/amd/test_deepseek_v3_mtp.py`
- `test/registered/spec/eagle/test_deepseek_v3_fp4_mtp_small.py`

### Stage V3R1-4: R1 W4AFP8 and DeepEP

R1 W4AFP8 is a separate ladder from native FP8.

- `W4AFp8Config` detects mixed precision and maps linear layers to FP8 while MoE experts use W4A8
- `cutlass_w4a8_moe.py` handles packed int4 expert weights and FP8 activations
- EP support maps global experts to local partitions
- normal DeepEP uses dispatch output metadata and `apply_deepep_normal`
- low-latency DeepEP has special communication behavior and must be checked against the revert/reland sequence

Key PRs:

- [#7762](https://github.com/sgl-project/sglang/pull/7762)
- [#8118](https://github.com/sgl-project/sglang/pull/8118)
- [#8247](https://github.com/sgl-project/sglang/pull/8247)
- [#8464](https://github.com/sgl-project/sglang/pull/8464)
- [#10027](https://github.com/sgl-project/sglang/pull/10027)
- [#14162](https://github.com/sgl-project/sglang/pull/14162)
- [#21719](https://github.com/sgl-project/sglang/pull/21719)
- [#22316](https://github.com/sgl-project/sglang/pull/22316)

### Stage V3R1-5: Quantized backend coverage

Treat each quantization format as a separate loader and backend contract.

- W4AFP8: `w4afp8.py`, `cutlass_w4a8_moe.py`, DeepEP and TP variants
- NVFP4 / ModelOpt FP4: server defaults and Blackwell backend selection
- MXFP4 / Quark: AMD R1 and open ROCm DeepSeek-architecture work
- MXFP8: FlashInfer TRTLLM routed MoE support
- LoRA: quant-info refactor and DeepSeek MLA LoRA support

Key PRs:

- [#11512](https://github.com/sgl-project/sglang/pull/11512)
- [#12778](https://github.com/sgl-project/sglang/pull/12778)
- [#12921](https://github.com/sgl-project/sglang/pull/12921)
- [#18242](https://github.com/sgl-project/sglang/pull/18242)
- [#19425](https://github.com/sgl-project/sglang/pull/19425)
- [#21280](https://github.com/sgl-project/sglang/pull/21280)
- [#22323](https://github.com/sgl-project/sglang/pull/22323)
- [#21529](https://github.com/sgl-project/sglang/pull/21529), open

### Stage V3R1-6: Distributed and backend hardening

The late-stage failures are usually topology bugs rather than model-architecture bugs.

- DP attention plus torch.compile GPU faults
- BF16 KV accuracy under DP
- AITER correction-bias dtype conversion
- XPU and MUSA backend compatibility
- OOM during weight loading
- PCG and deterministic test cleanup

Key PRs:

- [#10361](https://github.com/sgl-project/sglang/pull/10361)
- [#12000](https://github.com/sgl-project/sglang/pull/12000)
- [#17744](https://github.com/sgl-project/sglang/pull/17744)
- [#18461](https://github.com/sgl-project/sglang/pull/18461)
- [#19834](https://github.com/sgl-project/sglang/pull/19834)
- [#19843](https://github.com/sgl-project/sglang/pull/19843)
- [#20841](https://github.com/sgl-project/sglang/pull/20841)
- [#22774](https://github.com/sgl-project/sglang/pull/22774), open

## Validation Surface

Use the narrowest lane that matches the change:

- base V3: `test/registered/8-gpu-models/test_deepseek_v3_basic.py`
- V3 MTP: `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`
- AMD V3: `test/registered/amd/test_deepseek_v3_basic.py`
- AMD V3 KV FP8: `test/registered/amd/test_deepseek_v3_basic_kv_fp8.py`
- R1 MXFP4: `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py`
- R1 FP8 TRTLLM backend: `test/registered/backends/test_deepseek_r1_fp8_trtllm_backend.py`
- FP4: `test/registered/quant/test_deepseek_v3_fp4_4gpu.py`
- W4A8: `test/registered/quant/test_w4a8_deepseek_v3.py`
- MLA: `test/registered/mla/test_mla_deepseek_v3.py`
- INT8 MLA: `test/registered/mla/test_mla_int8_deepseek_v3.py`
- LoRA: `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py`
- router/top-k kernel: `test/registered/kernels/test_fused_topk_deepseek.py`
