---
name: sglang-qwen3-core-optimization
description: PR-diff-backed optimization manual for Qwen3 dense and Qwen3 MoE in SGLang. Use when Codex needs to recover, extend, audit, or write documentation for Qwen3/Qwen3-30B/Qwen3-235B-A22B, FP8/NVFP4/MXFP4/W4A4, fused QK-norm/RoPE/KV-store paths, FlashInfer TRTLLM-GEN-MoE, DeepEP/EPLB/TBO/context parallel, EAGLE3, LoRA, PP/tied embeddings, Ascend NPU/XPU/MLX support, or Qwen3 reasoning/tool-parser behavior.
---

# SGLang Qwen3 Core Optimization

This skill covers the non-hybrid Qwen3 text path: dense Qwen3, Qwen3 MoE, Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3 Instruct/Thinking variants, and the shared Qwen3 infrastructure reused by Qwen3.5, Qwen3-Next, Qwen3.6, Qwen3 Omni thinker-only, and Qwen-family quantization loaders.

## Mandatory Evidence Standard

When writing or revising Qwen3 Core optimization docs, use [../sglang-model-pr-diff-dossier/SKILL.md](../sglang-model-pr-diff-dossier/SKILL.md) as the production standard.

Do not fill PR history from a script, generated table, or title-level summary. For every PR you cite, read the PR diff or the final merge commit and record:

- why the PR existed,
- which runtime/docs/tests changed,
- the key implementation idea,
- the most important real code excerpt,
- validation impact and regression risk.

If a PR is docs-only, quote the exact launch/config line that changed and explain why that line matters for serving or validation.

## Evidence Files

- [references/pr-history.md](references/pr-history.md): canonical per-PR diff dossier for Qwen3 Core.
- [references/playbook.md](references/playbook.md): fast triage map from symptom to PR families and validation lanes.
- `model-pr-optimization-history/sglang/qwen3-core/README.zh.md`: Chinese PR optimization history.
- `model-pr-optimization-history/sglang/qwen3-core/README.en.md`: English PR optimization history.

Current evidence snapshot:

- SGLang mainline checked around `2026-04-22`: `b3e6cf60a`
- sgl-cookbook mainline checked around `2026-04-21`: `816bad5`
- Three-pass audit ledger: `model-pr-optimization-history/sglang/audits/qwen-glm-three-pass-audit-2026-04-23.md`

## Runtime Surfaces

Start from these files before making a Qwen3 Core change:

- `python/sglang/srt/models/qwen3.py`
- `python/sglang/srt/models/qwen3_moe.py`
- `python/sglang/srt/models/qwen2.py`
- `python/sglang/srt/layers/moe/`
- `python/sglang/srt/layers/layernorm.py`
- `python/sglang/srt/layers/quantization/`
- `python/sglang/srt/layers/attention/`
- `python/sglang/srt/distributed/`
- `python/sglang/srt/function_call/qwen25_detector.py`
- `test/registered/models/test_qwen_models.py`
- `test/registered/4-gpu-models/test_qwen3_30b.py`
- `test/registered/stress/test_stress_qwen3_235b.py`
- `test/srt/models/test_lora_qwen3.py`
- `test/registered/backends/test_qwen3_fp4_trtllm_gen_moe.py`
- NPU Qwen tests under `test/registered/npu/`
- `docs/basic_usage/qwen3.md`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx`

## Orientation

Treat Qwen3 Core as the compatibility baseline:

- Dense Qwen3 owns the common packed QKV, Q/K RMSNorm, RoPE fallback, LM-head, LoRA, embedding, parser, and many platform paths.
- Qwen3 MoE owns FusedMoE/EPMoE/DeepEP/EPLB/TBO/FlashInfer/FP8/NVFP4/CP behavior.
- Later Qwen families often reuse Qwen3 fixes: PP splitting, tied embeddings, packed module mapping, RoPE config fallback, fused QK norm, EAGLE3 hidden capture, and backend guards.
- Open PRs are radar only. Re-read their latest diffs before presenting them as supported behavior.

## Change Order

1. Identify the exact model class: `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, pooled output, embedding, or a downstream Qwen family reusing Qwen3 logic.
2. Confirm config compatibility: `rope_parameters`, top-level `rope_theta`, `rope_scaling`, `layer_types`, `tie_word_embeddings`, and quantization config.
3. Establish BF16 correctness with the smallest viable TP/PP shape.
4. Add or debug quantization only after BF16 works: ModelOpt FP8/NVFP4, MXFP4, W4AFP8, GPTQ, ModelSlim, or platform-native quant.
5. For MoE, separate routing/top-k, expert weight loading, A2A backend, all-reduce/reduce-scatter, and load balancing.
6. For fused kernels, prove fallback correctness first, then enable fused QK-norm/RoPE, fused KV store, fused all-reduce, or FP8 KV write.
7. For platform work, keep CUDA, NPU, XPU, MLX, and AMD logic behind backend-specific gates.
8. Update PR history docs with per-PR diff cards if the change affects model support or serving behavior.

## Open Radar to Recheck

- [#9147](https://github.com/sgl-project/sglang/pull/9147): Qwen3-MoE W4AFP8.
- [#20127](https://github.com/sgl-project/sglang/pull/20127): tied embeddings for Qwen MoE and Qwen3-Next.
- [#20474](https://github.com/sgl-project/sglang/pull/20474): Intel XPU Qwen3 layernorm/MRoPE support.
- [#20520](https://github.com/sgl-project/sglang/pull/20520): NPU TP communication compression.
- [#21412](https://github.com/sgl-project/sglang/pull/21412): dense Qwen3 old-style RoPE compatibility.
- [#21770](https://github.com/sgl-project/sglang/pull/21770): Apple MLX Qwen3 tests.
- [#22529](https://github.com/sgl-project/sglang/pull/22529): sliding window attention for Qwen3.
- [#22674](https://github.com/sgl-project/sglang/pull/22674): shared Qwen NPU quant packed mappings.
- [#22837](https://github.com/sgl-project/sglang/pull/22837): Qwen3 reasoning detector tool-call boundary.
- [#23372](https://github.com/sgl-project/sglang/pull/23372): NPU speculative decoding CI.
- [#23397](https://github.com/sgl-project/sglang/pull/23397): dense deterministic math for alignment.
- [#23434](https://github.com/sgl-project/sglang/pull/23434): Qwen3 pooled-output embedding accessor.

## Validation Lanes

- Dense correctness: Qwen3 tiny/0.6B/1.7B/4B/8B launch and deterministic prompt checks.
- MoE correctness: Qwen3-30B-A3B and Qwen3-235B-A22B under TP/EP/DP attention.
- Quantization: BF16 reference plus one target checkpoint per backend, especially ModelOpt FP8/NVFP4 and NPU GPTQ/ModelSlim.
- Fused kernels: compare fallback vs fused QK-norm/RoPE, fused KV store, fused FP8 KV write, and fused all-reduce.
- PP/tied embeddings: tied and untied checkpoints, with and without `lm_head.weight`.
- Parser: streaming and non-streaming reasoning plus `<tool_call>` before and after `</think>`.
- Platform: CUDA H100/H200/B200/GB200, Ascend NPU, Intel XPU, Apple MLX, and AMD lanes separately.
