---
name: sglang-qwen3-next-optimization
description: PR-diff-backed optimization manual for Qwen3-Next, Qwen3-Next MTP, and Qwen3-Coder-Next shared hybrid paths in SGLang. Use when Codex needs to audit, extend, or debug Qwen3-Next GDN/Mamba/RadixLinearAttention, MTP/EAGLE/NEXTN, FP8/NVFP4/ModelOpt loading, CPU offload, FlashInfer/CuTe/Gluon GDN kernels, AMD/NPU/Blackwell paths, mixed-chunk extra_buffer behavior, or Qwen3-Next cookbook deployment flags.
---

# SGLang Qwen3-Next Optimization

## Required Reading

Before writing or changing Qwen3-Next docs/code, read:

1. `references/pr-history.md`
2. `references/playbook.md`
3. The exact PR diffs for any PR you cite
4. The current SGLang runtime files you are touching

This skill follows the repository-level requirement captured in `skills/model-optimization/model-pr-diff-dossier`: every PR entry must be based on code diff inspection and must include motivation, implementation idea, and the key code fragment or pseudo-fragment that explains the change.

## Scope

Treat Qwen3-Next as an independent model family. It should not be collapsed into generic Qwen3 MoE unless the change is genuinely shared and no Qwen3-Next-specific runtime behavior is involved.

Covered surfaces:

- Base architecture: `Qwen3NextForCausalLM`
- MTP architecture: `Qwen3NextForCausalLMMTP`
- Shared Qwen3-Coder-Next architecture/runtime
- Hybrid GDN/Mamba state pools and RadixLinearAttention
- GDN projection fusion, fused norm/gate, FlashInfer, CuTe DSL, and Gluon kernels
- FP8/NVFP4/ModelOpt/checkpoint loading
- CPU offload tied-parameter and tensor-view aliasing
- AMD MI300/MI325/MI355, Ascend NPU, Hopper, Blackwell
- Mamba radix-cache scheduling, `extra_buffer`, mixed chunk, and prefix reuse

## First Checks

Capture these facts before debugging or documenting:

- Checkpoint: Qwen3-Next, Qwen3-Coder-Next, Qwen3.5 hybrid, or ModelOpt FP8/NVFP4 variant
- MTP/NEXTN/EAGLE enabled or disabled
- `SGLANG_ENABLE_SPEC_V2`
- `--mamba-scheduler-strategy`: `no_buffer` or `extra_buffer`
- `--enable-mixed-chunk`
- `--mamba-ssm-dtype`, `--page-size`, radix/prefix cache flags
- TP/DP/EP/PP and DeepEP/EPLB status
- Backend: CUDA SM90/SM100, AMD, Ascend NPU, CPU/AMX
- Quantization: BF16, FP8, W8A8, NVFP4, compressed-tensors
- CPU offload: `--cpu-offload-gb`

## PRs That Define the Lane

Initial architecture and state:

- `#10233`: initial Qwen3-Next support
- `#10322`: Gemma RMSNorm normalization fix
- `#10379`, `#11969`, `#16164`: Ascend NPU bring-up and W8A8 follow-ups
- `#10912`: PD disaggregation with Mamba extra pool transfer

GDN/Mamba performance:

- `#12508`: fused GDN gating
- `#13081`, `#17613`, `#19220`: PCG evolution
- `#15631`, `#17981`, `#17983`, `#23273`: CuTe/Gluon/FlashInfer GDN kernels
- `#18917`, `#19321`, `#19434`, `#21019`: fused projection and fused norm/gate path

MTP/speculative:

- `#10392`: MTP + DP fixes
- `#14607`: EAGLE3
- `#19767`: MTP + EPLB
- `#22458`: TP-rank broadcast to avoid MTP NCCL hang
- `#12892`, `#14502`, `#16488`: open state-copy/PCG/TBO optimization radar

Quantization/offload:

- `#10466`, `#10622`, `#17627`, `#18224`, `#21313`, `#21496`, `#21662`, `#21698`
- `#23474`: CPU offload for hybrid linear-attention tied params and cached tensor views

Scheduler/cache correctness:

- `#21684`: allocator clone for memory leak
- `#22876`: guard mixed chunk + `extra_buffer`
- `#23075`: root-cause metadata slicing fix for mixed chunk + `extra_buffer`

## Operating Rules

- Do not claim a PR is retained behavior if a later PR reverted or superseded it. For example, document `#21313` and `#21496` only as intermediate loader history; current behavior comes from `#21662`.
- Mark adjacent PRs clearly. `#22073` is Qwen3-ASR and only touches shared Qwen-family surfaces; do not treat it as a GDN optimization.
- For open PRs, label them as open radar and avoid presenting them as current-main behavior.
- When a PR title says Qwen3-Next but the current diff only touches Qwen3.5/shared code, say so. `#19812` is an example; merged Qwen3-Next MTP/EPLB behavior comes from `#19767`.
- For backend-specific PRs, state backend assumptions. NPU conv state layout, AMD dual stream behavior, and CUDA FlashInfer/CuTe kernels are not interchangeable.

## Validation Lanes

Run the narrowest relevant tests first:

```bash
python -m pytest test/registered/4-gpu-models/test_qwen3_next_models.py
python -m pytest test/registered/4-gpu-models/test_qwen3_next_models_mtp.py
python -m pytest test/registered/models/test_qwen3_next_models_fp4.py
```

Then add lane-specific checks:

- GDN fusion: logits parity, GSM8K/GPQA, TTFT, TPOT, and kernel profile
- MTP: base no-MTP, NEXTN, EAGLE3, radix cache on/off, TP>1 sampling
- Mixed chunk + Mamba: `extra_buffer`, `no_buffer`, concurrency, prefix reuse
- CPU offload: tied params, cached tensor views, greedy equivalence, quantized and BF16 checkpoints
- NPU/AMD: backend-native kernels and registered backend tests, not CUDA-only tests

## Output Format for Future Docs

Each PR card should include:

- PR link and merged/open status
- Motivation from the bug/performance/context, not only title
- Implementation idea in concrete code terms
- A short key code fragment
- Validation evidence or missing validation note
- Whether the PR is current behavior, adjacent, superseded, or open radar


## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.
